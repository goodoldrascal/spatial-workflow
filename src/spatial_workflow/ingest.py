"""Convert one spatial transcriptomics source into the project AnnData contract."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from .config import load_config, resolve_path


BUNDLE_FILES = ("counts.mtx.gz", "genes.tsv.gz", "barcodes.tsv.gz", "obs.csv.gz")
MERGED_SEURAT_FORMAT = "merged_seurat"
MERGED_BUNDLE_MANIFEST = "bundle_manifest.csv"
INT32_MAX = np.iinfo(np.int32).max
COUNT_VALIDATION_CHUNK_SIZE = 1_000_000


def _read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n").split("\t", 1)[0] for line in handle]


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _coordinate_columns(columns: pd.Index) -> tuple[str, str] | None:
    for pair in (("x_centroid", "y_centroid"), ("centroid_x", "centroid_y"), ("x", "y")):
        if set(pair).issubset(columns):
            return pair
    return None


def _validate_counts(counts: sp.spmatrix) -> sp.csr_matrix:
    counts = counts.tocsr()
    values = counts.data
    if not values.size:
        counts.data = values.astype(np.int32, copy=False)
        return counts

    if np.issubdtype(values.dtype, np.integer):
        if values.min() < 0:
            raise ValueError("Raw counts must be finite and non-negative")
        if values.max() > INT32_MAX:
            raise ValueError("Raw counts exceed the int32 storage range")
    elif np.issubdtype(values.dtype, np.floating):
        for start in range(0, values.size, COUNT_VALIDATION_CHUNK_SIZE):
            chunk = values[start : start + COUNT_VALIDATION_CHUNK_SIZE]
            if not np.isfinite(chunk).all() or np.any(chunk < 0):
                raise ValueError("Raw counts must be finite and non-negative")
            if np.any(chunk > INT32_MAX):
                raise ValueError("Raw counts exceed the int32 storage range")
            if not np.equal(chunk, np.floor(chunk)).all():
                raise ValueError("The configured count source contains non-integer values")
    else:
        raise ValueError("The configured count source contains non-integer values")

    counts.data = values.astype(np.int32, copy=False)
    return counts


def _normalized_log1p(counts: sp.csr_matrix, target_sum: float | None) -> sp.csr_matrix:
    matrix = counts.astype(np.float32)
    if target_sum is None:
        return matrix
    target_sum = float(target_sum)
    if target_sum <= 0:
        raise ValueError("conversion.normalize_target must be positive or null")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = sp.diags(scale.astype(np.float32)) @ matrix
    matrix = matrix.tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def _build_anndata(
    counts: sp.spmatrix,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    cell_ids: list[str],
    coordinates: np.ndarray,
    *,
    sample_id: str,
    condition: str | None,
    schema: dict[str, Any],
    normalize_target: float | None,
) -> ad.AnnData:
    counts = _validate_counts(counts)
    if counts.shape != (len(cell_ids), len(var)):
        raise ValueError(
            f"Matrix shape {counts.shape} does not match {len(cell_ids)} cells "
            f"and {len(var)} features"
        )
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Cell identifiers must be unique within a sample")
    if not var.index.is_unique:
        raise ValueError("Feature names must be unique")

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(cell_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Every cell must have two finite spatial coordinates")

    cell_key = schema["cell_id_key"]
    sample_key = schema["sample_key"]
    condition_key = schema["condition_key"]
    counts_layer = schema["counts_layer"]
    spatial_key = schema["spatial_key"]

    obs = obs.reset_index(drop=True).copy()
    obs[cell_key] = cell_ids
    if condition is None:
        missing = [key for key in (sample_key, condition_key) if key not in obs]
        if missing:
            raise ValueError(
                "Merged Seurat bundle metadata is missing configured columns: "
                f"{missing}"
            )
        if obs[[sample_key, condition_key]].isna().any(axis=None):
            raise ValueError("Merged Seurat sample and condition metadata cannot be missing")
        observed_samples = obs[sample_key].astype(str).unique().tolist()
        if observed_samples != [str(sample_id)]:
            raise ValueError(
                f"Merged Seurat bundle {sample_id!r} contains sample values "
                f"{observed_samples}"
            )
        obs[sample_key] = obs[sample_key].astype(str)
        obs[condition_key] = obs[condition_key].astype(str)
    else:
        obs[sample_key] = str(sample_id)
        obs[condition_key] = str(condition)
    obs.index = pd.Index([f"{sample_id}:{cell_id}" for cell_id in cell_ids], name="cell")

    data = ad.AnnData(X=_normalized_log1p(counts, normalize_target), obs=obs, var=var.copy())
    data.layers[counts_layer] = counts
    data.obsm[spatial_key] = coordinates
    return data


def _align_metadata(table: pd.DataFrame, cell_ids: list[str], id_column: str) -> pd.DataFrame:
    if id_column not in table:
        raise ValueError(f"Metadata must contain {id_column!r}")
    table = table.copy()
    table[id_column] = table[id_column].astype(str)
    if table[id_column].duplicated().any():
        raise ValueError(f"Metadata column {id_column!r} contains duplicate cell identifiers")
    table = table.set_index(id_column)
    missing = pd.Index(cell_ids).difference(table.index)
    if len(missing):
        raise ValueError(f"Metadata is missing {len(missing)} matrix cell identifiers")
    return table.loc[cell_ids].reset_index()


def read_bundle(
    path: str | Path,
    *,
    sample_id: str,
    condition: str | None,
    schema: dict[str, Any],
    normalize_target: float | None = 1e4,
) -> ad.AnnData:
    """Read one genes-by-cells MatrixMarket bundle."""

    bundle = Path(path)
    if not (bundle / BUNDLE_FILES[0]).exists():
        candidates = sorted(
            path
            for path in bundle.iterdir()
            if path.is_dir() and (path / BUNDLE_FILES[0]).exists()
        )
        if len(candidates) != 1:
            raise ValueError(f"Expected one bundle under {bundle}, found {len(candidates)}")
        bundle = candidates[0]
    missing = [name for name in BUNDLE_FILES if not (bundle / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing required files: {missing}")

    with gzip.open(bundle / "counts.mtx.gz", "rb") as handle:
        counts = scipy.io.mmread(handle).T.tocsr()
    genes = _read_lines(bundle / "genes.tsv.gz")
    cell_ids = _read_lines(bundle / "barcodes.tsv.gz")
    if counts.shape != (len(cell_ids), len(genes)):
        raise ValueError("Bundle matrix, genes, and barcodes have inconsistent dimensions")

    obs_raw = pd.read_csv(bundle / "obs.csv.gz")
    source_cell_key = schema["cell_id_key"] if schema["cell_id_key"] in obs_raw else "cell_id"
    obs = _align_metadata(obs_raw, cell_ids, source_cell_key)

    spatial_path = bundle / "spatial.csv.gz"
    if spatial_path.exists():
        spatial = _align_metadata(pd.read_csv(spatial_path), cell_ids, "cell_id")
        pair = _coordinate_columns(spatial.columns)
        if pair is None:
            raise ValueError("spatial.csv.gz must contain x/y coordinates")
        coordinates = spatial.loc[:, list(pair)].to_numpy()
    else:
        pair = _coordinate_columns(obs.columns)
        if pair is None:
            raise ValueError(
                "Bundle must provide spatial.csv.gz or coordinate columns in obs.csv.gz"
            )
        coordinates = obs.loc[:, list(pair)].to_numpy()

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    data = _build_anndata(
        counts,
        obs,
        var,
        cell_ids,
        coordinates,
        sample_id=sample_id,
        condition=condition,
        schema=schema,
        normalize_target=normalize_target,
    )
    for filename, key in (("pca.csv.gz", "X_pca"), ("umap.csv.gz", "X_umap")):
        embedding_path = bundle / filename
        if embedding_path.exists():
            table = _align_metadata(pd.read_csv(embedding_path), cell_ids, "cell_id")
            data.obsm[key] = table.select_dtypes(include="number").to_numpy()
    return data


def read_xenium(
    path: str | Path,
    *,
    sample_id: str,
    condition: str,
    schema: dict[str, Any],
    normalize_target: float | None = 1e4,
) -> ad.AnnData:
    """Read one raw Xenium output directory."""

    xenium = Path(path)
    matrix_path = xenium / "cell_feature_matrix.h5"
    cells_path = xenium / "cells.parquet"
    if not cells_path.exists():
        cells_path = xenium / "cells.csv.gz"
    if not matrix_path.exists() or not cells_path.exists():
        raise FileNotFoundError(
            f"Expected cell_feature_matrix.h5 and cells.parquet or cells.csv.gz under {xenium}"
        )

    with h5py.File(matrix_path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sp.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )
        cell_ids = _decode(group["barcodes"][:])
        names = np.asarray(_decode(group["features/name"][:]))
        feature_types = np.asarray(_decode(group["features/feature_type"][:]))
        feature_ids = (
            np.asarray(_decode(group["features/id"][:]))
            if "id" in group["features"]
            else names.copy()
        )

    keep = feature_types == "Gene Expression"
    if not keep.any():
        raise ValueError("Xenium matrix has no 'Gene Expression' features")
    counts = matrix[keep, :].T.tocsr()
    var = pd.DataFrame(
        {
            "feature_id": feature_ids[keep],
            "feature_type": feature_types[keep],
        },
        index=pd.Index(names[keep], name="gene"),
    )

    cells = _align_metadata(_read_table(cells_path), cell_ids, "cell_id")
    pair = _coordinate_columns(cells.columns)
    if pair is None:
        raise ValueError("Xenium cells table must contain x_centroid/y_centroid coordinates")
    return _build_anndata(
        counts,
        cells,
        var,
        cell_ids,
        cells.loc[:, list(pair)].to_numpy(),
        sample_id=sample_id,
        condition=condition,
        schema=schema,
        normalize_target=normalize_target,
    )


def _add_annotations(
    data: ad.AnnData,
    annotations: str | Path | dict[str, Any],
    cell_key: str,
) -> None:
    if isinstance(annotations, dict):
        path = Path(annotations["path"])
        id_column = annotations.get("cell_id_column", cell_key)
    else:
        path = Path(annotations)
        id_column = cell_key
    table = _read_table(path)
    aligned = _align_metadata(table, data.obs[cell_key].astype(str).tolist(), id_column)
    aligned = aligned.drop(columns=[id_column], errors="ignore")
    overlap = sorted(set(aligned.columns).intersection(data.obs.columns))
    if overlap:
        raise ValueError(f"Annotation columns already exist in obs: {overlap}")
    aligned.index = data.obs_names
    for column in aligned:
        data.obs[column] = aligned[column]


def _rscript_environment(r_libs_user: str | Path | None) -> dict[str, str] | None:
    if r_libs_user is None:
        return None
    environment = os.environ.copy()
    environment["R_LIBS_USER"] = str(r_libs_user)
    return environment


def _export_seurat(
    input_path: Path,
    bundle_root: Path,
    *,
    sample_id: str,
    assay: str,
    layer: str,
    rscript_bin: str,
    r_libs_user: str | Path | None = None,
) -> Path:
    script = Path(__file__).resolve().parent / "resources" / "export_seurat.R"
    subprocess.run(
        [
            rscript_bin,
            str(script),
            "--input",
            str(input_path),
            "--output-dir",
            str(bundle_root),
            "--sample-id",
            sample_id,
            "--assay",
            assay,
            "--layer",
            layer,
        ],
        check=True,
        env=_rscript_environment(r_libs_user),
    )
    return bundle_root / sample_id


def _export_merged_seurat(
    input_path: Path,
    bundle_root: Path,
    *,
    assay: str,
    layer: str,
    sample_key: str,
    condition_key: str,
    rscript_bin: str,
    r_libs_user: str | Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    script = Path(__file__).resolve().parent / "resources" / "export_seurat.R"
    subprocess.run(
        [
            rscript_bin,
            str(script),
            "--input",
            str(input_path),
            "--output-dir",
            str(bundle_root),
            "--sample-key",
            sample_key,
            "--condition-key",
            condition_key,
            "--assay",
            assay,
            "--layer",
            layer,
        ],
        check=True,
        env=_rscript_environment(r_libs_user),
    )

    manifest_path = bundle_root / MERGED_BUNDLE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Merged Seurat export did not write {manifest_path}")
    table = pd.read_csv(manifest_path, dtype=str)
    required = {"sample_id", "condition", "bundle_dir", "n_cells"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Merged Seurat bundle manifest is missing columns: {missing}")
    if table.empty:
        raise ValueError("Merged Seurat bundle manifest contains no samples")

    table["sample_id"] = table["sample_id"].astype(str)
    table["condition"] = table["condition"].astype(str)
    table["bundle_dir"] = table["bundle_dir"].astype(str)
    if table["sample_id"].duplicated().any():
        raise ValueError("Merged Seurat bundle manifest contains duplicate sample IDs")

    records = []
    for record in table.to_dict(orient="records"):
        bundle_dir = Path(record["bundle_dir"])
        if bundle_dir.is_absolute() or ".." in bundle_dir.parts:
            raise ValueError(f"Invalid bundle path in merged Seurat manifest: {bundle_dir}")
        records.append(
            {
                "sample_id": record["sample_id"],
                "condition": record["condition"],
                "bundle_path": bundle_root / bundle_dir,
                "n_cells": int(record["n_cells"]),
            }
        )
    return manifest_path, records


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("anndata", "h5py", "numpy", "pandas", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def _source_entries(conversion: dict[str, Any]) -> list[dict[str, Any]]:
    stage_format = str(conversion.get("input_format", "")).lower()
    if stage_format == MERGED_SEURAT_FORMAT:
        if "samples" in conversion:
            raise ValueError("merged_seurat uses conversion.input_path, not conversion.samples")
        if "input_path" not in conversion:
            raise ValueError("merged_seurat requires conversion.input_path")
        return [
            {
                "input_path": conversion["input_path"],
                "input_format": MERGED_SEURAT_FORMAT,
                "annotations": conversion.get("annotations"),
            }
        ]

    samples = conversion.get("samples")
    if samples is None:
        samples = [conversion]
    elif not isinstance(samples, list) or not samples:
        raise ValueError("conversion.samples must be a non-empty list")

    entries = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"conversion.samples[{index}] must be a mapping")
        missing = [key for key in ("input_path", "sample_id", "condition") if key not in sample]
        if missing:
            raise ValueError(f"conversion.samples[{index}] is missing: {missing}")
        input_format = sample.get("input_format", conversion.get("input_format"))
        if not input_format:
            raise ValueError(
                f"conversion.samples[{index}] requires input_format or a stage default"
            )
        if str(input_format).lower() == MERGED_SEURAT_FORMAT:
            raise ValueError(
                "merged_seurat must be configured once at the conversion stage"
            )
        entries.append(
            {
                "input_path": sample["input_path"],
                "sample_id": str(sample["sample_id"]),
                "condition": str(sample["condition"]),
                "input_format": str(input_format).lower(),
                "annotations": sample.get("annotations", conversion.get("annotations")),
            }
        )
    sample_ids = [entry["sample_id"] for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("conversion.samples must use unique sample_id values")
    return entries


def _apply_annotations_from_config(
    data: ad.AnnData,
    annotations: str | Path | dict[str, Any] | None,
    *,
    config_path: Path,
    data_root: Path,
    cell_key: str,
) -> str | dict[str, Any] | None:
    if not annotations:
        return None
    if isinstance(annotations, dict):
        annotation_path = resolve_path(config_path, annotations["path"], root=data_root)
        resolved = {**annotations, "path": str(annotation_path)}
        _add_annotations(data, {**annotations, "path": annotation_path}, cell_key)
        return resolved

    annotation_path = resolve_path(config_path, annotations, root=data_root)
    _add_annotations(data, annotation_path, cell_key)
    return str(annotation_path)


def run_ingest(config_path: str | Path) -> Path:
    """Run the configured conversion and return the written H5AD path."""

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    conversion = config["conversion"]
    schema = config["schema"]
    data_root = resolve_path(config_path, config["paths"]["data_root"])
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    output_dir = resolve_path(config_path, conversion["output_dir"], root=results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = str(conversion["output_name"])
    output_path = output_dir / (
        output_name if output_name.endswith(".h5ad") else f"{output_name}.h5ad"
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Conversion outputs already exist: {', '.join(map(str, existing))}")

    normalize_target = conversion.get("normalize_target")
    runtime = config.get("runtime", {})
    rscript_bin = str(runtime.get("rscript_bin", "Rscript"))
    r_libs_user = runtime.get("r_libs_user")
    if r_libs_user is not None:
        r_libs_user = resolve_path(config_path, r_libs_user)

    datasets = []
    resolved_sources = []
    input_objects = []
    source_entries = _source_entries(conversion)
    input_mode = (
        MERGED_SEURAT_FORMAT
        if source_entries[0]["input_format"] == MERGED_SEURAT_FORMAT
        else "per_sample"
    )
    for source in source_entries:
        input_path = resolve_path(config_path, source["input_path"], root=data_root)
        source_path = input_path
        input_format = source["input_format"]

        if input_format == MERGED_SEURAT_FORMAT:
            assay = conversion.get("assay")
            layer = conversion.get("layer")
            if not assay or not layer:
                raise ValueError(
                    "Seurat conversion requires explicit conversion.assay and conversion.layer"
                )
            bundle_manifest, exported_samples = _export_merged_seurat(
                input_path,
                output_dir / "bundles",
                assay=str(assay),
                layer=str(layer),
                sample_key=str(schema["sample_key"]),
                condition_key=str(schema["condition_key"]),
                rscript_bin=rscript_bin,
                r_libs_user=r_libs_user,
            )
            input_objects.append(
                {
                    "input_format": input_format,
                    "input_path": str(input_path),
                    "bundle_manifest": str(bundle_manifest),
                    "sample_key": schema["sample_key"],
                    "condition_key": schema["condition_key"],
                    "n_samples": len(exported_samples),
                }
            )
            for exported in exported_samples:
                sample_id = exported["sample_id"]
                expected_condition = exported["condition"]
                source_path = exported["bundle_path"]
                data = read_bundle(
                    source_path,
                    sample_id=sample_id,
                    condition=None,
                    schema=schema,
                    normalize_target=normalize_target,
                )
                conditions = data.obs[schema["condition_key"]].astype(str).unique().tolist()
                if conditions != [expected_condition]:
                    raise ValueError(
                        f"Merged Seurat sample {sample_id!r} has condition values "
                        f"{conditions}; export manifest records {expected_condition!r}"
                    )
                if data.n_obs != exported["n_cells"]:
                    raise ValueError(
                        f"Merged Seurat sample {sample_id!r} has {data.n_obs} cells; "
                        f"export manifest records {exported['n_cells']}"
                    )
                resolved_annotations = _apply_annotations_from_config(
                    data,
                    source["annotations"],
                    config_path=config_path,
                    data_root=data_root,
                    cell_key=schema["cell_id_key"],
                )
                datasets.append(data)
                resolved_sources.append(
                    {
                        "input_format": input_format,
                        "input_path": str(input_path),
                        "source_path": str(source_path),
                        "sample_id": sample_id,
                        "condition": expected_condition,
                        "condition_source": f"obs.{schema['condition_key']}",
                        "annotations": resolved_annotations,
                        "n_cells": data.n_obs,
                    }
                )
            continue

        sample_id = source["sample_id"]
        condition = source["condition"]

        if input_format in {"seurat_rds", "rds", "seurat_qs", "qs"}:
            assay = conversion.get("assay")
            layer = conversion.get("layer")
            if not assay or not layer:
                raise ValueError(
                    "Seurat conversion requires explicit conversion.assay and conversion.layer"
                )
            source_path = _export_seurat(
                input_path,
                output_dir / "bundles",
                sample_id=sample_id,
                assay=str(assay),
                layer=str(layer),
                rscript_bin=rscript_bin,
                r_libs_user=r_libs_user,
            )
            data = read_bundle(
                source_path,
                sample_id=sample_id,
                condition=condition,
                schema=schema,
                normalize_target=normalize_target,
            )
        elif input_format in {"bundle", "bundles"}:
            data = read_bundle(
                source_path,
                sample_id=sample_id,
                condition=condition,
                schema=schema,
                normalize_target=normalize_target,
            )
        elif input_format in {"xenium", "raw_xenium"}:
            data = read_xenium(
                source_path,
                sample_id=sample_id,
                condition=condition,
                schema=schema,
                normalize_target=normalize_target,
            )
        else:
            raise ValueError(f"Unsupported conversion.input_format: {input_format}")

        resolved_annotations = _apply_annotations_from_config(
            data,
            source["annotations"],
            config_path=config_path,
            data_root=data_root,
            cell_key=schema["cell_id_key"],
        )

        datasets.append(data)
        input_objects.append(
            {
                "input_format": input_format,
                "input_path": str(input_path),
                "n_samples": 1,
            }
        )
        resolved_sources.append(
            {
                "input_format": input_format,
                "input_path": str(input_path),
                "source_path": str(source_path),
                "sample_id": sample_id,
                "condition": condition,
                "condition_source": "configuration",
                "annotations": resolved_annotations,
                "n_cells": data.n_obs,
            }
        )

    join = str(conversion.get("join", "inner")).lower()
    if join not in {"inner", "outer"}:
        raise ValueError("conversion.join must be 'inner' or 'outer'")
    data = (
        datasets[0].copy()
        if len(datasets) == 1
        else ad.concat(datasets, join=join, merge="same", index_unique=None)
    )
    if not data.obs_names.is_unique:
        raise ValueError("Merged observation names are not globally unique")
    if schema["counts_layer"] not in data.layers:
        raise ValueError(f"Merged object lost counts layer {schema['counts_layer']!r}")
    data.uns["spatial_workflow_ingest"] = {
        "join": join,
        "n_sources": len(datasets),
        "n_input_objects": len(input_objects),
    }
    data.write_h5ad(output_path, compression="gzip")

    sample_counts = data.obs[schema["sample_key"]].astype(str).value_counts().sort_index()
    manifest = {
        "config_path": str(config_path),
        "configuration": config,
        "output_h5ad": str(output_path),
        "shape": list(data.shape),
        "join": join,
        "input_mode": input_mode,
        "input_objects": input_objects,
        "sources": resolved_sources,
        "sample_cell_counts": {sample: int(count) for sample, count in sample_counts.items()},
        "normalize_target": normalize_target,
        "schema": schema,
        "software": _package_versions(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(run_ingest(args.config))


if __name__ == "__main__":
    main()
