#!/usr/bin/env python3
"""Append curated cross-lineage cNMF usages to a derived master AnnData.

The selected usage matrix is sparse. Values outside a program's source
lineage are structurally absent, so a parallel applicability matrix is written
to distinguish "not applicable" from a measured zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


LINEAGE_DIRECTORIES = {
    "astrocyte": "astrocyte",
    "oligodendrocyte": "oligodendrocyte",
    "microglia": "microglia_cluster_sub_all",
    "inhibitory_neuron": "inhibitory_neuron",
    "excitatory_neuron": "excitatory_neuron",
    "perivascular": "perivascular",
    "epd": "epd",
    "chp": "chp",
}

REQUIRED_COLUMNS = [
    "lineage",
    "selected_k",
    "program",
    "proposed_label",
    "category",
    "confidence",
    "primary_include",
    "sensitivity_include",
    "decision",
    "rationale",
    "top_genes",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    cnmf_root = project_root / "results" / "ab_xenium" / "05_cnmf"
    parser = argparse.ArgumentParser(
        description=(
            "Build a derived CellCharter AnnData with curated cNMF usages in "
            "obsm and feature provenance in uns."
        )
    )
    parser.add_argument(
        "--master-h5ad",
        type=Path,
        default=project_root
        / "results"
        / "ab_xenium"
        / "02_cellcharter"
        / "ab_xenium_cellcharter.h5ad",
    )
    parser.add_argument("--cnmf-root", type=Path, default=cnmf_root)
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=cnmf_root
        / "program_review"
        / "cnmf_program_whitelist_human_reviewed.tsv",
    )
    parser.add_argument(
        "--whitelist-manifest",
        type=Path,
        default=None,
        help="Defaults to WHITELIST with .manifest.json suffix.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=cnmf_root
        / "integrated"
        / "ab_xenium_cellcharter_cnmf_human_reviewed.h5ad",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_H5AD with .manifest.json suffix.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("primary", "sensitivity"),
        default="sensitivity",
        help=(
            "primary writes only primary_include programs; sensitivity writes "
            "all sensitivity_include programs and marks the primary subset."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_whitelist(path: Path, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(table.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Unexpected whitelist columns: {list(table.columns)}; "
            f"expected {REQUIRED_COLUMNS}"
        )
    if table.duplicated(["lineage", "program"]).any():
        duplicate = table.loc[
            table.duplicated(["lineage", "program"], keep=False),
            ["lineage", "program"],
        ]
        raise ValueError(f"Duplicate lineage-program rows:\n{duplicate}")
    for column in ("primary_include", "sensitivity_include"):
        values = set(table[column])
        if not values <= {"TRUE", "FALSE"}:
            raise ValueError(f"{column} contains non-boolean values: {values}")
        table[column] = table[column].eq("TRUE")
    table["selected_k"] = table["selected_k"].astype(int)

    include_column = (
        "primary_include" if feature_set == "primary" else "sensitivity_include"
    )
    selected = table.loc[table[include_column]].copy()
    if selected.empty:
        raise ValueError(f"No programs selected for feature set {feature_set!r}")
    selected["feature_name"] = (
        "cnmf__"
        + selected["lineage"].str.replace("_", "-", regex=False)
        + "__"
        + selected["program"]
    )
    if selected["feature_name"].duplicated().any():
        raise ValueError("Generated cNMF feature names are not unique")
    selected.insert(0, "column_index", np.arange(len(selected), dtype=int))
    return table, selected


def usage_path_from_manifest(
    cnmf_root: Path, lineage: str
) -> tuple[Path, Path, int]:
    if lineage not in LINEAGE_DIRECTORIES:
        raise ValueError(
            f"No cNMF directory mapping for whitelist lineage {lineage!r}"
        )
    manifest_path = (
        cnmf_root / LINEAGE_DIRECTORIES[lineage] / "run_manifest.json"
    )
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    usage_path = Path(manifest["paths"]["usage_h5ad"])
    selected_k = int(manifest["selected"]["k"])
    if not usage_path.is_file():
        raise FileNotFoundError(usage_path)
    return usage_path, manifest_path, selected_k


def build_matrices(
    master_names: pd.Index,
    cnmf_root: Path,
    whitelist: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, list[dict]]:
    n_cells = len(master_names)
    n_features = len(selected)
    value_rows: list[np.ndarray] = []
    value_cols: list[np.ndarray] = []
    value_data: list[np.ndarray] = []
    applicable_rows: list[np.ndarray] = []
    applicable_cols: list[np.ndarray] = []
    source_lineage = np.full(n_cells, "unmodeled", dtype=object)
    claimed = np.zeros(n_cells, dtype=bool)
    audits: list[dict] = []

    for lineage in whitelist["lineage"].drop_duplicates():
        lineage_table = whitelist.loc[whitelist["lineage"].eq(lineage)]
        lineage_selected = selected.loc[selected["lineage"].eq(lineage)]
        usage_path, manifest_path, selected_k = usage_path_from_manifest(
            cnmf_root, lineage
        )
        expected_k = set(lineage_table["selected_k"])
        if expected_k != {selected_k}:
            raise ValueError(
                f"{lineage}: whitelist K {sorted(expected_k)} != "
                f"manifest K {selected_k}"
            )

        usage = ad.read_h5ad(usage_path, backed="r")
        try:
            usage_columns = sorted(
                [c for c in usage.obs.columns if c.startswith("Usage_")],
                key=lambda value: int(value.split("_", 1)[1]),
            )
            expected_programs = set(lineage_table["program"])
            if set(usage_columns) != expected_programs:
                raise ValueError(
                    f"{lineage}: usage columns and whitelist programs differ; "
                    f"missing={sorted(set(usage_columns) - expected_programs)}, "
                    f"extra={sorted(expected_programs - set(usage_columns))}"
                )
            if len(usage_columns) != selected_k:
                raise ValueError(
                    f"{lineage}: found {len(usage_columns)} usages for K={selected_k}"
                )

            usage_names = pd.Index(usage.obs_names)
            if not usage_names.is_unique:
                raise ValueError(f"{lineage}: usage obs_names are not unique")
            master_rows = master_names.get_indexer(usage_names)
            if np.any(master_rows < 0):
                raise ValueError(
                    f"{lineage}: {int(np.sum(master_rows < 0))} usage cells "
                    "are absent from the master object"
                )
            if claimed[master_rows].any():
                raise ValueError(
                    f"{lineage}: source-cell universe overlaps a prior lineage"
                )
            claimed[master_rows] = True
            source_lineage[master_rows] = lineage

            full_values = usage.obs[usage_columns].to_numpy(dtype=np.float32)
            if not np.isfinite(full_values).all():
                raise ValueError(f"{lineage}: non-finite usage values")
            if np.any(full_values < 0):
                raise ValueError(f"{lineage}: negative usage values")
            row_sums = full_values.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                raise ValueError(
                    f"{lineage}: usage row sums are not normalized to one"
                )

            selected_programs = lineage_selected["program"].tolist()
            if selected_programs:
                values = usage.obs[selected_programs].to_numpy(dtype=np.float32)
                feature_columns = lineage_selected["column_index"].to_numpy(
                    dtype=np.int64
                )
                rows = np.repeat(master_rows.astype(np.int64), len(feature_columns))
                columns = np.tile(feature_columns, len(master_rows))
                data = values.reshape(-1)
                nonzero = data != 0
                value_rows.append(rows[nonzero])
                value_cols.append(columns[nonzero])
                value_data.append(data[nonzero])
                applicable_rows.append(rows)
                applicable_cols.append(columns)

            sample_column = next(
                (
                    column
                    for column in ("sample_id", "sample", "library_id")
                    if column in usage.obs.columns
                ),
                None,
            )
            audits.append(
                {
                    "lineage": lineage,
                    "manifest": str(manifest_path.resolve()),
                    "usage_h5ad": str(usage_path.resolve()),
                    "selected_k": selected_k,
                    "n_cells": int(usage.n_obs),
                    "n_selected_features": int(len(lineage_selected)),
                    "n_primary_features": int(
                        lineage_selected["primary_include"].sum()
                    ),
                    "n_samples": (
                        int(usage.obs[sample_column].nunique())
                        if sample_column
                        else None
                    ),
                    "usage_row_sum_min": float(row_sums.min()),
                    "usage_row_sum_median": float(np.median(row_sums)),
                    "usage_row_sum_max": float(row_sums.max()),
                }
            )
        finally:
            usage.file.close()

    rows = np.concatenate(value_rows)
    columns = np.concatenate(value_cols)
    data = np.concatenate(value_data)
    usage_matrix = sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_cells, n_features), dtype=np.float32
    ).tocsr()
    app_rows = np.concatenate(applicable_rows)
    app_columns = np.concatenate(applicable_cols)
    applicability = sparse.coo_matrix(
        (
            np.ones(len(app_rows), dtype=np.uint8),
            (app_rows, app_columns),
        ),
        shape=(n_cells, n_features),
        dtype=np.uint8,
    ).tocsr()
    return usage_matrix, applicability, source_lineage, audits


def main() -> None:
    args = parse_args()
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.output_h5ad.with_suffix(".manifest.json")
    )
    whitelist_manifest_path = (
        args.whitelist_manifest
        if args.whitelist_manifest is not None
        else args.whitelist.with_suffix(".manifest.json")
    )
    for path in (args.master_h5ad, args.whitelist, whitelist_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_h5ad, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    with whitelist_manifest_path.open(encoding="utf-8") as handle:
        whitelist_manifest = json.load(handle)
    whitelist_hash = sha256(args.whitelist)
    if str(whitelist_manifest.get("output_whitelist_sha256", "")) != whitelist_hash:
        raise ValueError("Whitelist manifest hash does not match the whitelist")
    whitelist_frozen = bool(whitelist_manifest.get("frozen", False))

    whitelist, selected = load_whitelist(args.whitelist, args.feature_set)
    master = ad.read_h5ad(args.master_h5ad)
    if not master.obs_names.is_unique:
        raise ValueError("Master obs_names are not unique")
    for key in (
        "X_cnmf_usage_selected",
        "X_cnmf_usage_applicable",
    ):
        if key in master.obsm:
            raise ValueError(f"Master already contains obsm[{key!r}]")

    usage_matrix, applicability, source_lineage, audits = build_matrices(
        pd.Index(master.obs_names),
        args.cnmf_root,
        whitelist,
        selected,
    )
    master.obsm["X_cnmf_usage_selected"] = usage_matrix
    master.obsm["X_cnmf_usage_applicable"] = applicability
    master.obs["cnmf_usage_source_lineage"] = pd.Categorical(
        source_lineage,
        categories=[*whitelist["lineage"].drop_duplicates(), "unmodeled"],
    )

    feature_metadata = selected[
        [
            "column_index",
            "feature_name",
            "lineage",
            "selected_k",
            "program",
            "proposed_label",
            "category",
            "confidence",
            "primary_include",
            "sensitivity_include",
            "decision",
            "rationale",
            "top_genes",
        ]
    ].copy()
    feature_metadata.index = feature_metadata["feature_name"].astype(str)
    feature_metadata.index.name = "feature_name_index"
    master.uns["cnmf_usage_selected"] = feature_metadata
    generated_at = datetime.now(timezone.utc).isoformat()
    master.uns["cnmf_usage_selected_provenance"] = {
        "generated_at": generated_at,
        "master_h5ad": str(args.master_h5ad.resolve()),
        "whitelist": str(args.whitelist.resolve()),
        "whitelist_sha256": whitelist_hash,
        "whitelist_manifest": str(whitelist_manifest_path.resolve()),
        "whitelist_manifest_sha256": sha256(whitelist_manifest_path),
        "whitelist_frozen": whitelist_frozen,
        "feature_set": args.feature_set,
        "zero_semantics": (
            "Consult obsm['X_cnmf_usage_applicable']; zero with applicability "
            "0 means not applicable, not measured biological zero."
        ),
        "chp_cnmf_status": (
            "included" if "chp" in set(selected["lineage"]) else "not_selected"
        ),
    }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_h5ad.with_name(
        f".{args.output_h5ad.name}.tmp-{os.getpid()}"
    )
    try:
        master.write_h5ad(
            temporary_output,
            compression="gzip",
            compression_opts=4,
        )
        os.replace(temporary_output, args.output_h5ad)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    output_check = ad.read_h5ad(args.output_h5ad, backed="r")
    try:
        if output_check.obsm["X_cnmf_usage_selected"].shape != usage_matrix.shape:
            raise ValueError("Written usage-matrix shape failed verification")
        if output_check.obsm["X_cnmf_usage_applicable"].shape != applicability.shape:
            raise ValueError("Written applicability-matrix shape failed verification")
    finally:
        output_check.file.close()

    manifest = {
        "generated_at": generated_at,
        "status": (
            "frozen_feature_contract"
            if whitelist_frozen
            else "draft_feature_contract"
        ),
        "master_h5ad": str(args.master_h5ad.resolve()),
        "output_h5ad": str(args.output_h5ad.resolve()),
        "whitelist": str(args.whitelist.resolve()),
        "whitelist_sha256": whitelist_hash,
        "whitelist_manifest": str(whitelist_manifest_path.resolve()),
        "whitelist_manifest_sha256": sha256(whitelist_manifest_path),
        "whitelist_frozen": whitelist_frozen,
        "feature_set": args.feature_set,
        "n_cells": int(master.n_obs),
        "n_genes": int(master.n_vars),
        "n_whitelist_programs": int(len(whitelist)),
        "n_selected_features": int(len(selected)),
        "n_primary_features": int(selected["primary_include"].sum()),
        "n_usage_nonzero": int(usage_matrix.nnz),
        "n_applicable": int(applicability.nnz),
        "n_modeled_cells": int(np.sum(source_lineage != "unmodeled")),
        "n_unmodeled_cells": int(np.sum(source_lineage == "unmodeled")),
        "obsm": {
            "usage": "X_cnmf_usage_selected",
            "applicability": "X_cnmf_usage_applicable",
        },
        "uns": {
            "feature_metadata": "cnmf_usage_selected",
            "provenance": "cnmf_usage_selected_provenance",
        },
        "lineage_audits": audits,
        "chp_cnmf_status": (
            "included" if "chp" in set(selected["lineage"]) else "not_selected"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary_manifest.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
