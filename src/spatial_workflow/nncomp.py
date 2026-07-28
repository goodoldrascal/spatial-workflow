"""Nearest-neighbor composition tables for CellCharter-annotated AnnData."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve_path


ABUNDANCE_FILE = "compartment_abundance.csv"
NEIGHBOR_FILE = "neighbor_composition.csv"
SUMMARY_FILE = "neighbor_composition_summary.csv"
MANIFEST_FILE = "manifest.json"


def compartment_abundance(
    obs: pd.DataFrame,
    *,
    condition_col: str,
    sample_col: str,
    compartment_col: str,
    cell_type_col: str,
) -> pd.DataFrame:
    """Count cell types by sample and compartment, including whole-sample rows."""

    renamed = obs[
        [condition_col, sample_col, compartment_col, cell_type_col]
    ].rename(
        columns={
            condition_col: "condition",
            sample_col: "sample",
            compartment_col: "compartment",
            cell_type_col: "cell_type",
        }
    )
    for column in renamed:
        renamed[column] = renamed[column].astype("string")

    group_cols = ["condition", "sample", "compartment", "cell_type"]
    observed = (
        renamed.groupby(group_cols, observed=True, dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_meta = renamed[["condition", "sample"]].drop_duplicates()
    domain_cell_types = pd.MultiIndex.from_product(
        [
            sorted(renamed["compartment"].dropna().unique()),
            sorted(renamed["cell_type"].dropna().unique()),
        ],
        names=["compartment", "cell_type"],
    ).to_frame(index=False)
    detail = sample_meta.merge(domain_cell_types, how="cross").merge(
        observed, on=group_cols, how="left", validate="one_to_one"
    )
    detail["n_cells"] = detail["n_cells"].fillna(0).astype(int)
    whole_sample = (
        detail.groupby(["condition", "sample", "cell_type"], observed=True)[
            "n_cells"
        ]
        .sum()
        .reset_index()
        .assign(compartment="all")
    )
    abundance = pd.concat([detail, whole_sample], ignore_index=True)
    abundance["compartment_total_cells"] = abundance.groupby(
        ["condition", "sample", "compartment"], observed=True
    )["n_cells"].transform("sum")
    sample_totals = (
        detail.groupby(["condition", "sample"], observed=True)["n_cells"]
        .sum()
        .rename("sample_total_cells")
        .reset_index()
    )
    abundance = abundance.merge(
        sample_totals, on=["condition", "sample"], how="left", validate="many_to_one"
    )
    abundance["cell_type_fraction"] = abundance["n_cells"].div(
        abundance["compartment_total_cells"].where(
            abundance["compartment_total_cells"].ne(0)
        )
    ).fillna(0)
    abundance["compartment_fraction"] = (
        abundance["compartment_total_cells"] / abundance["sample_total_cells"]
    )
    return abundance.sort_values(group_cols, kind="stable").reset_index(drop=True)


def long_neighbor_composition(
    nn_df: pd.DataFrame,
    abundance: pd.DataFrame,
    *,
    condition_col: str,
    sample_col: str,
    compartment_col: str,
) -> pd.DataFrame:
    """Expand spatial-nncomp matrix columns into one row per directed pair."""

    parts: list[pd.DataFrame] = []
    for _, record in nn_df.iterrows():
        fractions = record["nn_frac"]
        index = fractions.index.astype(str)
        columns = fractions.columns.astype(str)
        counts = record["nn_counts"].reindex(
            index=fractions.index, columns=fractions.columns
        )
        presence = record["nn_presence"].reindex(
            index=fractions.index, columns=fractions.columns
        )
        part = pd.DataFrame(
            {
                "condition": str(record[condition_col]),
                "sample": str(record[sample_col]),
                "compartment": str(record[compartment_col]),
                "source_cell_type": index.repeat(len(columns)),
                "neighbor_cell_type": list(columns) * len(index),
                "neighbor_count": counts.to_numpy(dtype=float).ravel(),
                "neighbor_fraction": fractions.to_numpy(dtype=float).ravel(),
                "neighbor_presence": presence.to_numpy(dtype=float).ravel(),
            }
        )
        parts.append(part)

    columns = [
        "condition",
        "sample",
        "compartment",
        "source_cell_type",
        "neighbor_cell_type",
        "neighbor_count",
        "neighbor_fraction",
        "neighbor_presence",
    ]
    if not parts:
        return pd.DataFrame(
            columns=[*columns[:4], "source_n_cells", *columns[4:]]
        )

    neighbors = (
        pd.concat(parts, ignore_index=True)
        .sort_values(columns[:5], kind="stable")
        .reset_index(drop=True)
    )
    support = abundance[
        ["condition", "sample", "compartment", "cell_type", "n_cells"]
    ].rename(
        columns={
            "cell_type": "source_cell_type",
            "n_cells": "source_n_cells",
        }
    )
    neighbors = neighbors.merge(
        support,
        on=["condition", "sample", "compartment", "source_cell_type"],
        how="left",
        validate="many_to_one",
    )
    neighbors["source_n_cells"] = (
        neighbors["source_n_cells"].fillna(0).astype(int)
    )
    return neighbors[
        [
            "condition",
            "sample",
            "compartment",
            "source_cell_type",
            "source_n_cells",
            "neighbor_cell_type",
            "neighbor_count",
            "neighbor_fraction",
            "neighbor_presence",
        ]
    ]


def summarize_neighbor_composition(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Summarize directed neighborhood measurements across samples."""

    group_cols = [
        "condition",
        "compartment",
        "source_cell_type",
        "neighbor_cell_type",
    ]
    return (
        neighbors.groupby(group_cols, observed=True, dropna=False)
        .agg(
            n_samples=("sample", "nunique"),
            n_supporting_samples=(
                "source_n_cells",
                lambda values: int(values.gt(0).sum()),
            ),
            total_source_cells=("source_n_cells", "sum"),
            mean_neighbor_count=("neighbor_count", "mean"),
            mean_neighbor_fraction=("neighbor_fraction", "mean"),
            sd_neighbor_fraction=("neighbor_fraction", "std"),
            mean_neighbor_presence=("neighbor_presence", "mean"),
            sd_neighbor_presence=("neighbor_presence", "std"),
        )
        .reset_index()
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )


def validate_nncomp_obs(
    obs: pd.DataFrame,
    *,
    sample_col: str,
    condition_col: str,
    cell_type_col: str,
    compartment_col: str,
) -> None:
    """Validate the metadata needed for neighborhood composition."""

    required = [sample_col, condition_col, cell_type_col, compartment_col]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {', '.join(missing)}")

    null_columns = [column for column in required if obs[column].isna().any()]
    if null_columns:
        raise ValueError(
            f"Null labels in required obs columns: {', '.join(null_columns)}"
        )

    condition_counts = obs.groupby(sample_col, observed=True)[
        condition_col
    ].nunique()
    conflicts = condition_counts.loc[condition_counts.gt(1)].index.astype(str).tolist()
    if conflicts:
        raise ValueError(
            "Each sample must map to one condition; conflicts: "
            + ", ".join(conflicts)
        )


def _fresh_output_paths(output_dir: Path) -> dict[str, Path]:
    outputs = {
        "compartment_abundance": output_dir / ABUNDANCE_FILE,
        "neighbor_composition": output_dir / NEIGHBOR_FILE,
        "neighbor_composition_summary": output_dir / SUMMARY_FILE,
        "manifest": output_dir / MANIFEST_FILE,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite nncomp outputs: {names}")
    return outputs



def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _resolve_stage_paths(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, Path]:
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    stage = config["nncomp"]
    input_h5ad = resolve_path(
        config_path, stage["input_h5ad"], root=results_root
    )
    output_dir = resolve_path(
        config_path, stage["output_dir"], root=results_root
    )
    return input_h5ad, output_dir


def run_nncomp(config_path: str | Path) -> dict[str, Path]:
    """Run spatial graph construction and write tidy review tables."""

    try:
        import anndata as ad
        import spatial_nncomp as snn
    except ImportError as exc:
        raise ImportError(
            "The nncomp stage requires anndata and spatial-nncomp. "
            "Install the workflow environment before running it."
        ) from exc

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    stage = config["nncomp"]
    schema = config["schema"]
    input_h5ad, output_dir = _resolve_stage_paths(config_file, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _fresh_output_paths(output_dir)

    sample_col = schema["sample_key"]
    condition_col = schema["condition_key"]
    cell_type_col = schema["cell_type_key"]
    compartment_col = schema["spatial_domain_key"]
    spatial_key = schema["spatial_key"]
    graph = stage.get("graph", {})

    adata = ad.read_h5ad(input_h5ad)
    validate_nncomp_obs(
        adata.obs,
        sample_col=sample_col,
        condition_col=condition_col,
        cell_type_col=cell_type_col,
        compartment_col=compartment_col,
    )
    snn.build_spatial_graph(
        adata,
        method=graph.get("method", "squidpy_knn"),
        spatial_key=spatial_key,
        sample_col=sample_col,
        n_neighs=int(graph.get("n_neighs", 6)),
        radius=float(graph.get("radius", 30.0)),
    )
    nn_df, _, _, _, _ = snn.nn_composition_tables(
        adata,
        cluster_col=cell_type_col,
        condition_col=condition_col,
        sample_col=sample_col,
        comp_col=compartment_col,
    )

    abundance = compartment_abundance(
        adata.obs,
        condition_col=condition_col,
        sample_col=sample_col,
        compartment_col=compartment_col,
        cell_type_col=cell_type_col,
    )
    neighbors = long_neighbor_composition(
        nn_df,
        abundance,
        condition_col=condition_col,
        sample_col=sample_col,
        compartment_col=compartment_col,
    )
    summary = summarize_neighbor_composition(neighbors)

    abundance.to_csv(outputs["compartment_abundance"], index=False)
    neighbors.to_csv(outputs["neighbor_composition"], index=False)
    summary.to_csv(outputs["neighbor_composition_summary"], index=False)

    input_stat = input_h5ad.stat()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_file),
        "configuration": config,
        "input_h5ad": {
            "path": str(input_h5ad),
            "size_bytes": input_stat.st_size,
            "modified_time_ns": input_stat.st_mtime_ns,
        },
        "schema": {
            "sample_key": sample_col,
            "condition_key": condition_col,
            "cell_type_key": cell_type_col,
            "spatial_domain_key": compartment_col,
            "spatial_key": spatial_key,
        },
        "graph": {
            "method": graph.get("method", "squidpy_knn"),
            "n_neighs": int(graph.get("n_neighs", 6)),
            "radius": float(graph.get("radius", 30.0)),
        },
        "data": {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "n_samples": int(adata.obs[sample_col].nunique()),
            "conditions": sorted(
                adata.obs[condition_col].dropna().astype(str).unique()
            ),
            "compartments": sorted(
                adata.obs[compartment_col].dropna().astype(str).unique()
            ),
            "cell_types": sorted(
                adata.obs[cell_type_col].dropna().astype(str).unique()
            ),
            "graph_nonzero_entries": int(
                adata.obsp["spatial_connectivities"].nnz
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "anndata": _package_version("anndata"),
            "spatial_nncomp": _package_version("spatial-nncomp"),
            "spatial_workflow": _package_version("spatial-workflow"),
        },
        "outputs": {
            key: path.name for key, path in outputs.items() if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build tidy nearest-neighbor composition outputs."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run_nncomp(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
