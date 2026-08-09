"""Nearest-neighbor composition tables for CellCharter-annotated AnnData."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve_path


ABUNDANCE_FILE = "compartment_abundance.csv"
NEIGHBOR_FILE = "neighbor_composition.csv"
SUMMARY_FILE = "neighbor_composition_summary.csv"
RANKED_EDGES_FILE = "ranked_neighbor_edges.parquet"
OVERLAP_FILE = "directional_knn_overlap.parquet"
PERMUTATION_FILE = "directional_knn_overlap_permutation.parquet"
RECIPROCAL_CELLTYPE_FILE = "within_compartment_celltype_overlap_permutation.parquet"
WHOLE_SAMPLE_CELLTYPE_FILE = "whole_sample_celltype_overlap_permutation.parquet"
PERMUTATION_PLAN_FILE = "label_permutation_plan.parquet"
WHOLE_SAMPLE_PERMUTATION_PLAN_FILE = "whole_sample_label_permutation_plan.parquet"
MANIFEST_FILE = "manifest.json"

COLOCALIZATION_ANALYSES = ("within_compartment", "whole_sample")


def compartment_abundance(
    obs: pd.DataFrame,
    *,
    condition_col: str,
    sample_col: str,
    compartment_col: str,
    cell_type_col: str,
) -> pd.DataFrame:
    """Count cell types by sample and compartment, including whole-sample rows."""

    renamed = obs[[condition_col, sample_col, compartment_col, cell_type_col]].rename(
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
        detail.groupby(["condition", "sample", "cell_type"], observed=True)["n_cells"]
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
    abundance["cell_type_fraction"] = (
        abundance["n_cells"]
        .div(
            abundance["compartment_total_cells"].where(
                abundance["compartment_total_cells"].ne(0)
            )
        )
        .fillna(0)
    )
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
        return pd.DataFrame(columns=[*columns[:4], "source_n_cells", *columns[4:]])

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
    neighbors["source_n_cells"] = neighbors["source_n_cells"].fillna(0).astype(int)
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

    condition_counts = obs.groupby(sample_col, observed=True)[condition_col].nunique()
    conflicts = condition_counts.loc[condition_counts.gt(1)].index.astype(str).tolist()
    if conflicts:
        raise ValueError(
            "Each sample must map to one condition; conflicts: " + ", ".join(conflicts)
        )


def _fresh_output_paths(
    output_dir: Path,
    *,
    include_overlap: bool = False,
    include_reciprocal_celltypes: bool = False,
    reciprocal_analyses: Sequence[str] | None = None,
) -> dict[str, Path]:
    outputs = {
        "compartment_abundance": output_dir / ABUNDANCE_FILE,
        "neighbor_composition": output_dir / NEIGHBOR_FILE,
        "neighbor_composition_summary": output_dir / SUMMARY_FILE,
        "manifest": output_dir / MANIFEST_FILE,
    }
    if include_overlap:
        outputs.update(
            {
                "label_permutation_plan": output_dir / PERMUTATION_PLAN_FILE,
                "ranked_neighbor_edges": output_dir / RANKED_EDGES_FILE,
                "directional_knn_overlap": output_dir / OVERLAP_FILE,
                "directional_knn_overlap_permutation": (output_dir / PERMUTATION_FILE),
            }
        )
    if include_reciprocal_celltypes:
        analyses = tuple(reciprocal_analyses or ("within_compartment",))
        if "within_compartment" in analyses:
            outputs["within_compartment_celltype_permutation"] = (
                output_dir / RECIPROCAL_CELLTYPE_FILE
            )
        if "whole_sample" in analyses:
            outputs["whole_sample_celltype_permutation"] = (
                output_dir / WHOLE_SAMPLE_CELLTYPE_FILE
            )
            outputs["whole_sample_label_permutation_plan"] = (
                output_dir / WHOLE_SAMPLE_PERMUTATION_PLAN_FILE
            )
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite nncomp outputs: {names}")
    return outputs


def _fresh_overlap_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return overlap-only paths while protecting accepted artifacts."""

    required = [
        output_dir / ABUNDANCE_FILE,
        output_dir / NEIGHBOR_FILE,
        output_dir / SUMMARY_FILE,
        output_dir / MANIFEST_FILE,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Overlap-only mode requires accepted nncomp outputs: " + ", ".join(missing)
        )
    outputs = {
        "label_permutation_plan": output_dir / PERMUTATION_PLAN_FILE,
        "ranked_neighbor_edges": output_dir / RANKED_EDGES_FILE,
        "directional_knn_overlap": output_dir / OVERLAP_FILE,
        "directional_knn_overlap_permutation": output_dir / PERMUTATION_FILE,
    }
    existing = [path.name for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite directional-overlap outputs: " + ", ".join(existing)
        )
    return outputs


def _reciprocal_celltype_paths(
    output_dir: Path,
    analyses: Sequence[str],
) -> dict[str, Path]:
    """Return configured all-cell-type artifacts, including existing paths."""

    required = [
        output_dir / MANIFEST_FILE,
        output_dir / RANKED_EDGES_FILE,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Reciprocal cell-type mode requires existing outputs: " + ", ".join(missing)
        )
    outputs: dict[str, Path] = {}
    if "within_compartment" in analyses:
        outputs["within_compartment_celltype_permutation"] = (
            output_dir / RECIPROCAL_CELLTYPE_FILE
        )
    if "whole_sample" in analyses:
        outputs["whole_sample_celltype_permutation"] = (
            output_dir / WHOLE_SAMPLE_CELLTYPE_FILE
        )
        outputs["whole_sample_label_permutation_plan"] = (
            output_dir / WHOLE_SAMPLE_PERMUTATION_PLAN_FILE
        )
    return outputs


def _fresh_permutation_plan_path(output_dir: Path) -> Path:
    required = [output_dir / MANIFEST_FILE]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Permutation-plan mode requires existing outputs: " + ", ".join(missing)
        )
    output = output_dir / PERMUTATION_PLAN_FILE
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite permutation plan: {output.name}")
    return output


def _validate_manifest_input(manifest: dict[str, Any], input_h5ad: Path) -> None:
    """Require the accepted nncomp manifest to describe the current H5AD."""

    recorded = manifest.get("input_h5ad")
    if not isinstance(recorded, dict):
        raise ValueError("Existing nncomp manifest has no input_h5ad provenance")
    recorded_path = Path(str(recorded.get("path", ""))).expanduser().resolve()
    stat = input_h5ad.stat()
    mismatches = []
    if recorded_path != input_h5ad:
        mismatches.append("path")
    if recorded.get("size_bytes") != stat.st_size:
        mismatches.append("size_bytes")
    if recorded.get("modified_time_ns") != stat.st_mtime_ns:
        mismatches.append("modified_time_ns")
    if mismatches:
        raise ValueError(
            "Current H5AD does not match accepted nncomp manifest: "
            + ", ".join(mismatches)
        )


def _overlap_settings(
    overlap_config: dict[str, Any]
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    k_values = tuple(
        sorted({int(value) for value in overlap_config.get("k_values", [1, 2, 6])})
    )
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("nncomp.overlap.k_values must contain positive integers")
    analyses = tuple(
        overlap_config.get(
            "analyses",
            ["whole_sample", "within_compartment", "compartment_pair"],
        )
    )
    candidate_scopes = []
    if {"whole_sample", "compartment_pair"}.intersection(analyses):
        candidate_scopes.append("sample")
    if "within_compartment" in analyses:
        candidate_scopes.append("sample_compartment")
    if not candidate_scopes:
        raise ValueError(
            "nncomp.overlap.analyses must include whole_sample, "
            "within_compartment, or compartment_pair"
        )
    return k_values, analyses, tuple(candidate_scopes)


def _colocalization_analyses(overlap_config: dict[str, Any]) -> tuple[str, ...]:
    settings = overlap_config.get("reciprocal_cell_types", {})
    analyses = tuple(
        dict.fromkeys(
            str(value) for value in settings.get("analyses", ["within_compartment"])
        )
    )
    invalid = sorted(set(analyses).difference(COLOCALIZATION_ANALYSES))
    if invalid:
        raise ValueError(
            "nncomp.overlap.reciprocal_cell_types.analyses only supports "
            "within_compartment and whole_sample; invalid: " + ", ".join(invalid)
        )
    if not analyses:
        raise ValueError(
            "nncomp.overlap.reciprocal_cell_types.analyses cannot be empty"
        )
    _, overlap_analyses, _ = _overlap_settings(overlap_config)
    missing = sorted(set(analyses).difference(overlap_analyses))
    if missing:
        raise ValueError(
            "All-cell-type analyses require matching nncomp.overlap.analyses: "
            + ", ".join(missing)
        )
    return analyses


def _compute_directional_overlap(
    adata,
    *,
    snn,
    schema: dict[str, Any],
    overlap_config: dict[str, Any],
    permutation_plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute ranked edges, observed overlap, and its fixed-graph null."""

    sample_col = schema["sample_key"]
    condition_col = schema["condition_key"]
    cell_type_col = schema["cell_type_key"]
    compartment_col = schema["spatial_domain_key"]
    spatial_key = schema["spatial_key"]
    k_values, analyses, candidate_scopes = _overlap_settings(overlap_config)
    cell_id_col = schema.get("cell_id_key")
    if cell_id_col not in adata.obs:
        cell_id_col = None

    edge_parts = [
        snn.ranked_neighbor_edges(
            adata,
            sample_col=sample_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
            compartment_col=compartment_col,
            spatial_key=spatial_key,
            cell_id_col=cell_id_col,
            n_neighbors=max(k_values),
            candidate_scope=candidate_scope,
            workers=int(overlap_config.get("workers", 1)),
        )
        for candidate_scope in candidate_scopes
    ]
    ranked_edges = pd.concat(edge_parts, ignore_index=True)
    common = {
        "sample_col": sample_col,
        "condition_col": condition_col,
        "cell_type_col": cell_type_col,
        "compartment_col": compartment_col,
        "k_values": k_values,
        "analyses": analyses,
        "source_cell_types": overlap_config.get("source_cell_types"),
        "neighbor_cell_types": overlap_config.get("neighbor_cell_types"),
    }
    overlap = snn.directional_knn_overlap(
        ranked_edges,
        adata.obs,
        **common,
    )
    permutation = snn.permutation_knn_overlap(
        ranked_edges,
        adata.obs,
        **common,
        n_permutations=int(overlap_config.get("n_permutations", 1000)),
        seed=int(overlap_config.get("seed", 0)),
        permutation_plan=permutation_plan,
    )
    return ranked_edges, overlap, permutation


def _compute_reciprocal_celltype_permutations(
    adata,
    ranked_edges: pd.DataFrame,
    *,
    snn,
    schema: dict[str, Any],
    overlap_config: dict[str, Any],
    analyses: Sequence[str],
    permutation_plan: pd.DataFrame | None = None,
    whole_sample_permutation_plan: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Compute efficient all-cell-type nulls for requested spatial scopes."""

    settings = overlap_config.get("reciprocal_cell_types", {})
    k_values, _, _ = _overlap_settings(overlap_config)
    n_permutations = int(
        settings.get(
            "n_permutations",
            overlap_config.get("n_permutations", 1000),
        )
    )
    seed = int(settings.get("seed", overlap_config.get("seed", 0)))
    workers = int(settings.get("workers", overlap_config.get("workers", -1)))
    progress_every = int(settings.get("progress_every", 0))
    common = {
        "sample_col": schema["sample_key"],
        "condition_col": schema["condition_key"],
        "cell_type_col": schema["cell_type_key"],
        "k_values": k_values,
        "n_permutations": n_permutations,
        "seed": seed,
        "workers": workers,
        "progress_every": progress_every,
    }
    tables: dict[str, pd.DataFrame] = {}
    plans: dict[str, pd.DataFrame] = {}

    if "within_compartment" in analyses:
        if permutation_plan is None:
            permutation_plan = _build_permutation_plan(
                adata,
                snn=snn,
                schema=schema,
                overlap_config=overlap_config,
            )
        tables["within_compartment_celltype_permutation"] = (
            snn.permutation_within_compartment_all_pairs(
                ranked_edges,
                adata.obs,
                compartment_col=schema["spatial_domain_key"],
                permutation_plan=permutation_plan,
                **common,
            )
        )
        plans["label_permutation_plan"] = permutation_plan

    if "whole_sample" in analyses:
        synthetic_compartment = "__whole_sample__"
        whole_obs = adata.obs[
            [
                schema["sample_key"],
                schema["condition_key"],
                schema["cell_type_key"],
            ]
        ].copy()
        whole_obs[synthetic_compartment] = "all"
        whole_edges = ranked_edges.loc[
            ranked_edges["candidate_scope"].astype(str).eq("sample")
        ].copy()
        if whole_edges.empty:
            raise ValueError("No sample-scoped ranked edges are available")
        whole_edges["candidate_scope"] = "sample_compartment"
        if whole_sample_permutation_plan is None:
            plan_config = overlap_config.get("permutation_plan", {})
            whole_sample_permutation_plan = snn.label_permutation_plan(
                whole_obs,
                sample_col=schema["sample_key"],
                cell_type_col=schema["cell_type_key"],
                compartment_col=synthetic_compartment,
                n_permutations=n_permutations,
                seed=int(plan_config.get("seed", seed)),
                endpoint_roles=tuple(
                    plan_config.get("endpoint_roles", ["shared", "source", "target"])
                ),
            )
        whole = snn.permutation_within_compartment_all_pairs(
            whole_edges,
            whole_obs,
            compartment_col=synthetic_compartment,
            permutation_plan=whole_sample_permutation_plan,
            **common,
        )
        whole["analysis"] = "whole_sample"
        whole["source_compartment"] = "all"
        whole["neighbor_compartment"] = "all"
        whole["permutation_strata"] = "sample"
        tables["whole_sample_celltype_permutation"] = whole
        plans["whole_sample_label_permutation_plan"] = whole_sample_permutation_plan

    return tables, plans


def _build_permutation_plan(
    adata,
    *,
    snn,
    schema: dict[str, Any],
    overlap_config: dict[str, Any],
) -> pd.DataFrame:
    settings = overlap_config.get("reciprocal_cell_types", {})
    n_permutations = max(
        int(overlap_config.get("n_permutations", 1000)),
        int(settings.get("n_permutations", 0)),
    )
    plan_config = overlap_config.get("permutation_plan", {})
    return snn.label_permutation_plan(
        adata.obs,
        sample_col=schema["sample_key"],
        cell_type_col=schema["cell_type_key"],
        compartment_col=schema["spatial_domain_key"],
        n_permutations=n_permutations,
        seed=int(plan_config.get("seed", overlap_config.get("seed", 0))),
        endpoint_roles=tuple(
            plan_config.get("endpoint_roles", ["shared", "source", "target"])
        ),
    )


def _write_single_parquet(path: Path, table: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".parquet",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        table.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_overlap_parquets(
    outputs: dict[str, Path],
    permutation_plan: pd.DataFrame,
    ranked_edges: pd.DataFrame,
    overlap: pd.DataFrame,
    permutation: pd.DataFrame,
) -> None:
    """Write all overlap tables via temporary files in the destination."""

    output_dir = next(iter(outputs.values())).parent
    with tempfile.TemporaryDirectory(prefix=".overlap-", dir=output_dir) as tmp:
        temporary = {key: Path(tmp) / path.name for key, path in outputs.items()}
        permutation_plan.to_parquet(
            temporary["label_permutation_plan"],
            index=False,
            compression="zstd",
        )
        ranked_edges.to_parquet(
            temporary["ranked_neighbor_edges"], index=False, compression="zstd"
        )
        overlap.to_parquet(
            temporary["directional_knn_overlap"], index=False, compression="zstd"
        )
        permutation.to_parquet(
            temporary["directional_knn_overlap_permutation"],
            index=False,
            compression="zstd",
        )
        for key, path in outputs.items():
            temporary[key].replace(path)


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
    input_h5ad = resolve_path(config_path, stage["input_h5ad"], root=results_root)
    output_dir = resolve_path(config_path, stage["output_dir"], root=results_root)
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
    overlap_config = stage.get("overlap", {})
    overlap_enabled = bool(overlap_config.get("enabled", False))
    reciprocal_enabled = overlap_enabled and bool(
        overlap_config.get("reciprocal_cell_types", {}).get("enabled", False)
    )
    reciprocal_analyses = (
        _colocalization_analyses(overlap_config) if reciprocal_enabled else ()
    )
    input_h5ad, output_dir = _resolve_stage_paths(config_file, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _fresh_output_paths(
        output_dir,
        include_overlap=overlap_enabled,
        include_reciprocal_celltypes=reciprocal_enabled,
        reciprocal_analyses=reciprocal_analyses,
    )

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

    ranked_edges = None
    overlap = None
    permutation = None
    reciprocal_celltypes: dict[str, pd.DataFrame] = {}
    reciprocal_plans: dict[str, pd.DataFrame] = {}
    permutation_plan = None
    if overlap_enabled:
        permutation_plan = _build_permutation_plan(
            adata,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
        )
        ranked_edges, overlap, permutation = _compute_directional_overlap(
            adata,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
            permutation_plan=permutation_plan,
        )
        _write_overlap_parquets(
            {
                key: outputs[key]
                for key in (
                    "label_permutation_plan",
                    "ranked_neighbor_edges",
                    "directional_knn_overlap",
                    "directional_knn_overlap_permutation",
                )
            },
            permutation_plan,
            ranked_edges,
            overlap,
            permutation,
        )
        if reciprocal_enabled:
            reciprocal_celltypes, reciprocal_plans = (
                _compute_reciprocal_celltype_permutations(
                    adata,
                    ranked_edges,
                    snn=snn,
                    schema=schema,
                    overlap_config=overlap_config,
                    analyses=reciprocal_analyses,
                    permutation_plan=permutation_plan,
                )
            )
            for key, table in reciprocal_celltypes.items():
                _write_single_parquet(outputs[key], table)
            for key, table in reciprocal_plans.items():
                if key != "label_permutation_plan":
                    _write_single_parquet(outputs[key], table)

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
        "directional_overlap": {
            "enabled": overlap_enabled,
            "configuration": overlap_config if overlap_enabled else None,
            "ranked_edge_rows": (
                int(len(ranked_edges)) if ranked_edges is not None else 0
            ),
            "observed_rows": (int(len(overlap)) if overlap is not None else 0),
            "permutation_rows": (
                int(len(permutation)) if permutation is not None else 0
            ),
            "permutation_plan": {
                "rows": (
                    int(len(permutation_plan)) if permutation_plan is not None else 0
                ),
                "endpoint_roles": (
                    sorted(permutation_plan["endpoint_role"].unique())
                    if permutation_plan is not None
                    else []
                ),
                "algorithm": (
                    str(permutation_plan["algorithm"].iloc[0])
                    if permutation_plan is not None and not permutation_plan.empty
                    else None
                ),
            },
            "reciprocal_cell_types": {
                "enabled": reciprocal_enabled,
                "configuration": (
                    overlap_config.get("reciprocal_cell_types")
                    if reciprocal_enabled
                    else None
                ),
                "analyses": list(reciprocal_analyses),
                "rows_by_analysis": {
                    analysis: int(
                        len(reciprocal_celltypes[f"{analysis}_celltype_permutation"])
                    )
                    for analysis in reciprocal_analyses
                    if f"{analysis}_celltype_permutation" in reciprocal_celltypes
                },
            },
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
            "graph_nonzero_entries": int(adata.obsp["spatial_connectivities"].nnz),
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


def run_directional_overlap(config_path: str | Path) -> dict[str, Path]:
    """Backfill overlap tables from an accepted CellCharter H5AD and nncomp run."""

    try:
        import anndata as ad
        import spatial_nncomp as snn
    except ImportError as exc:
        raise ImportError(
            "The overlap-only stage requires anndata and spatial-nncomp."
        ) from exc

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    stage = config["nncomp"]
    schema = config["schema"]
    overlap_config = stage.get("overlap", {})
    if not bool(overlap_config.get("enabled", False)):
        raise ValueError("nncomp.overlap.enabled must be true")

    input_h5ad, output_dir = _resolve_stage_paths(config_file, config)
    outputs = _fresh_overlap_output_paths(output_dir)
    manifest_path = output_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_input(manifest, input_h5ad)

    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        validate_nncomp_obs(
            adata.obs,
            sample_col=schema["sample_key"],
            condition_col=schema["condition_key"],
            cell_type_col=schema["cell_type_key"],
            compartment_col=schema["spatial_domain_key"],
        )
        permutation_plan = _build_permutation_plan(
            adata,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
        )
        ranked_edges, overlap, permutation = _compute_directional_overlap(
            adata,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
            permutation_plan=permutation_plan,
        )
        _write_overlap_parquets(
            outputs,
            permutation_plan,
            ranked_edges,
            overlap,
            permutation,
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    manifest["config_path"] = str(config_file)
    manifest["configuration"] = config
    manifest["directional_overlap"] = {
        "enabled": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": overlap_config,
        "ranked_edge_rows": int(len(ranked_edges)),
        "observed_rows": int(len(overlap)),
        "permutation_rows": int(len(permutation)),
        "permutation_plan": {
            "rows": int(len(permutation_plan)),
            "endpoint_roles": sorted(permutation_plan["endpoint_role"].unique()),
            "algorithm": str(permutation_plan["algorithm"].iloc[0]),
        },
    }
    manifest.setdefault("outputs", {}).update(
        {key: path.name for key, path in outputs.items()}
    )
    manifest.setdefault("software", {}).update(
        {
            "anndata": _package_version("anndata"),
            "spatial_nncomp": _package_version("spatial-nncomp"),
            "spatial_workflow": _package_version("spatial-workflow"),
        }
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".manifest-",
        suffix=".json",
        dir=output_dir,
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest_path)
    return outputs


def run_permutation_plan(config_path: str | Path) -> dict[str, Path]:
    """Backfill a reusable permutation plan without rerunning analysis stages."""

    try:
        import anndata as ad
        import spatial_nncomp as snn
    except ImportError as exc:
        raise ImportError(
            "The permutation-plan stage requires anndata and spatial-nncomp."
        ) from exc

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    schema = config["schema"]
    overlap_config = config["nncomp"].get("overlap", {})
    if not bool(overlap_config.get("enabled", False)):
        raise ValueError("nncomp.overlap.enabled must be true")
    input_h5ad, output_dir = _resolve_stage_paths(config_file, config)
    output_path = _fresh_permutation_plan_path(output_dir)
    manifest_path = output_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_input(manifest, input_h5ad)

    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        permutation_plan = _build_permutation_plan(
            adata,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
        )
        _write_single_parquet(output_path, permutation_plan)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    directional = manifest.setdefault("directional_overlap", {})
    directional["permutation_plan"] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(permutation_plan)),
        "endpoint_roles": sorted(permutation_plan["endpoint_role"].unique()),
        "algorithm": str(permutation_plan["algorithm"].iloc[0]),
    }
    manifest.setdefault("outputs", {})["label_permutation_plan"] = output_path.name
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".manifest-",
        suffix=".json",
        dir=output_dir,
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest_path)
    return {"label_permutation_plan": output_path}


def run_reciprocal_celltype_permutation(
    config_path: str | Path,
) -> dict[str, Path]:
    """Backfill missing all-cell-type nulls from saved ranked edges."""

    try:
        import anndata as ad
        import spatial_nncomp as snn
    except ImportError as exc:
        raise ImportError(
            "The reciprocal cell-type stage requires anndata, numba, and "
            "spatial-nncomp."
        ) from exc

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    schema = config["schema"]
    overlap_config = config["nncomp"].get("overlap", {})
    settings = overlap_config.get("reciprocal_cell_types", {})
    if not bool(settings.get("enabled", False)):
        raise ValueError("nncomp.overlap.reciprocal_cell_types.enabled must be true")
    analyses = _colocalization_analyses(overlap_config)

    input_h5ad, output_dir = _resolve_stage_paths(config_file, config)
    output_paths = _reciprocal_celltype_paths(output_dir, analyses)
    manifest_path = output_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_input(manifest, input_h5ad)
    table_keys = {analysis: f"{analysis}_celltype_permutation" for analysis in analyses}
    pending = tuple(
        analysis
        for analysis, key in table_keys.items()
        if not output_paths[key].exists()
    )
    if not pending:
        return {key: path for key, path in output_paths.items() if path.exists()}

    ranked_edges = pd.read_parquet(
        output_dir / RANKED_EDGES_FILE,
        columns=[
            "candidate_scope",
            "source_index",
            "neighbor_index",
            "neighbor_rank",
        ],
    )
    permutation_plan = None
    plan_path = output_dir / PERMUTATION_PLAN_FILE
    if "within_compartment" in pending and plan_path.exists():
        permutation_plan = pd.read_parquet(plan_path)
    whole_sample_plan = None
    whole_sample_plan_path = output_dir / WHOLE_SAMPLE_PERMUTATION_PLAN_FILE
    if "whole_sample" in pending and whole_sample_plan_path.exists():
        whole_sample_plan = pd.read_parquet(whole_sample_plan_path)

    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        tables, plans = _compute_reciprocal_celltype_permutations(
            adata,
            ranked_edges,
            snn=snn,
            schema=schema,
            overlap_config=overlap_config,
            analyses=pending,
            permutation_plan=permutation_plan,
            whole_sample_permutation_plan=whole_sample_plan,
        )
        for key, table in tables.items():
            _write_single_parquet(output_paths[key], table)
        for key, table in plans.items():
            path = plan_path if key == "label_permutation_plan" else output_paths[key]
            if not path.exists():
                _write_single_parquet(path, table)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    manifest["config_path"] = str(config_file)
    manifest["configuration"] = config
    directional = manifest.setdefault("directional_overlap", {})
    summary = directional.setdefault("reciprocal_cell_types", {})
    summary.update(
        {
            "enabled": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "configuration": settings,
            "analyses": list(analyses),
        }
    )
    rows_by_analysis = summary.setdefault("rows_by_analysis", {})
    for analysis, key in table_keys.items():
        if key in tables:
            rows_by_analysis[analysis] = int(len(tables[key]))
    for key, path in output_paths.items():
        if path.exists():
            manifest.setdefault("outputs", {})[key] = path.name
    if plan_path.exists():
        manifest.setdefault("outputs", {})["label_permutation_plan"] = plan_path.name
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".manifest-",
        suffix=".json",
        dir=output_dir,
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest_path)
    return {key: path for key, path in output_paths.items() if path.exists()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build tidy nearest-neighbor composition outputs."
    )
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--permutation-plan-only",
        action="store_true",
        help="Backfill the reusable label-permutation plan only.",
    )
    modes.add_argument(
        "--overlap-only",
        action="store_true",
        help="Backfill ranked-overlap Parquets without rerunning CellCharter or nncomp.",
    )
    modes.add_argument(
        "--reciprocal-celltypes-only",
        action="store_true",
        help="Backfill missing all-cell-type permutation outputs for configured scopes.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.overlap_only:
        outputs = run_directional_overlap(args.config)
    elif args.permutation_plan_only:
        outputs = run_permutation_plan(args.config)
    elif args.reciprocal_celltypes_only:
        outputs = run_reciprocal_celltype_permutation(args.config)
    else:
        outputs = run_nncomp(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
