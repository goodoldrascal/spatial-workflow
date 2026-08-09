"""Reproducible cNMF preparation, execution, status, and review helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_config, resolve_path
from .launchers import write_tmux_launcher


@dataclass(frozen=True)
class CnmfLineageSpec:
    """Validated selection and cNMF parameters for one lineage analysis."""

    name: str
    cell_types: tuple[str, ...]
    exclude_cells: tuple[str, ...]
    compartments: tuple[str, ...] | None
    conditions: tuple[str, ...] | None
    sweep_k_values: tuple[int, ...]
    sweep_n_iter: int
    sweep_workers: int
    selected_k: int
    selected_n_iter: int
    selected_workers: int
    numgenes: int
    seed: int
    max_nmf_iter: int
    density_threshold: float
    local_neighborhood_size: float
    min_counts_per_cell: int
    min_cells_per_gene: int


@dataclass(frozen=True)
class CnmfWorkflowPaths:
    """Resolved paths for one lineage's input, sweep, and selected-K run."""

    source_h5ad: Path
    output_root: Path
    lineage_root: Path
    counts_h5ad: Path
    obs_csv: Path
    var_csv: Path
    selection_summary_csv: Path
    export_manifest: Path
    sweep_output_dir: Path
    sweep_name: str
    selected_output_dir: Path
    selected_name: str
    usage_h5ad: Path
    run_manifest: Path

    @property
    def sweep_run_dir(self) -> Path:
        return self.sweep_output_dir / self.sweep_name

    @property
    def selected_run_dir(self) -> Path:
        return self.selected_output_dir / self.selected_name


def _as_exact_values(value: Any, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        value = [value]
    elif isinstance(value, (int, float)):
        value = [value]
    normalized = tuple(dict.fromkeys(str(item) for item in value))
    if not normalized:
        raise ValueError(f"{name} must be 'all' or contain at least one value")
    if any(item.lower() == "all" for item in normalized):
        if len(normalized) == 1:
            return None
        raise ValueError(f"{name} cannot mix 'all' with explicit values")
    return normalized


def _positive_ints(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    normalized = tuple(sorted(dict.fromkeys(int(value) for value in values)))
    if not normalized or normalized[0] < 2:
        raise ValueError(f"{name} must contain integers greater than or equal to 2")
    return normalized


def load_lineage_spec(
    config: Mapping[str, Any],
    lineage: str,
    *,
    analysis_name: str | None = None,
    cell_types: Sequence[str] | None = None,
    compartments: Sequence[str] | str | None = None,
    conditions: Sequence[str] | str | None = None,
    selected_k: int | None = None,
    workers: int | None = None,
) -> CnmfLineageSpec:
    """Load one exact-list lineage specification with optional CLI overrides."""

    settings = config.get("cnmf")
    if not isinstance(settings, Mapping):
        raise ValueError("Configuration section 'cnmf' must be a mapping")
    lineages = settings.get("lineages")
    if not isinstance(lineages, Mapping) or lineage not in lineages:
        available = list(lineages) if isinstance(lineages, Mapping) else []
        raise KeyError(f"Unknown cNMF lineage {lineage!r}. Available: {available}")
    entry = lineages[lineage]
    if not isinstance(entry, Mapping):
        raise ValueError(f"cnmf.lineages.{lineage} must be a mapping")
    sweep = entry.get("sweep", {})
    selected = entry.get("selected", {})
    if not isinstance(sweep, Mapping) or not isinstance(selected, Mapping):
        raise ValueError(f"{lineage} sweep and selected settings must be mappings")
    global_preparation = settings.get("preparation", {})
    lineage_preparation = entry.get("preparation", {})
    if not isinstance(global_preparation, Mapping) or not isinstance(
        lineage_preparation, Mapping
    ):
        raise ValueError(
            f"cnmf preparation settings for {lineage} must be mappings"
        )

    configured_cell_types = _as_exact_values(
        entry.get("cell_types"), name="cell_types"
    )
    exact_cell_types = _as_exact_values(
        configured_cell_types if cell_types is None else cell_types,
        name="cell_types",
    )
    if exact_cell_types is None:
        raise ValueError(
            "A lineage must list every included cell type explicitly; "
            "cell_types='all' is not allowed"
        )
    compartment_values = (
        entry.get("compartments", "all")
        if compartments is None
        else compartments
    )
    condition_values = (
        entry.get("conditions", "all") if conditions is None else conditions
    )
    exact_compartments = _as_exact_values(
        compartment_values, name="compartments"
    )
    exact_conditions = _as_exact_values(condition_values, name="conditions")
    configured_compartments = _as_exact_values(
        entry.get("compartments", "all"), name="compartments"
    )
    configured_conditions = _as_exact_values(
        entry.get("conditions", "all"), name="conditions"
    )
    selection_changed = (
        set(exact_cell_types) != set(configured_cell_types)
        or (None if exact_compartments is None else set(exact_compartments))
        != (
            None
            if configured_compartments is None
            else set(configured_compartments)
        )
        or (None if exact_conditions is None else set(exact_conditions))
        != (None if configured_conditions is None else set(configured_conditions))
    )
    if selection_changed and analysis_name is None:
        raise ValueError(
            "A custom cell-type, compartment, or condition selection requires "
            "a distinct analysis_name"
        )
    sweep_workers = int(sweep.get("workers", 1) if workers is None else workers)
    selected_workers = int(
        selected.get("workers", 1) if workers is None else workers
    )
    chosen_k = selected.get("k") if selected_k is None else selected_k
    if chosen_k is None:
        raise ValueError(f"cnmf.lineages.{lineage}.selected.k is required")
    values = CnmfLineageSpec(
        name=str(analysis_name or lineage),
        cell_types=exact_cell_types,
        exclude_cells=tuple(
            dict.fromkeys(str(value) for value in entry.get("exclude_cells", []))
        ),
        compartments=exact_compartments,
        conditions=exact_conditions,
        sweep_k_values=_positive_ints(sweep.get("k_values", []), name="k_values"),
        sweep_n_iter=int(sweep.get("n_iter", 15)),
        sweep_workers=sweep_workers,
        selected_k=int(chosen_k),
        selected_n_iter=int(selected.get("n_iter", 100)),
        selected_workers=selected_workers,
        numgenes=int(entry.get("numgenes", sweep.get("numgenes", 2000))),
        seed=int(entry.get("seed", sweep.get("seed", 14))),
        max_nmf_iter=int(
            entry.get("max_nmf_iter", sweep.get("max_nmf_iter", 1000))
        ),
        density_threshold=float(selected.get("density_threshold", 0.1)),
        local_neighborhood_size=float(
            selected.get("local_neighborhood_size", 0.3)
        ),
        min_counts_per_cell=int(
            lineage_preparation.get(
                "min_counts_per_cell",
                global_preparation.get("min_counts_per_cell", 1),
            )
        ),
        min_cells_per_gene=int(
            lineage_preparation.get(
                "min_cells_per_gene",
                global_preparation.get(
                    "min_cells_per_gene", settings.get("min_cells_per_gene", 1)
                ),
            )
        ),
    )
    for field_name in (
        "sweep_n_iter",
        "sweep_workers",
        "selected_n_iter",
        "selected_workers",
        "numgenes",
        "max_nmf_iter",
    ):
        if getattr(values, field_name) < 1:
            raise ValueError(f"{field_name} must be at least 1")
    if values.selected_k < 2:
        raise ValueError("selected_k must be at least 2")
    if values.min_counts_per_cell < 0 or values.min_cells_per_gene < 0:
        raise ValueError("Invalid cNMF count filters")
    if not 0 < values.density_threshold <= 2:
        raise ValueError("density_threshold must be greater than 0 and at most 2")
    if not 0 < values.local_neighborhood_size <= 1:
        raise ValueError("local_neighborhood_size must be in (0, 1]")
    return values


def resolve_cnmf_paths(
    config_path: str | Path,
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
) -> CnmfWorkflowPaths:
    """Resolve source and result paths under the workflow results root."""

    settings = config["cnmf"]
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    source_h5ad = resolve_path(
        config_path,
        settings.get("input_h5ad", config["nncomp"]["input_h5ad"]),
        root=results_root,
    )
    output_root = resolve_path(
        config_path,
        settings.get("output_dir", "05_cnmf"),
        root=results_root,
    )
    lineage_root = output_root / spec.name
    input_dir = lineage_root / "input"
    sweep_output_dir = lineage_root / "sweep"
    selected_output_dir = lineage_root / (
        f"selected_k{spec.selected_k}_i{spec.selected_n_iter}"
    )
    sweep_name = f"{spec.name}_sweep"
    selected_name = f"{spec.name}_k{spec.selected_k}_i{spec.selected_n_iter}"
    return CnmfWorkflowPaths(
        source_h5ad=source_h5ad,
        output_root=output_root,
        lineage_root=lineage_root,
        counts_h5ad=input_dir / f"{spec.name}_counts.h5ad",
        obs_csv=input_dir / f"{spec.name}_obs.csv.gz",
        var_csv=input_dir / f"{spec.name}_var.csv.gz",
        selection_summary_csv=input_dir / f"{spec.name}_selection_summary.csv",
        export_manifest=input_dir / "manifest.json",
        sweep_output_dir=sweep_output_dir,
        sweep_name=sweep_name,
        selected_output_dir=selected_output_dir,
        selected_name=selected_name,
        usage_h5ad=selected_output_dir / f"{spec.name}_cnmf_usage.h5ad",
        run_manifest=lineage_root / "run_manifest.json",
    )


def load_cnmf_context(
    config_path: str | Path,
    lineage: str,
    *,
    cell_type_key: str | None = None,
    **overrides: Any,
) -> tuple[dict[str, Any], CnmfLineageSpec, CnmfWorkflowPaths]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    if cell_type_key is not None:
        resolved_key = str(cell_type_key).strip()
        if not resolved_key:
            raise ValueError("cell_type_key must be a non-empty column name")
        configured_key = _selection_keys(config)["cell_type_key"]
        if resolved_key != configured_key and not overrides.get("analysis_name"):
            raise ValueError(
                "A cell_type_key override requires a distinct analysis_name"
            )
        config["cnmf"]["cell_type_key"] = resolved_key
    spec = load_lineage_spec(config, lineage, **overrides)
    paths = resolve_cnmf_paths(config_path, config, spec)
    return config, spec, paths


def selection_mask(
    obs: pd.DataFrame,
    *,
    cell_type_key: str,
    cell_types: Sequence[str],
    compartment_key: str,
    compartments: Sequence[str] | None,
    condition_key: str,
    conditions: Sequence[str] | None,
    exclude_cells: Sequence[str] = (),
) -> np.ndarray:
    """Select exact cell-type values and all-or-explicit domains/conditions."""

    required = [cell_type_key, compartment_key, condition_key]
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError("Missing cNMF selection columns: " + ", ".join(missing))
    available_cell_types = set(obs[cell_type_key].dropna().astype(str))
    missing_cell_types = sorted(set(map(str, cell_types)) - available_cell_types)
    if missing_cell_types:
        raise ValueError("Requested cell types not found: " + ", ".join(missing_cell_types))
    mask = obs[cell_type_key].astype(str).isin(cell_types).to_numpy()
    if compartments is not None:
        available = set(obs[compartment_key].dropna().astype(str))
        missing_values = sorted(set(map(str, compartments)) - available)
        if missing_values:
            raise ValueError(
                "Requested compartments not found: " + ", ".join(missing_values)
            )
        mask &= obs[compartment_key].astype(str).isin(compartments).to_numpy()
    if conditions is not None:
        available = set(obs[condition_key].dropna().astype(str))
        missing_values = sorted(set(map(str, conditions)) - available)
        if missing_values:
            raise ValueError(
                "Requested conditions not found: " + ", ".join(missing_values)
            )
        mask &= obs[condition_key].astype(str).isin(conditions).to_numpy()
    if exclude_cells:
        missing_cells = sorted(set(map(str, exclude_cells)).difference(obs.index))
        if missing_cells:
            raise ValueError(
                "Configured excluded cell IDs not found: " + ", ".join(missing_cells)
            )
        mask &= ~obs.index.isin(exclude_cells)
    return mask


def _selection_keys(config: Mapping[str, Any]) -> dict[str, str]:
    settings = config["cnmf"]
    schema = config["schema"]
    return {
        "cell_type_key": str(
            settings.get("cell_type_key", schema["broad_cell_type_key"])
        ),
        "compartment_key": str(
            settings.get("compartment_key", schema["spatial_domain_key"])
        ),
        "condition_key": str(
            settings.get("condition_key", schema["condition_key"])
        ),
        "sample_key": str(settings.get("sample_key", schema["sample_key"])),
        "spatial_key": str(settings.get("spatial_key", schema["spatial_key"])),
        "counts_layer": str(
            settings.get("counts_layer", schema.get("counts_layer", "counts"))
        ),
    }


def cnmf_selection_keys(config: Mapping[str, Any]) -> dict[str, str]:
    """Return resolved metadata and count-layer keys for cNMF review code."""

    return _selection_keys(config)


def preview_lineage_selection(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    source_h5ad: str | Path,
) -> dict[str, pd.DataFrame]:
    """Return backed, read-only selection summaries before exporting counts."""

    import anndata as ad

    keys = _selection_keys(config)
    adata = ad.read_h5ad(source_h5ad, backed="r")
    mask = selection_mask(
        adata.obs,
        cell_type_key=keys["cell_type_key"],
        cell_types=spec.cell_types,
        compartment_key=keys["compartment_key"],
        compartments=spec.compartments,
        condition_key=keys["condition_key"],
        conditions=spec.conditions,
        exclude_cells=spec.exclude_cells,
    )
    selected = adata.obs.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No cells matched cNMF lineage {spec.name!r}")
    overview = pd.Series(
        {
            "lineage": spec.name,
            "cells": len(selected),
            "genes_before_filter": adata.n_vars,
            "cell_type_key": keys["cell_type_key"],
            "cell_types": ", ".join(spec.cell_types),
            "compartments": (
                "all" if spec.compartments is None else ", ".join(spec.compartments)
            ),
            "conditions": (
                "all" if spec.conditions is None else ", ".join(spec.conditions)
            ),
            "samples": selected[keys["sample_key"]].astype(str).nunique(),
        },
        name="selection",
    ).to_frame()
    by_sample = (
        selected.groupby(
            [keys["condition_key"], keys["sample_key"]],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    by_cell_type = (
        selected.groupby(keys["cell_type_key"], observed=True, dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    by_compartment = (
        selected.groupby(
            [keys["compartment_key"], keys["condition_key"]],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    return {
        "overview": overview,
        "by_sample": by_sample,
        "by_cell_type": by_cell_type,
        "by_compartment": by_compartment,
    }


def _hash_values(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".json",
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv_atomic(path: Path, table: pd.DataFrame, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes) or ".csv"
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=suffix,
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        compression = "gzip" if path.suffix == ".gz" else None
        table.to_csv(temporary, index=index, compression=compression)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_lineage_counts(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a compact count-scale H5AD and exact selection provenance."""

    import anndata as ad

    keys = _selection_keys(config)
    min_counts_per_cell = spec.min_counts_per_cell
    min_cells_per_gene = spec.min_cells_per_gene
    expected_selection = {
        "analysis_name": spec.name,
        "source_h5ad": str(paths.source_h5ad.resolve()),
        "counts_layer": keys["counts_layer"],
        "cell_type_key": keys["cell_type_key"],
        "cell_types": list(spec.cell_types),
        "exclude_cells": list(spec.exclude_cells),
        "compartment_key": keys["compartment_key"],
        "compartments": (
            "all" if spec.compartments is None else list(spec.compartments)
        ),
        "condition_key": keys["condition_key"],
        "conditions": "all" if spec.conditions is None else list(spec.conditions),
        "min_counts_per_cell": min_counts_per_cell,
        "min_cells_per_gene": min_cells_per_gene,
    }
    targets = [
        paths.counts_h5ad,
        paths.obs_csv,
        paths.var_csv,
        paths.selection_summary_csv,
        paths.export_manifest,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        if len(existing) == len(targets):
            with paths.export_manifest.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            source_stat = paths.source_h5ad.stat()
            source = manifest.get("source", {})
            source_matches = (
                source.get("path") == str(paths.source_h5ad.resolve())
                and source.get("size_bytes") == int(source_stat.st_size)
                and source.get("mtime_ns") == int(source_stat.st_mtime_ns)
            )
            if not source_matches or manifest.get("selection") != expected_selection:
                raise FileExistsError(
                    "Existing cNMF export does not match the current source and "
                    "selection; use a distinct analysis name or overwrite explicitly"
                )
            return manifest
        raise FileExistsError(
            "Partial cNMF export exists; choose a new analysis name or explicitly "
            "overwrite the generated export: "
            + ", ".join(path.name for path in existing)
        )

    source = ad.read_h5ad(paths.source_h5ad, backed="r")
    if keys["counts_layer"] not in source.layers:
        raise KeyError(
            f"Counts layer {keys['counts_layer']!r} is missing from {paths.source_h5ad}"
        )
    mask = selection_mask(
        source.obs,
        cell_type_key=keys["cell_type_key"],
        cell_types=spec.cell_types,
        compartment_key=keys["compartment_key"],
        compartments=spec.compartments,
        condition_key=keys["condition_key"],
        conditions=spec.conditions,
        exclude_cells=spec.exclude_cells,
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError(f"No cells matched cNMF lineage {spec.name!r}")
    counts = source.layers[keys["counts_layer"]][indices, :]
    counts = counts.tocsr() if sp.issparse(counts) else np.asarray(counts)
    values = counts.data if sp.issparse(counts) else counts.ravel()
    if values.size and (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.array_equal(values, np.rint(values))
    ):
        raise ValueError("Selected counts contain negative, non-finite, or non-integer values")

    if min_counts_per_cell < 0 or min_cells_per_gene < 0:
        raise ValueError("Invalid cNMF count filters")
    cell_sums = np.asarray(counts.sum(axis=1)).ravel()
    keep_cells = cell_sums >= min_counts_per_cell
    dropped_cells = int((~keep_cells).sum())
    if dropped_cells:
        indices = indices[keep_cells]
        counts = counts[keep_cells, :]
    gene_support = (
        np.asarray((counts > 0).sum(axis=0)).ravel()
        if sp.issparse(counts)
        else np.count_nonzero(counts > 0, axis=0)
    )
    keep_genes = gene_support >= min_cells_per_gene
    dropped_genes = int((~keep_genes).sum())
    counts = counts[:, keep_genes]
    if counts.shape[0] < max(spec.sweep_k_values + (spec.selected_k,)):
        raise ValueError("Fewer selected cells than the largest requested K")
    if counts.shape[1] < spec.numgenes:
        raise ValueError(
            f"Only {counts.shape[1]} genes remain, fewer than numgenes={spec.numgenes}"
        )

    obs = source.obs.iloc[indices].copy()
    var = source.var.iloc[np.flatnonzero(keep_genes)].copy()
    if not obs.index.is_unique:
        raise ValueError("Selected cNMF cell IDs must be unique")
    if not var.index.is_unique:
        raise ValueError("Selected cNMF gene IDs must be unique")
    exported = ad.AnnData(X=counts, obs=obs, var=var)
    if keys["spatial_key"] in source.obsm:
        exported.obsm[keys["spatial_key"]] = np.asarray(
            source.obsm[keys["spatial_key"]][indices]
        )
    exported.uns["cnmf_preparation"] = expected_selection

    paths.counts_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary_h5ad = paths.counts_h5ad.with_name(
        f".{paths.counts_h5ad.stem}-{os.getpid()}.h5ad"
    )
    try:
        exported.write_h5ad(temporary_h5ad, compression="gzip")
        temporary_h5ad.replace(paths.counts_h5ad)
    finally:
        if temporary_h5ad.exists():
            temporary_h5ad.unlink()
    _write_csv_atomic(paths.obs_csv, exported.obs, index=True)
    _write_csv_atomic(paths.var_csv, exported.var, index=True)
    summary = (
        exported.obs.groupby(
            [
                keys["condition_key"],
                keys["sample_key"],
                keys["compartment_key"],
                keys["cell_type_key"],
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    _write_csv_atomic(paths.selection_summary_csv, summary, index=False)
    source_stat = paths.source_h5ad.stat()
    manifest = {
        "stage": "cnmf_input_export",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_name": spec.name,
        "source": {
            "path": str(paths.source_h5ad.resolve()),
            "size_bytes": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
            "shape": [int(source.n_obs), int(source.n_vars)],
        },
        "selection": exported.uns["cnmf_preparation"],
        "output": {
            "counts_h5ad": str(paths.counts_h5ad.resolve()),
            "obs_csv": str(paths.obs_csv.resolve()),
            "var_csv": str(paths.var_csv.resolve()),
            "selection_summary_csv": str(paths.selection_summary_csv.resolve()),
            "shape": [int(exported.n_obs), int(exported.n_vars)],
            "dropped_low_count_cells": dropped_cells,
            "dropped_low_support_genes": dropped_genes,
            "cell_id_sha256": _hash_values(exported.obs_names),
            "gene_sha256": _hash_values(exported.var_names),
        },
    }
    _write_json_atomic(paths.export_manifest, manifest)
    return manifest


def _cnmf_binary(config: Mapping[str, Any]) -> Path:
    configured = config.get("cnmf", {}).get("binary") or config.get(
        "runtime", {}
    ).get("cnmf_bin")
    if configured:
        binary = Path(str(configured)).expanduser()
        if binary.exists():
            return binary.resolve()
        raise FileNotFoundError(f"Configured cNMF binary does not exist: {binary}")
    python_bin = Path(
        str(config.get("runtime", {}).get("python_bin", shutil.which("python3")))
    ).expanduser()
    sibling = python_bin.parent / "cnmf"
    if sibling.exists():
        return sibling.resolve()
    found = shutil.which("cnmf")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError("Could not locate the cNMF CLI")


def _base_prepare_command(
    binary: Path,
    *,
    output_dir: Path,
    name: str,
    counts_h5ad: Path,
    k_values: Sequence[int],
    n_iter: int,
    spec: CnmfLineageSpec,
) -> list[str]:
    return [
        str(binary),
        "prepare",
        "--output-dir",
        str(output_dir),
        "--name",
        name,
        "--counts",
        str(counts_h5ad),
        "--components",
        *map(str, k_values),
        "--n-iter",
        str(n_iter),
        "--seed",
        str(spec.seed),
        "--numgenes",
        str(spec.numgenes),
        "--max-nmf-iter",
        str(spec.max_nmf_iter),
    ]


def cnmf_stage_commands(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
) -> dict[str, list[str]]:
    """Return copy/pasteable single-worker commands for every stage."""

    binary = _cnmf_binary(config)
    return {
        "sweep_prepare": _base_prepare_command(
            binary,
            output_dir=paths.sweep_output_dir,
            name=paths.sweep_name,
            counts_h5ad=paths.counts_h5ad,
            k_values=spec.sweep_k_values,
            n_iter=spec.sweep_n_iter,
            spec=spec,
        ),
        "sweep_factorize_worker_0": [
            str(binary),
            "factorize",
            "--output-dir",
            str(paths.sweep_output_dir),
            "--name",
            paths.sweep_name,
            "--worker-index",
            "0",
            "--total-workers",
            str(spec.sweep_workers),
            "--skip-completed-runs",
        ],
        "sweep_combine": [
            str(binary),
            "combine",
            "--output-dir",
            str(paths.sweep_output_dir),
            "--name",
            paths.sweep_name,
        ],
        "sweep_kselect": [
            str(binary),
            "k_selection_plot",
            "--output-dir",
            str(paths.sweep_output_dir),
            "--name",
            paths.sweep_name,
        ],
        "selected_prepare": _base_prepare_command(
            binary,
            output_dir=paths.selected_output_dir,
            name=paths.selected_name,
            counts_h5ad=paths.counts_h5ad,
            k_values=[spec.selected_k],
            n_iter=spec.selected_n_iter,
            spec=spec,
        ),
        "selected_factorize_worker_0": [
            str(binary),
            "factorize",
            "--output-dir",
            str(paths.selected_output_dir),
            "--name",
            paths.selected_name,
            "--worker-index",
            "0",
            "--total-workers",
            str(spec.selected_workers),
            "--skip-completed-runs",
        ],
        "selected_combine": [
            str(binary),
            "combine",
            "--output-dir",
            str(paths.selected_output_dir),
            "--name",
            paths.selected_name,
        ],
        "selected_consensus": [
            str(binary),
            "consensus",
            "--output-dir",
            str(paths.selected_output_dir),
            "--name",
            paths.selected_name,
            "--components",
            str(spec.selected_k),
            "--local-density-threshold",
            str(spec.density_threshold),
            "--local-neighborhood-size",
            str(spec.local_neighborhood_size),
            "--show-clustering",
        ],
    }


def write_cnmf_tmux_scripts(
    config_path: str | Path,
    lineage: str,
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    mode: str,
) -> dict[str, str | Path]:
    """Write a reproducible cNMF CLI job and detached-tmux launcher."""

    if mode not in {"sweep", "selected"}:
        raise ValueError("mode must be 'sweep' or 'selected'")
    config_file = Path(config_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    workflow_script = repo_root / "scripts" / "run_cnmf_workflow.py"
    runtime = config.get("runtime", {})
    python_bin = runtime.get("python_bin", "python3")
    if not isinstance(python_bin, str) or not python_bin.strip():
        raise ValueError("runtime.python_bin must be a non-empty string")

    command = [
        python_bin,
        str(workflow_script),
        "--config",
        str(config_file),
        "--lineage",
        str(lineage),
        "--mode",
        mode,
        "--cell-type-key",
        _selection_keys(config)["cell_type_key"],
    ]
    if spec.name != lineage:
        command.extend(["--analysis-name", spec.name])
    for cell_type in spec.cell_types:
        command.extend(["--cell-type", cell_type])
    if spec.compartments is None:
        command.extend(["--compartment", "all"])
    else:
        for compartment in spec.compartments:
            command.extend(["--compartment", compartment])
    if spec.conditions is None:
        command.extend(["--condition", "all"])
    else:
        for condition in spec.conditions:
            command.extend(["--condition", condition])
    command.extend(["--selected-k", str(spec.selected_k)])
    workers = spec.sweep_workers if mode == "sweep" else spec.selected_workers
    command.extend(["--workers", str(workers)])

    launcher_dir = paths.lineage_root / "launchers"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    job_script = launcher_dir / f"run_{mode}.sh"
    job_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + shlex.join(command)
        + "\n",
        encoding="utf-8",
    )
    job_script.chmod(job_script.stat().st_mode | 0o111)

    session_fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name).strip("_.-")
    session_name = f"cnmf_{session_fragment or 'analysis'}_{mode}"
    log_path = launcher_dir / f"{mode}.tmux.log"
    launcher_script = write_tmux_launcher(
        job_script,
        launcher_dir / f"launch_{mode}_tmux.sh",
        session_name,
        log_path,
    )
    return {
        "mode": mode,
        "job_script": job_script,
        "launcher_script": launcher_script,
        "log_path": log_path,
        "session_name": session_name,
        "launch_command": shlex.join(["bash", str(launcher_script)]),
        "attach_command": shlex.join(
            ["tmux", "attach-session", "-t", session_name]
        ),
    }


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    dry_run: bool,
    env: Mapping[str, str] | None = None,
) -> None:
    print("+", shlex.join(map(str, command)), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + shlex.join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(
            list(map(str, command)),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(env) if env is not None else None,
        )


def _cnmf_internal_paths(output_dir: Path, name: str) -> dict[str, Path]:
    run_dir = output_dir / name
    temp_dir = run_dir / "cnmf_tmp"
    return {
        "run_dir": run_dir,
        "temp_dir": temp_dir,
        "params": temp_dir / f"{name}.nmf_params.df.npz",
        "normalized_counts": temp_dir / f"{name}.norm_counts.h5ad",
        "k_stats": run_dir / f"{name}.k_selection_stats.df.npz",
        "k_plot": run_dir / f"{name}.k_selection.png",
    }


def _completed_iterations(run_dir: Path, name: str) -> int:
    return len(list((run_dir / "cnmf_tmp").glob(f"{name}.spectra.k_*.iter_*.df.npz")))


def _run_prepare(
    command: Sequence[str],
    *,
    output_dir: Path,
    name: str,
    log_path: Path,
    dry_run: bool,
) -> None:
    internal = _cnmf_internal_paths(output_dir, name)
    if internal["params"].exists() and internal["normalized_counts"].exists():
        print(f"Reusing completed cNMF prepare stage: {internal['run_dir']}")
        return
    _run_logged(command, log_path=log_path, dry_run=dry_run)


def _run_factorize_workers(
    base_command: Sequence[str],
    *,
    workers: int,
    output_dir: Path,
    name: str,
    expected_iterations: int,
    dry_run: bool,
) -> None:
    run_dir = output_dir / name
    completed = _completed_iterations(run_dir, name)
    if completed >= expected_iterations:
        print(
            f"Reusing completed factorization: {completed}/{expected_iterations} spectra"
        )
        return
    commands = []
    for worker in range(workers):
        command = list(map(str, base_command))
        index = command.index("--worker-index") + 1
        command[index] = str(worker)
        commands.append(command)
        print("+", shlex.join(command), flush=True)
    if dry_run:
        return
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"
    processes: list[tuple[subprocess.Popen[Any], Any, Path]] = []
    try:
        for worker, command in enumerate(commands):
            log_path = log_dir / f"factorize.worker_{worker:02d}.log"
            handle = log_path.open("w", encoding="utf-8")
            handle.write("+ " + shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append((process, handle, log_path))
        failures = []
        for process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code:
                failures.append((return_code, log_path))
        if failures:
            details = ", ".join(f"rc={rc}: {path}" for rc, path in failures)
            raise RuntimeError("cNMF factorize workers failed: " + details)
    finally:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()
    completed = _completed_iterations(run_dir, name)
    if completed != expected_iterations:
        raise RuntimeError(
            f"Expected {expected_iterations} factorization spectra, found {completed}"
        )


def _run_combine(
    command: Sequence[str],
    *,
    output_dir: Path,
    name: str,
    k_values: Sequence[int],
    log_path: Path,
    dry_run: bool,
) -> None:
    temp_dir = output_dir / name / "cnmf_tmp"
    merged = [temp_dir / f"{name}.spectra.k_{k}.merged.df.npz" for k in k_values]
    if all(path.exists() for path in merged):
        print(f"Reusing combined spectra for {name}")
        return
    _run_logged(command, log_path=log_path, dry_run=dry_run)


def _ensure_collection_manifest(
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    collection: str,
    dry_run: bool,
) -> None:
    """Validate the cell universe and parameters before resuming a collection."""

    if dry_run:
        return
    if collection == "sweep":
        output_dir = paths.sweep_output_dir
        name = paths.sweep_name
        settings = {
            "k_values": list(spec.sweep_k_values),
            "n_iter": spec.sweep_n_iter,
            "workers": spec.sweep_workers,
        }
    elif collection == "selected":
        output_dir = paths.selected_output_dir
        name = paths.selected_name
        settings = {
            "k": spec.selected_k,
            "n_iter": spec.selected_n_iter,
            "workers": spec.selected_workers,
            "density_threshold": spec.density_threshold,
            "local_neighborhood_size": spec.local_neighborhood_size,
        }
    else:
        raise ValueError("collection must be 'sweep' or 'selected'")
    counts_stat = paths.counts_h5ad.stat()
    payload = {
        "stage": "cnmf_collection",
        "collection": collection,
        "name": name,
        "selection": {
            "cell_types": list(spec.cell_types),
            "exclude_cells": list(spec.exclude_cells),
            "compartments": (
                "all" if spec.compartments is None else list(spec.compartments)
            ),
            "conditions": (
                "all" if spec.conditions is None else list(spec.conditions)
            ),
        },
        "counts": {
            "path": str(paths.counts_h5ad.resolve()),
            "size_bytes": int(counts_stat.st_size),
            "mtime_ns": int(counts_stat.st_mtime_ns),
        },
        "settings": {
            **settings,
            "numgenes": spec.numgenes,
            "seed": spec.seed,
            "max_nmf_iter": spec.max_nmf_iter,
        },
    }
    manifest_path = output_dir / "workflow_manifest.json"
    internal = _cnmf_internal_paths(output_dir, name)
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"Existing {collection} manifest does not match current settings: "
                f"{manifest_path}"
            )
    elif internal["params"].exists():
        raise FileExistsError(
            f"Existing cNMF parameters lack a workflow manifest: {internal['params']}"
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(manifest_path, payload)


@contextmanager
def _collection_lock(output_dir: Path, *, collection: str, dry_run: bool):
    """Prevent two launchers from writing the same cNMF collection."""

    if dry_run:
        yield
        return
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "workflow.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            holder = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"Another {collection} launcher holds {lock_path}: {holder}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n"
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_sweep_unlocked(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    dry_run: bool = False,
) -> None:
    """Prepare, factorize, combine, and calculate K-selection statistics."""

    if not paths.counts_h5ad.exists() and not dry_run:
        raise FileNotFoundError(f"Missing prepared counts: {paths.counts_h5ad}")
    _ensure_collection_manifest(spec, paths, collection="sweep", dry_run=dry_run)
    commands = cnmf_stage_commands(config, spec, paths)
    _run_prepare(
        commands["sweep_prepare"],
        output_dir=paths.sweep_output_dir,
        name=paths.sweep_name,
        log_path=paths.sweep_output_dir / "prepare.log",
        dry_run=dry_run,
    )
    _run_factorize_workers(
        commands["sweep_factorize_worker_0"],
        workers=spec.sweep_workers,
        output_dir=paths.sweep_output_dir,
        name=paths.sweep_name,
        expected_iterations=len(spec.sweep_k_values) * spec.sweep_n_iter,
        dry_run=dry_run,
    )
    _run_combine(
        commands["sweep_combine"],
        output_dir=paths.sweep_output_dir,
        name=paths.sweep_name,
        k_values=spec.sweep_k_values,
        log_path=paths.sweep_output_dir / "combine.log",
        dry_run=dry_run,
    )
    internal = _cnmf_internal_paths(paths.sweep_output_dir, paths.sweep_name)
    if internal["k_stats"].exists() and internal["k_plot"].exists():
        print(f"Reusing K-selection outputs for {paths.sweep_name}")
    else:
        _run_logged(
            commands["sweep_kselect"],
            log_path=paths.sweep_output_dir / "k_selection.log",
            dry_run=dry_run,
        )


def run_sweep(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    dry_run: bool = False,
) -> None:
    """Run or resume one locked K-sweep collection."""

    with _collection_lock(
        paths.sweep_output_dir,
        collection="sweep",
        dry_run=dry_run,
    ):
        _run_sweep_unlocked(config, spec, paths, dry_run=dry_run)


def _density_tag(value: float | str) -> str:
    return str(value).replace(".", "_")


def consensus_result_paths(
    paths: CnmfWorkflowPaths,
    spec: CnmfLineageSpec,
) -> dict[str, Path]:
    tag = _density_tag(spec.density_threshold)
    run_dir = paths.selected_run_dir
    stem = paths.selected_name
    k = spec.selected_k
    return {
        "usage": run_dir / f"{stem}.usages.k_{k}.dt_{tag}.consensus.txt",
        "gene_scores": run_dir / f"{stem}.gene_spectra_score.k_{k}.dt_{tag}.txt",
        "gene_tpm": run_dir / f"{stem}.gene_spectra_tpm.k_{k}.dt_{tag}.txt",
        "spectra": run_dir / f"{stem}.spectra.k_{k}.dt_{tag}.consensus.txt",
        "clustering": run_dir / f"{stem}.clustering.k_{k}.dt_{tag}.png",
        "hvgs": run_dir / f"{stem}.overdispersed_genes.txt",
    }


def _run_selected_unlocked(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    dry_run: bool = False,
) -> None:
    """Run a distinct 100-replicate selected-K fit and consensus."""

    if not paths.counts_h5ad.exists() and not dry_run:
        raise FileNotFoundError(f"Missing prepared counts: {paths.counts_h5ad}")
    _ensure_collection_manifest(spec, paths, collection="selected", dry_run=dry_run)
    commands = cnmf_stage_commands(config, spec, paths)
    _run_prepare(
        commands["selected_prepare"],
        output_dir=paths.selected_output_dir,
        name=paths.selected_name,
        log_path=paths.selected_output_dir / "prepare.log",
        dry_run=dry_run,
    )
    _run_factorize_workers(
        commands["selected_factorize_worker_0"],
        workers=spec.selected_workers,
        output_dir=paths.selected_output_dir,
        name=paths.selected_name,
        expected_iterations=spec.selected_n_iter,
        dry_run=dry_run,
    )
    _run_combine(
        commands["selected_combine"],
        output_dir=paths.selected_output_dir,
        name=paths.selected_name,
        k_values=[spec.selected_k],
        log_path=paths.selected_output_dir / "combine.log",
        dry_run=dry_run,
    )
    results = consensus_result_paths(paths, spec)
    if all(result.exists() for result in results.values()):
        print(f"Reusing consensus outputs for {paths.selected_name}")
    else:
        _run_logged(
            commands["selected_consensus"],
            log_path=paths.selected_output_dir / "consensus.log",
            dry_run=dry_run,
        )


def run_selected(
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
    *,
    dry_run: bool = False,
) -> None:
    """Run or resume one locked selected-K collection and consensus."""

    with _collection_lock(
        paths.selected_output_dir,
        collection="selected-K",
        dry_run=dry_run,
    ):
        _run_selected_unlocked(config, spec, paths, dry_run=dry_run)
    if not dry_run:
        write_usage_h5ad(paths, spec)


def load_k_selection_stats(paths: CnmfWorkflowPaths) -> pd.DataFrame:
    """Load native cNMF stability and prediction-error statistics."""

    from cnmf import load_df_from_npz

    path = _cnmf_internal_paths(paths.sweep_output_dir, paths.sweep_name)["k_stats"]
    if not path.exists():
        raise FileNotFoundError(f"Missing K-selection statistics: {path}")
    return load_df_from_npz(path).sort_values("k").reset_index(drop=True)


def plot_k_selection(stats: pd.DataFrame, *, selected_k: int | None = None):
    """Plot cNMF stability and prediction error on aligned Plotly axes."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    required = {"k", "silhouette", "prediction_error"}
    missing = sorted(required.difference(stats.columns))
    if missing:
        raise KeyError("Missing K-selection columns: " + ", ".join(missing))
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=stats["k"],
            y=stats["silhouette"],
            mode="lines+markers",
            name="stability",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=stats["k"],
            y=stats["prediction_error"],
            mode="lines+markers",
            name="prediction error",
        ),
        secondary_y=True,
    )
    if selected_k is not None:
        figure.add_vline(x=int(selected_k), line_dash="dash", line_color="gray")
    figure.update_xaxes(title_text="K")
    figure.update_yaxes(title_text="Stability", secondary_y=False)
    figure.update_yaxes(title_text="Prediction error", secondary_y=True)
    figure.update_layout(template="plotly_white", height=500, width=900)
    return figure


def load_consensus_results(
    paths: CnmfWorkflowPaths,
    spec: CnmfLineageSpec,
    *,
    usage_prefix: str = "Usage_",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load normalized usage, gene scores, TPM spectra, and top genes."""

    from cnmf import cNMF

    obj = cNMF(output_dir=str(paths.selected_output_dir), name=paths.selected_name)
    usage, scores, tpm, top_genes = obj.load_results(
        K=spec.selected_k,
        density_threshold=spec.density_threshold,
    )
    usage = usage.copy()
    usage.columns = [f"{usage_prefix}{column}" for column in usage.columns]
    scores = scores.copy()
    scores.columns = list(usage.columns)
    tpm = tpm.copy()
    tpm.columns = list(usage.columns)
    top_genes = top_genes.copy()
    top_genes.columns = list(usage.columns)
    return usage, scores, tpm, top_genes


def join_usage_to_counts(
    counts_h5ad: str | Path,
    usage: pd.DataFrame,
):
    """Load the compact lineage H5AD and join consensus usages by cell ID."""

    import anndata as ad

    adata = ad.read_h5ad(counts_h5ad)
    if not adata.obs_names.is_unique or not usage.index.is_unique:
        raise ValueError("Counts and usage cell IDs must be unique")
    counts_index = pd.Index(adata.obs_names.astype(str))
    usage_index = pd.Index(usage.index.astype(str))
    missing_usage = counts_index.difference(usage_index)
    extra_usage = usage_index.difference(counts_index)
    if len(missing_usage) or len(extra_usage):
        raise RuntimeError(
            "Consensus usage cell IDs do not exactly match the counts export: "
            f"missing={len(missing_usage)}, extra={len(extra_usage)}"
        )
    usage = usage.copy()
    usage.index = usage_index
    adata.obs = adata.obs.join(usage.reindex(counts_index), how="left")
    return adata


def write_usage_h5ad(
    paths: CnmfWorkflowPaths,
    spec: CnmfLineageSpec,
    *,
    overwrite: bool = False,
) -> Path:
    """Materialize counts, metadata, spatial coordinates, and consensus usages."""

    if paths.usage_h5ad.exists() and not overwrite:
        return paths.usage_h5ad
    usage, _, _, _ = load_consensus_results(paths, spec)
    adata = join_usage_to_counts(paths.counts_h5ad, usage)
    adata.uns["cnmf_consensus"] = {
        "selected_k": spec.selected_k,
        "selected_n_iter": spec.selected_n_iter,
        "density_threshold": spec.density_threshold,
        "local_neighborhood_size": spec.local_neighborhood_size,
        "usage_columns": list(usage.columns),
    }
    paths.usage_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.usage_h5ad.with_name(
        f".{paths.usage_h5ad.stem}-{os.getpid()}.h5ad"
    )
    try:
        adata.write_h5ad(temporary, compression="gzip")
        temporary.replace(paths.usage_h5ad)
    finally:
        if temporary.exists():
            temporary.unlink()
    return paths.usage_h5ad


def usage_columns(frame_or_adata: Any, *, prefix: str = "Usage_") -> list[str]:
    frame = frame_or_adata.obs if hasattr(frame_or_adata, "obs") else frame_or_adata
    return [column for column in frame.columns if str(column).startswith(prefix)]


def subset_usage_by_cell_type(
    obs: pd.DataFrame,
    *,
    cell_type_key: str,
    cell_types: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Select exact review cell types and always exclude missing/blank labels.

    ``cell_types=None`` retains every non-missing value present in the cNMF
    analysis object. Supplying values restricts all downstream review summaries
    to those exact labels.
    """

    if cell_type_key not in obs:
        raise KeyError(f"Usage cell-type key {cell_type_key!r} is absent from obs")
    labels = obs[cell_type_key].astype("string")
    valid = labels.notna() & labels.str.strip().ne("")
    available = labels.loc[valid].astype(str).drop_duplicates().tolist()
    if cell_types is None:
        selected = available
    else:
        values = [cell_types] if isinstance(cell_types, str) else cell_types
        selected = list(dict.fromkeys(str(value) for value in values))
        if not selected:
            raise ValueError("cell_types must be None or contain at least one label")
        missing = sorted(set(selected).difference(available))
        if missing:
            raise KeyError("Unknown review cell types: " + ", ".join(missing))
    mask = valid & labels.astype(str).isin(selected)
    result = obs.loc[mask].copy()
    if result.empty:
        raise ValueError("The usage cell-type selection retained no cells")
    result[cell_type_key] = labels.loc[mask].astype(str).to_numpy()
    result.attrs["cell_type_key"] = cell_type_key
    result.attrs["cell_types"] = selected
    return result


def summarize_usage(
    obs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    usage_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return cell counts plus mean and median usage for review groups."""

    missing_groups = [column for column in group_columns if column not in obs]
    if missing_groups:
        raise KeyError("Missing usage grouping columns: " + ", ".join(missing_groups))
    usage_cols = list(usage_cols or usage_columns(obs))
    if not usage_cols:
        raise ValueError("No Usage_ columns are available")
    grouped = obs.groupby(list(group_columns), observed=True, dropna=False)
    counts = grouped.size().rename("n_cells").reset_index()
    mean = grouped[usage_cols].mean().reset_index().melt(
        id_vars=list(group_columns),
        var_name="program",
        value_name="mean_usage",
    )
    median = grouped[usage_cols].median().reset_index().melt(
        id_vars=list(group_columns),
        var_name="program",
        value_name="median_usage",
    )
    return mean.merge(median, on=[*group_columns, "program"]).merge(
        counts,
        on=list(group_columns),
        validate="many_to_one",
    )


def sample_usage_summary(
    obs: pd.DataFrame,
    *,
    sample_key: str,
    condition_key: str,
    usage_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize usages at the biological-sample level for comparisons."""

    summary = summarize_usage(
        obs,
        group_columns=[condition_key, sample_key],
        usage_cols=usage_cols,
    )
    mapping = summary[[condition_key, sample_key]].drop_duplicates()
    duplicated = mapping[sample_key].astype(str).duplicated(keep=False)
    if duplicated.any():
        problem = mapping.loc[duplicated, sample_key].astype(str).unique().tolist()
        raise ValueError(f"Samples map to multiple conditions: {problem}")
    return summary


def top_genes_long(top_genes: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Convert the cNMF top-gene matrix into a tidy ranked table."""

    shown = top_genes.head(int(n)).copy()
    shown.index = np.arange(1, len(shown) + 1)
    return (
        shown.rename_axis("rank")
        .reset_index()
        .melt(id_vars="rank", var_name="program", value_name="gene")
        .sort_values(["program", "rank"], kind="stable")
        .reset_index(drop=True)
    )


def plot_sample_usage(
    summary: pd.DataFrame,
    *,
    condition_key: str,
    sample_key: str,
    programs: Sequence[str] | None = None,
):
    """Plot sample-mean usages without treating cells as replicates."""

    import plotly.express as px

    available = summary["program"].astype(str).unique().tolist()
    selected = available if programs is None else [str(value) for value in programs]
    missing = sorted(set(selected).difference(available))
    if missing:
        raise KeyError("Unknown usage programs: " + ", ".join(missing))
    shown = summary.loc[summary["program"].astype(str).isin(selected)].copy()
    figure = px.box(
        shown,
        x=condition_key,
        y="mean_usage",
        color=condition_key,
        points="all",
        hover_data=[sample_key, "n_cells"],
        facet_col="program",
        facet_col_wrap=min(4, len(selected)),
        title="Biological-sample mean cNMF usage",
    )
    rows = (len(selected) + 3) // 4
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        height=max(430, 300 * rows),
    )
    return figure


def plot_usage_heatmap(
    summary: pd.DataFrame,
    *,
    group_column: str,
    value_column: str = "mean_usage",
    title: str | None = None,
):
    """Plot a group-by-program usage matrix with Plotly."""

    import plotly.express as px

    matrix = summary.pivot_table(
        index=group_column,
        columns="program",
        values=value_column,
        aggfunc="first",
    )
    figure = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": value_column.replace("_", " ")},
        title=title,
    )
    figure.update_layout(template="plotly_white")
    return figure


def plot_spatial_usage(
    adata: Any,
    *,
    program: str,
    sample: str,
    sample_key: str,
    spatial_key: str,
    point_size: float = 3.0,
):
    """Return an interactive spatial usage map for one biological sample."""

    import plotly.express as px

    if program not in adata.obs:
        raise KeyError(f"Program {program!r} not found in adata.obs")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key {spatial_key!r} not found in adata.obsm")
    mask = adata.obs[sample_key].astype(str).eq(str(sample)).to_numpy()
    if not mask.any():
        raise ValueError(f"Sample {sample!r} has no cells in this lineage")
    coords = np.asarray(adata.obsm[spatial_key])[mask, :2]
    plot = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            program: adata.obs.loc[mask, program].to_numpy(dtype=float),
            "cell_id": adata.obs_names[mask].astype(str),
        }
    )
    figure = px.scatter(
        plot,
        x="x",
        y="y",
        color=program,
        hover_data=["cell_id"],
        color_continuous_scale="Viridis",
        render_mode="webgl",
    )
    figure.update_traces(marker={"size": point_size, "opacity": 0.85})
    figure.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    figure.update_layout(
        title=f"{program}: {sample}",
        template="plotly_white",
        height=700,
        width=800,
    )
    return figure


def cnmf_status(
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
) -> pd.DataFrame:
    """Report exact artifact-backed completion for every workflow stage."""

    sweep_internal = _cnmf_internal_paths(paths.sweep_output_dir, paths.sweep_name)
    selected_internal = _cnmf_internal_paths(
        paths.selected_output_dir, paths.selected_name
    )
    sweep_iterations = _completed_iterations(paths.sweep_run_dir, paths.sweep_name)
    selected_iterations = _completed_iterations(
        paths.selected_run_dir, paths.selected_name
    )
    sweep_merged = [
        sweep_internal["temp_dir"]
        / f"{paths.sweep_name}.spectra.k_{k}.merged.df.npz"
        for k in spec.sweep_k_values
    ]
    selected_merged = (
        selected_internal["temp_dir"]
        / f"{paths.selected_name}.spectra.k_{spec.selected_k}.merged.df.npz"
    )
    consensus = consensus_result_paths(paths, spec)
    rows = [
        {
            "stage": "input_export",
            "complete": all(
                path.exists()
                for path in (
                    paths.counts_h5ad,
                    paths.obs_csv,
                    paths.var_csv,
                    paths.selection_summary_csv,
                    paths.export_manifest,
                )
            ),
            "observed": sum(
                path.exists()
                for path in (
                    paths.counts_h5ad,
                    paths.obs_csv,
                    paths.var_csv,
                    paths.selection_summary_csv,
                    paths.export_manifest,
                )
            ),
            "expected": 5,
            "artifact": str(paths.counts_h5ad),
        },
        {
            "stage": "sweep_prepare",
            "complete": sweep_internal["params"].exists()
            and sweep_internal["normalized_counts"].exists(),
            "observed": sum(
                path.exists()
                for path in (
                    sweep_internal["params"],
                    sweep_internal["normalized_counts"],
                )
            ),
            "expected": 2,
            "artifact": str(sweep_internal["params"]),
        },
        {
            "stage": "sweep_factorize",
            "complete": sweep_iterations
            == len(spec.sweep_k_values) * spec.sweep_n_iter,
            "observed": sweep_iterations,
            "expected": len(spec.sweep_k_values) * spec.sweep_n_iter,
            "artifact": str(sweep_internal["temp_dir"]),
        },
        {
            "stage": "sweep_combine",
            "complete": all(path.exists() for path in sweep_merged),
            "observed": sum(path.exists() for path in sweep_merged),
            "expected": len(sweep_merged),
            "artifact": str(sweep_internal["temp_dir"]),
        },
        {
            "stage": "k_selection",
            "complete": sweep_internal["k_stats"].exists()
            and sweep_internal["k_plot"].exists(),
            "observed": sum(
                path.exists()
                for path in (sweep_internal["k_stats"], sweep_internal["k_plot"])
            ),
            "expected": 2,
            "artifact": str(sweep_internal["k_plot"]),
        },
        {
            "stage": "selected_prepare",
            "complete": selected_internal["params"].exists()
            and selected_internal["normalized_counts"].exists(),
            "observed": sum(
                path.exists()
                for path in (
                    selected_internal["params"],
                    selected_internal["normalized_counts"],
                )
            ),
            "expected": 2,
            "artifact": str(selected_internal["params"]),
        },
        {
            "stage": "selected_factorize",
            "complete": selected_iterations == spec.selected_n_iter,
            "observed": selected_iterations,
            "expected": spec.selected_n_iter,
            "artifact": str(selected_internal["temp_dir"]),
        },
        {
            "stage": "selected_combine",
            "complete": selected_merged.exists(),
            "observed": int(selected_merged.exists()),
            "expected": 1,
            "artifact": str(selected_merged),
        },
        {
            "stage": "consensus",
            "complete": all(path.exists() for path in consensus.values()),
            "observed": sum(path.exists() for path in consensus.values()),
            "expected": len(consensus),
            "artifact": str(consensus["usage"]),
        },
        {
            "stage": "usage_h5ad",
            "complete": paths.usage_h5ad.exists(),
            "observed": int(paths.usage_h5ad.exists()),
            "expected": 1,
            "artifact": str(paths.usage_h5ad),
        },
    ]
    return pd.DataFrame(rows)


def write_run_manifest(
    config_path: str | Path,
    spec: CnmfLineageSpec,
    paths: CnmfWorkflowPaths,
) -> None:
    status = cnmf_status(spec, paths)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "analysis": {
            "name": spec.name,
            "cell_types": list(spec.cell_types),
            "exclude_cells": list(spec.exclude_cells),
            "compartments": (
                "all" if spec.compartments is None else list(spec.compartments)
            ),
            "conditions": "all" if spec.conditions is None else list(spec.conditions),
        },
        "sweep": {
            "k_values": list(spec.sweep_k_values),
            "n_iter": spec.sweep_n_iter,
            "workers": spec.sweep_workers,
            "numgenes": spec.numgenes,
            "seed": spec.seed,
        },
        "selected": {
            "k": spec.selected_k,
            "n_iter": spec.selected_n_iter,
            "workers": spec.selected_workers,
            "density_threshold": spec.density_threshold,
            "local_neighborhood_size": spec.local_neighborhood_size,
        },
        "paths": {
            field: str(value)
            for field, value in paths.__dict__.items()
            if isinstance(value, Path)
        },
        "status": status.to_dict(orient="records"),
    }
    _write_json_atomic(paths.run_manifest, payload)


def with_selected_k(
    config_path: str | Path,
    config: Mapping[str, Any],
    spec: CnmfLineageSpec,
    selected_k: int,
) -> tuple[CnmfLineageSpec, CnmfWorkflowPaths]:
    """Return a selected-K override and its distinct output paths."""

    updated = replace(spec, selected_k=int(selected_k))
    if updated.selected_k < 2:
        raise ValueError("selected_k must be at least 2")
    return updated, resolve_cnmf_paths(config_path, config, updated)
