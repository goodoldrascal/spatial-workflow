"""Whole-sample, non-graph adaptive-window LIANA rank aggregation.

The module separates the expensive window-level LIANA run from the biological-
replicate comparison.  Windows may overlap, but they are collapsed to one edge
vector per sample before any condition-level statistic is calculated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind


EDGE_COLUMNS = ["source", "target", "ligand_complex", "receptor_complex"]
CELLCHAT_PROTEIN_ANNOTATIONS = (
    "Secreted Signaling",
    "ECM-Receptor",
    "Cell-Cell Contact",
)
LIANA_SCORE_COLUMNS = [
    "magnitude_rank",
    "specificity_rank",
    "lr_means",
    "cellphone_pvals",
    "expr_prod",
    "scaled_weight",
    "lr_logfc",
    "spec_weight",
    "lrscore",
]


@dataclass(frozen=True)
class WindowParameters:
    """Parameters for the non-graph adaptive-grid estimator."""

    adaptive_k: int = 60
    grid_stride: float = 100.0
    expr_prop: float = 0.1
    min_cells: int = 5
    min_required_groups: int = 2
    spatial_bandwidth: float = 250.0
    spatial_kernel: str = "gaussian"
    spatial_trim_fraction: float = 0.1
    n_perms: int = 200
    seed: int = 1337
    n_jobs: int = 8
    anchor_groups: tuple[str, ...] = ()
    anchor_k_target: int = 60
    anchor_min_cells: int = 2
    anchor_max_radius: float = 150.0
    anchor_dedup_distance: float = 60.0
    return_all_lrs: bool = False

    def validate(self) -> None:
        if self.adaptive_k < 1:
            raise ValueError("adaptive_k must be >= 1")
        if self.grid_stride <= 0:
            raise ValueError("grid_stride must be > 0")
        if not 0 <= self.expr_prop <= 1:
            raise ValueError("expr_prop must be in [0, 1]")
        if self.min_cells < 1 or self.min_required_groups < 2:
            raise ValueError("min_cells must be >= 1 and min_required_groups >= 2")
        if self.spatial_bandwidth <= 0:
            raise ValueError("spatial_bandwidth must be > 0")
        if not 0 <= self.spatial_trim_fraction < 0.5:
            raise ValueError("spatial_trim_fraction must be in [0, 0.5)")
        if self.n_perms < 2 or self.n_jobs < 1:
            raise ValueError("n_perms must be >= 2 and n_jobs must be >= 1")
        if self.anchor_groups:
            if self.anchor_k_target < 1 or self.anchor_min_cells < 1:
                raise ValueError("anchor cell parameters must be >= 1")
            if self.anchor_max_radius <= 0 or self.anchor_dedup_distance < 0:
                raise ValueError("anchor radii must be positive/non-negative")


def parameter_table(parameters: WindowParameters | None = None) -> pd.DataFrame:
    """Return the user-facing parameter inventory shown in Notebook 07."""

    parameters = parameters or WindowParameters()
    descriptions = {
        "adaptive_k": "Nearest cells assigned to each adaptive grid center.",
        "grid_stride": "Spacing between candidate grid centers in spatial units.",
        "expr_prop": "Minimum within-group expression proportion for every LR subunit.",
        "min_cells": "Minimum cells in each LIANA source/target group.",
        "min_required_groups": "Groups with >= min_cells required for a valid window.",
        "spatial_bandwidth": "Gaussian proximity bandwidth passed to LIANA.",
        "spatial_kernel": "Spatial proximity kernel passed to LIANA.",
        "spatial_trim_fraction": "Trim fraction for nearest-target spatial distance.",
        "n_perms": "LIANA within-window permutations.",
        "seed": "LIANA/random seed.",
        "n_jobs": "Parallel jobs used inside one LIANA call.",
        "anchor_groups": "Optional sparse cell groups receiving extra anchored windows.",
        "anchor_k_target": "Total-cell target for each optional anchor window.",
        "anchor_min_cells": "Anchor-group cells required in an anchor window.",
        "anchor_max_radius": "Largest accepted anchor-window radius.",
        "anchor_dedup_distance": "Minimum distance between same-group anchor centers.",
        "return_all_lrs": (
            "Whether LIANA returns below-expr_prop LR rows at its worst score. "
            "False preserves explicit observed/absent support for later zero completion."
        ),
    }
    values = asdict(parameters)
    return pd.DataFrame(
        [
            {"parameter": key, "default": values[key], "meaning": descriptions[key]}
            for key in values
        ]
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def canonical_component(value: object) -> str:
    """Canonical key used only for joining LIANA complexes to CellChat metadata."""

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _symbol_complex(value: object, fallback: object) -> str:
    """Convert CellChat comma-separated gene symbols to LIANA complex syntax."""

    selected = fallback if pd.isna(value) or not str(value).strip() else value
    symbols = [symbol.strip() for symbol in str(selected).split(",") if symbol.strip()]
    return "_".join(symbols)


def _selection(values: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize an optional CLI/notebook selection while preserving its order."""

    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(dict.fromkeys(str(value) for value in values))


def _validate_selection(
    observed: pd.Series,
    requested: Sequence[str] | str | None,
    *,
    label: str,
) -> tuple[str, ...]:
    selected = _selection(requested)
    if not selected:
        return selected
    available = set(observed.dropna().astype(str))
    missing = sorted(set(selected).difference(available))
    if missing:
        raise KeyError(f"Requested {label} absent from input: {missing}")
    return selected


def load_cellchat_resource(
    path: str | Path,
    *,
    pathways: Sequence[str] | str | None = None,
    annotations: Sequence[str] | str | None = None,
    lr_pairs: Sequence[str] | str | None = None,
    include_non_protein: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and optionally subset CellChat before constructing the LIANA resource.

    By default this mirrors CellChat v2 ``subsetDB(CellChatDB)`` and retains
    Secreted Signaling, ECM-Receptor, and Cell-Cell Contact while excluding
    Non-protein Signaling. Set ``include_non_protein=True`` to use the complete
    database, or provide explicit annotations to select the requested categories.

    Filters across pathways, annotations, and explicit LR pairs are combined with
    AND. Multiple values within one filter are combined with OR. Explicit pairs
    may use either raw CellChat names or resolved LIANA gene-complex names in
    ``ligand^receptor`` form.
    """

    path = Path(path)
    cellchat = pd.read_csv(path, low_memory=False)
    total_database_rows = len(cellchat)
    required = {"ligand", "receptor", "pathway_name"}
    missing = sorted(required.difference(cellchat.columns))
    if missing:
        raise KeyError(f"{path}: missing CellChat columns {missing}")
    ligand_symbols = (
        cellchat["ligand.symbol"]
        if "ligand.symbol" in cellchat
        else pd.Series(np.nan, index=cellchat.index)
    )
    receptor_symbols = (
        cellchat["receptor.symbol"]
        if "receptor.symbol" in cellchat
        else pd.Series(np.nan, index=cellchat.index)
    )
    cellchat = cellchat.copy()
    cellchat["cellchat_ligand"] = cellchat["ligand"].astype(str)
    cellchat["cellchat_receptor"] = cellchat["receptor"].astype(str)
    cellchat["resource_ligand"] = [
        _symbol_complex(value, fallback)
        for value, fallback in zip(ligand_symbols, cellchat["ligand"])
    ]
    cellchat["resource_receptor"] = [
        _symbol_complex(value, fallback)
        for value, fallback in zip(receptor_symbols, cellchat["receptor"])
    ]
    if cellchat[["resource_ligand", "resource_receptor"]].eq("").any().any():
        raise ValueError(f"{path}: an LR row has no testable gene symbols")
    cellchat["cellchat_lr_pair"] = (
        cellchat["cellchat_ligand"] + "^" + cellchat["cellchat_receptor"]
    )
    cellchat["resource_lr_pair"] = (
        cellchat["resource_ligand"] + "^" + cellchat["resource_receptor"]
    )

    selected_pathways = _validate_selection(
        cellchat["pathway_name"], pathways, label="CellChat pathways"
    )
    selected_annotations = _selection(annotations)
    default_protein_subset = not selected_annotations and not include_non_protein
    if default_protein_subset:
        selected_annotations = CELLCHAT_PROTEIN_ANNOTATIONS
    if selected_annotations:
        if "annotation" not in cellchat:
            raise KeyError(f"{path}: annotations were requested but annotation is absent")
        if not default_protein_subset:
            _validate_selection(
                cellchat["annotation"], selected_annotations, label="CellChat annotations"
            )
    selected_lr_pairs = _selection(lr_pairs)
    if selected_lr_pairs:
        all_pairs = set(cellchat["cellchat_lr_pair"]) | set(cellchat["resource_lr_pair"])
        missing_pairs = sorted(set(selected_lr_pairs).difference(all_pairs))
        if missing_pairs:
            raise KeyError(f"Requested LR pairs absent from CellChat: {missing_pairs}")

    if selected_pathways:
        cellchat = cellchat[cellchat["pathway_name"].astype(str).isin(selected_pathways)]
    if selected_annotations:
        cellchat = cellchat[cellchat["annotation"].astype(str).isin(selected_annotations)]
    if selected_lr_pairs:
        cellchat = cellchat[
            cellchat["cellchat_lr_pair"].isin(selected_lr_pairs)
            | cellchat["resource_lr_pair"].isin(selected_lr_pairs)
        ]
    if cellchat.empty:
        raise ValueError("The CellChat pathway/annotation/LR filters have an empty intersection")

    cellchat = cellchat.copy()
    cellchat["_ligand_key"] = cellchat["resource_ligand"].map(canonical_component)
    cellchat["_receptor_key"] = cellchat["resource_receptor"].map(canonical_component)
    resource = (
        cellchat[["resource_ligand", "resource_receptor"]]
        .drop_duplicates()
        .rename(columns={"resource_ligand": "ligand", "resource_receptor": "receptor"})
        .reset_index(drop=True)
    )
    cellchat.attrs["total_database_rows"] = int(total_database_rows)
    cellchat.attrs["selected_database_rows"] = int(len(cellchat))
    return resource, cellchat


def load_custom_lr_resource(
    path: str | Path,
    *,
    pathways: Sequence[str] | str | None = None,
    annotations: Sequence[str] | str | None = None,
    lr_pairs: Sequence[str] | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a custom LR CSV with required ligand/receptor columns.

    ``pathway_name`` and ``annotation`` are optional metadata columns. Complexes
    should use LIANA underscore syntax, for example ``Itgav_Itgb3``.
    """

    path = Path(path)
    metadata = pd.read_csv(path, low_memory=False)
    total_rows = len(metadata)
    if {"ligand", "receptor"}.issubset(metadata.columns):
        ligand_col, receptor_col = "ligand", "receptor"
    elif {"ligand_complex", "receptor_complex"}.issubset(metadata.columns):
        ligand_col, receptor_col = "ligand_complex", "receptor_complex"
    else:
        raise KeyError(
            f"{path}: custom LR CSV needs ligand/receptor or "
            "ligand_complex/receptor_complex columns"
        )
    metadata = metadata.copy()
    metadata["resource_ligand"] = metadata[ligand_col].astype(str).str.strip()
    metadata["resource_receptor"] = metadata[receptor_col].astype(str).str.strip()
    if metadata[["resource_ligand", "resource_receptor"]].eq("").any().any():
        raise ValueError(f"{path}: custom LR CSV contains a blank ligand or receptor")
    metadata["cellchat_ligand"] = metadata["resource_ligand"]
    metadata["cellchat_receptor"] = metadata["resource_receptor"]
    metadata["cellchat_lr_pair"] = (
        metadata["resource_ligand"] + "^" + metadata["resource_receptor"]
    )
    metadata["resource_lr_pair"] = metadata["cellchat_lr_pair"]

    selected_pathways = _selection(pathways)
    selected_annotations = _selection(annotations)
    selected_lr_pairs = _selection(lr_pairs)
    if selected_pathways:
        if "pathway_name" not in metadata:
            raise KeyError("Custom LR pathway filtering requires a pathway_name column")
        _validate_selection(metadata["pathway_name"], selected_pathways, label="pathways")
        metadata = metadata[metadata["pathway_name"].astype(str).isin(selected_pathways)]
    if selected_annotations:
        if "annotation" not in metadata:
            raise KeyError("Custom LR annotation filtering requires an annotation column")
        _validate_selection(metadata["annotation"], selected_annotations, label="annotations")
        metadata = metadata[metadata["annotation"].astype(str).isin(selected_annotations)]
    if selected_lr_pairs:
        _validate_selection(metadata["resource_lr_pair"], selected_lr_pairs, label="LR pairs")
        metadata = metadata[metadata["resource_lr_pair"].isin(selected_lr_pairs)]
    if metadata.empty:
        raise ValueError("The custom LR filters have an empty intersection")
    if "pathway_name" not in metadata:
        metadata["pathway_name"] = pd.NA
    if "annotation" not in metadata:
        metadata["annotation"] = pd.NA
    metadata["_ligand_key"] = metadata["resource_ligand"].map(canonical_component)
    metadata["_receptor_key"] = metadata["resource_receptor"].map(canonical_component)
    resource = (
        metadata[["resource_ligand", "resource_receptor"]]
        .drop_duplicates()
        .rename(columns={"resource_ligand": "ligand", "resource_receptor": "receptor"})
        .reset_index(drop=True)
    )
    metadata.attrs["total_database_rows"] = int(total_rows)
    metadata.attrs["selected_database_rows"] = int(len(metadata))
    return resource, metadata


def load_rankagg_resource(
    li,
    *,
    mode: str = "cellchat",
    cellchat_csv: str | Path | None = None,
    custom_lr_csv: str | Path | None = None,
    liana_resource_name: str = "mouseconsensus",
    pathways: Sequence[str] | str | None = None,
    annotations: Sequence[str] | str | None = None,
    lr_pairs: Sequence[str] | str | None = None,
    include_non_protein: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve CellChat, custom CSV, or native LIANA LR space."""

    mode = str(mode).lower()
    if mode == "cellchat":
        if cellchat_csv is None:
            raise ValueError("cellchat_csv is required for resource_mode=cellchat")
        return load_cellchat_resource(
            cellchat_csv,
            pathways=pathways,
            annotations=annotations,
            lr_pairs=lr_pairs,
            include_non_protein=include_non_protein,
        )
    if mode == "custom":
        if custom_lr_csv is None:
            raise ValueError("custom_lr_csv is required for resource_mode=custom")
        return load_custom_lr_resource(
            custom_lr_csv, pathways=pathways, annotations=annotations, lr_pairs=lr_pairs
        )
    if mode != "liana":
        raise ValueError("resource_mode must be one of: cellchat, custom, liana")
    if _selection(pathways) or _selection(annotations):
        raise ValueError(
            "Native LIANA resources do not carry CellChat pathway/annotation metadata; "
            "use explicit lr_pairs or a custom annotated CSV instead"
        )
    resource = pd.DataFrame(li.resource.select_resource(liana_resource_name)).copy()
    if not {"ligand", "receptor"}.issubset(resource.columns):
        raise KeyError(
            f"LIANA resource {liana_resource_name!r} lacks ligand/receptor columns"
        )
    resource = resource[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
    total_rows = len(resource)
    selected_lr_pairs = _selection(lr_pairs)
    pair_names = resource["ligand"].astype(str) + "^" + resource["receptor"].astype(str)
    if selected_lr_pairs:
        _validate_selection(pair_names, selected_lr_pairs, label="LR pairs")
        resource = resource[pair_names.isin(selected_lr_pairs)].reset_index(drop=True)
    metadata = resource.rename(
        columns={"ligand": "resource_ligand", "receptor": "resource_receptor"}
    ).copy()
    metadata["cellchat_ligand"] = metadata["resource_ligand"]
    metadata["cellchat_receptor"] = metadata["resource_receptor"]
    metadata["cellchat_lr_pair"] = (
        metadata["resource_ligand"] + "^" + metadata["resource_receptor"]
    )
    metadata["resource_lr_pair"] = metadata["cellchat_lr_pair"]
    metadata["pathway_name"] = pd.NA
    metadata["annotation"] = pd.NA
    metadata["_ligand_key"] = metadata["resource_ligand"].map(canonical_component)
    metadata["_receptor_key"] = metadata["resource_receptor"].map(canonical_component)
    metadata.attrs["total_database_rows"] = int(total_rows)
    metadata.attrs["selected_database_rows"] = int(len(metadata))
    return resource, metadata


def make_grid_centers(coords: np.ndarray, stride: float) -> np.ndarray:
    """Create the original rectangular adaptive-grid candidate centers."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must have shape (n_cells, 2)")
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    xs = np.arange(lower[0], upper[0] + stride, stride)
    ys = np.arange(lower[1], upper[1] + stride, stride)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _window_tables(
    *,
    centers: np.ndarray,
    hits: Sequence[np.ndarray],
    groups: np.ndarray,
    obs_names: np.ndarray,
    min_cells: int,
    sample: str,
    family: str,
    window_type: str,
    anchor_group: str | None = None,
    radii: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict] = []
    membership_rows: list[dict] = []
    for candidate_id, indices in enumerate(hits):
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            continue
        counts = pd.Series(groups[indices]).value_counts()
        context = f"{_slug(sample)}__{_slug(family)}__w{candidate_id:05d}"
        row = {
            "context": context,
            "candidate_id": candidate_id,
            "sample": sample,
            "window_type": window_type,
            "window_family": family,
            "anchor_group": anchor_group,
            "center_x": float(centers[candidate_id, 0]),
            "center_y": float(centers[candidate_id, 1]),
            "n_cells_total": int(indices.size),
            "n_groups_present": int(len(counts)),
            "n_groups_ge_min": int((counts >= min_cells).sum()),
        }
        if radii is not None:
            row["effective_radius"] = float(radii[candidate_id])
        if anchor_group is not None:
            row["n_anchor_cells"] = int(counts.get(anchor_group, 0))
        window_rows.append(row)
        membership_rows.extend(
            {
                "context": context,
                "obs_name": str(obs_names[index]),
                "group": str(groups[index]),
            }
            for index in indices
        )
    return pd.DataFrame(window_rows), pd.DataFrame(membership_rows)


def build_adaptive_windows(
    adata_sample,
    *,
    sample: str,
    group_key: str,
    spatial_key: str,
    parameters: WindowParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build adaptive k-nearest-cell grid windows; no graph is constructed."""

    parameters.validate()
    coords = np.asarray(adata_sample.obsm[spatial_key], dtype=float)
    groups = adata_sample.obs[group_key].astype(str).to_numpy()
    obs_names = adata_sample.obs_names.astype(str).to_numpy()
    centers = make_grid_centers(coords, parameters.grid_stride)
    tree = cKDTree(coords)
    distances, members = tree.query(centers, k=min(parameters.adaptive_k, len(coords)))
    if np.ndim(distances) == 1:
        distances = distances[:, None]
        members = members[:, None]
    hits = [np.atleast_1d(row).astype(int) for row in members]
    radii = [float(np.max(np.atleast_1d(row))) for row in distances]
    windows, memberships = _window_tables(
        centers=centers,
        hits=hits,
        groups=groups,
        obs_names=obs_names,
        min_cells=parameters.min_cells,
        sample=sample,
        family=f"grid_adaptive_k{parameters.adaptive_k}",
        window_type="grid_adaptive",
        radii=radii,
    )
    valid_contexts = windows.loc[
        windows["n_groups_ge_min"].ge(parameters.min_required_groups), "context"
    ]
    windows = windows[windows["context"].isin(valid_contexts)].reset_index(drop=True)
    memberships = memberships[memberships["context"].isin(valid_contexts)].reset_index(drop=True)
    return windows, memberships


def _deduplicate_centers(coords: np.ndarray, min_distance: float) -> np.ndarray:
    if len(coords) == 0 or min_distance <= 0:
        return np.arange(len(coords), dtype=int)
    tree = cKDTree(coords)
    blocked = np.zeros(len(coords), dtype=bool)
    keep: list[int] = []
    for index in range(len(coords)):
        if blocked[index]:
            continue
        keep.append(index)
        blocked[tree.query_ball_point(coords[index], r=min_distance)] = True
    return np.asarray(keep, dtype=int)


def build_anchor_windows(
    adata_sample,
    *,
    sample: str,
    group_key: str,
    spatial_key: str,
    parameters: WindowParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build optional sparse-group rescue windows without using a graph."""

    if not parameters.anchor_groups:
        return pd.DataFrame(), pd.DataFrame()
    coords = np.asarray(adata_sample.obsm[spatial_key], dtype=float)
    groups = adata_sample.obs[group_key].astype(str).to_numpy()
    obs_names = adata_sample.obs_names.astype(str).to_numpy()
    tree = cKDTree(coords)
    all_windows: list[pd.DataFrame] = []
    all_memberships: list[pd.DataFrame] = []
    for anchor_group in parameters.anchor_groups:
        anchor_idx = np.flatnonzero(groups == anchor_group)
        if len(anchor_idx) < parameters.anchor_min_cells:
            continue
        selected = _deduplicate_centers(
            coords[anchor_idx], parameters.anchor_dedup_distance
        )
        centers = coords[anchor_idx[selected]]
        distances, members = tree.query(
            centers, k=min(parameters.anchor_k_target, len(coords))
        )
        if np.ndim(distances) == 1:
            distances = distances[:, None]
            members = members[:, None]
        hits = [np.atleast_1d(row).astype(int) for row in members]
        radii = [float(np.max(np.atleast_1d(row))) for row in distances]
        windows, memberships = _window_tables(
            centers=centers,
            hits=hits,
            groups=groups,
            obs_names=obs_names,
            min_cells=parameters.min_cells,
            sample=sample,
            family=f"anchor_{anchor_group}",
            window_type="anchored",
            anchor_group=anchor_group,
            radii=radii,
        )
        valid = (
            windows["n_groups_ge_min"].ge(parameters.min_required_groups)
            & windows["n_anchor_cells"].ge(parameters.anchor_min_cells)
            & windows["effective_radius"].le(parameters.anchor_max_radius)
        )
        valid_contexts = windows.loc[valid, "context"]
        all_windows.append(windows[windows["context"].isin(valid_contexts)])
        all_memberships.append(memberships[memberships["context"].isin(valid_contexts)])
    return (
        pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame(),
        pd.concat(all_memberships, ignore_index=True)
        if all_memberships
        else pd.DataFrame(),
    )


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame["n_cells_total"], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())


def aggregate_window_scores(window_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse windows to families, then families to one vector per sample."""

    if window_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    scores = [column for column in LIANA_SCORE_COLUMNS if column in window_results.columns]
    family_keys = [
        "sample",
        "condition",
        *EDGE_COLUMNS,
        "window_family",
        "window_type",
        "anchor_group",
    ]
    family_rows: list[dict] = []
    for keys, group in window_results.groupby(family_keys, dropna=False, observed=False):
        row = dict(zip(family_keys, keys))
        row["n_windows"] = int(group["context"].nunique())
        row["mean_window_cells"] = float(group["n_cells_total"].mean())
        row["total_window_cells"] = float(group["n_cells_total"].sum())
        for column in scores:
            row[column] = _weighted_mean(group, column)
        family_rows.append(row)
    family = pd.DataFrame(family_rows)

    sample_keys = ["sample", "condition", *EDGE_COLUMNS]
    sample_rows: list[dict] = []
    for keys, group in family.groupby(sample_keys, dropna=False, observed=False):
        row = dict(zip(sample_keys, keys))
        row["n_window_families"] = int(group["window_family"].nunique())
        row["n_windows_total"] = int(group["n_windows"].sum())
        row["supporting_families"] = "|".join(sorted(group["window_family"].astype(str)))
        for column in scores:
            row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        sample_rows.append(row)
    return family, pd.DataFrame(sample_rows)


def audit_liana_input(
    input_h5ad: str | Path,
    *,
    sample_key: str,
    condition_key: str,
    group_key: str,
    spatial_key: str,
    compartment_key: str | None = None,
    cell_types: Sequence[str] | str | None = None,
    compartments: Sequence[str] | str | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Read-only audit after applying the requested pre-window cell scope."""

    import anndata as ad

    adata = ad.read_h5ad(input_h5ad, backed="r")
    selected_cell_types = _selection(cell_types)
    selected_compartments = _selection(compartments)
    if selected_compartments and not compartment_key:
        raise ValueError("compartment_key is required when compartments are selected")
    required = list(
        dict.fromkeys(
            [
                sample_key,
                condition_key,
                group_key,
                *([compartment_key] if compartment_key else []),
            ]
        )
    )
    missing = [column for column in required if column not in adata.obs]
    if missing:
        raise KeyError(f"{input_h5ad}: missing obs columns {missing}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{input_h5ad}: missing obsm[{spatial_key!r}]")
    metadata = adata.obs[required].copy()
    selected_cell_types = _validate_selection(
        metadata[group_key], selected_cell_types, label="cell types"
    )
    if selected_compartments:
        selected_compartments = _validate_selection(
            metadata[compartment_key], selected_compartments, label="compartments"
        )
    mask = np.ones(len(metadata), dtype=bool)
    if selected_cell_types:
        mask &= metadata[group_key].astype(str).isin(selected_cell_types).to_numpy()
    if selected_compartments:
        mask &= metadata[compartment_key].astype(str).isin(selected_compartments).to_numpy()
    metadata = metadata.loc[mask].copy()
    if metadata.empty:
        raise ValueError("The cell-type and compartment filters retain no cells")
    sample_condition = metadata[[sample_key, condition_key]].drop_duplicates()
    if sample_condition.duplicated(sample_key).any():
        raise ValueError("Each biological sample must map to exactly one condition")
    result = {
        "shape": pd.Series(
            {
                "cells_before_scope": adata.n_obs,
                "cells_after_scope": len(metadata),
                "genes": adata.n_vars,
            }
        ),
        "samples": (
            metadata.groupby([condition_key, sample_key], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        ),
        "cell_groups": (
            metadata.groupby([condition_key, group_key], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        ),
    }
    if compartment_key:
        result["compartments"] = (
            metadata.groupby([condition_key, compartment_key], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
    adata.file.close()
    return result


def _load_sample_to_memory(
    adata,
    sample_key: str,
    sample: str,
    *,
    group_key: str,
    cell_types: Sequence[str] | str | None = None,
    compartment_key: str | None = None,
    compartments: Sequence[str] | str | None = None,
):
    mask = adata.obs[sample_key].astype(str).eq(str(sample)).to_numpy()
    selected_cell_types = _selection(cell_types)
    selected_compartments = _selection(compartments)
    if selected_cell_types:
        mask &= adata.obs[group_key].astype(str).isin(selected_cell_types).to_numpy()
    if selected_compartments:
        if not compartment_key:
            raise ValueError("compartment_key is required when compartments are selected")
        mask &= adata.obs[compartment_key].astype(str).isin(selected_compartments).to_numpy()
    if not mask.any():
        raise KeyError(f"Sample {sample!r} has no cells in the requested scope")
    view = adata[mask]
    return view.to_memory() if getattr(view, "isbacked", False) else view.copy()


def run_window_rankagg(
    input_h5ad: str | Path,
    cellchat_csv: str | Path | None,
    output_dir: str | Path,
    *,
    sample_key: str = "sample_id",
    condition_key: str = "condition",
    group_key: str = "cluster_sub",
    spatial_key: str = "spatial",
    compartment_key: str | None = None,
    cell_types: Sequence[str] | str | None = None,
    compartments: Sequence[str] | str | None = None,
    pathways: Sequence[str] | str | None = None,
    annotations: Sequence[str] | str | None = None,
    lr_pairs: Sequence[str] | str | None = None,
    resource_mode: str = "cellchat",
    custom_lr_csv: str | Path | None = None,
    liana_resource_name: str = "mouseconsensus",
    include_non_protein: bool = False,
    parameters: WindowParameters | None = None,
    samples: Sequence[str] | None = None,
    overwrite: bool = False,
    limit_windows_per_sample: int | None = None,
) -> pd.DataFrame:
    """Run/resume LIANA independently for each biological sample."""

    try:
        import anndata as ad
        import liana as li
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "The LIANA stage requires anndata and liana; install the project liana extra."
        ) from exc

    parameters = parameters or WindowParameters()
    parameters.validate()
    input_h5ad = Path(input_h5ad).resolve()
    output_dir = Path(output_dir).resolve()
    sample_root = output_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    resource_mode = str(resource_mode).lower()
    selected_cell_types = _selection(cell_types)
    selected_compartments = _selection(compartments)
    selected_pathways = _selection(pathways)
    selected_annotations = _selection(annotations)
    selected_lr_pairs = _selection(lr_pairs)
    if selected_compartments and not compartment_key:
        raise ValueError("compartment_key is required when compartments are selected")
    resource, resource_metadata = load_rankagg_resource(
        li,
        mode=resource_mode,
        cellchat_csv=cellchat_csv,
        custom_lr_csv=custom_lr_csv,
        liana_resource_name=liana_resource_name,
        pathways=selected_pathways,
        annotations=selected_annotations,
        lr_pairs=selected_lr_pairs,
        include_non_protein=include_non_protein,
    )
    effective_annotations = tuple(
        sorted(resource_metadata["annotation"].dropna().astype(str).unique())
    )

    adata = ad.read_h5ad(input_h5ad, backed="r")
    required = list(
        dict.fromkeys(
            [
                sample_key,
                condition_key,
                group_key,
                *([compartment_key] if compartment_key else []),
            ]
        )
    )
    missing = [column for column in required if column not in adata.obs]
    if missing:
        raise KeyError(f"{input_h5ad}: missing obs columns {missing}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{input_h5ad}: missing obsm[{spatial_key!r}]")
    selected_cell_types = _validate_selection(
        adata.obs[group_key], selected_cell_types, label="cell types"
    )
    if selected_compartments:
        selected_compartments = _validate_selection(
            adata.obs[compartment_key], selected_compartments, label="compartments"
        )
    scope_mask = np.ones(adata.n_obs, dtype=bool)
    if selected_cell_types:
        scope_mask &= adata.obs[group_key].astype(str).isin(selected_cell_types).to_numpy()
    if selected_compartments:
        scope_mask &= (
            adata.obs[compartment_key].astype(str).isin(selected_compartments).to_numpy()
        )
    if not scope_mask.any():
        raise ValueError("The cell-type and compartment filters retain no cells")
    sample_condition = (
        adata.obs.loc[scope_mask, [sample_key, condition_key]]
        .astype(str)
        .drop_duplicates()
    )
    if sample_condition.duplicated(sample_key).any():
        raise ValueError("Each sample must map to one condition")
    condition_lookup = sample_condition.set_index(sample_key)[condition_key].to_dict()
    observed_samples = sample_condition[sample_key].tolist()
    selected_samples = list(samples) if samples is not None else observed_samples
    missing_samples = sorted(set(selected_samples).difference(observed_samples))
    if missing_samples:
        raise KeyError(f"Requested samples absent from the selected cell scope: {missing_samples}")

    selection_manifest = {
        "cell_types": list(selected_cell_types) or None,
        "compartment_key": compartment_key,
        "compartments": list(selected_compartments) or None,
        "resource_mode": resource_mode,
        "custom_lr_csv": str(Path(custom_lr_csv).resolve()) if custom_lr_csv else None,
        "liana_resource_name": liana_resource_name if resource_mode == "liana" else None,
        "cellchat_pathways": list(selected_pathways) or None,
        "cellchat_annotations_requested": list(selected_annotations) or None,
        "cellchat_annotations": list(effective_annotations) or None,
        "cellchat_include_non_protein": (
            bool(include_non_protein) if resource_mode == "cellchat" else None
        ),
        "lr_pairs": list(selected_lr_pairs) or None,
    }

    statuses: list[dict] = []
    for sample in selected_samples:
        condition = condition_lookup[str(sample)]
        sample_dir = sample_root / _slug(sample)
        sample_dir.mkdir(parents=True, exist_ok=True)
        merged_path = sample_dir / "merged_by_sample.parquet"
        manifest_path = sample_dir / "manifest.json"
        if merged_path.exists() and manifest_path.exists() and not overwrite:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("status") == "complete":
                if prior.get("selection", selection_manifest) != selection_manifest:
                    raise RuntimeError(
                        f"Sample {sample}: completed outputs use a different selection; "
                        "choose another output directory or set overwrite=True"
                    )
                statuses.append(
                    {"sample": sample, "condition": condition, "status": "skipped_complete"}
                )
                continue

        sample_adata = _load_sample_to_memory(
            adata,
            sample_key,
            str(sample),
            group_key=group_key,
            cell_types=selected_cell_types,
            compartment_key=compartment_key,
            compartments=selected_compartments,
        )
        sample_adata.obs[group_key] = sample_adata.obs[group_key].astype("category")
        standard_windows, standard_memberships = build_adaptive_windows(
            sample_adata,
            sample=str(sample),
            group_key=group_key,
            spatial_key=spatial_key,
            parameters=parameters,
        )
        anchor_windows, anchor_memberships = build_anchor_windows(
            sample_adata,
            sample=str(sample),
            group_key=group_key,
            spatial_key=spatial_key,
            parameters=parameters,
        )
        window_parts = [frame for frame in [standard_windows, anchor_windows] if not frame.empty]
        membership_parts = [
            frame for frame in [standard_memberships, anchor_memberships] if not frame.empty
        ]
        windows = pd.concat(window_parts, ignore_index=True) if window_parts else pd.DataFrame()
        memberships = (
            pd.concat(membership_parts, ignore_index=True) if membership_parts else pd.DataFrame()
        )
        if windows.empty:
            raise RuntimeError(f"Sample {sample}: no valid adaptive windows")
        if limit_windows_per_sample is not None:
            keep_contexts = windows["context"].head(limit_windows_per_sample)
            windows = windows[windows["context"].isin(keep_contexts)].copy()
            memberships = memberships[memberships["context"].isin(keep_contexts)].copy()

        windows.to_parquet(sample_dir / "window_metadata.parquet", index=False)
        grouped = memberships.groupby("context", sort=False)["obs_name"].agg(list)
        window_meta = windows.set_index("context")
        results: list[pd.DataFrame] = []
        failures: list[dict] = []
        for context, obs_names in grouped.items():
            metadata = window_meta.loc[context]
            context_adata = sample_adata[obs_names].copy()
            context_adata.obs[group_key] = context_adata.obs[group_key].astype("category")
            try:
                li.mt.rank_aggregate(
                    context_adata,
                    groupby=group_key,
                    resource=resource,
                    expr_prop=parameters.expr_prop,
                    min_cells=parameters.min_cells,
                    use_raw=False,
                    spatial_key=spatial_key,
                    spatial_kwargs={
                        "bandwidth": parameters.spatial_bandwidth,
                        "kernel": parameters.spatial_kernel,
                        "trim_fraction": parameters.spatial_trim_fraction,
                    },
                    n_perms=parameters.n_perms,
                    seed=parameters.seed,
                    n_jobs=parameters.n_jobs,
                    return_all_lrs=parameters.return_all_lrs,
                    verbose=False,
                    inplace=True,
                    key_added="liana_window_res",
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                failures.append(
                    {
                        "sample": sample,
                        "condition": condition,
                        "context": context,
                        "error": repr(exc),
                    }
                )
                continue
            result = context_adata.uns.get("liana_window_res")
            if result is None:
                failures.append(
                    {
                        "sample": sample,
                        "condition": condition,
                        "context": context,
                        "error": "LIANA returned no liana_window_res table",
                    }
                )
                continue
            result = pd.DataFrame(result).copy()
            if result.empty:
                continue
            result["sample"] = sample
            result["condition"] = condition
            result["context"] = context
            for column in [
                "window_type",
                "window_family",
                "anchor_group",
                "n_cells_total",
                "n_groups_ge_min",
                "effective_radius",
            ]:
                if column in metadata.index:
                    result[column] = metadata[column]
            anchor_group = metadata.get("anchor_group")
            if pd.notna(anchor_group):
                result = result[
                    result["source"].astype(str).eq(str(anchor_group))
                    | result["target"].astype(str).eq(str(anchor_group))
                ]
            if not result.empty:
                results.append(result)

        by_window = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        family, merged = aggregate_window_scores(by_window)
        if merged.empty:
            pd.DataFrame(failures).to_csv(sample_dir / "window_failures.csv", index=False)
            raise RuntimeError(f"Sample {sample}: LIANA produced no merged sample edges")
        by_window.to_parquet(sample_dir / "rankagg_by_window.parquet", index=False)
        family.to_parquet(sample_dir / "rankagg_by_window_family.parquet", index=False)
        merged.to_parquet(merged_path, index=False)
        pd.DataFrame(failures).to_csv(sample_dir / "window_failures.csv", index=False)
        manifest = {
            "status": "complete",
            "scope": (
                "selected cell types and pooled spatial domains within each sample; "
                "all values when selection fields are null"
            ),
            "selection": selection_manifest,
            "window_estimator": "adaptive grid k-nearest cells; not graph windows",
            "sample": str(sample),
            "condition": str(condition),
            "input_h5ad": str(input_h5ad),
            "input_h5ad_sha256": _sha256(input_h5ad),
            "resource_mode": resource_mode,
            "resource_file": (
                str(Path(cellchat_csv).resolve())
                if resource_mode == "cellchat" and cellchat_csv
                else str(Path(custom_lr_csv).resolve())
                if resource_mode == "custom" and custom_lr_csv
                else None
            ),
            "resource_file_sha256": (
                _sha256(cellchat_csv)
                if resource_mode == "cellchat" and cellchat_csv
                else _sha256(custom_lr_csv)
                if resource_mode == "custom" and custom_lr_csv
                else None
            ),
            "liana_resource_name": (
                liana_resource_name if resource_mode == "liana" else None
            ),
            "resource_rows": int(len(resource)),
            "resource_metadata_rows_total": int(
                resource_metadata.attrs.get("total_database_rows", len(resource_metadata))
            ),
            "resource_metadata_rows_selected": int(len(resource_metadata)),
            "resource_pathways": int(resource_metadata["pathway_name"].nunique()),
            "parameters": asdict(parameters),
            "n_cells": int(sample_adata.n_obs),
            "n_valid_windows": int(windows["context"].nunique()),
            "n_failed_windows": int(len(failures)),
            "n_merged_edges": int(len(merged)),
        }
        _write_json(manifest, manifest_path)
        statuses.append(
            {"sample": sample, "condition": condition, "status": "complete", **manifest}
        )
    adata.file.close()
    status = pd.DataFrame(statuses)
    status.to_csv(output_dir / "sample_status.csv", index=False)
    return status


def liana_window_status(output_dir: str | Path) -> pd.DataFrame:
    """Report artifact-backed per-sample completion."""

    rows: list[dict] = []
    for manifest_path in sorted((Path(output_dir) / "samples").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged_path = manifest_path.parent / "merged_by_sample.parquet"
        rows.append(
            {
                "sample": manifest.get("sample"),
                "condition": manifest.get("condition"),
                "status": manifest.get("status"),
                "merged_output_exists": merged_path.exists(),
                "n_valid_windows": manifest.get("n_valid_windows"),
                "n_failed_windows": manifest.get("n_failed_windows"),
                "n_merged_edges": manifest.get("n_merged_edges"),
                "manifest": str(manifest_path),
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg correction without an additional dependency."""

    values = np.asarray(list(values), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0, 1)
    return result


def exact_partition_count(n_a: int, n_b: int) -> int:
    return math.comb(n_a + n_b, n_a)


def minimum_exact_p(n_a: int, n_b: int) -> float:
    """Minimum attainable exhaustive p; equal groups include complementary splits."""

    partitions = exact_partition_count(n_a, n_b)
    return (2.0 if n_a == n_b else 1.0) / partitions


def exact_mean_difference(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided exact label-permutation test for rows of features."""

    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must contain the same feature rows")
    observed = b.mean(axis=1) - a.mean(axis=1)
    pooled = np.concatenate([a, b], axis=1)
    indices = np.arange(pooled.shape[1])
    assignments = list(combinations(indices, a.shape[1]))
    exceed = np.zeros(a.shape[0], dtype=int)
    for selected in assignments:
        selected = np.asarray(selected, dtype=int)
        other = np.setdiff1d(indices, selected)
        null = pooled[:, other].mean(axis=1) - pooled[:, selected].mean(axis=1)
        exceed += np.abs(null) >= np.abs(observed) - 1e-12
    return observed, exceed / len(assignments)


def rms_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance between sample centroids over pathway edge features."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be sample-by-feature matrices with matching features")
    if a.shape[1] == 0:
        return float("nan")
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(a.shape[1]))


def exact_rms_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Exact label-permutation p-value for pathway RMS centroid distance."""

    observed = rms_centroid_distance(a, b)
    pooled = np.vstack([a, b])
    indices = np.arange(len(pooled))
    null = []
    for selected in combinations(indices, len(a)):
        selected = np.asarray(selected, dtype=int)
        other = np.setdiff1d(indices, selected)
        null.append(rms_centroid_distance(pooled[selected], pooled[other]))
    return observed, float(np.mean(np.asarray(null) >= observed - 1e-12))


def _presence_class(n_a: int, n_b: int, threshold: int) -> str:
    if n_a >= threshold and n_b >= threshold:
        return "supported_both"
    if n_a == 0 and n_b >= threshold:
        return "appears_in_b"
    if n_a >= threshold and n_b == 0:
        return "disappears_in_b"
    if n_b >= threshold:
        return "b_supported_a_sparse"
    if n_a >= threshold:
        return "a_supported_b_sparse"
    return "not_supported_in_pair"


def _condition_samples(observed: pd.DataFrame, conditions: Sequence[str]) -> dict[str, list[str]]:
    mapping = observed[["sample", "condition"]].drop_duplicates()
    if mapping.duplicated("sample").any():
        raise ValueError("Each sample must map to one condition")
    result = {
        condition: sorted(mapping.loc[mapping["condition"].eq(condition), "sample"].astype(str))
        for condition in conditions
    }
    empty = [condition for condition, samples in result.items() if not samples]
    if empty:
        raise ValueError(f"Selected conditions have no samples: {empty}")
    return result


def _annotate_observed(
    observed: pd.DataFrame, cellchat: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = observed.copy()
    observed["_ligand_key"] = observed["ligand_complex"].map(canonical_component)
    observed["_receptor_key"] = observed["receptor_complex"].map(canonical_component)
    metadata = cellchat.copy()
    metadata["ligand"] = metadata["resource_ligand"]
    metadata["receptor"] = metadata["resource_receptor"]
    aggregate = {
        "ligand": "first",
        "receptor": "first",
        "pathway_name": lambda values: "|".join(sorted(set(map(str, values)))),
    }
    for column in ["interaction_name", "interaction_name_2", "annotation"]:
        if column in metadata.columns:
            aggregate[column] = lambda values: "|".join(sorted(set(map(str, values))))
    collapsed = (
        metadata.groupby(["_ligand_key", "_receptor_key"], as_index=False, observed=True)
        .agg(aggregate)
    )
    pathway_counts = (
        metadata.groupby(["_ligand_key", "_receptor_key"], observed=True)["pathway_name"]
        .nunique()
        .rename("n_pathway_mappings")
        .reset_index()
    )
    collapsed = collapsed.merge(
        pathway_counts, on=["_ligand_key", "_receptor_key"], validate="one_to_one"
    )
    collapsed["pathway_ambiguous"] = collapsed["n_pathway_mappings"].gt(1)
    annotated = observed.merge(
        collapsed,
        on=["_ligand_key", "_receptor_key"],
        how="left",
        validate="many_to_one",
    )
    unmapped = annotated[annotated["pathway_name"].isna()].copy()
    missing_pathway = annotated["pathway_name"].isna()
    annotated.loc[missing_pathway, "ligand"] = annotated.loc[
        missing_pathway, "ligand_complex"
    ]
    annotated.loc[missing_pathway, "receptor"] = annotated.loc[
        missing_pathway, "receptor_complex"
    ]
    annotated.loc[missing_pathway, "n_pathway_mappings"] = 0
    annotated.loc[missing_pathway, "pathway_ambiguous"] = False
    annotated["n_pathway_mappings"] = annotated["n_pathway_mappings"].astype(int)
    annotated["pathway_ambiguous"] = annotated["pathway_ambiguous"].astype(bool)
    return annotated, unmapped

def _edge_support(
    observed: pd.DataFrame, conditions: Sequence[str], threshold: int
) -> pd.DataFrame:
    metadata_columns = [
        *EDGE_COLUMNS,
        "ligand",
        "receptor",
        "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
        *[
            column
            for column in ["interaction_name", "interaction_name_2", "annotation"]
            if column in observed.columns
        ],
    ]
    metadata = observed[metadata_columns].drop_duplicates()
    if metadata.duplicated(EDGE_COLUMNS).any():
        raise ValueError("An edge maps to conflicting CellChat metadata")
    counts = (
        observed.groupby([*EDGE_COLUMNS, "condition"], observed=True)["sample"]
        .nunique()
        .unstack("condition", fill_value=0)
        .reindex(columns=conditions, fill_value=0)
        .reset_index()
    )
    counts = counts.rename(columns={condition: f"n_observed__{condition}" for condition in conditions})
    support = metadata.merge(counts, on=EDGE_COLUMNS, validate="one_to_one")
    support["max_condition_support"] = support[
        [f"n_observed__{condition}" for condition in conditions]
    ].max(axis=1)
    support["retained"] = support["max_condition_support"].ge(threshold)
    support = support.sort_values(EDGE_COLUMNS).reset_index(drop=True)
    support.insert(0, "edge_id", [f"E{index:07d}" for index in range(1, len(support) + 1)])
    lr_keys = support[["ligand", "receptor"]].drop_duplicates().sort_values(["ligand", "receptor"])
    lr_keys["lr_id"] = [f"LR{index:05d}" for index in range(1, len(lr_keys) + 1)]
    support = support.merge(lr_keys, on=["ligand", "receptor"], validate="many_to_one")
    pathway_names = sorted(support["pathway_name"].astype(str).unique())
    pathway_ids = {name: f"PW{index:04d}" for index, name in enumerate(pathway_names, 1)}
    support["pathway_id"] = support["pathway_name"].map(pathway_ids)
    return support


def _complete_edges(
    observed: pd.DataFrame,
    support: pd.DataFrame,
    samples: Sequence[str],
    sample_condition: dict[str, str],
    rank_column: str,
    zero_fill_missing: bool = True,
) -> pd.DataFrame:
    retained = support[support["retained"]].copy()
    lookup = retained[["edge_id", *EDGE_COLUMNS]]
    values = observed.merge(lookup, on=EDGE_COLUMNS, how="inner", validate="many_to_one")
    if values.duplicated(["edge_id", "sample"]).any():
        raise ValueError("Observed sample-edge rows are not unique after window aggregation")
    if rank_column not in values.columns:
        raise KeyError(f"Observed LIANA output lacks rank column {rank_column!r}")
    grid = pd.MultiIndex.from_product(
        [retained["edge_id"], samples], names=["edge_id", "sample"]
    ).to_frame(index=False)
    values = values[["edge_id", "sample", rank_column]].copy()
    values["edge_observed"] = True
    completed = grid.merge(values, on=["edge_id", "sample"], how="left", validate="one_to_one")
    completed["edge_observed"] = completed["edge_observed"].eq(True)
    completed["rank_missing"] = ~completed["edge_observed"]
    completed["rank_imputed"] = completed["rank_missing"] & bool(zero_fill_missing)
    completed[rank_column] = pd.to_numeric(completed[rank_column], errors="coerce")
    if zero_fill_missing:
        completed[rank_column] = completed[rank_column].fillna(1.0)
    completed["magnitude_rank"] = completed[rank_column]
    completed["activity"] = 1.0 - completed["magnitude_rank"]
    completed["condition"] = completed["sample"].map(sample_condition)
    metadata_columns = [
        "edge_id",
        "lr_id",
        "pathway_id",
        *EDGE_COLUMNS,
        "ligand",
        "receptor",
        "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
        *[column for column in ["annotation"] if column in retained.columns],
    ]
    return completed.merge(
        retained[metadata_columns], on="edge_id", how="left", validate="many_to_one"
    ).sort_values(["edge_id", "sample"]).reset_index(drop=True)


def _edge_pairwise(
    completed: pd.DataFrame,
    support: pd.DataFrame,
    condition_samples: dict[str, list[str]],
    threshold: int,
    *,
    zero_fill_missing: bool = True,
) -> pd.DataFrame:
    matrix = completed.pivot(index="edge_id", columns="sample", values="activity")
    metadata_columns = [
        "edge_id",
        "lr_id",
        "pathway_id",
        *EDGE_COLUMNS,
        "ligand",
        "receptor",
        "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
        *[column for column in ["annotation"] if column in support.columns],
    ]
    metadata = support.loc[support["retained"], metadata_columns].set_index("edge_id")
    support_index = support.set_index("edge_id").loc[matrix.index]
    rows: list[pd.DataFrame] = []
    for condition_a, condition_b in combinations(condition_samples, 2):
        samples_a = condition_samples[condition_a]
        samples_b = condition_samples[condition_b]
        a = matrix[samples_a].to_numpy(float)
        b = matrix[samples_b].to_numpy(float)
        n_observed_a = support_index[f"n_observed__{condition_a}"].to_numpy(int)
        n_observed_b = support_index[f"n_observed__{condition_b}"].to_numpy(int)
        if zero_fill_missing:
            tested = np.maximum(n_observed_a, n_observed_b) >= threshold
            n_tested_a = np.full(len(matrix), len(samples_a), dtype=int)
            n_tested_b = np.full(len(matrix), len(samples_b), dtype=int)
        else:
            tested = (n_observed_a >= threshold) & (n_observed_b >= threshold)
            n_tested_a = np.isfinite(a).sum(axis=1).astype(int)
            n_tested_b = np.isfinite(b).sum(axis=1).astype(int)

        mean_a = np.divide(
            np.nansum(a, axis=1),
            np.isfinite(a).sum(axis=1),
            out=np.full(len(matrix), np.nan),
            where=np.isfinite(a).sum(axis=1) > 0,
        )
        mean_b = np.divide(
            np.nansum(b, axis=1),
            np.isfinite(b).sum(axis=1),
            out=np.full(len(matrix), np.nan),
            where=np.isfinite(b).sum(axis=1) > 0,
        )
        difference = np.full(len(matrix), np.nan)
        exact_p = np.full(len(matrix), np.nan)
        welch_t = np.full(len(matrix), np.nan)
        welch_p = np.full(len(matrix), np.nan)
        exact_partitions = np.full(len(matrix), np.nan)
        minimum_p = np.full(len(matrix), np.nan)

        pattern_rows: dict[tuple[tuple[bool, ...], tuple[bool, ...]], list[int]] = {}
        for row_index in np.flatnonzero(tested):
            pattern = (
                tuple(np.isfinite(a[row_index]).tolist()),
                tuple(np.isfinite(b[row_index]).tolist()),
            )
            pattern_rows.setdefault(pattern, []).append(int(row_index))
        for (mask_a_tuple, mask_b_tuple), row_indices in pattern_rows.items():
            mask_a = np.asarray(mask_a_tuple, dtype=bool)
            mask_b = np.asarray(mask_b_tuple, dtype=bool)
            selected_a = a[np.ix_(row_indices, mask_a)]
            selected_b = b[np.ix_(row_indices, mask_b)]
            group_difference, group_exact_p = exact_mean_difference(selected_a, selected_b)
            difference[row_indices] = group_difference
            exact_p[row_indices] = group_exact_p
            if mask_a.sum() >= 2 and mask_b.sum() >= 2:
                with np.errstate(invalid="ignore", divide="ignore"):
                    group_welch_t, group_welch_p = ttest_ind(
                        selected_b, selected_a, axis=1, equal_var=False
                    )
                welch_t[row_indices] = group_welch_t
                welch_p[row_indices] = group_welch_p
            exact_partitions[row_indices] = exact_partition_count(
                int(mask_a.sum()), int(mask_b.sum())
            )
            minimum_p[row_indices] = minimum_exact_p(
                int(mask_a.sum()), int(mask_b.sum())
            )

        frame = metadata.loc[matrix.index].reset_index()
        frame["contrast"] = f"{condition_b}_vs_{condition_a}"
        frame["condition_a"] = condition_a
        frame["condition_b"] = condition_b
        frame["n_a"] = len(samples_a)
        frame["n_b"] = len(samples_b)
        frame["n_observed_a"] = n_observed_a
        frame["n_observed_b"] = n_observed_b
        frame["n_tested_a"] = n_tested_a
        frame["n_tested_b"] = n_tested_b
        frame["presence_class"] = [
            _presence_class(left, right, threshold)
            for left, right in zip(n_observed_a, n_observed_b)
        ]
        frame["appearance_or_disappearance"] = frame["presence_class"].isin(
            ["appears_in_b", "disappears_in_b"]
        )
        frame["zero_fill_missing"] = bool(zero_fill_missing)
        frame["tested"] = tested
        frame["mean_activity_a"] = mean_a
        frame["mean_activity_b"] = mean_b
        frame["difference_b_minus_a"] = difference
        frame["abs_difference"] = np.abs(difference)
        frame["higher_condition"] = np.where(
            ~np.isfinite(difference),
            "not_tested",
            np.where(difference > 0, condition_b, np.where(difference < 0, condition_a, "tie")),
        )
        frame["welch_t"] = welch_t
        frame["welch_p"] = welch_p
        frame["exact_permutation_p"] = exact_p
        frame["welch_q"] = bh_adjust(welch_p)
        frame["exact_permutation_q"] = bh_adjust(exact_p)
        frame["exact_partitions"] = exact_partitions
        frame["minimum_attainable_p"] = minimum_p
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _pathway_tables(
    completed: pd.DataFrame,
    edge_pairwise: pd.DataFrame,
    condition_samples: dict[str, list[str]],
    *,
    zero_fill_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activity_matrix = completed.pivot(index="edge_id", columns="sample", values="activity")
    edge_metadata = completed[
        [
            "edge_id",
            "lr_id",
            "pathway_id",
            "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
            *EDGE_COLUMNS,
            "ligand",
            "receptor",
        ]
    ].drop_duplicates("edge_id").set_index("edge_id")
    pathway_rows: list[dict] = []
    sample_rows: list[dict] = []
    edge_driver_rows: list[pd.DataFrame] = []
    for contrast, pair_edges in edge_pairwise.groupby("contrast", sort=False):
        condition_a = str(pair_edges["condition_a"].iloc[0])
        condition_b = str(pair_edges["condition_b"].iloc[0])
        samples_a = condition_samples[condition_a]
        samples_b = condition_samples[condition_b]
        eligible_ids = pair_edges.loc[pair_edges["tested"], "edge_id"]
        if not zero_fill_missing and len(eligible_ids):
            pair_samples = [*samples_a, *samples_b]
            complete_case = activity_matrix.loc[eligible_ids, pair_samples].notna().all(axis=1)
            eligible_ids = complete_case.index[complete_case]
        eligible_meta = edge_metadata.loc[edge_metadata.index.intersection(eligible_ids)]
        eligible_meta = eligible_meta.loc[~eligible_meta["pathway_ambiguous"].astype(bool)]
        for (pathway_id, pathway_name), group in eligible_meta.groupby(
            ["pathway_id", "pathway_name"], observed=True
        ):
            edge_ids = group.index.tolist()
            sample_by_edge = activity_matrix.loc[edge_ids]
            a = sample_by_edge[samples_a].T.to_numpy(float)
            b = sample_by_edge[samples_b].T.to_numpy(float)
            sample_mean_a = a.mean(axis=1)
            sample_mean_b = b.mean(axis=1)
            mean_delta, mean_p = exact_mean_difference(
                sample_mean_a[None, :], sample_mean_b[None, :]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                welch_t, welch_p = ttest_ind(sample_mean_b, sample_mean_a, equal_var=False)
            rms, rms_p = exact_rms_distance(a, b)
            centroid_a = a.mean(axis=0)
            centroid_b = b.mean(axis=0)
            delta = centroid_b - centroid_a
            pathway_rows.append(
                {
                    "contrast": contrast,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                    "n_edges": len(edge_ids),
                    "zero_fill_missing": bool(zero_fill_missing),
                    "edge_missing_policy": (
                        "rank_1_activity_0" if zero_fill_missing else "complete_case_edges"
                    ),
                    "n_lr_pairs": int(group["lr_id"].nunique()),
                    "n_cell_pairs": int(group[["source", "target"]].drop_duplicates().shape[0]),
                    "mean_activity_a": float(sample_mean_a.mean()),
                    "mean_activity_b": float(sample_mean_b.mean()),
                    "mean_activity_difference_b_minus_a": float(mean_delta[0]),
                    "mean_activity_welch_t": float(welch_t),
                    "mean_activity_welch_p": float(welch_p),
                    "mean_activity_exact_p": float(mean_p[0]),
                    "rms_centroid_distance": rms,
                    "rms_exact_p": rms_p,
                    "exact_partitions": exact_partition_count(len(a), len(b)),
                    "minimum_attainable_p": minimum_exact_p(len(a), len(b)),
                }
            )
            for sample, values, condition in [
                *[(sample, a[index], condition_a) for index, sample in enumerate(samples_a)],
                *[(sample, b[index], condition_b) for index, sample in enumerate(samples_b)],
            ]:
                sample_rows.append(
                    {
                        "contrast": contrast,
                        "pathway_id": pathway_id,
                        "pathway_name": pathway_name,
                        "sample": sample,
                        "condition": condition,
                        "pathway_mean_activity": float(values.mean()),
                        "rms_distance_from_a_centroid": float(
                            np.linalg.norm(values - centroid_a) / np.sqrt(len(values))
                        ),
                    }
                )
            drivers = group.reset_index().copy()
            drivers["contrast"] = contrast
            drivers["condition_a"] = condition_a
            drivers["condition_b"] = condition_b
            drivers["mean_activity_a"] = centroid_a
            drivers["mean_activity_b"] = centroid_b
            drivers["difference_b_minus_a"] = delta
            drivers["abs_difference"] = np.abs(delta)
            drivers["squared_difference"] = delta**2
            abs_total = drivers["abs_difference"].sum()
            squared_total = drivers["squared_difference"].sum()
            drivers["absolute_effect_fraction"] = (
                drivers["abs_difference"] / abs_total if abs_total > 0 else 0.0
            )
            drivers["rms_squared_fraction"] = (
                drivers["squared_difference"] / squared_total if squared_total > 0 else 0.0
            )
            drivers["higher_condition"] = np.where(
                drivers["difference_b_minus_a"].gt(0),
                condition_b,
                np.where(drivers["difference_b_minus_a"].lt(0), condition_a, "tie"),
            )
            drivers["driver_rank"] = drivers["abs_difference"].rank(
                method="first", ascending=False
            ).astype(int)
            edge_driver_rows.append(drivers)

    pathway = pd.DataFrame(pathway_rows)
    if not pathway.empty:
        pathway["mean_activity_exact_q"] = pathway.groupby("contrast")[
            "mean_activity_exact_p"
        ].transform(lambda values: bh_adjust(values))
        pathway["mean_activity_welch_q"] = pathway.groupby("contrast")[
            "mean_activity_welch_p"
        ].transform(lambda values: bh_adjust(values))
        pathway["rms_exact_q"] = pathway.groupby("contrast")["rms_exact_p"].transform(
            lambda values: bh_adjust(values)
        )
        pathway["raw_rms_p_lt_0_05"] = pathway["rms_exact_p"].lt(0.05)
        pathway["fdr_rms_q_lt_0_05"] = pathway["rms_exact_q"].lt(0.05)
    edge_drivers = (
        pd.concat(edge_driver_rows, ignore_index=True) if edge_driver_rows else pd.DataFrame()
    )
    if edge_drivers.empty:
        return pathway, pd.DataFrame(sample_rows), edge_drivers, pd.DataFrame(), pd.DataFrame()
    lr_drivers = (
        edge_drivers.groupby(
            [
                "contrast",
                "condition_a",
                "condition_b",
                "pathway_id",
                "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
                "lr_id",
                "ligand",
                "receptor",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_cell_edges=("edge_id", "nunique"),
            signed_effect_sum=("difference_b_minus_a", "sum"),
            absolute_effect_sum=("abs_difference", "sum"),
            rms_squared_sum=("squared_difference", "sum"),
        )
    )
    lr_drivers["absolute_effect_fraction"] = lr_drivers.groupby(
        ["contrast", "pathway_id"]
    )["absolute_effect_sum"].transform(lambda values: values / values.sum() if values.sum() else 0)
    lr_drivers["driver_rank"] = lr_drivers.groupby(["contrast", "pathway_id"])[
        "absolute_effect_sum"
    ].rank(method="first", ascending=False).astype(int)
    cellpair_drivers = (
        edge_drivers.groupby(
            [
                "contrast",
                "condition_a",
                "condition_b",
                "pathway_id",
                "pathway_name",
        "n_pathway_mappings",
        "pathway_ambiguous",
                "source",
                "target",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_lr_edges=("edge_id", "nunique"),
            signed_effect_sum=("difference_b_minus_a", "sum"),
            absolute_effect_sum=("abs_difference", "sum"),
            rms_squared_sum=("squared_difference", "sum"),
        )
    )
    cellpair_drivers["absolute_effect_fraction"] = cellpair_drivers.groupby(
        ["contrast", "pathway_id"]
    )["absolute_effect_sum"].transform(lambda values: values / values.sum() if values.sum() else 0)
    cellpair_drivers["driver_rank"] = cellpair_drivers.groupby(
        ["contrast", "pathway_id"]
    )["absolute_effect_sum"].rank(method="first", ascending=False).astype(int)
    return pathway, pd.DataFrame(sample_rows), edge_drivers, lr_drivers, cellpair_drivers


def analyze_window_rankagg(
    run_dir: str | Path,
    cellchat_csv: str | Path | None,
    analysis_dir: str | Path | None = None,
    *,
    conditions: Sequence[str] | None = None,
    min_condition_samples: int = 3,
    rank_column: str = "magnitude_rank",
    zero_fill_missing: bool = True,
    include_non_protein: bool = False,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Filter, zero-complete, compare, and summarize whole-sample LIANA edges."""

    if min_condition_samples < 1:
        raise ValueError("min_condition_samples must be >= 1")
    run_dir = Path(run_dir).resolve()
    analysis_dir = Path(analysis_dir or run_dir / "analysis").resolve()
    manifest_path = analysis_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Analysis already exists: {manifest_path}; pass overwrite=True")
    parts = sorted((run_dir / "samples").glob("*/merged_by_sample.parquet"))
    if len(parts) < 2:
        raise FileNotFoundError(f"Need at least two completed sample outputs under {run_dir}")
    observed = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    required = {"sample", "condition", rank_column, *EDGE_COLUMNS}
    missing = sorted(required.difference(observed.columns))
    if missing:
        raise KeyError(f"Merged LIANA output missing columns {missing}")
    observed["sample"] = observed["sample"].astype(str)
    observed["condition"] = observed["condition"].astype(str)
    if conditions is None:
        conditions = observed[["sample", "condition"]].drop_duplicates()["condition"].drop_duplicates().tolist()
    conditions = list(conditions)
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required")
    observed = observed[observed["condition"].isin(conditions)].copy()
    condition_samples = _condition_samples(observed, conditions)
    samples = [sample for condition in conditions for sample in condition_samples[condition]]
    sample_condition = {
        sample: condition for condition, condition_sample in condition_samples.items() for sample in condition_sample
    }
    _, cellchat = load_cellchat_resource(
        cellchat_csv,
        include_non_protein=include_non_protein,
    )
    observed, unmapped = _annotate_observed(observed, cellchat)
    support = _edge_support(observed, conditions, min_condition_samples)
    completed = _complete_edges(
        observed,
        support,
        samples,
        sample_condition,
        rank_column,
        zero_fill_missing=zero_fill_missing,
    )
    edge_pairwise = _edge_pairwise(
        completed,
        support,
        condition_samples,
        min_condition_samples,
        zero_fill_missing=zero_fill_missing,
    )
    pathway, pathway_samples, edge_drivers, lr_drivers, cellpair_drivers = _pathway_tables(
        completed,
        edge_pairwise,
        condition_samples,
        zero_fill_missing=zero_fill_missing,
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(analysis_dir / "observed_sample_edges.parquet", index=False)
    unmapped.to_csv(analysis_dir / "unmapped_cellchat_edges.csv", index=False)
    support.to_csv(analysis_dir / "edge_support.csv.gz", index=False, compression="gzip")
    completed.to_parquet(analysis_dir / "completed_sample_edges.parquet", index=False)
    edge_pairwise.to_parquet(analysis_dir / "edge_pairwise.parquet", index=False)
    edge_pairwise.to_csv(
        analysis_dir / "edge_pairwise.csv.gz", index=False, compression="gzip"
    )
    pathway.to_csv(analysis_dir / "pathway_pairwise.csv", index=False)
    pathway_samples.to_csv(analysis_dir / "pathway_sample_activity.csv", index=False)
    edge_drivers.to_parquet(analysis_dir / "pathway_edge_drivers.parquet", index=False)
    edge_drivers.sort_values(["contrast", "pathway_id", "driver_rank"]).groupby(
        ["contrast", "pathway_id"], observed=True
    ).head(25).to_csv(analysis_dir / "pathway_edge_drivers_top25.csv", index=False)
    lr_drivers.to_csv(analysis_dir / "pathway_lr_drivers.csv", index=False)
    cellpair_drivers.to_csv(analysis_dir / "pathway_cellpair_drivers.csv", index=False)
    manifest = {
        "status": "complete",
        "scope": "sample-level outputs; cell and LR selections are recorded in run manifests",
        "conditions": conditions,
        "condition_samples": condition_samples,
        "min_condition_samples": min_condition_samples,
        "rank_column": rank_column,
        "zero_fill_missing": bool(zero_fill_missing),
        "cellchat_include_non_protein": bool(include_non_protein),
        "missing_rank_fill": 1.0 if zero_fill_missing else None,
        "missing_activity_fill": 0.0 if zero_fill_missing else None,
        "raw_score_fields_imputed": False,
        "edge_test": "two-sided exact label permutation; Welch retained as secondary",
        "pathway_tests": [
            "exact permutation of mean pathway activity",
            "exact permutation of RMS centroid distance over retained edge vectors",
        ],
        "multiple_testing": "Benjamini-Hochberg separately within each contrast and statistic",
        "n_observed_edges": int(observed[EDGE_COLUMNS].drop_duplicates().shape[0]),
        "n_retained_edges": int(support["retained"].sum()),
        "n_completed_rows": int(len(completed)),
        "n_imputed_rows": int(completed["rank_imputed"].sum()),
        "n_missing_rows": int(completed["rank_missing"].sum()),
        "n_edge_pairwise_rows": int(len(edge_pairwise)),
        "n_pathway_pairwise_rows": int(len(pathway)),
        "exact_test_resolution": {
            f"{condition_b}_vs_{condition_a}": {
                "partitions": exact_partition_count(
                    len(condition_samples[condition_a]), len(condition_samples[condition_b])
                ),
                "minimum_p": minimum_exact_p(
                    len(condition_samples[condition_a]), len(condition_samples[condition_b])
                ),
            }
            for condition_a, condition_b in combinations(conditions, 2)
        },
    }
    _write_json(manifest, manifest_path)
    return {
        "edge_support": support,
        "completed_edges": completed,
        "edge_pairwise": edge_pairwise,
        "pathway_pairwise": pathway,
        "pathway_samples": pathway_samples,
        "pathway_edge_drivers": edge_drivers,
        "pathway_lr_drivers": lr_drivers,
        "pathway_cellpair_drivers": cellpair_drivers,
    }


def load_window_analysis(analysis_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the compact review tables written by :func:`analyze_window_rankagg`."""

    analysis_dir = Path(analysis_dir)
    required = {
        "edge_pairwise": analysis_dir / "edge_pairwise.parquet",
        "pathway_pairwise": analysis_dir / "pathway_pairwise.csv",
        "pathway_samples": analysis_dir / "pathway_sample_activity.csv",
        "pathway_edge_drivers": analysis_dir / "pathway_edge_drivers.parquet",
        "pathway_lr_drivers": analysis_dir / "pathway_lr_drivers.csv",
        "pathway_cellpair_drivers": analysis_dir / "pathway_cellpair_drivers.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIANA analysis outputs: {missing}")
    return {
        name: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        for name, path in required.items()
    }


def prepare_chord_input(
    edge_drivers: pd.DataFrame,
    *,
    contrast: str,
    pathway_name: str,
    top_n: int = 15,
    focal: str | None = None,
) -> pd.DataFrame:
    """Translate pathway edge drivers to the established R/circlize schema."""

    required = {
        "contrast",
        "pathway_name",
        "source",
        "target",
        "ligand",
        "receptor",
        "abs_difference",
        "higher_condition",
    }
    missing = sorted(required.difference(edge_drivers.columns))
    if missing:
        raise KeyError(f"Driver table missing chord columns {missing}")
    selected = edge_drivers[
        edge_drivers["contrast"].eq(contrast)
        & edge_drivers["pathway_name"].eq(pathway_name)
    ].nlargest(top_n, "abs_difference")
    selected = selected[selected["abs_difference"].gt(0)].copy()
    if selected.empty:
        raise ValueError(f"No positive driver weights for {contrast} / {pathway_name}")
    output = selected[
        ["source", "target", "ligand", "receptor", "abs_difference", "higher_condition"]
    ].rename(columns={"abs_difference": "weight_abs"})
    if focal is not None:
        output["focal"] = focal
    return output.reset_index(drop=True)


def render_chord(
    chord_input: pd.DataFrame,
    *,
    script_path: str | Path,
    input_csv: str | Path,
    output_path: str | Path,
    title: str,
    rscript_bin: str = "Rscript",
    color_mode: str = "condition",
) -> Path:
    """Write chord input and call the shared directional big-arrow renderer."""

    input_csv = Path(input_csv)
    output_path = Path(output_path)
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chord_input.to_csv(input_csv, index=False)
    subprocess.run(
        [
            rscript_bin,
            str(Path(script_path)),
            str(input_csv),
            str(output_path),
            title,
            color_mode,
        ],
        check=True,
    )
    if not output_path.exists():
        raise RuntimeError(f"Chord renderer did not create {output_path}")
    return output_path
