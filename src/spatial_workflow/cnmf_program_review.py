"""Condition-blind review helpers for cNMF program admission.

These helpers deliberately omit condition and treatment fields. They support
biological review of high-usage cells, spatial localization, expected-gene
support, and neighboring-cell context before a program whitelist is frozen.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


DEFAULT_SAFE_COLUMNS = (
    "cluster_sub",
    "celltype_short",
    "celltype_full",
    "spatial_domain",
    "nCount_Xenium",
    "nFeature_Xenium",
)

FORBIDDEN_REVIEW_COLUMNS = {
    "condition",
    "exposure_group",
    "treatment_group",
    "group",
    "sample",
    "orig.ident",
}

REVIEW_DECISIONS = {
    "keep_primary",
    "keep_sensitivity",
    "exclude",
    "needs_followup",
}

REVIEW_COLUMNS = [
    "lineage",
    "program",
    "review_decision",
    "review_confidence",
    "review_notes",
    "reviewer",
    "top_fraction",
    "neighbor_radius_um",
    "whitelist_sha256",
    "reviewed_at_utc",
]


def _usage_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        str(column)
        for column in frame.columns
        if str(column).startswith("Usage_")
    ]
    return sorted(columns, key=lambda value: int(value.split("_", 1)[1]))


def _validate_fraction(value: float) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and at most 1")
    return fraction


def blind_sample_labels(
    values: Sequence[Any] | pd.Series,
    *,
    salt: str = "cnmf-program-review-v1",
) -> dict[str, str]:
    """Map real sample IDs to deterministic, non-condition aliases."""

    samples = sorted(
        {
            str(value)
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    if not samples:
        raise ValueError("No non-missing sample IDs are available")
    ranked = sorted(
        samples,
        key=lambda sample: hashlib.sha256(
            f"{salt}\0{sample}".encode("utf-8")
        ).hexdigest(),
    )
    return {
        sample: f"Section_{index:02d}"
        for index, sample in enumerate(ranked, start=1)
    }


def prepare_blinded_usage_obs(
    obs: pd.DataFrame,
    *,
    sample_key: str,
    usage_cols: Sequence[str] | None = None,
    safe_columns: Sequence[str] = DEFAULT_SAFE_COLUMNS,
    salt: str = "cnmf-program-review-v1",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a review table without sample IDs or experimental labels."""

    if sample_key not in obs:
        raise KeyError(f"Missing sample column {sample_key!r}")
    forbidden_requested = sorted(
        set(map(str, safe_columns)).intersection(FORBIDDEN_REVIEW_COLUMNS)
    )
    if forbidden_requested:
        raise ValueError(
            "Condition-blind review cannot include: "
            + ", ".join(forbidden_requested)
        )
    programs = list(usage_cols or _usage_columns(obs))
    if not programs:
        raise ValueError("No Usage_ columns are available")
    missing = [
        column
        for column in [*safe_columns, *programs]
        if column not in obs
    ]
    if missing:
        raise KeyError("Missing review columns: " + ", ".join(missing))
    mapping = blind_sample_labels(obs[sample_key], salt=salt)
    aliases = obs[sample_key].astype("string").map(mapping)
    if aliases.isna().any():
        raise ValueError("Some cells have missing or blank sample IDs")
    review = obs.loc[:, [*safe_columns, *programs]].copy()
    review.insert(0, "blinded_sample", aliases.astype(str).to_numpy())
    review.index = obs.index.astype(str)
    review.index.name = "cell_id"
    if set(review.columns).intersection(FORBIDDEN_REVIEW_COLUMNS):
        raise AssertionError("A forbidden experimental field entered review data")
    return review, mapping


def select_high_usage_cells(
    review_obs: pd.DataFrame,
    *,
    program: str,
    top_fraction: float = 0.05,
    extra_columns: Sequence[str] = DEFAULT_SAFE_COLUMNS,
) -> pd.DataFrame:
    """Return an exact top fraction of cells ranked by one program usage."""

    fraction = _validate_fraction(top_fraction)
    required = ["blinded_sample", program, *extra_columns]
    missing = [column for column in required if column not in review_obs]
    if missing:
        raise KeyError("Missing high-usage columns: " + ", ".join(missing))
    if not review_obs.index.is_unique:
        raise ValueError("Review cell IDs are not unique")
    values = pd.to_numeric(review_obs[program], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    if not valid.any():
        raise ValueError(f"{program} has no finite usage values")
    n_top = max(1, int(math.ceil(int(valid.sum()) * fraction)))
    ordered = (
        pd.DataFrame(
            {
                "cell_id": review_obs.index.astype(str),
                "usage": values.to_numpy(),
            },
            index=review_obs.index,
        )
        .rename_axis(None)
        .loc[valid]
        .sort_values(["usage", "cell_id"], ascending=[False, True], kind="stable")
        .head(n_top)
    )
    result = review_obs.loc[
        ordered.index, ["blinded_sample", *extra_columns]
    ].copy()
    result.insert(0, "program", str(program))
    result.insert(1, "usage", ordered["usage"].to_numpy(dtype=float))
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=int))
    result.insert(0, "cell_id", result.index.astype(str))
    return result.reset_index(drop=True)


def program_usage_qc_summary(
    review_obs: pd.DataFrame,
    *,
    usage_cols: Sequence[str] | None = None,
    top_fraction: float = 0.05,
) -> pd.DataFrame:
    """Summarize distribution and section concentration without conditions."""

    fraction = _validate_fraction(top_fraction)
    programs = list(usage_cols or _usage_columns(review_obs))
    if "blinded_sample" not in review_obs:
        raise KeyError("review_obs must contain blinded_sample")
    rows: list[dict[str, Any]] = []
    for program in programs:
        top = select_high_usage_cells(
            review_obs,
            program=program,
            top_fraction=fraction,
            extra_columns=(),
        )
        section_counts = top["blinded_sample"].value_counts()
        proportions = section_counts.to_numpy(dtype=float) / len(top)
        values = pd.to_numeric(review_obs[program], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "program": program,
                "n_cells": int(len(finite)),
                "mean_usage": float(finite.mean()),
                "median_usage": float(finite.median()),
                "q90_usage": float(finite.quantile(0.90)),
                "q95_usage": float(finite.quantile(0.95)),
                "q99_usage": float(finite.quantile(0.99)),
                "max_usage": float(finite.max()),
                "top_fraction": fraction,
                "n_top_cells": int(len(top)),
                "n_top_sections": int(len(section_counts)),
                "largest_section_fraction": float(proportions.max()),
                "effective_top_sections": float(1.0 / np.square(proportions).sum()),
            }
        )
    return pd.DataFrame(rows)


def program_usage_diagnostic_metrics(
    review_obs: pd.DataFrame,
    *,
    usage_cols: Sequence[str] | None = None,
    top_fraction: float = 0.05,
    subtype_key: str = "cluster_sub",
) -> pd.DataFrame:
    """Add variability, redundancy, QC, and subtype metrics for review.

    The input must already be condition-blinded. These diagnostics describe
    whether a program is measurable and reviewable; they do not decide whether
    it is biological or predictive.
    """

    if set(review_obs.columns).intersection(FORBIDDEN_REVIEW_COLUMNS):
        raise ValueError("review_obs contains forbidden experimental columns")
    programs = list(usage_cols or _usage_columns(review_obs))
    if not programs:
        raise ValueError("No Usage_ columns are available")
    missing = [program for program in programs if program not in review_obs]
    if missing:
        raise KeyError("Missing usage columns: " + ", ".join(missing))
    if subtype_key not in review_obs:
        raise KeyError(f"Missing subtype column {subtype_key!r}")

    summary = program_usage_qc_summary(
        review_obs,
        usage_cols=programs,
        top_fraction=top_fraction,
    ).set_index("program")
    numeric = review_obs.loc[:, programs].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for program in programs:
        values = numeric[program]
        finite = values[np.isfinite(values)]
        high = select_high_usage_cells(
            review_obs,
            program=program,
            top_fraction=top_fraction,
            extra_columns=(),
        )
        high_ids = pd.Index(high["cell_id"].astype(str))
        subtype_values = (
            review_obs.loc[high_ids, subtype_key]
            .astype("string")
            .dropna()
            .astype(str)
        )
        subtype_counts = subtype_values.value_counts()
        subtype_proportions = (
            subtype_counts.to_numpy(dtype=float) / subtype_counts.sum()
            if len(subtype_counts)
            else np.array([], dtype=float)
        )
        pairwise = correlations.loc[program].drop(index=program).dropna()
        largest_pairwise_program = (
            str(pairwise.idxmax()) if len(pairwise) else ""
        )
        max_abs_pairwise_program = (
            str(pairwise.abs().idxmax()) if len(pairwise) else ""
        )

        row = summary.loc[program].to_dict()
        row.update(
            {
                "program": program,
                "sd_usage": float(finite.std(ddof=1)),
                "iqr_usage": float(
                    finite.quantile(0.75) - finite.quantile(0.25)
                ),
                "nonzero_fraction": float((finite > 0).mean()),
                "largest_pairwise_spearman_program": largest_pairwise_program,
                "largest_pairwise_spearman": (
                    float(pairwise.loc[largest_pairwise_program])
                    if largest_pairwise_program
                    else np.nan
                ),
                "max_abs_pairwise_spearman_program": max_abs_pairwise_program,
                "max_abs_pairwise_spearman": (
                    float(pairwise.loc[max_abs_pairwise_program])
                    if max_abs_pairwise_program
                    else np.nan
                ),
                "spearman_nCount_Xenium": (
                    float(
                        values.corr(
                            pd.to_numeric(
                                review_obs["nCount_Xenium"], errors="coerce"
                            ),
                            method="spearman",
                        )
                    )
                    if "nCount_Xenium" in review_obs
                    else np.nan
                ),
                "spearman_nFeature_Xenium": (
                    float(
                        values.corr(
                            pd.to_numeric(
                                review_obs["nFeature_Xenium"], errors="coerce"
                            ),
                            method="spearman",
                        )
                    )
                    if "nFeature_Xenium" in review_obs
                    else np.nan
                ),
                "n_top_subtypes": int(len(subtype_counts)),
                "largest_top_subtype_fraction": (
                    float(subtype_proportions.max())
                    if len(subtype_proportions)
                    else np.nan
                ),
                "effective_top_subtypes": (
                    float(1.0 / np.square(subtype_proportions).sum())
                    if len(subtype_proportions)
                    else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def top_usage_spatial_knn_enrichment(
    review_obs: pd.DataFrame,
    spatial: np.ndarray,
    *,
    program: str,
    top_fraction: float = 0.05,
    n_neighbors: int = 10,
) -> dict[str, Any]:
    """Measure local clustering of top-usage cells within the source lineage.

    Nearest neighbors are found separately inside each blinded section. The
    observed fraction of top-usage neighbors is divided by the section-level
    fraction expected under spatial mixing. Values above one indicate local
    clustering, but do not distinguish intrinsic biology from spillover.
    """

    if set(review_obs.columns).intersection(FORBIDDEN_REVIEW_COLUMNS):
        raise ValueError("review_obs contains forbidden experimental columns")
    if "blinded_sample" not in review_obs:
        raise KeyError("review_obs must contain blinded_sample")
    if program not in review_obs:
        raise KeyError(f"Missing program {program!r}")
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be positive")
    coords = np.asarray(spatial)
    if coords.ndim != 2 or coords.shape[0] != len(review_obs) or coords.shape[1] < 2:
        raise ValueError(
            "spatial must have one row per review cell and at least 2 columns"
        )

    high = select_high_usage_cells(
        review_obs,
        program=program,
        top_fraction=top_fraction,
        extra_columns=(),
    )
    review_index = pd.Index(review_obs.index.astype(str))
    high_rows = review_index.get_indexer(pd.Index(high["cell_id"].astype(str)))
    if np.any(high_rows < 0):
        raise KeyError("Some high-usage cells are absent from review_obs")
    top_mask = np.zeros(len(review_obs), dtype=bool)
    top_mask[high_rows] = True
    sections = review_obs["blinded_sample"].astype(str).to_numpy()
    observed_top_neighbors = 0.0
    expected_top_neighbors = 0.0
    evaluated_neighbor_edges = 0
    evaluated_top_cells = 0

    for section in sorted(set(sections[high_rows])):
        section_rows = np.flatnonzero(sections == section)
        if len(section_rows) < 2:
            continue
        section_top_local = np.flatnonzero(top_mask[section_rows])
        if not len(section_top_local):
            continue
        tree = cKDTree(coords[section_rows, :2])
        query_k = min(int(n_neighbors) + 1, len(section_rows))
        _, neighbor_local = tree.query(
            coords[section_rows[section_top_local], :2],
            k=query_k,
        )
        neighbor_local = np.atleast_2d(neighbor_local)
        section_expected = float(
            (len(section_top_local) - 1) / (len(section_rows) - 1)
        )
        for focal_local, candidates in zip(section_top_local, neighbor_local):
            kept = [
                int(candidate)
                for candidate in np.atleast_1d(candidates)
                if int(candidate) != int(focal_local)
            ][: int(n_neighbors)]
            if not kept:
                continue
            n_kept = len(kept)
            observed_top_neighbors += float(
                top_mask[section_rows[np.asarray(kept, dtype=int)]].sum()
            )
            expected_top_neighbors += n_kept * section_expected
            evaluated_neighbor_edges += n_kept
            evaluated_top_cells += 1

    observed_fraction = (
        observed_top_neighbors / evaluated_neighbor_edges
        if evaluated_neighbor_edges
        else np.nan
    )
    expected_fraction = (
        expected_top_neighbors / evaluated_neighbor_edges
        if evaluated_neighbor_edges
        else np.nan
    )
    enrichment = (
        observed_top_neighbors / expected_top_neighbors
        if expected_top_neighbors > 0
        else np.nan
    )
    return {
        "program": str(program),
        "spatial_knn_neighbors": int(n_neighbors),
        "spatial_knn_evaluated_top_cells": int(evaluated_top_cells),
        "spatial_knn_evaluated_edges": int(evaluated_neighbor_edges),
        "spatial_knn_observed_top_fraction": float(observed_fraction),
        "spatial_knn_expected_top_fraction": float(expected_fraction),
        "spatial_knn_enrichment": float(enrichment),
    }


def program_spatial_knn_enrichment(
    review_obs: pd.DataFrame,
    spatial: np.ndarray,
    *,
    usage_cols: Sequence[str] | None = None,
    top_fraction: float = 0.05,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Compute top-usage spatial enrichment using one shared neighbor graph.

    This is the all-program counterpart of
    :func:`top_usage_spatial_knn_enrichment`. It constructs the section-scoped
    nearest-neighbor index once, then evaluates every usage program against it.
    """

    if set(review_obs.columns).intersection(FORBIDDEN_REVIEW_COLUMNS):
        raise ValueError("review_obs contains forbidden experimental columns")
    if "blinded_sample" not in review_obs:
        raise KeyError("review_obs must contain blinded_sample")
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be positive")
    programs = list(usage_cols or _usage_columns(review_obs))
    if not programs:
        raise ValueError("No Usage_ columns are available")
    coords = np.asarray(spatial)
    if coords.ndim != 2 or coords.shape[0] != len(review_obs) or coords.shape[1] < 2:
        raise ValueError(
            "spatial must have one row per review cell and at least 2 columns"
        )

    n_cells = len(review_obs)
    neighbor_rows = np.full(
        (n_cells, int(n_neighbors)),
        -1,
        dtype=np.int64,
    )
    sections = review_obs["blinded_sample"].astype(str).to_numpy()
    for section in sorted(set(sections)):
        section_rows = np.flatnonzero(sections == section)
        if len(section_rows) < 2:
            continue
        query_k = min(int(n_neighbors) + 1, len(section_rows))
        tree = cKDTree(coords[section_rows, :2])
        _, local_neighbors = tree.query(
            coords[section_rows, :2],
            k=query_k,
        )
        local_neighbors = np.atleast_2d(local_neighbors)
        local_rows = np.arange(len(section_rows), dtype=int)[:, None]
        keep = local_neighbors != local_rows
        kept_rank = np.cumsum(keep, axis=1) - 1
        keep &= kept_rank < int(n_neighbors)
        source_local = np.broadcast_to(local_rows, keep.shape)[keep]
        destination_column = kept_rank[keep]
        destination_local = local_neighbors[keep]
        neighbor_rows[
            section_rows[source_local], destination_column
        ] = section_rows[destination_local]

    review_index = pd.Index(review_obs.index.astype(str))
    rows: list[dict[str, Any]] = []
    for program in programs:
        high = select_high_usage_cells(
            review_obs,
            program=program,
            top_fraction=top_fraction,
            extra_columns=(),
        )
        high_rows = review_index.get_indexer(
            pd.Index(high["cell_id"].astype(str))
        )
        top_mask = np.zeros(n_cells, dtype=bool)
        top_mask[high_rows] = True
        observed_top_neighbors = 0.0
        expected_top_neighbors = 0.0
        evaluated_neighbor_edges = 0
        evaluated_top_cells = 0
        for section in sorted(set(sections[high_rows])):
            section_rows = np.flatnonzero(sections == section)
            section_top_rows = section_rows[top_mask[section_rows]]
            if not len(section_top_rows) or len(section_rows) < 2:
                continue
            focal_neighbors = neighbor_rows[section_top_rows]
            valid = focal_neighbors >= 0
            n_edges = int(valid.sum())
            if not n_edges:
                continue
            observed_top_neighbors += float(
                top_mask[focal_neighbors[valid]].sum()
            )
            section_expected = float(
                (len(section_top_rows) - 1) / (len(section_rows) - 1)
            )
            expected_top_neighbors += n_edges * section_expected
            evaluated_neighbor_edges += n_edges
            evaluated_top_cells += int(np.any(valid, axis=1).sum())
        rows.append(
            {
                "program": str(program),
                "spatial_knn_neighbors": int(n_neighbors),
                "spatial_knn_evaluated_top_cells": evaluated_top_cells,
                "spatial_knn_evaluated_edges": evaluated_neighbor_edges,
                "spatial_knn_observed_top_fraction": (
                    observed_top_neighbors / evaluated_neighbor_edges
                    if evaluated_neighbor_edges
                    else np.nan
                ),
                "spatial_knn_expected_top_fraction": (
                    expected_top_neighbors / evaluated_neighbor_edges
                    if evaluated_neighbor_edges
                    else np.nan
                ),
                "spatial_knn_enrichment": (
                    observed_top_neighbors / expected_top_neighbors
                    if expected_top_neighbors > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_blinded_spatial_usage(
    review_obs: pd.DataFrame,
    spatial: np.ndarray,
    *,
    program: str,
    blinded_sample: str,
    point_size: float = 3.0,
):
    """Return one interactive spatial map labeled only by blinded section."""

    import plotly.express as px

    if program not in review_obs:
        raise KeyError(f"Missing program {program!r}")
    coords = np.asarray(spatial)
    if coords.ndim != 2 or coords.shape[0] != len(review_obs) or coords.shape[1] < 2:
        raise ValueError("spatial must have one row per review cell and at least 2 columns")
    mask = review_obs["blinded_sample"].astype(str).eq(str(blinded_sample)).to_numpy()
    if not mask.any():
        raise ValueError(f"Unknown blinded section {blinded_sample!r}")
    plot = pd.DataFrame(
        {
            "x": coords[mask, 0],
            "y": coords[mask, 1],
            program: pd.to_numeric(
                review_obs.loc[mask, program], errors="coerce"
            ).to_numpy(),
            "cell_id": review_obs.index[mask].astype(str),
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
        title=f"{program}: {blinded_sample}",
    )
    figure.update_traces(marker={"size": point_size, "opacity": 0.85})
    figure.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    figure.update_layout(template="plotly_white", height=700, width=800)
    return figure


def high_usage_gene_support(
    adata: Any,
    *,
    high_cell_ids: Sequence[str],
    genes: Sequence[str],
    max_background_cells: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare raw-count support in high-usage versus other lineage cells."""

    if not adata.obs_names.is_unique:
        raise ValueError("AnnData cell IDs must be unique")
    high_ids = pd.Index(map(str, high_cell_ids))
    high_rows = pd.Index(adata.obs_names.astype(str)).get_indexer(high_ids)
    if np.any(high_rows < 0):
        raise KeyError(f"{int(np.sum(high_rows < 0))} high-usage cells are absent")
    available_genes = pd.Index(adata.var_names.astype(str))
    shown_genes = [str(gene) for gene in genes if str(gene) in available_genes]
    if not shown_genes:
        raise KeyError("None of the requested genes are present")
    gene_rows = available_genes.get_indexer(shown_genes)
    all_rows = np.arange(adata.n_obs, dtype=int)
    background = np.setdiff1d(all_rows, np.unique(high_rows), assume_unique=False)
    if len(background) > int(max_background_cells):
        rng = np.random.default_rng(random_state)
        background = np.sort(
            rng.choice(background, size=int(max_background_cells), replace=False)
        )

    def materialize(rows: np.ndarray) -> np.ndarray:
        matrix = adata[rows, gene_rows].X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=float)

    high_matrix = materialize(np.unique(high_rows))
    background_matrix = materialize(background)
    high_mean = high_matrix.mean(axis=0)
    background_mean = background_matrix.mean(axis=0)
    result = pd.DataFrame(
        {
            "gene": shown_genes,
            "high_usage_detection_fraction": (high_matrix > 0).mean(axis=0),
            "background_detection_fraction": (background_matrix > 0).mean(axis=0),
            "high_usage_mean_count": high_mean,
            "background_mean_count": background_mean,
            "log2_mean_count_ratio": np.log2(
                (high_mean + 0.1) / (background_mean + 0.1)
            ),
            "n_high_usage_cells": int(high_matrix.shape[0]),
            "n_background_cells": int(background_matrix.shape[0]),
        }
    )
    result.attrs["missing_genes"] = [
        str(gene) for gene in genes if str(gene) not in available_genes
    ]
    return result.sort_values(
        "log2_mean_count_ratio", ascending=False, kind="stable"
    ).reset_index(drop=True)


def high_usage_neighbor_context(
    master_obs: pd.DataFrame,
    spatial: np.ndarray,
    *,
    focal_cell_ids: Sequence[str],
    sample_key: str,
    cell_type_key: str,
    radius: float = 30.0,
) -> pd.DataFrame:
    """Summarize neighboring cell types around high-usage focal cells."""

    if float(radius) <= 0:
        raise ValueError("radius must be positive")
    for column in (sample_key, cell_type_key):
        if column not in master_obs:
            raise KeyError(f"Missing master column {column!r}")
    if not master_obs.index.is_unique:
        raise ValueError("Master cell IDs must be unique")
    coords = np.asarray(spatial)
    if coords.ndim != 2 or coords.shape[0] != len(master_obs) or coords.shape[1] < 2:
        raise ValueError("spatial must have one row per master cell and at least 2 columns")
    focal_ids = pd.Index(map(str, focal_cell_ids)).drop_duplicates()
    master_index = pd.Index(master_obs.index.astype(str))
    focal_rows = master_index.get_indexer(focal_ids)
    if np.any(focal_rows < 0):
        raise KeyError(f"{int(np.sum(focal_rows < 0))} focal cells are absent")

    occurrence_counts: Counter[str] = Counter()
    focal_presence_counts: Counter[str] = Counter()
    sample_values = master_obs[sample_key].astype(str).to_numpy()
    cell_types = (
        master_obs[cell_type_key]
        .astype("string")
        .fillna("<missing>")
        .astype(str)
        .to_numpy()
    )
    n_focal = 0
    for sample in sorted(set(sample_values[focal_rows])):
        sample_rows = np.flatnonzero(sample_values == sample)
        row_to_local = {int(row): index for index, row in enumerate(sample_rows)}
        sample_focal_rows = [
            int(row) for row in focal_rows if sample_values[row] == sample
        ]
        if not sample_focal_rows:
            continue
        tree = cKDTree(coords[sample_rows, :2])
        focal_local = np.array(
            [row_to_local[row] for row in sample_focal_rows], dtype=int
        )
        neighbors = tree.query_ball_point(coords[sample_focal_rows, :2], r=float(radius))
        for focal_position, neighbor_local in zip(focal_local, neighbors):
            neighbor_local = [
                int(value) for value in neighbor_local if int(value) != int(focal_position)
            ]
            labels = cell_types[sample_rows[neighbor_local]]
            occurrence_counts.update(labels)
            focal_presence_counts.update(set(labels))
            n_focal += 1
    if n_focal == 0:
        raise ValueError("No focal cells were available for neighbor review")
    rows = [
        {
            "neighbor_cell_type": label,
            "n_neighbor_occurrences": int(count),
            "n_focal_cells_with_neighbor": int(focal_presence_counts[label]),
            "fraction_focal_cells_with_neighbor": float(
                focal_presence_counts[label] / n_focal
            ),
            "mean_neighbors_per_focal_cell": float(count / n_focal),
            "n_focal_cells": int(n_focal),
            "radius": float(radius),
        }
        for label, count in occurrence_counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["n_focal_cells_with_neighbor", "n_neighbor_occurrences"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def build_review_decision(
    *,
    lineage: str,
    program: str,
    review_decision: str,
    review_confidence: str,
    review_notes: str,
    reviewer: str,
    top_fraction: float,
    neighbor_radius_um: float,
    whitelist_sha256: str,
) -> dict[str, Any]:
    """Build one validated, timestamped human review record."""

    if review_decision not in REVIEW_DECISIONS:
        raise ValueError(
            f"review_decision must be one of {sorted(REVIEW_DECISIONS)}"
        )
    if review_confidence not in {"low", "medium", "high"}:
        raise ValueError("review_confidence must be low, medium, or high")
    _validate_fraction(top_fraction)
    if float(neighbor_radius_um) <= 0:
        raise ValueError("neighbor_radius_um must be positive")
    return {
        "lineage": str(lineage),
        "program": str(program),
        "review_decision": str(review_decision),
        "review_confidence": str(review_confidence),
        "review_notes": str(review_notes),
        "reviewer": str(reviewer),
        "top_fraction": float(top_fraction),
        "neighbor_radius_um": float(neighbor_radius_um),
        "whitelist_sha256": str(whitelist_sha256),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def upsert_review_decision(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Atomically add or replace one lineage-program review decision."""

    path = Path(path)
    missing = [column for column in REVIEW_COLUMNS if column not in record]
    if missing:
        raise KeyError("Missing review-record fields: " + ", ".join(missing))
    incoming = pd.DataFrame([{column: record[column] for column in REVIEW_COLUMNS}])
    if path.exists():
        current = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if list(current.columns) != REVIEW_COLUMNS:
            raise ValueError("Existing review decision columns do not match contract")
        keep = ~(
            current["lineage"].eq(str(record["lineage"]))
            & current["program"].eq(str(record["program"]))
        )
        retained = current.loc[keep]
        result = (
            incoming
            if retained.empty
            else pd.concat([retained, incoming], ignore_index=True)
        )
    else:
        result = incoming
    result = result.sort_values(["lineage", "program"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        result.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
