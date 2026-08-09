"""Small loading and plotting helpers for spatial-overview review notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .nncomp import (
    ABUNDANCE_FILE,
    MANIFEST_FILE,
    NEIGHBOR_FILE,
    OVERLAP_FILE,
    PERMUTATION_FILE,
    PERMUTATION_PLAN_FILE,
    RANKED_EDGES_FILE,
    RECIPROCAL_CELLTYPE_FILE,
    SUMMARY_FILE,
)


def load_autok_outputs(output_dir: str | Path, output_name: str) -> dict[str, Any]:
    """Load CellCharter K-selection results for the review notebook."""

    root = Path(output_dir)
    stem = output_name.removesuffix(".h5ad")
    with (root / f"{stem}.summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    outputs = {
        "stability": pd.read_csv(root / f"{stem}.autok_stability.csv"),
        "summary": summary,
    }
    return outputs


def load_review_outputs(output_dir: str | Path) -> dict[str, Any]:
    """Load the tidy outputs produced by :mod:`spatial_workflow.nncomp`."""

    root = Path(output_dir)
    with (root / MANIFEST_FILE).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    outputs = {
        "compartment_abundance": pd.read_csv(
            root / ABUNDANCE_FILE,
            dtype={
                "condition": "string",
                "sample": "string",
                "compartment": "string",
                "cell_type": "string",
            },
        ),
        "neighbor_composition": pd.read_csv(
            root / NEIGHBOR_FILE,
            dtype={
                "condition": "string",
                "sample": "string",
                "compartment": "string",
                "source_cell_type": "string",
                "neighbor_cell_type": "string",
            },
        ),
        "neighbor_composition_summary": pd.read_csv(
            root / SUMMARY_FILE,
            dtype={
                "condition": "string",
                "compartment": "string",
                "source_cell_type": "string",
                "neighbor_cell_type": "string",
            },
        ),
        "manifest": manifest,
    }
    optional_parquet = {
        "label_permutation_plan": PERMUTATION_PLAN_FILE,
        "ranked_neighbor_edges": RANKED_EDGES_FILE,
        "directional_knn_overlap": OVERLAP_FILE,
        "directional_knn_overlap_permutation": PERMUTATION_FILE,
        "within_compartment_celltype_permutation": RECIPROCAL_CELLTYPE_FILE,
    }
    for key, filename in optional_parquet.items():
        path = root / filename
        outputs[key] = pd.read_parquet(path) if path.exists() else None
    return outputs


def available_compartments(abundance: pd.DataFrame) -> list[str]:
    """Return real compartment labels, excluding the whole-sample bucket."""

    values = abundance.loc[
        abundance["compartment"].astype(str).ne("all"), "compartment"
    ]
    return sorted(values.dropna().astype(str).unique())


def summarize_compartments(abundance: pd.DataFrame) -> pd.DataFrame:
    """Summarize domain size and sample coverage by condition."""

    sample_domains = abundance.loc[abundance["compartment"].astype(str).ne("all")][
        [
            "condition",
            "sample",
            "compartment",
            "compartment_total_cells",
            "compartment_fraction",
        ]
    ].drop_duplicates(["condition", "sample", "compartment"])
    return (
        sample_domains.groupby(
            ["condition", "compartment"], observed=True, dropna=False
        )
        .agg(
            n_samples=("sample", "nunique"),
            mean_cells=("compartment_total_cells", "mean"),
            sd_cells=("compartment_total_cells", "std"),
            mean_sample_fraction=("compartment_fraction", "mean"),
            sd_sample_fraction=("compartment_fraction", "std"),
        )
        .reset_index()
        .sort_values(["condition", "compartment"], kind="stable")
        .reset_index(drop=True)
    )


def compartment_composition(abundance: pd.DataFrame, compartment: str) -> pd.DataFrame:
    """Summarize cell-type composition for one compartment."""

    selected = abundance.loc[abundance["compartment"].astype(str).eq(str(compartment))]
    return (
        selected.groupby(["condition", "cell_type"], observed=True, dropna=False)
        .agg(
            n_samples=("sample", "nunique"),
            mean_fraction=("cell_type_fraction", "mean"),
            sd_fraction=("cell_type_fraction", "std"),
            mean_cells=("n_cells", "mean"),
        )
        .reset_index()
        .sort_values(["condition", "mean_fraction"], ascending=[True, False])
        .reset_index(drop=True)
    )


def neighbor_fraction_matrix(
    summary: pd.DataFrame,
    compartment: str,
    source_cell_type: str | None = None,
) -> pd.DataFrame:
    """Pivot mean neighbor fractions for a domain and optional source type."""

    selected = summary.loc[summary["compartment"].astype(str).eq(str(compartment))]
    index: str | list[str]
    if source_cell_type is None:
        index = ["condition", "source_cell_type"]
    else:
        selected = selected.loc[
            selected["source_cell_type"].astype(str).eq(str(source_cell_type))
        ]
        index = "condition"
    return selected.pivot(
        index=index,
        columns="neighbor_cell_type",
        values="mean_neighbor_fraction",
    ).fillna(0)


def reciprocal_celltype_zscores(
    overlap_table: pd.DataFrame,
    *,
    z_column: str,
    selected_compartment: str | None = None,
) -> pd.DataFrame:
    """Pair A-to-B and B-to-A cell-type z-scores in the same compartment."""

    identity = ["condition", "analysis", "k", "source_compartment"]
    if "sample" in overlap_table.columns:
        identity.insert(1, "sample")
    required = {
        *identity,
        "neighbor_compartment",
        "source_cell_type",
        "neighbor_cell_type",
        z_column,
    }
    missing = sorted(required.difference(overlap_table.columns))
    if missing:
        raise KeyError("Missing reciprocal cell-type columns: " + ", ".join(missing))

    selected = overlap_table.loc[
        overlap_table["analysis"].astype(str).eq("within_compartment")
        & overlap_table["source_compartment"]
        .astype(str)
        .eq(overlap_table["neighbor_compartment"].astype(str))
    ].copy()
    if selected_compartment is not None:
        selected = selected.loc[
            selected["source_compartment"].astype(str).eq(str(selected_compartment))
        ]
    paired = selected.merge(
        selected,
        left_on=[*identity, "source_cell_type", "neighbor_cell_type"],
        right_on=[*identity, "neighbor_cell_type", "source_cell_type"],
        suffixes=("_ab", "_ba"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["source_cell_type_ab"]
        .astype(str)
        .lt(paired["neighbor_cell_type_ab"].astype(str))
    ].rename(
        columns={
            "source_compartment": "compartment",
            "source_cell_type_ab": "cell_type_a",
            "neighbor_cell_type_ab": "cell_type_b",
            f"{z_column}_ab": "z_a_to_b",
            f"{z_column}_ba": "z_b_to_a",
        }
    )
    paired = paired.loc[
        np.isfinite(paired["z_a_to_b"]) & np.isfinite(paired["z_b_to_a"])
    ].copy()
    paired["cell_type_pair"] = paired["cell_type_a"] + " ↔ " + paired["cell_type_b"]
    columns = [
        *[
            "compartment" if value == "source_compartment" else value
            for value in identity
        ],
        "cell_type_a",
        "cell_type_b",
        "cell_type_pair",
        "z_a_to_b",
        "z_b_to_a",
    ]
    if "n_samples_ab" in paired:
        paired["n_samples"] = paired[["n_samples_ab", "n_samples_ba"]].min(axis=1)
        columns.append("n_samples")
    if {
        "n_source_cells_ab",
        "n_neighbor_cells_ab",
        "n_source_cells_ba",
        "n_neighbor_cells_ba",
    }.issubset(paired.columns):
        paired["n_cell_type_a"] = paired[
            ["n_source_cells_ab", "n_neighbor_cells_ba"]
        ].min(axis=1)
        paired["n_cell_type_b"] = paired[
            ["n_neighbor_cells_ab", "n_source_cells_ba"]
        ].min(axis=1)
        columns.extend(["n_cell_type_a", "n_cell_type_b"])
    if {
        "min_source_cells_ab",
        "min_neighbor_cells_ab",
        "min_source_cells_ba",
        "min_neighbor_cells_ba",
    }.issubset(paired.columns):
        paired["min_cell_type_a"] = paired[
            ["min_source_cells_ab", "min_neighbor_cells_ba"]
        ].min(axis=1)
        paired["min_cell_type_b"] = paired[
            ["min_neighbor_cells_ab", "min_source_cells_ba"]
        ].min(axis=1)
        columns.extend(["min_cell_type_a", "min_cell_type_b"])
    sort_columns = ["condition"]
    if "sample" in paired:
        sort_columns.append("sample")
    sort_columns.extend(["compartment", "k", "cell_type_a", "cell_type_b"])
    return (
        paired[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    )


def filter_overlap_support(
    overlap_table: pd.DataFrame,
    *,
    min_cells_per_type: int,
) -> pd.DataFrame:
    """Keep rows with adequate source and neighbor cell-type populations."""

    if min_cells_per_type < 1:
        raise ValueError("min_cells_per_type must be at least 1")
    required = {"n_source_cells", "n_neighbor_cells"}
    missing = sorted(required.difference(overlap_table.columns))
    if missing:
        raise KeyError("Missing overlap support columns: " + ", ".join(missing))
    return overlap_table.loc[
        overlap_table["n_source_cells"].ge(min_cells_per_type)
        & overlap_table["n_neighbor_cells"].ge(min_cells_per_type)
    ].copy()


def filter_condition_sample_support(
    condition_table: pd.DataFrame,
    *,
    min_samples: int,
) -> pd.DataFrame:
    """Keep condition rows supported by at least ``min_samples`` samples."""

    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if "n_samples" not in condition_table:
        raise KeyError("Missing condition support column: n_samples")
    return condition_table.loc[condition_table["n_samples"].ge(min_samples)].copy()


def _normalize_cell_type_filter(
    values: str | list[str] | tuple[str, ...] | set[str] | None,
    *,
    name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        normalized = (values,)
    else:
        normalized = tuple(dict.fromkeys(str(value) for value in values))
    if not normalized:
        raise ValueError(f"{name} must be None or contain at least one cell type")
    return normalized


def _filter_reciprocal_zscores(
    pairs: pd.DataFrame,
    *,
    min_abs_z_score: float,
    z_direction: str,
) -> pd.DataFrame:
    if min_abs_z_score < 0:
        raise ValueError("min_abs_z_score must be non-negative")
    if z_direction not in {"both", "positive", "negative"}:
        raise ValueError("z_direction must be 'both', 'positive', or 'negative'")
    mask = pairs["z_a_to_b"].abs().ge(min_abs_z_score) & pairs["z_b_to_a"].abs().ge(
        min_abs_z_score
    )
    if z_direction == "positive":
        mask &= pairs["z_a_to_b"].gt(0) & pairs["z_b_to_a"].gt(0)
    elif z_direction == "negative":
        mask &= pairs["z_a_to_b"].lt(0) & pairs["z_b_to_a"].lt(0)
    return pairs.loc[mask].copy()


def prepare_reciprocal_celltype_plot(
    overlap_table: pd.DataFrame,
    *,
    compartment: str | int | None = None,
    cell_types_a: str | list[str] | tuple[str, ...] | set[str] | None = None,
    cell_types_b: str | list[str] | tuple[str, ...] | set[str] | None = None,
    k_value: int = 2,
    min_cells_a: int = 10,
    min_cells_b: int = 10,
    min_samples: int = 4,
    min_abs_z_score: float = 0.0,
    z_direction: str = "both",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare matched sample and condition reciprocal cell-type z-scores.

    ``compartment=None`` retains every within-compartment context across the
    complete samples. Cell-type filters define the A-to-B axis orientation and
    accept either one name or multiple names. Condition means are calculated
    before z-score display filtering, avoiding selection-biased means.
    """

    if k_value < 1:
        raise ValueError("k_value must be at least 1")
    if min_cells_a < 1 or min_cells_b < 1:
        raise ValueError("min_cells_a and min_cells_b must be at least 1")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    cell_types_a = _normalize_cell_type_filter(cell_types_a, name="cell_types_a")
    cell_types_b = _normalize_cell_type_filter(cell_types_b, name="cell_types_b")
    required = {
        "condition",
        "sample",
        "analysis",
        "k",
        "source_compartment",
        "neighbor_compartment",
        "source_cell_type",
        "neighbor_cell_type",
        "n_source_cells",
        "n_neighbor_cells",
        "z_score",
    }
    missing = sorted(required.difference(overlap_table.columns))
    if missing:
        raise KeyError("Missing reciprocal plot columns: " + ", ".join(missing))

    selected = overlap_table.loc[
        overlap_table["analysis"].astype(str).eq("within_compartment")
        & overlap_table["source_compartment"]
        .astype(str)
        .eq(overlap_table["neighbor_compartment"].astype(str))
        & overlap_table["k"].eq(k_value)
    ].copy()
    if compartment is not None:
        selected = selected.loc[
            selected["source_compartment"].astype(str).eq(str(compartment))
        ]
    if cell_types_a is not None:
        selected = selected.loc[selected["source_cell_type"].isin(cell_types_a)]
    if cell_types_b is not None:
        selected = selected.loc[selected["neighbor_cell_type"].isin(cell_types_b)]
    selected = selected.loc[
        selected["source_cell_type"]
        .astype(str)
        .ne(selected["neighbor_cell_type"].astype(str))
        & selected["n_source_cells"].ge(min_cells_a)
        & selected["n_neighbor_cells"].ge(min_cells_b)
    ]

    identity = ["condition", "sample", "analysis", "k", "source_compartment"]
    reciprocal = overlap_table.loc[
        overlap_table["analysis"].astype(str).eq("within_compartment")
        & overlap_table["k"].eq(k_value)
    ].copy()
    paired = selected.merge(
        reciprocal,
        left_on=[*identity, "source_cell_type", "neighbor_cell_type"],
        right_on=[*identity, "neighbor_cell_type", "source_cell_type"],
        suffixes=("_ab", "_ba"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["n_source_cells_ba"].ge(min_cells_b)
        & paired["n_neighbor_cells_ba"].ge(min_cells_a)
    ]
    if cell_types_a is None and cell_types_b is None:
        paired = paired.loc[
            paired["source_cell_type_ab"]
            .astype(str)
            .lt(paired["neighbor_cell_type_ab"].astype(str))
        ]
    paired = paired.rename(
        columns={
            "source_compartment": "compartment",
            "source_cell_type_ab": "cell_type_a",
            "neighbor_cell_type_ab": "cell_type_b",
            "z_score_ab": "z_a_to_b",
            "z_score_ba": "z_b_to_a",
        }
    )
    paired = paired.loc[
        np.isfinite(paired["z_a_to_b"]) & np.isfinite(paired["z_b_to_a"])
    ].copy()
    paired["n_cell_type_a"] = paired[["n_source_cells_ab", "n_neighbor_cells_ba"]].min(
        axis=1
    )
    paired["n_cell_type_b"] = paired[["n_neighbor_cells_ab", "n_source_cells_ba"]].min(
        axis=1
    )
    paired["cell_type_pair"] = paired["cell_type_a"] + " ↔ " + paired["cell_type_b"]
    sample_columns = [
        "condition",
        "sample",
        "analysis",
        "k",
        "compartment",
        "cell_type_a",
        "cell_type_b",
        "cell_type_pair",
        "z_a_to_b",
        "z_b_to_a",
        "n_cell_type_a",
        "n_cell_type_b",
    ]
    sample_pairs = paired[sample_columns].copy()
    condition_keys = [
        "condition",
        "analysis",
        "k",
        "compartment",
        "cell_type_a",
        "cell_type_b",
        "cell_type_pair",
    ]
    condition_pairs = (
        sample_pairs.groupby(condition_keys, observed=True)
        .agg(
            n_samples=("sample", "nunique"),
            z_a_to_b=("z_a_to_b", "mean"),
            z_b_to_a=("z_b_to_a", "mean"),
            min_cell_type_a=("n_cell_type_a", "min"),
            min_cell_type_b=("n_cell_type_b", "min"),
        )
        .reset_index()
    )
    condition_pairs = filter_condition_sample_support(
        condition_pairs, min_samples=min_samples
    )
    supported_keys = [
        "condition",
        "analysis",
        "k",
        "compartment",
        "cell_type_a",
        "cell_type_b",
    ]
    sample_pairs = sample_pairs.merge(
        condition_pairs[supported_keys],
        on=supported_keys,
        how="inner",
        validate="many_to_one",
    )
    sample_pairs = _filter_reciprocal_zscores(
        sample_pairs,
        min_abs_z_score=min_abs_z_score,
        z_direction=z_direction,
    )
    condition_pairs = _filter_reciprocal_zscores(
        condition_pairs,
        min_abs_z_score=min_abs_z_score,
        z_direction=z_direction,
    )
    if sample_pairs.empty or condition_pairs.empty:
        raise ValueError("No reciprocal points remain after the requested filters")
    sort_columns = ["condition", "compartment", "cell_type_a", "cell_type_b"]
    return (
        sample_pairs.sort_values(["sample", *sort_columns], kind="stable").reset_index(
            drop=True
        ),
        condition_pairs.sort_values(sort_columns, kind="stable").reset_index(drop=True),
    )


def plot_reciprocal_celltype_zscores(
    sample_pairs: pd.DataFrame,
    condition_pairs: pd.DataFrame,
    *,
    k_value: int,
    title: str,
):
    """Plot reciprocal cell-type z-scores at one selected k value."""

    import plotly.express as px
    from plotly.subplots import make_subplots

    sample_pairs = sample_pairs.loc[sample_pairs["k"].eq(k_value)].copy()
    condition_pairs = condition_pairs.loc[condition_pairs["k"].eq(k_value)].copy()
    if sample_pairs.empty or condition_pairs.empty:
        raise ValueError(f"No reciprocal points are available for k={k_value}")

    hover = {
        "condition": True,
        "sample": True,
        "k": True,
        "compartment": True,
        "cell_type_pair": True,
        "cell_type_a": True,
        "cell_type_b": True,
        "z_a_to_b": ":.3f",
        "z_b_to_a": ":.3f",
    }
    for column in ("n_cell_type_a", "n_cell_type_b"):
        if column in sample_pairs:
            hover[column] = True
    condition_hover = {
        key: value
        for key, value in hover.items()
        if key not in {"sample", "n_cell_type_a", "n_cell_type_b"}
    }
    condition_hover["n_samples"] = True
    for column in ("min_cell_type_a", "min_cell_type_b"):
        if column in condition_pairs:
            condition_hover[column] = True
    left = px.scatter(
        sample_pairs,
        x="z_a_to_b",
        y="z_b_to_a",
        color="condition",
        hover_data=hover,
        opacity=0.72,
        render_mode="webgl",
    )
    right = px.scatter(
        condition_pairs,
        x="z_a_to_b",
        y="z_b_to_a",
        color="condition",
        size="n_samples",
        hover_data=condition_hover,
        opacity=0.82,
        render_mode="webgl",
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("All samples", "Condition means"),
        horizontal_spacing=0.1,
    )
    for trace in left.data:
        figure.add_trace(trace, row=1, col=1)
    for trace in right.data:
        trace.showlegend = False
        figure.add_trace(trace, row=1, col=2)

    x_label = "Cell type A → cell type B z-score"
    y_label = "Cell type B → cell type A z-score"
    figure.update_xaxes(title_text=x_label, zeroline=True, row=1, col=1)
    figure.update_yaxes(title_text=y_label, zeroline=True, row=1, col=1)
    figure.update_xaxes(title_text=x_label, zeroline=True, row=1, col=2)
    figure.update_yaxes(title_text=y_label, zeroline=True, row=1, col=2)
    values = pd.concat(
        [
            sample_pairs["z_a_to_b"],
            sample_pairs["z_b_to_a"],
            condition_pairs["z_a_to_b"],
            condition_pairs["z_b_to_a"],
        ],
        ignore_index=True,
    )
    low, high = float(values.min()), float(values.max())
    for column in (1, 2):
        figure.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line={"color": "gray", "dash": "dot", "width": 1},
            row=1,
            col=column,
        )
    figure.update_layout(
        title=title,
        height=650,
        width=1400,
        legend_title_text="Condition",
        template="plotly_white",
    )
    return figure


def plot_reciprocal_celltype_overlap(
    overlap_table: pd.DataFrame,
    *,
    compartment: str | int | None = None,
    cell_types_a: str | list[str] | tuple[str, ...] | set[str] | None = None,
    cell_types_b: str | list[str] | tuple[str, ...] | set[str] | None = None,
    k_value: int = 2,
    min_cells_a: int = 10,
    min_cells_b: int = 10,
    min_samples: int = 4,
    min_abs_z_score: float = 0.0,
    z_direction: str = "both",
    title: str | None = None,
):
    """Return sample and condition reciprocal z-score panels side by side.

    ``compartment=None`` is the whole-sample review view: all within-compartment
    contexts are retained. ``cell_types_a`` and ``cell_types_b`` accept a
    single cell type, multiple cell types, or ``None`` for all types.
    """

    sample_pairs, condition_pairs = prepare_reciprocal_celltype_plot(
        overlap_table,
        compartment=compartment,
        cell_types_a=cell_types_a,
        cell_types_b=cell_types_b,
        k_value=k_value,
        min_cells_a=min_cells_a,
        min_cells_b=min_cells_b,
        min_samples=min_samples,
        min_abs_z_score=min_abs_z_score,
        z_direction=z_direction,
    )
    if title is None:
        scope = "whole sample" if compartment is None else f"compartment {compartment}"
        title = f"Reciprocal cell-type z-scores: {scope}, k={k_value}"
    figure = plot_reciprocal_celltype_zscores(
        sample_pairs,
        condition_pairs,
        k_value=k_value,
        title=title,
    )
    figure.update_layout(
        meta={
            "compartment": "all" if compartment is None else str(compartment),
            "cell_types_a": (
                "all"
                if cell_types_a is None
                else list(
                    _normalize_cell_type_filter(cell_types_a, name="cell_types_a")
                )
            ),
            "cell_types_b": (
                "all"
                if cell_types_b is None
                else list(
                    _normalize_cell_type_filter(cell_types_b, name="cell_types_b")
                )
            ),
            "k": int(k_value),
            "min_cells_a": int(min_cells_a),
            "min_cells_b": int(min_cells_b),
            "min_samples": int(min_samples),
            "min_abs_z_score": float(min_abs_z_score),
            "z_direction": z_direction,
            "sample_points": int(len(sample_pairs)),
            "condition_points": int(len(condition_pairs)),
        }
    )
    return figure


def plot_spatial_domains(
    adata,
    *,
    sample_col: str,
    compartment_col: str,
    spatial_key: str,
    sample: str,
    ax=None,
    point_size: float = 2.0,
):
    """Plot CellCharter labels for one sample without modifying AnnData."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    mask = adata.obs[sample_col].astype(str).eq(str(sample)).to_numpy()
    coords = adata.obsm[spatial_key][mask, :2]
    labels = adata.obs.loc[mask, compartment_col].astype(str)
    for label in sorted(labels.unique()):
        selected = labels.eq(label).to_numpy()
        ax.scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=point_size,
            label=label,
            linewidths=0,
        )
    ax.set_aspect("equal")
    ax.set_title(f"{sample}: spatial domains")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(
        title=compartment_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=3,
        frameon=False,
    )
    return ax


def plot_autok_stability(
    stability: pd.DataFrame,
    *,
    best_k: int,
    peaks: list[int] | tuple[int, ...] = (),
    ax=None,
):
    """Plot mean AutoK stability and the K selected by CellCharter."""

    import matplotlib.pyplot as plt

    ordered = stability.sort_values("k")
    x = ordered["k"].to_numpy(dtype=int)
    mean = ordered["mean_stability"].to_numpy(dtype=float)
    sd = ordered["sd_stability"].to_numpy(dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean, marker="o", color="black", linewidth=1.5)
    ax.fill_between(x, mean - sd, mean + sd, color="black", alpha=0.12)
    peak_mask = ordered["k"].isin(peaks).to_numpy()
    ax.scatter(
        x[peak_mask],
        mean[peak_mask],
        s=65,
        facecolors="white",
        edgecolors="black",
        label="stability peak",
        zorder=3,
    )
    ax.axvline(best_k, color="tab:red", linestyle="--", label=f"selected K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean stability")
    ax.set_title("CellCharter AutoK stability")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    return ax


def plot_compartment_overview(summary: pd.DataFrame, ax=None):
    """Plot mean fraction of each sample assigned to each compartment."""

    import matplotlib.pyplot as plt

    matrix = summary.pivot(
        index="compartment",
        columns="condition",
        values="mean_sample_fraction",
    ).fillna(0)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.45 * len(matrix.index)), 4))
    matrix.plot.bar(ax=ax)
    ax.set_ylabel("Mean fraction of sample cells")
    ax.set_xlabel("Spatial domain")
    ax.legend(title="Condition", frameon=False)
    return ax


def plot_compartment_composition(
    composition: pd.DataFrame, *, compartment: str, ax=None
):
    """Plot mean cell-type fractions for a selected compartment."""

    import matplotlib.pyplot as plt

    matrix = composition.pivot(
        index="cell_type", columns="condition", values="mean_fraction"
    ).fillna(0)
    matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 0.45 * len(matrix.index)), 4))
    matrix.plot.bar(ax=ax)
    ax.set_title(f"Domain {compartment}: cell-type composition")
    ax.set_ylabel("Mean cell fraction")
    ax.set_xlabel("Cell type")
    ax.legend(title="Condition", frameon=False)
    return ax


def plot_neighbor_fraction_heatmap(
    matrix: pd.DataFrame, *, title: str | None = None, ax=None
):
    """Plot a compact heatmap from a neighbor-fraction matrix."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(
            figsize=(
                max(6, 0.45 * len(matrix.columns)),
                max(2.5, 0.45 * len(matrix.index)),
            )
        )
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), [str(x) for x in matrix.index])
    ax.set_xlabel("Neighbor cell type")
    ax.set_ylabel(
        "Condition" if matrix.index.nlevels == 1 else "Condition, source cell type"
    )
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Mean neighbor fraction")
    return ax
