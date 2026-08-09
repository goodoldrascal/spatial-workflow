"""Review plots for whole-sample LIANA pathway and edge comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


PATHWAY_RANK_METRICS = (
    "rms_centroid_distance",
    "mean_activity_difference_b_minus_a",
    "mean_activity_exact_p",
    "rms_exact_p",
)

_CONDITION_A_COLOR = "#2166AC"
_CONDITION_B_COLOR = "#B2182B"
_TIE_COLOR = "#8C8C8C"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns {missing}")


def _contrast_conditions(frame: pd.DataFrame) -> tuple[str, str]:
    pairs = frame[["condition_a", "condition_b"]].drop_duplicates()
    if len(pairs) != 1:
        raise ValueError("Expected exactly one condition pair for the selected contrast")
    row = pairs.iloc[0]
    return str(row["condition_a"]), str(row["condition_b"])


def prepare_pathway_ranking_plot_data(
    pathway_pairwise: pd.DataFrame,
    *,
    contrast: str,
    rank_by: str = "rms_centroid_distance",
    top_n: int = 20,
    exact_p_max: float | None = None,
) -> pd.DataFrame:
    """Select and transform pathways for an effect- or p-ranked bar chart."""

    if rank_by not in PATHWAY_RANK_METRICS:
        raise ValueError(
            f"rank_by must be one of {list(PATHWAY_RANK_METRICS)}, got {rank_by!r}"
        )
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if exact_p_max is not None and not 0 < exact_p_max <= 1:
        raise ValueError("exact_p_max must be in (0, 1] or None")
    required = {
        "contrast",
        "condition_a",
        "condition_b",
        "pathway_name",
        "mean_activity_difference_b_minus_a",
        rank_by,
    }
    if exact_p_max is not None:
        required.update({"mean_activity_exact_p", "rms_exact_p"})
    if rank_by == "rms_exact_p":
        required.add("rms_centroid_distance")
    _require_columns(pathway_pairwise, required, "pathway_pairwise")

    selected = pathway_pairwise.loc[
        pathway_pairwise["contrast"].eq(contrast)
    ].copy()
    if selected.empty:
        raise ValueError(f"No pathway rows found for contrast {contrast!r}")
    if exact_p_max is not None:
        selected = selected.loc[
            pd.to_numeric(selected["mean_activity_exact_p"], errors="coerce").lt(
                exact_p_max
            )
            | pd.to_numeric(selected["rms_exact_p"], errors="coerce").lt(exact_p_max)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No pathways have mean-activity or RMS exact p < {exact_p_max:g} "
                f"for contrast {contrast!r}"
            )
    _contrast_conditions(selected)

    metric = pd.to_numeric(selected[rank_by], errors="coerce")
    selected = selected.loc[np.isfinite(metric)].copy()
    selected[rank_by] = metric.loc[selected.index].astype(float)
    if rank_by == "mean_activity_difference_b_minus_a":
        selected["ranking_value"] = selected[rank_by].abs()
        selected["plot_value"] = selected[rank_by]
        selected = selected.sort_values(
            ["ranking_value", "pathway_name"],
            ascending=[False, True],
        )
    elif rank_by == "rms_centroid_distance":
        selected["ranking_value"] = selected[rank_by]
        selected["plot_value"] = selected[rank_by]
        selected = selected.sort_values(
            ["ranking_value", "pathway_name"],
            ascending=[False, True],
        )
    else:
        selected = selected.loc[selected[rank_by].between(0.0, 1.0)].copy()
        tie_breaker = (
            selected["mean_activity_difference_b_minus_a"].abs()
            if rank_by == "mean_activity_exact_p"
            else selected["rms_centroid_distance"]
        )
        selected["ranking_value"] = selected[rank_by]
        selected["tie_breaker"] = pd.to_numeric(tie_breaker, errors="coerce").fillna(0.0)
        selected["plot_value"] = -np.log10(
            selected[rank_by].clip(lower=np.finfo(float).tiny)
        )
        selected = selected.sort_values(
            ["ranking_value", "tie_breaker", "pathway_name"],
            ascending=[True, False, True],
        )

    selected = selected.head(top_n).reset_index(drop=True)
    selected.insert(0, "display_rank", np.arange(1, len(selected) + 1))
    return selected


def plot_pathway_ranking(
    pathway_pairwise: pd.DataFrame,
    *,
    contrast: str,
    rank_by: str = "rms_centroid_distance",
    top_n: int = 20,
    exact_p_max: float | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot top pathways while retaining the direction of mean activity change."""

    import matplotlib.pyplot as plt

    ranked = prepare_pathway_ranking_plot_data(
        pathway_pairwise,
        contrast=contrast,
        rank_by=rank_by,
        top_n=top_n,
        exact_p_max=exact_p_max,
    )
    condition_a, condition_b = _contrast_conditions(ranked)
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, max(4.5, 0.38 * len(ranked) + 1.8)))
    figure = ax.figure

    differences = ranked["mean_activity_difference_b_minus_a"].to_numpy(float)
    colors = np.where(
        differences > 0,
        _CONDITION_B_COLOR,
        np.where(differences < 0, _CONDITION_A_COLOR, _TIE_COLOR),
    )
    positions = np.arange(len(ranked))
    bars = ax.barh(positions, ranked["plot_value"], color=colors, alpha=0.9)
    ax.set_yticks(positions, labels=ranked["pathway_name"])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    if rank_by == "mean_activity_difference_b_minus_a":
        metric_label = "absolute mean-activity difference (bar retains B − A sign)"
        ax.set_xlabel(f"Mean activity difference ({condition_b} − {condition_a})")
        ax.axvline(0.0, color="#333333", linewidth=0.9)
        value_labels = [f"{value:+.3f}" for value in ranked[rank_by]]
    elif rank_by == "rms_centroid_distance":
        metric_label = "RMS centroid distance"
        ax.set_xlabel("RMS centroid distance across pathway edges")
        value_labels = [f"{value:.3f}" for value in ranked[rank_by]]
        ax.set_xlim(left=0.0)
    else:
        metric_label = {
            "mean_activity_exact_p": "mean-activity exact p",
            "rms_exact_p": "RMS-centroid exact p",
        }[rank_by]
        ax.set_xlabel(f"−log10({metric_label})")
        threshold = -np.log10(0.05)
        ax.axvline(threshold, color="#444444", linestyle="--", linewidth=0.9)
        value_labels = [f"p={value:.3g}" for value in ranked[rank_by]]
        ax.set_xlim(left=0.0)
    ax.bar_label(bars, labels=value_labels, padding=3, fontsize=8)
    ax.margins(x=0.16)

    filter_label = (
        ""
        if exact_p_max is None
        else f"; mean-activity or RMS exact p < {exact_p_max:g}"
    )
    ax.set_title(
        title
        or (
            f"Top {len(ranked)} pathways · {condition_b} vs {condition_a}\n"
            f"Ranked by {metric_label}{filter_label}\n"
            f"red: mean {condition_b} higher; blue: mean {condition_a} higher"
        ),
        loc="left",
        fontsize=12,
    )
    figure.tight_layout()
    return figure, ax, ranked


def prepare_pathway_driver_plot_data(
    pathway_lr_drivers: pd.DataFrame,
    pathway_edge_drivers: pd.DataFrame,
    edge_pairwise: pd.DataFrame,
    *,
    contrast: str,
    pathway_name: str,
    top_lr: int = 10,
    top_edges: int = 35,
    edge_p_max: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare aggregate LR drivers and individual directed pathway edges."""

    if top_lr < 1 or top_edges < 1:
        raise ValueError("top_lr and top_edges must be >= 1")
    if edge_p_max is not None and not 0 < edge_p_max <= 1:
        raise ValueError("edge_p_max must be in (0, 1] or None")
    _require_columns(
        pathway_lr_drivers,
        {
            "contrast",
            "condition_a",
            "condition_b",
            "pathway_name",
            "ligand",
            "receptor",
            "signed_effect_sum",
            "absolute_effect_sum",
            "absolute_effect_fraction",
            "driver_rank",
        },
        "pathway_lr_drivers",
    )
    _require_columns(
        pathway_edge_drivers,
        {
            "edge_id",
            "contrast",
            "condition_a",
            "condition_b",
            "pathway_name",
            "source",
            "target",
            "ligand",
            "receptor",
            "difference_b_minus_a",
            "abs_difference",
            "driver_rank",
        },
        "pathway_edge_drivers",
    )
    _require_columns(
        edge_pairwise,
        {
            "edge_id",
            "contrast",
            "exact_permutation_p",
            "exact_permutation_q",
            "tested",
            "presence_class",
            "minimum_attainable_p",
        },
        "edge_pairwise",
    )

    lr = pathway_lr_drivers.loc[
        pathway_lr_drivers["contrast"].eq(contrast)
        & pathway_lr_drivers["pathway_name"].eq(pathway_name)
    ].copy()
    edges = pathway_edge_drivers.loc[
        pathway_edge_drivers["contrast"].eq(contrast)
        & pathway_edge_drivers["pathway_name"].eq(pathway_name)
    ].copy()
    if lr.empty or edges.empty:
        raise ValueError(f"No driver rows found for {contrast} / {pathway_name}")
    _contrast_conditions(edges)

    lr = lr.loc[lr["absolute_effect_sum"].gt(0)].sort_values(
        ["driver_rank", "absolute_effect_sum"], ascending=[True, False]
    ).head(top_lr)
    lr["lr_label"] = lr["ligand"].astype(str) + " → " + lr["receptor"].astype(str)
    lr["absolute_effect_percent"] = 100.0 * lr["absolute_effect_fraction"]
    lr["direction_balance"] = (
        lr["signed_effect_sum"] / lr["absolute_effect_sum"]
    ).clip(-1.0, 1.0)

    statistics = edge_pairwise[
        [
            "edge_id",
            "contrast",
            "exact_permutation_p",
            "exact_permutation_q",
            "tested",
            "presence_class",
            "minimum_attainable_p",
        ]
    ].drop_duplicates(["edge_id", "contrast"])
    edges = edges.merge(
        statistics,
        on=["edge_id", "contrast"],
        how="left",
        validate="many_to_one",
    )
    p_values = pd.to_numeric(edges["exact_permutation_p"], errors="coerce")
    edges = edges.loc[np.isfinite(p_values) & p_values.between(0.0, 1.0)].copy()
    edges["exact_permutation_p"] = p_values.loc[edges.index].astype(float)
    if edge_p_max is not None:
        edges = edges.loc[edges["exact_permutation_p"].le(edge_p_max)].copy()
    edges = edges.sort_values(
        ["abs_difference", "exact_permutation_p", "driver_rank"],
        ascending=[False, True, True],
    ).head(top_edges)
    if not edges.empty:
        zero_p = edges["exact_permutation_p"].eq(0)
        replacement = pd.to_numeric(
            edges["minimum_attainable_p"], errors="coerce"
        ).where(lambda value: value.gt(0), np.finfo(float).tiny)
        display_p = edges["exact_permutation_p"].mask(zero_p, replacement)
        edges["minus_log10_exact_p"] = -np.log10(
            display_p.clip(lower=np.finfo(float).tiny)
        )
        edges["lr_label"] = (
            edges["ligand"].astype(str) + " → " + edges["receptor"].astype(str)
        )
        edges["cell_pair_label"] = (
            edges["source"].astype(str) + " → " + edges["target"].astype(str)
        )
    return lr.reset_index(drop=True), edges.reset_index(drop=True)


def plot_pathway_driver_summary(
    pathway_lr_drivers: pd.DataFrame,
    pathway_edge_drivers: pd.DataFrame,
    edge_pairwise: pd.DataFrame,
    *,
    contrast: str,
    pathway_name: str,
    top_lr: int = 10,
    top_edges: int = 35,
    edge_p_max: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot aggregate LR drivers beside a pathway-specific edge bubble chart."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    lr, edges = prepare_pathway_driver_plot_data(
        pathway_lr_drivers,
        pathway_edge_drivers,
        edge_pairwise,
        contrast=contrast,
        pathway_name=pathway_name,
        top_lr=top_lr,
        top_edges=top_edges,
        edge_p_max=edge_p_max,
    )
    source = edges if not edges.empty else pathway_edge_drivers.loc[
        pathway_edge_drivers["contrast"].eq(contrast)
        & pathway_edge_drivers["pathway_name"].eq(pathway_name)
    ]
    condition_a, condition_b = _contrast_conditions(source)
    n_cell_pairs = edges["cell_pair_label"].nunique() if not edges.empty else 1
    n_lr_labels = edges["lr_label"].nunique() if not edges.empty else 1
    if figsize is None:
        figsize = (
            min(22.0, max(14.0, 9.5 + 0.45 * n_lr_labels)),
            min(15.0, max(6.0, 3.4 + 0.34 * max(len(lr), n_cell_pairs))),
        )
    figure, (driver_ax, bubble_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [0.85, 1.65]},
        constrained_layout=True,
    )

    lr_positions = np.arange(len(lr))
    lr_colors = np.where(
        lr["signed_effect_sum"].to_numpy(float) > 0,
        _CONDITION_B_COLOR,
        np.where(
            lr["signed_effect_sum"].to_numpy(float) < 0,
            _CONDITION_A_COLOR,
            _TIE_COLOR,
        ),
    )
    lr_bars = driver_ax.barh(
        lr_positions,
        lr["absolute_effect_percent"],
        color=lr_colors,
        alpha=0.9,
    )
    driver_ax.set_yticks(lr_positions, labels=lr["lr_label"])
    driver_ax.invert_yaxis()
    driver_ax.set_xlabel("Share of pathway absolute effect (%)")
    driver_ax.set_title(
        (
            f"Top {len(lr)} ligand–receptor drivers\n"
            "bar = absolute contribution (%)\n"
            f"red: net {condition_b}; blue: net {condition_a}"
        ),
        loc="left",
        fontsize=10,
    )
    driver_ax.bar_label(
        lr_bars,
        labels=[f"{value:.1f}%" for value in lr["absolute_effect_percent"]],
        padding=3,
        fontsize=8,
    )
    driver_ax.margins(x=0.18)
    driver_ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    driver_ax.set_axisbelow(True)
    driver_ax.spines[["top", "right"]].set_visible(False)
    threshold_label = "all tested edges" if edge_p_max is None else f"exact p ≤ {edge_p_max:g}"
    if edges.empty:
        bubble_ax.text(
            0.5,
            0.5,
            f"No {pathway_name} edges meet {threshold_label}",
            ha="center",
            va="center",
            transform=bubble_ax.transAxes,
        )
        bubble_ax.set_axis_off()
        bubble_ax.set_title(
            f"Directed edges · {threshold_label}", loc="left", fontsize=11
        )
    else:
        lr_order = (
            edges.groupby("lr_label", observed=True)["abs_difference"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        cell_order = (
            edges.groupby("cell_pair_label", observed=True)["abs_difference"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        x_lookup = {label: index for index, label in enumerate(lr_order)}
        y_lookup = {label: index for index, label in enumerate(cell_order)}
        x = edges["lr_label"].map(x_lookup).to_numpy()
        y = edges["cell_pair_label"].map(y_lookup).to_numpy()
        differences = edges["difference_b_minus_a"].to_numpy(float)
        color_limit = max(float(np.nanmax(np.abs(differences))), np.finfo(float).eps)
        sizes = 35.0 + 110.0 * edges["minus_log10_exact_p"].to_numpy(float)
        scatter = bubble_ax.scatter(
            x,
            y,
            s=sizes,
            c=differences,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
            edgecolor="white",
            linewidth=0.6,
        )
        bubble_ax.set_xticks(np.arange(len(lr_order)), labels=lr_order, rotation=45, ha="right")
        bubble_ax.set_yticks(np.arange(len(cell_order)), labels=cell_order)
        bubble_ax.invert_yaxis()
        bubble_ax.set_xlabel("Ligand–receptor pair")
        bubble_ax.set_ylabel("Source → target")
        bubble_ax.set_title(
            f"Directed edges · {threshold_label}\n"
            "color = Δ activity; size = −log10(exact p)",
            loc="left",
            fontsize=11,
        )
        bubble_ax.grid(color="#E5E5E5", linewidth=0.8)
        bubble_ax.set_axisbelow(True)
        bubble_ax.spines[["top", "right"]].set_visible(False)
        colorbar = figure.colorbar(scatter, ax=bubble_ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"Δ activity ({condition_b} − {condition_a})")

        unique_p = np.sort(edges["exact_permutation_p"].unique())
        if len(unique_p) > 3:
            unique_p = unique_p[[0, len(unique_p) // 2, -1]]
        size_handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=np.sqrt(
                    35.0 + 110.0 * max(-np.log10(max(value, np.finfo(float).tiny)), 0.0)
                ),
                markerfacecolor="#777777",
                markeredgecolor="white",
                label=f"p={value:.3g}",
            )
            for value in unique_p
        ]
        bubble_ax.legend(
            handles=size_handles,
            title="Exact edge p",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.72),
        )

    figure.suptitle(
        f"{pathway_name} pathway drivers · {condition_b} vs {condition_a}",
        fontsize=16,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    return figure, (driver_ax, bubble_ax), lr, edges
