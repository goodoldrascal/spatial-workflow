"""Matplotlib/seaborn plots for sample-level cNMF usage comparisons."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def clean_sample_label(value: Any) -> str:
    """Remove common subcluster suffixes from a displayed sample label."""

    label = str(value)
    for token in ("_subclusters", "_subcluster", "subclusters", "subcluster"):
        label = label.replace(token, "")
    return label.strip("_")


def _p_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.2e}"


def plot_usage_condition_violin(
    obs: pd.DataFrame,
    *,
    result: pd.Series | Mapping[str, Any],
    cell_type_key: str,
    sample_key: str,
    condition_key: str,
    condition_order: Sequence[str] | None = None,
    condition_labels: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    sample_label_fn: Callable[[Any], str] | None = clean_sample_label,
    title: str | None = None,
    max_strip_points_per_sample: int = 1_000,
    random_state: int = 0,
    figsize: tuple[float, float] = (13.0, 7.5),
):
    """Plot per-sample cell distributions with sample-level inference.

    The violin and strip layers display cells, but the annotation is taken from
    ``test_usage_condition_changes`` and therefore uses biological-sample means
    as replicates. ``palette`` may be keyed by raw or displayed condition names.

    Returns
    -------
    figure, sample_summary
        The Matplotlib figure and the per-sample values displayed below it.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    result = pd.Series(result)
    required_result = {
        cell_type_key,
        "program",
        "condition_a",
        "condition_b",
        "welch_p",
        "permutation_p",
    }
    missing_result = sorted(required_result.difference(result.index))
    if missing_result:
        raise KeyError("Missing plot result fields: " + ", ".join(missing_result))
    program = str(result["program"])
    cell_type = str(result[cell_type_key])
    required_obs = [cell_type_key, sample_key, condition_key, program]
    missing_obs = [column for column in required_obs if column not in obs]
    if missing_obs:
        raise KeyError("Missing plot columns: " + ", ".join(missing_obs))

    plot_df = obs.loc[
        obs[cell_type_key].astype("string").eq(cell_type), required_obs
    ].copy()
    plot_df = plot_df.dropna(subset=[sample_key, condition_key, program])
    plot_df = plot_df.loc[
        plot_df[sample_key].astype(str).str.strip().ne("")
        & plot_df[condition_key].astype(str).str.strip().ne("")
    ].copy()
    if plot_df.empty:
        raise ValueError(f"No cells are available for {cell_type_key}={cell_type}")
    plot_df[sample_key] = plot_df[sample_key].astype(str)
    plot_df[condition_key] = plot_df[condition_key].astype(str)
    plot_df[program] = pd.to_numeric(plot_df[program], errors="coerce")
    plot_df = plot_df.dropna(subset=[program])

    observed_conditions = plot_df[condition_key].drop_duplicates().tolist()
    raw_order = (
        observed_conditions
        if condition_order is None
        else [str(value) for value in condition_order]
    )
    unknown_conditions = sorted(set(raw_order).difference(observed_conditions))
    if unknown_conditions:
        raise KeyError(
            "Condition order contains unavailable values: "
            + ", ".join(unknown_conditions)
        )
    comparison = [str(result["condition_a"]), str(result["condition_b"])]
    missing_comparison = sorted(set(comparison).difference(raw_order))
    if missing_comparison:
        raise KeyError(
            "The plotted condition order omits contrast values: "
            + ", ".join(missing_comparison)
        )
    plot_df = plot_df.loc[plot_df[condition_key].isin(raw_order)].copy()

    labels = {condition: condition for condition in raw_order}
    if condition_labels is not None:
        labels.update({str(key): str(value) for key, value in condition_labels.items()})
    display_order = [labels[condition] for condition in raw_order]
    plot_df["condition_raw"] = plot_df[condition_key]
    plot_df["condition_display"] = pd.Categorical(
        plot_df[condition_key].map(labels),
        categories=display_order,
        ordered=True,
    )

    sample_condition = (
        plot_df[[sample_key, "condition_raw", "condition_display"]]
        .drop_duplicates()
        .sort_values(["condition_display", sample_key])
    )
    duplicated = sample_condition[sample_key].duplicated(keep=False)
    if duplicated.any():
        bad = sorted(sample_condition.loc[duplicated, sample_key].unique())
        raise ValueError("Samples map to multiple conditions: " + ", ".join(bad))
    sample_order = sample_condition[sample_key].tolist()
    sample_to_raw_condition = dict(
        zip(sample_condition[sample_key], sample_condition["condition_raw"])
    )
    label_function = (lambda value: str(value)) if sample_label_fn is None else sample_label_fn
    sample_label_order = [label_function(sample) for sample in sample_order]

    if palette is None:
        colors = sns.color_palette("colorblind", n_colors=len(raw_order)).as_hex()
        raw_palette = dict(zip(raw_order, colors))
    elif isinstance(palette, Mapping):
        raw_palette = {}
        for condition in raw_order:
            if condition in palette:
                raw_palette[condition] = str(palette[condition])
            elif labels[condition] in palette:
                raw_palette[condition] = str(palette[labels[condition]])
            else:
                raise KeyError(f"No palette color was supplied for {condition!r}")
    else:
        colors = list(palette)
        if len(colors) < len(raw_order):
            raise ValueError("Palette sequence has fewer colors than conditions")
        raw_palette = dict(zip(raw_order, map(str, colors)))
    sample_palette = {
        sample: raw_palette[sample_to_raw_condition[sample]] for sample in sample_order
    }

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4.6, 1.0],
        hspace=0.42,
    )
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[1, 0])

    sns.violinplot(
        data=plot_df,
        x=sample_key,
        y=program,
        order=sample_order,
        hue=sample_key,
        hue_order=sample_order,
        cut=0,
        inner="quartile",
        palette=sample_palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    rng = np.random.default_rng(random_state)
    point_frames = []
    for sample in sample_order:
        sample_points = plot_df.loc[plot_df[sample_key].eq(sample)]
        if len(sample_points) > int(max_strip_points_per_sample):
            selected = rng.choice(
                len(sample_points),
                size=int(max_strip_points_per_sample),
                replace=False,
            )
            sample_points = sample_points.iloc[selected]
        point_frames.append(sample_points)
    points = pd.concat(point_frames, ignore_index=True)
    sns.stripplot(
        data=points,
        x=sample_key,
        y=program,
        order=sample_order,
        color="black",
        alpha=0.12,
        size=1.5,
        jitter=0.22,
        ax=ax,
    )

    sample_summary = (
        plot_df.groupby(
            [sample_key, "condition_raw", "condition_display"], observed=True
        )[program]
        .agg(mean="mean", median="median", n_cells="size")
        .reset_index()
    )
    sample_summary[sample_key] = pd.Categorical(
        sample_summary[sample_key], categories=sample_order, ordered=True
    )
    sample_summary = sample_summary.sort_values(sample_key).reset_index(drop=True)
    ax.scatter(
        np.arange(len(sample_order)),
        sample_summary["mean"],
        marker="D",
        color="white",
        edgecolor="black",
        linewidth=0.8,
        s=28,
        zorder=6,
        label="sample mean",
    )

    y_min = float(plot_df[program].min())
    y_max = float(plot_df[program].max())
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
    bracket_y = y_max + 0.10 * y_range
    tip_y = y_max + 0.06 * y_range
    text_y = y_max + 0.14 * y_range
    condition_a, condition_b = comparison
    condition_a_x = [
        index
        for index, sample in enumerate(sample_order)
        if sample_to_raw_condition[sample] == condition_a
    ]
    condition_b_x = [
        index
        for index, sample in enumerate(sample_order)
        if sample_to_raw_condition[sample] == condition_b
    ]
    if condition_a_x and condition_b_x:
        x1 = float(np.mean(condition_a_x))
        x2 = float(np.mean(condition_b_x))
        ax.plot(
            [x1, x1, x2, x2],
            [tip_y, bracket_y, bracket_y, tip_y],
            linewidth=2,
            color="black",
        )
        q_within = result.get("welch_q_within_contrast", np.nan)
        q_global = result.get("welch_q_global", np.nan)
        annotation = (
            f"{labels[condition_a]} vs {labels[condition_b]}\n"
            f"sample-level Welch p = {_p_text(result['welch_p'])}; "
            f"BH q (contrast) = {_p_text(q_within)}\n"
            f"BH q (global) = {_p_text(q_global)}; "
            f"permutation p = {_p_text(result['permutation_p'])}"
        )
        ax.text(
            (x1 + x2) / 2,
            text_y,
            annotation,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylim(y_min, y_max + 0.38 * y_range)

    ax.set_title(
        title or f"{program} usage in {cell_type}",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Usage score", fontsize=18, fontweight="bold", labelpad=16)
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels(sample_label_order, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(False)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    table_ax.axis("off")
    table_values = [
        sample_summary["condition_display"].astype(str).tolist(),
        [f"{value:.3f}" for value in sample_summary["mean"]],
        [str(int(value)) for value in sample_summary["n_cells"]],
    ]
    table = table_ax.table(
        cellText=table_values,
        rowLabels=["Condition", "Mean", "Cells"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0, 0.08, 1, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.45)

    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.86, bottom=0.08)
    return fig, sample_summary
