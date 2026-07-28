"""Small loading and plotting helpers for spatial-overview review notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .nncomp import ABUNDANCE_FILE, MANIFEST_FILE, NEIGHBOR_FILE, SUMMARY_FILE


def load_review_outputs(output_dir: str | Path) -> dict[str, Any]:
    """Load the tidy outputs produced by :mod:`spatial_workflow.nncomp`."""

    root = Path(output_dir)
    with (root / MANIFEST_FILE).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
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


def available_compartments(abundance: pd.DataFrame) -> list[str]:
    """Return real compartment labels, excluding the whole-sample bucket."""

    values = abundance.loc[
        abundance["compartment"].astype(str).ne("all"), "compartment"
    ]
    return sorted(values.dropna().astype(str).unique())


def summarize_compartments(abundance: pd.DataFrame) -> pd.DataFrame:
    """Summarize domain size and sample coverage by condition."""

    sample_domains = (
        abundance.loc[abundance["compartment"].astype(str).ne("all")]
        [
            [
                "condition",
                "sample",
                "compartment",
                "compartment_total_cells",
                "compartment_fraction",
            ]
        ]
        .drop_duplicates(["condition", "sample", "compartment"])
    )
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


def compartment_composition(
    abundance: pd.DataFrame, compartment: str
) -> pd.DataFrame:
    """Summarize cell-type composition for one compartment."""

    selected = abundance.loc[
        abundance["compartment"].astype(str).eq(str(compartment))
    ]
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

    selected = summary.loc[
        summary["compartment"].astype(str).eq(str(compartment))
    ]
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


def plot_compartment_overview(summary: pd.DataFrame, ax=None):
    """Plot mean fraction of each sample assigned to each compartment."""

    import matplotlib.pyplot as plt

    matrix = summary.pivot(
        index="compartment",
        columns="condition",
        values="mean_sample_fraction",
    ).fillna(0)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, 0.45 * len(matrix.index)), 4)
        )
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
        _, ax = plt.subplots(
            figsize=(max(7, 0.45 * len(matrix.index)), 4)
        )
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
        "Condition"
        if matrix.index.nlevels == 1
        else "Condition, source cell type"
    )
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Mean neighbor fraction")
    return ax
