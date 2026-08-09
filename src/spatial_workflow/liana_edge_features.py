"""Summaries of significant features in LIANA edge-pairwise results."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "ligands": ("ligand_complex",),
    "receptors": ("receptor_complex",),
    "lr_pairs": ("ligand_complex", "receptor_complex"),
    "pathways": ("pathway_name",),
    "sources": ("source",),
    "targets": ("target",),
    "source_target_pairs": ("source", "target"),
}


def _count_edges(
    significant: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    top_n: int,
) -> pd.DataFrame:
    count_column = "n_significant_edges"
    result = (
        significant.groupby(
            list(group_columns),
            observed=True,
            dropna=False,
        )["edge_id"]
        .nunique()
        .rename(count_column)
        .reset_index()
    )
    return (
        result.sort_values(
            [count_column, *group_columns],
            ascending=[False, *([True] * len(group_columns))],
            kind="stable",
            na_position="last",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def top_edge_features(
    edge_pairwise: pd.DataFrame,
    *,
    contrast: str,
    threshold: float = 0.05,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Return seven ranked feature-count tables for significant LIANA edges.

    An edge is included when it belongs to ``contrast``, passed the workflow's
    support gate (``tested``), and has ``exact_permutation_p < threshold``.
    Counts are unique ``edge_id`` counts, so duplicated input rows cannot inflate
    a feature. The returned dictionary contains tables named ``ligands``,
    ``receptors``, ``lr_pairs``, ``pathways``, ``sources``, ``targets``, and
    ``source_target_pairs``.
    """

    if not isinstance(edge_pairwise, pd.DataFrame):
        raise TypeError("edge_pairwise must be a pandas DataFrame")
    if not isinstance(contrast, str) or not contrast.strip():
        raise ValueError("contrast must be a non-empty string")
    if not 0 < float(threshold) <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    required = {
        "edge_id",
        "contrast",
        "tested",
        "exact_permutation_p",
        *(column for columns in FEATURE_COLUMNS.values() for column in columns),
    }
    missing = sorted(required.difference(edge_pairwise.columns))
    if missing:
        raise KeyError("edge_pairwise is missing columns: " + ", ".join(missing))

    available_contrasts = edge_pairwise["contrast"].dropna().astype(str).unique()
    if contrast not in available_contrasts:
        available = ", ".join(sorted(available_contrasts)) or "none"
        raise ValueError(
            f"Unknown contrast {contrast!r}; available contrasts: {available}"
        )

    tested = edge_pairwise["tested"].fillna(False).astype(bool)
    p_values = pd.to_numeric(
        edge_pairwise["exact_permutation_p"], errors="coerce"
    )
    significant = edge_pairwise.loc[
        edge_pairwise["contrast"].astype(str).eq(contrast)
        & tested
        & p_values.lt(float(threshold))
    ]

    return {
        name: _count_edges(significant, columns, top_n=top_n)
        for name, columns in FEATURE_COLUMNS.items()
    }
