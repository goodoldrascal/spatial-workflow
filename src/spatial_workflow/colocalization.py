"""Condition-level inference and interactive filters for colocalization review."""

from __future__ import annotations

import itertools
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


WITHIN_DIRECTIONAL_FILE = "all_directional_within_condition_stats.csv"
WITHIN_RECIPROCAL_FILE = "all_reciprocal_within_condition_stats.csv"
WITHIN_DIRECTIONAL_STRONG_FILE = "all_directional_within_condition_strong.csv"
WITHIN_RECIPROCAL_STRONG_FILE = "all_reciprocal_within_condition_strong.csv"
CONTRAST_DIRECTIONAL_FILE = "all_directional_limma.csv"
CONTRAST_RECIPROCAL_FILE = "all_reciprocal_limma.csv"
CONTRAST_SAMPLE_COUNTS_FILE = "all_sample_celltype_counts.csv"
CONTRAST_SUPPORT_FILE = "contrast_support_summary.csv"
MANIFEST_FILE = "manifest.json"

WITHIN_CONDITION_KEYS = [
    "condition",
    "analysis",
    "k",
    "source_compartment",
    "neighbor_compartment",
    "source_cell_type",
    "neighbor_cell_type",
]
WITHIN_FDR_FAMILY = [
    "condition",
    "analysis",
    "k",
    "source_compartment",
]
PAIR_KEYS = [
    "analysis",
    "k",
    "compartment",
    "cell_type_a",
    "cell_type_b",
    "cell_type_pair",
]


def _bh_with_nan(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    keep = values.notna()
    if keep.any():
        adjusted.loc[keep] = multipletests(
            values.loc[keep].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
    return adjusted


def summarize_within_condition_colocalization(
    sample_table: pd.DataFrame,
    *,
    min_source_cells: int = 10,
    min_neighbor_cells: int = 10,
    expected_samples: int = 4,
) -> pd.DataFrame:
    """Combine sample-level directional enrichment within each condition.

    Samples are filtered by cell counts before equal-weight Stouffer,
    three-of-four partial-conjunction, and four-of-four conjunction tests.
    Cell counts determine eligibility only; they are not statistical weights.
    """

    if min_source_cells < 1 or min_neighbor_cells < 1:
        raise ValueError("Cell-count thresholds must be at least 1")
    if expected_samples != 4:
        raise ValueError(
            "expected_samples must be 4 because the reported replication "
            "tests are explicitly three-of-four and four-of-four"
        )
    required = [
        *WITHIN_CONDITION_KEYS,
        "sample",
        "n_source_cells",
        "n_neighbor_cells",
        "n_permutations",
        "coefficient",
        "null_mean",
        "delta",
        "z_score",
        "p_enriched",
    ]
    missing = [column for column in required if column not in sample_table]
    if missing:
        raise KeyError("Missing all-pair columns: " + ", ".join(missing))

    work = sample_table.loc[:, required].copy()
    string_columns = [
        "condition",
        "analysis",
        "source_compartment",
        "neighbor_compartment",
        "source_cell_type",
        "neighbor_cell_type",
        "sample",
    ]
    for column in string_columns:
        work[column] = work[column].astype(str)
    numeric_columns = [
        "k",
        "n_source_cells",
        "n_neighbor_cells",
        "n_permutations",
        "coefficient",
        "null_mean",
        "delta",
        "z_score",
        "p_enriched",
    ]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    duplicate_keys = [*WITHIN_CONDITION_KEYS, "sample"]
    if work.duplicated(duplicate_keys, keep=False).any():
        raise ValueError(
            "Expected one row per condition, compartment, pair, k, and sample"
        )

    availability = (
        work.groupby(
            WITHIN_CONDITION_KEYS,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(n_samples_before_cell_filter=("sample", "nunique"))
        .reset_index()
    )
    work = work.loc[
        work["n_source_cells"].ge(min_source_cells)
        & work["n_neighbor_cells"].ge(min_neighbor_cells)
    ].copy()
    if work.empty:
        raise ValueError("No sample-level edges passed the cell-count thresholds")

    continuity = 0.5 / (work["n_permutations"] + 1.0)
    work["_p_enriched_for_combination"] = np.maximum(
        continuity,
        np.minimum(1.0 - continuity, work["p_enriched"]),
    )
    work["_sample_z_enriched"] = norm.isf(
        work["_p_enriched_for_combination"]
    )
    work["_positive_delta"] = work["delta"].gt(0)

    grouped = work.groupby(
        WITHIN_CONDITION_KEYS,
        observed=True,
        dropna=False,
        sort=False,
    )
    summary = grouped.agg(
        n_samples=("sample", "nunique"),
        n_p_values=("_sample_z_enriched", "count"),
        n_permutations=("n_permutations", "min"),
        mean_coefficient=("coefficient", "mean"),
        median_coefficient=("coefficient", "median"),
        mean_null=("null_mean", "mean"),
        mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        min_delta=("delta", "min"),
        max_delta=("delta", "max"),
        sd_delta=("delta", "std"),
        mean_z=("z_score", "mean"),
        median_z=("z_score", "median"),
        n_positive_delta=("_positive_delta", "sum"),
        positive_fraction=("_positive_delta", "mean"),
        min_source_cells=("n_source_cells", "min"),
        min_neighbor_cells=("n_neighbor_cells", "min"),
        sum_sample_z_enriched=("_sample_z_enriched", "sum"),
        min_sample_p_enriched=("_p_enriched_for_combination", "min"),
        max_sample_p_enriched=("_p_enriched_for_combination", "max"),
    ).reset_index()
    summary = summary.merge(
        availability,
        on=WITHIN_CONDITION_KEYS,
        how="left",
        validate="one_to_one",
    )
    summary["stouffer_z_enriched"] = (
        summary["sum_sample_z_enriched"] / np.sqrt(summary["n_p_values"])
    )
    summary["stouffer_p_enriched"] = norm.sf(
        summary["stouffer_z_enriched"]
    )

    ordered = work.loc[
        work["_p_enriched_for_combination"].notna(),
        [*WITHIN_CONDITION_KEYS, "_p_enriched_for_combination"],
    ].sort_values(
        [*WITHIN_CONDITION_KEYS, "_p_enriched_for_combination"],
        kind="stable",
    )
    ordered["_p_order"] = (
        ordered.groupby(
            WITHIN_CONDITION_KEYS,
            observed=True,
            dropna=False,
        )
        .cumcount()
        .add(1)
    )
    third = ordered.loc[
        ordered["_p_order"].eq(3),
        [*WITHIN_CONDITION_KEYS, "_p_enriched_for_combination"],
    ].rename(columns={"_p_enriched_for_combination": "sample_p_order_3"})
    fourth = ordered.loc[
        ordered["_p_order"].eq(4),
        [*WITHIN_CONDITION_KEYS, "_p_enriched_for_combination"],
    ].rename(columns={"_p_enriched_for_combination": "sample_p_order_4"})
    summary = summary.merge(
        third,
        on=WITHIN_CONDITION_KEYS,
        how="left",
        validate="one_to_one",
    ).merge(
        fourth,
        on=WITHIN_CONDITION_KEYS,
        how="left",
        validate="one_to_one",
    )
    complete = (
        summary["n_samples"].eq(expected_samples)
        & summary["n_p_values"].eq(expected_samples)
    )
    summary["partial_p_enriched_3_of_4"] = np.where(
        complete,
        np.minimum(1.0, 2.0 * summary["sample_p_order_3"]),
        np.nan,
    )
    summary["conjunction_p_enriched_4_of_4"] = np.where(
        complete,
        summary["sample_p_order_4"],
        np.nan,
    )
    summary = summary.drop(
        columns=[
            "sum_sample_z_enriched",
            "sample_p_order_3",
            "sample_p_order_4",
        ]
    )
    summary["compartment"] = summary["source_compartment"].astype(str)
    summary["is_self_pair"] = summary["source_cell_type"].eq(
        summary["neighbor_cell_type"]
    )
    summary["min_source_cells_threshold"] = int(min_source_cells)
    summary["min_neighbor_cells_threshold"] = int(min_neighbor_cells)

    for p_column, q_column in [
        ("stouffer_p_enriched", "stouffer_q_enriched"),
        ("partial_p_enriched_3_of_4", "partial_q_enriched_3_of_4"),
        (
            "conjunction_p_enriched_4_of_4",
            "conjunction_q_enriched_4_of_4",
        ),
    ]:
        summary[q_column] = summary.groupby(
            WITHIN_FDR_FAMILY,
            observed=True,
            dropna=False,
        )[p_column].transform(_bh_with_nan)

    return summary.sort_values(
        [
            "stouffer_q_enriched",
            "partial_q_enriched_3_of_4",
            "median_delta",
        ],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def build_reciprocal_within_condition_colocalization(
    directional_table: pd.DataFrame,
) -> pd.DataFrame:
    """Pair A-to-B and B-to-A within-condition evidence."""

    context = [
        "condition",
        "analysis",
        "k",
        "source_compartment",
        "neighbor_compartment",
    ]
    base = directional_table.loc[~directional_table["is_self_pair"]].copy()
    paired = base.merge(
        base,
        left_on=[*context, "source_cell_type", "neighbor_cell_type"],
        right_on=[*context, "neighbor_cell_type", "source_cell_type"],
        suffixes=("_a_to_b", "_b_to_a"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["source_cell_type_a_to_b"]
        .astype(str)
        .lt(paired["neighbor_cell_type_a_to_b"].astype(str))
    ].copy()

    result = paired.loc[:, context].copy()
    result["compartment"] = result["source_compartment"].astype(str)
    result["cell_type_a"] = paired["source_cell_type_a_to_b"].astype(str)
    result["cell_type_b"] = paired["neighbor_cell_type_a_to_b"].astype(str)
    result["cell_type_pair"] = result["cell_type_a"] + " ↔ " + result["cell_type_b"]
    directional_columns = [
        "n_samples_before_cell_filter",
        "n_samples",
        "n_p_values",
        "mean_coefficient",
        "median_coefficient",
        "mean_delta",
        "median_delta",
        "min_delta",
        "max_delta",
        "sd_delta",
        "mean_z",
        "median_z",
        "n_positive_delta",
        "positive_fraction",
        "min_source_cells",
        "min_neighbor_cells",
        "stouffer_z_enriched",
        "stouffer_p_enriched",
        "stouffer_q_enriched",
        "partial_p_enriched_3_of_4",
        "partial_q_enriched_3_of_4",
        "conjunction_p_enriched_4_of_4",
        "conjunction_q_enriched_4_of_4",
    ]
    for column in directional_columns:
        result[f"{column}_a_to_b"] = paired[f"{column}_a_to_b"]
        result[f"{column}_b_to_a"] = paired[f"{column}_b_to_a"]

    self_base = directional_table.loc[directional_table["is_self_pair"]].copy()
    if not self_base.empty:
        self_result = self_base.loc[:, context].copy()
        self_result["compartment"] = self_result["source_compartment"].astype(str)
        self_result["cell_type_a"] = self_base["source_cell_type"].astype(str)
        self_result["cell_type_b"] = self_base["neighbor_cell_type"].astype(str)
        self_result["cell_type_pair"] = (
            self_result["cell_type_a"] + " ↔ " + self_result["cell_type_b"]
        )
        for column in directional_columns:
            self_result[f"{column}_a_to_b"] = self_base[column].to_numpy()
            self_result[f"{column}_b_to_a"] = self_base[column].to_numpy()
        result = pd.concat([result, self_result], ignore_index=True)

    result["is_self_pair"] = result["cell_type_a"].eq(result["cell_type_b"])
    result["n_samples_min"] = result[
        ["n_samples_a_to_b", "n_samples_b_to_a"]
    ].min(axis=1)
    result["min_positive_samples"] = result[
        ["n_positive_delta_a_to_b", "n_positive_delta_b_to_a"]
    ].min(axis=1)
    result["min_median_delta"] = result[
        ["median_delta_a_to_b", "median_delta_b_to_a"]
    ].min(axis=1)
    result["min_cell_type_a"] = result[
        ["min_source_cells_a_to_b", "min_neighbor_cells_b_to_a"]
    ].min(axis=1)
    result["min_cell_type_b"] = result[
        ["min_neighbor_cells_a_to_b", "min_source_cells_b_to_a"]
    ].min(axis=1)

    for left_column, right_column, output_column in [
        (
            "stouffer_p_enriched_a_to_b",
            "stouffer_p_enriched_b_to_a",
            "both_direction_stouffer_p_max",
        ),
        (
            "partial_p_enriched_3_of_4_a_to_b",
            "partial_p_enriched_3_of_4_b_to_a",
            "both_direction_partial_p_3_of_4_max",
        ),
        (
            "conjunction_p_enriched_4_of_4_a_to_b",
            "conjunction_p_enriched_4_of_4_b_to_a",
            "both_direction_conjunction_p_4_of_4_max",
        ),
    ]:
        result[output_column] = result[[left_column, right_column]].max(
            axis=1, skipna=False
        )

    reciprocal_fdr_family = [
        "condition",
        "analysis",
        "k",
        "source_compartment",
    ]
    for p_column, q_column in [
        ("both_direction_stouffer_p_max", "both_direction_stouffer_q"),
        (
            "both_direction_partial_p_3_of_4_max",
            "both_direction_partial_q_3_of_4",
        ),
        (
            "both_direction_conjunction_p_4_of_4_max",
            "both_direction_conjunction_q_4_of_4",
        ),
    ]:
        result[q_column] = result.groupby(
            reciprocal_fdr_family,
            observed=True,
            dropna=False,
        )[p_column].transform(_bh_with_nan)

    return result.sort_values(
        [
            "both_direction_stouffer_q",
            "both_direction_partial_q_3_of_4",
            "min_median_delta",
        ],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def prepare_reciprocal_effect_samples(
    overlap_table: pd.DataFrame,
    *,
    k_value: int,
    min_cells_a: int,
    min_cells_b: int,
) -> pd.DataFrame:
    """Pair reciprocal sample effects and retain observed coefficients."""

    if k_value < 1:
        raise ValueError("k_value must be at least 1")
    if min_cells_a < 1 or min_cells_b < 1:
        raise ValueError("Cell-count thresholds must be at least 1")
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
        "coefficient",
        "null_mean",
        "delta",
        "z_score",
    }
    missing = sorted(required.difference(overlap_table.columns))
    if missing:
        raise KeyError("Missing reciprocal-effect columns: " + ", ".join(missing))

    selected = overlap_table.loc[
        overlap_table["analysis"].astype(str).eq("within_compartment")
        & overlap_table["source_compartment"]
        .astype(str)
        .eq(overlap_table["neighbor_compartment"].astype(str))
        & overlap_table["k"].eq(k_value)
        & overlap_table["source_cell_type"]
        .astype(str)
        .ne(overlap_table["neighbor_cell_type"].astype(str))
        & overlap_table["n_source_cells"].ge(min_cells_a)
        & overlap_table["n_neighbor_cells"].ge(min_cells_b)
        & np.isfinite(overlap_table["delta"])
    ].copy()

    identity = ["condition", "sample", "analysis", "k", "source_compartment"]
    paired = selected.merge(
        selected,
        left_on=[*identity, "source_cell_type", "neighbor_cell_type"],
        right_on=[*identity, "neighbor_cell_type", "source_cell_type"],
        suffixes=("_ab", "_ba"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["n_source_cells_ba"].ge(min_cells_b)
        & paired["n_neighbor_cells_ba"].ge(min_cells_a)
        & paired["source_cell_type_ab"]
        .astype(str)
        .lt(paired["neighbor_cell_type_ab"].astype(str))
    ].copy()
    paired = paired.rename(
        columns={
            "source_compartment": "compartment",
            "source_cell_type_ab": "cell_type_a",
            "neighbor_cell_type_ab": "cell_type_b",
            "coefficient_ab": "coefficient_a_to_b",
            "coefficient_ba": "coefficient_b_to_a",
            "null_mean_ab": "null_mean_a_to_b",
            "null_mean_ba": "null_mean_b_to_a",
            "delta_ab": "delta_a_to_b",
            "delta_ba": "delta_b_to_a",
            "z_score_ab": "z_a_to_b",
            "z_score_ba": "z_b_to_a",
        }
    )
    paired["cell_type_pair"] = (
        paired["cell_type_a"].astype(str)
        + " ↔ "
        + paired["cell_type_b"].astype(str)
    )
    paired["n_cell_type_a"] = paired[
        ["n_source_cells_ab", "n_neighbor_cells_ba"]
    ].min(axis=1)
    paired["n_cell_type_b"] = paired[
        ["n_neighbor_cells_ab", "n_source_cells_ba"]
    ].min(axis=1)
    keep = [
        "condition",
        "sample",
        *PAIR_KEYS,
        "coefficient_a_to_b",
        "coefficient_b_to_a",
        "null_mean_a_to_b",
        "null_mean_b_to_a",
        "delta_a_to_b",
        "delta_b_to_a",
        "z_a_to_b",
        "z_b_to_a",
        "n_cell_type_a",
        "n_cell_type_b",
    ]
    return paired[keep].sort_values(
        ["sample", "compartment", "cell_type_a", "cell_type_b"],
        kind="stable",
    ).reset_index(drop=True)


@lru_cache(maxsize=1)
def _limma_fit():
    try:
        from rpy2 import robjects as ro
    except ImportError as exc:
        raise ImportError(
            "Condition contrasts require rpy2 and an R installation with limma"
        ) from exc

    return ro.r(
        """
        function(y, condition, numerator, denominator) {
            suppressPackageStartupMessages(library(limma))
            condition <- factor(condition, levels=unique(condition))
            design <- model.matrix(~ 0 + condition)
            colnames(design) <- levels(condition)
            fit <- lmFit(y, design)
            contrast <- rep(0, ncol(design))
            names(contrast) <- colnames(design)
            contrast[numerator] <- 1
            contrast[denominator] <- -1
            contrast_matrix <- matrix(
                contrast,
                ncol=1,
                dimnames=list(
                    names(contrast),
                    paste0(numerator, "_vs_", denominator)
                )
            )
            fit <- contrasts.fit(fit, contrast_matrix)
            fit <- eBayes(fit, robust=TRUE, trend=TRUE)
            out <- topTable(fit, coef=1, number=Inf, sort.by="none")
            out$df_total <- fit$df.total
            out
        }
        """
    )


def _run_limma(
    effect_matrix: pd.DataFrame,
    conditions: Sequence[str],
    numerator: str,
    denominator: str,
) -> pd.DataFrame:
    from rpy2 import robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + numpy2ri.converter):
        r_y = conversion.py2rpy(effect_matrix.to_numpy(dtype=float))
    result = _limma_fit()(
        r_y,
        ro.StrVector(list(map(str, conditions))),
        str(numerator),
        str(denominator),
    )
    with localconverter(default_converter + pandas2ri.converter):
        result = conversion.rpy2py(result)
    return pd.DataFrame(result).reset_index(drop=True).rename(
        columns={
            "logFC": "contrast_delta",
            "AveExpr": "average_delta",
            "P.Value": "limma_p",
            "adj.P.Val": "limma_q",
            "B": "log_odds_differential",
        }
    )


def _exact_and_stability(
    effect_matrix: pd.DataFrame,
    n_denominator: int,
) -> pd.DataFrame:
    values = effect_matrix.to_numpy(dtype=float)
    n_total = values.shape[1]
    n_numerator = n_total - n_denominator
    if not 0 < n_denominator < n_total:
        raise ValueError("Both contrast groups must contain at least one sample")
    denominator = values[:, :n_denominator]
    numerator = values[:, n_denominator:]
    observed = numerator.mean(axis=1) - denominator.mean(axis=1)

    allocations = list(itertools.combinations(range(n_total), n_numerator))
    all_indices = np.arange(n_total)
    permutation_differences = []
    for numerator_indices in allocations:
        numerator_indices = np.asarray(numerator_indices)
        denominator_indices = np.setdiff1d(
            all_indices, numerator_indices, assume_unique=True
        )
        permutation_differences.append(
            values[:, numerator_indices].mean(axis=1)
            - values[:, denominator_indices].mean(axis=1)
        )
    permutation_differences = np.column_stack(permutation_differences)
    exact_p = (
        np.abs(permutation_differences)
        >= np.abs(observed[:, None]) - np.finfo(float).eps
    ).mean(axis=1)

    loo_differences = []
    for index in range(n_total):
        if index < n_denominator:
            loo_differences.append(
                numerator.mean(axis=1)
                - np.delete(denominator, index, axis=1).mean(axis=1)
            )
        else:
            loo_differences.append(
                np.delete(numerator, index - n_denominator, axis=1).mean(axis=1)
                - denominator.mean(axis=1)
            )
    loo_differences = np.column_stack(loo_differences)
    observed_sign = np.sign(observed)
    loo_sign_fraction = (
        np.sign(loo_differences) == observed_sign[:, None]
    ).mean(axis=1)
    superiority = (
        (numerator[:, :, None] > denominator[:, None, :]).mean(axis=(1, 2))
        + 0.5
        * (numerator[:, :, None] == denominator[:, None, :]).mean(axis=(1, 2))
    )
    exact_q = (
        multipletests(exact_p, method="fdr_bh")[1]
        if len(exact_p)
        else np.asarray([], dtype=float)
    )

    return pd.DataFrame(
        {
            "exact_permutation_p": exact_p,
            "exact_permutation_q": exact_q,
            "denominator_mean_delta": denominator.mean(axis=1),
            "numerator_mean_delta": numerator.mean(axis=1),
            "denominator_median_delta": np.median(denominator, axis=1),
            "numerator_median_delta": np.median(numerator, axis=1),
            "denominator_positive_samples": (denominator > 0).sum(axis=1),
            "numerator_positive_samples": (numerator > 0).sum(axis=1),
            "pairwise_superiority": superiority,
            "loo_sign_fraction": loo_sign_fraction,
            "loo_min_contrast": loo_differences.min(axis=1),
            "loo_max_contrast": loo_differences.max(axis=1),
        }
    )


def run_condition_contrast(
    sample_effects: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    sample_dict: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one sample-level condition contrast in both reciprocal directions."""

    if numerator == denominator:
        raise ValueError("Contrast numerator and denominator must differ")
    missing_conditions = [
        condition
        for condition in (denominator, numerator)
        if condition not in sample_dict
    ]
    if missing_conditions:
        raise KeyError("Missing contrast conditions: " + ", ".join(missing_conditions))

    directional = pd.concat(
        [
            sample_effects[
                [
                    "condition",
                    "sample",
                    *PAIR_KEYS,
                    "delta_a_to_b",
                    "coefficient_a_to_b",
                ]
            ]
            .rename(
                columns={
                    "delta_a_to_b": "delta",
                    "coefficient_a_to_b": "coefficient",
                }
            )
            .assign(direction="a_to_b"),
            sample_effects[
                [
                    "condition",
                    "sample",
                    *PAIR_KEYS,
                    "delta_b_to_a",
                    "coefficient_b_to_a",
                ]
            ]
            .rename(
                columns={
                    "delta_b_to_a": "delta",
                    "coefficient_b_to_a": "coefficient",
                }
            )
            .assign(direction="b_to_a"),
        ],
        ignore_index=True,
    )
    feature_keys = [*PAIR_KEYS, "direction"]
    sample_order = [
        *map(str, sample_dict[denominator]),
        *map(str, sample_dict[numerator]),
    ]
    if len(sample_order) != len(set(sample_order)):
        raise ValueError("Sample identifiers must be unique across contrast groups")
    contrast_rows = directional.loc[
        directional["condition"].isin([denominator, numerator])
    ].copy()
    effect_matrix = (
        contrast_rows.pivot_table(
            index=feature_keys,
            columns="sample",
            values="delta",
            aggfunc="first",
        )
        .reindex(columns=sample_order)
        .dropna()
    )
    if effect_matrix.empty:
        raise ValueError(
            f"No complete features for {numerator}_vs_{denominator}"
        )
    coefficient_matrix = (
        contrast_rows.pivot_table(
            index=feature_keys,
            columns="sample",
            values="coefficient",
            aggfunc="first",
        )
        .reindex(index=effect_matrix.index, columns=sample_order)
    )
    if coefficient_matrix.isna().any().any():
        raise ValueError("Observed coefficients are incomplete for contrast features")

    complete_pairs = (
        effect_matrix.index.droplevel("direction")
        .unique()
        .to_frame(index=False)
    )
    pair_support = (
        sample_effects.loc[
            sample_effects["condition"].isin([denominator, numerator]),
            [
                "condition",
                "sample",
                *PAIR_KEYS,
                "n_cell_type_a",
                "n_cell_type_b",
            ],
        ]
        .drop_duplicates(["condition", "sample", *PAIR_KEYS])
        .merge(
            complete_pairs,
            on=PAIR_KEYS,
            how="inner",
            validate="many_to_one",
        )
    )
    support_wide = pair_support.pivot(
        index=PAIR_KEYS,
        columns="sample",
        values=["n_cell_type_a", "n_cell_type_b"],
    )
    expected_support_columns = pd.MultiIndex.from_product(
        [["n_cell_type_a", "n_cell_type_b"], sample_order]
    )
    support_wide = support_wide.reindex(columns=expected_support_columns)
    support_wide.columns = [
        f"{metric}__{sample}" for metric, sample in support_wide.columns
    ]
    for condition in (denominator, numerator):
        for metric in ("n_cell_type_a", "n_cell_type_b"):
            sample_columns = [
                f"{metric}__{sample}" for sample in sample_dict[condition]
            ]
            support_wide[f"min_{metric}__{condition}"] = support_wide[
                sample_columns
            ].min(axis=1)
    count_columns = [
        column
        for column in support_wide.columns
        if column.startswith("n_cell_type_")
        or column.startswith("min_n_cell_type_")
    ]
    support_wide[count_columns] = support_wide[count_columns].astype("Int64")
    support_wide = support_wide.reset_index()
    pair_support.insert(0, "contrast", f"{numerator}_vs_{denominator}")
    pair_support = pair_support.sort_values(
        ["condition", "sample", "compartment", "cell_type_a", "cell_type_b"],
        kind="stable",
    ).reset_index(drop=True)

    n_denominator = len(sample_dict[denominator])
    conditions = [
        *([denominator] * n_denominator),
        *([numerator] * len(sample_dict[numerator])),
    ]
    limma_result = _run_limma(
        effect_matrix,
        conditions,
        numerator,
        denominator,
    )
    stability = _exact_and_stability(
        effect_matrix,
        n_denominator=n_denominator,
    )
    denominator_coefficient = coefficient_matrix.iloc[:, :n_denominator]
    numerator_coefficient = coefficient_matrix.iloc[:, n_denominator:]
    coefficient_summary = pd.DataFrame(
        {
            "denominator_mean_coefficient": denominator_coefficient.mean(axis=1),
            "numerator_mean_coefficient": numerator_coefficient.mean(axis=1),
            "denominator_median_coefficient": denominator_coefficient.median(axis=1),
            "numerator_median_coefficient": numerator_coefficient.median(axis=1),
        }
    ).reset_index(drop=True)
    coefficient_summary["max_condition_mean_coefficient"] = coefficient_summary[
        ["denominator_mean_coefficient", "numerator_mean_coefficient"]
    ].max(axis=1)

    directional_result = pd.concat(
        [
            effect_matrix.index.to_frame(index=False).reset_index(drop=True),
            limma_result,
            stability,
            coefficient_summary,
        ],
        axis=1,
    )
    directional_result.insert(0, "contrast", f"{numerator}_vs_{denominator}")
    directional_result = directional_result.merge(
        support_wide,
        on=PAIR_KEYS,
        how="left",
        validate="many_to_one",
    )

    statistic_columns = [
        "contrast_delta",
        "limma_p",
        "limma_q",
        "exact_permutation_p",
        "exact_permutation_q",
        "denominator_mean_delta",
        "numerator_mean_delta",
        "denominator_median_delta",
        "numerator_median_delta",
        "denominator_positive_samples",
        "numerator_positive_samples",
        "pairwise_superiority",
        "loo_sign_fraction",
        "loo_min_contrast",
        "loo_max_contrast",
        "denominator_mean_coefficient",
        "numerator_mean_coefficient",
        "denominator_median_coefficient",
        "numerator_median_coefficient",
        "max_condition_mean_coefficient",
    ]
    reciprocal_result = directional_result.pivot(
        index=["contrast", *PAIR_KEYS],
        columns="direction",
        values=statistic_columns,
    )
    reciprocal_result.columns = [
        f"{statistic}_{direction}"
        for statistic, direction in reciprocal_result.columns
    ]
    reciprocal_result = reciprocal_result.reset_index().merge(
        support_wide,
        on=PAIR_KEYS,
        how="left",
        validate="one_to_one",
    )
    reciprocal_result["reciprocal_same_direction"] = np.sign(
        reciprocal_result["contrast_delta_a_to_b"]
    ).eq(np.sign(reciprocal_result["contrast_delta_b_to_a"]))
    reciprocal_result["both_direction_limma_q_max"] = reciprocal_result[
        ["limma_q_a_to_b", "limma_q_b_to_a"]
    ].max(axis=1)
    reciprocal_result["both_direction_exact_p_max"] = reciprocal_result[
        ["exact_permutation_p_a_to_b", "exact_permutation_p_b_to_a"]
    ].max(axis=1)
    reciprocal_result["both_direction_loo_sign_fraction_min"] = reciprocal_result[
        ["loo_sign_fraction_a_to_b", "loo_sign_fraction_b_to_a"]
    ].min(axis=1)
    reciprocal_result[
        "min_direction_max_condition_mean_coefficient"
    ] = reciprocal_result[
        [
            "max_condition_mean_coefficient_a_to_b",
            "max_condition_mean_coefficient_b_to_a",
        ]
    ].min(axis=1)
    reciprocal_result.loc[
        ~reciprocal_result["reciprocal_same_direction"],
        ["both_direction_limma_q_max", "both_direction_exact_p_max"],
    ] = 1.0
    reciprocal_result = reciprocal_result.sort_values(
        [
            "both_direction_limma_q_max",
            "both_direction_loo_sign_fraction_min",
        ],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    return directional_result, reciprocal_result, pair_support


def build_colocalization_tables(
    sample_table: pd.DataFrame,
    *,
    k_value: int,
    min_cells_a: int,
    min_cells_b: int,
    contrasts: Sequence[tuple[str, str]],
    expected_samples: int = 4,
) -> dict[str, Any]:
    """Build all within-condition and between-condition review tables."""

    within_directional = summarize_within_condition_colocalization(
        sample_table,
        min_source_cells=min_cells_a,
        min_neighbor_cells=min_cells_b,
        expected_samples=expected_samples,
    )
    within_reciprocal = build_reciprocal_within_condition_colocalization(
        within_directional
    )
    within_directional_strong = within_directional.loc[
        within_directional["n_samples"].eq(expected_samples)
        & within_directional["median_delta"].gt(0)
        & within_directional["n_positive_delta"].ge(expected_samples - 1)
        & within_directional["stouffer_q_enriched"].lt(0.05)
        & within_directional["partial_q_enriched_3_of_4"].lt(0.10)
    ].reset_index(drop=True)
    within_reciprocal_strong = within_reciprocal.loc[
        within_reciprocal["n_samples_min"].eq(expected_samples)
        & within_reciprocal["min_median_delta"].gt(0)
        & within_reciprocal["min_positive_samples"].ge(expected_samples - 1)
        & within_reciprocal["both_direction_stouffer_q"].lt(0.05)
        & within_reciprocal["both_direction_partial_q_3_of_4"].lt(0.10)
    ].reset_index(drop=True)

    reciprocal_samples = prepare_reciprocal_effect_samples(
        sample_table,
        k_value=k_value,
        min_cells_a=min_cells_a,
        min_cells_b=min_cells_b,
    )
    sample_dict = (
        reciprocal_samples[["condition", "sample"]]
        .drop_duplicates()
        .sort_values(["condition", "sample"], kind="stable")
        .groupby("condition", observed=True)["sample"]
        .apply(list)
        .to_dict()
    )
    contrast_results: dict[str, dict[str, pd.DataFrame]] = {}
    for numerator, denominator in contrasts:
        directional, reciprocal, sample_counts = run_condition_contrast(
            reciprocal_samples,
            numerator=str(numerator),
            denominator=str(denominator),
            sample_dict=sample_dict,
        )
        contrast_results[f"{numerator}_vs_{denominator}"] = {
            "directional": directional,
            "reciprocal": reciprocal,
            "sample_counts": sample_counts,
        }
    if not contrast_results:
        raise ValueError("At least one condition contrast is required")

    all_directional = pd.concat(
        [entry["directional"] for entry in contrast_results.values()],
        ignore_index=True,
    )
    all_reciprocal = pd.concat(
        [entry["reciprocal"] for entry in contrast_results.values()],
        ignore_index=True,
    )
    all_sample_counts = pd.concat(
        [entry["sample_counts"] for entry in contrast_results.values()],
        ignore_index=True,
    )
    contrast_support = pd.DataFrame(
        [
            {
                "contrast": contrast,
                "complete_directional_features": len(result["directional"]),
                "complete_reciprocal_pairs": len(result["reciprocal"]),
                "limma_q_lt_0.05_directions": int(
                    result["directional"]["limma_q"].lt(0.05).sum()
                ),
                "same_direction_qmax_lt_0.05_pairs": int(
                    result["reciprocal"]["both_direction_limma_q_max"]
                    .lt(0.05)
                    .sum()
                ),
            }
            for contrast, result in contrast_results.items()
        ]
    )
    return {
        "within_directional": within_directional,
        "within_reciprocal": within_reciprocal,
        "within_directional_strong": within_directional_strong,
        "within_reciprocal_strong": within_reciprocal_strong,
        "reciprocal_samples": reciprocal_samples,
        "contrast_results": contrast_results,
        "all_directional": all_directional,
        "all_reciprocal": all_reciprocal,
        "all_sample_counts": all_sample_counts,
        "contrast_support": contrast_support,
    }


def _write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        table.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_colocalization_outputs(
    output_dir: str | Path,
    tables: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    """Write complete review tables and provenance with atomic replacement."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "within_directional": root / WITHIN_DIRECTIONAL_FILE,
        "within_reciprocal": root / WITHIN_RECIPROCAL_FILE,
        "within_directional_strong": root / WITHIN_DIRECTIONAL_STRONG_FILE,
        "within_reciprocal_strong": root / WITHIN_RECIPROCAL_STRONG_FILE,
        "all_directional": root / CONTRAST_DIRECTIONAL_FILE,
        "all_reciprocal": root / CONTRAST_RECIPROCAL_FILE,
        "all_sample_counts": root / CONTRAST_SAMPLE_COUNTS_FILE,
        "contrast_support": root / CONTRAST_SUPPORT_FILE,
    }
    for contrast in tables["contrast_results"]:
        paths[f"{contrast}_directional"] = root / f"{contrast}_directional_limma.csv"
        paths[f"{contrast}_reciprocal"] = root / f"{contrast}_reciprocal_limma.csv"
        paths[f"{contrast}_sample_counts"] = (
            root / f"{contrast}_sample_celltype_counts.csv"
        )
    manifest_path = root / MANIFEST_FILE
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite colocalization outputs: "
            + ", ".join(path.name for path in existing)
        )

    for key in [
        "within_directional",
        "within_reciprocal",
        "within_directional_strong",
        "within_reciprocal_strong",
        "all_directional",
        "all_reciprocal",
        "all_sample_counts",
        "contrast_support",
    ]:
        _write_csv_atomic(paths[key], tables[key])
    for contrast, result in tables["contrast_results"].items():
        _write_csv_atomic(paths[f"{contrast}_directional"], result["directional"])
        _write_csv_atomic(paths[f"{contrast}_reciprocal"], result["reciprocal"])
        _write_csv_atomic(
            paths[f"{contrast}_sample_counts"], result["sample_counts"]
        )

    payload = dict(manifest)
    payload["outputs"] = {
        key: {"path": str(path.resolve()), "rows": int(len(tables[key]))}
        for key, path in paths.items()
        if key in tables and isinstance(tables[key], pd.DataFrame)
    }
    payload["outputs"]["contrasts"] = {
        contrast: {
            key: {
                "path": str(paths[f"{contrast}_{key}"].resolve()),
                "rows": int(len(result[key])),
            }
            for key in ("directional", "reciprocal", "sample_counts")
        }
        for contrast, result in tables["contrast_results"].items()
    }
    with tempfile.NamedTemporaryFile(
        prefix=".colocalization-manifest-",
        suffix=".json",
        dir=root,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    try:
        temporary_manifest.replace(manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()


def load_colocalization_outputs(output_dir: str | Path) -> dict[str, Any]:
    """Load generated tables into the structure used by notebook filters."""

    root = Path(output_dir)
    required = {
        "within_directional": WITHIN_DIRECTIONAL_FILE,
        "within_reciprocal": WITHIN_RECIPROCAL_FILE,
        "all_directional": CONTRAST_DIRECTIONAL_FILE,
        "all_reciprocal": CONTRAST_RECIPROCAL_FILE,
        "all_sample_counts": CONTRAST_SAMPLE_COUNTS_FILE,
        "contrast_support": CONTRAST_SUPPORT_FILE,
    }
    missing = [filename for filename in required.values() if not (root / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing colocalization outputs: "
            + ", ".join(missing)
            + ". Run scripts/build_colocalization_tables.py first."
        )
    outputs: dict[str, Any] = {
        key: pd.read_csv(root / filename) for key, filename in required.items()
    }
    for optional_key, filename in [
        ("within_directional_strong", WITHIN_DIRECTIONAL_STRONG_FILE),
        ("within_reciprocal_strong", WITHIN_RECIPROCAL_STRONG_FILE),
    ]:
        path = root / filename
        outputs[optional_key] = pd.read_csv(path) if path.exists() else None
    manifest_path = root / MANIFEST_FILE
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            outputs["manifest"] = json.load(handle)
    else:
        outputs["manifest"] = None

    contrast_results = {}
    contrast_names = outputs["all_directional"]["contrast"].dropna().astype(str).unique()
    for contrast in contrast_names:
        contrast_results[contrast] = {
            "directional": outputs["all_directional"].loc[
                outputs["all_directional"]["contrast"].astype(str).eq(contrast)
            ].reset_index(drop=True),
            "reciprocal": outputs["all_reciprocal"].loc[
                outputs["all_reciprocal"]["contrast"].astype(str).eq(contrast)
            ].reset_index(drop=True),
            "sample_counts": outputs["all_sample_counts"].loc[
                outputs["all_sample_counts"]["contrast"].astype(str).eq(contrast)
            ].reset_index(drop=True),
        }
    outputs["contrast_results"] = contrast_results
    return outputs


def _filter_values(values: Any) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, int, float)):
        values = [values]
    return {str(value) for value in values}


def filter_within_condition_colocalization(
    outputs: Mapping[str, Any],
    *,
    table: str = "directional",
    conditions: Any = None,
    compartments: Any = None,
    k_values: Any = None,
    cell_types_a: Any = None,
    cell_types_b: Any = None,
    cell_types_any: Any = None,
    include_self_pairs: bool = True,
    min_samples: int | None = None,
    min_positive_samples: int | None = None,
    min_condition_cells: int | None = None,
    min_median_delta: float | None = None,
    max_stouffer_q: float | None = None,
    max_partial_q: float | None = None,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Filter complete within-condition evidence without recomputing it."""

    if table == "directional":
        filtered = outputs["within_directional"].copy()
        cell_type_columns = ["source_cell_type", "neighbor_cell_type"]
        sample_column = "n_samples"
        positive_column = "n_positive_delta"
        count_columns = ["min_source_cells", "min_neighbor_cells"]
        effect_column = "median_delta"
        stouffer_q_column = "stouffer_q_enriched"
        partial_q_column = "partial_q_enriched_3_of_4"
    elif table == "reciprocal":
        filtered = outputs["within_reciprocal"].copy()
        cell_type_columns = ["cell_type_a", "cell_type_b"]
        sample_column = "n_samples_min"
        positive_column = "min_positive_samples"
        count_columns = ["min_cell_type_a", "min_cell_type_b"]
        effect_column = "min_median_delta"
        stouffer_q_column = "both_direction_stouffer_q"
        partial_q_column = "both_direction_partial_q_3_of_4"
    else:
        raise ValueError("table must be 'directional' or 'reciprocal'")
    if not include_self_pairs:
        filtered = filtered.loc[~filtered["is_self_pair"]]

    for column, values in [
        ("condition", conditions),
        ("compartment", compartments),
        ("k", k_values),
        (cell_type_columns[0], cell_types_a),
        (cell_type_columns[1], cell_types_b),
    ]:
        allowed = _filter_values(values)
        if allowed is not None:
            filtered = filtered.loc[filtered[column].astype(str).isin(allowed)]
    any_cell_types = _filter_values(cell_types_any)
    if any_cell_types is not None:
        filtered = filtered.loc[
            filtered[cell_type_columns]
            .astype(str)
            .isin(any_cell_types)
            .any(axis=1)
        ]

    for value, column, description in [
        (min_samples, sample_column, "min_samples"),
        (min_positive_samples, positive_column, "min_positive_samples"),
    ]:
        if value is not None:
            if value < 0:
                raise ValueError(f"{description} must be non-negative")
            filtered = filtered.loc[filtered[column].ge(value)]
    if min_condition_cells is not None:
        if min_condition_cells < 1:
            raise ValueError("min_condition_cells must be at least 1")
        filtered = filtered.loc[
            filtered[count_columns].ge(min_condition_cells).all(axis=1)
        ]
    if min_median_delta is not None:
        filtered = filtered.loc[filtered[effect_column].ge(min_median_delta)]
    if max_stouffer_q is not None:
        filtered = filtered.loc[filtered[stouffer_q_column].le(max_stouffer_q)]
    if max_partial_q is not None:
        filtered = filtered.loc[filtered[partial_q_column].le(max_partial_q)]

    if sort_by is None:
        sort_by = stouffer_q_column
    if sort_by not in filtered:
        raise KeyError(f"sort_by={sort_by!r} is not a result column")
    return filtered.sort_values(
        [sort_by, effect_column],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def within_condition_display_columns(table: str) -> list[str]:
    if table == "directional":
        return [
            "condition",
            "compartment",
            "k",
            "source_cell_type",
            "neighbor_cell_type",
            "n_samples_before_cell_filter",
            "n_samples",
            "mean_coefficient",
            "median_delta",
            "n_positive_delta",
            "min_source_cells",
            "min_neighbor_cells",
            "stouffer_q_enriched",
            "partial_q_enriched_3_of_4",
            "conjunction_q_enriched_4_of_4",
        ]
    if table == "reciprocal":
        return [
            "condition",
            "compartment",
            "k",
            "cell_type_a",
            "cell_type_b",
            "is_self_pair",
            "n_samples_min",
            "mean_coefficient_a_to_b",
            "mean_coefficient_b_to_a",
            "median_delta_a_to_b",
            "median_delta_b_to_a",
            "min_positive_samples",
            "min_cell_type_a",
            "min_cell_type_b",
            "both_direction_stouffer_q",
            "both_direction_partial_q_3_of_4",
            "both_direction_conjunction_q_4_of_4",
        ]
    raise ValueError("table must be 'directional' or 'reciprocal'")


def display_within_condition_colocalization(
    outputs: Mapping[str, Any],
    *,
    display_columns: Sequence[str] | None = None,
    n: int | None = 100,
    **filters: Any,
) -> pd.DataFrame:
    """Filter, display, and return within-condition evidence."""

    from IPython.display import display

    table = str(filters.get("table", "directional"))
    filtered = filter_within_condition_colocalization(outputs, **filters)
    columns = (
        list(display_columns)
        if display_columns is not None
        else within_condition_display_columns(table)
    )
    missing = [column for column in columns if column not in filtered]
    if missing:
        raise KeyError("Unknown display columns: " + ", ".join(missing))
    shown = filtered if n is None else filtered.head(int(n))
    display(shown.loc[:, columns])
    print(f"Showing {len(shown):,} of {len(filtered):,} filtered rows")
    return filtered


def _attach_within_condition_colocalization(
    outputs: Mapping[str, Any],
    contrast: str,
    table: str,
    rows: pd.DataFrame,
    *,
    min_positive_samples: int,
    min_median_delta: float,
    max_stouffer_q: float,
    max_partial_q: float,
) -> pd.DataFrame:
    """Attach condition-specific evidence that a tested relationship is enriched."""

    denominator, numerator = _contrast_conditions(outputs, contrast)
    work = rows.copy()
    work["_analysis_key"] = work["analysis"].astype(str)
    work["_k_key"] = work["k"].astype(str)
    work["_compartment_key"] = work["compartment"].astype(str)
    work["_cell_type_a_key"] = work["cell_type_a"].astype(str)
    work["_cell_type_b_key"] = work["cell_type_b"].astype(str)

    if table == "directional":
        evidence = outputs.get("within_directional")
        if evidence is None:
            raise KeyError(
                "within_directional is required to verify positive colocalization"
            )
        evidence = evidence.copy()
        source = evidence["source_cell_type"].astype(str)
        neighbor = evidence["neighbor_cell_type"].astype(str)
        evidence["_cell_type_a_key"] = np.where(
            source.le(neighbor), source, neighbor
        )
        evidence["_cell_type_b_key"] = np.where(
            source.le(neighbor), neighbor, source
        )
        evidence["_direction_key"] = np.where(
            source.le(neighbor), "a_to_b", "b_to_a"
        )
        work["_direction_key"] = work["direction"].astype(str)
        metric_columns = [
            "median_delta",
            "n_positive_delta",
            "stouffer_q_enriched",
            "partial_q_enriched_3_of_4",
        ]
        positive_metric = "n_positive_delta"
        median_metrics = ["median_delta"]
        stouffer_metric = "stouffer_q_enriched"
        partial_metric = "partial_q_enriched_3_of_4"
        merge_keys = [
            "_analysis_key",
            "_k_key",
            "_compartment_key",
            "_cell_type_a_key",
            "_cell_type_b_key",
            "_direction_key",
        ]
        evidence["_compartment_key"] = evidence["source_compartment"].astype(str)
    else:
        evidence = outputs.get("within_reciprocal")
        if evidence is None:
            raise KeyError(
                "within_reciprocal is required to verify reciprocal colocalization"
            )
        evidence = evidence.copy()
        metric_columns = [
            "median_delta_a_to_b",
            "median_delta_b_to_a",
            "min_positive_samples",
            "both_direction_stouffer_q",
            "both_direction_partial_q_3_of_4",
        ]
        positive_metric = "min_positive_samples"
        median_metrics = ["median_delta_a_to_b", "median_delta_b_to_a"]
        stouffer_metric = "both_direction_stouffer_q"
        partial_metric = "both_direction_partial_q_3_of_4"
        merge_keys = [
            "_analysis_key",
            "_k_key",
            "_compartment_key",
            "_cell_type_a_key",
            "_cell_type_b_key",
        ]
        evidence["_compartment_key"] = evidence["compartment"].astype(str)
        evidence["_cell_type_a_key"] = evidence["cell_type_a"].astype(str)
        evidence["_cell_type_b_key"] = evidence["cell_type_b"].astype(str)

    evidence["_analysis_key"] = evidence["analysis"].astype(str)
    evidence["_k_key"] = evidence["k"].astype(str)
    for prefix, condition in [
        ("denominator", denominator),
        ("numerator", numerator),
    ]:
        condition_evidence = evidence.loc[
            evidence["condition"].astype(str).eq(condition),
            [*merge_keys, *metric_columns],
        ].copy()
        condition_evidence[f"{prefix}_within_colocalized"] = (
            condition_evidence[positive_metric].ge(min_positive_samples)
            & condition_evidence[median_metrics].gt(min_median_delta).all(axis=1)
            & condition_evidence[stouffer_metric].le(max_stouffer_q)
            & condition_evidence[partial_metric].le(max_partial_q)
        )
        condition_evidence = condition_evidence.rename(
            columns={
                column: f"{prefix}_within_{column}"
                for column in metric_columns
            }
        )
        work = work.merge(
            condition_evidence,
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )

    work["colocalized_in_denominator"] = (
        work["denominator_within_colocalized"].eq(True)
    )
    work["colocalized_in_numerator"] = (
        work["numerator_within_colocalized"].eq(True)
    )
    work["colocalized_in_either_condition"] = work[
        ["colocalized_in_denominator", "colocalized_in_numerator"]
    ].any(axis=1)
    return work.drop(
        columns=[column for column in work if column.startswith("_")],
    )


def filter_contrast_table(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    table: str = "directional",
    compartments: Any = None,
    cell_types_a: Any = None,
    cell_types_b: Any = None,
    directions: Any = None,
    effect_direction: str = "both",
    min_positive_samples: int | None = None,
    min_condition_cells: int | None = None,
    min_observed_colocalization: float | None = None,
    require_within_condition_colocalization: bool = False,
    within_min_positive_samples: int = 3,
    within_min_median_delta: float = 0.0,
    within_max_stouffer_q: float = 0.15,
    within_max_partial_q: float = 0.10,
    max_limma_q: float | None = None,
    max_exact_p: float | None = None,
    min_loo_sign_fraction: float | None = None,
    reciprocal_same_direction: bool | None = None,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Filter one condition contrast, including an observed-overlap floor.

    For a directional row, ``min_observed_colocalization`` requires the mean
    observed coefficient to reach the threshold in at least one condition.
    For a reciprocal row, that same rule must hold separately for A-to-B and
    B-to-A. Requiring a minimum in *both* conditions would discard biologically
    interesting appearance or loss of colocalization. When the within-condition
    gate is enabled, a directional result must pass the one-way enrichment
    criteria in at least one condition; a reciprocal result must pass those
    criteria in both directions in at least one condition.
    """

    contrast_results = outputs["contrast_results"]
    if contrast not in contrast_results:
        raise KeyError(
            f"Unknown contrast {contrast!r}. Available: {list(contrast_results)}"
        )
    if table not in {"directional", "reciprocal"}:
        raise ValueError("table must be 'directional' or 'reciprocal'")
    if effect_direction not in {"both", "positive", "negative"}:
        raise ValueError("effect_direction must be 'both', 'positive', or 'negative'")

    filtered = contrast_results[contrast][table].copy()
    for column, values in [
        ("compartment", compartments),
        ("cell_type_a", cell_types_a),
        ("cell_type_b", cell_types_b),
    ]:
        allowed = _filter_values(values)
        if allowed is not None:
            filtered = filtered.loc[filtered[column].astype(str).isin(allowed)]

    if table == "directional":
        allowed_directions = _filter_values(directions)
        if allowed_directions is not None:
            filtered = filtered.loc[
                filtered["direction"].astype(str).isin(allowed_directions)
            ]
        effect_columns = ["contrast_delta"]
        positive_columns = [
            "denominator_positive_samples",
            "numerator_positive_samples",
        ]
        limma_column = "limma_q"
        exact_column = "exact_permutation_p"
        loo_column = "loo_sign_fraction"
        colocalization_column = "max_condition_mean_coefficient"
    else:
        if directions is not None:
            raise ValueError("directions applies only to table='directional'")
        effect_columns = ["contrast_delta_a_to_b", "contrast_delta_b_to_a"]
        positive_columns = [
            "denominator_positive_samples_a_to_b",
            "numerator_positive_samples_a_to_b",
            "denominator_positive_samples_b_to_a",
            "numerator_positive_samples_b_to_a",
        ]
        limma_column = "both_direction_limma_q_max"
        exact_column = "both_direction_exact_p_max"
        loo_column = "both_direction_loo_sign_fraction_min"
        colocalization_column = "min_direction_max_condition_mean_coefficient"
        if reciprocal_same_direction is not None:
            filtered = filtered.loc[
                filtered["reciprocal_same_direction"].eq(
                    bool(reciprocal_same_direction)
                )
            ]

    if effect_direction == "positive":
        filtered = filtered.loc[filtered[effect_columns].gt(0).all(axis=1)]
    elif effect_direction == "negative":
        filtered = filtered.loc[filtered[effect_columns].lt(0).all(axis=1)]
    if min_positive_samples is not None:
        filtered = filtered.loc[
            filtered[positive_columns].ge(min_positive_samples).all(axis=1)
        ]
    minimum_count_columns = [
        column
        for column in filtered.columns
        if column.startswith("min_n_cell_type_")
    ]
    if min_condition_cells is not None:
        if not minimum_count_columns:
            raise KeyError("No condition-minimum cell-count columns are available")
        filtered = filtered.loc[
            filtered[minimum_count_columns].ge(min_condition_cells).all(axis=1)
        ]
    if min_observed_colocalization is not None:
        if not 0 <= min_observed_colocalization <= 1:
            raise ValueError("min_observed_colocalization must be between 0 and 1")
        if colocalization_column not in filtered:
            raise KeyError(
                f"Missing {colocalization_column!r}; regenerate the tables with "
                "scripts/build_colocalization_tables.py"
            )
        filtered = filtered.loc[
            filtered[colocalization_column].ge(min_observed_colocalization)
        ]
    if require_within_condition_colocalization:
        filtered = _attach_within_condition_colocalization(
            outputs,
            contrast,
            table,
            filtered,
            min_positive_samples=int(within_min_positive_samples),
            min_median_delta=float(within_min_median_delta),
            max_stouffer_q=float(within_max_stouffer_q),
            max_partial_q=float(within_max_partial_q),
        )
        filtered = filtered.loc[filtered["colocalized_in_either_condition"]]
    if max_limma_q is not None:
        filtered = filtered.loc[filtered[limma_column].le(max_limma_q)]
    if max_exact_p is not None:
        filtered = filtered.loc[filtered[exact_column].le(max_exact_p)]
    if min_loo_sign_fraction is not None:
        filtered = filtered.loc[filtered[loo_column].ge(min_loo_sign_fraction)]

    if sort_by is None:
        sort_by = limma_column
    if sort_by not in filtered:
        raise KeyError(f"sort_by={sort_by!r} is not a result column")
    return filtered.sort_values(sort_by, kind="stable").reset_index(drop=True)


def contrast_display_columns(
    table: str,
    columns: Sequence[str],
    *,
    include_sample_counts: bool,
) -> list[str]:
    minimum_count_columns = [
        column for column in columns if column.startswith("min_n_cell_type_")
    ]
    sample_count_columns = [
        column for column in columns if column.startswith("n_cell_type_")
    ]
    common = [
        "contrast",
        "compartment",
        "cell_type_a",
        "cell_type_b",
        "cell_type_pair",
    ]
    within_evidence = [
        column
        for column in [
            "colocalized_in_denominator",
            "colocalized_in_numerator",
            "colocalized_in_either_condition",
        ]
        if column in columns
    ]
    if table == "directional":
        statistics = [
            "direction",
            "denominator_mean_coefficient",
            "numerator_mean_coefficient",
            *within_evidence,
            "contrast_delta",
            "max_condition_mean_coefficient",
            "limma_p",
            "limma_q",
            "exact_permutation_p",
            "denominator_positive_samples",
            "numerator_positive_samples",
            "loo_sign_fraction",
        ]
    elif table == "reciprocal":
        statistics = [
            "denominator_mean_coefficient_a_to_b",
            "numerator_mean_coefficient_a_to_b",
            "denominator_mean_coefficient_b_to_a",
            "numerator_mean_coefficient_b_to_a",
            *within_evidence,
            "contrast_delta_a_to_b",
            "contrast_delta_b_to_a",
            "min_direction_max_condition_mean_coefficient",
            "limma_q_a_to_b",
            "limma_q_b_to_a",
            "both_direction_limma_q_max",
            "exact_permutation_p_a_to_b",
            "exact_permutation_p_b_to_a",
            "reciprocal_same_direction",
            "loo_sign_fraction_a_to_b",
            "loo_sign_fraction_b_to_a",
            "both_direction_loo_sign_fraction_min",
        ]
    else:
        raise ValueError("table must be 'directional' or 'reciprocal'")
    result = [*common, *statistics, *minimum_count_columns]
    if include_sample_counts:
        result.extend(sample_count_columns)
    return result


def display_contrast_table(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    table: str = "directional",
    include_sample_counts: bool = False,
    display_columns: Sequence[str] | None = None,
    n: int | None = 100,
    **filters: Any,
) -> pd.DataFrame:
    """Filter, display, and return one condition-contrast table."""

    from IPython.display import display

    filtered = filter_contrast_table(
        outputs,
        contrast,
        table=table,
        **filters,
    )
    columns = (
        list(display_columns)
        if display_columns is not None
        else contrast_display_columns(
            table,
            filtered.columns,
            include_sample_counts=include_sample_counts,
        )
    )
    missing = [column for column in columns if column not in filtered]
    if missing:
        raise KeyError("Unknown display columns: " + ", ".join(missing))
    shown = filtered if n is None else filtered.head(int(n))
    display(shown.loc[:, columns])
    print(f"Showing {len(shown):,} of {len(filtered):,} filtered rows")
    return filtered


def _contrast_conditions(
    outputs: Mapping[str, Any],
    contrast: str,
) -> tuple[str, str]:
    """Return denominator and numerator labels for one saved contrast."""

    manifest = outputs.get("manifest", {})
    specifications = (
        manifest.get("configuration", {}).get("contrasts", [])
        if isinstance(manifest, Mapping)
        else []
    )
    for specification in specifications:
        denominator = str(specification["denominator"])
        numerator = str(specification["numerator"])
        if f"{numerator}_vs_{denominator}" == contrast:
            return denominator, numerator
    if "_vs_" in contrast:
        numerator, denominator = contrast.rsplit("_vs_", maxsplit=1)
        return denominator, numerator
    raise ValueError(
        f"Cannot determine numerator and denominator for contrast {contrast!r}"
    )


def prepare_directional_contrast_plot(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    top_n: int = 15,
    rank_by: str = "max_abs_contrast",
    **filters: Any,
) -> pd.DataFrame:
    """Prepare top one-way relationships after directional colocalization filters."""

    if int(top_n) < 1:
        raise ValueError("top_n must be at least 1")
    filtered = filter_contrast_table(
        outputs,
        contrast,
        table="directional",
        **filters,
    ).copy()
    if filtered.empty:
        raise ValueError(f"No directional rows passed the filters for {contrast!r}")
    filtered["_max_abs_contrast"] = filtered["contrast_delta"].abs()
    if rank_by == "max_abs_contrast":
        filtered = filtered.sort_values(
            ["_max_abs_contrast", "exact_permutation_p"],
            ascending=[False, True],
            kind="stable",
        )
    elif rank_by == "exact_p":
        filtered = filtered.sort_values(
            ["exact_permutation_p", "_max_abs_contrast"],
            ascending=[True, False],
            kind="stable",
        )
    else:
        raise ValueError("rank_by must be 'max_abs_contrast' or 'exact_p'")

    selected = filtered.head(int(top_n)).reset_index(drop=True)
    denominator, numerator = _contrast_conditions(outputs, contrast)
    records: list[dict[str, Any]] = []
    for relationship_rank, row in selected.iterrows():
        if row["direction"] == "a_to_b":
            source, target = row["cell_type_a"], row["cell_type_b"]
        else:
            source, target = row["cell_type_b"], row["cell_type_a"]
        denominator_coefficient = row["denominator_mean_coefficient"]
        numerator_coefficient = row["numerator_mean_coefficient"]
        records.append(
            {
                "contrast": contrast,
                "compartment": str(row["compartment"]),
                "cell_type_pair": row["cell_type_pair"],
                "direction": row["direction"],
                "source_cell_type": str(source),
                "neighbor_cell_type": str(target),
                "display_label": (
                    f"Domain {row['compartment']} · {source} → {target}"
                ),
                "denominator_condition": denominator,
                "numerator_condition": numerator,
                "denominator_mean_coefficient": denominator_coefficient,
                "numerator_mean_coefficient": numerator_coefficient,
                "coefficient_change": (
                    numerator_coefficient - denominator_coefficient
                ),
                "denominator_mean_delta": row["denominator_mean_delta"],
                "numerator_mean_delta": row["numerator_mean_delta"],
                "contrast_delta": row["contrast_delta"],
                "exact_permutation_p": row["exact_permutation_p"],
                "limma_q": row["limma_q"],
                "loo_sign_fraction": row["loo_sign_fraction"],
                "colocalized_in_denominator": bool(
                    row.get("colocalized_in_denominator", False)
                ),
                "colocalized_in_numerator": bool(
                    row.get("colocalized_in_numerator", False)
                ),
                "relationship_rank": relationship_rank + 1,
                "max_abs_contrast": row["_max_abs_contrast"],
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_reciprocal_contrast_plot(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    top_n: int = 12,
    rank_by: str = "max_abs_contrast",
    **filters: Any,
) -> pd.DataFrame:
    """Prepare two directed rows per reciprocal pair for visual review.

    Pair selection happens after the ordinary contrast filters. The default
    ranking uses the larger absolute null-adjusted contrast across A-to-B and
    B-to-A, while retaining both directions next to each other.
    """

    if int(top_n) < 1:
        raise ValueError("top_n must be at least 1")
    filtered = filter_contrast_table(
        outputs,
        contrast,
        table="reciprocal",
        **filters,
    ).copy()
    if filtered.empty:
        raise ValueError(f"No reciprocal rows passed the filters for {contrast!r}")

    filtered["_max_abs_contrast"] = filtered[
        ["contrast_delta_a_to_b", "contrast_delta_b_to_a"]
    ].abs().max(axis=1)
    if rank_by == "max_abs_contrast":
        filtered = filtered.sort_values(
            ["_max_abs_contrast", "both_direction_exact_p_max"],
            ascending=[False, True],
            kind="stable",
        )
    elif rank_by == "exact_p":
        filtered = filtered.sort_values(
            ["both_direction_exact_p_max", "_max_abs_contrast"],
            ascending=[True, False],
            kind="stable",
        )
    else:
        raise ValueError("rank_by must be 'max_abs_contrast' or 'exact_p'")

    selected = filtered.head(int(top_n)).reset_index(drop=True)
    denominator, numerator = _contrast_conditions(outputs, contrast)
    records: list[dict[str, Any]] = []
    for pair_rank, row in selected.iterrows():
        for direction, source, target in [
            ("a_to_b", row["cell_type_a"], row["cell_type_b"]),
            ("b_to_a", row["cell_type_b"], row["cell_type_a"]),
        ]:
            denominator_coefficient = row[
                f"denominator_mean_coefficient_{direction}"
            ]
            numerator_coefficient = row[f"numerator_mean_coefficient_{direction}"]
            records.append(
                {
                    "contrast": contrast,
                    "compartment": str(row["compartment"]),
                    "cell_type_pair": row["cell_type_pair"],
                    "direction": direction,
                    "source_cell_type": str(source),
                    "neighbor_cell_type": str(target),
                    "display_label": (
                        f"Domain {row['compartment']} · {source} → {target}"
                    ),
                    "denominator_condition": denominator,
                    "numerator_condition": numerator,
                    "denominator_mean_coefficient": denominator_coefficient,
                    "numerator_mean_coefficient": numerator_coefficient,
                    "coefficient_change": (
                        numerator_coefficient - denominator_coefficient
                    ),
                    "denominator_mean_delta": row[
                        f"denominator_mean_delta_{direction}"
                    ],
                    "numerator_mean_delta": row[
                        f"numerator_mean_delta_{direction}"
                    ],
                    "contrast_delta": row[f"contrast_delta_{direction}"],
                    "exact_permutation_p": row[
                        f"exact_permutation_p_{direction}"
                    ],
                    "limma_q": row[f"limma_q_{direction}"],
                    "loo_sign_fraction": row[f"loo_sign_fraction_{direction}"],
                    "both_direction_exact_p_max": row[
                        "both_direction_exact_p_max"
                    ],
                    "both_direction_loo_sign_fraction_min": row[
                        "both_direction_loo_sign_fraction_min"
                    ],
                    "colocalized_in_denominator": bool(
                        row.get("colocalized_in_denominator", False)
                    ),
                    "colocalized_in_numerator": bool(
                        row.get("colocalized_in_numerator", False)
                    ),
                    "relationship_rank": pair_rank + 1,
                    "max_abs_contrast": row["_max_abs_contrast"],
                }
            )
    return pd.DataFrame.from_records(records)


def _plot_contrast_overview(
    plot_data: pd.DataFrame,
    contrast: str,
    *,
    top_n: int,
    rank_by: str = "max_abs_contrast",
    title: str | None = None,
    result_kind: str,
    filters: Mapping[str, Any],
):
    """Render aligned prevalence, colocalization, and contrast panels."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    denominator = str(plot_data["denominator_condition"].iloc[0])
    numerator = str(plot_data["numerator_condition"].iloc[0])
    labels = plot_data["display_label"].tolist()
    category_order = list(reversed(labels))
    figure = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        column_widths=[0.36, 0.34, 0.30],
        horizontal_spacing=0.06,
        subplot_titles=(
            "Observed neighborhood coefficient",
            "Colocalization relative to permutation null",
            f"Null-adjusted change ({numerator} − {denominator})",
        ),
    )

    for row in plot_data.itertuples(index=False):
        figure.add_trace(
            go.Scatter(
                x=[
                    row.denominator_mean_coefficient,
                    row.numerator_mean_coefficient,
                ],
                y=[row.display_label, row.display_label],
                mode="lines",
                line={"color": "#B8B8B8", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    hover_fields = plot_data[
        [
            "denominator_mean_coefficient",
            "numerator_mean_coefficient",
            "coefficient_change",
            "contrast_delta",
            "exact_permutation_p",
            "loo_sign_fraction",
        ]
    ].to_numpy()
    figure.add_trace(
        go.Scatter(
            x=plot_data["denominator_mean_coefficient"],
            y=plot_data["display_label"],
            mode="markers",
            name=denominator,
            marker={"color": "#4C78A8", "size": 10},
            customdata=hover_fields,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + denominator
                + ": %{x:.3f}<br>"
                + numerator
                + ": %{customdata[1]:.3f}<br>"
                "Raw coefficient change: %{customdata[2]:+.3f}<br>"
                "Null-adjusted change: %{customdata[3]:+.3f}<br>"
                "Exact p: %{customdata[4]:.4f}<br>"
                "LOO sign fraction: %{customdata[5]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=plot_data["numerator_mean_coefficient"],
            y=plot_data["display_label"],
            mode="markers",
            name=numerator,
            marker={"color": "#F58518", "size": 10},
            customdata=hover_fields,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + numerator
                + ": %{x:.3f}<br>"
                + denominator
                + ": %{customdata[0]:.3f}<br>"
                "Raw coefficient change: %{customdata[2]:+.3f}<br>"
                "Null-adjusted change: %{customdata[3]:+.3f}<br>"
                "Exact p: %{customdata[4]:.4f}<br>"
                "LOO sign fraction: %{customdata[5]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    delta_line_x: list[float | None] = []
    delta_line_y: list[str | None] = []
    for row in plot_data.itertuples(index=False):
        delta_line_x.extend(
            [
                float(row.denominator_mean_delta),
                float(row.numerator_mean_delta),
                None,
            ]
        )
        delta_line_y.extend([row.display_label, row.display_label, None])
    figure.add_trace(
        go.Scatter(
            x=delta_line_x,
            y=delta_line_y,
            mode="lines",
            line={"color": "#B8B8B8", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    delta_hover = plot_data[
        [
            "denominator_mean_delta",
            "numerator_mean_delta",
            "contrast_delta",
            "exact_permutation_p",
            "colocalized_in_denominator",
            "colocalized_in_numerator",
        ]
    ].to_numpy()
    figure.add_trace(
        go.Scatter(
            x=plot_data["denominator_mean_delta"],
            y=plot_data["display_label"],
            mode="markers",
            name=f"{denominator} null-adjusted",
            showlegend=False,
            marker={
                "color": "#4C78A8",
                "size": 10,
                "symbol": [
                    "circle" if passed else "circle-open"
                    for passed in plot_data["colocalized_in_denominator"]
                ],
                "line": {"color": "#4C78A8", "width": 2},
            },
            customdata=delta_hover,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + denominator
                + " observed − null: %{x:+.3f}<br>"
                + numerator
                + " observed − null: %{customdata[1]:+.3f}<br>"
                "Differential change: %{customdata[2]:+.3f}<br>"
                "Exact p: %{customdata[3]:.4f}<br>"
                + denominator
                + " passes colocalization gate: %{customdata[4]}<br>"
                + numerator
                + " passes colocalization gate: %{customdata[5]}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=plot_data["numerator_mean_delta"],
            y=plot_data["display_label"],
            mode="markers",
            name=f"{numerator} null-adjusted",
            showlegend=False,
            marker={
                "color": "#F58518",
                "size": 10,
                "symbol": [
                    "circle" if passed else "circle-open"
                    for passed in plot_data["colocalized_in_numerator"]
                ],
                "line": {"color": "#F58518", "width": 2},
            },
            customdata=delta_hover,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + numerator
                + " observed − null: %{x:+.3f}<br>"
                + denominator
                + " observed − null: %{customdata[0]:+.3f}<br>"
                "Differential change: %{customdata[2]:+.3f}<br>"
                "Exact p: %{customdata[3]:.4f}<br>"
                + denominator
                + " passes colocalization gate: %{customdata[4]}<br>"
                + numerator
                + " passes colocalization gate: %{customdata[5]}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    effect_line_x: list[float | None] = []
    effect_line_y: list[str | None] = []
    for row in plot_data.itertuples(index=False):
        effect_line_x.extend([0.0, float(row.contrast_delta), None])
        effect_line_y.extend([row.display_label, row.display_label, None])
    figure.add_trace(
        go.Scatter(
            x=effect_line_x,
            y=effect_line_y,
            mode="lines",
            line={"color": "#B8B8B8", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    effect_hover = plot_data[
        [
            "denominator_mean_coefficient",
            "numerator_mean_coefficient",
            "exact_permutation_p",
            "limma_q",
            "loo_sign_fraction",
        ]
    ].to_numpy()
    max_effect = max(float(plot_data["contrast_delta"].abs().max()), 0.01)
    figure.add_trace(
        go.Scatter(
            x=plot_data["contrast_delta"],
            y=plot_data["display_label"],
            mode="markers",
            name="Null-adjusted change",
            marker={
                "color": plot_data["contrast_delta"],
                "colorscale": "RdBu_r",
                "cmin": -max_effect,
                "cmax": max_effect,
                "cmid": 0,
                "line": {"color": "white", "width": 0.8},
                "size": 11,
                "showscale": False,
            },
            customdata=effect_hover,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Null-adjusted change: %{x:+.3f}<br>"
                + denominator
                + " coefficient: %{customdata[0]:.3f}<br>"
                + numerator
                + " coefficient: %{customdata[1]:.3f}<br>"
                "Exact p: %{customdata[2]:.4f}<br>"
                "Limma q: %{customdata[3]:.4f}<br>"
                "LOO sign fraction: %{customdata[4]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=3,
    )

    coefficient_high = min(
        1.0,
        max(
            0.10,
            float(
                plot_data[
                    [
                        "denominator_mean_coefficient",
                        "numerator_mean_coefficient",
                    ]
                ]
                .max()
                .max()
            )
            * 1.08,
        ),
    )
    max_condition_delta = max(
        0.01,
        float(
            plot_data[
                ["denominator_mean_delta", "numerator_mean_delta"]
            ].abs().max().max()
        ),
    )
    figure.update_xaxes(
        title_text="Fraction of source cells neighboring target",
        range=[0, coefficient_high],
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Mean observed − permutation-null overlap",
        range=[-1.15 * max_condition_delta, 1.15 * max_condition_delta],
        zeroline=True,
        zerolinecolor="#666666",
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="Between-condition change in observed − null",
        range=[-1.15 * max_effect, 1.15 * max_effect],
        zeroline=True,
        zerolinecolor="#666666",
        row=1,
        col=3,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=category_order,
        title_text="Directed cell-type relationship",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=category_order,
        row=1,
        col=2,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=category_order,
        row=1,
        col=3,
    )
    if title is None:
        title = (
            f"{result_kind.capitalize()} colocalization: "
            f"{numerator} vs {denominator}"
        )
    ranking_label = (
        "largest absolute null-adjusted change"
        if rank_by == "max_abs_contrast"
        else "exact permutation p-value"
    )
    reciprocal_note = (
        " Each reciprocal pair is shown in both directions."
        if result_kind == "reciprocal"
        else ""
    )
    if filters.get("require_within_condition_colocalization", False):
        selection_note = (
            "Rows first pass positive within-condition colocalization in at "
            "least one condition, then are ranked by "
            + ranking_label
            + ". Filled middle-panel points mark conditions that pass the "
            "colocalization gate."
        )
    else:
        selection_note = "Rows are ranked by " + ranking_label + "."
    figure.update_layout(
        title={
            "text": (
                title
                + "<br><sup>"
                + selection_note
                + reciprocal_note
                + "</sup>"
            )
        },
        template="plotly_white",
        height=max(520, 33 * len(plot_data) + 190),
        width=1650,
        legend_title_text="Condition",
        margin={"l": 250, "r": 40, "t": 105, "b": 70},
        meta={
            "contrast": contrast,
            "denominator": denominator,
            "numerator": numerator,
            "result_kind": result_kind,
            "top_n_relationships": int(top_n),
            "shown_relationships": int(
                plot_data["relationship_rank"].nunique()
            ),
            "shown_directed_rows": int(len(plot_data)),
            "rank_by": rank_by,
            "filters": {key: value for key, value in filters.items()},
        },
    )
    return figure


def plot_directional_contrast_overview(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    top_n: int = 15,
    rank_by: str = "max_abs_contrast",
    title: str | None = None,
    **filters: Any,
):
    """Plot directional colocalization as the primary contrast review."""

    filters = dict(filters)
    filters.setdefault("require_within_condition_colocalization", True)
    plot_data = prepare_directional_contrast_plot(
        outputs,
        contrast,
        top_n=top_n,
        rank_by=rank_by,
        **filters,
    )
    return _plot_contrast_overview(
        plot_data,
        contrast,
        top_n=top_n,
        rank_by=rank_by,
        title=title,
        result_kind="directional",
        filters=filters,
    )


def plot_reciprocal_contrast_overview(
    outputs: Mapping[str, Any],
    contrast: str,
    *,
    top_n: int = 12,
    rank_by: str = "max_abs_contrast",
    title: str | None = None,
    **filters: Any,
):
    """Plot reciprocal colocalization as a secondary confirmation view."""

    filters = dict(filters)
    filters.setdefault("require_within_condition_colocalization", True)
    plot_data = prepare_reciprocal_contrast_plot(
        outputs,
        contrast,
        top_n=top_n,
        rank_by=rank_by,
        **filters,
    )
    return _plot_contrast_overview(
        plot_data,
        contrast,
        top_n=top_n,
        rank_by=rank_by,
        title=title,
        result_kind="reciprocal",
        filters=filters,
    )
