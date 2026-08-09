"""Biological-sample condition tests for cNMF usage programs.

Cell-level usages are first averaged within each biological sample and cell
type.  Inferential tests therefore use samples, not cells, as replicates.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def _usage_columns(
    obs: pd.DataFrame,
    usage_cols: Sequence[str] | None,
) -> list[str]:
    columns = (
        [str(column) for column in usage_cols]
        if usage_cols is not None
        else [str(column) for column in obs if str(column).startswith("Usage_")]
    )
    if not columns:
        raise ValueError("No Usage_ columns are available")
    missing = [column for column in columns if column not in obs]
    if missing:
        raise KeyError("Unknown usage columns: " + ", ".join(missing))
    return columns


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment that preserves missing values and index."""

    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.notna() & np.isfinite(p_values.to_numpy(dtype=float))
    if not finite.any():
        return adjusted
    values = p_values.loc[finite].to_numpy(dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    n_tests = len(ranked)
    corrected = ranked * n_tests / np.arange(1, n_tests + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty(n_tests, dtype=float)
    restored[order] = corrected
    adjusted.loc[p_values.index[finite]] = restored
    return adjusted


def sample_celltype_usage_table(
    obs: pd.DataFrame,
    *,
    cell_type_key: str,
    sample_key: str,
    condition_key: str,
    usage_cols: Sequence[str] | None = None,
    cell_types: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Return one cNMF usage summary per cell type, sample, and program."""

    required = [cell_type_key, sample_key, condition_key]
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError("Missing usage metadata columns: " + ", ".join(missing))
    programs = _usage_columns(obs, usage_cols)
    frame = obs[[*required, *programs]].copy()

    for column in required:
        labels = frame[column].astype("string")
        frame = frame.loc[labels.notna() & labels.str.strip().ne("")].copy()
        frame[column] = frame[column].astype(str)
    if frame.empty:
        raise ValueError("No cells remain after excluding missing metadata")

    available = frame[cell_type_key].drop_duplicates().tolist()
    if cell_types is not None:
        requested = [cell_types] if isinstance(cell_types, str) else cell_types
        selected = list(dict.fromkeys(str(value) for value in requested))
        if not selected:
            raise ValueError("cell_types must contain at least one label")
        unknown = sorted(set(selected).difference(available))
        if unknown:
            raise KeyError("Unknown cell types: " + ", ".join(unknown))
        frame = frame.loc[frame[cell_type_key].isin(selected)].copy()

    sample_condition = frame[[sample_key, condition_key]].drop_duplicates()
    duplicated = sample_condition[sample_key].duplicated(keep=False)
    if duplicated.any():
        bad = sorted(sample_condition.loc[duplicated, sample_key].unique())
        raise ValueError("Samples map to multiple conditions: " + ", ".join(bad))

    group_columns = [cell_type_key, condition_key, sample_key]
    grouped = frame.groupby(group_columns, observed=True, sort=False)
    counts = grouped.size().rename("n_cells").reset_index()
    means = grouped[programs].mean().reset_index().melt(
        id_vars=group_columns,
        var_name="program",
        value_name="mean_usage",
    )
    medians = grouped[programs].median().reset_index().melt(
        id_vars=group_columns,
        var_name="program",
        value_name="median_usage",
    )
    return (
        means.merge(medians, on=[*group_columns, "program"], validate="one_to_one")
        .merge(counts, on=group_columns, validate="many_to_one")
        .sort_values([cell_type_key, "program", condition_key, sample_key])
        .reset_index(drop=True)
    )


def _permutation_pvalue(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    max_exact_combinations: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    """Two-sided permutation p-value for a difference in sample means."""

    values = np.concatenate([group_a, group_b]).astype(float, copy=False)
    n_a = len(group_a)
    n_total = len(values)
    observed = float(np.mean(group_a) - np.mean(group_b))
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8
    n_combinations = comb(n_total, n_a)

    if n_combinations <= int(max_exact_combinations):
        extreme = 0
        all_indices = np.arange(n_total)
        for selected_tuple in combinations(range(n_total), n_a):
            selected = np.fromiter(selected_tuple, dtype=int, count=n_a)
            mask = np.ones(n_total, dtype=bool)
            mask[selected] = False
            difference = values[selected].mean() - values[all_indices[mask]].mean()
            extreme += abs(difference) >= abs(observed) - tolerance
        return extreme / n_combinations, "exact", n_combinations

    extreme = 0
    for _ in range(int(n_permutations)):
        selected = rng.choice(n_total, size=n_a, replace=False)
        mask = np.ones(n_total, dtype=bool)
        mask[selected] = False
        difference = values[selected].mean() - values[mask].mean()
        extreme += abs(difference) >= abs(observed) - tolerance
    p_value = (extreme + 1) / (int(n_permutations) + 1)
    return p_value, "monte_carlo", int(n_permutations)


def _normalize_contrasts(
    conditions: Sequence[str],
    contrasts: Sequence[tuple[str, str]] | tuple[str, str] | None,
) -> list[tuple[str, str]]:
    available = list(dict.fromkeys(str(value) for value in conditions))
    if contrasts is None:
        selected = list(combinations(available, 2))
    elif (
        len(contrasts) == 2
        and all(isinstance(value, str) for value in contrasts)
    ):
        selected = [(str(contrasts[0]), str(contrasts[1]))]
    else:
        selected = [(str(a), str(b)) for a, b in contrasts]
    if not selected:
        raise ValueError("At least one condition contrast is required")
    bad = sorted({value for pair in selected for value in pair}.difference(available))
    if bad:
        raise KeyError("Unknown contrast conditions: " + ", ".join(bad))
    if any(a == b for a, b in selected):
        raise ValueError("A contrast must contain two different conditions")
    return list(dict.fromkeys(selected))


def test_usage_condition_changes(
    obs: pd.DataFrame,
    *,
    cell_type_key: str,
    sample_key: str,
    condition_key: str,
    contrasts: Sequence[tuple[str, str]] | tuple[str, str] | None = None,
    usage_cols: Sequence[str] | None = None,
    cell_types: Sequence[str] | str | None = None,
    min_cells_per_sample: int = 10,
    min_samples_per_condition: int = 3,
    max_exact_combinations: int = 200_000,
    n_permutations: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Test condition shifts for every cell-type and cNMF-program combination.

    The reported effect is the mean of biological-sample means in condition A
    minus the corresponding mean in condition B. Welch tests and permutation
    tests use those biological-sample means. Benjamini-Hochberg q-values are
    reported both within each contrast and globally across all requested
    contrasts.
    """

    if int(min_cells_per_sample) < 1:
        raise ValueError("min_cells_per_sample must be at least 1")
    if int(min_samples_per_condition) < 2:
        raise ValueError("min_samples_per_condition must be at least 2")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1")

    sample_table = sample_celltype_usage_table(
        obs,
        cell_type_key=cell_type_key,
        sample_key=sample_key,
        condition_key=condition_key,
        usage_cols=usage_cols,
        cell_types=cell_types,
    )
    selected_contrasts = _normalize_contrasts(
        sample_table[condition_key].drop_duplicates().tolist(), contrasts
    )
    retained = sample_table.loc[
        sample_table["n_cells"].ge(int(min_cells_per_sample))
    ].copy()
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    feature_columns = [cell_type_key, "program"]
    for feature, feature_table in sample_table.groupby(
        feature_columns, observed=True, sort=False
    ):
        cell_type, program = map(str, feature)
        retained_feature = retained.loc[
            retained[cell_type_key].astype(str).eq(cell_type)
            & retained["program"].astype(str).eq(program)
        ]
        for condition_a, condition_b in selected_contrasts:
            group_a = retained_feature.loc[
                retained_feature[condition_key].eq(condition_a), "mean_usage"
            ].dropna().to_numpy(dtype=float)
            group_b = retained_feature.loc[
                retained_feature[condition_key].eq(condition_b), "mean_usage"
            ].dropna().to_numpy(dtype=float)
            cells_a = retained_feature.loc[
                retained_feature[condition_key].eq(condition_a), "n_cells"
            ].to_numpy(dtype=int)
            cells_b = retained_feature.loc[
                retained_feature[condition_key].eq(condition_b), "n_cells"
            ].to_numpy(dtype=int)
            supported = (
                len(group_a) >= int(min_samples_per_condition)
                and len(group_b) >= int(min_samples_per_condition)
            )
            record: dict[str, object] = {
                cell_type_key: cell_type,
                "program": program,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "contrast": f"{condition_a}_vs_{condition_b}",
                "n_samples_a": len(group_a),
                "n_samples_b": len(group_b),
                "min_n_cells_a": int(cells_a.min()) if len(cells_a) else 0,
                "min_n_cells_b": int(cells_b.min()) if len(cells_b) else 0,
                "median_n_cells_a": float(np.median(cells_a)) if len(cells_a) else np.nan,
                "median_n_cells_b": float(np.median(cells_b)) if len(cells_b) else np.nan,
                "mean_usage_a": float(np.mean(group_a)) if len(group_a) else np.nan,
                "mean_usage_b": float(np.mean(group_b)) if len(group_b) else np.nan,
                "delta_a_minus_b": (
                    float(np.mean(group_a) - np.mean(group_b))
                    if len(group_a) and len(group_b)
                    else np.nan
                ),
                "tested": supported,
                "filter_reason": (
                    ""
                    if supported
                    else "insufficient samples after min_cells_per_sample filter"
                ),
                "welch_t": np.nan,
                "welch_p": np.nan,
                "permutation_p": np.nan,
                "permutation_method": "",
                "permutation_draws": 0,
            }
            if supported:
                welch = ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
                permutation_p, permutation_method, permutation_draws = (
                    _permutation_pvalue(
                        group_a,
                        group_b,
                        max_exact_combinations=int(max_exact_combinations),
                        n_permutations=int(n_permutations),
                        rng=rng,
                    )
                )
                record.update(
                    {
                        "welch_t": float(welch.statistic),
                        "welch_p": float(welch.pvalue),
                        "permutation_p": float(permutation_p),
                        "permutation_method": permutation_method,
                        "permutation_draws": permutation_draws,
                    }
                )
            rows.append(record)

    result = pd.DataFrame(rows)
    result["abs_delta"] = result["delta_a_minus_b"].abs()
    for p_column, q_stem in [
        ("welch_p", "welch_q"),
        ("permutation_p", "permutation_q"),
    ]:
        result[f"{q_stem}_within_contrast"] = (
            result.groupby("contrast", sort=False, group_keys=False)[p_column]
            .apply(_bh_adjust)
            .reindex(result.index)
        )
        result[f"{q_stem}_global"] = _bh_adjust(result[p_column])
    return result.sort_values(
        ["welch_q_global", "welch_p", "abs_delta"],
        ascending=[True, True, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def select_usage_condition_result(
    results: pd.DataFrame,
    *,
    contrast: tuple[str, str],
    cell_type_key: str,
    cell_type: str | None = None,
    program: str | None = None,
    sort_by: str = "welch_q_within_contrast",
) -> pd.Series:
    """Select an explicit result or the best supported result for a contrast."""

    required = {
        cell_type_key,
        "program",
        "condition_a",
        "condition_b",
        "tested",
        sort_by,
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError("Missing condition-test columns: " + ", ".join(missing))
    condition_a, condition_b = map(str, contrast)
    selected = results.loc[
        results["condition_a"].astype(str).eq(condition_a)
        & results["condition_b"].astype(str).eq(condition_b)
        & results["tested"].astype(bool)
    ].copy()
    if cell_type is not None:
        selected = selected.loc[
            selected[cell_type_key].astype(str).eq(str(cell_type))
        ]
    if program is not None:
        selected = selected.loc[selected["program"].astype(str).eq(str(program))]
    if selected.empty:
        details = f"contrast={condition_a} vs {condition_b}"
        if cell_type is not None:
            details += f", {cell_type_key}={cell_type}"
        if program is not None:
            details += f", program={program}"
        raise ValueError("No supported usage-condition result matches " + details)
    selected = selected.sort_values(
        [sort_by, "welch_p", "abs_delta"],
        ascending=[True, True, False],
        na_position="last",
        kind="stable",
    )
    return selected.iloc[0]
