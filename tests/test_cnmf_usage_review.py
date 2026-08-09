from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatial_workflow.cnmf import (
    sample_usage_summary,
    subset_usage_by_cell_type,
    summarize_usage,
)
from spatial_workflow.cnmf_usage_analysis import (
    sample_celltype_usage_table,
    select_usage_condition_result,
    test_usage_condition_changes as run_usage_condition_tests,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "condition": ["A", "A", "A", "B", "B", "B"],
            "spatial_domain": ["0", "0", "1", "0", "1", "1"],
            "cluster_sub": ["Ast1", "Ast2", None, "Ast1", "", "Ast2"],
            "Usage_1": [0.1, 0.3, 0.9, 0.5, 0.8, 0.7],
        },
        index=[f"cell_{index}" for index in range(6)],
    )


def test_usage_review_defaults_to_all_nonmissing_cell_types():
    selected = subset_usage_by_cell_type(
        _obs(), cell_type_key="cluster_sub", cell_types=None
    )
    assert selected.index.tolist() == ["cell_0", "cell_1", "cell_3", "cell_5"]
    assert selected.attrs["cell_types"] == ["Ast1", "Ast2"]

    sample = sample_usage_summary(
        selected,
        sample_key="sample_id",
        condition_key="condition",
        usage_cols=["Usage_1"],
    ).set_index("sample_id")
    assert np.isclose(sample.loc["s1", "mean_usage"], 0.2)
    assert np.isclose(sample.loc["s2", "mean_usage"], 0.6)

    domains = summarize_usage(
        selected,
        group_columns=["spatial_domain"],
        usage_cols=["Usage_1"],
    ).set_index("spatial_domain")
    assert domains.loc["0", "n_cells"] == 3
    assert domains.loc["1", "n_cells"] == 1


def test_usage_review_exact_subset_and_validation():
    selected = subset_usage_by_cell_type(
        _obs(), cell_type_key="cluster_sub", cell_types=["Ast2"]
    )
    assert selected["cluster_sub"].unique().tolist() == ["Ast2"]
    assert selected.index.tolist() == ["cell_1", "cell_5"]

    with pytest.raises(KeyError, match="Unknown review cell types"):
        subset_usage_by_cell_type(
            _obs(), cell_type_key="cluster_sub", cell_types=["Microglia"]
        )
    with pytest.raises(KeyError, match="absent from obs"):
        subset_usage_by_cell_type(_obs(), cell_type_key="missing")


def _condition_obs() -> pd.DataFrame:
    records = []
    sample_values = {
        "a1": ("A", 0.00),
        "a2": ("A", 0.01),
        "a3": ("A", 0.02),
        "a4": ("A", 0.03),
        "b1": ("B", 1.00),
        "b2": ("B", 1.01),
        "b3": ("B", 1.02),
        "b4": ("B", 1.03),
    }
    for sample, (condition, value) in sample_values.items():
        for cell in range(2):
            records.append(
                {
                    "sample_id": sample,
                    "condition": condition,
                    "cluster_sub": "MG",
                    "Usage_1": value + cell * 0.001,
                    "Usage_2": 0.2 + int(sample[-1]) * 0.01 + cell * 0.001,
                }
            )
    return pd.DataFrame(records)


def test_usage_condition_table_uses_samples_and_adjusts_all_programs():
    obs = _condition_obs()
    sample_table = sample_celltype_usage_table(
        obs,
        cell_type_key="cluster_sub",
        sample_key="sample_id",
        condition_key="condition",
    )
    assert len(sample_table) == 16
    assert set(sample_table["n_cells"]) == {2}

    results = run_usage_condition_tests(
        obs,
        cell_type_key="cluster_sub",
        sample_key="sample_id",
        condition_key="condition",
        contrasts=("B", "A"),
        min_cells_per_sample=2,
        min_samples_per_condition=4,
    )
    assert len(results) == 2
    focus = results.set_index("program").loc["Usage_1"]
    assert focus["n_samples_a"] == focus["n_samples_b"] == 4
    assert np.isclose(focus["delta_a_minus_b"], 1.0)
    assert focus["permutation_method"] == "exact"
    assert focus["permutation_draws"] == 70
    assert np.isclose(focus["permutation_p"], 2 / 70)
    assert focus["permutation_q_within_contrast"] >= focus["permutation_p"]
    assert focus["welch_q_global"] >= focus["welch_p"]

    selected = select_usage_condition_result(
        results,
        contrast=("B", "A"),
        cell_type_key="cluster_sub",
        cell_type="MG",
        program="Usage_1",
    )
    assert selected["program"] == "Usage_1"


def test_usage_condition_table_reports_insufficient_sample_support():
    results = run_usage_condition_tests(
        _condition_obs(),
        cell_type_key="cluster_sub",
        sample_key="sample_id",
        condition_key="condition",
        contrasts=("B", "A"),
        min_cells_per_sample=3,
        min_samples_per_condition=4,
    )
    assert not results["tested"].any()
    assert results["welch_p"].isna().all()
    assert results["filter_reason"].str.contains("insufficient samples").all()


def test_usage_condition_violin_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")
    from spatial_workflow.cnmf_usage_plotting import plot_usage_condition_violin

    obs = _condition_obs()
    results = run_usage_condition_tests(
        obs,
        cell_type_key="cluster_sub",
        sample_key="sample_id",
        condition_key="condition",
        contrasts=("B", "A"),
        min_cells_per_sample=2,
        min_samples_per_condition=4,
    )
    focus = select_usage_condition_result(
        results,
        contrast=("B", "A"),
        cell_type_key="cluster_sub",
        cell_type="MG",
        program="Usage_1",
    )
    figure, sample_summary = plot_usage_condition_violin(
        obs,
        result=focus,
        cell_type_key="cluster_sub",
        sample_key="sample_id",
        condition_key="condition",
        condition_order=["A", "B"],
        condition_labels={"A": "Control", "B": "Treated"},
        palette={"A": "#4C1D57", "B": "#F3B37A"},
    )
    assert len(figure.axes) == 2
    assert len(sample_summary) == 8
    figure.clear()
