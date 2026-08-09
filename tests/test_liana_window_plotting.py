from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatial_workflow.liana_window_plotting import (
    plot_pathway_driver_summary,
    plot_pathway_ranking,
    prepare_pathway_driver_plot_data,
    prepare_pathway_ranking_plot_data,
)


def _pathway_pairwise() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contrast": ["B_vs_A"] * 3,
            "condition_a": ["A"] * 3,
            "condition_b": ["B"] * 3,
            "pathway_name": ["PATH_A", "PATH_B", "PATH_C"],
            "mean_activity_difference_b_minus_a": [0.10, -0.35, 0.20],
            "mean_activity_exact_p": [0.20, 0.05, 0.01],
            "rms_centroid_distance": [0.4, 0.8, 0.6],
            "rms_exact_p": [0.10, 0.01, 0.01],
        }
    )


def _driver_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lr = pd.DataFrame(
        {
            "contrast": ["B_vs_A", "B_vs_A"],
            "condition_a": ["A", "A"],
            "condition_b": ["B", "B"],
            "pathway_name": ["PATH_A", "PATH_A"],
            "ligand": ["Lig1", "Lig2"],
            "receptor": ["Rec1", "Rec2"],
            "signed_effect_sum": [0.7, -0.3],
            "absolute_effect_sum": [1.0, 0.5],
            "absolute_effect_fraction": [2 / 3, 1 / 3],
            "driver_rank": [1, 2],
        }
    )
    edges = pd.DataFrame(
        {
            "edge_id": ["E1", "E2", "E3"],
            "contrast": ["B_vs_A"] * 3,
            "condition_a": ["A"] * 3,
            "condition_b": ["B"] * 3,
            "pathway_name": ["PATH_A"] * 3,
            "source": ["S1", "S2", "S1"],
            "target": ["T1", "T1", "T2"],
            "ligand": ["Lig1", "Lig1", "Lig2"],
            "receptor": ["Rec1", "Rec1", "Rec2"],
            "difference_b_minus_a": [0.5, -0.3, 0.2],
            "abs_difference": [0.5, 0.3, 0.2],
            "driver_rank": [1, 2, 3],
        }
    )
    pairwise = pd.DataFrame(
        {
            "edge_id": ["E1", "E2", "E3"],
            "contrast": ["B_vs_A"] * 3,
            "exact_permutation_p": [0.01, 0.04, 0.20],
            "exact_permutation_q": [0.03, 0.06, 0.25],
            "tested": [True, True, True],
            "presence_class": ["supported_both"] * 3,
            "minimum_attainable_p": [0.01] * 3,
        }
    )
    return lr, edges, pairwise


def test_pathway_ranking_respects_metric_semantics():
    frame = _pathway_pairwise()
    effect = prepare_pathway_ranking_plot_data(
        frame,
        contrast="B_vs_A",
        rank_by="mean_activity_difference_b_minus_a",
        top_n=2,
    )
    assert effect["pathway_name"].tolist() == ["PATH_B", "PATH_C"]
    assert effect["plot_value"].tolist() == [-0.35, 0.20]

    rms_distance = prepare_pathway_ranking_plot_data(
        frame,
        contrast="B_vs_A",
        rank_by="rms_centroid_distance",
        top_n=2,
    )
    assert rms_distance["pathway_name"].tolist() == ["PATH_B", "PATH_C"]
    assert rms_distance["plot_value"].tolist() == [0.8, 0.6]

    filtered = prepare_pathway_ranking_plot_data(
        frame,
        contrast="B_vs_A",
        rank_by="rms_centroid_distance",
        top_n=3,
        exact_p_max=0.05,
    )
    assert filtered["pathway_name"].tolist() == ["PATH_B", "PATH_C"]

    mean_p = prepare_pathway_ranking_plot_data(
        frame,
        contrast="B_vs_A",
        rank_by="mean_activity_exact_p",
        top_n=2,
    )
    assert mean_p["pathway_name"].tolist() == ["PATH_C", "PATH_B"]
    assert np.allclose(mean_p["plot_value"], -np.log10([0.01, 0.05]))

    rms_p = prepare_pathway_ranking_plot_data(
        frame,
        contrast="B_vs_A",
        rank_by="rms_exact_p",
        top_n=2,
    )
    assert rms_p["pathway_name"].tolist() == ["PATH_B", "PATH_C"]


def test_driver_data_joins_edge_p_values_and_applies_threshold():
    lr, edges, pairwise = _driver_tables()
    lr_plot, edge_plot = prepare_pathway_driver_plot_data(
        lr,
        edges,
        pairwise,
        contrast="B_vs_A",
        pathway_name="PATH_A",
        top_lr=2,
        top_edges=10,
        edge_p_max=0.05,
    )
    assert lr_plot["lr_label"].tolist() == ["Lig1 → Rec1", "Lig2 → Rec2"]
    assert np.allclose(lr_plot["absolute_effect_percent"], [200 / 3, 100 / 3])
    assert edge_plot["edge_id"].tolist() == ["E1", "E2"]
    assert set(edge_plot["cell_pair_label"]) == {"S1 → T1", "S2 → T1"}
    assert np.allclose(edge_plot["minus_log10_exact_p"], [2.0, -np.log10(0.04)])


def test_plot_titles_and_empty_threshold_panel_are_pathway_specific():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank_figure, rank_axis, _ = plot_pathway_ranking(
        _pathway_pairwise(),
        contrast="B_vs_A",
        rank_by="rms_exact_p",
        top_n=3,
    )
    assert "B vs A" in rank_axis.get_title(loc="left")

    lr, edges, pairwise = _driver_tables()
    figure, axes, _, edge_plot = plot_pathway_driver_summary(
        lr,
        edges,
        pairwise,
        contrast="B_vs_A",
        pathway_name="PATH_A",
        edge_p_max=0.001,
    )
    assert edge_plot.empty
    assert "PATH_A pathway drivers" in figure._suptitle.get_text()
    assert "No PATH_A edges" in axes[1].texts[0].get_text()
    plt.close(rank_figure)
    plt.close(figure)
