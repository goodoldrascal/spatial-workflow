import numpy as np
import pandas as pd

from spatial_workflow import colocalization
from spatial_workflow.colocalization import (
    build_reciprocal_within_condition_colocalization,
    filter_contrast_table,
    plot_directional_contrast_overview,
    plot_reciprocal_contrast_overview,
    prepare_directional_contrast_plot,
    prepare_reciprocal_contrast_plot,
    prepare_reciprocal_effect_samples,
    run_condition_contrast,
    select_colocalization_scope,
    summarize_within_condition_colocalization,
)


def _overlap_row(
    sample,
    source,
    neighbor,
    coefficient,
    null_mean,
    *,
    condition="control",
    compartment="3",
    analysis="within_compartment",
    k=2,
    n_source_cells=20,
    n_neighbor_cells=20,
):
    delta = coefficient - null_mean
    return {
        "condition": condition,
        "sample": sample,
        "analysis": analysis,
        "k": k,
        "source_compartment": compartment,
        "neighbor_compartment": compartment,
        "source_cell_type": source,
        "neighbor_cell_type": neighbor,
        "n_source_cells": n_source_cells,
        "n_neighbor_cells": n_neighbor_cells,
        "n_permutations": 999,
        "coefficient": coefficient,
        "null_mean": null_mean,
        "delta": delta,
        "z_score": delta / 0.01,
        "p_enriched": 0.01 if delta > 0 else 0.9,
    }


def test_within_condition_summary_retains_observed_coefficients_for_reciprocal_pairs():
    rows = []
    for sample in ["s1", "s2", "s3", "s4"]:
        rows.extend(
            [
                _overlap_row(sample, "A", "B", 0.20, 0.10),
                _overlap_row(sample, "B", "A", 0.30, 0.10),
            ]
        )
    directional = summarize_within_condition_colocalization(
        pd.DataFrame(rows),
        min_source_cells=10,
        min_neighbor_cells=10,
    )
    reciprocal = build_reciprocal_within_condition_colocalization(directional)

    assert directional["mean_coefficient"].sort_values().tolist() == [0.2, 0.3]
    assert reciprocal["cell_type_pair"].tolist() == ["A ↔ B"]
    assert reciprocal["mean_coefficient_a_to_b"].tolist() == [0.2]
    assert reciprocal["mean_coefficient_b_to_a"].tolist() == [0.3]


def test_prepare_reciprocal_effect_samples_retains_directional_coefficients():
    overlap = pd.DataFrame(
        [
            _overlap_row("s1", "A", "B", 0.08, 0.03),
            _overlap_row("s1", "B", "A", 0.06, 0.02),
            _overlap_row("s1", "A", "A", 0.50, 0.40),
        ]
    )

    paired = prepare_reciprocal_effect_samples(
        overlap,
        k_value=2,
        min_cells_a=10,
        min_cells_b=10,
    )

    assert paired["cell_type_pair"].tolist() == ["A ↔ B"]
    assert paired["coefficient_a_to_b"].tolist() == [0.08]
    assert paired["coefficient_b_to_a"].tolist() == [0.06]


def test_contrast_display_columns_lead_with_condition_coefficients():
    directional_columns = colocalization.contrast_display_columns(
        "directional",
        [
            "min_n_cell_type_a__control",
            "n_cell_type_a__c1",
        ],
        include_sample_counts=False,
    )
    assert directional_columns[5:9] == [
        "direction",
        "denominator_mean_coefficient",
        "numerator_mean_coefficient",
        "contrast_delta",
    ]

    reciprocal_columns = colocalization.contrast_display_columns(
        "reciprocal",
        [
            "min_n_cell_type_a__control",
            "n_cell_type_a__c1",
        ],
        include_sample_counts=False,
    )
    assert reciprocal_columns[5:11] == [
        "denominator_mean_coefficient_a_to_b",
        "numerator_mean_coefficient_a_to_b",
        "denominator_mean_coefficient_b_to_a",
        "numerator_mean_coefficient_b_to_a",
        "contrast_delta_a_to_b",
        "contrast_delta_b_to_a",
    ]


def test_reciprocal_contrast_plot_pairs_condition_coefficients_with_effects():
    reciprocal = pd.DataFrame(
        {
            "contrast": ["treated_vs_control", "treated_vs_control"],
            "analysis": ["within_compartment", "within_compartment"],
            "k": [2, 2],
            "compartment": ["3", "4"],
            "cell_type_a": ["A", "A"],
            "cell_type_b": ["B", "C"],
            "cell_type_pair": ["A ↔ B", "A ↔ C"],
            "contrast_delta_a_to_b": [0.08, 0.01],
            "contrast_delta_b_to_a": [0.04, -0.02],
            "denominator_mean_coefficient_a_to_b": [0.10, 0.20],
            "numerator_mean_coefficient_a_to_b": [0.18, 0.21],
            "denominator_mean_coefficient_b_to_a": [0.30, 0.15],
            "numerator_mean_coefficient_b_to_a": [0.35, 0.14],
            "denominator_mean_delta_a_to_b": [0.05, 0.03],
            "numerator_mean_delta_a_to_b": [0.13, 0.04],
            "denominator_mean_delta_b_to_a": [0.10, 0.02],
            "numerator_mean_delta_b_to_a": [0.14, 0.00],
            "min_direction_max_condition_mean_coefficient": [0.18, 0.15],
            "limma_q_a_to_b": [0.02, 0.50],
            "limma_q_b_to_a": [0.03, 0.50],
            "both_direction_limma_q_max": [0.03, 0.50],
            "exact_permutation_p_a_to_b": [0.03, 0.40],
            "exact_permutation_p_b_to_a": [0.06, 0.50],
            "both_direction_exact_p_max": [0.06, 0.50],
            "denominator_positive_samples_a_to_b": [4, 4],
            "numerator_positive_samples_a_to_b": [4, 4],
            "denominator_positive_samples_b_to_a": [4, 4],
            "numerator_positive_samples_b_to_a": [4, 4],
            "reciprocal_same_direction": [True, False],
            "loo_sign_fraction_a_to_b": [1.0, 0.5],
            "loo_sign_fraction_b_to_a": [1.0, 0.5],
            "both_direction_loo_sign_fraction_min": [1.0, 0.5],
        }
    )
    outputs = {
        "manifest": {
            "configuration": {
                "contrasts": [{"numerator": "treated", "denominator": "control"}]
            }
        },
        "contrast_results": {"treated_vs_control": {"reciprocal": reciprocal}},
    }

    plot_data = prepare_reciprocal_contrast_plot(
        outputs,
        "treated_vs_control",
        top_n=1,
    )

    assert plot_data["display_label"].tolist() == [
        "Domain 3 · A → B",
        "Domain 3 · B → A",
    ]
    assert plot_data["denominator_condition"].unique().tolist() == ["control"]
    assert plot_data["numerator_condition"].unique().tolist() == ["treated"]
    assert plot_data["denominator_mean_coefficient"].tolist() == [0.10, 0.30]
    assert plot_data["numerator_mean_coefficient"].tolist() == [0.18, 0.35]
    assert np.allclose(plot_data["contrast_delta"], [0.08, 0.04])
    assert np.allclose(plot_data["denominator_mean_delta"], [0.05, 0.10])
    assert np.allclose(plot_data["numerator_mean_delta"], [0.13, 0.14])

    figure = plot_reciprocal_contrast_overview(
        outputs,
        "treated_vs_control",
        top_n=1,
        require_within_condition_colocalization=False,
    )
    assert figure.layout.meta["shown_relationships"] == 1
    assert figure.layout.meta["shown_directed_rows"] == 2
    assert figure.layout.meta["denominator"] == "control"
    assert figure.layout.meta["numerator"] == "treated"


def test_directional_within_colocalization_gate_keeps_one_way_relationships():
    directional = pd.DataFrame(
        {
            "contrast": ["treated_vs_control", "treated_vs_control"],
            "analysis": ["within_compartment", "within_compartment"],
            "k": [2, 2],
            "compartment": ["3", "3"],
            "cell_type_a": ["A", "A"],
            "cell_type_b": ["B", "B"],
            "cell_type_pair": ["A ↔ B", "A ↔ B"],
            "direction": ["a_to_b", "b_to_a"],
            "contrast_delta": [0.08, 0.04],
            "denominator_mean_delta": [0.05, -0.02],
            "numerator_mean_delta": [0.13, 0.02],
            "denominator_mean_coefficient": [0.10, 0.30],
            "numerator_mean_coefficient": [0.18, 0.35],
            "max_condition_mean_coefficient": [0.18, 0.35],
            "limma_q": [0.02, 0.03],
            "exact_permutation_p": [0.03, 0.03],
            "denominator_positive_samples": [4, 1],
            "numerator_positive_samples": [4, 2],
            "loo_sign_fraction": [1.0, 1.0],
        }
    )
    within_directional = pd.DataFrame(
        {
            "condition": ["control", "treated", "control", "treated"],
            "analysis": ["within_compartment"] * 4,
            "k": [2] * 4,
            "source_compartment": ["3"] * 4,
            "source_cell_type": ["A", "A", "B", "B"],
            "neighbor_cell_type": ["B", "B", "A", "A"],
            "median_delta": [0.05, 0.13, -0.02, 0.02],
            "n_positive_delta": [4, 4, 1, 2],
            "stouffer_q_enriched": [0.01, 0.01, 1.0, 0.50],
            "partial_q_enriched_3_of_4": [0.02, 0.02, 1.0, 0.50],
        }
    )
    outputs = {
        "manifest": {
            "configuration": {
                "contrasts": [{"numerator": "treated", "denominator": "control"}]
            }
        },
        "within_directional": within_directional,
        "contrast_results": {"treated_vs_control": {"directional": directional}},
    }

    filtered = filter_contrast_table(
        outputs,
        "treated_vs_control",
        table="directional",
        require_within_condition_colocalization=True,
    )
    assert filtered["direction"].tolist() == ["a_to_b"]
    assert filtered["colocalized_in_denominator"].tolist() == [True]
    assert filtered["colocalized_in_numerator"].tolist() == [True]

    plot_data = prepare_directional_contrast_plot(
        outputs,
        "treated_vs_control",
        require_within_condition_colocalization=True,
    )
    assert plot_data["display_label"].tolist() == ["Domain 3 · A → B"]
    figure = plot_directional_contrast_overview(
        outputs,
        "treated_vs_control",
        require_within_condition_colocalization=True,
    )
    assert figure.layout.meta["result_kind"] == "directional"
    assert figure.layout.meta["shown_relationships"] == 1


def test_condition_contrast_exposes_adjustable_observed_colocalization_floor(
    monkeypatch,
):
    rows = []
    samples = {
        "control": ["c1", "c2", "c3", "c4"],
        "treated": ["t1", "t2", "t3", "t4"],
    }
    for condition, sample_names in samples.items():
        for index, sample in enumerate(sample_names):
            treated = condition == "treated"
            rows.append(
                {
                    "condition": condition,
                    "sample": sample,
                    "analysis": "within_compartment",
                    "k": 2,
                    "compartment": "3",
                    "cell_type_a": "A",
                    "cell_type_b": "B",
                    "cell_type_pair": "A ↔ B",
                    "coefficient_a_to_b": (0.08 if treated else 0.01),
                    "coefficient_b_to_a": (0.06 if treated else 0.02),
                    "delta_a_to_b": (0.04 if treated else 0.005) + index * 0.001,
                    "delta_b_to_a": (0.03 if treated else 0.004) + index * 0.001,
                    "n_cell_type_a": 20,
                    "n_cell_type_b": 25,
                }
            )
    sample_effects = pd.DataFrame(rows)

    def fake_limma(effect_matrix, conditions, numerator, denominator):
        contrast = effect_matrix.iloc[:, 4:].mean(axis=1) - effect_matrix.iloc[
            :, :4
        ].mean(axis=1)
        return pd.DataFrame(
            {
                "contrast_delta": contrast.to_numpy(),
                "average_delta": effect_matrix.mean(axis=1).to_numpy(),
                "limma_p": np.repeat(0.01, len(effect_matrix)),
                "limma_q": np.repeat(0.02, len(effect_matrix)),
                "log_odds_differential": np.repeat(1.0, len(effect_matrix)),
                "df_total": np.repeat(6.0, len(effect_matrix)),
            }
        )

    monkeypatch.setattr(colocalization, "_run_limma", fake_limma)
    directional, reciprocal, sample_counts = run_condition_contrast(
        sample_effects,
        numerator="treated",
        denominator="control",
        sample_dict=samples,
    )
    outputs = {
        "contrast_results": {
            "treated_vs_control": {
                "directional": directional,
                "reciprocal": reciprocal,
                "sample_counts": sample_counts,
            }
        }
    }

    assert sorted(directional["max_condition_mean_coefficient"]) == [0.06, 0.08]
    assert reciprocal["min_direction_max_condition_mean_coefficient"].tolist() == [0.06]
    assert (
        len(
            filter_contrast_table(
                outputs,
                "treated_vs_control",
                table="directional",
                min_observed_colocalization=0.07,
            )
        )
        == 1
    )
    assert (
        len(
            filter_contrast_table(
                outputs,
                "treated_vs_control",
                table="reciprocal",
                min_observed_colocalization=0.05,
            )
        )
        == 1
    )
    assert filter_contrast_table(
        outputs,
        "treated_vs_control",
        table="reciprocal",
        min_observed_colocalization=0.07,
    ).empty


def test_whole_sample_scope_is_not_a_mix_of_compartment_rows():
    rows = [
        _overlap_row("s1", "A", "B", 0.2, 0.1),
        _overlap_row(
            "s1",
            "A",
            "B",
            0.3,
            0.1,
            analysis="whole_sample",
            compartment="all",
        ),
    ]
    selected = select_colocalization_scope(pd.DataFrame(rows), "whole_sample")

    assert selected["analysis"].unique().tolist() == ["whole_sample"]
    assert selected["source_compartment"].unique().tolist() == ["all"]


def test_prepare_reciprocal_effect_samples_supports_whole_sample():
    overlap = pd.DataFrame(
        [
            _overlap_row(
                "s1",
                "A",
                "B",
                0.20,
                0.10,
                analysis="whole_sample",
                compartment="all",
            ),
            _overlap_row(
                "s1",
                "B",
                "A",
                0.30,
                0.10,
                analysis="whole_sample",
                compartment="all",
            ),
        ]
    )

    paired = prepare_reciprocal_effect_samples(
        overlap,
        k_value=2,
        min_cells_a=10,
        min_cells_b=10,
        analysis_scope="whole_sample",
    )

    assert paired["analysis"].unique().tolist() == ["whole_sample"]
    assert paired["compartment"].unique().tolist() == ["all"]
