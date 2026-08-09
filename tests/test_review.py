import numpy as np
import pandas as pd

from spatial_workflow.review import (
    filter_condition_sample_support,
    filter_overlap_support,
    plot_reciprocal_celltype_overlap,
    prepare_reciprocal_celltype_plot,
    reciprocal_celltype_zscores,
)


def _row(
    source_type,
    neighbor_type,
    z,
    *,
    sample="s1",
    condition="control",
    compartment="3",
    k=2,
    n_source_cells=10,
    n_neighbor_cells=20,
):
    return {
        "condition": condition,
        "sample": sample,
        "analysis": "within_compartment",
        "k": k,
        "source_compartment": compartment,
        "neighbor_compartment": compartment,
        "source_cell_type": source_type,
        "neighbor_cell_type": neighbor_type,
        "z_score": z,
        "n_source_cells": n_source_cells,
        "n_neighbor_cells": n_neighbor_cells,
    }


def test_reciprocal_celltype_scores_pair_source_and_neighbor_directions():
    selected_overlap = pd.DataFrame(
        [
            _row("B", "A", 2.0, n_source_cells=20, n_neighbor_cells=10),
            _row("A", "B", -1.0),
            _row("A", "C", np.nan),
            _row("C", "A", 1.0),
            _row("A", "D", 7.0, compartment="4"),
            _row("D", "A", 8.0, compartment="4"),
        ]
    )

    all_pairs = reciprocal_celltype_zscores(selected_overlap, z_column="z_score")
    assert all_pairs[["compartment", "cell_type_a", "cell_type_b"]].values.tolist() == [
        ["3", "A", "B"],
        ["4", "A", "D"],
    ]
    assert all_pairs[["z_a_to_b", "z_b_to_a"]].iloc[0].tolist() == [
        -1.0,
        2.0,
    ]
    assert all_pairs[["n_cell_type_a", "n_cell_type_b"]].iloc[0].tolist() == [
        10,
        20,
    ]

    compartment_three = reciprocal_celltype_zscores(
        selected_overlap,
        z_column="z_score",
        selected_compartment="3",
    )
    assert compartment_three["cell_type_pair"].tolist() == ["A ↔ B"]


def test_reciprocal_celltype_scores_accept_condition_overlap_mean_z():
    condition_overlap = pd.DataFrame(
        [
            {
                "condition": "control",
                "analysis": "within_compartment",
                "k": 2,
                "source_compartment": "3",
                "neighbor_compartment": "3",
                "source_cell_type": "A",
                "neighbor_cell_type": "B",
                "n_samples": 4,
                "min_source_cells": 11,
                "min_neighbor_cells": 17,
                "mean_z": 3.0,
            },
            {
                "condition": "control",
                "analysis": "within_compartment",
                "k": 2,
                "source_compartment": "3",
                "neighbor_compartment": "3",
                "source_cell_type": "B",
                "neighbor_cell_type": "A",
                "n_samples": 3,
                "min_source_cells": 17,
                "min_neighbor_cells": 11,
                "mean_z": 5.0,
            },
        ]
    )

    paired = reciprocal_celltype_zscores(condition_overlap, z_column="mean_z")

    assert paired["n_samples"].tolist() == [3]
    assert paired["z_a_to_b"].tolist() == [3.0]
    assert paired["z_b_to_a"].tolist() == [5.0]
    assert paired["min_cell_type_a"].tolist() == [11]
    assert paired["min_cell_type_b"].tolist() == [17]


def test_filter_overlap_support_requires_both_cell_type_populations():
    overlap = pd.DataFrame(
        {
            "n_source_cells": [10, 9, 20],
            "n_neighbor_cells": [10, 20, 8],
            "pair": ["keep", "low source", "low neighbor"],
        }
    )

    selected = filter_overlap_support(overlap, min_cells_per_type=10)

    assert selected["pair"].tolist() == ["keep"]


def test_filter_condition_sample_support_is_inclusive_at_threshold():
    condition = pd.DataFrame(
        {"n_samples": [3, 4, 5], "pair": ["low", "keep4", "keep5"]}
    )

    selected = filter_condition_sample_support(condition, min_samples=4)

    assert selected["pair"].tolist() == ["keep4", "keep5"]


def test_prepare_reciprocal_plot_supports_celltype_sign_and_magnitude_filters():
    rows = []
    for sample, condition, ab, ba, ac, ca in [
        ("s1", "control", 2.0, 3.0, -4.0, -3.0),
        ("s2", "control", 4.0, 5.0, -2.0, -2.5),
        ("s3", "treated", 2.5, 2.0, -3.5, -4.0),
        ("s4", "treated", 3.5, 4.0, -2.5, -3.0),
    ]:
        rows.extend(
            [
                _row("A", "B", ab, sample=sample, condition=condition),
                _row(
                    "B",
                    "A",
                    ba,
                    sample=sample,
                    condition=condition,
                    n_source_cells=20,
                    n_neighbor_cells=10,
                ),
                _row("A", "C", ac, sample=sample, condition=condition),
                _row(
                    "C",
                    "A",
                    ca,
                    sample=sample,
                    condition=condition,
                    n_source_cells=20,
                    n_neighbor_cells=10,
                ),
            ]
        )
    overlap = pd.DataFrame(rows)

    positive_sample, positive_condition = prepare_reciprocal_celltype_plot(
        overlap,
        compartment="3",
        cell_types_a="A",
        cell_types_b=["B", "C"],
        k_value=2,
        min_cells_a=10,
        min_cells_b=20,
        min_samples=2,
        min_abs_z_score=2,
        z_direction="positive",
    )
    negative_sample, negative_condition = prepare_reciprocal_celltype_plot(
        overlap,
        compartment="3",
        cell_types_a=["A"],
        cell_types_b=["B", "C"],
        min_cells_a=10,
        min_cells_b=20,
        min_samples=2,
        min_abs_z_score=2,
        z_direction="negative",
    )

    assert positive_sample["cell_type_b"].unique().tolist() == ["B"]
    assert positive_condition["cell_type_b"].unique().tolist() == ["B"]
    assert negative_sample["cell_type_b"].unique().tolist() == ["C"]
    assert negative_condition["cell_type_b"].unique().tolist() == ["C"]
    assert positive_condition["n_samples"].tolist() == [2, 2]

    figure = plot_reciprocal_celltype_overlap(
        overlap,
        compartment="3",
        cell_types_a="A",
        cell_types_b="B",
        min_cells_a=10,
        min_cells_b=20,
        min_samples=2,
        min_abs_z_score=2,
        z_direction="positive",
    )
    assert figure.layout.meta["k"] == 2
    assert figure.layout.meta["cell_types_a"] == ["A"]
    assert figure.layout.meta["condition_points"] == 2
