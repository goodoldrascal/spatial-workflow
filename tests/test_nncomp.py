import numpy as np
import pandas as pd
import pytest

from spatial_workflow.nncomp import (
    _fresh_output_paths,
    compartment_abundance,
    long_neighbor_composition,
    summarize_neighbor_composition,
    validate_nncomp_obs,
)


def test_compartment_abundance_is_tidy_and_includes_whole_sample():
    obs = pd.DataFrame(
        {
            "condition-name": ["control", "control", "control", "treated"],
            "sample-name": ["s1", "s1", "s1", "s2"],
            "spatial-domain": ["0", "0", "1", "0"],
            "cell-type": ["A", "B", "A", "A"],
        }
    )

    abundance = compartment_abundance(
        obs,
        condition_col="condition-name",
        sample_col="sample-name",
        compartment_col="spatial-domain",
        cell_type_col="cell-type",
    )

    assert set(abundance["compartment"]) == {"0", "1", "all"}
    s1_domain0 = abundance.loc[
        abundance["sample"].eq("s1") & abundance["compartment"].eq("0")
    ]
    assert s1_domain0["compartment_total_cells"].unique().tolist() == [2]
    assert np.isclose(s1_domain0["cell_type_fraction"].sum(), 1)
    assert s1_domain0["compartment_fraction"].unique().tolist() == [2 / 3]

    s1_all = abundance.loc[
        abundance["sample"].eq("s1") & abundance["compartment"].eq("all")
    ]
    assert s1_all["n_cells"].sum() == 3
    assert s1_all["compartment_fraction"].unique().tolist() == [1]


def test_neighbor_matrices_expand_to_clean_long_tables_and_summary():
    counts_1 = pd.DataFrame(
        [[0, 3], [1, 0]], index=["A", "B"], columns=["A", "B"]
    )
    counts_2 = pd.DataFrame(
        [[0, 0], [3, 0]], index=["A", "B"], columns=["A", "B"]
    )
    frac_1 = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"]
    )
    frac_2 = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"]
    )
    presence_1 = pd.DataFrame(
        [[0.0, 0.75], [0.5, 0.0]], index=["A", "B"], columns=["A", "B"]
    )
    presence_2 = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"]
    )
    nn_df = pd.DataFrame(
        {
            "condition-name": ["control", "control"],
            "sample-name": ["s1", "s2"],
            "spatial-domain": ["0", "0"],
            "nn_counts": [counts_1, counts_2],
            "nn_frac": [frac_1, frac_2],
            "nn_presence": [presence_1, presence_2],
        }
    )

    abundance = pd.DataFrame(
        {
            "condition": ["control"] * 4,
            "sample": ["s1", "s1", "s2", "s2"],
            "compartment": ["0"] * 4,
            "cell_type": ["A", "B", "A", "B"],
            "n_cells": [4, 2, 0, 3],
        }
    )

    long = long_neighbor_composition(
        nn_df,
        abundance,
        condition_col="condition-name",
        sample_col="sample-name",
        compartment_col="spatial-domain",
    )

    assert len(long) == 8
    assert list(long.columns) == [
        "condition",
        "sample",
        "compartment",
        "source_cell_type",
        "source_n_cells",
        "neighbor_cell_type",
        "neighbor_count",
        "neighbor_fraction",
        "neighbor_presence",
    ]
    a_to_b = long.loc[
        long["source_cell_type"].eq("A")
        & long["neighbor_cell_type"].eq("B")
    ]
    assert a_to_b["neighbor_count"].tolist() == [3.0, 0.0]
    assert a_to_b["neighbor_presence"].tolist() == [0.75, 0.0]
    assert a_to_b["source_n_cells"].tolist() == [4, 0]

    summary = summarize_neighbor_composition(long)
    a_to_b_summary = summary.loc[
        summary["source_cell_type"].eq("A")
        & summary["neighbor_cell_type"].eq("B")
    ].iloc[0]
    assert a_to_b_summary["n_samples"] == 2
    assert a_to_b_summary["n_supporting_samples"] == 1
    assert a_to_b_summary["total_source_cells"] == 4
    assert a_to_b_summary["mean_neighbor_count"] == 1.5
    assert a_to_b_summary["mean_neighbor_fraction"] == 0.5
    assert a_to_b_summary["mean_neighbor_presence"] == 0.375


def test_nncomp_obs_preflight_checks_labels_and_sample_condition_mapping():
    obs = pd.DataFrame(
        {
            "sample-name": ["s1", "s1"],
            "condition-name": ["control", "control"],
            "cell-type": ["A", "B"],
            "spatial-domain": ["0", "0"],
        }
    )
    kwargs = {
        "sample_col": "sample-name",
        "condition_col": "condition-name",
        "cell_type_col": "cell-type",
        "compartment_col": "spatial-domain",
    }

    validate_nncomp_obs(obs, **kwargs)
    with pytest.raises(KeyError, match="cell-type"):
        validate_nncomp_obs(obs.drop(columns="cell-type"), **kwargs)
    with pytest.raises(ValueError, match="Null labels"):
        validate_nncomp_obs(obs.assign(**{"cell-type": ["A", None]}), **kwargs)
    with pytest.raises(ValueError, match="one condition"):
        validate_nncomp_obs(
            obs.assign(**{"condition-name": ["control", "treated"]}), **kwargs
        )


def test_nncomp_output_paths_refuse_existing_files(tmp_path):
    outputs = _fresh_output_paths(tmp_path)
    assert len(outputs) == 4
    outputs["manifest"].write_text("{}")

    with pytest.raises(FileExistsError, match="manifest.json"):
        _fresh_output_paths(tmp_path)
