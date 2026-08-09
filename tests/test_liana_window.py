from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatial_workflow.liana_window import (
    WindowParameters,
    audit_liana_input,
    load_cellchat_resource,
    load_custom_lr_resource,
    load_rankagg_resource,
    analyze_window_rankagg,
    build_adaptive_windows,
    exact_mean_difference,
    minimum_exact_p,
    prepare_chord_input,
)


def test_adaptive_windows_are_spatial_grid_not_graph_windows():
    adata = ad.AnnData(np.ones((8, 2)))
    adata.obs_names = [f"cell_{index}" for index in range(8)]
    adata.obs["cell_type"] = ["A"] * 4 + ["B"] * 4
    adata.obsm["spatial"] = np.column_stack([np.arange(8), np.zeros(8)])
    parameters = WindowParameters(
        adaptive_k=8,
        grid_stride=20,
        min_cells=2,
        min_required_groups=2,
        n_perms=2,
        n_jobs=1,
    )

    windows, memberships = build_adaptive_windows(
        adata,
        sample="sample_1",
        group_key="cell_type",
        spatial_key="spatial",
        parameters=parameters,
    )

    assert len(windows) == 2
    assert windows.iloc[0]["window_type"] == "grid_adaptive"
    assert windows.iloc[0]["n_cells_total"] == 8
    assert windows.iloc[0]["n_groups_ge_min"] == 2
    assert memberships["obs_name"].nunique() == 8


def test_exact_four_vs_four_resolution():
    a = np.array([[0.0, 0.0, 0.0, 0.0]])
    b = np.array([[1.0, 1.0, 1.0, 1.0]])
    difference, p_value = exact_mean_difference(a, b)
    assert difference[0] == 1.0
    assert p_value[0] == minimum_exact_p(4, 4)
    assert p_value[0] == 2 / 70


def _write_cellchat(path: Path) -> None:
    pd.DataFrame(
        {
            "ligand": ["Lig1", "Lig2"],
            "receptor": ["Rec1", "Rec2"],
            "pathway_name": ["PATH_A", "PATH_A"],
            "interaction_name": ["Lig1_Rec1", "Lig2_Rec2"],
            "interaction_name_2": ["Lig1 - Rec1", "Lig2 - Rec2"],
            "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
        }
    ).to_csv(path, index=False)


def test_two_condition_support_zero_completion_and_pathway_drivers(tmp_path: Path):
    run_dir = tmp_path / "run"
    cellchat_path = tmp_path / "cellchat.csv"
    _write_cellchat(cellchat_path)
    samples = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "b1": "B",
        "b2": "B",
        "b3": "B",
    }
    for sample, condition in samples.items():
        rows = [
            {
                "sample": sample,
                "condition": condition,
                "source": "Sender",
                "target": "Receiver",
                "ligand_complex": "Lig1",
                "receptor_complex": "Rec1",
                "magnitude_rank": 0.8 if condition == "A" else 0.2,
            }
        ]
        if condition == "A":
            rows.append(
                {
                    "sample": sample,
                    "condition": condition,
                    "source": "Sender",
                    "target": "Receiver2",
                    "ligand_complex": "Lig2",
                    "receptor_complex": "Rec2",
                    "magnitude_rank": 0.4,
                }
            )
        sample_dir = run_dir / "samples" / sample
        sample_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(sample_dir / "merged_by_sample.parquet", index=False)

    tables = analyze_window_rankagg(
        run_dir,
        cellchat_path,
        conditions=["A", "B"],
        min_condition_samples=3,
    )

    completed = tables["completed_edges"]
    assert completed["edge_id"].nunique() == 2
    assert len(completed) == 12
    assert completed["rank_imputed"].sum() == 3
    assert completed.loc[completed["rank_imputed"], "activity"].eq(0).all()

    pairwise = tables["edge_pairwise"]
    assert pairwise["contrast"].unique().tolist() == ["B_vs_A"]
    disappearing = pairwise.loc[pairwise["ligand"].eq("Lig2")].iloc[0]
    assert disappearing["presence_class"] == "disappears_in_b"
    assert bool(disappearing["appearance_or_disappearance"])
    assert bool(disappearing["tested"])

    pathway = tables["pathway_pairwise"]
    assert pathway.iloc[0]["n_edges"] == 2
    assert np.isfinite(pathway.iloc[0]["rms_centroid_distance"])
    assert not tables["pathway_edge_drivers"].empty
    assert not tables["pathway_lr_drivers"].empty
    assert not tables["pathway_cellpair_drivers"].empty

    chord = prepare_chord_input(
        tables["pathway_edge_drivers"],
        contrast="B_vs_A",
        pathway_name="PATH_A",
        top_n=2,
    )
    assert set(chord.columns) == {
        "source",
        "target",
        "ligand",
        "receptor",
        "weight_abs",
        "higher_condition",
    }
    assert len(chord) == 2
