from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_workflow.liana_window import analyze_window_rankagg


def test_disabling_zero_fill_uses_observed_and_complete_case_edges(tmp_path: Path):
    run_dir = tmp_path / "run"
    cellchat_path = tmp_path / "cellchat.csv"
    pd.DataFrame(
        {
            "ligand": ["Lig1", "Lig2"],
            "receptor": ["Rec1", "Rec2"],
            "pathway_name": ["PATH_A", "PATH_A"],
            "annotation": ["Secreted Signaling", "Secreted Signaling"],
        }
    ).to_csv(cellchat_path, index=False)

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
        pd.DataFrame(rows).to_parquet(
            sample_dir / "merged_by_sample.parquet", index=False
        )

    tables = analyze_window_rankagg(
        run_dir,
        cellchat_path,
        conditions=["A", "B"],
        min_condition_samples=3,
        zero_fill_missing=False,
    )

    completed = tables["completed_edges"]
    assert completed["rank_missing"].sum() == 3
    assert completed["rank_imputed"].sum() == 0
    assert completed.loc[completed["rank_missing"], "activity"].isna().all()

    pairwise = tables["edge_pairwise"]
    shared = pairwise.loc[pairwise["ligand"].eq("Lig1")].iloc[0]
    disappearing = pairwise.loc[pairwise["ligand"].eq("Lig2")].iloc[0]
    assert bool(shared["tested"])
    assert shared["n_tested_a"] == 3
    assert shared["n_tested_b"] == 3
    assert disappearing["presence_class"] == "disappears_in_b"
    assert not bool(disappearing["tested"])
    assert np.isnan(disappearing["exact_permutation_p"])

    pathway = tables["pathway_pairwise"]
    assert pathway.iloc[0]["n_edges"] == 1
    assert pathway.iloc[0]["edge_missing_policy"] == "complete_case_edges"
