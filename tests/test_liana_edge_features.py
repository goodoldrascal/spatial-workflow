from __future__ import annotations

import pandas as pd
import pytest

from spatial_workflow.liana_edge_features import top_edge_features


def _edge_pairwise_fixture() -> pd.DataFrame:
    rows = [
        {
            "edge_id": "edge_1",
            "contrast": "B_vs_A",
            "tested": True,
            "exact_permutation_p": 0.01,
            "ligand_complex": "Lig1",
            "receptor_complex": "Rec1",
            "pathway_name": "Path1",
            "source": "Sender1",
            "target": "Receiver1",
        },
        {
            "edge_id": "edge_2",
            "contrast": "B_vs_A",
            "tested": True,
            "exact_permutation_p": 0.049,
            "ligand_complex": "Lig1",
            "receptor_complex": "Rec2",
            "pathway_name": "Path1",
            "source": "Sender1",
            "target": "Receiver2",
        },
        {
            "edge_id": "edge_3",
            "contrast": "B_vs_A",
            "tested": True,
            "exact_permutation_p": 0.05,
            "ligand_complex": "Lig2",
            "receptor_complex": "Rec1",
            "pathway_name": "Path2",
            "source": "Sender2",
            "target": "Receiver1",
        },
        {
            "edge_id": "edge_4",
            "contrast": "B_vs_A",
            "tested": False,
            "exact_permutation_p": 0.001,
            "ligand_complex": "Lig3",
            "receptor_complex": "Rec3",
            "pathway_name": "Path3",
            "source": "Sender3",
            "target": "Receiver3",
        },
        {
            "edge_id": "edge_5",
            "contrast": "C_vs_B",
            "tested": True,
            "exact_permutation_p": 0.01,
            "ligand_complex": "Lig4",
            "receptor_complex": "Rec4",
            "pathway_name": "Path4",
            "source": "Sender4",
            "target": "Receiver4",
        },
    ]
    # Duplicate a significant row to verify that edge IDs, rather than rows, are counted.
    return pd.DataFrame([*rows, rows[0].copy()])


def test_top_edge_features_returns_seven_ranked_tables():
    tables = top_edge_features(
        _edge_pairwise_fixture(),
        contrast="B_vs_A",
        threshold=0.05,
        top_n=20,
    )

    assert list(tables) == [
        "ligands",
        "receptors",
        "lr_pairs",
        "pathways",
        "sources",
        "targets",
        "source_target_pairs",
    ]
    assert tables["ligands"].to_dict("records") == [
        {"ligand_complex": "Lig1", "n_significant_edges": 2}
    ]
    assert tables["lr_pairs"]["n_significant_edges"].tolist() == [1, 1]
    assert tables["source_target_pairs"]["n_significant_edges"].tolist() == [1, 1]
    assert set(tables["pathways"]["pathway_name"]) == {"Path1"}


def test_top_edge_features_applies_top_n_after_count_sorting():
    tables = top_edge_features(
        _edge_pairwise_fixture(),
        contrast="B_vs_A",
        threshold=0.05,
        top_n=1,
    )

    assert all(len(table) == 1 for table in tables.values())
    assert tables["ligands"].iloc[0].to_dict() == {
        "ligand_complex": "Lig1",
        "n_significant_edges": 2,
    }


def test_top_edge_features_rejects_unknown_contrast():
    with pytest.raises(ValueError, match="Unknown contrast"):
        top_edge_features(
            _edge_pairwise_fixture(),
            contrast="missing",
        )
