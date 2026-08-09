from __future__ import annotations

from pathlib import Path

import pandas as pd

from spatial_workflow.liana_window import (
    CELLCHAT_PROTEIN_ANNOTATIONS,
    load_cellchat_resource,
)


def _write_cellchat_v2_fixture(path: Path) -> None:
    pd.DataFrame(
        {
            "ligand": ["Wnt1", "Col1a1", "Cadm1", "GABA-GAD1"],
            "receptor": ["Fzd1", "Itga1_Itgb1", "Cadm2", "Gabra1"],
            "pathway_name": ["WNT", "COLLAGEN", "CADM", "GABA-A"],
            "annotation": [
                "Secreted Signaling",
                "ECM-Receptor",
                "Cell-Cell Contact",
                "Non-protein Signaling",
            ],
        }
    ).to_csv(path, index=False)


def test_cellchat_default_matches_v2_subsetdb_and_keeps_wnt(tmp_path: Path):
    path = tmp_path / "cellchat_v2.csv"
    _write_cellchat_v2_fixture(path)

    resource, metadata = load_cellchat_resource(path)

    assert set(metadata["annotation"]) == set(CELLCHAT_PROTEIN_ANNOTATIONS)
    assert "WNT" in set(metadata["pathway_name"])
    assert "GABA-A" not in set(metadata["pathway_name"])
    assert len(resource) == 3
    assert metadata.attrs["total_database_rows"] == 4
    assert metadata.attrs["selected_database_rows"] == 3


def test_cellchat_non_protein_requires_explicit_opt_in(tmp_path: Path):
    path = tmp_path / "cellchat_v2.csv"
    _write_cellchat_v2_fixture(path)

    resource, metadata = load_cellchat_resource(path, include_non_protein=True)

    assert len(resource) == 4
    assert "Non-protein Signaling" in set(metadata["annotation"])
    assert "GABA-A" in set(metadata["pathway_name"])
