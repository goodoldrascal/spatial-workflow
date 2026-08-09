from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatial_workflow.liana_window import (
    audit_liana_input,
    load_cellchat_resource,
    load_custom_lr_resource,
    load_rankagg_resource,
)


def _write_cellchat(path: Path) -> None:
    pd.DataFrame(
        {
            "ligand": ["Lig1", "Lig2"],
            "receptor": ["Rec1", "Rec2"],
            "pathway_name": ["PATH_A", "PATH_B"],
            "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
        }
    ).to_csv(path, index=False)


def test_resource_scope_cellchat_custom_and_liana(tmp_path: Path):
    cellchat_path = tmp_path / "cellchat.csv"
    _write_cellchat(cellchat_path)

    resource, metadata = load_cellchat_resource(
        cellchat_path,
        annotations=["Secreted Signaling"],
        lr_pairs=["Lig1^Rec1"],
    )
    assert resource.to_dict("records") == [{"ligand": "Lig1", "receptor": "Rec1"}]
    assert len(metadata) == 1
    assert metadata.attrs["total_database_rows"] == 2

    custom_path = tmp_path / "custom.csv"
    pd.DataFrame(
        {
            "ligand_complex": ["Lig1", "LigX_Sub"],
            "receptor_complex": ["Rec1", "RecX"],
            "pathway_name": ["PATH_A", "PATH_X"],
        }
    ).to_csv(custom_path, index=False)
    custom, custom_metadata = load_custom_lr_resource(
        custom_path, pathways=["PATH_X"]
    )
    assert custom.to_dict("records") == [{"ligand": "LigX_Sub", "receptor": "RecX"}]
    assert custom_metadata["pathway_name"].tolist() == ["PATH_X"]

    class FakeResource:
        @staticmethod
        def select_resource(name):
            assert name == "mouseconsensus"
            return pd.DataFrame(
                {"ligand": ["Lig1", "Lig2"], "receptor": ["Rec1", "Rec2"]}
            )

    class FakeLiana:
        resource = FakeResource()

    native, native_metadata = load_rankagg_resource(
        FakeLiana(), mode="liana", lr_pairs=["Lig2^Rec2"]
    )
    assert native.to_dict("records") == [{"ligand": "Lig2", "receptor": "Rec2"}]
    assert native_metadata["pathway_name"].isna().all()


def test_input_audit_filters_cells_and_pools_compartments(tmp_path: Path):
    adata = ad.AnnData(np.ones((6, 2)))
    adata.obs["sample"] = ["s1", "s1", "s1", "s2", "s2", "s2"]
    adata.obs["condition"] = ["A", "A", "A", "B", "B", "B"]
    adata.obs["cell_type"] = ["Ast", "Ast", "Mic", "Ast", "Mic", "Mic"]
    adata.obs["compartment"] = ["0", "1", "1", "0", "0", "1"]
    adata.obsm["spatial"] = np.column_stack([np.arange(6), np.zeros(6)])
    path = tmp_path / "input.h5ad"
    adata.write_h5ad(path)

    audit = audit_liana_input(
        path,
        sample_key="sample",
        condition_key="condition",
        group_key="cell_type",
        spatial_key="spatial",
        compartment_key="compartment",
        cell_types=["Ast"],
        compartments=["0", "1"],
    )
    assert audit["shape"]["cells_before_scope"] == 6
    assert audit["shape"]["cells_after_scope"] == 3
    assert set(audit["compartments"]["compartment"].astype(str)) == {"0", "1"}
