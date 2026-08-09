import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from spatial_workflow.cellcharter import _abundance_table, run_cellcharter


def _write_reuse_fixture(tmp_path: Path, *, omit_source_cell: bool = False) -> Path:
    results = tmp_path / "results"
    input_dir = results / "01_anndata"
    input_dir.mkdir(parents=True)
    obs = pd.DataFrame(
        {
            "cell_id": ["c2", "c1", "c2", "c1"],
            "sample_id": ["s1", "s1", "s2", "s2"],
            "condition": ["control", "control", "treated", "treated"],
            "cell_type": ["B", "A", "A", "B"],
        },
        index=["s1:c2", "s1:c1", "s2:c2", "s2:c1"],
    )
    data = ad.AnnData(np.ones((4, 2)), obs=obs)
    data.obsm["spatial"] = np.arange(8).reshape(4, 2)
    data.write_h5ad(input_dir / "sample.h5ad")

    source = pd.DataFrame(
        {
            "sample": ["s2", "s1", "s2", "s1"],
            "cellcharter_autok": ["1", "0", "0", "1"],
        },
        index=["s2_subclusters:c1", "s1_subclusters:c1", "s2_subclusters:c2", "s1_subclusters:c2"],
    )
    if omit_source_cell:
        source = source.iloc[1:]
    source.to_csv(tmp_path / "legacy_obs.csv.gz")
    (tmp_path / "legacy_summary.json").write_text(
        json.dumps({"best_k": 2, "peaks": [2]})
    )
    pd.DataFrame(
        {"k": [2, 3], "0": [0.8, 0.6], "1": [1.0, 0.4]}
    ).to_csv(tmp_path / "legacy_stability.csv", index=False)

    config = {
        "project": {"id": "reuse-test"},
        "paths": {"results_root": str(results)},
        "schema": {
            "cell_id_key": "cell_id",
            "sample_key": "sample_id",
            "condition_key": "condition",
            "cell_type_key": "cell_type",
            "spatial_domain_key": "spatial_domain",
            "counts_layer": "counts",
            "spatial_key": "spatial",
        },
        "cellcharter": {
            "input_h5ad": "01_anndata/sample.h5ad",
            "output_dir": "02_cellcharter",
            "output_name": "sample_cellcharter",
            "reuse": {
                "enabled": True,
                "source_obs": str(tmp_path / "legacy_obs.csv.gz"),
                "source_summary": str(tmp_path / "legacy_summary.json"),
                "source_stability": str(tmp_path / "legacy_stability.csv"),
                "source_sample_key": "sample",
                "source_domain_key": "cellcharter_autok",
            },
        },
    }
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_reuse_materializes_current_cellcharter_contract(tmp_path: Path) -> None:
    config_path = _write_reuse_fixture(tmp_path)

    output = run_cellcharter(config_path)
    result = ad.read_h5ad(output)
    root = output.parent

    assert result.obs["spatial_domain"].astype(str).tolist() == ["1", "0", "0", "1"]
    assert "spatial_domain" not in ad.read_h5ad(
        tmp_path / "results/01_anndata/sample.h5ad", backed="r"
    ).obs
    stability = pd.read_csv(root / "sample_cellcharter.autok_stability.csv")
    assert stability["k"].tolist() == [2, 3]
    assert stability["n_comparisons"].tolist() == [2, 2]
    assert np.allclose(stability["mean_stability"], [0.9, 0.5])
    assert np.allclose(stability["sd_stability"], [0.1, 0.1])
    assert (root / "sample_cellcharter.autok_stability_comparisons.csv").exists()
    assert (root / "sample_cellcharter.domain_abundance.csv").exists()

    summary = json.loads(
        (root / "sample_cellcharter.summary.json").read_text()
    )
    assert summary["mode"] == "reused_prior_run"
    assert summary["best_k"] == 2
    assert summary["peaks"] == [2]
    assert summary["fitted_cellcharter"] is False
    assert summary["reuse"]["matched_cells"] == 4
    assert summary["reuse"]["target_domain_key"] == "spatial_domain"


def test_reuse_requires_an_exact_cell_identity_match(tmp_path: Path) -> None:
    config_path = _write_reuse_fixture(tmp_path, omit_source_cell=True)

    with pytest.raises(ValueError, match="missing=1, extra=0"):
        run_cellcharter(config_path)


def test_abundance_handles_categorical_samples_with_distinct_totals() -> None:
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["s1", "s2", "s2"]),
            "condition": pd.Categorical(["a", "b", "b"]),
            "domain": pd.Categorical(["0", "0", "1"]),
        }
    )

    abundance = _abundance_table(
        obs,
        domain_key="domain",
        sample_key="sample",
        condition_key="condition",
    )

    assert abundance.groupby("sample", observed=True)["sample_cells"].first().to_dict() == {
        "s1": 1,
        "s2": 2,
    }
    assert np.allclose(abundance["fraction"], [1.0, 0.5, 0.5])
