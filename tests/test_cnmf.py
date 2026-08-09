from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from spatial_workflow.cnmf import (
    _collection_lock,
    cnmf_status,
    export_lineage_counts,
    join_usage_to_counts,
    load_cnmf_context,
    load_lineage_spec,
    resolve_cnmf_paths,
    selection_mask,
    summarize_usage,
    write_cnmf_tmux_scripts,
)


def _config(tmp_path: Path, source_h5ad: Path) -> dict:
    return {
        "project": {"id": "test"},
        "paths": {"results_root": str(tmp_path / "results")},
        "schema": {
            "counts_layer": "counts",
            "broad_cell_type_key": "celltype_short",
            "spatial_domain_key": "spatial_domain",
            "condition_key": "condition",
            "sample_key": "sample_id",
            "spatial_key": "spatial",
        },
        "nncomp": {"input_h5ad": str(source_h5ad)},
        "cnmf": {
            "input_h5ad": str(source_h5ad),
            "output_dir": "cnmf",
            "counts_layer": "counts",
            "cell_type_key": "celltype_short",
            "compartment_key": "spatial_domain",
            "condition_key": "condition",
            "sample_key": "sample_id",
            "spatial_key": "spatial",
            "preparation": {
                "min_counts_per_cell": 1,
                "min_cells_per_gene": 1,
            },
            "lineages": {
                "test_lineage": {
                    "cell_types": ["A", "B"],
                    "exclude_cells": ["cell_1"],
                    "compartments": "all",
                    "conditions": "all",
                    "numgenes": 2,
                    "seed": 14,
                    "max_nmf_iter": 50,
                    "sweep": {
                        "k_values": [2, 3],
                        "n_iter": 2,
                        "workers": 2,
                    },
                    "selected": {
                        "k": 2,
                        "n_iter": 3,
                        "workers": 2,
                        "density_threshold": 0.5,
                        "local_neighborhood_size": 0.5,
                    },
                }
            },
        },
    }


def _write_source(path: Path) -> ad.AnnData:
    counts = sp.csr_matrix(
        np.array(
            [
                [2, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 3, 1, 0],
                [2, 1, 0, 0],
                [0, 1, 3, 0],
                [4, 0, 1, 0],
                [1, 2, 1, 0],
                [2, 0, 2, 0],
            ],
            dtype=np.int32,
        )
    )
    obs = pd.DataFrame(
        {
            "celltype_short": ["A", "A", "B", "B", "A", "B", "other", "A"],
            "cluster_sub": ["A1", "A1", "B1", "B1", "A2", "B2", "other", "A2"],
            "spatial_domain": ["0", "0", "1", "1", "0", "1", "0", "1"],
            "condition": ["control"] * 4 + ["treated"] * 4,
            "sample_id": ["s1"] * 4 + ["s2"] * 4,
        },
        index=[f"cell_{index}" for index in range(8)],
    )
    source = ad.AnnData(
        X=counts.astype(float),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(4)]),
    )
    source.layers["counts"] = counts
    source.obsm["spatial"] = np.arange(16, dtype=float).reshape(8, 2)
    source.write_h5ad(path)
    return source


def test_context_accepts_safe_cell_type_key_override(tmp_path: Path) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_source(source_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path, source_path)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct analysis_name"):
        load_cnmf_context(
            config_path,
            "test_lineage",
            cell_type_key="cluster_sub",
        )

    config, spec, _ = load_cnmf_context(
        config_path,
        "test_lineage",
        cell_type_key="cluster_sub",
        analysis_name="subtype_analysis",
        cell_types=["A1", "A2"],
    )
    assert config["cnmf"]["cell_type_key"] == "cluster_sub"
    assert spec.name == "subtype_analysis"
    assert spec.cell_types == ("A1", "A2")


def test_write_cnmf_tmux_scripts_preserves_resolved_selection(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_source(source_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path, source_path)),
        encoding="utf-8",
    )
    config, spec, paths = load_cnmf_context(
        config_path,
        "test_lineage",
        cell_type_key="cluster_sub",
        analysis_name="MG analysis",
        cell_types=["A1", "A2"],
        compartments="all",
        conditions=["control"],
        selected_k=3,
        workers=4,
    )

    generated = write_cnmf_tmux_scripts(
        config_path,
        "test_lineage",
        config,
        spec,
        paths,
        mode="sweep",
    )
    job_text = Path(generated["job_script"]).read_text(encoding="utf-8")
    launcher_text = Path(generated["launcher_script"]).read_text(encoding="utf-8")

    assert "--mode sweep" in job_text
    assert "--cell-type-key cluster_sub" in job_text
    assert "--analysis-name 'MG analysis'" in job_text
    assert "--cell-type A1 --cell-type A2" in job_text
    assert "--compartment all" in job_text
    assert "--condition control" in job_text
    assert "--selected-k 3" in job_text
    assert "--workers 4" in job_text
    assert generated["session_name"] == "cnmf_MG_analysis_sweep"
    assert "tmux new-session -d" in launcher_text
    assert str(generated["job_script"]) in launcher_text
    assert str(generated["log_path"]) in launcher_text
    assert generated["launch_command"].startswith("bash ")

    with pytest.raises(ValueError, match="mode"):
        write_cnmf_tmux_scripts(
            config_path,
            "test_lineage",
            config,
            spec,
            paths,
            mode="all",
        )


def test_lineage_spec_requires_safe_exact_selections(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "source.h5ad")
    spec = load_lineage_spec(config, "test_lineage", compartments=["all"])
    assert spec.cell_types == ("A", "B")
    assert spec.compartments is None
    assert spec.min_counts_per_cell == 1
    assert spec.min_cells_per_gene == 1

    with pytest.raises(ValueError, match="cannot mix 'all'"):
        load_lineage_spec(
            config,
            "test_lineage",
            analysis_name="bad",
            compartments=["all", "0"],
        )
    with pytest.raises(ValueError, match="distinct analysis_name"):
        load_lineage_spec(config, "test_lineage", compartments=["0"])

    custom = load_lineage_spec(
        config,
        "test_lineage",
        analysis_name="domain_0",
        cell_types=["A"],
        compartments=["0"],
    )
    assert custom.name == "domain_0"
    assert custom.cell_types == ("A",)
    assert custom.compartments == ("0",)

    with pytest.raises(ValueError, match="selected_k"):
        load_lineage_spec(config, "test_lineage", selected_k=0)
    with pytest.raises(ValueError, match="workers"):
        load_lineage_spec(config, "test_lineage", workers=0)


def test_lineage_spec_can_override_preparation_filters(tmp_path: Path) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_source(source_path)
    config = _config(tmp_path, source_path)
    config["cnmf"]["preparation"]["min_cells_per_gene"] = 4
    config["cnmf"]["lineages"]["test_lineage"]["preparation"] = {
        "min_cells_per_gene": 0,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder", encoding="utf-8")

    spec = load_lineage_spec(config, "test_lineage")
    assert spec.min_cells_per_gene == 0
    paths = resolve_cnmf_paths(config_path, config, spec)
    manifest = export_lineage_counts(config, spec, paths)
    assert manifest["selection"]["min_cells_per_gene"] == 0
    assert manifest["output"]["shape"] == [6, 4]




def test_selection_mask_is_exact_and_excludes_ids() -> None:
    obs = pd.DataFrame(
        {
            "celltype_short": ["AST", "AST-CX", "AST-CX"],
            "spatial_domain": ["3", "3", "4"],
            "condition": ["a", "a", "b"],
        },
        index=["x", "y", "z"],
    )
    mask = selection_mask(
        obs,
        cell_type_key="celltype_short",
        cell_types=["AST-CX"],
        compartment_key="spatial_domain",
        compartments=["3"],
        condition_key="condition",
        conditions=None,
        exclude_cells=["x"],
    )
    assert mask.tolist() == [False, True, False]


def test_collection_lock_rejects_duplicate_launcher(tmp_path: Path) -> None:
    output_dir = tmp_path / "collection"
    with _collection_lock(output_dir, collection="test", dry_run=False):
        with pytest.raises(RuntimeError, match="Another test launcher"):
            with _collection_lock(output_dir, collection="test", dry_run=False):
                pass


def test_export_status_and_usage_join(tmp_path: Path) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_source(source_path)
    config = _config(tmp_path, source_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    spec = load_lineage_spec(config, "test_lineage")
    paths = resolve_cnmf_paths(config_path, config, spec)

    manifest = export_lineage_counts(config, spec, paths)
    assert manifest["output"]["shape"] == [6, 3]
    exported = ad.read_h5ad(paths.counts_h5ad)
    assert exported.obs_names.tolist() == [
        "cell_0",
        "cell_2",
        "cell_3",
        "cell_4",
        "cell_5",
        "cell_7",
    ]
    assert exported.var_names.tolist() == ["gene_0", "gene_1", "gene_2"]
    assert sp.issparse(exported.X)
    assert np.array_equal(exported.obsm["spatial"], np.array([
        [0, 1], [4, 5], [6, 7], [8, 9], [10, 11], [14, 15]
    ], dtype=float))

    reused = export_lineage_counts(config, spec, paths)
    assert reused["output"]["cell_id_sha256"] == manifest["output"]["cell_id_sha256"]
    input_status = cnmf_status(spec, paths).set_index("stage").loc["input_export"]
    assert bool(input_status["complete"])
    assert input_status["observed"] == input_status["expected"] == 5

    usage = pd.DataFrame(
        {"Usage_1": np.arange(exported.n_obs, dtype=float)},
        index=exported.obs_names[::-1],
    )
    joined = join_usage_to_counts(paths.counts_h5ad, usage)
    assert joined.obs["Usage_1"].tolist() == [5, 4, 3, 2, 1, 0]
    with pytest.raises(RuntimeError, match="do not exactly match"):
        join_usage_to_counts(paths.counts_h5ad, usage.iloc[:-1])

    grouped = summarize_usage(
        joined.obs,
        group_columns=["condition"],
        usage_cols=["Usage_1"],
    )
    assert set(grouped.columns) == {
        "condition",
        "program",
        "mean_usage",
        "median_usage",
        "n_cells",
    }

    with paths.export_manifest.open(encoding="utf-8") as handle:
        tampered = json.load(handle)
    tampered["selection"]["cell_types"] = ["wrong"]
    paths.export_manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FileExistsError, match="does not match"):
        export_lineage_counts(config, spec, paths)
