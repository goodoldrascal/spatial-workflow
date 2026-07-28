from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import pytest
import yaml

import spatial_workflow.ingest as ingest
from spatial_workflow.ingest import read_bundle, read_xenium, run_ingest


SCHEMA = {
    "cell_id_key": "cell_id",
    "sample_key": "sample_id",
    "condition_key": "condition",
    "counts_layer": "counts",
    "spatial_key": "spatial",
}


def _gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def _write_bundle(
    path: Path,
    genes_by_cells: np.ndarray | None = None,
    genes: list[str] | None = None,
) -> np.ndarray:
    path.mkdir()
    if genes_by_cells is None:
        genes_by_cells = np.array([[1, 0], [2, 3], [0, 4]], dtype=np.int32)
    if genes is None:
        genes = ["GeneA", "GeneB", "GeneC"]
    matrix_path = path / "counts.mtx"
    scipy.io.mmwrite(matrix_path, sp.coo_matrix(genes_by_cells))
    with matrix_path.open("rb") as source, gzip.open(path / "counts.mtx.gz", "wb") as target:
        target.write(source.read())
    matrix_path.unlink()
    _gzip_text(path / "genes.tsv.gz", "\n".join(genes) + "\n")
    _gzip_text(path / "barcodes.tsv.gz", "cell1\ncell2\n")
    pd.DataFrame(
        {
            "cell_id": ["cell2", "cell1"],
            "cell_type": ["B", "A"],
        }
    ).to_csv(path / "obs.csv.gz", index=False)
    pd.DataFrame(
        {
            "cell_id": ["cell2", "cell1"],
            "x": [20.0, 10.0],
            "y": [2.0, 1.0],
        }
    ).to_csv(path / "spatial.csv.gz", index=False)
    return genes_by_cells.T


def test_read_bundle_preserves_counts_and_aligns_cells(tmp_path: Path) -> None:
    expected_counts = _write_bundle(tmp_path / "bundle")
    data = read_bundle(
        tmp_path / "bundle",
        sample_id="sample_1",
        condition="control",
        schema=SCHEMA,
        normalize_target=10,
    )

    np.testing.assert_array_equal(data.layers["counts"].toarray(), expected_counts)
    assert data.layers["counts"].dtype == np.dtype(np.int32)
    assert data.obs_names.tolist() == ["sample_1:cell1", "sample_1:cell2"]
    assert data.obs["cell_type"].tolist() == ["A", "B"]
    assert data.obs["sample_id"].tolist() == ["sample_1", "sample_1"]
    assert data.obs["condition"].tolist() == ["control", "control"]
    np.testing.assert_array_equal(data.obsm["spatial"], [[10.0, 1.0], [20.0, 2.0]])
    expected = np.log1p(expected_counts / expected_counts.sum(axis=1, keepdims=True) * 10)
    np.testing.assert_allclose(data.X.toarray(), expected, rtol=1e-6)


def test_validate_counts_uses_int32_and_guards_range() -> None:
    counts = sp.csr_matrix(np.array([[0, 7]], dtype=np.int64))

    validated = ingest._validate_counts(counts)

    assert validated.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(validated.toarray(), [[0, 7]])

    too_large = sp.csr_matrix(
        np.array([[np.iinfo(np.int32).max + 1]], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="int32 storage range"):
        ingest._validate_counts(too_large)


def _write_xenium(path: Path) -> np.ndarray:
    path.mkdir()
    features_by_cells = sp.csc_matrix(
        np.array(
            [
                [1, 0],
                [9, 9],
                [2, 3],
            ],
            dtype=np.int32,
        )
    )
    with h5py.File(path / "cell_feature_matrix.h5", "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=features_by_cells.data)
        group.create_dataset("indices", data=features_by_cells.indices)
        group.create_dataset("indptr", data=features_by_cells.indptr)
        group.create_dataset("shape", data=features_by_cells.shape)
        group.create_dataset("barcodes", data=np.asarray([b"c1", b"c2"]))
        features = group.create_group("features")
        features.create_dataset("name", data=np.asarray([b"GeneA", b"NegControl", b"GeneB"]))
        features.create_dataset("id", data=np.asarray([b"g1", b"neg1", b"g2"]))
        features.create_dataset(
            "feature_type",
            data=np.asarray([b"Gene Expression", b"Negative Control Codeword", b"Gene Expression"]),
        )
    pd.DataFrame(
        {
            "cell_id": ["c2", "c1"],
            "x_centroid": [2.0, 1.0],
            "y_centroid": [4.0, 3.0],
        }
    ).to_csv(path / "cells.csv.gz", index=False)
    return np.array([[1, 2], [0, 3]], dtype=np.int32)


def test_read_xenium_filters_features_and_preserves_raw_counts(tmp_path: Path) -> None:
    expected_counts = _write_xenium(tmp_path / "xenium")
    data = read_xenium(
        tmp_path / "xenium",
        sample_id="sample_x",
        condition="treated",
        schema=SCHEMA,
        normalize_target=None,
    )

    assert data.var_names.tolist() == ["GeneA", "GeneB"]
    assert data.var["feature_id"].tolist() == ["g1", "g2"]
    np.testing.assert_array_equal(data.layers["counts"].toarray(), expected_counts)
    np.testing.assert_array_equal(data.X.toarray(), expected_counts)
    np.testing.assert_array_equal(data.obsm["spatial"], [[1.0, 3.0], [2.0, 4.0]])
    assert data.obs_names.tolist() == ["sample_x:c1", "sample_x:c2"]


def test_run_ingest_merges_multiple_bundle_samples(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir()
    _write_bundle(
        data_root / "bundle_a",
        np.array([[1, 0], [2, 3]], dtype=np.int32),
        ["GeneA", "GeneB"],
    )
    _write_bundle(
        data_root / "bundle_b",
        np.array([[4, 0], [0, 5]], dtype=np.int32),
        ["GeneB", "GeneC"],
    )

    config = {
        "project": {"id": "multi_sample"},
        "paths": {
            "data_root": str(data_root),
            "results_root": str(results_root),
        },
        "schema": SCHEMA,
        "conversion": {
            "input_format": "bundle",
            "output_dir": "01_anndata",
            "output_name": "merged",
            "normalize_target": None,
            "join": "outer",
            "samples": [
                {"input_path": "bundle_a", "sample_id": "sample_a", "condition": "control"},
                {"input_path": "bundle_b", "sample_id": "sample_b", "condition": "treated"},
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    output_path = run_ingest(config_path)
    merged = ad.read_h5ad(output_path)

    assert merged.shape == (4, 3)
    assert set(merged.obs["sample_id"]) == {"sample_a", "sample_b"}
    assert set(merged.obs["condition"]) == {"control", "treated"}
    assert merged.obs_names.is_unique
    assert "counts" in merged.layers

    counts = pd.DataFrame(
        merged.layers["counts"].toarray(),
        index=merged.obs_names,
        columns=merged.var_names,
    )
    assert counts.loc["sample_a:cell1", "GeneC"] == 0
    assert counts.loc["sample_b:cell1", "GeneA"] == 0

    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    assert manifest["join"] == "outer"
    assert len(manifest["sources"]) == 2
    assert manifest["sample_cell_counts"] == {"sample_a": 2, "sample_b": 2}
    assert manifest["configuration"] == config


def test_export_merged_seurat_passes_metadata_keys_and_reads_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "bundles"
    captured: dict = {}

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        bundle_root.mkdir()
        pd.DataFrame(
            {
                "sample_id": ["sample_01", "sample_02"],
                "condition": ["control", "treated"],
                "bundle_dir": ["sample_0001", "sample_0002"],
                "n_cells": [2, 3],
            }
        ).to_csv(bundle_root / "bundle_manifest.csv", index=False)

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    manifest_path, records = ingest._export_merged_seurat(
        tmp_path / "merged.qs",
        bundle_root,
        assay="Xenium",
        layer="counts",
        sample_key="specimen",
        condition_key="treatment",
        rscript_bin="Rscript",
        r_libs_user=tmp_path / ".r-lib",
    )

    command = captured["command"]
    assert command[command.index("--sample-key") + 1] == "specimen"
    assert command[command.index("--condition-key") + 1] == "treatment"
    assert captured["check"] is True
    assert captured["env"]["R_LIBS_USER"] == str(tmp_path / ".r-lib")
    assert manifest_path == bundle_root / "bundle_manifest.csv"
    assert records == [
        {
            "sample_id": "sample_01",
            "condition": "control",
            "bundle_path": bundle_root / "sample_0001",
            "n_cells": 2,
        },
        {
            "sample_id": "sample_02",
            "condition": "treated",
            "bundle_path": bundle_root / "sample_0002",
            "n_cells": 3,
        },
    ]


def test_run_ingest_splits_merged_seurat_and_preserves_design_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir()
    (data_root / "merged.qs").touch()
    schema = {
        **SCHEMA,
        "sample_key": "specimen",
        "condition_key": "treatment",
    }
    captured: dict = {}

    def fake_export(
        input_path: Path,
        bundle_root: Path,
        **kwargs: object,
    ) -> tuple[Path, list[dict]]:
        bundle_root.mkdir(parents=True)
        captured.update(kwargs)
        records = []
        for index, (sample_id, condition) in enumerate(
            [("sample_a", "control"), ("sample_b", "treated")],
            start=1,
        ):
            bundle_path = bundle_root / f"sample_{index:04d}"
            _write_bundle(
                bundle_path,
                np.array([[index, 0], [2, 3]], dtype=np.int32),
                ["GeneA", "GeneB"],
            )
            obs = pd.read_csv(bundle_path / "obs.csv.gz")
            obs["specimen"] = sample_id
            obs["treatment"] = condition
            obs["cluster_sub"] = ["B", "A"]
            obs.to_csv(bundle_path / "obs.csv.gz", index=False)
            records.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "bundle_path": bundle_path,
                    "n_cells": 2,
                }
            )
        manifest_path = bundle_root / "bundle_manifest.csv"
        pd.DataFrame(
            {
                "sample_id": [record["sample_id"] for record in records],
                "condition": [record["condition"] for record in records],
                "bundle_dir": [record["bundle_path"].name for record in records],
                "n_cells": [record["n_cells"] for record in records],
            }
        ).to_csv(manifest_path, index=False)
        return manifest_path, records

    monkeypatch.setattr(ingest, "_export_merged_seurat", fake_export)
    config = {
        "project": {"id": "merged_seurat"},
        "paths": {"data_root": str(data_root), "results_root": str(results_root)},
        "schema": schema,
        "conversion": {
            "input_format": "merged_seurat",
            "input_path": "merged.qs",
            "assay": "Xenium",
            "layer": "counts",
            "output_dir": "01_anndata",
            "output_name": "merged",
            "normalize_target": None,
            "join": "inner",
        },
        "runtime": {"r_libs_user": "r-library"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    output_path = run_ingest(config_path)
    data = ad.read_h5ad(output_path)

    assert data.obs_names.tolist() == [
        "sample_a:cell1",
        "sample_a:cell2",
        "sample_b:cell1",
        "sample_b:cell2",
    ]
    assert data.obs["specimen"].tolist() == ["sample_a"] * 2 + ["sample_b"] * 2
    assert data.obs["treatment"].tolist() == ["control"] * 2 + ["treated"] * 2
    assert data.obs["cluster_sub"].tolist() == ["A", "B", "A", "B"]
    assert captured["sample_key"] == "specimen"
    assert captured["condition_key"] == "treatment"
    assert captured["r_libs_user"] == (tmp_path / "r-library").resolve()

    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    assert manifest["input_mode"] == "merged_seurat"
    assert len(manifest["input_objects"]) == 1
    assert manifest["input_objects"][0]["n_samples"] == 2
    assert manifest["sample_cell_counts"] == {"sample_a": 2, "sample_b": 2}
    assert {source["condition_source"] for source in manifest["sources"]} == {
        "obs.treatment"
    }
