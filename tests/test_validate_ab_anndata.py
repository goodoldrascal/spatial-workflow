from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_ab_anndata.py"


def _write_pair(
    tmp_path: Path,
    *,
    count_mismatch: bool = False,
    new_conditions: list[str] | None = None,
    reference_conditions: list[str] | None = None,
) -> tuple[Path, Path]:
    reference_path = tmp_path / "reference.h5ad"
    new_path = tmp_path / "new.h5ad"

    counts = np.asarray(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5],
            [0, 6, 0],
        ],
        dtype=np.int64,
    )
    samples = np.asarray(["sample_a", "sample_a", "sample_b", "sample_b"])
    cell_ids = np.asarray(["cell_1", "cell_2", "cell_3", "cell_4"])
    conditions = np.asarray(["control", "control", "treated", "treated"])
    new_conditions = np.asarray(
        conditions if new_conditions is None else new_conditions
    )
    reference_conditions = np.asarray(
        conditions if reference_conditions is None else reference_conditions
    )
    labels = {
        "cluster_sub": ["A1", "A2", "B1", "B2"],
        "celltype_short": ["A", "A", "B", "B"],
        "celltype_full": ["A one", "A two", "B one", "B two"],
    }
    coordinates = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )

    reference_obs = pd.DataFrame(
        {"sample": samples, "condition": reference_conditions, **labels},
        index=pd.Index(
            [
                f"{sample}_subclusters:{cell_id}"
                for sample, cell_id in zip(samples, cell_ids)
            ]
        ),
    )
    reference_obs[["centroid_x", "centroid_y"]] = coordinates
    reference = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=reference_obs,
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC"]),
    )
    reference.obsm["spatial"] = coordinates
    reference.write_h5ad(reference_path)

    order = np.asarray([2, 0, 3, 1])
    new_counts = counts[order].copy()
    if count_mismatch:
        new_counts[0, 0] += 1
    new_obs = pd.DataFrame(
        {
            "sample_id": samples[order],
            "cell_id": cell_ids[order],
            "condition": new_conditions[order],
            **{
                field: np.asarray(values)[order]
                for field, values in labels.items()
            },
        },
        index=pd.Index(
            [
                f"{sample}:{cell_id}"
                for sample, cell_id in zip(samples[order], cell_ids[order])
            ]
        ),
    )
    new_obs[["centroid_x", "centroid_y"]] = coordinates[order]
    converted = ad.AnnData(
        X=sp.csr_matrix(np.log1p(new_counts)),
        obs=new_obs,
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC"]),
    )
    converted.layers["counts"] = sp.csr_matrix(new_counts)
    converted.obsm["spatial"] = coordinates[order]
    converted.write_h5ad(new_path)
    return new_path, reference_path


def _run(
    new_path: Path, reference_path: Path, report_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(new_path),
            "--reference",
            str(reference_path),
            "--report",
            str(report_path),
            "--expected-samples",
            "2",
            "--chunk-size",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validator_aligns_composite_identity_without_legacy_obs_names(
    tmp_path: Path,
) -> None:
    new_path, reference_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"

    result = _run(new_path, reference_path, report_path)
    report = json.loads(report_path.read_text())

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["status"] == "pass"
    assert report["checks"]["samples"]["pass"]
    assert report["checks"]["cell_identity"]["pass"]
    assert not report["checks"]["cell_identity"]["same_order"]
    assert report["checks"]["conditions"]["pass"]
    assert report["checks"]["conditions"]["new_condition_by_sample"] == {
        "sample_a": "control",
        "sample_b": "treated",
    }
    assert report["checks"]["raw_counts"]["pass"]
    assert (
        report["checks"]["raw_counts"]["method"]
        == "blake2b_row_digest_after_identity_alignment"
    )


def test_validator_returns_nonzero_for_raw_count_mismatch(tmp_path: Path) -> None:
    new_path, reference_path = _write_pair(tmp_path, count_mismatch=True)
    report_path = tmp_path / "report.json"

    result = _run(new_path, reference_path, report_path)
    report = json.loads(report_path.read_text())

    assert result.returncode == 1
    assert report["status"] == "fail"
    assert not report["checks"]["raw_counts"]["pass"]
    assert report["checks"]["raw_counts"]["mismatched_rows"] == 1


def test_validator_rejects_aligned_condition_mismatch(tmp_path: Path) -> None:
    new_path, reference_path = _write_pair(
        tmp_path,
        new_conditions=["control", "control", "control", "control"],
    )
    report_path = tmp_path / "report.json"

    result = _run(new_path, reference_path, report_path)
    report = json.loads(report_path.read_text())
    conditions = report["checks"]["conditions"]

    assert result.returncode == 1
    assert report["status"] == "fail"
    assert not conditions["pass"]
    assert conditions["mismatched"] == 2
    assert conditions["new_samples_with_multiple_conditions"] == {}


def test_validator_rejects_multiple_conditions_per_sample(tmp_path: Path) -> None:
    new_path, reference_path = _write_pair(
        tmp_path,
        new_conditions=["control", "treated", "treated", "treated"],
    )
    report_path = tmp_path / "report.json"

    result = _run(new_path, reference_path, report_path)
    report = json.loads(report_path.read_text())
    conditions = report["checks"]["conditions"]

    assert result.returncode == 1
    assert not conditions["pass"]
    assert conditions["new_samples_with_multiple_conditions"] == {
        "sample_a": ["control", "treated"]
    }


def test_validator_rejects_multiple_reference_conditions_per_sample(
    tmp_path: Path,
) -> None:
    new_path, reference_path = _write_pair(
        tmp_path,
        reference_conditions=["control", "treated", "treated", "treated"],
    )
    report_path = tmp_path / "report.json"

    result = _run(new_path, reference_path, report_path)
    report = json.loads(report_path.read_text())
    conditions = report["checks"]["conditions"]

    assert result.returncode == 1
    assert not conditions["pass"]
    assert conditions["reference_samples_with_multiple_conditions"] == {
        "sample_a": ["control", "treated"]
    }
