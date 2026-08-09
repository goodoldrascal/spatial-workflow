#!/usr/bin/env python3
"""Validate a converted AB H5AD against the legacy raw-count reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem


DEFAULT_REFERENCE = Path("reference.h5ad")
REQUIRED_LABELS = ("cluster_sub", "celltype_short", "celltype_full")


class H5adMatrix:
    """Read contiguous row blocks from a dense or CSR-encoded H5AD matrix."""

    def __init__(self, node: h5py.Dataset | h5py.Group, source: str):
        self.node = node
        self.source = source
        encoding = node.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        self.encoding = str(encoding)

        if isinstance(node, h5py.Dataset):
            self.shape = tuple(int(value) for value in node.shape)
        elif self.encoding == "csr_matrix":
            self.shape = tuple(int(value) for value in node.attrs["shape"])
        else:
            raise ValueError(
                f"{source} must be dense or CSR encoded, found {self.encoding!r}"
            )

        if len(self.shape) != 2:
            raise ValueError(f"{source} is not a two-dimensional matrix")

    def rows(self, start: int, stop: int) -> sp.csr_matrix:
        if isinstance(self.node, h5py.Dataset):
            return sp.csr_matrix(np.asarray(self.node[start:stop, :]))

        indptr = np.asarray(self.node["indptr"][start : stop + 1], dtype=np.int64)
        data_start = int(indptr[0])
        data_stop = int(indptr[-1])
        data = np.asarray(self.node["data"][data_start:data_stop])
        indices = np.asarray(
            self.node["indices"][data_start:data_stop], dtype=np.int64
        )
        return sp.csr_matrix(
            (data, indices, indptr - data_start),
            shape=(stop - start, self.shape[1]),
        )


def _node(handle: h5py.File, path: str) -> h5py.Dataset | h5py.Group:
    current: h5py.File | h5py.Group = handle
    for part in path.strip("/").split("/"):
        current = current[part]
    return current


def _strings(values: pd.Series | pd.Index) -> np.ndarray:
    return pd.Series(values, copy=False).astype("string").fillna("").to_numpy(str)


def _reference_cell_ids(obs_names: pd.Index) -> np.ndarray:
    names = _strings(obs_names)
    cell_ids = np.asarray(
        [name.partition(":")[2] if ":" in name else "" for name in names],
        dtype=object,
    )
    if np.any(cell_ids == ""):
        raise ValueError(
            "Reference observation names must have the form "
            "'sample_subclusters:cell_id'"
        )
    return cell_ids.astype(str)


def _sample_counts(samples: np.ndarray) -> dict[str, int]:
    counts = pd.Series(samples).value_counts(sort=False).sort_index()
    return {str(sample): int(count) for sample, count in counts.items()}


def _key_example(key: tuple[object, object]) -> dict[str, str]:
    return {"sample_id": str(key[0]), "cell_id": str(key[1])}


def _first_gene_mismatch(
    new_names: np.ndarray, reference_names: np.ndarray
) -> dict[str, object] | None:
    limit = min(len(new_names), len(reference_names))
    mismatch = np.flatnonzero(new_names[:limit] != reference_names[:limit])
    if mismatch.size:
        index = int(mismatch[0])
        return {
            "index": index,
            "new": str(new_names[index]),
            "reference": str(reference_names[index]),
        }
    if len(new_names) != len(reference_names):
        return {"index": limit, "new": None, "reference": None}
    return None


def _canonical_counts(matrix: sp.csr_matrix, source: str) -> sp.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    values = matrix.data
    if values.size:
        rounded = np.rint(values)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.array_equal(values, rounded)
        ):
            raise ValueError(f"{source} contains non-integer or invalid raw counts")
        matrix.data = rounded.astype(np.int64)
    return matrix


def _row_digests(matrix: sp.csr_matrix) -> np.ndarray:
    digests = np.empty(matrix.shape[0], dtype="V16")
    for row in range(matrix.shape[0]):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray([stop - start], dtype="<i8").tobytes())
        digest.update(np.asarray(matrix.indices[start:stop], dtype="<i8").tobytes())
        digest.update(np.asarray(matrix.data[start:stop], dtype="<i8").tobytes())
        digests[row] = np.void(digest.digest())
    return digests


def _compare_counts(
    new_matrix: H5adMatrix,
    reference_matrix: H5adMatrix,
    reference_positions: np.ndarray,
    keys: pd.MultiIndex,
    chunk_size: int,
) -> dict[str, object]:
    if new_matrix.shape != reference_matrix.shape:
        return {
            "pass": False,
            "new_shape": list(new_matrix.shape),
            "reference_shape": list(reference_matrix.shape),
            "reason": "matrix shape mismatch",
        }

    n_rows = new_matrix.shape[0]
    same_order = np.array_equal(reference_positions, np.arange(n_rows))
    mismatched_rows = 0
    mismatched_entries = 0
    first_mismatch: dict[str, str] | None = None

    if same_order:
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            new = _canonical_counts(
                new_matrix.rows(start, stop), new_matrix.source
            )
            reference = _canonical_counts(
                reference_matrix.rows(start, stop), reference_matrix.source
            )
            difference = new != reference
            row_mismatch = np.asarray(difference.getnnz(axis=1)).ravel() > 0
            mismatched_rows += int(row_mismatch.sum())
            mismatched_entries += int(difference.nnz)
            if first_mismatch is None and row_mismatch.any():
                offset = int(np.flatnonzero(row_mismatch)[0])
                first_mismatch = _key_example(keys[start + offset])
        method = "exact_chunked"
    else:
        reference_digests = np.empty(n_rows, dtype="V16")
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            reference = _canonical_counts(
                reference_matrix.rows(start, stop), reference_matrix.source
            )
            reference_digests[start:stop] = _row_digests(reference)

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            new = _canonical_counts(
                new_matrix.rows(start, stop), new_matrix.source
            )
            observed = _row_digests(new)
            expected = reference_digests[reference_positions[start:stop]]
            row_mismatch = observed != expected
            mismatched_rows += int(row_mismatch.sum())
            if first_mismatch is None and row_mismatch.any():
                offset = int(np.flatnonzero(row_mismatch)[0])
                first_mismatch = _key_example(keys[start + offset])
        mismatched_entries = None
        method = "blake2b_row_digest_after_identity_alignment"

    return {
        "pass": mismatched_rows == 0,
        "method": method,
        "new_source": new_matrix.source,
        "reference_source": reference_matrix.source,
        "mismatched_rows": mismatched_rows,
        "mismatched_entries": mismatched_entries,
        "first_mismatch": first_mismatch,
    }


def _compare_labels(
    new_obs: pd.DataFrame,
    reference_obs: pd.DataFrame,
    reference_positions: np.ndarray,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for field in REQUIRED_LABELS:
        if field not in new_obs or field not in reference_obs:
            fields[field] = {
                "pass": False,
                "missing_from": [
                    name
                    for name, obs in (
                        ("new", new_obs),
                        ("reference", reference_obs),
                    )
                    if field not in obs
                ],
            }
            continue

        observed = _strings(new_obs[field])
        expected = _strings(reference_obs[field])[reference_positions]
        missing = observed == ""
        mismatch = observed != expected
        fields[field] = {
            "pass": not missing.any() and not mismatch.any(),
            "missing_or_empty": int(missing.sum()),
            "mismatched": int(mismatch.sum()),
        }

    return {
        "pass": all(bool(result["pass"]) for result in fields.values()),
        "fields": fields,
    }


def _condition_by_sample(
    samples: np.ndarray, conditions: np.ndarray
) -> tuple[dict[str, str | list[str]], dict[str, list[str]]]:
    table = pd.DataFrame({"sample": samples, "condition": conditions})
    grouped = table.groupby("sample", sort=True)["condition"].agg(
        lambda values: sorted(pd.unique(values).tolist())
    )
    mapping = {
        str(sample): values[0] if len(values) == 1 else values
        for sample, values in grouped.items()
    }
    multiple = {
        str(sample): values for sample, values in grouped.items() if len(values) > 1
    }
    return mapping, multiple


def _compare_conditions(
    new_obs: pd.DataFrame,
    reference_obs: pd.DataFrame,
    reference_positions: np.ndarray,
) -> dict[str, object]:
    required = {
        "new": ("sample_id", "condition"),
        "reference": ("sample", "condition"),
    }
    missing = {
        source: [column for column in columns if column not in obs]
        for source, columns, obs in (
            ("new", required["new"], new_obs),
            ("reference", required["reference"], reference_obs),
        )
    }
    if any(missing.values()):
        return {"pass": False, "missing_columns": missing}

    new_samples = _strings(new_obs["sample_id"])
    reference_samples = _strings(reference_obs["sample"])[reference_positions]
    observed = _strings(new_obs["condition"])
    expected = _strings(reference_obs["condition"])[reference_positions]
    new_mapping, new_multiple = _condition_by_sample(new_samples, observed)
    reference_mapping, reference_multiple = _condition_by_sample(
        reference_samples, expected
    )
    mismatch = observed != expected

    return {
        "pass": not (
            np.any(observed == "")
            or np.any(expected == "")
            or mismatch.any()
            or new_multiple
            or reference_multiple
        ),
        "new_column": "condition",
        "reference_column": "condition",
        "new_missing_or_empty": int(np.sum(observed == "")),
        "reference_missing_or_empty": int(np.sum(expected == "")),
        "mismatched": int(mismatch.sum()),
        "new_condition_by_sample": new_mapping,
        "reference_condition_by_sample": reference_mapping,
        "new_samples_with_multiple_conditions": new_multiple,
        "reference_samples_with_multiple_conditions": reference_multiple,
    }


def _coordinate_result(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
) -> dict[str, object]:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if observed.shape != expected.shape:
        return {
            "pass": False,
            "new_shape": list(observed.shape),
            "reference_shape": list(expected.shape),
        }
    finite = np.isfinite(observed).all(axis=1)
    difference = np.abs(observed - expected)
    mismatched = (~finite) | np.any(difference > atol, axis=1)
    return {
        "pass": not mismatched.any(),
        "nonfinite_rows": int((~finite).sum()),
        "mismatched_rows": int(mismatched.sum()),
        "max_abs_difference": (
            float(np.nanmax(difference)) if difference.size else 0.0
        ),
        "atol": atol,
    }


def _compare_coordinates(
    new_file: h5py.File,
    reference_file: h5py.File,
    new_obs: pd.DataFrame,
    reference_obs: pd.DataFrame,
    reference_positions: np.ndarray,
    *,
    spatial_key: str,
    atol: float,
) -> dict[str, object]:
    coordinate_columns = ("centroid_x", "centroid_y")
    missing_obs_columns = {
        "new": [column for column in coordinate_columns if column not in new_obs],
        "reference": [
            column for column in coordinate_columns if column not in reference_obs
        ],
    }
    spatial_path = f"obsm/{spatial_key}"
    missing_spatial = {
        "new": spatial_path not in new_file,
        "reference": spatial_path not in reference_file,
    }
    if any(missing_obs_columns.values()) or any(missing_spatial.values()):
        return {
            "pass": False,
            "missing_obs_columns": missing_obs_columns,
            "missing_spatial": missing_spatial,
        }

    new_centroids = new_obs.loc[:, coordinate_columns].to_numpy(dtype=np.float64)
    reference_centroids = reference_obs.loc[:, coordinate_columns].to_numpy(
        dtype=np.float64
    )[reference_positions]
    new_spatial = np.asarray(read_elem(_node(new_file, spatial_path)))
    reference_spatial = np.asarray(
        read_elem(_node(reference_file, spatial_path))
    )[reference_positions]

    comparisons = {
        "centroids_vs_reference": _coordinate_result(
            new_centroids, reference_centroids, atol=atol
        ),
        "spatial_vs_reference": _coordinate_result(
            new_spatial, reference_spatial, atol=atol
        ),
        "new_spatial_vs_centroids": _coordinate_result(
            new_spatial, new_centroids, atol=atol
        ),
    }
    return {
        "pass": all(bool(result["pass"]) for result in comparisons.values()),
        **comparisons,
    }


def validate_ab_anndata(
    new_path: Path,
    reference_path: Path = DEFAULT_REFERENCE,
    *,
    expected_samples: int = 12,
    chunk_size: int = 4096,
    coordinate_atol: float = 1e-6,
    new_counts_path: str = "layers/counts",
    reference_counts_path: str = "X",
    spatial_key: str = "spatial",
) -> dict[str, object]:
    """Return a compact validation report without modifying either H5AD."""

    checks: dict[str, object] = {}
    with h5py.File(new_path, "r") as new_file, h5py.File(
        reference_path, "r"
    ) as reference_file:
        new_obs = read_elem(new_file["obs"])
        new_var = read_elem(new_file["var"])
        reference_obs = read_elem(reference_file["obs"])
        reference_var = read_elem(reference_file["var"])

        new_shape = (len(new_obs), len(new_var))
        reference_shape = (len(reference_obs), len(reference_var))
        checks["shape"] = {
            "pass": new_shape == reference_shape,
            "new": list(new_shape),
            "reference": list(reference_shape),
        }

        new_genes = _strings(new_var.index)
        reference_genes = _strings(reference_var.index)
        first_gene_mismatch = _first_gene_mismatch(new_genes, reference_genes)
        checks["gene_order"] = {
            "pass": first_gene_mismatch is None,
            "first_mismatch": first_gene_mismatch,
        }

        identity_columns = {
            "new": [
                column
                for column in ("sample_id", "cell_id")
                if column not in new_obs
            ],
            "reference": [
                column for column in ("sample",) if column not in reference_obs
            ],
        }
        if any(identity_columns.values()):
            checks["samples"] = {
                "pass": False,
                "missing_columns": identity_columns,
            }
            checks["cell_identity"] = {
                "pass": False,
                "missing_columns": identity_columns,
            }
            reference_positions = None
            new_keys = None
        else:
            new_samples = _strings(new_obs["sample_id"])
            new_cell_ids = _strings(new_obs["cell_id"])
            reference_samples = _strings(reference_obs["sample"])
            reference_cell_ids = _reference_cell_ids(reference_obs.index)
            new_counts = _sample_counts(new_samples)
            reference_counts = _sample_counts(reference_samples)
            checks["samples"] = {
                "pass": (
                    len(new_counts) == expected_samples
                    and len(reference_counts) == expected_samples
                    and new_counts == reference_counts
                ),
                "expected_sample_count": expected_samples,
                "new": new_counts,
                "reference": reference_counts,
            }

            new_keys = pd.MultiIndex.from_arrays(
                [new_samples, new_cell_ids], names=("sample_id", "cell_id")
            )
            reference_keys = pd.MultiIndex.from_arrays(
                [reference_samples, reference_cell_ids],
                names=("sample_id", "cell_id"),
            )
            new_unique = new_keys.is_unique
            reference_unique = reference_keys.is_unique
            if new_unique and reference_unique:
                reference_positions = reference_keys.get_indexer(new_keys)
                new_positions = new_keys.get_indexer(reference_keys)
                extra = np.flatnonzero(reference_positions < 0)
                missing = np.flatnonzero(new_positions < 0)
                identity_pass = not extra.size and not missing.size
                checks["cell_identity"] = {
                    "pass": identity_pass,
                    "new_unique": True,
                    "reference_unique": True,
                    "extra_in_new": int(extra.size),
                    "missing_from_new": int(missing.size),
                    "first_extra_in_new": (
                        _key_example(new_keys[int(extra[0])])
                        if extra.size
                        else None
                    ),
                    "first_missing_from_new": (
                        _key_example(reference_keys[int(missing[0])])
                        if missing.size
                        else None
                    ),
                    "same_order": bool(
                        identity_pass
                        and np.array_equal(
                            reference_positions, np.arange(len(new_keys))
                        )
                    ),
                    "legacy_obs_name_equality_required": False,
                }
                if not identity_pass:
                    reference_positions = None
            else:
                reference_positions = None
                checks["cell_identity"] = {
                    "pass": False,
                    "new_unique": bool(new_unique),
                    "reference_unique": bool(reference_unique),
                    "legacy_obs_name_equality_required": False,
                }

        if reference_positions is None:
            checks["conditions"] = {
                "pass": False,
                "skipped": "cell identities could not be aligned",
            }
            checks["metadata"] = {
                "pass": False,
                "skipped": "cell identities could not be aligned",
            }
            checks["coordinates"] = {
                "pass": False,
                "skipped": "cell identities could not be aligned",
            }
            checks["raw_counts"] = {
                "pass": False,
                "skipped": "cell identities could not be aligned",
            }
        else:
            checks["conditions"] = _compare_conditions(
                new_obs, reference_obs, reference_positions
            )
            checks["metadata"] = _compare_labels(
                new_obs, reference_obs, reference_positions
            )
            checks["coordinates"] = _compare_coordinates(
                new_file,
                reference_file,
                new_obs,
                reference_obs,
                reference_positions,
                spatial_key=spatial_key,
                atol=coordinate_atol,
            )
            if not (
                checks["shape"]["pass"] and checks["gene_order"]["pass"]
            ):
                checks["raw_counts"] = {
                    "pass": False,
                    "skipped": "shape or gene order mismatch",
                }
            else:
                try:
                    new_matrix = H5adMatrix(
                        _node(new_file, new_counts_path), new_counts_path
                    )
                    reference_matrix = H5adMatrix(
                        _node(reference_file, reference_counts_path),
                        reference_counts_path,
                    )
                    checks["raw_counts"] = _compare_counts(
                        new_matrix,
                        reference_matrix,
                        reference_positions,
                        new_keys,
                        chunk_size,
                    )
                except (KeyError, ValueError) as error:
                    checks["raw_counts"] = {
                        "pass": False,
                        "error": str(error),
                        "new_source": new_counts_path,
                        "reference_source": reference_counts_path,
                    }

    passed = all(bool(check["pass"]) for check in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "new_h5ad": str(new_path.resolve()),
        "reference_h5ad": str(reference_path.resolve()),
        "checks": checks,
    }


def _write_report(report: dict[str, object], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a converted AB H5AD against the legacy raw-count H5AD. "
            "Inputs are opened read-only and matrices are compared in row chunks."
        )
    )
    parser.add_argument("new_h5ad", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Legacy raw-count H5AD used for parity validation",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--expected-samples", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--coordinate-atol", type=float, default=1e-6)
    parser.add_argument("--new-counts-path", default="layers/counts")
    parser.add_argument("--reference-counts-path", default="X")
    parser.add_argument("--spatial-key", default="spatial")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.expected_samples < 1:
        raise SystemExit("--expected-samples must be positive")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if args.coordinate_atol < 0:
        raise SystemExit("--coordinate-atol must be non-negative")

    try:
        report = validate_ab_anndata(
            args.new_h5ad,
            args.reference,
            expected_samples=args.expected_samples,
            chunk_size=args.chunk_size,
            coordinate_atol=args.coordinate_atol,
            new_counts_path=args.new_counts_path,
            reference_counts_path=args.reference_counts_path,
            spatial_key=args.spatial_key,
        )
    except (OSError, KeyError, ValueError) as error:
        report = {
            "status": "error",
            "new_h5ad": str(args.new_h5ad.resolve()),
            "reference_h5ad": str(args.reference.resolve()),
            "error": f"{type(error).__name__}: {error}",
        }
        _write_report(report, args.report)
        return 2

    _write_report(report, args.report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
