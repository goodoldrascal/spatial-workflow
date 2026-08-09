#!/usr/bin/env python3
"""Migrate the accepted legacy EPD cNMF run into the current workflow contract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnmf import cNMF, load_df_from_npz, save_df_to_npz


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_workflow.cnmf import (  # noqa: E402
    _ensure_collection_manifest,
    cnmf_selection_keys,
    cnmf_status,
    export_lineage_counts,
    load_cnmf_context,
    load_consensus_results,
    selection_mask,
    write_cnmf_tmux_scripts,
    write_run_manifest,
    write_usage_h5ad,
)


DEFAULT_LEGACY_ROOT = Path(
    "/stor/scratch/WCAAR/rhyan_scratch/liana/cnmf/runs/"
    "epd_cluster_sub_cnmf_run"
)
LEGACY_SWEEP_NAME = "epd_cluster_sub_all_conditions"
LEGACY_SELECTED_NAME = "epd_cluster_sub_k5_10_niter100"
LEGACY_ID_TOKEN = "_subclusters:"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.yaml",
    )
    parser.add_argument("--lineage", default="epd")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    return parser


def _canonical_ids(values: pd.Index | list[str]) -> pd.Index:
    original = pd.Index(values).astype(str)
    if not original.str.contains(LEGACY_ID_TOKEN, regex=False).all():
        raise ValueError(
            "Every legacy EPD cell ID must contain "
            f"{LEGACY_ID_TOKEN!r}"
        )
    canonical = original.str.replace(LEGACY_ID_TOKEN, ":", regex=False)
    if not canonical.is_unique:
        raise ValueError("Canonicalized EPD cell IDs are not unique")
    return canonical


def _as_csr(matrix):
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)


def _validate_legacy_counts(config, spec, paths, legacy_counts: Path):
    keys = cnmf_selection_keys(config)
    source = ad.read_h5ad(paths.source_h5ad, backed="r")
    mask = selection_mask(
        source.obs,
        cell_type_key=keys["cell_type_key"],
        cell_types=spec.cell_types,
        compartment_key=keys["compartment_key"],
        compartments=spec.compartments,
        condition_key=keys["condition_key"],
        conditions=spec.conditions,
        exclude_cells=spec.exclude_cells,
    )
    indices = np.flatnonzero(mask)
    accepted_ids = pd.Index(source.obs_names[indices]).astype(str)
    accepted_genes = pd.Index(source.var_names).astype(str)

    legacy = ad.read_h5ad(legacy_counts)
    canonical_ids = _canonical_ids(legacy.obs_names)
    if not canonical_ids.equals(accepted_ids):
        mismatch = np.flatnonzero(canonical_ids.to_numpy() != accepted_ids.to_numpy())
        raise RuntimeError(
            "Canonical legacy IDs do not match the accepted EPD selection: "
            f"mismatches={len(mismatch)}"
        )
    if not pd.Index(legacy.var_names).astype(str).equals(accepted_genes):
        raise RuntimeError("Legacy and accepted EPD gene order differs")

    accepted_counts = _as_csr(source.layers[keys["counts_layer"]][indices, :])
    legacy_counts_matrix = _as_csr(legacy.X)
    difference = legacy_counts_matrix - accepted_counts
    if difference.nnz:
        raise RuntimeError(
            "Legacy and accepted EPD raw counts differ: "
            f"nonzero_differences={difference.nnz}"
        )
    return accepted_ids, {
        "cells": int(legacy.n_obs),
        "genes": int(legacy.n_vars),
        "raw_counts_equal": True,
        "canonical_cell_ids_match": True,
    }


def _target_relative_path(relative: Path, old_name: str, new_name: str) -> Path:
    return Path(*(part.replace(old_name, new_name) for part in relative.parts))


def _copy_renamed_tree(
    source_root: Path,
    target_root: Path,
    *,
    old_name: str,
    new_name: str,
    include: Callable[[Path], bool] | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or (include is not None and not include(source)):
            continue
        relative = _target_relative_path(
            source.relative_to(source_root), old_name, new_name
        )
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _selected_file_filter(path: Path, selected_k: int) -> bool:
    name = path.name
    common_suffixes = (
        ".nmf_idvrun_params.yaml",
        ".norm_counts.h5ad",
        ".tpm.h5ad",
        ".tpm_stats.df.npz",
        ".overdispersed_genes.txt",
    )
    if name.endswith(".nmf_params.df.npz"):
        return False
    if name.endswith(common_suffixes):
        return True
    return f".k_{selected_k}." in name


def _rewrite_h5ad_ids(path: Path, accepted_ids: pd.Index) -> None:
    data = ad.read_h5ad(path)
    canonical = _canonical_ids(data.obs_names)
    if not canonical.equals(accepted_ids):
        raise RuntimeError(f"Unexpected cell order in migrated H5AD: {path}")
    data.obs_names = canonical
    temporary = path.with_name(f".{path.stem}.canonicalized.h5ad")
    try:
        data.write_h5ad(temporary, compression="gzip")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rewrite_usage_npz(path: Path, accepted_ids: pd.Index) -> None:
    usage = load_df_from_npz(path)
    canonical = _canonical_ids(usage.index)
    if not canonical.equals(accepted_ids):
        raise RuntimeError(f"Unexpected cell order in migrated usage table: {path}")
    usage.index = canonical
    save_df_to_npz(usage, path)


def _rewrite_usage_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if LEGACY_ID_TOKEN not in text:
        raise RuntimeError(f"Legacy cell-ID token is absent from {path}")
    path.write_text(text.replace(LEGACY_ID_TOKEN, ":"), encoding="utf-8")


def _write_selected_params(
    legacy_root: Path,
    target_path: Path,
    *,
    selected_k: int,
) -> None:
    source_path = (
        legacy_root
        / LEGACY_SELECTED_NAME
        / "cnmf_tmp"
        / f"{LEGACY_SELECTED_NAME}.nmf_params.df.npz"
    )
    params = load_df_from_npz(source_path)
    selected = (
        params.loc[params["n_components"].astype(int) == int(selected_k)]
        .reset_index(drop=True)
    )
    if len(selected) != 100 or selected["iter"].astype(int).tolist() != list(range(100)):
        raise RuntimeError("Legacy selected-K parameter table is not K=10 x 100")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_df_to_npz(selected, target_path)


def _validate_numeric_results(legacy_root: Path, paths, spec) -> dict[str, bool]:
    legacy = cNMF(output_dir=str(legacy_root), name=LEGACY_SELECTED_NAME)
    old_usage, old_scores, old_tpm, old_top = legacy.load_results(
        K=spec.selected_k,
        density_threshold=spec.density_threshold,
    )
    new_usage, new_scores, new_tpm, new_top = load_consensus_results(paths, spec)

    old_usage.index = _canonical_ids(old_usage.index)
    new_usage.columns = old_usage.columns
    checks = {
        "usage_values_equal": np.array_equal(
            old_usage.to_numpy(), new_usage.to_numpy()
        ),
        "usage_ids_equal": old_usage.index.equals(new_usage.index),
        "gene_scores_equal": np.array_equal(
            old_scores.to_numpy(), new_scores.to_numpy()
        ),
        "gene_tpm_equal": np.array_equal(old_tpm.to_numpy(), new_tpm.to_numpy()),
        "top_genes_equal": np.array_equal(
            old_top.to_numpy(), new_top.to_numpy()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Migrated EPD result validation failed: " + ", ".join(failed)
        )
    return checks


def main() -> None:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    legacy_root = args.legacy_root.expanduser().resolve()
    config, spec, paths = load_cnmf_context(config_path, args.lineage)

    if paths.lineage_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing EPD workflow tree: "
            f"{paths.lineage_root}"
        )
    if spec.selected_k != 10 or spec.selected_n_iter != 100:
        raise ValueError("The accepted legacy migration requires selected K=10, i100")
    if spec.sweep_k_values != tuple(range(4, 13)) or spec.sweep_n_iter != 10:
        raise ValueError("The accepted legacy sweep requires K=4..12, i10")

    legacy_counts = legacy_root / "epd_cluster_sub_counts.h5ad"
    legacy_sweep = legacy_root / LEGACY_SWEEP_NAME
    legacy_selected = legacy_root / LEGACY_SELECTED_NAME
    for required in (legacy_counts, legacy_sweep, legacy_selected):
        if not required.exists():
            raise FileNotFoundError(required)

    accepted_ids, count_validation = _validate_legacy_counts(
        config, spec, paths, legacy_counts
    )
    export_manifest = export_lineage_counts(config, spec, paths)
    exported = ad.read_h5ad(paths.counts_h5ad)
    if not pd.Index(exported.obs_names).astype(str).equals(accepted_ids):
        raise RuntimeError("Standardized EPD export has an unexpected cell order")
    legacy_export = ad.read_h5ad(legacy_counts)
    exported_difference = _as_csr(exported.X) - _as_csr(legacy_export.X)
    if exported_difference.nnz:
        raise RuntimeError(
            "Standardized EPD export differs from legacy raw counts: "
            f"nonzero_differences={exported_difference.nnz}"
        )

    _ensure_collection_manifest(
        spec, paths, collection="sweep", dry_run=False
    )
    _ensure_collection_manifest(
        spec, paths, collection="selected", dry_run=False
    )

    sweep_files = _copy_renamed_tree(
        legacy_sweep,
        paths.sweep_run_dir,
        old_name=LEGACY_SWEEP_NAME,
        new_name=paths.sweep_name,
    )
    selected_files = _copy_renamed_tree(
        legacy_selected,
        paths.selected_run_dir,
        old_name=LEGACY_SELECTED_NAME,
        new_name=paths.selected_name,
        include=lambda path: _selected_file_filter(path, spec.selected_k),
    )

    selected_params = (
        paths.selected_run_dir
        / "cnmf_tmp"
        / f"{paths.selected_name}.nmf_params.df.npz"
    )
    _write_selected_params(
        legacy_root, selected_params, selected_k=spec.selected_k
    )

    for run_dir, name in (
        (paths.sweep_run_dir, paths.sweep_name),
        (paths.selected_run_dir, paths.selected_name),
    ):
        for suffix in (".norm_counts.h5ad", ".tpm.h5ad"):
            _rewrite_h5ad_ids(run_dir / "cnmf_tmp" / f"{name}{suffix}", accepted_ids)

    usage_npz = (
        paths.selected_run_dir
        / "cnmf_tmp"
        / (
            f"{paths.selected_name}.usages.k_{spec.selected_k}."
            "dt_0_1.consensus.df.npz"
        )
    )
    usage_text = paths.selected_run_dir / (
        f"{paths.selected_name}.usages.k_{spec.selected_k}."
        "dt_0_1.consensus.txt"
    )
    _rewrite_usage_npz(usage_npz, accepted_ids)
    _rewrite_usage_text(usage_text)

    usage_h5ad = write_usage_h5ad(paths, spec)
    write_cnmf_tmux_scripts(
        config_path, args.lineage, config, spec, paths, mode="sweep"
    )
    write_cnmf_tmux_scripts(
        config_path, args.lineage, config, spec, paths, mode="selected"
    )
    write_run_manifest(config_path, spec, paths)

    numeric_validation = _validate_numeric_results(legacy_root, paths, spec)
    status = cnmf_status(spec, paths)
    if not status["complete"].all():
        incomplete = status.loc[~status["complete"], "stage"].tolist()
        raise RuntimeError(
            "Migrated workflow is incomplete: " + ", ".join(incomplete)
        )

    migration_manifest = {
        "stage": "legacy_cnmf_migration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "legacy_root": str(legacy_root),
            "counts_h5ad": str(legacy_counts),
            "sweep_name": LEGACY_SWEEP_NAME,
            "selected_name": LEGACY_SELECTED_NAME,
            "selected_collection_note": (
                "K=10 was extracted from the legacy joint K=5/K=10 i100 "
                "collection; its original replicate seeds were preserved."
            ),
        },
        "target": {
            "lineage_root": str(paths.lineage_root),
            "sweep_name": paths.sweep_name,
            "selected_name": paths.selected_name,
            "usage_h5ad": str(usage_h5ad),
        },
        "transforms": {
            "cell_id_rewrite": "sample_subclusters:barcode -> sample:barcode",
            "factor_values_recomputed": False,
            "selected_k": spec.selected_k,
            "selected_n_iter": spec.selected_n_iter,
            "sweep_k_values": list(spec.sweep_k_values),
            "sweep_n_iter": spec.sweep_n_iter,
            "copied_sweep_files": len(sweep_files),
            "copied_selected_files": len(selected_files),
        },
        "validation": {
            **count_validation,
            **numeric_validation,
            "export_shape": export_manifest["output"]["shape"],
            "all_pipeline_stages_complete": True,
        },
        "status": status.to_dict(orient="records"),
    }
    manifest_path = paths.lineage_root / "migration_manifest.json"
    manifest_path.write_text(
        json.dumps(migration_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(migration_manifest["validation"], indent=2, sort_keys=True))
    print(f"Migrated EPD cNMF workflow: {paths.lineage_root}")


if __name__ == "__main__":
    main()
