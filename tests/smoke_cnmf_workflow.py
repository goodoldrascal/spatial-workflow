#!/usr/bin/env python3
"""Run a tiny end-to-end cNMF sweep and selected-K consensus in /tmp."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_workflow.cnmf import (  # noqa: E402
    cnmf_status,
    export_lineage_counts,
    load_cnmf_context,
    load_consensus_results,
    load_k_selection_stats,
    run_selected,
    run_sweep,
)


def main() -> None:
    rng = np.random.default_rng(14)
    memberships = np.repeat([0, 1], 40)
    rates = np.full((80, 12), 0.4)
    rates[memberships == 0, :5] += 4
    rates[memberships == 1, 5:10] += 4
    counts = sp.csr_matrix(rng.poisson(rates).astype(np.int32))
    obs = pd.DataFrame(
        {
            "celltype_short": "smoke_type",
            "spatial_domain": np.where(np.arange(80) % 2, "1", "2"),
            "condition": np.where(np.arange(80) < 40, "control", "treated"),
            "sample_id": np.where(np.arange(80) < 40, "s1", "s2"),
        },
        index=[f"cell_{index:03d}" for index in range(80)],
    )
    adata = ad.AnnData(
        X=counts.astype(float),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{index:02d}" for index in range(12)]),
    )
    adata.layers["counts"] = counts
    adata.obsm["spatial"] = rng.normal(size=(80, 2))

    root = Path(tempfile.mkdtemp(prefix="spatial-workflow-cnmf-smoke-"))
    try:
        source_h5ad = root / "source.h5ad"
        adata.write_h5ad(source_h5ad)
        config = {
            "project": {"id": "cnmf_smoke"},
            "paths": {"results_root": str(root / "results")},
            "schema": {
                "counts_layer": "counts",
                "broad_cell_type_key": "celltype_short",
                "spatial_domain_key": "spatial_domain",
                "condition_key": "condition",
                "sample_key": "sample_id",
                "spatial_key": "spatial",
            },
            "nncomp": {"input_h5ad": str(source_h5ad)},
            "runtime": {"cnmf_bin": "cnmf"},
            "cnmf": {
                "input_h5ad": str(source_h5ad),
                "output_dir": "cnmf",
                "preparation": {
                    "min_counts_per_cell": 1,
                    "min_cells_per_gene": 1,
                },
                "lineages": {
                    "smoke": {
                        "cell_types": ["smoke_type"],
                        "exclude_cells": [],
                        "compartments": "all",
                        "conditions": "all",
                        "numgenes": 10,
                        "seed": 14,
                        "max_nmf_iter": 200,
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
        config_path = root / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        loaded_config, spec, paths = load_cnmf_context(config_path, "smoke")
        export_lineage_counts(loaded_config, spec, paths)
        run_sweep(loaded_config, spec, paths)
        run_selected(loaded_config, spec, paths)

        status = cnmf_status(spec, paths)
        if not status["complete"].all():
            raise RuntimeError("Smoke workflow incomplete:\n" + status.to_string())
        k_stats = load_k_selection_stats(paths)
        usage, _, _, _ = load_consensus_results(paths, spec)
        if set(k_stats["k"]) != {2, 3}:
            raise RuntimeError(f"Unexpected K-selection rows:\n{k_stats}")
        if usage.shape != (80, 2):
            raise RuntimeError(
                f"Unexpected consensus usage shape: {usage.shape}"
            )
        print(status.to_string(index=False))
        print(k_stats.to_string(index=False))
        print("usage shape:", usage.shape)
        print("cNMF smoke test passed")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
