"""Run or resume the local CellCharter and spatial-nncomp overview stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cellcharter import run_cellcharter
from .config import load_config, resolve_path
from .nncomp import (
    ABUNDANCE_FILE,
    MANIFEST_FILE,
    NEIGHBOR_FILE,
    OVERLAP_FILE,
    PERMUTATION_FILE,
    PERMUTATION_PLAN_FILE,
    RANKED_EDGES_FILE,
    RECIPROCAL_CELLTYPE_FILE,
    SUMMARY_FILE,
    WHOLE_SAMPLE_CELLTYPE_FILE,
    WHOLE_SAMPLE_PERMUTATION_PLAN_FILE,
    run_directional_overlap,
    run_nncomp,
    run_permutation_plan,
    run_reciprocal_celltype_permutation,
)


def _classify(paths: dict[str, Path], stage: str) -> str:
    present = [name for name, path in paths.items() if path.exists()]
    if not present:
        return "missing"
    if len(present) == len(paths):
        return "complete"
    missing = [name for name, path in paths.items() if not path.exists()]
    raise RuntimeError(
        f"Refusing to continue from partial {stage} outputs; "
        f"present={present}, missing={missing}"
    )


def _overview_paths(config_path: Path, config: dict[str, Any]) -> tuple[
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
]:
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    cellcharter = config["cellcharter"]
    stem = str(cellcharter["output_name"]).removesuffix(".h5ad")
    cellcharter_dir = resolve_path(
        config_path, cellcharter["output_dir"], root=results_root
    )
    cellcharter_paths = {
        "h5ad": cellcharter_dir / f"{stem}.h5ad",
        "summary": cellcharter_dir / f"{stem}.summary.json",
        "abundance": cellcharter_dir / f"{stem}.domain_abundance.csv",
        "stability": cellcharter_dir / f"{stem}.autok_stability.csv",
    }

    nncomp_dir = resolve_path(
        config_path, config["nncomp"]["output_dir"], root=results_root
    )
    nncomp_paths = {
        "compartment_abundance": nncomp_dir / ABUNDANCE_FILE,
        "neighbor_composition": nncomp_dir / NEIGHBOR_FILE,
        "neighbor_composition_summary": nncomp_dir / SUMMARY_FILE,
        "manifest": nncomp_dir / MANIFEST_FILE,
    }
    overlap_paths = {
        "ranked_neighbor_edges": nncomp_dir / RANKED_EDGES_FILE,
        "directional_knn_overlap": nncomp_dir / OVERLAP_FILE,
        "directional_knn_overlap_permutation": nncomp_dir / PERMUTATION_FILE,
    }
    reciprocal_settings = (
        config["nncomp"].get("overlap", {}).get("reciprocal_cell_types", {})
    )
    reciprocal_analyses = tuple(
        reciprocal_settings.get("analyses", ["within_compartment"])
    )
    reciprocal_paths = {}
    if "within_compartment" in reciprocal_analyses:
        reciprocal_paths["within_compartment_celltype_permutation"] = (
            nncomp_dir / RECIPROCAL_CELLTYPE_FILE
        )
    if "whole_sample" in reciprocal_analyses:
        reciprocal_paths["whole_sample_celltype_permutation"] = (
            nncomp_dir / WHOLE_SAMPLE_CELLTYPE_FILE
        )
        reciprocal_paths["whole_sample_label_permutation_plan"] = (
            nncomp_dir / WHOLE_SAMPLE_PERMUTATION_PLAN_FILE
        )
    permutation_plan_paths = {
        "label_permutation_plan": nncomp_dir / PERMUTATION_PLAN_FILE
    }
    return (
        cellcharter_paths,
        nncomp_paths,
        overlap_paths,
        permutation_plan_paths,
        reciprocal_paths,
    )


def run_spatial_overview(config_path: str | Path) -> dict[str, Any]:
    """Run missing stages and leave complete local stage outputs unchanged."""

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    (
        cellcharter_paths,
        nncomp_paths,
        overlap_paths,
        permutation_plan_paths,
        reciprocal_paths,
    ) = _overview_paths(config_file, config)
    actions: list[str] = []

    if _classify(cellcharter_paths, "CellCharter") == "missing":
        run_cellcharter(config_file)
        actions.append("ran_cellcharter")
    else:
        actions.append("used_existing_cellcharter")

    nncomp_state = _classify(nncomp_paths, "nncomp")
    overlap_enabled = bool(config["nncomp"].get("overlap", {}).get("enabled", False))
    reciprocal_enabled = overlap_enabled and bool(
        config["nncomp"]
        .get("overlap", {})
        .get("reciprocal_cell_types", {})
        .get("enabled", False)
    )
    overlap_state = _classify(overlap_paths, "directional overlap")
    reciprocal_complete = all(path.exists() for path in reciprocal_paths.values())
    reciprocal_present = any(path.exists() for path in reciprocal_paths.values())
    if nncomp_state == "missing":
        if overlap_state != "missing" or reciprocal_present:
            raise RuntimeError(
                "Directional-overlap outputs exist without the base nncomp contract"
            )
        run_nncomp(config_file)
        actions.append("ran_nncomp")
    else:
        actions.append("used_existing_nncomp")
        if overlap_enabled and overlap_state == "missing":
            run_directional_overlap(config_file)
            actions.append("ran_directional_overlap")
        elif overlap_enabled:
            actions.append("used_existing_directional_overlap")
        if overlap_enabled and not next(iter(permutation_plan_paths.values())).exists():
            run_permutation_plan(config_file)
            actions.append("ran_permutation_plan")
        elif overlap_enabled:
            actions.append("used_existing_permutation_plan")
        if reciprocal_enabled and not reciprocal_complete:
            run_reciprocal_celltype_permutation(config_file)
            actions.append("ran_reciprocal_celltype_permutation")
        elif reciprocal_enabled:
            actions.append("used_existing_reciprocal_celltype_permutation")

    return {
        "actions": actions,
        "cellcharter": cellcharter_paths,
        "nncomp": nncomp_paths,
        "directional_overlap": overlap_paths if overlap_enabled else {},
        "permutation_plan": (permutation_plan_paths if overlap_enabled else {}),
        "reciprocal_celltypes": (reciprocal_paths if reciprocal_enabled else {}),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_spatial_overview(args.config)
    print("actions: " + ", ".join(result["actions"]))
    for stage in (
        "cellcharter",
        "nncomp",
        "directional_overlap",
        "permutation_plan",
        "reciprocal_celltypes",
    ):
        for name, path in result[stage].items():
            print(f"{stage}.{name}: {path}")


if __name__ == "__main__":
    main()
