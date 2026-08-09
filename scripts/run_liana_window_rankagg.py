#!/usr/bin/env python3
"""Run or analyze whole-sample, non-graph adaptive-window LIANA rankagg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_workflow.config import load_config, resolve_path
from spatial_workflow.liana_window import (
    WindowParameters,
    analyze_window_rankagg,
    liana_window_status,
    run_window_rankagg,
)


def default_paths(config_path: Path) -> tuple[dict, Path, Path]:
    config = load_config(config_path)
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    section = config.get("liana_window", {})
    if section.get("input_h5ad"):
        input_h5ad = resolve_path(config_path, section["input_h5ad"], root=results_root)
    else:
        cellcharter = config["cellcharter"]
        input_h5ad = resolve_path(
            config_path,
            Path(cellcharter["output_dir"]) / f"{cellcharter['output_name']}.h5ad",
            root=results_root,
        )
    output_dir = resolve_path(
        config_path,
        section.get("output_dir", "07_liana_window_rankagg"),
        root=results_root,
    )
    return config, input_h5ad, output_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=REPO_ROOT / "configs/local.yaml")
    subparsers = result.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run/resume sample-level LIANA windows")
    run.add_argument("--input-h5ad", type=Path)
    run.add_argument("--cellchat-csv", type=Path)
    run.add_argument(
        "--resource-mode", choices=["cellchat", "custom", "liana"], default="cellchat"
    )
    run.add_argument("--custom-lr-csv", type=Path)
    run.add_argument("--liana-resource-name", default="mouseconsensus")
    run.add_argument("--cell-type", action="append", default=[])
    run.add_argument("--compartment", action="append", default=[])
    run.add_argument("--pathway", action="append", default=[])
    run.add_argument("--annotation", action="append", default=[])
    run.add_argument(
        "--include-non-protein",
        action="store_true",
        help="Include CellChat v2 Non-protein Signaling (excluded by default)",
    )
    run.add_argument(
        "--lr-pair", action="append", default=[], help="Exact ligand^receptor pair"
    )
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--sample", action="append", default=[])
    run.add_argument("--adaptive-k", type=int, default=60)
    run.add_argument("--grid-stride", type=float, default=100.0)
    run.add_argument("--expr-prop", type=float, default=0.1)
    run.add_argument("--min-cells", type=int, default=5)
    run.add_argument("--min-required-groups", type=int, default=2)
    run.add_argument("--spatial-bandwidth", type=float, default=250.0)
    run.add_argument(
        "--spatial-kernel",
        choices=["gaussian", "exponential", "linear", "misty_rbf"],
        default="gaussian",
    )
    run.add_argument("--spatial-trim-fraction", type=float, default=0.1)
    run.add_argument("--n-perms", type=int, default=200)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--n-jobs", type=int, default=8)
    run.add_argument("--return-all-lrs", action="store_true")
    run.add_argument("--anchor-group", action="append", default=[])
    run.add_argument("--anchor-k-target", type=int, default=60)
    run.add_argument("--anchor-min-cells", type=int, default=2)
    run.add_argument("--anchor-max-radius", type=float, default=150.0)
    run.add_argument("--anchor-dedup-distance", type=float, default=60.0)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--limit-windows-per-sample", type=int)

    analyze = subparsers.add_parser("analyze", help="Build pairwise/pathway tables")
    analyze.add_argument(
        "--cellchat-csv",
        type=Path,
        required=True,
        help="CellChat or custom annotated LR CSV used for pathway mapping",
    )
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument("--analysis-dir", type=Path)
    analyze.add_argument("--condition", action="append", default=[])
    analyze.add_argument("--min-condition-samples", type=int, default=3)
    analyze.add_argument("--rank-column", default="magnitude_rank")
    analyze.add_argument(
        "--zero-fill-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill missing retained ranks with 1/activity 0 (default: true)",
    )
    analyze.add_argument(
        "--include-non-protein",
        action="store_true",
        help="Include CellChat v2 Non-protein Signaling metadata",
    )
    analyze.add_argument("--overwrite", action="store_true")

    status = subparsers.add_parser("status", help="Inspect completed sample artifacts")
    status.add_argument("--output-dir", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    config, default_input, default_output = default_paths(args.config)
    schema = config["schema"]
    output_dir = (args.output_dir or default_output).resolve()

    if args.command == "status":
        table = liana_window_status(output_dir)
        print(table.to_string(index=False) if not table.empty else "No completed samples")
        return

    if args.command == "run":
        parameters = WindowParameters(
            adaptive_k=args.adaptive_k,
            grid_stride=args.grid_stride,
            expr_prop=args.expr_prop,
            min_cells=args.min_cells,
            min_required_groups=args.min_required_groups,
            spatial_bandwidth=args.spatial_bandwidth,
            spatial_kernel=args.spatial_kernel,
            spatial_trim_fraction=args.spatial_trim_fraction,
            n_perms=args.n_perms,
            seed=args.seed,
            n_jobs=args.n_jobs,
            return_all_lrs=args.return_all_lrs,
            anchor_groups=tuple(args.anchor_group),
            anchor_k_target=args.anchor_k_target,
            anchor_min_cells=args.anchor_min_cells,
            anchor_max_radius=args.anchor_max_radius,
            anchor_dedup_distance=args.anchor_dedup_distance,
        )
        status = run_window_rankagg(
            args.input_h5ad or default_input,
            args.cellchat_csv,
            output_dir,
            sample_key=schema["sample_key"],
            condition_key=schema["condition_key"],
            group_key=schema["cell_type_key"],
            spatial_key=schema["spatial_key"],
            compartment_key=schema.get("spatial_domain_key"),
            cell_types=args.cell_type or None,
            compartments=args.compartment or None,
            pathways=args.pathway or None,
            annotations=args.annotation or None,
            lr_pairs=args.lr_pair or None,
            resource_mode=args.resource_mode,
            custom_lr_csv=args.custom_lr_csv,
            liana_resource_name=args.liana_resource_name,
            include_non_protein=args.include_non_protein,
            parameters=parameters,
            samples=args.sample or None,
            overwrite=args.overwrite,
            limit_windows_per_sample=args.limit_windows_per_sample,
        )
        print(status.to_string(index=False))
        return

    tables = analyze_window_rankagg(
        output_dir,
        args.cellchat_csv,
        analysis_dir=args.analysis_dir,
        conditions=args.condition or None,
        min_condition_samples=args.min_condition_samples,
        rank_column=args.rank_column,
        zero_fill_missing=args.zero_fill_missing,
        include_non_protein=args.include_non_protein,
        overwrite=args.overwrite,
    )
    print(
        "\n".join(f"{name}: {table.shape}" for name, table in tables.items())
    )


if __name__ == "__main__":
    main()
