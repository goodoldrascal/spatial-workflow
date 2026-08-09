#!/usr/bin/env python3
"""Prepare, sweep, select, and inspect reproducible lineage cNMF runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_workflow.cnmf import (  # noqa: E402
    cnmf_stage_commands,
    cnmf_status,
    export_lineage_counts,
    load_cnmf_context,
    preview_lineage_selection,
    run_selected,
    run_sweep,
    write_run_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.yaml",
    )
    parser.add_argument(
        "--lineage",
        action="append",
        required=True,
        help="Configured lineage name; repeat or pass 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "preview",
            "plan",
            "export",
            "sweep",
            "selected",
            "all",
            "status",
            "commands",
        ],
        default="status",
    )
    parser.add_argument(
        "--analysis-name",
        help="Output analysis name; allowed only with one --lineage.",
    )
    parser.add_argument(
        "--cell-type",
        action="append",
        help="Exact included cell type; repeat to replace the configured lineage list.",
    )
    parser.add_argument(
        "--cell-type-key",
        help="Metadata column used for exact cell-type selection.",
    )
    parser.add_argument(
        "--compartment",
        action="append",
        help="Exact spatial domain; repeat, or pass 'all'.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help="Exact condition; repeat, or pass 'all'.",
    )
    parser.add_argument("--selected-k", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        help="Override both sweep and selected-run worker counts.",
    )
    parser.add_argument(
        "--overwrite-export",
        action="store_true",
        help="Atomically replace only the generated lineage input export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cNMF commands without launching factorization.",
    )
    return parser


def _lineages(config_path: Path, requested: list[str]) -> list[str]:
    from spatial_workflow.config import load_config

    config = load_config(config_path)
    configured = list(config.get("cnmf", {}).get("lineages", {}))
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError("--lineage all cannot be combined with other names")
        return configured
    unknown = [value for value in requested if value not in configured]
    if unknown:
        raise KeyError(
            f"Unknown lineages {unknown}. Available: {configured}"
        )
    return list(dict.fromkeys(requested))


def main() -> None:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    lineages = _lineages(config_path, args.lineage)
    if args.analysis_name and len(lineages) != 1:
        raise ValueError("--analysis-name requires exactly one lineage")
    if len(lineages) > 1 and any(
        value is not None
        for value in (
            args.cell_type,
            args.cell_type_key,
            args.compartment,
            args.condition,
            args.selected_k,
            args.workers,
        )
    ):
        raise ValueError("Overrides require exactly one lineage")
    for lineage in lineages:
        print(f"\n=== {lineage} ===", flush=True)
        overrides = {
            "analysis_name": args.analysis_name,
            "cell_type_key": args.cell_type_key,
            "cell_types": args.cell_type,
            "compartments": args.compartment,
            "conditions": args.condition,
            "selected_k": args.selected_k,
            "workers": args.workers,
        }
        config, spec, paths = load_cnmf_context(
            config_path,
            lineage,
            **overrides,
        )
        if args.mode in {"preview", "plan"}:
            summaries = preview_lineage_selection(config, spec, paths.source_h5ad)
            for name, table in summaries.items():
                print(f"\n{name}")
                print(table.to_string())
        elif args.mode == "commands":
            import shlex

            for stage, command in cnmf_stage_commands(config, spec, paths).items():
                print(f"{stage}:\n  {shlex.join(command)}")
        elif args.mode == "status":
            print(cnmf_status(spec, paths).to_string(index=False))
        else:
            if args.mode in {"export", "all"}:
                manifest = export_lineage_counts(
                    config,
                    spec,
                    paths,
                    overwrite=args.overwrite_export,
                )
                print(
                    "Exported shape:",
                    manifest["output"]["shape"],
                    "->",
                    paths.counts_h5ad,
                )
            if args.mode == "sweep":
                if not paths.counts_h5ad.exists() and not args.dry_run:
                    export_lineage_counts(config, spec, paths)
                run_sweep(config, spec, paths, dry_run=args.dry_run)
            elif args.mode == "selected":
                if not paths.counts_h5ad.exists() and not args.dry_run:
                    export_lineage_counts(config, spec, paths)
                run_selected(config, spec, paths, dry_run=args.dry_run)
            elif args.mode == "all":
                run_sweep(config, spec, paths, dry_run=args.dry_run)
                run_selected(config, spec, paths, dry_run=args.dry_run)
            if not args.dry_run:
                write_run_manifest(config_path, spec, paths)
            print(cnmf_status(spec, paths).to_string(index=False))


if __name__ == "__main__":
    main()
