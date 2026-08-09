#!/usr/bin/env python3
"""Build Notebook 04 colocalization tables from accepted nncomp outputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_workflow.colocalization import (  # noqa: E402
    build_colocalization_tables,
    write_colocalization_outputs,
)
from spatial_workflow.config import load_config, resolve_path  # noqa: E402
from spatial_workflow.nncomp import (  # noqa: E402
    RECIPROCAL_CELLTYPE_FILE,
    WHOLE_SAMPLE_CELLTYPE_FILE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build within-condition and between-condition colocalization "
            "tables outside the review notebook."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.yaml",
        help="Workflow YAML (default: configs/local.yaml).",
    )
    parser.add_argument(
        "--analysis-scope",
        choices=("within_compartment", "whole_sample"),
        help="Override colocalization.analysis_scope.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override colocalization.output_dir. Relative paths resolve under "
            "paths.results_root."
        ),
    )
    parser.add_argument(
        "--contrast",
        action="append",
        nargs=2,
        metavar=("NUMERATOR", "DENOMINATOR"),
        help="Condition contrast; repeat to override configured contrasts.",
    )
    parser.add_argument("--k", type=int, help="Override colocalization.k.")
    parser.add_argument(
        "--min-cells-a",
        type=int,
        help="Minimum cell-type-A count in every eligible sample.",
    )
    parser.add_argument(
        "--min-cells-b",
        type=int,
        help="Minimum cell-type-B count in every eligible sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing generated tables.",
    )
    return parser


def _configured_contrasts(
    settings: dict[str, Any],
    override: list[list[str]] | None,
) -> list[tuple[str, str]]:
    if override:
        return [
            (str(numerator), str(denominator)) for numerator, denominator in override
        ]
    configured = settings.get("contrasts", [])
    contrasts = []
    for index, entry in enumerate(configured):
        if not isinstance(entry, dict):
            raise ValueError(f"colocalization.contrasts[{index}] must be a mapping")
        try:
            contrasts.append((str(entry["numerator"]), str(entry["denominator"])))
        except KeyError as exc:
            raise ValueError(
                f"colocalization.contrasts[{index}] requires numerator and denominator"
            ) from exc
    if not contrasts:
        raise ValueError(
            "No contrasts configured; add colocalization.contrasts or pass --contrast"
        )
    return contrasts


def main() -> None:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    settings = config.get("colocalization", {})
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'colocalization' must be a mapping")
    review = config.get("review", {})

    results_root = resolve_path(config_path, config["paths"]["results_root"])
    nncomp_dir = resolve_path(
        config_path,
        config["nncomp"]["output_dir"],
        root=results_root,
    )
    analysis_scope = str(
        args.analysis_scope or settings.get("analysis_scope", "within_compartment")
    )
    input_filename = {
        "within_compartment": RECIPROCAL_CELLTYPE_FILE,
        "whole_sample": WHOLE_SAMPLE_CELLTYPE_FILE,
    }[analysis_scope]
    input_path = nncomp_dir / input_filename
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run the all-cell-type nncomp stage first."
        )

    output_value = args.output_dir or settings.get(
        "output_dir", "04_colocalization_analysis"
    )
    output_root = resolve_path(config_path, output_value, root=results_root)
    output_dir = output_root / analysis_scope
    k_value = int(
        args.k if args.k is not None else settings.get("k", review.get("overlap_k", 2))
    )
    min_cells_a = int(
        args.min_cells_a
        if args.min_cells_a is not None
        else settings.get("min_cells_a", review.get("min_overlap_cells_a", 10))
    )
    min_cells_b = int(
        args.min_cells_b
        if args.min_cells_b is not None
        else settings.get("min_cells_b", review.get("min_overlap_cells_b", 10))
    )
    expected_samples = int(settings.get("expected_samples", 4))
    contrasts = _configured_contrasts(settings, args.contrast)

    import pandas as pd

    sample_table = pd.read_parquet(input_path)
    tables = build_colocalization_tables(
        sample_table,
        k_value=k_value,
        min_cells_a=min_cells_a,
        min_cells_b=min_cells_b,
        contrasts=contrasts,
        expected_samples=expected_samples,
        analysis_scope=analysis_scope,
    )
    input_stat = input_path.stat()
    manifest = {
        "stage": "colocalization_analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "input": {
            "path": str(input_path.resolve()),
            "size_bytes": int(input_stat.st_size),
            "mtime_ns": int(input_stat.st_mtime_ns),
            "rows": int(len(sample_table)),
        },
        "configuration": {
            "analysis_scope": analysis_scope,
            "k": k_value,
            "min_cells_a": min_cells_a,
            "min_cells_b": min_cells_b,
            "expected_samples": expected_samples,
            "contrasts": [
                {"numerator": numerator, "denominator": denominator}
                for numerator, denominator in contrasts
            ],
        },
        "contrast_colocalization_filter": {
            "metric_directional": "max_condition_mean_coefficient",
            "metric_reciprocal": ("min_direction_max_condition_mean_coefficient"),
            "note": (
                "The complete tables are unfiltered. Notebook review applies an "
                "adjustable observed-coefficient floor after loading."
            ),
        },
    }
    write_colocalization_outputs(
        output_dir,
        tables,
        manifest=manifest,
        overwrite=args.overwrite,
    )

    print(f"Saved colocalization tables to {output_dir}")
    print(tables["contrast_support"].to_string(index=False))


if __name__ == "__main__":
    main()
