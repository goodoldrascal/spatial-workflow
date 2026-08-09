#!/usr/bin/env python3
"""Build condition-blind cNMF program diagnostics and a review queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatial_workflow.cnmf_program_review import (
    prepare_blinded_usage_obs,
    program_spatial_knn_enrichment,
    program_usage_diagnostic_metrics,
    select_high_usage_cells,
)


LINEAGE_DIRECTORIES = {
    "astrocyte": "astrocyte",
    "oligodendrocyte": "oligodendrocyte",
    "microglia": "microglia_cluster_sub_all",
    "inhibitory_neuron": "inhibitory_neuron",
    "excitatory_neuron": "excitatory_neuron",
    "perivascular": "perivascular",
    "epd": "epd",
    "chp": "chp",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    cnmf_root = project_root / "results" / "ab_xenium" / "05_cnmf"
    review_root = cnmf_root / "program_review"
    parser = argparse.ArgumentParser(
        description=(
            "Build label-free cNMF usage, section-breadth, raw-gene-support, "
            "technical-correlation, redundancy, and spatial-coherence metrics."
        )
    )
    parser.add_argument("--cnmf-root", type=Path, default=cnmf_root)
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=review_root / "cnmf_program_whitelist_draft.tsv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=review_root / "cnmf_program_review_metrics_condition_blind.tsv",
    )
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=review_root / "cnmf_program_review_queue_draft.tsv",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--spatial-neighbors", type=int, default=10)
    parser.add_argument("--max-background-cells", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def usage_path(cnmf_root: Path, lineage: str) -> tuple[Path, Path, int]:
    if lineage not in LINEAGE_DIRECTORIES:
        raise KeyError(f"No cNMF directory mapping for lineage {lineage!r}")
    manifest_path = cnmf_root / LINEAGE_DIRECTORIES[lineage] / "run_manifest.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    path = Path(manifest["paths"]["usage_h5ad"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, manifest_path, int(manifest["selected"]["k"])


def materialize_gene_subset(
    usage: ad.AnnData,
    genes: list[str],
    *,
    chunk_size: int = 10_000,
) -> tuple[np.ndarray, dict[str, int]]:
    """Read a small gene panel from a backed sparse matrix one row chunk at a time."""

    available = pd.Index(usage.var_names.astype(str))
    retained = [gene for gene in dict.fromkeys(genes) if gene in available]
    if not retained:
        raise KeyError("None of the annotation genes are present")
    gene_rows = available.get_indexer(retained)
    output = np.empty((usage.n_obs, len(retained)), dtype=np.float32)
    for start in range(0, usage.n_obs, int(chunk_size)):
        stop = min(start + int(chunk_size), usage.n_obs)
        block = usage.X[start:stop]
        if sparse.issparse(block):
            block = block[:, gene_rows].toarray()
        else:
            block = np.asarray(block)[:, gene_rows]
        output[start:stop] = np.asarray(block, dtype=np.float32)
    return output, {gene: index for index, gene in enumerate(retained)}


def summarize_gene_support(
    gene_matrix: np.ndarray,
    gene_to_column: dict[str, int],
    review_obs: pd.DataFrame,
    *,
    program: str,
    genes: list[str],
    top_fraction: float,
    max_background_cells: int,
) -> dict:
    high = select_high_usage_cells(
        review_obs,
        program=program,
        top_fraction=top_fraction,
        extra_columns=(),
    )
    review_index = pd.Index(review_obs.index.astype(str))
    high_rows = review_index.get_indexer(pd.Index(high["cell_id"].astype(str)))
    available_genes = [gene for gene in genes if gene in gene_to_column]
    gene_columns = np.asarray(
        [gene_to_column[gene] for gene in available_genes],
        dtype=int,
    )
    background = np.setdiff1d(
        np.arange(len(review_obs), dtype=int),
        np.unique(high_rows),
        assume_unique=False,
    )
    if len(background) > int(max_background_cells):
        rng = np.random.default_rng(14)
        background = np.sort(
            rng.choice(
                background,
                size=int(max_background_cells),
                replace=False,
            )
        )
    high_matrix = gene_matrix[np.unique(high_rows)][:, gene_columns]
    background_matrix = gene_matrix[background][:, gene_columns]
    high_mean = high_matrix.mean(axis=0)
    background_mean = background_matrix.mean(axis=0)
    ratios = np.log2((high_mean + 0.1) / (background_mean + 0.1))
    return {
        "n_annotation_genes_tested": int(len(available_genes)),
        "mean_top_gene_detection_fraction": float((high_matrix > 0).mean()),
        "mean_background_gene_detection_fraction": float(
            (background_matrix > 0).mean()
        ),
        "median_top_gene_log2_mean_count_ratio": float(np.median(ratios)),
        "fraction_top_genes_positive_log2_ratio": float((ratios > 0).mean()),
    }


def metric_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if float(row["largest_section_fraction"]) >= 0.50:
        flags.append("top_cells_concentrated_in_one_section")
    if float(row["effective_top_sections"]) < 3.0:
        flags.append("low_effective_section_breadth")
    technical = max(
        abs(float(row["spearman_nCount_Xenium"])),
        abs(float(row["spearman_nFeature_Xenium"])),
    )
    if np.isfinite(technical) and technical >= 0.30:
        flags.append("usage_correlated_with_cell_qc")
    redundancy = abs(float(row["max_abs_pairwise_spearman"]))
    if np.isfinite(redundancy) and redundancy >= 0.75:
        flags.append("strong_pairwise_usage_redundancy")
    if float(row["sd_usage"]) < 0.01:
        flags.append("very_low_usage_variability")
    spatial = float(row["spatial_knn_enrichment"])
    if np.isfinite(spatial) and spatial >= 2.0:
        flags.append("strong_within_lineage_spatial_clustering")
    if float(row["mean_top_gene_detection_fraction"]) < 0.10:
        flags.append("weak_raw_detection_of_annotation_genes")
    return flags


def build_review_queue(metrics: pd.DataFrame) -> pd.DataFrame:
    queue = metrics.copy()
    flag_values: list[str] = []
    tiers: list[str] = []
    reasons: list[str] = []
    for _, row in queue.iterrows():
        flags = metric_flags(row)
        flag_values.append("|".join(flags))
        retained = str(row["sensitivity_include"]) == "TRUE"
        needs_individual = (
            row["lineage"] == "chp"
            or row["confidence"] != "high"
            or row["decision"] == "include_sensitivity"
            or row["category"] in {"mixed", "uncertain", "cell_cycle"}
            or (retained and bool(flags))
        )
        if needs_individual:
            tiers.append("1_individual_attention")
            reasons.append(
                "new_or_ambiguous_program"
                if row["lineage"] == "chp" or row["confidence"] != "high"
                else "retained_program_with_review_flag"
            )
        elif retained:
            tiers.append("2_retained_high_confidence_confirmation")
            reasons.append("retained_high_confidence_draft")
        else:
            tiers.append("3_batch_contamination_confirmation")
            reasons.append("high_confidence_exclusion_draft")
    queue.insert(0, "review_tier", tiers)
    queue.insert(1, "review_reason", reasons)
    queue.insert(2, "metric_flags", flag_values)
    queue["_lineage_order"] = pd.Categorical(
        queue["lineage"],
        categories=list(LINEAGE_DIRECTORIES),
        ordered=True,
    )
    queue["_program_number"] = queue["program"].str.split("_").str[-1].astype(int)
    queue = queue.sort_values(
        ["review_tier", "_lineage_order", "_program_number"],
        kind="stable",
    ).drop(columns=["_lineage_order", "_program_number"])
    queue.insert(0, "review_order", np.arange(1, len(queue) + 1, dtype=int))
    columns = [
        "review_order",
        "review_tier",
        "review_reason",
        "lineage",
        "program",
        "proposed_label",
        "category",
        "confidence",
        "primary_include",
        "sensitivity_include",
        "decision",
        "metric_flags",
        "sd_usage",
        "iqr_usage",
        "q95_usage",
        "nonzero_fraction",
        "n_top_sections",
        "largest_section_fraction",
        "effective_top_sections",
        "spatial_knn_enrichment",
        "spearman_nCount_Xenium",
        "spearman_nFeature_Xenium",
        "max_abs_pairwise_spearman_program",
        "max_abs_pairwise_spearman",
        "mean_top_gene_detection_fraction",
        "median_top_gene_log2_mean_count_ratio",
        "rationale",
        "top_genes",
    ]
    return queue.loc[:, columns]


def main() -> None:
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("top_fraction must be greater than zero and at most one")
    if args.spatial_neighbors < 1:
        raise ValueError("spatial_neighbors must be positive")
    if not args.whitelist.is_file():
        raise FileNotFoundError(args.whitelist)
    manifest_path = args.manifest or args.metrics_output.with_suffix(
        ".manifest.json"
    )
    whitelist = pd.read_csv(
        args.whitelist,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if whitelist.duplicated(["lineage", "program"]).any():
        raise ValueError("Whitelist contains duplicate lineage-program rows")

    lineage_results: list[pd.DataFrame] = []
    lineage_audits: list[dict] = []
    for lineage in whitelist["lineage"].drop_duplicates():
        lineage_whitelist = whitelist.loc[whitelist["lineage"].eq(lineage)].copy()
        path, source_manifest, selected_k = usage_path(args.cnmf_root, lineage)
        expected_k = set(lineage_whitelist["selected_k"].astype(int))
        if expected_k != {selected_k}:
            raise ValueError(
                f"{lineage}: whitelist K {sorted(expected_k)} != manifest K {selected_k}"
            )
        usage = ad.read_h5ad(path, backed="r")
        try:
            programs = sorted(
                [column for column in usage.obs.columns if column.startswith("Usage_")],
                key=lambda value: int(value.split("_", 1)[1]),
            )
            if set(programs) != set(lineage_whitelist["program"]):
                raise ValueError(f"{lineage}: whitelist and usage programs differ")
            review_obs, _ = prepare_blinded_usage_obs(
                usage.obs,
                sample_key="sample_id",
                usage_cols=programs,
            )
            diagnostics = program_usage_diagnostic_metrics(
                review_obs,
                usage_cols=programs,
                top_fraction=args.top_fraction,
            )
            spatial = program_spatial_knn_enrichment(
                review_obs,
                np.asarray(usage.obsm["spatial"]),
                usage_cols=programs,
                top_fraction=args.top_fraction,
                n_neighbors=args.spatial_neighbors,
            )
            support_rows: list[dict] = []
            whitelist_by_program = lineage_whitelist.set_index("program")
            annotation_genes = [
                gene
                for value in lineage_whitelist["top_genes"]
                for gene in value.split("|")
                if gene
            ]
            gene_matrix, gene_to_column = materialize_gene_subset(
                usage,
                annotation_genes,
            )
            for program in programs:
                genes = [
                    gene
                    for gene in whitelist_by_program.loc[program, "top_genes"].split("|")
                    if gene
                ]
                support_rows.append(
                    {
                        "program": program,
                        **summarize_gene_support(
                            gene_matrix,
                            gene_to_column,
                            review_obs,
                            program=program,
                            genes=genes,
                            top_fraction=args.top_fraction,
                            max_background_cells=args.max_background_cells,
                        ),
                    }
                )
            combined = (
                lineage_whitelist.merge(diagnostics, on="program", validate="one_to_one")
                .merge(spatial, on="program", validate="one_to_one")
                .merge(pd.DataFrame(support_rows), on="program", validate="one_to_one")
            )
            combined["source_usage_h5ad"] = str(path.resolve())
            lineage_results.append(combined)
            lineage_audits.append(
                {
                    "lineage": lineage,
                    "selected_k": selected_k,
                    "n_cells": int(usage.n_obs),
                    "n_programs": len(programs),
                    "n_blinded_sections": int(review_obs["blinded_sample"].nunique()),
                    "usage_h5ad": str(path.resolve()),
                    "run_manifest": str(source_manifest.resolve()),
                }
            )
        finally:
            usage.file.close()

    metrics = pd.concat(lineage_results, ignore_index=True)
    metrics.insert(0, "metric_condition_blind", True)
    queue = build_review_queue(metrics)
    atomic_tsv(metrics, args.metrics_output)
    atomic_tsv(queue, args.queue_output)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at": generated_at,
        "condition_blind": True,
        "whitelist": str(args.whitelist.resolve()),
        "whitelist_sha256": sha256(args.whitelist),
        "metrics_output": str(args.metrics_output.resolve()),
        "queue_output": str(args.queue_output.resolve()),
        "n_programs": int(len(metrics)),
        "n_primary": int(metrics["primary_include"].eq("TRUE").sum()),
        "n_sensitivity": int(metrics["sensitivity_include"].eq("TRUE").sum()),
        "top_fraction": args.top_fraction,
        "spatial_neighbors": args.spatial_neighbors,
        "max_background_cells": args.max_background_cells,
        "safe_fields_accessed": [
            "sample_id (converted immediately to blinded_sample)",
            "cluster_sub",
            "celltype_short",
            "celltype_full",
            "spatial_domain",
            "nCount_Xenium",
            "nFeature_Xenium",
            "Usage_k",
            "spatial coordinates",
            "raw count matrix",
        ],
        "forbidden_fields_not_accessed": [
            "condition",
            "exposure_group",
            "treatment_group",
            "group",
        ],
        "interpretation": {
            "usage_variability": (
                "Low variance can be uninformative, but model features are later "
                "standardized within training folds."
            ),
            "section_breadth": (
                "Concentration in one blinded section is a generalization warning, "
                "not an automatic exclusion."
            ),
            "raw_gene_support": (
                "Top-gene support is circular because cNMF was learned from these "
                "counts; use it as a QC check, not independent validation."
            ),
            "spatial_knn_enrichment": (
                "Values above one indicate local clustering within the source "
                "lineage; clustering can reflect biology or spatial spillover."
            ),
            "technical_correlation": (
                "Large correlations with detected counts or genes can indicate a "
                "technical or cell-quality axis."
            ),
            "pairwise_redundancy": (
                "Large absolute within-lineage Spearman correlation indicates "
                "redundant programs; cNMF closure can also induce anticorrelation."
            ),
        },
        "lineage_audits": lineage_audits,
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
