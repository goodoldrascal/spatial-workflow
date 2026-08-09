#!/usr/bin/env python3
"""Build the condition-blind cNMF program whitelist review notebook."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import nbformat as nbf

from notebook_presentation import apply_notebook_presentation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "notebooks" / "08_cnmf_program_whitelist_review.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        markdown(
            """
# Condition-blind cNMF program whitelist review

This notebook is the biological-QC gate before the cNMF neighborhood feature
whitelist is frozen. It asks whether each program is a coherent state of its
source lineage and whether its spatial pattern looks biological rather than
contamination, segmentation spillover, a section edge, or a single-sample
artifact.

The review is deliberately blind to experimental condition:

- real sample IDs are replaced by deterministic Section_XX aliases;
- condition, exposure, treatment, and group columns never enter review tables;
- no condition comparison, p-value, classifier, or feature importance is run;
- decisions use genes, high-usage cells, spatial localization, and neighboring
  cell identity only.

The draft whitelist is not frozen by opening or executing this notebook.
Saving a human decision requires the explicit SAVE_REVIEW_DECISION=True switch
in the final section.
"""
        ),
        code(
            """
from pathlib import Path
import hashlib
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "src" / "spatial_workflow").is_dir():
    candidate = REPO_ROOT.parent
    if (candidate / "src" / "spatial_workflow").is_dir():
        REPO_ROOT = candidate
    else:
        raise RuntimeError("Start Jupyter from the spatial-workflow repository")
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial_workflow.cnmf import usage_columns
from spatial_workflow.cnmf_program_review import (
    build_review_decision,
    high_usage_gene_support,
    high_usage_neighbor_context,
    plot_blinded_spatial_usage,
    prepare_blinded_usage_obs,
    program_usage_qc_summary,
    select_high_usage_cells,
    upsert_review_decision,
)

CNMF_ROOT = REPO_ROOT / "results" / "ab_xenium" / "05_cnmf"
WHITELIST_PATH = (
    CNMF_ROOT / "program_review" / "cnmf_program_whitelist_draft.tsv"
)
METRICS_PATH = (
    CNMF_ROOT
    / "program_review"
    / "cnmf_program_review_metrics_condition_blind.tsv"
)
QUEUE_PATH = (
    CNMF_ROOT / "program_review" / "cnmf_program_review_queue_draft.tsv"
)
DECISION_PATH = (
    CNMF_ROOT / "program_review" / "human_review_decisions_draft.tsv"
)
MASTER_H5AD = (
    CNMF_ROOT
    / "integrated"
    / "ab_xenium_cellcharter_cnmf_selected_draft.h5ad"
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

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
"""
        ),
        markdown(
            """
## Inventory and draft decisions

This section confirms the exact whitelist and source usage objects. Counts are
shown by lineage and draft admission category, without condition statistics.
"""
        ),
        code(
            """
whitelist = pd.read_csv(
    WHITELIST_PATH,
    sep="\\t",
    dtype=str,
    keep_default_na=False,
)
whitelist_sha256 = file_sha256(WHITELIST_PATH)

inventory = (
    whitelist.groupby("lineage", sort=False)
    .agg(
        selected_k=("selected_k", "first"),
        raw_programs=("program", "size"),
        primary_programs=("primary_include", lambda s: int(s.eq("TRUE").sum())),
        sensitivity_total=("sensitivity_include", lambda s: int(s.eq("TRUE").sum())),
        excluded_programs=("decision", lambda s: int(s.eq("exclude").sum())),
    )
    .reset_index()
)
display(inventory)
print("whitelist SHA-256:", whitelist_sha256)

program_metrics = pd.read_csv(METRICS_PATH, sep="\t")
review_queue = pd.read_csv(QUEUE_PATH, sep="\t", keep_default_na=False)
if not program_metrics["metric_condition_blind"].astype(bool).all():
    raise RuntimeError("Program metric table is not marked condition-blind")
display(
    review_queue[
        [
            "review_order",
            "review_tier",
            "lineage",
            "program",
            "decision",
            "metric_flags",
        ]
    ].head(25)
)
"""
        ),
        markdown(
            """
## Review controls

Review one lineage-program pair at a time. TOP_FRACTION=0.05 means the
highest-usage 5% of cells within that lineage. The notebook takes an exact
number of ranked cells rather than including arbitrary quantile ties.

The 30-um neighbor radius is a contamination diagnostic, not the future
predictive-model neighborhood definition.

Scale is handled differently in review and modeling. This notebook ranks cells
within each program, so a low-baseline program is not disadvantaged. It keeps
raw cNMF usages and does not renormalize the retained whitelist. During model
training, each constructed focal or neighbor feature will instead be
standardized with training-fold statistics. Sparse-group-lasso does not
correct unequal feature scales on its own.
"""
        ),
        code(
            """
LINEAGE = "chp"
PROGRAM = "Usage_1"
TOP_FRACTION = 0.05
NEIGHBOR_RADIUS_UM = 30.0
MAX_CONTEXT_FOCAL_CELLS = 2_000
MAX_CORRELATION_CELLS = 50_000

# None selects the blinded section containing the most high-usage cells.
SPATIAL_SECTION = None
SPATIAL_POINT_SIZE = 3.0
"""
        ),
        code(
            """
if LINEAGE not in LINEAGE_DIRECTORIES:
    raise KeyError(f"Unknown lineage {LINEAGE!r}: {list(LINEAGE_DIRECTORIES)}")

lineage_root = CNMF_ROOT / LINEAGE_DIRECTORIES[LINEAGE]
manifest_path = lineage_root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text())
usage_path = Path(manifest["paths"]["usage_h5ad"])
usage_adata = ad.read_h5ad(usage_path, backed="r")
programs = usage_columns(usage_adata)
if PROGRAM not in programs:
    raise KeyError(f"{PROGRAM!r} is unavailable; choose one of {programs}")

review_obs, sample_aliases = prepare_blinded_usage_obs(
    usage_adata.obs,
    sample_key="sample_id",
    usage_cols=programs,
)
assert "condition" not in review_obs
assert "sample_id" not in review_obs

program_row = whitelist.loc[
    whitelist["lineage"].eq(LINEAGE)
    & whitelist["program"].eq(PROGRAM)
]
if len(program_row) != 1:
    raise RuntimeError(
        f"Expected one whitelist row for {LINEAGE}/{PROGRAM}; found {len(program_row)}"
    )
program_row = program_row.iloc[0]

display(
    program_row[
        [
            "lineage",
            "program",
            "proposed_label",
            "category",
            "confidence",
            "decision",
            "rationale",
            "top_genes",
        ]
    ].to_frame("draft_value")
)
current_metrics = program_metrics.loc[
    program_metrics["lineage"].eq(LINEAGE)
    & program_metrics["program"].eq(PROGRAM)
]
if len(current_metrics) != 1:
    raise RuntimeError(
        f"Expected one metric row for {LINEAGE}/{PROGRAM}; found {len(current_metrics)}"
    )
display(
    current_metrics.iloc[0][
        [
            "sd_usage",
            "iqr_usage",
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
        ]
    ].to_frame("condition_blind_metric")
)
print(
    f"{LINEAGE}: {usage_adata.n_obs:,} cells, "
    f"{len(programs)} programs, {review_obs['blinded_sample'].nunique()} blinded sections"
)
"""
        ),
        markdown(
            """
## Distribution and section-concentration checks

A credible program can vary between sections, but a factor whose high-usage
cells occur almost entirely in one section deserves inspection for section
edges, segmentation quality, or a sample-specific technical effect. Section
concentration is a warning flag, not an automatic exclusion.
"""
        ),
        code(
            """
qc_summary = program_usage_qc_summary(
    review_obs,
    usage_cols=programs,
    top_fraction=TOP_FRACTION,
)
display(qc_summary.loc[qc_summary["program"].eq(PROGRAM)])

usage_plot = review_obs[["blinded_sample", PROGRAM]].copy()
usage_plot[PROGRAM] = pd.to_numeric(usage_plot[PROGRAM], errors="coerce")
figure = px.box(
    usage_plot,
    x="blinded_sample",
    y=PROGRAM,
    points=False,
    title=f"{LINEAGE} {PROGRAM}: distribution by blinded section",
)
figure.update_layout(template="plotly_white", showlegend=False)
figure
"""
        ),
        markdown(
            """
## High-usage cells

These cells most strongly define the program. Check whether their annotated
lineage, spatial domain, count depth, and feature count are coherent. Very
low-quality cells, contradictory identities, or dominance by one annotation
subtype are reasons to inspect more closely.
"""
        ),
        code(
            """
high_usage = select_high_usage_cells(
    review_obs,
    program=PROGRAM,
    top_fraction=TOP_FRACTION,
)
display(high_usage.head(50))
display(
    high_usage.groupby(
        ["blinded_sample", "cluster_sub", "spatial_domain"],
        dropna=False,
        observed=True,
    )
    .size()
    .rename("n_high_usage_cells")
    .reset_index()
    .sort_values("n_high_usage_cells", ascending=False)
    .head(50)
)
"""
        ),
        markdown(
            """
## Expected-gene support

For the genes recorded in the draft whitelist, compare raw-count detection and
mean counts in high-usage cells against other cells from the same cNMF lineage.
This is not differential-expression inference. It is a direct coherence check.
"""
        ),
        code(
            """
expected_genes = [
    gene for gene in str(program_row["top_genes"]).split("|") if gene
]
gene_support = high_usage_gene_support(
    usage_adata,
    high_cell_ids=high_usage["cell_id"],
    genes=expected_genes,
)
display(gene_support)
if gene_support.attrs.get("missing_genes"):
    print("Genes absent from this panel:", gene_support.attrs["missing_genes"])
"""
        ),
        markdown(
            """
## Spatial localization

The map uses a blinded section label and never colors or facets by condition.
Look for plausible anatomical localization or local multicellular patches.
Warning patterns include section borders, segmentation halos, exact tracing of
a different lineage, or a handful of extreme cells.
"""
        ),
        code(
            """
high_section_counts = high_usage["blinded_sample"].value_counts()
shown_section = (
    str(high_section_counts.index[0])
    if SPATIAL_SECTION is None
    else str(SPATIAL_SECTION)
)
spatial_figure = plot_blinded_spatial_usage(
    review_obs,
    np.asarray(usage_adata.obsm["spatial"]),
    program=PROGRAM,
    blinded_sample=shown_section,
    point_size=SPATIAL_POINT_SIZE,
)
spatial_figure
"""
        ),
        markdown(
            """
## Neighboring-cell context

The table counts cell types within 30 um of high-usage focal cells, separately
within each tissue section but pooled only after neighbor lookup. Real sample
IDs and conditions are not returned. A neuronal program in astrocyte-labeled
cells that is strongest immediately beside neurons, for example, is more
consistent with segmentation spillover than an intrinsic astrocyte state.
"""
        ),
        code(
            """
master = ad.read_h5ad(MASTER_H5AD, backed="r")
try:
    context_ids = high_usage["cell_id"].head(MAX_CONTEXT_FOCAL_CELLS).tolist()
    neighbor_context = high_usage_neighbor_context(
        master.obs[["sample_id", "cluster_sub"]],
        np.asarray(master.obsm["spatial"]),
        focal_cell_ids=context_ids,
        sample_key="sample_id",
        cell_type_key="cluster_sub",
        radius=NEIGHBOR_RADIUS_UM,
    )
finally:
    master.file.close()
display(neighbor_context.head(40))
"""
        ),
        markdown(
            """
## Relationship to other programs

Strong positive correlation with another coherent program can indicate a
shared state; strong correlation with a known contaminant can be a warning.
This uses a condition-blind sample of lineage cells and is descriptive only.
"""
        ),
        code(
            """
correlation_obs = review_obs[programs]
if len(correlation_obs) > MAX_CORRELATION_CELLS:
    correlation_obs = correlation_obs.sample(
        MAX_CORRELATION_CELLS,
        random_state=0,
    )
program_correlations = (
    correlation_obs.corr(method="spearman")[PROGRAM]
    .drop(PROGRAM)
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .rename_axis("other_program")
    .reset_index(name="spearman_correlation")
)
display(
    program_correlations.merge(
        whitelist.loc[
            whitelist["lineage"].eq(LINEAGE),
            ["program", "proposed_label", "decision"],
        ],
        left_on="other_program",
        right_on="program",
        how="left",
    ).drop(columns="program").head(15)
)
"""
        ),
        markdown(
            """
## Decision rubric

- keep_primary: coherent lineage identity, subtype, or activity, supported by
  high-usage cells and spatial localization.
- keep_sensitivity: biologically plausible but less secure, cell-cycle related,
  spatially restricted, or otherwise deserving sensitivity analysis.
- exclude: wrong-lineage, mixed, technical, doublet-like, or spatially
  consistent with spillover or segmentation contamination.
- needs_followup: evidence is insufficient; do not admit it yet.

Record biological evidence in REVIEW_NOTES. Do not mention condition
differences or predictiveness. Saving updates a separate human-decision table;
it does not edit or freeze the whitelist.
"""
        ),
        code(
            """
DRAFT_TO_REVIEW = {
    "include_primary": "keep_primary",
    "include_sensitivity": "keep_sensitivity",
    "exclude": "exclude",
}

REVIEW_DECISION = DRAFT_TO_REVIEW[str(program_row["decision"])]
REVIEW_CONFIDENCE = str(program_row["confidence"])
REVIEW_NOTES = ""
REVIEWER = ""
SAVE_REVIEW_DECISION = False

review_record = build_review_decision(
    lineage=LINEAGE,
    program=PROGRAM,
    review_decision=REVIEW_DECISION,
    review_confidence=REVIEW_CONFIDENCE,
    review_notes=REVIEW_NOTES,
    reviewer=REVIEWER,
    top_fraction=TOP_FRACTION,
    neighbor_radius_um=NEIGHBOR_RADIUS_UM,
    whitelist_sha256=whitelist_sha256,
)
display(pd.Series(review_record).to_frame("review_value"))

if SAVE_REVIEW_DECISION:
    saved_path = upsert_review_decision(DECISION_PATH, review_record)
    print("saved:", saved_path)
else:
    print("Not saved. Set SAVE_REVIEW_DECISION=True only after review.")
"""
        ),
        markdown(
            """
## Freeze gate

Do not construct the model matrix from human decisions until:

1. every primary or sensitivity program has a saved review record;
2. plausible biological exclusions have been reviewed or explicitly deferred;
3. the decision table whitelist SHA-256 matches the current draft;
4. a frozen whitelist version and manifest are written;
5. only then are condition labels exposed for model training.

Close the backed lineage object when finished with:

    usage_adata.file.close()
"""
        ),
    ]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    return apply_notebook_presentation(notebook, "08")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
