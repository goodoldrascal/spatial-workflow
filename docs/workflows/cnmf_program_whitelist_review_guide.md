# Using Notebook 08: condition-blind cNMF program review

This guide explains how to use
`notebooks/08_cnmf_program_whitelist_review.ipynb` to review cNMF programs one
at a time before they enter the neighborhood exposure model.

Notebook 08 is a biological quality-control gate. It asks whether a program is
a coherent state of its source lineage or instead reflects contamination,
segmentation spillover, mixed cells, a section edge, or another technical
effect. It does **not** test whether a program predicts VAP exposure.

The review is condition-blind:

- real sample IDs are replaced by deterministic `Section_XX` aliases;
- condition, exposure, treatment, and group fields are omitted;
- no condition comparison, p-value, classifier, or feature importance is run;
- decisions use program genes, high-usage cells, spatial localization, and
  neighboring-cell identity.

## Files used by the notebook

| File | Purpose |
| --- | --- |
| `notebooks/08_cnmf_program_whitelist_review.ipynb` | Interactive review notebook |
| `scripts/build_cnmf_program_whitelist_review_notebook.py` | Reproducible notebook builder and source of truth |
| `results/ab_xenium/05_cnmf/program_review/cnmf_program_whitelist_draft.tsv` | Draft biological labels and admission calls |
| `results/ab_xenium/05_cnmf/program_review/cnmf_program_review_metrics_condition_blind.tsv` | Full condition-blind diagnostic metrics |
| `results/ab_xenium/05_cnmf/program_review/cnmf_program_review_queue_draft.tsv` | Prioritized one-row-per-program review queue |
| `results/ab_xenium/05_cnmf/program_review/human_review_decisions_draft.tsv` | Complete saved human reviews for all 150 programs |
| `results/ab_xenium/05_cnmf/program_review/cnmf_program_whitelist_human_reviewed_draft.tsv` | Reviewed feature-admission table; two unresolved programs are admitted to neither model |
| `results/ab_xenium/05_cnmf/program_review/cnmf_program_whitelist_human_reviewed_draft.manifest.json` | Hashes, counts, unresolved keys, and freeze status |
| `scripts/finalize_cnmf_program_reviews.py` | Reproducible validator and writer for the approved reviews |
| `results/ab_xenium/05_cnmf/integrated/ab_xenium_cellcharter_cnmf_selected_draft.h5ad` | Master spatial object used for neighboring-cell context |

If the notebook itself needs a durable change, edit its builder and regenerate
it rather than editing notebook JSON directly:

```bash
/stor/home/ncr828/cci_venv/bin/python3 \
  scripts/build_cnmf_program_whitelist_review_notebook.py
```

## Starting the notebook

Start Jupyter from the `spatial-workflow` repository root so every relative
path resolves correctly:

```bash
cd /stor/scratch/WCAAR/rhyan_scratch/spatial-workflow
jupyter lab
```

Open `notebooks/08_cnmf_program_whitelist_review.ipynb` and execute the import
and inventory cells first. These cells create:

- `whitelist`: draft labels and admission decisions;
- `program_metrics`: all condition-blind metrics;
- `review_queue`: the prioritized order;
- `whitelist_sha256`: the exact whitelist version being reviewed.

## Reviewing the queue one program at a time

Use `review_order` as a one-based cursor. Replace the hardcoded `LINEAGE` and
`PROGRAM` assignments in the review-control cell with:

```python
# Change only this number to move through the review queue.
REVIEW_ORDER = 1

matches = review_queue.loc[
    review_queue["review_order"].eq(REVIEW_ORDER)
]
if len(matches) != 1:
    raise ValueError(
        f"Expected one queue row for review_order={REVIEW_ORDER}"
    )

review_item = matches.iloc[0]
LINEAGE = str(review_item["lineage"])
PROGRAM = str(review_item["program"])

display(review_item.to_frame("queue_value"))
print(
    f"Reviewing {REVIEW_ORDER}/{len(review_queue)}: "
    f"{LINEAGE} / {PROGRAM}"
)

TOP_FRACTION = 0.05
NEIGHBOR_RADIUS_UM = 30.0
MAX_CONTEXT_FOCAL_CELLS = 2_000
MAX_CORRELATION_CELLS = 50_000
SPATIAL_SECTION = None
SPATIAL_POINT_SIZE = 3.0
```

After finishing that program, change `REVIEW_ORDER` to the next number and
rerun the control cell and every review cell below it. An explicit cursor is
preferable to `next(iterator)` because the position remains visible and can be
recovered after restarting the kernel.

When repeatedly switching lineages in one kernel, close the previous backed
usage object before rerunning the lineage-loading cell:

```python
if "usage_adata" in globals() and usage_adata.isbacked:
    usage_adata.file.close()
```

## Understanding the queue columns

| Column | Meaning |
| --- | --- |
| `review_order` | Stable one-based position in the current queue |
| `review_tier` | Review urgency, not biological importance |
| `review_reason` | Why the row was assigned to that tier |
| `lineage`, `program` | Exact variables used to load the cNMF factor |
| `proposed_label` | Draft biological interpretation |
| `category` | Identity, activity, cell cycle, contamination, mixed, or uncertain |
| `confidence` | Confidence in the draft biological interpretation, not predictiveness |
| `primary_include` | Whether the draft primary model retains the program |
| `sensitivity_include` | Whether any draft model retains the program |
| `decision` | Draft admission call: primary, sensitivity, or exclude |
| `metric_flags` | Diagnostics that deserve attention; never automatic decisions |

### Meaning of `confidence`

`confidence` describes how secure the draft biological interpretation is.

- `high`: the program has a clear identity or is clearly contamination;
- `medium`: the label is plausible, but lineage assignment, spatial context,
  or section breadth needs review;
- `low`: the program is mixed or difficult to interpret reliably.

High confidence does not mean “include.” An endothelial program found in
ChP-labeled cells can be a high-confidence exclusion. Likewise, confidence is
not a cNMF stability score, probability, p-value, or exposure association.

## Which cells define the displayed metrics?

Most summary metrics, including `sd_usage`, are calculated across **all cells
in the program's source-lineage cNMF object**. The notebook then selects the
top `TOP_FRACTION` of those cells for detailed inspection.

With `TOP_FRACTION = 0.05`:

- `sd_usage`, `iqr_usage`, and `nonzero_fraction` use all lineage cells;
- section-concentration metrics describe the top 5%;
- raw-gene support compares the top 5% against other lineage cells;
- the spatial metric asks whether top-5% cells neighbor other top-5% cells.

Do not calculate the main SD from only the high-usage cells. Selecting the
upper tail truncates the distribution and produces a different, generally
misleading measure of program variability.

The selected queue row already contains the SD:

```python
PROGRAM_SD = float(review_item["sd_usage"])
```

The complete metric row, including the mean, is available from
`program_metrics`:

```python
metric_row = program_metrics.loc[
    program_metrics["lineage"].eq(LINEAGE)
    & program_metrics["program"].eq(PROGRAM)
].iloc[0]

PROGRAM_MEAN = float(metric_row["mean_usage"])
PROGRAM_SD = float(metric_row["sd_usage"])
```

## Condition-blind metric reference

No single metric admits or excludes a program. The metrics identify what to
inspect and make review consistent across programs with different usage
scales.

### Usage variability

| Metric | Interpretation |
| --- | --- |
| `sd_usage` | Standard deviation of raw usage across all source-lineage cells |
| `iqr_usage` | 75th minus 25th percentile; a robust measure of spread |
| `nonzero_fraction` | Fraction of source-lineage cells with usage greater than zero |
| `mean_usage`, `median_usage` | Overall raw usage location |
| `q90_usage`, `q95_usage`, `q99_usage`, `max_usage` | Upper-tail usage levels |

A low-magnitude program can still be useful if it varies consistently. The
future predictive model standardizes constructed features inside each training
fold, so raw magnitude does not determine the penalty. Standardization cannot,
however, rescue a nearly constant feature. The queue flags `sd_usage < 0.01`
as very low variability; this is a review warning rather than a universal
biological cutoff.

### Breadth across blinded sections

| Metric | Interpretation |
| --- | --- |
| `n_top_sections` | Number of blinded sections containing at least one top-usage cell |
| `largest_section_fraction` | Fraction of all top-usage cells contributed by the dominant section |
| `effective_top_sections` | Diversity-adjusted section count, calculated as `1 / sum(p_section^2)` |

If top cells are evenly distributed across 12 sections,
`effective_top_sections` approaches 12. If one section supplies nearly all top
cells, it approaches 1.

The queue currently flags:

- `largest_section_fraction >= 0.50`;
- `effective_top_sections < 3`.

These are generalization warnings. A real localized response may be
section-restricted, but it should not enter the primary model without careful
spatial and biological review.

### Within-lineage spatial coherence

| Metric | Interpretation |
| --- | --- |
| `spatial_knn_observed_top_fraction` | Fraction of evaluated neighbors that are also top-usage cells |
| `spatial_knn_expected_top_fraction` | Expected fraction after accounting for top-cell prevalence within each blinded section |
| `spatial_knn_enrichment` | Observed divided by expected top-neighbor fraction |

The diagnostic uses the ten nearest cells from the **same source lineage** and
performs neighbor lookup separately inside each blinded section.

- approximately 1: no more local clustering than expected;
- greater than 1: top-usage cells cluster spatially;
- less than 1: top-usage cells are spatially dispersed.

The queue flags enrichment `>= 2`. Strong spatial coherence is double-edged: it
can support a real anatomical state, but contamination or segmentation halos
around another lineage can also cluster strongly. Always interpret this metric
with the spatial map and neighboring-cell context.

### Correlation with cell-level QC

| Metric | Interpretation |
| --- | --- |
| `spearman_nCount_Xenium` | Rank correlation between usage and detected transcript count |
| `spearman_nFeature_Xenium` | Rank correlation between usage and detected gene count |

Values near zero are reassuring. A large absolute correlation can mean the
factor tracks cell size, segmentation quality, or measurement depth. The queue
flags an absolute correlation `>= 0.30`.

### Redundancy with other cNMF programs

| Metric | Interpretation |
| --- | --- |
| `max_abs_pairwise_spearman_program` | Other program with the largest absolute correlation |
| `max_abs_pairwise_spearman` | Signed Spearman correlation with that program |

The queue flags absolute correlation `>= 0.75`. A strong positive value can
indicate redundant factors or a shared state. Negative correlations are common
because each cell's complete set of cNMF usages sums to one: increasing one
usage necessarily leaves less mass for others. Modest anticorrelation alone is
not evidence that one program should be removed.

### Raw-count support for annotation genes

| Metric | Interpretation |
| --- | --- |
| `mean_top_gene_detection_fraction` | Mean fraction of top-usage cell-by-gene combinations with a nonzero count |
| `mean_background_gene_detection_fraction` | Corresponding fraction in other source-lineage cells |
| `median_top_gene_log2_mean_count_ratio` | Median across annotation genes of the log2 ratio of top-cell to background mean counts |
| `fraction_top_genes_positive_log2_ratio` | Fraction of annotation genes with higher mean counts in top cells |

Strong support means the annotated genes are actually detected in the cells
that receive high usage. It is still a circular QC check because cNMF was
learned from the same count matrix; it is not independent validation or a
differential-expression test.

For intuition, a median log2 ratio of 1 corresponds to roughly twofold higher
mean counts. The implementation adds a small pseudocount, so this conversion is
approximate for very sparse genes.

## Reading the notebook sections

### 1. Inventory and draft call

Confirm that the expected lineage and program were loaded. Read the draft
label, category, confidence, rationale, top genes, queue flags, and metric row
before looking at individual cells.

### 2. Distribution by blinded section

The box plot shows the complete usage distribution within each blinded
section. Look for:

- similar ranges across many sections;
- one section with an entirely different distribution;
- a program driven by a handful of extreme values;
- values compressed close to zero in nearly every section.

### 3. High-usage cells

The `high_usage` table contains the exact top fraction ranked by usage. Check:

- `cluster_sub` and broader cell-type identity;
- spatial domain;
- `nCount_Xenium` and `nFeature_Xenium`;
- whether one blinded section or subtype dominates;
- whether the top cells look like intact representatives of the source
  lineage.

### 4. Expected-gene support

The per-gene `gene_support` table compares top cells with other cells from the
same cNMF lineage. Inspect detection fractions and log2 mean-count ratios gene
by gene. A label supported by only one of eight annotation genes is less secure
than one supported coherently across the set.

### 5. Spatial map

By default, `SPATIAL_SECTION = None` displays the blinded section containing
the largest number of high-usage cells. Change it to another `Section_XX` to
inspect breadth across sections.

Biologically plausible patterns include anatomical layers, boundaries,
ventricular structures, vascular niches, or local multicellular patches.
Warnings include section borders, segmentation halos, exact tracing of a
different lineage, and isolated extreme cells.

### 6. Neighboring-cell context

The notebook reports cell types within `NEIGHBOR_RADIUS_UM = 30` micrometers of
up to `MAX_CONTEXT_FOCAL_CELLS` high-usage cells. Important columns include:

- `fraction_focal_cells_with_neighbor`: fraction of focal cells with at least
  one neighbor of that type;
- `mean_neighbors_per_focal_cell`: average number of that neighbor type;
- `n_neighbor_occurrences`: total pooled neighbor count.

This is a contamination diagnostic, not the final model neighborhood
definition. For example, a neuronal program in astrocyte-labeled cells that is
strongest immediately beside neurons is more consistent with spillover than an
intrinsic astrocyte program.

### 7. Relationship to other programs

Read the largest absolute Spearman correlations together with the other
programs' proposed labels and draft decisions. Correlation with a coherent
same-lineage activity program may be biologically sensible. Correlation with a
known contaminant deserves closer spatial and boundary review.

## Assigning the human decision

The allowed review decisions are:

| Decision | Use when |
| --- | --- |
| `keep_primary` | Coherent lineage identity, subtype, or activity with convincing cellular and spatial support |
| `keep_sensitivity` | Biologically plausible but less secure, cell-cycle related, spatially restricted, or section-concentrated |
| `exclude` | Wrong-lineage, mixed, technical, doublet-like, or consistent with spillover/segmentation contamination |
| `needs_followup` | Evidence is insufficient; do not admit the program yet |

The final notebook cell initially maps the draft decision into
`REVIEW_DECISION`. Treat that as a starting suggestion, not the human result.
Set the fields explicitly after review:

```python
REVIEW_DECISION = "keep_primary"
REVIEW_CONFIDENCE = "high"
REVIEW_NOTES = (
    "Top cells retain the expected lineage identity; annotation genes are "
    "coherently detected; usage spans blinded sections and forms plausible "
    "spatial patches without enrichment beside a conflicting source lineage."
)
REVIEWER = "your_name"
SAVE_REVIEW_DECISION = False
```

`REVIEW_CONFIDENCE` is your confidence after completing the review. It is
separate from the draft queue's `confidence` column.

Useful notes should record the evidence that would let someone else reproduce
the call:

- whether top cells retain source-lineage identity;
- which expected genes support or contradict the label;
- section breadth or concentration;
- spatial pattern;
- relevant neighboring cell types;
- why the evidence supports primary, sensitivity, exclusion, or follow-up.

Do not mention condition differences, exposure association, or model
predictiveness in the admission note.

## Saving safely

Keep `SAVE_REVIEW_DECISION = False` while composing and inspecting the record.
When it is correct:

1. set `SAVE_REVIEW_DECISION = True`;
2. run only the final decision cell once;
3. confirm the printed output path;
4. immediately return the flag to `False`.

Saving upserts one `(lineage, program)` record into
`human_review_decisions_draft.tsv`. It does not edit or freeze the whitelist.
Each record stores the whitelist SHA-256 so decisions cannot silently be
treated as belonging to a different whitelist version.

Inspect saved decisions with:

```python
saved_decisions = pd.read_csv(
    DECISION_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)
display(saved_decisions)
```

## Finding the next unreviewed row

After decisions exist, filter the queue to the current-whitelist decisions:

```python
if DECISION_PATH.exists():
    saved_decisions = pd.read_csv(
        DECISION_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    current_reviewed = saved_decisions.loc[
        saved_decisions["whitelist_sha256"].eq(whitelist_sha256),
        ["lineage", "program"],
    ].drop_duplicates()

    pending_queue = review_queue.merge(
        current_reviewed.assign(reviewed=True),
        on=["lineage", "program"],
        how="left",
    )
    pending_queue = pending_queue.loc[
        pending_queue["reviewed"].ne(True)
    ].drop(columns="reviewed")
else:
    pending_queue = review_queue.copy()

display(pending_queue.head(20))
```

Then set `REVIEW_ORDER` to the first displayed `review_order` rather than
renumbering the queue.

## Freeze gate

Do not construct the predictive neighborhood matrix from human decisions
until:

1. every primary or sensitivity program has a saved decision;
2. plausible biological exclusions have been reviewed or explicitly deferred;
3. every accepted decision has the current whitelist SHA-256;
4. a frozen whitelist and manifest are written;
5. only then are condition labels exposed for model training.

The model phase introduces different diagnostics—sample-held-out predictive
performance, coefficient sign and selection stability, and model ablations.
Those metrics must not be used retroactively to rescue a biologically invalid
program.

## Troubleshooting

### Paths do not resolve

Start Jupyter from the repository root. The notebook raises an explicit error
if it cannot find `src/spatial_workflow`.

### A program is unavailable

Confirm that `LINEAGE` and `PROGRAM` came from the same queue row. Program names
restart at `Usage_1` within each lineage.

### The notebook retains an old lineage file

Close `usage_adata.file`, rerun the cursor/control cell, then rerun the
lineage-loading cell.

### The decision did not save

Confirm that `REVIEWER` and `REVIEW_NOTES` are populated and that
`SAVE_REVIEW_DECISION` was set to `True` before running the final cell. Return
the flag to `False` afterward.

### Metrics and decisions refer to different whitelist versions

Rebuild the condition-blind metrics and queue from the current whitelist:

```bash
/stor/home/ncr828/cci_venv/bin/python3 \
  scripts/build_cnmf_program_review_metrics.py
```

Do not copy old decisions onto a new whitelist hash without reviewing what
changed.
