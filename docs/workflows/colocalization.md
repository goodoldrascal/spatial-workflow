# Directional colocalization review

Notebook 04 reviews whether one annotated cell population is spatially
associated with another and whether that association changes between
conditions. Directional relationships are the primary result. Reciprocal
relationships are a stricter secondary confirmation.

## Run the stage

Start from the repository root after Notebook 03 and the all-source nearest
neighbor permutation stage have completed:

```bash
python3 scripts/build_colocalization_tables.py \
  --config configs/local.yaml \
  --overwrite

jupyter lab notebooks/04_colocalization_analysis.ipynb
```

The table builder performs the sample-level aggregation, limma comparison,
exact permutation test, leave-one-out stability calculation, and CSV writing.
Notebook 04 is a review layer: changing its filters or plotting controls does
not rewrite the complete result tables.

## Three quantities answer different questions

| Quantity | Interpretation |
| --- | --- |
| `coefficient` | Observed fraction of source cells with at least one target cell among their first `k` neighbors |
| `delta` | Observed coefficient minus its within-sample, within-domain label-permutation null |
| `contrast_delta` | Numerator-condition mean delta minus denominator-condition mean delta |

A nonzero coefficient means the cell types touch often enough to measure. It
does **not** establish positive colocalization. Positive colocalization requires
delta above zero with consistent sample-level enrichment evidence.

Likewise, a small between-condition p-value only establishes a difference. A
pair can differ significantly while being depleted relative to the null in
both conditions. Notebook 04 therefore requires positive within-condition
colocalization in at least one contrast condition before displaying a
differential relationship as colocalized.

## Default evidence gate

The default directional gate requires, in at least one condition:

- positive delta in at least 3 samples;
- median delta greater than 0;
- Stouffer enrichment q-value no greater than 0.15; and
- partial-conjunction 3-of-4 enrichment q-value no greater than 0.10.

The reciprocal gate applies those criteria to both A→B and B→A in the same
condition. It is deliberately secondary because a valid directional
relationship need not pass in reverse.

Between-condition plots additionally use:

- mean observed coefficient of at least 0.05 in either condition;
- exact permutation p-value no greater than 0.05; and
- leave-one-out sign fraction of 1.0.

For a 4-versus-4 exact test, the two-sided p-value floor is
`0.02857142857142857`. Exact p-values are unadjusted; use the limma q-value in
the hover text and detail table when making multiple-testing claims.

## Read the primary plot

Each row is one directed source→target relationship in one spatial domain:

1. **Observed neighborhood coefficient** shows how common the relationship is
   in each condition.
2. **Colocalization relative to permutation null** shows each condition's mean
   delta. Filled points mark conditions that pass the full within-condition
   colocalization gate; open points do not.
3. **Between-condition change** shows the contrast delta. Positive values favor
   the numerator condition and negative values favor the denominator.

Rows are ranked by absolute contrast delta after all evidence filters. Hover
text provides condition coefficients, exact p-values, limma q-values, and
leave-one-out stability.

## Review order

1. Use the within-condition directional table to establish positive
   source→target colocalization.
2. Review the directional contrast plot and its compact detail table.
3. Use reciprocal results only as confirmation that both directional
   denominators support the same biological pattern.
4. Enable `show_z_diagnostics` only when sample-level null-standardized
   geometry is needed for troubleshooting.

Complete unfiltered tables remain under the configured colocalization output
directory. The principal files are:

- `all_directional_within_condition_stats.csv`;
- `all_reciprocal_within_condition_stats.csv`;
- `all_directional_limma.csv`;
- `all_reciprocal_limma.csv`; and
- one directional, reciprocal, and sample-support table per configured
  contrast.
