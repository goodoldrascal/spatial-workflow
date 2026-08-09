# CellCharter and neighbor composition

This stage performs three ordered operations:

1. train or reuse a transcript representation;
2. build a within-sample spatial graph and fit CellCharter AutoK;
3. calculate nearest-neighbor composition using `spatial-nncomp`.

The generated job remains an ordinary bash script. A separate tmux wrapper can
launch it on whichever compute node the user chooses; the repository does not
encode SSH behavior or a host name.

Notebook 02 runs a resume-aware local overview and writes an equivalent
fail-fast job. A complete local stage is left unchanged; a partial stage is an
error. In the accepted AB workspace it uses the already materialized standard
`02_cellcharter` contract and backfills missing nncomp overlap tables. For a
new user without CellCharter outputs, the active no-reuse YAML runs a genuine
fresh CellCharter stage before nncomp. When `nncomp.overlap.enabled` is true,
nncomp also constructs ranked neighbors for k=1, 2, and 6 and runs the
configured compartment-stratified label permutations. In the normal mode,
CellCharter fits every candidate K in
the configured range and predicts domains at the K with the highest mean
cross-run stability. The review curve is therefore a stability-selection
curve, not a reconstruction-error elbow.

For a new AnnData object containing exactly the same cells as a previously
accepted CellCharter run, `cellcharter.reuse.enabled` can materialize the same
output contract without refitting. Labels are joined by configured sample and
cell IDs, written under the current spatial-domain key, and accompanied by an
explicit provenance record. The prior comparison-level stability sweep is
preserved and summarized into the same table produced by a fresh fit. This mode
does not copy the prior latent representation, graph, or fitted model; none is
used by the downstream spatial-nncomp stage.

The reuse API remains supported, but the AB example does not enable it. The
accepted historical evidence is copied into
`results/ab_xenium/00_cellcharter_reference` for migration audit only, while
the normal downstream path is the standard local `02_cellcharter` output.

CellCharter labels are written to the configured spatial-domain key. They are
data-driven contexts rather than curated anatomical region names.

The nearest-neighbor stage writes tidy composition and neighbor-fraction tables
for the read-only compartment review notebook.
Sample-domain-cell-type combinations are zero-filled, and outputs report both
total and supporting sample counts so condition means have an explicit support
interpretation.

The optional directional-overlap contract adds these Parquet files:

- `label_permutation_plan.parquet`, a reusable deterministic plan keyed by
  permutation, sample, compartment, and endpoint role;
- `ranked_neighbor_edges.parquet`, with source and neighbor cell IDs, rank,
  distance, labels, compartments, and candidate scope;
- `directional_knn_overlap.parquet`, with per-sample observed coefficients;
- `directional_knn_overlap_permutation.parquet`, with null means and standard
  deviations, deltas, fold enrichment, z-scores, empirical p-values, and FDR;
- `within_compartment_celltype_overlap_permutation.parquet`, the optimized
  all-source within-compartment result, including both `n_source_cells` and
  `n_neighbor_cells` in each sample-domain;
- `whole_sample_celltype_overlap_permutation.parquet`, the optimized all-source
  result from sample-wide graphs and sample-wide label shuffles; and
- `whole_sample_label_permutation_plan.parquet`, the deterministic sample-level
  draw plan used by that whole-sample result.

The compact plan stores one deterministic seed per sample-compartment draw
rather than one row per cell and permutation. Cell-ID and label-count
fingerprints prevent reuse on a changed cell universe. The `shared` endpoint
role permutes vertex labels once and is used by NN overlap. Separate `source`
and `target` roles make an independent-endpoint null available to future cell
communication analyses without conflating the two null hypotheses.

The within-compartment search is recomputed inside every sample-domain stratum;
it is not obtained by filtering whole-sample neighbors. Whole-sample and
compartment-pair results use the ranked whole-sample candidates, with the latter
retaining source-domain to target-domain direction.

The CellCharter stage writes:

- an annotated H5AD;
- domain abundance by sample;
- mean and standard deviation of stability for every candidate K;
- a summary containing the selected K and stability peaks;
- comparison-level stability values when an accepted run is reused.

The compartment review notebook displays the K selection before the brief
domain overview, selected-domain composition, neighbor-composition panels, and
the ranked directional-overlap permutation results when those artifacts exist.
Its final Plotly section defines `selected_overlap` as the sample-level
all-source within-compartment table and `condition_overlap` as its condition
aggregation. Each point pairs A-to-B and B-to-A cell-type z-scores inside the
same compartment; compartments are grouping and filtering variables, not the
entities paired on the axes.
`plot_reciprocal_celltype_overlap` takes one raw all-source table and returns
sample and condition panels side by side. `analysis_scope="whole_sample"`
selects the true pooled result; the default remains `within_compartment`.
Within the latter, `compartment=None` displays every compartment-specific row
without pooling them. An optional compartment selects one context;
`cell_types_a` and `cell_types_b` accept one or multiple cell types. Separate
`min_cells_a` and `min_cells_b` thresholds are applied before matched condition
means, which must meet `min_samples` per condition. `min_abs_z_score` and
`z_direction` (`both`, `positive`, or `negative`) filter displayed points only,
after unbiased condition means have been computed.

```python
figure = plot_reciprocal_celltype_overlap(
    celltype_permutation,
    compartment="3",  # omit for all compartments
    cell_types_a=["AST_sub0", "AST_sub1"],
    cell_types_b=["OLIG_sub0", "OLIG_sub1"],
    k_value=2,
    min_cells_a=10,
    min_cells_b=10,
    min_samples=4,
    min_abs_z_score=2.0,
    z_direction="positive",
)
```

For a sample-level overlap row, `delta = coefficient - null_mean` and
`fold_enrichment = coefficient / null_mean`. Delta is the absolute change in
the fraction of source cells with a top-k neighbor of the requested type; fold
enrichment is the relative ratio. The z-score is `delta / null_sd`.
