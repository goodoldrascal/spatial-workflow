# Configuration

A project uses one YAML file. Start from `configs/example.yaml` for raw Xenium
inputs or `configs/ab_xenium.example.yaml` for the curated AB merged-Seurat
workflow. Keep local absolute paths in a gitignored `configs/local*.yaml` file.

The major sections are:

- `project`: stable dataset identifier
- `paths`: data and result roots
- `schema`: semantic AnnData keys
- `conversion`: source defaults, per-sample inputs, merge policy, and output
- `cellcharter`: representation, graph, aggregation, and AutoK settings
- `nncomp`: spatial graph and neighbor-composition settings
- `runtime`: executables and tmux session name
- `review`: default selections for the read-only notebook
- `colocalization`: generated-table settings, contrasts, and review thresholds
- `cnmf`: exact lineage selections, domain/condition scope, sweep, and consensus

`conversion.samples` contains one mapping per biological sample. Each entry
requires `input_path`, `sample_id`, and `condition`; `input_format` may be set
once at the stage level or overridden per sample. An `annotations` mapping
contains `path` and `cell_id_column`. Seurat input additionally requires the
stage-level `assay` and `layer` fields.

For one multi-sample Seurat RDS or QS object, use `merged_seurat` instead of
`conversion.samples`:

```yaml
conversion:
  input_format: merged_seurat
  input_path: merged_xenium_seurat.qs
  assay: Xenium
  layer: counts
  output_dir: 01_anndata
  output_name: cohort
  normalize_target: 10000
  join: inner
```

The Seurat metadata must contain the columns named by `schema.sample_key` and
`schema.condition_key`. The exporter writes one bundle per sample, and the
AnnData builder retains the condition values from that metadata. Each sample
must map to one condition. Other metadata columns, including curated cluster
assignments, remain in `obs`.
If `meta.data$cell_id` is present, it is the exported cell ID; Seurat column
names are used only to index matrices and reductions. Raw cell IDs must be
unique within a sample and may repeat across samples.

`conversion.join` controls the gene axis of the single merged H5AD: `inner`
keeps genes shared across samples, while `outer` keeps their union.

To reuse labels and K-selection evidence from an accepted CellCharter run on
an AnnData object with identical cells, add an explicit reuse mapping:

```yaml
cellcharter:
  reuse:
    enabled: true
    source_obs: /path/to/prior.cellcharter.obs.csv.gz
    source_summary: /path/to/prior.cellcharter.summary.json
    source_stability: /path/to/prior.cellcharter.autok_stability.csv
    source_sample_key: sample
    source_domain_key: cellcharter_autok
```

The first column of `source_obs` is the prior observation index, with the raw
cell ID after `:`. Reuse requires an exact one-to-one match on
`schema.sample_key` and `schema.cell_id_key`; it never joins on raw cell ID
alone. Leave `reuse.enabled` absent or false for a fresh scVI and CellCharter
fit.

The tracked AB example and active local AB configuration intentionally contain
no `reuse` block. The accepted workspace already carries the standard local
`02_cellcharter` output contract, so the resume-aware overview skips that
complete stage. On a workspace without those outputs, the same configuration
runs a fresh scVI and CellCharter fit. The optional reuse implementation above
remains available for other migrations but is not part of the published AB
execution path.

Enable ranked directional overlap as part of the nncomp stage with:

```yaml
nncomp:
  overlap:
    enabled: true
    k_values: [1, 2, 6]
    analyses:
      - whole_sample
      - within_compartment
      - compartment_pair
    source_cell_types:
      - OLIG_sub7
    neighbor_cell_types: null
    n_permutations: 1000
    seed: 0
    workers: -1
    permutation_plan:
      seed: 0
      endpoint_roles: [shared, source, target]
    reciprocal_cell_types:
      enabled: true
      n_permutations: 1000
      seed: 0
      workers: -1
      progress_every: 100
```

`whole_sample` and `compartment_pair` rank candidates across each complete
sample. `within_compartment` performs a separate nearest-neighbor search inside
each sample-compartment stratum. `source_cell_types` and
`neighbor_cell_types` may be null for every type, but filtering sources is
recommended for large permutation runs. Labels are shuffled within
sample-compartment strata while coordinates, compartments, and ranked edges
remain fixed.

`permutation_plan` writes reusable deterministic draws with cell-universe
fingerprints. Use the `shared` role when both ends of an edge must refer to one
permuted vertex-label assignment, as in NN overlap. The `source` and `target`
roles are independent streams intended for communication analyses that
explicitly choose an independent-endpoint null. `reciprocal_cell_types`
enables the optimized all-source within-compartment summary consumed by the
reciprocal Plotly panels in Notebook 03.

Configure reciprocal-review support and the one plotted k value with:

```yaml
review:
  overlap_cell_types_a: null
  overlap_cell_types_b: null
  min_overlap_cells_a: 10
  min_overlap_cells_b: 10
  min_overlap_samples: 4
  min_abs_z_score: 0.0
  z_direction: both
  overlap_k: 2
```

The A and B minima are applied to their respective cell-type populations before
reciprocal pairing. `overlap_cell_types_a` and `overlap_cell_types_b` may each
be null, one name, or a YAML list. Condition means use all count-eligible
matched samples and must meet `min_overlap_samples`. `min_abs_z_score` is then
applied to both plotted axes without biasing those means. `z_direction` accepts
`both`, `positive`, or `negative`; the sign-specific modes require both
reciprocal axes to have that sign. The plotting default is k=2.

Build the condition-level colocalization tables outside Notebook 04 with:

```bash
python3 scripts/build_colocalization_tables.py \
  --config configs/local.yaml \
  --overwrite
```

The corresponding configuration is:

```yaml
colocalization:
  output_dir: 04_colocalization_analysis
  k: 2
  min_cells_a: 10
  min_cells_b: 10
  expected_samples: 4
  contrasts:
    - numerator: vap_igg
      denominator: naive_igg
    - numerator: vap_ab
      denominator: vap_igg
  review:
    min_observed_colocalization: 0.05
```

The builder writes complete, unfiltered result tables. Notebook 04 applies
`min_observed_colocalization` interactively. For a directional contrast, the
mean observed coefficient must reach this floor in at least one condition.
For a reciprocal contrast, both directions must independently reach the floor
in at least one condition. This retains true gains or losses while excluding
stable differences between pairs that almost never co-occur.

The default notebook view also requires positive within-condition permutation
evidence in at least one contrast condition before calling a differential
relationship colocalized. This gate is applied directionally first; reciprocal
evidence is a stricter secondary result. See
[the colocalization review guide](workflows/colocalization.md) for the complete
criteria and plot interpretation.

Stage-relative result paths resolve under `paths.results_root`. Conversion input
paths resolve under `paths.data_root`.

Configure cNMF with exact lineage labels and independently parameterized sweep
and selected-K collections:

```yaml
runtime:
  cnmf_bin: cnmf

cnmf:
  input_h5ad: 02_cellcharter/ab_xenium_cellcharter.h5ad
  output_dir: 05_cnmf
  binary: cnmf
  counts_layer: counts
  cell_type_key: cluster_sub
  compartment_key: spatial_domain
  condition_key: condition
  sample_key: sample_id
  spatial_key: spatial
  preparation:
    min_counts_per_cell: 1
    min_cells_per_gene: 1
  lineages:
    astrocyte:
      cell_types: [AST-CX, AST-TH]
      exclude_cells: ["vap_46_igg:dhkkjcam-1"]
      compartments: all
      conditions: all
      numgenes: 2000
      seed: 14
      max_nmf_iter: 1000
      sweep:
        k_values: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        n_iter: 15
        workers: 10
      selected:
        k: 18
        n_iter: 100
        workers: 10
        density_threshold: 0.1
        local_neighborhood_size: 0.3
```

Use `all` for every observed domain/condition or a YAML list of exact values.
Every cell type included in a lineage must be listed explicitly. The runner's
repeatable CLI overrides support one or more cell types and domains while
requiring a distinct `--analysis-name` for safe custom output naming.
The tracked configuration also provides exact `cluster_sub` templates and
reviewed selected K values for inhibitory neurons (K=19), excitatory neurons
(K=25), and EC/PVF/PERICYTE cells (K=26).

Set `runtime.r_libs_user` when Seurat conversion should use an isolated R
library. Relative values resolve from the configuration directory, so
`../.r-lib` points to the repository-local library from a config under
`configs/`.
