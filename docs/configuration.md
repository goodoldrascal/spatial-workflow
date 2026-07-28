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

Stage-relative result paths resolve under `paths.results_root`. Conversion input
paths resolve under `paths.data_root`.

Set `runtime.r_libs_user` when Seurat conversion should use an isolated R
library. Relative values resolve from the configuration directory, so
`../.r-lib` points to the repository-local library from a config under
`configs/`.
