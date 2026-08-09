# Curated AB merged Seurat object

This step creates a new merged Seurat QS object with the final AB annotations.
It uses the existing merged object for assays, counts, reductions, and images,
and uses the twelve final `*_subclusters.qs` objects only as authoritative
cell-level metadata sources.

The builder never modifies or overwrites an input. It also does not fall back
to a fuzzy barcode join. Before changing metadata, it requires an exact
one-to-one match of every composite `sample_id + cell_id` key in the merged
object and the twelve sources.

## Inputs

- Base object:
  `/path/to/merged_xenium_seurat.qs`
- Source manifest:
  `configs/ab_curated_seurat_sources.local.tsv`
- Portable template: `configs/ab_curated_seurat_sources.example.tsv`
- Builder:
  `scripts/build_ab_curated_merged_seurat.R`

The source manifest defines conditions explicitly. They are not inferred from
filenames. Relative `input_path` values are resolved from the manifest
directory.

## Build command

Choose a new output path with enough free space. The build loads the 33 GB QS
object and should be launched as a batch or tmux job, not run interactively in
the preparation notebook.

```bash
R_LIBS_USER="$PWD/.r-lib" Rscript scripts/build_ab_curated_merged_seurat.R \
  --base /path/to/merged_xenium_seurat.qs \
  --sample-sheet configs/ab_curated_seurat_sources.local.tsv \
  --output results/ab_xenium/00_curated_seurat/ab_xenium_curated.qs
```
If Seurat appended a terminal chain of `_<integer>` suffixes during successive
merges, the first run stops before writing and reports whether stripping that
complete suffix chain would yield a complete bijection. Only then authorize the
transformation explicitly:

```bash
R_LIBS_USER="$PWD/.r-lib" Rscript scripts/build_ab_curated_merged_seurat.R \
  --base /path/to/merged_xenium_seurat.qs \
  --sample-sheet configs/ab_curated_seurat_sources.local.tsv \
  --output results/ab_xenium/00_curated_seurat/ab_xenium_curated.qs \
  --allow-seurat-merge-suffix
```

If the base object already has an authoritative within-sample cell-ID column,
pass it with `--base-cell-id-column`. No suffix recovery is needed in that
case.

## Metadata written

The builder overwrites or creates only these columns in `meta.data`:

- `cell_id`
- `sample_id`
- `condition`
- `cluster_sub`
- `celltype_short`
- `celltype_full`
- `centroid_x`
- `centroid_y`

Labels are stored as character values and coordinates as numeric values. All
source labels must be nonmissing and all coordinates must be finite.

The base cell order and assay set must remain identical after the patch. The
configured `Xenium` assay and `counts` layer or slot must be present.
The base object's `DefaultAssay` remains `SCT`; do not change it for this step.
AnnData conversion deliberately reads the `Xenium` assay and `counts` layer
specified in configuration.

## Audit outputs

For `ab_xenium_curated.qs`, the builder also writes:

- `ab_xenium_curated_manifest.tsv`: paths, package versions, assay dimensions,
  identity method, and output size;
- `ab_xenium_curated_sample_audit.tsv`: source path, condition, cell count, and
  annotation cardinalities for each sample.

All three output paths must be absent when the build starts.
