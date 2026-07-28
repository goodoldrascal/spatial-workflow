# Ingestion

The ingestion stage produces one standardized H5AD merged across configured
samples and a compact manifest.

Supported source contracts in the first stage are:

- an exported bundle containing count matrix, genes, barcodes, metadata, and
  optional coordinates;
- a raw Xenium output directory containing `cell_feature_matrix.h5` and
  `cells.parquet` or `cells.csv.gz`;
- a Seurat RDS or QS object exported through the repository R script with an
  explicit assay and layer;
- one merged, multi-sample Seurat RDS or QS object using `input_format:
  merged_seurat`.

For per-sample inputs, biological conditions come from configuration and are
not inferred from filenames. For `merged_seurat`, sample and condition columns
come from the Seurat metadata keys named in `schema`. The R exporter splits the
object into deterministic per-sample bundles before Python creates AnnData, so
the cohort cannot be stamped with one synthetic sample ID. Each sample must map
to one condition.

Each configured sample can provide an `annotations` mapping. Its table must
contain the configured cell-ID column and the column named by
`schema.cell_type_key`; this cell-type label is required before CellCharter and
neighbor-composition review.

The conversion manifest distinguishes the physical input objects from the
logical sample sources. A merged Seurat run records `input_mode`, the resolved
input object and bundle manifest, the configured sample and condition keys,
per-sample bundle paths, cell counts, and `condition_source`. The original
configuration and package versions remain embedded as before.

The notebooks resolve the repository root when Jupyter starts from either the
repository root or `notebooks/`. `notebooks/01_prepare_anndata.ipynb` writes the
conversion job script but does not execute it automatically.

