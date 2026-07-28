# AnnData contract

The standardized H5AD merges all configured samples and has one globally unique
observation name per cell. The within-sample ID remains in `obs["cell_id"]`.

## Observation roles

- `sample_id`: biological replicate
- `condition`: experimental group
- `cell_type`: primary cell-type label, required before the spatial overview
- `cell_type_broad`: optional broad label
- `spatial_domain`: CellCharter-derived spatial context
- `region`: optional curated anatomical annotation

Source columns may have different names. The `schema` section of the project
configuration maps these roles once for every downstream stage.

An external annotation table must contain its configured cell-ID column and
the column named by `schema.cell_type_key`. Ingestion must populate that
cell-type role before CellCharter and neighbor-composition review.

## Matrices

- `layers["counts"]`: raw non-negative integer transcript counts, stored as
  `int32` after an explicit range check
- `.X`: total-count normalized, log1p expression
- `obsm["spatial"]`: native two-dimensional coordinates in microns
- `obsm["spatial_aligned"]`: optional aligned coordinate frame
- `obsm["X_scVI"]`: optional learned transcript representation

Native spatial coordinates are not overwritten by alignment.

## Identity and provenance

`obs_names` use the deterministic form `<sample_id>:<cell_id>`. Each run writes
a compact manifest containing the resolved configuration, physical inputs,
logical sample sources, software versions, and random seeds. For per-sample
inputs, `sample_id` and `condition` come from configuration. For a merged
Seurat input, both roles come from the configured Seurat metadata columns and
are retained per cell.

