# Provenance

This file records which legacy analysis sources were used when a component was
promoted into this repository.

| Component | Primary source | Migration status |
|---|---|---|
| Curated AB Seurat builder | `/stor/scratch/WCAAR/rhyan_scratch/merged_nihal/merged_xenium_seurat.qs` and `configs/ab_curated_seurat_sources.local.tsv` | accepted (AB, 2026-07-27) |
| Seurat bundle export | `liana/spatial-nncomp/scripts/export_seurat_*_to_bundles.R` | accepted (AB merged mode, 2026-07-27) |
| Merged Seurat-to-AnnData conversion | `liana/spatial-nncomp/src/spatial_nncomp/bundles.py` | accepted (AB, 2026-07-27) |
| Raw Xenium matrix reader | `liana/workflows/celladmix/scripts/validate_current_xenium_count_universe.py` | scaffold |
| scVI representation | `liana/workflows/compartments/scripts/build_ab_whole_sample_scvi_input.py` | scaffold |
| CellCharter AutoK | `liana/workflows/compartments/scripts/run_ab_focus_compartment_3_cellcharter.py` | scaffold |
| Neighbor composition | `liana/spatial-nncomp` | external dependency |

Legacy files remain untouched until the migrated component passes a parity
check against a known output.

## AB curated Seurat and AnnData acceptance

Accepted on 2026-07-27 using:

- base Seurat object:
  `/stor/scratch/WCAAR/rhyan_scratch/merged_nihal/merged_xenium_seurat.qs`;
- authoritative source manifest:
  `configs/ab_curated_seurat_sources.local.tsv`;
- curated outputs:
  `results/ab_xenium/00_curated_seurat/ab_xenium_curated.qs`,
  `results/ab_xenium/00_curated_seurat/ab_xenium_curated_manifest.tsv`, and
  `results/ab_xenium/00_curated_seurat/ab_xenium_curated_sample_audit.tsv`;
- final AnnData outputs:
  `results/ab_xenium/01_anndata/ab_xenium.h5ad`,
  `results/ab_xenium/01_anndata/ab_xenium.manifest.json`, and
  `results/ab_xenium/01_anndata/ab_xenium.parity.json`;
- legacy parity reference:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/AB_merged_raw.h5ad`.

The final shape is 870,645 cells × 5,006 genes. Parity found zero mismatches
for cell identities, conditions, annotations, coordinates, and every raw count
entry. All base and authoritative source inputs were left untouched.

## VAP46 compartment 3 single-cell communication reference

The following artifacts are the concrete reference set for the future
single-cell communication module. Recording them here does not mark that module
as implemented.

- Notebook:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/notebooks/vap46_comp3_single_cell_signaling_networks.ipynb`
- Final LR-resolved positive-edge ledger:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/outputs/first_hop_profile_permutation_atlas/vap_46_igg/compartment_3_combined_pool-all_cells_perm250_minopp10_pos10_lsend2_prec2_no-self/single_cell_networks/gate-both_atlas_fdr_mode-combined_norm-equal_total_leiden-r1_dbscan-eps50-min5/lr_cell_edges.parquet`
  — 31,439 positive cell-to-cell edges, one per LR and signaling mode, across
  262 channels passing `both_atlas_fdr`; includes cell IDs and types, LR
  identity, expression, distance, raw score, enrichment, q-values, and graph
  weights.
- Contact opportunity denominator:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/outputs/landmark_comp3_raw_edges/vap_46_igg/landmark_contact_graph_opportunities_vap_46_igg_comp3.parquet`
  — 32,630 directed 30-µm spatial cell-pair opportunities.
- Contact raw positive LR edges:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/outputs/landmark_comp3_raw_edges/vap_46_igg/landmark_contact_raw_edges_vap_46_igg_comp3.parquet`
  — LR-resolved contact and ECM edges before the final atlas channel gate.
- Secreted opportunity denominator:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/outputs/comp3_secreted_edge_permutation_cache/vap_46_igg/secreted_graph_opportunities.parquet`
  — 301,806 directed pairs with cell IDs and types, distance, kernel weight,
  and normalized receiver weight; LR scores are reconstructed from the
  same-directory ligand and receptor score tables.
- Channel-level atlas table:
  `/stor/scratch/WCAAR/rhyan_scratch/liana/workflows/single_cell/outputs/first_hop_profile_permutation_atlas/vap_46_igg/compartment_3_combined_pool-all_cells_perm250_minopp10_pos10_lsend2_prec2_no-self/canonical_lr_edges.parquet`
  — 3,090 LR × sender type × receiver type × mode rows with positive-edge and
  opportunity counts, observed and null statistics, enrichment, p/q-values,
  and gate columns.

The eventual module must support contact and secreted modes, retain individual
sender cell IDs for secreted edges, and make the permutation-null context
configurable. The currently scattered implementations should be consolidated
behind one clean module and entrypoint. For condition modeling, channel
inclusion must be defined independently of significance in VAP46.
