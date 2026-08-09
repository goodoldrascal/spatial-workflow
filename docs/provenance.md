# Provenance

This file records which legacy analysis sources were used when a component was
promoted into this repository.

| Component | Primary source | Migration status |
|---|---|---|
| Curated AB Seurat builder | `/path/to/merged_xenium_seurat.qs` and `configs/ab_curated_seurat_sources.local.tsv` | accepted (AB, 2026-07-27) |
| Seurat bundle export | `liana/spatial-nncomp/scripts/export_seurat_*_to_bundles.R` | accepted (AB merged mode, 2026-07-27) |
| Merged Seurat-to-AnnData conversion | `liana/spatial-nncomp/src/spatial_nncomp/bundles.py` | accepted (AB, 2026-07-27) |
| Raw Xenium matrix reader | `liana/workflows/celladmix/scripts/validate_current_xenium_count_universe.py` | scaffold |
| scVI representation | `liana/workflows/compartments/scripts/build_ab_whole_sample_scvi_input.py` | scaffold |
| CellCharter AutoK | `liana/workflows/compartments/notebooks/AB_whole_samples_scvi_latent30_cellcharter_review.ipynb` | accepted (AB label reuse, 2026-07-28) |
| Neighbor composition | `liana/spatial-nncomp` at `a5b7c4a5f4c52ecc5f0148901d5ee5b76fc523cc` | accepted (AB, 2026-07-28) |

Legacy files remain untouched until the migrated component passes a parity
check against a known output.

## AB curated Seurat and AnnData acceptance

Accepted on 2026-07-27 using:

- base Seurat object:
  `/path/to/merged_xenium_seurat.qs`;
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
  `/path/to/liana/AB_merged_raw.h5ad`.

The final shape is 870,645 cells × 5,006 genes. Parity found zero mismatches
for cell identities, conditions, annotations, coordinates, and every raw count
entry. All base and authoritative source inputs were left untouched.

## AB spatial overview acceptance

Accepted on 2026-07-28 without refitting the long-running CellCharter model.
The new AnnData and the prior accepted run matched all 870,645 cells exactly on
composite sample and raw cell ID, with zero missing, extra, duplicate, or null
assignments. Labels were written as `obs["spatial_domain"]`; all 13 label counts
match the source run. The archived AutoK sweep covers K=2 through K=16 with 90
comparisons per K. The accepted selection is K=13, with stability peaks at 4,
8, and 13.

Current outputs are:

- `results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.h5ad`;
- `results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.summary.json`;
- `results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.domain_abundance.csv`;
- `results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.autok_stability.csv`;
- `results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.autok_stability_comparisons.csv`;
- the three composition CSVs, three directional-overlap Parquets, and manifest
  under `results/ab_xenium/03_nncomp/`.

The accepted label table, AutoK summary, and comparison-level stability table
were copied byte-for-byte into
`results/ab_xenium/00_cellcharter_reference`; hashes and original location are
recorded in that directory's README. This is a migration evidence bundle, not
an active `cellcharter.reuse` input. The active and tracked AB YAML files have
no reuse mapping. Notebook 02 treats the complete standard local
`02_cellcharter` contract as finished here, while the same example performs a
fresh fit when those outputs are absent.

The spatial-nncomp run used the pinned package revision, a within-sample
six-neighbor graph, 12 samples, and 5,223,870 nonzero graph entries. It produced
8,232 abundance rows, 403,368 directed neighbor-composition rows, and 100,842
condition-summary rows. Notebook 02 passed generation and bash-syntax smoke
tests; notebook 03 executed end to end against these outputs.

Notebook 02 completed the directional-overlap backfill on 2026-07-29 without
touching the local CellCharter or accepted composition artifacts. It wrote
10,447,740 ranked edges plus 14,662 observed and 14,662 permutation-summary
rows. Coverage includes `whole_sample`, `within_compartment`, and
`compartment_pair` for k=1, 2, and 6, with `OLIG_sub7` as the configured source
and 1,000 sample-compartment-stratified label permutations. Notebook 03 then
executed end to end with the populated review tables.

The all-source within-compartment extension writes 241,272 summary rows. Its
current null was regenerated from `label_permutation_plan.parquet`, which has
468,000 rows covering 1,000 draws, 156 sample-compartment strata, and the
`shared`, `source`, and `target` endpoint roles. The derived overlap table uses
the `shared` role. In Notebook 03, reciprocal pairing yields 106,095 finite
sample-level and 41,668 condition-level cell-type points across all
compartments; compartment 3 yields 6,051 and 2,157 respectively.

After adding explicit neighbor-type populations, the 241,272-row all-source
table was regenerated from the same reusable plan. With the review minimum of
10 cells for both source and neighbor types, 187,879 rows remain eligible
before condition aggregation. At the single plotted value k=6, the reciprocal
views contain 34,353 sample points and 11,909 condition points overall; the
compartment-3 view contains 1,924 and 536.

Applying the additional strict requirement of at least four eligible samples
reduces the 67,619 condition rows to 29,884 before reciprocal pairing. At k=6,
the aligned sample and condition panels contain 23,144 and 5,786 reciprocal
points overall; compartment 3 contains 1,696 and 424.

The generalized function now defaults to all A/B types, every compartment,
k=2, 10 cells on each side, four samples per condition, no z cutoff, and both
signs. Those defaults yield 16,964 sample points and 4,241 condition points;
restricting to compartment 3 yields 1,320 and 330.

## VAP46 compartment 3 single-cell communication reference

The following artifacts are the concrete reference set for the future
single-cell communication module. Recording them here does not mark that module
as implemented.

- Notebook:
  `/path/to/liana/workflows/single_cell/notebooks/vap46_comp3_single_cell_signaling_networks.ipynb`
- Final LR-resolved positive-edge ledger:
  `/path/to/liana/workflows/single_cell/outputs/first_hop_profile_permutation_atlas/vap_46_igg/compartment_3_combined_pool-all_cells_perm250_minopp10_pos10_lsend2_prec2_no-self/single_cell_networks/gate-both_atlas_fdr_mode-combined_norm-equal_total_leiden-r1_dbscan-eps50-min5/lr_cell_edges.parquet`
  — 31,439 positive cell-to-cell edges, one per LR and signaling mode, across
  262 channels passing `both_atlas_fdr`; includes cell IDs and types, LR
  identity, expression, distance, raw score, enrichment, q-values, and graph
  weights.
- Contact opportunity denominator:
  `/path/to/liana/workflows/single_cell/outputs/landmark_comp3_raw_edges/vap_46_igg/landmark_contact_graph_opportunities_vap_46_igg_comp3.parquet`
  — 32,630 directed 30-µm spatial cell-pair opportunities.
- Contact raw positive LR edges:
  `/path/to/liana/workflows/single_cell/outputs/landmark_comp3_raw_edges/vap_46_igg/landmark_contact_raw_edges_vap_46_igg_comp3.parquet`
  — LR-resolved contact and ECM edges before the final atlas channel gate.
- Secreted opportunity denominator:
  `/path/to/liana/workflows/single_cell/outputs/comp3_secreted_edge_permutation_cache/vap_46_igg/secreted_graph_opportunities.parquet`
  — 301,806 directed pairs with cell IDs and types, distance, kernel weight,
  and normalized receiver weight; LR scores are reconstructed from the
  same-directory ligand and receptor score tables.
- Channel-level atlas table:
  `/path/to/liana/workflows/single_cell/outputs/first_hop_profile_permutation_atlas/vap_46_igg/compartment_3_combined_pool-all_cells_perm250_minopp10_pos10_lsend2_prec2_no-self/canonical_lr_edges.parquet`
  — 3,090 LR × sender type × receiver type × mode rows with positive-edge and
  opportunity counts, observed and null statistics, enrichment, p/q-values,
  and gate columns.

The eventual module must support contact and secreted modes, retain individual
sender cell IDs for secreted edges, and make the permutation-null context
configurable. The currently scattered implementations should be consolidated
behind one clean module and entrypoint. For condition modeling, channel
inclusion must be defined independently of significance in VAP46.
