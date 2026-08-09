# cNMF lineage workflow

This stage runs reproducible cNMF programs on exact cell-type selections from
the accepted CellCharter AnnData object. Heavy preparation, execution, status,
and general review logic lives in `src/spatial_workflow/cnmf.py`.
Condition-associated usage tables and plots live in
`src/spatial_workflow/cnmf_usage_analysis.py` and
`src/spatial_workflow/cnmf_usage_plotting.py`; the notebooks are thin controls
over those functions.

## Published AB selections

The tracked configuration reproduces the earlier lineage settings across all
conditions and all spatial domains:

| lineage | exact `cluster_sub` values | cells | sweep | selected run |
| --- | --- | ---: | --- | --- |
| astrocyte | `AST-CX`, `AST-TH` | 105,329 | K=10–30, 15 replicates, 2,000 genes, seed 14 | K=18, 100 replicates, density threshold 0.1 |
| oligodendrocyte | `OLIG-CC`, `OLIG-GLOB`, `OLIG-STR`, `OPC` | 189,654 | K=10–30, 15 replicates, 2,000 genes, seed 14 | K=25, 100 replicates, density threshold 0.1 |
| inhibitory_neuron | `IN-GABA-HT-AMY`, `IN-SST-RTN`, `IN-PV`, `IN-VIP-ARC`, `IN-ChAT`, `MSN-D1`, `MSN-D2` | 162,691 | K=10–30, 15 replicates, 2,000 genes, seed 14 | K=19, 100 replicates, density threshold 0.1 |
| excitatory_neuron | `EX-L23-IT`, `EX-TH`, `EX-HPC`, `EX-HB`, `EX-HT-ENDO`, `EX-OXT` | 235,192 | K=10–30, 15 replicates, 2,000 genes, seed 14 | K=25, 100 replicates, density threshold 0.1 |
| perivascular | `EC`, `PVF`, `PERICYTE` | 126,014 | K=10–30, 15 replicates, 2,000 genes, seed 14 | K=26, 100 replicates, density threshold 0.1 |
| epd | `EPD` | 12,535 | migrated K=4–12 sweep, 10 replicates, 2,000 genes, seed 14 | migrated K=10, 100 replicates, density threshold 0.1 |
| chp | `ChP` | 8,550 | K=4–20, 15 replicates, 2,000 genes, seed 14 | K=9, 100 replicates, density threshold 0.1 |

Lineages are explicit lists. No substring or regular-expression matching is
used. Domains and conditions accept one value, repeated values, or `all`.
The configuration also records the one astrocyte and five oligodendrocytes
excluded from the earlier accepted inputs because they have only 1–3 total
transcripts and zero counts among cNMF's selected overdispersed genes.
The counts above were verified from the current accepted H5AD across all 12
biological samples and all 13 spatial domains. Selected K values in the table
are verified from each lineage's current run manifest. EPD is the exception to
a native rerun: its accepted legacy K=4–12 sweep and K=10, 100-replicate
consensus were migrated without recomputing factor values. Its lineage-specific
`min_cells_per_gene: 0` export preserves the full 5,006-gene accepted input,
including 55 genes with zero EPD support.

## Stages

The runner reports completion from artifacts, not process names:

1. `export` selects cells and writes raw integer counts in `X` plus metadata,
   coordinates, a selection table, and a manifest.
2. `sweep` performs `prepare`, factorization, `combine`, and K-selection for
   every configured K.
3. `selected` creates a separate higher-replicate collection for the chosen K,
   combines it, and generates consensus usages and gene spectra.

Cell universe, K values, replicates, gene count, seed, worker partition, and
consensus parameters are protected by manifests and cannot silently share
incompatible outputs. Choose worker count before the first run; resume with the
same value so each replicate remains assigned to the same partition.
Each collection also holds a nonblocking process lock for the duration of a
run, so a second notebook or CLI invocation fails instead of launching
duplicate workers against the same artifacts.

The sweep and selected run deliberately have separate directories. A complete
K sweep is not a complete selected-K consensus analysis.

## Commands

Inspect the exact selection without writing:

```bash
python3 scripts/run_cnmf_workflow.py --config configs/local.yaml --lineage astrocyte --mode plan
```

Run or resume a configured lineage:

```bash
python3 scripts/run_cnmf_workflow.py --config configs/local.yaml --lineage astrocyte --mode all
python3 scripts/run_cnmf_workflow.py --config configs/local.yaml --lineage oligodendrocyte --mode all
```

For a sweep-only launch of one of the newer configured lineages, use
`--mode sweep`; this cannot run the placeholder selected K.

Define a custom exact lineage and a subset of domains. Use a unique analysis
name so it cannot overwrite the configured lineage outputs:

```bash
python3 scripts/run_cnmf_workflow.py --config configs/local.yaml \
  --lineage astrocyte --analysis-name ast_domains_3_7 \
  --cell-type AST-CX --cell-type AST-TH \
  --compartment 3 --compartment 7 --mode all
```

Repeat `--cell-type`, `--compartment`, or `--condition`. Pass a single
`--compartment all` or `--condition all` for the full cohort. `--lineage all`
runs every configured lineage, but selection overrides require one lineage.

## Notebooks

- `notebooks/05_cnmf_run.ipynb` exposes selection, K, worker, and stage controls.
  Its run flags default to false, so opening the notebook cannot launch work.
- `notebooks/06_cnmf_inspection.ipynb` is read-only and provides Plotly views
  for K selection, grouped usage heatmaps, top genes, and spatial program maps.
  It also builds a condition-change table across every selected
  cell-type × program combination and a Matplotlib/seaborn sample violin/table
  plot for the best corrected or user-selected result. Inference uses
  biological-sample means, reports Welch and permutation p-values, and applies
  Benjamini-Hochberg correction within contrasts and globally.
- `notebooks/08_cnmf_program_whitelist_review.ipynb` is the condition-blind
  admission review for reusable cNMF program features. It replaces real sample IDs
  with stable section aliases and shows within-program high-usage cells,
  expected-gene support, spatial localization, neighboring-cell context, and
  correlations with other programs. It never uses condition labels or model
  predictiveness. Human decisions are saved only behind an explicit flag. See
  [the Notebook 08 user guide](cnmf_program_whitelist_review_guide.md) for queue
  navigation, metric interpretation, and the decision-saving workflow.

Rebuild both notebooks with:

```bash
python3 scripts/build_cnmf_notebooks.py
```

Rebuild the whitelist-review notebook with:

```bash
python3 scripts/build_cnmf_program_whitelist_review_notebook.py
```

Build the condition-blind diagnostic table and prioritized review queue with:

```bash
/stor/home/ncr828/cci_venv/bin/python3 \
  scripts/build_cnmf_program_review_metrics.py
```

This writes
`results/ab_xenium/05_cnmf/program_review/cnmf_program_review_metrics_condition_blind.tsv`,
`cnmf_program_review_queue_draft.tsv`, and a manifest containing the whitelist
hash and safe-field contract. The metrics cover usage variability, blinded
section breadth, ten-nearest-neighbor spatial coherence within the source
lineage, correlation with cell count/feature QC, pairwise program redundancy,
and raw-count support for annotation genes. They prioritize review but never
automatically admit a program. Raw-gene support is circular because the same
counts were used by cNMF, and spatial coherence can reflect biology or
spillover.

After editing the draft whitelist, rebuild the derived usage-enriched master
object explicitly:

```bash
/stor/home/ncr828/cci_venv/bin/python3 \
  scripts/build_selected_cnmf_anndata.py --overwrite
```

The human-reviewed draft whitelist covers all 150 programs from eight
lineages. It selects 55 primary programs and seven additional sensitivity-only
programs. Two unresolved programs remain excluded, so the whitelist is not
yet frozen. The reviewed development H5AD has 62 columns in
`obsm["X_cnmf_usage_selected"]`, paired with
`obsm["X_cnmf_usage_applicable"]`; 870,639 of 870,645 master cells have a
source-lineage usage vector. ChP contributes four selected features from all
8,550 ChP cells. The remaining six unmodeled cells are the documented
ultra-low-count astrocyte/oligodendrocyte exclusions.

The review ranks cells within each program, so raw program magnitude does not
control which cells are inspected. Do not renormalize the retained whitelist:
the native usages already sum to one across each lineage's complete K. The
future model must standardize every constructed feature with training-fold
statistics because sparse-group-lasso penalties are not scale invariant.

Run the disposable end-to-end smoke workflow with the cNMF environment:

```bash
/stor/home/ncr828/cci_venv/bin/python3 tests/smoke_cnmf_workflow.py
```

It creates an 80-cell synthetic count object under `/tmp`, exercises sweep and
selected consensus, validates result dimensions and artifacts, then removes the
temporary directory.

## Output contract

Each analysis is rooted at
`results/ab_xenium/05_cnmf/<analysis-name>/` and contains:

- `input/`: prepared counts H5AD, observation and variable tables, exact
  selection summary, and provenance manifest;
- `sweep/`: per-replicate spectra, combined spectra, K statistics, and the
  cNMF K-selection plot;
- `selected_k<K>_i<N>/`: the independent higher-replicate collection and final
  consensus usages, gene scores, gene TPM and consensus spectra, HVG list, and
  clustering diagnostic, plus a usage-enriched lineage H5AD. The inspection
  notebook derives ranked top-gene tables from these native cNMF results.

The migrated `epd/` tree follows the same contract. Legacy cell IDs were
canonicalized from `sample_subclusters:barcode` to `sample:barcode`, and
program columns use the current `Usage_1` through `Usage_10` names. The raw
counts, usages, gene scores, gene TPM values, and top-gene values are unchanged.
`epd/migration_manifest.json` records the source paths, structural transforms,
and equality checks. K=10 retains the seed schedule from the legacy joint
K=5/K=10 preparation rather than pretending to be a fresh single-K run.

All stages are resumable. Existing prepared selections are reused unless
`--overwrite-export` is passed explicitly. `--dry-run` prints commands without
writing manifests or launching cNMF.
