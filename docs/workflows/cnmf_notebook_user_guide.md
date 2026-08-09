# Using the cNMF notebooks

This guide describes how to run cNMF for a configured lineage or a new exact
cell-type selection with Notebook 05, then inspect the results with Notebook
06.

For the condition-blind program-admission workflow in Notebook 08, use the
[Notebook 08 program-review guide](cnmf_program_whitelist_review_guide.md).

- Run notebook: notebooks/05_cnmf_run.ipynb
- Inspection notebook: notebooks/06_cnmf_inspection.ipynb
- Local configuration: configs/local.yaml

The notebooks are deliberately thin. Selection, export, cNMF execution, and
artifact checks are implemented in `src/spatial_workflow/cnmf.py`.
Sample-level condition tables are implemented in
`src/spatial_workflow/cnmf_usage_analysis.py`, and the sample violin/table plot
is implemented in `src/spatial_workflow/cnmf_usage_plotting.py`. Run flags
default to False, so opening or executing Notebook 05 does not start a cNMF job
by itself.

## Before starting

Create configs/local.yaml from the tracked example if it does not already
exist. Verify these entries under cnmf:

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

cell_type_key determines which adata.obs column is filtered. The configured
lineages use exact `cluster_sub` labels. Notebook 05 can override the key for
one custom analysis without changing the YAML:

    CELL_TYPE_KEY = "cluster_sub"

After changing the annotation key, enter exact values from that column in
CELL_TYPES. Use a new analysis name whenever the annotation key or selected
cell universe changes.

## Notebook 05 controls

The main parameter cell contains:

    LINEAGE = "astrocyte"
    ANALYSIS_NAME = None
    CELL_TYPE_KEY = None
    CELL_TYPES = None
    COMPARTMENTS = "all"
    CONDITIONS = "all"
    SELECTED_K = None
    WORKERS = None

    RUN_EXPORT = False
    RUN_SWEEP = False
    RUN_SELECTED_CONSENSUS = False
    RUN_IN_TMUX = False
    OVERWRITE_EXPORT = False

The controls have the following meanings:

| Control | Meaning |
| --- | --- |
| LINEAGE | A configured lineage and parameter template. The current choices are astrocyte, oligodendrocyte, inhibitory_neuron, excitatory_neuron, and perivascular. |
| ANALYSIS_NAME | Output-directory name. Supply a unique name for any custom cell-type, compartment, condition, or annotation-key selection. |
| CELL_TYPE_KEY | Optional adata.obs column override. None uses cnmf.cell_type_key from the YAML. |
| CELL_TYPES | Exact included values from the configured cell_type_key. None uses the configured lineage list. A custom lineage must explicitly list every included cell type; all is not accepted. |
| COMPARTMENTS | `"all"` for one pooled no-compartment cNMF input, or one/list of exact `spatial_domain` values to filter before the fit. |
| CONDITIONS | "all", one exact value, or a list of exact condition values. |
| SELECTED_K | K used only for the high-replicate selected-K consensus run. None uses the configured template K. It does not control the sweep. |
| WORKERS | Optional worker override. None uses the configuration. Keep the same worker count when resuming an existing collection. |
| RUN_EXPORT | Write the selected raw-count input and provenance files. |
| RUN_SWEEP | Run or resume the configured K sweep and generate K-selection evidence. |
| RUN_SELECTED_CONSENSUS | Run or resume the high-replicate consensus analysis for SELECTED_K. |
| RUN_IN_TMUX | When True, generate scripts and print a terminal command instead of running the sweep or selected-K stage in Jupyter. |
| OVERWRITE_EXPORT | Rebuild an existing input export. Leave this False unless the replacement is intentional. |

For the established templates, `SELECTED_K=None` uses the reviewed configured
value: K=18 for astrocytes, K=25 for oligodendrocytes, K=19 for inhibitory
neurons, K=25 for excitatory neurons, and K=26 for EC/PVF/PERICYTE cells. The
sweep remains independent and currently uses K=10 through K=30 for these
lineages.

## Recommended two-pass workflow

### 1. Define and preview the cell universe

Set the selection controls while leaving every run flag False. Execute the
parameter and preview cells. Review the reported counts by cell type, sample,
condition, and compartment before launching work.

Example: all microglia across every compartment and condition:

    LINEAGE = "astrocyte"  # use its numerical settings as a template
    ANALYSIS_NAME = "microglia_cluster_sub_all"
    CELL_TYPE_KEY = None
    CELL_TYPES = ["MG"]
    COMPARTMENTS = "all"
    CONDITIONS = "all"
    SELECTED_K = None
    WORKERS = None

    RUN_EXPORT = False
    RUN_SWEEP = False
    RUN_SELECTED_CONSENSUS = False
    RUN_IN_TMUX = False
    OVERWRITE_EXPORT = False

Here, LINEAGE="astrocyte" supplies numerical defaults such as the sweep range,
number of replicates, number of genes, seed, and worker count. It does not
select astrocytes because CELL_TYPES=["MG"] replaces that selection.

### 2. Export counts and run the K sweep

After confirming the preview, use:

    SELECTED_K = None

    RUN_EXPORT = True
    RUN_SWEEP = True
    RUN_SELECTED_CONSENSUS = False
    RUN_IN_TMUX = False
    OVERWRITE_EXPORT = False

Rerun the parameter cell and continue through the export, sweep, status, and
K-selection cells. SELECTED_K=None is appropriate here because selected K does
not affect the sweep.

For an unattended sweep, use:

    RUN_EXPORT = False
    RUN_SWEEP = True
    RUN_SELECTED_CONSENSUS = False
    RUN_IN_TMUX = True

The sweep cell then writes `run_sweep.sh` and `launch_sweep_tmux.sh` under the
analysis `launchers/` directory. It prints a `bash .../launch_sweep_tmux.sh`
command to paste into a terminal. The terminal job exports counts first if the
input export is absent, then runs or resumes the sweep. The notebook does not
start tmux itself.

The tmux job writes the K-selection statistics and plot to disk, but it cannot
update an already-finished notebook cell. After the terminal job completes,
return to Notebook 05 and set:

    RUN_EXPORT = False
    RUN_SWEEP = False
    RUN_SELECTED_CONSENSUS = False
    RUN_IN_TMUX = False

Rerun the parameter cell and the K-sweep cell. With `RUN_SWEEP=False`, the cell
does not relaunch cNMF; it detects the completed artifacts and displays the
K-selection table and plot.

The K-selection view plots stability and prediction error for every configured
K. Choosing K is a scientific decision; the notebook does not automatically
replace SELECTED_K from the plot.

### 3. Run consensus at the chosen K

After choosing a value, keep the exact same ANALYSIS_NAME, CELL_TYPES,
COMPARTMENTS, and CONDITIONS, then change the controls to:

    SELECTED_K = 12  # example; replace with the chosen K

    RUN_EXPORT = False
    RUN_SWEEP = False
    RUN_SELECTED_CONSENSUS = True
    RUN_IN_TMUX = False
    OVERWRITE_EXPORT = False

Rerun the parameter cell and the selected-consensus section. The selected-K
collection is intentionally separate from the sweep and normally uses more
replicates. A completed sweep is not a completed consensus analysis.

Set `RUN_IN_TMUX=True` to generate the analogous `run_selected.sh` and
`launch_selected_tmux.sh` scripts rather than blocking the notebook kernel.
The generated command includes the selected K, worker count, annotation key,
cell types, compartments, conditions, and analysis name.

The selected-K tmux job likewise writes consensus artifacts without updating
the notebook display. After it finishes, reset every run flag to `False`, then
open Notebook 06 with the same `LINEAGE`, `ANALYSIS_NAME`, `CELL_TYPE_KEY`,
`CELL_TYPES`, `COMPARTMENTS`, `CONDITIONS`, and `SELECTED_K`. Notebook 06 reads
the completed artifacts and displays the consensus tables, usage plots,
heatmaps, and spatial views; it does not rerun cNMF.

### 4. Return all run flags to False

After the run finishes, reset all three run flags to False. This prevents an
accidental launch when the notebook is reopened or executed from the top.

## Selecting compartments, conditions, and multiple cell types

All compartments and all conditions:

    CELL_TYPES = ["AST-CX", "AST-TH"]
    COMPARTMENTS = "all"
    CONDITIONS = "all"

Several compartments:

    ANALYSIS_NAME = "astrocytes_domains_3_7"
    CELL_TYPES = ["AST-CX", "AST-TH"]
    COMPARTMENTS = ["3", "7"]
    CONDITIONS = "all"

One condition and one compartment:

    ANALYSIS_NAME = "microglia_vap_igg_domain_3"
    CELL_TYPES = ["MG"]
    COMPARTMENTS = ["3"]
    CONDITIONS = ["vap_igg"]

Cell types are matched exactly; substring and regular-expression matching are
not used. For a lineage, explicitly list every included label.

## Notebook 06 inspection

Open Notebook 06 after selected-K consensus is complete. Reproduce the same
result-selection controls used in Notebook 05:

    LINEAGE = "astrocyte"
    ANALYSIS_NAME = "microglia_cluster_sub_all"
    CELL_TYPE_KEY = "cluster_sub"
    USAGE_CELL_TYPE_KEY = "cluster_sub"
    REVIEW_CELL_TYPES = ["MG"]
    CELL_TYPES = ["MG"]
    COMPARTMENTS = "all"
    CONDITIONS = "all"
    SELECTED_K = 18

Notebook 06 is read-only. It checks completion and then provides:

- the K-selection stability and prediction-error plot;
- consensus usage, gene-score, gene-TPM, and top-gene tables;
- biological-sample-level usage comparisons;
- a condition-change table across every selected cell-type × usage-program
  combination;
- a sample violin/table plot for the best corrected result or an explicitly
  requested cell-type, usage, and contrast;
- grouped usage heatmaps; and
- spatial Plotly program maps.

The condition plots summarize at the biological-sample level rather than
treating individual cells as independent replicates.

### Condition-associated usage controls

Notebook 06 exposes these controls immediately above the new test table:

    # Each tuple is (condition A, condition B); delta = A - B.
    USAGE_CONDITION_CONTRASTS = [
        ("vap_ab", "vap_igg"),
        ("vap_igg", "naive_igg"),
        ("vap_ab", "naive_igg"),
    ]
    MIN_USAGE_CELLS_PER_SAMPLE = 10
    MIN_USAGE_SAMPLES_PER_CONDITION = 3

    PLOT_USAGE_CONTRAST = ("vap_ab", "vap_igg")
    PLOT_USAGE_CELL_TYPE = None
    PLOT_USAGE_PROGRAM = None
    PLOT_USAGE_SORT_BY = "welch_q_global"

`PLOT_USAGE_CELL_TYPE=None` and `PLOT_USAGE_PROGRAM=None` select the best
corrected result within `PLOT_USAGE_CONTRAST`. Set both explicitly to inspect a
planned combination:

    PLOT_USAGE_CELL_TYPE = "MG"
    PLOT_USAGE_PROGRAM = "Usage_9"

Condition labels and colors are controlled by `USAGE_CONDITION_LABELS` and
`USAGE_CONDITION_PALETTE`. The violin layer shows the cell-level distribution
within each sample, but the diamond, test, and reported p-values use one mean
per biological sample.

For every requested contrast, the table reports:

- mean usage in each condition and the A-minus-B effect;
- retained sample counts and minimum cell support;
- sample-level Welch p-values;
- two-sided sample-label permutation p-values;
- Benjamini-Hochberg q-values within each contrast; and
- global q-values across every requested contrast.

Welch's unequal-variance t-test is reasonable as a sample-level summary, but it
is not sufficient by itself for discovery with four samples per condition.
Its variance estimate is unstable at that sample size. The permutation result
is a useful finite-sample check, although a 4-versus-4 exhaustive two-sided test
has only 70 allocations and cannot produce p-values below
`2 / 70 = 0.028571`. Multiple-testing correction is required whenever the
result is selected after scanning many programs or cell types. It does not
destroy significance; it prevents the best-looking random result from being
reported as though it were pre-specified.

If one cell-type × program × contrast hypothesis was specified before looking
at this table, its raw sample-level test can be reported as a planned test,
alongside its effect and sample distribution. If the notebook chooses the most
significant row, use the appropriate q-value and describe the result as an
exploratory screen.

## Resuming and protecting outputs

Every stage is resumable. Reusing the same controls and enabling an unfinished
stage skips completed artifacts. The workflow protects the cell universe and
numerical settings with manifests and refuses to silently combine incompatible
runs.

Important resume rules:

1. Keep the same ANALYSIS_NAME and selection controls.
2. Keep the same worker count for an existing sweep or selected-K collection.
3. Use a new ANALYSIS_NAME when changing cell types, annotation key,
   compartments, or conditions.
4. Do not start the same collection from two notebook or CLI sessions. A
   process lock rejects duplicate launch attempts.
5. Leave OVERWRITE_EXPORT=False unless deliberately replacing the prepared
   input.

## Outputs

Each analysis is written under:

    results/ab_xenium/05_cnmf/<analysis-name>/

The principal directories are:

    input/                 exact raw-count selection and provenance
    sweep/                 K sweep, merged spectra, statistics, and K plot
    selected_k<K>_i<N>/    high-replicate consensus and usage-enriched H5AD
    launchers/              generated cNMF jobs, tmux launchers, and tmux logs

Notebook 06 reads these artifacts directly; it does not rerun cNMF.

## Notebook 08 condition-blind whitelist review

Notebook 08 reviews one lineage-program pair at a time before the predictor
whitelist is frozen. It anonymizes samples as `Section_XX` and does not expose
condition, exposure, treatment, group, condition comparisons, p-values, or
model feature importance. For each program it provides:

- exact top-fraction high-usage cells and their QC/annotation fields;
- section-concentration warnings;
- raw-count support for the proposed genes;
- a spatial map in a blinded section;
- neighboring-cell-type context around high-usage cells; and
- descriptive correlation with the lineage's other programs.

The default save flag is false. When a review is complete, the explicit save
cell upserts a separate human-decision table without modifying the whitelist.

For review, compare cells within one program using ranks or percentiles. For
modeling, retain the native cNMF values and standardize each constructed focal
or neighborhood feature using only the training folds. Do not renormalize only
the retained programs to sum to one, because removing excluded factors from
the denominator would artificially inflate the remaining usages. Regularized
regression does not correct unequal feature scales automatically.

## Common problems

### The preview returns zero cells

Confirm that cnmf.cell_type_key names the intended adata.obs column and that
every CELL_TYPES entry is an exact value from that column. Also verify the
selected compartments and conditions.

### Consensus uses an unexpected K

SELECTED_K=None uses the configured lineage-template K. Enter the chosen
integer explicitly before setting RUN_SELECTED_CONSENSUS=True.

### A manifest mismatch is reported

The analysis name already belongs to a different cell universe or parameter
set. Restore the original settings to resume it, or use a new unique
ANALYSIS_NAME.

### The inspection notebook says consensus is incomplete

Finish the selected-K stage in Notebook 05. Completing only the sweep does not
produce consensus usages.

### A collection is locked

Another process is already running that exact sweep or selected-K collection.
Wait for it to finish and then rerun the status cell.
