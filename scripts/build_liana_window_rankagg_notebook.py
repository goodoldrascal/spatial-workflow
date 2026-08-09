#!/usr/bin/env python3
"""Build Notebook 07 for whole-sample adaptive-window LIANA rank aggregation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

from notebook_presentation import apply_notebook_presentation


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "07_liana_window_rankagg.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
            # 07 · Whole-sample LIANA adaptive-window rank aggregation

            By default this notebook runs LIANA across **all spatial domains/compartments
            together**. It uses the original adaptive spatial-grid estimator and
            does **not** construct or consume graph windows. Candidate centers are
            laid on a regular spatial grid; each context contains the nearest
            `adaptive_k` cells. Overlapping contexts are collapsed to one edge
            vector per biological sample before condition statistics.

            The default LR resource matches CellChat v2 `subsetDB(CellChatDB)`:
            Secreted Signaling, ECM-Receptor, and Cell-Cell Contact are retained,
            while Non-protein Signaling is excluded unless explicitly requested.
            In the current local file this retains 2,377 metadata rows, 2,365
            distinct gene-complex tests, and 254 pathways, including WNT. LIANA still
            applies assay-gene availability, complex-subunit, `expr_prop`, and
            `min_cells` checks, so “all CellChat LRs” means the tested resource
            universe—not that every LR must be emitted in every window.
            The current file has two gene-complex identities shared by the `Aldo`
            and `Cort` pathway rows. They remain in edge-level results but are
            excluded from pathway aggregation rather than being double-counted or
            assigned to an arbitrary pathway.
            """
        ),
        code(
            """
            import json
            import os
            import shlex
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import liana as li
            from IPython.display import Image, Markdown, display

            REPO_ROOT = Path.cwd().resolve()
            if REPO_ROOT.name == "notebooks":
                REPO_ROOT = REPO_ROOT.parent
            SOURCE_ROOT = str(REPO_ROOT / "src")
            if SOURCE_ROOT not in sys.path:
                sys.path.insert(0, SOURCE_ROOT)

            from spatial_workflow.config import load_config, resolve_path
            from spatial_workflow.liana_edge_features import top_edge_features
            from spatial_workflow.liana_window import (
                WindowParameters,
                analyze_window_rankagg,
                audit_liana_input,
                liana_window_status,
                load_cellchat_resource,
                load_rankagg_resource,
                load_window_analysis,
                parameter_table,
                prepare_chord_input,
                render_chord,
                run_window_rankagg,
            )
            from spatial_workflow.liana_window_plotting import (
                plot_pathway_driver_summary,
                plot_pathway_ranking,
            )

            CONFIG_PATH = REPO_ROOT / "configs" / "local.yaml"
            config = load_config(CONFIG_PATH)
            schema = config["schema"]
            results_root = resolve_path(CONFIG_PATH, config["paths"]["results_root"])
            cellcharter = config["cellcharter"]
            INPUT_H5AD = resolve_path(
                CONFIG_PATH,
                Path(cellcharter["output_dir"]) / f"{cellcharter['output_name']}.h5ad",
                root=results_root,
            )
            OUTPUT_DIR = results_root / "07_liana_window_rankagg"
            ANALYSIS_DIR = OUTPUT_DIR / "analysis"
            CELLCHAT_CSV = Path(
                os.environ.get(
                    "CELLCHAT_CSV",
                    REPO_ROOT / "resources" / "CellChatDB_interaction.csv",
                )
            )
            CHORD_SCRIPT = REPO_ROOT / "scripts" / "plot_liana_window_chord.R"

            print("input:", INPUT_H5AD)
            print("output:", OUTPUT_DIR)
            print("CellChat:", CELLCHAT_CSV)
            """
        ),
        markdown(
            """
            ## Analysis scope and workflow parameters

            `CELL_TYPES` and `COMPARTMENTS` restrict cells before any window is
            built. `COMPARTMENTS=None` is a true whole-sample analysis: all cells in
            each biological sample are eligible and window construction does not use
            compartment labels. An explicit compartment list filters cells first;
            the selected domains are then pooled rather than analyzed as separate
            runs. The LR space can be the local
            CellChat table, an arbitrary custom CSV with `ligand` and `receptor`
            columns, or a native LIANA resource such as `mouseconsensus`. Pathway
            summaries require pathway metadata: CellChat supplies it, while a
            custom CSV may provide `pathway_name`; native `mouseconsensus` itself
            has no pathway labels. Filters in different LR dimensions are ANDed;
            multiple choices within one dimension are ORed.


            The defaults reproduce the non-graph adaptive contract. `min_cells=5`
            is the parameter you remembered: within a window, each cell identity
            must have at least five cells to participate as a LIANA source or
            target. A window is retained only if at least two identities meet that
            threshold.

            `n_perms=200` belongs to LIANA inside each window. In LIANA 1.7.1,
            the default rank-aggregate consensus permutes whole cell expression
            profiles against the fixed cell-type label slots, preserving group
            sizes but breaking the expression/identity association. It recomputes
            group means and asks, one-sided, how often a permuted ligand-source /
            receptor-target mean is at least the observed score. In this installed
            consensus only the CellPhoneDB component uses those permutations; they
            feed `cellphone_pvals` and then `specificity_rank`. They do not test
            condition labels, move coordinates, rebuild windows, or determine the
            downstream `magnitude_rank` condition p-values. Spatial proximity is a
            fixed source-target weight applied to observed and permuted scores.
            With 200 draws the Monte Carlo p-value grid is 0.005, and this
            LIANA implementation can return 0 because it uses `count / n_perms`
            without a plus-one correction. Also, choosing CellChatDB here selects
            the LR catalog; it does not switch the consensus method to CellChat.



            Optional anchor windows are disabled here. The earlier
            `AST_sub11/12/13` rescue families were a compartment-specific analysis
            decision. Add exact labels to `anchor_groups` only if the whole-sample
            analysis needs a predeclared sparse-cell rescue.
            """
        ),
        code(
            """
            # None means all available values. These filters are applied before windows.
            CELL_TYPES = None          # e.g. ["Astrocyte", "Microglia"]
            COMPARTMENTS = None        # true whole sample; or e.g. ["0", "3"] pooled

            # LR_RESOURCE_MODE: "cellchat", "custom", or "liana".
            LR_RESOURCE_MODE = "cellchat"
            CUSTOM_LR_CSV = None        # CSV with ligand/receptor columns
            LIANA_RESOURCE_NAME = "mouseconsensus"
            LR_PATHWAYS = None          # pathway_name values; CellChat/annotated custom only
            LR_ANNOTATIONS = None       # annotation values; CellChat/annotated custom only
            LR_PAIRS = None             # exact ligand^receptor strings
            INCLUDE_NON_PROTEIN = False # CellChat v2 default; True adds metabolic/synaptic rows
            # Pathway annotation used downstream; for an annotated custom LR CSV,
            # set this to CUSTOM_LR_CSV. Native mouseconsensus has no pathways, so
            # the CellChat default annotates only its overlapping pairs.
            PATHWAY_METADATA_CSV = CELLCHAT_CSV


            PARAMETERS = WindowParameters(
                adaptive_k=60,
                grid_stride=100.0,
                expr_prop=0.1,
                min_cells=5,
                min_required_groups=2,
                spatial_bandwidth=250.0,
                spatial_kernel="gaussian",
                spatial_trim_fraction=0.1,
                n_perms=200,
                seed=1337,
                n_jobs=8,
                anchor_groups=(),
                anchor_k_target=60,
                anchor_min_cells=2,
                anchor_max_radius=150.0,
                anchor_dedup_distance=60.0,
                return_all_lrs=False,
            )
            display(parameter_table(PARAMETERS))

            resource, resource_metadata = load_rankagg_resource(
                li,
                mode=LR_RESOURCE_MODE,
                cellchat_csv=CELLCHAT_CSV,
                custom_lr_csv=CUSTOM_LR_CSV,
                liana_resource_name=LIANA_RESOURCE_NAME,
                pathways=LR_PATHWAYS,
                annotations=LR_ANNOTATIONS,
                lr_pairs=LR_PAIRS,
                include_non_protein=INCLUDE_NON_PROTEIN,
            )
            display(
                pd.Series(
                    {
                        "selected resource metadata rows": len(resource_metadata),
                        "distinct LIANA gene-complex tests": len(resource),
                        "annotated pathways": resource_metadata["pathway_name"].nunique(),
                        "annotations": resource_metadata["annotation"].nunique(),
                    },
                    name="resource audit",
                ).to_frame()
            )
            """
        ),
        markdown("## Read-only input audit after applying the requested scope"),
        code(
            """
            RUN_INPUT_AUDIT = True
            if RUN_INPUT_AUDIT:
                input_audit = audit_liana_input(
                    INPUT_H5AD,
                    sample_key=schema["sample_key"],
                    condition_key=schema["condition_key"],
                    group_key=schema["cell_type_key"],
                    spatial_key=schema["spatial_key"],
                    compartment_key=schema.get("spatial_domain_key"),
                    cell_types=CELL_TYPES,
                    compartments=COMPARTMENTS,
                )
                display(input_audit["shape"].to_frame("value"))
                display(input_audit["samples"])
                display(input_audit["cell_groups"].head(50))
            """
        ),
        markdown(
            """
            ## Run or resume sample-level LIANA

            This is the expensive stage. Outputs and manifests are written per
            biological sample, so a rerun skips completed samples unless
            `OVERWRITE_LIANA=True`. Use `SAMPLES=[...]` for a one-sample smoke test;
            `SAMPLES=None` schedules every sample. `LIMIT_WINDOWS=2` is useful only
            for a smoke test and must not be mixed with production outputs.

            `LAUNCH_TMUX=True` is the default. After reviewing the printed CLI
            preview, set `RUN_LIANA=True` to create a detached tmux session and log.
            Set `LAUNCH_TMUX=False` only when an in-kernel run is intentional. The
            cell refuses to reuse an existing tmux session name.


            For unattended work, the equivalent CLI is

            ```bash
            python3 \
              scripts/run_liana_window_rankagg.py run \
              --config configs/local.yaml \
              --cellchat-csv /path/to/CellChatDB_interaction.csv
            ```
            """
        ),
        code(
            """
            RUN_LIANA = False
            LAUNCH_TMUX = True         # default: detach the expensive run
            TMUX_SESSION = "liana_window_rankagg"
            TMUX_LOG = OUTPUT_DIR / "logs" / f"{TMUX_SESSION}.log"
            SAMPLES = None             # e.g. ["vap_46_igg"] for a smoke test
            LIMIT_WINDOWS = None       # e.g. 2 for smoke testing only
            OVERWRITE_LIANA = False

            def _append_repeated(arguments, flag, values):
                for value in values or []:
                    arguments.extend([flag, str(value)])

            cli_arguments = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_liana_window_rankagg.py"),
                "--config",
                str(CONFIG_PATH),
                "run",
                "--input-h5ad",
                str(INPUT_H5AD),
                "--output-dir",
                str(OUTPUT_DIR),
                "--resource-mode",
                LR_RESOURCE_MODE,
                "--liana-resource-name",
                LIANA_RESOURCE_NAME,
                "--adaptive-k",
                str(PARAMETERS.adaptive_k),
                "--grid-stride",
                str(PARAMETERS.grid_stride),
                "--expr-prop",
                str(PARAMETERS.expr_prop),
                "--min-cells",
                str(PARAMETERS.min_cells),
                "--min-required-groups",
                str(PARAMETERS.min_required_groups),
                "--spatial-bandwidth",
                str(PARAMETERS.spatial_bandwidth),
                "--spatial-kernel",
                PARAMETERS.spatial_kernel,
                "--spatial-trim-fraction",
                str(PARAMETERS.spatial_trim_fraction),
                "--n-perms",
                str(PARAMETERS.n_perms),
                "--seed",
                str(PARAMETERS.seed),
                "--n-jobs",
                str(PARAMETERS.n_jobs),
                "--anchor-k-target",
                str(PARAMETERS.anchor_k_target),
                "--anchor-min-cells",
                str(PARAMETERS.anchor_min_cells),
                "--anchor-max-radius",
                str(PARAMETERS.anchor_max_radius),
                "--anchor-dedup-distance",
                str(PARAMETERS.anchor_dedup_distance),
            ]
            if LR_RESOURCE_MODE == "cellchat":
                cli_arguments.extend(["--cellchat-csv", str(CELLCHAT_CSV)])
            elif LR_RESOURCE_MODE == "custom":
                if CUSTOM_LR_CSV is None:
                    raise ValueError("CUSTOM_LR_CSV is required for custom LR mode")
                cli_arguments.extend(["--custom-lr-csv", str(CUSTOM_LR_CSV)])
            _append_repeated(cli_arguments, "--sample", SAMPLES)
            _append_repeated(cli_arguments, "--cell-type", CELL_TYPES)
            _append_repeated(cli_arguments, "--compartment", COMPARTMENTS)
            _append_repeated(cli_arguments, "--pathway", LR_PATHWAYS)
            _append_repeated(cli_arguments, "--annotation", LR_ANNOTATIONS)
            _append_repeated(cli_arguments, "--lr-pair", LR_PAIRS)
            _append_repeated(cli_arguments, "--anchor-group", PARAMETERS.anchor_groups)
            if INCLUDE_NON_PROTEIN:
                cli_arguments.append("--include-non-protein")
            if PARAMETERS.return_all_lrs:
                cli_arguments.append("--return-all-lrs")
            if OVERWRITE_LIANA:
                cli_arguments.append("--overwrite")
            if LIMIT_WINDOWS is not None:
                cli_arguments.extend(["--limit-windows-per-sample", str(LIMIT_WINDOWS)])

            print("CLI preview:", shlex.join(cli_arguments))
            if RUN_LIANA and LAUNCH_TMUX:
                TMUX_LOG.parent.mkdir(parents=True, exist_ok=True)
                session_exists = subprocess.run(
                    ["tmux", "has-session", "-t", TMUX_SESSION],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode == 0
                if session_exists:
                    raise RuntimeError(f"tmux session already exists: {TMUX_SESSION}")
                shell_command = (
                    f"cd {shlex.quote(str(REPO_ROOT))} && "
                    f"{shlex.join(cli_arguments)} 2>&1 | tee {shlex.quote(str(TMUX_LOG))}"
                )
                subprocess.run(
                    ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "bash", "-lc", shell_command],
                    check=True,
                )
                display(pd.Series({
                    "tmux_session": TMUX_SESSION,
                    "log": str(TMUX_LOG),
                    "attach": f"tmux attach -t {TMUX_SESSION}",
                }))
            elif RUN_LIANA:
                run_status = run_window_rankagg(
                    INPUT_H5AD,
                    CELLCHAT_CSV,
                    OUTPUT_DIR,
                    sample_key=schema["sample_key"],
                    condition_key=schema["condition_key"],
                    group_key=schema["cell_type_key"],
                    spatial_key=schema["spatial_key"],
                    compartment_key=schema.get("spatial_domain_key"),
                    cell_types=CELL_TYPES,
                    compartments=COMPARTMENTS,
                    pathways=LR_PATHWAYS,
                    annotations=LR_ANNOTATIONS,
                    lr_pairs=LR_PAIRS,
                    resource_mode=LR_RESOURCE_MODE,
                    custom_lr_csv=CUSTOM_LR_CSV,
                    liana_resource_name=LIANA_RESOURCE_NAME,
                    include_non_protein=INCLUDE_NON_PROTEIN,
                    parameters=PARAMETERS,
                    samples=SAMPLES,
                    overwrite=OVERWRITE_LIANA,
                    limit_windows_per_sample=LIMIT_WINDOWS,
                )
                display(run_status)
            else:
                print("Set RUN_LIANA=True; LAUNCH_TMUX=True is the default execution mode.")

            artifact_status = liana_window_status(OUTPUT_DIR)
            display(artifact_status if not artifact_status.empty else pd.DataFrame(
                {"status": ["No completed sample manifests yet"]}
            ))
            """
        ),
        markdown(
            """
            ## Edge universe, missing values, and significance contract

            The analysis makes the support decision **before** calculating
            pairwise statistics:

            1. An edge is `(source, target, ligand_complex, receptor_complex)`.
            2. Keep it globally if it is observed in at least
               `MIN_CONDITION_SAMPLES=3` biological samples of any included
               condition. This reproduces the prior “three samples in one
               condition” rule; change it to 4 for a stricter all-replicate screen.
            3. For a particular contrast, test it only if either side reaches that
               support threshold. A third-condition-only edge is retained in the
               global audit but is not tested in an unrelated contrast.
            4. `ZERO_FILL_MISSING=True` is the default. It completes each retained
               edge across biological samples: a missing normalized
               `magnitude_rank` becomes 1, hence activity `1 - rank` becomes 0.
               Raw score columns are never indiscriminately imputed. If set to
               `False`, missing ranks remain NA; an edge is tested only when both
               conditions meet the support threshold, and pathway RMS/mean tests
               use edges observed in every sample of that contrast.
            5. Preserve observed-support counts and label `appears_in_b`,
               `disappears_in_b`, and partial-support cases. Thus a zero-filled
               effect never loses its appearance/disappearance provenance.
            6. The primary edge test pools the biological-sample activities from
               the two conditions and enumerates every allocation of `n_a` samples
               to condition A. For each allocation it recomputes `mean(B)-mean(A)`;
               the exact two-sided p-value is the fraction whose absolute null
               difference is at least the observed absolute difference. Cells and
               windows are never treated as replicates. This tests exchangeability
               of independent biological samples under the null.
            7. Welch is retained only as a secondary parametric sensitivity check
               on the same sample-level values. It allows unequal variances and
               uses a Satterthwaite degrees-of-freedom approximation, but with 3-4
               samples per condition it has little power and unstable variance
               estimates; structural zeros can make it especially unhelpful.
            8. BH correction is applied separately within each condition contrast
               and test family across all tested edges. It converts raw p-values
               to q-values that control the expected false-discovery proportion;
               it does not rescue the coarse exact-test resolution. With 4 versus
               4 samples there are 70 allocations and the two-sided floor is
               `2/70 = 0.02857`. Therefore exact effect sizes, support/provenance,
               raw p-values, and BH q-values should be reviewed together.

            `CONDITIONS` may contain two or three names. Every pairwise combination
            is generated automatically.
            """
        ),
        code(
            """
            CONDITIONS = ["naive_igg", "vap_igg", "vap_ab"]
            MIN_CONDITION_SAMPLES = 3
            RANK_COLUMN = "magnitude_rank"
            ZERO_FILL_MISSING = True   # default; False uses observed/complete cases
            RUN_PAIRWISE_ANALYSIS = False
            OVERWRITE_ANALYSIS = False

            if RUN_PAIRWISE_ANALYSIS:
                analysis_tables = analyze_window_rankagg(
                    OUTPUT_DIR,
                    PATHWAY_METADATA_CSV,
                    ANALYSIS_DIR,
                    conditions=CONDITIONS,
                    min_condition_samples=MIN_CONDITION_SAMPLES,
                    rank_column=RANK_COLUMN,
                    zero_fill_missing=ZERO_FILL_MISSING,
                    include_non_protein=INCLUDE_NON_PROTEIN,
                    overwrite=OVERWRITE_ANALYSIS,
                )
                display(
                    pd.Series(
                        {name: table.shape for name, table in analysis_tables.items()},
                        name="shape",
                    ).to_frame()
                )
            else:
                print("Set RUN_PAIRWISE_ANALYSIS=True after all intended samples finish.")
            """
        ),
        markdown(
            """
            ## Edge-level review, including appearance/disappearance

            `tested` marks the pair-specific support gate. The primary table keeps
            zero-filled means, exact p/q, Welch p/q, support counts, and the explicit
            presence class together.
            """
        ),
        code(
            """
            analysis_tables = load_window_analysis(ANALYSIS_DIR) if (
                ANALYSIS_DIR / "manifest.json"
            ).exists() else None

            if analysis_tables is not None:
                edge_pairwise = analysis_tables["edge_pairwise"]
                selected_contrast = edge_pairwise["contrast"].iloc[0]
                edge_review = edge_pairwise[
                    edge_pairwise["contrast"].eq(selected_contrast)
                    & edge_pairwise["tested"]
                ].sort_values(
                    ["exact_permutation_p", "abs_difference"],
                    ascending=[True, False],
                )
                display(edge_review.head(50))
                display(
                    edge_review.groupby("presence_class", observed=True)
                    .agg(
                        n_edges=("edge_id", "nunique"),
                        median_abs_difference=("abs_difference", "median"),
                        raw_p_lt_0_05=("exact_permutation_p", lambda x: (x < 0.05).sum()),
                        fdr_q_lt_0_05=("exact_permutation_q", lambda x: (x < 0.05).sum()),
                    )
                    .sort_values("n_edges", ascending=False)
                )
            else:
                print("Analysis tables are not present yet.")
            """
        ),
        markdown(
            """
            ## Top significant edge features

            `top_edge_features` filters one contrast by raw exact permutation
            p-value and returns seven independently displayable count tables:
            ligands, receptors, LR pairs, pathways, sources, targets, and
            source-target pairs. Counts are unique significant `edge_id` values. The cell below runs it for both treatment-versus-naive contrasts.
            """
        ),
        code(
            """
            EDGE_FEATURE_CONTRASTS = (
                "vap_igg_vs_naive_igg",
                "vap_ab_vs_naive_igg",
            )
            EDGE_FEATURE_THRESHOLD = 0.05
            EDGE_FEATURE_TOP_N = 20

            edge_feature_tables_by_contrast = None
            if analysis_tables is not None:
                edge_feature_tables_by_contrast = {}
                for contrast in EDGE_FEATURE_CONTRASTS:
                    display(Markdown(f"### {contrast}"))
                    feature_tables = top_edge_features(
                        analysis_tables["edge_pairwise"],
                        contrast=contrast,
                        threshold=EDGE_FEATURE_THRESHOLD,
                        top_n=EDGE_FEATURE_TOP_N,
                    )
                    edge_feature_tables_by_contrast[contrast] = feature_tables
                    for table_name, table in feature_tables.items():
                        display(Markdown(f"#### {table_name.replace("_", " ").title()}"))
                        display(table)
            else:
                print("Analysis tables are not present yet.")
            """
        ),
        markdown(
            """
            ## Pathway changes and their drivers

            Both pathway views are retained:

            - **mean activity** averages retained edge activity within each
              sample/pathway, then compares condition means by exact permutation;
            - **RMS centroid distance** treats a pathway as its sample-by-edge
              vector and tests the multivariate separation by exact permutation.

            At pathway level, the mean-activity exact test permutes sample labels
            and recomputes the scalar pathway mean difference. The RMS exact test
            permutes those same labels but recomputes both multivariate condition
            centroids and their RMS edge-space distance for every allocation.
            Welch is defined only for the scalar mean-activity view; there is no
            Welch analogue for the RMS statistic here. BH is run separately for
            mean-exact, mean-Welch, and RMS-exact p-values within each contrast.

            RMS distance is the primary ranking because it preserves coordinated
            edge changes that cancel in the mean. Mean activity remains as the
            directional summary. Driver tables decompose each pathway contrast at
            three levels: individual directed edges, CellChat LR pairs aggregated
            over cell-state pairs, and source-target cell-state pairs aggregated
            over LR identities. `absolute_effect_fraction` and
            `rms_squared_fraction` show which rows contribute to the pathway shift.

            The ranking plot can order pathways by RMS centroid distance, signed
            mean-activity difference, mean-activity exact p, or RMS-centroid exact
            p. RMS distance is the default because it remains continuous when many
            pathways share the exact-p floor. Effect ranking uses the absolute
            difference but keeps signed bars. P-value ranking displays `-log10(p)`
            while bar color retains the direction of the mean change. The default
            review retains a pathway when either mean-activity exact p or RMS exact
            p is strictly below 0.05, then ranks that union by RMS distance.
            """
        ),
        code(
            """
            PATHWAY_RANK_BY = "rms_centroid_distance"  # also rms_exact_p / mean_activity_exact_p / mean_activity_difference_b_minus_a
            PATHWAY_TOP_N = 20
            PATHWAY_EXACT_P_MAX = 0.05  # mean-activity OR RMS exact p must be strictly below this

            if analysis_tables is not None:
                pathway_pairwise = analysis_tables["pathway_pairwise"]
                pathway_rank_figure, pathway_rank_axis, pathway_review = plot_pathway_ranking(
                    pathway_pairwise,
                    contrast=selected_contrast,
                    rank_by=PATHWAY_RANK_BY,
                    top_n=PATHWAY_TOP_N,
                    exact_p_max=PATHWAY_EXACT_P_MAX,
                )
                selected_pathway = pathway_review.iloc[0]["pathway_name"]
                display(pathway_rank_figure)
                plt.close(pathway_rank_figure)
                display(
                    pathway_review[
                        [
                            "display_rank",
                            "pathway_name",
                            "mean_activity_difference_b_minus_a",
                            "mean_activity_exact_p",
                            "rms_centroid_distance",
                            "rms_exact_p",
                        ]
                    ]
                )
            else:
                print("Analysis tables are not present yet.")
            """
        ),
        markdown(
            """
            ## Selected pathway drivers and directed edges

            The left panel ranks CellChat ligand-receptor identities by their
            share of the pathway's total absolute edge change; bar color shows
            whether their net change favors condition B or A. The right panel is
            the sparse source-target by ligand-receptor edge grid: color is signed
            activity change and bubble size is `-log10(edge exact p)`. Set
            `DRIVER_EDGE_P_MAX=None` to retain the top edges regardless of nominal
            p-value.
            """
        ),
        code(
            """
            SELECTED_PATHWAY = "ICAM"  # set to None to use the top-ranked pathway
            DRIVER_TOP_LR = 10
            DRIVER_TOP_EDGES = 35
            DRIVER_EDGE_P_MAX = 0.05

            if analysis_tables is not None:
                selected_pathway = SELECTED_PATHWAY or pathway_review.iloc[0]["pathway_name"]
                driver_figure, driver_axes, pathway_lr_review, pathway_edge_review = (
                    plot_pathway_driver_summary(
                        analysis_tables["pathway_lr_drivers"],
                        analysis_tables["pathway_edge_drivers"],
                        analysis_tables["edge_pairwise"],
                        contrast=selected_contrast,
                        pathway_name=selected_pathway,
                        top_lr=DRIVER_TOP_LR,
                        top_edges=DRIVER_TOP_EDGES,
                        edge_p_max=DRIVER_EDGE_P_MAX,
                    )
                )
                display(driver_figure)
                plt.close(driver_figure)
                display(
                    pathway_lr_review[
                        [
                            "lr_label",
                            "absolute_effect_percent",
                            "signed_effect_sum",
                            "direction_balance",
                        ]
                    ]
                )
                display(
                    pathway_edge_review[
                        [
                            "source",
                            "target",
                            "ligand",
                            "receptor",
                            "difference_b_minus_a",
                            "exact_permutation_p",
                            "exact_permutation_q",
                            "presence_class",
                        ]
                    ]
                )
                print("Selected pathway:", selected_pathway)
            else:
                print("Analysis tables are not present yet.")
            """
        ),
        markdown(
            """
            ## Directional chord plot

            `scripts/plot_liana_window_chord.R` contains the reusable plotting
            function and command-line entrypoint. It is the established
            R/circlize directional `big.arrow` geometry, generalized so arbitrary
            two- or three-condition labels receive colors automatically.

            By default, the cell below renders the currently selected pathway
            and contrast. Set `RENDER_RESULT_CHORD=False` only when you want to
            display the checked-in renderer demonstration instead of a result
            from this analysis.
            """
        ),
        code(
            """
            RENDER_RESULT_CHORD = True
            CHORD_TOP_N = 15
            if RENDER_RESULT_CHORD:
                if analysis_tables is None:
                    raise RuntimeError("Run or load the pairwise analysis before rendering a chord")
                chord_input = prepare_chord_input(
                    analysis_tables["pathway_edge_drivers"],
                    contrast=selected_contrast,
                    pathway_name=selected_pathway,
                    top_n=CHORD_TOP_N,
                )
                chord_dir = ANALYSIS_DIR / "figures/chords"
                rendered = render_chord(
                    chord_input,
                    script_path=CHORD_SCRIPT,
                    input_csv=chord_dir / f"{selected_contrast}__{selected_pathway}.csv",
                    output_path=chord_dir / f"{selected_contrast}__{selected_pathway}.png",
                    title=f"{selected_pathway}: {selected_contrast}",
                    rscript_bin=config.get("runtime", {}).get("rscript_bin", "Rscript"),
                )
                display(Image(filename=str(rendered), width=900))
            else:
                example_plot = REPO_ROOT / "docs/figures/liana_window_rankagg_example_chord.png"
                if example_plot.exists():
                    display(Image(filename=str(example_plot), width=900))
                else:
                    print("Example chord has not been rendered yet:", example_plot)
            """
        ),
        markdown(
            """
            ## Interpretation boundaries

            - Windows are local spatial contexts, not biological replicates and
              not CellCharter compartments. Statistical replication is the sample.
            - Overlap changes the sample-level spatial weighting even after windows
              are collapsed; it does not increase the condition sample size.
            - A zero-filled missing rank encodes “not emitted under the configured
              expression/support gates,” not proof that molecular signaling is
              literally absent. Use the support and presence columns in every claim.
            - With four samples per condition, exact raw p-values have coarse
              resolution. Emphasize effect size, direction, support class, and FDR;
              treat raw-p pathway rankings as discovery-oriented.
            """
        ),
    ]
    return apply_notebook_presentation(notebook, "07")


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
