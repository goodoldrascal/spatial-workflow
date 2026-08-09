#!/usr/bin/env python3
"""Refresh the maintained cNMF notebooks without discarding review outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from notebook_presentation import apply_notebook_presentation


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_NOTEBOOK = REPO_ROOT / "notebooks" / "05_cnmf_run.ipynb"
INSPECTION_NOTEBOOK = REPO_ROOT / "notebooks" / "06_cnmf_inspection.ipynb"


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Maintained notebook template is missing: {path}")
    return nbf.read(path, as_version=4)


def _find_cell(notebook, *, cell_type: str, needle: str):
    matches = [
        cell
        for cell in notebook.cells
        if cell.cell_type == cell_type and needle in cell.source
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {cell_type} cell containing {needle!r}; found {len(matches)}"
        )
    return matches[0]


def _clear_code_output(cell) -> None:
    cell.outputs = []
    cell.execution_count = None


def build_run_notebook():
    """Return the maintained execution notebook unchanged."""

    return apply_notebook_presentation(_load(RUN_NOTEBOOK), "05")


def build_inspection_notebook():
    notebook = _load(INSPECTION_NOTEBOOK)

    import_cell = _find_cell(
        notebook,
        cell_type="code",
        needle="from spatial_workflow.cnmf import (",
    )
    if "subset_usage_by_cell_type" not in import_cell.source:
        import_cell.source = import_cell.source.replace(
            "    summarize_usage,\n",
            "    summarize_usage,\n    subset_usage_by_cell_type,\n",
        )
        _clear_code_output(import_cell)
    if "from spatial_workflow.cnmf_usage_analysis import" not in import_cell.source:
        import_cell.source += """

from spatial_workflow.cnmf_usage_analysis import (
    select_usage_condition_result,
    test_usage_condition_changes,
)
from spatial_workflow.cnmf_usage_plotting import plot_usage_condition_violin
""".rstrip()
        _clear_code_output(import_cell)

    result_markdown = _find_cell(
        notebook,
        cell_type="markdown",
        needle="## Select a completed result",
    )
    if "USAGE_CELL_TYPE_KEY" not in result_markdown.source:
        result_markdown.source += (
            "\n\n`CELL_TYPE_KEY` identifies the cNMF run and its cell universe. "
            "`USAGE_CELL_TYPE_KEY` is the annotation column used for review "
            "grouping, and may be another retained column such as `cluster_sub`. "
            "`REVIEW_CELL_TYPES=None` includes every non-missing value of that "
            "column; an explicit list subsets every usage summary and boxplot."
        )

    selection_cell = _find_cell(
        notebook,
        cell_type="code",
        needle="LINEAGE = ",
    )
    if "USAGE_CELL_TYPE_KEY" not in selection_cell.source:
        selection_cell.source = selection_cell.source.replace(
            "CELL_TYPE_KEY = None\n",
            "CELL_TYPE_KEY = None\n"
            "USAGE_CELL_TYPE_KEY = None  # None uses resolved cNMF key; e.g. cluster_sub\n"
            "REVIEW_CELL_TYPES = None    # None includes all non-missing labels\n",
        )
        _clear_code_output(selection_cell)

    consensus_cell = _find_cell(
        notebook,
        cell_type="code",
        needle="usage, gene_scores, gene_tpm, top_genes",
    )
    if "resolved_usage_cell_type_key" not in consensus_cell.source:
        review_block = """

resolved_usage_cell_type_key = (
    config["cnmf"]["cell_type_key"]
    if USAGE_CELL_TYPE_KEY is None
    else str(USAGE_CELL_TYPE_KEY)
)
usage_obs = subset_usage_by_cell_type(
    adata.obs,
    cell_type_key=resolved_usage_cell_type_key,
    cell_types=REVIEW_CELL_TYPES,
)
included_usage_cell_types = usage_obs.attrs["cell_types"]
""".rstrip()
        consensus_cell.source = consensus_cell.source.replace(
            "programs = usage_columns(adata)",
            "programs = usage_columns(adata)" + review_block,
        )
        consensus_cell.source = consensus_cell.source.replace(
            '        "programs": len(programs),\n',
            '        "programs": len(programs),\n'
            '        "usage cell-type key": resolved_usage_cell_type_key,\n'
            '        "usage-review cells": len(usage_obs),\n'
            '        "usage-review cell types": included_usage_cell_types,\n',
        )
        _clear_code_output(consensus_cell)

    sample_markdown = _find_cell(
        notebook,
        cell_type="markdown",
        needle="## Usage by biological sample",
    )
    sample_markdown.source = """## Biological-sample usage

Condition boxplots use one mean per biological sample, avoiding cell-level
pseudoreplication. They are calculated from `usage_obs`, so
`USAGE_CELL_TYPE_KEY` and `REVIEW_CELL_TYPES` apply before sample means are
computed. Cells with missing or blank values in the selected cell-type column
are excluded. Select any subset of programs."""

    sample_cell = _find_cell(
        notebook,
        cell_type="code",
        needle="sample_summary = sample_usage_summary(",
    )
    sample_cell.source = sample_cell.source.replace(
        "    adata.obs,\n",
        "    usage_obs,\n",
    )
    _clear_code_output(sample_cell)

    condition_markdown_source = """## Condition-associated usage changes

This table tests every selected cell-type × usage-program combination using
**biological-sample means** as replicates. The effect is condition A minus
condition B. Cell-level distributions appear in the violin plot only; cells
are not treated as independent observations.

The table reports sample-level Welch tests and two-sided permutation tests.
Benjamini-Hochberg q-values are calculated within each contrast and globally
across all requested contrasts. The automatic plot is selected only after this
correction and therefore annotates both the raw Welch p-value and corrected
q-value. With four samples per group, an exhaustive two-sided permutation test
has a minimum attainable p-value of 2/70 = 0.02857."""
    condition_parameter_source = """# Each tuple is (condition A, condition B); delta = A - B.
USAGE_CONDITION_CONTRASTS = [
    ("vap_ab", "vap_igg"),
    ("vap_igg", "naive_igg"),
    ("vap_ab", "naive_igg"),
]
MIN_USAGE_CELLS_PER_SAMPLE = 10
MIN_USAGE_SAMPLES_PER_CONDITION = 3

# None selects the best corrected result for PLOT_USAGE_CONTRAST.
PLOT_USAGE_CONTRAST = ("vap_ab", "vap_igg")
PLOT_USAGE_CELL_TYPE = None
PLOT_USAGE_PROGRAM = None
PLOT_USAGE_SORT_BY = "welch_q_global"

USAGE_CONDITION_ORDER = ["naive_igg", "vap_igg", "vap_ab"]
USAGE_CONDITION_LABELS = {
    "naive_igg": "no vapor IgG",
    "vap_igg": "vapor IgG",
    "vap_ab": "vapor Ab",
}
USAGE_CONDITION_PALETTE = {
    "naive_igg": "#4C1D57",
    "vap_igg": "#F3B37A",
    "vap_ab": "#2A9D8F",
}"""
    condition_table_source = """usage_condition_tests = test_usage_condition_changes(
    usage_obs,
    cell_type_key=resolved_usage_cell_type_key,
    sample_key=config["cnmf"]["sample_key"],
    condition_key=config["cnmf"]["condition_key"],
    contrasts=USAGE_CONDITION_CONTRASTS,
    usage_cols=programs,
    min_cells_per_sample=MIN_USAGE_CELLS_PER_SAMPLE,
    min_samples_per_condition=MIN_USAGE_SAMPLES_PER_CONDITION,
)

condition_test_display_columns = [
    resolved_usage_cell_type_key,
    "program",
    "contrast",
    "delta_a_minus_b",
    "mean_usage_a",
    "mean_usage_b",
    "n_samples_a",
    "n_samples_b",
    "min_n_cells_a",
    "min_n_cells_b",
    "welch_p",
    "welch_q_within_contrast",
    "welch_q_global",
    "permutation_p",
    "permutation_q_within_contrast",
    "permutation_q_global",
]
display(usage_condition_tests[condition_test_display_columns].head(50))"""
    condition_plot_source = """focus_usage_condition_result = select_usage_condition_result(
    usage_condition_tests,
    contrast=PLOT_USAGE_CONTRAST,
    cell_type_key=resolved_usage_cell_type_key,
    cell_type=PLOT_USAGE_CELL_TYPE,
    program=PLOT_USAGE_PROGRAM,
    sort_by=PLOT_USAGE_SORT_BY,
)
display(focus_usage_condition_result.to_frame("value"))

available_usage_conditions = set(usage_obs[config["cnmf"]["condition_key"]].astype(str))
shown_usage_condition_order = [
    condition
    for condition in USAGE_CONDITION_ORDER
    if condition in available_usage_conditions
]
focus_usage_figure, focus_usage_sample_summary = plot_usage_condition_violin(
    usage_obs,
    result=focus_usage_condition_result,
    cell_type_key=resolved_usage_cell_type_key,
    sample_key=config["cnmf"]["sample_key"],
    condition_key=config["cnmf"]["condition_key"],
    condition_order=shown_usage_condition_order,
    condition_labels=USAGE_CONDITION_LABELS,
    palette=USAGE_CONDITION_PALETTE,
)
display(focus_usage_sample_summary)
focus_usage_figure"""

    existing_condition_markdown = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and "## Condition-associated usage" in cell.source
    ]
    if len(existing_condition_markdown) > 1:
        raise RuntimeError("Multiple condition-associated usage sections found")
    existing_condition_code = {
        needle: [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and needle in cell.source
        ]
        for needle in [
            "USAGE_CONDITION_CONTRASTS =",
            "usage_condition_tests = test_usage_condition_changes(",
            "focus_usage_condition_result = select_usage_condition_result(",
        ]
    }
    if any(len(cells) > 1 for cells in existing_condition_code.values()):
        raise RuntimeError("Duplicate condition-associated usage code cells found")

    if existing_condition_markdown:
        existing_condition_markdown[0].source = condition_markdown_source
        replacement_sources = [
            condition_parameter_source,
            condition_table_source,
            condition_plot_source,
        ]
        for cells, source in zip(existing_condition_code.values(), replacement_sources):
            if not cells:
                raise RuntimeError(
                    "Condition-associated usage section is missing a maintained code cell"
                )
            cells[0].source = source
            _clear_code_output(cells[0])
    else:
        insertion_index = notebook.cells.index(sample_cell) + 1
        notebook.cells[insertion_index:insertion_index] = [
            nbf.v4.new_markdown_cell(condition_markdown_source),
            nbf.v4.new_code_cell(condition_parameter_source),
            nbf.v4.new_code_cell(condition_table_source),
            nbf.v4.new_code_cell(condition_plot_source),
        ]

    grouped_markdown = _find_cell(
        notebook,
        cell_type="markdown",
        needle="## Grouped usage summaries",
    )
    grouped_markdown.source = """## Descriptive grouped summaries

Condition, spatial-domain, and cell-type means use the same filtered
`usage_obs`. The final heatmap groups by the resolved `USAGE_CELL_TYPE_KEY`, not
a hard-coded `celltype_short`. With `REVIEW_CELL_TYPES=None`, it contains every
non-missing label represented in the analyzed cells."""

    grouped_cell = _find_cell(
        notebook,
        cell_type="code",
        needle="for group_column in [",
    )
    grouped_cell.source = """for group_column in [
    config["cnmf"]["condition_key"],
    config["cnmf"]["compartment_key"],
    resolved_usage_cell_type_key,
]:
    grouped = summarize_usage(
        usage_obs,
        group_columns=[group_column],
        usage_cols=SELECTED_PROGRAMS,
    )
    display(grouped)
    plot_usage_heatmap(
        grouped,
        group_column=group_column,
        title=f"Mean usage by {group_column}",
    ).show()"""
    _clear_code_output(grouped_cell)

    return apply_notebook_presentation(notebook, "06")


def main() -> None:
    nbf.write(build_run_notebook(), RUN_NOTEBOOK)
    nbf.write(build_inspection_notebook(), INSPECTION_NOTEBOOK)
    print(RUN_NOTEBOOK)
    print(INSPECTION_NOTEBOOK)


if __name__ == "__main__":
    main()
