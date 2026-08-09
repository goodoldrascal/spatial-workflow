#!/usr/bin/env python3
"""Apply public-facing Markdown and compact code layout to workflow notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_PATH = Path(__file__).with_suffix(".json")
NOTEBOOKS = {
    "03": REPO_ROOT / "notebooks" / "03_compartment_review.ipynb",
    "04": REPO_ROOT / "notebooks" / "04_colocalization_analysis.ipynb",
    "05": REPO_ROOT / "notebooks" / "05_cnmf_run.ipynb",
    "06": REPO_ROOT / "notebooks" / "06_cnmf_inspection.ipynb",
    "07": REPO_ROOT / "notebooks" / "07_liana_window_rankagg.ipynb",
    "08": REPO_ROOT / "notebooks" / "08_cnmf_program_whitelist_review.ipynb",
}

_PRESENTATION = json.loads(PRESENTATION_PATH.read_text(encoding="utf-8"))
PUBLIC_MARKDOWN = _PRESENTATION["public_markdown"]
COMPACT_CODE = _PRESENTATION["compact_code"]
DROP_MARKDOWN_HEADINGS = {
    key: set(values) for key, values in _PRESENTATION["drop_markdown_headings"].items()
}
DROP_CODE_PREFIXES = {
    key: set(values) for key, values in _PRESENTATION["drop_code_prefixes"].items()
}


def _first_nonempty_line(source: str) -> str:
    return next((line.strip() for line in source.splitlines() if line.strip()), "")


def _with_canonical_aliases(replacements: dict[str, str]) -> dict[str, str]:
    aliases = dict(replacements)
    for source in replacements.values():
        aliases[_first_nonempty_line(source)] = source
    return aliases


def apply_public_markdown(notebook, notebook_id: str):
    """Replace maintained Markdown without modifying code or outputs."""

    notebook_id = str(notebook_id)
    replacements = _with_canonical_aliases(PUBLIC_MARKDOWN[notebook_id])
    dropped = DROP_MARKDOWN_HEADINGS.get(notebook_id, set())
    cells = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            cells.append(cell)
            continue
        heading = _first_nonempty_line(cell.source)
        if heading in dropped:
            continue
        if heading in replacements:
            cell.source = replacements[heading].strip()
        if cell.source.strip():
            cells.append(cell)
    notebook.cells = cells
    return notebook


def apply_compact_code(notebook, notebook_id: str):
    """Compact selected display cells without changing analysis behavior."""

    notebook_id = str(notebook_id)
    replacements = _with_canonical_aliases(COMPACT_CODE.get(notebook_id, {}))
    dropped = DROP_CODE_PREFIXES.get(notebook_id, set())
    cells = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            cells.append(cell)
            continue
        first_line = _first_nonempty_line(cell.source)
        if any(first_line.startswith(prefix) for prefix in dropped):
            continue
        replacement = replacements.get(first_line)
        if replacement is None:
            replacement = next(
                (
                    source
                    for prefix, source in replacements.items()
                    if first_line.startswith(prefix)
                ),
                None,
            )
        if replacement is not None:
            cell.source = replacement.strip()
        cells.append(cell)
    notebook.cells = cells
    return notebook


def apply_notebook_presentation(notebook, notebook_id: str):
    apply_public_markdown(notebook, notebook_id)
    apply_compact_code(notebook, notebook_id)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            output.get("data", {}).pop(
                "application/vnd.microsoft.datawrangler.viewer.v0+json", None
            )
    return notebook


def update_notebook(path: Path, notebook_id: str) -> None:
    notebook = nbf.read(path, as_version=4)
    apply_notebook_presentation(notebook, notebook_id)
    nbf.write(notebook, path)


def main() -> None:
    for notebook_id, path in NOTEBOOKS.items():
        update_notebook(path, notebook_id)
        print(path)


if __name__ == "__main__":
    main()
