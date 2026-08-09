# spatial-workflow

`spatial-workflow` is a reproducible, configuration-driven workflow for Xenium
spatial transcriptomics developed from the Mayfield Lab analysis workflow.

The implemented scope is:

1. convert one merged Seurat object or multiple Seurat, exported-bundle, or raw
   Xenium samples into one standardized AnnData object;
2. train a transcript representation, identify CellCharter spatial domains,
   and run nearest-neighbor composition analysis;
3. review domain structure, directional colocalization, and sample-level
   support in compact notebooks;
4. run exact-lineage cNMF K sweeps, selected-K consensus, and condition-blind
   program-whitelist review; and
5. build local graph windows for LIANA rank aggregation and review.

## Design

- Configuration defines paths and semantic column roles once.
- Production code lives under `src/spatial_workflow`.
- Notebooks configure jobs or review completed outputs; they do not contain the
  workflow implementation.
- Raw counts remain in `layers["counts"]`; biological replication is defined by
  `sample_id`.
- CellCharter output is called a spatial domain. Anatomical region names remain
  a separate, curated annotation.

See [the documentation index](docs/index.md),
[the staged roadmap](docs/roadmap.md), and
[the AnnData contract](docs/data_contract.md).

## Install for development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cellcharter,nncomp,cnmf,notebooks,test]"
```

The `nncomp` extra pins the current Git revision. To develop both projects
together, replace it with the local checkout:

```bash
python -m pip install -e ../spatial-nncomp
```

Copy `configs/example.yaml` to a gitignored local configuration and update its
paths:

```bash
cp configs/example.yaml configs/local.yaml
```

Each sample annotation table must contain its configured cell-ID column and the
column named by `schema.cell_type_key`. The cell-type annotation is required
before running the spatial overview.

Start Jupyter from the repository root so configuration and generated-script
paths resolve consistently:

```bash
cd /path/to/spatial-workflow
jupyter lab
```

Then open the notebooks in order:

- `notebooks/01_prepare_anndata.ipynb`
- `notebooks/02_run_spatial_overview.ipynb`
- `notebooks/03_compartment_review.ipynb`
- `notebooks/04_colocalization_analysis.ipynb`
- `notebooks/05_cnmf_run.ipynb`
- `notebooks/06_cnmf_inspection.ipynb`
- `notebooks/07_liana_window_rankagg.ipynb`
- `notebooks/08_cnmf_program_whitelist_review.ipynb`

The execution notebooks write ordinary bash scripts or configured outputs.
The compartment, cNMF inspection, and whitelist-review notebooks are
read-only by default; Notebook 08 writes a human decision only when its
explicit save flag is enabled.

Notebook 04 treats directional colocalization as the primary result and
requires positive within-condition permutation evidence before interpreting a
between-condition difference. See the
[directional colocalization guide](docs/workflows/colocalization.md).
Notebook 04 supports both a true `whole_sample` graph and separate
`within_compartment` graphs; their artifacts and result directories are kept
separate.

Whole-sample scope is also explicit downstream. In Notebooks 05–06,
`COMPARTMENTS="all"` pools the selected cells into one cNMF fit. In Notebook
07, `COMPARTMENTS=None` builds LIANA windows across the complete biological
sample without using compartment labels. Explicit compartment lists filter and
pool cells; they do not silently launch one analysis per domain.

For the cNMF admission workflow,
`scripts/build_cnmf_program_review_metrics.py` generates condition-blind
program diagnostics and a prioritized queue, and
`scripts/build_selected_cnmf_anndata.py` rebuilds the derived master object
after the draft whitelist changes. See
[`docs/workflows/cnmf.md`](docs/workflows/cnmf.md) for the feature and
provenance contracts.

## Status

The repository provides public workflow code and review surfaces through
Notebook 08. The full test suite passes locally; large inputs and generated
results remain intentionally gitignored and are not distributed with the code.
