# spatial-workflow

`spatial-workflow` is a reproducible workflow for Xenium spatial
transcriptomics. It is being assembled in verified stages from the Mayfield Lab
analysis workflow.

The initial scope is:

1. convert one merged Seurat object or multiple Seurat, exported-bundle, or raw
   Xenium samples into one standardized AnnData object;
2. train a transcript representation, identify CellCharter spatial domains,
   and run nearest-neighbor composition analysis;
3. review the domain structure and neighboring cell types in compact
   notebooks.

Later stages will add differential expression, cNMF, adaptive-window LIANA,
pathway analysis, and single-cell communication.

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
python -m pip install -e ".[cellcharter,nncomp,notebooks,test]"
```

The `nncomp` extra pins the current Git revision. To develop both projects
together, replace it with the local checkout:

```bash
python -m pip install -e ../liana/spatial-nncomp
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

The first two notebooks write ordinary bash scripts. Run those scripts directly
or launch the second through its generated tmux wrapper on the compute node.
The third notebook is read-only.

## Status

This repository is an early scaffold. Each workflow stage will be promoted only
after a small test-dataset run succeeds and a parity check against an existing
Mayfield analysis artifact passes.

