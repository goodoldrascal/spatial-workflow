# Verified environment

The initial ingestion and spatial-overview stage was verified with Python 3.13
and these direct dependency versions:

| Package | Version |
|---|---:|
| anndata | 0.12.10 |
| h5py | 3.15.1 |
| numpy | 2.4.3 |
| pandas | 2.3.3 |
| pyarrow | 23.0.1 |
| pyyaml | 6.0.3 |
| scipy | 1.16.3 |

The volatile CellCharter stack is pinned in `pyproject.toml`. A complete
transitive lock file will be added after the first reference-dataset parity
check.

Seurat RDS and QS export was verified with R 4.3.3 and the project-local
compatibility library shown below.

| Package | Version |
|---|---:|
| Seurat | 5.0.2 |
| SeuratObject | 5.0.2 |
| Matrix | 1.6.5 |
| RcppParallel | 6.1.1 |
| stringfish | 0.17.0 |
| qs | 0.27.2 |
| spatstat.utils | 3.2-4 |
| jsonlite | 2.0.0 |

`scripts/install_r_conversion_deps.R` installs the five pinned compatibility
packages into `.r-lib` without introducing a project-wide R lockfile:

```bash
R_LIBS_USER="$PWD/.r-lib" Rscript scripts/install_r_conversion_deps.R "$PWD/.r-lib"
```

Route Seurat conversion through that library in the project configuration:

```yaml
runtime:
  rscript_bin: Rscript
  r_libs_user: ../.r-lib
```

Relative `runtime.r_libs_user` paths resolve from the configuration directory;
the example therefore targets the repository root for configs under `configs/`.
