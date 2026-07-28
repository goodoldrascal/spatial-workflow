#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
})

if (!requireNamespace("qs", quietly = TRUE)) stop("qs is required")

repo_root <- normalizePath(getwd(), mustWork = TRUE)
builder <- file.path(repo_root, "scripts", "build_ab_curated_merged_seurat.R")
if (!file.exists(builder)) stop("Run this smoke test from the repository root")

work_dir <- tempfile("curated-seurat-smoke-")
dir.create(work_dir)
on.exit(unlink(work_dir, recursive = TRUE), add = TRUE)

make_source <- function(sample_id, condition, cells, offset) {
  counts <- Matrix(
    matrix(c(1, 0, 2, 3, 1, 0), nrow = 3) + offset,
    sparse = TRUE
  )
  rownames(counts) <- c("GeneA", "GeneB", "GeneC")
  colnames(counts) <- cells
  metadata <- data.frame(
    sample = sample_id,
    group = condition,
    cluster_sub = c("AST_sub1", "OLIG_sub7"),
    celltype_short = c("AST", "OLIG"),
    celltype_full = c("Astrocytes", "Oligodendrocytes"),
    centroid_x = c(10, 20) + offset,
    centroid_y = c(30, 40) + offset,
    row.names = cells,
    stringsAsFactors = FALSE
  )
  CreateSeuratObject(counts = counts, assay = "Xenium", meta.data = metadata)
}

source_a <- make_source("sample_a", "control", c("shared-1", "a-only-1"), 0)
source_b <- make_source("sample_b", "treated", c("shared-1", "b-only-1"), 1)
source_a_path <- file.path(work_dir, "sample_a.qs")
source_b_path <- file.path(work_dir, "sample_b.qs")
qs::qsave(source_a, source_a_path)
qs::qsave(source_b, source_b_path)

base_counts <- cbind(
  LayerData(source_a, assay = "Xenium", layer = "counts"),
  LayerData(source_b, assay = "Xenium", layer = "counts")
)
base_cells <- c("shared-1_1_1_1", "a-only-1_1_1_1", "shared-1_2_2_2", "b-only-1_2_2_2")
colnames(base_counts) <- base_cells
base_metadata <- data.frame(
  sample = c("sample_a", "sample_a", "sample_b", "sample_b"),
  row.names = base_cells,
  stringsAsFactors = FALSE
)
base <- CreateSeuratObject(
  counts = base_counts,
  assay = "Xenium",
  meta.data = base_metadata
)
base_path <- file.path(work_dir, "base.qs")
output_path <- file.path(work_dir, "curated.qs")
sample_sheet_path <- file.path(work_dir, "sources.tsv")
qs::qsave(base, base_path)
write.table(
  data.frame(
    sample_id = c("sample_a", "sample_b"),
    condition = c("control", "treated"),
    input_path = c(source_a_path, source_b_path)
  ),
  sample_sheet_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

common_args <- c(
  builder,
  "--base", base_path,
  "--sample-sheet", sample_sheet_path,
  "--output", output_path,
  "--nthreads", "1"
)
strict_log <- suppressWarnings(
  system2("Rscript", common_args, stdout = TRUE, stderr = TRUE)
)
strict_status <- attr(strict_log, "status")
if (is.null(strict_status)) strict_status <- 0L
stopifnot(strict_status != 0L, !file.exists(output_path))
stopifnot(any(grepl("--allow-seurat-merge-suffix", strict_log, fixed = TRUE)))

authorized_log <- system2(
  "Rscript",
  c(common_args, "--allow-seurat-merge-suffix"),
  stdout = TRUE,
  stderr = TRUE
)
authorized_status <- attr(authorized_log, "status")
if (is.null(authorized_status)) authorized_status <- 0L
if (authorized_status != 0L) {
  stop("Authorized smoke build failed:\n", paste(authorized_log, collapse = "\n"))
}

curated <- qs::qread(output_path)
stopifnot(
  identical(Cells(curated), Cells(base)),
  identical(Assays(curated), Assays(base)),
  identical(
    LayerData(curated, assay = "Xenium", layer = "counts"),
    LayerData(base, assay = "Xenium", layer = "counts")
  ),
  identical(
    curated@meta.data$cell_id,
    c("shared-1", "a-only-1", "shared-1", "b-only-1")
  ),
  identical(
    curated@meta.data$sample_id,
    c("sample_a", "sample_a", "sample_b", "sample_b")
  ),
  identical(
    curated@meta.data$condition,
    c("control", "control", "treated", "treated")
  ),
  file.exists(file.path(work_dir, "curated_manifest.tsv")),
  file.exists(file.path(work_dir, "curated_sample_audit.tsv"))
)

cat("curated Seurat builder smoke test passed\n")
manifest <- read.delim(file.path(work_dir, "curated_manifest.tsv"), stringsAsFactors = FALSE)
stopifnot(manifest$value[manifest$key == "max_seurat_suffix_depth"] == "3")
