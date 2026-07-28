#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: test_export_seurat_merged.R export_seurat.R")
export_script <- normalizePath(args[[1]], mustWork = TRUE)

work_dir <- tempfile("spatial_workflow_merged_seurat_")
dir.create(work_dir)
on.exit(unlink(work_dir, recursive = TRUE), add = TRUE)

object_cells <- c("cell_1-1", "cell_2-1", "cell_1-2", "cell_2-2")
raw_cell_ids <- c("cell_1", "cell_2", "cell_1", "cell_2")
counts <- Matrix(
  matrix(
    c(1, 0, 2, 3, 4, 0, 1, 5, 0, 2, 1, 1),
    nrow = 3,
    dimnames = list(c("GeneA", "GeneB", "GeneC"), object_cells)
  ),
  sparse = TRUE
)
metadata <- data.frame(
  cell_id = raw_cell_ids,
  sample_id = rep(c("sample_a", "sample_b"), each = 2),
  condition = rep(c("control", "treated"), each = 2),
  cluster_sub = c("A1", "A2", "B1", "B2"),
  centroid_x = c(1, 2, 11, 12),
  centroid_y = c(3, 4, 13, 14),
  row.names = object_cells,
  check.names = FALSE
)
object <- CreateSeuratObject(counts = counts, assay = "Xenium", meta.data = metadata)
embedding <- matrix(
  seq_len(8),
  nrow = 4,
  dimnames = list(object_cells, c("PC_1", "PC_2"))
)
object[["pca"]] <- CreateDimReducObject(
  embeddings = embedding,
  key = "PC_",
  assay = "Xenium"
)

input_path <- file.path(work_dir, "merged.rds")
output_dir <- file.path(work_dir, "bundles")
saveRDS(object, input_path)
command_output <- system2(
  file.path(R.home("bin"), "Rscript"),
  c(
    export_script,
    "--input", input_path,
    "--output-dir", output_dir,
    "--sample-key", "sample_id",
    "--condition-key", "condition",
    "--assay", "Xenium",
    "--layer", "counts"
  ),
  stdout = TRUE,
  stderr = TRUE
)
status <- attr(command_output, "status")
if (!is.null(status) && status != 0) stop(paste(command_output, collapse = "\n"))

manifest <- read.csv(file.path(output_dir, "bundle_manifest.csv"), check.names = FALSE)
stopifnot(identical(manifest$sample_id, c("sample_a", "sample_b")))
stopifnot(identical(manifest$condition, c("control", "treated")))
stopifnot(identical(manifest$n_cells, c(2L, 2L)))

read_gzip_lines <- function(path) {
  connection <- gzfile(path, "rt")
  on.exit(close(connection), add = TRUE)
  readLines(connection)
}

for (index in seq_len(nrow(manifest))) {
  bundle <- file.path(output_dir, manifest$bundle_dir[[index]])
  cell_ids <- read_gzip_lines(file.path(bundle, "barcodes.tsv.gz"))
  obs <- read.csv(gzfile(file.path(bundle, "obs.csv.gz")), check.names = FALSE)
  spatial <- read.csv(gzfile(file.path(bundle, "spatial.csv.gz")), check.names = FALSE)
  pca <- read.csv(gzfile(file.path(bundle, "pca.csv.gz")), check.names = FALSE)

  stopifnot(identical(cell_ids, c("cell_1", "cell_2")))
  stopifnot(identical(as.character(obs$cell_id), cell_ids))
  stopifnot(identical(as.character(spatial$cell_id), cell_ids))
  stopifnot(identical(as.character(pca$cell_id), cell_ids))
  stopifnot(all(as.character(obs$sample_id) == manifest$sample_id[[index]]))
  stopifnot(all(as.character(obs$condition) == manifest$condition[[index]]))
  stopifnot(!any(object_cells %in% cell_ids))
}

cat("merged Seurat raw-cell-ID integration test passed\n")

