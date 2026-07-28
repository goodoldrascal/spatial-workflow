#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(name) {
  exact <- which(args == name)
  if (length(exact) == 1 && exact < length(args)) return(args[[exact + 1]])
  prefix <- paste0(name, "=")
  inline <- grep(paste0("^", prefix), args, value = TRUE)
  if (length(inline) == 1) return(sub(paste0("^", prefix), "", inline))
  NULL
}

input_path <- arg_value("--input")
output_dir <- arg_value("--output-dir")
sample_id <- arg_value("--sample-id")
sample_key <- arg_value("--sample-key")
condition_key <- arg_value("--condition-key")
assay <- arg_value("--assay")
layer <- arg_value("--layer")

if (any(vapply(list(input_path, output_dir, assay, layer), is.null, logical(1)))) {
  stop(
    "Usage: export_seurat.R --input object.rds|object.qs --output-dir bundles ",
    "--assay Xenium --layer counts and either --sample-id sample_01 or ",
    "--sample-key sample_id --condition-key condition"
  )
}
merged_mode <- !is.null(sample_key) || !is.null(condition_key)
if (merged_mode && (is.null(sample_key) || is.null(condition_key))) {
  stop("Merged export requires both --sample-key and --condition-key")
}
if (merged_mode && !is.null(sample_id)) {
  stop("Use --sample-id for one sample or --sample-key/--condition-key for a merged object")
}
if (!merged_mode && is.null(sample_id)) {
  stop(
    "Supply --sample-id for one sample or --sample-key/--condition-key for a merged object"
  )
}
if (!file.exists(input_path)) stop("Input does not exist: ", input_path)

extension <- tolower(tools::file_ext(input_path))
if (extension == "rds") {
  object <- readRDS(input_path)
} else if (extension == "qs") {
  if (!requireNamespace("qs", quietly = TRUE)) stop("The qs package is required for .qs input")
  object <- qs::qread(input_path)
} else {
  stop("Input must end in .rds or .qs")
}
if (!inherits(object, "Seurat")) stop("Input is not a Seurat object")
if (!assay %in% Assays(object)) stop("Assay not found: ", assay)

assay_object <- object[[assay]]
available_layers <- tryCatch(Layers(assay_object), error = function(e) character())
if (length(available_layers) > 0) {
  if (!layer %in% available_layers) {
    stop(
      "Layer '", layer, "' not found. Available layers: ",
      paste(available_layers, collapse = ", ")
    )
  }
  counts <- LayerData(assay_object, layer = layer)
} else {
  counts <- tryCatch(
    GetAssayData(object, assay = assay, slot = layer),
    error = function(e) stop("Could not read explicit slot '", layer, "' from assay '", assay, "'")
  )
}
counts <- as(counts, "dgCMatrix")
if (length(counts@x) && (any(!is.finite(counts@x)) || any(counts@x < 0))) {
  stop("Configured layer contains non-finite or negative values")
}
if (length(counts@x) && any(abs(counts@x - round(counts@x)) > 1e-8)) {
  stop("Configured layer is not an integer raw-count layer")
}

write_csv_gz <- function(table, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.csv(table, connection, row.names = FALSE, quote = TRUE)
}

write_lines_gz <- function(values, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  writeLines(values, connection)
}

coordinate_pair <- function(table) {
  for (pair in list(
    c("x_centroid", "y_centroid"),
    c("centroid_x", "centroid_y"),
    c("x", "y"),
    c("imagecol", "imagerow")
  )) {
    if (all(pair %in% colnames(table))) return(pair)
  }
  NULL
}

aligned_coordinates <- function(table, object_cells, output_cell_ids) {
  if (is.null(table) || nrow(table) == 0) return(NULL)
  table <- as.data.frame(table, check.names = FALSE)
  pair <- coordinate_pair(table)
  if (is.null(pair)) return(NULL)

  index <- match(object_cells, rownames(table))
  if (any(is.na(index)) && "cell_id" %in% colnames(table)) {
    index <- match(object_cells, as.character(table$cell_id))
  }
  if (any(is.na(index))) return(NULL)
  coordinates <- data.frame(
    cell_id = output_cell_ids,
    x = as.numeric(table[[pair[[1]]]][index]),
    y = as.numeric(table[[pair[[2]]]][index]),
    check.names = FALSE
  )
  if (!all(is.finite(coordinates$x) & is.finite(coordinates$y))) return(NULL)
  coordinates
}

object_cells <- colnames(counts)
metadata <- object[[]]
metadata_index <- match(object_cells, rownames(metadata))
if (any(is.na(metadata_index))) stop("Seurat metadata is missing assay cells")
metadata <- metadata[metadata_index, , drop = FALSE]
for (column in colnames(metadata)) {
  if (is.list(metadata[[column]])) {
    metadata[[column]] <- vapply(
      metadata[[column]],
      paste,
      collapse = ";",
      FUN.VALUE = character(1)
    )
  }
}
output_cell_ids <- if ("cell_id" %in% colnames(metadata)) {
  as.character(metadata$cell_id)
} else {
  object_cells
}
if (any(is.na(output_cell_ids) | trimws(output_cell_ids) == "")) {
  stop("Seurat cell_id metadata contains missing or empty values")
}
metadata$cell_id <- output_cell_ids

spatial <- aligned_coordinates(metadata, object_cells, output_cell_ids)
if (is.null(spatial)) {
  spatial <- aligned_coordinates(
    tryCatch(GetTissueCoordinates(object), error = function(e) NULL),
    object_cells,
    output_cell_ids
  )
}
if (is.null(spatial)) {
  for (image_name in Images(object)) {
    spatial <- aligned_coordinates(
      tryCatch(GetTissueCoordinates(object[[image_name]]), error = function(e) NULL),
      object_cells,
      output_cell_ids
    )
    if (!is.null(spatial)) break
  }
}
if (is.null(spatial)) stop("Could not recover finite spatial coordinates for every exported cell")

write_bundle <- function(sample_dir, selected_indices) {
  sample_object_cells <- object_cells[selected_indices]
  sample_cell_ids <- output_cell_ids[selected_indices]
  if (anyDuplicated(sample_cell_ids)) {
    stop("Raw cell_id values must be unique within each sample: ", sample_dir)
  }
  sample_counts <- counts[, selected_indices, drop = FALSE]
  sample_metadata <- metadata[selected_indices, , drop = FALSE]
  sample_spatial <- spatial[selected_indices, , drop = FALSE]

  dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)
  matrix_path <- file.path(sample_dir, "counts.mtx")
  Matrix::writeMM(sample_counts, matrix_path)
  status <- system2("gzip", c("-f", shQuote(matrix_path)))
  if (status != 0) stop("gzip failed while writing counts.mtx.gz")

  write_lines_gz(rownames(sample_counts), file.path(sample_dir, "genes.tsv.gz"))
  write_lines_gz(sample_cell_ids, file.path(sample_dir, "barcodes.tsv.gz"))
  write_csv_gz(sample_metadata, file.path(sample_dir, "obs.csv.gz"))
  write_csv_gz(sample_spatial, file.path(sample_dir, "spatial.csv.gz"))

  for (reduction in c("pca", "umap")) {
    if (reduction %in% Reductions(object)) {
      embedding <- Embeddings(object, reduction)
      embedding <- embedding[match(sample_object_cells, rownames(embedding)), , drop = FALSE]
      write_csv_gz(
        data.frame(cell_id = sample_cell_ids, embedding, check.names = FALSE),
        file.path(sample_dir, paste0(reduction, ".csv.gz"))
      )
    }
  }

  cat(
    "Wrote", sample_dir, "with", ncol(sample_counts), "cells and",
    nrow(sample_counts), "features\n"
  )
}

if (!merged_mode) {
  write_bundle(file.path(output_dir, sample_id), seq_along(object_cells))
} else {
  missing_columns <- setdiff(c(sample_key, condition_key), colnames(metadata))
  if (length(missing_columns)) {
    stop("Seurat metadata is missing configured columns: ", paste(missing_columns, collapse = ", "))
  }

  sample_values <- as.character(metadata[[sample_key]])
  condition_values <- as.character(metadata[[condition_key]])
  if (any(is.na(sample_values) | trimws(sample_values) == "")) {
    stop("Configured sample metadata contains missing or empty values")
  }
  if (any(is.na(condition_values) | trimws(condition_values) == "")) {
    stop("Configured condition metadata contains missing or empty values")
  }

  design <- unique(data.frame(
    sample_id = sample_values,
    condition = condition_values,
    stringsAsFactors = FALSE
  ))
  if (any(duplicated(design$sample_id))) {
    stop("Each sample must map to exactly one condition in Seurat metadata")
  }
  design <- design[order(design$sample_id), , drop = FALSE]
  design$bundle_dir <- sprintf("sample_%04d", seq_len(nrow(design)))
  design$n_cells <- integer(nrow(design))

  for (index in seq_len(nrow(design))) {
    selected_indices <- which(sample_values == design$sample_id[[index]])
    design$n_cells[[index]] <- length(selected_indices)
    write_bundle(
      file.path(output_dir, design$bundle_dir[[index]]),
      selected_indices
    )
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(
    design[, c("sample_id", "condition", "bundle_dir", "n_cells")],
    file.path(output_dir, "bundle_manifest.csv"),
    row.names = FALSE,
    quote = TRUE
  )
  cat("Wrote", nrow(design), "sample bundles from merged Seurat metadata\n")
}
