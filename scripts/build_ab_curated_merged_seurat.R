#!/usr/bin/env Rscript

SCRIPT_VERSION <- "0.1.0"
args <- commandArgs(trailingOnly = TRUE)

usage <- function(status = 0L) {
  cat(
    "Build a curated AB merged Seurat object by patching metadata only.\n\n",
    "Usage:\n",
    "  Rscript scripts/build_ab_curated_merged_seurat.R \\\n",
    "    --base merged_xenium_seurat.qs \\\n",
    "    --sample-sheet configs/ab_curated_seurat_sources.tsv \\\n",
    "    --output results/ab_xenium_curated.qs \\\n",
    "    [--base-sample-column sample] \\\n",
    "    [--base-cell-id-column cell_id] \\\n",
    "    [--source-sample-column sample] \\\n",
    "    [--source-condition-column group] \\\n",
    "    [--assay Xenium] [--layer counts] \\\n",
    "    [--preset high] [--nthreads 4] \\\n",
    "    [--allow-seurat-merge-suffix]\n\n",
    "The sample sheet must contain: sample_id, condition, input_path.\n",
    "Existing outputs are never overwritten. The optional suffix flag permits\n",
    "removing a terminal chain of _<integer> suffixes added by Seurat merge, but only when the\n",
    "result is an exact one-to-one match to all authoritative source cells.\n",
    sep = ""
  )
  quit(save = "no", status = status)
}

if (any(args %in% c("-h", "--help"))) usage()

arg_value <- function(name, default = NULL, required = FALSE) {
  exact <- which(args == name)
  inline <- grep(paste0("^", name, "="), args, value = TRUE)
  values <- character()

  if (length(exact) > 0) {
    if (any(exact == length(args))) stop("Missing value after ", name)
    values <- c(values, args[exact + 1L])
  }
  if (length(inline) > 0) {
    values <- c(values, sub(paste0("^", name, "="), "", inline))
  }
  if (length(values) > 1L) stop("Argument supplied more than once: ", name)
  if (length(values) == 0L) {
    if (required) stop("Missing required argument: ", name)
    return(default)
  }
  values[[1]]
}

base_path <- arg_value("--base", required = TRUE)
sample_sheet_path <- arg_value("--sample-sheet", required = TRUE)
output_path <- arg_value("--output", required = TRUE)
base_sample_column <- arg_value("--base-sample-column", "sample")
base_cell_id_column <- arg_value("--base-cell-id-column")
source_sample_column <- arg_value("--source-sample-column", "sample")
source_condition_column <- arg_value("--source-condition-column", "group")
assay_name <- arg_value("--assay", "Xenium")
counts_layer <- arg_value("--layer", "counts")
preset <- arg_value("--preset", "high")
nthreads <- as.integer(arg_value("--nthreads", "4"))
allow_merge_suffix <- "--allow-seurat-merge-suffix" %in% args

if (is.na(nthreads) || nthreads < 1L) stop("--nthreads must be a positive integer")

missing_packages <- c("qs", "SeuratObject")[
  !vapply(c("qs", "SeuratObject"), requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Required R package(s) unavailable: ",
    paste(missing_packages, collapse = ", "),
    "\n.libPaths(): ",
    paste(.libPaths(), collapse = " | ")
  )
}

package_diagnostic <- paste0(
  "qs=", as.character(utils::packageVersion("qs")),
  "; SeuratObject=", as.character(utils::packageVersion("SeuratObject")),
  "; R=", paste(R.version$major, R.version$minor, sep = ".")
)
message("Runtime: ", package_diagnostic)

normalize_input <- function(path, label) {
  if (!file.exists(path)) stop(label, " does not exist: ", path)
  normalizePath(path, mustWork = TRUE)
}

base_path <- normalize_input(base_path, "Base object")
sample_sheet_path <- normalize_input(sample_sheet_path, "Sample sheet")
if (tolower(tools::file_ext(base_path)) != "qs") stop("--base must be a .qs file")
if (tolower(tools::file_ext(output_path)) != "qs") stop("--output must end in .qs")

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(
  normalizePath(dirname(output_path), mustWork = TRUE),
  basename(output_path)
)
audit_stem <- sub("\\.qs$", "", output_path, ignore.case = TRUE)
manifest_path <- paste0(audit_stem, "_manifest.tsv")
sample_audit_path <- paste0(audit_stem, "_sample_audit.tsv")

for (path in c(output_path, manifest_path, sample_audit_path)) {
  if (file.exists(path)) stop("Refusing to overwrite existing output: ", path)
}
if (identical(base_path, output_path)) stop("--output must differ from --base")

sample_sheet <- utils::read.delim(
  sample_sheet_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  quote = "",
  comment.char = ""
)
required_sheet_columns <- c("sample_id", "condition", "input_path")
missing_sheet_columns <- setdiff(required_sheet_columns, colnames(sample_sheet))
if (length(missing_sheet_columns) > 0L) {
  stop(
    "Sample sheet is missing column(s): ",
    paste(missing_sheet_columns, collapse = ", ")
  )
}
sample_sheet <- sample_sheet[, required_sheet_columns, drop = FALSE]
if (nrow(sample_sheet) == 0L) stop("Sample sheet has no rows")
for (column in required_sheet_columns) {
  sample_sheet[[column]] <- trimws(as.character(sample_sheet[[column]]))
  if (any(is.na(sample_sheet[[column]]) | !nzchar(sample_sheet[[column]]))) {
    stop("Sample sheet contains an empty value in ", column)
  }
}
if (anyDuplicated(sample_sheet$sample_id)) stop("sample_id values must be unique")

sheet_dir <- dirname(sample_sheet_path)
is_absolute <- grepl("^(/|[A-Za-z]:[/\\\\])", sample_sheet$input_path)
sample_sheet$input_path[!is_absolute] <- file.path(
  sheet_dir,
  sample_sheet$input_path[!is_absolute]
)
sample_sheet$input_path <- vapply(
  seq_len(nrow(sample_sheet)),
  function(i) normalize_input(
    sample_sheet$input_path[[i]],
    paste0("Source for ", sample_sheet$sample_id[[i]])
  ),
  character(1)
)
if (any(tolower(tools::file_ext(sample_sheet$input_path)) != "qs")) {
  stop("Every source input_path must be a .qs file")
}
if (anyDuplicated(sample_sheet$input_path)) stop("Source input paths must be unique")
if (base_path %in% sample_sheet$input_path) {
  stop("The base object cannot also be an authoritative per-sample source")
}
sample_sheet <- sample_sheet[order(sample_sheet$sample_id), , drop = FALSE]
rownames(sample_sheet) <- NULL

read_seurat_qs <- function(path, label) {
  object <- tryCatch(
    qs::qread(path, nthreads = nthreads),
    error = function(error) {
      stop(
        "qs::qread failed for ", label, ": ", path,
        "\nRuntime: ", package_diagnostic,
        "\nOriginal error: ", conditionMessage(error)
      )
    }
  )
  if (!inherits(object, "Seurat")) stop(label, " is not a Seurat object: ", path)
  object
}

nonempty_character <- function(values, label, sample_id) {
  values <- as.character(values)
  if (any(is.na(values) | !nzchar(values))) {
    stop("Missing ", label, " value(s) in source sample ", sample_id)
  }
  values
}

extract_source <- function(index) {
  specification <- sample_sheet[index, , drop = FALSE]
  sample_id <- specification$sample_id[[1]]
  condition <- specification$condition[[1]]
  source_path <- specification$input_path[[1]]
  message("Reading annotation source: ", sample_id)
  object <- read_seurat_qs(source_path, paste0("Source ", sample_id))
  cells <- as.character(SeuratObject::Cells(object))
  metadata <- object@meta.data

  if (!identical(rownames(metadata), cells)) {
    stop("Source metadata row names are not identical to cell order: ", sample_id)
  }
  if (anyDuplicated(cells)) stop("Source cell IDs are duplicated: ", sample_id)

  required_metadata <- c(
    source_sample_column,
    source_condition_column,
    "cluster_sub",
    "celltype_short",
    "celltype_full",
    "centroid_x",
    "centroid_y"
  )
  missing_metadata <- setdiff(required_metadata, colnames(metadata))
  if (length(missing_metadata) > 0L) {
    stop(
      "Source ", sample_id, " is missing metadata column(s): ",
      paste(missing_metadata, collapse = ", ")
    )
  }

  stored_samples <- unique(nonempty_character(
    metadata[[source_sample_column]],
    source_sample_column,
    sample_id
  ))
  if (!identical(stored_samples, sample_id)) {
    stop(
      "Source ", sample_id, " stores unexpected ", source_sample_column,
      " value(s): ", paste(stored_samples, collapse = ", ")
    )
  }
  stored_conditions <- unique(nonempty_character(
    metadata[[source_condition_column]],
    source_condition_column,
    sample_id
  ))
  if (!identical(stored_conditions, condition)) {
    stop(
      "Source ", sample_id, " stores unexpected ", source_condition_column,
      " value(s): ", paste(stored_conditions, collapse = ", "),
      "; sample sheet specifies ", condition
    )
  }

  annotation <- data.frame(
    sample_id = rep(sample_id, length(cells)),
    condition = rep(condition, length(cells)),
    cell_id = cells,
    cluster_sub = nonempty_character(metadata$cluster_sub, "cluster_sub", sample_id),
    celltype_short = nonempty_character(
      metadata$celltype_short,
      "celltype_short",
      sample_id
    ),
    celltype_full = nonempty_character(
      metadata$celltype_full,
      "celltype_full",
      sample_id
    ),
    centroid_x = as.numeric(metadata$centroid_x),
    centroid_y = as.numeric(metadata$centroid_y),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (!all(is.finite(annotation$centroid_x) & is.finite(annotation$centroid_y))) {
    stop("Source ", sample_id, " has missing or non-finite centroid coordinates")
  }

  audit <- data.frame(
    sample_id = sample_id,
    condition = condition,
    input_path = source_path,
    input_bytes = file.info(source_path)$size,
    n_cells = nrow(annotation),
    n_cluster_sub = length(unique(annotation$cluster_sub)),
    n_celltype_short = length(unique(annotation$celltype_short)),
    n_celltype_full = length(unique(annotation$celltype_full)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  rm(object, metadata)
  invisible(gc(verbose = FALSE))
  list(annotation = annotation, audit = audit)
}

source_results <- lapply(seq_len(nrow(sample_sheet)), extract_source)
annotation <- do.call(rbind, lapply(source_results, `[[`, "annotation"))
sample_audit <- do.call(rbind, lapply(source_results, `[[`, "audit"))
rm(source_results)
invisible(gc(verbose = FALSE))

key_separator <- "\u001f"
make_key <- function(sample_id, cell_id) {
  if (any(grepl(key_separator, sample_id, fixed = TRUE)) ||
      any(grepl(key_separator, cell_id, fixed = TRUE))) {
    stop("Cell or sample IDs contain the reserved identity separator")
  }
  paste(sample_id, cell_id, sep = key_separator)
}

source_keys <- make_key(annotation$sample_id, annotation$cell_id)
if (anyDuplicated(source_keys)) {
  stop("Authoritative source sample_id + cell_id keys are not unique")
}

message("Reading base merged object")
base <- read_seurat_qs(base_path, "Base object")
base_cells <- as.character(SeuratObject::Cells(base))
base_metadata <- base@meta.data
if (!identical(rownames(base_metadata), base_cells)) {
  stop("Base metadata row names are not identical to cell order")
}
if (!base_sample_column %in% colnames(base_metadata)) {
  stop("Base metadata is missing sample column: ", base_sample_column)
}
if (!assay_name %in% SeuratObject::Assays(base)) {
  stop("Base object is missing assay: ", assay_name)
}
available_layers <- tryCatch(
  SeuratObject::Layers(base@assays[[assay_name]]),
  error = function(error) character()
)
if (length(available_layers) > 0L && !counts_layer %in% available_layers) {
  stop(
    "Base assay ", assay_name, " is missing layer ", counts_layer,
    ". Available layers: ", paste(available_layers, collapse = ", ")
  )
}
if (length(available_layers) == 0L &&
    !counts_layer %in% methods::slotNames(base@assays[[assay_name]])) {
  stop("Base assay ", assay_name, " has no layer or slot named ", counts_layer)
}

base_samples <- nonempty_character(
  base_metadata[[base_sample_column]],
  base_sample_column,
  "base object"
)
unexpected_base_samples <- setdiff(unique(base_samples), sample_sheet$sample_id)
missing_base_samples <- setdiff(sample_sheet$sample_id, unique(base_samples))
if (length(unexpected_base_samples) > 0L || length(missing_base_samples) > 0L) {
  stop(
    "Base/source sample sets differ. Unexpected in base: ",
    paste(unexpected_base_samples, collapse = ", "),
    "; missing from base: ", paste(missing_base_samples, collapse = ", ")
  )
}

if (is.null(base_cell_id_column)) {
  base_cell_ids <- base_cells
  identity_method <- "base_cell_names"
} else {
  if (!base_cell_id_column %in% colnames(base_metadata)) {
    stop("Base metadata is missing cell-ID column: ", base_cell_id_column)
  }
  base_cell_ids <- nonempty_character(
    base_metadata[[base_cell_id_column]],
    base_cell_id_column,
    "base object"
  )
  identity_method <- paste0("base_metadata:", base_cell_id_column)
}

is_exact_bijection <- function(keys) {
  length(keys) == length(source_keys) &&
    !anyDuplicated(keys) &&
    !anyNA(match(keys, source_keys)) &&
    !anyNA(match(source_keys, keys))
}

base_keys <- make_key(base_samples, base_cell_ids)
n_suffix_resolved <- 0L
max_suffix_depth <- 0L
if (!is_exact_bijection(base_keys)) {
  stripped_ids <- sub("(_[0-9]+)+$", "", base_cell_ids, perl = TRUE)
  stripped_keys <- make_key(base_samples, stripped_ids)
  suffix_would_match <- is_exact_bijection(stripped_keys)

  if (!allow_merge_suffix || !suffix_would_match) {
    unmatched_base <- sum(is.na(match(base_keys, source_keys)))
    unmatched_source <- sum(is.na(match(source_keys, base_keys)))
    suffix_hint <- if (suffix_would_match) {
      paste0(
        "\nA complete bijection is possible after removing a terminal chain of ",
        "Seurat _<integer> merge suffixes. Re-run with ",
        "--allow-seurat-merge-suffix to authorize that exact transformation."
      )
    } else {
      ""
    }
    stop(
      "Cell identity validation failed before metadata patching.",
      "\nBase cells: ", length(base_keys),
      "; source cells: ", length(source_keys),
      "; unmatched base keys: ", unmatched_base,
      "; unmatched source keys: ", unmatched_source,
      suffix_hint
    )
  }

  n_suffix_resolved <- sum(stripped_ids != base_cell_ids)
  changed <- stripped_ids != base_cell_ids
  suffix_start <- regexpr("(_[0-9]+)+$", base_cell_ids[changed], perl = TRUE)
  suffix_text <- substring(base_cell_ids[changed], suffix_start)
  max_suffix_depth <- max(
    lengths(strsplit(sub("^_", "", suffix_text), "_", fixed = TRUE))
  )
  base_cell_ids <- stripped_ids
  base_keys <- stripped_keys
  identity_method <- paste0(identity_method, "+explicit_seurat_merge_suffix")
}

annotation_index <- match(base_keys, source_keys)
if (anyNA(annotation_index)) {
  stop("Internal error: exact identity validation passed but annotation match is incomplete")
}
ordered_annotation <- annotation[annotation_index, , drop = FALSE]
if (!identical(ordered_annotation$sample_id, base_samples)) {
  stop("Internal error: matched annotations are not aligned to base samples")
}

assays_before <- SeuratObject::Assays(base)
assay_dimensions <- vapply(
  assays_before,
  function(name) paste(dim(base@assays[[name]]), collapse = "x"),
  character(1)
)

base_metadata$cell_id <- ordered_annotation$cell_id
base_metadata$sample_id <- ordered_annotation$sample_id
base_metadata$condition <- ordered_annotation$condition
base_metadata$cluster_sub <- ordered_annotation$cluster_sub
base_metadata$celltype_short <- ordered_annotation$celltype_short
base_metadata$celltype_full <- ordered_annotation$celltype_full
base_metadata$centroid_x <- ordered_annotation$centroid_x
base_metadata$centroid_y <- ordered_annotation$centroid_y
base@meta.data <- base_metadata

if (!identical(base_cells, as.character(SeuratObject::Cells(base)))) {
  stop("Cell order changed while patching metadata")
}
if (!identical(assays_before, SeuratObject::Assays(base))) {
  stop("Assay set changed while patching metadata")
}

base_counts <- as.data.frame(table(sample_id = base_samples), stringsAsFactors = FALSE)
colnames(base_counts)[[2]] <- "n_cells_in_base"
sample_audit <- merge(
  sample_audit,
  base_counts,
  by = "sample_id",
  all.x = TRUE,
  sort = FALSE
)
sample_audit <- sample_audit[
  match(sample_sheet$sample_id, sample_audit$sample_id),
  ,
  drop = FALSE
]
if (!identical(sample_audit$n_cells, sample_audit$n_cells_in_base)) {
  stop("Per-sample cell counts differ after exact identity matching")
}

partial_path <- file.path(
  dirname(output_path),
  paste0(".", basename(output_path), ".partial-", Sys.getpid())
)
on.exit({
  if (file.exists(partial_path)) unlink(partial_path)
}, add = TRUE)

message("Writing new curated object: ", output_path)
tryCatch(
  qs::qsave(base, partial_path, preset = preset, nthreads = nthreads),
  error = function(error) {
    stop(
      "qs::qsave failed for ", partial_path,
      "\nRuntime: ", package_diagnostic,
      "\nOriginal error: ", conditionMessage(error)
    )
  }
)
if (!file.rename(partial_path, output_path)) {
  stop("Could not atomically move completed QS file to: ", output_path)
}

manifest <- data.frame(
  key = c(
    "script_version",
    "created_utc",
    "base_path",
    "base_bytes",
    "sample_sheet",
    "output_path",
    "output_bytes",
    "n_samples",
    "n_cells",
    "identity_method",
    "n_seurat_suffixes_removed",
    "max_seurat_suffix_depth",
    "base_sample_column",
    "source_sample_column",
    "source_condition_column",
    "assay",
    "counts_layer",
    "assay_dimensions",
    "qs_preset",
    "qs_nthreads",
    "runtime"
  ),
  value = c(
    SCRIPT_VERSION,
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    base_path,
    as.character(file.info(base_path)$size),
    sample_sheet_path,
    output_path,
    as.character(file.info(output_path)$size),
    as.character(nrow(sample_sheet)),
    as.character(length(base_cells)),
    identity_method,
    as.character(n_suffix_resolved),
    as.character(max_suffix_depth),
    base_sample_column,
    source_sample_column,
    source_condition_column,
    assay_name,
    counts_layer,
    paste(paste(names(assay_dimensions), assay_dimensions, sep = ":"), collapse = ","),
    preset,
    as.character(nthreads),
    package_diagnostic
  ),
  stringsAsFactors = FALSE
)
utils::write.table(
  manifest,
  manifest_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
utils::write.table(
  sample_audit,
  sample_audit_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

message(
  "Completed: ", length(base_cells), " cells across ", nrow(sample_sheet),
  " samples. Assays and cell order were preserved."
)
message("Manifest: ", manifest_path)
message("Sample audit: ", sample_audit_path)
