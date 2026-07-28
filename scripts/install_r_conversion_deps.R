#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
library_path <- if (length(args)) args[[1]] else file.path(getwd(), ".r-lib")
library_path <- normalizePath(library_path, mustWork = FALSE)
dir.create(library_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_path, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", lib = library_path, repos = "https://cloud.r-project.org")
}

versions <- c(
  RcppParallel = "6.1.1",
  stringfish = "0.17.0",
  qs = "0.27.2",
  spatstat.utils = "3.2-4",
  jsonlite = "2.0.0"
)

for (package in names(versions)) {
  installed <- tryCatch(
    packageDescription(package, lib.loc = library_path)$Version,
    error = function(error) NULL
  )
  if (identical(installed, versions[[package]])) {
    message(package, " ", installed, " already installed")
    next
  }
  remotes::install_version(
    package,
    version = versions[[package]],
    lib = library_path,
    repos = "https://cloud.r-project.org",
    dependencies = NA,
    upgrade = "never"
  )
}

message("R conversion dependencies installed in ", library_path)
