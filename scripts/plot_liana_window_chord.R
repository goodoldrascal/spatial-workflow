#!/usr/bin/env Rscript

# Directional gene-level chord plot used by the LIANA window-rankagg notebook.
# The geometry matches the manuscript circlize renderer: directional big-arrow
# ribbons, target-proportional arrowheads, cell-state sectors, and explicit
# condition or sender-cell link colors.

suppressPackageStartupMessages({
  library(circlize)
  library(ComplexHeatmap)
  library(grid)
})

plot_liana_window_chord <- function(
    input_path,
    output_path,
    title_text = "LIANA pathway drivers",
    link_color_mode = "condition",
    width_inches = 14,
    height_inches = 11,
    png_res = 250L) {
  if (!link_color_mode %in% c("condition", "source_cell")) {
    stop("link_color_mode must be 'condition' or 'source_cell'")
  }

  edges <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c(
    "source", "target", "ligand", "receptor", "weight_abs", "higher_condition"
  )
  missing <- setdiff(required, names(edges))
  if (length(missing) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))
  }
  edges$weight_abs <- as.numeric(edges$weight_abs)
  edges <- edges[is.finite(edges$weight_abs) & edges$weight_abs > 0, , drop = FALSE]
  if (nrow(edges) == 0) {
    stop("No positive chord weights remain")
  }

  cell_states <- sort(unique(c(edges$source, edges$target)))
  cell_colors <- grDevices::hcl.colors(
    max(length(cell_states), 3L), palette = "Dark 3"
  )[seq_along(cell_states)]
  names(cell_colors) <- cell_states

  condition_levels <- sort(unique(edges$higher_condition))
  condition_colors <- grDevices::hcl.colors(
    max(length(condition_levels), 3L), palette = "Set 2"
  )[seq_along(condition_levels)]
  names(condition_colors) <- condition_levels

  edges$ligand_sector <- paste("L", edges$source, edges$ligand, sep = "::")
  edges$receptor_sector <- paste("R", edges$target, edges$receptor, sep = "::")
  focal <- if ("focal" %in% names(edges)) unique(edges$focal)[[1]] else NULL
  cell_order <- unique(c(focal, setdiff(cell_states, focal)))
  edges$source <- factor(edges$source, levels = cell_order)
  edges$target <- factor(edges$target, levels = cell_order)
  edges <- edges[order(edges$source, -edges$weight_abs), , drop = FALSE]

  source_sectors <- unique(
    edges[, c("ligand_sector", "ligand", "source"), drop = FALSE]
  )
  target_sectors <- unique(
    edges[, c("receptor_sector", "receptor", "target"), drop = FALSE]
  )
  sector_order <- unique(c(source_sectors$ligand_sector, target_sectors$receptor_sector))
  sector_labels <- c(
    setNames(source_sectors$ligand, source_sectors$ligand_sector),
    setNames(target_sectors$receptor, target_sectors$receptor_sector)
  )[sector_order]
  grid_colors <- c(
    setNames(cell_colors[as.character(source_sectors$source)], source_sectors$ligand_sector),
    setNames(cell_colors[as.character(target_sectors$target)], target_sectors$receptor_sector)
  )[sector_order]
  link_colors <- if (link_color_mode == "source_cell") {
    unname(cell_colors[as.character(edges$source)])
  } else {
    unname(condition_colors[edges$higher_condition])
  }
  plot_edges <- data.frame(
    ligand = edges$ligand_sector,
    receptor = edges$receptor_sector,
    weight_abs = edges$weight_abs,
    stringsAsFactors = FALSE
  )

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  extension <- tolower(tools::file_ext(output_path))
  if (extension == "png") {
    grDevices::png(
      output_path, width = width_inches, height = height_inches,
      units = "in", res = png_res
    )
  } else if (extension == "pdf") {
    grDevices::pdf(
      output_path, width = width_inches, height = height_inches,
      useDingbats = FALSE
    )
  } else {
    stop("Output extension must be png or pdf")
  }

  circos.clear()
  circos.par(canvas.xlim = c(-1.35, 1.65), canvas.ylim = c(-1.3, 1.3))
  chordDiagram(
    plot_edges,
    order = sector_order,
    col = link_colors,
    grid.col = grid_colors,
    transparency = if (link_color_mode == "source_cell") 0.18 else 0.25,
    directional = 1,
    direction.type = "arrows",
    link.arr.type = "big.arrow",
    link.arr.col = link_colors,
    link.arr.length = 0.12,
    link.arr.width = 0.8,
    link.arr.lwd = 0,
    link.target.prop = TRUE,
    annotationTrack = "grid",
    annotationTrackHeight = 0.03,
    preAllocateTracks = list(track.height = max(strwidth(unname(sector_labels)))),
    small.gap = 1,
    big.gap = 10,
    reduce = -1
  )
  circos.track(track.index = 1, panel.fun = function(x, y) {
    xlim <- get.cell.meta.data("xlim")
    ylim <- get.cell.meta.data("ylim")
    sector_name <- get.cell.meta.data("sector.index")
    circos.text(
      mean(xlim), ylim[1], sector_labels[[sector_name]],
      facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5), cex = 0.85
    )
  }, bg.border = NA)

  cell_legend <- Legend(
    at = cell_states,
    type = "grid",
    legend_gp = gpar(fill = cell_colors[cell_states]),
    title = "Cell state"
  )
  condition_legend <- Legend(
    at = condition_levels,
    type = "grid",
    legend_gp = gpar(fill = condition_colors[condition_levels]),
    title = "Higher activity"
  )
  legend <- if (link_color_mode == "source_cell") {
    cell_legend
  } else {
    packLegend(condition_legend, cell_legend, direction = "vertical")
  }
  draw(
    legend,
    x = unit(1, "npc") - unit(7, "mm"),
    y = unit(7, "mm"),
    just = c("right", "bottom")
  )
  title(title_text, cex.main = 1.25, line = -1)
  circos.clear()
  grDevices::dev.off()
  invisible(normalizePath(output_path, mustWork = FALSE))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2L || length(args) > 5L) {
    stop(
      paste(
        "Usage: Rscript plot_liana_window_chord.R <input.csv> <output.png|pdf>",
        "[title] [condition|source_cell]"
      )
    )
  }
  output <- plot_liana_window_chord(
    input_path = args[[1]],
    output_path = args[[2]],
    title_text = if (length(args) >= 3L) args[[3]] else "LIANA pathway drivers",
    link_color_mode = if (length(args) >= 4L) args[[4]] else "condition"
  )
  cat(sprintf("wrote LIANA chord to %s\n", output))
}
