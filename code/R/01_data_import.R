# ============================================================================
# 01_data_import.R — Import core NetITH datasets: CollecTRI network, GDSC expression, focused genes, cached results.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-08-19
# Purpose : Import all core datasets for the NetITH R re-analysis:
#           (1) CollecTRI TF-target network; (2) GDSC expression + annotation;
#           (3) GDSC drug response (IC50/AUC); (4) cached NetITH scores and
#           downstream result tables produced by the Python pipeline, used for
#           number-consistency checks.
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv,
#           data/gdsc/cell_annot.csv, data/gdsc/ensg_symbol_map.csv,
#           results/*.csv (cached analysis tables)
# Outputs : list() of imported data.frame objects (returned, not written)
# Usage   : dat <- source("01_data_import.R")$value
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
})


# locate project root first (config needs it)
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))

import_collectri <- function() {
  net <- fread(file.path(DATA_DIR, "collectri_network.csv"))
  stopifnot(all(c("source", "target", "weight") %in% names(net)))
  net[, weight := as.numeric(weight)]
  message(sprintf("[import] CollecTRI: %d edges, %d unique TFs, %d unique targets",
                  nrow(net), uniqueN(net$source), uniqueN(net$target)))
  net[]
}

import_gdsc_expression <- function() {
  expr <- as.data.frame(fread(file.path(DATA_DIR, "gdsc", "rna_expr.csv"),
                              header = TRUE))
  rownames(expr) <- as.character(expr[[1]])
  expr <- expr[, -1, drop = FALSE]
  annot <- fread(file.path(DATA_DIR, "gdsc", "cell_annot.csv"))
  gene_map <- fread(file.path(DATA_DIR, "gdsc", "ensg_symbol_map.csv"))

  # 1) CEL file -> cell line name (Characteristics.cell.line.)
  stopifnot("Characteristics.cell.line." %in% names(annot))
  cel_map <- setNames(annot[["Characteristics.cell.line."]], annot[[1]])
  cel_map <- cel_map[!is.na(cel_map) & nzchar(cel_map)]
  common_cels <- intersect(colnames(expr), names(cel_map))
  expr <- expr[, common_cels, drop = FALSE]
  colnames(expr) <- unname(cel_map[colnames(expr)])
  # drop duplicate cell lines (keep first)
  expr <- expr[, !duplicated(colnames(expr)), drop = FALSE]

  # 2) ENSG -> symbol mapping (drop NA/duplicate symbols BEFORE assigning rownames)
  ensg_to_sym <- setNames(gene_map$symbol, gene_map$ensg)
  ensg_to_sym <- ensg_to_sym[!is.na(ensg_to_sym) & nzchar(ensg_to_sym)]
  matched <- rownames(expr) %in% names(ensg_to_sym)
  expr <- expr[matched, , drop = FALSE]
  new_sym <- unname(ensg_to_sym[rownames(expr)])
  keep <- !duplicated(new_sym)
  expr <- expr[keep, , drop = FALSE]
  rownames(expr) <- new_sym[keep]

  message(sprintf("[import] GDSC expression: %d genes x %d cell lines (ENSG->symbol, CEL->cell line)",
                  nrow(expr), ncol(expr)))
  list(expr = expr, annot = annot, gene_map = gene_map)
}

import_focused_genes <- function() {
  f <- file.path(RESULTS_DIR, "focused_genes_collectri.txt")
  if (!file.exists(f)) stop("focused_genes_collectri.txt not found: ", f)
  genes <- readLines(f)
  genes <- genes[nzchar(genes)]
  message(sprintf("[import] Focused genes: %d", length(genes)))
  genes
}

# Cached Python-pipeline results used for consistency checks
import_cached <- function() {
  gdsc_netith <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
  drug_corr   <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations.csv"))
  tcga_netith <- fread(file.path(RESULTS_DIR, "tcga", "tcga_netith.csv"))
  message("[import] Cached Python results loaded (consistency-check targets)")
  list(gdsc_netith = gdsc_netith, drug_corr = drug_corr, tcga_netith = tcga_netith)
}

main <- function() {
  list(
    collectri = import_collectri(),
    gdsc      = import_gdsc_expression(),
    focused   = import_focused_genes(),
    cached    = import_cached()
  )
}

main()
