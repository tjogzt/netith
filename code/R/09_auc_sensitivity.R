# ============================================================================
# 09_auc_sensitivity.R — GDSC readout sensitivity: NetITH drug associations using AUC vs ln(IC50).
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : Review-mandated analysis 2 (PRISM root-cause substitute): GDSC
#           readout sensitivity — recompute the 286-drug NetITH associations
#           using AUC instead of ln(IC50). If the association persists with AUC,
#           it is not an artefact of the IC50 readout alone; the PRISM raw matrix
#           is unavailable (403 placeholder on the data disk), so the assay-
#           specificity statement rests on this within-GDSC readout comparison
#           plus the published PRISM summary.
# Inputs  : results/gdsc/netith_r_cell_lines.csv, GDSC2_IC50_all.csv (has AUC)
# Outputs : results/control/auc_sensitivity.json
# Usage   : Rscript code/R/09_auc_sensitivity.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages(library(data.table))

netith <- fread(file.path(RESULTS_DIR, "gdsc", "netith_r_cell_lines.csv"))
ic50 <- fread(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"))
stopifnot("AUC" %in% names(ic50))

auc_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "AUC", fun.aggregate = mean)
ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)

assoc <- function(mat) {
  common_cl <- intersect(netith$cell_line, mat$CELL_LINE_NAME)
  ntv <- netith$NetITH[match(common_cl, netith$cell_line)]
  drugs <- setdiff(names(mat), "CELL_LINE_NAME")
  rhos <- vapply(drugs, function(d) {
    y <- mat[[d]][match(common_cl, mat$CELL_LINE_NAME)]
    ok <- is.finite(ntv) & is.finite(y)
    if (sum(ok) < 10) return(NA_real_)
    suppressWarnings(cor(ntv[ok], y[ok], method = "spearman"))
  }, numeric(1))
  rhos[!is.na(rhos)]
}
r_ic50 <- assoc(ic50_mat); r_auc <- assoc(auc_mat)
common_d <- intersect(names(r_ic50), names(r_auc))
r_cross <- suppressWarnings(cor(r_ic50[common_d], r_auc[common_d], method = "spearman"))

out <- list(n_drugs = length(r_auc),
            ic50_median_rho = median(r_ic50), ic50_pos = sum(r_ic50 > 0),
            auc_median_rho = median(r_auc), auc_pos = sum(r_auc > 0),
            ic50_auc_rho = r_cross,
            prism_raw_unavailable = TRUE,
            note = "PRISM_19Q4_secondary.csv on data disk is a 403 placeholder; root-cause limited to GDSC readout sensitivity (AUC vs IC50) plus published PRISM summary")
jsonlite::write_json(out, file.path(RESULTS_DIR, "control", "auc_sensitivity.json"), auto_unbox = TRUE, digits = 6)
cat(sprintf("[auc] IC50 median rho=%.4f (%d/%d pos) | AUC median rho=%.4f (%d/%d pos) | cross rho=%.4f\n",
            median(r_ic50), sum(r_ic50 > 0), length(r_ic50), median(r_auc), sum(r_auc > 0),
            length(r_auc), r_cross))
