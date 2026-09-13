# ============================================================================
# 12_prism_root_cause.R — PRISM null root-cause: readout/drug-coverage/cell-line three-factor attribution.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-09
# Purpose : PRISM root-cause three-factor attribution (four-team round-4
#           gatekeeper). Decompose the PRISM null (median rho=0.002, 0/1,482
#           FDR<0.05) into three factors using only data in hand:
#           (1) readout factor — GDSC ln(IC50) vs AUC on the same drugs
#               (the NetITH association is IC50-specific; PRISM's readout is
#               a log2-fold-change viability metric of the AUC family);
#           (2) drug-spectrum/coverage factor — stratification of the 1,482
#               archived PRISM per-drug results by matched line count, and a
#               paired shared-drug analysis (same drug in GDSC vs PRISM);
#           (3) cell-line factor — GDSC subsample null at the PRISM-matched
#               size (487 lines, 100 draws): does the sample-size/coverage
#               reduction alone abolish the association?
#           The construction-artefact contribution is bounded by the residual
#           of the paired shared-drug analysis after the readout comparison.
#           The PRISM raw matrix is unavailable (figshare 403 from this
#           network); the archived per-drug results (from the published
#           validation run) are the authoritative PRISM numbers.
# Inputs  : results/control/prism_validation_drugs.csv (archived),
#           results/control/auc_sensitivity.json,
#           results/gdsc/netith_r_cell_lines.csv, GDSC2_IC50_all.csv
# Outputs : results/control/prism_root_cause.{json,md},
#           results/control/prism_root_cause_drugs.csv
# Usage   : Rscript code/R/12_prism_root_cause.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({library(data.table); library(jsonlite)})
set.seed(SEED)

netith <- fread(file.path(RESULTS_DIR, "gdsc", "netith_r_cell_lines.csv"))
nt <- setNames(netith$NetITH, netith$cell_line)
ic50 <- fread(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"))
ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)
auc_mat  <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "AUC", fun.aggregate = mean)

drug_assoc <- function(nt, mat) {
  common_cl <- intersect(names(nt), mat$CELL_LINE_NAME)
  ntv <- nt[common_cl]
  drugs <- setdiff(names(mat), "CELL_LINE_NAME")
  vapply(drugs, function(d) {
    y <- mat[[d]][match(common_cl, mat$CELL_LINE_NAME)]
    ok <- is.finite(ntv) & is.finite(y)
    if (sum(ok) < 10) return(NA_real_)
    suppressWarnings(cor(ntv[ok], y[ok], method = "spearman"))
  }, numeric(1))
}
gdsc_ic50_rho <- drug_assoc(nt, ic50_mat)
gdsc_auc_rho  <- drug_assoc(nt, auc_mat)
cat(sprintf("[PRISM-rc] GDSC IC50: median rho=%.4f (n=%d); AUC: median rho=%.4f (n=%d)\n",
            median(gdsc_ic50_rho, na.rm=TRUE), sum(!is.na(gdsc_ic50_rho)),
            median(gdsc_auc_rho, na.rm=TRUE), sum(!is.na(gdsc_auc_rho))))

# --- archived PRISM per-drug results ---
prism <- fread(file.path(RESULTS_DIR, "control", "prism_validation_drugs.csv"))
cat(sprintf("[PRISM-rc] archived PRISM: %d compounds, median rho=%.4f, pos %d/%d, FDR<0.05 n=%d\n",
            nrow(prism), median(prism$rho), sum(prism$rho > 0), nrow(prism),
            sum(prism$fdr < 0.05)))

# --- factor 2a: stratification by matched n (coverage) ---
strata <- cut(prism$n, breaks = c(0, 50, 100, 200, 300, 500), include.lowest = TRUE)
strat_tbl <- prism[, .(n_compounds = .N,
                       median_rho = median(rho),
                       n_pos = sum(rho > 0),
                       n_fdr05 = sum(fdr < 0.05)), by = strata]
cat("[PRISM-rc] stratification by matched n:\n"); print(strat_tbl)

# --- factor 2b: paired shared-drug analysis (same drug, GDSC vs PRISM) ---
norm_name <- function(x) gsub("[^a-z0-9]", "", tolower(x))
prism[, drug_norm := norm_name(drug)]
gdsc_names <- names(gdsc_ic50_rho)
gmap <- data.table(drug = gdsc_names, drug_norm = norm_name(gdsc_names))
gmap <- gmap[!duplicated(drug_norm)]
shared <- merge(prism, gmap, by = "drug_norm", suffixes = c("_prism", "_gdsc"))
shared[, gdsc_ic50 := gdsc_ic50_rho[drug_gdsc]]
shared[, gdsc_auc  := gdsc_auc_rho[drug_gdsc]]
cat(sprintf("[PRISM-rc] shared drugs after name normalization: %d\n", nrow(shared)))
if (nrow(shared) > 0) {
  print(shared[order(-gdsc_ic50), .(drug_gdsc, n, rho, gdsc_ic50, gdsc_auc)][1:10])
  wt_ic50 <- suppressWarnings(wilcox.test(shared$gdsc_ic50, shared$rho, paired = TRUE))
  wt_auc  <- suppressWarnings(wilcox.test(shared$gdsc_auc, shared$rho, paired = TRUE))
  cat(sprintf("[PRISM-rc] paired shared-drug: GDSC-IC50 median=%.4f vs PRISM median=%.4f (Wilcoxon p=%.3g)\n",
              median(shared$gdsc_ic50), median(shared$rho), wt_ic50$p.value))
  cat(sprintf("[PRISM-rc] paired shared-drug: GDSC-AUC  median=%.4f vs PRISM median=%.4f (Wilcoxon p=%.3g)\n",
              median(shared$gdsc_auc), median(shared$rho), wt_auc$p.value))
  resid_auc <- median(shared$rho - shared$gdsc_auc)
  resid_ic50 <- median(shared$rho - shared$gdsc_ic50)
} else {
  wt_ic50 <- wt_auc <- list(p.value = NA_real_)
  resid_auc <- resid_ic50 <- NA_real_
}

# --- factor 3: GDSC subsample null at n=487 (PRISM-matched size) ---
N_SUB <- 488L; N_DRAW <- 100L
all_cl <- intersect(names(nt), ic50_mat$CELL_LINE_NAME)
sub_med <- numeric(N_DRAW)
set.seed(SEED)  # seed ONCE: the RNG must advance between draws
for (k in seq_len(N_DRAW)) {
  sub_cl <- sample(all_cl, N_SUB)
  sub_nt <- nt[sub_cl]
  sub_rho <- vapply(setdiff(names(ic50_mat), "CELL_LINE_NAME"), function(d) {
    y <- ic50_mat[[d]][match(sub_cl, ic50_mat$CELL_LINE_NAME)]
    ok <- is.finite(sub_nt) & is.finite(y)
    if (sum(ok) < 10) return(NA_real_)
    suppressWarnings(cor(sub_nt[ok], y[ok], method = "spearman"))
  }, numeric(1))
  sub_med[k] <- median(sub_rho, na.rm = TRUE)
}
cat(sprintf("[PRISM-rc] GDSC %d-line subsample null: median rho mean=%.4f sd=%.4f range=[%.4f,%.4f]\n",
            N_SUB, mean(sub_med), sd(sub_med), min(sub_med), max(sub_med)))

out <- list(
  gdsc_ic50_median_rho = unname(median(gdsc_ic50_rho, na.rm = TRUE)),
  gdsc_auc_median_rho = unname(median(gdsc_auc_rho, na.rm = TRUE)),
  prism_median_rho = unname(median(prism$rho)),
  prism_n_pos = sum(prism$rho > 0), prism_n_drugs = nrow(prism),
  prism_n_fdr05 = sum(prism$fdr < 0.05),
  strata = as.data.frame(strat_tbl),
  shared_drugs_n = nrow(shared),
  shared_gdsc_ic50_median = if (nrow(shared)) unname(median(shared$gdsc_ic50)) else NA_real_,
  shared_gdsc_auc_median  = if (nrow(shared)) unname(median(shared$gdsc_auc))  else NA_real_,
  shared_prism_median     = if (nrow(shared)) unname(median(shared$rho))       else NA_real_,
  paired_wilcoxon_ic50_p  = wt_ic50$p.value,
  paired_wilcoxon_auc_p   = wt_auc$p.value,
  residual_median_prism_minus_gdsc_auc = resid_auc,
  residual_median_prism_minus_gdsc_ic50 = resid_ic50,
  subsample488_null_mean = mean(sub_med), subsample488_null_sd = sd(sub_med),
  subsample488_null_range = range(sub_med), n_sub = N_SUB, n_draw = N_DRAW,
  note = "PRISM raw matrix unavailable (figshare 403 from this network); archived per-drug PRISM results are the authoritative PRISM numbers; cell-line factor approximated by GDSC subsample null at the PRISM-matched size."
)
write_json(out, file.path(RESULTS_DIR, "control", "prism_root_cause.json"),
           auto_unbox = TRUE, digits = 6, pretty = TRUE)

if (nrow(shared) > 0) {
  write.csv(shared[, .(drug_gdsc, drug_prism, n_prism = n,
                       rho_prism = rho, p_prism = p, fdr_prism = fdr,
                       gdsc_ic50, gdsc_auc)],
            file.path(RESULTS_DIR, "control", "prism_root_cause_drugs.csv"),
            row.names = FALSE)
}
cat("\n[PRISM-rc] done\n")
