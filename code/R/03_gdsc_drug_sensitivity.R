# ============================================================================
# 03_gdsc_drug_sensitivity.R — GDSC2 drug sensitivity: NetITH vs ln(IC50) Spearman correlations with BH FDR.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : GDSC2 drug-sensitivity analysis: NetITH vs. drug response (IC50)
#           Spearman correlations across cell lines, with BH/FDR control.
#           Reproduces scripts/01_core/run_gdsc_drug_sensitivity.py in R and
#           cross-checks against the cached Python results.
# Inputs  : NetITH scores (results/gdsc/netith_r_cell_lines.csv or cached),
#           GDSC2_IC50_all.csv (data disk or data/gdsc)
# Outputs : results/gdsc/gdsc_drug_netith_correlations_r.csv
# Usage   : Rscript code/R/03_gdsc_drug_sensitivity.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))

suppressPackageStartupMessages(library(data.table))

# ---- locate IC50 table (data disk first, then local copy) -------------------
find_ic50 <- function() {
  cand <- c(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"),
            file.path(DATA_DIR, "gdsc", "GDSC2_IC50_all.csv"))
  hit <- cand[file.exists(cand)]
  if (length(hit) == 0L) stop(paste("GDSC2_IC50_all.csv not found (mount", DATA_DISK, "or copy into data/gdsc/)"))
  hit[1]
}

main <- function() {
  ic50_path <- find_ic50()
  ic50 <- fread(ic50_path)
  stopifnot(all(c("CELL_LINE_NAME", "DRUG_NAME", "LN_IC50", "CANCER_TYPE") %in% names(ic50)))
  message(sprintf("[ic50] %d records, %d drugs, %d cell lines",
                  nrow(ic50), uniqueN(ic50$DRUG_NAME), uniqueN(ic50$CELL_LINE_NAME)))

  # NetITH scores (prefer the freshly computed R table; fall back to cached Python)
  r_path <- file.path(RESULTS_DIR, "gdsc", "netith_r_cell_lines.csv")
  if (file.exists(r_path)) {
    netith <- fread(r_path)
  } else {
    netith <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
  }
  setnames(netith, c("cell_line", "NetITH"))

  # pivot LN_IC50 to cell line x drug (mean over replicates)
  ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)

  common_cl <- intersect(netith$cell_line, ic50_mat$CELL_LINE_NAME)
  message(sprintf("[drug] %d cell lines with both NetITH and IC50", length(common_cl)))
  stopifnot(length(common_cl) > 50)

  nt <- netith$NetITH[match(common_cl, netith$cell_line)]
  drugs <- setdiff(names(ic50_mat), "CELL_LINE_NAME")

  # Spearman per drug (two-sided), then BH FDR
  res <- lapply(drugs, function(d) {
    y <- ic50_mat[[d]][match(common_cl, ic50_mat$CELL_LINE_NAME)]
    st <- spearman_test(nt, y)
    # Mann-Whitney: top vs bottom NetITH tertile IC50
    q <- stats::quantile(nt, c(1/3, 2/3), na.rm = TRUE)
    grp <- ifelse(nt <= q[1], "low", ifelse(nt >= q[2], "high", NA_character_))
    yg <- y[!is.na(grp) & !is.na(y)]
    gg <- grp[!is.na(grp) & !is.na(y)]
    mw <- NA_real_
    if (length(yg) >= 6 && length(unique(gg)) == 2) {
      mw <- suppressWarnings(wilcox.test(yg ~ factor(gg))$p.value)
    }
    data.table(drug = d, n_cells = st[["n"]], rho = st[["rho"]], p_spearman = st[["p"]],
               p_mannwhitney = mw)
  })
  res <- rbindlist(res)
  # Benjamini-Hochberg FDR control across the per-drug Spearman p-values
  res[, fdr := p.adjust(p_spearman, method = "BH")]
  res[, neg_log10_p := -log10(p_spearman)]
  setorder(res, p_spearman)

  out_path <- file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations_r.csv")
  fwrite(res, out_path)
  n_sig <- sum(res$fdr < 0.05, na.rm = TRUE)
  n_pos <- sum(res$rho > 0, na.rm = TRUE)
  cat(sprintf("[drug] %d drugs tested; rho>0: %d; FDR<0.05: %d\n", nrow(res), n_pos, n_sig))

  # ---- consistency vs cached Python table ----
  cached <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations.csv"))
  m <- merge(res, cached[, .(drug, rho_py = rho, fdr_py = fdr)], by = "drug")
  cc <- cor(m$rho, m$rho_py, method = "spearman", use = "complete.obs")
  cat(sprintf("[check] R vs Python drug rho: Spearman=%.6f (%d drugs)\n", cc, nrow(m)))
  invisible(res)
}

main()
