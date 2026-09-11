# ============================================================================
# figED10_calibration.R — Extended Data Figure 10: calibration / DCA / nomogram. Panels A-C: (A) cross-platform calibration (Wasserstein), (B) decision curve analysis, (C) nomogram discrimination (C-index).
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/depmap/crossplatform_calibration_stats.csv,
#           results/tcga/dca_summary.csv,
#           results/tcga/nomogram_cox_results.csv
# Outputs : results/figures/r/EDFig10_calibration_dca_nomogram.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figED10_calibration.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "EDFig10_calibration_dca_nomogram"
H_MM <- 2.4297 / 7.0866 * 180   # 61.7 mm (Python figsize 7.0866 x 2.4297 in)

cal <- fread(file.path(RESULTS_DIR, "depmap", "crossplatform_calibration_stats.csv"))
qn  <- cal[grepl("quantile", method, ignore.case = TRUE)]
dca <- fread(file.path(RESULTS_DIR, "tcga", "dca_summary.csv"))
setorder(dca, delta_vs_treatall_tau20)
nm  <- fread(file.path(RESULTS_DIR, "tcga", "nomogram_cox_results.csv"))
stopifnot(nrow(qn) == 1L, nrow(dca) == 7L, nrow(nm) == 7L)

# ---- Panel A: cross-platform calibration (Wasserstein) ----------------------------
cal_d <- data.table(
  x = factor(c(0, 1), levels = c(0, 1)),
  w = c(qn$wasserstein_before[1], qn$wasserstein_after[1]),
  col = c(COL_GDSC, COL_TCGA))
cat(sprintf("  [A] Wasserstein: raw=%.4f -> quantile_norm=%.4f (reduction %.1f%%)\n",
            cal_d$w[1], cal_d$w[2], 100 * qn$wasserstein_reduction[1]))
pA <- ggplot(cal_d, aes(x, w)) +
  geom_col(aes(fill = x), width = 0.5, alpha = 0.75) +
  scale_fill_manual(values = c(COL_GDSC, COL_TCGA), guide = "none") +
  scale_x_discrete(labels = c("raw", "quantile\nnormalized")) +
  geom_text(aes(label = ifelse(w >= 1, sprintf("%.2f", w), sprintf("%.3f", w))),
            vjust = -0.4, size = FONT_BASE/.pt) +
  labs(title = "A  Cross-platform calibration\n(GDSC to TCGA)",
       x = NULL, y = "Wasserstein distance") +
  theme_pub()

# ---- Panel B: decision curve analysis (delta net benefit vs treat-all) -------------
dca[, col := ifelse(delta_vs_treatall_tau20 > 0, COL_GDSC, COL_NS)]
dca[, y := seq_len(.N)]
pB <- ggplot(dca, aes(delta_vs_treatall_tau20, y)) +
  geom_col(aes(fill = col), width = 0.7, alpha = 0.75) +
  scale_fill_identity() +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  scale_y_continuous(breaks = dca$y, labels = sprintf("%s (n=%d)", dca$cancer, dca$n)) +
  labs(title = "B  DCA (delta NB vs. treat-all,\ntau=0.20; n per cancer as labeled)",
       x = "Delta net benefit (NetITH model vs. treat-all)", y = NULL) +
  theme_pub()

# ---- Panel C: nomogram discrimination (C-index) --------------------------------------
nmc <- melt(nm[, .(cancer, uv_concordance, mv_concordance)], id.vars = "cancer",
            variable.name = "model", value.name = "cindex")
nmc[, model := factor(model, levels = c("uv_concordance", "mv_concordance"),
                      labels = c("univariate", "multivariable"))]
nmc[, cancer := factor(cancer, levels = rev(nm$cancer))]
cat(sprintf("  [C] 7 cancers, uv C-index %.3f-%.3f, mv C-index %.3f-%.3f\n",
            min(nm$uv_concordance), max(nm$uv_concordance),
            min(nm$mv_concordance), max(nm$mv_concordance)))
pC <- ggplot(nmc, aes(cindex, cancer, shape = model, colour = model)) +
  geom_point(size = 2.2) +
  scale_shape_manual(values = c(univariate = 16, multivariable = 15)) +
  scale_colour_manual(values = c(univariate = COL_NS, multivariable = COL_SIG)) +
  labs(title = "C  Nomogram discrimination\n(7 cancers; NetITH not\nindependently prognostic)",
       x = "C-index", y = NULL) +
  theme_pub() + theme(legend.position = "bottom", legend.title = element_blank())

# ---- composite + save -------------------------------------------------------------------
p <- pA | pB | pC
save_fig(p, NAME, height_mm = H_MM)
save_panels(list(pA, pB, pC), NAME)

# ---- print key numbers ---------------------------------------------------------------------
cat("\n=== EDFig10 key numbers ===\n")
cat(sprintf("  Wasserstein: before=%.3f, after=%.3f (quantile_norm)\n",
            qn$wasserstein_before[1], qn$wasserstein_after[1]))
cat(sprintf("  DCA cancers (delta NB > 0): %s\n",
            paste(dca[delta_vs_treatall_tau20 > 0, cancer], collapse = ", ")))
cat(sprintf("  Nomogram mv C-index range: %.3f-%.3f\n", min(nm$mv_concordance), max(nm$mv_concordance)))
