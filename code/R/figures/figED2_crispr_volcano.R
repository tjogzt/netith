# ============================================================================
# figED2_crispr_volcano.R — Extended Data Figure 2: genome-wide CRISPR-CERES vs NetITH volcano (single panel). Significance by Bonferroni threshold 0.05/n.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/depmap/ext_d1_crispr_netith_correlations.csv
# Outputs : results/figures/r/EDFig2_crispr_volcano.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figED2_crispr_volcano.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "EDFig2_crispr_volcano"
DEP <- file.path(RESULTS_DIR, "depmap")

cat("=== EDFig2: CRISPR-NetITH volcano ===\n")

cr <- fread(file.path(DEP, "ext_d1_crispr_netith_correlations.csv"))
stopifnot(all(c("gene", "rho_crispr_netith", "p_crispr_netith") %in% names(cr)))
cr <- cr[is.finite(rho_crispr_netith) & is.finite(p_crispr_netith) & p_crispr_netith > 0]
n_g <- nrow(cr)
bonf <- 0.05 / n_g
cr[, neg_log_p := -log10(pmax(p_crispr_netith, 1e-100))]
cr[, sig := p_crispr_netith < bonf]
n_sig <- sum(cr$sig)

pA <- ggplot(cr, aes(rho_crispr_netith, neg_log_p)) +
  geom_point(data = cr[sig == FALSE], colour = COL_NS, size = 0.3, alpha = 0.15, shape = 16) +
  geom_point(data = cr[sig == TRUE], colour = COL_SIG, size = 1.5, alpha = 0.6, shape = 16) +
  geom_hline(yintercept = -log10(bonf), linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  annotate("text", x = min(cr$rho_crispr_netith), y = -log10(bonf), hjust = 0, vjust = -0.6,
           size = FONT_BASE/.pt, colour = "grey40",
           label = sprintf("Bonferroni (%.2e)", bonf)) +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = FONT_BASE/.pt,
           fontface = "bold", colour = COL_SIG,
           label = sprintf("%d / %d genes FDR<0.05", n_sig, n_g)) +
  labs(title = "CRISPR\u2013NetITH genome-wide (DepMap)",
       x = "\u03c1 (CRISPR CERES vs. NetITH)",
       y = "-log\u2081\u2080(p)") +
  theme_pub()

save_fig(pA, NAME, height_mm = 139.7, width_mm = 85)
save_panels(list(pA), NAME)

cat(sprintf("  Genes tested: %d\n", n_g))
cat(sprintf("  Bonferroni threshold: %.3e\n", bonf))
cat(sprintf("  Significant (p < bonf): %d / %d (%.1f%%)\n",
            n_sig, n_g, 100 * n_sig / n_g))
cat(sprintf("  rho range: %.3f .. %.3f\n", min(cr$rho_crispr_netith), max(cr$rho_crispr_netith)))
cat(sprintf("  PDF: %s\n", file.path(FIG_DIR, paste0(NAME, ".pdf"))))
