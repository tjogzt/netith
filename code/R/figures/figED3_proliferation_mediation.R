# ============================================================================
# figED3_proliferation_mediation.R — Extended Data Figure 3: proliferation mediation. Panels A-C: (A) bootstrap median mediated proportion by pathway, (B) per-drug bootstrap indirect-effect waterfall, (C) NetITH-IC50 Spearman rho distribution.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/depmap/ext_d3_proliferation_mediation.csv,
#           results/control/mediation_bootstrap.csv,
#           results/gdsc/gdsc_drug_netith_correlations.csv
# Outputs : results/figures/r/EDFig3_proliferation_mediation.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figED3_proliferation_mediation.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "EDFig3_proliferation_mediation"
DEP <- file.path(RESULTS_DIR, "depmap")
GDSC <- file.path(RESULTS_DIR, "gdsc")

cat("=== EDFig3: Proliferation mediation ===\n")

med  <- fread(file.path(DEP, "ext_d3_proliferation_mediation.csv"))
medb <- fread(file.path(RESULTS_DIR, "control", "mediation_bootstrap.csv"))
dc   <- fread(file.path(GDSC, "gdsc_drug_netith_correlations.csv"))
stopifnot(all(c("drug", "sobel_p", "prop_mediated") %in% names(med)),
          all(c("drug", "prop_median", "ind_ci_excludes_0") %in% names(medb)),
          all(c("drug", "rho", "pathway") %in% names(dc)))

# ---- Panel A: mean mediated proportion by drug pathway -----------------------
medb[, drug_upper := toupper(drug)]
dc[, drug_upper := toupper(drug)]
mb <- merge(medb, dc[, .(drug_upper, pathway)], by = "drug_upper", allow.cartesian = TRUE)
cp <- mb[, .(m = median(prop_median)), by = pathway][order(-m)]
overall <- median(mb$prop_median)
pA <- ggplot(cp, aes(pathway, 100 * m)) +
  geom_col(fill = COL_NER, alpha = 0.85, width = 0.6) +
  geom_hline(yintercept = 100 * overall, linetype = "dashed", colour = COL_SIG, linewidth = 0.4) +
  coord_flip() +
  annotate("text", x = 0.6, y = Inf, hjust = 1.05, vjust = 1.1, size = FONT_BASE/.pt,
           label = sprintf("Overall median = %.1f%%", 100 * overall)) +
  labs(title = "A  Bootstrap median mediated proportion by drug class",
       x = NULL, y = "Median mediated proportion (%)") +
  theme_pub() +
  theme(axis.text.y = element_text(size = FONT_BASE))

# ---- Panel B: per-drug bootstrap indirect-effect estimates (waterfall) ------
ms <- medb[order(-prop_median)]
ms[, rank := .I]
ms[, ci0 := ind_ci_excludes_0 %in% c(TRUE, "True", "TRUE")]
ms[, col := ifelse(ci0, COL_SIG, COL_NS)]
pB <- ggplot(ms, aes(rank, 100 * prop_median)) +
  geom_col(aes(fill = col), width = 0.8) +
  scale_fill_identity() +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey30") +
  annotate("text", x = 0.02 * nrow(ms), y = Inf, hjust = 0, vjust = 1.4,
           size = FONT_BASE/.pt, colour = COL_SIG,
           label = sprintf("%d/286 CI excludes zero", sum(ms$ci0))) +
  labs(title = "B  Per-drug bootstrap indirect-effect estimates",
       x = "Drug rank", y = "Bootstrap median mediated proportion (%)") +
  theme_pub()

# ---- Panel C: NetITH-IC50 rho distribution -----------------------------------
rho_b <- dc$rho[is.finite(dc$rho)]
n_pos <- sum(rho_b > 0)
mean_rho <- mean(rho_b)

pC <- ggplot(data.frame(rho = rho_b), aes(rho)) +
  geom_histogram(bins = 40, fill = COL_NETITH, alpha = 0.6, colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = mean_rho, linetype = "dashed", colour = COL_NER, linewidth = 0.5) +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = FONT_BASE/.pt,
           fontface = "bold",
           label = sprintf("%d/%d positive\nmean \u03c1=%.3f", n_pos, length(rho_b), mean_rho)) +
  labs(title = "C  NetITH\u2013IC\u2085\u2080 \u03c1: before vs. after proliferation adjustment",
       x = "Spearman \u03c1 (NetITH vs. ln IC\u2085\u2080)", y = "Drug count") +
  theme_pub()

p <- pA / pB / pC
save_fig(p, NAME, height_mm = 185.7)
save_panels(list(pA, pB, pC), NAME)

cat(sprintf("  A: %d drugs, %d pathways; overall bootstrap median = %.1f%%\n",
            nrow(medb), nrow(cp), 100 * overall))
cat(sprintf("  B: bootstrap CI excludes zero: %d / %d (%.1f%%)\n",
            sum(ms$ci0), nrow(ms), 100 * sum(ms$ci0) / nrow(ms)))
cat(sprintf("  C: rho distribution: %d/%d positive, mean rho = %.3f\n",
            n_pos, length(rho_b), mean_rho))
cat(sprintf("  PDF: %s\n", file.path(FIG_DIR, paste0(NAME, ".pdf"))))
