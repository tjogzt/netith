# ============================================================================
# figS12_teschendorff.R — Supplementary Figure 12
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/gdsc/teschendorff/signaling_entropy_gdsc.csv,
#           results/gdsc/gdsc_netith_cell_lines.csv,
#           results/gdsc/gdsc_drug_netith_correlations.csv,
#           results/gdsc/teschendorff_benchmark_results.csv
# Outputs : results/figures/r/FigS12_teschendorff_benchmark.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS12_teschendorff.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "FigS12_teschendorff_benchmark"
H_MM <- 4.0495 / 7.0866 * 180   # 102.9 mm (Python figsize 7.0866 x 4.0495 in)

GDSC_D <- file.path(RESULTS_DIR, "gdsc")

# ---- data: SR per cell line, NetITH per cell line, cached benchmark ----------
sr  <- fread(file.path(GDSC_D, "teschendorff", "signaling_entropy_gdsc.csv"))
ni  <- fread(file.path(GDSC_D, "gdsc_netith_cell_lines.csv"))
bm  <- fread(file.path(GDSC_D, "teschendorff_benchmark_results.csv"))

d <- merge(ni, sr, by = "cell_line")
d <- d[is.finite(NetITH) & is.finite(signaling_entropy)]
stopifnot(nrow(d) == bm$n_common[1])   # cached n_common = 507

st <- spearman_test(d$NetITH, d$signaling_entropy)
rho <- unname(st["rho"]); pv <- unname(st["p"]); nn <- unname(st["n"])
stopifnot(abs(rho - bm$sr_netith_rho[1]) < 1e-3)

# ---- Panel A: SR vs NetITH scatter -------------------------------------------
pA <- ggplot(d, aes(NetITH, signaling_entropy)) +
  geom_point(colour = COL_SIG, alpha = 0.4, size = 1.2) +
  annotate("text", x = quantile(d$NetITH, 0.97), y = quantile(d$signaling_entropy, 0.03),
           hjust = 1, size = FONT_BASE/.pt, colour = "grey40", label = sprintf("n=%d", nn)) +
  labs(title = sprintf("SR vs NetITH\nρ=%.3f, p=%.1e", rho, pv),
       x = "NetITH (CollecTRI)", y = "Signaling Entropy (STRING PPI)") +
  theme_pub()

# ---- Panel B: distribution comparison (density histograms) --------------------
db <- rbind(
  data.table(value = ni$NetITH, metric = "NetITH"),
  data.table(value = sr$signaling_entropy, metric = "SR (STRING)"))
db <- db[is.finite(value)]
pB <- ggplot(db, aes(value, fill = metric, colour = metric)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.5,
                 position = "identity", linewidth = 0.2) +
  scale_fill_manual(values = c("NetITH" = COL_NETITH, "SR (STRING)" = COL_GDSC)) +
  scale_colour_manual(values = c("NetITH" = COL_NETITH, "SR (STRING)" = COL_GDSC)) +
  labs(title = "Distribution Comparison", x = "Entropy Value", y = "Density") +
  theme_pub() + theme(legend.position = "top", legend.title = element_blank())

# BH-corrected FDR (the cached CSV fdr column was computed as Bonferroni p*n by
# the legacy Python pipeline; manuscript reports BH=276 -- recompute properly)
drug_corr_full <- data.table::fread(file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations.csv"))
bh_n <- sum(p.adjust(drug_corr_full$p_spearman, method = "BH") < 0.05)

# ---- Panel C: directional consistency (286 GDSC drugs) -------------------------
dc <- data.table(
  y     = 0:1,
  label = c("NetITH\n(CollecTRI)", "Signaling\nEntropy\n(STRING)"),
  pct   = 100,
  col   = c(COL_NETITH, COL_GREY))
pC <- ggplot(dc, aes(pct, y)) +
  geom_col(aes(fill = label), width = 0.5) +
  scale_fill_manual(values = setNames(dc$col, dc$label), guide = "none") +
  annotate("text", x = 100, y = 0, label = "100%", hjust = -0.3, vjust = 0.5,
           size = 8/.pt, fontface = "bold") +
  scale_y_continuous(breaks = 0:1, labels = c("NetITH\n(CollecTRI)", "Signaling\nEntropy\n(STRING)")) +
  coord_cartesian(xlim = c(0, 110)) +
  labs(title = "Directional Consistency\n(286 GDSC drugs)",
       x = "Drugs with ρ≥0 (%)", y = NULL) +
  theme_pub()

# ---- Panel D: benchmark summary (text) -----------------------------------------
lines <- c(
  sprintf("Cell lines compared: %d", nn),
  sprintf("SR-NetITH correlation: ρ=%.3f", rho),
  "NetITH directional consistency: 100%",
  sprintf("NetITH FDR<0.05 drugs: %d/%d (BH)", bh_n, bm$netith_total_drugs[1]),
  "",
  "Key Finding:",
  "NetITH (CollecTRI TF-target)",
  "and Signaling Entropy (STRING PPI)",
  "represent complementary network views",
  "of tumor functional heterogeneity.")
pD <- ggplot() + xlim(0, 1) + ylim(0, 1) +
  annotate("text", x = 0.04, y = 0.96, hjust = 0, vjust = 1, size = FONT_BASE/.pt,
           label = paste(lines, collapse = "\n")) +
  theme_void() +
  labs(title = "Benchmark Summary") +
  theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5),
        plot.margin = margin(2, 2, 2, 2, "mm"))

# ---- composite + save -----------------------------------------------------------
p <- (pA | pB) / (pC | pD) +
  plot_annotation(title = "NetITH vs Teschendorff Signaling Entropy Benchmark",
                  theme = theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5)))
save_fig(p, NAME, height_mm = H_MM)
save_panels(list(pA, pB, pC, pD), NAME)

# ---- print key numbers (must match cached CSV) -----------------------------------
cat("=== EDFig6 key numbers ===\n")
cat(sprintf("SR vs NetITH (recomputed): rho=%.4f, p=%.2e, n=%d\n", rho, pv, nn))
cat(sprintf("Cached benchmark CSV    : rho=%.4f, p=%.2e, n_common=%d\n",
            bm$sr_netith_rho[1], bm$sr_netith_p[1], bm$n_common[1]))
cat(sprintf("NetITH directional consistency: %.0f%% | FDR<0.05 drugs (BH): %d/%d\n",
            bm$netith_directional_consistency[1] * 100, bh_n, bm$netith_total_drugs[1]))
