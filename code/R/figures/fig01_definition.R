# ============================================================================
# fig01_definition.R — Figure 1: NetITH definition and distributions. Panels A-D: (A) computation schematic, (B) GDSC/TCGA distributions, (C) TCGA primary tumour vs matched normal, (D) drug-association strength vs degree-preserving null.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/gdsc/gdsc_netith_cell_lines.csv, results/tcga/tcga_netith.csv,
#           results/control/random_graph_null_summary.json
# Outputs : results/figures/r/Fig1_netith_definition_distribution.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/fig01_definition.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "Fig1_netith_definition_distribution"

gdsc <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
tcga <- fread(file.path(RESULTS_DIR, "tcga", "tcga_netith.csv")); tcga[, 1 := NULL]
null <- jsonlite::fromJSON(file.path(RESULTS_DIR, "control", "random_graph_null_summary.json"))

# ---- Panel A: schematic (pure ggplot annotation) ----------------------------
pA <- ggplot() + xlim(0, 8) + ylim(0, 6) +
  annotate("rect", xmin = c(0.5, 3.0, 5.5), xmax = c(2.5, 5.0, 7.5),
           ymin = c(4.2, 4.2, 4.2), ymax = c(5.35, 5.35, 5.35),
           fill = COL_NETITH, alpha = 0.12, colour = COL_NETITH, linewidth = 1.0) +
  annotate("rect", xmin = c(0.5, 3.0, 5.5), xmax = c(2.5, 5.0, 7.5),
           ymin = c(1.2, 1.2, 1.2), ymax = c(2.35, 2.35, 2.35),
           fill = c(COL_NETITH, COL_NER, COL_NER), alpha = 0.12,
           colour = c(COL_NETITH, COL_NER, COL_NER), linewidth = 1.0) +
  annotate("text", x = c(1.5, 4.0, 6.5, 1.5, 4.0, 6.5),
           y = c(4.77, 4.77, 4.77, 1.77, 1.77, 1.77),
           label = c("TF\u2192Target\n(CollecTRI)", "Weighted graph\nA = |z|\u00b7|z\u2032|",
                     "Laplacian\nL = D \u2212 A", "Eigendecomp.\n\u03bb\u2081 \u2026 \u03bb\u2099",
                     "vN entropy\nH = \u2212\u03a3 \u03bb\u0303\u1d62 log\u2082 \u03bb\u0303\u1d62", "NetITH"),
           size = FONT_BASE/.pt, fontface = "bold", colour = c(COL_NETITH, COL_NETITH, COL_NETITH,
                                                         COL_NETITH, COL_NER, COL_NER)) +
  annotate("segment", x = c(2.5, 7.5, 7.5), xend = c(3.0, 5.5, 5.5),
           y = c(4.8, 4.8, 3.4), yend = c(4.8, 3.4, 3.4),
           arrow = arrow(length = unit(1.5, "mm"), type = "closed"),
           colour = "grey50", linewidth = 0.5) +
  labs(title = "A  NetITH computation") +
  theme_void(base_size = FONT_BASE) +
  theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0))

# ---- Panel B: GDSC / TCGA distributions --------------------------------------
g_vals <- gdsc$NetITH[!is.na(gdsc$NetITH)]
t_vals <- tcga$netith_bulk[!is.na(tcga$netith_bulk)]
pB <- ggplot() +
  geom_histogram(aes(g_vals, after_stat(density)), bins = 60, fill = COL_GDSC,
                 alpha = 0.45, colour = NA) +
  geom_histogram(aes(t_vals, after_stat(density)), bins = 60, fill = COL_TCGA,
                 alpha = 0.45, colour = NA) +
  annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.1, size = FONT_BASE/.pt,
           label = sprintf("GDSC (n=%s cell lines)\nmedian=%.2f\n\nTCGA (n=%s tumour + normal samples)\nmedian=%.2f",
                           format(length(g_vals), big.mark = ","), median(g_vals),
                           format(length(t_vals), big.mark = ","), median(t_vals)),
           fill = "white", label.size = NA) +
  labs(title = "B  NetITH distribution", x = "NetITH", y = "Density") +
  theme_pub()

# ---- Panel C: TCGA tumour vs normal -------------------------------------------
samp <- as.character(tcga$sample)
# Primary tumours = sample-type code "01", matched normals = "11"
# (TCGA barcode positions 14-15; other codes (02/03/05/06/07 metastases/other) excluded)
is_tum <- endsWith(samp, "01")
is_norm <- endsWith(samp, "11")
tv <- tcga$netith_bulk[is_tum]; nv <- tcga$netith_bulk[is_norm]
mw_p <- suppressWarnings(wilcox.test(tv, nv)$p.value)
pC <- ggplot() +
  geom_histogram(aes(tv, after_stat(density)), bins = 60, fill = COL_TCGA,
                 alpha = 0.5, colour = NA) +
  geom_histogram(aes(nv, after_stat(density)), bins = 60, fill = COL_GDSC,
                 alpha = 0.5, colour = NA) +
  annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.1, size = FONT_BASE/.pt,
           label = sprintf("tumour (n=%s)\nmedian=%.2f\n\nnormal (n=%s)\nmedian=%.2f",
                           format(length(tv), big.mark = ","), median(tv),
                           format(length(nv), big.mark = ","), median(nv)),
           fill = "white", label.size = NA) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.05, vjust = -0.6, size = FONT_BASE/.pt,
           colour = "dimgrey",
           label = if (mw_p < 0.001) sprintf("Mann-Whitney p=%.1e", mw_p) else sprintf("Mann-Whitney p=%.2f", mw_p)) +
  labs(title = "C  TCGA primary tumour vs. matched normal",
       x = "NetITH", y = "Density") +
  theme_pub()

# ---- Panel D: real vs null drug-association strength ---------------------------
pD <- ggplot(data.frame(x = c("CollecTRI\nnetwork", "50 rewired\nnetworks"),
                        y = c(null$real_median_rho, null$null_median_rho_mean),
                        sd = c(0, null$null_median_rho_sd)),
             aes(x, y, fill = x)) +
  geom_col(width = 0.55, alpha = 0.8) +
  geom_errorbar(aes(ymin = y - sd, ymax = y + sd), width = 0.15, linewidth = 0.3) +
  scale_fill_manual(values = c(COL_NER, "grey70"), guide = "none") +
  labs(title = "D  Drug-association strength vs. degree-preserving null",
       x = NULL, y = "median Spearman \u03c1 (286 drugs)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  annotate("text", x = 1, y = null$real_median_rho, vjust = -0.8, size = FONT_BASE/.pt,
           label = sprintf("%.3f", null$real_median_rho)) +
  annotate("text", x = 2, y = null$null_median_rho_mean + null$null_median_rho_sd,
           vjust = -0.8, size = FONT_BASE/.pt,
           label = sprintf("%.3f \u00b1 %.3f", null$null_median_rho_mean, null$null_median_rho_sd)) +
  theme_pub()

# ---- assemble: 2x2 grid, double-column 180 mm ----------------------------------
panels <- list(pA, pB, pC, pD)
composite <- (pA | pB) / (pC | pD)
HEIGHT_MM <- 111.3   # matches original aspect (7.0866 x 4.3808 in)
save_fig(composite, NAME, height_mm = HEIGHT_MM)
save_panels(panels, NAME)
cat(sprintf("[fig01] done; p(tumour vs normal) = %.3e\n", mw_p))
