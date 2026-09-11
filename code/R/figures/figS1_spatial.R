# ============================================================================
# figS1_spatial.R — Supplementary Figure S1: spatial transcriptomics. Panels A-C: (A) BRCA Visium radial gradient, (B) Xenium tumor vs non-tumor NetITH, (C) ovarian Visium negative control.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/depmap/spatial_visium_radial_profile.csv,
#           results/depmap/spatial_visium_netith.csv,
#           results/depmap/spatial_xenium_netith.csv,
#           results/depmap/spatial_visium_OV_netith.csv
# Outputs : results/figures/r/FigS1_spatial_transcriptomics.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS1_spatial.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "FigS1_spatial_transcriptomics"

# ---- unicode superscript helper: 8.6e-19 -> "8.6x10^-19" -------------------
sup_digits <- function(s) {
  sup <- c("0" = "\u2070", "1" = "\u00b9", "2" = "\u00b2", "3" = "\u00b3",
           "4" = "\u2074", "5" = "\u2075", "6" = "\u2076", "7" = "\u2077",
           "8" = "\u2078", "9" = "\u2079", "-" = "\u207b", "+" = "\u207a")
  chars <- strsplit(s, "")[[1]]
  paste0(vapply(chars, function(ch) if (ch %in% names(sup)) sup[[ch]] else ch, character(1)),
         collapse = "")
}
fmt_exp <- function(p, digits = 1) {
  parts <- strsplit(formatC(p, format = "e", digits = digits), "e")[[1]]
  paste0(parts[1], "\u00d710", sup_digits(parts[2]))
}

# ---- data ------------------------------------------------------------------
rad <- fread(file.path(RESULTS_DIR, "depmap", "spatial_visium_radial_profile.csv"))
vis <- fread(file.path(RESULTS_DIR, "depmap", "spatial_visium_netith.csv"))
xn  <- fread(file.path(RESULTS_DIR, "depmap", "spatial_xenium_netith.csv"))
ov  <- fread(file.path(RESULTS_DIR, "depmap", "spatial_visium_OV_netith.csv"))

# ---- Panel A: BRCA Visium radial gradient -----------------------------------
stA <- spearman_test(vis$dist_norm, vis$NetITH)          # rho=+0.152, p=8.6e-19
n_spots <- format(sum(rad$n_spots), big.mark = ",")
pA <- ggplot(rad, aes(dist_mid, netith_mean)) +
  geom_line(colour = COL_NETITH, linewidth = 0.6) +
  geom_point(colour = COL_NETITH, size = 1.4) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.05, vjust = -0.35, size = FONT_BASE/.pt,
           colour = COL_SIG,
           label = sprintf("\u03c1=%+.3f\np=%s", stA["rho"], fmt_exp(stA["p"], 1))) +
  labs(title = sprintf("A  BRCA Visium (1 section,\nn = %s spots)", n_spots),
       x = "distance from centroid (normalized)",
       y = "mean NetITH (spot-level)") +
  theme_pub()

# ---- Panel B: Xenium tumor vs non-tumor -------------------------------------
xn <- xn[!is.na(cell_type) & !is.na(NetITH)]
tumor_mask <- grepl("tumor|epithelial|malignant", xn$cell_type, ignore.case = TRUE)
tu <- xn$NetITH[tumor_mask]
nt <- xn$NetITH[!tumor_mask]
stopifnot(length(tu) == 15000L, length(nt) == 15000L)
mw_p <- suppressWarnings(wilcox.test(tu, nt)$p.value)     # p=2.07e-95
bar_df <- data.frame(
  group  = factor(c("tumor", "non-tumor"), levels = c("tumor", "non-tumor")),
  median = c(median(tu), median(nt)),
  se3    = c(3 * sd(tu) / sqrt(length(tu)), 3 * sd(nt) / sqrt(length(nt))))
ymin <- min(bar_df$median - bar_df$se3) - 0.01
ymax <- max(bar_df$median + bar_df$se3) + 0.035
pB <- ggplot(bar_df, aes(group, median, fill = group)) +
  geom_col(width = 0.5, alpha = 0.75) +
  geom_errorbar(aes(ymin = median - se3, ymax = median + se3),
                width = 0.12, linewidth = 0.4) +
  scale_fill_manual(values = c(COL_GDSC, COL_NS), guide = "none") +
  scale_x_discrete(labels = c(sprintf("tumor\n(n=%s)", format(length(tu), big.mark = ",")),
                              sprintf("non-tumor\n(n=%s)", format(length(nt), big.mark = ",")))) +
  coord_cartesian(ylim = c(ymin, ymax)) +
  annotate("text", x = 1.5, y = ymax - 0.004, size = FONT_BASE/.pt, colour = COL_SIG,
           label = sprintf("tumor > non-tumor\np=%s (Mann-Whitney)", fmt_exp(mw_p, 0))) +
  labs(title = "B  Xenium (1 IDC section;\n30,000-cell sample: 15k+15k)",
       x = NULL, y = "NetITH (per-cell)") +
  theme_pub()

# ---- Panel C: Ovarian Visium negative control --------------------------------
stC <- spearman_test(ov$dist_norm, ov$NetITH)            # rho=-0.017, p=0.28
pC <- ggplot(ov, aes(dist_norm, NetITH)) +
  geom_point(colour = COL_NETITH, alpha = 0.25, size = 0.5) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.05, vjust = -0.35, size = FONT_BASE/.pt,
           colour = "grey45",
           label = sprintf("\u03c1=%.3f\np=%.2f (no gradient)", stC["rho"], stC["p"])) +
  labs(title = "C  Ovarian Visium\n(1 section; negative control)",
       x = "distance from centroid (normalized)",
       y = "NetITH (spot-level)") +
  theme_pub()

# ---- assemble: 1x3, double-column 180 mm (original figsize 7.0866 x 2.4297) --
panels <- list(pA, pB, pC)
composite <- pA | pB | pC
HEIGHT_MM <- round(2.4297 / 7.0866 * 180, 1)             # 61.7 mm
save_fig(composite, NAME, height_mm = HEIGHT_MM)
save_panels(panels, NAME)

# ---- key numbers (must match cached data / manuscript) ----------------------
cat("=== FigS1 key numbers ===\n")
cat(sprintf("A BRCA Visium: %s spots; per-spot Spearman rho=%.3f, p=%.1e\n",
            n_spots, stA["rho"], stA["p"]))
cat(sprintf("B Xenium: tumor n=%d median=%.4f | non-tumor n=%d median=%.4f | Mann-Whitney p=%.2e\n",
            length(tu), median(tu), length(nt), median(nt), mw_p))
cat(sprintf("C OV Visium: n=%d spots; Spearman rho=%.3f, p=%.3f (no gradient)\n",
            nrow(ov), stC["rho"], stC["p"]))
cat(sprintf("[figS1] composite %.0f mm wide x %.1f mm tall\n", 180, HEIGHT_MM))
