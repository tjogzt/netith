# ============================================================================
# figS3_tf_subset.R — Supplementary Figure S3: random 50% TF-subset robustness (GDSC). Panels A-C: (A) association strength (per-subset median rho), (B) directionality (drugs with rho>0), (C) construct consistency (subset vs full-set rho).
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/control/tf_subset_50pct_per_subset.csv,
#           results/control/tf_subset_50pct_summary.json
# Outputs : results/figures/r/FigS3_tf_subset_50pct.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS3_tf_subset.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork); library(jsonlite) })

NAME <- "FigS3_tf_subset_50pct"

df <- fread(file.path(RESULTS_DIR, "control", "tf_subset_50pct_per_subset.csv"))
summ <- fromJSON(file.path(RESULTS_DIR, "control", "tf_subset_50pct_summary.json"))

stopifnot(nrow(df) == summ$n_subsets)          # 50 subsets
stopifnot(summ$n_drugs == 286L)                # 286 drugs

# ---- Panel A: association strength (per-subset median rho) -------------------
pA <- ggplot(df, aes(x = "", y = median_rho)) +
  geom_boxplot(fill = COL_NETITH, alpha = 0.35, width = 0.5, outlier.size = 0.7) +
  geom_hline(yintercept = summ$full_median_rho, linetype = "dashed",
             colour = COL_GDSC, linewidth = 0.5) +
  scale_y_continuous(limits = c(0, 0.35)) +
  labs(title = "A  Association strength",
       x = "50 random 50% TF subsets",
       y = "median Spearman \u03c1 (286 drugs)") +
  theme_pub()

# ---- Panel B: directionality (count of drugs with rho > 0) -------------------
pB <- ggplot(df, aes(n_pos)) +
  geom_histogram(bins = 12, fill = COL_NETITH, alpha = 0.4,
                 colour = "black", linewidth = 0.2) +
  geom_vline(xintercept = summ$full_n_pos, linetype = "dashed",
             colour = COL_GDSC, linewidth = 0.5) +
  labs(title = "B  Directionality",
       x = "drugs with \u03c1>0 (of 286)",
       y = "subsets") +
  theme_pub()

# ---- Panel C: construct consistency (subset vs full-set NetITH) --------------
pC <- ggplot(df, aes(rho_with_full)) +
  geom_histogram(bins = 12, fill = COL_NETITH, alpha = 0.35,
                 colour = "black", linewidth = 0.2) +
  labs(title = "C  Construct consistency",
       x = "Spearman \u03c1 (subset vs full-set NetITH)",
       y = "subsets") +
  theme_pub()

# ---- assemble: 1x3, double-column 180 mm (original 7.0866 x 2.0247) ---------
panels <- list(pA, pB, pC)
composite <- pA / pB / pC
HEIGHT_MM <- round(3 * (2.0247 / 7.0866 * 180), 1)     # 154.2 mm (vertical A/B/C)
save_fig(composite, NAME, height_mm = HEIGHT_MM, width_mm = 85)
save_panels(panels, NAME)

# ---- key numbers (must match cached CSV / JSON / manuscript) -----------------
n_ge <- sum(df$n_pos >= 283L)
cat("=== FigS3 key numbers ===\n")
cat(sprintf("Subsets: %d (of %d drugs); full-set median rho = %.4f (dashed line in A)\n",
            nrow(df), summ$n_drugs, summ$full_median_rho))
cat(sprintf("Per-subset median rho: median %.4f (range %.4f-%.4f)\n",
            median(df$median_rho), min(df$median_rho), max(df$median_rho)))
cat(sprintf("Drugs with rho>0: median %d / 286 (range %d-%d); %d/50 subsets >= 283\n",
            as.integer(median(df$n_pos)), min(df$n_pos), max(df$n_pos), n_ge))
cat(sprintf("Subset vs full-set NetITH rho: median %.4f (range %.4f-%.4f)\n",
            median(df$rho_with_full), min(df$rho_with_full), max(df$rho_with_full)))
cat(sprintf("[figS3] composite %.0f mm wide x %.1f mm tall\n", 180, HEIGHT_MM))
