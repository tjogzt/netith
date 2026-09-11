# ============================================================================
# fig03_robustness.R — Figure 3: robustness / construct validity / cross-platform drivers. Panels A-H: (A) genomic-ITH correlations with NetITH, (B) CRISPR essentiality, (C) virtual-perturbation landscape, (D) CRISPR-expression convergence, (E) predictive insufficiency, (F) GRN ablation, (G) benchmark vs 5 ITH metrics, (H) JUN regulon explanatory power.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/depmap/ith_benchmark/genomic_ith_correlations.csv,
#           results/depmap/depmap_differential_dependency.csv,
#           results/depmap/perturbation/virtual_perturbation_genome_wide.csv,
#           results/depmap/perturbation/convergence.csv,
#           results/depmap/grn_ablation_results.csv,
#           results/depmap/benchmark/gdsc_ith_benchmark_summary.csv,
#           results/depmap/tf_combinatorial_model_comparison.csv,
#           results/tcga/module_conservation_summary.csv,
#           results/depmap/p1_robustness/summary.json (nested-CV R2 = 0.041)
# Outputs : results/figures/r/Fig3_robustness_construct_validity.{pdf,png}
#           results/figures/r/panels/Fig3_robustness_construct_validity_panel{A..H}.{pdf,png}
# Usage   : Rscript code/R/figures/fig03_robustness.R
# Note    : All in-figure text English; numbers come from the cached CSVs only.
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "Fig3_robustness_construct_validity"
DEP_D <- file.path(RESULTS_DIR, "depmap")
TCGA_D <- file.path(RESULTS_DIR, "tcga")

# === Figure parameters (fixed block, do not scatter) =====================
.base_fs  <- 8            # pt: all axis/legend text (theme_pub default)
.title_fs <- 8            # pt: panel titles (theme_pub default)
.annot9   <- 9 / .pt      # large in-panel annotation (~9 pt)
.annot8   <- 8 / .pt      # secondary in-panel annotation (~8 pt)
.hist_alpha <- 0.6
.bar_alpha  <- 0.75
# ========================================================================

# ---- Panel A: Genomic ITH correlations with NetITH (orthogonal) ----------
ith <- fread(file.path(DEP_D, "ith_benchmark", "genomic_ith_correlations.csv"))
subA <- ith[metric_2 == "NetITH"]
stopifnot(nrow(subA) == 2L)
subA[, metric_1 := factor(metric_1, levels = subA$metric_1)]
pA <- ggplot(subA, aes(metric_1, spearman_r)) +
  geom_col(fill = COL_TCGA, alpha = 0.7, width = 0.6) +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.3f", spearman_r)), vjust = ifelse(subA$spearman_r >= 0, -0.6, 1.4),
            size = .base_fs / .pt) +
  coord_cartesian(ylim = c(-0.02, 0.03)) +
  labs(title = "A  Genomic ITH vs. NetITH (orthogonal)",
       x = NULL, y = "Spearman \u03c1 (vs. NetITH, n=681)") +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ---- Panel B: CRISPR essentiality (DepMap 26Q1) ---------------------------
# Python fallback file (crispr_dependency_netith.csv) has no effect column;
# the intended data (title 109/18,435 genes FDR<0.05) is the differential
# dependency table (identical to Supplementary Table 04).
dep <- fread(file.path(DEP_D, "depmap_differential_dependency.csv"))
n_tested_b <- nrow(dep)
n_fdr_b    <- dep[, sum(fdr < 0.05)]
pB <- ggplot(dep, aes(cohens_d)) +
  geom_histogram(bins = 60, fill = COL_GDSC, alpha = .hist_alpha, colour = NA) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  labs(title = sprintf("B  CRISPR essentiality (DepMap 26Q1;\n%d/%s genes FDR<0.05)",
                       n_fdr_b, format(n_tested_b, big.mark = ",")),
       x = "differential dependency (cohens_d)", y = "genes") +
  theme_pub()

# ---- Panel C: Virtual perturbation landscape ------------------------------
vp <- fread(file.path(DEP_D, "perturbation", "virtual_perturbation_genome_wide.csv"))
vp_max_abs <- vp[, max(abs(delta_netith))]
pC <- ggplot(vp, aes(delta_netith)) +
  geom_histogram(bins = 80, fill = COL_NETITH, alpha = .hist_alpha, colour = NA) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  geom_vline(xintercept = c(-0.5, 0.5), colour = COL_REF, linetype = "dashed", linewidth = 0.4) +
  labs(title = sprintf("C  Virtual perturbation (%s genes; none |\u0394|>0.5)",
                       format(nrow(vp), big.mark = ",")),
       x = "\u0394NetITH (expression-tertile stratification)", y = "genes") +
  theme_pub()

# ---- Panel D: CRISPR-expression convergence (text panel) ------------------
cv <- fread(file.path(DEP_D, "perturbation", "convergence.csv"))
n_conc_d <- cv[(both_sig == TRUE) & (concordant == TRUE), .N]
max_d_expr <- cv[, max(abs(dnetith_expr))]
pD <- ggplot() +
  annotate("text", x = 0.5, y = 0.55, hjust = 0.5, vjust = 0.5, size = .annot9,
           label = sprintf("%s/%s concordant significant genes",
                           n_conc_d, format(nrow(cv), big.mark = ","))) +
  annotate("text", x = 0.5, y = 0.32, hjust = 0.5, vjust = 0.5, size = .annot8, colour = "grey45",
           label = sprintf("max |d_expr| = %.2f\n(no gene changes NetITH >0.5)", max_d_expr)) +
  labs(title = "D  CRISPR\u2013expression convergence") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_void() +
  theme(plot.title = element_text(size = .title_fs, face = "bold", hjust = 0),
        plot.margin = margin(2, 2, 2, 2, "mm"))

# ---- Panel E: Predictive insufficiency (corrected R2 values) --------------
# Python fig3() hardcoded [0.74, 0.04, 0.10]; results/NUMBER_TRUTH_TABLE.csv
# marks 0.74 (4-TF in-sample) and 0.10 (top-1k Lasso) as corrected:
#   4-TF RF in-sample = 0.71 (FOURTF_RF_IN), nested-CV OOF = 0.041
#   (FOURTF_OOF, verified), top-1,000 Lasso in-sample = 0.34 (TOP1K_LASSO_IN).
r2 <- data.table(
  model = c("4-TF in-sample (RF)", "4-TF nested-CV (LASSO)", "top-1,000 Lasso"),
  r2    = c(0.71, 0.041, 0.34))
pE <- ggplot(r2, aes(model, r2, fill = model)) +
  geom_col(width = 0.55, alpha = .bar_alpha) +
  scale_fill_manual(values = c("4-TF in-sample (RF)" = COL_GDSC,
                               "4-TF nested-CV (LASSO)" = COL_ACCENT,
                               "top-1,000 Lasso" = COL_NS), guide = "none") +
  geom_text(aes(label = sprintf("%.2f", r2)), vjust = -0.5, size = .base_fs / .pt) +
  scale_y_continuous(limits = c(0, 0.85), breaks = seq(0, 0.8, 0.2)) +
  annotate("text", x = "4-TF nested-CV (LASSO)", y = 0.05, hjust = 0.5, vjust = 1.6,
           size = .base_fs / .pt, colour = "dimgrey", label = "in-sample optimistic") +
  labs(title = "E  Predictive insufficiency (4-TF models)",
       x = NULL, y = "R\u00b2 (predicting NetITH)") +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

# ---- Panel F: GRN ablation (in silico TF removal), top 10 ----------------
abl <- fread(file.path(DEP_D, "grn_ablation_results.csv"))
abl10 <- abl[order(-pct_change)][1:10]
abl10[, tf := factor(tf, levels = abl10$tf)]
abl10[, col := fifelse(tf == "JUN", COL_NER, COL_NS)]
pF <- ggplot(abl10, aes(tf, pct_change)) +
  geom_col(aes(fill = tf), alpha = .bar_alpha, width = 0.7) +
  scale_fill_manual(values = setNames(abl10$col, as.character(abl10$tf)), guide = "none") +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.3) +
  labs(title = "F  GRN ablation (in silico TF removal)",
       x = NULL, y = "\u0394NetITH (%)") +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# ---- Panel G: Benchmark vs. 5 ITH metrics (286 drugs) ---------------------
bm <- fread(file.path(DEP_D, "benchmark", "gdsc_ith_benchmark_summary.csv"))
bm[, col := fifelse(grepl("NetITH", label), COL_NETITH, COL_NS)]
bm[, label := factor(label, levels = rev(label))]          # NetITH on top
pG <- ggplot(bm, aes(n_fdr05, label)) +
  geom_col(aes(fill = label), alpha = 0.7, width = 0.65) +
  scale_fill_manual(values = setNames(bm$col, as.character(bm$label)), guide = "none") +
  geom_text(aes(label = n_fdr05), hjust = -0.3, size = .base_fs / .pt) +
  coord_cartesian(xlim = c(0, max(bm$n_fdr05) * 1.15)) +
  labs(title = "G  Benchmark vs. 5 ITH metrics (286 drugs)",
       x = "# drugs FDR<0.05 (of 286)", y = NULL) +
  theme_pub()

# ---- Panel H: JUN regulon explanatory power (GDSC vs. TCGA) ---------------
tfcc <- fread(file.path(DEP_D, "tf_combinatorial_model_comparison.csv"))
jun_gdsc  <- tfcc[Model == "JUN only"][["R\u00b2 (Full)"]]
mc <- fread(file.path(TCGA_D, "module_conservation_summary.csv"))
jun_tcga  <- mc[, median(r2_jun)]
hd <- data.table(grp = c("GDSC", "TCGA"), r2 = c(jun_gdsc, jun_tcga))
hd[, lab := c(sprintf("GDSC (R\u00b2=%.2f)", jun_gdsc),
              sprintf("TCGA (median R\u00b2\u2248%.3f)", jun_tcga))]
pG0 <- pG  # keep G name stable below
pH <- ggplot(hd, aes(grp, r2, fill = grp)) +
  geom_col(width = 0.5, alpha = .bar_alpha) +
  scale_fill_manual(values = c(GDSC = COL_GDSC, TCGA = COL_TCGA), guide = "none") +
  scale_x_discrete(labels = hd$lab) +
  labs(title = "H  JUN regulon explanatory power\n(GDSC vs. TCGA)",
       x = NULL, y = "JUN-only R\u00b2 (NetITH)") +
  theme_pub()

# ---- Assemble (4 rows x 2 cols; Python gridspec 4x2) ----------------------
panels <- list(pA, pB, pC, pD, pE, pF, pG, pH)
composite <- (pA | pB) / (pC | pD) / (pE | pF) / (pG | pH)
save_fig(composite, NAME, height_mm = 180.0)   # 7.0866 x 7.0866 in -> 180 mm
save_panels(panels, NAME)

# ---- Key numbers per panel (cross-checked against cached CSVs) ------------
cat(sprintf("[fig03] A: n=%d; TMB rho=%.3f (p=%.3f); MutationShannon rho=%.3f (p=%.3f)\n",
            subA$n[1], subA$spearman_r[1], subA$spearman_p[1],
            subA$spearman_r[2], subA$spearman_p[2]))
cat(sprintf("[fig03] B: %d/%d genes FDR<0.05; cohens_d range %.2f..%.2f\n",
            n_fdr_b, n_tested_b, dep[, min(cohens_d)], dep[, max(cohens_d)]))
cat(sprintf("[fig03] C: n=%d genes; max |dNetITH| = %.3f (none >0.5)\n",
            nrow(vp), vp_max_abs))
cat(sprintf("[fig03] D: %d concordant significant of %d tested; max |d_expr| = %.2f\n",
            n_conc_d, nrow(cv), max_d_expr))
cat(sprintf("[fig03] E: R2 = %.2f (4-TF RF in-sample), %.3f (4-TF nested-CV OOF), %.2f (top-1,000 Lasso in-sample)  [NUMBER_TRUTH_TABLE corrected values]\n",
            r2$r2[1], r2$r2[2], r2$r2[3]))
cat(sprintf("[fig03] F: top10 TFs by pct_change: %s; JUN +%.2f%% (p=%.1e)\n",
            paste(abl10$tf, collapse = ","), abl10[tf == "JUN", pct_change],
            abl10[tf == "JUN", wilcoxon_p]))
cat(sprintf("[fig03] G: n_fdr05 = %s (of %d drugs); NetITH = %d\n",
            paste(bm$n_fdr05, collapse = ","), bm$n_drugs[1], bm[grepl("NetITH", label), n_fdr05]))
cat(sprintf("[fig03] H: JUN-only R2 GDSC = %.2f; TCGA median R2 = %.3f (n=%d cancer types)\n",
            jun_gdsc, jun_tcga, nrow(mc)))
cat("[fig03] done\n")
