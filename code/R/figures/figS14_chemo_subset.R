# ============================================================================
# figS14_chemo_subset.R — Supplementary Figure 14
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/chemo_subset/chemo_results.json,
#           results/chemo_subset/chemo_per_cancer_results.csv,
#           results/chemo_subset/tcga_treatment_classified.csv
# Outputs : results/figures/r/FigS14_treatment_stratified.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS14_chemo_subset.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork); library(jsonlite) })

NAME <- "FigS14_treatment_stratified"
H_MM <- 5.3150 / 7.0866 * 180   # 135.0 mm (Python figsize 7.0866 x 5.3150 in)

CHEMO_D <- file.path(RESULTS_DIR, "chemo_subset")
pan <- fromJSON(file.path(CHEMO_D, "chemo_results.json"), simplifyVector = FALSE)
pc <- fread(file.path(CHEMO_D, "chemo_per_cancer_results.csv"))
tc <- fread(file.path(CHEMO_D, "tcga_treatment_classified.csv"))

# darken helper (keeps NPG hues, alters value only)
shade <- function(col, f) {
  m <- col2rgb(col)[, 1]
  rgb(m[1] * f, m[2] * f, m[3] * f, maxColorValue = 255)
}

sig_star <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "ns")))

# ---- Panel A: forest plot (All / Treated / Untreated) --------------------------
fa <- data.table(
  label = factor(c("All", "Treated", "Untreated"), levels = c("Untreated", "Treated", "All")),
  hr = c(pan$All$hr, pan$Treated$hr, pan$Untreated$hr),
  lo = c(pan$All$ci_lower, pan$Treated$ci_lower, pan$Untreated$ci_lower),
  hi = c(pan$All$ci_upper, pan$Treated$ci_upper, pan$Untreated$ci_upper),
  p  = c(pan$All$p, pan$Treated$p, pan$Untreated$p),
  col = c(COL_SIG, COL_GDSC, COL_NETITH))
fa[, star := sig_star(p)]

pA <- ggplot(fa, aes(hr, label)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_errorbar(aes(xmin = lo, xmax = hi, y = label), width = 0.2, linewidth = 0.5,
                orientation = "y") +
  geom_point(aes(colour = label), size = 2.2) +
  scale_colour_manual(values = setNames(fa$col, fa$label), guide = "none") +
  annotate("text", x = fa$hi + 0.02, y = fa$label,
           label = sprintf("HR=%.3f %s", fa$hr, fa$star), size = FONT_BASE/.pt, hjust = 0) +
  coord_cartesian(xlim = c(0.55, 1.70)) +
  labs(title = "(A) NetITH HR by\nTreatment Status", x = "Hazard Ratio (95% CI)", y = NULL) +
  theme_pub()

# ---- Panel B: 3-year survival by NetITH median split x treatment ---------------
tc2 <- tc[!is.na(netith_bulk) & !is.na(OS.time) & !is.na(OS.event) & OS.time > 0]
med <- median(tc2$netith_bulk)
tc2[, nt := ifelse(netith_bulk > med, "High", "Low")]
tc2[, trt := ifelse(treated == 1, "Trt", "Unt")]
g <- tc2[, .(surv3 = mean(OS.time >= 1095), n = .N), by = .(trt, nt)][n > 50]
g[, lbl := paste0(trt, "-", nt)]
g[, col := ifelse(trt == "Trt", shade(COL_GDSC, ifelse(nt == "Low", 1, 0.8)),
                                  shade(COL_NETITH, ifelse(nt == "Low", 1, 0.8)))]
g[, lbl := factor(lbl, levels = c("Trt-Low", "Trt-High", "Unt-Low", "Unt-High"))]
cat(sprintf("  [B] 3-year survival groups: %s\n",
            paste(sprintf("%s=%.3f (n=%d)", g$lbl, g$surv3, g$n), collapse = ", ")))

pB <- ggplot(g, aes(lbl, surv3)) +
  geom_col(aes(fill = lbl), width = 0.6) +
  scale_fill_manual(values = setNames(g$col, g$lbl), guide = "none") +
  geom_text(aes(label = sprintf("%.2f", surv3)), vjust = -0.4, size = FONT_BASE/.pt) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(title = "(B) 3-Year Survival by\nNetITH x Treatment",
       x = NULL, y = "3-Year Survival Rate") +
  theme_pub()

# ---- Panel C: per-cancer treated vs untreated HR --------------------------------
pv <- dcast(pc, cancer ~ group, value.var = "HR")
pv <- pv[!is.na(Treated) & !is.na(Untreated)]
pv[, col := ifelse(Treated < Untreated, COL_GDSC, COL_NETITH)]
cat(sprintf("  [C] per-cancer pairs analysed: %d\n", nrow(pv)))

pC <- ggplot(pv, aes(Treated, Untreated)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_hline(yintercept = 1, linetype = "dotted", colour = "grey60", linewidth = 0.3) +
  geom_vline(xintercept = 1, linetype = "dotted", colour = "grey60", linewidth = 0.3) +
  geom_point(aes(colour = col), size = 1.8, alpha = 0.7) +
  scale_colour_identity() +
  geom_text(aes(label = cancer), size = FONT_BASE/.pt, alpha = 0.7, vjust = -0.8) +
  coord_cartesian(xlim = c(0.2, 2.0), ylim = c(0.2, 2.0)) +
  labs(title = "(C) Per-Cancer: Treated vs\nUntreated HR",
       x = "HR Treated", y = "HR Untreated") +
  theme_pub()

# ---- Panel D: NetITH distribution by treatment ----------------------------------
pD <- ggplot(tc2, aes(netith_bulk, fill = treated_label)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.5, position = "identity",
                 linewidth = 0.2) +
  scale_fill_manual(values = c("Treated" = COL_GDSC, "Untreated/Unknown" = COL_NETITH)) +
  labs(title = "(D) NetITH Distribution\nby Treatment", x = "NetITH", y = "Density") +
  theme_pub() + theme(legend.position = "top", legend.title = element_blank())

# ---- Panel E: NetITH HR in treated patients (by cancer) --------------------------
trt <- pc[group == "Treated"][order(-n)][1:min(20, .N)]
trt[, y := .N:1]
trt[, col := ifelse(p < 0.05, COL_GDSC, COL_GREY)]
cat(sprintf("  [E] treated cancers plotted: %d\n", nrow(trt)))

pE <- ggplot(trt, aes(log2(pmin(pmax(HR, 0.125), 8)), y)) +
  geom_col(aes(fill = col), width = 0.7) +
  scale_fill_identity() +
  geom_vline(xintercept = 0, linewidth = 0.5) +
  scale_y_continuous(breaks = trt$y, labels = sprintf("%s (n=%d)", trt$cancer, trt$n)) +
  labs(title = "(E) NetITH HR in\nTreated Patients", x = "log2(HR)", y = NULL) +
  theme_pub()

# ---- Panel F: summary text --------------------------------------------------------
int_p <- as.numeric(pan$interaction_p)
lines <- c(
  "Chemotherapy-Stratified", "NetITH Survival Analysis", "",
  sprintf("  All:     HR = %.3f", pan$All$hr),
  sprintf("  Treated: HR = %.3f", pan$Treated$hr),
  sprintf("  Untreated: HR = %.3f", pan$Untreated$hr),
  sprintf("  Interaction p = %.2f", int_p), "",
  "  Interpretation:")
if (is.finite(int_p) && int_p < 0.05) {
  lines <- c(lines, "  Significant treatment interaction,",
             "  effect direction differs by group.")
} else {
  lines <- c(lines, "  No significant treatment",
             "  interaction. NetITH survival",
             "  effect is similar regardless of",
             "  chemotherapy status. This supports",
             "  the immune-mediated mechanism",
             "  over chemosensitivity.")
}
pF <- ggplot() + xlim(0, 1) + ylim(0, 1) +
  annotate("text", x = 0.04, y = 0.96, hjust = 0, vjust = 1, size = FONT_BASE/.pt,
           label = paste(lines, collapse = "\n")) +
  theme_void()

# ---- composite + save --------------------------------------------------------------
p <- (pA | pB | pC) / (pD | pE | pF) +
  plot_annotation(title = "TCGA Chemotherapy-Stratified NetITH Survival Analysis",
                  theme = theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5)))
save_fig(p, NAME, height_mm = H_MM)
save_panels(list(pA, pB, pC, pD, pE, pF), NAME)

# ---- print key numbers ---------------------------------------------------------------
cat("\n=== EDFig8 key numbers ===\n")
cat(sprintf("  All:      HR=%.3f [%.3f-%.3f], p=%.3f, n=%d\n",
            pan$All$hr, pan$All$ci_lower, pan$All$ci_upper, pan$All$p, pan$All$n))
cat(sprintf("  Treated:  HR=%.3f [%.3f-%.3f], p=%.3f, n=%d\n",
            pan$Treated$hr, pan$Treated$ci_lower, pan$Treated$ci_upper, pan$Treated$p, pan$Treated$n))
cat(sprintf("  Untreated:HR=%.3f [%.3f-%.3f], p=%.3f, n=%d\n",
            pan$Untreated$hr, pan$Untreated$ci_lower, pan$Untreated$ci_upper, pan$Untreated$p, pan$Untreated$n))
cat(sprintf("  Interaction p = %.4f\n", int_p))
