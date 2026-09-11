# ============================================================================
# figS5_sc_cohort_forest.R — Supplementary Figure S5: cross-cohort single-cell outcome forest. Panels A-B: (A) Cohen's d (95% CI) per cohort, (B) OS hazard ratios (95% CI) for the two outcome-annotated cohorts.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-09
# Inputs  : results/scrnaseq_validation/*.json (single source for replication/OS values;
#           discovery values follow NUMBER_TRUTH_TABLE SC_D; GSE72056 CI computed in-script)
# Outputs : results/figures/r/FigS5_sc_cohort_forest.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS5_sc_cohort_forest.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(jsonlite) })

NAME <- "FigS5_sc_cohort_forest"
SCV <- file.path(RESULTS_DIR, "scrnaseq_validation")

read_j <- function(f) fromJSON(file.path(SCV, f))

# ---- cohort data (single source: replication JSONs + truth table) -----------
# class: discovery / replication / localisation / OS-replication
d_cohorts <- data.frame(
  cohort = c("Compendium (7 cancers, 113 patients)", "GSE115978 (31 patients)",
             "GSE120575 (immune, 48 points)", "GSE123139 (IT vs N, 19 patients)",
             "GSE72056 (tumour vs immune cells)"),
  d      = c(0.687, -0.503, 0.091, -0.196, 0.571),
  lo     = c(0.37, -1.367, -0.453, -1.219, NA),   # GSE72056 CI computed below
  hi     = c(0.87, 0.187, 0.684, 0.901, NA),
  p      = c(0.017, 0.198, 0.714, 0.778, 2.6e-47),
  n      = c("n=113", "n=29", "n=48", "n=19", "n=4,645 cells"),
  class  = c("discovery", "replication", "replication", "replication", "localisation")
)

# analytic 95% CI for GSE72056 cell-level d (n1=1,257 tumour / n2=3,388 immune)
n1 <- 1257; n2 <- 3388; d72056 <- 0.571
se72056 <- sqrt(1/n1 + 1/n2 + d72056^2 / (2 * (n1 + n2)))
d_cohorts$lo[d_cohorts$cohort == "GSE72056 (tumour vs immune cells)"] <- d72056 - 1.96 * se72056
d_cohorts$hi[d_cohorts$cohort == "GSE72056 (tumour vs immune cells)"] <- d72056 + 1.96 * se72056

# stopifnot against the JSON single source
j575 <- read_j("gse120575_replication.json"); j139 <- read_j("gse123139_replication.json")
j978 <- read_j("gse115978_replication.json"); j656 <- read_j("gse72056_replication.json")
stopifnot(abs(j575$cohen_d - 0.091) < 0.01, abs(j575$ci95[1] - (-0.453)) < 0.01)
stopifnot(abs(j139$contrast_IT_vs_N$cohen_d - (-0.196)) < 0.01)
stopifnot(abs(j978$contrast_sample_res_vs_untreated$cohen_d - (-0.503)) < 0.01)
stopifnot(abs(j656$tumour_vs_immune$cohen_d - 0.571) < 0.01)

hr_cohorts <- data.frame(
  cohort = c("GSE120575 OS (32 patients, 9 events)", "Xue2022 HCC OS (26 patients, 22 events)"),
  hr  = c(1.394, 1.083),
  lo  = c(0.635, 0.681),
  hi  = c(3.063, 1.721),
  p   = c(0.408, 0.737),
  class = c("OS-replication", "OS-replication")
)
j575os <- read_j("gse120575_os_replication.json")
jxue <- read_j("xue2022_hcc_replication.json")
stopifnot(abs(j575os$cox_patient$hr - 1.394) < 0.01)
stopifnot(abs(jxue$cox_patient$hr - 1.083) < 0.01)

CLASS_COL <- c(discovery = COL_NETITH, replication = COL_SIG,
               localisation = COL_NS, `OS-replication` = COL_TCGA)

fmt_p <- function(p) ifelse(p < 0.001, "p<0.001", sprintf("p=%.3f", p))
d_cohorts$plab <- fmt_p(d_cohorts$p)
hr_cohorts$plab <- fmt_p(hr_cohorts$p)
d_cohorts$lab <- sprintf("d=%.2f [%.2f, %.2f], %s", d_cohorts$d, d_cohorts$lo, d_cohorts$hi, d_cohorts$plab)
hr_cohorts$lab <- sprintf("HR=%.2f [%.2f, %.2f], %s", hr_cohorts$hr, hr_cohorts$lo, hr_cohorts$hi, hr_cohorts$plab)

# ---- Panel A: Cohen's d ------------------------------------------------------
# labels right-aligned at x=2.55 (panel limit 2.6): bars end <=0.87, labels
# extend leftwards to ~1.6 -> no overlap; 8pt minimum font kept.
d_cohorts$cohort <- factor(d_cohorts$cohort, levels = rev(d_cohorts$cohort))
pA <- ggplot(d_cohorts, aes(x = d, y = cohort, colour = class)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = COL_REF, linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.25, linewidth = 1.4) +
  geom_point(size = 2.4) +
  geom_text(aes(x = 2.55, label = lab), hjust = 1, size = 8/2.834, colour = "black",
            family = "Arial") +
  scale_colour_manual(values = CLASS_COL, name = "Cohort class") +
  scale_x_continuous(limits = c(-1.6, 2.6), breaks = seq(-1.5, 1, 0.5)) +
  labs(title = "A  Outcome contrast (Cohen's d)", x = "Cohen's d (95% CI)", y = NULL) +
  theme_pub() +
  theme(legend.position = "none", axis.text.y = element_text(size = 8))

# ---- Panel B: OS hazard ratios (log2 axis) ------------------------------------
hr_cohorts$cohort <- factor(hr_cohorts$cohort, levels = rev(hr_cohorts$cohort))
pB <- ggplot(hr_cohorts, aes(x = hr, y = cohort, colour = class)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = COL_REF, linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.25, linewidth = 1.4) +
  geom_point(size = 2.4) +
  geom_text(aes(x = 9.5, label = lab), hjust = 1, size = 8/2.834, colour = "black",
            family = "Arial") +
  scale_colour_manual(values = CLASS_COL, name = "Cohort class") +
  scale_x_log10(limits = c(0.5, 10), breaks = c(0.5, 1, 2, 4, 8)) +
  labs(title = "B  Overall survival (hazard ratio)", x = "HR per SD (95% CI, log scale)", y = NULL) +
  theme_pub() +
  theme(axis.text.y = element_text(size = 8))

# ---- assemble: stacked, single column ----------------------------------------
composite <- pA / pB
save_fig(composite, NAME, height_mm = 118, width_mm = 85)
save_panels(list(pA, pB), NAME)

cat("=== FigS5 key numbers ===\n")
print(d_cohorts[, c("cohort", "d", "lo", "hi", "p")])
print(hr_cohorts[, c("cohort", "hr", "lo", "hi", "p")])
cat(sprintf("[figS5] composite 85 mm wide x 118 mm tall\n"))
