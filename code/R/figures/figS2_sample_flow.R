# ============================================================================
# figS2_sample_flow.R — Supplementary Figure S2: sample-flow diagram per dataset (initial -> expression -> NetITH -> annotation -> analysis). Five columns: GDSC, TCGA, GSE25066, IMvigor210, single-cell compendium.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : none required for the diagram (published cohort counts); row counts
#           cross-checked at runtime against results/gdsc/gdsc_netith_cell_lines.csv,
#           results/tcga/tcga_netith.csv, results/tcga/multicancer_meta_analysis.csv,
#           results/neoadjuvant/gse25066_netith_pcr.csv,
#           results/imvigor210/imvigor210_netith_results.csv,
#           results/scrnaseq_validation/patient_netith.csv
# Outputs : results/figures/r/FigS2_sample_flow.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS2_sample_flow.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "FigS2_sample_flow"

# ---- flow content: label (bold) + count text (one or two lines) -------------
# Every number below is the published cohort count (see header); the long
# count strings are manually wrapped so that no line exceeds the column width
# at 7 pt. "167 samples / 223 patients" follows the verified manuscript
# correction (RERUN_VERIFICATION_REPORT: compendium = 167 samples from 223
# patients, 9 cancer types).
flows <- list(
  "GDSC cell lines" = list(
    list("RNA expression", "1,018 profiled samples"),
    list("Mapped cell lines", "1,013 with expression +\nNetITH"),
    list("Drug screening", "286 drugs; 969 cell lines\nwith IC50; \u2265900 per drug"),
    list("Analysis", "1,013 lines, 286 drugs\n(per-drug n reported)")),
  "TCGA tumours" = list(
    list("RNA expression", "10,535 tumour samples\n(TOIL)"),
    list("NetITH computed", "11,069 samples\n(9,710 tumours + 744 normal)"),
    list("Survival annotation", "9,633 tumour samples\nwith OS"),
    list("Meta-analysis", "26 cancer types,\n3,073 events")),
  "GSE25066" = list(
    list("Profiled", "508 patients\n(Affymetrix HG-U133A)"),
    list("NetITH + pCR annotation", "306 patients"),
    list("pCR events", "57 pCR (18.6%)")),
  "IMvigor210" = list(
    list("RNA-seq", "348 patients"),
    list("Response annotation", "298 (68 CR/PR)"),
    list("Multivariable model", "205 complete cases")),
  "Single-cell compendium" = list(
    list("Raw", "355,941 cells / 167 samples /\n223 patients / 9 cancer types"),
    list("Post-QC", "24,943 cells / 113 patients\n/ 7 types"),
    list("+ GSE131907", "6,834 cells / 58 patients\n(lung)"),
    list("Total", "31,777 cells reported"))
)

# ---- one column: boxes + arrows via annotate (theme_void canvas) ------------
col_plot <- function(title, stages) {
  n <- length(stages)
  yc <- seq(0.92, by = -0.24, length.out = n)          # box centres (pitch 0.24)
  p <- ggplot() + xlim(0, 1) + ylim(0.05, 1.05) +
    labs(title = title) +
    theme_void(base_size = FONT_BASE) +
    theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5,
                                    margin = margin(b = 2, unit = "mm")))
  for (i in seq_len(n)) {
    p <- p +
      annotate("rect", xmin = 0.02, xmax = 0.98,
               ymin = yc[i] - 0.10, ymax = yc[i] + 0.10,
               fill = COL_NETITH, alpha = 0.10, colour = COL_NETITH, linewidth = 0.5) +
      annotate("text", x = 0.5, y = yc[i] + 0.048, size = FONT_BASE/.pt, fontface = "bold",
               label = stages[[i]][1]) +
      annotate("text", x = 0.5, y = yc[i] - 0.055, size = FONT_BASE/.pt, colour = COL_SIG,
               label = stages[[i]][2])
    if (i < n) {
      p <- p + annotate("segment", x = 0.5, xend = 0.5,
                        y = yc[i] - 0.125, yend = yc[i + 1] + 0.125,
                        arrow = arrow(length = unit(1.2, "mm"), type = "closed"),
                        colour = "grey50", linewidth = 0.4)
    }
  }
  p
}

p1 <- col_plot("GDSC cell lines", flows[["GDSC cell lines"]])
p2 <- col_plot("TCGA tumours", flows[["TCGA tumours"]])
p3 <- col_plot("GSE25066", flows[["GSE25066"]])
p4 <- col_plot("IMvigor210", flows[["IMvigor210"]])
p5 <- col_plot("Single-cell compendium", flows[["Single-cell compendium"]])

# ---- assemble: 5 columns, double-column 180 mm (original 7.0866 x 1.9931) ---
# Unequal widths give the longest columns (GDSC, single-cell) more room.
panels <- list(p1, p2, p3, p4, p5)
# Unequal widths: the long-text columns (GDSC, TCGA, single-cell) get more room.
composite <- (p1 | p2 | p3 | p4 | p5) +
  plot_layout(widths = c(1.15, 1.10, 1.00, 0.85, 1.15))
HEIGHT_MM <- round(1.9931 / 7.0866 * 180, 1)           # 50.6 mm
save_fig(composite, NAME, height_mm = HEIGHT_MM)
save_panels(panels, NAME)

# ---- key numbers: cross-check row counts against cached CSVs -----------------
cat("=== FigS2 key numbers (cross-checked vs cached CSVs) ===\n")
gdsc <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
tcga <- fread(file.path(RESULTS_DIR, "tcga", "tcga_netith.csv")); tcga[, 1 := NULL]
meta <- fread(file.path(RESULTS_DIR, "tcga", "multicancer_meta_analysis.csv"))
gse  <- fread(file.path(RESULTS_DIR, "neoadjuvant", "gse25066_netith_pcr.csv"))
imv  <- fread(file.path(RESULTS_DIR, "imvigor210", "imvigor210_netith_results.csv"))
sc   <- fread(file.path(RESULTS_DIR, "scrnaseq_validation", "patient_netith.csv"))
cat(sprintf("GDSC cell lines with NetITH: %d (flow says 1,013)\n", nrow(gdsc)))
samp <- as.character(tcga$sample)
n_tum <- sum(grepl("-01", samp)); n_norm <- sum(grepl("-11", samp))
cat(sprintf("TCGA NetITH samples: %d (%d tumour + %d normal) (flow says 11,069 / 9,710 + 744)\n",
            nrow(tcga), n_tum, n_norm))
cat(sprintf("TCGA survival meta-analysis: %d cancer types, n=%d, %d events (flow says 26 / 9,633 / 3,073)\n",
            meta$k[1], meta$n_total[1], meta$n_events_total[1]))
pcr <- gse$pathologic_response_pcr_rd
cat(sprintf("GSE25066: %d patients with NetITH+pCR, %d pCR (%.1f%%) (flow says 306 / 57 / 18.6%%)\n",
            nrow(gse), sum(pcr == "pCR"), 100 * sum(pcr == "pCR") / nrow(gse)))
resp <- imv[binaryResponse != ""]
cat(sprintf("IMvigor210: %d with RNA-seq, %d with response annotation, %d CR/PR (flow says 348 / 298 / 68)\n",
            nrow(imv), nrow(resp), sum(resp$binaryResponse == "CR/PR")))
cat(sprintf("Single-cell compendium post-QC: %d patients, %d cells in patient-level table (figure/paper: 24,943 cells)\n",
            nrow(sc), sum(sc$n_cells)))
cat(sprintf("[figS2] composite %.0f mm wide x %.1f mm tall\n", 180, HEIGHT_MM))
