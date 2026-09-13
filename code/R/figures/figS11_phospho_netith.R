# ============================================================================
# figS11_phospho_netith.R — Supplementary Figure 11
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/depmap/ext_d6_phospho_netith.csv,
#           results/depmap/ext_d6_drug_comparison.csv,
#           results/depmap/ext_d6_kinase_pathway_netith.csv
# Outputs : results/figures/r/FigS11_phospho_netith.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS11_phospho_netith.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "FigS11_phospho_netith"
DEP <- file.path(RESULTS_DIR, "depmap")
COL_PHOSPHO <- NPG_COLORS[8]        # "#FF7F00" - phospho orange (manuscript colour)

cat("=== EDFig5: Phospho-NetITH ===\n")

# ---- Panel A: phospho vs expression NetITH scatter ---------------------------
ph <- fread(file.path(DEP, "ext_d6_phospho_netith.csv"))
stopifnot(all(c("cell_line", "phospho_netith", "original_netith") %in% names(ph)))
ph <- ph[is.finite(phospho_netith) & is.finite(original_netith)]
rho5 <- cor(ph$original_netith, ph$phospho_netith, method = "spearman")
n_ph <- nrow(ph)

pA <- ggplot(ph, aes(original_netith, phospho_netith)) +
  geom_point(colour = COL_NETITH, size = 0.5, alpha = 0.3, shape = 16) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.1, vjust = -0.8, size = FONT_BASE/.pt,
           fontface = "bold",
           label = sprintf("\u03c1=%.2f\nn=%d", rho5, n_ph)) +
  labs(title = "A  Phospho-NetITH vs. expression NetITH",
       x = "Expression NetITH", y = "Phospho-NetITH") +
  theme_pub()

# ---- Panel B: top-12 drugs by |rho| improvement with Phospho-NetITH ----------
dcomp <- fread(file.path(DEP, "ext_d6_drug_comparison.csv"))
stopifnot(all(c("drug", "rho_original", "rho_phospho") %in% names(dcomp)))
dcomp[, improvement := (abs(rho_phospho) - abs(rho_original)) /
        pmax(abs(rho_original), 0.001) * 100]
top12 <- dcomp[order(-improvement)][1:12]
top12[, drug := factor(drug, levels = rev(drug))]
top12[, ymax := pmax(abs(rho_phospho), abs(rho_original))]

pB <- ggplot(top12) +
  geom_col(aes(abs(rho_original), drug, fill = "Expression NetITH |\u03c1|"),
           alpha = 0.4, width = 0.5) +
  geom_col(aes(abs(rho_phospho), drug, fill = "Phospho-NetITH |\u03c1|"),
           alpha = 0.6, width = 0.5) +
  geom_text(aes(ymax + 0.005, drug, label = sprintf("%+.0f%%", improvement)),
            size = FONT_BASE/.pt, fontface = "bold", colour = COL_NER, hjust = 0) +
  scale_fill_manual(values = c("Phospho-NetITH |\u03c1|" = COL_PHOSPHO,
                               "Expression NetITH |\u03c1|" = COL_NS),
                    breaks = c("Phospho-NetITH |\u03c1|", "Expression NetITH |\u03c1|"),
                    name = NULL) +
  labs(title = "B  Top drugs: |\u03c1| improvement with Phospho-NetITH",
       x = "|Spearman \u03c1| with IC\u2085\u2080", y = NULL) +
  theme_pub() +
  theme(legend.position = "bottom")

# ---- Panel C: pathway-level kinase counts (phosphosite decomposition) --------
kp <- fread(file.path(DEP, "ext_d6_kinase_pathway_netith.csv"))
stopifnot(all(c("n_kinases", "mean_phospho_netith", "rho_with_original") %in% names(kp)))
kp[, pathway := V1]                       # unnamed first column = pathway name
kp[, pathway := factor(pathway, levels = rev(V1))]   # file order: top = last (barh-like)

pC <- ggplot(kp, aes(n_kinases, pathway)) +
  geom_col(fill = COL_PHOSPHO, alpha = 0.6, width = 0.6) +
  labs(title = "C  Pathway-level phosphosite variance decomposition",
       x = "Kinases per pathway (n)", y = NULL) +
  theme_pub()

p <- pA / pB / pC
save_fig(p, NAME, height_mm = 195.9)
save_panels(list(pA, pB, pC), NAME)

cat(sprintf("  A: n = %d cell lines, Spearman rho = %.4f\n", n_ph, rho5))
cat(sprintf("  B: top-12 improvement: %s +%.0f%% ... %s %+.0f%%\n",
            top12$drug[1], top12$improvement[1],
            top12$drug[12], top12$improvement[12]))
for (i in seq_len(nrow(top12))) {
  cat(sprintf("     %-22s orig|r|=%.3f phospho|r|=%.3f %+.1f%%\n",
              as.character(top12$drug[i]), abs(top12$rho_original[i]),
              abs(top12$rho_phospho[i]), top12$improvement[i]))
}
cat("  C: kinase counts per pathway:", paste(sprintf("%s=%d", kp$V1, kp$n_kinases), collapse = " "), "\n")
cat(sprintf("  PDF: %s\n", file.path(FIG_DIR, paste0(NAME, ".pdf"))))
