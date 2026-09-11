# ============================================================================
# fig04_ner_hypothesis.R — Figure 4: NER hypothesis generation. Panels A-D: (A) NER five-module workflow schematic, (B) NER-2 TF-combination perturbation search (6,175 combos), (C) NER-3 drug-sensitization ranking (top 20 of 286), (D) NER-5 DrugComb v1.4 concordance.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/depmap/ner/ner2_tf_perturbation_search.csv,
#           results/depmap/ner/ner3_drug_sensitization.csv
#           (Panel D numbers come from results/NUMBER_TRUTH_TABLE.csv:
#            DC_MATCHED=11, DC_PERM=6/11 (54.5%, global 18.1%),
#            DC_STRICT=1/11 (9.1%, global 13.1%)).
# Outputs : results/figures/r/Fig4_ner_hypothesis_generation.{pdf,png}
#           results/figures/r/panels/Fig4_ner_hypothesis_generation_panel{A..D}.{pdf,png}
# Usage   : Rscript code/R/figures/fig04_ner_hypothesis.R
# Note    : All in-figure text English; numbers come from the cached CSVs only.
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "Fig4_ner_hypothesis_generation"
DEP_D <- file.path(RESULTS_DIR, "depmap")

# === Figure parameters (fixed block) =====================================
.base_fs  <- 8            # pt: axis/legend text (theme_pub default)
.title_fs <- 8            # pt: panel titles
.annot7   <- FONT_BASE/.pt      # in-panel annotation (~7 pt)
.hist_alpha <- 0.6
.bar_alpha  <- 0.7
# ========================================================================

# ---- Panel A: NER five-module workflow (schematic) -----------------------
steps <- data.table(
  x    = c(1.0, 3.4, 5.8, 8.2, 10.6),
  name = c("NER-1", "NER-2", "NER-3", "NER-4", "NER-5"),
  sub  = c("drug landscape", "TF search", "sensitization", "pathway ERI", "DrugComb"))
pA <- ggplot() +
  annotate("rect", xmin = steps$x - 0.9, xmax = steps$x + 0.9,
           ymin = 2.4, ymax = 4.2, fill = COL_NER, alpha = 0.15,
           colour = COL_NER, linewidth = 0.5) +
  annotate("text", x = steps$x, y = 3.65, label = steps$name,
           size = .annot7, fontface = "bold", colour = COL_NER) +
  annotate("text", x = steps$x, y = 2.95, label = steps$sub,
           size = .annot7, colour = COL_NER) +
  annotate("segment", x = steps$x[-5] + 0.9, xend = steps$x[-1] - 0.9,
           y = 3.3, yend = 3.3, colour = "grey55", linewidth = 0.5,
           arrow = arrow(length = unit(1.5, "mm"), type = "closed")) +
  annotate("text", x = 5.8, y = 1.15, hjust = 0.5, size = .annot7,
           fontface = "italic", colour = "grey45",
           label = "All outputs are hypotheses for prospective experimental testing") +
  labs(title = "A  NER five-module workflow (hypothesis-generating)") +
  coord_cartesian(xlim = c(0, 11.6), ylim = c(0.5, 4.6), clip = "off") +
  theme_void() +
  theme(plot.title = element_text(size = .title_fs, face = "bold", hjust = 0),
        plot.margin = margin(2, 2, 2, 2, "mm"))

# ---- Panel B: NER-2 TF combination perturbation search --------------------
# Python fig4() histogrammed n2.columns[0] ("k", a constant); the meaningful
# column is the predicted dNetITH (delta_netith_pred), used here.
n2 <- fread(file.path(DEP_D, "ner", "ner2_tf_perturbation_search.csv"))
n2_range <- n2[, range(delta_netith_pred)]
pB <- ggplot(n2, aes(delta_netith_pred)) +
  geom_histogram(bins = 50, fill = COL_NETITH, alpha = .hist_alpha, colour = NA) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  labs(title = sprintf("B  NER-2: TF combination search (%s combinations)",
                       format(nrow(n2), big.mark = ",")),
       x = "predicted \u0394NetITH", y = "combinations") +
  theme_pub()

# ---- Panel C: NER-3 drug sensitization ranking (top 20 of 286) ------------
n3 <- fread(file.path(DEP_D, "ner", "ner3_drug_sensitization.csv"))
slope_col <- names(n3)[grepl("slope|sens", names(n3), ignore.case = TRUE)][1]
stopifnot(!is.na(slope_col))
top20 <- n3[order(-get(slope_col))][1:20]
top20[, drug := factor(drug, levels = rev(drug))]     # largest slope on top
pC <- ggplot(top20, aes(get(slope_col), drug)) +
  geom_col(fill = COL_GDSC, alpha = .bar_alpha, width = 0.7) +
  labs(title = sprintf("C  NER-3: drug sensitization ranking (top 20 of %d)", nrow(n3)),
       x = slope_col, y = NULL) +
  theme_pub()

# ---- Panel D: NER-5 DrugComb v1.4 concordance (threshold-dependent) ------
# Numbers per results/NUMBER_TRUTH_TABLE.csv (DC_MATCHED=11, DC_PERM=6/11,
# DC_STRICT=1/11; global 18.1% / 13.1%). Python hardcodes 6/5/1.
dc <- data.table(cat = c("permissive\n(any ZIP>10)", "not concordant",
                         "strict\n(median ZIP>10)"),
                 n   = c(6, 5, 1))
pD <- ggplot(dc, aes(cat, n)) +
  geom_col(aes(fill = cat), width = 0.5,
           alpha = c(0.7, 0.7, 0.85)) +
  scale_fill_manual(values = c("permissive\n(any ZIP>10)" = COL_SIG,
                               "not concordant" = COL_NS,
                               "strict\n(median ZIP>10)" = COL_GDSC), guide = "none") +
  annotate("text", x = 1, y = 6.35, hjust = 0.5, size = .annot7, label = "6 (54.5%)") +
  annotate("text", x = 3, y = 1.35, hjust = 0.5, size = .annot7, label = "1 (9.1%)") +
  annotate("text", x = 2, y = -0.55, hjust = 0.5, vjust = 1, size = .annot7,
           colour = "grey45", fontface = "italic",
           label = "global rates: 18.1% permissive / 13.1% strict") +
  scale_y_continuous(breaks = seq(0, 6, 2)) +
  coord_cartesian(ylim = c(-1.1, 7.3)) +
  labs(title = "D  NER-5: DrugComb v1.4 database concordance\n(threshold-dependent)",
       x = NULL, y = "concordant pairs (of 11 matched in DrugComb v1.4)") +
  theme_pub()

# ---- Assemble (2 x 2 grid; Python gridspec 2x2) ---------------------------
panels <- list(pA, pB, pC, pD)
composite <- (pA | pB) / (pC | pD)
save_fig(composite, NAME, height_mm = 158.8)   # 7.0866 x 6.2529 in -> 158.8 mm
save_panels(panels, NAME)

# ---- Key numbers per panel (cross-checked against cached CSVs) ------------
cat(sprintf("[fig04] A: schematic, 5 modules (NER-1..NER-5)\n"))
cat(sprintf("[fig04] B: n=%d combinations; predicted dNetITH range %.3f..%.3f; top combo %s\n",
            nrow(n2), n2_range[1], n2_range[2], n2[which.max(abs_delta), combo]))
cat(sprintf("[fig04] C: n=%d drugs; top-20 slopes %.2f..%.2f (%s); top drug %s (slope=%.2f)\n",
            nrow(n3), top20[1, get(slope_col)], top20[20, get(slope_col)], slope_col,
            top20[1, drug], top20[1, get(slope_col)]))
cat(sprintf("[fig04] D: 11 matched in DrugComb v1.4; permissive 6/11 (54.5%%), strict 1/11 (9.1%%); global 18.1%% / 13.1%%\n"))
cat("[fig04] done\n")
