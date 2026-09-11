# ============================================================================
# figS6_descriptor_census.R — Supplementary Figure S6: four-test protocol descriptor census. Panels A-D: (A) drug association (IC50/AUC), (B) dynamic-range strata, (C) Test-1 topology null, (D) Test-4 GSE25066 transfer.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-10
# Inputs  : results/control/census/drug_assoc_*.csv, null_sr_dne_activity.json,
#           results/control/census/gse25066_transfer.json, purity_corr.json;
#           GDSC2_IC50_all.csv via NETITH_DATA_DISK env (DATA_DISK)
# Outputs : results/figures/r/FigS6_descriptor_census.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS6_descriptor_census.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork); library(jsonlite) })

NAME <- "FigS6_descriptor_census"
CEN <- file.path(RESULTS_DIR, "control", "census")

DESC <- c(netith = "NetITH", sr = "Signaling entropy", dne = "Differential network entropy",
          activity = "TF-activity aggregate", shannon = "Expression Shannon",
          mad800 = "Expression MAD", cytotrace = "CytoTRACE proxy")
DESC_LEVELS <- rev(names(DESC))

load_med <- function(nm) {
  da <- fread(file.path(CEN, paste0("drug_assoc_", nm, ".csv")))
  list(med_ic50 = median(da$rho_ic50), med_auc = median(da$rho_auc),
       n_pos = sum(da$rho_ic50 > 0), n_fdr = sum(da$fdr_ic50 < 0.05))
}

# ---- assemble descriptor-level stats -----------------------------------------
stats <- rbindlist(lapply(names(DESC), function(nm) {
  s <- load_med(nm)
  data.table(descriptor = nm, med_ic50 = s$med_ic50, med_auc = s$med_auc,
             n_pos = s$n_pos, n_fdr = s$n_fdr)
}))
stats[, descriptor := factor(descriptor, levels = DESC_LEVELS)]

# strata (IC50 IQR tertiles) per descriptor
ic50_all <- fread(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"))
iqr <- ic50_all[, .(iqr = quantile(LN_IC50, .75, na.rm = TRUE) - quantile(LN_IC50, .25, na.rm = TRUE)),
                by = DRUG_NAME]
strata_long <- rbindlist(lapply(names(DESC), function(nm) {
  da <- fread(file.path(CEN, paste0("drug_assoc_", nm, ".csv")))
  da <- merge(da, iqr, by.x = "drug", by.y = "DRUG_NAME")
  da[, stratum := cut(rank(iqr, ties.method = "first"), 3, labels = c("low", "mid", "high"))]
  da[, .(median_rho = median(rho_ic50)), by = stratum][, descriptor := nm]
}))
strata_long[, descriptor := factor(descriptor, levels = DESC_LEVELS)]
strata_long[, stratum := factor(stratum, levels = c("low", "mid", "high"))]

# Test-1 nulls
null <- fromJSON(file.path(CEN, "null_sr_dne_activity.json"))
null_df <- rbindlist(lapply(c("sr", "dne", "activity"), function(nm) {
  data.table(descriptor = nm,
             real = load_med(nm)$med_ic50,
             null_mean = null[[nm]]$mean, null_sd = null[[nm]]$sd,
             null_lo = null[[nm]]$range[1], null_hi = null[[nm]]$range[2])
}))
null_df[, descriptor := factor(descriptor, levels = DESC_LEVELS)]

# Test-4 transfer
tr <- fromJSON(file.path(CEN, "gse25066_transfer.json"))
tr_df <- rbindlist(lapply(names(DESC), function(nm) {
  v <- tr[[nm]]
  data.table(descriptor = nm, or = v$or, lo = v$ci[1], hi = v$ci[2], p = v$p)
}), fill = TRUE)
tr_df[, descriptor := factor(descriptor, levels = DESC_LEVELS)]

COLS_DESC <- c(netith = COL_NETITH, sr = COL_SIG, dne = COL_NER,
               activity = COL_TCGA, shannon = COL_NS, mad800 = COL_REF,
               cytotrace = COL_ACCENT)

# panel titles anchored at the plot (not panel) left edge so long titles do
# not overflow the 85-mm canvas past wide y-axis labels
title_left <- theme(plot.title.position = "plot",
                    plot.title = element_text(hjust = 0))

# ---- Panel A: median rho (IC50 filled / AUC open) ----------------------------
stats_long <- rbind(
  stats[, .(descriptor, med = med_ic50, readout = "ln(IC50)")],
  stats[, .(descriptor, med = med_auc, readout = "AUC")]
)
pA <- ggplot(stats_long) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = COL_GREY, linewidth = 0.4) +
  geom_point(aes(med, descriptor, colour = descriptor, shape = readout), size = 2.2) +
  scale_colour_manual(values = COLS_DESC, guide = "none") +
  scale_shape_manual(values = c("ln(IC50)" = 19, "AUC" = 1), name = NULL) +
  scale_x_continuous(limits = c(-0.30, 0.45), breaks = seq(-0.3, 0.4, 0.1)) +
  scale_y_discrete(labels = DESC[levels(stats$descriptor)]) +
  labs(title = "A  Drug association (286 compounds)",
       x = "median Spearman \u03c1", y = NULL) +
  theme_pub() + title_left + theme(legend.position = "top")

# ---- Panel B: dynamic-range strata --------------------------------------------
pB <- ggplot(strata_long, aes(descriptor, median_rho, fill = stratum)) +
  geom_col(position = position_dodge(0.7), width = 0.65, alpha = 0.9) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_fill_manual(values = c(low = COL_REF, mid = COL_SIG, high = COL_GDSC),
                    name = "IC50 range") +
  scale_x_discrete(labels = DESC[levels(strata_long$descriptor)]) +
  coord_flip() +
  labs(title = "B  Dynamic-range strata (IC50 IQR tertiles)",
       x = NULL, y = "median Spearman \u03c1 (ln(IC50))") +
  theme_pub() + title_left + theme(legend.position = "top")

# ---- Panel C: Test-1 null ------------------------------------------------------
pC <- ggplot(null_df) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = COL_GREY, linewidth = 0.4) +
  geom_errorbarh(aes(y = descriptor, xmin = null_lo, xmax = null_hi),
                 colour = COL_NS, height = 0.2, linewidth = 1.2) +
  geom_point(aes(null_mean, descriptor), colour = COL_NS, size = 2.0) +
  geom_point(aes(real, descriptor, colour = descriptor), size = 2.6) +
  scale_colour_manual(values = COLS_DESC, guide = "none") +
  scale_y_discrete(labels = DESC[levels(null_df$descriptor)]) +
  scale_x_continuous(limits = c(-0.45, 0.45), breaks = seq(-0.4, 0.4, 0.2)) +
  labs(title = "C  Test 1: topology vs null (50 draws)",
       x = "median Spearman \u03c1 (ln(IC50))", y = NULL) +
  theme_pub() + title_left

# ---- Panel D: GSE25066 transfer -------------------------------------------------
pD <- ggplot(tr_df[!is.na(or)]) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = COL_GREY, linewidth = 0.4) +
  geom_errorbarh(aes(y = descriptor, xmin = lo, xmax = hi), height = 0.25, linewidth = 1.2) +
  geom_point(aes(or, descriptor, colour = descriptor), size = 2.6) +
  scale_colour_manual(values = COLS_DESC, guide = "none") +
  scale_y_discrete(labels = DESC[levels(tr_df$descriptor)]) +
  scale_x_log10(limits = c(0.5, 2.2), breaks = c(0.5, 0.75, 1, 1.5, 2)) +
  labs(title = "D  Test 4: GSE25066 neoadjuvant pCR",
       x = "odds ratio (95% CI, log scale)", y = NULL) +
  theme_pub() + title_left

composite <- pA / pB / pC / pD
save_fig(composite, NAME, height_mm = 300, width_mm = 85)
save_panels(list(pA, pB, pC, pD), NAME)

cat("=== FigS6 key numbers ===\n")
print(stats[, .(descriptor, med_ic50 = round(med_ic50, 3), med_auc = round(med_auc, 3),
                n_pos, n_fdr)])
cat("[figS6] done\n")
