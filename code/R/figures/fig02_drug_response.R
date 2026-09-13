# ============================================================================
# fig02_drug_response.R — Figure 2: drug-response and clinical associations. Panels A-D: (A) TCGA pan-cancer survival meta-analysis forest, (B) GDSC pan-drug resistance (286 drugs), (C) GSE25066 neoadjuvant pCR by NetITH tertile, (D) IMvigor210 immunotherapy response by tertile.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/tcga/{tcga_survival_results,multicancer_meta_analysis}.csv,
#           results/gdsc/gdsc_drug_netith_correlations.csv,
#           results/neoadjuvant/gse25066_netith_pcr.csv,
#           results/neoadjuvant/gse25066_fixed_network_netith.csv,
#           results/neoadjuvant/gse25066_fixed_network_summary.json,
#           results/imvigor210/imvigor210_netith_results.csv
# Outputs : results/figures/r/Fig2_drug_response_clinical.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/fig02_drug_response.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "Fig2_drug_response_clinical"
TCGA_D <- file.path(RESULTS_DIR, "tcga"); GDSC_D <- file.path(RESULTS_DIR, "gdsc")

# ---- Panel A: pan-cancer survival forest plot with RE diamond -----------------
surv <- fread(file.path(TCGA_D, "tcga_survival_results.csv"))
surv <- surv[is.finite(cox_hr) & cox_hr > 0]
setorder(surv, cox_hr)
hrs <- surv$cox_hr
se_logHR <- abs(log(pmax(hrs, 0.01))) / pmax(qnorm(1 - pmin(pmax(surv$cox_p, 1e-300), 0.5) / 2), 0.01)
surv[, ci_lo := exp(log(hrs) - 1.96 * se_logHR)][, ci_hi := exp(log(hrs) + 1.96 * se_logHR)]
surv[, sig := (hrs < 1 & ci_hi < 1) | (hrs > 1 & ci_lo > 1)]
surv[, cancer := factor(cancer_type, levels = rev(cancer_type))]

meta <- fread(file.path(TCGA_D, "multicancer_meta_analysis.csv"))[1]
pA <- ggplot(surv, aes(cox_hr, cancer, colour = sig)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.25, linewidth = 0.5) +
  geom_point(size = 1.6) +
  scale_colour_manual(values = c(`FALSE` = COL_NS, `TRUE` = COL_SIG), guide = "none") +
  annotate("point", x = meta$re_hr, y = 0.3, shape = 23, size = 2.6, fill = COL_NER, colour = COL_NER) +
  annotate("errorbarh", xmin = meta$re_ci_lower, xmax = meta$re_ci_upper, y = 0.3,
           height = 0.2, colour = COL_NER, linewidth = 0.9) +
  annotate("text", x = meta$re_hr, y = 1.2, size = FONT_BASE/.pt, fontface = "bold", colour = COL_NER,
           label = sprintf("RE HR=%.3f\nI\u00b2=%.1f%%", meta$re_hr, meta$I2)) +
  annotate("text", x = 0.02, y = 0.05, hjust = 0, size = FONT_BASE/.pt, colour = "grey40",
           label = sprintf("FE HR=%.3f, p=%.3f\n(RE p=%.3f, n=26 cancers)", meta$fe_hr, meta$fe_p, meta$re_p)) +
  scale_x_log10(labels = scales::label_number(accuracy = 0.5)) +
  coord_cartesian(xlim = c(0.1, 10), ylim = c(0, length(surv) + 1.5)) +
  labs(title = "A  TCGA pan-cancer survival meta-analysis",
       x = "Hazard Ratio, high vs. low NetITH (tertile Cox, per cancer)", y = NULL) +
  theme_pub()

# ---- Panel B: GDSC pan-drug resistance (pathway-grouped horizontal bars) ------
dc <- fread(file.path(GDSC_D, "gdsc_drug_netith_correlations.csv"))
dc[is.na(pathway), pathway := "Unclassified"]
ord <- dc[, .(m = mean(rho, na.rm = TRUE)), by = pathway][order(-m), pathway]
dc[, pathway := factor(pathway, levels = ord)]
dc <- dc[order(pathway, rho)]
dc[, y := seq_len(.N)]
pB <- ggplot(dc, aes(rho, y, fill = pathway)) +
  geom_col(width = 0.8, alpha = 0.55) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  scale_fill_manual(values = rep(NPG_COLORS, length.out = uniqueN(dc$pathway)), guide = "none") +
  annotate("text", x = Inf, y = -Inf, hjust = 1.05, vjust = -0.6, size = FONT_BASE/.pt,
           fontface = "bold", colour = COL_NER,
           label = sprintf("%.0f%% positive\ndirection", 100 * mean(dc$rho > 0))) +
  labs(title = "B  GDSC pan-drug resistance (n=286 drugs)",
       x = "Spearman \u03c1 (NetITH vs. ln IC\u2085\u2080); 276/286 FDR<0.05 (BH)", y = NULL) +
  theme_pub() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

# ---- Panel C: GSE25066 neoadjuvant pCR ----------------------------------------
pcr <- fread(file.path(RESULTS_DIR, "neoadjuvant", "gse25066_netith_pcr.csv"))
pcr <- pcr[!is.na(pcr_bin) & !is.na(netith_z)]
pcr[, tert := cut(netith_z, breaks = quantile(netith_z, c(0, 1/3, 2/3, 1)),
                  labels = c("Low", "Mid", "High"), include.lowest = TRUE)]
# Fixed 239-gene network construction (R2 recomputation; cached 62-edge variant was a
# construction artefact: OR 1.27 -> 0.86 when the network was held fixed)
fix <- data.table::fread(file.path(RESULTS_DIR, "neoadjuvant", "gse25066_fixed_network_netith.csv"))
fix[, tert := cut(NetITH_239, breaks = quantile(NetITH_239, c(0, 1/3, 2/3, 1)),
                  labels = c("Low", "Mid", "High"), include.lowest = TRUE)]
rate <- fix[, .(rate = 100 * mean(pcr_bin), n = .N), by = tert]
rate[, tert := factor(tert, levels = c("Low", "Mid", "High"))]
nj <- jsonlite::fromJSON(file.path(RESULTS_DIR, "neoadjuvant", "gse25066_fixed_network_summary.json"))$construction_A
pC <- ggplot(rate, aes(tert, rate, fill = tert)) +
  geom_col(width = 0.6, alpha = 0.75) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", rate, n)), vjust = -0.4, size = FONT_BASE/.pt) +
  scale_fill_manual(values = c(Low = COL_TCGA, Mid = COL_NS, High = COL_GDSC), guide = "none") +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = FONT_BASE/.pt, colour = COL_SIG,
           label = sprintf("OR=%.2f per SD [%.2f-%.2f]\np=%.2f (logistic)", nj$or_per_sd, nj$ci[1], nj$ci[2], nj$p)) +
  coord_cartesian(ylim = c(0, max(rate$rate) * 1.35)) +
  labs(title = "C  Neoadjuvant chemotherapy (GSE25066; n=306 patients, fixed network)",
       x = "NetITH tertile", y = "pCR rate (%)") +
  theme_pub()

# ---- Panel D: IMvigor210 immunotherapy ------------------------------------------
imv <- fread(file.path(RESULTS_DIR, "imvigor210", "imvigor210_netith_results.csv"))
imv[, bres := fcase(binaryResponse == "CR/PR", 1, binaryResponse == "SD/PD", 0, default = NA_real_)]
imv <- imv[!is.na(bres) & !is.na(NetITH)]
imv[, z := (NetITH - mean(NetITH)) / sd(NetITH)]
imv[, tert := cut(z, breaks = quantile(z, c(0, 1/3, 2/3, 1)),
                  labels = c("Low", "Mid", "High"), include.lowest = TRUE)]
rate2 <- imv[, .(rate = 100 * mean(bres), n = .N), by = tert]
rate2[, tert := factor(tert, levels = c("Low", "Mid", "High"))]
glm_d <- suppressWarnings(glm(bres ~ z, family = binomial(), data = imv))
or_d <- exp(coef(glm_d)[["z"]]); p_d <- summary(glm_d)$coefficients["z", 4]
pD <- ggplot(rate2, aes(tert, rate, fill = tert)) +
  geom_col(width = 0.6, alpha = 0.75) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", rate, n)), vjust = -0.4, size = FONT_BASE/.pt) +
  scale_fill_manual(values = c(Low = COL_TCGA, Mid = COL_NS, High = COL_GDSC), guide = "none") +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = FONT_BASE/.pt, colour = "grey40",
           label = sprintf("OR=%.2f per SD\np=%.2f", or_d, p_d)) +
  coord_cartesian(ylim = c(0, max(rate2$rate) * 1.35)) +
  labs(title = "D  Immunotherapy (IMvigor210; n=298 with response)",
       x = "NetITH tertile", y = "Response rate (%)") +
  theme_pub()

panels <- list(pA, pB, pC, pD)
composite <- (pA | pB) / (pC | pD)
save_fig(composite, NAME, height_mm = 148.2)   # 7.0866 x 5.8360 in
save_panels(panels, NAME)
cat(sprintf("[fig02] done; IMvigor210 OR=%.2f p=%.2f\n", or_d, p_d))
