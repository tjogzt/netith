# ============================================================================
# figS15_neoadjuvant.R — Supplementary Figure 15
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/neoadjuvant/gse25066_fixed_network_netith.csv,
#           results/neoadjuvant/gse25066_fixed_network_summary.json (cached stats)
# Outputs : results/figures/r/FigS15_neoadjuvant_pcr.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS15_neoadjuvant.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork); library(jsonlite) })

NAME <- "FigS15_neoadjuvant_pcr"
H_MM <- 2.8906 / 7.0866 * 180   # 73.4 mm (Python figsize 7.0866 x 2.8906 in)

NEO_D <- file.path(RESULTS_DIR, "neoadjuvant")
# Fixed 239-gene network construction (R2 recomputation; cached 62-edge variant was a
# construction artefact). Use fixed-network NetITH and summary for all panels.
df <- fread(file.path(NEO_D, "gse25066_fixed_network_netith.csv"))
df[, NetITH := NetITH_239]
sj <- fromJSON(file.path(NEO_D, "gse25066_fixed_network_summary.json"))$construction_A
# keep sj compatible with cached summary accessors used below
sj$n <- nrow(df); sj$n_pcr <- sum(df$pcr_bin == 1)

dd <- df[!is.na(pcr_bin)]
dd[, netith_z := (NetITH - mean(NetITH)) / sd(NetITH)]
cat(sprintf("  Cohort: n=%d, n_pcr=%d\n", nrow(dd), sum(dd$pcr_bin == 1)))
stopifnot(nrow(dd) == sj$n, sum(dd$pcr_bin == 1) == sj$n_pcr)

# ---- OR per SD (logistic regression, Wald CI — same as statsmodels Logit) ------
or_per_sd <- function(sub) {
  if (nrow(sub) < 20) return(c(or = NA_real_, lo = NA_real_, hi = NA_real_, p = NA_real_))
  m <- glm(pcr_bin ~ netith_z, data = sub, family = binomial)
  ci <- suppressMessages(confint.default(m))
  r <- c(exp(coef(m)[2]), exp(ci[2, 1]), exp(ci[2, 2]), summary(m)$coefficients[2, 4])
  names(r) <- c("or", "lo", "hi", "p")
  r
}
rows <- rbindlist(list(
  `Pooled (n=306)` = as.list(or_per_sd(dd)),
  `MD Anderson (n=227)` = as.list(or_per_sd(dd[source == "MDACC"])),
  `I-SPY (n=79)` = as.list(or_per_sd(dd[source == "ISPY"]))
), idcol = "label")
rows[, n := c(nrow(dd), nrow(dd[source == "MDACC"]), nrow(dd[source == "ISPY"]))]
rows[, star := ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "ns")))]
cat(sprintf("  [A] Pooled OR=%.3f [%.3f-%.3f] p=%.4f (cached: OR=%.3f [%.3f-%.3f] p=%.4f)\n",
            rows[1, or], rows[1, lo], rows[1, hi], rows[1, p],
            sj$or_per_sd, sj$ci[1], sj$ci[2], sj$p))
stopifnot(abs(rows[1, or] - sj$or_per_sd) < 0.01)

# ---- tertile pCR rates -----------------------------------------------------------
dd[, tert := cut(NetITH, breaks = quantile(NetITH, c(0, 1/3, 2/3, 1)),
                include.lowest = TRUE, labels = c("Low", "Mid", "High"))]
tr <- dd[, .(rate = mean(pcr_bin), n = .N), by = tert]
tr[, tert := factor(tert, levels = c("Low", "Mid", "High"))]
tr[, rate_pct := rate * 100]
cat(sprintf("  [B] tertile pCR rates (Low/Mid/High): %s (n=%s)\n",
            paste(sprintf("%.1f%%", tr[order(tert), rate_pct]), collapse = " / "),
            paste(tr[order(tert), n], collapse = " / ")))
cat(sprintf("  [B] high-vs-low tertile rate diff: %.4f (cached: %.4f)\n",
            tr[tert == "High", rate] - tr[tert == "Low", rate],
            sj$pcr_rate_high_vs_low_tertile))

# ---- Mann-Whitney: NetITH by pCR status ------------------------------------------
mw <- wilcox.test(dd[pcr_bin == 1, NetITH], dd[pcr_bin == 0, NetITH])
cat(sprintf("  [C] Mann-Whitney p=%.4f (cached: %.4f) | pCR n=%d, RD n=%d\n",
            mw$p.value, sj$mw_p, sum(dd$pcr_bin == 1), sum(dd$pcr_bin == 0)))
stopifnot(abs(mw$p.value - sj$mw_p) < 0.01)

# ---- Panel A: forest plot ---------------------------------------------------------
fa <- rows[order(-n)]
fa[, y := rev(seq_len(.N))]
pA <- ggplot(fa, aes(or, y)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_errorbar(aes(xmin = lo, xmax = hi, y = y), width = 0.15, linewidth = 0.5, orientation = "y",
                 colour = ifelse(fa$p < 0.05, COL_NETITH, COL_NS)) +
  geom_point(size = 2.2, colour = ifelse(fa$p < 0.05, COL_NETITH, COL_NS)) +
  annotate("text", x = fa$hi + 0.06, y = fa$y, hjust = 0, vjust = 0.5, size = FONT_BASE/.pt,
           label = sprintf("%.2f\n[%.2f-%.2f] %s", fa$or, fa$lo, fa$hi, fa$star)) +
  scale_y_continuous(breaks = fa$y, labels = fa$label) +
  scale_x_log10(limits = c(0.45, 4.2), breaks = c(0.5, 1, 2), labels = c("0.5", "1", "2")) +
  labs(title = "(A)  NetITH to pCR", x = "OR per SD (pCR, 95% CI)", y = NULL) +
  theme_pub()

# ---- Panel B: tertile pCR rates -----------------------------------------------------
pB <- ggplot(tr, aes(tert, rate_pct, fill = tert)) +
  geom_col(width = 0.55, alpha = 0.8, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(Low = COL_TCGA, Mid = COL_NS, High = COL_GDSC), guide = "none") +
  geom_text(aes(label = sprintf("%.1f%%", rate_pct)), vjust = -0.8, size = FONT_BASE/.pt, fontface = "bold") +
  geom_text(aes(label = sprintf("n=%d", n)), y = -3, size = FONT_BASE/.pt, colour = "dimgray") +
  scale_y_continuous(limits = c(0, 32)) +
  labs(title = "(B)  pCR by NetITH tertile", x = NULL, y = "pCR rate (%)") +
  theme_pub()

# ---- Panel C: NetITH by response (pCR vs RD) ------------------------------------------
n1 <- sum(dd$pcr_bin == 1); n0 <- sum(dd$pcr_bin == 0)
dd[, resp := factor(ifelse(pcr_bin == 1, sprintf("pCR\n(n=%d)", n1), sprintf("RD\n(n=%d)", n0)),
                   levels = c(sprintf("pCR\n(n=%d)", n1), sprintf("RD\n(n=%d)", n0)))]
set.seed(SEED)
mw_lab <- ifelse(mw$p.value < 0.05, "*", "ns")
pC <- ggplot(dd, aes(resp, NetITH)) +
  geom_boxplot(aes(fill = resp), width = 0.45, outlier.shape = NA, alpha = 0.75,
               linewidth = 0.3) +
  scale_fill_manual(values = setNames(c(COL_GDSC, COL_TCGA),
                                     c(sprintf("pCR\n(n=%d)", n1), sprintf("RD\n(n=%d)", n0))),
                    guide = "none") +
  geom_jitter(width = 0.06, size = 0.3, alpha = 0.35) +
  annotate("text", x = 1.5, y = max(dd$NetITH) + 0.1 * diff(range(dd$NetITH)), hjust = 0.5,
           size = FONT_BASE/.pt, label = sprintf("Mann-Whitney %s (p=%.3f)", mw_lab, mw$p.value)) +
  labs(title = "(C)  NetITH by response", x = NULL, y = "NetITH") +
  theme_pub()

# ---- composite + save -------------------------------------------------------------------
p <- pA | pB | pC
save_fig(p, NAME, height_mm = H_MM)
save_panels(list(pA, pB, pC), NAME)

# ---- print key numbers ---------------------------------------------------------------------
cat("\n=== EDFig9 key numbers ===\n")
for (i in seq_len(nrow(rows)))
  cat(sprintf("  %-20s OR=%.3f [%.3f-%.3f] p=%.4f %s (n=%d)\n",
              rows[i, label], rows[i, or], rows[i, lo], rows[i, hi], rows[i, p], rows[i, star], rows[i, n]))
cat(sprintf("  Tertile pCR rates: %s%%\n", paste(sprintf("%.1f", tr$rate_pct), collapse = " / ")))
cat(sprintf("  Mann-Whitney p=%.4f (%s)\n", mw$p.value, mw_lab))
