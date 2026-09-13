# ============================================================================
# figS4_signed_vs_unsigned.R — Supplementary Figure S4: signed vs unsigned (|w|) construction comparison. Panels A-D: (A) Laplacian eigenvalue spectra, (B) information-gain decomposition, (C) NetITH distributions, (D) channel decomposition (E2 amplitude-matched null).
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-09
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv (via 01_data_import.R),
#           results/gdsc/gdsc_drug_netith_correlations{,_abs_weight}.csv,
#           results/control/amplitude_equalized_null.json,
#           results/control/focused_mad_baseline.json
# Outputs : results/figures/r/FigS4_signed_vs_unsigned.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS4_signed_vs_unsigned.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(Matrix); library(ggplot2); library(patchwork) })

NAME <- "FigS4_signed_vs_unsigned"
dat <- source(file.path(CODE_DIR, "01_data_import.R"))$value
edges <- dat$collectri; expr <- as.matrix(dat$gdsc$expr); focused <- dat$focused
common <- intersect(focused, rownames(expr))
gene_to_idx <- setNames(seq_along(common), common)
e <- edges[source %in% common & target %in% common]
e <- e[, .(tf_idx = gene_to_idx[source], target_idx = gene_to_idx[target], weight = as.numeric(weight))]
z_all <- t(apply(expr[common, , drop = FALSE], 1, function(g) {
  m <- mean(g); s <- sd(g) + 1e-10; pmin(pmax((g - m) / s, -3), 3)
}))

spectrum <- function(z, wsign) {
  vals <- wsign * abs(z[e$tf_idx]) * abs(z[e$target_idx])
  keep <- which(vals != 0)
  A <- Matrix::sparseMatrix(i = e$tf_idx[keep], j = e$target_idx[keep], x = vals[keep],
                            dims = c(length(common), length(common)))
  A <- A + Matrix::t(A)
  deg <- Matrix::rowSums(A); L <- Matrix::Diagonal(x = deg) - A
  tr <- sum(deg)
  ev <- sort(pmax(eigen(L, symmetric = TRUE, only.values = TRUE)$values, 0) / tr, decreasing = TRUE)
  ev[ev > 1e-12]
}

# representative cell lines: 10/50/90th percentile of signed NetITH
nt_signed <- vapply(seq_len(ncol(z_all)), function(i) {
  z <- z_all[, i]; ev <- spectrum(z, e$weight); if (length(ev) == 0) return(0)
  rho <- pmin(pmax(ev, 1e-12), 1); -sum(rho * log2(rho))
}, numeric(1))
names(nt_signed) <- colnames(z_all)
q <- quantile(nt_signed, c(0.10, 0.50, 0.90))
rep_lines <- names(nt_signed)[vapply(q, function(x) which.min(abs(nt_signed - x)), integer(1))]

spec_df <- rbindlist(lapply(rep_lines, function(cl) {
  z <- z_all[, cl]
  rbind(
    data.table(cell = cl, construction = "signed", idx = seq_along(spectrum(z, e$weight)), lambda = spectrum(z, e$weight)),
    data.table(cell = cl, construction = "unsigned (|w|)", idx = seq_along(spectrum(z, abs(e$weight))), lambda = spectrum(z, abs(e$weight)))
  )
}))
spec_df[, cell := factor(cell, levels = rep_lines, labels = paste0("NetITH p", c(10, 50, 90)))]

pA <- ggplot(spec_df, aes(idx, lambda, colour = construction, linetype = construction)) +
  geom_line(linewidth = 0.5) +
  facet_wrap(~cell, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c("signed" = COL_NETITH, "unsigned (|w|)" = COL_GDSC)) +
  scale_linetype_manual(values = c("signed" = "solid", "unsigned (|w|)" = "dashed")) +
  labs(title = "A  Normalised Laplacian eigenvalue spectra\n(representative GDSC cell lines)",
       x = "eigenvalue rank", y = "normalised eigenvalue (lambda/tr(L))") +
  theme_pub() + theme(legend.position = "top", legend.title = element_blank())

gain <- data.table(
  construction = factor(c("sign-null\n(signs randomised)", "unsigned (|w|)", "signed"),
                        levels = c("sign-null\n(signs randomised)", "unsigned (|w|)", "signed")),
  median_rho = c(0.016, 0.174, 0.232))
pB <- ggplot(gain, aes(construction, median_rho, fill = construction)) +
  geom_col(width = 0.6, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.3f", median_rho)), vjust = -0.5, size = FONT_BASE/.pt) +
  scale_fill_manual(values = c("sign-null\n(signs randomised)" = COL_NS, "unsigned (|w|)" = COL_GDSC,
                               "signed" = COL_NETITH), guide = "none") +
  coord_cartesian(ylim = c(0, 0.27)) +
  annotate("text", x = 2.0, y = 0.255, hjust = 0.5, size = FONT_BASE/.pt, colour = COL_SIG,
           label = "signed - unsigned = 0.058\n(~58% of association strength\nattributable to edge signs)") +
  labs(title = "B  Information-gain decomposition\n(286 GDSC drugs)",
       x = NULL, y = "median Spearman \u03c1 (NetITH vs ln(IC50))") +
  theme_pub()

nt_abs <- vapply(seq_len(ncol(z_all)), function(i) {
  z <- z_all[, i]; ev <- spectrum(z, abs(e$weight)); if (length(ev) == 0) return(0)
  rho <- pmin(pmax(ev, 1e-12), 1); -sum(rho * log2(rho))
}, numeric(1))
dist_df <- rbind(
  data.table(construction = "signed", NetITH = nt_signed),
  data.table(construction = "unsigned (|w|)", NetITH = nt_abs))
pC <- ggplot(dist_df, aes(NetITH, fill = construction)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60, alpha = 0.45, colour = NA) +
  scale_fill_manual(values = c("signed" = COL_NETITH, "unsigned (|w|)" = COL_GDSC)) +
  labs(title = "C  NetITH distributions\n(n=1,013 GDSC cell lines)",
       x = "NetITH", y = "Density") +
  theme_pub() + theme(legend.position = "top", legend.title = element_blank())

# ---- Panel D: channel decomposition (E2 amplitude-matched null) ---------------
e2 <- jsonlite::fromJSON(file.path(RESULTS_DIR, "control", "amplitude_equalized_null.json"))
mad_ref <- jsonlite::fromJSON(file.path(RESULTS_DIR, "control", "focused_mad_baseline.json"))
chan <- data.table(
  channel = factor(c("MAD\n(amplitude-only)", "NetITH\n(full)", "amplitude-equalized\n(wiring+sign)", "rewired\nequalized null"),
                   levels = c("MAD\n(amplitude-only)", "NetITH\n(full)", "amplitude-equalized\n(wiring+sign)", "rewired\nequalized null")),
  median_rho = c(mad_ref$median_rho, e2$full_median_rho, e2$equalized_median_rho, e2$rewire_null_mean),
  sd = c(NA, NA, NA, e2$rewire_null_sd))
pD <- ggplot(chan, aes(channel, median_rho, fill = channel)) +
  geom_col(width = 0.62, alpha = 0.85) +
  geom_errorbar(aes(ymin = median_rho - ifelse(is.na(sd), 0, sd),
                    ymax = median_rho + ifelse(is.na(sd), 0, sd)), width = 0.15) +
  geom_text(aes(label = sprintf("%.3f", median_rho)), vjust = -0.6, size = FONT_BASE/.pt) +
  scale_fill_manual(values = c("MAD\n(amplitude-only)" = COL_NS, "NetITH\n(full)" = COL_NETITH,
                               "amplitude-equalized\n(wiring+sign)" = COL_NER, "rewired\nequalized null" = COL_REF),
                    guide = "none") +
  coord_cartesian(ylim = c(0, 0.32)) +
  labs(title = "D  Channel decomposition\n(286 GDSC drugs)",
       x = NULL, y = "median Spearman \u03c1 (vs ln(IC50))") +
  theme_pub()

panels <- list(pA, pB, pC, pD)
composite <- pA / pB / pC / pD
save_fig(composite, NAME, height_mm = 320, width_mm = 85)
save_panels(panels, NAME)
cat(sprintf("[figS4] done; signed median=%.4f | unsigned median=%.4f | gain=%.4f | equalized=%.4f\n",
            median(nt_signed), median(nt_abs), median(nt_signed) - median(nt_abs), e2$equalized_median_rho))
