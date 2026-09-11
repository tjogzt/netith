# ============================================================================
# figED4_perturb_seq.R — Extended Data Figure 4: Perturb-seq landscape. Panels A-D: (A) virtual-perturbation volcano (Cohen's d vs FDR), (B) top TF mediator (text), (C) perturbation-drug signature coherence, (D) combinatorial TF interaction screen.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Inputs  : results/depmap/perturbation/virtual_perturbation_genome_wide.csv,
#           results/depmap/ext_d5_tf_perturb_mediation.csv,
#           results/depmap/ext_d5_perturb_drug_signature.csv,
#           results/depmap/ext_d5_combinatorial_screen.csv
# Outputs : results/figures/r/EDFig4_perturb_seq.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figED4_perturb_seq.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "EDFig4_perturb_seq"
DEP <- file.path(RESULTS_DIR, "depmap")
PERT <- file.path(DEP, "perturbation")

cat("=== EDFig4: Perturb-seq landscape ===\n")

# ---- Panel A: virtual perturbation volcano -----------------------------------
vp <- fread(file.path(PERT, "virtual_perturbation_genome_wide.csv"))
stopifnot(all(c("gene", "cohens_d", "fdr") %in% names(vp)))
vp <- vp[is.finite(cohens_d) & is.finite(fdr) & fdr > 0]
n_vp <- nrow(vp)
bonf_vp <- 0.05 / n_vp
vp[, neg_log_p := -log10(pmax(fdr, 1e-50))]
vp[, sig := fdr < bonf_vp]
n_sig_vp <- sum(vp$sig)

pA <- ggplot(vp, aes(cohens_d, neg_log_p)) +
  geom_point(data = vp[sig == FALSE], colour = COL_NS, size = 0.3, alpha = 0.15, shape = 16) +
  geom_point(data = vp[sig == TRUE], colour = COL_SIG, size = 1.2, alpha = 0.6, shape = 16) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", colour = COL_NER, linewidth = 0.4) +
  geom_hline(yintercept = -log10(bonf_vp), linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  labs(title = "A  Virtual perturbation effect sizes",
       x = "Cohen's d", y = "-log\u2081\u2080(p)") +
  theme_pub()

# ---- Panel B: TF perturbation mediation (top mediator, text panel) -----------
tf <- fread(file.path(DEP, "ext_d5_tf_perturb_mediation.csv"))
stopifnot(all(c("tf", "prop_mediated", "ci95_low", "ci95_high") %in% names(tf)))
top <- tf[which.max(prop_mediated)]
b_label <- sprintf("%s\n%.1f%% mediated\nCI95 [%.1f%%, %.1f%%]",
                   top$tf, 100 * top$prop_mediated,
                   100 * top$ci95_low, 100 * top$ci95_high)
pB <- ggplot() +
  xlim(0, 1) + ylim(0, 1) +
  annotate("text", x = 0.5, y = 0.5, label = b_label, size = 10 / .pt,
           fontface = "bold", colour = COL_NETITH, hjust = 0.5, vjust = 0.5) +
  labs(title = "B  TF perturbation mediation") +
  theme_void(base_size = FONT_BASE) +
  theme(plot.title = element_text(size = FONT_TITLE, face = "bold"))

# ---- Panel C: perturbation-drug signature coherence ---------------------------
pds <- fread(file.path(DEP, "ext_d5_perturb_drug_signature.csv"))
stopifnot(all(c("drug", "rho_perturb_ic50") %in% names(pds)))
rp <- pds$rho_perturb_ic50[is.finite(pds$rho_perturb_ic50)]
n_neg <- sum(rp < 0)
mean_rp <- mean(rp)

pC <- ggplot(data.frame(rho = rp), aes(rho)) +
  geom_histogram(bins = 40, fill = COL_NETITH, alpha = 0.55, colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = mean_rp, linetype = "dashed", colour = COL_NER, linewidth = 0.5) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.4, size = FONT_BASE/.pt,
           fontface = "bold", colour = COL_NER,
           label = sprintf("%d/%d negative\nmean \u03c1=%.3f", n_neg, length(rp), mean_rp)) +
  labs(title = "C  Perturbation-drug signature coherence",
       x = "\u03c1 (perturbation effect vs. drug sensitivity)", y = "Drug count") +
  theme_pub()

# ---- Panel D: combinatorial TF interactions -----------------------------------
comb <- fread(file.path(DEP, "ext_d5_combinatorial_screen.csv"))
stopifnot(all(c("tf_a", "tf_b", "p_interaction") %in% names(comb)))
n_sig_comb <- sum(comb$p_interaction < 0.05)
n_comb <- nrow(comb)
db <- data.frame(status = factor(c("Significant", "Not significant"),
                                 levels = c("Significant", "Not significant")),
                 n = c(n_sig_comb, n_comb - n_sig_comb))

pD <- ggplot(db, aes(status, n, fill = status)) +
  geom_col(alpha = 0.6, width = 0.5) +
  scale_fill_manual(values = c("Significant" = COL_SIG, "Not significant" = COL_NS), guide = "none") +
  annotate("text", x = 1, y = n_sig_comb + 2, size = FONT_BASE/.pt, fontface = "bold",
           colour = COL_SIG,
           label = sprintf("%d\n(%.1f%%)", n_sig_comb, 100 * n_sig_comb / n_comb)) +
  labs(title = "D  Combinatorial TF interactions",
       x = NULL, y = "TF pairs") +
  theme_pub() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

p <- (pA | pB) / (pC | pD)
save_fig(p, NAME, height_mm = 213.4)
save_panels(list(pA, pB, pC, pD), NAME)

cat(sprintf("  A: %d perturbations; Bonferroni %.2e; significant %d (%.1f%%)\n",
            n_vp, bonf_vp, n_sig_vp, 100 * n_sig_vp / n_vp))
cat(sprintf("  B: top mediator: %s, %.1f%% mediated, CI95 [%.1f%%, %.1f%%]\n",
            top$tf, 100 * top$prop_mediated, 100 * top$ci95_low, 100 * top$ci95_high))
cat(sprintf("  C: %d/%d negative, mean rho = %.3f\n", n_neg, length(rp), mean_rp))
cat(sprintf("  D: %d/%d interaction pairs significant (%.1f%%)\n",
            n_sig_comb, n_comb, 100 * n_sig_comb / n_comb))
cat(sprintf("  PDF: %s\n", file.path(FIG_DIR, paste0(NAME, ".pdf"))))
