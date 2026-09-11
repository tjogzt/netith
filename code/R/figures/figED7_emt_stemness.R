# ============================================================================
# figED7_emt_stemness.R — Extended Data Figure 7: NetITH biological anchoring (EMT, stemness mRNAsi/EREG, DTP). Panels A-D: (A) GDSC 76-gene EMT score vs NetITH, (B) TCGA mRNAsi vs bulk NetITH, (C) DTP gene-enrichment volcano, (D) summary bars.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : data/gdsc/{rna_expr,cell_annot,ensg_symbol_map}.csv,
#           results/gdsc/gdsc_netith_cell_lines.csv, results/tcga/tcga_netith.csv,
#           results/depmap/emt_stemness_results.csv (cached headline stats),
#           $NETITH_DATA_ROOT/xena/tcgapancan/StemnessScores_RNAexp_20170127.2.tsv.gz
#           (data disk via NETITH_DATA_DISK env; see Usage)
# Outputs : results/figures/r/EDFig7_emt_stemness_dtp.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : NETITH_DATA_ROOT=$NETITH_DATA_DISK/data Rscript code/R/figures/figED7_emt_stemness.R
# Note    : Former placeholder panel B (TCGA Hallmark EMT ssGSEA) removed: ssGSEA scores unavailable; GDSC EMT panel A suffices.
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })

NAME <- "EDFig7_emt_stemness_dtp"
H_MM <- 4.4291 / 7.0866 * 180   # 112.5 mm (Python figsize 7.0866 x 4.4291 in)

DATA_ROOT <- Sys.getenv("NETITH_DATA_ROOT", unset = file.path(PROJECT_ROOT, "data"))
GDSC_D <- file.path(RESULTS_DIR, "gdsc"); TCGA_D <- file.path(RESULTS_DIR, "tcga")
DEP_D  <- file.path(RESULTS_DIR, "depmap")

# cached headline statistics (authoritative, from Python pipeline)
res <- fread(file.path(DEP_D, "emt_stemness_results.csv"))
stopifnot(nrow(res) == 1L)

# ---- 76-gene EMT signature (Byers/EMT consensus; from Python source) ----------
EMT_EPITHELIAL <- c('CDH1','OCLN','CLDN4','CLDN7','TJP1','TJP2','TJP3','DSP','PKP3',
  'KRT5','KRT7','KRT8','KRT18','KRT19','EPCAM','ESRP1','ESRP2','RAB25','MUC1','ST14',
  'MARVELD2','MARVELD3','GRHL2','OVOL2','ELF3','ELF5','SPINT1','SPINT2','CDS1','LLGL2')
EMT_MESENCHYMAL <- c('CDH2','VIM','FN1','SNAI1','SNAI2','ZEB1','ZEB2','TWIST1','TWIST2',
  'FOXC2','GSC','LEF1','TCF3','TCF4','MMP2','MMP3','MMP9','COL1A1','COL1A2','COL3A1',
  'COL5A2','FBN1','THBS1','THBS2','SPARC','TNC','ITGA5','ITGB1','POSTN','LOX','LOXL2',
  'PDGFRB','TAGLN','ACTA2','ACTG2','MYL9','CNN1','DES','S100A4','FAP','VCAN','CD44',
  'ABCC4','PLOD2','PCOLCE','SERPINE1')
stopifnot(length(EMT_EPITHELIAL) + length(EMT_MESENCHYMAL) == 76L)

# ---- DTP up-regulated markers (Sharma 2010 / Ramirez 2016 / DTP consensus) ----
DTP_UPREGULATED <- c('KDM5A','KDM5B','KDM6A','KDM6B','CDKN1A','CDKN1B','SOX2','SOX10',
  'NGFR','NES','ALDH1A1','ALDH1A3','ABCG2','ABCB1','HES1','HEY1','NOTCH1','NOTCH3',
  'IGF1R','IGFBP2','IGFBP3','WNT5A','FZD7','AXIN2','HDAC1','HDAC2','HDAC3','EZH2',
  'SUZ12','BCL2L1','MCL1','BIRC5','XIAP','NFKB1','NFKB2','RELA','RELB','GDF15',
  'TGFB1','TGFBR1','EPHA2','MET','AXL','EGFR','ATF4','DDIT3','XBP1','GPX4','SLC7A11','NFE2L2')

# ==============================================================================
#  load GDSC expression (ENSG x cell lines) and NetITH
# ==============================================================================
cat("=== [EDFig7] loading GDSC expression ===\n")
expr <- fread(file.path(DATA_DIR, "gdsc", "rna_expr.csv"))
setnames(expr, 1, "ensg")
annot <- fread(file.path(DATA_DIR, "gdsc", "cell_annot.csv"))
setnames(annot, 1, "cel_id")
cell_map <- annot[, .(cel_id, cl = `Characteristics.cell.line.`)][cl != "" & !is.na(cl)]

cols_keep <- intersect(names(expr)[-1], cell_map$cel_id)
expr <- expr[, c("ensg", cols_keep), with = FALSE]
setnames(expr, cols_keep, cell_map[match(cols_keep, cel_id), cl])
dup_cl <- duplicated(names(expr))
expr <- expr[, !dup_cl, with = FALSE]
cat(sprintf("  GDSC expression: %d genes x %d cell lines\n", nrow(expr), ncol(expr) - 1))

gmap <- fread(file.path(DATA_DIR, "gdsc", "ensg_symbol_map.csv"))
ensg2sym <- setNames(gmap$symbol, gmap$ensg)
matched <- expr$ensg %in% names(ensg2sym)
cat(sprintf("  Gene coverage: %d/%d (%.1f%%)\n", sum(matched), nrow(expr),
            100 * sum(matched) / nrow(expr)))
expr <- expr[matched]
expr[, ensg := ensg2sym[ensg]]
expr <- expr[!duplicated(ensg)]

m <- as.matrix(expr[, -1, with = FALSE])
rownames(m) <- expr$ensg
rm(expr); gc()

ni <- fread(file.path(GDSC_D, "gdsc_netith_cell_lines.csv"))
common_cells <- intersect(colnames(m), ni$cell_line)
m <- m[, common_cells, drop = FALSE]
netith_vals <- ni[match(common_cells, cell_line), NetITH]
cat(sprintf("  Common cell lines (expr x NetITH): %d\n", length(common_cells)))

# ==============================================================================
#  Panel A: GDSC EMT score (76-gene) vs NetITH
# ==============================================================================
epi <- intersect(EMT_EPITHELIAL, rownames(m)); mes <- intersect(EMT_MESENCHYMAL, rownames(m))
cat(sprintf("  EMT genes available: %d epi + %d mes = %d/76\n",
            length(epi), length(mes), length(epi) + length(mes)))
emt_score <- colMeans(m[mes, , drop = FALSE]) - colMeans(m[epi, , drop = FALSE])
emt_d <- data.table(cell_line = common_cells, emt = emt_score, NetITH = netith_vals)
emt_d <- emt_d[is.finite(emt) & is.finite(NetITH)]
st_emt <- spearman_test(emt_d$emt, emt_d$NetITH)
cat(sprintf("  [A] EMT (GDSC): rho=%.4f, p=%.2e, n=%d (cached: %.4f, %.2e, %d)\n",
            st_emt["rho"], st_emt["p"], st_emt["n"], res$emt_gdsc_rho, res$emt_gdsc_p, res$emt_gdsc_n))
stopifnot(abs(st_emt["rho"] - res$emt_gdsc_rho) < 0.01, st_emt["n"] == res$emt_gdsc_n)

pA <- ggplot(emt_d, aes(emt, NetITH)) +
  geom_point(colour = COL_SIG, alpha = 0.4, size = 1.2) +
  geom_smooth(method = "lm", se = FALSE, colour = COL_GDSC, linewidth = 0.5) +
  annotate("text", x = quantile(emt_d$emt, 0.97), y = quantile(emt_d$NetITH, 0.02),
           hjust = 1, size = FONT_BASE/.pt, colour = "grey40",
           label = sprintf("n=%d", st_emt["n"])) +
  labs(title = sprintf("GDSC: EMT vs NetITH\nρ=%.2f, p=%s", st_emt["rho"],
          ifelse(st_emt["p"] < 0.001, sprintf("%.1e", st_emt["p"]), sprintf("%.2f", st_emt["p"]))),
       x = "EMT Score (76-gene)", y = "NetITH") +
  theme_pub()

# ==============================================================================
#  Panel B: TCGA mRNAsi / EREG (Malta 2018) vs bulk NetITH
# ==============================================================================
cat("=== [EDFig7] loading TCGA mRNAsi (Xena stemness) ===\n")
stem_file <- file.path(DATA_ROOT, "xena", "tcgapancan", "StemnessScores_RNAexp_20170127.2.tsv.gz")
stopifnot(file.exists(stem_file))
# NOTE: file layout is features-as-rows x samples-as-columns (Xena); transpose
stem <- fread(stem_file)
feat_names <- stem[[1]]
samp_ids   <- names(stem)[-1]
m_stem <- as.matrix(stem[, -1, with = FALSE])        # features x samples
stopifnot(all(c("RNAss", "EREG.EXPss") %in% feat_names))
keep <- grepl("-01$", samp_ids)                       # primary tumours only
m_stem <- m_stem[, keep]; samp_ids <- samp_ids[keep]
patient <- substr(samp_ids, 1, 12)                    # first sample per patient
first <- !duplicated(patient)
m_stem <- m_stem[, first]; samp_ids <- samp_ids[first]
i_rna  <- which(feat_names == "RNAss")
i_ereg <- which(feat_names == "EREG.EXPss")
stem <- data.table(sample_id = samp_ids,
                   RNAss     = as.numeric(m_stem[i_rna, ]),
                   EREG.EXPss = as.numeric(m_stem[i_ereg, ]))
cat(sprintf("  mRNAsi primary tumours (deduped): %d\n", nrow(stem)))

tnet <- fread(file.path(TCGA_D, "tcga_netith.csv"))
setnames(tnet, 1, "sample_id")          # unnamed index col duplicates sample
md <- merge(stem[, .(sample_id, RNAss, EREG.EXPss)], tnet, by = "sample_id")
md <- md[is.finite(RNAss) & is.finite(EREG.EXPss) & is.finite(netith_bulk)]
st_m <- spearman_test(md$RNAss, md$netith_bulk)
st_e <- spearman_test(md$EREG.EXPss, md$netith_bulk)
cat(sprintf("  [B] mRNAsi: rho=%.4f, p=%.2e, n=%d (cached: %.4f, %.2e, %d)\n",
            st_m["rho"], st_m["p"], st_m["n"], res$mrnasi_rho, res$mrnasi_p, res$mrnasi_n))
cat(sprintf("  [B] EREG  : rho=%.4f, p=%.2e, n=%d (cached: %.4f, %.2e)\n",
            st_e["rho"], st_e["p"], st_e["n"], res$ereg_rho, res$ereg_p))
stopifnot(abs(st_m["rho"] - res$mrnasi_rho) < 0.01, st_m["n"] == res$mrnasi_n)

pB <- ggplot(md, aes(RNAss, netith_bulk)) +
  geom_point(colour = NPG_COLORS[7], alpha = 0.3, size = 0.8) +
  geom_smooth(method = "lm", se = FALSE, colour = COL_GDSC, linewidth = 0.5) +
  annotate("text", x = quantile(md$RNAss, 0.97), y = quantile(md$netith_bulk, 0.02),
           hjust = 1, size = FONT_BASE/.pt, colour = "grey40",
           label = sprintf("n=%d", st_m["n"])) +
  labs(title = sprintf("TCGA: Stemness vs NetITH\nρ=%.2f, p=%s", st_m["rho"],
          ifelse(st_m["p"] < 0.001, sprintf("%.1e", st_m["p"]), sprintf("%.2f", st_m["p"]))),
       x = "mRNAsi (Malta 2018)", y = "NetITH (bulk)") +
  theme_pub()

# ==============================================================================
#  Panel C: DTP gene enrichment volcano (per-gene NetITH correlations, GDSC)
# ==============================================================================
cat("=== [EDFig7] per-gene NetITH correlations (Spearman) ===\n")
rnk <- t(apply(m, 1, rank, ties.method = "average"))
n_cells <- ncol(m)
netith_r <- rank(netith_vals, ties.method = "average")
rho_g <- as.numeric(cor(t(rnk), netith_r))
tval <- rho_g * sqrt((n_cells - 2) / pmax(1 - rho_g^2, 1e-12))
pval <- 2 * pt(-abs(tval), df = n_cells - 2)
corr_df <- data.table(gene = rownames(m), rho = rho_g, pval = pval)
corr_df[, fdr := p.adjust(pval, method = "BH")]
rm(rnk); gc()

# top-10% |rho| gene set (primary DTP enrichment definition)
thr <- as.numeric(quantile(abs(corr_df$rho), 0.90))
top10 <- corr_df[abs(rho) >= thr, gene]
dtp_expr <- intersect(DTP_UPREGULATED, corr_df$gene)
a_ <- sum(dtp_expr %in% top10); b_ <- length(dtp_expr) - a_
c_ <- length(setdiff(top10, dtp_expr)); d_ <- nrow(corr_df) - a_ - b_ - c_
ft <- fisher.test(matrix(c(a_, b_, c_, d_), nrow = 2), alternative = "greater")
cat(sprintf("  [C] DTP-up in expr: %d | overlap with top10%%|rho|: %d\n", length(dtp_expr), a_))
cat(sprintf("  [C] Fisher exact (one-sided greater): OR=%.3f, p=%.4f (cached: OR=%.3f, p=%.4f, overlap=%d/%d)\n",
            ft$estimate, ft$p.value, res$dtp_odds_ratio, res$dtp_fisher_p,
            res$dtp_overlap_n, res$dtp_total_up))
stopifnot(abs(ft$p.value - res$dtp_fisher_p) < 0.05, a_ == res$dtp_overlap_n,
          length(dtp_expr) == res$dtp_total_up)

set.seed(SEED)   # project-wide seed (49) for the gray subsample
sample_n <- min(5000, nrow(corr_df))
idx <- sample(nrow(corr_df), sample_n)
vol <- rbind(
  corr_df[idx, .(rho, pval, grp = "All genes")],
  corr_df[gene %in% dtp_expr, .(rho, pval, grp = sprintf("DTP-up (n=%d)", length(dtp_expr)))])
vol[, neglogp := -log10(pmax(pval, 1e-300))]

pC <- ggplot(vol, aes(rho, neglogp, colour = grp)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.3) +
  geom_point(data = vol[grp == "All genes"], alpha = 0.3, size = 0.4) +
  geom_point(data = vol[grp != "All genes"], alpha = 0.85, size = 1.6) +
  scale_colour_manual(values = c("All genes" = COL_GREY,
                                 "DTP-up (n=49)" = NPG_COLORS[8],
                                 "DTP-up (n=50)" = NPG_COLORS[8])) +
  labs(title = sprintf("DTP Gene Enrichment Among NetITH-Correlated Targets\nFisher p=%.3f, OR=%.2f",
                       res$dtp_fisher_p, res$dtp_odds_ratio),
       x = "Spearman ρ (gene vs NetITH)", y = "-log10(p)") +
  theme_pub() + theme(legend.position = "top", legend.title = element_blank())

# ==============================================================================
#  Panel D: biological anchoring summary bars (cached statistics)
# ==============================================================================
bar_d <- data.table(
  label = c("EMT\n(GDSC)", "mRNAsi\n(TCGA)", "EREG\n(TCGA)"),
  rho   = c(res$emt_gdsc_rho, res$mrnasi_rho, res$ereg_rho),
  p     = c(res$emt_gdsc_p,   res$mrnasi_p,   res$ereg_p),
  col   = c(COL_SIG, NPG_COLORS[7], NPG_COLORS[8]))
bar_d[, sig := ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "ns")))]

pD <- ggplot(bar_d, aes(label, rho)) +
  geom_col(aes(fill = label), width = 0.6, alpha = 0.75, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = setNames(bar_d$col, bar_d$label), guide = "none") +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_text(aes(y = rho + ifelse(rho >= 0, 0.012, -0.045),
                label = sprintf("r=%.3f\n%s", rho, sig)),
            size = FONT_BASE/.pt, fontface = "bold") +
  labs(title = "Biological Anchoring Summary", x = NULL, y = "Spearman ρ with NetITH") +
  theme_pub()

# ==============================================================================
#  composite + save
# ==============================================================================
p <- wrap_plots(list(pA, pB, pC, pD), design = "AB\nCD") +
  plot_annotation(title = paste("NetITH Biological Anchoring: EMT, Stemness, and",
                                "Drug-Tolerant Persister Connections"),
                  theme = theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5)))
save_fig(p, NAME, height_mm = H_MM)
save_panels(list(pA, pB, pC, pD), NAME)

# ---- print key numbers ----------------------------------------------------------
cat("\n=== EDFig7 key numbers ===\n")
cat(sprintf("  EMT (GDSC):    rho=%.4f, p=%.2e, n=%d\n", st_emt["rho"], st_emt["p"], st_emt["n"]))
cat(sprintf("  mRNAsi (TCGA): rho=%.4f, p=%.2e, n=%d\n", st_m["rho"], st_m["p"], st_m["n"]))
cat(sprintf("  EREG.EXPss:    rho=%.4f, p=%.2e, n=%d\n", st_e["rho"], st_e["p"], st_e["n"]))
cat(sprintf("  DTP enrichment: Fisher p=%.4f, OR=%.2f, overlap=%d/%d\n",
            res$dtp_fisher_p, res$dtp_odds_ratio, res$dtp_overlap_n, res$dtp_total_up))
