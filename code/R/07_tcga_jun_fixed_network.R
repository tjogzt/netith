# ============================================================================
# 07_tcga_jun_fixed_network.R — JUN edge-weight contribution to NetITH (GDSC + TCGA) on the fixed construction.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : R2 recomputation (reviewer-requested, fixed-network): JUN transfer
#           analysis.
#           (1) GDSC side: reproduce the cached JUN-only R2 = 0.28 (Pearson^2
#               between JUN edge-weight contribution and NetITH, signed
#               weights, fixed 224-gene/613-edge construction) as validation
#               of the definition used in run_tf_combinatorial.py.
#           (2) TCGA side: recompute NetITH on the FIXED construction
#               (focused 239-gene set, signed CollecTRI edges within the
#               focused genes, w*|z|*|z'|, L = D - A, vN entropy), compute the
#               JUN edge-weight contribution with the identical definition,
#               and report R2 pan-cancer and per-cancer-type (median).
#           Context: the manuscript's "0.007" used per-cancer regression of
#           NetITH on JUN *expression* under the top-300 construction; this
#           script additionally reports that manuscript-style quantity on the
#           fixed construction for decomposition of the collapse.
# Inputs  : data/gdsc/rna_expr.csv, data/gdsc/cell_annot.csv,
#           data/gdsc/ensg_symbol_map.csv, data/collectri_network.csv,
#           results/focused_genes_collectri.txt,
#           results/gdsc/gdsc_netith_cell_lines.csv (cached GDSC NetITH),
#           $NETITH_DATA_DISK/data/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz,
#           $NETITH_DATA_DISK/data/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp
# Outputs : results/tcga/tcga_netith_fixed_network.csv,
#           results/tcga/tcga_jun_activity_fixed_network.csv,
#           results/tcga/jun_transfer_fixed_network_summary.json,
#           results/tcga/jun_transfer_fixed_network_recomputation.md
# Usage   : Rscript code/R/07_tcga_jun_fixed_network.R   (run from repo root)
# ============================================================================
suppressPackageStartupMessages({ library(data.table); library(Matrix) })

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
source(file.path(CODE_DIR, "02_netith_core.R"))

TCGA_XENA <- file.path(DATA_DISK, "data", "xena", "tcgapancan", "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz")
TCGA_SURV <- file.path(DATA_DISK, "data", "xena", "tcgapancan", "Survival_SupplementalTable_S1_20171025_xena_sp")
OUT_DIR <- file.path(RESULTS_DIR, "tcga")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# shared helper: per-gene z-scores across samples, clipped to [-3,3];
# NA/NaN expression values are set to z = 0 (matches run_tcga_validation.py's
# .fillna(0) and is equivalent to dropping the affected edge contributions)
zscore_clip <- function(mat) {
  z <- t(apply(mat, 1, function(g) {
    m <- mean(g, na.rm = TRUE); s <- sd(g, na.rm = TRUE) + 1e-10
    pmin(pmax((g - m) / s, -3), 3)
  }))
  z[is.na(z)] <- 0
  z
}
jun_activity <- function(zmat, jun_edges, n_samples) {
  # jun_edges: data.table with columns target_idx (row index into zmat), weight
  if (nrow(jun_edges) == 0L) return(rep(0, n_samples))
  jun_z <- zmat["JUN", ]
  out <- numeric(n_samples)
  for (k in seq_len(nrow(jun_edges))) {
    tgt <- jun_edges$target_idx[k]; w <- jun_edges$weight[k]
    out <- out + w * abs(jun_z) * abs(zmat[tgt, ])
  }
  out
}

net <- fread(file.path(DATA_DIR, "collectri_network.csv"))
net[, weight := as.numeric(weight)]
focused <- readLines(file.path(RESULTS_DIR, "focused_genes_collectri.txt"))
focused <- focused[nzchar(focused)]

cat("=== [1/6] GDSC side: reproduce cached JUN-only R2 = 0.28 ===\n")
expr_raw <- as.data.frame(fread(file.path(DATA_DIR, "gdsc", "rna_expr.csv"), header = TRUE))
rownames(expr_raw) <- as.character(expr_raw[[1]])
expr_raw <- expr_raw[, -1, drop = FALSE]
annot <- fread(file.path(DATA_DIR, "gdsc", "cell_annot.csv"))
gene_map <- fread(file.path(DATA_DIR, "gdsc", "ensg_symbol_map.csv"))
cel_map <- setNames(annot[["Characteristics.cell.line."]], annot[[1]])
cel_map <- cel_map[!is.na(cel_map) & nzchar(cel_map)]
common_cels <- intersect(colnames(expr_raw), names(cel_map))
expr_raw <- expr_raw[, common_cels, drop = FALSE]
colnames(expr_raw) <- unname(cel_map[colnames(expr_raw)])
expr_raw <- expr_raw[, !duplicated(colnames(expr_raw)), drop = FALSE]
ensg_to_sym <- setNames(gene_map$symbol, gene_map$ensg)
ensg_to_sym <- ensg_to_sym[!is.na(ensg_to_sym) & nzchar(ensg_to_sym)]
matched <- rownames(expr_raw) %in% names(ensg_to_sym)
expr_raw <- expr_raw[matched, , drop = FALSE]
new_sym <- unname(ensg_to_sym[rownames(expr_raw)])
keep <- !duplicated(new_sym)
expr_raw <- expr_raw[keep, , drop = FALSE]
rownames(expr_raw) <- new_sym[keep]
cat(sprintf("  GDSC expression: %d genes x %d cell lines\n", nrow(expr_raw), ncol(expr_raw)))

gdsc_cached <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
common_g <- intersect(focused, rownames(expr_raw))
cat(sprintf("  focused genes present in GDSC: %d/239\n", length(common_g)))
gdsc_mat <- as.matrix(expr_raw[common_g, ])
# NetITH on the fixed construction (signed) as an additional consistency check
gdsc_netith_r <- compute_netith(gdsc_mat, net, focused)
common_cl <- intersect(names(gdsc_netith_r), gdsc_cached$cell_line)
rho_g <- cor(gdsc_netith_r[common_cl], gdsc_cached$NetITH[match(common_cl, gdsc_cached$cell_line)],
             method = "spearman")
cat(sprintf("  R fixed-construction NetITH vs cached GDSC NetITH: Spearman = %.6f (n=%d)\n",
            rho_g, length(common_cl)))

gdsc_z <- zscore_clip(gdsc_mat)
gi <- setNames(seq_len(nrow(gdsc_mat)), rownames(gdsc_mat))
edges_g <- net[source %in% common_g & target %in% common_g]
edges_g[, tf_idx := gi[source]][, target_idx := gi[target]]
jun_edges_g <- edges_g[source == "JUN"]
cat(sprintf("  GDSC construction: %d genes, %d edges (%d negative), JUN out-edges: %d\n",
            length(common_g), nrow(edges_g), sum(edges_g$weight < 0), nrow(jun_edges_g)))
jun_act_g <- jun_activity(gdsc_z, jun_edges_g, ncol(gdsc_z))
names(jun_act_g) <- colnames(gdsc_z)
ok_g <- intersect(names(jun_act_g), gdsc_cached$cell_line)
jun_act_g <- jun_act_g[ok_g]
netith_g <- gdsc_netith_r[ok_g]
r2_jun_gdsc <- cor(jun_act_g, netith_g, method = "pearson")^2
cat(sprintf("  GDSC JUN-only R2 (activity, fixed construction) = %.4f (n=%d)\n", r2_jun_gdsc, length(ok_g)))

cat("=== [2/6] loading TCGA Xena expression ===\n")
t0 <- proc.time()
xena <- fread(TCGA_XENA, sep = "\t", header = TRUE, check.names = FALSE)
cat(sprintf("  Xena dims: %d genes x %d samples (load %.1f s)\n",
            nrow(xena), ncol(xena) - 1L, (proc.time() - t0)[["elapsed"]]))
gene_ids <- as.character(xena[[1]])
num_mask <- grepl("^[0-9.]+$", gene_ids)
xena <- xena[!num_mask, ]
gene_ids <- gene_ids[!num_mask]
xena <- xena[!duplicated(gene_ids), ]
gene_ids <- gene_ids[!duplicated(gene_ids)]
tcga_mat <- as.matrix(xena[, -1, with = FALSE])
rownames(tcga_mat) <- gene_ids
rm(xena); gc()
cat(sprintf("  TCGA symbol genes x samples: %d x %d\n", nrow(tcga_mat), ncol(tcga_mat)))

cat("=== [3/6] fixed-construction TCGA NetITH ===\n")
common_t <- intersect(focused, rownames(tcga_mat))
cat(sprintf("  focused genes present in TCGA: %d/239\n", length(common_t)))
tcga_mat_f <- tcga_mat[common_t, , drop = FALSE]
netith_t <- compute_netith(tcga_mat_f, net, focused)
cat(sprintf("  TCGA NetITH (fixed): n=%d, median=%.4f, range=[%.4f, %.4f]\n",
            length(netith_t), median(netith_t), min(netith_t), max(netith_t)))

cat("=== [4/6] TCGA JUN edge-weight contribution ===\n")
tcga_z <- zscore_clip(tcga_mat_f)
gi_t <- setNames(seq_len(nrow(tcga_mat_f)), rownames(tcga_mat_f))
edges_t <- net[source %in% common_t & target %in% common_t]
edges_t[, tf_idx := gi_t[source]][, target_idx := gi_t[target]]
jun_edges_t <- edges_t[source == "JUN"]
cat(sprintf("  TCGA construction: %d genes, %d edges (%d negative), JUN out-edges: %d\n",
            length(common_t), nrow(edges_t), sum(edges_t$weight < 0), nrow(jun_edges_t)))
jun_act_t <- jun_activity(tcga_z, jun_edges_t, ncol(tcga_z))
names(jun_act_t) <- colnames(tcga_z)

# keep all samples positionally (names are colnames of the same matrix, in the
# same order; duplicate TCGA barcodes are retained to mirror the manuscript's
# 11,069-sample set)
sample_names_t <- names(netith_t)
netith_tv <- unname(netith_t); jun_act_tv <- unname(jun_act_t)
r2_pan <- cor(jun_act_tv, netith_tv, method = "pearson")^2
rho_pan <- cor(jun_act_tv, netith_tv, method = "spearman")
cat(sprintf("  TCGA pan-cancer JUN-activity R2 (fixed construction) = %.4f (Spearman rho=%.4f, n=%d)\n",
            r2_pan, rho_pan, length(netith_tv)))

cat("=== [5/6] per-cancer JUN R2 (activity, fixed construction; + manuscript-style expression) ===\n")
surv_lines <- readLines(TCGA_SURV)
tab_i <- grep("\t", surv_lines, fixed = TRUE)
start <- tab_i[1]
hdr <- strsplit(surv_lines[start], "\t", fixed = TRUE)[[1]]
hdr[1] <- sub("^.*sample$", "sample", hdr[1])
surv <- fread(text = paste(surv_lines[start:length(surv_lines)], collapse = "\n"),
              sep = "\t", header = TRUE, check.names = FALSE)
names(surv)[1] <- "sample"
setnames(surv, "cancer type abbreviation", "cancer")
surv <- surv[, .(sample, cancer)]
samp2cancer <- setNames(surv$cancer, surv$sample)
ct_full <- samp2cancer[sample_names_t]
ok_s <- !is.na(ct_full)
ct <- ct_full[ok_s]

per_cancer <- function(score_y, pred_x, ct_vec, label) {
  res <- data.table(cancer = names(table(ct_vec)), n = as.integer(table(ct_vec)), r2 = NA_real_)
  for (cn in res$cancer) {
    idx <- which(ct_vec == cn)
    if (length(idx) < 30) next
    r2v <- cor(pred_x[idx], score_y[idx], method = "pearson")^2
    res[cancer == cn, r2 := r2v]
  }
  res <- res[!is.na(r2)]
  med <- median(res$r2)
  cat(sprintf("  %s: %d cancers >=30 samples; median per-cancer R2 = %.4f (min %.4f, max %.4f)\n",
              label, nrow(res), med, min(res$r2), max(res$r2)))
  list(tbl = res, median = med)
}
pc_act <- per_cancer(netith_tv[ok_s], jun_act_tv[ok_s], ct, "JUN-activity R2 (fixed construction)")
# manuscript-style: regress NetITH on JUN raw expression (linear regression, in-sample R2)
jun_expr_t <- tcga_mat["JUN", sample_names_t[ok_s]]
pc_expr <- per_cancer(netith_tv[ok_s], jun_expr_t, ct, "JUN-expression R2 (manuscript style, fixed construction)")

cat("=== [6/6] saving outputs ===\n")
out_df <- data.table(sample = sample_names_t, netith_fixed = unname(netith_tv),
                     jun_activity = unname(jun_act_tv))
fwrite(out_df, file.path(OUT_DIR, "tcga_netith_fixed_network.csv"))
fwrite(data.table(sample = sample_names_t, jun_activity = unname(jun_act_tv)),
       file.path(OUT_DIR, "tcga_jun_activity_fixed_network.csv"))

summary_json <- list(
  gdsc_side = list(
    n_genes = length(common_g), n_edges = nrow(edges_g),
    n_neg_edges = sum(edges_g$weight < 0), n_jun_out_edges = nrow(jun_edges_g),
    n_cell_lines = length(ok_g),
    r2_jun_activity_vs_cached_netith = r2_jun_gdsc,
    spearman_fixed_netith_vs_cached = rho_g,
    cached_manuscript_r2_jun = 0.279744427497574),
  tcga_side = list(
    n_genes = length(common_t), n_edges = nrow(edges_t),
    n_neg_edges = sum(edges_t$weight < 0), n_jun_out_edges = nrow(jun_edges_t),
    n_samples = length(netith_tv),
    r2_jun_activity_pancancer = r2_pan,
    spearman_jun_activity_pancancer = rho_pan,
    median_per_cancer_r2_activity = pc_act$median,
    n_cancers_activity = nrow(pc_act$tbl),
    median_per_cancer_r2_expression_manuscript_style = pc_expr$median,
    n_cancers_expression = nrow(pc_expr$tbl),
    manuscript_reported_tcga = 0.007)
)
jsonlite::write_json(summary_json, file.path(OUT_DIR, "jun_transfer_fixed_network_summary.json"),
                     auto_unbox = TRUE, digits = 8)
fwrite(pc_act$tbl, file.path(OUT_DIR, "tcga_jun_activity_r2_per_cancer_fixed.csv"))
fwrite(pc_expr$tbl, file.path(OUT_DIR, "tcga_jun_expression_r2_per_cancer_fixed.csv"))
cat("JSON summary written.\n")
cat("\n===== KEY VALUES =====\n")
cat(sprintf("GDSC: genes=%d edges=%d jun_edges=%d n=%d r2=%.4f netith_rho=%.6f\n",
            length(common_g), nrow(edges_g), nrow(jun_edges_g), length(ok_g),
            r2_jun_gdsc, rho_g))
cat(sprintf("TCGA: genes=%d edges=%d jun_edges=%d n=%d pan_r2=%.4f spearman=%.4f med_percancer_act=%.4f med_percancer_expr=%.4f\n",
            length(common_t), nrow(edges_t), nrow(jun_edges_t), length(netith_tv),
            r2_pan, rho_pan, pc_act$median, pc_expr$median))
cat("===== END KEY VALUES =====\n")
