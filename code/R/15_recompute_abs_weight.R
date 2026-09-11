# ============================================================================
# 15_recompute_abs_weight.R — |w| vs signed-w NetITH recomputation (GDSC) with 286-drug correlations.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : R1 recomputation: NetITH with |w| (absolute CollecTRI weight)
#           construction vs the signed-w construction used by the validated
#           pipeline. Computes (1) GDSC |w| NetITH for all cell lines,
#           (2) the 286-drug sensitivity correlations under |w|, and
#           (3) summary statistics feeding the abs_vs_signed comparison table.
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv,
#           data/gdsc/cell_annot.csv, data/gdsc/ensg_symbol_map.csv,
#           results/focused_genes_collectri.txt,
#           results/gdsc/gdsc_netith_cell_lines.csv (cached Python signed),
#           results/gdsc/gdsc_drug_netith_correlations.csv (cached Python signed)
# Outputs : results/gdsc/netith_r_abs_weight.csv,
#           results/gdsc/gdsc_drug_netith_correlations_abs_weight.csv
# Usage   : Rscript code/R/15_recompute_abs_weight.R
# Note    : Single deliberate deviation from 02_netith_core.R: edge weights
#           take absolute value (vals <- abs(w) * za instead of w * za).
# Note    : ONE-OFF audit script (R1 recomputation for the abs-vs-signed review). Kept for
#           audit traceability; not part of the run_all chain. Prefer 02/03/04 for
#           reproducible analysis.
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
dat <- source(file.path(CODE_DIR, "01_data_import.R"))$value  # list() of imported tables

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
})

# ---- per-sample entropy with a weight-transform function --------------------
# wfun: function mapping CollecTRI weight vector -> effective weight vector.
#   signed: wfun = identity        (validated pipeline, matches cache)
#   abs   : wfun = abs             (manuscript formula |w|)
netith_one <- function(z, edges, n_genes, wfun) {
  A <- Matrix::sparseMatrix(i = integer(0), j = integer(0),
                            dims = c(n_genes, n_genes))
  tf_idx <- edges$tf_idx
  tgt_idx <- edges$target_idx
  w <- wfun(edges$weight)
  za <- abs(z[tf_idx]) * abs(z[tgt_idx])
  vals <- w * za
  keep <- which(vals != 0)
  if (length(keep) == 0L) return(0)
  A <- Matrix::sparseMatrix(i = tf_idx[keep], j = tgt_idx[keep], x = vals[keep],
                            dims = c(n_genes, n_genes))
  A <- A + Matrix::t(A)
  deg <- Matrix::rowSums(A)
  L <- Matrix::Diagonal(x = deg) - A
  trace <- sum(deg)
  if (trace < 1e-10) return(0)
  eigs <- eigen(L, symmetric = TRUE, only.values = TRUE)$values
  # Laplacian eigenvalues are >= 0; clip tiny negative round-off to 0
  eigs <- pmax(eigs, 0)
  # normalize by trace (= sum of eigenvalues) so rho is a probability vector
  rho <- eigs / (trace + 1e-10)
  # clip to [1e-12, 1] so log2 is finite; -sum(rho*log2(rho)) is vN entropy
  rho <- pmin(pmax(rho, 1e-12), 1.0)
  -sum(rho * log2(rho))
}

compute_netith_w <- function(expr, edges_df, focused, wfun = identity) {
  stopifnot(is.matrix(expr) || is.data.frame(expr),
            all(c("source", "target", "weight") %in% names(edges_df)))
  expr <- as.matrix(expr)
  common <- intersect(focused, rownames(expr))
  if (length(common) < 10) stop("Too few focused genes present in expression matrix")
  expr_sub <- expr[common, , drop = FALSE]
  gene_to_idx <- setNames(seq_along(common), common)
  n_genes <- length(common)
  n_samples <- ncol(expr_sub)

  edges <- edges_df[source %in% common & target %in% common]
  edges <- edges[, .(tf_idx = gene_to_idx[source],
                     target_idx = gene_to_idx[target],
                     weight = as.numeric(weight))]
  message(sprintf("[netith] %d focused genes, %d edges, %d samples",
                  n_genes, nrow(edges), n_samples))
  stopifnot(nrow(edges) > 0)

  z_all <- t(apply(expr_sub, 1, function(g) {
    m <- mean(g); s <- stats::sd(g) + 1e-10
    pmin(pmax((g - m) / s, -3), 3)
  }))

  scores <- vapply(seq_len(n_samples), function(i) {
    netith_one(z_all[, i], edges, n_genes, wfun)
  }, numeric(1))
  names(scores) <- colnames(expr_sub)
  scores
}

# ---- drug sensitivity correlation (mirrors 03_gdsc_drug_sensitivity.R) ------
drug_correlations <- function(netith_vec, netith_names) {
  find_ic50 <- function() {
    cand <- c(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"),
              file.path(DATA_DIR, "gdsc", "GDSC2_IC50_all.csv"))
    hit <- cand[file.exists(cand)]
    if (length(hit) == 0L) stop("GDSC2_IC50_all.csv not found")
    hit[1]
  }
  ic50 <- fread(find_ic50())
  stopifnot(all(c("CELL_LINE_NAME", "DRUG_NAME", "LN_IC50", "CANCER_TYPE") %in% names(ic50)))
  ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)
  common_cl <- intersect(netith_names, ic50_mat$CELL_LINE_NAME)
  stopifnot(length(common_cl) > 50)
  nt <- netith_vec[match(common_cl, netith_names)]
  drugs <- setdiff(names(ic50_mat), "CELL_LINE_NAME")
  res <- lapply(drugs, function(d) {
    y <- ic50_mat[[d]][match(common_cl, ic50_mat$CELL_LINE_NAME)]
    st <- spearman_test(nt, y)
    data.table(drug = d, n_cells = st[["n"]], rho = st[["rho"]],
               p_spearman = st[["p"]])
  })
  res <- rbindlist(res)
  res[, fdr := p.adjust(p_spearman, method = "BH")]
  setorder(res, p_spearman)
  res[]
}

# ============================================================================
main <- function() {
  edges   <- dat$collectri
  expr    <- dat$gdsc$expr
  focused <- dat$focused

  # ---- 1) GDSC NetITH under both constructions ------------------------------
  nt_signed <- compute_netith_w(expr, edges, focused, wfun = identity)
  nt_abs    <- compute_netith_w(expr, edges, focused, wfun = abs)

  # consistency of our signed recomputation vs cached Python signed
  cached <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
  common <- intersect(names(nt_signed), cached$cell_line)
  r_check <- cor(nt_signed[common], cached$NetITH[match(common, cached$cell_line)],
                 method = "spearman")
  r_cross <- cor(nt_abs[common], nt_signed[common], method = "spearman")
  cat(sprintf("[abs] signed R vs Python cache rho=%.6f (n=%d)\n", r_check, length(common)))
  cat(sprintf("[abs] |w| vs signed NetITH rho=%.6f (n=%d)\n", r_cross, length(common)))

  out <- data.table(cell_line = names(nt_abs), NetITH = unname(nt_abs))
  fwrite(out, file.path(RESULTS_DIR, "gdsc", "netith_r_abs_weight.csv"))

  # ---- 2) 286-drug correlations under |w| -----------------------------------
  drugs_abs <- drug_correlations(nt_abs, names(nt_abs))
  fwrite(drugs_abs, file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations_abs_weight.csv"))

  # ---- 3) comparison statistics ---------------------------------------------
  drugs_signed <- fread(file.path(RESULTS_DIR, "gdsc", "gdsc_drug_netith_correlations.csv"))
  setorder(drugs_signed, p_spearman)
  m <- merge(drugs_abs[, .(drug, rho_abs = rho, fdr_abs = fdr)],
             drugs_signed[, .(drug, rho_signed = rho, fdr_signed = fdr)], by = "drug")
  cat(sprintf("[abs] drugs: %d (%d merged)\n", nrow(drugs_abs), nrow(m)))
  cat(sprintf("[abs] 286-drug median rho |w|=%.4f signed=%.4f\n",
              median(m$rho_abs), median(m$rho_signed)))
  cat(sprintf("[abs] rho>0: |w|=%d signed=%d; FDR<0.05: |w|=%d signed=%d\n",
              sum(m$rho_abs > 0), sum(m$rho_signed > 0),
              sum(m$fdr_abs < 0.05), sum(m$fdr_signed < 0.05)))

  # distribution stats for the comparison table
  cat(sprintf("[abs] NetITH median |w|=%.4f signed=%.4f\n",
              median(nt_abs), median(nt_signed)))
  cat(sprintf("[abs] NetITH IQR  |w|=[%.4f, %.4f] signed=[%.4f, %.4f]\n",
              quantile(nt_abs, 0.25), quantile(nt_abs, 0.75),
              quantile(nt_signed, 0.25), quantile(nt_signed, 0.75)))
  cat(sprintf("[abs] NetITH range |w|=[%.4f, %.4f] signed=[%.4f, %.4f]\n",
              min(nt_abs), max(nt_abs), min(nt_signed), max(nt_signed)))

  invisible(list(nt_abs = nt_abs, nt_signed = nt_signed,
                 drugs_abs = drugs_abs, drugs_signed = drugs_signed,
                 r_cross = r_cross))
}

main()
