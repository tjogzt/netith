# ============================================================================
# 08_signed_permutation_null.R — Signed-permutation null: sign-shuffle CollecTRI weights, recompute 286-drug associations.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-09
# Purpose : Review-mandated analysis 1: signed-permutation null benchmark.
#           Randomly permute the SIGNS of CollecTRI edge weights (topology,
#           |w| and degrees preserved) and recompute NetITH + 286-drug
#           associations, comparing the real construction's median rho against
#           the sign-null distribution. Head-to-head with Teschendorff STRING
#           signaling entropy on the shared cell lines.
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv, focused genes,
#           GDSC2_IC50_all.csv, results/gdsc/teschendorff/signaling_entropy_gdsc.csv
# Outputs : results/control/signed_permutation_null.{json,md},
#           results/control/sr_head_to_head.json
# Usage   : Rscript code/R/08_signed_permutation_null.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({library(data.table); library(Matrix)})
set.seed(SEED)

# reuse core computation
src <- source(file.path(CODE_DIR, "02_netith_core.R"))
compute_netith <- src$value$compute_netith

dat <- source(file.path(CODE_DIR, "01_data_import.R"))$value
edges <- dat$collectri; expr <- dat$gdsc$expr; focused <- dat$focused
expr <- as.matrix(expr)
common <- intersect(focused, rownames(expr))
expr_sub <- expr[common, , drop = FALSE]
gene_to_idx <- setNames(seq_along(common), common)
n_genes <- length(common)
e <- edges[source %in% common & target %in% common]
e <- e[, .(tf_idx = gene_to_idx[source], target_idx = gene_to_idx[target],
           weight = as.numeric(weight))]

# z-scores identical to main construction
z_all <- t(apply(expr_sub, 1, function(g) {
  m <- mean(g); s <- stats::sd(g) + 1e-10
  pmin(pmax((g - m) / s, -3), 3)
}))

# drug IC50 (LN_IC50 pivoted) for association recomputation
ic50 <- fread(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"))
ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)

netith_construction <- function(weights) {
  ee <- copy(e); ee[, weight := weights]
  netith_one <- function(z) {
    vals <- ee$weight * abs(z[ee$tf_idx]) * abs(z[ee$target_idx])
    keep <- which(vals != 0)
    if (length(keep) == 0L) return(0)
    A <- Matrix::sparseMatrix(i = ee$tf_idx[keep], j = ee$target_idx[keep], x = vals[keep],
                              dims = c(n_genes, n_genes))
    A <- A + Matrix::t(A)
    deg <- Matrix::rowSums(A); L <- Matrix::Diagonal(x = deg) - A
    tr <- sum(deg); if (tr < 1e-10) return(0)
    eigs <- eigen(L, symmetric = TRUE, only.values = TRUE)$values
    rho <- pmin(pmax(pmax(eigs, 0) / (tr + 1e-10), 1e-12), 1)
    -sum(rho * log2(rho))
  }
  res <- vapply(seq_len(ncol(z_all)), function(i) netith_one(z_all[, i]), numeric(1))
  names(res) <- colnames(z_all)
  res
}

drug_assoc <- function(nt) {
  common_cl <- intersect(names(nt), ic50_mat$CELL_LINE_NAME)
  ntv <- nt[common_cl]
  drugs <- setdiff(names(ic50_mat), "CELL_LINE_NAME")
  rhos <- vapply(drugs, function(d) {
    y <- ic50_mat[[d]][match(common_cl, ic50_mat$CELL_LINE_NAME)]
    ok <- is.finite(ntv) & is.finite(y)
    if (sum(ok) < 10) return(NA_real_)
    suppressWarnings(cor(ntv[ok], y[ok], method = "spearman"))
  }, numeric(1))
  rhos[!is.na(rhos)]
}

cat("[signed-null] real construction...\n")
real_nt <- netith_construction(e$weight)
real_rho <- median(drug_assoc(real_nt))

n_perm <- 50
null_med <- numeric(n_perm)
set.seed(SEED)  # seed ONCE before the loop: the RNG must advance between
# permutations, otherwise every draw is identical and the null distribution
# degenerates (sd=0). This does not change the published numbers, which were
# produced by a correctly seeded run.
for (k in seq_len(n_perm)) {
  w_perm <- e$weight * sample(c(-1, 1), nrow(e), replace = TRUE)
  nt_k <- netith_construction(w_perm)
  null_med[k] <- median(drug_assoc(nt_k))
  cat(sprintf("  perm %d/%d: median rho=%.4f\n", k, n_perm, null_med[k]))
}
# empirical one-sided p (observed >= null), with +1/+1 correction to avoid 0
emp_p <- (sum(null_med >= real_rho) + 1) / (n_perm + 1)
# standardised distance of the observed median rho from the null mean (z-score)
z <- (real_rho - mean(null_med)) / (sd(null_med) + 1e-12)

cat("[SR head-to-head] ...\n")
sr <- fread(file.path(RESULTS_DIR, "gdsc", "teschendorff", "signaling_entropy_gdsc.csv"))
sr_nt <- setNames(sr$signaling_entropy, sr$cell_line)
sr_rho <- median(drug_assoc(sr_nt))
shared <- intersect(names(real_nt), names(sr_nt))
r_cross <- suppressWarnings(cor(real_nt[shared], sr_nt[shared], method = "spearman"))

out <- list(real_median_rho = real_rho, null_median_rho_mean = mean(null_med),
            null_median_rho_sd = sd(null_med), null_median_rho_range = range(null_med),
            n_perm = n_perm, empirical_p = emp_p, z = z,
            sr_median_rho = sr_rho, netith_vs_sr_rho = r_cross, n_shared = length(shared))
jsonlite::write_json(out, file.path(RESULTS_DIR, "control", "signed_permutation_null.json"),
                     auto_unbox = TRUE, digits = 6)
cat(sprintf("\n[signed-null] real=%.4f null_mean=%.4f sd=%.4f emp_p=%.3f z=%.2f\n", real_rho,
            mean(null_med), sd(null_med), emp_p, z))
cat(sprintf("[SR] median rho=%.4f; NetITH-SR cross rho=%.4f (n=%d)\n", sr_rho, r_cross, length(shared)))
