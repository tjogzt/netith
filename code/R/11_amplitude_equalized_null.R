# ============================================================================
# 11_amplitude_equalized_null.R — Amplitude-equalized null: rank-equalize |z| amplitudes, configuration-model rewire.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-09
# Purpose : E2 amplitude-matched null (four-team round-4 gatekeeper).
#           Quantify how much of NetITH's drug association is carried by the
#           node-amplitude channel (per-sample gene magnitudes) versus the
#           wiring/sign channel. NetITH is recomputed with per-sample
#           rank-equalized node amplitudes (|z| replaced by rank(|z|)/n_genes),
#           so that within each sample every node carries identical marginal
#           amplitude information; only the curated wiring and its signs
#           remain. Complemented by 100 configuration-model rewirings of the
#           equalized graph (degree sequence preserved, signs shuffled) to
#           test whether the specific wiring matters in the amplitude-free
#           regime. Compared against the full construction (median rho 0.232)
#           and the focused-MAD amplitude-only baseline (median rho 0.274).
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv, focused genes,
#           GDSC2_IC50_all.csv
# Outputs : results/control/amplitude_equalized_null.{json,md},
#           results/control/amplitude_equalized_null_drugs.csv
# Usage   : Rscript code/R/11_amplitude_equalized_null.R
# ============================================================================
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({library(data.table); library(Matrix); library(jsonlite)})
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
n_edges <- nrow(e)
cat(sprintf("[E2] genes=%d edges=%d cells=%d\n", n_genes, n_edges, ncol(expr_sub)))

# z-scores identical to main construction (per gene across cells, clip +-3)
z_all <- t(apply(expr_sub, 1, function(g) {
  m <- mean(g); s <- stats::sd(g) + 1e-10
  pmin(pmax((g - m) / s, -3), 3)
}))

# per-sample rank-equalized amplitudes: replace |z| by rank(|z|)/n_genes
# (ties -> average rank). Removes all cross-node amplitude information.
amp_full <- abs(z_all)
amp_eq <- apply(amp_full, 2, function(a) rank(a, ties.method = "average") / n_genes)

# drug IC50 (LN_IC50 pivoted) for association recomputation
ic50 <- fread(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"))
ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)

# NetITH under a given amplitude matrix (rows = genes, cols = cells)
netith_with_amplitudes <- function(amp, weights, tf_idx, tgt_idx) {
  n_cells <- ncol(amp)
  res <- vapply(seq_len(n_cells), function(i) {
    vals <- weights * amp[tf_idx, i] * amp[tgt_idx, i]
    keep <- which(vals != 0)
    if (length(keep) == 0L) return(0)
    A <- Matrix::sparseMatrix(i = tf_idx[keep], j = tgt_idx[keep], x = vals[keep],
                              dims = c(n_genes, n_genes))
    A <- A + Matrix::t(A)
    deg <- Matrix::rowSums(A); L <- Matrix::Diagonal(x = deg) - A
    tr <- sum(deg); if (tr < 1e-10) return(0)
    eigs <- eigen(L, symmetric = TRUE, only.values = TRUE)$values
    rho <- pmin(pmax(pmax(eigs, 0) / (tr + 1e-10), 1e-12), 1)
    -sum(rho * log2(rho))
  }, numeric(1))
  names(res) <- colnames(amp)
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

# configuration-model rewire: exact in/out degree sequence, signs shuffled,
# self-loops rejected (mirrors scripts/02_controls/run_control_random_graph_null.py)
rewire_edges <- function(tf_idx, tgt_idx, weights, rng = NULL) {
  n <- max(c(tf_idx, tgt_idx)); m <- length(tf_idx)
  out_deg <- tabulate(tf_idx, nbins = n); in_deg <- tabulate(tgt_idx, nbins = n)
  out_stubs <- rep(seq_len(n), out_deg); in_stubs <- rep(seq_len(n), in_deg)
  ok <- FALSE
  for (attempt in seq_len(200)) {
    if (!is.null(rng)) in_stubs <- rng(in_stubs) else in_stubs <- sample(in_stubs)
    bad <- which(out_stubs == in_stubs)
    iter <- 0
    while (length(bad) > 0 && iter < 2000) {
      i <- bad[1]; j <- sample.int(m, 1)
      if (i == j || in_stubs[j] == out_stubs[i] || in_stubs[i] == out_stubs[j]) {
        iter <- iter + 1; next
      }
      tmp <- in_stubs[i]; in_stubs[i] <- in_stubs[j]; in_stubs[j] <- tmp
      bad <- which(out_stubs == in_stubs); iter <- iter + 1
    }
    if (length(bad) == 0) { ok <- TRUE; break }
  }
  if (!ok) stop("configuration model failed")
  list(tf = out_stubs, tgt = in_stubs, w = sample(weights))
}

cat("[E2] full construction (real |z|, signed weights)...\n")
t0 <- proc.time()
full_nt <- netith_with_amplitudes(amp_full, e$weight, e$tf_idx, e$target_idx)
full_rho <- drug_assoc(full_nt)
cat(sprintf("  full: median rho=%.4f, pos=%d/%d (%.1fs)\n",
            median(full_rho), sum(full_rho > 0), length(full_rho),
            (proc.time() - t0)[3]))

cat("[E2] equalized construction (rank amplitudes, signed weights)...\n")
t0 <- proc.time()
eq_nt <- netith_with_amplitudes(amp_eq, e$weight, e$tf_idx, e$target_idx)
eq_rho <- drug_assoc(eq_nt)
cat(sprintf("  equalized: median rho=%.4f, pos=%d/%d (%.1fs)\n",
            median(eq_rho), sum(eq_rho > 0), length(eq_rho),
            (proc.time() - t0)[3]))

# per-sample focused-gene MAD (amplitude-only reference, same cells)
mad_per_cell <- apply(z_all, 2, function(z) stats::median(abs(z - stats::median(z))))
names(mad_per_cell) <- colnames(z_all)

sh <- intersect(names(full_nt), names(eq_nt))
r_full_eq <- suppressWarnings(cor(full_nt[sh], eq_nt[sh], method = "spearman"))
sh2 <- intersect(names(eq_nt), names(mad_per_cell))
r_eq_mad <- suppressWarnings(cor(eq_nt[sh2], mad_per_cell[sh2], method = "spearman"))
r_full_mad <- suppressWarnings(cor(full_nt[sh2], mad_per_cell[sh2], method = "spearman"))

# 100 configuration-model rewirings of the equalized graph
# (seed ONCE before the loop: the RNG must advance between draws, otherwise
# every rewire is identical and the null distribution degenerates)
N_REWIRE <- 100L
null_med <- numeric(N_REWIRE)
set.seed(SEED)
for (k in seq_len(N_REWIRE)) {
  rw <- rewire_edges(e$tf_idx, e$target_idx, e$weight)
  nt_k <- netith_with_amplitudes(amp_eq, rw$w, rw$tf, rw$tgt)
  null_med[k] <- median(drug_assoc(nt_k))
  if (k %% 10 == 0) cat(sprintf("  rewire %d/%d: median rho=%.4f\n", k, N_REWIRE, null_med[k]))
}
# empirical one-sided p: how often a rewired null median equals/exceeds the
# equalized construction's median (observed), +1/+1 correction to avoid 0
emp_p <- (sum(null_med >= median(eq_rho)) + 1) / (N_REWIRE + 1)

out <- list(
  full_median_rho = unname(median(full_rho)),
  full_n_pos = unname(sum(full_rho > 0)),
  full_n_drugs = length(full_rho),
  equalized_median_rho = unname(median(eq_rho)),
  equalized_n_pos = unname(sum(eq_rho > 0)),
  equalized_n_drugs = length(eq_rho),
  rewire_null_mean = mean(null_med),
  rewire_null_sd = sd(null_med),
  rewire_null_range = range(null_med),
  rewire_empirical_p = emp_p,
  n_rewire = N_REWIRE,
  cor_equalized_vs_full = unname(r_full_eq),
  cor_equalized_vs_mad = unname(r_eq_mad),
  cor_full_vs_mad = unname(r_full_mad),
  mad_median_rho_reference = 0.2736,   # results/control/focused_mad_baseline.json
  n_genes = n_genes, n_edges = n_edges, n_cells = ncol(expr_sub)
)
write_json(out, file.path(RESULTS_DIR, "control", "amplitude_equalized_null.json"),
           auto_unbox = TRUE, digits = 6)

# per-drug CSV (full vs equalized)
drugs <- names(full_rho)
write.csv(data.frame(drug = drugs,
                     rho_full = as.numeric(full_rho[drugs]),
                     rho_equalized = as.numeric(eq_rho[drugs])),
          file.path(RESULTS_DIR, "control", "amplitude_equalized_null_drugs.csv"),
          row.names = FALSE)

cat(sprintf("\n[E2] full=%.4f | equalized=%.4f | rewire-null=%.4f (sd %.4f) | emp_p=%.3f\n",
            median(full_rho), median(eq_rho), mean(null_med), sd(null_med), emp_p))
cat(sprintf("[E2] cor(eq,full)=%.3f cor(eq,MAD)=%.3f cor(full,MAD)=%.3f\n",
            r_full_eq, r_eq_mad, r_full_mad))
