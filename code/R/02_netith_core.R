# ============================================================================
# 02_netith_core.R — Core NetITH computation: von Neumann entropy of the graph Laplacian from CollecTRI.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-08-19
# Purpose : Core NetITH computation in R: von Neumann entropy of the
#           non-normalised graph Laplacian L = D - A built from the CollecTRI
#           TF-target network weighted by per-sample expression z-scores.
#           Mirrors the validated Python implementation exactly
#           (scripts/01_core/run_gdsc_drug_sensitivity.py::compute_netith).
# Inputs  : expression matrix (genes x samples), CollecTRI edges
#           (source, target, weight), focused gene list
# Outputs : named numeric vector of NetITH scores per sample
# Usage   : scores <- compute_netith(expr, edges, focused)
# Validation: spearman consistency against results/gdsc/gdsc_netith_cell_lines.csv
# ============================================================================

# locate project root + load config
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))

suppressPackageStartupMessages(library(Matrix))

# ---- core entropy for one sample -------------------------------------------
netith_one <- function(z, edges, n_genes) {
  # z: numeric vector of clipped z-scores for focused genes (names = gene)
  A <- Matrix::sparseMatrix(i = integer(0), j = integer(0),
                            dims = c(n_genes, n_genes))
  # build weighted adjacency from edges
  tf_idx <- edges$tf_idx
  tgt_idx <- edges$target_idx
  w <- edges$weight
  za <- abs(z[tf_idx]) * abs(z[tgt_idx])
  vals <- w * za
  keep <- which(vals != 0)
  if (length(keep) == 0L) return(0)
  A <- Matrix::sparseMatrix(i = tf_idx[keep], j = tgt_idx[keep], x = vals[keep],
                            dims = c(n_genes, n_genes))
  A <- A + Matrix::t(A)                       # symmetrise (eigvalsh audit fix)
  deg <- Matrix::rowSums(A)
  L <- Matrix::Diagonal(x = deg) - A          # non-normalised Laplacian
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

# ---- full computation --------------------------------------------------------
compute_netith <- function(expr, edges_df, focused) {
  # expr: data.frame/matrix, genes in rows, samples in columns
  stopifnot(is.matrix(expr) || is.data.frame(expr),
            all(c("source", "target", "weight") %in% names(edges_df)))
  expr <- as.matrix(expr)
  common <- intersect(focused, rownames(expr))
  if (length(common) < 10) stop("Too few focused genes present in expression matrix")
  expr_sub <- expr[common, , drop = FALSE]
  gene_to_idx <- setNames(seq_along(common), common)
  n_genes <- length(common)
  n_samples <- ncol(expr_sub)

  # map edges onto the focused-gene index space
  edges <- edges_df[source %in% common & target %in% common]
  edges <- edges[, .(tf_idx = gene_to_idx[source],
                     target_idx = gene_to_idx[target],
                     weight = as.numeric(weight))]
  message(sprintf("[netith] %d focused genes, %d edges, %d samples",
                  n_genes, nrow(edges), n_samples))
  stopifnot(nrow(edges) > 0)

  # per-sample z-scores (clipped to [-3, 3]), matching Python row-wise scaling
  z_all <- t(apply(expr_sub, 1, function(g) {
    m <- mean(g); s <- stats::sd(g) + 1e-10
    pmin(pmax((g - m) / s, -3), 3)
  }))

  scores <- vapply(seq_len(n_samples), function(i) {
    netith_one(z_all[, i], edges, n_genes)
  }, numeric(1))
  names(scores) <- colnames(expr_sub)
  scores
}

# ---- consistency check against the cached Python output ----------------------
check_consistency <- function(scores_r, cached_csv) {
  cached <- data.table::fread(cached_csv)
  common <- intersect(names(scores_r), cached$cell_line)
  stopifnot(length(common) > 10)
  r <- stats::cor(scores_r[common], cached$NetITH[match(common, cached$cell_line)],
                  method = "spearman")
  cat(sprintf("[check] R vs Python NetITH: n=%d, Spearman rho=%.6f\n",
              length(common), r))
  invisible(r)
}

main <- function() {
  library(data.table)
  dat <- source(file.path(CODE_DIR, "01_data_import.R"))$value
  edges <- dat$collectri
  expr  <- dat$gdsc$expr
  focused <- dat$focused
  scores <- compute_netith(expr, edges, focused)
  out <- data.table(cell_line = names(scores), NetITH = unname(scores))
  data.table::fwrite(out, file.path(RESULTS_DIR, "gdsc", "netith_r_cell_lines.csv"))
  check_consistency(scores, file.path(RESULTS_DIR, "gdsc", "gdsc_netith_cell_lines.csv"))
  out
}

if (sys.nframe() == 0L) main()
