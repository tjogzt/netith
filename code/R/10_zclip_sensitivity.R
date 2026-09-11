# ============================================================================
# 10_zclip_sensitivity.R — z-clip threshold sensitivity of the GDSC drug association (2/3/4/none).
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-06
# Purpose : z-clip sensitivity of the GDSC drug association (review-batch-A
#           item 12): recompute NetITH with z-clip thresholds +/-2, +/-3
#           (reference), +/-4 and no clipping, then recompute the 286-drug
#           Spearman correlations with ln(IC50). Reports median rho, count of
#           rho>0 and BH FDR<0.05 per threshold.
# Inputs  : data/collectri_network.csv, data/gdsc/rna_expr.csv,
#           data/gdsc/cell_annot.csv, data/gdsc/ensg_symbol_map.csv,
#           results/focused_genes_collectri.txt,
#           GDSC2_IC50_all.csv (data disk $NETITH_DATA_DISK/data/gdsc_download)
# Outputs : results/control/zclip_sensitivity.json
#           results/control/zclip_sensitivity_median_rho.csv
# Usage   : Rscript code/R/10_zclip_sensitivity.R
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(Matrix); library(jsonlite) })

dat <- source(file.path(CODE_DIR, "01_data_import.R"))$value
edges_all <- dat$collectri
expr <- dat$gdsc$expr
focused <- dat$focused

# ---- compute_netith with a configurable z-clip threshold --------------------
compute_netith_clip <- function(expr, edges_df, focused, clip) {
  expr <- as.matrix(expr)
  common <- intersect(focused, rownames(expr))
  expr_sub <- expr[common, , drop = FALSE]
  gene_to_idx <- setNames(seq_along(common), common)
  n_genes <- length(common)
  edges <- edges_df[source %in% common & target %in% common]
  edges <- edges[, .(tf_idx = gene_to_idx[source], target_idx = gene_to_idx[target],
                     weight = as.numeric(weight))]
  message(sprintf("[zclip=%s] %d focused genes, %d edges, %d samples",
                  ifelse(is.infinite(clip), "none", clip), n_genes, nrow(edges), ncol(expr_sub)))
  z_all <- t(apply(expr_sub, 1, function(g) {
    m <- mean(g); s <- stats::sd(g) + 1e-10
    z <- (g - m) / s
    if (is.infinite(clip)) z else pmin(pmax(z, -clip), clip)
  }))
  scores <- vapply(seq_len(ncol(expr_sub)), function(i) {
    z <- z_all[, i]
    A <- Matrix::sparseMatrix(i = edges$tf_idx, j = edges$target_idx,
                              x = edges$weight * abs(z[edges$tf_idx]) * abs(z[edges$target_idx]),
                              dims = c(n_genes, n_genes))
    A <- A + Matrix::t(A)
    deg <- Matrix::rowSums(A)
    L <- Matrix::Diagonal(x = deg) - A
    trace <- sum(deg)
    if (trace < 1e-10) return(0)
    eigs <- eigen(L, symmetric = TRUE, only.values = TRUE)$values
    eigs <- pmax(eigs, 0)
    rho <- eigs / (trace + 1e-10)
    rho <- pmin(pmax(rho, 1e-12), 1.0)
    -sum(rho * log2(rho))
  }, numeric(1))
  names(scores) <- colnames(expr_sub)
  scores
}

# ---- IC50 table --------------------------------------------------------------
ic50_path <- c(file.path(DATA_DISK, "data", "gdsc_download", "GDSC2_IC50_all.csv"),
               file.path(DATA_DIR, "gdsc", "GDSC2_IC50_all.csv"))
ic50_path <- ic50_path[file.exists(ic50_path)][1]
stopifnot(!is.na(ic50_path))
ic50 <- fread(ic50_path)
ic50_mat <- dcast(ic50, CELL_LINE_NAME ~ DRUG_NAME, value.var = "LN_IC50", fun.aggregate = mean)

clips <- c(2, 3, 4, Inf)
res <- lapply(clips, function(clip) {
  scores <- compute_netith_clip(expr, edges_all, focused, clip)
  nt <- data.table(cell_line = names(scores), NetITH = unname(scores))
  common_cl <- intersect(nt$cell_line, ic50_mat$CELL_LINE_NAME)
  ntv <- nt$NetITH[match(common_cl, nt$cell_line)]
  drugs <- setdiff(names(ic50_mat), "CELL_LINE_NAME")
  rhos <- vapply(drugs, function(d) {
    y <- ic50_mat[[d]][match(common_cl, ic50_mat$CELL_LINE_NAME)]
    cc <- suppressWarnings(cor(ntv, y, method = "spearman", use = "complete.obs"))
    ifelse(is.na(cc), 0, cc)
  }, numeric(1))
  ps <- vapply(drugs, function(d) {
    y <- ic50_mat[[d]][match(common_cl, ic50_mat$CELL_LINE_NAME)]
    ok <- !is.na(ntv) & !is.na(y)
    if (sum(ok) < 10) return(1)
    suppressWarnings(cor.test(ntv[ok], y[ok], method = "spearman")$p.value)
  }, numeric(1))
  fdr <- p.adjust(ps, method = "BH")
  data.table(clip = ifelse(is.infinite(clip), "none", as.character(clip)),
             median_rho = round(median(rhos), 4),
             n_pos = sum(rhos > 0),
             n_fdr05 = sum(fdr < 0.05),
             n_drugs = length(rhos))
})
res <- rbindlist(res)
fwrite(res, file.path(RESULTS_DIR, "control", "zclip_sensitivity_median_rho.csv"))
out <- list(
  note = "NetITH z-clip sensitivity (review-batch-A item 12): GDSC cell lines, 286 drugs, Spearman rho vs ln(IC50), BH FDR; clip=3 is the manuscript reference",
  n_cell_lines = ncol(expr),
  n_drugs = 286,
  rows = as.list(setDT(res))
)
write_json(out, file.path(RESULTS_DIR, "control", "zclip_sensitivity.json"), pretty = TRUE)
cat("z-clip sensitivity done:\n"); print(res)
