# ============================================================================
# 06_gse25066_fixed_network.R — GSE25066 neoadjuvant pCR on the fixed 239-gene signed CollecTRI construction.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-08-23
# Purpose : R2 recomputation (reviewer-requested, fixed-network test):
#           GSE25066 neoadjuvant pCR analysis on the SAME construction as the
#           GDSC primary analysis -- fixed focused 239-gene set, CollecTRI
#           edges within the focused genes, SIGNED weights (w*|z_TF|*|z_tgt|),
#           L = D - A, von Neumann entropy (log2). Replaces the cached
#           top-300-variable/62-edge construction used in the manuscript.
#           Also computes a GDSC-identical sensitivity (224 genes / 613 edges)
#           and validates the probe->gene parsing by reproducing the cached
#           top-300/62-edge construction.
# Inputs  : data/external/gse25066/GSE25066_series_matrix.txt,
#           data/external/gse25066/GPL96_full.txt,
#           data/collectri_network.csv,
#           results/focused_genes_collectri.txt,
#           results/neoadjuvant/gse25066_netith_pcr.csv (cached reference)
# Outputs : results/neoadjuvant/gse25066_fixed_network_netith.csv,
#           results/neoadjuvant/gse25066_fixed_network_summary.json,
#           results/neoadjuvant/gse25066_fixed_network_recomputation.md
# Usage   : Rscript code/R/06_gse25066_fixed_network.R   (run from repo root)
# ============================================================================
suppressPackageStartupMessages({ library(data.table); library(Matrix) })

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
if (!file.exists(file.path(.d, "data", "collectri_network.csv"))) stop("Project root not found")
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
source(file.path(CODE_DIR, "02_netith_core.R"))   # compute_netith / netith_one

GSE_DIR <- file.path(DATA_DIR, "external", "gse25066")
SM_PATH <- file.path(GSE_DIR, "GSE25066_series_matrix.txt")
GPL_PATH <- file.path(GSE_DIR, "GPL96_full.txt")
OUT_DIR <- file.path(RESULTS_DIR, "neoadjuvant")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== [1/7] parsing GSE25066 series matrix metadata ===\n")
sm_lines <- readLines(SM_PATH)

field_cells <- function(pat) {
  i <- grep(pat, sm_lines)[1]
  if (is.na(i)) return(NULL)
  cells <- strsplit(sm_lines[i], "\t", fixed = TRUE)[[1]]
  cells <- cells[-1]                     # drop the field-name cell
  sub('^"', '', sub('"$', '', cells))
}
titles     <- field_cells("^!Sample_title")
accessions <- field_cells("^!Sample_geo_accession")
stopifnot(!is.null(titles), !is.null(accessions), length(titles) == length(accessions))
n_samp <- length(accessions)
cat(sprintf("  samples: %d\n", n_samp))

char_idx <- grep("^!Sample_characteristics", sm_lines)
clin <- list()
for (i in char_idx) {
  cells <- strsplit(sm_lines[i], "\t", fixed = TRUE)[[1]]
  cells <- cells[-1]                     # drop the field-name cell
  cells <- sub('^"', '', sub('"$', '', cells))
  if (!grepl(":", cells[1])) next
  key <- sub(":.*$", "", cells[1])
  vals <- vapply(cells, function(c) {
    if (grepl(":", c)) sub("^[^:]*:\\s*", "", c) else c
  }, character(1))
  vals <- unname(vals)
  if (length(vals) < n_samp) vals <- c(vals, rep("", n_samp - length(vals)))
  clin[[key]] <- vals
}
cat(sprintf("  clinical attributes: %s\n", paste(names(clin), collapse = ", ")))
stopifnot("pathologic_response_pcr_rd" %in% names(clin))

cat("=== [2/7] parsing expression table (fread) ===\n")
tbl <- fread(SM_PATH, sep = "\t", skip = "!series_matrix_table_begin",
             header = TRUE, check.names = FALSE)
stopifnot(ncol(tbl) == n_samp + 1L)
probe_ids <- as.character(tbl[[1]])
expr_tab <- as.data.frame(tbl[, -1, with = FALSE])
rownames(expr_tab) <- probe_ids
expr_tab <- as.data.frame(lapply(expr_tab, as.numeric))
cat(sprintf("  probes x samples: %d x %d\n", nrow(expr_tab), ncol(expr_tab)))

cat("=== [3/7] parsing GPL96 probe->symbol annotation ===\n")
gpl <- fread(GPL_PATH, sep = "\t", skip = "!platform_table_begin",
             header = TRUE, check.names = FALSE)
stopifnot("Gene Symbol" %in% names(gpl))
sym <- gpl[["Gene Symbol"]][match(probe_ids, gpl[["ID"]])]
expr_tab$symbol <- sym
expr_tab <- expr_tab[!is.na(expr_tab$symbol) & nzchar(expr_tab$symbol), ]
sym_tab <- expr_tab$symbol
expr_tab$symbol <- NULL
gexpr_df <- aggregate(expr_tab, by = list(symbol = sym_tab), FUN = mean)
rownames(gexpr_df) <- gexpr_df$symbol
gexpr_df$symbol <- NULL
gexpr <- as.matrix(gexpr_df)
cat(sprintf("  genes x samples (probe-averaged): %d x %d\n", nrow(gexpr), ncol(gexpr)))

net <- fread(file.path(DATA_DIR, "collectri_network.csv"))
net[, weight := as.numeric(weight)]
focused_file <- file.path(RESULTS_DIR, "focused_genes_collectri.txt")
focused <- readLines(focused_file); focused <- focused[nzchar(focused)]
cat(sprintf("  focused gene list: %d\n", length(focused)))

cat("=== [4/7] validation: reproduce cached top-300/62-edge construction ===\n")
cached <- fread(file.path(RESULTS_DIR, "neoadjuvant", "gse25066_netith_pcr.csv"))
all_g <- rownames(gexpr)
v <- apply(gexpr, 1, var, na.rm = TRUE)
edge_genes <- names(sort(v, decreasing = TRUE))[1:min(300, length(v))]
net_val <- net[source %in% edge_genes & target %in% edge_genes]
net_val[, w_abs := abs(weight)]
Z_val <- t(apply(gexpr[edge_genes, , drop = FALSE], 1, function(g) {
  m <- mean(g); s <- sd(g) + 1e-10
  pmin(pmax((g - m) / s, -3), 3)
}))
gi <- setNames(seq_along(edge_genes), edge_genes)
edges_val <- net_val[, .(tf = gi[source], tg = gi[target], w = w_abs)]
ent_val <- vapply(seq_len(ncol(Z_val)), function(i) {
  A <- Matrix::sparseMatrix(i = edges_val$tf, j = edges_val$tg,
                            x = edges_val$w * abs(Z_val[edges_val$tf, i]) * abs(Z_val[edges_val$tg, i]),
                            dims = c(length(edge_genes), length(edge_genes)))
  A <- A + Matrix::t(A)
  deg <- Matrix::rowSums(A)
  L <- Matrix::Diagonal(x = deg) - A
  ev <- eigen(L, symmetric = TRUE, only.values = TRUE)$values
  ev <- ev[ev > 1e-12]
  s <- sum(ev)
  if (s <= 0) 0 else { rho <- ev / s; -sum(rho * log2(rho)) }
}, numeric(1))
names(ent_val) <- colnames(Z_val)
common_v <- intersect(names(ent_val), cached$accession)
rho_v <- cor(ent_val[common_v], cached$NetITH[match(common_v, cached$accession)], method = "spearman")
cat(sprintf("  cached-construction replicate: %d genes, %d edges; Spearman vs cached NetITH = %.6f (n=%d)\n",
            length(edge_genes), nrow(net_val), rho_v, length(common_v)))
stopifnot(rho_v > 0.99)

cat("=== [5/7] fixed-construction NetITH (signed weights) ===\n")
netith_239 <- compute_netith(gexpr, net, focused)
common_239 <- intersect(focused, rownames(gexpr))
edges_239 <- net[source %in% common_239 & target %in% common_239]
cat(sprintf("  construction A (focused 239): %d genes present, %d edges (signed; %d negative; %d self-loops)\n",
            length(common_239), nrow(edges_239), sum(edges_239$weight < 0), sum(edges_239$source == edges_239$target)))

# GDSC-identical sensitivity: the 224 focused genes present in GDSC expression
missing_gdsc <- c("CHRD","GH1","HBA2","IFITM1","IL1RL1","IL32","KRT16","MIF",
                  "NEUROD1","POU3F3","SERPINA3","SERPINB2","SLC18A3","SOX2","ZSCAN26")
focused_224 <- setdiff(focused, missing_gdsc)
stopifnot(length(focused_224) == 224L)
netith_224 <- compute_netith(gexpr, net, focused_224)
common_224 <- intersect(focused_224, rownames(gexpr))
edges_224 <- net[source %in% common_224 & target %in% common_224]
cat(sprintf("  construction B (GDSC-identical 224): %d genes present, %d edges (signed; %d negative)\n",
            length(common_224), nrow(edges_224), sum(edges_224$weight < 0)))

cat("=== [6/7] pCR statistics on fixed constructions ===\n")
pcr_map <- c("pCR" = 1L, "RD" = 0L)
cdf <- data.frame(accession = accessions, title = titles,
                  NetITH_239 = unname(netith_239[accessions]),
                  NetITH_224 = unname(netith_224[accessions]),
                  stringsAsFactors = FALSE)
for (k in names(clin)) if (length(clin[[k]]) == n_samp) cdf[[k]] <- clin[[k]]
cdf$pcr_bin <- pcr_map[cdf$pathologic_response_pcr_rd]
cdf <- cdf[!is.na(cdf$pcr_bin) & is.finite(cdf$NetITH_239), ]
cat(sprintf("  analysable patients: %d (pCR: %d)\n", nrow(cdf), sum(cdf$pcr_bin)))
stopifnot(nrow(cdf) == 306L, sum(cdf$pcr_bin) == 57L)

stats_for <- function(score, label) {
  z <- as.numeric(scale(score))
  m <- glm(cdf$pcr_bin ~ z, family = binomial())
  ci <- confint.default(m)["z", ]
  or_v <- exp(coef(m)["z"]); lo <- exp(ci[1]); hi <- exp(ci[2])
  p <- summary(m)$coefficients["z", 4]
  mw <- wilcox.test(score[cdf$pcr_bin == 1], score[cdf$pcr_bin == 0])
  tert <- cut(score, breaks = quantile(score, c(0, 1/3, 2/3, 1)), include.lowest = TRUE,
              labels = c("low", "mid", "high"))
  rates <- tapply(cdf$pcr_bin, tert, function(x) c(rate = mean(x), n = length(x)))
  rate_lo <- rates[["low"]]["rate"]; rate_mid <- rates[["mid"]]["rate"]; rate_hi <- rates[["high"]]["rate"]
  cat(sprintf("  [%s] OR per SD = %.3f [%.3f-%.3f] p=%.4f | MW p=%.4f | tertile pCR: %.1f/%.1f/%.1f%%\n",
              label, or_v, lo, hi, p, mw$p.value, 100*rate_lo, 100*rate_mid, 100*rate_hi))
  list(or = or_v, lo = lo, hi = hi, p = p, mw_p = mw$p.value,
       tert = c(low = unname(rate_lo), mid = unname(rate_mid), high = unname(rate_hi)))
}
s239 <- stats_for(cdf$NetITH_239, "A: 239-gene fixed")
s224 <- stats_for(cdf$NetITH_224, "B: GDSC-identical 224")

cat("=== [7/7] saving outputs ===\n")
fwrite(cdf, file.path(OUT_DIR, "gse25066_fixed_network_netith.csv"))

summary_json <- list(
  cohort = "GSE25066",
  n = nrow(cdf), n_pcr = sum(cdf$pcr_bin),
  construction_A = list(n_genes = length(common_239), n_genes_present = length(common_239),
                        n_edges = nrow(edges_239), n_neg_edges = sum(edges_239$weight < 0),
                        n_self_loops = sum(edges_239$source == edges_239$target),
                        or_per_sd = s239$or, ci = c(s239$lo, s239$hi), p = s239$p,
                        mw_p = s239$mw_p,
                        pcr_rate_tertile = unname(s239$tert),
                        pcr_rate_high_minus_low = unname(s239$tert["high"] - s239$tert["low"])),
  construction_B_gdsc_identical = list(n_genes = length(common_224),
                        n_edges = nrow(edges_224), n_neg_edges = sum(edges_224$weight < 0),
                        or_per_sd = s224$or, ci = c(s224$lo, s224$hi), p = s224$p,
                        mw_p = s224$mw_p,
                        pcr_rate_tertile = unname(s224$tert),
                        pcr_rate_high_minus_low = unname(s224$tert["high"] - s224$tert["low"])),
  cached_62edge = list(n_genes = 300, n_edges = 62,
                       or_per_sd = 1.2712696236546965, ci = c(0.9310780877967061, 1.7357582325361556),
                       p = 0.13090723376078117, mw_p = 0.108181181274866,
                       pcr_rate_tertile = c(low = 0.14705882352941177, mid = 0.17647058823529413, high = 0.23529411764705882)),
  validation_replicate = list(n_genes = length(edge_genes), n_edges = nrow(net_val),
                              spearman_vs_cached = rho_v, n = length(common_v))
)
jsonlite::write_json(summary_json, file.path(OUT_DIR, "gse25066_fixed_network_summary.json"),
                     auto_unbox = TRUE, digits = 8)
cat("JSON summary written.\n")
cat("\n===== KEY VALUES =====\n")
cat(sprintf("A239: genes=%d edges=%d neg=%d OR=%.4f [%.4f-%.4f] p=%.4f mw=%.4f tert=%.4f/%.4f/%.4f\n",
            length(common_239), nrow(edges_239), sum(edges_239$weight < 0),
            s239$or, s239$lo, s239$hi, s239$p, s239$mw_p, s239$tert[1], s239$tert[2], s239$tert[3]))
cat(sprintf("B224: genes=%d edges=%d neg=%d OR=%.4f [%.4f-%.4f] p=%.4f mw=%.4f tert=%.4f/%.4f/%.4f\n",
            length(common_224), nrow(edges_224), sum(edges_224$weight < 0),
            s224$or, s224$lo, s224$hi, s224$p, s224$mw_p, s224$tert[1], s224$tert[2], s224$tert[3]))
cat(sprintf("VAL: replicate genes=%d edges=%d spearman=%.6f n=%d\n",
            length(edge_genes), nrow(net_val), rho_v, length(common_v)))
cat("===== END KEY VALUES =====\n")
