# ============================================================================
# 14_xue2022_hcc_convert.R — Convert SexTumorDB Xue2022 HCC Seurat RDS to focused-gene matrix + metadata.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : Convert SexTumorDB LIHC_Xue2022 Seurat RDS (26 patients) into a
#           focused-239-gene x cell matrix + per-cell metadata.
# Inputs  : data/external/xue2022_hcc/SexTumorDB_LIHC_Xue2022.rds,
#           results/focused_genes_collectri.txt
# Outputs : data/external/xue2022_hcc/expr.tsv.gz + meta.tsv
# Usage   : Rscript code/R/14_xue2022_hcc_convert.R
# ============================================================================
suppressMessages({library(Seurat)})
rds <- "data/external/xue2022_hcc/SexTumorDB_LIHC_Xue2022.rds"
focused <- readLines("results/focused_genes_collectri.txt")
focused <- focused[nzchar(trimws(focused))]
cat("loading RDS...\n")
obj <- readRDS(rds)
cat("class:", class(obj), " dims:", dim(obj), "\n")
cnt <- tryCatch(LayerData(obj, assay = "RNA", layer = "counts"), error = function(e) NULL)
if (is.null(cnt)) cnt <- LayerData(obj, assay = "RNA", layer = "data")
meta <- obj@meta.data
cat("meta cols:", paste(colnames(meta), collapse = ", "), "\n")
keep <- intersect(focused, rownames(cnt))
cat("focused genes present:", length(keep), "/", length(focused), "\n")
sub <- as.matrix(cnt[keep, , drop = FALSE])
cat("writing expr.tsv.gz (", nrow(sub), "x", ncol(sub), ")...\n")
write.table(sub, gzfile("data/external/xue2022_hcc/expr.tsv.gz"), sep = "\t", quote = FALSE, col.names = NA)
write.table(meta, "data/external/xue2022_hcc/meta.tsv", sep = "\t", quote = FALSE, col.names = NA)
cat("done\n")
