# ============================================================================
# figS7_causal_graph.R — Supplementary Figure 7
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Inputs  : results/depmap/ext_d1_causal_discovery.csv
# Outputs : results/figures/r/FigS7_causal_graph.{pdf,png}
#           + per-panel exports in results/figures/r/panels/
# Usage   : Rscript code/R/figures/figS7_causal_graph.R
# Note    : Node positions come from a Fruchterman-Reingold layout (seed 49); all numbers (edge count, AP-1 count, weights) are read from cache.
# ============================================================================

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork); library(igraph) })

NAME <- "FigS7_causal_graph"
DEP <- file.path(RESULTS_DIR, "depmap")

cat("=== EDFig1: DirectLiNGAM causal graph ===\n")

# ---- data: causal edges with inferred directions ----------------------------
edges <- fread(file.path(DEP, "ext_d1_causal_discovery.csv"))
stopifnot(nrow(edges) > 0, all(c("var1", "var2", "pearson_r", "direction") %in% names(edges)))
edges[, c("from", "to") := tstrsplit(direction, "\u2192", fixed = TRUE)]
edges[, `:=`(from = trimws(from), to = trimws(to))]
edges[, w := abs(pearson_r)]
edges[, ap1 := grepl("JUN|FOS|ATF", from)]      # AP-1-family source (same rule as manuscript)

# ---- network layout (Fruchterman-Reingold; project seed) ---------------------
set.seed(SEED)
g <- igraph::graph_from_data_frame(edges[, .(from, to, w, ap1)], directed = TRUE)
lo <- igraph::layout_with_fr(g, niter = 1000)
nd <- data.frame(name = igraph::V(g)$name, x = lo[, 1], y = lo[, 2])
edf <- igraph::as_data_frame(g, what = "edges")
edf$x <- nd$x[match(edf$from, nd$name)]
edf$y <- nd$y[match(edf$from, nd$name)]
edf$xend <- nd$x[match(edf$to, nd$name)]
edf$yend <- nd$y[match(edf$to, nd$name)]
edf <- edf[order(edf$w), ]                       # draw weak edges first

pA <- ggplot() +
  geom_curve(data = edf, aes(x, y, xend = xend, yend = yend, colour = ap1, linewidth = w),
             curvature = 0.15, alpha = 0.55, lineend = "round",
             arrow = arrow(type = "closed", length = unit(0.09, "inches"))) +
  # AP-1-family edges (JUN/FOS/ATF sources) in COL_NER, matching the JUN/AP-1
  # emphasis colour used in Fig3F; COL_TCGA/COL_GDSC are reserved for cohorts.
  scale_colour_manual(values = c(`TRUE` = COL_NER, `FALSE` = COL_REF), guide = "none") +
  scale_linewidth_continuous(range = c(0.3, 2.0), guide = "none") +
  geom_point(data = nd, aes(x, y), size = 7, shape = 21, fill = COL_NETITH,
             colour = COL_NETITH, alpha = 0.85) +
  geom_text(data = nd, aes(x, y, label = name), size = FONT_BASE/.pt, fontface = "bold") +
  coord_equal(xlim = range(nd$x) + c(-1, 1) * 0.5 * diff(range(nd$x)),
              ylim = range(nd$y) + c(-1, 1) * 0.5 * diff(range(nd$y))) +
  labs(title = sprintf("DirectLiNGAM: %d directed edges | AP-1 family dominant", nrow(edges))) +
  theme_void(base_size = FONT_BASE) +
  theme(plot.title = element_text(size = FONT_TITLE, face = "bold", hjust = 0.5),
        plot.margin = margin(2, 2, 2, 2, "mm"))

save_fig(pA, NAME, height_mm = 139.7, width_mm = 85)
save_panels(list(pA), NAME)

cat(sprintf("  %d directed edges, %d nodes\n", nrow(edges), igraph::vcount(g)))
cat(sprintf("  AP-1-family edges (source in JUN/FOS/ATF): %d / %d\n",
            sum(edges$ap1), nrow(edges)))
cat(sprintf("  Edge weight |r| range: %.3f - %.3f\n", min(edges$w), max(edges$w)))
cat(sprintf("  PDF: %s\n", file.path(FIG_DIR, paste0(NAME, ".pdf"))))
