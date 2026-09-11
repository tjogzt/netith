# ============================================================================
# 00_global_config.R — Global configuration: project paths, NPG palette, theme, figure helpers, seed.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-07
# Purpose : Global configuration for the NetITH R re-analysis: project paths,
#           unified NPG colour palette, journal figure canvas constants,
#           ggplot2 theme, and random seed. Loaded by every analysis/plot script.
# Inputs  : none (pure configuration)
# Outputs : side effects only (defines constants used across the pipeline)
# Usage   : source("00_global_config.R")
# Note    : All figure text is English; all documents/interaction are Chinese.
#           R >= 4.3.0; ggplot2 >= 3.4; patchwork >= 1.2
# ============================================================================

# ---- project paths (three-tier layout: data/ code/ results/) --------------
find_project_root <- function() {
  d <- getwd()
  while (!file.exists(file.path(d, "data", "collectri_network.csv")) && nchar(d) > 1) {
    d <- dirname(d)
  }
  if (!file.exists(file.path(d, "data", "collectri_network.csv"))) stop("Project root not found")
  d
}
PROJECT_ROOT <- find_project_root()
DATA_DIR    <- file.path(PROJECT_ROOT, "data")
CODE_DIR    <- file.path(PROJECT_ROOT, "code", "R")
RESULTS_DIR <- file.path(PROJECT_ROOT, "results")
FIG_DIR     <- file.path(RESULTS_DIR, "figures", "r")           # R-generated figures
PANEL_DIR   <- file.path(RESULTS_DIR, "figures", "r", "panels") # per-panel exports

# ---- random seed (project-wide standard, see project norms) ---------------
SEED <- 49L
set.seed(SEED)
DATA_DISK <- Sys.getenv("NETITH_DATA_DISK", unset = "/Volumes/tjogzt4T")  # external data-disk mount root (override via env var)


# ---- unified NPG (Nature Publishing Group) colour palette ----------------
# NetITH is #4DBBD5 (cyan) throughout; GDSC red; TCGA green; signature blue.
COL_NETITH <- "#2E6F8E"   # NetITH (main)
COL_GDSC   <- "#C03A2B"   # GDSC / drug-response datasets
COL_TCGA   <- "#4A7C59"   # TCGA / clinical datasets
COL_SIG    <- "#374E6E"   # signature / comparison methods
COL_NER    <- "#D9922E"   # NER / AP-1 related
COL_NS     <- "#9C8AA5"   # non-significant / neutral
COL_REF    <- "#8A7B6C"   # reference line / secondary
COL_GREY   <- "#E0E0E0"   # background points
COL_ACCENT <- "#5B8C5A"   # semantic-free emphasis (e.g. concordant/selected marks)
COL_BLACK  <- "#000000"

NPG_COLORS <- c(COL_NETITH, COL_GDSC, COL_TCGA, COL_SIG, COL_NER, COL_NS,
                "#668B6E", "#D9922E", "#7A6F9E", "#8C6D5D")

# ---- journal canvas constants (double column = 180 mm, direct output) ------
# Rule: target size straight out; NEVER generate large then shrink.
MM2IN      <- 1 / 25.4
PANEL_W_FULL    <- 180 * MM2IN   # 7.0866 in double-column full width
PANEL_W_HALF    <-  85 * MM2IN   # single column
PANEL_W_THIRD   <-  57 * MM2IN   # one third of double column
PANEL_W_QUARTER <-  45 * MM2IN   # one quarter of double column
FONT_BASE   <- 8    # base font size (pt) at final print size
FONT_TITLE  <- 8    # panel titles
FONT_LABEL  <- 9    # panel letters (A/B/C, bold)

# ---- ggplot2 publication theme -------------------------------------------------
theme_pub <- function(base_size = FONT_BASE, base_family = "Arial") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      axis.line          = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks         = element_line(linewidth = 0.3, colour = "black"),
      axis.text          = element_text(size = base_size, colour = "black"),
      axis.title         = element_text(size = base_size),
      legend.text        = element_text(size = base_size),
      legend.title       = element_text(size = base_size),
      legend.key.size    = unit(0.35, "cm"),
      legend.background  = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(linewidth = 0.2, colour = "grey92"),
      panel.border       = element_blank(),
      strip.text         = element_text(size = base_size, face = "bold"),
      plot.title         = element_text(size = FONT_TITLE, face = "bold"),
      plot.margin        = margin(2, 2, 2, 2, "mm")
    )
}

# ---- device helpers ------------------------------------------------------------
# Save a ggplot at real double-column size: vector PDF (cairo, embedded fonts)
# + 300-dpi PNG. width_mm defaults to 180 mm; height_mm must be supplied.
save_fig <- function(plot, name, height_mm, width_mm = 180, dir = FIG_DIR) {
  stopifnot(is.numeric(height_mm), height_mm > 0, is.numeric(width_mm), width_mm > 0)
  dir.create(dir, showWarnings = FALSE)
  w_in <- width_mm * MM2IN
  h_in <- height_mm * MM2IN
  pdf_path <- file.path(dir, paste0(name, ".pdf"))
  png_path <- file.path(dir, paste0(name, ".png"))
  grDevices::cairo_pdf(pdf_path, width = w_in, height = h_in, family = "Arial")
  print(plot)
  grDevices::dev.off()
  ggplot2::ggsave(png_path, plot = plot, width = w_in, height = h_in,
                  dpi = 300, bg = "white")
  message(sprintf("[save_fig] %s: %.1f x %.1f mm (PDF + 300-dpi PNG)", name,
                  width_mm, height_mm))
  invisible(pdf_path)
}

# Export each panel ggplot (explicit list, patchwork-internal structure is
# version-fragile) as its own PDF + 300-dpi PNG for review and reassembly.
save_panels <- function(plots, name, dir = PANEL_DIR, dpi = 300) {
  stopifnot(is.list(plots), length(plots) > 0L)
  dir.create(dir, showWarnings = FALSE)
  letters <- LETTERS[seq_along(plots)]
  for (i in seq_along(plots)) {
    stopifnot(inherits(plots[[i]], "ggplot"))
    ggplot2::ggsave(file.path(dir, sprintf("%s_panel%s.pdf", name, letters[i])),
                    plot = plots[[i]], device = cairo_pdf, width = 3.5, height = 3.5, dpi = 300)
    ggplot2::ggsave(file.path(dir, sprintf("%s_panel%s.png", name, letters[i])),
                    plot = plots[[i]], dpi = dpi, width = 3.5, height = 3.5, bg = "white")
  }
  message(sprintf("[save_panels] %s: %d panels exported", name, length(plots)))
  invisible(letters)
}

# ---- small utilities --------------------------------------------------------------
`%||%` <- function(a, b) if (is.null(a)) b else a

# Format p values: scientific below 0.001, 3 decimals otherwise
fmt_p <- function(p) ifelse(p < 0.001, formatC(p, format = "e", digits = 1),
                            sprintf("%.3f", p))

# Spearman rho with p value (two-sided), returns named vector
spearman_test <- function(x, y, method = "spearman") {
  stopifnot(is.numeric(x), is.numeric(y), length(x) == length(y))
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(c(rho = NA_real_, p = NA_real_, n = sum(ok)))
  ct <- cor.test(x[ok], y[ok], method = method, exact = FALSE)
  c(rho = unname(ct$estimate), p = unname(ct$p.value), n = sum(ok))
}
