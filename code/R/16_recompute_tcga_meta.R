# ============================================================================
# 16_recompute_tcga_meta.R — TCGA random-effects meta-analysis under |w| and signed-w constructions.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : R1 recomputation: TCGA pan-cancer survival + random-effects
#           meta-analysis under BOTH the |w| and signed-w NetITH constructions.
#           The cached Python TCGA pipeline used abs(weight) (|w|), so the
#           cached tcga_survival_results.csv / multicancer_meta_analysis.csv
#           are |w|-based; this script recomputes per-cancer Cox from
#           freshly computed NetITH vectors (both constructions) and reports
#           the DerSimonian-Laird random-effects meta HR for each.
# Inputs  : /tmp/nt_abs.npy, /tmp/nt_sig.npy (NetITH, both constructions),
#           results/tcga/tcga_survival_results.csv (cancer-type grouping),
#           data disk Xena survival file (OS/time + cancer type)
# Outputs : results/tcga/tcga_meta_abs_vs_signed.csv
# Usage   : Rscript code/R/16_recompute_tcga_meta.R
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

suppressPackageStartupMessages({
  library(data.table)
  library(survival)
})

# ---- load survival data (Xena format: header glued after 'head(data)') -----
load_survival <- function(path) {
  con <- file(path, "r")
  lines <- readLines(con, warn = FALSE)
  close(con)
  hdr_idx <- grep("\tsample\t", lines)[1]
  stopifnot(!is.na(hdr_idx))
  hdr_line <- lines[hdr_idx]
  hdr <- strsplit(hdr_line, "\t")[[1]]
  # first token is 'head(data)sample' -> fix
  hdr[1] <- sub("^head\\(data\\)", "", hdr[1])
  body <- lapply(lines[(hdr_idx + 1):length(lines)], function(l) {
    parts <- strsplit(l, "\t")[[1]]
    if (length(parts) < length(hdr)) parts <- c(parts, rep(NA, length(hdr) - length(parts)))
    parts[seq_len(length(hdr))]
  })
  surv <- as.data.frame(do.call(rbind, body), stringsAsFactors = FALSE)
  names(surv) <- hdr
  data.table::setDT(surv)
  surv[, OS := as.numeric(OS)]
  surv[, OS.time := as.numeric(OS.time)]
  surv <- surv[!is.na(OS) & !is.na(OS.time) & OS.time > 0]
  surv[]
}

# ---- DerSimonian-Laird random-effects meta (mirrors run_multicancer_validation) --
re_meta <- function(log_hr, se) {
  valid <- is.finite(log_hr) & is.finite(se) & se > 0
  log_hr <- log_hr[valid]; se <- se[valid]
  k <- length(se)
  if (k < 3) return(c(re_hr = NA_real_, re_lo = NA_real_, re_hi = NA_real_,
                      re_p = NA_real_, fe_hr = NA_real_, I2 = NA_real_, k = k))
  wf <- 1 / se^2
  fe <- sum(wf * log_hr) / sum(wf)
  Q <- sum(wf * (log_hr - fe)^2)
  tau2 <- max(0, (Q - (k - 1)) / (sum(wf) - sum(wf^2) / sum(wf)))
  wr <- 1 / (se^2 + tau2)
  re <- sum(wr * log_hr) / sum(wr)
  re_se <- sqrt(1 / sum(wr))
  I2 <- max(0, (Q - (k - 1)) / Q * 100)
  c(re_hr = exp(re), re_lo = exp(re - 1.96 * re_se), re_hi = exp(re + 1.96 * re_se),
    re_p = 2 * pnorm(-abs(re / re_se)), fe_hr = exp(fe), I2 = I2, k = k)
}

# ---- per-cancer Cox (continuous NetITH) + meta -------------------------------
run_analysis <- function(netith_vec, netith_names, surv, cached) {
  dt <- data.table(sample = netith_names, netith = unname(netith_vec))
  m <- merge(surv[, .(sample, cancer_type, OS, OS.time)], dt, by = "sample")
  m <- m[!is.na(netith)]
  message(sprintf("[tcga] merged %d samples", nrow(m)))
  cts <- unique(cached$cancer_type)  # reuse cached cancer-type grouping
  rows <- lapply(cts, function(ct) {
    sub <- m[cancer_type == ct]
    if (nrow(sub) < 30 || sum(sub$OS) < 10) return(NULL)
    fit <- coxph(Surv(OS.time, OS) ~ netith, data = sub)
    s <- summary(fit)
    cs <- coef(summary(fit))
    data.table(cancer_type = ct, n = nrow(sub), n_events = sum(sub$OS),
               hr = exp(coef(fit)), hr_se = cs[1, 3], p = cs[1, 5])
  })
  res <- rbindlist(rows)
  res
}

main <- function() {
  # fresh NetITH (both constructions) from the Python recomputation
  if (!file.exists("/tmp/nt_abs.npy") || !file.exists("/tmp/nt_sig.npy"))
    stop("Missing /tmp/nt_abs.npy or /tmp/nt_sig.npy (run the TCGA recomputation first)")
  py <- reticulate::import("numpy")
  nt_abs <- py$load("/tmp/nt_abs.npy")
  nt_sig <- py$load("/tmp/nt_sig.npy")

  # sample order must be recovered from the recomputation script: expr_z.index
  # (gene-symbol-filtered TCGA sample order). We recover it by matching against
  # the cached tcga_netith.csv sample column (same order in practice).
  cache_net <- fread(file.path(RESULTS_DIR, "tcga", "tcga_netith.csv"))
  samples <- cache_net$sample
  stopifnot(length(samples) == length(nt_abs))

  surv_path <- file.path(DATA_DISK, "data", "xena", "tcgapancan",
                         "Survival_SupplementalTable_S1_20171025_xena_sp")
  if (!file.exists(surv_path)) stop("Xena survival file not found")
  surv <- load_survival(surv_path)
  cached <- fread(file.path(RESULTS_DIR, "tcga", "tcga_survival_results.csv"))

  res_abs <- run_analysis(nt_abs, samples, surv, cached)
  res_sig <- run_analysis(nt_sig, samples, surv, cached)

  meta_abs <- re_meta(log(res_abs$hr), res_abs$hr_se)
  meta_sig <- re_meta(log(res_sig$hr), res_sig$hr_se)

  out <- data.table(
    construction = c("abs_w", "signed_w", "cached_python_abs_w"),
    re_hr = c(meta_abs["re_hr"], meta_sig["re_hr"], NA_real_),
    re_lo = c(meta_abs["re_lo"], meta_sig["re_lo"], NA_real_),
    re_hi = c(meta_abs["re_hi"], meta_sig["re_hi"], NA_real_),
    re_p  = c(meta_abs["re_p"], meta_sig["re_p"], NA_real_),
    fe_hr = c(meta_abs["fe_hr"], meta_sig["fe_hr"], NA_real_),
    I2    = c(meta_abs["I2"], meta_sig["I2"], NA_real_),
    k     = c(meta_abs["k"], meta_sig["k"], NA_real_)
  )
  # cached meta from results/tcga/multicancer_meta_analysis.csv (already |w|)
  cm <- fread(file.path(RESULTS_DIR, "tcga", "multicancer_meta_analysis.csv"))
  out[construction == "cached_python_abs_w", `:=`(
    re_hr = cm$re_hr, re_lo = cm$hk_ci_lower, re_hi = cm$hk_ci_upper,
    re_p = cm$hk_p, fe_hr = cm$fe_hr, I2 = cm$I2, k = cm$k)]

  fwrite(out, file.path(RESULTS_DIR, "tcga", "tcga_meta_abs_vs_signed.csv"))
  cat(sprintf("[tcga-meta] |w|: RE HR=%.4f [%.4f-%.4f] p=%.3g (k=%d)\n",
              meta_abs["re_hr"], meta_abs["re_lo"], meta_abs["re_hi"], meta_abs["re_p"], meta_abs["k"]))
  cat(sprintf("[tcga-meta] signed: RE HR=%.4f [%.4f-%.4f] p=%.3g (k=%d)\n",
              meta_sig["re_hr"], meta_sig["re_lo"], meta_sig["re_hi"], meta_sig["re_p"], meta_sig["k"]))
  # consistency of recomputed |w| against cached per-cancer (should match closely)
  cc <- cor(res_abs$hr, cached$cox_hr, method = "spearman", use = "complete.obs")
  cat(sprintf("[tcga-meta] recomputed |w| per-cancer HR vs cached: rho=%.4f\n", cc))
  invisible(out)
}

main()
