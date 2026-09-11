# ============================================================================
# run_all.R — Master runner: R analysis chain + Python replication + figures, logged to CSV.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : Master runner for the full NetITH analysis chain. Executes the R
#           analysis scripts (core + validation + controls), the Python
#           scRNA-seq/spatial/replication scripts, and all figure scripts in
#           dependency order. Every step is recorded in
#           results/pipeline_run_log.csv (script, start, end, status,
#           key_metrics); failures do not abort the remaining steps.
# Inputs  : all data under data/ and results/ (see each script header);
#           optional external data-disk mount $NETITH_DATA_DISK (default
#           /Volumes/tjogzt4T) for GDSC/TCGA/scRNA-seq raw files.
# Outputs : results/{gdsc,tcga,neoadjuvant,control,scrnaseq_validation,
#           depmap}/... plus results/figures/r/* and pipeline_run_log.csv
# Usage   : Rscript code/R/run_all.R
# ============================================================================
suppressPackageStartupMessages({library(data.table)})

.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
FIG_DIR_R <- file.path(CODE_DIR, "figures")
RESULTS_DIR <- file.path(PROJECT_ROOT, "results")
LOG_FILE <- file.path(RESULTS_DIR, "pipeline_run_log.csv")
PYTHON <- Sys.getenv("NETITH_PYTHON", unset = "/opt/anaconda3/bin/python")

# ---- log helpers -----------------------------------------------------------
if (!file.exists(LOG_FILE)) {
  cat("script,start_time,end_time,status,key_metrics\n", file = LOG_FILE)
}
log_row <- function(script, start, end, status, metrics = "") {
  row <- sprintf("%s,%s,%s,%s,%s", script, start, end, status,
                 gsub(",", ";", metrics, fixed = TRUE))
  cat(row, "\n", file = LOG_FILE, append = TRUE)
}
run_r <- function(script) {
  st <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(sprintf("\n=== %s ===", script))
  ok <- tryCatch({
    sys.source(file.path(CODE_DIR, script), envir = globalenv())
    TRUE
  }, error = function(e) {
    message(sprintf("[FAIL] %s: %s", script, conditionMessage(e)))
    FALSE
  })
  log_row(script, st, format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          if (ok) "ok" else "error")
  invisible(ok)
}
run_py <- function(script, optional = FALSE) {
  st <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  full <- file.path(PROJECT_ROOT, script)
  if (optional && !file.exists(full)) {
    log_row(script, st, format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "skipped", "input missing")
    return(invisible(FALSE))
  }
  message(sprintf("\n=== %s (python) ===", script))
  ok <- tryCatch({
    system2(PYTHON, shQuote(full), stdout = TRUE, stderr = TRUE)
    TRUE
  }, error = function(e) {
    message(sprintf("[FAIL] %s: %s", script, conditionMessage(e)))
    FALSE
  })
  log_row(script, st, format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          if (ok) "ok" else "error")
  invisible(ok)
}

# ---- 1) R analysis chain (core + validation + controls) --------------------
run_r("02_netith_core.R")
run_r("03_gdsc_drug_sensitivity.R")
run_r("04_tcga_validation.R")
run_r("05_focused_mad_baseline.R")
run_r("06_gse25066_fixed_network.R")
run_r("07_tcga_jun_fixed_network.R")
run_r("08_signed_permutation_null.R")
run_r("09_auc_sensitivity.R")

# ---- 2) Python scRNA-seq / spatial / replication chain ---------------------
run_py("scripts/05_replications/01_scrnaseq_hardening.py")
run_py("scripts/05_replications/02_spatial_autocorrelation.py")
run_py("scripts/05_replications/03_gse120575_replication.py")
run_py("scripts/05_replications/04_gse72056_replication.py")
run_py("scripts/05_replications/05_gse123139_replication.py")
# (former 14_gse123139_sensitivity.py superseded: --sensitivity stage of 05_gse123139_replication.py)
# (former 15_gse120575_os_replication.py superseded: --os stage of 03_gse120575_replication.py)
# 16 (Seurat RDS conversion) runs only when the source RDS is present
if (file.exists(file.path(PROJECT_ROOT, "data", "external", "xue2022_hcc", "SexTumorDB_LIHC_Xue2022.rds"))) {
  run_r("14_xue2022_hcc_convert.R")
}
run_py("scripts/05_replications/06_gse115978_replication.py")
if (file.exists(file.path(PROJECT_ROOT, "data", "external", "xue2022_hcc", "expr.tsv.gz"))) {
  run_py("scripts/05_replications/07_xue2022_hcc_replication.py")
}

# ---- 3) figures (each also exports per-panel files) ------------------------
figs <- sort(list.files(FIG_DIR_R, pattern = "^fig.*\\.R$", full.names = TRUE))
for (f in figs) run_r(basename(f))

message("\n[run_all] complete: log at results/pipeline_run_log.csv")
