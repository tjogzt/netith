# ============================================================================
# 13_gse120575_os_cox.R — GSE120575 OS: Cox PH and median-split log-rank for patient/timepoint NetITH_E.
# Project : NetITH — spectral-entropy descriptor of transcription-factor networks
# Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
# Created : 2026-09-05
# Purpose : R survival companion for 15_gse120575_os_replication.py: reads the
#           per-patient / per-timepoint NetITH_E + OS table and writes exact
#           Cox PH (per-SD HR, 95% CI, Wald p) and median-split log-rank
#           results as JSON. The Python script previously emitted numerically
#           unstable Cox estimates; R survival is authoritative here.
# Inputs  : data/external/gse120575/os_netith_levels.csv (written by the .py)
# Outputs : data/external/gse120575/os_cox_results.json
# Usage   : Rscript code/R/13_gse120575_os_cox.R
# ============================================================================
suppressMessages({library(survival); library(jsonlite)})
d <- read.csv("data/external/gse120575/os_netith_levels.csv", row.names = 1)
out <- list()
for (lvl in c("patient", "timepoint")) {
  dd <- d[d$level == lvl, ]
  if (nrow(dd) >= 10 && sum(dd$os_event) >= 3) {
    c1 <- coxph(Surv(os_days, os_event) ~ scale(netith_e), data = dd)
    lr <- survdiff(Surv(os_days, os_event) ~ I(netith_e >= median(netith_e)), data = dd)
    out[[lvl]] <- list(
      hr = unname(exp(coef(c1))),
      ci95 = as.vector(exp(confint(c1))),
      wald_p = coef(summary(c1))[5],
      logrank_p = 1 - pchisq(lr$chisq, 1),
      n = nrow(dd),
      events = sum(dd$os_event)
    )
  } else {
    out[[lvl]] <- list(hr = NA, ci95 = c(NA, NA), wald_p = NA, logrank_p = NA,
                       n = nrow(dd), events = sum(dd$os_event))
  }
}
write(toJSON(out, auto_unbox = TRUE, digits = NA),
      "data/external/gse120575/os_cox_results.json")
cat("written os_cox_results.json\n")
