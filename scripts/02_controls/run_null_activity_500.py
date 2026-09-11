#!/usr/bin/env python3
"""run_null_activity_500.py — stabilized activity-aggregate regulon-shuffle null (500 draws, seed 42).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-10
Pipeline: control / test stage — see repository README

The 50-draw censuses under two seed-consistent RNG streams gave empirical
p = 0.333 and 0.020 for the TF-activity variance aggregate's Test-1 null —
realization-sensitive because the null band is narrow ([-0.18, -0.23]) and the
real value (-0.238) sits at its edge. This run draws 500 independent
regulon-membership shuffles (identical design and pipeline) to stabilize the
empirical p.

Outputs: results/control/census/null_activity_500draws.json
Usage: NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_null_activity_500.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_descriptor_census import (load_gdsc_expr, load_ic50, build_tf_targets,
                                   activity_per_sample, drug_assoc)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "control" / "census" / "null_activity_500draws.json"
RNG = np.random.default_rng(42)
N = 500

expr = load_gdsc_expr()
X = expr.values.astype(float)
mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-10
Z_full = np.clip((X - mu) / sd, -3, 3)
tf_targets = build_tf_targets(expr.index)
gene_perm_base = np.arange(X.shape[0])
ic50_mat, _ = load_ic50()
cells = list(expr.columns)

# real
real = pd.Series({cl: activity_per_sample(Z_full[:, j], tf_targets)
                  for j, cl in enumerate(cells)})
real_rho = drug_assoc(real, ic50_mat, ic50_mat)["rho_ic50"].median()
print(f"real activity median rho = {real_rho:.4f}")

draws = []
t0 = time.time()
for k in range(N):
    perm = RNG.permutation(gene_perm_base)
    tgt = [[perm[t] for t in tt] for tt in tf_targets]
    desc = {}
    for j, cl in enumerate(cells):
        acts = np.array([Z_full[t, j].mean() for t in tgt if len(t) > 0])
        desc[cl] = float(acts.var())
    rho = drug_assoc(pd.Series(desc), ic50_mat, ic50_mat)["rho_ic50"].median()
    draws.append(float(rho))
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{N} ({time.time()-t0:.0f}s)")

draws = np.array(draws)
# Empirical two-sided p from the 500-draw null via the finite-null (+1)/(N+1)
# convention, so p is never exactly 0 when the real value falls outside the band.
emp = (1 + int((np.abs(draws) >= abs(real_rho)).sum())) / (N + 1)
out = {
    "design": "regulon-membership shuffle (gene labels permuted, regulon sizes kept), 500 draws, seed 42",
    "real_median_rho": float(real_rho),
    "null": {"n": N, "mean": float(draws.mean()), "sd": float(draws.std()),
             "range": [float(draws.min()), float(draws.max())]},
    "emp_p": float(emp),
    "stream_sensitivity": {"50-draw_run1_emp_p": 0.333, "50-draw_run2_emp_p": 0.0196,
                           "note": "both seed-consistent realizations of the same design"},
}
json.dump(out, open(OUT, "w"), indent=2)
print(f"[done] null mean {draws.mean():.4f} sd {draws.std():.4f} "
      f"range [{draws.min():.4f},{draws.max():.4f}] emp p={emp:.4f}")
