"""
02_spatial_autocorrelation.py — spatial orthogonal validation (four-way verdict step 3).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : results/depmap/spatial_visium_netith.csv (BRCA Visium spot-level NetITH),
          results/depmap/spatial_visium_OV_netith.csv (ovarian negative control)
Outputs : results/depmap/spatial_autocorrelation.json
Pipeline: replication stage — see repository README
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "depmap", "spatial_autocorrelation.json")


def morans_i(x, coords, k=8):
    """Moran's I over the k nearest spatial neighbours (cKDTree)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(x)))
    idx = idx[:, 1:]
    z = x - x.mean()
    n = len(x)
    w_sum = 0.0
    num = 0.0
    for i in range(n):
        for j in idx[i]:
            if j < n:
                w_sum += 1.0
                num += z[i] * z[j]
    denom = np.sum(z ** 2)
    return (n / w_sum) * (num / denom) if denom > 0 else np.nan


def block_perm_rho(df, n_block=10, n_perm=999, seed=49):
    """Block permutation of the core-to-margin gradient (10x10 blocks, fixed seed).

    Shuffles NetITH within spatial blocks (not globally) to give a space-aware
    empirical p for the Spearman correlation of NetITH vs dist_norm.
    """
    rng = np.random.RandomState(seed)
    x = df["x"].values; y = df["y"].values
    d = df["dist_norm"].values; nt = df["NetITH"].values
    bx = np.digitize(x, np.quantile(x, np.linspace(0, 1, n_block + 1)[1:-1]))
    by = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_block + 1)[1:-1]))
    block = bx * n_block + by
    obs = stats.spearmanr(d, nt)[0]
    nulls = []
    for _ in range(n_perm):
        perm_nt = nt.copy()
        for b in np.unique(block):
            m = block == b
            perm_nt[m] = rng.permutation(perm_nt[m])
        nulls.append(stats.spearmanr(d, perm_nt)[0])
    nulls = np.array(nulls)
    emp_p = (np.sum(np.abs(nulls) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, emp_p, nulls, block


res = {}
for label, path in [("BRCA", os.path.join(ROOT, "results", "depmap", "spatial_visium_netith.csv")),
                    ("OV", os.path.join(ROOT, "results", "depmap", "spatial_visium_OV_netith.csv"))]:
    df = pd.read_csv(path)
    # align alternative column names (spatial_x/spatial_y -> x/y)
    if "spatial_x" in df.columns:
        df = df.rename(columns={"spatial_x": "x", "spatial_y": "y"})
    x = df["x"].values; y = df["y"].values
    nt = df["NetITH"].values
    d = df["dist_norm"].values
    rho_nom, p_nom = stats.spearmanr(d, nt)
    mi = morans_i(nt, np.column_stack([x, y]))
    obs, emp_p, nulls, block = block_perm_rho(df)
    res[label] = {"n_spots": int(len(df)),
                  "gradient_rho_nominal": float(rho_nom),
                  "gradient_p_nominal": float(p_nom),
                  "morans_i_netith": float(mi),
                  "block_perm_empirical_p": float(emp_p),
                  "block_perm_null_mean": float(nulls.mean()),
                  "block_perm_null_sd": float(nulls.std()),
                  "n_blocks": int(len(np.unique(block)))}
    print(f"[{label}] n={len(df)} rho={rho_nom:.3f} p_nom={p_nom:.2e} Moran I={mi:.3f} "
          f"block-perm emp_p={emp_p:.3f} (null {nulls.mean():.3f}+-{nulls.std():.3f})")

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print("written:", OUT)
