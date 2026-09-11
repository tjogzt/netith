"""
replication_common.py — shared utilities for the scRNA-seq replication scripts (09, 11–18).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : results/focused_genes_collectri.txt (239-gene CollecTRI focused set),
          data/collectri_network.csv (signed CollecTRI TF→target edges)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : (none — imported as `replication_common` by the sibling scripts)
Pipeline: replication stage — see repository README
"""
import gzip, os, json
import numpy as np
import pandas as pd
from scipy import stats, linalg

SEED = 49
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("NETITH_DATA_ROOT", os.path.join(ROOT, "data"))


def new_rng():
    """Fresh RandomState seeded with the project seed (49)."""
    return np.random.RandomState(SEED)


def focused_gene_set():
    """239-gene CollecTRI focused set from results/focused_genes_collectri.txt."""
    with open(os.path.join(ROOT, "results", "focused_genes_collectri.txt")) as f:
        return {l.strip() for l in f if l.strip()}


def load_expression_matrix(gz_path):
    """Load a gzipped gene x cell TSV/CSV as a pandas DataFrame (genes x cells)."""
    with gzip.open(gz_path, "rt", errors="replace") as f:
        return pd.read_csv(f, sep="\t", index_col=0)


def save_focused_cache(cache_path, expr_df, cell_meta=None, **extra_arrays):
    """Persist a focused expression matrix plus optional per-cell arrays."""
    np.savez(cache_path,
             expr=expr_df.values.astype(np.float32),
             genes=np.array(expr_df.index, dtype=object),
             cells=np.array(expr_df.columns, dtype=object),
             **{k: np.array(v, dtype=object) for k, v in (extra_arrays or {}).items()})


def load_focused_cache(cache_path):
    """Load a focused cache written by save_focused_cache.

    Returns (expr_df, extra_dict) where expr_df is genes x cells.
    """
    arr = np.load(cache_path, allow_pickle=True)
    df = pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"])
    extra = {k: arr[k] for k in arr.files if k not in ("expr", "genes", "cells")}
    return df, extra


def build_edges(common_genes):
    """Signed CollecTRI edges restricted to common genes.

    Returns (e_tf, e_tg, e_w, n_genes) numpy arrays (edge vectors + gene count).
    """
    net = pd.read_csv(os.path.join(ROOT, "data", "collectri_network.csv"))
    g2i = {g: i for i, g in enumerate(common_genes)}
    edges = [(g2i[r["source"]], g2i[r["target"]], float(r["weight"]))
             for _, r in net.iterrows() if r["source"] in g2i and r["target"] in g2i]
    e_tf = np.array([e[0] for e in edges])
    e_tg = np.array([e[1] for e in edges])
    e_w = np.array([e[2] for e in edges])
    return e_tf, e_tg, e_w, len(common_genes)


def vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes):
    """Per-cell von-Neumann entropy of the signed weighted Laplacian.

    z: cells x genes clipped z-scores. Weights w = e_w * |z_tf| * |z_tg|.
    Entropy from L = D - A on the symmetrised signed adjacency (only the
    non-negative part of the spectrum, normalised by trace(L)).
    """
    n_cells = z.shape[0]
    ent = np.zeros(n_cells)
    for i in range(n_cells):
        za = np.abs(z[i, e_tf]) * np.abs(z[i, e_tg])
        w = e_w * za
        k = w != 0
        if k.sum() == 0:
            continue
        A = np.zeros((n_genes, n_genes))
        A[e_tf[k], e_tg[k]] = w[k]
        A = A + A.T
        deg = A.sum(1)
        L = np.diag(deg) - A
        tr = deg.sum()
        if tr < 1e-10:
            continue
        eigs = linalg.eigvalsh(L)
        rho = np.clip(np.clip(eigs, 0, None) / (tr + 1e-10), 1e-12, 1.0)
        ent[i] = -np.sum(rho * np.log2(rho))
    return ent


def cohort_zscore(expr_values, clip=3.0):
    """Cohort-level z-scores (mean/std over cells per gene), clipped."""
    mu = expr_values.mean(axis=0)
    sd = expr_values.std(axis=0) + 1e-10
    return np.clip((expr_values - mu) / sd, -clip, clip).astype(np.float32)


def shannon_hist(x, n_bins=20):
    """Shannon entropy (base 2) of a histogram of x with n_bins."""
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return 0.0
    counts, _ = np.histogram(x, bins=n_bins, range=(lo, hi))
    p = (counts + 1e-10) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def contrast(a, b, label_a="A", label_b="B", n_boot=2000, seed=SEED):
    """Two-group contrast: pooled-SD Cohen d, 95% bootstrap CI, Mann-Whitney p.

    Returns a dict {label, n_a, n_b, cohen_d, ci95, mw_p}; NaN when n < 3 either
    side. Bootstrap uses a fresh RandomState(seed) per call.
    """
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    if len(a) < 3 or len(b) < 3:
        return {"label": f"{label_a} vs {label_b}", "n_a": int(len(a)), "n_b": int(len(b)),
                "cohen_d": np.nan, "ci95": [np.nan, np.nan], "mw_p": np.nan}
    d = (a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ba = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        boots[i] = (ba.mean() - bb.mean()) / np.sqrt(((len(ba) - 1) * ba.var(ddof=1) + (len(bb) - 1) * bb.var(ddof=1)) / (len(ba) + len(bb) - 2))
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    return {"label": f"{label_a} vs {label_b}", "n_a": int(len(a)), "n_b": int(len(b)),
            "cohen_d": float(d), "ci95": [float(ci[0]), float(ci[1])], "mw_p": float(p)}
