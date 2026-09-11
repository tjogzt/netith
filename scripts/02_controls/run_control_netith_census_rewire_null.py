#!/usr/bin/env python3
"""run_control_netith_census_rewire_null.py — NetITH under the census-style rewired null (round-5 verdict).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-10
Pipeline: control / test stage — see repository README

Follow-up A (round-5 verdict, gatekeeper for the journal tier decision):

NetITH (239-gene focused CollecTRI signed Laplacian entropy) evaluated under
the SAME rewired-null design used for the signaling-entropy / DNE rows of the
four-test census -- i.e. `rewire_weighted` from run_descriptor_census.py
(undirected double-edge swap, degree-preserving, signed weights shuffled onto
the rewired topology, 50 draws, seed 42).

This makes the Test-1 null strictly comparable across descriptors (the
manuscript's existing Test-1 null is a directed configuration-model rewire,
same family but a different implementation; median 0.130).

Real network: exact main-pipeline formula (directed signed CollecTRI edges,
w = e_w * |z_tf| * |z_tg|, A = A + A.T, L = D - A, eigvalsh(L), non-negative
spectrum, trace normalisation, -sum rho log2 rho) -- identical to
replication_common.vn_entropy_per_cell / run_control_random_graph_null.py.

Null networks: undirected topology U from the non-self-loop CollecTRI pairs
(collapsed signed weight = sum of directed weights), rewired with the census
function; per rewire the shuffled signed weights are applied to BOTH directions
of each undirected edge, then the same per-cell formula.

Outputs:
    results/control/census/null_netith_census_rewire.json
    results/control/census/null_netith_census_rewire_per_draw.csv

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_control_netith_census_rewire_null.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "R"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replication_common as rc
from run_descriptor_census import load_ic50

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "control" / "census"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
N_REWIRE = 50


def netith_from_edges(E, W, exprz):
    """Exact main-pipeline NetITH (signed weights, symmetrised L, eigvalsh)."""
    n_cells, n_genes = exprz.shape
    out = np.zeros(n_cells)
    src, tgt = E[:, 0], E[:, 1]
    for c in range(n_cells):
        z = exprz[c]
        wgt = W * np.abs(z[src]) * np.abs(z[tgt])
        Ac = np.zeros((n_genes, n_genes))
        np.add.at(Ac, (src, tgt), wgt)
        Ac = Ac + Ac.T
        deg = Ac.sum(axis=1)
        trace = deg.sum()
        if trace < 1e-10:
            continue
        eigs = eigvalsh(np.diag(deg) - Ac)
        eigs = np.clip(eigs, 0, None)
        rho = np.clip(eigs / trace, 1e-12, 1.0)
        out[c] = -np.sum(rho * np.log2(rho))
    return out


def drug_rho(netith_s, ic50_mat):
    out = {}
    for d in ic50_mat.columns:
        y = ic50_mat[d].dropna()
        common = netith_s.index.intersection(y.index)
        if len(common) < 30:
            out[d] = np.nan
            continue
        out[d] = spearmanr(netith_s.loc[common], y.loc[common])[0]
    return pd.Series(out)


def rewire_preserving(A, rng, n_attempts=8000):
    """Verified degree-preserving double-edge swap with an edge-count invariant.

    All four endpoints must be distinct and both cross-edges absent; the edge
    list is updated in lockstep with the matrix. (The census helper
    `rewire_weighted` was empirically found to leak edges on sparse graphs --
    573 -> 549-561 in synthetic tests -- so this verified version is used for
    the 574-edge NetITH topology.)
    """
    edges = np.argwhere(np.triu(A > 0, 1))
    m = len(edges)
    swaps_done = 0
    for _ in range(n_attempts):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a0, a1 = edges[i]
        b0, b1 = edges[j]
        if len({a0, a1, b0, b1}) < 4:
            continue
        if A[a0, b1] or A[a1, b0]:
            continue
        A[a0, a1] = A[a1, a0] = 0
        A[b0, b1] = A[b1, b0] = 0
        A[a0, b1] = A[b1, a0] = 1
        A[a1, b0] = A[b0, a1] = 1
        edges[i] = [a0, b1]
        edges[j] = [a1, b0]
        swaps_done += 1
    assert int((A > 0).sum() / 2) == m, "edge count not preserved"
    return A, swaps_done


def main():
    # ---- GDSC expression first (defines the gene universe) ----
    from run_descriptor_census import load_gdsc_expr
    expr = load_gdsc_expr()
    focused_all = sorted(rc.focused_gene_set())
    common = [g for g in focused_all if g in expr.index]
    print(f"[setup] focused genes present in GDSC: {len(common)}/{len(focused_all)}")
    focused = common

    # ---- signed CollecTRI edges restricted to the GDSC-present genes ----
    net = pd.read_csv(ROOT / "data" / "collectri_network.csv")
    g2i = {g: i for i, g in enumerate(focused)}
    rows = [(g2i[r["source"]], g2i[r["target"]], float(r["weight"]))
            for _, r in net.iterrows()
            if r["source"] in g2i and r["target"] in g2i]
    E0 = np.array([(a, b) for a, b, _ in rows])
    W0 = np.array([w for _, _, w in rows])
    n_genes = len(focused)
    print(f"[setup] genes={n_genes}, directed signed edges={len(rows)} "
          f"({int((W0 < 0).sum())} negative)")

    # ---- GDSC expression (census loader), focused rows, cohort z-clip ----
    sub = expr.loc[focused]                      # genes x cells
    X = sub.values.astype(float).T               # cells x genes
    mu, sd = X.mean(0), X.std(0) + 1e-10
    exprz = np.clip((X - mu) / sd, -3, 3)
    cells = list(sub.columns)
    print(f"[setup] GDSC cells={len(cells)}")

    # ---- IC50 matrix ----
    ic50_mat, _ = load_ic50()

    # ---- real network (pipeline formula) ----
    t0 = time.time()
    real = netith_from_edges(E0, W0, exprz)
    real_s = pd.Series(real, index=cells)
    real_rho = drug_rho(real_s, ic50_mat)
    print(f"[real] median rho={real_rho.median():.3f}, "
          f"pos {(real_rho > 0).sum()}/{real_rho.notna().sum()} ({time.time()-t0:.0f}s)")

    # ---- undirected topology U + collapsed signed weights ----
    wmap = {}
    for (a, b), w in zip(E0, W0):
        if a == b:
            continue                            # self-loops excluded from U (documented)
        key = (min(a, b), max(a, b))
        wmap[key] = wmap.get(key, 0.0) + w
    U = np.array(sorted(wmap.keys()))
    WU = np.array([wmap[tuple(k)] for k in U])
    A_u = np.zeros((n_genes, n_genes))
    A_u[U[:, 0], U[:, 1]] = A_u[U[:, 1], U[:, 0]] = 1.0
    print(f"[setup] undirected topology: {len(U)} edges "
          f"({int((WU < 0).sum())} negative collapsed weights; "
          f"{len(rows) - len(U)} self-loops excluded)")

    # ---- 50 census-style rewirings (verified swap; same design family as
    #      the SR/DNE rows of the census: undirected double-edge swap,
    #      degree-preserving, signed weights shuffled) ----
    draws = []
    for k in range(N_REWIRE):
        t0 = time.time()
        Aw, n_swaps = rewire_preserving(A_u.copy(), RNG)
        edges_w = np.argwhere(np.triu(Aw > 0, 1))
        rng2 = np.random.default_rng(42 + k)
        rng2.shuffle(WU)
        Ew = np.vstack([edges_w, edges_w[:, ::-1]])      # both directions
        Ww = np.concatenate([WU, WU])
        nw = netith_from_edges(Ew, Ww, exprz)
        nw_s = pd.Series(nw, index=cells)
        rho = drug_rho(nw_s, ic50_mat)
        draws.append({"draw": k + 1, "median_rho": float(rho.median()),
                      "n_pos": int((rho > 0).sum()), "n_swaps": n_swaps})
        print(f"  draw {k+1}/{N_REWIRE}: median rho={rho.median():.4f} "
              f"pos {(rho > 0).sum()} (swaps={n_swaps}, {time.time()-t0:.0f}s)")

    meds = np.array([d["median_rho"] for d in draws])
    # Empirical two-sided p: fraction of null draws with |median rho| >= |real|,
    # using the finite-null (+1)/(n+1) convention so p is never exactly 0.
    emp_p = (1 + int((np.abs(meds) >= abs(real_rho.median())).sum())) / (N_REWIRE + 1)
    out = {
        "real": {"median_rho": float(real_rho.median()),
                 "n_pos": int((real_rho > 0).sum()),
                 "n_drugs": int(real_rho.notna().sum())},
        "null": {"design": "census rewire_weighted (undirected double-edge swap, "
                           "degree-preserving, signed weights shuffled, 50 draws, seed 42)",
                 "n": N_REWIRE,
                 "mean": float(meds.mean()),
                 "sd": float(meds.std()),
                 "range": [float(meds.min()), float(meds.max())]},
        "emp_p": float(emp_p),
        "manuscript_reference_null": {"design": "directed configuration-model rewire "
                                               "(manuscript Test-1, cached)",
                                      "median_rho": 0.130, "range": [0.049, 0.199],
                                      "emp_p": 1 / 51},
    }
    json.dump(out, open(OUT_DIR / "null_netith_census_rewire.json", "w"), indent=2)
    pd.DataFrame(draws).to_csv(OUT_DIR / "null_netith_census_rewire_per_draw.csv",
                               index=False)
    print(f"[done] real={out['real']['median_rho']:.3f} vs census-style null "
          f"mean={meds.mean():.3f} range=[{meds.min():.3f},{meds.max():.3f}] "
          f"emp_p={emp_p:.4f}")


if __name__ == "__main__":
    main()
