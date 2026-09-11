"""run_control_random_graph_null.py — degree-preserving random-network null model (Control 1, GDSC).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Pipeline: control / test stage — see repository README

Question: is the NetITH-drug-association signal (286/286 rho>0, median 0.232)
attributable to the curated CollecTRI topology, or would any network with the
same degree sequence reproduce it given the same expression data?

NetITH is recomputed per cell line for each of N_REWIRE degree-preserving
rewired networks (double-edge swaps, signed CollecTRI weights carried with
their edges), using the exact main-pipeline formula (symmetrized Laplacian,
eigvalsh(L), rho=eigs/trace, signed weights). Spearman rho vs ln(IC50) is
recomputed for all 286 drugs; real-network values are compared with the null
distribution.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - /tmp/nullmodel_focused_edges_signed.npy
    - /tmp/nullmodel_exprz.npy
    - /tmp/nullmodel_cells.npy
    - /tmp/nullmodel_ic50.csv
    - results/control/random_graph_null_summary.json
Outputs:
    - results/control/random_graph_null_per_rewire.csv
Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_control_random_graph_null.py
"""
import numpy as np, pandas as pd, json, os
from pathlib import Path
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr
import time

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
OUT = f"{ROOT}/results/control"
# Intermediate null-model artifacts prepared upstream (see README; staged under /tmp).
EDGES_NPY = '/tmp/nullmodel_focused_edges_signed.npy'  # (E,3) tf,target,SIGNED weight
EXPRZ_NPY = '/tmp/nullmodel_exprz.npy'                 # cells × genes (per-gene z, clip3)
CELLS_NPY = '/tmp/nullmodel_cells.npy'
IC50_CSV = '/tmp/nullmodel_ic50.csv'

edges = np.load(EDGES_NPY)
exprz = np.load(EXPRZ_NPY)
cells = list(np.load(CELLS_NPY))
ic50 = pd.read_csv(IC50_CSV, index_col=0)
N_REWIRE = 50
RNG = np.random.default_rng(42)

n_genes = exprz.shape[1]
E0 = np.array([(int(e[0]), int(e[1])) for e in edges])
W0 = np.array([float(e[2]) for e in edges])   # signed

def netith_from_edges(E, W, exprz):
    """Exact main-pipeline NetITH (signed weights, symmetrized L, eigvalsh(L))."""
    n_cells, n_genes = exprz.shape
    out = np.zeros(n_cells)
    src, tgt = E[:,0], E[:,1]
    for c in range(n_cells):
        z = exprz[c]
        wgt = W * np.abs(z[src]) * np.abs(z[tgt])
        Ac = np.zeros((n_genes, n_genes))
        np.add.at(Ac, (src, tgt), wgt)
        Ac = Ac + Ac.T
        deg = Ac.sum(axis=1); trace = deg.sum()
        if trace < 1e-10:
            continue
        eigs = eigvalsh(np.diag(deg) - Ac)
        eigs = np.clip(eigs, 0, None)
        rho = np.clip(eigs / trace, 1e-12, 1.0)
        out[c] = -np.sum(rho * np.log2(rho))
    return out

def degree_preserving_rewire(E, W, rng, n_swap=100):
    """Degree-preserving random network (configuration model).

    Exact in/out degree sequence preserved; parallel edges allowed (reported);
    self-loops rejected via a derangement-style fix; signed weights are
    randomly permuted onto the rewired topology (weight distribution kept,
    weight-gene associations broken). This mixes fully even for the highly
    heterogeneous degree sequence of CollecTRI (verified: null-null overlap
    ~= null-original overlap ~19-22%).
    """
    E = E.copy(); W = W.copy()
    n = int(E.max()) + 1
    m = len(E)
    out_deg = np.bincount(E[:,0], minlength=n)
    in_deg = np.bincount(E[:,1], minlength=n)
    out_stubs = np.repeat(np.arange(n), out_deg)
    in_stubs = np.repeat(np.arange(n), in_deg)
    for _ in range(100):
        rng.shuffle(in_stubs)
        bad = np.where(out_stubs == in_stubs)[0]
        it = 0
        while len(bad) > 0 and it < 500:
            i = bad[0]; j = rng.integers(0, m)
            if i == j or in_stubs[j] == out_stubs[i] or in_stubs[i] == out_stubs[j]:
                it += 1; continue
            in_stubs[i], in_stubs[j] = in_stubs[j], in_stubs[i]
            bad = np.where(out_stubs == in_stubs)[0]; it += 1
        if len(bad) == 0:
            rng.shuffle(W)  # break weight-gene association
            return np.column_stack([out_stubs, in_stubs]), W
    raise RuntimeError('configuration model failed')

# --- real network ---
t0 = time.time()
real_netith = netith_from_edges(E0, W0, exprz)
real_s = pd.Series(real_netith, index=cells)
print(f"real NetITH done ({time.time()-t0:.0f}s)")

def drug_rho(netith_s):
    out = {}
    for d in ic50.columns:
        y = ic50[d].dropna()
        common = netith_s.index.intersection(y.index)
        if len(common) < 30:
            out[d] = np.nan; continue
        r, _ = spearmanr(netith_s.loc[common], y.loc[common])
        out[d] = r
    return pd.Series(out)

real_rho = drug_rho(real_s)
n_pos_real = int((real_rho > 0).sum())
med_real = float(real_rho.median())
print(f"REAL: median rho={med_real:.3f}, rho>0 {n_pos_real}/{len(real_rho)}")

rows = []
for k in range(N_REWIRE):
    t0 = time.time()
    Ew, Ww = degree_preserving_rewire(E0, W0, RNG)
    nw = netith_from_edges(Ew, Ww, exprz)
    nw_s = pd.Series(nw, index=cells)
    rho_w = drug_rho(nw_s)
    rows.append({
        'rewire': k+1,
        'median_rho': float(rho_w.median()),
        'n_pos': int((rho_w > 0).sum()),
        'n_pos_frac': float((rho_w > 0).mean()),
        'netith_mean': float(nw.mean()),
        'netith_std': float(nw.std()),
    })
    print(f"  rewire {k+1}/{N_REWIRE}: med rho={rows[-1]['median_rho']:.3f}, "
          f"pos {rows[-1]['n_pos']}/286 ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows)
null_pos = df['n_pos'].values
# Empirical one-sided p: fraction of null rewirings with n_pos >= the real
# network's n_pos (a real network is extreme if no null reaches its count).
p_pos = float((np.array([n_pos_real]) <= null_pos).mean())  # empirical p
summary = {
    'real_median_rho': med_real,
    'real_n_pos': n_pos_real,
    'real_n_pos_frac': float((real_rho > 0).mean()),
    'null_median_rho_mean': float(df['median_rho'].mean()),
    'null_median_rho_sd': float(df['median_rho'].std()),
    'null_n_pos_mean': float(df['n_pos'].mean()),
    'null_n_pos_sd': float(df['n_pos'].std()),
    'null_max_n_pos': int(df['n_pos'].max()),
    'empirical_p_npos': p_pos,
    'null_netith_mean': float(df['netith_mean'].mean()),
    'real_netith_mean': float(real_netith.mean()),
    'real_netith_std': float(real_netith.std()),
    'n_rewire': int(N_REWIRE),
}
os.makedirs(OUT, exist_ok=True)
df.to_csv(f'{OUT}/random_graph_null_per_rewire.csv', index=False)
with open(f'{OUT}/random_graph_null_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
