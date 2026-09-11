"""run_tf_subset_50pct.py — random 50% TF-subset robustness of NetITH and drug associations (SI-M1).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    For B=50 random 50% subsets of the focused TFs, recomputes NetITH over the
    1,013 GDSC cell lines keeping only edges whose TF is in the subset (the
    node set is unchanged), then re-evaluates the drug-sensitivity association
    across the same compounds. Reports per-subset median Spearman rho vs
    ln(IC50), count of rho>0 drugs, and the correlation of subset NetITH with
    the full-set NetITH, quantifying the robustness of the primary association.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - data/gdsc/rna_expr.csv + ensg_symbol_map.csv + cell_annot.csv: GDSC expression & names
    - results/focused_genes_collectri.txt: focused gene list
    - /tmp/collectri_net.pkl: pickled CollecTRI network (prepared upstream)
    - $DATA_ROOT/gdsc_download/GDSC2_IC50_all.csv: GDSC2 lnIC50 (cell line x drug)

Outputs:
    - results/control/tf_subset_50pct_per_subset.csv: per-subset NetITH/drug metrics
    - results/control/tf_subset_50pct_summary.json: summary (SI-M1 / SI Figure 3)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_tf_subset_50pct.py
"""
import os
import numpy as np, pandas as pd, pickle, sys, json
from pathlib import Path
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
# Pickled CollecTRI network (source/target/weight; prepared upstream — see README).
COLLECTRI_PKL = "/tmp/collectri_net.pkl"

SEED = 42
B = 50
np.random.seed(SEED)

DATA_DIR = f"{ROOT}/data/gdsc"
GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
OUT = Path(f"{ROOT}/results/control")
OUT.mkdir(parents=True, exist_ok=True)

# Step 1: load GDSC expression, rename columns to cell-line names, collapse duplicates
# 1. Expression (ENSG x CEL) -> cell-line names
expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
cell_line_map = {}
for cel, row in annot.iterrows():
    cl = str(row.get('Characteristics.cell.line.', ''))
    if cl and cl != 'nan' and cl != 'NA':
        cell_line_map[cel] = cl
common_cels = [c for c in expr_raw.columns if c in cell_line_map]
expr_named = expr_raw[common_cels].copy()
expr_named.columns = [cell_line_map[c] for c in common_cels]
# collapse duplicate cell-line names (mean)
expr_named = expr_named.T.groupby(level=0).mean().T

# Map ENSG ids to gene symbols
# gene id -> symbol
gmap = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
id2sym = dict(zip(gmap.iloc[:, 0], gmap.iloc[:, 1]))
expr_sym = expr_named.rename(index=id2sym)

# Step 2: focused TF list from results/, CollecTRI network from pickle
# 2. Focused genes + CollecTRI net
focused = set(l.strip() for l in open(f"{ROOT}/results/focused_genes_collectri.txt") if l.strip())
with open(COLLECTRI_PKL, 'rb') as f:
    net = pickle.load(f)
tfs = sorted({r['source'] for _, r in net.iterrows() if r['source'] in focused})
print(f"Focused TFs: {len(tfs)}")

# Step 3: lnIC50 matrix; same >=30-obs per-drug rule as run_gdsc_drug_sensitivity.py
# 3. IC50 matrix (cell line x drug, LN_IC50); drugs kept per-drug by >=30 valid obs (main-script rule)
ic50 = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
print(f"IC50: {ic50_mat.shape[0]} cell lines x {ic50_mat.shape[1]} drugs (pre-filter)")

# Shared cell lines between expression and IC50
# shared cell lines
shared = sorted(set(expr_sym.columns) & set(ic50_mat.index))
print(f"Shared cell lines: {len(shared)}")
expr_shared = expr_sym[shared]
ic50_shared = ic50_mat.loc[shared]

# NetITH for a gene set + edge list: weighted adjacency w*|z_i||z_j|, entropy of
# the normalized Laplacian spectrum per cell (vectorized over cells)
def netith_for(genes, edges):
    """Vectorized-ish per-cell NetITH for a gene set + edge list."""
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    vals = expr_shared.loc[genes].values.T  # cells x genes
    mean = vals.mean(axis=0); std = vals.std(axis=0) + 1e-10
    z = np.clip((vals - mean) / std, -3, 3)
    # edge list as idx pairs
    eidx = [(gene_to_idx[e['tf']], gene_to_idx[e['tgt']], e['w']) for e in edges
            if e['tf'] in gene_to_idx and e['tgt'] in gene_to_idx]
    ent = np.full(len(shared), np.nan)
    for i in range(len(shared)):
        A = np.zeros((n, n))
        for (ti, tj, w) in eidx:
            A[ti, tj] = w * abs(z[i, ti]) * abs(z[i, tj])
        A = A + A.T
        deg = A.sum(axis=1); trace = deg.sum()
        if trace < 1e-10:
            ent[i] = 0.0; continue
        eigs = np.clip(eigvalsh(np.diag(deg) - A), 0, None)
        rho = np.clip(eigs / (trace + 1e-10), 1e-12, 1.0)
        ent[i] = -np.sum(rho * np.log2(rho))
    return ent

# Full-set edges: TF in the focused list, both endpoints in the expression matrix
# full-set edges (TF in focused, both endpoints present)
all_edges = []
for _, r in net.iterrows():
    tf, tgt = r['source'], r['target']
    if tf in focused and tgt in focused:
        all_edges.append({'tf': tf, 'tgt': tgt, 'w': float(r.get('weight', 1.0))})
print(f"Full-set edges: {len(all_edges)}")

# Reference: full-set NetITH for comparison with the subsets
# full-set NetITH (reference)
full_genes = sorted(focused & set(expr_shared.index))
print(f"Full gene set: {len(full_genes)}")
full_netith = netith_for(full_genes, all_edges)
print(f"Full NetITH range: [{np.nanmin(full_netith):.3f}, {np.nanmax(full_netith):.3f}]")

# Per-drug Spearman rho vs lnIC50 with the >=30-obs rule (identical to main script)
# drug rho for full set (per-drug dropna, >=30 obs; identical rule to run_gdsc_drug_sensitivity.py)
def drug_rho(netith_vec, drug):
    x = netith_vec; y = ic50_shared[drug].values
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 30:
        return np.nan
    return spearmanr(x[m], y[m])[0]

drugs = list(ic50_shared.columns)
full_rho = np.array([drug_rho(full_netith, d) for d in drugs])
keep = ~np.isnan(full_rho)
drugs = [d for d, k in zip(drugs, keep) if k]
full_rho = full_rho[keep]
print(f"Drugs with >=30 obs: {len(drugs)}")
print(f"Full: median rho={np.median(full_rho):.3f}, rho>0: {(full_rho>0).sum()}/{len(drugs)}")

# Step 4: B=50 random 50% TF subsets; keep the full node set, subset TF edges only,
# then recompute NetITH, drug rho, and correlation with full-set NetITH
# 50 random 50% TF subsets
rows = []
for b in range(B):
    sub_tfs = set(np.random.choice(tfs, size=len(tfs)//2, replace=False))
    sub_edges = [e for e in all_edges if e['tf'] in sub_tfs]
    sub_genes = sorted(set(full_genes))  # keep full node set; only TF edges subset
    s_netith = netith_for(sub_genes, sub_edges)
    s_rho = np.array([drug_rho(s_netith, d) for d in drugs])
    with_full = spearmanr(s_netith, full_netith)[0]
    rows.append({
        'subset': b, 'n_tfs': len(sub_tfs), 'n_edges': len(sub_edges),
        'median_rho': np.median(s_rho), 'n_pos': int((s_rho > 0).sum()),
        'rho_with_full': with_full,
        'netith_min': float(np.nanmin(s_netith)), 'netith_max': float(np.nanmax(s_netith)),
    })
    print(f"subset {b}: edges={len(sub_edges)} median_rho={np.median(s_rho):.3f} pos={int((s_rho>0).sum())}/{len(drugs)} rho_full={with_full:.3f}")

# Save per-subset table and summary JSON (SI-M1 / SI Figure 3)
res = pd.DataFrame(rows)
res.to_csv(OUT / 'tf_subset_50pct_per_subset.csv', index=False)
summary = {
    'n_subsets': B, 'full_median_rho': float(np.median(full_rho)),
    'full_n_pos': int((full_rho > 0).sum()), 'n_drugs': len(drugs),
    'subset_median_rho_median': float(np.median(res.median_rho)),
    'subset_median_rho_range': [float(res.median_rho.min()), float(res.median_rho.max())],
    'subset_n_pos_min': int(res.n_pos.min()), 'subset_n_pos_max': int(res.n_pos.max()),
    'subset_rho_with_full_median': float(np.median(res.rho_with_full)),
    'subset_rho_with_full_range': [float(res.rho_with_full.min()), float(res.rho_with_full.max())],
    'subset_netith_min': [float(res.netith_min.min()), float(res.netith_min.max())],
    'subset_netith_max': [float(res.netith_max.min()), float(res.netith_max.max())],
}
json.dump(summary, open(OUT / 'tf_subset_50pct_summary.json', 'w'), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
