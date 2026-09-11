"""run_expression_permutation.py — expression permutation: fix topology, shuffle z (B-2.3).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-23

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Recomputes NetITH for the focused 239-gene CollecTRI subgraph (613 edges in
    GDSC) from z-scored expression, then permutes the expression values either
    gene-wise (each gene's z shuffled across samples) or sample-wise (whole
    cell vectors shuffled) B=100 times and recomputes NetITH each time. The
    real rho(NetITH, published NetITH) is compared against the permutation null
    to show that the signal requires the real expression-topology coupling.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - data/gdsc/rna_expr.csv + ensg_symbol_map.csv + cell_annot.csv: GDSC expression & names
    - data/collectri_network.csv: CollecTRI TF-target prior with signed weights
    - results/focused_genes_collectri.txt: focused gene list
    - results/gdsc/gdsc_netith_cell_lines.csv: published GDSC NetITH (reference)

Outputs:
    - results/control/expression_permutation.json: real rho, null distributions, empirical p (B-2.3)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_expression_permutation.py
"""
from pathlib import Path
import os
import pandas as pd, numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr
import pickle

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
# Load GDSC expression; map ENSG -> symbols and rename columns to cell-line names
expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
ensg = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg.columns=['ensg','symbol']
sym = dict(zip(ensg['ensg'], ensg['symbol']))
annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
cl_map = {cel: str(row.get('Characteristics.cell.line.','')) for cel,row in annot.iterrows()}
common_cels = [c for c in expr.columns if c in cl_map]
en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
en = en.loc[:, ~en.columns.duplicated()]
matched = en.index.isin(ensg['ensg'])
en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
en_sym = en_sym[~en_sym.index.duplicated(keep='first')]

# CollecTRI prior + focused gene list -> gene-indexed signed edge list
net = pd.read_csv(f'{ROOT}/data/collectri_network.csv')
focused = set(l.strip() for l in open(f'{ROOT}/results/focused_genes_collectri.txt') if l.strip())
common = sorted(focused & set(en_sym.index))
gi = {g:i for i,g in enumerate(common)}
edges = [(gi[r['source']], gi[r['target']], float(r['weight'])) for _, r in net.iterrows()
         if r['source'] in gi and r['target'] in gi]
E = np.array([(a,b) for a,b,_ in edges]); W = np.array([float(w) for _,_,w in edges])  # signed, as main pipeline
n_genes = len(common)
expr_sub = en_sym.loc[common]
# Z-score each gene across cells (clip to [-3,3]); cells x genes matrix
Z = np.clip(((expr_sub.T - expr_sub.mean(axis=1)) / (expr_sub.std(axis=1)+1e-10)).values, -3, 3)
cells = expr_sub.columns.tolist()
print(f"genes {n_genes}, edges {len(E)}, cells {len(Z)}")

# NetITH from a z-matrix: adjacency A_ij = w_ij*|z_i||z_j|, entropy of the
# normalized Laplacian spectrum (same construction as the main pipeline)
def netith_from_z(Z):
    out = np.zeros(len(Z))
    src, tgt = E[:,0], E[:,1]
    for c in range(len(Z)):
        z = Z[c]
        wgt = W * np.abs(z[src]) * np.abs(z[tgt])
        A = np.zeros((n_genes, n_genes))
        np.add.at(A, (src, tgt), wgt)
        A = A + A.T
        D = A.sum(axis=1); tr = D.sum()
        if tr < 1e-10: continue
        eigs = np.clip(eigvalsh(np.diag(D) - A), 0, None)
        rho = np.clip(eigs / tr, 1e-12, 1.0)
        out[c] = -np.sum(rho * np.log2(rho))
    return out

# Real NetITH on observed z; sanity check against published GDSC NetITH
real = netith_from_z(Z)
netith_gdsc = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH']
r_real, _ = spearmanr(real, netith_gdsc.reindex(cells))
print(f"real vs published NetITH: rho={r_real:.4f}")

# Permutation null (B=100, seed 42): gene-wise shuffle breaks gene-sample coupling
# while keeping the edge set fixed; sample-wise shuffle permutes whole cell vectors
RNG = np.random.default_rng(42)
# gene-wise shuffle: permute each gene's z values across samples
N_PERM = 100
rs_gene, rs_sample = [], []
for k in range(N_PERM):
    Zg = Z.copy()
    for j in range(Zg.shape[1]):
        Zg[:, j] = RNG.permutation(Zg[:, j])
    s = netith_from_z(Zg)
    rs_gene.append(spearmanr(s, netith_gdsc.reindex(cells))[0])
    # sample-wise shuffle: permute cells' z vectors
    Zs = Z[RNG.permutation(len(Z))]
    s2 = netith_from_z(Zs)
    rs_sample.append(spearmanr(s2, netith_gdsc.reindex(cells))[0])

# Empirical p: fraction of permuted |rho| >= |real rho|, per shuffle scheme.
rs_gene, rs_sample = np.array(rs_gene), np.array(rs_sample)
print(f"\ngene-wise shuffle: rho vs published: {rs_gene.mean():.3f} ± {rs_gene.std():.3f} (real {r_real:.3f})")
print(f"sample-wise shuffle: {rs_sample.mean():.3f} ± {rs_sample.std():.3f}")
# FIX(review-batch-A): empirical p cannot be exactly 0 with a finite null;
# use (count_ge + 1)/(B + 1), the standard finite-null convention (same as the
# 1/51 null-model p in SI-N14).
B = 100
p_gene = (np.abs(rs_gene) >= np.abs(r_real)).sum() + 1
p_samp = (np.abs(rs_sample) >= np.abs(r_real)).sum() + 1
p_gene = p_gene / (B + 1)
p_samp = p_samp / (B + 1)
print(f"empirical p (finite-null, +1/(B+1)): gene-wise {p_gene:.4f}, sample-wise {p_samp:.4f}")

import json, os
os.makedirs(f'{ROOT}/results/control', exist_ok=True)
# Save real-vs-null summary to results/control/
json.dump({'real_rho': float(r_real), 'gene_shuffle_mean': float(rs_gene.mean()),
           'gene_shuffle_sd': float(rs_gene.std()), 'sample_shuffle_mean': float(rs_sample.mean()),
           'sample_shuffle_sd': float(rs_sample.std()),
           'p_gene': float(p_gene), 'p_sample': float(p_samp)},
          open(f'{ROOT}/results/control/expression_permutation.json','w'), indent=2)
print("saved results/control/expression_permutation.json")
