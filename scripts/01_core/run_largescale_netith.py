"""
run_largescale_netith.py — Validate the focused 239-gene NetITH against an ~800-gene CollecTRI NetITH.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/gdsc/rna_expr.csv: GDSC expression (ENSG rows x cell-line columns)
    - data/gdsc/cell_annot.csv: cell-line annotation (array-ID to cell-line-name mapping)
    - data/gdsc/ensg_symbol_map.csv: ENSG -> gene-symbol mapping
    - results/focused_genes_collectri.txt: focused 239-gene CollecTRI set
    - results/gdsc/gdsc_netith_cell_lines.csv: focused-set NetITH per cell line
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 drug sensitivity (LN_IC50)
    - <NETITH_COLLECTRI_PKL, default /tmp/collectri_net.pkl>: pickled CollecTRI network
Outputs :
    - results/depmap/netith_large_800.csv, netith_focused_vs_large.csv, drug_large_vs_focused_netith.csv, randomized_svd_entropy_validation.csv
    - results/depmap/figures/largescale_netith_comparison.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""

import os

import numpy as np
import pandas as pd
import pickle, os, sys, warnings
from pathlib import Path
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr, pearsonr
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
COLLECTRI_PKL = Path(os.environ.get("NETITH_COLLECTRI_PKL", "/tmp/collectri_net.pkl"))

def load_collectri():
    """Return the CollecTRI network DataFrame, building the pickle from the
    shipped data/collectri_network.csv when NETITH_COLLECTRI_PKL is absent."""
    if not Path(COLLECTRI_PKL).exists():
        Path(COLLECTRI_PKL).parent.mkdir(parents=True, exist_ok=True)
        net = pd.read_csv(Path(ROOT) / "data" / "collectri_network.csv")
        with open(COLLECTRI_PKL, "wb") as fh:
            pickle.dump(net, fh)
    with open(COLLECTRI_PKL, "rb") as fh:
        return pickle.load(fh)




DATA_DIR = DATA_ROOT / "gdsc"
GDSC_OUTPUT = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
MIN_DRUG_CELLS = 30  # minimum cell lines per drug for the large-vs-focused comparison
N_SVD_CELLS = 30        # number of cells used to benchmark the randomized-SVD approximation
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 1. Build Full CollecTRI Gene Set
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("[1/5] Building full CollecTRI gene set...")

# Load GDSC expression
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
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]

# Gene symbol mapping
gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
ensg_to_sym = dict(zip(gene_map['ensg'].astype(str), gene_map['symbol']))
expr_named.index = expr_named.index.astype(str)

matched = expr_named.index.isin(ensg_to_sym.keys())
expr_sym = expr_named.loc[matched].copy()
expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]

# Load the CollecTRI TF-target network (pickled DataFrame of source/target/weight edges)
# Load CollecTRI
with open(COLLECTRI_PKL, 'rb') as f:
    net = pickle.load(f)

all_collectri_genes = set(net['source']) | set(net['target'])
gdsc_genes = set(expr_sym.index)
collectri_in_gdsc = sorted(all_collectri_genes & gdsc_genes)
print(f"  CollecTRI genes in GDSC: {len(collectri_in_gdsc)}")

# Load focused genes for comparison
focused_file = f"{ROOT}/results/focused_genes_collectri.txt"
with open(focused_file) as f:
    focused = set(line.strip() for line in f if line.strip())

# Select top N genes by expression variance × mean
# Rank CollecTRI genes by variance x mean expression to select the ~800-gene set
expr_sub_all = expr_sym.loc[collectri_in_gdsc]
gene_var = expr_sub_all.var(axis=1)
gene_mean = expr_sub_all.mean(axis=1)
gene_score = gene_var * gene_mean  # prioritize highly expressed + variable genes

N_LARGE = 800  # target gene set size
top_genes = gene_score.nlargest(N_LARGE).index.tolist()
print(f"  Large gene set: {len(top_genes)} (top by var×mean)")
print(f"  Focused genes in large set: {len(set(top_genes) & focused)}/{len(focused)}")

# Build CollecTRI edges for large gene set
# Subset the CollecTRI network to edges whose both endpoints are in the large gene set
gene_to_idx_large = {g: i for i, g in enumerate(top_genes)}
edges_large = []
tf_set_large = set()
for _, row in net.iterrows():
    tf = row['source']; target = row['target']
    if tf in gene_to_idx_large and target in gene_to_idx_large:
        edges_large.append({
            'tf': tf, 'target': target,
            'tf_idx': gene_to_idx_large[tf],
            'target_idx': gene_to_idx_large[target],
            'weight': float(row.get('weight', 1.0)),
        })
        tf_set_large.add(tf)

print(f"  Large GRN: {len(edges_large)} edges, {len(tf_set_large)} TFs")
n_genes_large = len(top_genes)

# ═══════════════════════════════════════════════════════════
# 2. Compute Large-Scale NetITH
# ═══════════════════════════════════════════════════════════
print("\n[2/5] Computing large-scale NetITH (von Neumann entropy)...")

# Prepare expression matrix
expr_large = expr_sym.loc[top_genes]
cell_names = expr_large.columns.tolist()
n_cells = len(cell_names)

# Z-score
# Z-score the 800-gene expression matrix per gene, clipped to [-3, 3]
expr_vals = expr_large.values.T
expr_mean = expr_vals.mean(axis=0)
expr_std = expr_vals.std(axis=0) + 1e-10
expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)

# Build adjacency matrix per cell and compute entropy
# Per cell line: edge weight = CollecTRI weight x |z(TF)| x |z(target)|, then entropy over the graph Laplacian
n_edges = len(edges_large)
entropies_large = np.zeros(n_cells)

t0 = time.time()
for cell_i in range(n_cells):
    # Build adjacency matrix
    A = np.zeros((n_genes_large, n_genes_large))
    for e in edges_large:
        w = e['weight'] * abs(expr_z[cell_i, e['tf_idx']]) * abs(expr_z[cell_i, e['target_idx']])
        if w > 0:
            A[e['tf_idx'], e['target_idx']] += w
    
    # Symmetrize (undirected GRN)
    A = A + A.T
    
    # Laplacian: L = D - A
    D_diag = A.sum(axis=1)
    trace_L = D_diag.sum()
    
    if trace_L > 1e-10:
        L = np.diag(D_diag) - A
        # Density matrix
        rho = L / trace_L
        # von Neumann entropy S = -sum(lambda * log2(lambda)) over positive eigenvalues
        # von Neumann entropy
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
    else:
        S = 0.0
    
    entropies_large[cell_i] = S
    
    if (cell_i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (cell_i + 1) / elapsed
        remaining = (n_cells - cell_i - 1) / rate
        print(f"  {cell_i+1}/{n_cells} cells ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

elapsed_total = time.time() - t0
print(f"  Complete: {n_cells} cells in {elapsed_total:.0f}s ({elapsed_total/n_cells:.3f}s/cell)")

large_netith = pd.Series(entropies_large, index=cell_names, name='NetITH_large')

# ═══════════════════════════════════════════════════════════
# 3. Load Focused NetITH & Compare
# ═══════════════════════════════════════════════════════════
print("\n[3/5] Comparing with focused-gene NetITH...")

netith_focused = pd.read_csv(GDSC_OUTPUT / "gdsc_netith_cell_lines.csv", index_col=0)
netith_focused = netith_focused['NetITH']

# Align
common_cells = sorted(set(large_netith.index) & set(netith_focused.index))
large_aligned = large_netith.loc[common_cells]
focused_aligned = netith_focused.loc[common_cells]

print(f"  Common cells: {len(common_cells)}")

# Concordance tests between large- and focused-set NetITH (null: no monotonic / linear association)
# Correlation
r_s, p_s = spearmanr(large_aligned, focused_aligned)
r_p, p_p = pearsonr(large_aligned, focused_aligned)
print(f"  Spearman ρ = {r_s:.4f} (p={p_s:.2g})")
print(f"  Pearson r  = {r_p:.4f} (p={p_p:.2g})")

# Rank preservation: what fraction of top/bottom quartile preserved?
# Quartile preservation: fraction of top/bottom-quartile cell lines shared between the two gene sets
n_q = len(common_cells) // 4
top_focused = set(focused_aligned.nlargest(n_q).index)
top_large = set(large_aligned.nlargest(n_q).index)
bottom_focused = set(focused_aligned.nsmallest(n_q).index)
bottom_large = set(large_aligned.nsmallest(n_q).index)

top_overlap = len(top_focused & top_large) / n_q
bottom_overlap = len(bottom_focused & bottom_large) / n_q
print(f"  Top quartile overlap: {top_overlap:.1%}")
print(f"  Bottom quartile overlap: {bottom_overlap:.1%}")

# ═══════════════════════════════════════════════════════════
# 4. Drug Sensitivity Comparison
# ═══════════════════════════════════════════════════════════
print("\n[4/5] Large-scale NetITH vs Drug Sensitivity...")

# Load GDSC2 LN_IC50 and pivot to a cell-line x drug matrix
# Load IC50
GDSC_DIR_IC50 = f"{DATA_ROOT}/gdsc_download"
ic50_raw = pd.read_csv(f"{GDSC_DIR_IC50}/GDSC2_IC50_all.csv")
ic50_mat = ic50_raw.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)

drug_cells = sorted(set(common_cells) & set(ic50_mat.index))
print(f"  Drug data cells: {len(drug_cells)}")

large_drug = large_aligned.loc[drug_cells]
focused_drug = focused_aligned.loc[drug_cells]
ic50_drug = ic50_mat.loc[drug_cells]

# Compare drug correlations: large NetITH vs focused NetITH
# Per drug (n >= 30 cell lines): Spearman r of NetITH (large and focused) with LN_IC50
drug_comparison = []
for drug in ic50_drug.columns:
    ic50_vals = ic50_drug[drug].dropna()
    common_idx = ic50_vals.index.intersection(drug_cells)
    if len(common_idx) < MIN_DRUG_CELLS:
        continue
    
    r_large, p_large = spearmanr(large_drug.loc[common_idx], ic50_vals.loc[common_idx])
    r_focused, p_focused = spearmanr(focused_drug.loc[common_idx], ic50_vals.loc[common_idx])
    
    drug_comparison.append({
        'drug': drug,
        'r_large': r_large,
        'r_focused': r_focused,
        'delta_r': r_large - r_focused,
        'n': len(common_idx)
    })

drug_df = pd.DataFrame(drug_comparison)
drug_df['abs_delta'] = drug_df['delta_r'].abs()

print(f"  Drugs compared: {len(drug_df)}")
print(f"  Mean |Δr|: {drug_df['abs_delta'].mean():.4f}")
print(f"  Max |Δr|: {drug_df['abs_delta'].max():.4f}")

# Spearman concordance of per-drug effects between the two gene sets (null: no association)
# Correlation between large and focused drug effects
r_drug, p_drug = spearmanr(drug_df['r_large'], drug_df['r_focused'])
print(f"  Drug effect correlation (large vs focused): ρ = {r_drug:.4f}")

# Drugs where large NetITH outperforms
better_large = drug_df[drug_df['delta_r'].abs() > 0.05].sort_values('delta_r')
print(f"  Drugs with |Δr| > 0.05: {len(better_large)}")
if len(better_large) > 0:
    print(f"  Top 5 drugs (large NetITH has stronger signal):")
    for _, row in better_large.head(5).iterrows():
        print(f"    {row['drug']:30s}: r_large={row['r_large']:+.3f}, r_focused={row['r_focused']:+.3f}, Δ={row['delta_r']:+.3f}")

# ═══════════════════════════════════════════════════════════
# 5. Randomized SVD Approximation Validation
# ═══════════════════════════════════════════════════════════
print("\n[5/5] Validating randomized SVD entropy approximation...")

from sklearn.utils.extmath import randomized_svd

# Test on a subset of cells: how well does k-rank approximation preserve entropy?
# Benchmark k-rank randomized-SVD entropy against exact eigenvalues on a random subset of 30 cells
test_cells = np.random.choice(n_cells, min(N_SVD_CELLS, n_cells), replace=False)
approx_results = []

for cell_i in test_cells:
    # Build exact density matrix
    A = np.zeros((n_genes_large, n_genes_large))
    for e in edges_large:
        w = e['weight'] * abs(expr_z[cell_i, e['tf_idx']]) * abs(expr_z[cell_i, e['target_idx']])
        if w > 0:
            A[e['tf_idx'], e['target_idx']] += w
    A = A + A.T
    D_diag = A.sum(axis=1)
    trace_L = D_diag.sum()
    if trace_L < 1e-10:
        continue
    
    L = np.diag(D_diag) - A
    rho = L / trace_L
    
    # Exact entropy
    eig_exact = eigvalsh(rho)
    eig_exact = eig_exact[eig_exact > 1e-12]
    S_exact = -np.sum(eig_exact * np.log2(eig_exact))
    
    # Approximate with k-rank SVD
    for k in [10, 20, 50, 100, 200]:
        try:
            U, S_svd, Vt = randomized_svd(rho, n_components=min(k, n_genes_large-1),
                                          random_state=SEED)
            # Normalized singular values → approximate eigenvalues
            sv_norm = S_svd / S_svd.sum()
            
            # Tail approximation: remaining mass distributed uniformly
            tail_mass = 1.0 - sv_norm.sum()
            n_tail = n_genes_large - k
            if n_tail > 0 and tail_mass > 1e-12:
                tail_eig = tail_mass / n_tail
                S_approx = -np.sum(sv_norm * np.log2(np.maximum(sv_norm, 1e-15)))
                S_approx += -tail_mass * np.log2(max(tail_eig, 1e-15))
            else:
                S_approx = -np.sum(sv_norm[sv_norm > 1e-12] * np.log2(sv_norm[sv_norm > 1e-12]))
            
            approx_results.append({
                'cell_idx': cell_i,
                'k': k,
                'S_exact': S_exact,
                'S_approx': S_approx,
                'error': abs(S_exact - S_approx),
                'rel_error': abs(S_exact - S_approx) / max(S_exact, 1e-10)
            })
        except Exception as ex:
            pass

approx_df = pd.DataFrame(approx_results)

if len(approx_df) > 0:
    print(f"\n  Randomized SVD entropy approximation:")
    for k in sorted(approx_df['k'].unique()):
        sub = approx_df[approx_df['k'] == k]
        print(f"    k={k:3d}: mean abs error={sub['error'].mean():.4f}, rel error={sub['rel_error'].mean():.4f}")

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Saving results...")

# Write all result tables: large NetITH, focused-vs-large, drug comparison, SVD validation
# Save large NetITH
large_netith_df = pd.DataFrame({'cell_line': large_netith.index, 'NetITH_large': large_netith.values})
large_netith_df.to_csv(OUTPUT_DIR / "netith_large_800.csv", index=False)

# Save comparison
comparison_df = pd.DataFrame({
    'cell_line': common_cells,
    'NetITH_focused': focused_aligned.values,
    'NetITH_large': large_aligned.values,
})
comparison_df.to_csv(OUTPUT_DIR / "netith_focused_vs_large.csv", index=False)

# Drug comparison
drug_df.to_csv(OUTPUT_DIR / "drug_large_vs_focused_netith.csv", index=False)

# SVD approximation results
approx_df.to_csv(OUTPUT_DIR / "randomized_svd_entropy_validation.csv", index=False)

# ═══════════════════════════════════════════════════════════
# VISUALIZATION — 6-panel figure
# ═══════════════════════════════════════════════════════════
print("Creating visualizations...")

# Six-panel figure: scatter, rank preservation, drug effects, distributions, SVD accuracy, summary
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.suptitle('Large-Scale NetITH (800 genes) vs Focused (239 genes)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Scatter: focused vs large NetITH
ax = axes[0, 0]
ax.scatter(focused_aligned, large_aligned, c='steelblue', alpha=0.3, s=15, edgecolors='none')
z = np.polyfit(focused_aligned, large_aligned, 1)
p_line = np.poly1d(z)
x_range = np.linspace(focused_aligned.min(), focused_aligned.max(), 100)
ax.plot(x_range, p_line(x_range), 'r-', linewidth=2)
ax.set_xlabel('Focused NetITH (239 genes)', fontsize=11)
ax.set_ylabel('Large NetITH (800 genes)', fontsize=11)
ax.set_title(f'Focused vs Large NetITH\nρ = {r_s:.3f}, r = {r_p:.3f}', fontsize=10)
# Identity line
lims = [min(focused_aligned.min(), large_aligned.min()), max(focused_aligned.max(), large_aligned.max())]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

# Panel B: Rank preservation
ax = axes[0, 1]
ranks = np.arange(1, len(common_cells)+1)
ax.scatter(focused_aligned.rank(), large_aligned.rank(), c='#ff7f0e', alpha=0.3, s=10, edgecolors='none')
ax.plot([1, len(common_cells)], [1, len(common_cells)], 'k--', alpha=0.5)
ax.set_xlabel('Focused NetITH Rank', fontsize=11)
ax.set_ylabel('Large NetITH Rank', fontsize=11)
ax.set_title(f'Rank Preservation\nTop Q overlap={top_overlap:.1%}, Bot Q={bottom_overlap:.1%}', fontsize=10)

# Panel C: Drug correlation comparison
ax = axes[0, 2]
ax.scatter(drug_df['r_focused'], drug_df['r_large'], c='#2ca02c', alpha=0.3, s=12, edgecolors='none')
lims_d = [min(drug_df['r_focused'].min(), drug_df['r_large'].min()),
          max(drug_df['r_focused'].max(), drug_df['r_large'].max())]
ax.plot(lims_d, lims_d, 'k--', alpha=0.5)
ax.set_xlabel('r (Focused NetITH × IC50)', fontsize=11)
ax.set_ylabel('r (Large NetITH × IC50)', fontsize=11)
ax.set_title(f'Drug Sensitivity: Focused vs Large\nρ = {r_drug:.3f}', fontsize=10)
ax.axhline(y=0, color='gray', alpha=0.3)
ax.axvline(x=0, color='gray', alpha=0.3)

# Panel D: Distribution comparison
ax = axes[1, 0]
ax.hist(focused_aligned, bins=40, alpha=0.5, label='Focused (239)', color='#1f77b4', density=True)
ax.hist(large_aligned, bins=40, alpha=0.5, label='Large (800)', color='#ff7f0e', density=True)
ax.set_xlabel('NetITH', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('NetITH Distribution: Focused vs Large', fontsize=10)
ax.legend(fontsize=9)

# Panel E: SVD approximation accuracy
ax = axes[1, 1]
if len(approx_df) > 0:
    ks = sorted(approx_df['k'].unique())
    means = [approx_df[approx_df['k']==k]['rel_error'].mean() for k in ks]
    stds = [approx_df[approx_df['k']==k]['rel_error'].std() for k in ks]
    ax.errorbar(ks, means, yerr=stds, marker='o', capsize=4, color='#d62728', linewidth=2)
    ax.set_xlabel('k (rank of SVD approximation)', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Randomized SVD Entropy Approximation', fontsize=10)
    ax.set_xscale('log')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% error')
    ax.legend(fontsize=9)

# Panel F: Summary
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
Large-Scale NetITH: 800 vs 239 Genes

━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRN: {len(edges_large)} edges, {len(tf_set_large)} TFs, {n_genes_large} genes

Correlation (Focused vs Large):
  Spearman ρ = {r_s:.4f}
  Pearson r  = {r_p:.4f}

Quartile Preservation:
  Top 25%:    {top_overlap:.1%}
  Bottom 25%: {bottom_overlap:.1%}

Drug Sensitivity:
  Compared:   {len(drug_df)} drugs
  Mean |Δr|:  {drug_df['abs_delta'].mean():.4f}
  Drug ρ:     {r_drug:.4f}

Computation (800 genes):
  Time:       {elapsed_total:.0f}s ({elapsed_total/n_cells:.3f}s/cell)
  Cells:      {n_cells}

SVD Approximation (best k):
"""
if len(approx_df) > 0:
    best_k = min(approx_df.groupby('k')['rel_error'].mean().items(), key=lambda x: x[1])
    summary_text += f"  k={best_k[0]}: mean rel error={best_k[1]:.4f}\n"

summary_text += f"""
Conclusion:
  Focused (239) NetITH captures the essential
  signal with {'high' if r_s > 0.9 else 'moderate'} fidelity.
  Large-scale (800) provides {'significant' if drug_df['abs_delta'].mean() > 0.02 else 'marginal'} additional
  drug sensitivity resolution.
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=8, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = OUTPUT_DIR / "figures" / "largescale_netith_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LARGE-SCALE NetITH — COMPLETE")
print("=" * 70)
print(f"""
Results:
  Focused vs Large ρ:  {r_s:.4f}
  Top Q overlap:       {top_overlap:.1%}
  Drug ρ (foc vs lrg): {r_drug:.4f}
  
Output files:
  {OUTPUT_DIR}/netith_large_800.csv
  {OUTPUT_DIR}/netith_focused_vs_large.csv
  {OUTPUT_DIR}/drug_large_vs_focused_netith.csv
  {OUTPUT_DIR}/randomized_svd_entropy_validation.csv
  {OUTPUT_DIR}/figures/largescale_netith_comparison.png
""")
