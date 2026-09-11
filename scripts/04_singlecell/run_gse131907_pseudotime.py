"""run_gse131907_pseudotime.py — GSE131907 diffusion pseudotime: NetITH along the lung differentiation trajectory.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz; results/focused_genes_collectri.txt; results/gse131907/cell_entropy_results.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/figures/gse131907_pseudotime_netith.png; results/depmap/{gse131907_pseudotime_origin.csv, gse131907_pseudotime_cells.csv}
Pipeline: single-cell stage — see repository README
"""
import os

import numpy as np
import pandas as pd
import scanpy as sc
import subprocess, pickle, os, sys, warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

GSE_DIR = Path(f"{ROOT}/results/gse131907")
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 1. Extract epithelial cell expression
# ═══════════════════════════════════════════════════════════
print("[1/5] Extracting epithelial cells from GSE131907...")

EXPR_PATH = f"{DATA_ROOT}/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz"
FOCUSED_FILE = f"{ROOT}/results/focused_genes_collectri.txt"

with open(FOCUSED_FILE) as f:
    target_genes = set(line.strip() for line in f if line.strip())

# Load cell entropy + annotations
cell_entropy = pd.read_csv(GSE_DIR / "cell_entropy_results.csv")
cell_entropy['full_barcode'] = cell_entropy['Barcode'] + '_' + cell_entropy['Sample']

# Restrict to epithelial cells for the differentiation-trajectory analysis
# Filter to epithelial cells only
epi_cells = cell_entropy[cell_entropy['Cell_type'] == 'Epithelial cells'].copy()
epi_barcodes = set(epi_cells['full_barcode'].values)
print(f"  Epithelial cells: {len(epi_barcodes)}")

# Stream the gzipped matrix row by row; keep only focused-gene rows for the matched epithelial cells
# Stream expression for epithelial cells
proc = subprocess.Popen(
    ['gunzip', '-c', EXPR_PATH],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1024*1024
)

header_line = proc.stdout.readline()
header = header_line.strip().split('\t')
col_to_idx = {name: i for i, name in enumerate(header)}
keep_cells = [c for c in epi_barcodes if c in col_to_idx]
keep_indices = [col_to_idx[c] for c in keep_cells]
print(f"  Matched: {len(keep_cells)} cells")

gene_rows = {}
for line in proc.stdout:
    gene = line.split('\t', 1)[0]
    if gene in target_genes:
        parts = line.strip().split('\t')
        values = [float(parts[ci]) if ci < len(parts) else 0.0 for ci in keep_indices]
        gene_rows[gene] = values
        if len(gene_rows) == len(target_genes):
            break
proc.terminate()
proc.wait()

gene_list = sorted(gene_rows.keys())
print(f"  Genes: {len(gene_list)}")

# Assemble the cells x genes expression matrix aligned to the entropy table
expr_arr = np.column_stack([gene_rows[g] for g in gene_list])
expr_df = pd.DataFrame(expr_arr, index=keep_cells, columns=gene_list)

# Align with entropy
epi_sub = epi_cells[epi_cells['full_barcode'].isin(keep_cells)].set_index('full_barcode')
epi_sub = epi_sub.loc[keep_cells]
n_cells = len(keep_cells)
print(f"  Expression: {n_cells} cells × {len(gene_list)} genes")

# ═══════════════════════════════════════════════════════════
# 2. Scanpy preprocessing + diffusion pseudotime
# ═══════════════════════════════════════════════════════════
print(f"\n[2/5] Scanpy preprocessing ({n_cells} cells)...")

adata = sc.AnnData(expr_df.values)
adata.obs_names = keep_cells
adata.var_names = gene_list
adata.obs['netith'] = epi_sub['vn_entropy'].values
adata.obs['sample_origin'] = epi_sub['Sample_Origin'].values

# Normalize + log
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=100, flavor='seurat_v3')
adata = adata[:, adata.var.highly_variable]

# PCA
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack', n_comps=min(30, n_cells-1, len(gene_list)-1))

# Neighbors
sc.pp.neighbors(adata, n_neighbors=min(15, n_cells-1), n_pcs=min(20, adata.obsm['X_pca'].shape[1]))

# Diffusion pseudotime — root = normal lung epithelial cell
print("\n[3/5] Computing diffusion pseudotime...")

# Root: normal lung epithelial cell with lowest NetITH (most "normal-like")
normal_mask = adata.obs['sample_origin'] == 'nLung'
if normal_mask.sum() > 0:
    # DPT root = normal lung (nLung) epithelial cell with the lowest NetITH
    root_idx = adata.obs.loc[normal_mask, 'netith'].idxmin()
else:
    root_idx = adata.obs['netith'].idxmin()

root_cell = list(adata.obs_names).index(root_idx)
print(f"  Root cell: {root_idx} (nLung, NetITH={adata.obs.loc[root_idx, 'netith']:.3f})")

# scanpy 1.11.x API: set iroot in uns, then call dpt
adata.uns['iroot'] = root_cell
# Diffusion pseudotime: diffusion-map distances define cell ordering along the
# differentiation trajectory, rooted at the normal-lung reference cell (DPT = distance from root).
sc.tl.diffmap(adata, n_comps=10)
sc.tl.dpt(adata, n_branchings=0)

print(f"  Pseudotime range: [{adata.obs['dpt_pseudotime'].min():.3f}, {adata.obs['dpt_pseudotime'].max():.3f}]")

# ═══════════════════════════════════════════════════════════
# 4. NetITH × Pseudotime correlation
# ═══════════════════════════════════════════════════════════
print("\n[4/5] NetITH × Pseudotime analysis...")

pt = adata.obs['dpt_pseudotime'].values
vn = adata.obs['netith'].values

# Spearman test NetITH vs pseudotime across all epithelial cells (null: rho = 0)
rho_all, p_all = spearmanr(pt, vn)
print(f"  Overall NetITH×Pseudotime: ρ={rho_all:.3f}, p={p_all:.2e}")

# By tissue origin
origins = adata.obs['sample_origin'].unique()
# Repeat the Spearman test within each tissue origin (n >= 10 cells)
origin_results = []
for origin in origins:
    mask = adata.obs['sample_origin'] == origin
    if mask.sum() < 10:
        continue
    rho, p = spearmanr(pt[mask], vn[mask])
    origin_results.append({
        'origin': origin, 'rho': rho, 'p': p, 'n': mask.sum()
    })

origin_df = pd.DataFrame(origin_results)
for _, row in origin_df.iterrows():
    print(f"    {row['origin']:8s}: ρ={row['rho']:+.3f}, p={row['p']:.3e}, n={row['n']}")

# Sliding window: NetITH mean ± SD along pseudotime
n_bins = 20
# Binned NetITH mean +/- SD along pseudotime (sliding-window profile)
bins = np.linspace(pt.min(), pt.max(), n_bins + 1)
bin_means, bin_stds, bin_centers = [], [], []
for i in range(n_bins):
    mask = (pt >= bins[i]) & (pt < bins[i+1])
    if mask.sum() >= 5:
        bin_means.append(vn[mask].mean())
        bin_stds.append(vn[mask].std())
        bin_centers.append((bins[i] + bins[i+1]) / 2)

# ═══════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════
print("\n[5/5] Creating visualizations...")

# 6-panel figure: pseudotime/NetITH on PCA, scatter, binned profile, by origin
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('GSE131907 Epithelial Cell Pseudotime: NetITH Along Differentiation Trajectory',
             fontsize=14, fontweight='bold', y=0.99)

# Panel A: Pseudotime in PCA space
ax = axes[0, 0]
pca_coords = adata.obsm['X_pca']
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=pt, cmap='viridis', s=3, alpha=0.6, edgecolors='none')
ax.set_xlabel(f'PC1', fontsize=10)
ax.set_ylabel(f'PC2', fontsize=10)
ax.set_title(f'A: Pseudotime on PCA\n({n_cells} epithelial cells)', fontsize=10)
plt.colorbar(scatter, ax=ax, label='Pseudotime', shrink=0.8)

# Panel B: NetITH on pseudotime
ax = axes[0, 1]
scatter2 = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                      c=vn, cmap='RdBu_r', s=3, alpha=0.6, edgecolors='none')
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)
ax.set_title(f'B: NetITH on PCA\n(ρ×PT={rho_all:+.3f})', fontsize=10)
plt.colorbar(scatter2, ax=ax, label='NetITH', shrink=0.8)

# Panel C: NetITH vs Pseudotime scatter
ax = axes[0, 2]
ax.scatter(pt, vn, c=pt, cmap='viridis', s=5, alpha=0.4, edgecolors='none')
# Smoothed trend
sort_idx = np.argsort(pt)
window = max(20, n_cells // 30)
smooth_pt = np.convolve(pt[sort_idx], np.ones(window)/window, mode='valid')
smooth_vn = np.convolve(vn[sort_idx], np.ones(window)/window, mode='valid')
ax.plot(smooth_pt, smooth_vn, 'r-', linewidth=2, alpha=0.8)
ax.set_xlabel('Diffusion Pseudotime', fontsize=11)
ax.set_ylabel('NetITH', fontsize=11)
ax.set_title(f'C: NetITH vs Pseudotime\nρ={rho_all:+.3f}, p={p_all:.2e}', fontsize=10)

# Panel D: Binned NetITH along pseudotime
ax = axes[1, 0]
if bin_centers:
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                fmt='o-', color='#8c564b', capsize=3, markersize=6, linewidth=1.5)
ax.axhline(y=np.median(vn), color='gray', linestyle=':', alpha=0.5, label=f'Median={np.median(vn):.3f}')
ax.set_xlabel('Pseudotime (binned)', fontsize=11)
ax.set_ylabel('Mean NetITH ± SD', fontsize=11)
ax.set_title('D: NetITH Along Pseudotime Bins', fontsize=10)
ax.legend(fontsize=8)

# Panel E: NetITH by tissue origin along pseudotime
ax = axes[1, 1]
origin_colors = {'nLung': '#1f77b4', 'tLung': '#d62728', 'tL/B': '#ff7f0e',
                 'nLN': '#2ca02c', 'mLN': '#9467bd', 'mBrain': '#8c564b', 'PE': '#e377c2'}
for origin in ['nLung', 'tLung', 'tL/B', 'mBrain']:
    o_mask = adata.obs['sample_origin'] == origin
    if o_mask.sum() > 10:
        ax.scatter(pt[o_mask], vn[o_mask],
                   c=origin_colors.get(origin, '#7f7f7f'),
                   s=3, alpha=0.5, label=f'{origin} (n={o_mask.sum()})')
ax.set_xlabel('Pseudotime', fontsize=11)
ax.set_ylabel('NetITH', fontsize=11)
ax.set_title('E: NetITH×PT by Tissue Origin', fontsize=10)
ax.legend(fontsize=7, markerscale=3)

# Panel F: Summary
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
GSE131907 Epithelial Pseudotime
NetITH Along Differentiation

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells: {n_cells} (Epithelial)
Genes: {len(gene_list)} focused CollecTRI
Root: {root_idx} (nLung, {adata.obs.loc[root_idx, 'netith']:.3f})

Overall NetITH×Pseudotime:
  ρ = {rho_all:+.4f}
  p = {p_all:.2e}

By Tissue Origin:
"""
for _, row in origin_df.iterrows():
    sig = '*' if row['p'] < 0.05 else ''
    summary_text += f"  {row['origin']:8s}: ρ={row['rho']:+.3f}{sig} (n={row['n']})\n"

summary_text += f"""
Hypothesis Test:
If high NetITH = differentiation
potential, then NetITH should
PEAK at intermediate pseudotime
(branch points), not monotonically
increase.
"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=7, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUTPUT_DIR / "figures" / "gse131907_pseudotime_netith.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# SAVE
# Save per-origin stats and per-cell pseudotime/NetITH tables
origin_df.to_csv(OUTPUT_DIR / "gse131907_pseudotime_origin.csv", index=False)
pseudotime_out = pd.DataFrame({
    'cell': keep_cells,
    'pseudotime': pt,
    'netith': vn,
    'sample_origin': adata.obs['sample_origin'].values,
})
pseudotime_out.to_csv(OUTPUT_DIR / "gse131907_pseudotime_cells.csv", index=False)

print("\n" + "=" * 70)
print("GSE131907 PSEUDOTIME — COMPLETE")
print("=" * 70)
print(f"""
  Epithelial cells: {n_cells}
  NetITH×Pseudotime: ρ={rho_all:+.3f}, p={p_all:.2e}
""")
