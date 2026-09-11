"""run_spatial_visium_netith.py — BRCA Visium spatial NetITH via per-spot k-NN co-expression.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/GraphHMM/processed/multi_cancer/BRCA_standardized.h5ad; CollecTRI (via decoupler dc.op.collectri) (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{spatial_visium_netith.csv, spatial_visium_leiden_stats.csv, spatial_visium_immune_stats.csv, spatial_visium_radial_profile.csv, spatial_visium_immune_corrs.csv}; results/depmap/figures/{spatial_visium_netith_main.png, spatial_visium_krt_overlay.png}
Pipeline: spatial stage — see repository README
"""
import os

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr as sp_corr_func
from scipy.stats import pearsonr, mannwhitneyu, f_oneway
from scipy.spatial import KDTree
from scipy.sparse import issparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import warnings, os, sys, time
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Paths ────────────────────────────────────────────────────
BRCA_PATH = f"{DATA_ROOT}/GraphHMM/processed/multi_cancer/BRCA_standardized.h5ad"
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

K_NEIGHBORS = 50          # Spatial neighbors for local GRN estimation
SEED = 42
np.random.seed(SEED)

# ─── Load Data ──────────────────────────────────────────────
print("=" * 70)
print("[1/5] Loading BRCA Visium data...")
t0 = time.time()

# Load BRCA Visium h5ad: spots x genes, spatial coordinates and obs annotations
adata = sc.read_h5ad(BRCA_PATH)
print(f"  {adata.n_obs} spots × {adata.n_vars} genes")

# Extract dense expression
if issparse(adata.X):
    X_full = adata.X.toarray()
else:
    X_full = adata.X
coords = adata.obsm['spatial']  # (3377, 2)
obs = adata.obs

print(f"  Spatial range X: [{coords[:,0].min():.0f}, {coords[:,0].max():.0f}]")
print(f"  Spatial range Y: [{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]")
print(f"  Expression: min={X_full.min():.4f}, max={X_full.max():.4f}, "
      f"mean={X_full.mean():.4f}, sparsity={np.mean(X_full==0):.2%}")

# ─── Load & Filter CollecTRI ──────────────────────────────────
print("\n[2/5] Loading CollecTRI & filtering to BRCA gene overlap...")

# Load human CollecTRI, drop default-sign edges, filter to genes on the array
collectri = dc.op.collectri(organism='human')
collectri = collectri[~collectri['sign_decision'].str.startswith('default')]

brca_gene_set = set(adata.var_names)
collectri_brca = collectri[
    collectri['source'].isin(brca_gene_set) &
    collectri['target'].isin(brca_gene_set)
].copy()

tfs_brca = sorted(collectri_brca['source'].unique())
targets_brca = sorted(collectri_brca['target'].unique())
all_genes_brca = sorted(set(tfs_brca + targets_brca))

print(f"  CollecTRI edges in BRCA: {len(collectri_brca)}")
print(f"  TFs: {len(tfs_brca)}, Targets: {len(targets_brca)}")
print(f"  Unique genes: {len(all_genes_brca)}")

# Focus on top 300 targets (by TF in-degree) to keep computation tractable
# while retaining the strongest regulatory signals
# Focused set: all TFs + top 300 targets by in-degree + epithelial marker genes
tf_target_counts = collectri_brca['target'].value_counts()
top_targets = tf_target_counts.head(300).index.tolist()
# Ensure all TFs are included (some may be targets too)
# Add key epithelial/marker genes even if not in CollecTRI top targets
marker_genes = ['KRT8', 'KRT18', 'KRT19', 'EPCAM', 'VIM', 'MKI67',
                'CDH1', 'CDH2', 'COL1A1', 'ACTA2', 'PECAM1', 'CD3E',
                'CD8A', 'CD4', 'CD68', 'CD163']
marker_genes_found = [g for g in marker_genes if g in brca_gene_set]
focused_genes = sorted(set(tfs_brca + top_targets + marker_genes_found))
print(f"  Focused gene set: {len(focused_genes)} (all {len(tfs_brca)} TFs + top 300 targets)")

# Build focused CollecTRI edges
collectri_focused = collectri_brca[
    collectri_brca['source'].isin(focused_genes) &
    collectri_brca['target'].isin(focused_genes)
].copy()
n_edges_focused = len(collectri_focused)
print(f"  Focused edges: {n_edges_focused}")

# Map gene symbols → array indices for the focused set
gene_to_idx = {g: i for i, g in enumerate(focused_genes)}
# Also map for full BRCA genes (for expression extraction)
brca_var_names = list(adata.var_names)
gene_to_brca_idx = {g: brca_var_names.index(g) for g in focused_genes
                    if g in brca_var_names}

# Build edge index arrays (tf_idx → target_idx in focused gene space)
tf_indices = []
tgt_indices = []
for _, row in collectri_focused.iterrows():
    src = row['source']
    tgt = row['target']
    if src in gene_to_idx and tgt in gene_to_idx:
        tf_indices.append(gene_to_idx[src])
        tgt_indices.append(gene_to_idx[tgt])
tf_indices = np.array(tf_indices, dtype=np.int32)
tgt_indices = np.array(tgt_indices, dtype=np.int32)
n_edges = len(tf_indices)
print(f"  Edge index arrays: {n_edges} edges")

# Extract focused expression matrix (3377 × n_focused_genes)
n_spots = adata.n_obs
n_genes_focused = len(focused_genes)
X_focused = np.zeros((n_spots, n_genes_focused), dtype=np.float32)
for g, brca_idx in gene_to_brca_idx.items():
    j = gene_to_idx[g]
    X_focused[:, j] = X_full[:, brca_idx]

# Z-score normalize per gene (robust to outliers via clipping)
X_mean = np.mean(X_focused, axis=0)
X_std = np.std(X_focused, axis=0) + 1e-8
# Z-score each gene across spots, clipped to [-3, 3]
X_z = np.clip((X_focused - X_mean) / X_std, -3, 3)
del X_focused, X_full
print(f"  Z-scored expression: ready ({X_z.shape})")

# ─── Spatial KDTree ───────────────────────────────────────────
print("\n[3/5] Building spatial KDTree & computing per-spot NetITH...")
# KDTree over spot coordinates for spatial neighbour queries.
# Pooling each spot's k-nearest spatial neighbours estimates a *local* TF-target
# correlation matrix, i.e. the spatial autocorrelation of the GRN across the tissue.
tree = KDTree(coords)

# ─── Compute Per-Spot von Neumann Entropy ─────────────────────
# For each spot:
#   1. Find k nearest spatial neighbors
#   2. Rank-transform neighbor expression (for Spearman)
#   3. Compute TF×target |Spearman ρ| → adjacency A
#   4. Compute W = A @ A^T (gene co-regulation matrix)
#   5. L = diag(sum(W,1)) - W, ρ = L/tr(L)
#   6. S = -sum(eigenvalues * log2(eigenvalues))

def rankdata_matrix(X: np.ndarray) -> np.ndarray:
    """Rank-transform each column of X (for Spearman correlation)."""
    n, p = X.shape
    R = np.zeros_like(X, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        # argsort twice = rank (ties averaged)
        order = np.argsort(col)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        # Handle ties: average rank
        _, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
        for u in range(len(cnt)):
            mask = inv == u
            ranks[mask] = np.mean(ranks[mask])
        R[:, j] = ranks
    return R

def compute_spearman_fast(X: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix from rank-transformed, centered data.

    Args:
        X: (n_samples, n_features)

    Returns:
        R: (n_features, n_features) Spearman correlation matrix.
    """
    R = rankdata_matrix(X)
    # Center ranks
    R_c = R - R.mean(axis=0, keepdims=True)
    # Normalize
    norms = np.sqrt(np.sum(R_c ** 2, axis=0))
    norms[norms < 1e-12] = 1.0
    R_n = R_c / norms
    # Correlation = R_n.T @ R_n
    corr = R_n.T @ R_n
    # Clamp to [-1, 1]
    corr = np.clip(corr, -1.0, 1.0)
    return corr

def compute_vn_entropy(A: np.ndarray) -> float:
    """Compute von Neumann entropy from adjacency A.

    W = A @ A^T, L = diag(sum(W,1)) - W, S = -sum(λ log2 λ) on ρ=L/tr(L).
    """
    # W = A A^T; L = D - W; entropy of the normalised Laplacian rho = L/tr(L)
    W = A @ A.T
    D_diag = W.sum(axis=1)
    trace_L = D_diag.sum()
    if trace_L < 1e-12:
        return 0.0
    L = np.diag(D_diag) - W
    rho = L / trace_L
    eigenvalues = eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

# Map TFs to focused gene indices for building A matrix
tf_list_in_focused = [g for g in tfs_brca if g in gene_to_idx]
tf_focused_indices = np.array([gene_to_idx[g] for g in tf_list_in_focused], dtype=np.int32)
tgt_list_in_focused = [g for g in top_targets if g in gene_to_idx]
tgt_focused_indices = np.array([gene_to_idx[g] for g in tgt_list_in_focused], dtype=np.int32)
n_tfs_focused = len(tf_focused_indices)
n_tgts_focused = len(tgt_focused_indices)
print(f"  A matrix: {n_tfs_focused} TFs × {n_tgts_focused} targets")

spot_netith = np.zeros(n_spots, dtype=np.float64)
spot_n_edges_used = np.zeros(n_spots, dtype=np.int32)

t_start = time.time()
# Main per-spot loop: k-NN pooling -> |Spearman rho| adjacency -> von Neumann entropy
for spot_i in range(n_spots):
    # Find k nearest spatial neighbors (including self)
    dists, neighbor_idx = tree.query(coords[spot_i], k=K_NEIGHBORS+1)
    neighbors = neighbor_idx  # include self for density estimation

    # Extract neighbor expression
    X_neigh = X_z[neighbors, :]  # (k+1, n_genes_focused)

    # Spearman correlation matrix (only for focused genes)
    R_full = compute_spearman_fast(X_neigh)  # (n_genes_focused, n_genes_focused)

    # Extract A[i, j] = |ρ(TF_i, target_j)|
    A = np.abs(R_full[tf_focused_indices, :][:, tgt_focused_indices])

    # Count edges with non-zero correlation
    nz_edges = np.sum(A > 0.05)
    spot_n_edges_used[spot_i] = nz_edges

    # von Neumann entropy
    S = compute_vn_entropy(A)
    spot_netith[spot_i] = S

    if (spot_i + 1) % 500 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (spot_i + 1) * (n_spots - spot_i - 1)
        print(f"  Spot {spot_i+1}/{n_spots} | NetITH={S:.3f} | "
              f"active_edges={nz_edges} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

elapsed = time.time() - t_start
print(f"  Completed in {elapsed:.0f}s ({elapsed/n_spots:.2f}s/spot)")

# ─── Assemble Results ─────────────────────────────────────────
# Assemble results and compute distance from the tissue centroid (core->margin axis)
obs['NetITH'] = spot_netith
obs['NetITH_n_edges'] = spot_n_edges_used

# Compute distance from tissue centroid
centroid = np.median(coords, axis=0)
obs['dist_from_centroid'] = np.sqrt(
    (coords[:, 0] - centroid[0])**2 + (coords[:, 1] - centroid[1])**2
)

# Normalize distance to [0, 1]
obs['dist_norm'] = (obs['dist_from_centroid'] - obs['dist_from_centroid'].min()) / \
                   (obs['dist_from_centroid'].max() - obs['dist_from_centroid'].min() + 1e-8)

print(f"\n  NetITH summary: mean={spot_netith.mean():.3f}, "
      f"std={spot_netith.std():.3f}, "
      f"range=[{spot_netith.min():.3f}, {spot_netith.max():.3f}]")
print(f"  Active edges: mean={spot_n_edges_used.mean():.0f}, "
      f"min={spot_n_edges_used.min()}, max={spot_n_edges_used.max()}")

# ─── Spatial Gradient Analysis ────────────────────────────────
print("\n[4/5] Spatial gradient & association analyses...")

# 4a: NetITH by leiden cluster
print("\n  4a: NetITH by leiden cluster")
# 4a: summarise NetITH per leiden cluster
leiden_stats = []
for cl in sorted(obs['leiden_clusters'].unique()):
    sub = obs[obs['leiden_clusters'] == cl]
    leiden_stats.append({
        'cluster': cl,
        'n_spots': len(sub),
        'netith_mean': sub['NetITH'].mean(),
        'netith_std': sub['NetITH'].std(),
        'dist_mean': sub['dist_norm'].mean(),
        'KRT8_mean': X_z[obs['leiden_clusters'] == cl, gene_to_idx.get('KRT8', 0)].mean()
        if 'KRT8' in gene_to_idx else np.nan,
        'KRT19_mean': X_z[obs['leiden_clusters'] == cl, gene_to_idx.get('KRT19', 0)].mean()
        if 'KRT19' in gene_to_idx else np.nan,
    })
leiden_df = pd.DataFrame(leiden_stats)
print(leiden_df.to_string(index=False))

# Identify "core" clusters: highest KRT8+KRT19 expression
if 'KRT8' in gene_to_idx and 'KRT19' in gene_to_idx:
    leiden_df['epithelial_score'] = (leiden_df['KRT8_mean'] + leiden_df['KRT19_mean']) / 2
else:
    leiden_df['epithelial_score'] = leiden_df['netith_mean']  # fallback
# Define tumour core / margin clusters by the KRT8+KRT19 epithelial score
core_clusters = leiden_df.nlargest(3, 'epithelial_score')['cluster'].tolist()
margin_clusters = leiden_df.nsmallest(3, 'epithelial_score')['cluster'].tolist()
print(f"  Tumor core clusters (high epithelial): {core_clusters}")
print(f"  Margin-like clusters (low epithelial): {margin_clusters}")

# 4b: NetITH by immune state
print("\n  4b: NetITH by predicted immune state")
# 4b: NetITH by predicted immune state
immune_stats = []
for state in sorted(obs['predicted_immune_state'].unique()):
    sub = obs[obs['predicted_immune_state'] == state]
    immune_stats.append({
        'immune_state': state,
        'n_spots': len(sub),
        'netith_mean': sub['NetITH'].mean(),
        'netith_std': sub['NetITH'].std(),
    })
immune_df = pd.DataFrame(immune_stats)
print(immune_df.to_string(index=False))

# 4c: Radial NetITH gradient (binned by distance from centroid)
print("\n  4c: Radial NetITH profile")
obs['dist_bin'] = pd.cut(obs['dist_norm'], bins=15)
dist_profile = obs.groupby('dist_bin', observed=False).agg(
    netith_mean=('NetITH', 'mean'),
    netith_std=('NetITH', 'std'),
    netith_se=('NetITH', 'sem'),
    dist_mid=('dist_norm', 'mean'),
    n_spots=('NetITH', 'count')
).dropna().reset_index(drop=True)

print(dist_profile.to_string(index=False))

# Spearman correlation: NetITH vs distance
# 4c: Spearman NetITH vs normalised distance from centroid (null: no radial gradient)
r_dist, p_dist = sp_corr_func(obs['dist_norm'].values, obs['NetITH'].values)
print(f"\n  NetITH vs Distance: ρ={r_dist:.4f}, p={p_dist:.2e}")

# 4d: NetITH by core vs margin (leiden-based)
core_mask = obs['leiden_clusters'].isin(core_clusters)
margin_mask = obs['leiden_clusters'].isin(margin_clusters)
core_netith = obs.loc[core_mask, 'NetITH'].values
margin_netith = obs.loc[margin_mask, 'NetITH'].values
if len(core_netith) > 0 and len(margin_netith) > 0:
    # 4d: Mann-Whitney U core vs margin NetITH (null: identical distributions)
    stat_mw, p_mw = mannwhitneyu(core_netith, margin_netith, alternative='two-sided')
    delta_cm = core_netith.mean() - margin_netith.mean()
    print(f"  Core vs Margin: Δ={delta_cm:+.3f}, MW p={p_mw:.2e}")
else:
    delta_cm = np.nan
    stat_mw = np.nan
    p_mw = np.nan

# 4e: NetITH vs immune infiltration scores
print("\n  4e: NetITH vs immune scores")
immune_scores = ['T_active_score', 'T_exhausted_score', 'M1_macro_score',
                 'M2_macro_score', 'CAF_rich_score', 'B_cells_score',
                 'NK_cells_score', 'Treg_score']
# 4e: Spearman NetITH vs each immune infiltration score; p < 0.05 flagged
immune_corrs = []
for score in immune_scores:
    if score in obs.columns:
        vals = obs[score].values
        valid = ~(np.isnan(vals) | np.isnan(spot_netith))
        if valid.sum() > 10:
            r, p = sp_corr_func(vals[valid], spot_netith[valid])
            immune_corrs.append({
                'score': score,
                'spearman_r': r,
                'p_value': p,
                'significant': p < 0.05
            })
immune_corr_df = pd.DataFrame(immune_corrs)
print(immune_corr_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION — 6-panel (2×3) figure
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] Creating visualization...")

# 6-panel figure: spatial map, cluster/immune boxplots, radial profile, immune correlations
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('BRCA Visium Spatial NetITH: Tumor Core→Margin Gradient Validation',
             fontsize=14, fontweight='bold', y=0.98)

# Panel A: Spatial NetITH heatmap
ax = axes[0, 0]
sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                     c=spot_netith, cmap='plasma', s=12, alpha=0.85,
                     edgecolors='none')
ax.set_xlabel('Spatial X (µm)', fontsize=9)
ax.set_ylabel('Spatial Y (µm)', fontsize=9)
ax.set_title(f'A: Spatial NetITH Map\n(mean={spot_netith.mean():.2f}, n={n_spots})', fontsize=10)
ax.set_aspect('equal')
cbar = plt.colorbar(sc_plot, ax=ax, shrink=0.8)
cbar.set_label('NetITH (bits)', fontsize=9)

# Panel B: NetITH by leiden cluster
ax = axes[0, 1]
clusters_sorted = sorted(obs['leiden_clusters'].unique())
cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters_sorted)))
bp_data = [obs[obs['leiden_clusters'] == c]['NetITH'].values for c in clusters_sorted]
bp = ax.boxplot(bp_data, labels=[str(c) for c in clusters_sorted],
                patch_artist=True)
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
# Star core clusters
for i, c in enumerate(clusters_sorted):
    if c in core_clusters:
        ax.text(i+1, ax.get_ylim()[1] * 0.95, '★', ha='center', fontsize=14, color='red')
ax.set_xlabel('Leiden Cluster', fontsize=9)
ax.set_ylabel('NetITH', fontsize=9)
ax.set_title('B: NetITH by Leiden Cluster', fontsize=10)

# Panel C: NetITH by immune state
ax = axes[0, 2]
states_sorted = sorted(obs['predicted_immune_state'].unique(),
                       key=lambda s: obs[obs['predicted_immune_state']==s]['NetITH'].mean())
state_data = [obs[obs['predicted_immune_state'] == s]['NetITH'].values for s in states_sorted]
bp2 = ax.boxplot(state_data, labels=[s.replace('_', '\n') for s in states_sorted],
                 patch_artist=True, vert=True)
state_colors = plt.cm.Set2(np.linspace(0, 1, len(states_sorted)))
for patch, color in zip(bp2['boxes'], state_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_ylabel('NetITH', fontsize=9)
ax.set_title('C: NetITH by Immune State', fontsize=10)
ax.tick_params(axis='x', labelsize=7)

# Panel D: Radial NetITH profile (core → margin)
ax = axes[1, 0]
ax.fill_between(dist_profile['dist_mid'],
                dist_profile['netith_mean'] - dist_profile['netith_se'],
                dist_profile['netith_mean'] + dist_profile['netith_se'],
                alpha=0.3, color='#d62728')
ax.plot(dist_profile['dist_mid'], dist_profile['netith_mean'], 'o-',
        color='#d62728', linewidth=2.5, markersize=6, label='NetITH')

# Dual y-axis: spot count
ax2 = ax.twinx()
ax2.bar(dist_profile['dist_mid'], dist_profile['n_spots'],
        width=0.03, alpha=0.2, color='gray', label='n spots')
ax2.set_ylabel('Number of spots', fontsize=8, color='gray')

# Add correlation annotation
ax.annotate(f"Spearman ρ={r_dist:.3f}, p={p_dist:.1e}",
            xy=(0.5, 0.05), xycoords='axes fraction',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Normalized Distance from Centroid (core→margin)', fontsize=9)
ax.set_ylabel('NetITH (mean ± SE)', fontsize=9, color='#d62728')
ax.set_title('D: Radial NetITH Profile', fontsize=10)
ax.axhline(y=spot_netith.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Panel E: Immune score correlation barplot
ax = axes[1, 1]
if len(immune_corr_df) > 0:
    colors_e = ['#d62728' if r < 0 else '#2ca02c' for r in immune_corr_df['spearman_r']]
    ax.barh(range(len(immune_corr_df)), immune_corr_df['spearman_r'], color=colors_e, alpha=0.8)
    ax.set_yticks(range(len(immune_corr_df)))
    ax.set_yticklabels([s.replace('_score', '').replace('_', ' ') for s in immune_corr_df['score']],
                       fontsize=8)
    ax.set_xlabel('Spearman ρ with NetITH', fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    # Mark significant ones
    for i, (_, row) in enumerate(immune_corr_df.iterrows()):
        if row['significant']:
            ax.annotate('*', xy=(row['spearman_r'] + 0.01*np.sign(row['spearman_r']), i),
                        fontsize=14, va='center', ha='center', color='darkred')
    ax.set_title('E: Immune Infiltration \nvs NetITH', fontsize=10)

# Panel F: Core vs Margin summary + key statistics
ax = axes[1, 2]
ax.axis('off')

core_n = len(core_netith) if len(core_netith) > 0 else 0
margin_n = len(margin_netith) if len(margin_netith) > 0 else 0

if len(immune_corr_df) > 0:
    top_pos = immune_corr_df.nlargest(1, 'spearman_r')
    top_neg = immune_corr_df.nsmallest(1, 'spearman_r')
    top_pos_name = top_pos['score'].values[0].replace('_score', '')
    top_pos_r = top_pos['spearman_r'].values[0]
    top_neg_name = top_neg['score'].values[0].replace('_score', '')
    top_neg_r = top_neg['spearman_r'].values[0]
else:
    top_pos_name = top_neg_name = 'N/A'
    top_pos_r = top_neg_r = 0.0

summary_text = f"""
BRCA Visium Spatial NetITH Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data: 10x Visium BRCA (breast cancer)
  {n_spots} spots × {n_genes_focused} genes (CollecTRI-focused)
  CollecTRI: {n_edges_focused} edges ({n_tfs_focused} TFs × {n_tgts_focused} targets)
  k={K_NEIGHBORS} spatial neighbors per spot

Global NetITH:
  Mean ± SD: {spot_netith.mean():.3f} ± {spot_netith.std():.3f}
  Range: [{spot_netith.min():.3f}, {spot_netith.max():.3f}]

Leiden Clusters ({len(clusters_sorted)} total):
  Core (epithelial-high): {core_clusters}
    NetITH: {obs.loc[core_mask, 'NetITH'].mean():.3f} ± {obs.loc[core_mask, 'NetITH'].std():.3f}
  Margin (epithelial-low): {margin_clusters}
    NetITH: {obs.loc[margin_mask, 'NetITH'].mean():.3f} ± {obs.loc[margin_mask, 'NetITH'].std():.3f}
  Δ(Core − Margin): {delta_cm:+.3f}
  Mann-Whitney p: {p_mw:.2e}

Spatial Gradient:
  NetITH vs Distance: ρ={r_dist:.4f} (p={p_dist:.2e})

Immune Associations:
  Top positive: {top_pos_name} (ρ={top_pos_r:.3f})
  Top negative: {top_neg_name} (ρ={top_neg_r:.3f})

Conclusion:
  {'✓ Tumor core→margin NetITH gradient CONFIRMED' if p_dist < 0.05 else '△ Gradient not significant'}
  {'✓ Core-margin difference significant' if p_mw < 0.05 else '△ Core-margin difference not significant'}
"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=7.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = FIG_DIR / "spatial_visium_netith_main.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Main figure saved: {fig_path}")

# ═══════════════════════════════════════════════════════════════
# Supplementary: NetITH × KRT expression overlay
# ═══════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('BRCA Visium: NetITH vs Epithelial Markers (KRT8/KRT19)',
              fontsize=12, fontweight='bold')

# KRT8
ax = axes2[0]
if 'KRT8' in gene_to_idx:
    krt8_expr = X_z[:, gene_to_idx['KRT8']]
    sc1 = ax.scatter(coords[:, 0], coords[:, 1], c=krt8_expr,
                     cmap='Reds', s=12, alpha=0.85, edgecolors='none')
    ax.set_title(f'KRT8 Expression\n(ρ vs NetITH: {sp_corr_func(krt8_expr, spot_netith)[0]:.3f})',
                 fontsize=10)
    plt.colorbar(sc1, ax=ax, shrink=0.8)
ax.set_xlabel('Spatial X', fontsize=9)
ax.set_ylabel('Spatial Y', fontsize=9)
ax.set_aspect('equal')

# KRT19
ax = axes2[1]
if 'KRT19' in gene_to_idx:
    krt19_expr = X_z[:, gene_to_idx['KRT19']]
    sc2 = ax.scatter(coords[:, 0], coords[:, 1], c=krt19_expr,
                     cmap='Reds', s=12, alpha=0.85, edgecolors='none')
    ax.set_title(f'KRT19 Expression\n(ρ vs NetITH: {sp_corr_func(krt19_expr, spot_netith)[0]:.3f})',
                 fontsize=10)
    plt.colorbar(sc2, ax=ax, shrink=0.8)
ax.set_xlabel('Spatial X', fontsize=9)
ax.set_ylabel('Spatial Y', fontsize=9)
ax.set_aspect('equal')

# NetITH vs KRT8 scatter
ax = axes2[2]
if 'KRT8' in gene_to_idx:
    ax.scatter(krt8_expr, spot_netith, c=obs['dist_norm'], cmap='plasma',
               s=8, alpha=0.5, edgecolors='none')
    r_krt, p_krt = sp_corr_func(krt8_expr, spot_netith)
    ax.set_xlabel('KRT8 Expression (Z-score)', fontsize=9)
    ax.set_ylabel('NetITH', fontsize=9)
    ax.set_title(f'NetITH vs KRT8\nρ={r_krt:.4f}, p={p_krt:.2e}', fontsize=10)
    cbar3 = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar3.set_label('Distance from centroid', fontsize=8)

plt.tight_layout()
fig_path2 = FIG_DIR / "spatial_visium_krt_overlay.png"
fig2.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"  KRT overlay saved: {fig_path2}")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print("\nSaving results...")

# Per-spot results
results_df = pd.DataFrame({
    'spot_id': obs.index,
    'spatial_x': coords[:, 0],
    'spatial_y': coords[:, 1],
    'dist_from_centroid': obs['dist_from_centroid'].values,
    'dist_norm': obs['dist_norm'].values,
    'leiden_cluster': obs['leiden_clusters'].values,
    'immune_state': obs['predicted_immune_state'].values,
    'NetITH': spot_netith,
    'n_active_edges': spot_n_edges_used,
})
# Write per-spot NetITH and all summary tables
results_df.to_csv(OUTPUT_DIR / "spatial_visium_netith.csv", index=False)

# Leiden cluster stats
leiden_df.to_csv(OUTPUT_DIR / "spatial_visium_leiden_stats.csv", index=False)

# Immune state stats
immune_df.to_csv(OUTPUT_DIR / "spatial_visium_immune_stats.csv", index=False)

# Radial profile
dist_profile.to_csv(OUTPUT_DIR / "spatial_visium_radial_profile.csv", index=False)

# Immune correlations
immune_corr_df.to_csv(OUTPUT_DIR / "spatial_visium_immune_corrs.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BRCA VISIUM SPATIAL NetITH VALIDATION — COMPLETE")
print("=" * 70)
print(f"""
Key Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data: {n_spots} spots × {n_genes_focused} genes
  CollecTRI: {n_edges_focused} focused edges ({n_tfs_focused} TFs × {n_tgts_focused} targets)
  k={K_NEIGHBORS} spatial neighbors

Global NetITH: {spot_netith.mean():.3f} ± {spot_netith.std():.3f}

Leiden Clusters ({len(clusters_sorted)}):
  Core (epithelial): {core_clusters}
  Margin (epithelial-low): {margin_clusters}
  Core NetITH: {obs.loc[core_mask, 'NetITH'].mean():.3f} ± {obs.loc[core_mask, 'NetITH'].std():.3f}
  Margin NetITH: {obs.loc[margin_mask, 'NetITH'].mean():.3f} ± {obs.loc[margin_mask, 'NetITH'].std():.3f}
  Δ(Core−Margin): {delta_cm:+.3f}, MW p={p_mw:.2e}

Spatial Gradient:
  NetITH vs Distance: ρ={r_dist:.4f}, p={p_dist:.2e}

Immune Associations:
  {len(immune_corr_df[immune_corr_df['significant']])}/{len(immune_corr_df)} significant

Output files:
  {OUTPUT_DIR}/spatial_visium_netith.csv             — per-spot NetITH
  {OUTPUT_DIR}/spatial_visium_leiden_stats.csv       — leiden cluster stats
  {OUTPUT_DIR}/spatial_visium_immune_stats.csv       — immune state stats
  {OUTPUT_DIR}/spatial_visium_radial_profile.csv     — radial gradient
  {OUTPUT_DIR}/spatial_visium_immune_corrs.csv       — immune correlations
  {FIG_DIR}/spatial_visium_netith_main.png           — 6-panel figure
  {FIG_DIR}/spatial_visium_krt_overlay.png           — KRT overlay
""")
