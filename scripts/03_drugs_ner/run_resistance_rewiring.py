#!/usr/bin/env python3
"""run_resistance_rewiring.py — resistance GRN rewiring: edge-level CollecTRI rewiring between resistant and sensitive states.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{ensg_symbol_map.csv, rna_expr.csv, cell_annot.csv}; results/focused_genes_collectri.txt; /tmp/collectri_net.pkl; data/gdsc_download/GDSC2_IC50_all.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{rewiring_edge_delta.csv, rewiring_consensus.csv, rewiring_tf_summary.csv}; results/depmap/figures/rewiring_{consensus_heatmap,drug_clusters,tf_bubble}.png
Pipeline: drug-ner stage — see repository README
"""
from pathlib import Path
import os

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import pickle, time, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_DATA = f"{DATA_ROOT}/gdsc"
DEPMAP_OUTPUT = f"{ROOT}/results/depmap"
GDSC_IC50 = f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv'
N_TERTILE_MIN = 12

os.makedirs(f'{DEPMAP_OUTPUT}/figures', exist_ok=True)


def load_data():
    print("[1] Loading data...")
    
    # ENSG→Symbol map
    ensg_map = pd.read_csv(f'{GDSC_DATA}/ensg_symbol_map.csv')
    ensg_map['ensg'] = ensg_map['ensg'].str.strip('"')
    ensg_map['symbol'] = ensg_map['symbol'].str.strip('"')
    symbol_to_ensg = dict(zip(ensg_map['symbol'], ensg_map['ensg']))
    ensg_to_symbol = dict(zip(ensg_map['ensg'], ensg_map['symbol']))
    
    # Expression
    expr_ensg = pd.read_csv(f'{GDSC_DATA}/rna_expr.csv', index_col=0)
    expr_ensg.index = expr_ensg.index.str.strip('"')
    
    # Cell line annotation
    cell_annot = pd.read_csv(f'{GDSC_DATA}/cell_annot.csv', index_col=0)
    cel_to_cell_line = cell_annot['Factor.Value.cell_line.'].to_dict()
    
    # Focused genes
    with open('results/focused_genes_collectri.txt') as f:
        focused_symbols = set(line.strip() for line in f if line.strip())
    focused_ensg = set()
    for sym in focused_symbols:
        if sym in symbol_to_ensg:
            focused_ensg.add(symbol_to_ensg[sym])
    
    common_ensg = sorted(focused_ensg & set(expr_ensg.index))
    common_symbols = [ensg_to_symbol.get(e, e) for e in common_ensg]
    n_genes = len(common_ensg)
    
    # Build expression matrix
    expr = expr_ensg.loc[common_ensg]
    vals = expr.values.T
    mean = vals.mean(axis=0)
    std = vals.std(axis=0) + 1e-10
    expr_z = np.clip((vals - mean) / std, -3, 3)
    
    cel_names = expr.columns.tolist()
    cell_line_names = [cel_to_cell_line.get(c, c) for c in cel_names]
    cel_to_cl = dict(zip(cel_names, cell_line_names))
    
    # CollecTRI edges restricted to the focused gene set, with edge IDs (TF→target)
    # CollecTRI edges
    with open('/tmp/collectri_net.pkl', 'rb') as f:
        net = pickle.load(f)
    
    symbol_to_idx = {}
    for e, s in zip(common_ensg, common_symbols):
        symbol_to_idx[s] = list(common_ensg).index(e)
    
    edges = []
    for _, row in net.iterrows():
        tf = row['source']; target = row['target']
        if tf in symbol_to_idx and target in symbol_to_idx:
            edges.append({
                'tf_idx': symbol_to_idx[tf],
                'target_idx': symbol_to_idx[target],
                'weight': float(row.get('weight', 1.0)),
                'tf_name': tf,
                'target_name': target,
                'edge_id': f"{tf}→{target}"
            })
    
    print(f"  Genes: {n_genes}, Edges: {len(edges)}")
    
    # IC50
    ic50_raw = pd.read_csv(GDSC_IC50)
    ic50_mat = ic50_raw.pivot_table(
        values='LN_IC50', index='CELL_LINE_NAME',
        columns='DRUG_NAME', aggfunc='mean')
    print(f"  IC50: {ic50_mat.shape[1]} drugs × {ic50_mat.shape[0]} cell lines")
    
    # Map cells to IC50
    ic50_cells = set(ic50_mat.index)
    cel_idx_to_ic50 = {}
    for i, cel in enumerate(cel_names):
        cl = cel_to_cl.get(cel, cel)
        if cl in ic50_cells:
            cel_idx_to_ic50[i] = cl
        elif cl.strip() in ic50_cells:
            cel_idx_to_ic50[i] = cl.strip()
    
    print(f"  Mapped {len(cel_idx_to_ic50)} cells to IC50")
    
    return (expr_z, n_genes, edges, ic50_mat, cel_idx_to_ic50, cel_names)


def compute_edge_weights(expr_z, cel_indices, edges, n_genes):
    """Pre-compute edge weights for all specified cells."""
    n_cells = len(cel_indices)
    n_edges = len(edges)
    edge_weights = np.zeros((n_cells, n_edges))
    
    # Edge weight per cell = CollecTRI weight × |z_TF| × |z_target| (expression-weighted)
    for j, ci in enumerate(cel_indices):
        for k, e in enumerate(edges):
            tf_a = abs(expr_z[ci, e['tf_idx']])
            tgt_a = abs(expr_z[ci, e['target_idx']])
            edge_weights[j, k] = e['weight'] * tf_a * tgt_a
    
    return edge_weights


def run_edge_rewiring(expr_z, n_genes, edges, ic50_mat, cel_idx_to_ic50):
    """V1+V2: Per-drug edge Δ and consensus rewiring."""
    print("\n" + "=" * 60)
    print("[V1] Edge-Level Rewiring: Sensitive vs Resistant")
    print("=" * 60)
    
    n_edges = len(edges)
    edge_ids = [e['edge_id'] for e in edges]
    
    # Build mapping from IC50 cell names → cel indices
    ic50_to_cel_idx = {}
    for cel_idx, ic50_name in cel_idx_to_ic50.items():
        if ic50_name not in ic50_to_cel_idx:
            ic50_to_cel_idx[ic50_name] = []
        ic50_to_cel_idx[ic50_name].append(cel_idx)
    
    # Pre-compute edge weights for ALL mapped cells
    all_mapped_indices = sorted(set(cel_idx_to_ic50.keys()))
    idx_map = {orig: new for new, orig in enumerate(all_mapped_indices)}
    
    print(f"  Computing edge weights for {len(all_mapped_indices)} cells × {n_edges} edges...")
    t0 = time.time()
    edge_weights_all = compute_edge_weights(expr_z, all_mapped_indices, edges, n_genes)
    print(f"  Done in {time.time()-t0:.0f}s, shape={edge_weights_all.shape}")
    
    # Aggregate per drug
    drugs = ic50_mat.columns.tolist()
    edge_delta_all = []  # drug × edge
    drug_names_used = []
    
    for di, drug in enumerate(drugs):
        drug_ic50 = ic50_mat[drug].dropna()
        
        # Find which of our mapped cells have IC50 data for this drug
        cell_indices = []
        ic50_vals = []
        for ic50_name, cel_idxs in ic50_to_cel_idx.items():
            if ic50_name in drug_ic50.index:
                cell_indices.extend(cel_idxs)
                ic50_vals.extend([drug_ic50[ic50_name]] * len(cel_idxs))
        
        if len(cell_indices) < N_TERTILE_MIN * 3:
            continue
        
        # Split cells into IC50 tertiles: bottom = sensitive, top = resistant
        ic50_arr = np.array(ic50_vals)
        lo_thresh = np.quantile(ic50_arr, 1/3)
        hi_thresh = np.quantile(ic50_arr, 2/3)
        
        sensitive_mask = ic50_arr <= lo_thresh
        resistant_mask = ic50_arr >= hi_thresh
        
        if sensitive_mask.sum() < N_TERTILE_MIN or resistant_mask.sum() < N_TERTILE_MIN:
            continue
        
        # Map to pre-computed indices
        s_indices = [idx_map[cell_indices[i]] for i in range(len(cell_indices)) if sensitive_mask[i]]
        r_indices = [idx_map[cell_indices[i]] for i in range(len(cell_indices)) if resistant_mask[i]]
        
        # Per-edge rewiring = mean edge weight in the resistant group minus the sensitive group
        # Mean edge weights per group
        s_means = edge_weights_all[s_indices].mean(axis=0)
        r_means = edge_weights_all[r_indices].mean(axis=0)
        deltas = r_means - s_means
        
        edge_delta_all.append(deltas)
        drug_names_used.append(drug)
        
        if (di + 1) % 50 == 0:
            print(f"  {di+1}/{len(drugs)} drugs processed...")
    
    # Write the drug × edge rewiring matrix (Δ edge weight per drug)
    delta_df = pd.DataFrame(edge_delta_all, index=drug_names_used, columns=edge_ids)
    delta_df.to_csv(f'{DEPMAP_OUTPUT}/rewiring_edge_delta.csv')
    print(f"  -> Saved rewiring_edge_delta.csv ({len(drug_names_used)} drugs × {n_edges} edges)")
    
    # V2: Consensus rewiring
    print("\n[V2] Consensus rewiring across drugs...")
    
    # For each edge: mean Δ, % of drugs with Δ > 0, Wilcoxon test
    consensus = []
    for k, edge_id in enumerate(edge_ids):
        vals = delta_df[edge_id].dropna().values
        if len(vals) < 10:
            continue
        
        # Consensus across drugs: mean Δ, fraction of drugs gaining the edge, Wilcoxon test (null: median Δ = 0)
        mean_delta = np.mean(vals)
        pct_up = (vals > 0).mean()
        
        try:
            stat, p = wilcoxon(vals)
        except:
            p = np.nan
        
        # Effect size: mean / std
        cohens_d = mean_delta / (np.std(vals) + 1e-10)
        
        consensus.append({
            'edge_id': edge_id,
            'tf': edges[k]['tf_name'],
            'target': edges[k]['target_name'],
            'mean_delta': mean_delta,
            'cohens_d': cohens_d,
            'pct_drugs_up': pct_up,
            'p_wilcoxon': p,
            'n_drugs': len(vals),
        })
    
    # Write consensus rewiring results, ranked by Wilcoxon p-value
    consensus_df = pd.DataFrame(consensus).sort_values('p_wilcoxon')
    consensus_df.to_csv(f'{DEPMAP_OUTPUT}/rewiring_consensus.csv', index=False)
    
    sig_up = consensus_df[(consensus_df['p_wilcoxon'] < 0.05) & (consensus_df['mean_delta'] > 0)]
    sig_down = consensus_df[(consensus_df['p_wilcoxon'] < 0.05) & (consensus_df['mean_delta'] < 0)]
    
    print(f"  Significantly gained in resistance: {len(sig_up)} edges")
    print(f"  Significantly lost in resistance: {len(sig_down)} edges")
    
    print(f"\n  Top 10 edges GAINED in resistance:")
    for _, r in sig_up.nlargest(10, 'cohens_d').iterrows():
        print(f"    {r['edge_id']:<30} Δ={r['mean_delta']:+.4f} d={r['cohens_d']:+.3f} "
              f"↑{r['pct_drugs_up']:.0%}")
    
    print(f"\n  Top 10 edges LOST in resistance:")
    for _, r in sig_down.nsmallest(10, 'cohens_d').iterrows():
        print(f"    {r['edge_id']:<30} Δ={r['mean_delta']:+.4f} d={r['cohens_d']:+.3f} "
              f"↑{r['pct_drugs_up']:.0%}")
    
    return delta_df, consensus_df, sig_up, sig_down


def run_tf_rewiring_summary(consensus_df):
    """V3: Which TF regulons are most rewired?"""
    print("\n[V3] TF Regulon Rewiring Summary...")
    
    # Aggregate consensus rewiring per TF (regulon): edges gained/lost, mean |Δ|, % significant
    tf_summary = consensus_df.groupby('tf').agg(
        n_edges=('edge_id', 'nunique'),
        mean_delta=('mean_delta', 'mean'),
        mean_abs_delta=('mean_delta', lambda x: np.abs(x).mean()),
        n_gained=('mean_delta', lambda x: ((x > 0) & (consensus_df.loc[x.index, 'p_wilcoxon'] < 0.05)).sum()),
        n_lost=('mean_delta', lambda x: ((x < 0) & (consensus_df.loc[x.index, 'p_wilcoxon'] < 0.05)).sum()),
        pct_sig=('p_wilcoxon', lambda x: (x < 0.05).mean()),
    ).sort_values('mean_abs_delta', ascending=False)
    
    # Write per-TF regulon rewiring summary
    tf_summary.to_csv(f'{DEPMAP_OUTPUT}/rewiring_tf_summary.csv')
    
    print(f"\n  Top 15 most rewired TFs:")
    for tf, row in tf_summary.head(15).iterrows():
        print(f"    {tf:<12} |Δ̅|={row['mean_abs_delta']:.4f} "
              f"gained={int(row['n_gained'])} lost={int(row['n_lost'])} "
              f"edges={int(row['n_edges'])} sig={row['pct_sig']:.0%}")
    
    return tf_summary


def cluster_drugs_by_rewiring(delta_df, output_dir):
    """V4: Cluster drugs by similar rewiring patterns → resistance subtypes."""
    print("\n[V4] Drug clustering by rewiring pattern...")
    
    # Use top variable edges
    edge_var = delta_df.var().nlargest(50)
    top_edges = edge_var.index.tolist()
    mat = delta_df[top_edges].dropna()
    
    if len(mat) < 10:
        print("  Not enough drugs for clustering")
        return
    
    # Ward hierarchical clustering of drugs by rewiring pattern → resistance subtypes
    # Hierarchical clustering
    dist = pdist(mat.values, metric='euclidean')
    Z = linkage(dist, method='ward')
    
    # Cut into clusters
    n_clusters = min(5, len(mat) // 10)
    if n_clusters < 2:
        n_clusters = 2
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    mat_clust = mat.copy()
    mat_clust['cluster'] = clusters
    
    print(f"  {n_clusters} rewiring-based resistance subtypes:")
    for c in range(1, n_clusters + 1):
        n = (clusters == c).sum()
        drugs_in_c = mat_clust[mat_clust['cluster'] == c].index.tolist()
        print(f"    Cluster {c}: {n} drugs, e.g. {', '.join(drugs_in_c[:5])}...")
    
    # PCA projection of the rewiring matrix for cluster visualization
    # PCA for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat.values)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for c in range(1, n_clusters + 1):
        mask = clusters == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[c-1]], label=f'Cluster {c} (n={mask.sum()})',
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title('Drug Clusters by GRN Rewiring Pattern\n'
                 '(Resistance subtypes based on edge-level changes)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/rewiring_drug_clusters.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved rewiring_drug_clusters.png")
    
    return mat_clust


def plot_consensus_heatmap(consensus_df, output_dir):
    """Heatmap of top rewired edges."""
    sig = consensus_df[consensus_df['p_wilcoxon'] < 0.05].copy()
    if len(sig) < 5:
        print("  Not enough significant edges for heatmap")
        return
    
    # Top 40 edges by |cohens_d|
    top = sig.copy()
    top['abs_d'] = top['cohens_d'].abs()
    top = top.nlargest(40, 'abs_d')
    
    # Sort: gained first, then lost
    top = top.sort_values('mean_delta', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(top) * 0.3)))
    
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top['mean_delta']]
    bars = ax.barh(range(len(top)), top['mean_delta'].values, color=colors, alpha=0.8)
    
    # Add edge labels
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['edge_id'].values, fontsize=7)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Mean Δ Edge Weight (Resistant − Sensitive)', fontsize=11)
    ax.set_title('Consensus GRN Rewiring in Drug Resistance\n'
                 f'(Top {len(top)} rewired edges, p<0.05)',
                 fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#d62728', label='Gained in resistance'),
        Patch(color='#1f77b4', label='Lost in resistance'),
    ], fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/rewiring_consensus_heatmap.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved rewiring_consensus_heatmap.png")


def plot_tf_rewiring_bubble(tf_summary, output_dir):
    """Bubble plot: TF regulon rewiring summary."""
    top_tfs = tf_summary.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    total = top_tfs['n_gained'] + top_tfs['n_lost']
    gained_pct = top_tfs['n_gained'] / (total + 1e-10)
    
    sizes = total.values * 30 + 20
    
    colors = []
    for gp in gained_pct:
        if gp > 0.6:
            colors.append('#d62728')
        elif gp < 0.4:
            colors.append('#1f77b4')
        else:
            colors.append('#9467bd')
    
    scatter = ax.scatter(
        top_tfs['mean_delta'], top_tfs['mean_abs_delta'],
        s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5
    )
    
    for i, (tf, row) in enumerate(top_tfs.iterrows()):
        ax.annotate(tf, (row['mean_delta'], row['mean_abs_delta']),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Δ Edge Weight (Resistant − Sensitive)', fontsize=11)
    ax.set_ylabel('Mean |Δ| Edge Weight', fontsize=11)
    ax.set_title('TF Regulon Rewiring in Drug Resistance\n'
                 '(Size = total rewired edges, Color = gain bias)',
                 fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#d62728', label='Mostly gained'),
        Patch(color='#1f77b4', label='Mostly lost'),
        Patch(color='#9467bd', label='Mixed'),
    ], fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/rewiring_tf_bubble.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved rewiring_tf_bubble.png")


def main():
    print("=" * 60)
    print("V: Resistance GRN Rewiring — Edge-Level Mechanism")
    print("=" * 60)
    
    expr_z, n_genes, edges, ic50_mat, cel_idx_to_ic50, cel_names = load_data()
    
    # V1+V2: Edge rewiring + consensus
    delta_df, consensus_df, sig_up, sig_down = run_edge_rewiring(
        expr_z, n_genes, edges, ic50_mat, cel_idx_to_ic50)
    
    # V3: TF regulon summary
    tf_summary = run_tf_rewiring_summary(consensus_df)
    
    # V4: Drug clustering
    mat_clust = cluster_drugs_by_rewiring(delta_df, DEPMAP_OUTPUT)
    
    # Visualization
    print("\n[Viz] Generating figures...")
    plot_consensus_heatmap(consensus_df, DEPMAP_OUTPUT)
    plot_tf_rewiring_bubble(tf_summary, DEPMAP_OUTPUT)
    
    print("\nDone! Outputs in results/depmap/")


if __name__ == '__main__':
    main()
