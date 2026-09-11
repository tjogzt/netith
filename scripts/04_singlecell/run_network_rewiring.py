"""run_network_rewiring.py — single-cell GRN rewiring: edge strength in high vs low NetITH cells.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz; results/focused_genes_collectri.txt; results/gse131907/cell_entropy_results.csv; /tmp/collectri_net.pkl (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{network_rewiring_edges.csv, network_rewiring_tf_summary.csv}; results/depmap/figures/network_rewiring_high_vs_low.png
Pipeline: single-cell stage — see repository README
"""
import os

import numpy as np
import pandas as pd
import pickle, os, sys, warnings, subprocess
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
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
# 1. Stream GSE131907 focused gene expression + load NetITH
# ═══════════════════════════════════════════════════════════
print("[1/4] Streaming GSE131907 expression...")

EXPR_PATH = f"{DATA_ROOT}/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz"
FOCUSED_FILE = f"{ROOT}/results/focused_genes_collectri.txt"

with open(FOCUSED_FILE) as f:
    target_genes = set(line.strip() for line in f if line.strip())

# Load cell entropy
# Per-cell NetITH (vn_entropy); barcodes keyed as {Barcode}_{Sample}
cell_entropy = pd.read_csv(GSE_DIR / "cell_entropy_results.csv")
cell_entropy['full_barcode'] = cell_entropy['Barcode'] + '_' + cell_entropy['Sample']
all_barcodes = set(cell_entropy['full_barcode'].values)

# Stream
# Stream the gzipped expression matrix, keeping focused-gene rows for the matched cells
proc = subprocess.Popen(
    ['gunzip', '-c', EXPR_PATH],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1024*1024
)

header_line = proc.stdout.readline()
header = header_line.strip().split('\t')
col_to_idx = {name: i for i, name in enumerate(header)}
keep_cells = [c for c in all_barcodes if c in col_to_idx]
keep_indices = [col_to_idx[c] for c in keep_cells]
print(f"  Matched cells: {len(keep_cells)}")

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

expr_arr = np.column_stack([gene_rows[g] for g in gene_list])
expr_df = pd.DataFrame(expr_arr, index=keep_cells, columns=gene_list)

# Align NetITH
sub_entropy = cell_entropy[cell_entropy['full_barcode'].isin(keep_cells)].set_index('full_barcode')
sub_entropy = sub_entropy.loc[keep_cells]
vn = sub_entropy['vn_entropy'].values
n_cells = len(keep_cells)
print(f"  Matrix: {n_cells} cells × {len(gene_list)} genes")

# ═══════════════════════════════════════════════════════════
# 2. Split by NetITH tertile + build CollecTRI edges
# ═══════════════════════════════════════════════════════════
print("\n[2/4] Splitting cells by NetITH + building GRN...")

# Split: top 1/3 vs bottom 1/3
# Split cells into tertiles: top third (high) vs bottom third (low) NetITH
high_thresh = np.quantile(vn, 2/3)
low_thresh = np.quantile(vn, 1/3)
high_mask = vn >= high_thresh
low_mask = vn <= low_thresh
print(f"  High group (≥{high_thresh:.2f}): {high_mask.sum()} cells")
print(f"  Low group (≤{low_thresh:.2f}): {low_mask.sum()} cells")

gene_to_idx = {g: i for i, g in enumerate(gene_list)}

# Load the CollecTRI GRN restricted to focused genes present in the matrix
with open('/tmp/collectri_net.pkl', 'rb') as f:
    net = pickle.load(f)

edges = []
for _, row in net.iterrows():
    tf = row['source']; target = row['target']
    if tf in gene_to_idx and target in gene_to_idx:
        edges.append({
            'tf': tf, 'target': target,
            'tf_idx': gene_to_idx[tf],
            'target_idx': gene_to_idx[target],
            'weight': float(row.get('weight', 1.0)),
        })

tf_set = set(e['tf'] for e in edges)
print(f"  GRN: {len(edges)} edges, {len(tf_set)} TFs")

# ═══════════════════════════════════════════════════════════
# 3. Per-edge correlation in high vs low NetITH groups
# ═══════════════════════════════════════════════════════════
print("\n[3/4] Computing edge correlations (high vs low NetITH)...")

expr_vals = expr_df.values
high_expr = expr_vals[high_mask]
low_expr = expr_vals[low_mask]

edge_results = []
for e in edges:
    ti = e['tf_idx']; gi = e['target_idx']
    
    # Per edge: Spearman correlation of TF and target within the high and low NetITH groups
    rho_high, p_high = spearmanr(high_expr[:, ti], high_expr[:, gi])
    rho_low, p_low = spearmanr(low_expr[:, ti], low_expr[:, gi])
    
    # Edge strength change delta|rho|; sign flips beyond +/-0.15 count as rewired
    delta_rho = abs(rho_high) - abs(rho_low)
    
    # Sign flip detection
    rewiring = 0
    if (rho_high > 0.15 and rho_low < -0.15) or (rho_high < -0.15 and rho_low > 0.15):
        rewiring = 1
    
    edge_results.append({
        'tf': e['tf'], 'target': e['target'],
        'edge': f"{e['tf']}→{e['target']}",
        'rho_high': rho_high, 'rho_low': rho_low,
        'delta_abs_rho': delta_rho, 'rewiring': rewiring,
        'weight': e['weight'],
    })

edge_df = pd.DataFrame(edge_results)

# Classify
# Classify edges: high-only / low-only / both / rewired by |rho| thresholds
edge_df['category'] = 'neutral'
edge_df.loc[(abs(edge_df['rho_high']) > 0.25) & (abs(edge_df['rho_low']) < 0.12), 'category'] = 'high_only'
edge_df.loc[(abs(edge_df['rho_low']) > 0.25) & (abs(edge_df['rho_high']) < 0.12), 'category'] = 'low_only'
edge_df.loc[(abs(edge_df['rho_high']) > 0.25) & (abs(edge_df['rho_low']) > 0.25), 'category'] = 'both'
edge_df.loc[edge_df['rewiring'] == 1, 'category'] = 'rewired'

cat_counts = edge_df['category'].value_counts()
print(f"  Edge categories:")
for cat, count in cat_counts.items():
    print(f"    {cat}: {count}")

# Per-TF summary
# Aggregate per-TF mean delta|rho| and category counts to rank rewired TFs
tf_summary = []
for tf_name in tf_set:
    tf_edges = edge_df[edge_df['tf'] == tf_name]
    if len(tf_edges) == 0:
        continue
    tf_summary.append({
        'tf': tf_name, 'n_edges': len(tf_edges),
        'delta_mean': tf_edges['delta_abs_rho'].mean(),
        'rho_high_mean': tf_edges['rho_high'].mean(),
        'rho_low_mean': tf_edges['rho_low'].mean(),
        'n_high_only': (tf_edges['category'] == 'high_only').sum(),
        'n_low_only': (tf_edges['category'] == 'low_only').sum(),
        'n_rewired': tf_edges['rewiring'].sum(),
    })

tf_summ_df = pd.DataFrame(tf_summary).sort_values('delta_mean', ascending=False)

print(f"\n  Top 10 TFs with largest network change:")
for _, row in tf_summ_df.head(10).iterrows():
    print(f"    {row['tf']:12s}: {row['n_edges']} edges, Δ|ρ|={row['delta_mean']:+.3f}, "
          f"h={row['n_high_only']}, l={row['n_low_only']}, r={row['n_rewired']}")

# ═══════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════
print("\n[4/4] Creating visualizations...")

# 6-panel figure: rho scatter, delta histogram, per-TF heatmap, JUN edges, top rewired network
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('Single-Cell GRN Network Rewiring: High vs Low NetITH Edge Structure\n(GSE131907 Lung Cancer)',
             fontsize=14, fontweight='bold', y=0.99)

# Panel A: ρ_high vs ρ_low scatter
ax = axes[0, 0]
colors_map = {'high_only': '#d62728', 'low_only': '#1f77b4', 'both': '#7f7f7f',
              'rewired': '#ff7f0e', 'neutral': '#e0e0e0'}
for cat in ['neutral', 'both', 'high_only', 'low_only', 'rewired']:
    sub = edge_df[edge_df['category'] == cat]
    if len(sub) > 0:
        ax.scatter(sub['rho_low'], sub['rho_high'],
                   c=colors_map[cat], s=5, alpha=0.6, label=f'{cat} ({len(sub)})')
ax.plot([-1, 1], [-1, 1], 'k-', linewidth=1, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('ρ (Low NetITH cells)', fontsize=11)
ax.set_ylabel('ρ (High NetITH cells)', fontsize=11)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_title(f'A: Edge Correlation: High vs Low NetITH\n({len(edge_df)} edges, GSE131907 SC)', fontsize=10)
ax.legend(fontsize=6, markerscale=3)

# Panel B: Δ|ρ| distribution
ax = axes[0, 1]
ax.hist(edge_df['delta_abs_rho'], bins=50, color='#8c564b', alpha=0.7, edgecolor='gray')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Δ|ρ| = |ρ_high| − |ρ_low|', fontsize=11)
ax.set_ylabel('Edges', fontsize=11)
ax.set_title(f'B: Edge Strength Change\n(μ={edge_df["delta_abs_rho"].mean():.4f})', fontsize=10)

# Panel C: Per-TF rewiring
ax = axes[0, 2]
top_tfs = tf_summ_df.nlargest(15, 'delta_mean')
tf_names = top_tfs['tf'].values
heat_data = np.column_stack([
    top_tfs['rho_high_mean'].values,
    top_tfs['rho_low_mean'].values,
    top_tfs['delta_mean'].values,
])
im = ax.imshow(heat_data.T, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
ax.set_yticks(range(3))
ax.set_yticklabels(['ρ_high', 'ρ_low', 'Δ|ρ|'], fontsize=8)
ax.set_xticks(range(len(tf_names)))
ax.set_xticklabels(tf_names, rotation=45, ha='right', fontsize=8)
ax.set_title('C: Top 15 TFs by Network Change', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel D: JUN sub-network
ax = axes[1, 0]
jun_edges = edge_df[edge_df['tf'] == 'JUN'].sort_values('delta_abs_rho').head(20)
if len(jun_edges) > 0:
    colors_j = ['#d62728' if v > 0 else '#1f77b4' for v in jun_edges['delta_abs_rho'].values]
    ax.barh(range(len(jun_edges)), jun_edges['delta_abs_rho'].values,
            color=colors_j, alpha=0.8, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(jun_edges)))
    ax.set_yticklabels([f"{r['target']}" for _, r in jun_edges.iterrows()], fontsize=6)
    ax.set_xlabel('Δ|ρ|', fontsize=11)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_title(f'D: JUN Target Edges\n(stronger in High=red, Low=blue)', fontsize=10)

# Panel E: Top rewired edge network
ax = axes[1, 1]
top_rewired = edge_df.nlargest(25, 'delta_abs_rho')
unique_nodes = list(set(top_rewired['tf'].unique()) | set(top_rewired['target'].unique()))[:25]
n_nodes = len(unique_nodes)
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
positions = {node: (np.cos(a), np.sin(a)) for node, a in zip(unique_nodes, angles)}

for _, row in top_rewired.iterrows():
    if row['tf'] in positions and row['target'] in positions:
        x1, y1 = positions[row['tf']]
        x2, y2 = positions[row['target']]
        c = '#d62728' if row['delta_abs_rho'] > 0 else '#1f77b4'
        w = min(abs(row['delta_abs_rho']) * 8, 3)
        ax.plot([x1, x2], [y1, y2], color=c, alpha=0.5, linewidth=w)

for node, (x, y) in positions.items():
    ax.scatter(x, y, s=50, c='#2ca02c', edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(node, (x, y), fontsize=4.5, ha='center', va='center', fontweight='bold')

ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
ax.axis('off')
ax.set_title('E: Top Δ|ρ| Edges Network', fontsize=10)

# Panel F: Summary
ax = axes[1, 2]
ax.axis('off')

n_high_only = (edge_df['category'] == 'high_only').sum()
n_low_only = (edge_df['category'] == 'low_only').sum()
n_rewired = edge_df['rewiring'].sum()

summary_text = f"""
Single-Cell GRN Rewiring
GSE131907 High vs Low NetITH

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells: {n_cells}
  High: {high_mask.sum()} (NetITH>{high_thresh:.2f})
  Low: {low_mask.sum()} (NetITH<{low_thresh:.2f})

GRN: {len(edges)} edges, {len(tf_set)} TFs

Edge Categories:
  High-only: {n_high_only} ({n_high_only/len(edge_df):.0%})
  Low-only: {n_low_only} ({n_low_only/len(edge_df):.0%})
  Rewired (sign-flip): {n_rewired}

Mean Δ|ρ|: {edge_df['delta_abs_rho'].mean():.4f}

Top Changed TFs:
"""
for _, row in tf_summ_df.head(5).iterrows():
    summary_text += f"  {row['tf']:10s}: {row['n_edges']}ed, Δ={row['delta_mean']:+.3f}\n"

summary_text += f"""
Insight:
Single-cell GRN structure is NOT
static — edges strengthen/weaken
depending on cellular NetITH state.
This explains "emergent property":
the same gene set produces different
network behaviors at different
entropy levels.
"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=7, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.4))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUTPUT_DIR / "figures" / "network_rewiring_high_vs_low.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# Write per-edge and per-TF rewiring tables
edge_df.to_csv(OUTPUT_DIR / "network_rewiring_edges.csv", index=False)
tf_summ_df.to_csv(OUTPUT_DIR / "network_rewiring_tf_summary.csv", index=False)

print("\n" + "=" * 70)
print("NETWORK REWIRING (SINGLE-CELL) — COMPLETE")
print("=" * 70)
print(f"""
  Edges: {len(edge_df)}, TFs: {len(tf_set)}
  High-only: {n_high_only}, Low-only: {n_low_only}, Rewired: {n_rewired}
  Mean Δ|ρ|: {edge_df['delta_abs_rho'].mean():.4f}
""")
