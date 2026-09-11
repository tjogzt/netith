"""run_gse131907_tf_decomp.py — GSE131907 single-cell TF contribution decomposition.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz; results/focused_genes_collectri.txt; results/gse131907/cell_entropy_results.csv; /tmp/collectri_net.pkl; results/depmap/tf_lasso_coefficients.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/figures/gse131907_tf_decomposition.png; results/depmap/{gse131907_tf_celltype_correlation.csv, gse131907_gdsc_sc_concordance.csv, gse131907_per_cell_tf_activity.csv}
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

# ═══════════════════════════════════════════════════════════
# 1. Stream focused genes from GSE131907
# ═══════════════════════════════════════════════════════════
print("[1/4] Streaming focused gene expression from GSE131907...")

EXPR_PATH = f"{DATA_ROOT}/geo/GSE131907/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz"
FOCUSED_FILE = f"{ROOT}/results/focused_genes_collectri.txt"

# Load focused genes
with open(FOCUSED_FILE) as f:
    target_genes = set(line.strip() for line in f if line.strip())
print(f"  Target genes: {len(target_genes)}")

# Load cell entropy to get barcodes — need to match {Barcode}_{Sample} format
# Per-cell NetITH (vn_entropy) with cell-type and sample-origin annotations
cell_entropy = pd.read_csv(GSE_DIR / "cell_entropy_results.csv")
cell_entropy['full_barcode'] = cell_entropy['Barcode'] + '_' + cell_entropy['Sample']
sampled_cells = set(cell_entropy['full_barcode'].values)
print(f"  Sampled cells: {len(sampled_cells)}")

# Open pipe and read header
# Stream the gzipped matrix, keeping focused-gene rows for the matched cells
proc = subprocess.Popen(
    ['gunzip', '-c', EXPR_PATH],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1024*1024
)

header_line = proc.stdout.readline()
header = header_line.strip().split('\t')

# Map barcode → column index
col_to_idx = {name: i for i, name in enumerate(header)}
keep_cells = [c for c in sampled_cells if c in col_to_idx]
keep_indices = [col_to_idx[c] for c in keep_cells]
print(f"  Matched cells: {len(keep_cells)}")

# Stream gene rows, filter for focused genes
gene_rows = {}
line_count = 0
for line in proc.stdout:
    line_count += 1
    gene = line.split('\t', 1)[0]
    if gene in target_genes:
        parts = line.strip().split('\t')
        values = []
        for ci in keep_indices:
            if ci < len(parts):
                try:
                    values.append(float(parts[ci]))
                except ValueError:
                    values.append(0.0)
            else:
                values.append(0.0)
        gene_rows[gene] = values
        if len(gene_rows) == len(target_genes):
            break  # all found

proc.terminate()
proc.wait()

gene_list = sorted(gene_rows.keys())
print(f"  Extracted {len(gene_list)}/{len(target_genes)} genes")
print(f"  Scanned {line_count} lines")

# Build expression matrix: cells × genes
expr_arr = np.column_stack([gene_rows[g] for g in gene_list])
expr_df = pd.DataFrame(expr_arr, index=keep_cells, columns=gene_list)

# Align with cell_entropy
# Align the entropy rows to the same cell order as the expression matrix
sub_entropy = cell_entropy[cell_entropy['full_barcode'].isin(keep_cells)].set_index('full_barcode')
sub_entropy = sub_entropy.loc[keep_cells]  # reorder

n_cells = len(keep_cells)
n_genes = len(gene_list)
gene_to_idx = {g: i for i, g in enumerate(gene_list)}
print(f"  Expression: {n_cells} cells × {n_genes} genes")

# ═══════════════════════════════════════════════════════════
# 2. Build CollecTRI GRN + compute per-cell per-TF contribution
# ═══════════════════════════════════════════════════════════
print("\n[2/4] Building CollecTRI GRN and computing TF activity...")

with open('/tmp/collectri_net.pkl', 'rb') as f:
    net = pickle.load(f)

edges = []
tf_set = set()
for _, row in net.iterrows():
    tf = row['source']; target = row['target']
    if tf in gene_to_idx and target in gene_to_idx:
        edges.append({
            'tf': tf, 'target': target,
            'tf_idx': gene_to_idx[tf],
            'target_idx': gene_to_idx[target],
            'weight': float(row.get('weight', 1.0)),
        })
        tf_set.add(tf)

tf_list = sorted(tf_set)
n_tfs = len(tf_list)
n_edges = len(edges)
print(f"  GRN: {n_edges} edges, {n_tfs} TFs")

# Z-score expression
expr_vals = expr_df.values
expr_mean = expr_vals.mean(axis=0)
expr_std = expr_vals.std(axis=0) + 1e-10
# Z-score expression per gene across cells, clipped to [-3, 3]
expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)

# TF → edge index map
# Index edges by source TF for the per-TF activity loop
tf_to_edge_idx = defaultdict(list)
for e_idx, e in enumerate(edges):
    tf_to_edge_idx[e['tf']].append(e_idx)

# Per-cell per-TF GRN contribution
# Per-cell per-TF GRN contribution: sum over the TF's edges of w * |z_TF| * |z_target|
tf_activity = np.zeros((n_cells, n_tfs))
for tf_i, tf_name in enumerate(tf_list):
    e_indices = tf_to_edge_idx[tf_name]
    if not e_indices:
        continue
    for e_idx in e_indices:
        e = edges[e_idx]
        bw = e['weight']
        for cell_i in range(n_cells):
            tf_activity[cell_i, tf_i] += bw * abs(expr_z[cell_i, e['tf_idx']]) * abs(expr_z[cell_i, e['target_idx']])

tf_act_df = pd.DataFrame(tf_activity, index=keep_cells, columns=tf_list)
cell_vn = sub_entropy['vn_entropy'].values
print(f"  TF activity matrix: {tf_act_df.shape}")

# ═══════════════════════════════════════════════════════════
# 3. TF × Entropy correlation by cell type & tissue origin
# ═══════════════════════════════════════════════════════════
print("\n[3/4] TF-NetITH correlation by cell type...")

cell_types = sorted(sub_entropy['Cell_type'].unique())
# Per cell type, Spearman TF activity vs NetITH (null: rho = 0; n >= 20 cells)
ct_results = []

for ct in cell_types:
    ct_mask = sub_entropy['Cell_type'] == ct
    ct_indices = np.where(ct_mask)[0]
    if len(ct_indices) < 20:
        continue
    
    ct_act = tf_act_df.iloc[ct_indices]
    ct_vn = cell_vn[ct_indices]
    
    for tf in tf_list:
        rho, p = spearmanr(ct_act[tf], ct_vn)
        ct_results.append({
            'cell_type': ct, 'tf': tf,
            'rho': rho, 'p_value': p,
            'n_cells': len(ct_indices)
        })

ct_tf_df = pd.DataFrame(ct_results)

print("  Top 3 TFs per cell type (|ρ|):")
for ct in cell_types:
    ct_sub = ct_tf_df[ct_tf_df['cell_type'] == ct]
    if len(ct_sub) == 0:
        continue
    top3 = ct_sub.iloc[ct_sub['rho'].abs().argsort()[-3:][::-1]]
    print(f"    {ct:20s}: " + ", ".join(
        f"{row['tf']}(ρ={row['rho']:+.3f})" for _, row in top3.iterrows()
    ))

# ═══════════════════════════════════════════════════════════
# 4. Compare with GDSC bulk TF module
# ═══════════════════════════════════════════════════════════
print("\n[4/4] GDSC bulk → Single-cell concordance...")

# Load GDSC LASSO coefficients
# Load GDSC bulk LASSO TF coefficients as the bulk reference module
gdsc_lasso = pd.read_csv(OUTPUT_DIR / "tf_lasso_coefficients.csv")
gdsc_top = gdsc_lasso.head(12)

# Map GDSC TFs to single-cell
sc_tf_summary = []
for _, row in gdsc_top.iterrows():
    tf = row['TF']
    if tf not in tf_list:
        continue
    gdsc_coef = row['lasso_coef']
    for ct in cell_types:
        ct_sub = ct_tf_df[(ct_tf_df['cell_type'] == ct) & (ct_tf_df['tf'] == tf)]
        if len(ct_sub) > 0:
            sc_tf_summary.append({
                'tf': tf,
                'gdsc_lasso_beta': gdsc_coef,
                'cell_type': ct,
                'sc_rho': ct_sub.iloc[0]['rho'],
                'sc_p': ct_sub.iloc[0]['p_value'],
                'sc_n': ct_sub.iloc[0]['n_cells'],
            })

sc_summary_df = pd.DataFrame(sc_tf_summary)

# Aggregate per TF
if len(sc_summary_df) > 0:
    tf_concord = sc_summary_df.groupby('tf').agg(
        mean_sc_rho=('sc_rho', 'mean'),
        gdsc_beta=('gdsc_lasso_beta', 'first'),
        n_ct=('cell_type', 'count'),
    ).reset_index()
    
    # Direction concordance: sign(GDSC lasso beta) == sign(mean single-cell rho)
    tf_concord['same_sign'] = (np.sign(tf_concord['gdsc_beta']) == np.sign(tf_concord['mean_sc_rho']))
    n_concordant = tf_concord['same_sign'].sum()
    print(f"  GDSC→SC direction concordance: {n_concordant}/{len(tf_concord)}")
    for _, row in tf_concord.iterrows():
        status = '✓' if row['same_sign'] else '✗'
        print(f"    {status} {row['tf']:12s}: GDSC β={row['gdsc_beta']:+.3f}, SC ρ={row['mean_sc_rho']:+.3f}")
else:
    tf_concord = pd.DataFrame(columns=['tf', 'mean_sc_rho', 'gdsc_beta', 'n_ct', 'same_sign'])
    n_concordant = 0

# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════
print("\nCreating visualizations...")

cell_types_plot = [ct for ct in cell_types if ct in ct_tf_df['cell_type'].values]
n_ct = len(cell_types_plot)

# 6-panel figure: TF x cell-type heatmap, JUN by origin, GDSC vs SC, R^2 bars
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('GSE131907 Single-Cell TF Contribution: GDSC Bulk → Single-Cell Bridge',
             fontsize=14, fontweight='bold', y=0.99)

# Panel A: TF-Cell Type ρ heatmap
ax = axes[0, 0]
top_tfs_all = gdsc_top['TF'].values[:10]
top_tfs_present = [t for t in top_tfs_all if t in tf_list]

if top_tfs_present:
    heatmap_data = np.zeros((len(top_tfs_present), n_ct))
    for i, tf in enumerate(top_tfs_present):
        for j, ct in enumerate(cell_types_plot):
            ct_sub = ct_tf_df[(ct_tf_df['cell_type'] == ct) & (ct_tf_df['tf'] == tf)]
            if len(ct_sub) > 0:
                heatmap_data[i, j] = ct_sub.iloc[0]['rho']
    
    im = ax.imshow(heatmap_data, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(n_ct))
    ax.set_xticklabels([ct[:15] for ct in cell_types_plot], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top_tfs_present)))
    ax.set_yticklabels(top_tfs_present, fontsize=8)
    ax.set_title('A: TF-Cell Type ρ (GDSC Top 10 TFs)', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

# Panel B: JUN activity across tissue origins
ax = axes[0, 1]
if 'JUN' in tf_list:
    jun_act = tf_act_df['JUN'].values
    origins = sub_entropy['Sample_Origin'].values
    
    origin_order = ['nLung', 'tLung', 'tL/B', 'nLN', 'mLN', 'mBrain', 'PE']
    origin_data, origin_labels = [], []
    for o in origin_order:
        o_mask = origins == o
        if o_mask.sum() > 20:
            origin_data.append(jun_act[o_mask])
            origin_labels.append(o)
    
    if origin_data:
        bp = ax.boxplot(origin_data, labels=origin_labels, patch_artist=True)
        colors_o = plt.cm.viridis(np.linspace(0.2, 0.9, len(origin_data)))
        for patch, c in zip(bp['boxes'], colors_o):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel('JUN GRN Contribution', fontsize=10)
        ax.set_title('B: JUN Activity Across Tissue Origins', fontsize=10)
        ax.tick_params(axis='x', rotation=45)

# Panel C: GDSC bulk β vs SC ρ
ax = axes[0, 2]
if len(tf_concord) > 0:
    ax.scatter(tf_concord['gdsc_beta'], tf_concord['mean_sc_rho'],
               c=['#2ca02c' if s else '#d62728' for s in tf_concord['same_sign']],
               s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    for _, row in tf_concord.iterrows():
        ax.annotate(row['tf'], (row['gdsc_beta'], row['mean_sc_rho']),
                    fontsize=7, ha='center', va='bottom')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('GDSC Bulk LASSO β', fontsize=11)
    ax.set_ylabel('GSE131907 SC Mean ρ', fontsize=11)
    ax.set_title(f'C: GDSC vs SC TF Effects\n({n_concordant}/{len(tf_concord)} concordant)', fontsize=10)

# Panel D: NetITH by cell type
ax = axes[1, 0]
ct_vn_data, ct_labels_plot = [], []
for ct in cell_types_plot[:8]:
    ct_mask = sub_entropy['Cell_type'] == ct
    if ct_mask.sum() > 20:
        ct_vn_data.append(cell_vn[ct_mask])
        ct_labels_plot.append(ct[:15])

if ct_vn_data:
    bp_ct = ax.boxplot(ct_vn_data, labels=ct_labels_plot, patch_artist=True)
    colors_ct = plt.cm.Set3(np.linspace(0, 1, len(ct_labels_plot)))
    for patch, c in zip(bp_ct['boxes'], colors_ct):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel('von Neumann Entropy', fontsize=10)
    ax.set_title('D: Single-Cell Entropy by Cell Type', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

# Panel E: TF R² with entropy
ax = axes[1, 1]
# Per-TF squared Spearman rho with NetITH as the entropy-variance explained
tf_var_explained = []
for tf in tf_list:
    rho, _ = spearmanr(tf_act_df[tf], cell_vn)
    tf_var_explained.append({'tf': tf, 'rho_sq': rho**2})
tf_var_df = pd.DataFrame(tf_var_explained).sort_values('rho_sq', ascending=False)

top_tf_var = tf_var_df.head(15)
ax.barh(range(len(top_tf_var)), top_tf_var['rho_sq'].values[::-1],
        color='#8c564b', alpha=0.8, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(top_tf_var)))
ax.set_yticklabels(top_tf_var['tf'].values[::-1], fontsize=8)
ax.set_xlabel('R² (ρ² × Entropy)', fontsize=11)
ax.set_title('E: TF Contribution to SC Entropy (R²)', fontsize=10)

# Panel F: Summary
ax = axes[1, 2]
ax.axis('off')

if len(tf_concord) > 0:
    concord_rate = f"{n_concordant/len(tf_concord):.0%}"
    concord_str = f"  Concordant: {n_concordant}/{len(tf_concord)}\n  Rate: {concord_rate}"
else:
    concord_rate = "N/A"
    concord_str = "  No GDSC TFs present in SC data"

summary_text = f"""
GSE131907 Single-Cell TF Decomposition

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells: {n_cells}
Cell types: {n_ct}
TFs: {n_tfs}, Edges: {n_edges}

GDSC → GSE131907 Concordance:
{concord_str}

Top 5 TFs by SC R²:
"""
for _, row_ in tf_var_df.head(5).iterrows():
    summary_text += f"  {row_['tf']:12s}: R²={row_['rho_sq']:.4f}\n"

summary_text += f"""
Key Question:
Does the GDSC "JUN-ATF4-FOS-STAT1"
module operate similarly at
single-cell resolution in tissue?

Concordance: {concord_rate} of GDSC
top TFs show consistent sign in SC.
"""
if 'JUN' in tf_list:
    jun_rho = tf_var_df[tf_var_df['tf']=='JUN']['rho_sq'].values[0]
    summary_text += f"\nJUN R² (SC): {jun_rho:.4f}"

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=7.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUTPUT_DIR / "figures" / "gse131907_tf_decomposition.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# SAVE
# Save per-cell-type TF correlations, GDSC-SC concordance and per-cell TF activity
ct_tf_df.to_csv(OUTPUT_DIR / "gse131907_tf_celltype_correlation.csv", index=False)
sc_summary_df.to_csv(OUTPUT_DIR / "gse131907_gdsc_sc_concordance.csv", index=False)
tf_act_df.to_csv(OUTPUT_DIR / "gse131907_per_cell_tf_activity.csv")

print("\n" + "=" * 70)
print("GSE131907 SINGLE-CELL TF DECOMPOSITION — COMPLETE")
print("=" * 70)
print(f"""
  Cells: {n_cells}, Cell types: {n_ct}, TFs: {n_tfs}
  Concordance: {n_concordant}/{len(tf_concord)}
""")
