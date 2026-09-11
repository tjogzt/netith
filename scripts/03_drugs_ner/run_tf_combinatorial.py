"""run_tf_combinatorial.py — TF combinatorial determinants of NetITH (LASSO + Random Forest).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/focused_genes_collectri.txt; results/gdsc/gdsc_netith_cell_lines.csv; /tmp/collectri_net.pkl (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{tf_lasso_coefficients.csv, tf_rf_importance.csv, tf_combinatorial_model_comparison.csv}; results/depmap/figures/tf_combinatorial_determinants.png
Pipeline: drug-ner stage — see repository README
"""

import numpy as np
import pandas as pd
import os, sys, pickle, warnings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
DATA_DIR = f"{DATA_ROOT}/gdsc"
GDSC_OUTPUT = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 1. Build per-cell-line TF GRN contribution matrix
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("[1/6] Building TF activity matrix (GRN contribution per cell line)...")

# Load expression
expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)

# Build cell name map
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

# Load focused genes
focused_file = f"{ROOT}/results/focused_genes_collectri.txt"
with open(focused_file) as f:
    focused = set(line.strip() for line in f if line.strip())

common_genes = sorted(focused & set(expr_sym.index))
print(f"  Focused genes: {len(common_genes)}")

expr_sub = expr_sym.loc[common_genes]
gene_to_idx = {g: i for i, g in enumerate(common_genes)}
n_genes = len(common_genes)
n_cells = expr_sub.shape[1]

# Load NetITH
netith_df = pd.read_csv(GDSC_OUTPUT / "gdsc_netith_cell_lines.csv", index_col=0)
netith = netith_df['NetITH']

# Load CollecTRI edges
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
print(f"  CollecTRI edges: {len(edges)}, Unique TFs: {n_tfs}")

# Z-score expression
expr_vals = expr_sub.values.T
expr_mean = expr_vals.mean(axis=0)
expr_std = expr_vals.std(axis=0) + 1e-10
expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)

# Common cell lines
cell_names = expr_sub.columns.tolist()
common_cells = sorted(set(cell_names) & set(netith.index))
cell_idx_map = {c: i for i, c in enumerate(cell_names) if c in common_cells}
common_indices = [cell_idx_map[c] for c in common_cells]
n_common = len(common_cells)
print(f"  Cell lines: {n_common}")

# Build TF → edges index map
tf_to_edge_idx = defaultdict(list)
for e_idx, e in enumerate(edges):
    tf_to_edge_idx[e['tf']].append(e_idx)

# Compute per-cell-line TF total outgoing weight
# For each TF: sum over all outgoing edges of weight * |TF_z| * |target_z|
tf_activity = np.zeros((n_common, n_tfs))
for tf_i, tf_name in enumerate(tf_list):
    e_indices = tf_to_edge_idx[tf_name]
    if not e_indices:
        continue
    for e_idx in e_indices:
        e = edges[e_idx]
        bw = e['weight']
        tgt_idx = e['target_idx']
        for j, cell_i in enumerate(common_indices):
            tf_activity[j, tf_i] += bw * abs(expr_z[cell_i, e['tf_idx']]) * abs(expr_z[cell_i, tgt_idx])

# Build DataFrame
tf_act_df = pd.DataFrame(tf_activity, index=common_cells, columns=tf_list)
netith_sub = netith.loc[common_cells]

print(f"  TF activity matrix: {tf_act_df.shape}")
print(f"  NetITH range: [{netith_sub.min():.3f}, {netith_sub.max():.3f}]")

# ═══════════════════════════════════════════════════════════
# 2. Baseline: JUN-only model
# ═══════════════════════════════════════════════════════════
print("\n[2/6] Baseline: JUN-only model...")

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

jun_col = 'JUN' if 'JUN' in tf_act_df.columns else None
if jun_col:
    r2_jun_simple = pearsonr(tf_act_df[jun_col], netith_sub)[0] ** 2
    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        LinearRegression(), tf_act_df[[jun_col]].values, netith_sub.values,
        cv=cv, scoring='r2'
    )
    print(f"  JUN-only R²: {r2_jun_simple:.4f}")
    print(f"  JUN-only CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
else:
    print("  WARNING: JUN not found in TF list!")
    r2_jun_simple = 0
    cv_scores = None

# ═══════════════════════════════════════════════════════════
# 3. LASSO Regression — Sparse TF Selection
# ═══════════════════════════════════════════════════════════
print("\n[3/6] LASSO regression (sparse TF selection)...")

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

X = tf_act_df.values
y = netith_sub.values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO CV for optimal alpha
lasso_cv = LassoCV(cv=5, random_state=SEED, max_iter=5000, n_alphas=100,
                    eps=1e-4, n_jobs=1)
lasso_cv.fit(X_scaled, y)
print(f"  Optimal alpha: {lasso_cv.alpha_:.6f}")
print(f"  CV R²: {lasso_cv.score(X_scaled, y):.4f}")

# Get selected TFs (non-zero coefficients)
lasso_coef = pd.Series(lasso_cv.coef_, index=tf_list)
selected_tfs = lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=False)
print(f"  Selected TFs: {len(selected_tfs)}/{n_tfs}")
print(f"\n  Top 15 selected TFs:")
for tf_name, coef in selected_tfs.head(15).items():
    print(f"    {tf_name:15s}: β={coef:+.4f}")

# ═══════════════════════════════════════════════════════════
# 4. Random Forest — Non-linear feature importance
# ═══════════════════════════════════════════════════════════
print("\n[4/6] Random Forest (non-linear interactions)...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=SEED,
                            n_jobs=-1, min_samples_leaf=5)
rf.fit(X_scaled, y)

# OOB score if available, otherwise CV
cv_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
rf.fit(X_scaled, y)  # re-fit on full data
print(f"  RF CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(f"  RF Full R²: {rf.score(X_scaled, y):.4f}")

# Feature importance (impurity-based)
rf_importance = pd.Series(rf.feature_importances_, index=tf_list).sort_values(ascending=False)

# Permutation importance (more robust)
perm_imp = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=SEED, n_jobs=-1)
perm_importance_df = pd.Series(perm_imp.importances_mean, index=tf_list).sort_values(ascending=False)

print(f"\n  Top 15 TFs (RF impurity importance):")
for tf_name, imp in rf_importance.head(15).items():
    print(f"    {tf_name:15s}: {imp:.4f}")

print(f"\n  Top 15 TFs (Permutation importance):")
for tf_name, imp in perm_importance_df.head(15).items():
    print(f"    {tf_name:15s}: {imp:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. TF-TF Interaction Network
# ═══════════════════════════════════════════════════════════
print("\n[5/6] TF-TF interaction & clustering...")

# Correlation among top TFs
top_n = min(20, n_tfs)
top_tfs = rf_importance.head(top_n).index.tolist()
tf_corr = tf_act_df[top_tfs].corr(method='spearman')

# Hierarchical clustering of TFs
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

dist = 1 - tf_corr.abs()
linkage_matrix = linkage(squareform(dist.values), method='ward')
# Cut into 3-5 clusters
for k in [3, 4, 5]:
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    cluster_map = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_map[c].append(tf_corr.index[i])
    print(f"\n  k={k} clusters:")
    for c_id, members in cluster_map.items():
        print(f"    Cluster {c_id}: {', '.join(members[:8])}{'...' if len(members)>8 else ''}")

# ═══════════════════════════════════════════════════════════
# 6. Model Comparison & Summary
# ═══════════════════════════════════════════════════════════
print("\n[6/6] Model comparison...")

# Compare models
results = {
    'Model': [],
    'R² (Full)': [],
    'R² (CV mean)': [],
    'R² (CV std)': [],
    'n_features': [],
}

# Null model (mean prediction)
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean')
dummy_cv = cross_val_score(dummy, X_scaled, y, cv=5, scoring='r2')
results['Model'].append('Mean (null)')
results['R² (Full)'].append(0)
results['R² (CV mean)'].append(dummy_cv.mean())
results['R² (CV std)'].append(dummy_cv.std())
results['n_features'].append(0)

# JUN-only
if jun_col:
    jun_idx = tf_list.index(jun_col)
    jun_cv = cross_val_score(LinearRegression(), X_scaled[:, [jun_idx]], y, cv=5, scoring='r2')
    results['Model'].append('JUN only')
    results['R² (Full)'].append(r2_jun_simple)
    results['R² (CV mean)'].append(jun_cv.mean())
    results['R² (CV std)'].append(jun_cv.std())
    results['n_features'].append(1)

# All TFs linear
lin_cv = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring='r2')
lin_r2 = LinearRegression().fit(X_scaled, y).score(X_scaled, y)
results['Model'].append('Linear (all TFs)')
results['R² (Full)'].append(lin_r2)
results['R² (CV mean)'].append(lin_cv.mean())
results['R² (CV std)'].append(lin_cv.std())
results['n_features'].append(n_tfs)

# LASSO
lasso_cv_score = cross_val_score(
    Lasso(alpha=lasso_cv.alpha_, max_iter=5000, random_state=SEED),
    X_scaled, y, cv=5, scoring='r2'
)
results['Model'].append(f'LASSO ({len(selected_tfs)} TFs)')
results['R² (Full)'].append(lasso_cv.score(X_scaled, y))
results['R² (CV mean)'].append(lasso_cv_score.mean())
results['R² (CV std)'].append(lasso_cv_score.std())
results['n_features'].append(len(selected_tfs))

# Random Forest
results['Model'].append('Random Forest')
results['R² (Full)'].append(rf.score(X_scaled, y))
results['R² (CV mean)'].append(cv_rf.mean())
results['R² (CV std)'].append(cv_rf.std())
results['n_features'].append(n_tfs)

model_df = pd.DataFrame(results)
print("\n  Model Comparison:")
print(model_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Saving results...")

# LASSO coefficients
lasso_coef_df = lasso_coef.reset_index()
lasso_coef_df.columns = ['TF', 'lasso_coef']
lasso_coef_df['abs_coef'] = lasso_coef_df['lasso_coef'].abs()
lasso_coef_df = lasso_coef_df.sort_values('abs_coef', ascending=False)
lasso_coef_df.to_csv(OUTPUT_DIR / "tf_lasso_coefficients.csv", index=False)

# RF importance
rf_imp_df = pd.DataFrame({
    'TF': rf_importance.index,
    'rf_impurity': rf_importance.values,
    'rf_permutation': perm_importance_df.loc[rf_importance.index].values
})
rf_imp_df.to_csv(OUTPUT_DIR / "tf_rf_importance.csv", index=False)

# Model comparison
model_df.to_csv(OUTPUT_DIR / "tf_combinatorial_model_comparison.csv", index=False)

# ═══════════════════════════════════════════════════════════
# VISUALIZATION — 6-panel figure
# ═══════════════════════════════════════════════════════════
print("Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.suptitle('TF Combinatorial Determinants of NetITH', fontsize=14, fontweight='bold', y=0.98)

# Panel A: LASSO coefficient path
ax = axes[0, 0]
alphas = lasso_cv.alphas_
coef_paths = np.zeros((n_tfs, len(alphas)))
for i, alpha in enumerate(alphas):
    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
    lasso.fit(X_scaled, y)
    coef_paths[:, i] = lasso.coef_

# Plot top 10 TFs
top10_path = lasso_coef.abs().nlargest(10).index
colors_path = plt.cm.tab10(np.linspace(0, 1, 10))
for i, tf in enumerate(top10_path):
    tf_idx = tf_list.index(tf)
    ax.plot(np.log10(alphas), coef_paths[tf_idx, :], color=colors_path[i],
            label=tf, linewidth=1.5, alpha=0.8)
ax.axvline(x=np.log10(lasso_cv.alpha_), color='red', linestyle='--', linewidth=1.5, label='CV optimal')
ax.set_xlabel('log₁₀(α)', fontsize=11)
ax.set_ylabel('Coefficient', fontsize=11)
ax.set_title('LASSO Coefficient Paths (Top 10 TFs)', fontsize=10)
ax.legend(fontsize=6, loc='upper right', ncol=2)
ax.axhline(y=0, color='gray', linewidth=0.5)

# Panel B: LASSO coefficients barplot
ax = axes[0, 1]
top_coef = lasso_coef_df.head(20)
colors_b = ['#d62728' if c < 0 else '#2ca02c' for c in top_coef['lasso_coef']]
ax.barh(range(len(top_coef)), top_coef['lasso_coef'].values[::-1], color=colors_b[::-1],
        edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(top_coef)))
ax.set_yticklabels(top_coef['TF'].values[::-1], fontsize=8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('LASSO Coefficient (β)', fontsize=11)
ax.set_title(f'LASSO: {len(selected_tfs)} Selected TFs', fontsize=10)

# Panel C: Model comparison R²
ax = axes[0, 2]
x_pos = np.arange(len(model_df))
bars = ax.bar(x_pos, model_df['R² (CV mean)'], yerr=model_df['R² (CV std)'],
              capsize=5, color=['gray', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
              edgecolor='darkgray', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_df['Model'], rotation=25, ha='right', fontsize=8)
ax.set_ylabel('CV R²', fontsize=11)
ax.set_title('Model Comparison (5-fold CV)', fontsize=10)
ax.axhline(y=0, color='gray', linewidth=0.5)
# Add R² values on bars
for bar, r2 in zip(bars, model_df['R² (CV mean)']):
    ax.text(bar.get_x() + bar.get_width()/2, max(0, r2) + 0.02,
            f'{r2:.3f}', ha='center', fontsize=8, fontweight='bold')

# Panel D: RF vs LASSO importance comparison
ax = axes[1, 0]
common_tfs = sorted(set(rf_importance.head(25).index) | set(lasso_coef.abs().nlargest(25).index))[:25]
rf_vals = [rf_importance.get(tf, 0) for tf in common_tfs]
lasso_vals = [abs(lasso_coef.get(tf, 0)) for tf in common_tfs]
# Normalize to [0,1]
rf_norm = np.array(rf_vals) / max(rf_vals) if max(rf_vals) > 0 else np.array(rf_vals)
lasso_norm = np.array(lasso_vals) / max(lasso_vals) if max(lasso_vals) > 0 else np.array(lasso_vals)

x = np.arange(len(common_tfs))
w = 0.35
ax.barh(x + w/2, rf_norm, w, label='RF Importance', color='#1f77b4', alpha=0.8)
ax.barh(x - w/2, lasso_norm, w, label='|LASSO β|', color='#ff7f0e', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(common_tfs, fontsize=7)
ax.set_xlabel('Normalized Importance', fontsize=11)
ax.set_title('RF vs LASSO: Top TF Convergence', fontsize=10)
ax.legend(fontsize=9)

# Panel E: TF-TF correlation heatmap (top TFs)
ax = axes[1, 1]
top_heat = min(15, len(top_tfs))
hm_tfs = top_tfs[:top_heat]
hm = tf_corr.loc[hm_tfs, hm_tfs]
im = ax.imshow(hm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(hm_tfs)))
ax.set_xticklabels(hm_tfs, rotation=90, fontsize=6)
ax.set_yticks(range(len(hm_tfs)))
ax.set_yticklabels(hm_tfs, fontsize=6)
ax.set_title('TF-TF Activity Correlations (Top TFs)', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel F: Summary text
ax = axes[1, 2]
ax.axis('off')
# Determine dominant TF clusters
cluster_text = ""
for c_id, members in sorted(cluster_map.items()):
    cluster_text += f"  C{c_id}: {', '.join(members[:5])}{'...' if len(members)>5 else ''}\n"

summary_text = f"""
TF-combination determinants: NetITH prediction models

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cell lines: {n_common}, TFs: {n_tfs}

Model comparison (CV R²):
  Null:   {dummy_cv.mean():.4f}
  JUN:    {jun_cv.mean():.4f} (single TF)
  LASSO:  {lasso_cv_score.mean():.4f} ({len(selected_tfs)} TFs)
  RF:     {cv_rf.mean():.4f} (all TFs)

LASSO key findings:
  Selected TFs: {len(selected_tfs)}/{n_tfs}
  Optimal alpha: {lasso_cv.alpha_:.6f}

Top 5 TFs (LASSO |β|):
"""
for i, (tf, coef) in enumerate(lasso_coef.abs().nlargest(5).items()):
    summary_text += f"  {i+1}. {tf}: |β|={coef:.4f}\n"

summary_text += f"\nTF modules (k={len(cluster_map)}):\n{cluster_text}"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=8.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = OUTPUT_DIR / "figures" / "tf_combinatorial_determinants.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TF COMBINATORIAL DETERMINANTS — COMPLETE")
print("=" * 70)
print(f"""
Key Results:
  Cell lines:           {n_common}
  TFs analyzed:         {n_tfs}
  JUN R² (single TF):   {r2_jun_simple:.4f}
  LASSO R²:             {lasso_cv.score(X_scaled, y):.4f} ({len(selected_tfs)} TFs)
  RF R²:                {rf.score(X_scaled, y):.4f}
  
  Top TF by LASSO:      {selected_tfs.head(1).index[0]}
  RF Permutation top:   {perm_importance_df.head(1).index[0]}
  
  TF Clusters found:    {len(cluster_map)}
  
Output files:
  {OUTPUT_DIR}/tf_lasso_coefficients.csv
  {OUTPUT_DIR}/tf_rf_importance.csv
  {OUTPUT_DIR}/tf_combinatorial_model_comparison.csv
  {OUTPUT_DIR}/figures/tf_combinatorial_determinants.png
""")
