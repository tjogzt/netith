"""complete_perturbation_analysis.py — Elastic-Net top-feature NetITH regression and perturbation independence tests.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/rna_expr.csv, data/gdsc/cell_annot.csv, data/gdsc/ensg_symbol_map.csv; results/gdsc/gdsc_netith_cell_lines.csv; results/depmap/perturbation/virtual_perturbation_genome_wide.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/perturbation/{lasso_top1000.csv, independence_netith_vs_exprvar.csv, perturbation_effect_sizes.csv}
Pipeline: drug-ner stage — see repository README
"""
import os
import numpy as np
import pandas as pd
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

SEED = 42
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
GDSC_DIR = DATA_ROOT / "gdsc"
RESULTS_GDSC = ROOT / "results" / "gdsc"
OUT_DIR = ROOT / "results" / "depmap" / "perturbation"

# ─── Load GDSC expression + NetITH ─────────────────────────
# Load GDSC expression (ENSG rows), rename CEL columns to cell line names, and map ENSG to symbols
expr_raw = pd.read_csv(GDSC_DIR / "rna_expr.csv", index_col=0)
annot = pd.read_csv(GDSC_DIR / "cell_annot.csv", index_col=0)
gene_map = pd.read_csv(GDSC_DIR / "ensg_symbol_map.csv")

cel_to_cl = {}
for cel, row in annot.iterrows():
    cl = str(row.get('Characteristics.cell.line.', ''))
    if cl and cl != 'nan' and cl != 'NA':
        cel_to_cl[cel] = cl

common_cels = [c for c in expr_raw.columns if c in cel_to_cl]
expr_named = expr_raw[common_cels].copy()
expr_named.columns = [cel_to_cl[c] for c in common_cels]
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]
ensg_to_sym = dict(zip(gene_map['ensg'], gene_map['symbol']))
expr_named.index = expr_named.index.astype(str)
matched = expr_named.index.isin(ensg_to_sym.keys())
expr_sym = expr_named.loc[matched].copy()
expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]

# Load per-cell-line NetITH and restrict expression and NetITH to common cell lines
netith_df = pd.read_csv(RESULTS_GDSC / "gdsc_netith_cell_lines.csv", index_col=0)
netith_map = netith_df['NetITH'].to_dict()
common_cl = sorted(set(expr_sym.columns) & set(netith_map.keys()))
expr_sym = expr_sym[common_cl]
cell_lines = common_cl
netith_vals = np.array([netith_map[cl] for cl in cell_lines])
all_genes = sorted(expr_sym.index.tolist())

# ─── Elastic Net Top5K ──────────────────────────────────────
print("=" * 60)
print("Elastic Net: Top 5,000 Most Variable Genes → NetITH")
print("=" * 60)

# Feature selection: rank genes by cross-cell-line expression variance, keep the top N_FEAT
gene_vars = np.nanvar(expr_sym.values.astype(float), axis=1)
topN_idx = np.argsort(gene_vars)[-1000:]
topN_genes = [all_genes[i] for i in topN_idx]
N_FEAT = 1000

Xbig = expr_sym.iloc[topN_idx].T.values.astype(float)
y = netith_vals
validB = ~np.isnan(Xbig).any(axis=1) & ~np.isnan(y)
X_c, y_c = Xbig[validB], y[validB]

# Use Lasso (L1 only, faster) + Ridge for comparison
from sklearn.linear_model import LassoCV, RidgeCV
s_big = StandardScaler()
X_s = s_big.fit_transform(X_c)

# Ridge baseline: 3-fold CV R2 and Spearman rho between predicted and observed NetITH
# Ridge (fast)
ridge = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=3)
ridge.fit(X_s, y_c)
cv_ridge = cross_val_score(ridge, X_s, y_c, cv=3, scoring='r2')
pred_r = ridge.predict(X_s)
rho_r, _ = spearmanr(pred_r, y_c)

# Lasso (L1-regularized) with 3-fold CV; nonzero coefficients mark genes most predictive of NetITH
# Lasso
lasso = LassoCV(cv=3, random_state=SEED, max_iter=2000, n_alphas=20, eps=1e-3)
lasso.fit(X_s, y_c)
cv_lasso = cross_val_score(lasso, X_s, y_c, cv=3, scoring='r2')
pred_l = lasso.predict(X_s)
rho_l, _ = spearmanr(pred_l, y_c)

coef_l = pd.DataFrame({'gene': topN_genes, 'coef': lasso.coef_})
coef_l = coef_l[coef_l['coef'].abs() > 1e-6].sort_values('coef', key=abs, ascending=False)
print(f"  Ridge:  R²={cv_ridge.mean():.4f}±{cv_ridge.std():.4f}, ρ_pred={rho_r:.4f}")
print(f"  Lasso:  R²={cv_lasso.mean():.4f}±{cv_lasso.std():.4f}, ρ_pred={rho_l:.4f}, {len(coef_l)} non-zero")

print("  Top 25 Lasso β:")
for _, row in coef_l.head(25).iterrows():
    print(f"    {row['gene']:15s}: β={row['coef']:+.5f}")

# Write the Lasso coefficients (nonzero entries only) for the top-feature model
coef_l.to_csv(OUT_DIR / f"lasso_top{N_FEAT}.csv", index=False)

# ─── Independence Test: NetITH Δ vs Expression Variance Δ ──
print("\n" + "=" * 60)
print("Independence Test: NetITH Δ vs Expression Variance Δ")
print("=" * 60)

# Independence test: is the NetITH effect of a gene (from virtual perturbation)
# associated with how much that gene's expression changes cell-level variance?
# For each gene, compute (1) NetITH Δ between high/low quartiles
# and (2) Expression variance change
vp_df = pd.read_csv(OUT_DIR / "virtual_perturbation_genome_wide.csv")

# Gene-level expression variance
expr_vals_all = expr_sym.values.astype(float)
gene_var = np.nanvar(expr_vals_all, axis=1)  # across cell lines

# Compute per-cell expression variance (as a proxy for transcriptomic entropy)
# and see if genes with large NetITH effects also have large effects on
# cell-level expression variance
cell_expr_var = np.nanvar(expr_vals_all, axis=0)

# For top hits, check if high-vs-low expression stratification
# also creates large changes in cellular expression variance
results_ind = []
for i in range(min(500, len(vp_df))):
    row = vp_df.iloc[i]
    gene = row['gene']
    if gene not in all_genes:
        continue
    gidx = all_genes.index(gene)
    # Stratify cell lines into low/high expression quartiles (require >=10 cells per group)
    expr_vec = expr_sym.iloc[gidx].values.astype(float)
    valid = ~np.isnan(expr_vec)
    cutoff_low = np.percentile(expr_vec[valid], 25)
    cutoff_high = np.percentile(expr_vec[valid], 75)
    low_mask = (expr_vec <= cutoff_low) & valid
    high_mask = (expr_vec >= cutoff_high) & valid
    if low_mask.sum() < 10 or high_mask.sum() < 10:
        continue
    
    # Compute cell-level expression variance in low vs high groups
    var_low = np.nanvar(expr_vals_all[:, low_mask], axis=1)
    var_high = np.nanvar(expr_vals_all[:, high_mask], axis=1)
    
    # Mean per-cell variance in each group
    mean_var_low = np.nanmean(np.nanvar(expr_vals_all[:, low_mask], axis=0))
    mean_var_high = np.nanmean(np.nanvar(expr_vals_all[:, high_mask], axis=0))
    
    results_ind.append({
        'gene': gene,
        'netith_delta': row['delta_netith'],
        'netith_cohens_d': row['cohens_d'],
        'expr_variance_delta': mean_var_high - mean_var_low,
        'expr_variance_ratio': mean_var_high / mean_var_low if mean_var_low > 1e-10 else 1.0,
    })

ind_df = pd.DataFrame(results_ind)
ind_df.to_csv(OUT_DIR / "independence_netith_vs_exprvar.csv", index=False)

# Spearman test (null: no monotonic association) between |NetITH effect| and |expression-variance change|
rho_ind, p_ind = spearmanr(ind_df['netith_cohens_d'].abs(), ind_df['expr_variance_delta'].abs())
print(f"  Top 500: |d_NetITH| vs |ΔVar_expr|: ρ={rho_ind:.4f}, p={p_ind:.2e}")
print(f"  # genes with |ΔVar| > 0.5: {(ind_df['expr_variance_delta'].abs()>0.5).sum()}")

# Global: for all 17K genes, is NetITH Δ correlated with the gene's
# own expression Δ between high/low groups?
# This tests: "do genes whose expression changes NetITH also show larger expression swings?"
delta_data = vp_df[['gene','delta_netith','cohens_d']].copy()
delta_data['abs_d'] = delta_data['cohens_d'].abs()
# Write per-gene perturbation effect sizes for downstream use
delta_data.to_csv(OUT_DIR / "perturbation_effect_sizes.csv", index=False)

print("\nDone.")
