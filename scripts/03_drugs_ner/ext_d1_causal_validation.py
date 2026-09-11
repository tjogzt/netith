"""ext_d1_causal_validation.py — NetITH causal validation via PC/LiNGAM, 2SLS IV and CRISPR stratification (Extended Data Fig. 1).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv; data/CRISPRGeneEffect.csv; data/external/depmap_metadata.csv; results/depmap/tf_rf_importance.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/ext_d1_{crispr_netith_correlations, iv_analysis, causal_discovery, crispr_stratification}.csv
Pipeline: drug-ner stage — see repository README
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import warnings
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
warnings.filterwarnings("ignore")

GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
DATA_DIR = f"{DATA_ROOT}/gdsc"
DEPMAP_CRISPR = f"{DATA_ROOT}/CRISPRGeneEffect.csv"
RESULTS_DIR = Path(f"{ROOT}/results/depmap")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("Track 1: NetITH Causal Validation")
print("=" * 70)

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
print("\n[1] Loading data...", flush=True)

# --- GDSC expression ---
# Rename CEL columns to cell line names and convert the ENSG index to gene symbols
expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
ensg_to_sym = dict(zip(gene_map['ensg'].astype(str), gene_map['symbol']))
sym_to_ensg = dict(zip(gene_map['symbol'], gene_map['ensg'].astype(str)))

# Map GDSC cell IDs to cell line names
cell_line_map = {}
for cel, row in annot.iterrows():
    cl = str(row.get('Characteristics.cell.line.', ''))
    if cl and cl != 'nan' and cl != 'NA':
        cell_line_map[cel] = cl

expr_named = expr_raw.copy()
expr_named.columns = [cell_line_map.get(c, c) for c in expr_raw.columns]
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]
expr_named.index = expr_named.index.astype(str)

# Convert to gene symbols
matched_idx = expr_named.index.isin(ensg_to_sym.keys())
expr_sym = expr_named.loc[matched_idx].copy()
expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]
print(f"  GDSC expression (symbols): {expr_sym.shape[0]} genes × {expr_sym.shape[1]} cells")

# --- NetITH ---
netith = pd.read_csv(
    f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv",
    index_col=0
)
print(f"  NetITH: {len(netith)} cell lines")

# --- Drug IC50 ---
ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
ic50_mat = ic50_raw.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)
print(f"  IC50: {ic50_mat.shape[0]} cells × {ic50_mat.shape[1]} drugs")

# --- DepMap CRISPR ---
print("  Loading DepMap CRISPR (this may take a minute)...", flush=True)
# Load DepMap CRISPR gene-effect scores (negative values = stronger dependency) as IV instruments
crispr_raw = pd.read_csv(DEPMAP_CRISPR, index_col=0)
# Parse gene names from columns: "A1BG (1)" → "A1BG"
crispr_raw.columns = [c.split(' (')[0] for c in crispr_raw.columns]
# Strip trailing space/newlines
crispr_raw.index = crispr_raw.index.str.strip()
print(f"  CRISPR: {crispr_raw.shape[0]} cell lines × {crispr_raw.shape[1]} genes")

# --- Map DepMap cell lines to GDSC ---
# DepMap uses COSMIC IDs or cell line names. Try direct name matching first.
depmap_names = set(crispr_raw.index)
gdsc_names = set(netith.index)
common_names = sorted(depmap_names & gdsc_names)
print(f"  Direct name match: {len(common_names)} cell lines")

# Use DepMap metadata to map ACH IDs → cell line names → GDSC names
print("  Mapping DepMap ACH IDs → GDSC cell line names via metadata...")
meta = pd.read_csv(
    f"{DATA_ROOT}/external/depmap_metadata.csv"
)
# Build: ACH ID → stripped_cell_line_name
ach_to_stripped = {}
for _, row in meta.iterrows():
    aid = str(row.get('depmap_id', ''))
    sname = str(row.get('stripped_cell_line_name', ''))
    if aid and sname and sname != 'nan':
        ach_to_stripped[aid] = sname.replace(' ', '').replace('-', '').replace('.', '').upper()

# Build: GDSC stripped → original name
gdsc_stripped_to_orig = {}
for n in gdsc_names:
    gdsc_stripped_to_orig[n.replace(' ', '').replace('-', '').replace('.', '').upper()] = n

# Align DepMap and GDSC cell lines: ACH ID → stripped cell line name → GDSC name
# Map: DepMap index (ACH ID) → GDSC cell line name
mapper = {}
for dep_idx in crispr_raw.index:
    dep_stripped = dep_idx.strip().replace(' ', '').replace('-', '').upper()
    # First try direct ACH ID lookup in metadata
    if dep_idx in ach_to_stripped:
        stripped_name = ach_to_stripped[dep_idx]
        if stripped_name in gdsc_stripped_to_orig:
            mapper[dep_idx] = gdsc_stripped_to_orig[stripped_name]
            continue
    # Then try direct matching
    if dep_stripped in gdsc_stripped_to_orig:
        mapper[dep_idx] = gdsc_stripped_to_orig[dep_stripped]

print(f"  Mapped: {len(mapper)} DepMap → GDSC cell lines")

common_names = sorted(set(mapper.values()) & gdsc_names)
print(f"  Final common: {len(common_names)} cell lines")

# Remap CRISPR index to GDSC names
crispr_rev = {v: k for k, v in mapper.items() if v in common_names}
crispr = crispr_raw.loc[[crispr_rev[n] for n in common_names]]
crispr.index = [mapper.get(c, c) for c in crispr.index]

# Align all three datasets
common_all = sorted(set(common_names) & set(ic50_mat.index))
print(f"  Common (expr+NetITH+IC50+CRISPR): {len(common_all)}")

if len(common_all) < 30:
    print("  FATAL: Too few common cell lines. Check DepMap-GDSC mapping.")
    import sys
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# 2. IDENTIFY CRISPR-INSTRUMENT GENES
# ═══════════════════════════════════════════════════════════
print(f"\n[2] Identifying CRISPR-instrument genes...", flush=True)

# Genes where CRISPR dependency correlates with NetITH
# These are "instruments" for IV analysis
n_common_crispr = len([n for n in common_all if n in crispr.index])
crispr_common = crispr.loc[[n for n in common_all if n in crispr.index]]

netith_vals = netith.loc[crispr_common.index, 'NetITH'].values

# Instrument discovery: Spearman test (null: no association) of each gene's CRISPR dependency vs NetITH
crispr_netith_corrs = []
for gene in crispr_common.columns:
    gene_vals = crispr_common[gene].values
    valid = ~(np.isnan(gene_vals) | np.isnan(netith_vals))
    if valid.sum() < 50:
        continue
    rho, p = spearmanr(gene_vals[valid], netith_vals[valid])
    crispr_netith_corrs.append({
        'gene': gene,
        'rho_crispr_netith': rho,
        'p_crispr_netith': p,
        'n': int(valid.sum())
    })

crispr_netith_df = pd.DataFrame(crispr_netith_corrs).sort_values(
    'p_crispr_netith'
)
sig_genes = crispr_netith_df[crispr_netith_df['p_crispr_netith'] < 0.01]
print(f"  Genes with CRISPR→NetITH association (p<0.01): {len(sig_genes)}")
print(f"  Top 10 instrument genes:")
for _, row in sig_genes.head(10).iterrows():
    print(f"    {row['gene']:<25} ρ={row['rho_crispr_netith']:+.4f}  p={row['p_crispr_netith']:.2e}")

# Save
crispr_netith_df.to_csv(RESULTS_DIR / "ext_d1_crispr_netith_correlations.csv", index=False)

# ═══════════════════════════════════════════════════════════
# 3. INSTRUMENTAL VARIABLE (2SLS) ANALYSIS
# ═══════════════════════════════════════════════════════════
print(f"\n[3] Instrumental Variable (2SLS) Analysis...", flush=True)

# For top instrument genes, run 2SLS:
# (Validity assumption: CRISPR dependency is a strong, exogenous instrument whose effect on IC50 runs through NetITH)
# Stage 1: NetITH ~ CRISPR_dependency
# Stage 2: IC50 ~ NetITH_hat
# Causal effect = Stage 2 coefficient

top_instruments = sig_genes.head(20)['gene'].tolist()
iv_results = []

for drug in sorted(ic50_mat.columns)[:200]:  # top 200 drugs
    ic50_col = ic50_mat.loc[common_all, drug]
    ic50_vals = ic50_col.values.astype(float)
    
    for gene in top_instruments:
        if gene not in crispr.columns:
            continue
        
        crispr_vals = crispr.loc[
            [n for n in common_all if n in crispr.index], gene
        ].values
        n_vals = netith.loc[common_all, 'NetITH'].values
        
        # Align: remove NaN
        valid = ~(np.isnan(ic50_vals) | np.isnan(n_vals))
        if gene in crispr.loc[[n for n in common_all if n in crispr.index]].columns:
            crispr_aligned = np.full(len(common_all), np.nan)
            for i, name in enumerate(common_all):
                if name in crispr.index:
                    crispr_aligned[i] = crispr.loc[name, gene]
            valid &= ~np.isnan(crispr_aligned)
        else:
            continue
        
        if valid.sum() < 50:
            continue
        
        ic50_v = ic50_vals[valid]
        n_v = n_vals[valid]
        z_v = crispr_aligned[valid]
        
        # Standardize
        z_s = (z_v - np.nanmean(z_v)) / np.nanstd(z_v)
        n_s = (n_v - np.nanmean(n_v)) / np.nanstd(n_v)
        y_s = (ic50_v - np.nanmean(ic50_v)) / np.nanstd(ic50_v)
        
        # Stage 1: NetITH = α + β*z + ε
        beta_1, _, _, _, _ = stats.linregress(z_s, n_s)
        n_hat = beta_1 * z_s  # predicted NetITH
        
        # F-stat for weak instrument
        ss_reg = np.sum((n_hat - np.mean(n_s))**2)
        ss_res = np.sum((n_s - n_hat)**2)
        F_stat = (ss_reg / 1) / (ss_res / (len(z_s) - 2)) if ss_res > 0 else np.inf
        
        # Stage 2: IC50 = γ + δ*NetITH_hat + ε
        delta, _, _, _, _ = stats.linregress(n_hat, y_s)
        
        # OLS (naive) for comparison
        beta_ols, _, _, _, _ = stats.linregress(n_s, y_s)
        
        # Durbin-Wu-Hausman test: IV vs OLS difference
        # Simplified: compare OLS and IV coefficients
        iv_results.append({
            'drug': drug,
            'instrument_gene': gene,
            'n': int(valid.sum()),
            'F_stat_stage1': F_stat,
            'beta_ols': beta_ols,
            'beta_iv': delta,
            'delta_iv_ols': delta - beta_ols,
            'rho_crispr_netith': spearmanr(z_v, n_v)[0]
        })

iv_df = pd.DataFrame(iv_results)
# Keep only drug–gene pairs whose stage-1 F > 10 (rule-of-thumb: not a weak instrument)
iv_df = iv_df[iv_df['F_stat_stage1'] > 10]  # F>10: not weak instrument
iv_df = iv_df.sort_values('F_stat_stage1', ascending=False)

print(f"  IV analyses with F>10: {len(iv_df)}")
print(f"  IV analyses total: {len(iv_results)}")

if len(iv_df) > 0:
    print(f"\n  IV effect summary:")
    print(f"    Mean β_OLS = {iv_df['beta_ols'].mean():.4f}")
    print(f"    Mean β_IV  = {iv_df['beta_iv'].mean():.4f}")
    print(f"    Mean Δ(IV-OLS) = {iv_df['delta_iv_ols'].mean():.4f}")
    
    # Write 2SLS instrumental-variable results (drug × instrument gene)
    iv_df.to_csv(RESULTS_DIR / "ext_d1_iv_analysis.csv", index=False)
    
    # Top significant IV findings
    sig_iv = iv_df[iv_df['F_stat_stage1'] > 20].head(20)
    print(f"\n  Top 20 IV results (F>20):")
    for _, row in sig_iv.iterrows():
        print(f"    {row['drug']:<30} IV={row['instrument_gene']:<20} "
              f"β_OLS={row['beta_ols']:+.4f} β_IV={row['beta_iv']:+.4f} "
              f"F={row['F_stat_stage1']:.1f}")

# ═══════════════════════════════════════════════════════════
# 4. CAUSAL DISCOVERY (PC + LiNGAM)
# ═══════════════════════════════════════════════════════════
print(f"\n[4] Causal Discovery (PC algorithm)...", flush=True)

# Build causal variables for a simplified model:
# Variables: NetITH, JUN (top TF), FOS, proliferation, drug sensitivity
# We test whether NetITH is a cause or consequence of TF expression

# Select top TFs from existing LASSO results
try:
    tf_importance = pd.read_csv(
        RESULTS_DIR / "tf_rf_importance.csv"
    )
    top_tfs = tf_importance.sort_values('importance', ascending=False).head(6)['tf'].tolist()
except:
    top_tfs = ['JUN', 'FOS', 'JUNB', 'FOSL1', 'ATF3', 'STAT3']

# Build data matrix — all aligned to common_all
causal_vars = {}
for tf in top_tfs:
    if tf in expr_sym.index:
        vals = expr_sym.loc[tf, [c for c in common_all if c in expr_sym.columns]]
        causal_vars[tf] = vals.values.astype(float)

causal_vars['NetITH'] = netith.loc[common_all, 'NetITH'].values

# Proliferation metagene
PROLIF = ['MKI67', 'PCNA', 'MCM2', 'TOP2A', 'CDK1', 'BIRC5', 'CDC20',
          'AURKA', 'AURKB', 'MCM3', 'MCM4', 'MCM5', 'TYMS', 'RRM2']
avail_prolif = [g for g in PROLIF if g in expr_sym.index]
prolif_z = expr_sym.loc[avail_prolif, [c for c in common_all if c in expr_sym.columns]].astype(float)
prolif_z = prolif_z.subtract(prolif_z.mean(axis=1), axis=0).divide(
    prolif_z.std(axis=1) + 1e-10, axis=0)
causal_vars['Proliferation'] = prolif_z.mean(axis=0).values

# Drug sensitivity: use median IC50 across top drugs (aligned to common_all)
drug_n = ic50_mat.loc[common_all].notna().sum()
top_drugs_list = drug_n.sort_values(ascending=False).head(50).index
ic50_subset = ic50_mat.loc[common_all, top_drugs_list]
causal_vars['DrugSensitivity'] = ic50_subset.median(axis=1).values

# Build matrix and remove NaN rows
var_names = list(causal_vars.keys())
X = np.column_stack([causal_vars[v] for v in var_names])
valid_rows = ~np.any(np.isnan(X), axis=1)
X = X[valid_rows]
print(f"  Variables: {var_names}")
print(f"  Complete cases: {X.shape[0]}")

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0)
n_vars = X.shape[1]

# --- PC Algorithm (simplified skeleton discovery) ---
# Step 1: Full graph
# Step 2: Test conditional independence for each edge
# Remove edge if X_i ⟂ X_j | any subset

# Partial correlation of Xi, Xj given Z: regress each on Z and correlate the residuals
def partial_corr(X, i, j, cond_set=None):
    """Compute partial correlation: ρ(Xi, Xj | Z)"""
    if cond_set is None or len(cond_set) == 0:
        return pearsonr(X[:, i], X[:, j])[0]
    # Regress Xi and Xj on Z, then correlate residuals
    Z = np.column_stack([np.ones(X.shape[0]), X[:, list(cond_set)]])
    resid_i = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
    resid_j = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
    return pearsonr(resid_i, resid_j)[0]

# Skeleton discovery (simplified)
print("  Running PC algorithm skeleton discovery...")
alpha = 0.01
n = X.shape[0]
adj = np.ones((n_vars, n_vars)) - np.eye(n_vars)  # full graph, no self-loops
sep_set = {}  # separation sets

# PC skeleton discovery: Fisher z-test of (partial) correlation; remove edges whose independence p > alpha
# Level 0: marginal independence
for i in range(n_vars):
    for j in range(i+1, n_vars):
        r = partial_corr(X, i, j, set())
        z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10)) * np.sqrt(n - 3)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        if p_val > alpha:
            adj[i, j] = adj[j, i] = 0
            sep_set[(i, j)] = set()
            sep_set[(j, i)] = set()

# Level 1+: conditional independence
max_cond = min(3, n_vars - 2)
for l in range(1, max_cond + 1):
    for i in range(n_vars):
        neighbors = [k for k in range(n_vars) if adj[i, k] > 0 and k != i]
        for j in neighbors:
            if j <= i:
                continue
            neighbors_without_j = [k for k in neighbors if k != j]
            if len(neighbors_without_j) < l:
                continue
            # Test subsets of size l
            from itertools import combinations
            found_indep = False
            for cond in combinations(neighbors_without_j, l):
                r = partial_corr(X, i, j, set(cond))
                z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10)) * np.sqrt(n - 3 - l)
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                if p_val > alpha:
                    adj[i, j] = adj[j, i] = 0
                    sep_set[(i, j)] = set(cond)
                    sep_set[(j, i)] = set(cond)
                    found_indep = True
                    break

# Report skeleton
print("\n  PC Algorithm — Causal Skeleton (edges):")
for i in range(n_vars):
    for j in range(i+1, n_vars):
        if adj[i, j] > 0:
            r = pearsonr(X[:, i], X[:, j])[0]
            sep = sep_set.get((i, j), '∅')
            if isinstance(sep, set):
                sep_str = ', '.join([var_names[k] for k in sep]) if sep else '∅'
            else:
                sep_str = str(sep)
            print(f"    {var_names[i]:<20} --- {var_names[j]:<20}  "
                  f"r={r:+.4f}  sep_set={{{sep_str}}}")

# --- LiNGAM-style directional inference ---
# Direction inference assumes a linear model; the causal direction is the one with lower |corr(cause, residual)|
# For each edge, test directionality:
# If X→Y: X is independent of residual Y - b*X
# If Y→X: Y is independent of residual X - b*Y
print("\n  LiNGAM Direction Inference:")
edge_directions = []
for i in range(n_vars):
    for j in range(i+1, n_vars):
        if adj[i, j] > 0:
            # Test X→Y: residual = Y - b*X
            b_ij, _, _, _, _ = stats.linregress(X[:, i], X[:, j])
            resid_j = X[:, j] - b_ij * X[:, i]
            r_i_resid, p_i_resid = pearsonr(X[:, i], resid_j)
            
            # Test Y→X: residual = X - b*Y
            b_ji, _, _, _, _ = stats.linregress(X[:, j], X[:, i])
            resid_i = X[:, i] - b_ji * X[:, j]
            r_j_resid, p_j_resid = pearsonr(X[:, j], resid_i)
            
            # Lower correlation with residual = more likely causal direction
            if abs(r_i_resid) < abs(r_j_resid):
                direction = f"{var_names[i]} → {var_names[j]}"
                conf = abs(r_j_resid) - abs(r_i_resid)
            elif abs(r_j_resid) < abs(r_i_resid):
                direction = f"{var_names[j]} → {var_names[i]}"
                conf = abs(r_i_resid) - abs(r_j_resid)
            else:
                direction = f"{var_names[i]} ↔ {var_names[j]} (bidirectional)"
                conf = 0
            
            print(f"    {direction:<45} "
                  f"|r(X,resid_Y)|={abs(r_i_resid):.4f}  "
                  f"|r(Y,resid_X)|={abs(r_j_resid):.4f}  "
                  f"confidence={conf:.4f}")
            
            edge_directions.append({
                'var1': var_names[i], 'var2': var_names[j],
                'pearson_r': pearsonr(X[:, i], X[:, j])[0],
                'direction': direction,
                'confidence': conf,
                'resid_corr_ij': r_i_resid,
                'resid_corr_ji': r_j_resid
            })

edge_dir_df = pd.DataFrame(edge_directions)
# Write causal skeleton with inferred edge directions
edge_dir_df.to_csv(RESULTS_DIR / "ext_d1_causal_discovery.csv", index=False)

# ═══════════════════════════════════════════════════════════
# 5. CRISPR STRATIFICATION VALIDATION
# ═══════════════════════════════════════════════════════════
print(f"\n[5] CRISPR Stratification Validation...", flush=True)

# For top instrument genes that are also TFs: split cells by CRISPR dependency
# Test whether NetITH-drug sensitivity association differs by dependency group

top_tf_instruments = [g for g in top_instruments if g in top_tfs]
if not top_tf_instruments:
    top_tf_instruments = [g for g in top_instruments[:5]]

print(f"  Testing TF instruments: {top_tf_instruments}")

strat_results = []
for gene in top_tf_instruments[:6]:  # limit to top 6
    if gene not in crispr.columns:
        continue
    
    crispr_gene = crispr.loc[[n for n in common_all if n in crispr.index], gene]
    gene_vals = crispr_gene.reindex(common_all).values
    netith_vals_aligned = netith.loc[common_all, 'NetITH'].values
    
    # Split at median CRISPR dependency
    valid = ~(np.isnan(gene_vals) | np.isnan(netith_vals_aligned))
    gene_valid = gene_vals[valid]
    n_valid = netith_vals_aligned[valid]
    
    if len(gene_valid) < 100:
        continue
    
    # Stratify cell lines by median CRISPR dependency (more negative = more dependent)
    median_dep = np.median(gene_valid)
    high_dep = gene_valid < median_dep  # more negative = more dependent
    low_dep = ~high_dep
    
    for drug in sorted(ic50_mat.columns):
        ic50_col = ic50_mat.loc[common_all, drug]
        ic50_valid = ic50_col.values[valid].astype(float)
        drug_valid = ~np.isnan(ic50_valid)
        
        if drug_valid.sum() < 30:
            continue
        
        high_idx = high_dep & drug_valid
        low_idx = low_dep & drug_valid
        
        if high_idx.sum() < 20 or low_idx.sum() < 20:
            continue
        
        rho_high, _ = spearmanr(n_valid[high_idx], ic50_valid[high_idx])
        rho_low, _ = spearmanr(n_valid[low_idx], ic50_valid[low_idx])
        
        # Fisher z-transformation test (null: equal correlations in the two dependency groups)
        # Fisher Z-test for difference
        z_high = np.arctanh(rho_high)
        z_low = np.arctanh(rho_low)
        se_diff = np.sqrt(1/(high_idx.sum()-3) + 1/(low_idx.sum()-3))
        z_stat = (z_high - z_low) / se_diff if se_diff > 0 else 0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        strat_results.append({
            'gene': gene,
            'drug': drug,
            'n_high': int(high_idx.sum()),
            'n_low': int(low_idx.sum()),
            'rho_high_dependency': rho_high,
            'rho_low_dependency': rho_low,
            'delta_rho': rho_high - rho_low,
            'z_diff': z_stat,
            'p_diff': p_diff
        })

strat_df = pd.DataFrame(strat_results)
# Write dependency-stratified NetITH–IC50 correlations and interaction tests
strat_df.to_csv(RESULTS_DIR / "ext_d1_crispr_stratification.csv", index=False)

if len(strat_df) > 0:
    strat_df = strat_df.sort_values('p_diff')
    sig_strat = strat_df[strat_df['p_diff'] < 0.05]
    print(f"  Total drug-gene tests: {len(strat_df)}")
    print(f"  Significant differences (p<0.05): {len(sig_strat)}")
    print(f"  Mean Δρ (high-low dependency): {strat_df['delta_rho'].mean():.4f}")

# ═══════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Direction 1: NetITH Causal Validation")
print(f"{'='*70}")

print(f"\n  A. Causal Discovery (PC + LiNGAM):")
if len(edge_dir_df) > 0:
    netith_edges = edge_dir_df[
        (edge_dir_df['var1'] == 'NetITH') | (edge_dir_df['var2'] == 'NetITH')
    ]
    for _, row in netith_edges.iterrows():
        print(f"    {row['direction']} (confidence={row['confidence']:.4f})")

print(f"\n  B. Instrumental Variable Analysis:")
print(f"    CRISPR genes with NetITH association (p<0.01): {len(sig_genes)}")
if len(iv_df) > 0:
    print(f"    IV analyses (F>10): {len(iv_df)}")
    print(f"    Mean β_OLS (naive) = {iv_df['beta_ols'].mean():.4f}")
    print(f"    Mean β_IV  (causal) = {iv_df['beta_iv'].mean():.4f}")

print(f"\n  C. CRISPR Stratification:")
if len(strat_df) > 0:
    print(f"    Total tests: {len(strat_df)}")
    print(f"    Sig. dependency×NetITH interactions (p<0.05): {len(sig_strat) if len(sig_strat)>0 else 0}")
    mean_delta = strat_df['delta_rho'].mean()
    print(f"    Mean Δρ (high vs low dependency): {mean_delta:+.4f}")
    if mean_delta > 0:
        print(f"    → Higher CRISPR dependency STRENGTHENS NetITH-drug association")
        print(f"      (consistent with NetITH as causal mediator of dependency)")
    else:
        print(f"    → Higher CRISPR dependency WEAKENS NetITH-drug association")

print(f"\nDone. All results in {RESULTS_DIR}/")
