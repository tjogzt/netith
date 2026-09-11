"""
run_ith_benchmark.py — traditional ITH benchmark: NetITH vs expression Shannon entropy / variance.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Pipeline: control / test stage — see repository README

Traditional ITH Benchmark: NetITH vs Expression Shannon Entropy vs Expression Variance

Part A (GDSC): Drug sensitivity prediction comparison (286 drugs × 1013 cell lines)
Part B (TCGA): Survival prognosis comparison (33 cancer types × 8384 samples)

Metrics compared:
  1. NetITH (von Neumann entropy of CollecTRI GRN Laplacian) — our method
  2. Expression Shannon Entropy — gene-wise expression binning + Shannon H
  3. Expression Variance — mean variance of top 5000 variable genes
  4. Expression MAD — median absolute deviation (robust variance)

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - {ROOT}/results/gdsc/gdsc_netith_cell_lines.csv
    - {ROOT}/results/tcga/tcga_netith.csv
    - {ROOT}/results/tcga/tcga_tri_modal_merged.csv
Outputs:
    - results/gdsc/gdsc_ith_benchmark_summary.csv (5 ITH metrics x 286 drugs)
Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_ith_benchmark.py
"""

import numpy as np
import pandas as pd
import os, sys, pickle, warnings, gzip
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from scipy.linalg import eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

DATA_DIR = Path(f"{ROOT}/data/gdsc")
GDSC_DIR = Path(f"{DATA_ROOT}/gdsc_download")
TCGA_DIR = Path(f"{DATA_ROOT}/xena/tcgapancan")
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
BENCH_DIR = OUTPUT_DIR / "benchmark"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
(BENCH_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
N_TOP_GENES = 5000  # For variance-based metrics
N_SHANNON_BINS = 20  # Bins for Shannon entropy of expression

print("=" * 70)
print("  Traditional ITH Benchmark: NetITH vs Classic Metrics")
print("=" * 70)

# ═══════════════════════════════════════════════════════════
# PART A: GDSC Drug Sensitivity Benchmark
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("Part A: GDSC Drug Sensitivity Benchmark")
print("─" * 50)

# ─── A1. Load GDSC Expression ──────────────────────────────
print("[A1] Loading GDSC expression...")

expr_raw = pd.read_csv(DATA_DIR / "rna_expr.csv", index_col=0)
print(f"  Raw expression: {expr_raw.shape}")

# CEL → cell line mapping
annot = pd.read_csv(DATA_DIR / "cell_annot.csv", index_col=0)
cell_line_map = {}
for cel, row in annot.iterrows():
    cl = str(row.get('Characteristics.cell.line.', ''))
    if cl and cl != 'nan' and cl != 'NA':
        cell_line_map[cel] = cl

common_cels = [c for c in expr_raw.columns if c in cell_line_map]
expr_named = expr_raw[common_cels].copy()
expr_named.columns = [cell_line_map[c] for c in common_cels]

dup_cols = expr_named.columns.duplicated()
if dup_cols.any():
    expr_named = expr_named.loc[:, ~dup_cols]

# ENSG → Symbol mapping
gene_map = pd.read_csv(DATA_DIR / "ensg_symbol_map.csv")
ensg_to_sym = dict(zip(gene_map['ensg'], gene_map['symbol']))

expr_named.index = expr_named.index.astype(str)
matched = expr_named.index.isin(ensg_to_sym.keys())
expr_sym = expr_named.loc[matched].copy()
expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]

print(f"  Final: {expr_sym.shape}")

# ─── A2. Compute Traditional ITH Metrics ───────────────────
print("[A2] Computing traditional ITH metrics...")

expr_mat = expr_sym.values.T  # cells × genes
n_cells = expr_mat.shape[0]
n_genes = expr_mat.shape[1]
cell_names_gdsc = expr_sym.columns.tolist()

results_a = pd.DataFrame({'cell_line': cell_names_gdsc})

# --- Shannon Entropy of Expression ---
# For each cell, bin the expression values into N_SHANNON_BINS, compute H
print("  Computing expression Shannon entropy...")
shannon_vals = np.full(n_cells, np.nan)
for i in range(n_cells):
    vals = expr_mat[i, :]
    # Remove NaN
    vals = vals[~np.isnan(vals)]
    if len(vals) < 100:
        continue
    # Z-score
    vals_z = (vals - np.mean(vals)) / (np.std(vals) + 1e-10)
    # Bin into equal-width bins
    hist, _ = np.histogram(vals_z, bins=N_SHANNON_BINS, density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    shannon_vals[i] = -np.sum(hist * np.log2(hist))

results_a['shannon_entropy'] = shannon_vals

# --- Expression Variance (mean of top N_TOP_GENES) ---
print("  Computing expression variance...")
gene_vars = np.var(expr_mat, axis=0)
top_idx = np.argsort(gene_vars)[-N_TOP_GENES:]
var_vals = np.mean(expr_mat[:, top_idx], axis=1)  # mean expression
var_var_vals = np.var(expr_mat[:, top_idx], axis=1).mean()  # not per-cell

# Actually, per-cell metric: for each cell, take the variance of its top-N genes' expression
# Let's use per-gene variance as selection, then per-cell mean of top genes
# Better: per-cell, select the genes with highest variance across all cells, then compute variance
# of those genes within each cell

# Let's do: variance of expression within each cell, across all genes
per_cell_var = np.var(expr_mat, axis=1)  # variance of all genes per cell
results_a['expression_variance'] = per_cell_var

# --- Expression MAD (median absolute deviation, more robust) ---
print("  Computing expression MAD...")
per_cell_med = np.median(expr_mat, axis=1)
per_cell_mad = np.median(np.abs(expr_mat - per_cell_med[:, np.newaxis]), axis=1)
results_a['expression_mad'] = per_cell_mad

# --- CV (coefficient of variation) ---
per_cell_mean = np.mean(expr_mat, axis=1)
per_cell_std = np.std(expr_mat, axis=1)
per_cell_cv = per_cell_std / (np.abs(per_cell_mean) + 1e-10)
results_a['expression_cv'] = per_cell_cv

# --- Gene-wise Shannon (genes as "states", each gene's expression → probability) ---
# For each cell, normalize expression to sum=1, treat as probability distribution
print("  Computing expression-based probability entropy...")
prob_vals = np.full(n_cells, np.nan)
for i in range(n_cells):
    vals = expr_mat[i, :].copy()
    vals = vals[~np.isnan(vals)]
    vals = vals - vals.min() + 1e-10  # shift positive
    vals = vals / vals.sum()
    vals = vals[vals > 0]
    prob_vals[i] = -np.sum(vals * np.log2(vals))
results_a['prob_entropy'] = prob_vals

results_a.set_index('cell_line', inplace=True)
results_a = results_a.dropna()
print(f"  Cell lines with all metrics: {len(results_a)}")

# ─── A3. Load NetITH (pre-computed) ────────────────────────
print("[A3] Loading NetITH (pre-computed)...")
netith_gdsc = pd.read_csv(
    f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv",
    index_col=0
)
print(f"  NetITH cell lines: {len(netith_gdsc)}")

# Merge
common_cl = sorted(set(results_a.index) & set(netith_gdsc.index))
results_a = results_a.loc[common_cl].copy()
results_a['NetITH'] = netith_gdsc.loc[common_cl, 'NetITH']
print(f"  Common cell lines: {len(results_a)}")

# ─── A4. Load IC50 & Correlate ─────────────────────────────
print("[A4] Loading IC50 data...")
ic50 = pd.read_csv(GDSC_DIR / "GDSC2_IC50_all.csv")
ic50_mat = ic50.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)
print(f"  IC50 matrix: {ic50_mat.shape}")

# Find common cell lines
common_drug = sorted(set(results_a.index) & set(ic50_mat.index))
print(f"  Common cell lines (drug): {len(common_drug)}")

results_sub = results_a.loc[common_drug]
ic50_sub = ic50_mat.loc[common_drug]

# Filter drugs with ≥30 cell lines
min_cells = 30
drug_counts = ic50_sub.notna().sum()
valid_drugs = drug_counts[drug_counts >= min_cells].index.tolist()
print(f"  Drugs with ≥{min_cells} cell lines: {len(valid_drugs)}")

# Correlate each metric with each drug
metrics = ['shannon_entropy', 'expression_variance', 'expression_mad',
           'expression_cv', 'prob_entropy', 'NetITH']
metric_labels = {
    'shannon_entropy': 'Expression Shannon Entropy',
    'expression_variance': 'Expression Variance',
    'expression_mad': 'Expression MAD',
    'expression_cv': 'Expression CV',
    'prob_entropy': 'Expression Prob Entropy',
    'NetITH': 'NetITH (GRN von Neumann)',
}

# Drug → pathway mapping (for later)
drug_pathways = {}
if 'PUTATIVE_TARGET' in ic50.columns:
    for drug in valid_drugs:
        targets = ic50[ic50['DRUG_NAME'] == drug]['PUTATIVE_TARGET'].dropna().unique()
        if len(targets) > 0:
            drug_pathways[drug] = targets[0]

corr_rows = []
for drug in valid_drugs:
    drug_ic50 = ic50_sub[drug].dropna()
    common = sorted(set(results_sub.index) & set(drug_ic50.index))
    if len(common) < min_cells:
        continue
    
    for metric in metrics:
        metric_vals = results_sub.loc[common, metric]
        ic50_vals = drug_ic50[common]
        
        valid_mask = metric_vals.notna() & ic50_vals.notna()
        if valid_mask.sum() < min_cells:
            continue
        
        rho, p = spearmanr(metric_vals[valid_mask], ic50_vals[valid_mask])
        corr_rows.append({
            'drug': drug, 'metric': metric, 'n': valid_mask.sum(),
            'spearman_r': rho, 'spearman_p': p
        })

corr_df = pd.DataFrame(corr_rows)
print(f"  Total correlations: {len(corr_df)}")

# FDR correction per metric
from statsmodels.stats.multitest import multipletests
for metric in metrics:
    mask = corr_df['metric'] == metric
    if mask.sum() > 0:
        _, fdr, _, _ = multipletests(corr_df.loc[mask, 'spearman_p'].values, method='fdr_bh')
        corr_df.loc[mask, 'fdr'] = fdr

corr_df.to_csv(BENCH_DIR / "gdsc_ith_benchmark_correlations.csv", index=False)

# ─── A5. Summary Statistics ────────────────────────────────
print("\n[A5] GDSC Benchmark Summary:")
print("-" * 50)

summary_a = []
for metric in metrics:
    mask = corr_df['metric'] == metric
    n_drugs = mask.sum()
    n_fdr05 = (corr_df.loc[mask, 'fdr'] < 0.05).sum()
    mean_abs_r = corr_df.loc[mask, 'spearman_r'].abs().mean()
    mean_r = corr_df.loc[mask, 'spearman_r'].mean()
    n_positive = (corr_df.loc[mask, 'spearman_r'] > 0).sum()
    
    summary_a.append({
        'metric': metric,
        'label': metric_labels.get(metric, metric),
        'n_drugs': n_drugs,
        'n_fdr05': n_fdr05,
        'pct_fdr05': n_fdr05 / n_drugs * 100 if n_drugs > 0 else 0,
        'mean_abs_r': mean_abs_r,
        'mean_r': mean_r,
        'n_positive': n_positive,
        'pct_positive': n_positive / n_drugs * 100 if n_drugs > 0 else 0,
    })

summary_a_df = pd.DataFrame(summary_a)
summary_a_df.to_csv(BENCH_DIR / "gdsc_ith_benchmark_summary.csv", index=False)

for _, row in summary_a_df.iterrows():
    print(f"  {row['label']:30s}: "
          f"FDR<0.05={row['n_fdr05']:3d}/{row['n_drugs']:3d} ({row['pct_fdr05']:4.1f}%), "
          f"mean|r|={row['mean_abs_r']:.4f}, "
          f"positive={row['pct_positive']:.1f}%")

# ─── A6. Partial Correlation (NetITH vs Expression Variance + MAD + Shannon) ─
print("\n[A6] Partial correlations (NetITH controlling for traditional metrics)...")
from scipy.stats import linregress

# Helper: compute partial Spearman (control for one covariate)
def partial_spearman(x, y, z):
    """Returns (rho_partial, p_partial): Spearman correlation of x-resid vs y-resid, both regressed on z"""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_v, y_v, z_v = x[valid], y[valid], z[valid]
    if len(x_v) < 30:
        return np.nan, np.nan
    # x ~ z
    s_xz, i_xz, _, _, _ = linregress(z_v, x_v)
    x_resid = x_v - (s_xz * z_v + i_xz)
    # y ~ z
    s_yz, i_yz, _, _, _ = linregress(z_v, y_v)
    y_resid = y_v - (s_yz * z_v + i_yz)
    return spearmanr(x_resid, y_resid)

partial_rows = []
for drug in valid_drugs:
    drug_ic50 = ic50_sub[drug].dropna()
    common = sorted(set(results_sub.index) & set(drug_ic50.index))
    if len(common) < min_cells:
        continue
    
    netith_v = results_sub.loc[common, 'NetITH'].values
    ic50_v = drug_ic50[common].values
    ev_v = results_sub.loc[common, 'expression_variance'].values
    mad_v = results_sub.loc[common, 'expression_mad'].values
    shannon_v = results_sub.loc[common, 'shannon_entropy'].values
    
    mask = ~(np.isnan(netith_v) | np.isnan(ic50_v))
    if mask.sum() < min_cells:
        continue
    
    rho_n_raw, p_n_raw = spearmanr(netith_v[mask], ic50_v[mask])
    rho_n_ev, p_n_ev = partial_spearman(netith_v, ic50_v, ev_v)
    rho_n_mad, p_n_mad = partial_spearman(netith_v, ic50_v, mad_v)
    rho_n_shannon, p_n_shannon = partial_spearman(netith_v, ic50_v, shannon_v)
    
    partial_rows.append({
        'drug': drug, 'n': mask.sum(),
        'rho_netith_raw': rho_n_raw, 'p_netith_raw': p_n_raw,
        'rho_netith_ctrl_ev': rho_n_ev, 'p_netith_ctrl_ev': p_n_ev,
        'rho_netith_ctrl_mad': rho_n_mad, 'p_netith_ctrl_mad': p_n_mad,
        'rho_netith_ctrl_shannon': rho_n_shannon, 'p_netith_ctrl_shannon': p_n_shannon,
    })

partial_df = pd.DataFrame(partial_rows)
# FDR
for pcol in ['p_netith_raw', 'p_netith_ctrl_ev', 'p_netith_ctrl_mad', 'p_netith_ctrl_shannon']:
    mask = partial_df[pcol].notna()
    if mask.sum() > 0:
        _, fdr, _, _ = multipletests(partial_df.loc[mask, pcol].values, method='fdr_bh')
        partial_df.loc[mask, pcol.replace('p_', 'fdr_')] = fdr

partial_df.to_csv(BENCH_DIR / "gdsc_ith_partial_correlations.csv", index=False)

n_raw = (partial_df['fdr_netith_raw'] < 0.05).sum() if 'fdr_netith_raw' in partial_df.columns else 0
n_ev = (partial_df['fdr_netith_ctrl_ev'] < 0.05).sum() if 'fdr_netith_ctrl_ev' in partial_df.columns else 0
n_mad = (partial_df['fdr_netith_ctrl_mad'] < 0.05).sum() if 'fdr_netith_ctrl_mad' in partial_df.columns else 0
n_shannon = (partial_df['fdr_netith_ctrl_shannon'] < 0.05).sum() if 'fdr_netith_ctrl_shannon' in partial_df.columns else 0

print(f"  NetITH raw significant:              {n_raw}/{len(partial_df)}")
print(f"  NetITH partial (ctrl ExprVar):       {n_ev}/{len(partial_df)}")
print(f"  NetITH partial (ctrl MAD):           {n_mad}/{len(partial_df)}")
print(f"  NetITH partial (ctrl Shannon):       {n_shannon}/{len(partial_df)}  ← independent of ALL traditional metrics")


# ═══════════════════════════════════════════════════════════
# PART B: TCGA Survival Benchmark
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("Part B: TCGA Survival Benchmark")
print("─" * 50)

# ─── B1. Load TCGA Expression (stream Xena file) ───────────
print("[B1] Loading TCGA expression (streaming Xena EB++ file)...")

xena_file = TCGA_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"

# Read header to get sample list and gene list
with gzip.open(xena_file, 'rt') as f:
    header = f.readline().strip().split('\t')
    
print(f"  Columns in file: {len(header)}")
samples_tcga = header[1:]  # first col is 'sample'
print(f"  TCGA samples: {len(samples_tcga)}")

# Map TCGA sample IDs (TCGA-XX-XXXX-XX) to aliquot barcodes (TCGA.XX.XXXX.XX)
# Xena uses TCGA-XX-XXXX-XX format, our NetITH data uses TCGA.XX.XXXX.XX format
def to_aliquot(s):
    parts = s.split('-')
    if len(parts) >= 3:
        return '.'.join(parts[:3]) + '.' + parts[3][:2]
    return s

sample_aliquot_map = {s: to_aliquot(s) for s in samples_tcga}

# ─── B2. Load TCGA NetITH ──────────────────────────────────
print("[B2] Loading TCGA NetITH...")
tcga_netith = pd.read_csv(
    f"{ROOT}/results/tcga/tcga_netith.csv",
    index_col=0
)
tcga_tri = pd.read_csv(
    f"{ROOT}/results/tcga/tcga_tri_modal_merged.csv"
)

# Helper: convert Xena hyphen format (TCGA-OR-A5J1-01) to dot format (TCGA.OR.A5J1.01)
def hyphens_to_dots(s):
    parts = s.split('-')
    if len(parts) >= 4:
        return '.'.join(parts[:3]) + '.' + parts[3][:2]
    return s

# Get NetITH map — convert to dot format for consistency with tri_modal
netith_map = {}
for _, row in tcga_netith.iterrows():
    sid = str(row['sample'])
    netith_map[hyphens_to_dots(sid)] = row['netith_bulk']

# Get survival map from tri_modal
surv_map = {}
for _, row in tcga_tri.iterrows():
    sid = row['sample']
    surv_map[sid] = {
        'OS': row.get('OS', np.nan),
        'OS.time': row.get('OS.time', np.nan),
        'cancer': row.get('cancer', 'UNKNOWN'),
    }

print(f"  NetITH samples: {len(netith_map)}")
print(f"  Survival samples: {len(surv_map)}")

# ─── B3. Stream Xena & Compute Traditional Metrics ─────────
print("[B3] Computing TCGA traditional ITH metrics (streaming)...")

# We'll compute metrics for samples that have NetITH and survival
target_aliquots = set(netith_map.keys()) & set(surv_map.keys())
print(f"  Target samples (with NetITH + survival): {len(target_aliquots)}")

# Build reverse map: aliquot → xena sample ID
aliquot_to_xena = {}
for xena_id, aliquot in sample_aliquot_map.items():
    aliquot_to_xena[aliquot] = xena_id

target_xena = [aliquot_to_xena[a] for a in target_aliquots if a in aliquot_to_xena]
print(f"  Matched Xena sample IDs: {len(target_xena)}")

# Build: xena_id → column index
xena_set = set(target_xena)
xena_col_idx = {}
with gzip.open(xena_file, 'rt') as f:
    header = f.readline().strip().split('\t')
    for i, col in enumerate(header):
        if col in xena_set:
            xena_col_idx[col] = i

print(f"  Columns to extract: {len(xena_col_idx)}")

# Stream and build expression matrix directly (genes × target samples)
# First pass: count genes and build gene list
gene_symbols_tcga = []
with gzip.open(xena_file, 'rt') as f:
    f.readline()  # skip header
    for line in f:
        gene = line.split('\t', 1)[0]
        if '|' in gene:
            gene = gene.split('|')[0]
        gene_symbols_tcga.append(gene)

n_genes_tcga = len(gene_symbols_tcga)
n_target = len(target_xena)
print(f"  Genes: {n_genes_tcga}, Target samples: {n_target}")

# Build column index array for fast lookup
col_indices = np.array([xena_col_idx.get(xid, -1) for xid in target_xena], dtype=np.int32)
valid_col_mask = col_indices >= 0

# Second pass: fill matrix
expr_matrix = np.empty((n_genes_tcga, n_target), dtype=np.float32)
expr_matrix.fill(np.nan)

with gzip.open(xena_file, 'rt') as f:
    f.readline()  # skip header
    for line_no, line in enumerate(f):
        if line_no % 5000 == 0 and line_no > 0:
            print(f"    Gene {line_no}/{n_genes_tcga}...")
        
        parts = line.strip().split('\t')
        
        row = np.full(n_target, np.nan, dtype=np.float32)
        for j in range(n_target):
            ci = col_indices[j]
            if ci >= 0 and ci < len(parts):
                try:
                    row[j] = float(parts[ci])
                except (ValueError, IndexError):
                    pass
        expr_matrix[line_no, :] = row

print(f"  Expression matrix: {expr_matrix.shape}")

# Free memory
del col_indices, valid_col_mask

# ─── B4. Compute Traditional Metrics for TCGA ──────────────
print("[B4] Computing TCGA traditional ITH metrics...")

# Map back: Xena ID → aliquot (dot format) for matched samples
ordered_aliquots = []
valid_col_indices = []
for j, xid in enumerate(target_xena):
    # Convert Xena hyphen format to dot format for matching
    al = sample_aliquot_map.get(xid, None)
    if al and al in target_aliquots:
        ordered_aliquots.append(al)
        valid_col_indices.append(j)

print(f"  Final ordered samples: {len(ordered_aliquots)}")

# Extract relevant columns and transpose to samples × genes
expr_mat_tcga = expr_matrix[:, valid_col_indices].T  # samples × genes
print(f"  TCGA expression: {expr_mat_tcga.shape}")

n_tcga = expr_mat_tcga.shape[0]

# Clean up large matrix
del expr_matrix

# Compute metrics per sample
tcga_results = pd.DataFrame({'aliquot': ordered_aliquots})

# Shannon entropy
tcga_shannon = np.full(n_tcga, np.nan)
for i in range(n_tcga):
    vals = expr_mat_tcga[i, :]
    vals = vals[~np.isnan(vals)]
    if len(vals) < 100:
        continue
    vals_z = (vals - np.mean(vals)) / (np.std(vals) + 1e-10)
    hist, _ = np.histogram(vals_z, bins=N_SHANNON_BINS, density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    tcga_shannon[i] = -np.sum(hist * np.log2(hist))
tcga_results['shannon_entropy'] = tcga_shannon

# Expression variance
tcga_var = np.var(expr_mat_tcga, axis=1)
tcga_results['expression_variance'] = tcga_var

# Expression MAD
tcga_med = np.median(expr_mat_tcga, axis=1)
tcga_mad = np.median(np.abs(expr_mat_tcga - tcga_med[:, np.newaxis]), axis=1)
tcga_results['expression_mad'] = tcga_mad

# Prob entropy
tcga_prob = np.full(n_tcga, np.nan)
for i in range(n_tcga):
    vals = expr_mat_tcga[i, :].copy()
    vals = vals[~np.isnan(vals)]
    vals = vals - vals.min() + 1e-10
    vals = vals / vals.sum()
    vals = vals[vals > 0]
    tcga_prob[i] = -np.sum(vals * np.log2(vals))
tcga_results['prob_entropy'] = tcga_prob

# CV (coefficient of variation)
per_cell_mean_tcga = np.mean(expr_mat_tcga, axis=1)
per_cell_std_tcga = np.std(expr_mat_tcga, axis=1)
tcga_cv = per_cell_std_tcga / (np.abs(per_cell_mean_tcga) + 1e-10)
tcga_results['expression_cv'] = tcga_cv

# Add NetITH and survival
tcga_netith_vals = []
tcga_os = []; tcga_os_time = []; tcga_cancer = []
for a in ordered_aliquots:
    tcga_netith_vals.append(netith_map.get(a, np.nan))
    surv = surv_map.get(a, {})
    tcga_os.append(surv.get('OS', np.nan))
    tcga_os_time.append(surv.get('OS.time', np.nan))
    tcga_cancer.append(surv.get('cancer', 'UNKNOWN'))

tcga_results['NetITH'] = tcga_netith_vals
tcga_results['OS'] = tcga_os
tcga_results['OS.time'] = tcga_os_time
tcga_results['cancer'] = tcga_cancer

tcga_results = tcga_results.dropna(subset=['NetITH', 'OS', 'OS.time'])
tcga_results = tcga_results[tcga_results['OS.time'] > 0]
print(f"  TCGA samples with all data: {len(tcga_results)}")

tcga_results.to_csv(BENCH_DIR / "tcga_ith_benchmark_samples.csv", index=False)

# ─── B5. Per-Cancer Survival Analysis ──────────────────────
print("[B5] TCGA per-cancer survival benchmark...")

from lifelines import CoxPHFitter

cancer_stats = tcga_results['cancer'].value_counts()
valid_cancers = cancer_stats[cancer_stats >= 30].index.tolist()
print(f"  Cancer types with ≥30 samples: {len(valid_cancers)}")

surv_rows = []
for cancer in valid_cancers:
    sub = tcga_results[tcga_results['cancer'] == cancer].copy()
    n = len(sub)
    n_events = int(sub['OS'].sum())
    
    for metric in ['shannon_entropy', 'expression_variance', 'expression_mad', 'expression_cv', 'prob_entropy', 'NetITH']:
        df = sub[['OS.time', 'OS', metric]].dropna().copy()
        df = df[df['OS.time'] > 0]
        if len(df) < 30 or df['OS'].sum() < 5:
            continue
        
        # Z-score metric
        df[metric] = (df[metric] - df[metric].mean()) / (df[metric].std(ddof=0) + 1e-10)
        
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(
                df.rename(columns={'OS.time': 'duration', 'OS': 'event'}),
                duration_col='duration', event_col='event',
                formula=metric
            )
            hr = np.exp(cph.params_[metric])
            p = cph.summary.loc[metric, 'p']
            concordance = cph.concordance_index_
        except Exception as e:
            hr, p, concordance = np.nan, np.nan, np.nan
        
        surv_rows.append({
            'cancer': cancer, 'n': n, 'n_events': n_events,
            'metric': metric,
            'hr': hr, 'p': p, 'c_index': concordance,
        })

surv_df = pd.DataFrame(surv_rows)
surv_df.to_csv(BENCH_DIR / "tcga_ith_benchmark_survival.csv", index=False)

# ─── B6. TCGA Summary ──────────────────────────────────────
print("\n[B6] TCGA Benchmark Summary:")
print("-" * 60)

# Per-metric summary
for metric in metrics:
    mask = surv_df['metric'] == metric
    n_cancers = mask.sum()
    n_sig = (surv_df.loc[mask, 'p'] < 0.05).sum()
    mean_hr = surv_df.loc[mask, 'hr'].mean()
    n_protective = (surv_df.loc[mask, 'hr'] < 1).sum()
    mean_cindex = surv_df.loc[mask, 'c_index'].mean()
    
    print(f"  {metric_labels.get(metric, metric):30s}: "
          f"sig={n_sig:2d}/{n_cancers:2d}, "
          f"mean HR={mean_hr:.3f}, "
          f"protective={n_protective}/{n_cancers}, "
          f"mean C-index={mean_cindex:.3f}")

# Pan-cancer
print("\n  Pan-Cancer Cox (all cancers pooled, stratified):")
pan_data = tcga_results.dropna(subset=['OS.time', 'OS', 'NetITH', 'shannon_entropy', 'expression_variance']).copy()
pan_data = pan_data[pan_data['OS.time'] > 0]

for metric in ['NetITH', 'shannon_entropy', 'expression_variance']:
    df = pan_data[['OS.time', 'OS', metric, 'cancer']].dropna().copy()
    df[metric] = (df[metric] - df[metric].mean()) / (df[metric].std(ddof=0) + 1e-10)
    
    try:
        # Stratified Cox by cancer type
        cph = CoxPHFitter(penalizer=0.1)
        fit_df = df.rename(columns={'OS.time': 'duration', 'OS': 'event'})
        cph.fit(
            fit_df,
            duration_col='duration', event_col='event',
            formula=metric, strata='cancer'
        )
        hr = np.exp(cph.params_[metric])
        p = cph.summary.loc[metric, 'p']
        c_index = cph.concordance_index_
        print(f"  {metric_labels.get(metric, metric):30s}: HR={hr:.4f}, p={p:.4f}, C={c_index:.3f}, n={len(df)}")
    except Exception as e:
        print(f"  {metric_labels.get(metric, metric):30s}: ERROR - {e}")


# ═══════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("Generating figures...")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# --- Fig 1: GDSC Bar Chart (FDR<0.05 %) ---
ax = axes[0, 0]
labels = [metric_labels[m] for m in metrics]
pct_fdr = [summary_a_df[summary_a_df['metric'] == m]['pct_fdr05'].values[0] for m in metrics]
colors = ['#2196F3' if m != 'NetITH' else '#F44336' for m in metrics]
bars = ax.bar(range(len(labels)), pct_fdr, color=colors, edgecolor='white', linewidth=0.5)
for i, (bar, v) in enumerate(zip(bars, pct_fdr)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('% Drugs FDR < 0.05', fontsize=12)
ax.set_title('GDSC Drug Sensitivity: Significant Drugs', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(pct_fdr) * 1.2)
ax.grid(axis='y', alpha=0.3)

# --- Fig 2: GDSC Mean |r| ---
ax = axes[0, 1]
mean_abs_r = [summary_a_df[summary_a_df['metric'] == m]['mean_abs_r'].values[0] for m in metrics]
bars = ax.bar(range(len(labels)), mean_abs_r, color=colors, edgecolor='white', linewidth=0.5)
for i, (bar, v) in enumerate(zip(bars, mean_abs_r)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{v:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Mean |Spearman ρ|', fontsize=12)
ax.set_title('GDSC: Mean Effect Size', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# --- Fig 3: GDSC Partial Correlation (NetITH vs MAD, best traditional metric) ---
ax = axes[0, 2]
if len(partial_df) > 0:
    # Use MAD as reference since it's the best traditional metric
    mad_corr = []
    netith_corr = []
    for drug in partial_df['drug']:
        m = corr_df[(corr_df['drug'] == drug) & (corr_df['metric'] == 'expression_mad')]
        n = corr_df[(corr_df['drug'] == drug) & (corr_df['metric'] == 'NetITH')]
        if len(m) > 0 and len(n) > 0:
            mad_corr.append(m['spearman_r'].values[0])
            netith_corr.append(n['spearman_r'].values[0])
    
    if len(mad_corr) > 0:
        ax.scatter(mad_corr, netith_corr, c='steelblue', alpha=0.5, s=20)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(0, color='grey', linewidth=0.5)
        # Quadrant labels
        top_right = sum(1 for m, n in zip(mad_corr, netith_corr) if m > 0 and n > 0)
        top_left = sum(1 for m, n in zip(mad_corr, netith_corr) if m < 0 and n > 0)
        ax.set_xlabel(f'Expression MAD ρ with IC50', fontsize=11)
        ax.set_ylabel('NetITH ρ with IC50', fontsize=11)
        ax.set_title(f'GDSC: NetITH vs MAD per Drug\nNetITH+ only: {top_right+top_left}/{len(mad_corr)}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

# --- Fig 4: TCGA Per-Cancer HR Forest ---
ax = axes[1, 0]
# Pick top cancers by sample size
top_cancers = cancer_stats.head(10).index.tolist()
plot_data = surv_df[surv_df['cancer'].isin(top_cancers)]

netith_hr = []
shannon_hr = []
cancer_labels = []
for cancer in top_cancers:
    n_row = plot_data[(plot_data['cancer'] == cancer) & (plot_data['metric'] == 'NetITH')]
    s_row = plot_data[(plot_data['cancer'] == cancer) & (plot_data['metric'] == 'shannon_entropy')]
    if len(n_row) > 0 and len(s_row) > 0:
        netith_hr.append(n_row['hr'].values[0])
        shannon_hr.append(s_row['hr'].values[0])
        cancer_labels.append(f"{cancer}\n(n={n_row['n'].values[0]})")

if len(cancer_labels) > 0:
    y_pos = np.arange(len(cancer_labels))
    ax.scatter(netith_hr, y_pos, color='#F44336', s=80, zorder=5, label='NetITH', edgecolors='white', linewidth=0.5)
    ax.scatter(shannon_hr, y_pos, color='#2196F3', s=80, zorder=5, label='Shannon Entropy', edgecolors='white', linewidth=0.5)
    ax.axvline(1.0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cancer_labels, fontsize=9)
    ax.set_xlabel('Cox Hazard Ratio', fontsize=11)
    ax.set_title('TCGA Per-Cancer HR: NetITH vs Shannon', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

# --- Fig 5: TCGA Mean C-index Comparison ---
ax = axes[1, 1]
cindex_means = []
cindex_stds = []
for metric in metrics:
    mask = surv_df['metric'] == metric
    vals = surv_df.loc[mask, 'c_index'].dropna()
    cindex_means.append(vals.mean())
    cindex_stds.append(vals.std())

bars = ax.bar(range(len(labels)), cindex_means, color=colors, edgecolor='white', linewidth=0.5,
              yerr=cindex_stds, capsize=4)
for i, (bar, v) in enumerate(zip(bars, cindex_means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Mean C-index', fontsize=12)
ax.set_title('TCGA: Mean Prognostic C-index', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# --- Fig 6: Scatter NetITH vs Expression Variance (TCGA) ---
ax = axes[1, 2]
# Drop NaN for scatter
scatter_mask = tcga_results['expression_variance'].notna() & tcga_results['NetITH'].notna()
ax.scatter(tcga_results.loc[scatter_mask, 'expression_variance'],
           tcga_results.loc[scatter_mask, 'NetITH'],
           c='steelblue', alpha=0.15, s=2)
rho_tcga, p_tcga = spearmanr(
    tcga_results.loc[scatter_mask, 'expression_variance'],
    tcga_results.loc[scatter_mask, 'NetITH']
)
ax.set_xlabel('Expression Variance', fontsize=11)
ax.set_ylabel('NetITH', fontsize=11)
ax.set_title(f'TCGA: NetITH vs Expression Variance\nρ={rho_tcga:.4f}, p={p_tcga:.2e}', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

fig.suptitle('Traditional ITH Benchmark: NetITH vs Classic Metrics',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(BENCH_DIR / "figures" / "ith_benchmark_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figure saved.")

# ─── Additional Plot: Per-drug scatter comparing NetITH vs best traditional ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Find the best traditional metric by mean |r|
best_trad_idx = np.argmax(mean_abs_r[:-1])  # exclude NetITH
best_trad_metric = metrics[best_trad_idx]
best_trad_label = metric_labels[best_trad_metric]

# Pivot correlations by drug
drug_pivot = corr_df.pivot_table(
    index='drug', columns='metric', values='spearman_r', aggfunc='first'
).dropna()

if len(drug_pivot) > 0:
    ax = axes[0]
    ax.scatter(drug_pivot[best_trad_metric], drug_pivot['NetITH'],
               c='steelblue', alpha=0.6, s=15)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_xlabel(f'{best_trad_label} ρ with IC50', fontsize=11)
    ax.set_ylabel('NetITH ρ with IC50', fontsize=11)
    rho_drug, p_drug = spearmanr(drug_pivot[best_trad_metric], drug_pivot['NetITH'])
    ax.set_title(f'GDSC Per-Drug: NetITH vs {best_trad_label}\nρ={rho_drug:.3f}, p={p_drug:.2e}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1]
    # Distribution of delta |r|
    drug_pivot['delta_abs_r'] = drug_pivot['NetITH'].abs() - drug_pivot[best_trad_metric].abs()
    ax.hist(drug_pivot['delta_abs_r'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.set_xlabel(f'|ρ_NetITH| - |ρ_{best_trad_label.split()[0]}|', fontsize=11)
    ax.set_ylabel('Number of Drugs', fontsize=11)
    better = (drug_pivot['delta_abs_r'] > 0).sum()
    worse = (drug_pivot['delta_abs_r'] < 0).sum()
    ax.set_title(f'NetITH > {best_trad_label.split()[0]}: {better} drugs\n{best_trad_label.split()[0]} > NetITH: {worse} drugs', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(BENCH_DIR / "figures" / "ith_benchmark_drug_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Drug scatter figure saved.")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  FINAL BENCHMARK SUMMARY")
print("=" * 70)

print("\n── GDSC Drug Sensitivity ──")
print(f"  Best traditional metric: {best_trad_label}")
print(f"  NetITH FDR<0.05: {summary_a_df[summary_a_df['metric']=='NetITH']['pct_fdr05'].values[0]:.1f}%")
print(f"  {best_trad_label} FDR<0.05: {summary_a_df[summary_a_df['metric']==best_trad_metric]['pct_fdr05'].values[0]:.1f}%")
print(f"  NetITH independent signal (partial, controlling ExprVar):  {n_ev}/{len(partial_df)} drugs")
print(f"  NetITH independent signal (partial, controlling MAD):      {n_mad}/{len(partial_df)} drugs")
print(f"  NetITH independent signal (partial, controlling Shannon):  {n_shannon}/{len(partial_df)} drugs")

print("\n── TCGA Survival ──")
# Compare NetITH vs Shannon for pan-cancer
print("  (see per-cancer details above)")

print(f"\n  All results saved to: {BENCH_DIR}")
print("  Done!")
