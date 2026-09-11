"""run_depmap_26q1_differential.py — DepMap 26Q1 CRISPR differential-dependency replication audit.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Recomputes the manuscript's CRISPR differential-dependency analysis
    (run_depmap_crispr.py section [7/7]) on the DepMap 26Q1 Chronos gene-effect
    matrix: GDSC cell lines are matched to DepMap IDs, split at the median
    NetITH, and each gene is tested with a two-sided Mann-Whitney U with BH FDR
    and a std>0.05 filter, to establish the true manuscript numbers.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - $DATA_ROOT/CRISPRGeneEffect.csv: DepMap 26Q1 Chronos gene-effect matrix
    - results/gdsc/gdsc_netith_cell_lines.csv: GDSC NetITH per cell line
    - data/gdsc/cell_annot.csv: GDSC cell-line names
    - data/external/depmap_metadata.csv: DepMap cell-line metadata (name <-> ID)

Outputs:
    - results/control/depmap_26q1_differential.csv: per-gene Cohen's d, p, BH FDR (CRISPR audit)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_depmap_26q1_differential.py
"""
from pathlib import Path
import os
import pandas as pd, numpy as np, re
from scipy.stats import mannwhitneyu
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
CRISPR = f'{DATA_ROOT}/CRISPRGeneEffect.csv'

# Normalize names for matching: uppercase, drop non-alphanumeric characters
def normalize_name(s):
    if not isinstance(s, str): return ""
    return re.sub(r'[^A-Z0-9]', '', s.upper().strip())

# Load GDSC NetITH and cell-line annotations for name-based matching
netith_df = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)
cell_annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv')
cell_annot['cell_line_name'] = cell_annot['Characteristics.cell.line.']
gdsc_names = set(cell_annot['cell_line_name'].dropna().unique())

# DepMap metadata: build normalized-name -> DepMap ID (ACH) maps
meta = pd.read_csv(f'{ROOT}/data/external/depmap_metadata.csv')
meta['name_key'] = meta['stripped_cell_line_name'].apply(normalize_name)
meta['name_key2'] = meta['cell_line_name'].apply(normalize_name)
name_to_ach = dict(zip(meta['name_key'], meta['depmap_id']))
name_to_ach2 = dict(zip(meta['name_key2'], meta['depmap_id']))
combined = {**name_to_ach, **name_to_ach2}

# Match GDSC cell lines to DepMap IDs via normalized names
gdsc_norm = {normalize_name(n): n for n in gdsc_names if isinstance(n, str)}
matches = [(orig, combined[norm]) for norm, orig in gdsc_norm.items() if norm and combined.get(norm)]
print("matched:", len(matches))

# Map GDSC NetITH onto matched DepMap IDs
netith_series = netith_df['NetITH']
netith_series.index = netith_series.index.str.strip()
netith_mapped = {}
for gdsc_name, depmap_id in matches:
    if gdsc_name in netith_series.index:
        netith_mapped[depmap_id] = netith_series[gdsc_name]
netith_sub = pd.Series(netith_mapped)
print("netith mapped:", len(netith_sub))

# Load 26Q1 Chronos gene effect; restrict to lines with NetITH
crispr = pd.read_csv(CRISPR, index_col=0)
crispr.index = crispr.index.str.upper()
crispr_sub = crispr.loc[crispr.index.intersection(netith_sub.index)]
print("crispr_sub:", crispr_sub.shape)

# Gene filter: require SD > 0.05 across lines (same rule as run_depmap_crispr.py)
gene_std = crispr_sub.std()
valid_genes = gene_std[gene_std > 0.05].index
print(f"valid genes (std>0.05): {len(valid_genes)} / {len(gene_std)}")

# Median NetITH split: high (> median) vs low (<= median) dependency groups
median_netith = netith_sub.median()
high_mask = netith_sub > median_netith
low_mask = netith_sub <= median_netith
print(f"high: {high_mask.sum()}, low: {low_mask.sum()}")

# Per gene: two-sided Mann-Whitney U (null = equal effect distributions across
# NetITH groups) plus Cohen's d on pooled SD; require >=5 lines per group
diff = []
for gene in valid_genes:
    effect = crispr_sub[gene]
    hv = effect.loc[effect.index.intersection(netith_sub[high_mask].index)].dropna()
    lv = effect.loc[effect.index.intersection(netith_sub[low_mask].index)].dropna()
    if len(hv) < 5 or len(lv) < 5: continue
    try:
        _, p = mannwhitneyu(hv, lv, alternative='two-sided')
    except Exception:
        p = 1.0
    md = hv.mean() - lv.mean()
    ps = np.sqrt((hv.std()**2 + lv.std()**2) / 2)
    d = md / ps if ps > 0 else 0
    diff.append({'gene': gene.split(' (')[0], 'cohens_d': d, 'p_value': p,
                 'n_high': len(hv), 'n_low': len(lv)})
# BH FDR correction across genes, then save the differential-dependency table
df = pd.DataFrame(diff).sort_values('p_value')
df['fdr'] = (df['p_value'] * len(df) / df['p_value'].rank(method='average')).clip(upper=1.0)
print(f"\ntested: {len(df)}; FDR<0.05: {(df['fdr']<0.05).sum()}; nominal: {(df['p_value']<0.05).sum()}")
print(df.head(8)[['gene','cohens_d','p_value','fdr']].to_string())
# Report the manuscript target gene NCKAP1L explicitly
nc = df[df['gene']=='NCKAP1L']
print("NCKAP1L:", nc[['gene','cohens_d','p_value','fdr']].to_string() if len(nc) else "ABSENT")
df.to_csv(f'{ROOT}/results/control/depmap_26q1_differential.csv', index=False)
print("saved results/control/depmap_26q1_differential.csv")
