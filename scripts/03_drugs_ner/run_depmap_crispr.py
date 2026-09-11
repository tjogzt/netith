"""run_depmap_crispr.py — DepMap CRISPR validation: JUN gene dependency vs NetITH.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc/cell_annot.csv; data/CRISPRGeneEffect.csv; data/external/depmap_metadata.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{depmap_ap1_netith.csv, depmap_genome_wide_netith.csv, depmap_differential_dependency.csv, depmap_cell_level.csv}; results/depmap/figures/depmap_crispr_validation.png
Pipeline: drug-ner stage — see repository README
"""

import numpy as np
import pandas as pd
import os, sys, warnings, gzip, re
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from scipy.linalg import eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Paths ───
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
GDSC_DIR = ROOT / "results" / "gdsc"
DEPMAP_CSV = DATA_ROOT / "CRISPRGeneEffect.csv"
META_CSV = DATA_ROOT / "external" / "depmap_metadata.csv"
OUTPUT_DIR = ROOT / "results" / "depmap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Gene of interest
JUN_GENE = "JUN (3725)"
AP1_GENES = ["JUN (3725)", "JUNB (3726)", "JUND (3727)",
             "FOS (2353)", "FOSB (2354)", "FOSL1 (8061)", "FOSL2 (2355)"]

# ─── Helper: fuzzy match cell line names ───
def normalize_name(s):
    """Normalize cell line name for matching."""
    if not isinstance(s, str):
        return ""
    s = s.upper().strip()
    # Remove common prefixes/suffixes
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

# ═══════════════════════════════════════════════════════════
# 1. Load GDSC NetITH
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("[1/7] Loading GDSC NetITH data...")
netith_df = pd.read_csv(GDSC_DIR / "gdsc_netith_cell_lines.csv", index_col=0)
print(f"  GDSC cell lines: {len(netith_df)}")

# Also load cell annotations for tissue info
cell_annot = pd.read_csv(DATA_ROOT / "gdsc" / "cell_annot.csv")
# Extract cell line name from GDSC annotations
cell_annot['cell_line_name'] = cell_annot['Characteristics.cell.line.']
gdsc_names = set(cell_annot['cell_line_name'].dropna().unique())
print(f"  GDSC annotated cell lines: {len(gdsc_names)}")

# ═══════════════════════════════════════════════════════════
# 2. Load DepMap Metadata → ACH ID → cell line name map
# ═══════════════════════════════════════════════════════════
print("\n[2/7] Loading DepMap metadata & building name map...")
meta = pd.read_csv(META_CSV)
print(f"  DepMap metadata: {len(meta)} models")

# Build lookup: stripped_cell_line_name → depmap_id
meta['name_key'] = meta['stripped_cell_line_name'].apply(normalize_name)
name_to_ach = dict(zip(meta['name_key'], meta['depmap_id']))
# Also build from cell_line_name (original)
meta['name_key2'] = meta['cell_line_name'].apply(normalize_name)
name_to_ach2 = dict(zip(meta['name_key2'], meta['depmap_id']))
# Merge both lookups
name_to_ach_combined = {**name_to_ach, **name_to_ach2}

# ═══════════════════════════════════════════════════════════
# 3. Match GDSC ↔ DepMap cell lines
# ═══════════════════════════════════════════════════════════
print("\n[3/7] Matching GDSC cell lines to DepMap ACH IDs...")

# Build normalized GDSC names
gdsc_norm = {normalize_name(n): n for n in gdsc_names if isinstance(n, str)}

matches = []
unmatched = []
for norm_name, orig_name in gdsc_norm.items():
    if not norm_name:
        continue
    ach_id = name_to_ach_combined.get(norm_name)
    if ach_id:
        matches.append((orig_name, ach_id))
    else:
        unmatched.append(orig_name)

print(f"  Direct matches: {len(matches)} / {len(gdsc_norm)}")
print(f"  Unmatched: {len(unmatched)}")

# Build mapping DataFrame
match_df = pd.DataFrame(matches, columns=['gdsc_name', 'depmap_id'])
match_df = match_df.set_index('depmap_id')

# Merge with NetITH
netith_series = netith_df['NetITH']
netith_series.index = netith_series.index.str.strip()

# Map GDSC names → NetITH via match
match_df['gdsc_name_norm'] = match_df['gdsc_name'].str.strip()
# Some GDSC names may differ slightly; try direct index match
netith_mapped = {}
for depmap_id, row in match_df.iterrows():
    gdsc_name = row['gdsc_name']
    if gdsc_name in netith_series.index:
        netith_mapped[depmap_id] = netith_series[gdsc_name]

netith_by_ach = pd.Series(netith_mapped, name='NetITH')
print(f"  Final matched (NetITH available): {len(netith_by_ach)}")

# ═══════════════════════════════════════════════════════════
# 4. Load CRISPR Gene Effect Data
# ═══════════════════════════════════════════════════════════
print("\n[4/7] Loading CRISPR gene effect data...")

# Read the CSV (it's large: ~1209 lines × 18532 columns)
crispr_raw = pd.read_csv(DEPMAP_CSV, index_col=0)
print(f"  CRISPR matrix: {crispr_raw.shape[0]} cell lines × {crispr_raw.shape[1]} genes")

# Verify JUN is present
if JUN_GENE not in crispr_raw.columns:
    available_genes = [c for c in crispr_raw.columns if 'JUN' in c.upper()]
    print(f"  WARNING: {JUN_GENE} not found. Available JUN-like: {available_genes}")
    if available_genes:
        JUN_GENE_USE = available_genes[0]
    else:
        raise ValueError("No JUN gene found in CRISPR data!")
else:
    JUN_GENE_USE = JUN_GENE

# ─── Merge CRISPR + NetITH ───
common_achs = sorted(set(crispr_raw.index) & set(netith_by_ach.index))
print(f"  Common ACH IDs (CRISPR ∩ NetITH): {len(common_achs)}")

if len(common_achs) < 30:
    print("  WARNING: Very few common cell lines. The matching may need improvement.")
    print("  Trying fuzzy matching on original cell_line_name...")
    # Build reverse: DepMap cell_line_name → ach_id
    ach_to_name = dict(zip(meta['depmap_id'], meta['cell_line_name']))
    ach_to_stripped = dict(zip(meta['depmap_id'], meta['stripped_cell_line_name']))
    
    # Try matching GDSC names against DepMap names directly
    extra_matches = {}
    for gdsc_name in gdsc_names:
        gdsc_n = normalize_name(gdsc_name)
        if not gdsc_n:
            continue
        for ach, stripped in ach_to_stripped.items():
            if normalize_name(stripped) == gdsc_n and ach not in netith_by_ach.index:
                if gdsc_name in netith_series.index:
                    extra_matches[ach] = netith_series[gdsc_name]
    
    # Add extra matches
    for ach, val in extra_matches.items():
        if ach not in netith_by_ach.index:
            netith_by_ach[ach] = val
    
    common_achs = sorted(set(crispr_raw.index) & set(netith_by_ach.index))
    print(f"  After expanded matching: {len(common_achs)} cell lines")

# Subset data
crispr_sub = crispr_raw.loc[common_achs]
netith_sub = netith_by_ach.loc[common_achs]

print(f"  Final analysis: {len(common_achs)} cell lines × {crispr_sub.shape[1]} genes")

# ═══════════════════════════════════════════════════════════
# 5. JUN Dependency vs NetITH — Primary Analysis
# ═══════════════════════════════════════════════════════════
print("\n[5/7] JUN dependency vs NetITH analysis...")

jun_effect = crispr_sub[JUN_GENE_USE]

# Spearman correlation
rho_jun, p_jun = spearmanr(jun_effect, netith_sub)
print(f"  JUN CERES × NetITH: ρ = {rho_jun:.4f}, p = {p_jun:.4g}")

# Pearson for comparison
r_pearson = np.corrcoef(jun_effect, netith_sub)[0,1]
print(f"  JUN CERES × NetITH (Pearson): r = {r_pearson:.4f}")

# AP-1 family analysis
ap1_results = []
ap1_present = [g for g in AP1_GENES if g in crispr_sub.columns]
print(f"\n  AP-1 family genes available: {len(ap1_present)}/{len(AP1_GENES)}")

for gene in ap1_present:
    effect = crispr_sub[gene]
    rho, p = spearmanr(effect, netith_sub)
    ap1_results.append({
        'gene': gene.split(' (')[0],
        'rho': rho,
        'p_value': p,
        'n_cells': len(effect.dropna())
    })

ap1_df = pd.DataFrame(ap1_results).sort_values('rho')
print("\n  AP-1 Family Dependency × NetITH:")
for _, row in ap1_df.iterrows():
    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"    {row['gene']:10s}: ρ={row['rho']:+.4f}, p={row['p_value']:.4g} {sig}")

# ═══════════════════════════════════════════════════════════
# 6. Genome-wide CRISPR Scan
# ═══════════════════════════════════════════════════════════
print("\n[6/7] Genome-wide CRISPR × NetITH scan...")

# Remove genes with too many zeros (non-expressed)
gene_var = crispr_sub.var()
gene_std = crispr_sub.std()
valid_genes = gene_std[gene_std > 0.05].index  # filter near-constant genes
print(f"  Genes with std > 0.05: {len(valid_genes)} / {len(gene_std)}")

gw_results = []
for gene in valid_genes:
    effect = crispr_sub[gene].dropna()
    if len(effect) < 20:
        continue
    # Align with netith
    common_idx = effect.index.intersection(netith_sub.index)
    if len(common_idx) < 20:
        continue
    rho, p = spearmanr(effect.loc[common_idx], netith_sub.loc[common_idx])
    gene_name = gene.split(' (')[0]
    gw_results.append({
        'gene': gene_name,
        'entrez': gene.split('(')[1].rstrip(')') if '(' in gene else '',
        'rho': rho,
        'p_value': p,
        'n_cells': len(common_idx)
    })

gw_df = pd.DataFrame(gw_results)
# Benjamini-Hochberg FDR
gw_df = gw_df.sort_values('p_value')
gw_df['fdr'] = gw_df['p_value'] * len(gw_df) / gw_df['p_value'].rank(method='average')
gw_df['fdr'] = gw_df['fdr'].clip(upper=1.0)
gw_df['neg_log10_p'] = -np.log10(gw_df['p_value'].clip(lower=1e-300))

n_sig = (gw_df['fdr'] < 0.05).sum()
n_nominal = (gw_df['p_value'] < 0.05).sum()
print(f"  FDR < 0.05: {n_sig} genes")
print(f"  Nominal p < 0.05: {n_nominal} genes")

# Top hits
print("\n  Top 20 genes (by |ρ|):")
top20 = gw_df.nlargest(20, 'rho').head(10)
bottom20 = gw_df.nsmallest(20, 'rho').head(10)
for _, row in pd.concat([bottom20, top20]).iterrows():
    sig = "***" if row['fdr'] < 0.001 else ("**" if row['fdr'] < 0.01 else ("*" if row['fdr'] < 0.05 else ""))
    print(f"    {row['gene']:12s}: ρ={row['rho']:+.4f}, p={row['p_value']:.4g}, FDR={row['fdr']:.4g} {sig}")

# ═══════════════════════════════════════════════════════════
# 7. High vs Low NetITH — Differential Dependency
# ═══════════════════════════════════════════════════════════
print("\n[7/7] High vs Low NetITH differential dependency...")

# Split into high/low NetITH (median)
median_netith = netith_sub.median()
high_mask = netith_sub > median_netith
low_mask = netith_sub <= median_netith

n_high = high_mask.sum()
n_low = low_mask.sum()
print(f"  High NetITH: {n_high}, Low NetITH: {n_low}")

# Differential analysis
diff_results = []
for gene in valid_genes:
    effect = crispr_sub[gene]
    high_vals = effect.loc[effect.index.intersection(netith_sub[high_mask].index)].dropna()
    low_vals = effect.loc[effect.index.intersection(netith_sub[low_mask].index)].dropna()
    if len(high_vals) < 5 or len(low_vals) < 5:
        continue
    
    mean_diff = high_vals.mean() - low_vals.mean()
    # Mann-Whitney U test
    try:
        stat, p = mannwhitneyu(high_vals, low_vals, alternative='two-sided')
    except:
        p = 1.0
    
    # Cohen's d
    pooled_std = np.sqrt((high_vals.std()**2 + low_vals.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    gene_name = gene.split(' (')[0]
    diff_results.append({
        'gene': gene_name,
        'mean_high': high_vals.mean(),
        'mean_low': low_vals.mean(),
        'delta': mean_diff,
        'cohens_d': cohens_d,
        'p_value': p,
        'n_high': len(high_vals),
        'n_low': len(low_vals)
    })

diff_df = pd.DataFrame(diff_results)
diff_df = diff_df.sort_values('p_value')
diff_df['fdr'] = diff_df['p_value'] * len(diff_df) / diff_df['p_value'].rank(method='average')
diff_df['fdr'] = diff_df['fdr'].clip(upper=1.0)
diff_df['neg_log10_p'] = -np.log10(diff_df['p_value'].clip(lower=1e-300))

n_sig_diff = (diff_df['fdr'] < 0.05).sum()
print(f"  FDR < 0.05: {n_sig_diff} genes")
print(f"  Nominal p < 0.05: {(diff_df['p_value']<0.05).sum()} genes")

# JUN in diff analysis
jun_diff = diff_df[diff_df['gene'] == 'JUN']
if len(jun_diff) > 0:
    jd = jun_diff.iloc[0]
    print(f"\n  JUN: Δ = {jd['delta']:+.4f} (high vs low NetITH), d = {jd['cohens_d']:+.4f}, p = {jd['p_value']:.4g}")

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Saving results...")

# AP-1 results
ap1_df.to_csv(OUTPUT_DIR / "depmap_ap1_netith.csv", index=False)

# Genome-wide results
gw_df.to_csv(OUTPUT_DIR / "depmap_genome_wide_netith.csv", index=False)

# Differential results
diff_df.to_csv(OUTPUT_DIR / "depmap_differential_dependency.csv", index=False)

# Cell-level data
cell_data = pd.DataFrame({
    'depmap_id': common_achs,
    'NetITH': netith_sub.values,
    'JUN_CERES': jun_effect.loc[common_achs].values
}, index=common_achs)
# Add cell line names
cell_data['cell_line_name'] = cell_data.index.map(
    lambda x: meta.set_index('depmap_id').loc[x, 'cell_line_name'] if x in meta['depmap_id'].values else x
)
cell_data.to_csv(OUTPUT_DIR / "depmap_cell_level.csv", index=False)

# ═══════════════════════════════════════════════════════════
# VISUALIZATION — 6-panel figure
# ═══════════════════════════════════════════════════════════
print("Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('DepMap CRISPR Validation: Gene Dependency vs NetITH', fontsize=14, fontweight='bold', y=0.98)

# Panel A: JUN CERES vs NetITH scatter
ax = axes[0, 0]
ax.scatter(netith_sub, jun_effect, c='steelblue', alpha=0.5, edgecolors='none', s=30)
# Add regression line
z = np.polyfit(netith_sub, jun_effect, 1)
p_line = np.poly1d(z)
x_range = np.linspace(netith_sub.min(), netith_sub.max(), 100)
ax.plot(x_range, p_line(x_range), 'r-', linewidth=2)
ax.set_xlabel('NetITH', fontsize=11)
ax.set_ylabel('JUN CERES Score', fontsize=11)
ax.set_title(f'JUN Dependency vs NetITH\nρ = {rho_jun:.3f}, p = {p_jun:.2g}', fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Panel B: AP-1 Family barplot
ax = axes[0, 1]
colors = ['#d62728' if r < 0 else '#2ca02c' for r in ap1_df['rho']]
bars = ax.barh(ap1_df['gene'], ap1_df['rho'], color=colors, edgecolor='gray', linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Spearman ρ (vs NetITH)', fontsize=11)
ax.set_title('AP-1 Family Dependency × NetITH', fontsize=10)
# Add significance stars
for i, (_, row) in enumerate(ap1_df.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    if sig:
        x_pos = row['rho'] + (0.02 if row['rho'] >= 0 else -0.06)
        ax.text(x_pos, i, sig, va='center', fontsize=9, fontweight='bold')

# Panel C: JUN boxplot by NetITH tertile
ax = axes[0, 2]
tertile_labels = ['Low NetITH', 'Mid NetITH', 'High NetITH']
tertile_data = []
tertile_bounds = np.percentile(netith_sub, [0, 33.33, 66.67, 100])
for i in range(3):
    mask = (netith_sub >= tertile_bounds[i]) & (netith_sub < tertile_bounds[i+1])
    if i == 2:
        mask = (netith_sub >= tertile_bounds[i])
    tertile_data.append(jun_effect[mask].dropna().values)

bp = ax.boxplot(tertile_data, labels=tertile_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('JUN CERES Score', fontsize=11)
ax.set_title('JUN Dependency by NetITH Tertile', fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Kruskal-Wallis test
from scipy.stats import kruskal
h_stat, p_kw = kruskal(*tertile_data)
ax.text(0.5, 0.95, f'Kruskal H={h_stat:.2f}, p={p_kw:.2g}',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Panel D: Volcano plot (genome-wide)
ax = axes[1, 0]
ax.scatter(gw_df['rho'], gw_df['neg_log10_p'], c='lightgray', alpha=0.4, s=8, edgecolors='none')
# Highlight significant
sig_mask = gw_df['fdr'] < 0.05
ax.scatter(gw_df.loc[sig_mask, 'rho'], gw_df.loc[sig_mask, 'neg_log10_p'],
           c='#d62728', alpha=0.7, s=15, edgecolors='none')
# Label top genes
top_genes = gw_df.nlargest(5, 'neg_log10_p')
for _, row in top_genes.iterrows():
    ax.annotate(row['gene'], (row['rho'], row['neg_log10_p']),
                fontsize=7, alpha=0.8, ha='center', va='bottom')
# Also label top by |rho|
top_rho = pd.concat([gw_df.nlargest(3, 'rho'), gw_df.nsmallest(3, 'rho')])
for _, row in top_rho.iterrows():
    if row['gene'] not in top_genes['gene'].values:
        ax.annotate(row['gene'], (row['rho'], row['neg_log10_p']),
                    fontsize=7, alpha=0.8, ha='center', va='bottom')

ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xlabel('Spearman ρ (vs NetITH)', fontsize=11)
ax.set_ylabel('-log₁₀(p)', fontsize=11)
ax.set_title(f'Genome-Wide CRISPR × NetITH\n{n_sig} genes FDR<0.05', fontsize=10)

# Panel E: Differential volcano (High vs Low NetITH)
ax = axes[1, 1]
ax.scatter(diff_df['cohens_d'], diff_df['neg_log10_p'], c='lightgray', alpha=0.4, s=8, edgecolors='none')
sig_mask = diff_df['fdr'] < 0.05
ax.scatter(diff_df.loc[sig_mask, 'cohens_d'], diff_df.loc[sig_mask, 'neg_log10_p'],
           c='#ff7f0e', alpha=0.7, s=15, edgecolors='none')
# Label top
top_diff = diff_df.nlargest(8, 'neg_log10_p')
for _, row in top_diff.iterrows():
    ax.annotate(row['gene'], (row['cohens_d'], row['neg_log10_p']),
                fontsize=7, alpha=0.8, ha='center', va='bottom')

ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
ax.set_xlabel("Cohen's d (High − Low NetITH)", fontsize=11)
ax.set_ylabel('-log₁₀(p)', fontsize=11)
ax.set_title(f'Differential Dependency\n{n_sig_diff} genes FDR<0.05', fontsize=10)

# Panel F: Summary of key genes
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
DepMap CRISPR x NetITH validation results

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Matched cell lines: {len(common_achs)} (GDSC-DepMap intersection)

JUN CERES × NetITH:
  ρ = {rho_jun:.4f} (p = {p_jun:.2g})

Genome-wide scan ({len(gw_df)} genes):
  FDR<0.05: {n_sig} genes
  Nominal p<0.05: {n_nominal} genes

Differential dependency (High vs Low NetITH):
  FDR<0.05: {n_sig_diff} genes

AP-1 family:
  JUN:  ρ = {ap1_df[ap1_df['gene']=='JUN']['rho'].values[0] if 'JUN' in ap1_df['gene'].values else 'N/A':+.4f}
"""
if 'FOS' in ap1_df['gene'].values:
    summary_text += f"  FOS:  ρ = {ap1_df[ap1_df['gene']=='FOS']['rho'].values[0]:+.4f}\n"
if 'JUNB' in ap1_df['gene'].values:
    summary_text += f"  JUNB: ρ = {ap1_df[ap1_df['gene']=='JUNB']['rho'].values[0]:+.4f}\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = OUTPUT_DIR / "figures" / "depmap_crispr_validation.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DEPMAP CRISPR VALIDATION — COMPLETE")
print("=" * 70)
print(f"""
Results:
  Cell lines matched:    {len(common_achs)}
  JUN × NetITH:          ρ={rho_jun:.4f}, p={p_jun:.2g}
  GW significant genes:  {n_sig} (FDR<0.05)
  Diff dep significant:  {n_sig_diff} (FDR<0.05)

Output files:
  {OUTPUT_DIR}/depmap_cell_level.csv
  {OUTPUT_DIR}/depmap_ap1_netith.csv
  {OUTPUT_DIR}/depmap_genome_wide_netith.csv
  {OUTPUT_DIR}/depmap_differential_dependency.csv
  {OUTPUT_DIR}/figures/depmap_crispr_validation.png
""")
