#!/usr/bin/env python3
"""
run_crossplatform.py — Compare NetITH and its determinants across GDSC, DepMap/CCLE, and TCGA platforms.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/gdsc/cell_annot.csv: cell-line tissue annotation (cancer-type mapping)
    - results/gdsc/gdsc_netith_cell_lines.csv: precomputed GDSC NetITH per cell line
    - results/depmap/depmap_cell_level.csv: precomputed DepMap/CCLE NetITH per cell line
    - results/depmap/tf_lasso_coefficients.csv: GDSC LASSO TF-importance coefficients
    - results/tcga/tcga_netith.csv: precomputed TCGA NetITH per tumor sample
    - results/tcga/module_conservation_tf_coefs.csv: TCGA module-conservation TF coefficients
    - <DATA_ROOT>/depmap/Model.csv: DepMap OncotreeLineage mapping, optional
Outputs :
    - results/depmap/crossplatform_distribution.csv, crossplatform_tf_concordance.csv, crossplatform_cancer_profile.csv
    - results/depmap/figures/crossplatform_{density,tf_scatter,cancer_heatmap}.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_DATA = DATA_ROOT / "gdsc"
DEPMAP_OUTPUT = 'results/depmap'
TCGA_OUTPUT = 'results/tcga'

# Create output directories (results/depmap and its figures subfolder) up front
os.makedirs(f'{DEPMAP_OUTPUT}/figures', exist_ok=True)


def load_gdsc_data():
    """Load GDSC NetITH + cancer type info."""
    print("[W1a] Loading GDSC...")
    # Load precomputed GDSC NetITH (per cell line) from the GDSC entropy pipeline
    netith = pd.read_csv(f'results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)
    
    # Get cancer type from cell annotation
    # Read the cell-line annotation to map each line to a tissue/cancer type
    cell_annot = pd.read_csv(f'{GDSC_DATA}/cell_annot.csv', index_col=0)
    
    # Find tissue column
    tissue_map = {}
    if 'Characteristics.cell.line.' in cell_annot.columns:
        for idx, row in cell_annot.iterrows():
            tissue_map[row.name] = str(row['Characteristics.cell.line.'])
    
    # Attach cancer type + platform label so all three platforms share one schema
    netith['cancer_type'] = netith.index.map(lambda x: tissue_map.get(x, 'Unknown'))
    netith['platform'] = 'GDSC'
    print(f"  {len(netith)} cell lines, NetITH range [{netith['NetITH'].min():.2f}, {netith['NetITH'].max():.2f}]")
    return netith


def load_depmap_data():
    """Load DepMap NetITH."""
    print("[W1b] Loading DepMap/CCLE...")
    # Load precomputed DepMap/CCLE NetITH (per cell line)
    netith = pd.read_csv(f'{DEPMAP_OUTPUT}/depmap_cell_level.csv')
    
    # DepMap has cell_line_name, map to cancer type using CCLE annotation if available
    netith['platform'] = 'DepMap'
    netith['cancer_type'] = 'Unknown'
    
    # Try to get lineage from DepMap sample info
    # Best-effort lineage mapping: join DepMap Model.csv (OncotreeLineage) by ModelID
    sample_info_path = f'{DATA_ROOT}/depmap/Model.csv'
    try:
        model_info = pd.read_csv(sample_info_path)
        model_info['ModelID'] = model_info['ModelID'].str.strip()
        lineage_map = dict(zip(model_info['ModelID'], model_info.get('OncotreeLineage', 'Unknown')))
        netith['cancer_type'] = netith['depmap_id'].map(lineage_map).fillna('Unknown')
    except Exception:
        pass
    
    print(f"  {len(netith)} cell lines, NetITH range [{netith['NetITH'].min():.2f}, {netith['NetITH'].max():.2f}]")
    return netith


def load_tcga_data():
    """Load TCGA NetITH."""
    print("[W1c] Loading TCGA...")
    # Load precomputed TCGA NetITH (per tumor) and unify the NetITH column name
    netith = pd.read_csv(f'{TCGA_OUTPUT}/tcga_netith.csv', index_col=0)
    netith = netith.rename(columns={'netith_bulk': 'NetITH'})
    
    # Get cancer type from TCGA barcode (e.g., TCGA-BRCA-...)
    # Derive cancer type from the TCGA barcode project code (e.g., TCGA-BRCA-* -> BRCA)
    netith['cancer_type'] = netith['sample'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) and len(x.split('-')) > 1 else 'Unknown')
    netith['platform'] = 'TCGA'
    
    print(f"  {len(netith)} samples, NetITH range [{netith['NetITH'].min():.2f}, {netith['NetITH'].max():.2f}]")
    return netith


def compare_distributions(df_gdsc, df_depmap, df_tcga):
    """W1: Compare NetITH distributions across platforms."""
    print("\n[W1] Distribution comparison...")
    
    # Combine
    # Stack the three platforms into one long table for distribution statistics
    cols = ['NetITH', 'platform', 'cancer_type']
    combined = pd.concat([
        df_gdsc[cols],
        df_depmap[cols],
        df_tcga[cols]
    ], ignore_index=True)
    
    # Stats per platform
    # Per-platform summary statistics (n, mean, median, std, min, max, IQR)
    stats = combined.groupby('platform').agg(
        n=('NetITH', 'count'),
        mean=('NetITH', 'mean'),
        median=('NetITH', 'median'),
        std=('NetITH', 'std'),
        min=('NetITH', 'min'),
        max=('NetITH', 'max'),
        iqr=('NetITH', lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25)),
    ).reset_index()
    
    # Write W1 output: per-platform NetITH distribution summary
    stats.to_csv(f'{DEPMAP_OUTPUT}/crossplatform_distribution.csv', index=False)
    
    # Between-platform differences
    print(f"\n  Platform comparison:")
    for _, r in stats.iterrows():
        print(f"    {r['platform']:<8} n={int(r['n']):<6} "
              f"μ={r['mean']:.2f} σ={r['std']:.2f} "
              f"range=[{r['min']:.2f}, {r['max']:.2f}]")
    
    # Test GDSC vs TCGA (Mann-Whitney)
    np.random.seed(42)
    gdsc_vals = df_gdsc['NetITH'].dropna().values
    tcga_vals = df_tcga['NetITH'].dropna().values
    # Two-sided Mann-Whitney U test of GDSC vs TCGA NetITH (null: distributions are equal)
    stat, p = mannwhitneyu(gdsc_vals, tcga_vals)
    
    print(f"\n  GDSC vs TCGA NetITH: p={p:.2e} (Mann-Whitney, sampled)")
    
    return combined, stats


def compare_tf_importance():
    """W2: Compare TF importance rankings across GDSC and TCGA."""
    print("\n[W2] TF importance concordance...")
    
    # GDSC LASSO
    # GDSC TF importance: rank TFs by absolute LASSO coefficient
    gdsc_lasso = pd.read_csv(f'{DEPMAP_OUTPUT}/tf_lasso_coefficients.csv')
    gdsc_lasso = gdsc_lasso.set_index('TF')
    gdsc_lasso['rank_gdsc'] = gdsc_lasso['abs_coef'].rank(ascending=False)
    
    # TCGA module conservation
    # TCGA TF importance: aggregate per-TF mean |coefficient| across cancers
    tcga_mod = pd.read_csv(f'{TCGA_OUTPUT}/module_conservation_tf_coefs.csv')
    # Aggregate TCGA coefficients across cancers (mean absolute coef)
    tcga_tf = tcga_mod.groupby('tf').agg(
        mean_abs_coef=('coef', lambda x: np.abs(x).mean()),
        n_sig=('coef', lambda x: (np.abs(x) > 0.01).sum()),
        n_cancers=('cancer', 'nunique'),
    ).reset_index()
    tcga_tf['rank_tcga'] = tcga_tf['mean_abs_coef'].rank(ascending=False)
    tcga_tf = tcga_tf.rename(columns={'tf': 'TF'}).set_index('TF')
    
    # Merge
    # Inner-join the two TF rankings on shared TFs
    merged = gdsc_lasso[['lasso_coef', 'abs_coef', 'rank_gdsc']].join(
        tcga_tf[['mean_abs_coef', 'n_sig', 'n_cancers', 'rank_tcga']],
        how='inner')
    
    if len(merged) >= 5:
        # Rank concordance tests between platforms (null: no monotonic / linear association)
        rho, p = spearmanr(merged['abs_coef'], merged['mean_abs_coef'])
        r_pearson, p_pearson = pearsonr(merged['abs_coef'], merged['mean_abs_coef'])
        print(f"  TF importance concordance (GDSC vs TCGA):")
        print(f"    Spearman ρ = {rho:+.3f} (p={p:.4f})")
        print(f"    Pearson r = {r_pearson:+.3f} (p={p_pearson:.4f})")
        print(f"    N = {len(merged)} shared TFs")
    
    # Write W2 output: per-TF cross-platform concordance table
    merged.to_csv(f'{DEPMAP_OUTPUT}/crossplatform_tf_concordance.csv')
    
    # Top TFs by platform
    print(f"\n  Top 10 GDSC TFs:      Top 10 TCGA TFs:")
    gdsc_top = gdsc_lasso.nlargest(10, 'abs_coef').index.tolist()
    tcga_top = tcga_tf.nlargest(10, 'mean_abs_coef').index.tolist()
    for i in range(10):
        g = gdsc_top[i] if i < len(gdsc_top) else ''
        t = tcga_top[i] if i < len(tcga_top) else ''
        in_both = '*' if g in tcga_top or t in gdsc_top else ''
        print(f"    {g:<16} {t:<16} {in_both}")
    
    shared_top = set(gdsc_top[:10]) & set(tcga_top[:10])
    print(f"  Shared top-10: {len(shared_top)} TFs: {shared_top}")
    
    return merged, gdsc_lasso, tcga_tf


def compare_cancer_profiles(df_gdsc, df_tcga):
    """W3: Compare NetITH profiles across cancer types."""
    print("\n[W3] Cancer-type NetITH profiles...")
    
    # Map GDSC cancer types to TCGA codes
    # GDSC uses cell line tissue names, TCGA uses project codes
    # Build a rough mapping
    # Map free-text GDSC tissue labels to TCGA project codes (keyword match)
    tissue_to_tcga = {
        'lung': 'LUAD', 'breast': 'BRCA', 'colon': 'COAD',
        'ovary': 'OV', 'skin': 'SKCM', 'pancreas': 'PAAD',
        'prostate': 'PRAD', 'stomach': 'STAD', 'liver': 'LIHC',
        'kidney': 'KIRC', 'brain': 'GBM', 'thyroid': 'THCA',
        'endometrium': 'UCEC', 'bladder': 'BLCA', 'oesophagus': 'ESCA',
        'head and neck': 'HNSC', 'bone': 'SARC', 'blood': 'LAML',
        'haematopoietic': 'LAML', 'lymphoid': 'DLBC',
    }
    
    # GDSC cancer-type NetITH
    def map_gdsc_tissue(ct):
        ct_lower = str(ct).lower()
        for tissue, tcga in tissue_to_tcga.items():
            if tissue in ct_lower:
                return tcga
        return None
    
    df_gdsc_mapped = df_gdsc.copy()
    df_gdsc_mapped['tcga_type'] = df_gdsc_mapped['cancer_type'].apply(map_gdsc_tissue)
    
    # Per-cancer-type NetITH means within GDSC (after tissue->TCGA mapping)
    gdsc_ct = df_gdsc_mapped.dropna(subset=['tcga_type']).groupby('tcga_type').agg(
        gdsc_mean=('NetITH', 'mean'),
        gdsc_std=('NetITH', 'std'),
        gdsc_n=('NetITH', 'count'),
    )
    
    # TCGA cancer-type NetITH
    # Per-cancer-type NetITH means within TCGA
    tcga_ct = df_tcga.groupby('cancer_type').agg(
        tcga_mean=('NetITH', 'mean'),
        tcga_std=('NetITH', 'std'),
        tcga_n=('NetITH', 'count'),
    )
    
    # Merge on common cancer types
    # Keep only cancer types measured in both platforms
    common = gdsc_ct.join(tcga_ct, how='inner')
    
    if len(common) >= 5:
        # Correlation of cancer-type means
        # Spearman correlation of cancer-type NetITH means (null: no monotonic concordance)
        rho, p = spearmanr(common['gdsc_mean'], common['tcga_mean'])
        print(f"  Cancer-type NetITH concordance (GDSC vs TCGA):")
        print(f"    Spearman ρ = {rho:+.3f} (p={p:.4f})")
        print(f"    N = {len(common)} shared cancer types")
    
    # Write W3 output: per-cancer-type cross-platform table
    common.to_csv(f'{DEPMAP_OUTPUT}/crossplatform_cancer_profile.csv')
    
    print(f"\n  Cancer-type NetITH (shared types):")
    for ct, row in common.sort_values('gdsc_mean').iterrows():
        print(f"    {ct:<8} GDSC={row['gdsc_mean']:.2f} (n={int(row['gdsc_n']):<4})  "
              f"TCGA={row['tcga_mean']:.2f} (n={int(row['tcga_n']):<6})")
    
    return common


def plot_density_comparison(combined, output_dir):
    """Density plot of NetITH across platforms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: density
    # W1 figure, left panel: overlaid density histograms of NetITH per platform
    ax = axes[0]
    colors = {'GDSC': '#d62728', 'DepMap': '#1f77b4', 'TCGA': '#2ca02c'}
    for platform in ['GDSC', 'DepMap', 'TCGA']:
        vals = combined[combined['platform'] == platform]['NetITH'].dropna()
        ax.hist(vals, bins=50, alpha=0.4, color=colors[platform], 
                label=f'{platform} (n={len(vals)})', density=True)
    
    ax.set_xlabel('NetITH', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('NetITH Distribution Across Platforms', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    
    # Right: boxplot
    # W1 figure, right panel: per-platform boxplots of NetITH
    ax = axes[1]
    platforms = ['GDSC', 'DepMap', 'TCGA']
    data = [combined[combined['platform'] == p]['NetITH'].dropna().values for p in platforms]
    bp = ax.boxplot(data, labels=platforms, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[p] for p in platforms]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_ylabel('NetITH', fontsize=12)
    ax.set_title('NetITH by Platform', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    # Save W1 figure (density + boxplot)
    fig.savefig(f'{output_dir}/figures/crossplatform_density.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved crossplatform_density.png")


def plot_tf_scatter(merged, output_dir):
    """Scatter plot: GDSC TF importance vs TCGA TF importance."""
    if len(merged) < 5:
        print("  Not enough TFs for scatter")
        return
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # W2 figure: scatter of GDSC vs TCGA TF importance with top-TF labels
    ax.scatter(merged['abs_coef'], merged['mean_abs_coef'],
               s=60, c='#1f77b4', alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # Label top TFs
    top_tfs = set(merged.nlargest(8, 'abs_coef').index) | \
              set(merged.nlargest(8, 'mean_abs_coef').index)
    for tf in top_tfs:
        if tf in merged.index:
            ax.annotate(tf, (merged.loc[tf, 'abs_coef'], merged.loc[tf, 'mean_abs_coef']),
                        fontsize=8, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')
    
    # Add correlation line
    if len(merged) >= 5:
        rho, p = spearmanr(merged['abs_coef'], merged['mean_abs_coef'])
        ax.text(0.05, 0.95, f'Spearman ρ = {rho:+.3f}\np = {p:.4f}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('GDSC |LASSO Coefficient|', fontsize=12)
    ax.set_ylabel('TCGA Mean |Module Coefficient|', fontsize=12)
    ax.set_title('Cross-Platform TF Importance Concordance', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    # Save W2 figure
    fig.savefig(f'{output_dir}/figures/crossplatform_tf_scatter.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved crossplatform_tf_scatter.png")


def plot_cancer_heatmap(common, output_dir):
    """Heatmap: cancer-type NetITH across platforms."""
    if len(common) < 3:
        print("  Not enough cancer types for heatmap")
        return
    
    # Normalize per platform
    # W3 figure: cancer-type NetITH heatmap; Z-score each column so scales are comparable
    mat = common[['gdsc_mean', 'tcga_mean']].copy()
    mat = mat.sort_values('gdsc_mean')
    
    fig, ax = plt.subplots(figsize=(8, max(5, len(mat) * 0.35)))
    
    # Z-score within each column for visualization
    mat_z = (mat - mat.mean()) / (mat.std() + 1e-10)
    
    im = ax.imshow(mat_z.values, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['GDSC', 'TCGA'], fontsize=11)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title('Cancer-Type NetITH Profile\n(Cross-Platform Comparison, Z-scored)',
                 fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(mat)):
        for j, col in enumerate(['gdsc_mean', 'tcga_mean']):
            ax.text(j, i, f'{mat.iloc[i][col]:.2f}',
                    ha='center', va='center', fontsize=7)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Z-score')
    plt.tight_layout()
    # Save W3 figure
    fig.savefig(f'{output_dir}/figures/crossplatform_cancer_heatmap.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved crossplatform_cancer_heatmap.png")


def main():
    print("=" * 60)
    print("W: Cross-Platform NetITH Determinants")
    print("=" * 60)
    
    # Load all three platforms
    df_gdsc = load_gdsc_data()
    df_depmap = load_depmap_data()
    df_tcga = load_tcga_data()
    
    # W1: Distribution comparison
    combined, stats = compare_distributions(df_gdsc, df_depmap, df_tcga)
    
    # W2: TF importance concordance
    merged, gdsc_lasso, tcga_tf = compare_tf_importance()
    
    # W3: Cancer-type profiles
    common = compare_cancer_profiles(df_gdsc, df_tcga)
    
    # W4: Integrated summary
    print("\n[W4] Cross-Platform Summary...")
    print(f"  GDSC cell lines: {len(df_gdsc)}, TCGA tumors: {len(df_tcga)}, DepMap: {len(df_depmap)}")
    print(f"  GDSC NetITH scale: {df_gdsc['NetITH'].mean():.2f} ± {df_gdsc['NetITH'].std():.2f}")
    print(f"  TCGA NetITH scale: {df_tcga['NetITH'].mean():.2f} ± {df_tcga['NetITH'].std():.2f}")
    
    # Visualization
    # Render all three cross-platform figures
    print("\n[Viz] Generating figures...")
    plot_density_comparison(combined, DEPMAP_OUTPUT)
    plot_tf_scatter(merged, DEPMAP_OUTPUT)
    plot_cancer_heatmap(common, DEPMAP_OUTPUT)
    
    print("\nDone! Outputs in results/depmap/")


if __name__ == '__main__':
    main()
