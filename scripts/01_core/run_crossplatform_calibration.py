#!/usr/bin/env python3
"""
run_crossplatform_calibration.py — Develop and validate algorithms that map NetITH between GDSC/DepMap and TCGA scales.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - results/gdsc/gdsc_netith_cell_lines.csv: precomputed GDSC NetITH per cell line
    - results/depmap/depmap_cell_level.csv: precomputed DepMap/CCLE NetITH per cell line
    - results/tcga/tcga_netith.csv: precomputed TCGA NetITH (calibration reference)
    - data/gdsc/cell_annot.csv: cell-line tissue annotation (cancer-type mapping)
    - <DATA_ROOT>/depmap/Model.csv: DepMap OncotreeLineage mapping, optional
Outputs :
    - results/depmap/crossplatform_calibrated_netith.csv, crossplatform_calibration_stats.csv
    - results/depmap/figures/calibration_{before_after,concordance}.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr, pearsonr
from scipy.interpolate import interp1d
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_OUTPUT = 'results/gdsc'
TCGA_OUTPUT = 'results/tcga'
DEPMAP_OUTPUT = 'results/depmap'
OUTPUT_DIR = DEPMAP_OUTPUT
# Outputs land in results/depmap; create the figures subfolder up front
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_all_platforms():
    """Load NetITH from all three platforms."""
    print("[Load] Loading NetITH from all platforms...")
    
    # Load each platform's NetITH and tag it with a platform label for downstream pooling
    # GDSC
    gdsc = pd.read_csv(f'{GDSC_OUTPUT}/gdsc_netith_cell_lines.csv', index_col=0)
    gdsc = gdsc.rename(columns={'NetITH': 'netith'})
    gdsc['platform'] = 'GDSC'
    
    # DepMap
    depmap = pd.read_csv(f'{DEPMAP_OUTPUT}/depmap_cell_level.csv')
    depmap = depmap.rename(columns={'NetITH': 'netith'})
    depmap['platform'] = 'DepMap'
    
    # TCGA
    tcga = pd.read_csv(f'{TCGA_OUTPUT}/tcga_netith.csv')
    tcga = tcga.rename(columns={'netith_bulk': 'netith'})
    tcga['platform'] = 'TCGA'
    
    # Cancer type mapping for TCGA (from barcode)
    tcga['cancer_type'] = tcga['sample'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) and len(x.split('-')) > 1 else 'Unknown')
    
    # Cancer type for GDSC
    cell_annot_path = DATA_ROOT / "gdsc" / "cell_annot.csv"
    try:
        cell_annot = pd.read_csv(cell_annot_path, index_col=0)
        if 'Characteristics.cell.line.' in cell_annot.columns:
            tissue_map = {}
            for idx, row in cell_annot.iterrows():
                tissue_map[row.name] = str(row['Characteristics.cell.line.'])
            gdsc['cancer_type'] = gdsc.index.map(lambda x: tissue_map.get(x, 'Unknown'))
        else:
            gdsc['cancer_type'] = 'Unknown'
    except Exception:
        gdsc['cancer_type'] = 'Unknown'
    
    # DepMap lineage
    try:
        model_info = pd.read_csv(f'{DATA_ROOT}/depmap/Model.csv')
        lineage_map = dict(zip(model_info['ModelID'].str.strip(),
                               model_info.get('OncotreeLineage', 'Unknown')))
        depmap['cancer_type'] = depmap['depmap_id'].map(lineage_map).fillna('Unknown')
    except Exception:
        depmap['cancer_type'] = 'Unknown'
    
    print(f"  GDSC: {len(gdsc)} lines, NetITH [{gdsc['netith'].min():.2f}, {gdsc['netith'].max():.2f}]")
    print(f"  DepMap: {len(depmap)} lines, NetITH [{depmap['netith'].min():.2f}, {depmap['netith'].max():.2f}]")
    print(f"  TCGA: {len(tcga)} samples, NetITH [{tcga['netith'].min():.2f}, {tcga['netith'].max():.2f}]")
    
    return gdsc, depmap, tcga


# ============================================================
# 2. QUANTILE NORMALIZATION
# ============================================================

def quantile_normalize(source_values: np.ndarray,
                        reference_values: np.ndarray) -> np.ndarray:
    """Map source distribution to reference via quantile-quantile mapping.
    
    For each value x in source:
        q = rank(x) / N_source  (empirical CDF)
        x' = reference_quantile(q)
    
    Uses linear interpolation for values between reference quantiles.
    """
    # Sort both
    # Quantile-quantile map: each source value -> its empirical rank -> reference quantile (linear interpolation)
    source_sorted = np.sort(source_values)
    ref_sorted = np.sort(reference_values)
    
    # Build interpolation: quantile → reference value
    n_ref = len(ref_sorted)
    quantiles_ref = (np.arange(n_ref) + 0.5) / n_ref  # mid-point quantiles
    
    interp = interp1d(quantiles_ref, ref_sorted,
                       bounds_error=False,
                       fill_value=(ref_sorted[0], ref_sorted[-1]))
    
    # Map each source value
    n_src = len(source_values)
    ranks = np.searchsorted(source_sorted, source_values, side='right')
    # Use mid-quantile for stability
    quantiles_src = (ranks + 0.5) / n_src
    quantiles_src = np.clip(quantiles_src, 1e-10, 1 - 1e-10)
    
    calibrated = interp(quantiles_src)
    return calibrated


def apply_quantile_calibration(gdsc, depmap, tcga):
    """Apply quantile normalization with TCGA as reference."""
    print("\n[Q1] Quantile normalization (reference: TCGA)...")
    
    # TCGA (largest N) serves as the reference distribution for quantile calibration
    ref_vals = tcga['netith'].dropna().values

    gdsc_cal = gdsc.copy()
    depmap_cal = depmap.copy()
    tcga_cal = tcga.copy()
    
    gdsc_cal['netith_qn'] = quantile_normalize(
        gdsc['netith'].dropna().values, ref_vals)
    gdsc_cal['method'] = 'quantile_norm'
    
    depmap_cal['netith_qn'] = quantile_normalize(
        depmap['netith'].dropna().values, ref_vals)
    depmap_cal['method'] = 'quantile_norm'
    
    tcga_cal['netith_qn'] = tcga['netith'].values
    tcga_cal['method'] = 'quantile_norm'
    
    return gdsc_cal, depmap_cal, tcga_cal


# ============================================================
# 3. LOCATION-SCALE CALIBRATION
# ============================================================

def location_scale_calibrate(source_values: np.ndarray,
                              ref_mean: float, ref_std: float) -> np.ndarray:
    """Simple mean-variance shift: (x - μ_src) * (σ_ref / σ_src) + μ_ref."""
    src_mean = np.mean(source_values)
    src_std = np.std(source_values)
    
    if src_std < 1e-10:
        return np.full_like(source_values, ref_mean)
    
    calibrated = (source_values - src_mean) * (ref_std / src_std) + ref_mean
    return calibrated


def apply_location_scale_calibration(gdsc, depmap, tcga):
    """Apply location-scale calibration with TCGA as reference."""
    print("\n[LS] Location-scale calibration (reference: TCGA)...")
    
    # Reference = TCGA mean/std; GDSC and DepMap are affinely shifted onto that scale
    ref_vals = tcga['netith'].dropna().values
    ref_mean = np.mean(ref_vals)
    ref_std = np.std(ref_vals)
    
    gdsc_cal = gdsc.copy()
    depmap_cal = depmap.copy()
    tcga_cal = tcga.copy()
    
    gdsc_cal['netith_ls'] = location_scale_calibrate(
        gdsc['netith'].dropna().values, ref_mean, ref_std)
    gdsc_cal['method'] = 'location_scale'
    
    depmap_cal['netith_ls'] = location_scale_calibrate(
        depmap['netith'].dropna().values, ref_mean, ref_std)
    depmap_cal['method'] = 'location_scale'
    
    tcga_cal['netith_ls'] = tcga['netith'].values
    tcga_cal['method'] = 'location_scale'
    
    return gdsc_cal, depmap_cal, tcga_cal


# ============================================================
# 4. PER-CANCER Z-SCORING
# ============================================================

def apply_cancer_zscore(gdsc, tcga):
    """Within-cancer-type z-scoring for platform-independent ranking."""
    print("\n[CZ] Per-cancer-type Z-scoring...")
    
    # Map GDSC tissue names to TCGA codes
    tissue_to_tcga = {
        'lung': 'LUAD', 'breast': 'BRCA', 'colon': 'COAD',
        'ovary': 'OV', 'skin': 'SKCM', 'pancreas': 'PAAD',
        'prostate': 'PRAD', 'stomach': 'STAD', 'liver': 'LIHC',
        'kidney': 'KIRC', 'brain': 'GBM', 'thyroid': 'THCA',
        'endometrium': 'UCEC', 'bladder': 'BLCA', 'oesophagus': 'ESCA',
        'head and neck': 'HNSC', 'bone': 'SARC', 'blood': 'LAML',
        'haematopoietic': 'LAML', 'lymphoid': 'DLBC',
    }
    
    def map_tissue(ct):
        ct_lower = str(ct).lower()
        for tissue, tcga_code in tissue_to_tcga.items():
            if tissue in ct_lower:
                return tcga_code
        return None
    
    # Within-cancer-type z-scores make platform-independent ranking possible (GDSC scored with TCGA stats)
    gdsc_cal = gdsc.copy()
    gdsc_cal['tcga_type'] = gdsc_cal['cancer_type'].apply(map_tissue)
    
    tcga_cal = tcga.copy()
    
    # Compute per-cancer mean/std in TCGA
    tcga_stats = tcga_cal.groupby('cancer_type').agg(
        tcga_mean=('netith', 'mean'),
        tcga_std=('netith', 'std'),
    ).reset_index()
    
    # Apply to TCGA
    tcga_cal = tcga_cal.merge(tcga_stats, on='cancer_type', how='left')
    tcga_cal['netith_z'] = ((tcga_cal['netith'] - tcga_cal['tcga_mean']) / 
                              tcga_cal['tcga_std'].replace(0, 1))
    
    # Apply to GDSC using TCGA stats
    gdsc_cal = gdsc_cal.merge(tcga_stats, left_on='tcga_type', 
                                right_on='cancer_type', how='left',
                                suffixes=('', '_ref'))
    gdsc_cal['netith_z'] = ((gdsc_cal['netith'] - gdsc_cal['tcga_mean']) / 
                              gdsc_cal['tcga_std'].replace(0, 1))
    
    gdsc_cal['method'] = 'cancer_zscore'
    tcga_cal['method'] = 'cancer_zscore'
    
    return gdsc_cal, tcga_cal


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def evaluate_calibration(gdsc_raw, tcga_raw,
                          gdsc_cal, tcga_cal,
                          method_label: str,
                          cal_col: str = 'netith_qn') -> dict:
    """Evaluate calibration quality with multiple metrics."""
    gdsc_vals = gdsc_raw['netith'].dropna().values
    tcga_vals = tcga_raw['netith'].dropna().values
    gdsc_cal_vals = gdsc_cal[cal_col].dropna().values
    tcga_cal_vals = tcga_cal[cal_col].dropna().values
    
    results = {'method': method_label}
    
    # 1. Distribution similarity (Wasserstein, KS)
    # Distribution similarity: Wasserstein distance + two-sample Kolmogorov-Smirnov test (null: identical distributions)
    # Before
    ws_before = wasserstein_distance(gdsc_vals, tcga_vals)
    ks_before = ks_2samp(gdsc_vals, tcga_vals)
    # After
    ws_after = wasserstein_distance(gdsc_cal_vals, tcga_cal_vals)
    ks_after = ks_2samp(gdsc_cal_vals, tcga_cal_vals)
    
    results['wasserstein_before'] = ws_before
    results['wasserstein_after'] = ws_after
    results['wasserstein_reduction'] = (ws_before - ws_after) / max(ws_before, 1e-10)
    results['ks_stat_before'] = ks_before.statistic
    results['ks_stat_after'] = ks_after.statistic
    results['ks_p_before'] = ks_before.pvalue
    results['ks_p_after'] = ks_after.pvalue
    
    print(f"\n  [{method_label}]")
    print(f"    Wasserstein: {ws_before:.3f} → {ws_after:.3f} "
          f"({results['wasserstein_reduction']:+.1%})")
    print(f"    KS stat: {ks_before.statistic:.3f} → {ks_after.statistic:.3f} "
          f"(p: {ks_before.pvalue:.2e} → {ks_after.pvalue:.2e})")
    
    # 2. Mean/variance alignment
    results['gdsc_mean_before'] = np.mean(gdsc_vals)
    results['gdsc_mean_after'] = np.mean(gdsc_cal_vals)
    results['gdsc_std_before'] = np.std(gdsc_vals)
    results['gdsc_std_after'] = np.std(gdsc_cal_vals)
    results['tcga_mean'] = np.mean(tcga_vals)
    results['tcga_std'] = np.std(tcga_vals)
    
    print(f"    GDSC mean: {results['gdsc_mean_before']:.2f} → "
          f"{results['gdsc_mean_after']:.2f} (TCGA: {results['tcga_mean']:.2f})")
    print(f"    GDSC std:  {results['gdsc_std_before']:.2f} → "
          f"{results['gdsc_std_after']:.2f} (TCGA: {results['tcga_std']:.2f})")
    
    return results


def evaluate_cancer_ranking(gdsc_cal, tcga_cal, cal_col: str = 'netith_qn') -> dict:
    """Evaluate cross-platform cancer-type ranking concordance."""
    # Map GDSC tissues to TCGA codes
    tissue_to_tcga = {
        'lung': 'LUAD', 'breast': 'BRCA', 'colon': 'COAD',
        'ovary': 'OV', 'skin': 'SKCM', 'pancreas': 'PAAD',
        'prostate': 'PRAD', 'stomach': 'STAD', 'liver': 'LIHC',
        'kidney': 'KIRC', 'brain': 'GBM', 'thyroid': 'THCA',
        'endometrium': 'UCEC', 'bladder': 'BLCA', 'head and neck': 'HNSC',
    }
    
    def map_tissue(ct):
        ct_lower = str(ct).lower()
        for tissue, tcga_code in tissue_to_tcga.items():
            if tissue in ct_lower:
                return tcga_code
        return None
    
    gdsc_cal_copy = gdsc_cal.copy()
    gdsc_cal_copy['tcga_type'] = gdsc_cal_copy['cancer_type'].apply(map_tissue)
    
    # Aggregate by cancer type
    if 'cancer_type' not in tcga_cal.columns:
        return {}
    
    gdsc_ct = gdsc_cal_copy.dropna(subset=['tcga_type']).groupby('tcga_type').agg(
        gdsc_mean=(cal_col, 'mean'),
        gdsc_n=(cal_col, 'count'),
    )
    
    tcga_ct = tcga_cal.groupby('cancer_type').agg(
        tcga_mean=(cal_col, 'mean'),
        tcga_n=(cal_col, 'count'),
    )
    
    common = gdsc_ct.join(tcga_ct, how='inner')
    
    if len(common) < 5:
        return {'n_shared_types': len(common)}
    
    # Cancer-type ranking concordance tests (null: no monotonic / linear association across platforms)
    rho, p = spearmanr(common['gdsc_mean'], common['tcga_mean'])
    r, rp = pearsonr(common['gdsc_mean'], common['tcga_mean'])
    
    print(f"    Cancer-type ranking: ρ={rho:+.3f} (p={p:.4f}), "
          f"r={r:+.3f}, N={len(common)} shared types")
    
    return {
        'n_shared_types': len(common),
        'spearman_rho': rho,
        'spearman_p': p,
        'pearson_r': r,
        'pearson_p': rp,
    }


# ============================================================
# 6. VISUALIZATION
# ============================================================

def plot_calibration_before_after(gdsc_raw, tcga_raw,
                                    gdsc_cal_qn, tcga_cal_qn,
                                    gdsc_cal_ls, tcga_cal_ls,
                                    output_dir: str):
    """Comprehensive before/after calibration comparison."""
    # Six panels: raw / quantile-norm / location-scale densities, Q-Q plots before and after, Wasserstein bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # Row 1: Before calibration
    # Row 2: After calibration (quantile norm)
    
    colors = {'GDSC': '#d62728', 'TCGA': '#1f77b4'}
    
    # A: Density before
    ax = axes[0, 0]
    ax.hist(gdsc_raw['netith'].dropna(), bins=50, alpha=0.5,
            color=colors['GDSC'], density=True, label=f'GDSC (n={len(gdsc_raw)})')
    ax.hist(tcga_raw['netith'].dropna(), bins=50, alpha=0.5,
            color=colors['TCGA'], density=True, label=f'TCGA (n={len(tcga_raw)})')
    ax.set_xlabel('NetITH (raw)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('A: Before Calibration', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    # B: Density after QN
    ax = axes[0, 1]
    ax.hist(gdsc_cal_qn['netith_qn'].dropna(), bins=50, alpha=0.5,
            color=colors['GDSC'], density=True, label=f'GDSC (QN)')
    ax.hist(tcga_cal_qn['netith_qn'].dropna(), bins=50, alpha=0.5,
            color=colors['TCGA'], density=True, label=f'TCGA (ref)')
    ax.set_xlabel('NetITH (quantile normalized)', fontsize=11)
    ax.set_title('B: Quantile Normalization', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    # C: Density after LS
    ax = axes[0, 2]
    ax.hist(gdsc_cal_ls['netith_ls'].dropna(), bins=50, alpha=0.5,
            color=colors['GDSC'], density=True, label=f'GDSC (LS)')
    ax.hist(tcga_cal_ls['netith_ls'].dropna(), bins=50, alpha=0.5,
            color=colors['TCGA'], density=True, label=f'TCGA (ref)')
    ax.set_xlabel('NetITH (location-scale)', fontsize=11)
    ax.set_title('C: Location-Scale Adjustment', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    # D: Q-Q plot before
    ax = axes[1, 0]
    gdsc_sorted = np.sort(gdsc_raw['netith'].dropna().values)
    tcga_sorted = np.sort(tcga_raw['netith'].dropna().values)
    # Sample to same length for Q-Q
    n_qq = min(len(gdsc_sorted), len(tcga_sorted), 2000)
    gdsc_qq = np.quantile(gdsc_sorted, np.linspace(0, 1, n_qq))
    tcga_qq = np.quantile(tcga_sorted, np.linspace(0, 1, n_qq))
    ax.scatter(tcga_qq, gdsc_qq, s=10, c='#9467bd', alpha=0.4)
    ax.plot([tcga_qq.min(), tcga_qq.max()],
            [tcga_qq.min(), tcga_qq.max()],
            'k--', linewidth=1.5, label='y=x')
    ax.set_xlabel('TCGA NetITH Quantiles', fontsize=11)
    ax.set_ylabel('GDSC NetITH Quantiles', fontsize=11)
    ax.set_title('D: Q-Q Plot (Before)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    # E: Q-Q plot after QN
    ax = axes[1, 1]
    gdsc_qn_sorted = np.sort(gdsc_cal_qn['netith_qn'].dropna().values)
    tcga_qn_sorted = np.sort(tcga_cal_qn['netith_qn'].dropna().values)
    n_qq2 = min(len(gdsc_qn_sorted), len(tcga_qn_sorted), 2000)
    gdsc_qq2 = np.quantile(gdsc_qn_sorted, np.linspace(0, 1, n_qq2))
    tcga_qq2 = np.quantile(tcga_qn_sorted, np.linspace(0, 1, n_qq2))
    ax.scatter(tcga_qq2, gdsc_qq2, s=10, c='#2ca02c', alpha=0.4)
    ax.plot([tcga_qq2.min(), tcga_qq2.max()],
            [tcga_qq2.min(), tcga_qq2.max()],
            'k--', linewidth=1.5, label='y=x')
    ax.set_xlabel('TCGA NetITH Quantiles', fontsize=11)
    ax.set_ylabel('GDSC NetITH Quantiles (QN)', fontsize=11)
    ax.set_title('E: Q-Q Plot (After QN)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    # F: Method comparison — Wasserstein reduction
    ax = axes[1, 2]
    # Compute Wasserstein for all three methods
    methods = ['Raw', 'Quantile Norm', 'Location-Scale']
    ws_vals = [
        wasserstein_distance(
            gdsc_raw['netith'].dropna().values,
            tcga_raw['netith'].dropna().values),
        wasserstein_distance(
            gdsc_cal_qn['netith_qn'].dropna().values,
            tcga_cal_qn['netith_qn'].dropna().values),
        wasserstein_distance(
            gdsc_cal_ls['netith_ls'].dropna().values,
            tcga_cal_ls['netith_ls'].dropna().values),
    ]
    bar_colors = ['#d62728', '#2ca02c', '#1f77b4']
    bars = ax.bar(methods, ws_vals, color=bar_colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, ws_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Wasserstein Distance (GDSC vs TCGA)', fontsize=11)
    ax.set_title('F: Distribution Alignment (↓ better)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    out = f'{output_dir}/figures/calibration_before_after.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


def plot_concordance_improvement(gdsc_raw, tcga_raw,
                                   gdsc_cal_qn, tcga_cal_qn,
                                   output_dir: str):
    """Show cancer-type ranking concordance improvement."""
    # Map GDSC cancer types to TCGA
    tissue_to_tcga = {
        'lung': 'LUAD', 'breast': 'BRCA', 'colon': 'COAD',
        'ovary': 'OV', 'skin': 'SKCM', 'pancreas': 'PAAD',
        'prostate': 'PRAD', 'stomach': 'STAD', 'liver': 'LIHC',
        'kidney': 'KIRC', 'brain': 'GBM', 'thyroid': 'THCA',
        'endometrium': 'UCEC', 'bladder': 'BLCA', 'head and neck': 'HNSC',
    }
    
    def map_tissue(ct):
        ct_lower = str(ct).lower()
        for tissue, tcga_code in tissue_to_tcga.items():
            if tissue in ct_lower:
                return tcga_code
        return None
    
    # Side-by-side cancer-type ranking concordance before and after quantile normalization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for method_idx, (gdsc_df, tcga_df, col, title) in enumerate([
        (gdsc_raw, tcga_raw, 'netith', 'Before Calibration'),
        (gdsc_cal_qn, tcga_cal_qn, 'netith_qn', 'After Quantile Normalization'),
    ]):
        ax = axes[method_idx]
        
        gdsc_c = gdsc_df.copy()
        gdsc_c['tcga_type'] = gdsc_c['cancer_type'].apply(map_tissue)
        
        gdsc_ct = gdsc_c.dropna(subset=['tcga_type']).groupby('tcga_type').agg(
            gdsc_mean=(col, 'mean'))
        
        tcga_ct = tcga_df.groupby('cancer_type').agg(
            tcga_mean=(col, 'mean'))
        
        common = gdsc_ct.join(tcga_ct, how='inner')
        
        if len(common) >= 5:
            rho, p = spearmanr(common['gdsc_mean'], common['tcga_mean'])
            
            ax.scatter(common['gdsc_mean'], common['tcga_mean'],
                       s=100, c='#1f77b4', alpha=0.7, edgecolors='white',
                       linewidth=0.5)
            
            # Label cancer types
            for ct in common.index:
                ax.annotate(ct, (common.loc[ct, 'gdsc_mean'],
                                 common.loc[ct, 'tcga_mean']),
                            fontsize=8, ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points')
            
            # Trend line
            z = np.polyfit(common['gdsc_mean'], common['tcga_mean'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(common['gdsc_mean'].min(), 
                                  common['gdsc_mean'].max(), 100)
            ax.plot(x_line, p_line(x_line), '--', color='#d62728',
                    linewidth=1.5, alpha=0.6)
            
            ax.text(0.05, 0.95,
                    f'Spearman ρ = {rho:+.3f}\np = {p:.4f}\nN = {len(common)}',
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel('GDSC Mean NetITH', fontsize=12)
        ax.set_ylabel('TCGA Mean NetITH', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    out = f'{output_dir}/figures/calibration_concordance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


# ============================================================
# 7. SAVE CALIBRATED DATA
# ============================================================

def save_calibrated_data(gdsc_cal_qn, depmap_cal_qn, tcga_cal_qn,
                          gdsc_cal_ls, depmap_cal_ls, tcga_cal_ls):
    """Save calibrated NetITH values with all methods."""
    print("\n[Save] Writing calibrated NetITH data...")
    
    # Combine all
    def prep_df(df, cal_col, method):
        out = pd.DataFrame()
        if 'cell_line' in df.columns or df.index.name == 'cell_line':
            out['sample_id'] = df.index if df.index.name else df.index
        elif 'depmap_id' in df.columns:
            out['sample_id'] = df['depmap_id']
        elif 'sample' in df.columns:
            out['sample_id'] = df['sample']
        else:
            out['sample_id'] = df.index
        
        out['platform'] = df['platform']
        out['cancer_type'] = df['cancer_type'] if 'cancer_type' in df.columns else 'Unknown'
        out['netith_raw'] = df['netith']
        out[f'netith_{method}'] = df[cal_col]
        out['method'] = method
        return out
    
    qn_data = pd.concat([
        prep_df(gdsc_cal_qn, 'netith_qn', 'quantile_norm'),
        prep_df(depmap_cal_qn, 'netith_qn', 'quantile_norm'),
        prep_df(tcga_cal_qn, 'netith_qn', 'quantile_norm'),
    ], ignore_index=True)
    
    ls_data = pd.concat([
        prep_df(gdsc_cal_ls, 'netith_ls', 'location_scale'),
        prep_df(depmap_cal_ls, 'netith_ls', 'location_scale'),
        prep_df(tcga_cal_ls, 'netith_ls', 'location_scale'),
    ], ignore_index=True)
    
    # Merge methods
    qn_keep = qn_data[['sample_id', 'platform', 'cancer_type', 
                         'netith_raw', 'netith_quantile_norm']]
    ls_keep = ls_data[['sample_id', 'netith_location_scale']]
    
    # Merge quantile-norm and location-scale outputs per sample into one calibrated table
    combined = qn_keep.merge(ls_keep, on='sample_id', how='outer')
    
    # Write calibrated NetITH table (raw + both calibration methods per sample)
    combined.to_csv(f'{OUTPUT_DIR}/crossplatform_calibrated_netith.csv', 
                    index=False)
    print(f"  -> Saved crossplatform_calibrated_netith.csv "
          f"({len(combined)} samples)")
    
    return combined


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("P2: Cross-Platform NetITH Calibration")
    print("=" * 60)
    
    # 1. Load all platforms
    gdsc, depmap, tcga = load_all_platforms()
    
    # 2. Quantile normalization
    gdsc_qn, depmap_qn, tcga_qn = apply_quantile_calibration(gdsc, depmap, tcga)
    qn_eval = evaluate_calibration(gdsc, tcga, gdsc_qn, tcga_qn,
                                     'Quantile Normalization', 'netith_qn')
    qn_rank = evaluate_cancer_ranking(gdsc_qn, tcga_qn, 'netith_qn')
    
    # 3. Location-scale calibration
    gdsc_ls, depmap_ls, tcga_ls = apply_location_scale_calibration(gdsc, depmap, tcga)
    ls_eval = evaluate_calibration(gdsc, tcga, gdsc_ls, tcga_ls,
                                     'Location-Scale', 'netith_ls')
    ls_rank = evaluate_cancer_ranking(gdsc_ls, tcga_ls, 'netith_ls')
    
    # 4. Per-cancer z-scoring
    gdsc_z, tcga_z = apply_cancer_zscore(gdsc, tcga)
    
    # 5. Raw (no calibration) baseline
    raw_eval = evaluate_calibration(gdsc, tcga, gdsc, tcga,
                                      'Raw (No Calibration)', 'netith')
    raw_rank = evaluate_cancer_ranking(gdsc, tcga, 'netith')
    
    # 6. Save calibration stats
    print("\n[Stats] Calibration comparison summary...")
    all_stats = []
    for eval_dict, rank_dict, method in [
        (raw_eval, raw_rank, 'raw'),
        (qn_eval, qn_rank, 'quantile_norm'),
        (ls_eval, ls_rank, 'location_scale'),
    ]:
        stat_row = {**eval_dict}
        if rank_dict:
            stat_row.update({f'rank_{k}': v for k, v in rank_dict.items()})
        stat_row['method'] = method
        all_stats.append(stat_row)
    
    # Assemble per-method evaluation rows (raw / quantile_norm / location_scale) and write the comparison table
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(f'{OUTPUT_DIR}/crossplatform_calibration_stats.csv',
                    index=False)
    print(f"  -> Saved crossplatform_calibration_stats.csv")
    
    # 7. Save calibrated data
    save_calibrated_data(gdsc_qn, depmap_qn, tcga_qn,
                          gdsc_ls, depmap_ls, tcga_ls)
    
    # 8. Visualization
    print("\n[Viz] Generating calibration figures...")
    plot_calibration_before_after(gdsc, tcga,
                                    gdsc_qn, tcga_qn,
                                    gdsc_ls, tcga_ls,
                                    OUTPUT_DIR)
    plot_concordance_improvement(gdsc, tcga,
                                   gdsc_qn, tcga_qn,
                                   OUTPUT_DIR)
    
    # 9. Final summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Method':<25} {'Wasserstein':>12} {'KS Stat':>10} {'Reduction':>10}")
    print(f"  {'-'*57}")
    for s in all_stats:
        print(f"  {s['method']:<25} {s['wasserstein_before']:>6.3f}→{s['wasserstein_after']:<6.3f} "
              f"{s['ks_stat_before']:>5.3f}→{s['ks_stat_after']:<5.3f} "
              f"{s['wasserstein_reduction']:>+9.1%}")
    
    # Recommend best method
    # Pick the recommended method as the one with the smallest post-calibration Wasserstein distance
    best = min(all_stats, key=lambda x: x['wasserstein_after'])
    print(f"\n  ★ Recommended: {best['method']} "
          f"(Wasserstein={best['wasserstein_after']:.3f}, "
          f"Δ={best['wasserstein_reduction']:+.1%})")
    
    print("\nDone! All outputs in results/depmap/")


if __name__ == '__main__':
    main()
