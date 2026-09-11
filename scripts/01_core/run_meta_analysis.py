#!/usr/bin/env python3
"""
run_meta_analysis.py — Pool NetITH prognostic hazard ratios across cohorts via random-effects meta-analysis.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - results/tcga/tcga_survival_results.csv: per-cancer TCGA Cox HRs
    - results/metabric_validation/metabric_cox.csv: METABRIC OS Cox results
    - results/metabric_validation/extensions/rfs_cox.csv: METABRIC RFS Cox results
    - results/imvigor210/imvigor210_netith_results.csv: IMvigor210 NetITH + survival
Outputs :
    - results/meta_analysis/cross_cohort_results.csv
    - results/meta_analysis/figures/meta_analysis_forest.pdf
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'results/meta_analysis'
FIG_DIR = OUTPUT_DIR / 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def random_effects_meta(log_hrs, se_log_hrs):
    """DerSimonian-Laird random-effects meta-analysis."""
    # Fixed-effect weights
    w_fixed = 1.0 / (se_log_hrs ** 2)
    # FE summary
    fe_summary = np.sum(w_fixed * log_hrs) / np.sum(w_fixed)
    # Heterogeneity Q
    Q = np.sum(w_fixed * (log_hrs - fe_summary) ** 2)
    df = len(log_hrs) - 1
    p_het = 1 - chi2.cdf(Q, df) if df > 0 else 1.0
    # I²
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    # Between-study variance (tau²)
    C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / C) if df > 0 and C > 0 else 0
    # Random-effects weights
    w_random = 1.0 / (se_log_hrs**2 + tau2)
    # RE summary
    re_summary = np.sum(w_random * log_hrs) / np.sum(w_random)
    se_re = np.sqrt(1.0 / np.sum(w_random))
    z_re = re_summary / se_re
    p_re = 2 * (1 - norm.cdf(abs(z_re)))
    hr_re = np.exp(re_summary)
    ci_lower = np.exp(re_summary - 1.96 * se_re)
    ci_upper = np.exp(re_summary + 1.96 * se_re)
    
    return {
        'fe_hr': np.exp(fe_summary),
        're_hr': hr_re,
        're_ci_lower': ci_lower,
        're_ci_upper': ci_upper,
        're_p': p_re,
        'Q': Q,
        'p_het': p_het,
        'I2': I2,
        'tau2': tau2,
    }


def build_dataset():
    """Compile all cohort results into a unified DataFrame."""
    records = []
    
    # --- TCGA pan-cancer (per-cancer HRs) ---
    tcga = pd.read_csv(PROJECT_ROOT / 'results/tcga/tcga_survival_results.csv')
    for _, row in tcga.iterrows():
        hr = row['cox_hr']
        p = row['cox_p']
        if pd.isna(hr) or pd.isna(p) or hr <= 0:
            continue
        log_hr = np.log(hr)
        # Approximate SE from p-value
        z = abs(norm.ppf(min(p, 0.99999) / 2))
        se = abs(log_hr) / z if z > 0 else 0.5
        
        records.append({
            'cohort': 'TCGA',
            'cancer': row['cancer_type'],
            'n': int(row['n']),
            'n_events': int(row['n_events']),
            'hr': hr,
            'ci_lower': np.exp(log_hr - 1.96 * se),
            'ci_upper': np.exp(log_hr + 1.96 * se),
            'p': p,
            'log_hr': log_hr,
            'se': se,
            'endpoint': 'OS',
            'category': 'per_cancer',
        })
    
    # --- TCGA BRCA standalone ---
    brca = tcga[tcga['cancer_type'] == 'BRCA'].iloc[0]
    records.append({
        'cohort': 'TCGA BRCA',
        'cancer': 'BRCA',
        'n': int(brca['n']),
        'n_events': int(brca['n_events']),
        'hr': brca['cox_hr'],
        'ci_lower': np.exp(np.log(brca['cox_hr']) - 1.96 * abs(np.log(brca['cox_hr'])) / abs(norm.ppf(min(brca['cox_p'], 0.999) / 2))),
        'ci_upper': np.exp(np.log(brca['cox_hr']) + 1.96 * abs(np.log(brca['cox_hr'])) / abs(norm.ppf(min(brca['cox_p'], 0.999) / 2))),
        'p': brca['cox_p'],
        'log_hr': np.log(brca['cox_hr']),
        'se': abs(np.log(brca['cox_hr'])) / abs(norm.ppf(min(brca['cox_p'], 0.999) / 2)),
        'endpoint': 'OS',
        'category': 'breast',
    })
    
    # --- METABRIC OS ---
    met_os = pd.read_csv(PROJECT_ROOT / 'results/metabric_validation/metabric_cox.csv')
    r_os = met_os[met_os['model'] == 'NetITH only'].iloc[0]
    log_hr_os = np.log(r_os['hr_netith'])
    se_os = abs(log_hr_os) / abs(norm.ppf(min(r_os['p_netith'], 0.999) / 2))
    records.append({
        'cohort': 'METABRIC',
        'cancer': 'BRCA',
        'n': int(r_os['n']),
        'n_events': int(r_os['n_events']),
        'hr': r_os['hr_netith'],
        'ci_lower': np.exp(log_hr_os - 1.96 * se_os),
        'ci_upper': np.exp(log_hr_os + 1.96 * se_os),
        'p': r_os['p_netith'],
        'log_hr': log_hr_os,
        'se': se_os,
        'endpoint': 'OS',
        'category': 'breast',
    })
    
    # --- METABRIC RFS ---
    rfs = pd.read_csv(PROJECT_ROOT / 'results/metabric_validation/extensions/rfs_cox.csv')
    if len(rfs) > 0:
        r_rfs = rfs[rfs['model'] == 'NetITH only'].iloc[0]
        log_hr_rfs = np.log(r_rfs['hr_netith'])
        se_rfs = abs(log_hr_rfs) / abs(norm.ppf(min(r_rfs['p_netith'], 0.999) / 2))
        records.append({
            'cohort': 'METABRIC',
            'cancer': 'BRCA',
            'n': int(r_rfs['n']),
            'n_events': int(r_rfs['n_events']),
            'hr': r_rfs['hr_netith'],
            'ci_lower': np.exp(log_hr_rfs - 1.96 * se_rfs),
            'ci_upper': np.exp(log_hr_rfs + 1.96 * se_rfs),
            'p': r_rfs['p_netith'],
            'log_hr': log_hr_rfs,
            'se': se_rfs,
            'endpoint': 'RFS',
            'category': 'breast',
        })
    
    # --- IMvigor210 ---
    from lifelines import CoxPHFitter
    imv = pd.read_csv(PROJECT_ROOT / 'results/imvigor210/imvigor210_netith_results.csv')
    imv_sub = imv[['os', 'censOS', 'NetITH']].dropna().rename(
        columns={'os': 'time', 'censOS': 'event', 'NetITH': 'netith'})
    imv_sub['time'] = imv_sub['time'].clip(lower=1)
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(imv_sub, 'time', 'event', formula='netith')
    hr_imv = float(cph.hazard_ratios_['netith'])
    p_imv = float(cph.summary.loc['netith', 'p'])
    log_hr_imv = np.log(hr_imv)
    se_imv = float(cph.summary.loc['netith', 'se(coef)'])
    
    records.append({
        'cohort': 'IMvigor210',
        'cancer': 'Bladder',
        'n': len(imv_sub),
        'n_events': int(imv_sub['event'].sum()),
        'hr': hr_imv,
        'ci_lower': np.exp(log_hr_imv - 1.96 * se_imv),
        'ci_upper': np.exp(log_hr_imv + 1.96 * se_imv),
        'p': p_imv,
        'log_hr': log_hr_imv,
        'se': se_imv,
        'endpoint': 'OS',
        'category': 'immunotherapy',
    })
    
    df = pd.DataFrame(records)
    return df


def make_forest_plot(df):
    """Two-panel forest plot: (A) All per-cancer TCGA, (B) Summary by cohort."""
    
    fig = plt.figure(figsize=(22, 14))
    
    # --- Panel A: All TCGA per-cancer ---
    ax_a = fig.add_subplot(1, 2, 1)
    tcga_df = df[df['category'] == 'per_cancer'].sort_values('hr')
    
    y_positions = range(len(tcga_df))
    for i, (_, row) in enumerate(tcga_df.iterrows()):
        color = '#c0392b' if row['p'] < 0.05 else '#3498db' if row['p'] < 0.1 else '#95a5a6'
        ax_a.errorbar(row['hr'], i, 
                     xerr=[[row['hr'] - row['ci_lower']], [row['ci_upper'] - row['hr']]],
                     fmt='o', color=color, capsize=2, markersize=6, linewidth=1.5)
    
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([f"{r['cancer']} (n={r['n']})" for _, r in tcga_df.iterrows()], fontsize=7)
    ax_a.axvline(1, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax_a.set_xlabel('Hazard Ratio (OS)', fontsize=10)
    
    sig_n = (tcga_df['p'] < 0.05).sum()
    ax_a.set_title(f'(A) TCGA Pan-Cancer: NetITH → OS\n{sig_n}/{len(tcga_df)} cancers p<0.05', 
                   fontsize=11, fontweight='bold')
    ax_a.set_xscale('log')
    ax_a.set_xlim(0.1, 60)
    
    # --- Panel B: Summary cohorts ---
    ax_b = fig.add_subplot(1, 2, 2)
    summary_df = df[df['category'] != 'per_cancer'].copy()
    
    # Order: TCGA BRCA, METABRIC OS, METABRIC RFS, IMvigor210
    order = ['TCGA BRCA', 'METABRIC', 'METABRIC', 'IMvigor210']
    type_order = ['breast', 'breast', 'breast', 'immunotherapy']
    
    y_b = []
    labels_b = []
    colors_b = []
    
    y_idx = len(summary_df) + 2  # leave room for meta
    
    for _, row in summary_df.iterrows():
        y_idx -= 1
        is_sig = row['p'] < 0.05
        color = '#c0392b' if is_sig else '#2c3e50'
        
        ax_b.errorbar(row['hr'], y_idx,
                     xerr=[[row['hr'] - row['ci_lower']], [row['ci_upper'] - row['hr']]],
                     fmt='s', color=color, capsize=4, markersize=10, linewidth=2)
        
        endpoint_str = f" [{row['endpoint']}]" if row['endpoint'] != 'OS' else ''
        label = f"{row['cohort']} {row['cancer']}{endpoint_str}\nn={row['n']}, events={row['n_events']}"
        sig = '***' if row['p']<0.001 else '**' if row['p']<0.01 else '*' if row['p']<0.05 else ''
        if sig:
            label += f' {sig}'
        labels_b.append(label)
        y_b.append(y_idx)
        colors_b.append(color)
        
        # Annotate HR
        ax_b.text(row['ci_upper'] * 1.1, y_idx, f'HR={row["hr"]:.2f}', va='center', fontsize=8)
    
    # Meta-analysis (random-effects) for breast cohorts
    breast_os = summary_df[(summary_df['category'] == 'breast') & (summary_df['endpoint'] == 'OS')]
    if len(breast_os) >= 2:
        meta = random_effects_meta(breast_os['log_hr'].values, breast_os['se'].values)
        
        y_meta = 0.5
        ax_b.errorbar(meta['re_hr'], y_meta,
                     xerr=[[meta['re_hr'] - meta['re_ci_lower']], [meta['re_ci_upper'] - meta['re_hr']]],
                     fmt='D', color='#e74c3c', capsize=6, markersize=14, linewidth=3, zorder=10)
        
        meta_label = f"RE Meta (Breast)\nHR={meta['re_hr']:.2f} [{meta['re_ci_lower']:.2f}-{meta['re_ci_upper']:.2f}]\n"
        meta_label += f"I²={meta['I2']:.0f}%, p={meta['re_p']:.3f}"
        labels_b.insert(0, meta_label)
        y_b.insert(0, y_meta)
        
        ax_b.text(meta['re_ci_upper'] * 1.1, y_meta, 
                 f'HR={meta["re_hr"]:.2f} ***' if meta['re_p']<0.001 else f'HR={meta["re_hr"]:.2f}',
                 va='center', fontsize=9, fontweight='bold')
    
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(labels_b, fontsize=8)
    ax_b.axvline(1, color='grey', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_b.set_xlabel('Hazard Ratio', fontsize=10)
    ax_b.set_title('(B) Cross-Cohort Validation: NetITH → Survival', fontsize=11, fontweight='bold')
    ax_b.set_xscale('log')
    ax_b.set_xlim(0.5, 50)
    
    fig.suptitle('Meta-Analysis: NetITH Prognostic Value Across Independent Cohorts\n'
                 'TCGA (29 cancers) · METABRIC (breast) · IMvigor210 (bladder immunotherapy)',
                 fontsize=13, fontweight='bold')
    
    pdf_path = FIG_DIR / 'meta_analysis_forest.pdf'
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [Figure] → {pdf_path}")
    return fig


def main():
    print("="*60)
    print("Meta-Analysis: NetITH Across Cohorts")
    print("="*60)
    
    df = build_dataset()
    print(f"\n  Compiled {len(df)} results from {df['cohort'].nunique()} cohorts")
    print(f"  Cohorts: {sorted(df['cohort'].unique())}")
    
    # Summary table
    print("\n  Cross-cohort summary:")
    summary = df[df['category'] != 'per_cancer']
    for _, row in summary.iterrows():
        sig = '***' if row['p']<0.001 else '**' if row['p']<0.01 else '*' if row['p']<0.05 else ''
        print(f"    {row['cohort']:<15} {row['cancer']:<8} {row['endpoint']:<5} "
              f"n={row['n']:<5} HR={row['hr']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}] p={row['p']:.3f} {sig}")
    
    # Meta-analysis
    breast_os = summary[(summary['category'] == 'breast') & (summary['endpoint'] == 'OS')]
    if len(breast_os) >= 2:
        meta = random_effects_meta(breast_os['log_hr'].values, breast_os['se'].values)
        print(f"\n  Random-effects meta (breast OS, {len(breast_os)} studies):")
        print(f"    RE HR: {meta['re_hr']:.2f} [{meta['re_ci_lower']:.2f}-{meta['re_ci_upper']:.2f}]")
        print(f"    p: {meta['re_p']:.4f}")
        print(f"    I²: {meta['I2']:.1f}%")
        print(f"    Q: {meta['Q']:.2f}, p_het: {meta['p_het']:.4f}")
    
    # Forest plot
    make_forest_plot(df)
    
    # Save
    df.to_csv(OUTPUT_DIR / 'cross_cohort_results.csv', index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
