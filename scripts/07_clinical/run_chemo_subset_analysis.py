#!/usr/bin/env python3
"""run_chemo_subset_analysis.py — TCGA chemotherapy-treated subset Cox analysis.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : results/tcga/tmb_vs_netith/tmb_netith_merged.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/chemo_subset/{tcga_treatment_classified.csv, chemo_results.json}; results/chemo_subset/figures/chemo_netith_analysis.pdf
Pipeline: clinical stage — see repository README
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TCGA_TMB = PROJECT_ROOT / 'results/tcga/tmb_vs_netith/tmb_netith_merged.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/chemo_subset'
FIGURE_DIR = OUTPUT_DIR / 'figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def classify_treatment(tcga):
    """Classify patients as treated vs untreated based on treatment outcome."""
    treated_outcomes = [
        'Complete Remission/Response',
        'Partial Remission/Response',
        'Progressive Disease',
        'Stable Disease',
        'Persistent Disease',
        'Normalization of Tumor Markers, but Residual Tumor Mass',
    ]
    tcga = tcga.copy()
    tcga['treated'] = tcga['treatment_outcome_first_course'].isin(treated_outcomes).astype(int)
    tcga['treated_label'] = tcga['treated'].map({1: 'Treated', 0: 'Untreated/Unknown'})
    
    print(f"\n  Treatment classification:")
    print(f"    Treated:     {tcga['treated'].sum():,}")
    print(f"    Untreated:   {(tcga['treated']==0).sum():,}")
    
    return tcga


def cox_by_treatment(tcga):
    """Run Cox PH models stratified by treatment status."""
    print("\n" + "=" * 65)
    print("PAN-CANCER: NetITH Survival by Treatment Status")
    print("=" * 65)
    
    results = {}
    
    for label, subset in [
        ('All', tcga),
        ('Treated', tcga[tcga['treated'] == 1]),
        ('Untreated', tcga[tcga['treated'] == 0])
    ]:
        cox_df = subset[['OS.time', 'OS.event', 'netith_bulk', 'age',
                          'stage_numeric', 'gender_male']].dropna()
        cox_df['OS.event'] = cox_df['OS.event'].astype(int)
        cox_df = cox_df[cox_df['OS.time'] > 0]
        
        if len(cox_df) < 50:
            results[label] = {'n': len(cox_df), 'hr': np.nan, 'p': np.nan}
            continue
        
        try:
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(cox_df, duration_col='OS.time', event_col='OS.event',
                    formula='netith_bulk + age + stage_numeric + gender_male')
            hr = np.exp(cph.params_['netith_bulk'])
            p = cph.summary.loc['netith_bulk', 'p']
            results[label] = {
                'n': len(cox_df),
                'hr': float(hr),
                'p': float(p),
                'ci_lower': float(np.exp(cph.confidence_intervals_.loc['netith_bulk', '95% lower-bound'])),
                'ci_upper': float(np.exp(cph.confidence_intervals_.loc['netith_bulk', '95% upper-bound'])),
                'concordance': float(cph.concordance_index_),
            }
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {label:<12} HR={hr:.3f} [{results[label]['ci_lower']:.3f}-{results[label]['ci_upper']:.3f}] p={p:.2e} n={len(cox_df):,} {sig}")
        except (ConvergenceError, Exception) as e:
            results[label] = {'n': len(cox_df), 'hr': np.nan, 'p': np.nan}
            print(f"  {label:<12} FAILED: {e}")
    
    # Interaction test
    print("\n  Interaction Test: NetITH × Treatment")
    cox_int = tcga[['OS.time', 'OS.event', 'netith_bulk', 'treated', 'age',
                     'stage_numeric', 'gender_male']].dropna().copy()
    cox_int['OS.event'] = cox_int['OS.event'].astype(int)
    cox_int = cox_int[cox_int['OS.time'] > 0]
    cox_int['netith_x_treated'] = cox_int['netith_bulk'] * cox_int['treated']
    
    try:
        cph_int = CoxPHFitter(penalizer=0.05)
        cph_int.fit(cox_int, duration_col='OS.time', event_col='OS.event',
                    formula='netith_bulk + treated + netith_x_treated + age + stage_numeric + gender_male')
        int_p = cph_int.summary.loc['netith_x_treated', 'p']
        results['interaction_p'] = float(int_p)
        print(f"  Interaction p = {int_p:.4f}")
        if int_p < 0.05:
            print(f"  → Significant interaction: treatment modifies NetITH-survival association")
        else:
            print(f"  → No significant interaction: NetITH effect is similar across groups")
    except (ConvergenceError, Exception) as e:
        results['interaction_p'] = None
        print(f"  Interaction test FAILED: {e}")
    
    return results


def per_cancer_analysis(tcga, min_treated=20):
    """Per-cancer NetITH HR in treated vs untreated."""
    print("\n" + "=" * 65)
    print("PER-CANCER: Treatment-Stratified Analysis")
    print("=" * 65)
    
    cancer_results = []
    
    for cancer in sorted(tcga['cancer_type'].unique()):
        cdf = tcga[tcga['cancer_type'] == cancer]
        n_treated = cdf['treated'].sum()
        n_untreated = (cdf['treated'] == 0).sum()
        
        if n_treated < min_treated or n_untreated < min_treated:
            continue
        
        for label, subset in [('Treated', cdf[cdf['treated'] == 1]),
                               ('Untreated', cdf[cdf['treated'] == 0])]:
            cox_df = subset[['OS.time', 'OS.event', 'netith_bulk', 'age',
                              'stage_numeric', 'gender_male']].dropna()
            cox_df['OS.event'] = cox_df['OS.event'].astype(int)
            cox_df = cox_df[cox_df['OS.time'] > 0]
            
            if len(cox_df) < 20:
                continue
            
            try:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cox_df, duration_col='OS.time', event_col='OS.event',
                        formula='netith_bulk + age + stage_numeric + gender_male')
                hr = np.exp(cph.params_['netith_bulk'])
                p = cph.summary.loc['netith_bulk', 'p']
                cancer_results.append({
                    'cancer': cancer, 'group': label,
                    'n': len(cox_df), 'hr': float(hr), 'p': float(p),
                })
            except (ConvergenceError, Exception):
                pass
    
    if cancer_results:
        cr_df = pd.DataFrame(cancer_results)
        # Show cancers where treated vs untreated HR differs
        pivot = cr_df.pivot_table(index='cancer', columns='group', values=['hr', 'p', 'n'], aggfunc='first')
        print(f"\n  Analyzed {cr_df['cancer'].nunique()} cancers with ≥{min_treated} per group:")
        for cancer in sorted(cr_df['cancer'].unique()):
            c = cr_df[cr_df['cancer'] == cancer]
            if len(c) == 2:
                hr_t = c[c['group']=='Treated']['hr'].values[0]
                hr_u = c[c['group']=='Untreated']['hr'].values[0]
                p_t = c[c['group']=='Treated']['p'].values[0]
                p_u = c[c['group']=='Untreated']['p'].values[0]
                delta = hr_t - hr_u
                if abs(delta) > 0.1:
                    dir_t = 'PROTECTIVE' if hr_t < 1 else 'RISK'
                    dir_u = 'PROTECTIVE' if hr_u < 1 else 'RISK'
                    print(f"    {cancer:<6} Treated: HR={hr_t:.3f} (p={p_t:.3f}) {dir_t} | "
                          f"Untreated: HR={hr_u:.3f} (p={p_u:.3f}) {dir_u}")
        
        return cr_df
    return None


def make_figure(tcga, pan_results, cancer_results, output_path):
    """Visualize chemotherapy-stratified NetITH analysis."""
    fig = plt.figure(figsize=(7.0866, 5.3150), constrained_layout=True)
    
    # ── A: Forest plot ──
    ax_a = fig.add_subplot(2, 3, 1)
    labels = ['All', 'Treated', 'Untreated']
    hrs = [pan_results[l]['hr'] for l in labels if 'hr' in pan_results[l]]
    ci_lower = [pan_results[l].get('ci_lower', np.nan) for l in labels if 'hr' in pan_results[l]]
    ci_upper = [pan_results[l].get('ci_upper', np.nan) for l in labels if 'hr' in pan_results[l]]
    ps = [pan_results[l]['p'] for l in labels if 'p' in pan_results[l]]
    
    y_pos = range(len(labels))
    colors = ['#2c3e50', '#e74c3c', '#3498db']
    for i, (lbl, hr, lo, hi, c) in enumerate(zip(labels, hrs, ci_lower, ci_upper, colors)):
        ax_a.errorbar(hr, i, xerr=[[hr-lo], [hi-hr]], fmt='o', color=c, capsize=5, markersize=10)
    ax_a.set_yticks(y_pos); ax_a.set_yticklabels(labels)
    ax_a.axvline(1, color='grey', linestyle='--', alpha=0.5)
    ax_a.set_xlabel('Hazard Ratio (95% CI)'); ax_a.set_title('(A) NetITH HR by Treatment Status')
    for i, (hr, p) in enumerate(zip(hrs, ps)):
        s = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        ax_a.text(hr + 0.02, i, f'HR={hr:.3f} {s}', va='center', fontsize=7)
    
    # ── B: KM-style split by NetITH tertile × treatment ──
    ax_b = fig.add_subplot(2, 3, 2)
    # Use DSS (disease-specific survival) as cleaner endpoint
    tcga_plot = tcga.dropna(subset=['netith_bulk', 'OS.time', 'OS.event']).copy()
    tcga_plot['netith_tertile'] = pd.qcut(tcga_plot['netith_bulk'], 2, labels=['Low', 'High'])
    
    groups_data = {}
    for trt in [0, 1]:
        for nt in ['Low', 'High']:
            s = tcga_plot[(tcga_plot['treated']==trt) & (tcga_plot['netith_tertile']==nt)]
            if len(s) > 50:
                # Simple 3-year survival
                surv_3yr = (s['OS.time'] >= 1095).mean()
                groups_data[f"{'Trt' if trt else 'Unt'}-{nt}"] = surv_3yr
    
    if groups_data:
        bars = ax_b.bar(range(len(groups_data)), list(groups_data.values()),
                        color=['#e74c3c', '#c0392b', '#3498db', '#2980b9'])
        ax_b.set_xticks(range(len(groups_data)))
        ax_b.set_xticklabels(list(groups_data.keys()), fontsize=8)
        ax_b.set_ylabel('3-Year Survival Rate')
        ax_b.set_title('(B) 3-Year Survival by NetITH × Treatment')
        ax_b.set_ylim(0, 1)
    
    # ── C: Per-cancer treated vs untreated HR ──
    ax_c = fig.add_subplot(2, 3, 3)
    if cancer_results is not None and len(cancer_results) > 0:
        pivot = cancer_results.pivot_table(index='cancer', columns='group', values='hr', aggfunc='first').dropna()
        if len(pivot) > 0:
            x = pivot['Treated'].values
            y = pivot['Untreated'].values
            ax_c.scatter(x, y, c=np.where(x < y, '#e74c3c', '#3498db'), s=60, alpha=0.7)
            ax_c.plot([0.5, 2], [0.5, 2], 'k--', alpha=0.3)
            ax_c.axhline(1, color='grey', linestyle=':', alpha=0.3)
            ax_c.axvline(1, color='grey', linestyle=':', alpha=0.3)
            ax_c.set_xlabel('HR Treated'); ax_c.set_ylabel('HR Untreated')
            ax_c.set_title('(C) Per-Cancer: Treated vs Untreated HR')
            for i, cancer in enumerate(pivot.index[:15]):
                ax_c.annotate(cancer, (x[i], y[i]), fontsize=7, alpha=0.7)
    
    # ── D: NetITH distribution by treatment ──
    ax_d = fig.add_subplot(2, 3, 4)
    for trt, label, color in [(1, 'Treated', '#e74c3c'), (0, 'Untreated', '#3498db')]:
        values = tcga[tcga['treated'] == trt]['netith_bulk'].dropna()
        ax_d.hist(values, bins=50, alpha=0.5, label=label, color=color, density=True)
    ax_d.set_xlabel('NetITH'); ax_d.set_ylabel('Density')
    ax_d.set_title('(D) NetITH Distribution by Treatment')
    ax_d.legend()
    
    # ── E: HR by cancer (treated only) ──
    ax_e = fig.add_subplot(2, 3, 5)
    if cancer_results is not None and len(cancer_results) > 0:
        treated_hr = cancer_results[cancer_results['group'] == 'Treated'].nlargest(20, 'n')
        y_pos = range(len(treated_hr))
        hrs_t = treated_hr['hr'].values
        ps_t = treated_hr['p'].values
        ax_e.barh(y_pos, np.log2(np.clip(hrs_t, 0.125, 8)),
                  color=['#c0392b' if p < 0.05 else '#95a5a6' for p in ps_t])
        ax_e.set_yticks(y_pos)
        ax_e.set_yticklabels([f"{r['cancer']} (n={int(r['n'])})" 
                               for _, r in treated_hr.iterrows()], fontsize=7)
        ax_e.set_xlabel('log₂(HR)'); ax_e.axvline(0, color='black', lw=0.8)
        ax_e.set_title('(E) NetITH HR in Treated Patients')
    
    # ── F: Summary ──
    ax_f = fig.add_subplot(2, 3, 6)
    ax_f.axis('off')
    
    all_hr = pan_results.get('All', {}).get('hr', np.nan)
    trt_hr = pan_results.get('Treated', {}).get('hr', np.nan)
    unt_hr = pan_results.get('Untreated', {}).get('hr', np.nan)
    int_p = pan_results.get('interaction_p', None)
    
    lines = [
        "Chemotherapy-Stratified NetITH Analysis",
        "",
        f"  All patients:   HR = {all_hr:.3f}" if not np.isnan(all_hr) else "  All patients:   N/A",
        f"  Treated:        HR = {trt_hr:.3f}" if not np.isnan(trt_hr) else "  Treated:        N/A",
        f"  Untreated:      HR = {unt_hr:.3f}" if not np.isnan(unt_hr) else "  Untreated:      N/A",
        f"  Interaction p = {int_p:.4f}" if int_p is not None else "  Interaction: N/A",
        "",
        "  Interpretation:",
    ]
    
    if int_p is not None and int_p < 0.05:
        if trt_hr < unt_hr:
            lines.append("  NetITH is more protective in treated")
            lines.append("  patients → supports chemosensitivity")
        else:
            lines.append("  NetITH is more protective in untreated")
            lines.append("  patients → immune effect dominates")
    else:
        lines.append("  No significant treatment interaction.")
        lines.append("  NetITH survival effect is similar")
        lines.append("  regardless of chemotherapy status.")
        lines.append("  This supports the immune-mediated")
        lines.append("  mechanism over chemosensitivity.")
    
    ax_f.text(0.05, 0.95, '\n'.join(lines), transform=ax_f.transAxes,
              fontfamily='monospace', fontsize=8, va='top')
    
    fig.suptitle('TCGA Chemotherapy-Stratified NetITH Survival Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.savefig(output_path, dpi=300, facecolor='white')

    export_panels(fig, "EDFig8_treatment_stratified")
    plt.close()
    print(f"\n[Figure] → {output_path}")


def main():
    print("=" * 65)
    print("Analysis 3/6: Chemotherapy-Stratified NetITH Survival")
    print("=" * 65)
    
    # Load
    print("\n[1] Loading TCGA data...")
    tcga = pd.read_csv(TCGA_TMB)
    tcga['OS.event'] = tcga['OS'].astype(int)
    tcga = tcga[tcga['OS.time'] > 0].copy()
    print(f"  Samples: {len(tcga)}")
    
    # Classify
    print("\n[2] Classifying treatment status...")
    tcga = classify_treatment(tcga)
    
    # Pan-cancer Cox
    print("\n[3] Pan-cancer Cox analysis...")
    pan_results = cox_by_treatment(tcga)
    
    # Per-cancer
    print("\n[4] Per-cancer analysis...")
    cancer_results = per_cancer_analysis(tcga)
    
    # Save
    tcga.to_csv(OUTPUT_DIR / 'tcga_treatment_classified.csv', index=False)
    print(f"\n[Save] {OUTPUT_DIR / 'tcga_treatment_classified.csv'}")
    
    with open(OUTPUT_DIR / 'chemo_results.json', 'w') as f:
        json.dump(pan_results, f, indent=2, default=str)
    print(f"[Save] {OUTPUT_DIR / 'chemo_results.json'}")
    
    # Figure
    make_figure(tcga, pan_results, cancer_results, str(FIGURE_DIR / 'chemo_netith_analysis.pdf'))
    
    print("\n[Done] Analysis 3/6 complete.\n")


# --- panel export (per-panel PDF + 300dpi PNG for review/patchwork assembly) ---
PANEL_DIR = PROJECT_ROOT / "results/figures/panels"
def export_panels(fig, base_name):
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        letter = chr(65 + i)
        extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(PANEL_DIR / f"{base_name}_panel{letter}.{ext}",
                        bbox_inches=extent, facecolor="white", **kw)

if __name__ == '__main__':
    main()
