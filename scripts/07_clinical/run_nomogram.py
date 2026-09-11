#!/usr/bin/env python3
"""run_nomogram.py — NetITH clinical prognostic nomogram (multivariate Cox across 7 TCGA cancer types).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : results/tcga/tcga_tri_modal_merged.csv; data/GraphHMM/clinical_real_tcga/{cancer}_tcga_clinical_survival.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/tcga/{nomogram_cox_results.csv, nomogram_cindex.csv, nomogram_patient_scores.csv}; results/tcga/figures/nomogram_{cancer}.png, nomogram_calibration_{cancer}.png, nomogram_forest.png, nomogram_cindex_bar.png
Pipeline: clinical stage — see repository README
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & MERGE DATA
# ============================================================

def load_tri_modal(path: str) -> pd.DataFrame:
    """Load tri-modal NetITH data and derive patient ID."""
    df = pd.read_csv(path)
    # sample: TCGA.OR.A5J1.01 -> patient: TCGA-OR-A5J1
    # Derive patient IDs from sample barcodes so tri-modal rows merge with per-cancer clinical files.
    def sample_to_patient(s):
        parts = s.split('.')
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        return None
    df['patient'] = df['sample'].apply(sample_to_patient)
    df = df.dropna(subset=['patient'])
    print(f"  Tri-modal: {len(df)} samples, {df['patient'].nunique()} patients")
    return df


def load_clinical_data(cancer: str, base_dir: str) -> pd.DataFrame:
    """Load per-cancer clinical data."""
    path = f"{base_dir}/{cancer}_tcga_clinical_survival.csv"
    try:
        df = pd.read_csv(path)
        # Rename submitter_id to patient for merging
        df['patient'] = df['submitter_id']
        print(f"  [{cancer}] Clinical: {len(df)} patients")
        return df
    except FileNotFoundError:
        print(f"  [{cancer}] No clinical data file")
        return None


def parse_ajcc_stage(stage_str):
    """Convert AJCC stage string to numeric.
    Stage I -> 1, Stage IA -> 1, Stage II -> 2, etc.
    Stage X -> NaN (unknown)
    """
    if pd.isna(stage_str) or not isinstance(stage_str, str):
        return np.nan
    s = stage_str.strip().lower()
    if 'stage x' in s or 'not reported' in s or s == '':
        return np.nan
    # Extract roman numeral
    for roman, arabic in [('iv', 4), ('iii', 3), ('ii', 2), ('i', 1)]:
        if roman in s:
            return float(arabic)
    return np.nan


def parse_ajcc_t(t_str):
    """Convert T-stage to numeric: T1->1, T1a->1, T2->2, etc."""
    if pd.isna(t_str) or not isinstance(t_str, str):
        return np.nan
    s = t_str.strip()
    if s.upper() in ['TX', 'T?', '']:
        return np.nan
    # Extract number after T
    import re
    # Leading digit of the T stage (T1a -> 1); TX / 'T?' already mapped to NaN above.
    m = re.match(r'[Tt](\d)', s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_ajcc_n(n_str):
    """Convert N-stage to numeric: N0->0, N1->1, N2->2, etc."""
    if pd.isna(n_str) or not isinstance(n_str, str):
        return np.nan
    s = n_str.strip()
    if s.upper() in ['NX', 'N?', '']:
        return np.nan
    import re
    # Leading digit of the N stage (N0 -> 0, N1 -> 1, ...).
    m = re.match(r'[Nn](\d)', s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_grade(grade_str):
    """Convert tumor grade to numeric."""
    if pd.isna(grade_str) or not isinstance(grade_str, str):
        return np.nan
    s = grade_str.strip().lower()
    # Map grade labels (G1-G4 or low/intermediate/high) to numeric 1-4; unparseable -> NaN.
    if 'grade 1' in s or 'g1' in s or s == 'low':
        return 1.0
    elif 'grade 2' in s or 'g2' in s or s == 'intermediate':
        return 2.0
    elif 'grade 3' in s or 'g3' in s or s == 'high':
        return 3.0
    elif 'grade 4' in s or 'g4' in s:
        return 4.0
    return np.nan


# ============================================================
# 2. COX MODEL & NOMOGRAM
# ============================================================

def build_nomogram_for_cancer(cancer: str, merged: pd.DataFrame,
                               output_dir: str) -> dict:
    """Build univariate and multivariate Cox models for one cancer."""
    from lifelines import CoxPHFitter

    results = {'cancer': cancer, 'n_total': len(merged)}

    # Identify available clinical variables
    clinical_vars = []
    # Keep only covariates with >= 20 non-missing observations in this cancer type.
    for col, label in [('age', 'Age'),
                        ('stage_numeric', 'Stage'),
                        ('t_numeric', 'T-stage'),
                        ('n_numeric', 'N-stage'),
                        ('grade_numeric', 'Grade'),
                        ('gender_male', 'Gender (Male)')]:
        if col in merged.columns and merged[col].notna().sum() >= 20:
            clinical_vars.append((col, label))

    if len(merged) < 50:
        results['status'] = 'insufficient_samples'
        print(f"  [{cancer}] Insufficient samples: {len(merged)}")
        return results

    # ---- Univariate: NetITH alone ----
    uv = merged[['patient', 'OS.time', 'OS', 'netith_bulk']].dropna()
    uv = uv[uv['OS.time'] > 0].copy()
    if len(uv) < 30 or uv['OS'].sum() < 5:
        results['status'] = 'insufficient_events'
        print(f"  [{cancer}] Insufficient events: {uv['OS'].sum()}")
        return results

    try:
        # Univariate Cox: NetITH alone; ridge penalty (penalizer=0.1) stabilizes fits on small cohorts.
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(uv[['OS.time', 'OS', 'netith_bulk']],
                duration_col='OS.time', event_col='OS')
        uv_summary = cph.summary
        results['uv_netith_hr'] = np.exp(uv_summary.loc['netith_bulk', 'coef'])
        results['uv_netith_p'] = uv_summary.loc['netith_bulk', 'p']
        results['uv_netith_ci_lower'] = np.exp(
            uv_summary.loc['netith_bulk', 'coef lower 95%'])
        results['uv_netith_ci_upper'] = np.exp(
            uv_summary.loc['netith_bulk', 'coef upper 95%'])
        results['uv_concordance'] = cph.concordance_index_
        print(f"  [{cancer}] UV NetITH HR={results['uv_netith_hr']:.3f}, "
              f"p={results['uv_netith_p']:.4f}, C={results['uv_concordance']:.3f}")
    except Exception as e:
        print(f"  [{cancer}] UV Cox failed: {e}")
        results['status'] = 'uv_failed'
        return results

    # ---- Multivariate: NetITH + clinical ----
    mv_cols = ['OS.time', 'OS', 'netith_bulk']
    for col, _ in clinical_vars:
        mv_cols.append(col)
    mv = merged[mv_cols].dropna()
    mv = mv[mv['OS.time'] > 0].copy()

    if len(mv) < 30:
        results['status'] = 'mv_insufficient'
        results['mv_n'] = len(mv)
        print(f"  [{cancer}] MV insufficient: {len(mv)}")
        return results

    try:
        # Multivariate Cox: NetITH adjusted for the available clinical covariates (stage, T/N, grade, age, gender).
        cph_mv = CoxPHFitter(penalizer=0.1)
        cph_mv.fit(mv[['OS.time', 'OS', 'netith_bulk'] + [c for c, _ in clinical_vars]],
                   duration_col='OS.time', event_col='OS')
        mv_summary = cph_mv.summary

        results['mv_n'] = len(mv)
        results['mv_netith_hr'] = np.exp(mv_summary.loc['netith_bulk', 'coef'])
        results['mv_netith_p'] = mv_summary.loc['netith_bulk', 'p']
        results['mv_netith_ci_lower'] = np.exp(
            mv_summary.loc['netith_bulk', 'coef lower 95%'])
        results['mv_netith_ci_upper'] = np.exp(
            mv_summary.loc['netith_bulk', 'coef upper 95%'])
        results['mv_concordance'] = cph_mv.concordance_index_

        # Model-level stats
        # Partial log-likelihood and AIC summarize overall multivariate model fit.
        results['mv_log_likelihood'] = cph_mv.log_likelihood_
        results['mv_aic'] = cph_mv.AIC_partial_

        # Clinical variable HRs
        for col in mv_summary.index:
            if col != 'netith_bulk':
                results[f'mv_{col}_hr'] = np.exp(mv_summary.loc[col, 'coef'])
                results[f'mv_{col}_p'] = mv_summary.loc[col, 'p']
                results[f'mv_{col}_ci_lower'] = np.exp(
                    mv_summary.loc[col, 'coef lower 95%'])
                results[f'mv_{col}_ci_upper'] = np.exp(
                    mv_summary.loc[col, 'coef upper 95%'])

        # ---- Predict risk scores ----
        # Per-patient nomogram risk scores (partial hazards); exported for run_dca_analysis.py.
        mv['risk_score'] = cph_mv.predict_partial_hazard(mv)
        mv['patient'] = merged.loc[mv.index, 'patient'] if 'patient' in merged.columns else None
        results['mv_patients'] = mv[['patient', 'risk_score', 'OS.time', 'OS']]

        print(f"  [{cancer}] MV NetITH HR={results['mv_netith_hr']:.3f}, "
              f"p={results['mv_netith_p']:.4f}, C={results['mv_concordance']:.3f}, "
              f"n={len(mv)}")

        # ---- Nomogram Plot ----
        try:
            plot_nomogram(cancer, cph_mv, mv, clinical_vars, output_dir)
        except Exception as pe:
            print(f"    -> Nomogram plot failed: {pe}")

        # ---- Calibration (by risk tertile) ----
        try:
            plot_calibration(cancer, mv, output_dir)
        except Exception as ce:
            print(f"    -> Calibration plot failed: {ce}")

        results['status'] = 'success'

    except Exception as e:
        print(f"  [{cancer}] MV Cox failed: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'mv_failed'

    return results


# ============================================================
# 3. VISUALIZATION
# ============================================================

def plot_nomogram(cancer: str, cph, data: pd.DataFrame,
                  clinical_vars: list, output_dir: str):
    """Create a custom nomogram-style visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 0.5], hspace=0.3)

    # ---- Panel A: Hazard Ratios (forest plot) ----
    ax_hr = fig.add_subplot(gs[0])
    summary = cph.summary
    vars_to_plot = []

    # NetITH first
    if 'netith_bulk' in summary.index:
        r = summary.loc['netith_bulk']
        vars_to_plot.append(('NetITH', np.exp(r['coef']),
                             np.exp(r['coef lower 95%']),
                             np.exp(r['coef upper 95%']), r['p']))

    # Clinical vars
    label_map = {'age': 'Age', 'stage_numeric': 'AJCC Stage',
                 't_numeric': 'T-stage', 'n_numeric': 'N-stage',
                 'grade_numeric': 'Tumor Grade', 'gender_male': 'Gender (M)'}
    for col, _ in clinical_vars:
        if col in summary.index:
            r = summary.loc[col]
            lbl = label_map.get(col, col)
            vars_to_plot.append((lbl, np.exp(r['coef']),
                                 np.exp(r['coef lower 95%']),
                                 np.exp(r['coef upper 95%']), r['p']))

    y_pos = range(len(vars_to_plot))
    hrs = [v[1] for v in vars_to_plot]
    # xerr must be absolute CI bounds, not offsets
    ci_abs_lower = [max(0.001, v[2]) for v in vars_to_plot]
    ci_abs_upper = [v[3] for v in vars_to_plot]

    # Use errorbar directly for proper CI display
    colors = ['#d62728' if hr > 1 else '#1f77b4' for hr in hrs]
    ax_hr.barh(y_pos, hrs, color=colors, edgecolor='black', height=0.5, alpha=0.9)
    # Add error bars
    for i, (hr, lo, hi) in enumerate(zip(hrs, ci_abs_lower, ci_abs_upper)):
        ax_hr.plot([lo, hi], [i, i], 'k-', linewidth=2)
        ax_hr.plot([lo, lo], [i-0.15, i+0.15], 'k-', linewidth=1.5)
        ax_hr.plot([hi, hi], [i-0.15, i+0.15], 'k-', linewidth=1.5)
    ax_hr.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    ax_hr.set_yticks(y_pos)
    ax_hr.set_yticklabels([v[0] for v in vars_to_plot], fontsize=11)
    ax_hr.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax_hr.set_title(f'{cancer}: Multivariate Cox Hazard Ratios', fontsize=14,
                    fontweight='bold')
    ax_hr.set_xlim(0, max(hrs) * 1.4 + 0.5)
    ax_hr.grid(axis='x', alpha=0.3)

    # Add p-values
    for i, (name, hr, lo, hi, p) in enumerate(vars_to_plot):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax_hr.text(hr + 0.05, i, f' {hr:.2f} [{lo:.2f}-{hi:.2f}] {sig}',
                   va='center', fontsize=9, color='#333333')

    # ---- Panel B: Risk Score Distribution ----
    ax_risk = fig.add_subplot(gs[1])
    risk = data['risk_score'].values
    median_risk = np.median(risk)
    # Median split into low- vs high-risk groups for visualization.
    high_mask = risk > median_risk

    ax_risk.hist(risk[~high_mask], bins=30, alpha=0.6, color='#1f77b4',
                 label=f'Low Risk (n={sum(~high_mask)})', edgecolor='white')
    ax_risk.hist(risk[high_mask], bins=30, alpha=0.6, color='#d62728',
                 label=f'High Risk (n={sum(high_mask)})', edgecolor='white')
    ax_risk.axvline(median_risk, color='black', linestyle='--', linewidth=1.5,
                    label=f'Median = {median_risk:.3f}')
    ax_risk.set_xlabel('Risk Score (Partial Hazard)', fontsize=12)
    ax_risk.set_ylabel('Count', fontsize=12)
    ax_risk.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
    ax_risk.legend(fontsize=9, loc='upper right')

    # ---- Panel C: Score Table ----
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis('off')

    # Points calculation (simplified scaling)
    # Nomogram points: each covariate contributes |log HR| scaled so the largest effect equals 100 points per unit.
    coefs = {}
    for name, hr, lo, hi, p in vars_to_plot:
        if name == 'NetITH':
            coefs['NetITH'] = np.log(hr)
        else:
            coefs[name] = np.log(hr)

    max_abs = max(abs(v) for v in coefs.values())
    points_scale = 100 / max_abs if max_abs > 0 else 100

    table_data = [['Variable', 'β (log HR)', 'Points/Unit', 'HR']]
    for name, hr, lo, hi, p in vars_to_plot:
        beta = np.log(hr)
        pts = abs(beta) * points_scale
        table_data.append([
            name, f'{beta:.3f}', f'{pts:.1f}', f'{hr:.3f}'
        ])

    table = ax_table.table(cellText=table_data, loc='center',
                           cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    plt.tight_layout()
    out = f'{output_dir}/figures/nomogram_{cancer}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    -> Saved {out}")


def plot_calibration(cancer: str, data: pd.DataFrame, output_dir: str):
    """Plot calibration: observed vs predicted survival by risk tertile."""
    from lifelines import KaplanMeierFitter

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Split into tertiles
    # Risk tertiles define low/medium/high groups whose observed Kaplan-Meier survival is compared.
    data = data.copy()
    data['tertile'] = pd.qcut(data['risk_score'], q=3, labels=['Low', 'Medium', 'High'])
    colors = {'Low': '#1f77b4', 'Medium': '#ff7f0e', 'High': '#d62728'}

    # ---- Panel A: KM curves by tertile ----
    ax = axes[0]
    kmf = KaplanMeierFitter()
    for label in ['Low', 'Medium', 'High']:
        subset = data[data['tertile'] == label]
        kmf.fit(subset['OS.time'], subset['OS'], label=label)
        kmf.plot_survival_function(ax=ax, color=colors[label], linewidth=2)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'{cancer}: KM by Risk Tertile', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ---- Panel B: Calibration plot (observed vs expected) ----
    ax = axes[1]
    # Bin by predicted risk
    # Quintiles: mean predicted risk vs observed event rate; points near the diagonal = well calibrated.
    data['risk_bin'] = pd.qcut(data['risk_score'], q=5, labels=False)
    expected = data.groupby('risk_bin')['risk_score'].mean()
    observed = data.groupby('risk_bin')['OS'].mean()

    ax.plot(expected, observed, 'o-', color='#d62728', markersize=8,
            linewidth=2, label='Observed')
    ax.plot([expected.min(), expected.max()],
            [expected.min(), expected.max()],
            '--', color='gray', linewidth=1, label='Ideal')
    ax.set_xlabel('Mean Predicted Risk (by quintile)', fontsize=12)
    ax.set_ylabel('Observed Event Rate', fontsize=12)
    ax.set_title(f'{cancer}: Calibration', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = f'{output_dir}/figures/nomogram_calibration_{cancer}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    -> Saved {out}")


def plot_summary_figures(all_results: list, output_dir: str):
    """Create pan-cancer summary figures."""
    success = [r for r in all_results if r.get('status') == 'success']

    if not success:
        print("No successful results to plot")
        return

    # ---- Pan-cancer Forest Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(success) * 0.5)))

    # Panel A: Univariate NetITH HR
    ax = axes[0]
    cancers = [r['cancer'] for r in success]
    # Forest-style bars: univariate NetITH HR per cancer with 95% CI error bars; dashed line at HR = 1.
    uv_hrs = [r['uv_netith_hr'] for r in success]
    uv_lo = [r['uv_netith_ci_lower'] for r in success]
    uv_hi = [r['uv_netith_ci_upper'] for r in success]

    y_pos = range(len(cancers))
    colors = ['#d62728' if hr > 1 else '#1f77b4' for hr in uv_hrs]
    ax.barh(y_pos, uv_hrs, color=colors, edgecolor='black', height=0.5, alpha=0.9)
    for i, (hr, lo, hi) in enumerate(zip(uv_hrs, uv_lo, uv_hi)):
        ax.plot([lo, hi], [i, i], 'k-', linewidth=2)
        ax.plot([lo, lo], [i-0.15, i+0.15], 'k-', linewidth=1.5)
        ax.plot([hi, hi], [i-0.15, i+0.15], 'k-', linewidth=1.5)
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cancers, fontsize=10)
    ax.set_xlabel('Hazard Ratio (Univariate NetITH)', fontsize=12)
    ax.set_title('Pan-Cancer: Univariate NetITH Prognostic Value',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Panel B: C-index comparison (UV vs MV)
    ax = axes[1]
    x = np.arange(len(cancers))
    w = 0.3
    ax.bar(x - w/2, [r['uv_concordance'] for r in success], w,
           color='#1f77b4', alpha=0.8, label='Univariate (NetITH only)')
    ax.bar(x + w/2, [r['mv_concordance'] for r in success], w,
           color='#d62728', alpha=0.8, label='Multivariate (+Clinical)')
    ax.set_xticks(x)
    ax.set_xticklabels(cancers, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Concordance Index', fontsize=12)
    ax.set_title('Model Discrimination: C-Index', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/nomogram_forest.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved nomogram_forest.png")

    # ---- C-index improvement bar chart ----
    fig, ax = plt.subplots(figsize=(10, max(4, len(success) * 0.4)))
    # Delta C-index (multivariate - univariate) shows how much clinical covariates add over NetITH alone.
    improvements = [r['mv_concordance'] - r['uv_concordance'] for r in success]
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    ax.barh(range(len(cancers)), improvements, color=colors, edgecolor='black')
    ax.set_yticks(range(len(cancers)))
    ax.set_yticklabels(cancers, fontsize=11)
    ax.set_xlabel('Δ C-index (MV - UV)', fontsize=12)
    ax.set_title('C-index Improvement with Clinical Variables',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    for i, imp in enumerate(improvements):
        ax.text(imp + 0.002, i, f'{imp:+.3f}', va='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/nomogram_cindex_bar.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved nomogram_cindex_bar.png")


# ============================================================
# 4. MAIN
# ============================================================

def main():
    import os

    # Paths
    tri_modal_path = 'results/tcga/tcga_tri_modal_merged.csv'
    clinical_dir = f'{DATA_ROOT}/GraphHMM/clinical_real_tcga'
    output_dir = 'results/tcga'
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Cancers with per-cancer clinical data
    # CRC clinical data covers COAD+READ
    cancers = ['BLCA', 'BRCA', 'CRC', 'HNSC', 'KIRC', 'LIHC', 'LUAD']  # PDAC missing
    # Mapping from clinical cancer label to tri-modal cancer label(s)
    cancer_map = {'CRC': ['COAD', 'READ']}

    print("=" * 60)
    print("M: NetITH Clinical Prognostic Nomogram")
    print("=" * 60)

    # 1. Load tri-modal data
    print("\n[1] Loading tri-modal NetITH data...")
    tri = load_tri_modal(tri_modal_path)

    # 2. Process each cancer
    all_results = []
    all_patients = []

    for cancer in cancers:
        print(f"\n[2] Processing {cancer}...")

        # Load clinical
        clin = load_clinical_data(cancer, clinical_dir)
        if clin is None:
            continue

        # Derive clinical variables
        # Encode raw clinical fields (stage, T/N, grade, age, gender) as numeric Cox covariates.
        clin['stage_numeric'] = clin['ajcc_pathologic_stage'].apply(parse_ajcc_stage)
        clin['t_numeric'] = clin['ajcc_pathologic_t'].apply(parse_ajcc_t)
        clin['n_numeric'] = clin['ajcc_pathologic_n'].apply(parse_ajcc_n)
        clin['grade_numeric'] = clin['tumor_grade'].apply(parse_grade)
        clin['age'] = clin['age_at_diagnosis'] / 365.25 if 'age_at_diagnosis' in clin.columns else np.nan
        clin['gender_male'] = (clin['gender'].str.lower() == 'male').astype(float)

        # Merge with tri-modal (handle cancer name mapping)
        # CRC clinical data is analyzed jointly over COAD + READ tri-modal samples.
        if cancer in cancer_map:
            tri_labels = cancer_map[cancer]
            tri_cancer = tri[tri['cancer'].isin(tri_labels)].copy()
        else:
            tri_cancer = tri[tri['cancer'] == cancer].copy()
        merged = tri_cancer.merge(clin, on='patient', how='inner')
        print(f"  Merged: {len(merged)} samples")

        if len(merged) < 30:
            print(f"  [{cancer}] Insufficient merged samples")
            continue

        # Build nomogram
        result = build_nomogram_for_cancer(cancer, merged, output_dir)
        all_results.append(result)

        # Collect patient scores
        if 'mv_patients' in result and result['mv_patients'] is not None:
            pts = result['mv_patients'].copy()
            pts['cancer'] = cancer
            all_patients.append(pts)

    # 3. Save results
    print("\n[3] Saving results...")

    # Cox summary
    # Flatten per-cancer results into rows (drop embedded DataFrames) for the Cox summary CSV.
    result_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if not isinstance(v, pd.DataFrame)}
        result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(f'{output_dir}/nomogram_cox_results.csv', index=False)
    print(f"  -> Saved nomogram_cox_results.csv ({len(results_df)} cancers)")

    # C-index comparison
    # UV vs MV C-indices and NetITH HRs per cancer feed Supplementary Table 9 (SI-N13).
    if 'status' in results_df.columns:
        success_df = results_df[results_df['status'] == 'success']
    else:
        success_df = results_df
    cindex_cols = ['cancer', 'uv_concordance', 'mv_concordance',
                   'uv_netith_hr', 'uv_netith_p',
                   'mv_netith_hr', 'mv_netith_p', 'mv_n']
    cindex_available = [c for c in cindex_cols if c in success_df.columns]
    success_df[cindex_available].to_csv(
        f'{output_dir}/nomogram_cindex.csv', index=False)
    print(f"  -> Saved nomogram_cindex.csv")

    # Patient scores
    # Concatenate per-cancer risk scores; this table is the input to run_dca_analysis.py.
    if all_patients:
        pts_all = pd.concat(all_patients, ignore_index=True)
        pts_all.to_csv(f'{output_dir}/nomogram_patient_scores.csv', index=False)
        print(f"  -> Saved nomogram_patient_scores.csv ({len(pts_all)} patients)")

    # 4. Summary figures
    print("\n[4] Creating summary figures...")
    plot_summary_figures(all_results, output_dir)

    # 5. Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Per-Cancer Nomogram Results")
    print("=" * 60)
    success = [r for r in all_results if r.get('status') == 'success']
    if success:
        print(f"{'Cancer':<8} {'UV HR':>8} {'UV p':>8} {'UV C':>6} "
              f"{'MV HR':>8} {'MV p':>8} {'MV C':>6} {'N':>5}")
        print("-" * 65)
        for r in success:
            print(f"{r['cancer']:<8} {r['uv_netith_hr']:>8.3f} "
                  f"{r['uv_netith_p']:>8.4f} {r['uv_concordance']:>6.3f} "
                  f"{r['mv_netith_hr']:>8.3f} {r['mv_netith_p']:>8.4f} "
                  f"{r['mv_concordance']:>6.3f} {r['mv_n']:>5}")

    print("\nDone! All outputs in results/tcga/")


if __name__ == '__main__':
    main()
