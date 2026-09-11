#!/usr/bin/env python3
"""run_dca_analysis.py — decision curve analysis of NetITH nomogram net benefit.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : results/tcga/{nomogram_patient_scores.csv, tcga_tri_modal_merged.csv}; data/GraphHMM/clinical_real_tcga/{cancer}_tcga_clinical_survival.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/tcga/{dca_net_benefit.csv, dca_summary.csv}; results/tcga/figures/{dca_{cancer}.png, dca_pancancer.png}
Pipeline: clinical stage — see repository README
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import norm
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

SEED = 42
# Decision thresholds tau over [0.01, 0.50]; the manuscript summary reports tau = 0.10/0.20/0.30.
THRESHOLDS = np.linspace(0.01, 0.50, 50)  # clinical decision thresholds
OUTPUT_DIR = 'results/tcga'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_patient_scores() -> pd.DataFrame:
    """Load nomogram patient-level risk scores."""
    path = f'{OUTPUT_DIR}/nomogram_patient_scores.csv'
    df = pd.read_csv(path)
    print(f"[Load] {len(df)} patients from {df['cancer'].nunique()} cancers")
    return df


def load_tri_modal() -> pd.DataFrame:
    """Load tri-modal NetITH data with clinical variables."""
    path = f'{OUTPUT_DIR}/tcga_tri_modal_merged.csv'
    df = pd.read_csv(path)
    # Derive patient ID
    # TCGA sample barcodes (TCGA.XX.AAAA.01) -> patient IDs (TCGA-XX-AAAA) for merging with clinical tables.
    def sample_to_patient(s):
        parts = s.split('.')
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        return None
    df['patient'] = df['sample'].apply(sample_to_patient)
    df = df.dropna(subset=['patient'])
    return df


def load_clinical_data(cancer: str, base_dir: str) -> pd.DataFrame:
    """Load per-cancer clinical data."""
    path = f"{base_dir}/{cancer}_tcga_clinical_survival.csv"
    try:
        df = pd.read_csv(path)
        # Alias submitter_id to patient so clinical rows merge with NetITH rows on patient ID.
        df['patient'] = df['submitter_id']
        return df
    except FileNotFoundError:
        return None


def parse_ajcc_stage(stage_str):
    if pd.isna(stage_str) or not isinstance(stage_str, str):
        return np.nan
    s = stage_str.strip().lower()
    if 'stage x' in s or 'not reported' in s or s == '':
        return np.nan
    # AJCC roman numerals I-IV -> 1-4; 'Stage X' / 'not reported' already mapped to NaN above.
    for roman, arabic in [('iv', 4), ('iii', 3), ('ii', 2), ('i', 1)]:
        if roman in s:
            return float(arabic)
    return np.nan


# ============================================================
# 2. BUILD CLINICAL-ONLY MODEL (for comparison)
# ============================================================

def build_clinical_only_risk(cancer: str, df: pd.DataFrame,
                              clinical_dir: str) -> pd.DataFrame:
    """Build a clinical-only Cox model for DCA comparison."""
    from lifelines import CoxPHFitter

    clin = load_clinical_data(cancer, clinical_dir)
    if clin is None:
        return None

    # Derive clinical vars
    clin['stage_numeric'] = clin['ajcc_pathologic_stage'].apply(parse_ajcc_stage)
    clin['age'] = clin['age_at_diagnosis'] / 365.25 if 'age_at_diagnosis' in clin.columns else np.nan
    clin['gender_male'] = (clin['gender'].str.lower() == 'male').astype(float)

    # Inner-join NetITH patients with clinical records; too few matched patients -> no clinical model.
    merged = df.merge(clin, on='patient', how='inner')
    if len(merged) < 50:
        return None

    clinical_vars = []
    # Keep clinical covariates (age, AJCC stage, gender) only if >= 20 non-missing values.
    for col in ['age', 'stage_numeric', 'gender_male']:
        if col in merged.columns and merged[col].notna().sum() >= 20:
            clinical_vars.append(col)

    mv_cols = ['OS.time', 'OS'] + clinical_vars
    mv = merged[mv_cols].dropna()
    mv = mv[mv['OS.time'] > 0].copy()

    if len(mv) < 30:
        return None

    try:
        # Clinical-only reference model: ridge-penalized Cox PH on clinical covariates alone; risk = partial hazard.
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(mv[['OS.time', 'OS'] + clinical_vars],
                duration_col='OS.time', event_col='OS')
        mv['risk_clinical'] = cph.predict_partial_hazard(mv)
        mv['patient'] = merged.loc[mv.index, 'patient']
        return mv[['patient', 'risk_clinical']]
    except Exception as e:
        print(f"  [{cancer}] Clinical-only Cox failed: {e}")
        return None


# ============================================================
# 3. DECISION CURVE ANALYSIS
# ============================================================

def compute_net_benefit(risk: np.ndarray, event: np.ndarray,
                         time: np.ndarray,
                         thresholds: np.ndarray,
                         time_horizon: int = 1825) -> np.ndarray:
    """Compute net benefit across thresholds.
    
    Uses 5-year (1825 day) survival as the clinical endpoint.
    For each threshold τ, classify high-risk = risk > percentile(1-τ),
    then compute NB = (TP_rate - w * FP_rate).
    
    Args:
        risk: predicted risk scores (higher = higher risk)
        event: 1/0 death event indicator
        time: overall survival time in days
        thresholds: array of decision thresholds
        time_horizon: days for defining "event" (default 5yr)
    
    Returns:
        nb: net benefit at each threshold
    """
    # Define 5-year event: died within 5 years = 1, otherwise 0
    event_5yr = ((event == 1) & (time <= time_horizon)).astype(int)
    # Censored before 5yr but alive at last FU: exclude from analysis
    # (conservative: treat as non-event)
    
    n = len(risk)
    # Vickers & Elkin DCA: NB(tau) = (TP - w*FP) / N with w = tau/(1-tau); treat-none has NB = 0.
    nb = np.full(len(thresholds), np.nan)
    
    for i, tau in enumerate(thresholds):
        # Classify: risk above (1-τ) quantile = "test positive"
        cutoff = np.quantile(risk, 1 - tau)
        predicted_positive = risk >= cutoff
        
        tp = np.sum(predicted_positive & (event_5yr == 1))
        fp = np.sum(predicted_positive & (event_5yr == 0))
        
        if np.sum(predicted_positive) == 0:
            nb[i] = 0.0
            continue
        
        w = tau / (1 - tau)  # odds at threshold
        nb[i] = (tp - w * fp) / n
    
    return nb


def compute_treat_all_nb(event: np.ndarray, time: np.ndarray,
                          thresholds: np.ndarray,
                          time_horizon: int = 1825) -> np.ndarray:
    """Net benefit of treating all patients (= event rate - w)."""
    event_5yr = ((event == 1) & (time <= time_horizon)).astype(int)
    event_rate = event_5yr.mean()
    n = len(event)
    
    # Treat-all baseline: everyone is positive, so NB(tau) = event_rate - w * (1 - event_rate).
    nb = np.zeros(len(thresholds))
    for i, tau in enumerate(thresholds):
        w = tau / (1 - tau)
        # Treat all: TP_rate = event_rate, FP_rate = 1 - event_rate
        nb[i] = event_rate - w * (1 - event_rate)
    
    return nb


def run_dca_for_cancer(cancer: str, df_patients: pd.DataFrame,
                        df_tri: pd.DataFrame,
                        clinical_dir: str) -> dict:
    """Run full DCA for one cancer type."""
    from lifelines import CoxPHFitter
    
    # Get this cancer's patients
    pts = df_patients[df_patients['cancer'] == cancer].copy()
    print(f"\n[DCA] {cancer}: {len(pts)} patients, "
          f"events={pts['OS'].sum():.0f}")
    
    # Require >= 30 patients and >= 5 events per cancer for stable DCA curves.
    if len(pts) < 30 or pts['OS'].sum() < 5:
        print(f"  Insufficient data, skipping")
        return {'cancer': cancer, 'status': 'insufficient'}
    
    event = pts['OS'].values
    time = pts['OS.time'].values
    
    # 1. NetITH-only risk (univariate Cox)
    # Univariate Cox refit on the nomogram risk score; falls back to the raw score if the fit fails.
    try:
        cph_uv = CoxPHFitter(penalizer=0.1)
        uv_data = pts[['OS.time', 'OS', 'risk_score']].dropna()
        cph_uv.fit(uv_data[['OS.time', 'OS', 'risk_score']],
                   duration_col='OS.time', event_col='OS')
        risk_netith = cph_uv.predict_partial_hazard(uv_data)
    except Exception:
        risk_netith = pts['risk_score'].values
    
    # 2. Clinical-only risk
    tri_cancer = df_tri[df_tri['cancer'] == cancer] if 'cancer' in df_tri.columns else df_tri
    clin_risk_df = build_clinical_only_risk(cancer, tri_cancer, clinical_dir)
    
    if clin_risk_df is not None:
        pts_clin = pts[['patient', 'OS.time', 'OS']].merge(
            clin_risk_df, on='patient', how='inner')
        risk_clinical = pts_clin['risk_clinical'].values
    else:
        risk_clinical = None
    
    # 3. NetITH + Clinical (full nomogram risk = existing risk_score)
    # Full-model risk = the multivariate nomogram partial hazard already saved by run_nomogram.py.
    risk_full = pts['risk_score'].values
    
    # 4. Compute net benefits
    nb_netith = compute_net_benefit(risk_netith, event, time, THRESHOLDS)
    nb_full = compute_net_benefit(risk_full, event, time, THRESHOLDS)
    nb_treat_all = compute_treat_all_nb(event, time, THRESHOLDS)
    nb_treat_none = np.zeros(len(THRESHOLDS))
    
    if risk_clinical is not None:
        nb_clinical = compute_net_benefit(risk_clinical, 
                                           pts_clin['OS'].values,
                                           pts_clin['OS.time'].values,
                                           THRESHOLDS)
    else:
        nb_clinical = None
    
    # 5. Clinical utility range: where NetITH+Clinical > max(Clinical, TreatAll)
    if nb_clinical is not None:
        ref = np.maximum(nb_clinical, nb_treat_all)
    else:
        ref = nb_treat_all
    
    # Clinical utility = thresholds where the full model beats the best available reference.
    utility_mask = nb_full > ref
    utility_range = (THRESHOLDS[utility_mask].min() if utility_mask.any() else None,
                     THRESHOLDS[utility_mask].max() if utility_mask.any() else None)
    
    print(f"  Clinical utility range: {utility_range}")
    
    # 6. Store results
    result = {
        'cancer': cancer,
        'n': len(pts),
        'n_events': int(pts['OS'].sum()),
        'thresholds': THRESHOLDS.tolist(),
        'nb_netith': nb_netith.tolist(),
        'nb_full': nb_full.tolist(),
        'nb_treat_all': nb_treat_all.tolist(),
        'nb_treat_none': nb_treat_none.tolist(),
        'nb_clinical': nb_clinical.tolist() if nb_clinical is not None else None,
        'utility_min': utility_range[0],
        'utility_max': utility_range[1],
        'status': 'success',
    }
    
    # Also store as wide-format for CSV
    rows = []
    for i, tau in enumerate(THRESHOLDS):
        row = {
            'cancer': cancer,
            'threshold': round(tau, 3),
            'nb_treat_all': round(nb_treat_all[i], 6),
            'nb_netith': round(nb_netith[i], 6),
            'nb_full': round(nb_full[i], 6),
        }
        if nb_clinical is not None:
            row['nb_clinical'] = round(nb_clinical[i], 6)
        else:
            row['nb_clinical'] = np.nan
        rows.append(row)
    
    result['df_rows'] = rows
    
    return result


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_dca_single(cancer: str, result: dict, output_dir: str):
    """Plot DCA curve for a single cancer."""
    thresholds = np.array(result['thresholds'])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Reference lines
    # Treat-all and treat-none define the decision boundaries every model curve is judged against.
    ax.plot(thresholds, result['nb_treat_all'], 'k--', linewidth=1.5,
            label='Treat All', alpha=0.7)
    ax.plot(thresholds, result['nb_treat_none'], 'k:', linewidth=1.5,
            label='Treat None', alpha=0.7)
    
    # Model curves
    ax.plot(thresholds, result['nb_netith'], color='#1f77b4', linewidth=2.5,
            label='NetITH-only')
    
    if result['nb_clinical'] is not None:
        ax.plot(thresholds, result['nb_clinical'], color='#ff7f0e', linewidth=2.5,
                label='Clinical-only')
    
    ax.plot(thresholds, result['nb_full'], color='#d62728', linewidth=3,
            label='NetITH + Clinical (Nomogram)')
    
    # Highlight clinical utility region
    umin, umax = result['utility_min'], result['utility_max']
    if umin is not None and umax is not None and umin < umax:
        ax.axvspan(umin, umax, alpha=0.1, color='#2ca02c')
        ax.text((umin + umax) / 2, ax.get_ylim()[1] * 0.95,
                'Clinical\nUtility', ha='center', va='top',
                fontsize=11, fontweight='bold', color='#2ca02c',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Threshold Probability', fontsize=13)
    ax.set_ylabel('Net Benefit', fontsize=13)
    ax.set_title(f'{cancer}: Decision Curve Analysis\n'
                 f'(n={result["n"]}, events={result["n_events"]})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, 0.50)
    ax.grid(alpha=0.2)
    
    # Set ylim to reasonable range
    y_max = max(
        np.nanmax(result['nb_full']),
        np.nanmax(result['nb_treat_all']),
        0.05
    )
    y_min = min(
        np.nanmin(result['nb_full']),
        np.nanmin(result['nb_netith']),
        -0.02
    )
    ax.set_ylim(y_min - 0.01, y_max + 0.02)
    
    plt.tight_layout()
    out = f'{output_dir}/figures/dca_{cancer}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


def plot_dca_pancancer(results: list, output_dir: str):
    """Pan-cancer DCA summary: Net benefit difference at key thresholds."""
    success = [r for r in results if r.get('status') == 'success']
    if len(success) < 2:
        print("Not enough cancer types for pan-cancer DCA")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    
    # Panel A: DCA curves overlaid for all cancers (full model only)
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(success)))
    for i, r in enumerate(success):
        thresholds = np.array(r['thresholds'])
        ax.plot(thresholds, r['nb_full'], color=colors[i], linewidth=2,
                label=f"{r['cancer']} (n={r['n']})")
    
    ax.plot(thresholds, success[0]['nb_treat_all'], 'k--', linewidth=2,
            label='Treat All', alpha=0.6)
    ax.plot(thresholds, success[0]['nb_treat_none'], 'k:', linewidth=2,
            label='Treat None', alpha=0.6)
    
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('A: Pan-Cancer Decision Curves (Full Nomogram)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_xlim(0, 0.50)
    ax.grid(alpha=0.2)
    
    # Panel B: ΔNet Benefit (Full - Clinical) at τ=0.10, 0.20, 0.30
    ax = axes[0, 1]
    # Manuscript thresholds for the incremental-net-benefit comparison (Supplementary Table 20, tau = 0.10-0.30).
    tau_points = [0.10, 0.20, 0.30]
    x = np.arange(len(success))
    width = 0.25
    
    for j, tau in enumerate(tau_points):
        deltas = []
        for r in success:
            idx = np.argmin(np.abs(np.array(r['thresholds']) - tau))
            nb_full = r['nb_full'][idx]
            nb_clin = r['nb_clinical'][idx] if r['nb_clinical'] is not None else r['nb_treat_all'][idx]
            deltas.append(nb_full - nb_clin)
        
        bars = ax.bar(x + j * width, deltas, width,
                      label=f'τ={tau:.2f}',
                      color=['#1f77b4', '#ff7f0e', '#d62728'][j],
                      edgecolor='white', alpha=0.85)
        
        # Value labels
        for bi, d in enumerate(deltas):
            if d > 0:
                ax.text(x[bi] + j * width, d + 0.001,
                        f'{d:+.3f}', ha='center', fontsize=7,
                        fontweight='bold' if d > 0.005 else 'normal')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([r['cancer'] for r in success], fontsize=9)
    ax.set_ylabel('Δ Net Benefit (Full - Clinical)', fontsize=12)
    ax.set_title('B: Incremental Net Benefit of Adding NetITH',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    
    # Panel C: Clinical utility range (thresholds where NetITH adds value)
    ax = axes[1, 0]
    y_pos = range(len(success))
    for i, r in enumerate(success):
        umin, umax = r.get('utility_min'), r.get('utility_max')
        if umin is not None and umax is not None and not np.isnan(umin) and not np.isnan(umax):
            ax.barh(i, umax - umin, left=umin, color='#2ca02c', alpha=0.7,
                    height=0.6,
                    label='Clinical Utility Range' if i == 0 else '')
            ax.text(umax + 0.01, i,
                    f'[{umin:.2f}, {umax:.2f}]',
                    va='center', fontsize=9)
        else:
            ax.text(0.25, i, 'No utility range',
                    va='center', fontsize=9, color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['cancer'] for r in success], fontsize=10)
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_title('C: Clinical Utility Range (NetITH > Reference)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.2)
    
    # Panel D: Net benefit at τ=0.20 (common screening threshold)
    ax = axes[1, 1]
    tau = 0.20
    models = ['NetITH-only', 'Clinical-only', 'Full Nomogram']
    x = np.arange(len(success))
    width = 0.25
    
    for j, (model_name, color) in enumerate(zip(
        models, ['#1f77b4', '#ff7f0e', '#d62728'])):
        nbs = []
        for r in success:
            idx = np.argmin(np.abs(np.array(r['thresholds']) - tau))
            if model_name == 'NetITH-only':
                nbs.append(r['nb_netith'][idx])
            elif model_name == 'Clinical-only':
                nb = r['nb_clinical'][idx] if r['nb_clinical'] is not None else np.nan
                nbs.append(nb)
            else:
                nbs.append(r['nb_full'][idx])
        
        ax.bar(x + j * width, nbs, width, label=model_name,
               color=color, edgecolor='white', alpha=0.85)
    
    # Treat All reference
    treat_all_nb = success[0]['nb_treat_all'][
        np.argmin(np.abs(np.array(success[0]['thresholds']) - tau))
    ]
    ax.axhline(y=treat_all_nb, color='gray', linestyle='--', linewidth=1.5,
               label=f'Treat All (NB={treat_all_nb:.3f})')
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([r['cancer'] for r in success], fontsize=9)
    ax.set_ylabel(f'Net Benefit at τ={tau:.2f}', fontsize=12)
    ax.set_title(f'D: Model Comparison at τ={tau:.2f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    out = f'{output_dir}/figures/dca_pancancer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DCA: Decision Curve Analysis for NetITH Nomogram")
    print("=" * 60)
    
    clinical_dir = '{DATA_ROOT}/GraphHMM/clinical_real_tcga'
    
    # 1. Load data
    print("\n[1] Loading data...")
    # Nomogram risk scores (run_nomogram.py) and the tri-modal NetITH table feed all per-cancer DCAs.
    df_patients = load_patient_scores()
    df_tri = load_tri_modal()
    
    cancers = sorted(df_patients['cancer'].unique())
    print(f"  Cancers: {cancers}")
    
    # 2. Run DCA per cancer
    all_results = []
    all_rows = []
    
    for cancer in cancers:
        result = run_dca_for_cancer(cancer, df_patients, df_tri, clinical_dir)
        all_results.append(result)
        
        if 'df_rows' in result:
            all_rows.extend(result['df_rows'])
        
        if result.get('status') == 'success':
            plot_dca_single(cancer, result, OUTPUT_DIR)
    
    # 3. Save results
    print("\n[3] Saving results...")
    df_nb = pd.DataFrame(all_rows)
    df_nb.to_csv(f'{OUTPUT_DIR}/dca_net_benefit.csv', index=False)
    print(f"  -> Saved dca_net_benefit.csv ({len(df_nb)} rows)")
    
    # Summary
    # Headline numbers at tau = 0.10/0.20/0.30: NB of the full model and deltas vs clinical-only / treat-all (Supplementary Table 20).
    summary_rows = []
    for r in all_results:
        if r.get('status') == 'success':
            # Compute NB improvement at key thresholds
            tau_10 = np.argmin(np.abs(np.array(r['thresholds']) - 0.10))
            tau_20 = np.argmin(np.abs(np.array(r['thresholds']) - 0.20))
            tau_30 = np.argmin(np.abs(np.array(r['thresholds']) - 0.30))
            
            nb_clin_20 = r['nb_clinical'][tau_20] if r['nb_clinical'] is not None else np.nan
            summary_rows.append({
                'cancer': r['cancer'],
                'n': r['n'],
                'n_events': r['n_events'],
                'nb_full_tau10': r['nb_full'][tau_10],
                'nb_full_tau20': r['nb_full'][tau_20],
                'nb_full_tau30': r['nb_full'][tau_30],
                'delta_vs_clinical_tau20': (r['nb_full'][tau_20] - nb_clin_20
                                            if not np.isnan(nb_clin_20) else np.nan),
                'delta_vs_treatall_tau20': r['nb_full'][tau_20] - r['nb_treat_all'][tau_20],
                'utility_min': r.get('utility_min'),
                'utility_max': r.get('utility_max'),
            })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{OUTPUT_DIR}/dca_summary.csv', index=False)
    print(f"  -> Saved dca_summary.csv ({len(df_summary)} cancers)")
    
    # 4. Pan-cancer figure
    print("\n[4] Creating pan-cancer DCA summary...")
    plot_dca_pancancer(all_results, OUTPUT_DIR)
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("DCA SUMMARY: Net Benefit at τ=0.20")
    print("=" * 60)
    success = [r for r in all_results if r.get('status') == 'success']
    for r in success:
        idx = np.argmin(np.abs(np.array(r['thresholds']) - 0.20))
        nb_full = r['nb_full'][idx]
        nb_ta = r['nb_treat_all'][idx]
        nb_clin = r['nb_clinical'][idx] if r['nb_clinical'] is not None else np.nan
        
        delta_ta = nb_full - nb_ta
        delta_s = '✓ clinical utility' if delta_ta > 0 else '× no benefit'
        
        print(f"  {r['cancer']:<8} NB_full={nb_full:.4f}  "
              f"NB_treatall={nb_ta:.4f}  "
              f"Δ={delta_ta:+.4f}  {delta_s}")
    
    print("\nDone! All outputs in results/tcga/")


if __name__ == '__main__':
    main()
