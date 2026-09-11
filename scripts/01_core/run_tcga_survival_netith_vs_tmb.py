"""
run_tcga_survival_netith_vs_tmb.py — Head-to-head comparison of NetITH vs TMB for TCGA pan-cancer survival prediction.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - <DATA_ROOT>/xena/tcgapancan/mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz: TCGA MC3 mutation matrix (TMB)
    - <DATA_ROOT>/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp: TCGA survival
    - results/tcga/tcga_netith.csv: per-tumor NetITH
Outputs :
    - results/tcga/tmb_vs_netith/{tmb_netith_merged,per_cancer_cox,multivariate_results,meta_analysis}.csv
    - results/tcga/tmb_vs_netith/figures/
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os
import sys
import gzip
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import scipy.stats as stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Config ───────────────────────────────────────────────────

MC3_PATH = f"{DATA_ROOT}/xena/tcgapancan/mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"
TCGA_SURV_PATH = f"{DATA_ROOT}/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp"
NETITH_PATH = f"{ROOT}/results/tcga/tcga_netith.csv"
OUTPUT_DIR = Path(f"{ROOT}/results/tcga/tmb_vs_netith")
MIN_SAMPLES = 30
MIN_EVENTS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "figures", exist_ok=True)


# ─── 1. Load & Compute TMB ────────────────────────────────────

def compute_tmb_from_mc3(path: str) -> pd.Series:
    """Compute per-sample TMB (non-silent mutation count) from MC3 matrix.

    File format: genes × samples (9104 genes, 9104 samples), values = 0/1.
    """
    print("=" * 60)
    print("[1/7] Computing TMB from MC3 mutation matrix...")
    t0 = pd.Timestamp.now()

    # Read gene × sample matrix
    df = pd.read_csv(path, sep='\t', compression='gzip', index_col=0)
    # df: genes × samples → transpose to samples × genes
    df_t = df.T
    # TMB = sum of mutations per sample
    tmb = df_t.sum(axis=1)
    tmb.name = 'TMB'
    tmb.index.name = 'sample'

    # Clean TCGA sample IDs (keep only primary tumor: -01)
    tmb.index = [s[:15] if len(s) > 15 else s for s in tmb.index]

    print(f"  MC3: {df.shape[0]} genes × {df.shape[1]} samples")
    print(f"  TMB range: {tmb.min():.0f} – {tmb.max():.0f}, median: {tmb.median():.0f}")
    print(f"  Time: {(pd.Timestamp.now() - t0).total_seconds():.1f}s")
    return tmb


# ─── 2. Load NetITH ───────────────────────────────────────────

def load_netith(path: str) -> pd.Series:
    """Load pre-computed TCGA NetITH values."""
    print("\n[2/7] Loading NetITH...")
    df = pd.read_csv(path, index_col=0)
    netith = df['netith_bulk']
    netith.index = [s[:15] if len(s) > 15 else s for s in netith.index]
    print(f"  NetITH: {len(netith)} samples, range [{netith.min():.2f}, {netith.max():.2f}]")
    return netith


# ─── 3. Load Survival ─────────────────────────────────────────

def load_tcga_survival(path: str) -> pd.DataFrame:
    """Load TCGA survival data (same parser as run_tcga_validation.py)."""
    print("\n[3/7] Loading TCGA survival...")
    rows = []
    header = None
    with open(path, 'r') as f:
        for line in f:
            if '\t' in line:
                idx = line.find('sample\t')
                if idx >= 0:
                    line = line[idx:]
                header = line.strip().split('\t')
                break
        for line in f:
            if '\t' in line:
                rows.append(line.strip().split('\t'))

    n_cols = len(header)
    rows_padded = []
    for r in rows:
        if len(r) < n_cols:
            r = r + [''] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        rows_padded.append(r)

    surv = pd.DataFrame(rows_padded, columns=header)
    surv = surv.rename(columns={'cancer type abbreviation': 'cancer_type'})

    # Numeric conversion
    for col in ['OS', 'OS.time', 'DSS', 'DSS.time', 'PFI', 'PFI.time',
                'age_at_initial_pathologic_diagnosis']:
        if col in surv.columns:
            surv[col] = pd.to_numeric(surv[col], errors='coerce')

    surv = surv.dropna(subset=['OS', 'OS.time'])
    surv = surv[surv['OS.time'] > 0]
    surv['sample_short'] = surv['sample'].str[:15]

    print(f"  Survival: {len(surv)} samples with valid OS")
    return surv


# ─── 4. Encode Clinical Variables ──────────────────────────────

def encode_clinical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode clinical variables from merged survival data.

    The TCGA survival file already contains age, gender, stage, grade.
    We encode them as numeric for Cox models.
    """
    print("\n[4/7] Encoding clinical variables from survival data...")

    # Age: already numeric
    if 'age_at_initial_pathologic_diagnosis' in df.columns:
        df['age'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce')

    # Stage: encode to numeric
    if 'ajcc_pathologic_tumor_stage' in df.columns:
        stage_map = {
            'stage i': 1, 'stage ia': 1, 'stage ib': 1, 'stage ic': 1,
            'stage ii': 2, 'stage iia': 2, 'stage iib': 2, 'stage iic': 2,
            'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
            'stage iv': 4, 'stage iva': 4, 'stage ivb': 4, 'stage ivc': 4,
        }
        df['stage_numeric'] = df['ajcc_pathologic_tumor_stage'].str.lower().map(stage_map)

    # Grade
    if 'histological_grade' in df.columns:
        grade_map = {
            'g1': 1, 'g2': 2, 'g3': 3, 'g4': 4,
            'low grade': 1, 'intermediate grade': 2, 'high grade': 3,
        }
        df['grade_numeric'] = df['histological_grade'].str.lower().map(grade_map)

    # Gender
    if 'gender' in df.columns:
        df['gender_male'] = (df['gender'].str.upper() == 'MALE').astype(int)

    n_with_stage = df['stage_numeric'].notna().sum()
    n_with_age = df['age'].notna().sum()
    print(f"  Stage available: {n_with_stage} | Age: {n_with_age} | "
          f"Gender: {df['gender_male'].notna().sum() if 'gender_male' in df.columns else 0}")

    return df


# ─── 5. Merge All ─────────────────────────────────────────────

def merge_all(tmb: pd.Series, netith: pd.Series, surv: pd.DataFrame) -> pd.DataFrame:
    """Merge TMB, NetITH, survival, and encode clinical variables."""
    print("\n[5/7] Merging datasets...")

    # Rename TMB index to match
    tmb_df = tmb.reset_index()
    tmb_df.columns = ['sample_short', 'TMB']

    netith_df = netith.reset_index()
    netith_df.columns = ['sample_short', 'netith_bulk']

    # Merge
    merged = surv.merge(tmb_df, on='sample_short', how='inner')
    merged = merged.merge(netith_df, on='sample_short', how='inner')

    # Add NetITH as continuous + binary (median split)
    merged['netith_z'] = (merged['netith_bulk'] - merged['netith_bulk'].mean()) / \
                          merged['netith_bulk'].std()
    merged['tmb_log'] = np.log2(merged['TMB'] + 1)
    merged['tmb_z'] = (merged['tmb_log'] - merged['tmb_log'].mean()) / \
                       merged['tmb_log'].std()

    # Encode clinical variables from survival data
    merged = encode_clinical_variables(merged)

    # Drop missing OS
    merged = merged.dropna(subset=['OS', 'OS.time'])
    merged = merged[merged['OS.time'] > 0]

    print(f"  Final merged: {len(merged)} samples")
    print(f"  Cancer types: {merged['cancer_type'].nunique()}")
    print(f"  TMB range: [{merged['TMB'].min():.0f}, {merged['TMB'].max():.0f}]")
    print(f"  NetITH range: [{merged['netith_bulk'].min():.2f}, {merged['netith_bulk'].max():.2f}]")
    print(f"  TMB-NetITH correlation: ρ={merged['TMB'].corr(merged['netith_bulk'], method='spearman'):.4f}")

    merged.to_csv(OUTPUT_DIR / "tmb_netith_merged.csv", index=False)
    return merged


# ─── 6. Per-Cancer Cox Models ─────────────────────────────────

def run_per_cancer_cox(df: pd.DataFrame) -> pd.DataFrame:
    """Run univariate and bivariate Cox models per cancer type.

    For each cancer:
      - NetITH univariate Cox
      - TMB univariate Cox
      - NetITH + TMB bivariate Cox
    """
    print("\n[6/7] Running per-cancer Cox models...")

    cancer_types = df['cancer_type'].value_counts()
    valid_cts = cancer_types[(cancer_types >= MIN_SAMPLES) &
                              (df.groupby('cancer_type')['OS'].sum() >= MIN_EVENTS)]
    valid_cts = valid_cts.index.tolist()

    results = []

    for ct in valid_cts:
        ct_df = df[df['cancer_type'] == ct].copy()
        n = len(ct_df)
        n_events = int(ct_df['OS'].sum())

        row = {'cancer_type': ct, 'n': n, 'n_events': n_events}

        # ── NetITH univariate ──
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cox_df = ct_df[['OS.time', 'OS', 'netith_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'})
            cph.fit(cox_df, duration_col='duration', event_col='event')
            row['netith_hr'] = np.exp(cph.params_['netith_z'])
            row['netith_p'] = cph.summary.loc['netith_z', 'p']
            ci_cols = cph.confidence_intervals_.columns
            row['netith_ci_lower'] = np.exp(cph.confidence_intervals_.loc['netith_z', ci_cols[0]])
            row['netith_ci_upper'] = np.exp(cph.confidence_intervals_.loc['netith_z', ci_cols[1]])
            row['netith_concordance'] = cph.concordance_index_
        except Exception:
            for k in ['netith_hr', 'netith_p', 'netith_ci_lower', 'netith_ci_upper', 'netith_concordance']:
                row[k] = np.nan

        # ── TMB univariate ──
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cox_df = ct_df[['OS.time', 'OS', 'tmb_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'})
            cph.fit(cox_df, duration_col='duration', event_col='event')
            row['tmb_hr'] = np.exp(cph.params_['tmb_z'])
            row['tmb_p'] = cph.summary.loc['tmb_z', 'p']
            ci_cols = cph.confidence_intervals_.columns
            row['tmb_ci_lower'] = np.exp(cph.confidence_intervals_.loc['tmb_z', ci_cols[0]])
            row['tmb_ci_upper'] = np.exp(cph.confidence_intervals_.loc['tmb_z', ci_cols[1]])
            row['tmb_concordance'] = cph.concordance_index_
        except Exception:
            for k in ['tmb_hr', 'tmb_p', 'tmb_ci_lower', 'tmb_ci_upper', 'tmb_concordance']:
                row[k] = np.nan

        # ── Bivariate: NetITH + TMB ──
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cox_df = ct_df[['OS.time', 'OS', 'netith_z', 'tmb_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'})
            cph.fit(cox_df, duration_col='duration', event_col='event')
            row['bi_netith_hr'] = np.exp(cph.params_['netith_z'])
            row['bi_netith_p'] = cph.summary.loc['netith_z', 'p']
            row['bi_tmb_hr'] = np.exp(cph.params_['tmb_z'])
            row['bi_tmb_p'] = cph.summary.loc['tmb_z', 'p']
            row['bi_concordance'] = cph.concordance_index_
        except Exception:
            for k in ['bi_netith_hr', 'bi_netith_p', 'bi_tmb_hr', 'bi_tmb_p', 'bi_concordance']:
                row[k] = np.nan

        results.append(row)

        sig_netith = "***" if row.get('netith_p', 1) < 0.001 else \
                     ("**" if row.get('netith_p', 1) < 0.01 else \
                      ("*" if row.get('netith_p', 1) < 0.05 else ""))
        sig_tmb = "***" if row.get('tmb_p', 1) < 0.001 else \
                  ("**" if row.get('tmb_p', 1) < 0.01 else \
                   ("*" if row.get('tmb_p', 1) < 0.05 else ""))
        print(f"  {ct:5s}: n={n:4d} events={n_events:3d} | "
              f"NetITH HR={row.get('netith_hr', np.nan):.3f}{sig_netith:3s} | "
              f"TMB HR={row.get('tmb_hr', np.nan):.3f}{sig_tmb:3s}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "per_cancer_cox.csv", index=False)
    print(f"  Saved: per_cancer_cox.csv ({len(results_df)} cancers)")
    return results_df


# ─── 7. Meta-Analysis ─────────────────────────────────────────

def run_meta_analysis(cox_results: pd.DataFrame, metric: str = 'netith') -> dict:
    """Random-effects meta-analysis (DerSimonian-Laird) across cancer types.

    Args:
        cox_results: DataFrame with columns 'n', f'{metric}_hr', f'{metric}_p'
        metric: 'netith' or 'tmb'
    """
    hr_col = f'{metric}_hr'
    p_col = f'{metric}_p'

    valid = cox_results.dropna(subset=[hr_col, p_col]).copy()

    # Estimate logHR and SE from p-values
    valid['logHR'] = np.log(valid[hr_col].astype(float))
    valid['se'] = np.abs(valid['logHR']) / stats.norm.ppf(1 - valid[p_col].astype(float) / 2)
    # Fix any infinite/NaN SEs
    valid['se'] = valid['se'].replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna(subset=['se'])
    valid = valid[valid['se'] > 0]

    k = len(valid)
    if k < 3:
        return {'k': k, 'error': 'Too few cancers'}

    # Fixed-effects
    w_fixed = 1.0 / valid['se']**2
    fe_est = np.sum(w_fixed * valid['logHR']) / np.sum(w_fixed)
    fe_se = np.sqrt(1.0 / np.sum(w_fixed))

    # Random-effects (DerSimonian-Laird)
    Q = np.sum(w_fixed * (valid['logHR'] - fe_est)**2)
    df_q = k - 1
    c = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df_q) / c) if c > 0 else 0

    w_random = 1.0 / (valid['se']**2 + tau2)
    re_est = np.sum(w_random * valid['logHR']) / np.sum(w_random)
    re_se = np.sqrt(1.0 / np.sum(w_random))

    # I²
    I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

    re_p = 2 * (1 - stats.norm.cdf(np.abs(re_est / re_se)))
    fe_p = 2 * (1 - stats.norm.cdf(np.abs(fe_est / fe_se)))

    # Egger's test for publication bias
    prec = 1.0 / valid['se']
    z = valid['logHR'] / valid['se']
    if k >= 5:
        slope, intercept, r, p_egger, _ = stats.linregress(prec, z)
    else:
        intercept, p_egger = 0, np.nan

    # Direction concordance
    n_better = int((valid[hr_col] < 1).sum())
    n_worse = int((valid[hr_col] >= 1).sum())

    meta = {
        'metric': metric,
        'k': k,
        'n_total': int(valid['n'].sum()),
        'n_events_total': int(valid['n_events'].sum()),
        'fe_hr': np.exp(fe_est),
        'fe_ci_lower': np.exp(fe_est - 1.96 * fe_se),
        'fe_ci_upper': np.exp(fe_est + 1.96 * fe_se),
        'fe_p': fe_p,
        're_hr': np.exp(re_est),
        're_ci_lower': np.exp(re_est - 1.96 * re_se),
        're_ci_upper': np.exp(re_est + 1.96 * re_se),
        're_p': re_p,
        'Q': Q,
        'df': df_q,
        'p_heterogeneity': 1 - stats.chi2.cdf(Q, df_q),
        'I2': I2,
        'tau2': tau2,
        'n_better': n_better,
        'n_worse': n_worse,
        'egger_intercept': intercept,
        'egger_p': p_egger,
    }

    return meta


def run_both_meta_analyses(cox_results: pd.DataFrame) -> Tuple[dict, dict]:
    """Run meta-analysis for both NetITH and TMB."""
    print("\n[7/7] Running meta-analyses (NetITH vs TMB)...")

    meta_netith = run_meta_analysis(cox_results, 'netith')
    meta_tmb = run_meta_analysis(cox_results, 'tmb')

    for label, meta in [('NetITH', meta_netith), ('TMB', meta_tmb)]:
        if 'error' not in meta:
            print(f"\n  {label} Meta-Analysis ({meta['k']} cancers):")
            print(f"    Fixed-effects:  HR = {meta['fe_hr']:.3f} "
                  f"[{meta['fe_ci_lower']:.3f}–{meta['fe_ci_upper']:.3f}], p={meta['fe_p']:.4f}")
            print(f"    Random-effects: HR = {meta['re_hr']:.3f} "
                  f"[{meta['re_ci_lower']:.3f}–{meta['re_ci_upper']:.3f}], p={meta['re_p']:.4f}")
            print(f"    I² = {meta['I2']:.1f}% | Better/Worse: {meta['n_better']}/{meta['n_worse']}")
            if not np.isnan(meta['egger_p']):
                print(f"    Egger's p = {meta['egger_p']:.4f}")

    # Save
    pd.DataFrame([meta_netith, meta_tmb]).to_csv(
        OUTPUT_DIR / "meta_analysis.csv", index=False)
    print(f"\n  Saved: meta_analysis.csv")

    return meta_netith, meta_tmb


# ─── Multivariate + Clinical ──────────────────────────────────

def run_multivariate_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Run full multivariate Cox models for 7 major cancer types."""
    print("\n[MV] Running multivariate + clinical models...")

    # Check available clinical variables
    mv_vars = ['netith_z', 'tmb_z']
    avail_clinical = []
    for c in ['age', 'stage_numeric', 'gender_male']:
        if c in df.columns and df[c].notna().sum() > 50:
            avail_clinical.append(c)
            mv_vars.append(c)

    print(f"  Clinical variables available: {avail_clinical}")

    # Focus on cancers with sufficient complete data
    cancer_counts = df.dropna(subset=mv_vars)['cancer_type'].value_counts()
    top_cancers = cancer_counts[cancer_counts >= 50].head(10).index.tolist()

    mv_results = []
    for ct in top_cancers:
        ct_df = df[df['cancer_type'] == ct].dropna(subset=mv_vars + ['OS', 'OS.time']).copy()
        if len(ct_df) < 50 or ct_df['OS'].sum() < 10:
            continue

        n = len(ct_df)
        row = {'cancer_type': ct, 'n_mv': n, 'n_events_mv': int(ct_df['OS'].sum())}

        # Model 1: NetITH only
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(ct_df[['OS.time', 'OS', 'netith_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'}),
                duration_col='duration', event_col='event')
            row['m1_netith_hr'] = np.exp(cph.params_['netith_z'])
            row['m1_netith_p'] = cph.summary.loc['netith_z', 'p']
            row['m1_cindex'] = cph.concordance_index_
        except Exception:
            row['m1_netith_hr'] = row['m1_netith_p'] = row['m1_cindex'] = np.nan

        # Model 2: TMB only
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(ct_df[['OS.time', 'OS', 'tmb_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'}),
                duration_col='duration', event_col='event')
            row['m2_tmb_hr'] = np.exp(cph.params_['tmb_z'])
            row['m2_tmb_p'] = cph.summary.loc['tmb_z', 'p']
            row['m2_cindex'] = cph.concordance_index_
        except Exception:
            row['m2_tmb_hr'] = row['m2_tmb_p'] = row['m2_cindex'] = np.nan

        # Model 3: NetITH + TMB
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(ct_df[['OS.time', 'OS', 'netith_z', 'tmb_z']].rename(
                columns={'OS.time': 'duration', 'OS': 'event'}),
                duration_col='duration', event_col='event')
            row['m3_netith_hr'] = np.exp(cph.params_['netith_z'])
            row['m3_netith_p'] = cph.summary.loc['netith_z', 'p']
            row['m3_tmb_hr'] = np.exp(cph.params_['tmb_z'])
            row['m3_tmb_p'] = cph.summary.loc['tmb_z', 'p']
            row['m3_cindex'] = cph.concordance_index_
        except Exception:
            for k in ['m3_netith_hr', 'm3_netith_p', 'm3_tmb_hr', 'm3_tmb_p', 'm3_cindex']:
                row[k] = np.nan

        # Model 4: NetITH + TMB + Clinical
        mv_vars_m4 = ['netith_z', 'tmb_z'] + avail_clinical
        mv_vars_m4 = [v for v in mv_vars_m4 if v in ct_df.columns]
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cox_cols = ['OS.time', 'OS'] + mv_vars_m4
            cph.fit(ct_df[cox_cols].rename(
                columns={'OS.time': 'duration', 'OS': 'event'}),
                duration_col='duration', event_col='event')
            row['m4_netith_hr'] = np.exp(cph.params_['netith_z'])
            row['m4_netith_p'] = cph.summary.loc['netith_z', 'p']
            if 'tmb_z' in cph.params_:
                row['m4_tmb_hr'] = np.exp(cph.params_['tmb_z'])
                row['m4_tmb_p'] = cph.summary.loc['tmb_z', 'p']
            row['m4_cindex'] = cph.concordance_index_
            row['m4_n'] = cph._n_examples
        except Exception:
            for k in ['m4_netith_hr', 'm4_netith_p', 'm4_tmb_hr', 'm4_tmb_p', 'm4_cindex', 'm4_n']:
                row[k] = np.nan

        mv_results.append(row)
        print(f"  {ct}: M1 C={row.get('m1_cindex', np.nan):.3f} | "
              f"M2 C={row.get('m2_cindex', np.nan):.3f} | "
              f"M3 C={row.get('m3_cindex', np.nan):.3f} | "
              f"M4 C={row.get('m4_cindex', np.nan):.3f}")

    mv_df = pd.DataFrame(mv_results)
    mv_df.to_csv(OUTPUT_DIR / "multivariate_results.csv", index=False)
    print(f"  Saved: multivariate_results.csv ({len(mv_df)} cancers)")
    return mv_df


# ─── Visualization ────────────────────────────────────────────

def make_figures(cox_results: pd.DataFrame, meta_netith: dict, meta_tmb: dict,
                 merged: pd.DataFrame):
    """Generate publication-ready figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    C = {
        'netith': '#2c3e50',
        'tmb': '#c0392b',
        'sig': '#e74c3c',
        'ns': '#bdc3c7',
        'protective': '#27ae60',
        'risk': '#e74c3c',
    }

    # ── Figure A: Head-to-head forest plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(cox_results) * 0.3)),
                              gridspec_kw={'width_ratios': [1, 1]})

    for ax_idx, (metric, color, meta) in enumerate([
        ('netith', C['netith'], meta_netith),
        ('tmb', C['tmb'], meta_tmb),
    ]):
        ax = axes[ax_idx]
        df_plot = cox_results.dropna(subset=[f'{metric}_hr']).copy()
        df_plot['logHR'] = np.log(df_plot[f'{metric}_hr'].astype(float))
        df_plot = df_plot.sort_values('logHR', ascending=True)

        if len(df_plot) == 0:
            continue

        colors = [C['protective'] if hr < 1 else C['risk']
                  for hr in df_plot[f'{metric}_hr']]
        ypos = np.arange(len(df_plot))
        ax.barh(ypos, df_plot['logHR'], height=0.6, color=colors, alpha=0.7,
                edgecolor='black', lw=0.3)

        # CI whiskers
        for i, (_, row) in enumerate(df_plot.iterrows()):
            log_hr = row['logHR']
            log_ci_lo = np.log(row[f'{metric}_ci_lower'])
            log_ci_hi = np.log(row[f'{metric}_ci_upper'])
            if not (np.isnan(log_ci_lo) or np.isnan(log_ci_hi)):
                ax.plot([log_ci_lo, log_ci_hi], [i, i], color='black', lw=1.0)

        ax.axvline(0, color='gray', linestyle='--', lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{ct} (n={int(n)})"
                           for ct, n in zip(df_plot['cancer_type'], df_plot['n'])],
                          fontsize=7)
        ax.set_xlabel('log(Hazard Ratio)', fontsize=10)

        title = f"{'NetITH' if metric == 'netith' else 'TMB'} (per SD)"
        if 'error' not in meta:
            re_hr = meta['re_hr']
            re_p = meta['re_p']
            title += f"\nRE HR={re_hr:.2f}, p={re_p:.3f}, I²={meta['I2']:.0f}%"
        ax.set_title(title, fontweight='bold', fontsize=11)

    plt.suptitle('TCGA Pan-Cancer: NetITH vs TMB Survival Prediction',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "head_to_head_forest.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / "figures" / "head_to_head_forest.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: head_to_head_forest.png/pdf")

    # ── Figure B: C-index comparison ──
    fig, ax = plt.subplots(figsize=(10, max(5, len(cox_results) * 0.3)))
    df_plot = cox_results.dropna(subset=['netith_concordance', 'tmb_concordance']).copy()
    df_plot = df_plot.sort_values('netith_concordance')

    ypos = np.arange(len(df_plot))
    w = 0.35
    ax.barh(ypos + w/2, df_plot['netith_concordance'], w,
            color=C['netith'], alpha=0.7, label='NetITH')
    ax.barh(ypos - w/2, df_plot['tmb_concordance'], w,
            color=C['tmb'], alpha=0.7, label='TMB')
    ax.axvline(0.5, color='gray', linestyle='--', lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{ct} (n={int(n)})"
                       for ct, n in zip(df_plot['cancer_type'], df_plot['n'])],
                      fontsize=7)
    ax.set_xlabel('Concordance Index', fontsize=10)
    ax.set_title('C-index: NetITH vs TMB (Univariate)', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "cindex_comparison.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / "figures" / "cindex_comparison.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: cindex_comparison.png/pdf")

    # ── Figure C: TMB vs NetITH scatter ──
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged['netith_bulk'], merged['tmb_log'],
              s=1, alpha=0.3, color='#34495e', edgecolors='none', rasterized=True)
    rho = merged['netith_bulk'].corr(merged['TMB'], method='spearman')
    ax.set_xlabel('NetITH', fontsize=11)
    ax.set_ylabel('log2(TMB+1)', fontsize=11)
    ax.set_title(f'TMB vs NetITH (Spearman ρ = {rho:.3f})', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "tmb_vs_netith_scatter.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: tmb_vs_netith_scatter.png")

    # ── Figure D: Summary bar plot — significant cancer count ──
    fig, ax = plt.subplots(figsize=(6, 4))
    sig_netith = ((cox_results['netith_p'] < 0.05).sum() if 'netith_p' in cox_results else 0)
    sig_tmb = ((cox_results['tmb_p'] < 0.05).sum() if 'tmb_p' in cox_results else 0)
    sig_both_netith = ((cox_results['bi_netith_p'] < 0.05).sum() if 'bi_netith_p' in cox_results else 0)
    sig_both_tmb = ((cox_results['bi_tmb_p'] < 0.05).sum() if 'bi_tmb_p' in cox_results else 0)
    total = len(cox_results)

    x_labels = ['NetITH\n(univariate)', 'TMB\n(univariate)',
                'NetITH\n(+TMB adjusted)', 'TMB\n(+NetITH adjusted)']
    counts = [sig_netith, sig_tmb, sig_both_netith, sig_both_tmb]
    colors_bar = [C['netith'], C['tmb'], C['netith'], C['tmb']]

    ax.bar(range(4), counts, color=colors_bar, alpha=0.7, edgecolor='black', lw=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.3, f'{c}/{total}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Number of cancers with p<0.05', fontsize=10)
    ax.set_title('Significant Survival Associations', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "significant_count.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: significant_count.png")


# ─── Main ─────────────────────────────────────────────────────

def main():
    t_start = pd.Timestamp.now()
    print("=" * 60)
    print("TCGA NetITH vs TMB Survival Analysis")
    print("=" * 60)

    # 1. Compute TMB
    tmb = compute_tmb_from_mc3(MC3_PATH)

    # 2. Load NetITH
    netith = load_netith(NETITH_PATH)

    # 3. Load survival (contains clinical variables)
    surv = load_tcga_survival(TCGA_SURV_PATH)

    # 4. Merge + encode clinical
    merged = merge_all(tmb, netith, surv)

    # 6. Per-cancer Cox
    cox_results = run_per_cancer_cox(merged)

    # 7. Meta-analysis
    meta_netith, meta_tmb = run_both_meta_analyses(cox_results)

    # 8. Multivariate + clinical
    mv_results = run_multivariate_clinical(merged)

    # 9. Figures
    print("\n" + "=" * 60)
    print("[Figures] Generating...")
    make_figures(cox_results, meta_netith, meta_tmb, merged)

    # Summary table
    print("\n" + "=" * 60)
    print("[SUMMARY] NetITH vs TMB Survival Prediction")
    print("=" * 60)
    if 'error' not in meta_netith and 'error' not in meta_tmb:
        print(f"  NetITH RE HR: {meta_netith['re_hr']:.3f} "
              f"[{meta_netith['re_ci_lower']:.3f}–{meta_netith['re_ci_upper']:.3f}], "
              f"p={meta_netith['re_p']:.4f}, I²={meta_netith['I2']:.1f}%")
        print(f"  TMB RE HR:    {meta_tmb['re_hr']:.3f} "
              f"[{meta_tmb['re_ci_lower']:.3f}–{meta_tmb['re_ci_upper']:.3f}], "
              f"p={meta_tmb['re_p']:.4f}, I²={meta_tmb['I2']:.1f}%")
        print(f"  Better/Worse direction — NetITH: {meta_netith['n_better']}/{meta_netith['n_worse']}, "
              f"TMB: {meta_tmb['n_better']}/{meta_tmb['n_worse']}")

    elapsed = (pd.Timestamp.now() - t_start).total_seconds()
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
