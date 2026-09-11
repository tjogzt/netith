"""run_nomogram_bootstrap.py — bootstrap optimism-corrected C-index for TCGA nomograms (D-06, B=1000).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Reuses run_nomogram.py's data-loading helpers to build, per TCGA cancer
    type, a Cox model of OS on NetITH plus clinical variables (age, stage,
    T/N, grade, sex). The apparent C-index is computed on the full data and
    bootstrap-corrected with the .632 formula (0.368*C_app + 0.632*C_oob),
    where C_oob scores each bootstrap-refitted model on out-of-bag rows.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - results/tcga/tcga_tri_modal_merged.csv: tri-modal TCGA table with netith_bulk
    - $DATA_ROOT/GraphHMM/clinical_real_tcga/: TCGA clinical files (AJCC stage, T, N, grade)
    - scripts/06_clinical/run_nomogram.py: load_tri_modal / load_clinical_data / parse_* helpers

Outputs:
    - results/control/nomogram_bootstrap_summary.csv + .json: per-cancer C_apparent, C_oob, C_632 (D-06)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_nomogram_bootstrap.py
"""
from pathlib import Path
import os
import numpy as np, pandas as pd, sys, os, json, time
sys.path.insert(0, 'scripts')
from run_nomogram import load_tri_modal, load_clinical_data, parse_ajcc_stage, parse_ajcc_t, parse_ajcc_n, parse_grade
from lifelines import CoxPHFitter
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

CLINICAL_DIR = f'{DATA_ROOT}/GraphHMM/clinical_real_tcga'
TRI_MODAL = f'{ROOT}/results/tcga/tcga_tri_modal_merged.csv'
CANCERS = ['BLCA','BRCA','CRC','HNSC','KIRC','LIHC','LUAD']
B = 1000
RNG = np.random.default_rng(42)

# Load the merged tri-modal TCGA table (contains netith_bulk)
tri = load_tri_modal(TRI_MODAL)
out = []
t0 = time.time()
CANCER_MAP = {'CRC': ['COAD', 'READ']}
# Per cancer type: parse AJCC stage/T/N and grade to numeric, add age and sex
for cancer in CANCERS:
    clin = load_clinical_data(cancer, CLINICAL_DIR)
    if clin is None:
        continue
    clin['stage_numeric'] = clin['ajcc_pathologic_stage'].apply(parse_ajcc_stage)
    clin['t_numeric'] = clin['ajcc_pathologic_t'].apply(parse_ajcc_t)
    clin['n_numeric'] = clin['ajcc_pathologic_n'].apply(parse_ajcc_n)
    clin['grade_numeric'] = clin['tumor_grade'].apply(parse_grade)
    clin['age'] = clin['age_at_diagnosis'] / 365.25 if 'age_at_diagnosis' in clin.columns else np.nan
    clin['gender_male'] = (clin['gender'].str.lower() == 'male').astype(float)
    tri_labels = CANCER_MAP.get(cancer, [cancer])
    tri_cancer = tri[tri['cancer'].isin(tri_labels)].copy()
    merged = tri_cancer.merge(clin, on='patient', how='inner')
    if len(merged) < 30:
        out.append({'cancer': cancer, 'status': 'insufficient_merged'})
        continue

    clinical_vars = []
    for col in ['age','stage_numeric','t_numeric','n_numeric','grade_numeric','gender_male']:
        if col in merged.columns and merged[col].notna().sum() >= 20:
            clinical_vars.append(col)

    # Build the survival modelling matrix; require >=30 rows and >=5 events
    mv_cols = ['OS.time','OS','netith_bulk'] + clinical_vars
    mv = merged[mv_cols].dropna()
    mv = mv[mv['OS.time'] > 0].copy()
    if len(mv) < 30 or mv['OS'].sum() < 5:
        out.append({'cancer': cancer, 'status': 'insufficient'})
        continue

    # Apparent C-index: Cox PH (ridge penalizer 0.1) fitted on the full data
    # apparent C-index (full data)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(mv, duration_col='OS.time', event_col='OS')
    c_app = cph.concordance_index_
    n_events = int(mv['OS'].sum())

    # bootstrap .632 correction (out-of-bag testing avoids tied-time artefact
    # from duplicated rows in the bootstrap sample)
    # Bootstrap (B=1000): refit Cox on the resample, score concordance on
    # out-of-bag rows only (avoids tied-time artefact from duplicated rows)
    oob_cs = []
    for b in range(B):
        idx = RNG.integers(0, len(mv), len(mv))
        bs = mv.iloc[idx]
        oob_mask = ~np.isin(np.arange(len(mv)), np.unique(idx))
        oob = mv.iloc[oob_mask]
        if bs['OS'].sum() < 5 or len(oob) < 30 or oob['OS'].sum() < 5:
            continue
        try:
            cph_b = CoxPHFitter(penalizer=0.1)
            cph_b.fit(bs, duration_col='OS.time', event_col='OS')
            oob_cs.append(cph_b.score(oob, scoring_method='concordance_index'))
        except Exception:
            continue
    if len(oob_cs) < 50:
        out.append({'cancer': cancer, 'status': 'bootstrap_failed', 'n_boot': len(oob_cs)})
        continue
    # .632 optimism correction: C_632 = 0.368*C_apparent + 0.632*C_oob
    c_oob = np.mean(oob_cs)
    c_632 = 0.368 * c_app + 0.632 * c_oob
    out.append({'cancer': cancer, 'n': len(mv), 'n_events': n_events,
                'c_apparent': c_app, 'c_oob': c_oob,
                'c_632_corrected': c_632, 'n_boot': len(oob_cs)})
    print(f"[{cancer}] n={len(mv)} events={n_events} C_app={c_app:.3f} C_oob={c_oob:.3f} C_632={c_632:.3f} ({time.time()-t0:.0f}s)")

# Save per-cancer apparent/OOB/.632-corrected C-index (CSV + JSON)
df = pd.DataFrame(out)
os.makedirs(f'{ROOT}/results/control', exist_ok=True)
df.to_csv(f'{ROOT}/results/control/nomogram_bootstrap_summary.csv', index=False)
json.dump(df.to_dict('records'), open(f'{ROOT}/results/control/nomogram_bootstrap_summary.json','w'), indent=2)
print("saved", len(df))
