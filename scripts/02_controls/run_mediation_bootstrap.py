"""run_mediation_bootstrap.py — proliferation-mediation bootstrap with percentile CIs (P1, GDSC).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    For each of ~200 GDSC drugs with enough data, fits the mediation model
    NetITH = a*proliferation + e and lnIC50 = b*NetITH + c'*proliferation + e
    on B=1,000 bootstrap resamples of (NetITH, proliferation metagene, lnIC50).
    Reports the mediated proportion ab/total and the indirect effect ab with
    95% percentile CIs, replacing the earlier Sobel normal approximation.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - data/gdsc/rna_expr.csv + ensg_symbol_map.csv + cell_annot.csv: GDSC expression & names
    - results/gdsc/gdsc_netith_cell_lines.csv: GDSC NetITH
    - $DATA_ROOT/gdsc_download/GDSC2_IC50_all.csv: GDSC2 lnIC50 (cell line x drug)

Outputs:
    - results/control/mediation_bootstrap.csv: per-drug mediated proportion & indirect effect with CIs
    - results/control/mediation_bootstrap.json: summary (P1, replaces Sobel CI)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_mediation_bootstrap.py
"""
from pathlib import Path
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
ensg = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg.columns=['ensg','symbol']
sym = dict(zip(ensg['ensg'], ensg['symbol']))
annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
cl_map = {cel: str(row.get('Characteristics.cell.line.','')) for cel,row in annot.iterrows()}
common_cels = [c for c in expr.columns if c in cl_map]
en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
en = en.loc[:, ~en.columns.duplicated()]
matched = en.index.isin(ensg['ensg'])
en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
en_sym = en_sym[~en_sym.index.duplicated(keep='first')]

# Proliferation metagene: mean expression z of MKI67/PCNA/MCM2/TOP2A (mediator)
# proliferation metagene: mean z of MKI67, PCNA, MCM2, TOP2A
prolif_genes = ['MKI67','PCNA','MCM2','TOP2A']
pg = en_sym.loc[[g for g in prolif_genes if g in en_sym.index]]
prolif = pg.mean(axis=0)
prolif.name = 'prolif'

# Load NetITH (exposure) and lnIC50 (outcome); pivot to cell line x drug
netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH']
ic50 = pd.read_csv(f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv')
ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
# Bootstrap design: B=1000 resamples of the (NetITH, prolif, lnIC50) triple, seed 42
B = 1000
RNG = np.random.default_rng(42)

# Per drug (n>=40): OLS mediation paths estimated on each bootstrap resample
rows = []
for d in ic50_mat.columns:
    y = ic50_mat[d].dropna()
    c = netith.dropna().index.intersection(y.index).intersection(prolif.dropna().index)
    if len(c) < 40: continue
    X = pd.DataFrame({'netith': netith[c], 'prolif': prolif[c], 'y': y[c]}).dropna()
    if len(X) < 40: continue
    props = np.zeros(B)
    inds = np.zeros(B)
    for b in range(B):
        idx = RNG.integers(0, len(X), len(X))
        bs = X.iloc[idx]
        # Path a (prolif->NetITH), path b (NetITH->lnIC50 | prolif), direct c';
        # indirect effect ab = a*b, total effect = ab + c'
        # path a: prolif -> netith
        a = np.polyfit(bs['prolif'], bs['netith'], 1)[0]
        # path b: netith -> y adjusted for prolif
        Xb = np.column_stack([bs['netith'], bs['prolif'], np.ones(len(bs))])
        coef = np.linalg.lstsq(Xb, bs['y'], rcond=None)[0]
        b_path = coef[0]
        # total effect: prolif -> y? No: total = c' + ab where c' = direct (prolif -> y adjusted netith)
        cprime = coef[1]
        ab = a * b_path
        total = ab + cprime
        if abs(total) > 1e-12:
            props[b] = ab / total
            inds[b] = ab
        else:
            props[b] = np.nan
            inds[b] = np.nan
    # Percentile CI (2.5-97.5%) of the mediated proportion and the indirect effect
    # ab; a drug is flagged as significant mediation when the ab CI excludes 0.
    props = props[~np.isnan(props)]
    inds = inds[~np.isnan(inds)]
    if len(props) < 100: continue
    lo, hi = np.percentile(props, [2.5, 97.5])
    ilo, ihi = np.percentile(inds, [2.5, 97.5])
    rows.append({'drug': d, 'n': len(X),
                 'prop_median': np.median(props), 'prop_ci_lo': lo, 'prop_ci_hi': hi,
                 'ind_median': np.median(inds), 'ind_ci_lo': ilo, 'ind_ci_hi': ihi,
                 'ind_ci_excludes_0': bool((ilo > 0) or (ihi < 0))})
# Summarize: drugs whose indirect-effect CI excludes 0 = significant mediation
df = pd.DataFrame(rows)
print(f"drugs: {len(df)}; median mediated proportion (bootstrap): {df['prop_median'].median():.4f}")
print(f"IQR of medians: {df['prop_median'].quantile(0.25):.4f}-{df['prop_median'].quantile(0.75):.4f}")
print(f"indirect CI excludes 0: {df['ind_ci_excludes_0'].sum()}/{len(df)}")
print(f"prop_median > 0: {(df['prop_median']>0).sum()}/{len(df)}")
import json, os
os.makedirs(f'{ROOT}/results/control', exist_ok=True)
# Save per-drug bootstrap CIs (CSV) and summary statistics (JSON)
df.to_csv(f'{ROOT}/results/control/mediation_bootstrap.csv', index=False)
json.dump({'n_drugs': int(len(df)),
           'median_prop': float(df['prop_median'].median()),
           'iqr_prop': [float(df['prop_median'].quantile(0.25)), float(df['prop_median'].quantile(0.75))],
           'n_ind_ci_excl0': int(df['ind_ci_excludes_0'].sum()),
           'n_prop_pos': int((df['prop_median']>0).sum())},
          open(f'{ROOT}/results/control/mediation_bootstrap.json','w'), indent=2)
print("saved")
