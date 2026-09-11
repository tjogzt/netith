"""run_control_mad_purity.py — expression-MAD increment and tumour-purity confounding (Controls 2 & 3).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Control 2 tests whether NetITH merely reflects global expression
    dispersion: Spearman rho(NetITH, per-sample expression MAD) in GDSC and
    TCGA, and, for each GDSC drug, compares the raw rho(NetITH, lnIC50) with
    the MAD-adjusted rank-based partial rho. Control 3 tests tumour-purity
    confounding by correlating TCGA NetITH with Aran 2015 purity estimates
    (ESTIMATE/ABSOLUTE/LUMP/IHC/CPE).

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - data/gdsc/rna_expr.csv + ensg_symbol_map.csv + cell_annot.csv: GDSC expression & names
    - $DATA_ROOT/gdsc_download/GDSC2_IC50_all.csv: GDSC2 lnIC50 per cell line and drug
    - $DATA_ROOT/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz: TCGA expression
    - results/gdsc/gdsc_netith_cell_lines.csv, results/tcga/tcga_netith.csv: NetITH
    - /tmp/tcga_purity_aran2015.csv: Aran 2015 purity estimates (downloaded separately)

Outputs:
    - results/control/gdsc_drug_rho_raw_vs_partialMAD.csv: raw vs MAD-partial rho per drug
    - results/control/tcga_expr_mad.csv: per-sample TCGA expression MAD
    - results/control/mad_purity_summary.json: Controls 2 & 3 summary

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_control_mad_purity.py
"""
from pathlib import Path
import os
import numpy as np, pandas as pd, gzip, time
from scipy.stats import spearmanr, rankdata
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
OUT = f"{ROOT}/results/control"
# Aran 2015 purity estimates (exported separately from TCGAbiolinks::Tumor.purity).
PURITY_CSV = "/tmp/tcga_purity_aran2015.csv"
os.makedirs(OUT, exist_ok=True)

# Rank-based partial Spearman: correlate OLS residuals of rank(x)/rank(y) on rank(z)
def partial_spearman(x, y, z):
    """Rank-based partial Spearman: correlation of OLS residuals."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(len(rz)), rz])
    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    ex, ey = resid(rx), resid(ry)
    r, _ = pearsonr(ex, ey)
    return r

# ───────── Control 2: GDSC ─────────
# Control 2 (GDSC): load expression, map ENSG -> symbols, rename to cell-line names
print("[GDSC] loading...")
expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
ensg_map = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg_map.columns=['ensg','symbol']
sym = dict(zip(ensg_map['ensg'], ensg_map['symbol']))
annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
cl_map = {cel: str(row.get('Characteristics.cell.line.','')) for cel,row in annot.iterrows()}
common_cels = [c for c in expr.columns if c in cl_map]
en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
en = en.loc[:, ~en.columns.duplicated()]
matched = en.index.isin(ensg_map['ensg'])
en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
en_sym = en_sym[~en_sym.index.duplicated(keep='first')]

# Load GDSC NetITH (the measure tested against expression dispersion)
netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH']
def col_mad(df, axis):
    arr = df.values
    med = np.median(arr, axis=axis, keepdims=True)
    return pd.Series(np.median(np.abs(arr - med), axis=axis), index=df.columns if axis==0 else df.index)
mad_gdsc = col_mad(en_sym, 0)  # per cell line across genes
mad_gdsc = mad_gdsc.reindex(netith.index)
common = netith.dropna().index.intersection(mad_gdsc.dropna().index)
# Spearman rho(NetITH, expr MAD); null = NetITH not driven by expression dispersion
r_mad, p_mad = spearmanr(netith.loc[common], mad_gdsc.loc[common])
print(f"GDSC NetITH vs expression MAD: rho={r_mad:.3f} (p={p_mad:.2g}), n={len(common)}")

# Load GDSC2 lnIC50 and pivot to a cell-line x drug matrix
ic50 = pd.read_csv(f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv')
ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
# Per drug (n>=30): raw rho(NetITH, lnIC50) vs MAD-adjusted partial rho
rows = []
for d in ic50_mat.columns:
    y = ic50_mat[d].dropna()
    c = netith.dropna().index.intersection(y.index).intersection(mad_gdsc.dropna().index)
    if len(c) < 30: continue
    c = [i for i in c if i in y.index and i in mad_gdsc.index]  # dedup guard
    xv = np.asarray([netith[i] for i in c]); yv = np.asarray([y[i] for i in c]); zv = np.asarray([mad_gdsc[i] for i in c])
    r0, _ = spearmanr(xv, yv)
    r1 = partial_spearman(xv, yv, zv)
    rows.append({'drug': d, 'n': len(c), 'rho_raw': r0, 'rho_partial_MAD': r1})
df_drug = pd.DataFrame(rows)
print(f"GDSC drugs: n={len(df_drug)}; median raw rho={df_drug['rho_raw'].median():.3f}, "
      f"median partial rho={df_drug['rho_partial_MAD'].median():.3f}; "
      f"raw>0 {int((df_drug['rho_raw']>0).sum())}/{len(df_drug)}, "
      f"partial>0 {int((df_drug['rho_partial_MAD']>0).sum())}/{len(df_drug)}")

# ───────── Control 2: TCGA MAD ─────────
# Control 2 (TCGA): per-sample expression MAD from the Xena pan-cancer matrix
print("[TCGA] loading Xena expression...")
t0 = time.time()
expr_tcga = pd.read_csv(f'{DATA_ROOT}/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz',
                        sep='\t', compression='gzip', index_col=0).T
print(f"  TCGA expr {expr_tcga.shape} ({time.time()-t0:.0f}s)")
# Per-sample MAD across genes (rows of the transposed expression matrix)
mad_tcga = col_mad(expr_tcga, 1)
mad_tcga.name = 'expr_MAD'
nt_tcga = pd.read_csv(f'{ROOT}/results/tcga/tcga_netith.csv', index_col=0)['netith_bulk']
# Normalize both NetITH and MAD indices to 15-char TCGA barcodes and deduplicate
# barcode normalize to 15 chars
def b15(x):
    x = str(x)
    return x[:15] if x.startswith('TCGA') else x
nt_tcga.index = [b15(i) for i in nt_tcga.index]
mad_tcga.index = [b15(i) for i in mad_tcga.index]
nt_tcga = nt_tcga[~nt_tcga.index.duplicated(keep='first')]
mad_tcga = mad_tcga[~mad_tcga.index.duplicated(keep='first')]
c = nt_tcga.dropna().index.intersection(mad_tcga.dropna().index)
# Spearman rho(TCGA NetITH, expr MAD); null = no relation to expression dispersion
r_mad_t, p_mad_t = spearmanr(nt_tcga.loc[c], mad_tcga.loc[c])
print(f"TCGA NetITH vs expr MAD: rho={r_mad_t:.3f} (p={p_mad_t:.2g}), n={len(c)}")

# ───────── Control 3: TCGA purity ─────────
# Control 3: Aran 2015 purity estimates; parse ',' decimals, normalize barcodes
purity = pd.read_csv(PURITY_CSV)
for col in ['ESTIMATE','ABSOLUTE','LUMP','IHC','CPE']:
    purity[col] = purity[col].astype(str).str.replace(',', '.').astype(float)
purity['Sample.ID'] = purity['Sample.ID'].map(b15)
pur = purity.set_index('Sample.ID')[['ESTIMATE','ABSOLUTE','LUMP','IHC','CPE']]
pur = pur[~pur.index.duplicated(keep='first')]
# Per purity estimator: Spearman rho vs NetITH (require n>=100)
pur_res = {}
for col in pur.columns:
    sub = pur[col].dropna()
    c = nt_tcga.dropna().index.intersection(sub.index)
    if len(c) < 100: continue
    r, p = spearmanr(nt_tcga.loc[c], sub.loc[c])
    pur_res[col] = {'rho': r, 'p': p, 'n': len(c)}
    print(f"TCGA NetITH vs {col} purity: rho={r:.3f} (p={p:.2g}), n={len(c)}")

# ───────── save ─────────
summary = {
    'gdsc_netith_vs_mad_rho': r_mad, 'gdsc_netith_vs_mad_p': p_mad,
    'gdsc_drugs_n': len(df_drug),
    'gdsc_drug_rho_raw_median': df_drug['rho_raw'].median(),
    'gdsc_drug_rho_partialMAD_median': df_drug['rho_partial_MAD'].median(),
    'gdsc_drug_raw_pos': int((df_drug['rho_raw']>0).sum()),
    'gdsc_drug_partialMAD_pos': int((df_drug['rho_partial_MAD']>0).sum()),
    'gdsc_drug_raw_fdr_lt_005': 0,  # placeholder; FDR recomputed in verify
    'tcga_netith_vs_mad_rho': r_mad_t, 'tcga_netith_vs_mad_p': p_mad_t, 'tcga_mad_n': len(c),
    'purity': pur_res,
}
import json
# Save per-drug raw vs partial rho, TCGA MAD, and the summary JSON
df_drug.to_csv(f'{OUT}/gdsc_drug_rho_raw_vs_partialMAD.csv', index=False)
mad_tcga.to_csv(f'{OUT}/tcga_expr_mad.csv')
with open(f'{OUT}/mad_purity_summary.json','w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSUMMARY:", json.dumps(summary, indent=2, default=str))
