"""run_prism_validation.py — PRISM Repurposing secondary validation of the NetITH-drug association (B-L11).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Correlates GDSC NetITH with PRISM 19Q4 secondary-screen log2AUC drug
    response (lower log2AUC = more sensitive) across drugs with >=30 matched
    cell lines. Lines are matched between GDSC and PRISM by normalized
    cell-line name, falling back to DepMap ModelID, and the direction of
    rho(NetITH, log2AUC) is checked for consistency with GDSC lnIC50.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - $PRISM_PATH (default $DATA_ROOT/prism/PRISM_log2AUC.csv): PRISM 19Q4 log2AUC matrix
    - results/gdsc/gdsc_netith_cell_lines.csv: GDSC NetITH
    - data/external/depmap_metadata.csv: DepMap metadata (name <-> ModelID)
    - data/gdsc/cell_annot.csv: GDSC cell-line names

Outputs:
    - results/control/prism_validation_drugs.csv: per-drug rho(NetITH, log2AUC), p, BH FDR
    - results/control/prism_validation.json: matched lines, median rho, FDR<0.05 count (B-L11)

Usage:
    NETITH_DATA_ROOT=/path/to/data PRISM_PATH=/path/to/PRISM_log2AUC.csv \\
        python3 scripts/02_controls/run_prism_validation.py
"""
from pathlib import Path
import os
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
# PRISM 19Q4 log2AUC matrix (not redistributed; place under $NETITH_DATA_ROOT/prism/
# or set PRISM_PATH explicitly). Original source: DepMap PRISM Repurposing dataset.
PRISM = os.environ.get("PRISM_PATH", str(DATA_ROOT / "prism" / "PRISM_log2AUC.csv"))

# Load PRISM 19Q4 log2AUC matrix (DepMap ModelID rows x drug columns)
prism = pd.read_csv(PRISM, index_col=0)
print("PRISM:", prism.shape)

# NetITH (GDSC) with cell-line names
# Load GDSC NetITH (the primary measure whose drug associations are re-tested)
netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH'].dropna()

# Match: PRISM rows are DepMap ModelIDs (ACH-...) typically
# DepMap metadata: build cell-line name -> ModelID map for cross-dataset matching
meta = pd.read_csv(f'{ROOT}/data/external/depmap_metadata.csv')
# Build model->name and name->model maps
name_map = {}
for _, r in meta.iterrows():
    for c in ['cell_line_name', 'stripped_cell_line_name']:
        nm = str(r[c]).strip().upper()
        if nm and nm != 'NAN':
            name_map[nm] = r['ModelID'] if 'ModelID' in r.index else r['depmap_id']

# GDSC names
# GDSC annotations: cell-line names keyed by expression-column ID
annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
gdsc_names = {str(row.get('Characteristics.cell.line.','')).strip().upper(): cel
              for cel, row in annot.iterrows()}
# Map GDSC NetITH to PRISM rows
import re
def norm(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())

# Match GDSC lines to PRISM rows: normalized-name lookup, then DepMap ModelID
prism_norm = {norm(i): i for i in prism.index}
matched = []
for gname in netith.index:
    n = norm(gname)
    if n in prism_norm:
        matched.append((gname, prism_norm[n]))
    else:
        mid = name_map.get(gname.upper()) or name_map.get(n)
        if mid and mid in prism.index:
            matched.append((gname, mid))
print("matched lines:", len(matched))
if len(matched) < 30:
    print("WARNING: too few matches")
    raise SystemExit(1)

# Subset NetITH to matched PRISM lines and PRISM to those lines
net_prism = pd.Series({pid: netith[g] for g, pid in matched})
net_prism = net_prism[~net_prism.index.duplicated(keep='first')]
prism_sub = prism.loc[net_prism.index]
print("PRISM subset:", prism_sub.shape)

# per-drug association
# Per drug (>=30 lines): Spearman rho(NetITH, log2AUC); null = no association
rows = []
for drug in prism_sub.columns:
    y = prism_sub[drug].dropna()
    common = net_prism.index.intersection(y.index)
    if len(common) < 30: continue
    r, p = spearmanr(net_prism.loc[common], y.loc[common])
    rows.append({'drug': drug, 'n': len(common), 'rho': r, 'p': p})
df = pd.DataFrame(rows)
if len(df) == 0:
    print("no drugs with >=30 lines"); raise SystemExit(1)
# BH FDR across drugs; log2AUC lower = more sensitive, so a positive rho means
# higher NetITH -> higher AUC (more resistant), consistent with GDSC lnIC50
fdr = multipletests(df['p'], method='fdr_bh')[1]
df['fdr'] = fdr
print(f"\ndrugs tested: {len(df)}; median rho = {df['rho'].median():.3f}; "
      f"rho>0: {(df['rho']>0).sum()} ({100*(df['rho']>0).mean():.0f}%); "
      f"FDR<0.05: {(fdr<0.05).sum()}")
# PRISM log2AUC lower = more sensitive -> positive rho = higher NetITH -> higher AUC (more resistant)
print("direction: positive rho = NetITH↑ -> AUC↑ = more resistant (consistent with GDSC)")

import json, os
os.makedirs(f'{ROOT}/results/control', exist_ok=True)
# Save per-drug results (CSV) and summary statistics (JSON)
df.to_csv(f'{ROOT}/results/control/prism_validation_drugs.csv', index=False)
json.dump({'n_lines': int(len(matched)), 'n_drugs': int(len(df)),
           'median_rho': float(df['rho'].median()),
           'n_rho_pos': int((df['rho']>0).sum()),
           'n_fdr05': int((fdr<0.05).sum())},
          open(f'{ROOT}/results/control/prism_validation.json','w'), indent=2)
print("saved")
