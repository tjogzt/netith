"""run_control_depth.py — sequencing-depth confounding check (Control 3, TCGA pan-cancer).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Uses per-sample total RSEM expected counts from the Xena pan-cancer TCGA
    matrix as a proxy for sequencing depth, then tests whether NetITH depends
    on depth via Spearman's rho, both overall and per cancer type. A rho near
    zero supports that NetITH is not a sequencing-depth artefact.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - $DATA_ROOT/xena/tcgapancan/tcga_expected_count.gz: per-sample RSEM expected counts (depth proxy)
    - results/tcga/tcga_netith.csv: precomputed pan-cancer NetITH (netith_bulk column)

Outputs:
    - results/control/tcga_depth_total_counts.csv: per-sample total expected counts
    - results/control/depth_summary.json: overall + per-cancer-type Spearman rho (Control 3)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_control_depth.py
"""
from pathlib import Path
import os
import numpy as np, pandas as pd, gzip, time, json, os
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
OUT = f"{ROOT}/results/control"
os.makedirs(OUT, exist_ok=True)

# Load Xena TCGA pan-cancer expected counts as the sequencing-depth proxy
t0 = time.time()
expr = pd.read_csv(f'{DATA_ROOT}/xena/tcgapancan/tcga_expected_count.gz',
                   sep='\t', compression='gzip', index_col=0)
print(f"counts {expr.shape} ({time.time()-t0:.0f}s)")
# Depth proxy: per-sample total counts; barcodes trimmed to 15 chars, deduped, >0 kept
total = expr.sum(axis=0)
total.name = 'total_expected_counts'
total.index = total.index.astype(str).str[:15]
total = total[~total.index.duplicated(keep='first')]
total = total[total > 0]

# Load precomputed pan-cancer NetITH and normalize barcodes the same way
nt = pd.read_csv(f'{ROOT}/results/tcga/tcga_netith.csv', index_col=0)['netith_bulk']
nt.index = nt.index.astype(str).str[:15]
nt = nt[~nt.index.duplicated(keep='first')]

c = nt.dropna().index.intersection(total.index)
# Spearman rho(NetITH, log10 total counts); null = NetITH independent of depth
r, p = spearmanr(nt.loc[c], np.log10(total.loc[c]))
print(f"ALL: NetITH vs log10 total counts: rho={r:.3f} p={p:.2g} n={len(c)}")

# Per-cancer-type Spearman rho (cancer code = TCGA barcode positions 6-7), n>=30
per_cancer = {}
cancer = pd.Series([i[5:7] for i in c], index=c)
for ct, idx in cancer.groupby(cancer).groups.items():
    if len(idx) < 30: continue
    rr, pp = spearmanr(nt.loc[idx], np.log10(total.loc[idx]))
    per_cancer[ct] = {'n': int(len(idx)), 'rho': float(rr), 'p': float(pp)}
    print(f"  {ct}: rho={rr:.3f} p={pp:.2g} n={len(idx)}")

summary = {
    'all_rho': float(r), 'all_p': float(p), 'n': int(len(c)),
    'n_cancer_types': len(per_cancer),
    'per_cancer': per_cancer,
    'depth_note': 'proxy = total RSEM expected counts per sample (Xena tcga_expected_count)',
}
# Write the per-sample depth proxy and the overall/per-cancer summary JSON
pd.Series({'total_expected_counts': total}).to_csv(f'{OUT}/tcga_depth_total_counts.csv', header=True)
with open(f'{OUT}/depth_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
