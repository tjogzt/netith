"""
04_gse72056_replication.py — GSE72056 replication (tumour-vs-immune contrast, construction consistency).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : {DATA_ROOT}/external/gse72056/matrix.txt.gz (annotations + gene x cell)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/gse72056_replication.json
Pipeline: replication stage — see repository README
"""
import gzip, json, os
import numpy as np
import pandas as pd
from scipy import stats, linalg
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared replication_common module
import replication_common as rc

ROOT = rc.ROOT
DATA_ROOT = rc.DATA_ROOT
RNG = np.random.RandomState(49)

print("[1/5] loading GSE72056...", flush=True)
with gzip.open(os.path.join(DATA_ROOT, "external", "gse72056", "matrix.txt.gz"), "rt", errors="replace") as f:
    lines = f.read().split("\n")
cells = lines[0].rstrip("\t").split("\t")[1:]
tumor_row = lines[1].rstrip("\t").split("\t")[1:]
mal_row = lines[2].rstrip("\t").split("\t")[1:]     # 1=no,2=yes,0=unresolved
ct_row = lines[3].rstrip("\t").split("\t")[1:]      # non-malignant type
print(f"  cells: {len(cells)}, patients: {len(set(tumor_row))}", flush=True)

# keep only the focused 239 genes, one value per cell
focused = rc.focused_gene_set()
rows = {}
for ln in lines[4:]:
    if not ln.strip():
        continue
    p = ln.rstrip("\t").split("\t")
    if p[0] in focused:
        rows[p[0]] = [float(x) for x in p[1:1 + len(cells)]]
df = pd.DataFrame.from_dict(rows, orient="index", columns=cells)
print(f"  focused genes: {df.shape[0]}/239", flush=True)

common = list(df.index)
e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
print(f"  edges: {len(e_tf)}", flush=True)

# per-cell cohort z-score + vN entropy
vals = df.T.values.astype(np.float32)
z = rc.cohort_zscore(vals)

n_cells = vals.shape[0]
ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
print(f"[2/5] per-cell entropy done [{ent.min():.3f},{ent.max():.3f}]", flush=True)

# annotations: malignant (2) vs immune (1); unresolved (0) excluded from the contrast
mal = np.array([1 if x == "2" else 0 for x in mal_row])
unres = np.array([x == "0" for x in mal_row])
ct = np.array(ct_row)
mal_ent = ent[mal == 1]; imm_ent = ent[mal == 0]
d_ti = (mal_ent.mean() - imm_ent.mean()) / np.sqrt(((len(mal_ent)-1)*mal_ent.var(ddof=1) + (len(imm_ent)-1)*imm_ent.var(ddof=1)) / (len(mal_ent)+len(imm_ent)-2))
p_ti = stats.mannwhitneyu(mal_ent, imm_ent, alternative="two-sided").pvalue
print(f"[3/5] tumour vs immune: d={d_ti:.3f} p={p_ti:.2e} (n={len(mal_ent)}/{len(imm_ent)})", flush=True)

# patient-level NetITH_E + malignant fraction
pat = {}
for pid in sorted(set(tumor_row)):
    m = np.array([p == pid for p in tumor_row])
    if m.sum() < 30:
        continue
    frac_mal = (mal[m] == 1).mean()
    pat[pid] = {"netith_e": rc.shannon_hist(ent[m]), "n_cells": int(m.sum()),
                "malignant_frac": float(frac_mal)}
pats = pd.DataFrame(pat).T
r_frac = stats.spearmanr(pats["netith_e"].dropna(), pats["malignant_frac"].dropna())
print(f"[4/5] patients: {len(pats)}; NetITH_E vs malignant fraction rho={r_frac[0]:.3f} p={r_frac[1]:.3f}", flush=True)

out = {"dataset": "GSE72056", "n_cells": int(n_cells), "n_patients": int(len(pats)),
       "n_focused_genes": df.shape[0], "n_edges": int(len(e_tf)),
       "tumour_vs_immune": {"n_mal": int(len(mal_ent)), "n_imm": int(len(imm_ent)),
                            "cohen_d": float(d_ti), "mw_p": float(p_ti),
                            "mal_median": float(np.median(mal_ent)), "imm_median": float(np.median(imm_ent))},
       "patient_netith_e_vs_malignant_frac": {"spearman_rho": float(r_frac[0]), "p": float(r_frac[1])},
       "per_patient": pats.reset_index().rename(columns={"index": "patient"}).to_dict("records"),
       "note": "survival table (Tirosh 2016 Table S1) not programmatically accessible (Science Cloudflare); outcome replication deferred; tumour-vs-immune contrast and construction consistency tested instead"}
out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse72056_replication.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2, default=str)
print("[5/5] written:", out_json)
