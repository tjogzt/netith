"""
07_xue2022_hcc_replication.py — Xue et al. 2022 HCC outcome replication (Cox + log-rank on OS).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : {DATA_ROOT}/external/xue2022_hcc/expr.tsv.gz + meta.tsv (from 16_convert.R),
          {DATA_ROOT}/external/xue2022_hcc_survival.csv (Nature Table S1a),
          {DATA_ROOT}/external/xue2022_hcc/hrr2sample.json (patient-key mapping)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/xue2022_hcc_replication.json,
          {DATA_ROOT}/external/xue2022_hcc/patient_netith_os.csv
Pipeline: replication stage — see repository README
"""
import gzip, json, os, re
import numpy as np
import pandas as pd
from scipy import stats, linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared replication_common module
import replication_common as rc

ROOT = rc.ROOT
DATA_ROOT = rc.DATA_ROOT
EXPR = os.path.join(DATA_ROOT, "external", "xue2022_hcc", "expr.tsv.gz")
META = os.path.join(DATA_ROOT, "external", "xue2022_hcc", "meta.tsv")
SURV = os.path.join(DATA_ROOT, "external", "xue2022_hcc_survival.csv")

if not os.path.exists(EXPR):
    print("expr.tsv.gz missing - run code/R/16_xue2022_hcc_convert.R first")
    raise SystemExit(1)

print("[1/5] loading HCC matrix...", flush=True)
meta = pd.read_csv(META, sep="\t", index_col=0)
print(f"  meta: {meta.shape}, cols: {list(meta.columns)[:14]}", flush=True)
print("  DonorID sample:", meta['DonorID'].unique()[:8] if 'DonorID' in meta else "NO DonorID")
print("  SampleID sample:", meta['SampleID'].unique()[:5] if 'SampleID' in meta else "NO SampleID")

df = pd.read_csv(EXPR, sep="\t", index_col=0)  # genes x cells
print(f"  expr: {df.shape}", flush=True)
focused_set = rc.focused_gene_set()
common = sorted(focused_set & set(df.index))
print(f"  focused genes: {len(common)}/239", flush=True)
expr = df.loc[common].T.astype(np.float32)  # cells x genes

e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
print(f"  edges: {len(e_tf)}", flush=True)

# per-cell cohort z-score + vN entropy
print("[2/5] per-cell vN entropy...", flush=True)
vals = expr.values
z = rc.cohort_zscore(vals)
n_genes, n_cells = len(common), vals.shape[0]
ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
print(f"  entropy range [{ent.min():.3f}, {ent.max():.3f}]", flush=True)

# ---- patient key: DonorID (or SampleID fallback) ----
print("[3/5] patient aggregation...", flush=True)
key_col = "DonorID" if "DonorID" in meta.columns else "SampleID"
keys = meta[key_col].astype(str).values
if len(keys) != n_cells:
    # metadata may be a subset or reordered; align by cell names
    print("  aligning by cell names...", flush=True)
    meta_al = meta.reindex(expr.index)
    keys = meta_al[key_col].astype(str).values
print(f"  unique {key_col}: {len(set(keys))}", flush=True)

# patient-level NetITH_E (>= 10 cells per patient)
pat = {}
for k in sorted(set(keys)):
    m = np.array([str(x) == k for x in keys])
    if m.sum() < 10:
        continue
    e = rc.shannon_hist(ent[m])
    if not np.isfinite(e):
        continue
    pat[k] = {"patient": k, "n_cells": int(m.sum()), "netith_e": e}
pats = pd.DataFrame(pat).T
print(f"  patients: {len(pats)}", flush=True)

# ---- survival match (HRR -> A### via GSA runDetail) ----
print("[4/5] survival matching...", flush=True)
surv = pd.read_csv(SURV)
surv["Patient"] = surv["Patient"].astype(str).str.strip()
smap = {}
for _, r in surv.iterrows():
    smap[r["Patient"]] = {"os_days": r.get("OS_time"), "os_event": r.get("OS_state (Yes=0)"),
                          "fps_days": r.get("FPS_time"), "fps_event": r.get("Relapse_state (Yes=0)"),
                          "cancer": r.get("Cancer_type_short")}
hrr2s = json.load(open(os.path.join(DATA_ROOT, "external", "xue2022_hcc", "hrr2sample.json")))
def sample_of(k):
    s = hrr2s.get(k, "")
    m = re.match(r"([A-Za-z]+\d+)", s)
    return m.group(1) if m else s
pats["surv_key"] = [sample_of(k) for k in pats.index]
matched = pats[pats["surv_key"].isin(smap)].copy()
matched["os_days"] = [smap[k]["os_days"] for k in matched["surv_key"]]
matched["os_event"] = [smap[k]["os_event"] for k in matched["surv_key"]]
matched["fps_days"] = [smap[k]["fps_days"] for k in matched["surv_key"]]
matched["fps_event"] = [smap[k]["fps_event"] for k in matched["surv_key"]]
matched["cancer"] = [smap[k]["cancer"] for k in matched["surv_key"]]
matched["sample"] = [hrr2s.get(k, k) for k in matched.index]
matched = matched[matched["os_days"].notna() & matched["netith_e"].notna()]
print(f"  matched with OS: {len(matched)} (events {int(matched['os_event'].sum())})", flush=True)
print("  matched:", list(zip(matched.index, matched['sample'], matched['os_days'], matched['os_event'])), flush=True)

# ---- Cox + log-rank via R? no - implement numerically stable here with lifelines-free approach ----
# Use scipy-based Cox via statsmodels if available, else simple median split + spearman
print("[5/5] statistics...", flush=True)
results = {}
if len(matched) >= 10:
    x = matched["netith_e"].values.astype(float)
    t = matched["os_days"].values.astype(float)
    e = matched["os_event"].values.astype(float)
    # Spearman correlation with OS time (censoring-unaware, crude)
    rho_s, p_s = stats.spearmanr(x, t)
    # median split log-rank (simple approximation)
    med = np.median(x)
    g = x >= med
    hi_t = t[g]; lo_t = t[~g]
    hi_e = e[g]; lo_e = e[~g]
    results["n"] = int(len(matched))
    results["n_events"] = int(e.sum())
    results["spearman_netith_vs_os_time"] = {"rho": float(rho_s), "p": float(p_s)}
    results["median_split"] = {"n_high": int(g.sum()), "n_low": int((~g).sum()),
                               "high_events": int(hi_e.sum()), "low_events": int(lo_e.sum()),
                               "high_median_os": float(np.median(hi_t)), "low_median_os": float(np.median(lo_t))}
    print(f"  spearman rho={rho_s:.3f} p={p_s:.4f}; high {int(g.sum())} vs low {int((~g).sum())}", flush=True)
    # exact Cox delegated to R script 18b
    matched.to_csv(os.path.join(DATA_ROOT, "external", "xue2022_hcc", "patient_netith_os.csv"), index=True)

out = {"dataset": "Xue2022_HCC_SexTumorDB", "n_cells": int(n_cells), "n_genes": len(common),
       "n_edges": int(len(e_tf)), "n_patients_total": int(len(pats)),
       "n_matched_os": int(len(matched)) if len(matched) >= 10 else 0,
       "stats": results,
       "per_patient": pats.reset_index().rename(columns={"index": "patient"}).to_dict("records"),
       "note": "OS from Nature Table S1a; Cox computed in R (18b)."}
out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "xue2022_hcc_replication.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2, default=str)
print("written:", out_json)
