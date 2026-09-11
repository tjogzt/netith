"""
run_p1_robustness.py — P1 robustness: PH-assumption checks and 4-TF nested-CV/external validation.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - results/tcga/tcga_survival_results.csv: per-cancer TCGA Cox summaries
    - results/tcga/tcga_netith.csv: TCGA NetITH per sample
    - results/tcga/tcga_tri_modal_merged.csv: tri-modal table (OS/OS.time)
    - data/gdsc/rna_expr.csv, data/gdsc/cell_annot.csv, data/gdsc/ensg_symbol_map.csv: GDSC expression
    - results/gdsc/gdsc_netith_cell_lines.csv: GDSC NetITH per cell line
    - data/external/gse25066/GSE25066_series_matrix.txt.gz + GPL96_full.txt: GSE25066 expression
    - results/neoadjuvant/gse25066_summary.json + gse25066_netith_pcr.csv: GSE25066 NetITH
Outputs :
    - results/depmap/p1_robustness/ph_assumption.csv, summary.json
Pipeline: stage 2 — see repository README for the full pipeline order
"""

import pandas as pd, numpy as np, json, os, re, sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
OUT = ROOT / "results/depmap/p1_robustness"
OUT.mkdir(parents=True, exist_ok=True)
report = {}

# ── 1) PH assumption checks ───────────────────────────────────────────
print("== 1) Proportional-hazards assumption (26-cancer Cox) ==")
from lifelines import CoxPHFitter
sv = pd.read_csv(ROOT / "results/tcga/tcga_survival_results.csv")
ph_rows = []
for _, r in sv.iterrows():
    ct = r["cancer_type"]
    # refit per-cancer Cox (same convention as tcga_validation: tertile encoding)
    df = pd.read_csv(ROOT / "results/tcga/tcga_netith.csv")
    df = df[df["sample"].str.contains("-01", na=False)].copy()
    # per-cancer samples + survival (rebuilding from the summary is fragile; use NetITH tertiles)
    # check_assumptions would require the original fits, which are not saved for the 26-cancer Cox
    # instead rebuild a tertile Cox per cancer with lifelines
    pass

# simplification: the saved 26-cancer Cox fits are not available (models not stored) —
# rebuild from tcga_netith + survival table
surv_raw = pd.read_csv(ROOT / "results/tcga/tcga_netith.csv")
# survival source: tcga_survival_results holds only summaries; use tri_modal (contains OS/OS.time)
tri = pd.read_csv(ROOT / "results/tcga/tcga_tri_modal_merged.csv")
tri["sample"] = tri["sample"].str.replace(".", "-")
ph_rows = []
for ct in sv["cancer_type"]:
    sub = tri[tri["cancer"] == ct].dropna(subset=["OS.time", "OS", "netith_bulk"])
    if len(sub) < 30 or sub["OS"].sum() < 10:
        continue
    # tertile
    sub = sub.copy()
    sub["grp"] = pd.qcut(sub["netith_bulk"], 3, labels=False, duplicates="drop")
    if sub["grp"].nunique() < 2:
        continue
    d = sub[["OS.time", "OS", "grp"]].rename(columns={"OS.time": "duration", "OS": "event"})
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(d, duration_col="duration", event_col="event")
        from lifelines.statistics import proportional_hazard_test
        try:
            ph_res = proportional_hazard_test(cph, d, time_transform="rank")
            p_ph = float(np.atleast_1d(ph_res.p_value)[0])
        except Exception:
            p_ph = np.nan
        ph_rows.append(dict(cancer=ct, n=len(sub), events=int(sub["OS"].sum()),
                            ph_p=round(float(p_ph), 4) if p_ph == p_ph else np.nan))
    except Exception as e:
        ph_rows.append(dict(cancer=ct, n=len(sub), events=int(sub["OS"].sum()), ph_p=np.nan, err=str(e)[:50]))

ph_df = pd.DataFrame(ph_rows)
ph_df.to_csv(OUT / "ph_assumption.csv", index=False)
n_ph = int((ph_df["ph_p"] < 0.05).sum()) if len(ph_df) else 0
n_ph_nan = int(ph_df["ph_p"].isna().sum()) if len(ph_df) else 0
print(f"  PH tested: {len(ph_df)} cancers; violated (p<0.05): {n_ph}; not estimable: {n_ph_nan}")
report["ph"] = dict(n_tested=len(ph_df), n_violated=n_ph, n_not_estimable=n_ph_nan)

# ── 2) 4-TF nested CV + GSE25066 external ─────────────────────────────
print("\n== 2) 4-TF model: nested CV (GDSC) + external validation (GSE25066) ==")
# GDSC expression (4 TFs)
expr_raw = pd.read_csv(DATA_ROOT / "gdsc/rna_expr.csv", index_col=0)
annot = pd.read_csv(DATA_ROOT / "gdsc/cell_annot.csv", index_col=0)
cmap = {}
for cel, row in annot.iterrows():
    cl = str(row.get("Characteristics.cell.line.", ""))
    if cl and cl != "nan" and cl != "NA":
        cmap[cel] = cl
expr_named = expr_raw[[c for c in expr_raw.columns if c in cmap]].copy()
expr_named.columns = [cmap[c] for c in expr_named.columns]
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]
ensg_map = pd.read_csv(DATA_ROOT / "gdsc/ensg_symbol_map.csv")
sym = dict(zip(ensg_map.iloc[:, 0], ensg_map.iloc[:, 1]))
expr_named.index = [sym.get(g, g) for g in expr_named.index]
netith_g = pd.read_csv(ROOT / "results/gdsc/gdsc_netith_cell_lines.csv", index_col=0)["NetITH"]
common = expr_named.columns.intersection(netith_g.index)
TFS = ["JUN", "ATF4", "FOS", "STAT1"]
X = expr_named.loc[[t for t in TFS if t in expr_named.index]].T.loc[common]
X = (X - X.mean()) / (X.std() + 1e-10)
y = netith_g.loc[common]
print(f"  GDSC: {len(common)} cell lines, 4 TFs present: {[t for t in TFS if t in X.columns]}")

# nested CV: outer 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oob = np.zeros(len(X))
for tr, te in kf.split(X):
    lcv = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, random_state=42, max_iter=5000)
    lcv.fit(X.iloc[tr], y.iloc[tr])
    oob[te] = lcv.predict(X.iloc[te])
r2_oob = 1 - np.sum((y - oob) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"  Nested-CV R² (out-of-fold): {r2_oob:.3f} (vs. in-sample ~0.74)")
report["nested_cv"] = dict(r2_oob=round(float(r2_oob), 3), n=len(X))

# GSE25066 external: GDSC-trained coefficients -> GSE25066 predicted NetITH
import gzip
def parse_gse25066_expr(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    header = lines[header_idx + 1].strip().split("\t")
    rows = []
    for l in lines[header_idx + 2:]:
        if l.startswith("!series_matrix_table_end"):
            break
        parts = l.strip().split("\t")
        rows.append(parts)
    df = pd.DataFrame(rows)
    df.columns = [str(h).strip('"') for h in header]
    df = df.set_index(df.columns[0])
    df.columns = [str(c).strip('"') for c in df.columns]
    return df

gex = parse_gse25066_expr(DATA_ROOT / "external/gse25066/GSE25066_series_matrix.txt.gz")
# probe -> gene (GPL96 annotation)
# GPL96 SOFT annotation: skip ^ metadata lines, find header with ID/Gene Symbol
soft_lines = open(DATA_ROOT / "external/gse25066/GPL96_full.txt", errors="replace").readlines()
start = next(i for i, l in enumerate(soft_lines) if l.startswith("ID\t") or l.startswith('"ID"\t') or l.startswith("ID,"))
header_parts = soft_lines[start].strip().split("\t")
header_parts = [h.strip('"') for h in header_parts]
gene_col = next((j for j, h in enumerate(header_parts) if "Gene Symbol" in h or h == "Symbol"), 1)
rows = []
for l in soft_lines[start+1:]:
    parts = l.strip().split("\t")
    if len(parts) > gene_col:
        rows.append([parts[0].strip('"'), parts[gene_col].strip('"')])
ann = pd.DataFrame(rows, columns=["ID", "GeneSymbol"])
probe_gene = dict(zip(ann["ID"], ann["GeneSymbol"]))
gex.index = [probe_gene.get(str(i).strip('"'), str(i).strip('"')) for i in gex.index]
gex = gex[~gex.index.duplicated()]
tf_expr = gex.loc[[t for t in TFS if t in gex.index]].T
tf_expr = tf_expr.apply(pd.to_numeric, errors="coerce").dropna()
# GSE25066 observed NetITH (neoadj results)
nj = json.load(open(ROOT / "results/neoadjuvant/gse25066_summary.json"))
pcr_df = pd.read_csv(ROOT / "results/neoadjuvant/gse25066_netith_pcr.csv")
pcr_df["sample id"] = pcr_df["sample id"].astype(str)
# expression columns are GSM ids? neoadj script sample ids are GSM -- compare
netith_map = dict(zip(pcr_df["accession"].astype(str), pcr_df["NetITH"]))
matched = [c for c in tf_expr.index.tolist() if c in netith_map]
if len(matched) >= 30:
    # GDSC fitted coefficients
    lcv_full = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, random_state=42, max_iter=5000)
    lcv_full.fit(X, y)
    z_ext = (tf_expr.loc[matched] - tf_expr.loc[matched].mean()) / (tf_expr.loc[matched].std() + 1e-10)
    pred = lcv_full.predict(z_ext.values)
    obs = np.array([netith_map[c] for c in matched])
    rho, pval = spearmanr(pred, obs)
    r2_ext = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    print(f"  GSE25066 external: n={len(matched)}, Spearman(pred,obs)={rho:.3f} (p={pval:.4f}), R²={r2_ext:.3f}")
    report["external_gse25066"] = dict(n=len(matched), rho=round(float(rho), 3), p=round(float(pval), 4), r2=round(float(r2_ext), 3))
else:
    print(f"  GSE25066 matched samples: {len(matched)} (<30; external validation limited)")
    report["external_gse25066"] = dict(n=len(matched), note="insufficient matched samples")

json.dump(report, open(OUT / "summary.json", "w"), indent=1)
print("\nDone. Outputs in results/depmap/p1_robustness/")
