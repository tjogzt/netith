"""
01_scrnaseq_hardening.py — single-cell statistical hardening (round-3 verdict, step 1).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : results/scrnaseq_validation/patient_netith.csv,
          results/scrnaseq_validation/coexpr_formulation_patients.csv,
          {DATA_ROOT}/scrnaseq_reference/134d34af-cbcd-4837-9310-3d1f83ec6f18.h5ad,
          {DATA_ROOT}/geo/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/statistical_hardening.json
Pipeline: replication stage — see repository README
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("NETITH_DATA_ROOT", os.path.join(ROOT, "data"))
OUT = os.path.join(ROOT, "results", "scrnaseq_validation")


def cohen_d_pooled(a, b):
    """Pooled-standard-deviation Cohen's d between two groups."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def boot_ci_d(a, b, n_boot=2000, seed=49):
    """Bootstrap 95% CI of Cohen's d (fixed seed for reproducibility)."""
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_boot):
        da = rng.choice(a, len(a), replace=True)
        db = rng.choice(b, len(b), replace=True)
        ds.append(cohen_d_pooled(da, db))
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def mw_p(a, b):
    """Two-sided Mann-Whitney U p-value."""
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


res = {}

# ---- 1) primary test: pooled MAIN construction (NetITH_E = shannon, per manuscript) ----
pt = pd.read_csv(os.path.join(OUT, "patient_netith.csv"))
fav = pt.loc[pt["outcome"] == "Favourable", "netith_e"].values
unf = pt.loc[pt["outcome"] == "Unfavourable", "netith_e"].values
d_prim = cohen_d_pooled(fav, unf)
ci_prim = boot_ci_d(fav, unf)
p_prim = mw_p(fav, unf)
lev_p = float(stats.levene(fav, unf).pvalue)
res["primary"] = {"n_fav": int(len(fav)), "n_unfav": int(len(unf)),
                  "cohen_d": d_prim, "ci95": ci_prim, "mw_p": p_prim, "levene_p": lev_p}
print(f"[1] primary pooled: d={d_prim:.3f} CI={ci_prim} p={p_prim:.4f} (n={len(fav)}/{len(unf)})")

# ---- 2) exploratory BH across 15 tests (7 cancer types x main/coexpr + pooled coexpr) ----
co = pd.read_csv(os.path.join(OUT, "coexpr_formulation_patients.csv"))
tests = []
for ct in sorted(pt["cancer_type"].unique()):
    sub_m = pt[pt["cancer_type"] == ct]
    sub_c = co[co["cancer_type"] == ct]
    if len(sub_m) >= 4:
        f = sub_m.loc[sub_m["outcome"] == "Favourable", "netith_e"].values
        u = sub_m.loc[sub_m["outcome"] == "Unfavourable", "netith_e"].values
        if len(f) >= 2 and len(u) >= 2:
            tests.append({"label": f"main:{ct}", "p": mw_p(f, u)})
    if len(sub_c) >= 4:
        f = sub_c.loc[sub_c["outcome"] == "Favourable", "netith_coexpr"].values
        u = sub_c.loc[sub_c["outcome"] == "Unfavourable", "netith_coexpr"].values
        if len(f) >= 2 and len(u) >= 2:
            tests.append({"label": f"coexpr:{ct}", "p": mw_p(f, u)})
f = co.loc[co["outcome"] == "Favourable", "netith_coexpr"].values
u = co.loc[co["outcome"] == "Unfavourable", "netith_coexpr"].values
tests.append({"label": "coexpr:pooled", "p": mw_p(f, u)})
ps = np.array([t["p"] for t in tests])
from statsmodels.stats.multitest import multipletests
try:
    bh = multipletests(ps, method="fdr_bh")[1]
except Exception:
    # Fallback BH-FDR (manual Benjamini-Hochberg) if statsmodels is unavailable.
    o = np.argsort(ps); ro = np.argsort(o); n = len(ps)
    q = ps[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    bh = np.clip(q[ro], 0, 1)
for t, q in zip(tests, bh):
    t["fdr"] = float(q)
res["exploratory"] = {"n_tests": len(tests), "tests": tests,
                      "n_fdr_lt_01": int(sum(bh < 0.1)), "n_fdr_lt_05": int(sum(bh < 0.05))}
print(f"[2] exploratory {len(tests)} tests; FDR<0.1: {int(sum(bh<0.1))}, FDR<0.05: {int(sum(bh<0.05))}")
for t in tests:
    if t["fdr"] < 0.1:
        print(f"    {t['label']}: p={t['p']:.4f} FDR={t['fdr']:.3f}")

# ---- 3) construction stability: 5 variants + correlation matrix ----
variants = {"mean_vn": "main", "median_vn": "median", "netith_e": "shannon", "var_vn": "variance"}
stab = {}
for col, lab in variants.items():
    f = pt.loc[pt["outcome"] == "Favourable", col].values
    u = pt.loc[pt["outcome"] == "Unfavourable", col].values
    stab[lab] = {"cohen_d": cohen_d_pooled(f, u), "ci95": boot_ci_d(f, u), "mw_p": mw_p(f, u)}
f = co.loc[co["outcome"] == "Favourable", "netith_coexpr"].values
u = co.loc[co["outcome"] == "Unfavourable", "netith_coexpr"].values
stab["coexpr"] = {"cohen_d": cohen_d_pooled(f, u), "ci95": boot_ci_d(f, u), "mw_p": mw_p(f, u)}
# correlation matrix across the 113 patients (mean_vn/median_vn/netith_e/var_vn + coexpr matched)
corr_df = pt.set_index("patient")[["mean_vn", "median_vn", "netith_e", "var_vn"]].rename(
    columns={"mean_vn": "main", "median_vn": "median", "netith_e": "shannon", "var_vn": "variance"})
corr_df = corr_df.join(co.set_index("patient")[["netith_coexpr"]].rename(columns={"netith_coexpr": "coexpr"}))
corr_mat = corr_df.corr(method="spearman")
res["stability"] = {"variants": stab,
                    "correlation_matrix": corr_mat.round(3).to_dict()}
print("[3] stability:")
for k, v in stab.items():
    print(f"    {k}: d={v['cohen_d']:.3f} CI={tuple(round(x,3) for x in v['ci95'])} p={v['mw_p']:.4f}")
print("    corr matrix:\n", corr_mat.round(2).to_string())

# ---- 4) cell-type diagnostic (exploratory, pre-specified) ----
ct_res = {"status": "not-run"}
try:
    import anndata as ad
    adata = ad.read_h5ad(os.path.join(DATA_ROOT, "scrnaseq_reference", "134d34af-cbcd-4837-9310-3d1f83ec6f18.h5ad"))
    obs = adata.obs
    # find patient id field matching patient_netith.csv
    pt_ids = set(pt["patient"])
    id_cols = [c for c in obs.columns if obs[c].astype(str).isin(pt_ids).sum() > 50]
    id_col = id_cols[0] if id_cols else None
    if id_col is None:
        # try combining PMID_donor_id
        cand = [c for c in obs.columns if "donor" in c.lower() or "patient" in c.lower()]
        ct_res = {"status": "patient-id field not found", "candidates": cand}
    else:
        sub = obs[obs[id_col].astype(str).isin(pt_ids)].copy()
        sub["patient"] = sub[id_col].astype(str)
        merged = sub.merge(pt[["patient", "outcome"]], on="patient", how="left")
        # per-cell NetITH: check existing columns
        net_cols = [c for c in obs.columns if "netith" in c.lower()]
        if net_cols:
            ct_res = {"status": "per-cell NetITH columns exist", "netith_cols": net_cols}
        else:
            # pre-specified types: tumor/epithelial vs T cells
            ct_vals = merged["Cell_type_broad"].astype(str)
            tum = ct_vals.str.contains("Tumor|Epithelial|epithelial|Malignant|Cancer", case=False, regex=True)
            tcell = ct_vals.str.contains("^T cell|T-cell|CD8|CD4", case=False, regex=True)
            # use patient-level mean_vn stratified by dominant type share
            dom = merged.groupby("patient").apply(
                lambda g: pd.Series({"tumor_frac": tum.reindex(g.index).mean() if len(g) else np.nan,
                                     "n": len(g)})).reset_index()
            dom = dom.merge(pt[["patient", "outcome", "netith_e"]], on="patient")
            hi = dom[dom["tumor_frac"] > 0.5]
            lo = dom[dom["tumor_frac"] <= 0.5]
            ct_res = {"status": "tumor-fraction stratification (patient-level, exploratory)",
                      "n_hi": int(len(hi)), "n_lo": int(len(lo)),
                      "d_hi_vs_lo_netith_e": (cohen_d_pooled(hi["netith_e"].values, lo["netith_e"].values)
                      if len(hi) > 2 and len(lo) > 2 else None)}
            if len(hi) > 2 and len(lo) > 2:
                ct_res["mw_p"] = mw_p(hi["netith_e"].values, lo["netith_e"].values)
            # outcome within tumor-high
            if len(hi) >= 6:
                f = hi.loc[hi["outcome"] == "Favourable", "netith_e"].values
                u = hi.loc[hi["outcome"] == "Unfavourable", "netith_e"].values
                if len(f) >= 2 and len(u) >= 2:
                    ct_res["tumor_high_outcome_n"] = [int(len(f)), int(len(u))]
                    ct_res["tumor_high_outcome_d"] = cohen_d_pooled(f, u)
                    ct_res["tumor_high_outcome_p"] = mw_p(f, u)
except Exception as e:
    ct_res = {"status": f"error: {e}"}
res["cell_type"] = ct_res
print(f"[4] cell type: {ct_res.get('status', '?')}")

# ---- 5) GSE131907 construction consistency ----
gse = {"status": "not-run"}
try:
    gz = os.path.join(DATA_ROOT, "geo", "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz")
    if os.path.exists(gz):
        m = pd.read_csv(gz, sep="\t", index_col=0, compression="gzip", nrows=5)
        gse = {"status": "file readable", "shape_preview": list(m.shape), "index_head": list(m.index[:3])[:3]}
        # note: full computation requires gene-id mapping; report coverage estimate
    else:
        gse = {"status": "file not found"}
except Exception as e:
    gse = {"status": f"error: {e}"}
res["gse131907"] = gse
print(f"[5] GSE131907: {gse.get('status')}")

with open(os.path.join(OUT, "statistical_hardening.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print("JSON written:", os.path.join(OUT, "statistical_hardening.json"))
