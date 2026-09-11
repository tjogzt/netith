"""
run_p08_drug_robustness.py — Robustness analyses of the pan-drug NetITH-IC50 association on GDSC2.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - results/gdsc/gdsc_netith_cell_lines.csv: per-cell-line NetITH
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 LN_IC50 with CANCER_TYPE metadata
    - data/gdsc/rna_expr.csv: GDSC expression (for MAD, PC1, proliferation covariates)
    - data/gdsc/cell_annot.csv: cell-line annotation (array-ID to cell-line-name mapping)
    - data/gdsc/ensg_symbol_map.csv: ENSG -> symbol mapping
Outputs :
    - results/depmap/p08_drug_robustness/{lineage_stratified,lineage_adjusted_partial,incremental_prediction,summary}.csv
Pipeline: stage 2 — see repository README for the full pipeline order
"""

import os
import pandas as pd, numpy as np, os, sys
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
OUT = ROOT / "results/depmap/p08_drug_robustness"
OUT.mkdir(parents=True, exist_ok=True)

def load_netith():
    df = pd.read_csv(ROOT / "results/gdsc/gdsc_netith_cell_lines.csv", index_col=0)
    return df["NetITH"]

def load_ic50():
    ic50 = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    ic50_mat = ic50.pivot_table(index="CELL_LINE_NAME", columns="DRUG_NAME",
                                values="LN_IC50", aggfunc="mean")
    meta = ic50[["CELL_LINE_NAME", "CANCER_TYPE"]].drop_duplicates().set_index("CELL_LINE_NAME")
    return ic50_mat, meta

netith = load_netith()
ic50_mat, meta = load_ic50()
common = netith.index.intersection(ic50_mat.index).intersection(meta.index)
netith_c = netith.loc[common]
meta_c = meta.loc[common]
ic50_c = ic50_mat.loc[common]
print(f"common cell lines: {len(common)}; drugs: {ic50_c.shape[1]}; lineages: {meta_c['CANCER_TYPE'].nunique()}")

summary = []

# ── 1) Lineage-stratified correlations ───────────────────────────────
# 1) Per-lineage Spearman rho of NetITH vs LN_IC50 across drugs (lineages with >= 15 cell lines)
lineage_rows = []
for lin in meta_c["CANCER_TYPE"].unique():
    mask = meta_c["CANCER_TYPE"] == lin
    if mask.sum() < 15:
        continue
    n_sub = netith_c[mask]
    ic_sub = ic50_c.loc[mask]
    rhos = []
    for drug in ic_sub.columns:
        d = ic_sub[drug].dropna()
        idx = d.index.intersection(n_sub.index)
        if len(idx) >= 10:
            r, _ = spearmanr(n_sub.loc[idx], d.loc[idx])
            rhos.append(r)
    if rhos:
        lineage_rows.append(dict(lineage=lin, n_cell_lines=int(mask.sum()),
                                 n_drugs=len(rhos),
                                 median_rho=float(np.median(rhos)),
                                 pct_positive=float(np.mean(np.array(rhos) > 0) * 100)))
lin_df = pd.DataFrame(lineage_rows).sort_values("median_rho", ascending=False)
# Write output: lineage-stratified correlation table
lin_df.to_csv(OUT / "lineage_stratified.csv", index=False)
print("\n[1] Lineage-stratified median rho (n_cell_lines >= 15):")
print(lin_df.head(10).to_string(index=False))
med_all = float(lin_df["median_rho"].median())
pos_all = float(lin_df["pct_positive"].mean())
summary.append(dict(analysis="lineage_stratified", median_lineage_median_rho=med_all,
                    mean_pct_positive_across_lineages=pos_all, n_lineages=len(lin_df)))

# ── 2) Lineage-adjusted partial correlations ──────────────────────────
from numpy.linalg import lstsq
def rankdata_np(x):
    return pd.Series(x).rank().values
def partial_corr(x, y, Zmat):
    """Spearman-style partial correlation via rank + OLS residualization."""
    def resid(v, Zmat):
        X = np.column_stack([np.ones(len(v)), Zmat])
        beta, *_ = lstsq(X, v, rcond=None)
        return v - X @ beta
    rx, ry = rankdata_np(x), rankdata_np(y)
    rx_r, ry_r = resid(rx, Zmat), resid(ry, Zmat)
    if np.std(rx_r) == 0 or np.std(ry_r) == 0:
        return np.nan, np.nan
    r, p = spearmanr(rx_r, ry_r)
    return r, p
# 2) Rank-based partial correlation: residualize NetITH and IC50 on lineage dummies (OLS), then Spearman
lineage_dummies = pd.get_dummies(meta_c["CANCER_TYPE"], drop_first=True)
part_rows = []
for drug in ic50_c.columns:
    d = ic50_c[drug].dropna()
    idx = d.index.intersection(netith_c.index)
    if len(idx) < 50:
        continue
    r_raw, _ = spearmanr(netith_c.loc[idx], d.loc[idx])
    Z_idx = lineage_dummies.loc[idx].values
    r_adj, p_adj = partial_corr(netith_c.loc[idx].values, d.loc[idx].values, Z_idx)
    part_rows.append(dict(drug=drug, n=len(idx), rho_raw=r_raw, rho_lineage_adj=r_adj))
part_df = pd.DataFrame(part_rows)
# Write output: raw vs lineage-adjusted partial correlation table
part_df.to_csv(OUT / "lineage_adjusted_partial.csv", index=False)
n_pos_raw = int((part_df["rho_raw"] > 0).sum())
n_pos_adj = int((part_df["rho_lineage_adj"] > 0).sum())
print(f"\n[2] Lineage-adjusted partial correlations ({len(part_df)} drugs, n>=50):")
print(f"    rho>0 raw: {n_pos_raw}/{len(part_df)}; rho>0 after lineage adjustment: {n_pos_adj}/{len(part_df)}")
summary.append(dict(analysis="lineage_adjusted_partial", n_drugs=len(part_df),
                    pos_raw=n_pos_raw, pos_adj=n_pos_adj,
                    median_rho_raw=float(part_df["rho_raw"].median()),
                    median_rho_adj=float(part_df["rho_lineage_adj"].median())))

# ── 3) Incremental prediction (30-drug sample) ────────────────────────
# 3) Build covariates (expression MAD, PC1, proliferation score) to test the incremental R2 of NetITH
expr_raw = pd.read_csv(DATA_ROOT / "gdsc/rna_expr.csv", index_col=0)
annot = pd.read_csv(DATA_ROOT / "gdsc/cell_annot.csv", index_col=0)
cell_line_map = {}
for cel, row in annot.iterrows():
    cl = str(row.get("Characteristics.cell.line.", ""))
    if cl and cl != "nan" and cl != "NA":
        cell_line_map[cel] = cl
common_cels = [c for c in expr_raw.columns if c in cell_line_map]
expr_named = expr_raw[common_cels].copy()
expr_named.columns = [cell_line_map[c] for c in common_cels]
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]
ensg_map = pd.read_csv(DATA_ROOT / "gdsc/ensg_symbol_map.csv")
sym_map = dict(zip(ensg_map.iloc[:, 0], ensg_map.iloc[:, 1])) if ensg_map.shape[1] >= 2 else {}
expr_named.index = [sym_map.get(g, g) for g in expr_named.index]
expr_t = expr_named.T  # cell lines x genes (symbols)
expr_t = expr_t.loc[expr_t.index.intersection(common)]
top_genes = expr_t.var().sort_values(ascending=False).index[:5000]
base_expr = expr_t[top_genes]
line_mad = base_expr.sub(base_expr.median(axis=1), axis=0).abs().median(axis=1)
from numpy.linalg import svd
Xc = base_expr.sub(base_expr.mean(axis=0)).div(base_expr.std(axis=0) + 1e-10).fillna(0)
U, S, Vt = svd(Xc, full_matrices=False)
pc1_series = pd.Series(U[:, 0], index=base_expr.index)
prolif_genes = [g for g in top_genes if g in ("MKI67", "PCNA", "MCM2", "TOP2A")]
prolif = base_expr[[g for g in prolif_genes if g in base_expr.columns]].mean(axis=1) if any(g in base_expr.columns for g in prolif_genes) else pd.Series(0.0, index=base_expr.index)
lin_dummies = pd.get_dummies(meta_c["CANCER_TYPE"], drop_first=True)

rng = np.random.default_rng(42)
drugs_sample = rng.choice(ic50_c.columns, size=min(30, ic50_c.shape[1]), replace=False)
inc_rows = []
for drug in drugs_sample:
    d = ic50_c[drug].dropna()
    idx = d.index.intersection(base_expr.index)
    if len(idx) < 60:
        continue
    y = d.loc[idx]
    feats = pd.DataFrame({"MAD": line_mad.loc[idx], "PC1": pc1_series.loc[idx], "prolif": prolif.loc[idx]})
    feats["NetITH"] = netith_c.loc[idx]
    lin = lin_dummies.loc[idx]
    X_base = pd.concat([feats[["MAD", "PC1", "prolif"]], lin], axis=1).fillna(0).astype(float)
    X_full = pd.concat([feats, lin], axis=1).fillna(0).astype(float)
    def r2(X, yv):
        Xm = np.column_stack([np.ones(len(yv)), X])
        beta, *_ = lstsq(Xm, yv, rcond=None)
        resid = yv - Xm @ beta
        return 1 - resid.var() / yv.var()
    # Compare OLS R2 of base covariates (lineage + MAD + PC1 + proliferation) vs base + NetITH (in-sample)
    r2b, r2f = r2(X_base.values, y.values), r2(X_full.values, y.values)
    inc_rows.append(dict(drug=drug, n=len(idx), r2_base=r2b, r2_full=r2f, delta_r2=r2f - r2b))
inc_df = pd.DataFrame(inc_rows)
# Write output: incremental prediction table
inc_df.to_csv(OUT / "incremental_prediction.csv", index=False)
print(f"\n[3] Incremental prediction (30-drug sample; base = lineage + MAD + PC1 + proliferation):")
print(f"    mean delta-R2 adding NetITH: {inc_df['delta_r2'].mean():.4f}; "
      f"median: {inc_df['delta_r2'].median():.4f}; positive: {(inc_df['delta_r2']>0).mean()*100:.0f}%")
summary.append(dict(analysis="incremental_prediction", n_drugs=len(inc_df),
                    mean_delta_r2=float(inc_df["delta_r2"].mean()),
                    median_delta_r2=float(inc_df["delta_r2"].median()),
                    pct_positive=float((inc_df["delta_r2"]>0).mean()*100)))

# ── 4) IC50 missingness vs NetITH ─────────────────────────────────────
# 4) Mann-Whitney U test: NetITH in cell lines with high vs low drug coverage (null: equal medians)
n_obs = ic50_c.notna().sum(axis=1)
obs_mask = n_obs >= ic50_c.shape[1] * 0.5
hi, lo = netith_c[obs_mask], netith_c[~obs_mask]
if len(hi) > 5 and len(lo) > 5:
    u, p_miss = mannwhitneyu(hi, lo)
    print(f"\n[4] IC50 missingness: cell lines with >=50% drug coverage n={len(hi)} (NetITH med {hi.median():.2f}) "
          f"vs <50% n={len(lo)} (med {lo.median():.2f}); MW p={p_miss:.4f}")
    summary.append(dict(analysis="ic50_missingness", n_high=len(hi), n_low=len(lo),
                        med_high=float(hi.median()), med_low=float(lo.median()), p=float(p_miss)))
else:
    print("\n[4] IC50 missingness: insufficient strata")

# ── 5) Drug inter-correlation / effective number of tests ─────────────
# 5) Median |Spearman rho| between drugs, and Cheverud-Nyholt effective number of tests (eigenvalue-based)
dmat = ic50_c.T.dropna(axis=0, thresh=50)
if dmat.shape[0] > 5:
    rho_mat = dmat.T.corr(method="spearman")
    tri = rho_mat.values[np.triu_indices_from(rho_mat.values, k=1)]
    tri = tri[~np.isnan(tri)]
    med_inter = float(np.median(np.abs(tri)))
    # effective tests via eigenvalue method (Cheverud-Nyholt)
    eig = np.linalg.eigvalsh(np.corrcoef(dmat.T.corr().fillna(0).values))
    eig = eig[eig > 1]
    n_eff = int(1 + (len(eig) - 1) * (1 - (len(eig) - 1) / dmat.shape[0])) if len(eig) > 1 else 1
    print(f"\n[5] Drug inter-correlation: median |rho| between drugs = {med_inter:.3f}; "
          f"effective tests ~{n_eff} (of {dmat.shape[0]})")
    summary.append(dict(analysis="drug_intercorrelation", n_drugs=dmat.shape[0],
                        median_abs_inter_rho=med_inter, n_effective_tests=n_eff))

# Write the consolidated per-analysis summary table
pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
print("\nDone. Outputs in results/depmap/p08_drug_robustness/")
