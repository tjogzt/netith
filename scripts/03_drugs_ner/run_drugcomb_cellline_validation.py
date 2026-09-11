"""run_drugcomb_cellline_validation.py — DrugComb cell-line-level validation of predicted drug combinations.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/external/drugcomb/summary_table_v1.4.csv; results/gdsc/gdsc_netith_cell_lines.csv; results/depmap/drug_combination_predictions.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/drugcomb_cellline_validation/{summary.json, detailed.csv}
Pipeline: drug-ner stage — see repository README
"""
import pandas as pd, numpy as np, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
DC = DATA_ROOT / "external" / "drugcomb" / "summary_table_v1.4.csv"
OUT = ROOT / "results/depmap/drugcomb_cellline_validation"
OUT.mkdir(parents=True, exist_ok=True)
N_PERM = 1000
RNG = np.random.default_rng(7)
TOL = 5

# --- load DrugComb v1.4 raw records; keep only the fields needed for pair-cell synergy ---
dc = pd.read_csv(DC, usecols=["drug_row", "drug_col", "cell_line_name", "synergy_zip"],
                 dtype={"drug_row": str, "drug_col": str, "cell_line_name": str,
                        "synergy_zip": "float32"}).dropna(subset=["synergy_zip"])
dc = dc.dropna(subset=["drug_row", "drug_col"])
dc["a"] = dc["drug_row"].str.strip().str.lower()
dc["b"] = dc["drug_col"].str.strip().str.lower()
dc["cell"] = dc["cell_line_name"].str.strip().str.upper()
# canonical pair key (lowercase, sorted by column order) for joining with predictions
dc["pair"] = dc["a"] + " + " + dc["b"]

netith = pd.read_csv(ROOT / "results/gdsc/gdsc_netith_cell_lines.csv", index_col=0)
# restrict to cell lines present in both DrugComb and the GDSC NetITH panel
shared = set(netith.index.str.strip().str.upper()) & set(dc["cell"].unique())
dc_s = dc[dc["cell"].isin(shared)].copy()
# per-drug degree (number of distinct partners) for the degree-matched permutation null
deg = pd.concat([dc_s.groupby("a")["b"].nunique(), dc_s.groupby("b")["a"].nunique()]).groupby(level=0).max()

pred = pd.read_csv(ROOT / "results/depmap/drug_combination_predictions.csv")
all_pairs = set(dc_s["pair"].unique())
pred["p"] = pred["drug1"].astype(str).str.strip().str.lower() + " + " + pred["drug2"].astype(str).str.strip().str.lower()
pred["pr"] = pred["drug2"].astype(str).str.strip().str.lower() + " + " + pred["drug1"].astype(str).str.strip().str.lower()
# match the 154 GDSC predictions to DrugComb pairs (either drug order) and canonicalize
matched = pred[(pred["p"].isin(all_pairs)) | (pred["pr"].isin(all_pairs))].copy()
matched["canon"] = np.where(matched["p"].isin(all_pairs), matched["p"], matched["pr"])
targets = matched["canon"].tolist()

obs = dc_s[dc_s["pair"].isin(targets)].copy()
# permissive hit: any tested cell line with ZIP synergy >10; strict hit: pair-median ZIP >10
obs_units = obs[["pair", "cell", "synergy_zip"]].drop_duplicates()
obs_any_hit = int((obs_units["synergy_zip"] > 10).sum())
obs_med_hits = int((obs_units.groupby("pair")["synergy_zip"].median() > 10).sum())
print(f"shared cells: {len(shared)}; target pairs: {len(targets)}; "
      f"pair-cell units: {len(obs_units)}; observed any>10: {obs_any_hit}; pair-median>10: {obs_med_hits}")

# precompute per-cell-line pools (pairs, deg1, deg2, synergy arrays)
pools = {}
for cell, g in dc_s.groupby("cell"):
    g = g[["pair", "synergy_zip"]].drop_duplicates()
    pv = g["pair"].values
    p1 = pd.Series(pv).str.split(r" \+ ", expand=True)[0].map(deg).fillna(0).values.astype(int)
    p2 = pd.Series(pv).str.split(r" \+ ", expand=True)[1].map(deg).fillna(0).values.astype(int)
    pools[cell] = (pv, p1, p2, g["synergy_zip"].values)
print(f"pools built: {len(pools)} cells")

obs_pairs = obs_units["pair"].values
obs_cells = obs_units["cell"].values
n_units = len(obs_units)

def draw_once():
    hits = 0
    for i in range(n_units):
        cell = obs_cells[i]
        if cell not in pools:
            continue
        pv, p1, p2, syn = pools[cell]
        d1, d2 = obs_pairs[i].split(" + ")
        dd1, dd2 = int(deg.get(d1, 0)), int(deg.get(d2, 0))
        mask = (np.abs(p1 - dd1) <= TOL) & (np.abs(p2 - dd2) <= TOL)
        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue
        j = idx[RNG.integers(0, len(idx))]
        hits += int(syn[j] > 10)
    return hits

null = np.array([draw_once() for _ in range(N_PERM)])
p_any = (np.sum(null >= obs_any_hit) + 1) / (N_PERM + 1)
base_rate = float((dc_s["synergy_zip"] > 10).mean())
print(f"permutation: obs any-hits={obs_any_hit}; null mean={null.mean():.2f} "
      f"CI=[{np.percentile(null,2.5):.0f},{np.percentile(null,97.5):.0f}]; p={p_any:.4f}")
print(f"global shared-cell ZIP>10 rate: {base_rate:.4f}")

json.dump(dict(shared_cell_lines=len(shared), n_target_pairs=len(targets),
               observed_pair_cell_units=n_units, observed_any_hits=obs_any_hit,
               observed_med_hits=obs_med_hits, null_mean=float(null.mean()),
               null_ci=[float(np.percentile(null,2.5)), float(np.percentile(null,97.5))],
               permutation_p=float(p_any), global_any_rate=base_rate,
               n_permutations=N_PERM, seed=7, degree_tolerance=TOL),
          open(OUT / "summary.json", "w"), indent=1)
print("Done.")
