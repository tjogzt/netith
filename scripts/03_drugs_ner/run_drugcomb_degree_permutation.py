"""run_drugcomb_degree_permutation.py — DrugComb v1.4 degree-matched permutation test for database concordance.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/external/drugcomb/summary_table_v1.4.csv; results/depmap/drug_combination_predictions.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/drugcomb_degree_permutation/summary.json
Pipeline: drug-ner stage — see repository README
"""
import pandas as pd, numpy as np, json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
DC = DATA_ROOT / "external" / "drugcomb" / "summary_table_v1.4.csv"
OUT = ROOT / "results/depmap/drugcomb_degree_permutation"
OUT.mkdir(parents=True, exist_ok=True)
SYN_THRESH = 10.0
N_PERM = 1000
RNG = np.random.default_rng(42)

# ── 1) Load DrugComb v1.4 (needed columns only) ───────────────────────
print("== 1) Loading DrugComb v1.4 ==")
dc = pd.read_csv(DC, usecols=["drug_row", "drug_col", "cell_line_name", "synergy_zip"],
                 dtype={"drug_row": str, "drug_col": str, "cell_line_name": str,
                        "synergy_zip": "float32"})
dc = dc.dropna(subset=["synergy_zip"]).copy()
dc = dc.dropna(subset=["drug_row", "drug_col"])
dc["a"] = dc["drug_row"].str.strip().str.lower()
dc["b"] = dc["drug_col"].str.strip().str.lower()
dc["pair"] = dc["a"] + " + " + dc["b"]
print(f"  records: {len(dc):,}; unique pairs: {dc['pair'].nunique():,}; drugs: {pd.unique(pd.concat([dc['a'], dc['b']])).size:,}")

# pair-level median ZIP
pair_med = dc.groupby("pair")["synergy_zip"].median()
pair_any = dc.groupby("pair")["synergy_zip"].max()
pairs = pair_med.index
known_med = set(pairs[pair_med > SYN_THRESH])
known_any = set(pairs[pair_any > SYN_THRESH])
print(f"  pairs with median ZIP>{SYN_THRESH}: {len(known_med):,}; any ZIP>{SYN_THRESH}: {len(known_any):,}")

# drug degree = number of unique partners
deg_a = dc.groupby("a")["b"].nunique()
deg_b = dc.groupby("b")["a"].nunique()
degree = pd.concat([deg_a, deg_b]).groupby(level=0).max()
print(f"  drug degree quantiles: {degree.quantile([0.25,0.5,0.75,0.9]).round(0).to_dict()}")

# ── 2) GDSC predictions matching ──────────────────────────────────────
print("\n== 2) Matching GDSC predictions ==")
pred = pd.read_csv(ROOT / "results/depmap/drug_combination_predictions.csv")
pred["pair"] = (pred["drug1"].astype(str).str.strip().str.lower() + " + " +
                pred["drug2"].astype(str).str.strip().str.lower())
pred["pair_rev"] = (pred["drug2"].astype(str).str.strip().str.lower() + " + " +
                    pred["drug1"].astype(str).str.strip().str.lower())
all_pairs = set(pairs)
matched = pred[(pred["pair"].isin(all_pairs)) | (pred["pair_rev"].isin(all_pairs))].copy()
matched["canon"] = np.where(matched["pair"].isin(all_pairs), matched["pair"], matched["pair_rev"])
print(f"  predictions: {len(pred)}; matched in DrugComb: {len(matched)}")

obs_med = int(matched["canon"].isin(known_med).sum())
obs_any = int(matched["canon"].isin(known_any).sum())
print(f"  observed concordance: median-threshold {obs_med}/{len(matched)}; any-threshold {obs_any}/{len(matched)}")

# ── 3) Degree-matched permutation ─────────────────────────────────────
print(f"\n== 3) Degree-matched permutation ({N_PERM} draws) ==")
deg = degree
tol = 5
targets = matched["canon"].tolist()


pair_series = pd.Series(pairs)
parts = pair_series.str.split(r" \+ ", expand=True)
deg1_all = parts[0].map(deg).fillna(0).values.astype(int)
deg2_all = parts[1].map(deg).fillna(0).values.astype(int)
pair_arr = pairs.values
known_arr = np.array([p in known_med for p in pair_arr])

def perm_once():
    k = 0
    for t in targets:
        d1, d2 = t.split(" + ")
        dd1, dd2 = int(deg.get(d1, 0)), int(deg.get(d2, 0))
        mask = (np.abs(deg1_all - dd1) <= tol) & (np.abs(deg2_all - dd2) <= tol)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        c = pair_arr[RNG.choice(idx)]
        k += int(c in known_med)
    return k

null = np.array([perm_once() for _ in range(N_PERM)])
p_perm = (np.sum(null >= obs_med) + 1) / (N_PERM + 1)
print(f"  observed concordance: {obs_med}/{len(matched)}")
print(f"  degree-matched null: mean={null.mean():.2f}, median={np.median(null):.0f}, "
      f"95% CI=[{np.percentile(null,2.5):.0f},{np.percentile(null,97.5):.0f}]")
print(f"  permutation p (obs >= null): {p_perm:.4f}")

# fully random baseline (no degree matching)
null_all = np.array([int(pair_arr[RNG.integers(0, len(pair_arr))] in known_med) for _ in range(N_PERM)])
base_rate = known_arr.mean()
print(f"  global baseline synergy rate: {base_rate:.4f}")

report = dict(
    n_records=int(len(dc)), n_pairs=int(pairs.nunique()), n_drugs=int(degree.size),
    synergy_threshold=SYN_THRESH, n_known_pairs_median=int(len(known_med)),
    n_predictions=int(len(pred)), n_matched=int(len(matched)),
    observed_concordance_median=int(obs_med), observed_concordance_any=int(obs_any),
    null_mean=float(null.mean()), null_median=float(np.median(null)),
    null_ci=[float(np.percentile(null,2.5)), float(np.percentile(null,97.5))],
    permutation_p=float(p_perm), global_baseline_rate=float(base_rate),
    degree_tolerance=tol, n_permutations=N_PERM, seed=42,
)
json.dump(report, open(OUT / "summary.json", "w"), indent=1)
print("\nDone. Outputs in results/depmap/drugcomb_degree_permutation/")
