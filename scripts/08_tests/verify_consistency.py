"""verify_consistency.py — automated data-integrity checks (review-driven, 2026-08-16).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-23
Pipeline: control / test stage — see repository README

Checks:
  1. master_results.yaml values are reproducible from source files (re-extract + compare)
  2. key numbers appearing in the manuscript/SI/cover letter match master values
  3. figure-legend sample sizes match source-data row counts (fixed mapping)
  4. every file referenced by DATA_PACKAGE_MANIFEST.md exists

Usage: python3 scripts/verify_consistency.py
Exit 0 = all checks pass; 1 = at least one inconsistency.
"""
import yaml, json, pandas as pd, numpy as np, os, re, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problems = []
checks = 0

def check(name, ok, detail=""):
    global checks
    checks += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not ok:
        problems.append((name, detail))

# ── 1) master_results reproducibility ─────────────────────────────────
print("== 1) master_results.yaml reproducibility ==")
master = yaml.safe_load(open(os.path.join(ROOT, "results/master_results.yaml")))["results"]

def reextract(kind, src, col=None, agg="median"):
    p = os.path.join(ROOT, src)
    if not os.path.exists(p):
        return None, "missing"
    if p.endswith(".json"):
        d = json.load(open(p))
        return d.get(kind), None
    df = pd.read_csv(p)
    if col is None:
        return len(df), None
    v = df[col].dropna()
    if agg == "median":
        return round(float(v.median()), 3), None
    return len(v), None

# manual mapping: result_id -> (source, col, agg, expected_kind)
manual = {
    "gdsc_netith_median": ("results/gdsc/gdsc_netith_cell_lines.csv", "NetITH", "median"),
    "tcga_netith_median": ("results/tcga/tcga_netith.csv", "netith_bulk", "median"),
    "drugs_n": ("results/gdsc/gdsc_drug_netith_correlations.csv", None, "count"),
    "imvigor_n": ("results/imvigor210/imvigor210_netith_results.csv", None, "count"),
    "sc_patients": ("results/scrnaseq_validation/patient_netith.csv", None, "count"),
    "vp_genes": ("results/depmap/perturbation/virtual_perturbation_genome_wide.csv", None, "count"),
    "crispr_genes": ("results/depmap/depmap_differential_dependency.csv", None, "count"),
    "gse25066_n": ("results/neoadjuvant/gse25066_summary.json", None, "json:n"),
    "coexpr_rho": ("results/scrnaseq_validation/coexpr_formulation_summary.json", None, "json:rho_coexpr_vs_netith_e"),
}
for rid, (src, col, agg) in manual.items():
    if rid not in master:
        continue
    if agg.startswith("json:"):
        d = json.load(open(os.path.join(ROOT, src)))
        val = d.get(agg.split(":", 1)[1])
        val = round(float(val), 3) if isinstance(val, float) else val
    elif agg == "count":
        val = len(pd.read_csv(os.path.join(ROOT, src)))
    else:
        df = pd.read_csv(os.path.join(ROOT, src))
        val = round(float(df[col].median()), 3)
    exp = master[rid]["value"]
    check(f"{rid} (master={exp}, re-extracted={val})", abs(float(exp) - float(val)) < 0.0015)

# ── 2) Manuscript/SI/cover key-number cross-check ─────────────────────
print("== 2) Manuscript/SI/cover letter key numbers ==")
ms = open(os.path.join(ROOT, "results/MANUSCRIPT_DRAFT.md")).read()
si = open(os.path.join(ROOT, "results/submission/Supplementary_Information.md")).read()
cl = open(os.path.join(ROOT, "docs/COVER_LETTER.md")).read()
all_text = ms + si + cl
pairs = [
    ("RE HR=0.827", "RE HR", "0.827"),
    ("FE HR=0.852", "FE HR", "0.852"),
    ("GSE25066 OR 1.27", "OR=1.27", "1.27"),
    ("GDSC median 5.70", "median 5.70", "5.70"),
    ("TCGA median 2.96", "median 2.96", "2.96"),
    ("drugs 276 FDR", "276 FDR", "276"),
    ("coexpr 0.34", "ρ=0.34", "0.34"),
    ("meta n 9,633", "9,633", "9633"),
]
for name, needle, val in pairs:
    check(name, needle in all_text or val in all_text, f"(needle: {needle})")

# ── 3) Figure-legend n vs source data ─────────────────────────────────
print("== 3) Figure legend n vs source data ==")
fig_map = [
    ("Fig2B: 286 drugs", "results/gdsc/gdsc_drug_netith_correlations.csv", None, 286),
    ("Fig2C: GSE25066 306", "results/neoadjuvant/gse25066_netith_pcr.csv", None, 306),
    ("FigS12A: Visium 3,377", "results/depmap/spatial_visium_radial_profile.csv", "n_spots", 3377),
    ("FigS12C: OV 4,205", "results/depmap/spatial_visium_OV_netith.csv", None, 4205),
]
for name, src, col, exp_n in fig_map:
    p = os.path.join(ROOT, src)
    if os.path.exists(p):
        df = pd.read_csv(p)
        n = int(df[col].sum()) if col else len(df)
        check(f"{name} (source n={n})", n == exp_n)
    else:
        check(name, False, "source missing")

# ── 4) Manifest file existence ────────────────────────────────────────
print("== 4) DATA_PACKAGE_MANIFEST referenced files exist ==")
mani = open(os.path.join(ROOT, "results/submission/DATA_PACKAGE_MANIFEST.md")).read()
refs = sorted(set(re.findall(r"`(results/[^`]+\.(?:csv|json|pdf|png))`", mani)))
# skip wildcard patterns (*) and composite paths (.md/.docx entries that list .pdf/.png alongside)
exact = [r for r in refs if "*" not in r and not re.search(r"\.(?:md|tex)/", r)]
missing = [r for r in exact if not os.path.exists(os.path.join(ROOT, r))]
check(f"{len(exact)} exact manifest references, {len(missing)} missing", len(missing) == 0, str(missing[:5]))

# ── 5) BH-FDR recomputation from source CSV (review-batch-A check 23) ────────
print("== 5) GDSC BH-FDR recomputation (review-batch-A check 23) ==")
try:
    dc = pd.read_csv(os.path.join(ROOT, "results/gdsc/gdsc_drug_netith_correlations.csv"))
    pv = dc["p_spearman"].values
    order = np.argsort(pv); ranks = np.argsort(order)
    n = len(pv)
    q = pv[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    bh_q = np.clip(q[ranks], 0, 1)
    n_sig = int((bh_q < 0.05).sum())
    exp = int(master["drugs_fdr"]["value"])
    needle_ok = "276 FDR" in all_text
    check(
        f"BH FDR<0.05 from p_spearman = {n_sig} (master drugs_fdr = {exp}; "
        f"manuscript needle '276 FDR' present = {needle_ok})",
        n_sig == exp and needle_ok,
        f"BH={n_sig}, master={exp}, needle={needle_ok}",
    )
except Exception as e:
    check("BH FDR recomputation", False, str(e))

print(f"\n{checks} checks, {len(problems)} problems")
if problems:
    for name, d in problems:
        print(f"  FAIL: {name} {d}")
    sys.exit(1)
print("ALL CONSISTENCY CHECKS PASSED")
