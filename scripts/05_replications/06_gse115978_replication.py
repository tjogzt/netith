"""
06_gse115978_replication.py — GSE115978 replication (Post-ICI resistant vs untreated, malignant stratification).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : {DATA_ROOT}/external/gse115978/GSE115978_counts.csv.gz (gene x cell counts),
          {DATA_ROOT}/external/gse115978/GSE115978_cell.annotations.csv.gz (cell annotations),
          {DATA_ROOT}/external/gse115978/sample_response.tsv (Table S1A response groups)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/gse115978_replication.json
Pipeline: replication stage — see repository README
"""
import gzip, json, io, os
import numpy as np
import pandas as pd
from scipy import stats, linalg
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared replication_common module
import replication_common as rc

ROOT = rc.ROOT
DATA_ROOT = rc.DATA_ROOT

print("[1/5] loading GSE115978...", flush=True)
ann = pd.read_csv(os.path.join(DATA_ROOT, "external", "gse115978", "GSE115978_cell.annotations.csv.gz"))
ann["samples"] = ann["samples"].str.strip('"')
# counts: CSV with quoted headers; stream and keep only the focused 239 genes
with gzip.open(os.path.join(DATA_ROOT, "external", "gse115978", "GSE115978_counts.csv.gz"), "rt") as f:
    hdr = f.readline().strip().split(",")
    cells = [c.strip('"') for c in hdr[1:]]
    focused0 = sorted(rc.focused_gene_set())
    focused_set = set(focused0)
    rows = {}
    for ln in f:
        p = ln.rstrip("\n").split(",")
        g = p[0].strip('"')
        if g in focused_set:
            rows[g] = [float(x) for x in p[1:1 + len(cells)]]
df = pd.DataFrame.from_dict(rows, orient="index", columns=cells)
print(f"  cells: {len(cells)}, focused genes: {df.shape[0]}/239", flush=True)
assert len(cells) == len(ann), f"cell mismatch {len(cells)} vs {len(ann)}"

# align annotation order to matrix columns
ann = ann.set_index("cells").loc[cells]
common = sorted(focused_set & set(df.index))
expr = df.loc[common].T.astype(np.float32)

# ---- edges ----
e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
print(f"  edges: {len(e_tf)}", flush=True)

# ---- per-cell z + vN entropy ----
print("[2/5] per-cell vN entropy...", flush=True)
vals = expr.values
z = rc.cohort_zscore(vals)
n_genes, n_cells = len(common), vals.shape[0]
ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
print(f"  entropy range [{ent.min():.3f}, {ent.max():.3f}]", flush=True)

# ---- sample/patient response ----
print("[3/5] response mapping...", flush=True)
resp = pd.read_csv(os.path.join(DATA_ROOT, "external", "gse115978", "sample_response.tsv"), sep="\t")
s2g = dict(zip(resp.Sample, resp["Treatment group"]))
# map each cell's sample -> group; patient = sample base (Mel75.1 -> Mel75)
def patient_of(s):
    import re
    m = re.match(r"(Mel\d+)", s)
    return m.group(1) if m else s
sample_arr = ann["samples"].values
cell_group = np.array([s2g.get(s, "") for s in sample_arr])
cell_pat = np.array([patient_of(s) for s in sample_arr])
cell_mal = np.array([1 if ct == "Mal" else 0 for ct in ann["cell.types"].values])
print(f"  group counts: {dict(Counter(cell_group))}", flush=True)

# sample-level NetITH_E
samples = sorted(set(sample_arr))
samp = {}
for s in samples:
    m = sample_arr == s
    if m.sum() < 10:
        continue
    samp[s] = {"sample": s, "patient": patient_of(s), "group": s2g.get(s, ""),
               "n_cells": int(m.sum()), "netith_e": rc.shannon_hist(ent[m])}
sdf = pd.DataFrame(samp).T
print(f"  samples: {len(sdf)} (resistant {sum(sdf.group=='Post-ICI (resistant)')}, untreated {sum(sdf.group=='Untreated')})", flush=True)

# patient-level (pool samples of same patient)
pats = {}
for p in sorted(set(patient_of(s) for s in samples)):
    m = np.array([patient_of(s) == p for s in sample_arr])
    if m.sum() < 10:
        continue
    g = [s2g.get(s, "") for s in set(sample_arr[m])]
    g = [x for x in g if x]
    grp = Counter(g).most_common(1)[0][0] if g else ""
    mal_cells_frac = float((cell_mal[m]).mean())
    pats[p] = {"patient": p, "group": grp, "n_cells": int(m.sum()),
               "netith_e": rc.shannon_hist(ent[m]), "mal_frac": mal_cells_frac}
pdf = pd.DataFrame(pats).T
print(f"  patients: {len(pdf)} (resistant {sum(pdf.group=='Post-ICI (resistant)')}, untreated {sum(pdf.group=='Untreated')})", flush=True)

print("[4/5] contrasts...", flush=True)
rs = sdf.loc[sdf.group == "Post-ICI (resistant)", "netith_e"]
ut = sdf.loc[sdf.group == "Untreated", "netith_e"]
res_samp = rc.contrast(rs, ut, "Resistant", "Untreated")
print(f"  sample-level: n_R={res_samp['n_a']} n_U={res_samp['n_b']} d={res_samp['cohen_d']:.3f} p={res_samp['mw_p']:.4f}", flush=True)
rp = pdf.loc[pdf.group == "Post-ICI (resistant)", "netith_e"]
up = pdf.loc[pdf.group == "Untreated", "netith_e"]
res_pat = rc.contrast(rp, up, "Resistant", "Untreated")
print(f"  patient-level: n_R={res_pat['n_a']} n_U={res_pat['n_b']} d={res_pat['cohen_d']:.3f} p={res_pat['mw_p']:.4f}", flush=True)

# malignant-only vs non-malignant-only contrasts (tumour-cell hypothesis)
res_mal_s, res_nmal_s = {}, {}
for label, mask in [("malignant", cell_mal == 1), ("non-malignant", cell_mal == 0)]:
    smp = {}
    for s in samples:
        m = (sample_arr == s) & mask
        if m.sum() < 10:
            continue
        smp[s] = {"sample": s, "group": s2g.get(s, ""), "n_cells": int(m.sum()),
                  "netith_e": rc.shannon_hist(ent[m])}
    s2 = pd.DataFrame(smp).T
    r = s2.loc[s2.group == "Post-ICI (resistant)", "netith_e"]
    u = s2.loc[s2.group == "Untreated", "netith_e"]
    res_mal_s[label] = rc.contrast(r, u, "Resistant", "Untreated")
    print(f"  [{label}] sample-level: d={res_mal_s[label]['cohen_d']:.3f} p={res_mal_s[label]['mw_p']:.4f} (n={res_mal_s[label]['n_a']}/{res_mal_s[label]['n_b']})", flush=True)

print("[5/5] output...", flush=True)
out = {"dataset": "GSE115978", "n_cells": int(n_cells), "n_genes": len(common), "n_edges": int(len(e_tf)),
       "n_samples": int(len(sdf)), "n_patients": int(len(pdf)),
       "entropy_range": [round(float(ent.min()), 3), round(float(ent.max()), 3)],
       "contrast_sample_res_vs_untreated": res_samp,
       "contrast_patient_res_vs_untreated": res_pat,
       "contrast_sample_malignant": res_mal_s["malignant"],
       "contrast_sample_nonmalignant": res_mal_s["non-malignant"],
       "per_sample": sdf.reset_index().rename(columns={"index": "key"}).to_dict("records"),
       "per_patient": pdf.reset_index().rename(columns={"index": "key"}).to_dict("records"),
       "note": "Whole-tumour cohort; Post-ICI resistant vs Untreated (only 1 OR sample excluded); construction identical to main compendium."}
out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse115978_replication.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print("written:", out_json)
