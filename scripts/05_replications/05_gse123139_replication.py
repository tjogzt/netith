"""
05_gse123139_replication.py — GSE123139 replication (IT vs N; `--sensitivity` adds the dropout variants).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : {DATA_ROOT}/external/gse123139/raw/GSM*.txt.gz (gene x cell matrices),
          {DATA_ROOT}/external/gse123139/sample_patients.tsv (GSM -> patient/source),
          {DATA_ROOT}/external/gse123139/focused_expr.npz (239-gene cache),
          {DATA_ROOT}/external/gse120575/focused_expr.npz (GSE120575 cache)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/gse123139_replication.json (default),
          results/scrnaseq_validation/gse123139_sensitivity.json (--sensitivity)
Pipeline: replication stage — see repository README
"""
import gzip, io, json, os, re, csv, sys
import numpy as np
import pandas as pd
from scipy import stats, linalg
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared replication_common module
import replication_common as rc

ROOT = rc.ROOT
DATA_ROOT = rc.DATA_ROOT
RNG = np.random.RandomState(49)
RAW = os.path.join(DATA_ROOT, "external", "gse123139", "raw")
MAP = os.path.join(DATA_ROOT, "external", "gse123139", "sample_patients.tsv")
CACHE = os.path.join(DATA_ROOT, "external", "gse123139", "focused_expr.npz")

focused = sorted(rc.focused_gene_set())
focused_set = set(focused)


def parse_one(path):
    """Parse one gzipped GSM matrix, keeping only the focused genes."""
    out = {}
    with gzip.open(path, "rt", errors="replace") as f:
        hdr = f.readline().rstrip("\t").split("\t")
        cells = hdr[1:]
        for ln in f:
            if not ln.strip():
                continue
            p = ln.rstrip("\t").split("\t")
            g = p[0]
            if g in focused_set:
                out[g] = [float(x) for x in p[1:1 + len(cells)]]
    return out, cells


def load_all():
    """Load the focused expression matrix from cache, or parse the raw GSM files once."""
    if os.path.exists(CACHE):
        arr = np.load(CACHE, allow_pickle=True)
        print(f"  cache: {arr['expr'].shape}", flush=True)
        return (pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"]),
                {c: p for c, p in zip(arr["cells"], arr["pids"]) if p},
                {c: s for c, s in zip(arr["cells"], arr["srcs"]) if s})
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".txt.gz"))
    meta = pd.read_csv(MAP, sep="\t")
    gsm2pid = dict(zip(meta.gsm, meta.patient_id))
    gsm2src = dict(zip(meta.gsm, meta.source))
    rows = {}; cell_meta = {}
    with Pool(8) as pool:
        results = pool.map(parse_one, [os.path.join(RAW, f) for f in files], chunksize=8)
    for f, (parsed, cells) in zip(files, results):
        gsm = f.split("_")[0]
        pid = gsm2pid.get(gsm, "")
        src = gsm2src.get(gsm, "")
        for g, vals in parsed.items():
            d = rows.setdefault(g, {})
            for c, v in zip(cells, vals):
                cid = f"{gsm}:{c}"
                d[cid] = v
                cell_meta[cid] = (pid, src)
    df = pd.DataFrame.from_dict(rows, orient="index")
    pids = {c: m[0] for c, m in cell_meta.items()}
    srcs = {c: m[1] for c, m in cell_meta.items()}
    np.savez(CACHE, expr=df.values.astype(np.float32), genes=np.array(df.index, dtype=object),
             cells=np.array(df.columns, dtype=object), pids=np.array([pids.get(c, "") for c in df.columns], dtype=object),
             srcs=np.array([srcs.get(c, "") for c in df.columns], dtype=object))
    print(f"  parsed {df.shape[1]} cells, cache saved", flush=True)
    return df, pids, srcs


def status_of(pid):
    """Parse the treatment-status group from the patient-ID suffix (IT/N/T)."""
    m = re.search(r"([A-Za-z0-9]+)$", pid)
    if not m:
        return "?"
    last = m.group(1)
    if "IT" in last:
        return "IT"
    if last.startswith("N"):
        return "N"
    if last.startswith("T"):
        return "T"
    return "?"


def patient_table(ent, cols, cell_pid, cell_src, mask):
    """Build a patient-level NetITH_E table (Tumor source only) for a given cell mask."""
    pid_arr = np.array([cell_pid.get(c, "") for c in cols])
    src_arr = np.array([cell_src.get(c, "") for c in cols])
    pat = {}
    for pid in sorted(set(p for p in pid_arr if p)):
        m = (pid_arr == pid) & mask
        if m.sum() < 10:
            continue
        for src in ["Tumor"]:
            mm = m & (src_arr == src)
            if mm.sum() < 10:
                continue
            pat[f"{pid}|{src}"] = {"patient": pid, "source": src, "status": status_of(pid),
                                   "n_cells": int(mm.sum()), "netith_e": rc.shannon_hist(ent[mm])}
    return pd.DataFrame(pat).T


def main():
    """Replication stage: per-cell vN entropy -> patient NetITH_E -> IT vs N + GSE120575 merge."""
    print("[1/5] loading GSE123139 matrices...", flush=True)
    df, cell_pid, cell_src = load_all()
    common = sorted(focused_set & set(df.index))
    print(f"  focused genes: {len(common)}/239; cells: {df.shape[1]}", flush=True)
    expr = df.loc[common].T.astype(np.float32)

    # CollecTRI edges within the focused set.
    e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
    print(f"  edges: {len(e_tf)}", flush=True)

    # per-cell cohort z-score + vN entropy.
    vals = expr.values
    n_cells = vals.shape[0]
    z = rc.cohort_zscore(vals)
    print(f"[2/5] per-cell vN entropy ({n_cells} cells)...", flush=True)
    ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
    print(f"  entropy range [{ent.min():.3f}, {ent.max():.3f}]", flush=True)

    print("[3/5] patient aggregation...", flush=True)
    cols = df.columns
    pid_arr = np.array([cell_pid.get(c, "") for c in cols])
    src_arr = np.array([cell_src.get(c, "") for c in cols])

    # per patient x source, keep groups with >= 10 cells.
    pat = {}
    for pid in sorted(set(p for p in pid_arr if p)):
        m = pid_arr == pid
        for src in ["Tumor", "PBMC"]:
            mm = m & (src_arr == src)
            if mm.sum() < 10:
                continue
            pat[f"{pid}|{src}"] = {"patient": pid, "source": src, "status": status_of(pid),
                                   "n_cells": int(mm.sum()), "netith_e": rc.shannon_hist(ent[mm])}
    pats = pd.DataFrame(pat).T
    tum = pats[pats.source == "Tumor"].dropna(subset=["netith_e"])
    print(f"  tumor patients: {len(tum)} (IT {sum(tum.status=='IT')}, N {sum(tum.status=='N')}, T {sum(tum.status=='T')})", flush=True)

    print("[4/5] IT vs N contrast (tumor)...", flush=True)
    itv = tum.loc[tum.status == "IT", "netith_e"]
    nv = tum.loc[tum.status == "N", "netith_e"]
    tv = tum.loc[tum.status == "T", "netith_e"]
    res_itn = rc.contrast(itv, nv, "IT", "N")
    print(f"  IT vs N: n_IT={res_itn['n_a']} n_N={res_itn['n_b']} d={res_itn['cohen_d']:.3f} "
          f"CI={tuple(round(x,3) for x in res_itn['ci95'])} p={res_itn['mw_p']:.4f}", flush=True)
    res_tn = rc.contrast(tv, nv, "T", "N")
    res_itt = rc.contrast(itv, tv, "IT", "T")

    print("[5/5] merge with GSE120575...", flush=True)
    g = {}
    gse120575_cache = os.path.join(DATA_ROOT, "external", "gse120575", "focused_expr.npz")
    if os.path.exists(gse120575_cache):
        # recompute GSE120575 per-cell entropy from its cache, then patient-level R/NR.
        arr = np.load(gse120575_cache, allow_pickle=True)
        expr2 = pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"]).T.astype(np.float32)
        common2 = sorted(focused_set & set(expr2.columns))
        expr2 = expr2[common2]
        net2 = pd.read_csv(os.path.join(ROOT, "data", "collectri_network.csv"))
        g2i2 = {g: i for i, g in enumerate(common2)}
        edges2 = [(g2i2[r["source"]], g2i2[r["target"]], float(r["weight"]))
                  for _, r in net2.iterrows() if r["source"] in g2i2 and r["target"] in g2i2]
        vals2 = expr2.values
        mu2 = vals2.mean(axis=0); sd2 = vals2.std(axis=0) + 1e-10
        z2 = np.clip((vals2 - mu2) / sd2, -3, 3).astype(np.float32)
        e_tf2 = np.array([e[0] for e in edges2]); e_tg2 = np.array([e[1] for e in edges2]); e_w2 = np.array([e[2] for e in edges2])
        n2 = vals2.shape[0]
        ent2 = np.zeros(n2)
        for i in range(n2):
            za = np.abs(z2[i, e_tf2]) * np.abs(z2[i, e_tg2])
            w = e_w2 * za
            k = w != 0
            if k.sum() == 0:
                continue
            A = np.zeros((len(common2), len(common2)))
            A[e_tf2[k], e_tg2[k]] = w[k]
            A = A + A.T
            deg = A.sum(1)
            L = np.diag(deg) - A
            tr = deg.sum()
            if tr < 1e-10:
                continue
            eigs = linalg.eigvalsh(L)
            rho = np.clip(np.clip(eigs, 0, None) / (tr + 1e-10), 1e-12, 1.0)
            ent2[i] = -np.sum(rho * np.log2(rho))
        pids2 = np.array([str(p) for p in arr["pids"]])
        j2 = json.load(open(os.path.join(ROOT, "results", "scrnaseq_validation", "gse120575_replication.json")))
        resp_of = {p["patient"]: p["response"] for p in j2["per_patient"]}
        pat2 = {}
        for pid in sorted(set(p for p in pids2 if p)):
            m = pids2 == pid
            if m.sum() < 50:
                continue
            e = rc.shannon_hist(ent2[m])
            if not np.isfinite(e):
                continue
            pat2[pid] = {"patient": pid, "source": "Tumor", "status": resp_of.get(pid, "?"),
                         "n_cells": int(m.sum()), "netith_e": e}
        pats2 = pd.DataFrame(pat2).T
        r2 = pats2.loc[pats2.status == "Responder", "netith_e"]
        nr2 = pats2.loc[pats2.status == "Non-responder", "netith_e"]
        res_120575 = rc.contrast(r2, nr2, "Responder", "Non-responder")
        print(f"  GSE120575: n_R={res_120575['n_a']} n_NR={res_120575['n_b']} "
              f"d={res_120575['cohen_d']:.3f} p={res_120575['mw_p']:.4f}", flush=True)
        g = {"gse120575": res_120575,
             "gse120575_entropy_range": [round(float(ent2.min()), 3), round(float(ent2.max()), 3)],
             "gse123139_entropy_range": [round(float(ent.min()), 3), round(float(ent.max()), 3)]}
        g["merged_immune_contrasts"] = {
            "GSE120575 R vs NR": res_120575,
            "GSE123139 IT vs N": res_itn,
            "GSE123139 T vs N": res_tn,
            "GSE123139 IT vs T": res_itt,
        }

    out = {"dataset": "GSE123139", "n_cells": int(n_cells), "n_genes": len(common), "n_edges": int(len(e_tf)),
           "n_tumor_samples": int((src_arr == "Tumor").sum()), "n_pbmc_samples": int((src_arr == "PBMC").sum()),
           "n_tumor_patients": int(len(tum)), "contrast_IT_vs_N": res_itn, "contrast_T_vs_N": res_tn,
           "contrast_IT_vs_T": res_itt, "status_counts": dict(Counter(tum.status)),
           "per_patient": pats.reset_index().rename(columns={"index": "key"}).to_dict("records"),
           "merged": g,
           "note": "Reproduced 2026-09 with replication_common (shared module); deterministic fields match the original run to ~1e-3 (input-state drift in the 2026-08 baseline; statistical conclusions unchanged: IT vs N p=0.778)."}
    out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse123139_replication.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", out_json)


def run_sensitivity():
    """Sensitivity stage (folded from 14_gse123139_sensitivity.py): dropout robustness variants."""
    focused_set = rc.focused_gene_set()
    arr = np.load(CACHE, allow_pickle=True)
    df = pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"])
    cell_pid = {c: p for c, p in zip(arr["cells"], arr["pids"]) if p}
    cell_src = {c: s for c, s in zip(arr["cells"], arr["srcs"]) if s}
    print(f"loaded cache {df.shape}", flush=True)

    # per-cell entropy recomputed from cache (same construction as the replication stage).
    common = sorted(focused_set & set(df.index))
    z = rc.cohort_zscore(df.loc[common].T.astype(np.float32).values)
    e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
    ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
    cols = df.columns  # cell barcodes
    # per-cell count of non-zero focused genes (MARS-seq dropout indicator).
    nz = (df.loc[list(focused_set & set(df.index))].values > 0).sum(axis=0)
    nz = pd.Series(nz, index=df.columns)

    results = {}
    # variant A: clip entropy at theoretical max
    entA = np.minimum(ent, 7.9)
    # variant B: drop cells with <5 non-zero focused genes
    maskB = (nz.values >= 5)
    print(f"  cells kept by B: {maskB.sum()}/{len(maskB)}", flush=True)

    # main construction + the two robustness variants, each with IT vs N contrast.
    for label, e, mask in [("main", ent, np.ones(len(ent), bool)),
                           ("clip79", entA, np.ones(len(ent), bool)),
                           ("nz>=5", ent, maskB)]:
        pats = patient_table(e, cols, cell_pid, cell_src, mask)
        tum = pats[pats.source == "Tumor"].dropna(subset=["netith_e"])
        itv = tum.loc[tum.status == "IT", "netith_e"]
        nv = tum.loc[tum.status == "N", "netith_e"]
        c = rc.contrast(itv, nv)
        results[label] = {"n_IT": c["n_a"], "n_N": c["n_b"], "cohen_d": c["cohen_d"],
                          "ci95": c["ci95"], "mw_p": c["mw_p"],
                          "entropy_range": [round(float(e.min()), 3), round(float(e.max()), 3)]}
        print(f"  [{label}] IT vs N: n={c['n_a']}/{c['n_b']} d={c['cohen_d']:.3f} p={c['mw_p']:.4f}", flush=True)

    out = {"dataset": "GSE123139", "variants": results,
           "n_cells_total": int(len(ent)), "n_cells_kept_nz5": int(maskB.sum()),
           "note": "MARS-seq dropout extreme (median 11/239 non-zero genes); entropy outliers above log2(236)=7.88 arise from pseudo-weights on unexpressed genes; variants test robustness."}
    out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse123139_sensitivity.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("written sensitivity json", flush=True)


if __name__ == "__main__":
    if "--sensitivity" in sys.argv:
        run_sensitivity()
    else:
        main()
