"""
03_gse120575_replication.py — GSE120575 replication (R/NR; `--os` adds the OS outcome stage).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : {DATA_ROOT}/external/gse120575/TPM.txt.gz (gene x cell TPM),
          {DATA_ROOT}/external/gse120575/patient_ids.txt.gz (cell metadata + response),
          {DATA_ROOT}/external/gse120575/focused_expr.npz (239-gene cache),
          {DATA_ROOT}/external/gse120575/sample_to_os_mapping.tsv (OS stage only)
          (DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data)
Outputs : results/scrnaseq_validation/gse120575_replication.json (default),
          results/scrnaseq_validation/gse120575_os_replication.json (--os),
          {DATA_ROOT}/external/gse120575/os_netith_levels.csv + os_cox_results.json (--os)
Pipeline: replication stage — see repository README
"""
import gzip, io, json, csv, os, sys
import numpy as np
import pandas as pd
from scipy import stats, linalg
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared replication_common module
import replication_common as rc

ROOT = rc.ROOT
DATA_ROOT = rc.DATA_ROOT


def run_replication():
    """R/NR responder replication: per-cell vN entropy -> patient NetITH_E -> contrast."""
    # ---- load expression (gene x cell TPM) ----
    tpm_path = os.path.join(DATA_ROOT, "external", "gse120575", "TPM.txt.gz")
    print("[1/5] loading TPM...", flush=True)
    # robust parse: some rows have ragged fields, so cache the parsed focused subset
    cache = os.path.join(DATA_ROOT, "external", "gse120575", "focused_expr.npz")
    if os.path.exists(cache):
        arr = np.load(cache, allow_pickle=True)
        df = pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"])
        cell_pid = {c: p for c, p in zip(arr["cells"], arr["pids"]) if p}
        print(f"  TPM cache: {df.shape}", flush=True)
    else:
        focused0 = sorted(rc.focused_gene_set())
        # header row = cell IDs; second row embeds per-cell patient/timepoint IDs
        with gzip.open(tpm_path, "rt", errors="replace") as f:
            hdr = f.readline().rstrip("\t").split("\t")
            pid_line = f.readline().rstrip("\t").split("\t")
        cells = [c for c in hdr if c.strip()]
        n_cells = len(cells)
        pid_vals = [x for x in pid_line if x.strip() or True][1:1 + n_cells]
        cell_pid = {}
        for c, pid in zip(cells, pid_vals):
            if pid and pid.strip():
                cell_pid[c] = pid.strip()
        # streaming: keep only focused-gene rows (55,737 -> 239), each row split ~16.3k fields
        rows = {}
        with gzip.open(tpm_path, "rt", errors="replace") as f:
            f.readline(); f.readline()
            for ln in f:
                if not ln.strip():
                    continue
                parts = ln.rstrip("\t").split("\t")
                g = parts[0]
                if g in focused0:
                    rows[g] = [float(x) for x in parts[1:1 + n_cells]]
        df = pd.DataFrame.from_dict(rows, orient="index", columns=cells)
        print(f"  TPM: {df.shape} (patient row: {len(cell_pid)} cells)", flush=True)
        np.savez(cache, expr=df.values.astype(np.float32), genes=np.array(df.index, dtype=object),
                 cells=np.array(df.columns, dtype=object), pids=np.array([cell_pid.get(c, "") for c in df.columns], dtype=object))
        print("  cache saved", flush=True)

    # ---- focused 239 genes ----
    focused = sorted(rc.focused_gene_set())
    common = sorted(set(focused) & set(df.index))
    print(f"[2/5] focused genes present: {len(common)}/239", flush=True)
    expr = df.loc[common].T.astype(np.float32)  # cells x genes

    # ---- CollecTRI edges within focused set ----
    e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
    print(f"  edges: {len(e_tf)}", flush=True)

    # ---- per-cell z-score (cohort-level, clip) ----
    vals = expr.values
    z = rc.cohort_zscore(vals)

    # ---- per-cell vN entropy ----
    n_cells = vals.shape[0]
    print(f"[3/5] per-cell vN entropy ({n_cells} cells)...", flush=True)
    ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
    print(f"  entropy range [{ent.min():.3f}, {ent.max():.3f}]", flush=True)

    # ---- patient mapping + response ----
    print("[4/5] patient mapping...", flush=True)
    with gzip.open(os.path.join(DATA_ROOT, "external", "gse120575", "patient_ids.txt.gz"), "rt", errors="replace") as f:
        raw = f.read()
    rows = [r for r in csv.reader(io.StringIO(raw), delimiter="\t") if r and not r[0].startswith("#")]
    hdr = next(r for r in rows if any("Sample name" in x for x in r))
    data_rows = [r for r in rows[rows.index(hdr) + 1:] if r and r[0].strip()]
    pid_col = next(i for i, h in enumerate(hdr) if "patinet" in h or "patient" in h)
    resp_col = next(i for i, h in enumerate(hdr) if "response" in h)
    cell2pid = {}; cell2resp = {}
    for r in data_rows:
        sname = r[0].strip()
        if sname:
            cell2pid[sname] = r[pid_col].strip()
            cell2resp[sname] = r[resp_col].strip()
    # TPM columns -> patient via the embedded patient-ID row (Pre_P1, Post_P1_2, ...)
    print(f"  metadata samples: {len(cell2pid)}; TPM cols: {n_cells}", flush=True)
    tpm_cols = df.columns
    pid_from_col = [cell_pid.get(c, "") for c in tpm_cols]
    print("  pid example:", dict(zip(list(tpm_cols[:3]), pid_from_col[:3])), flush=True)

    # ---- patient-level NetITH_E (Shannon of within-patient entropy histogram) ----
    # patient-level response: metadata maps each Pre_Px/Post_Px label to a response
    meta_resp = {}
    for r in data_rows:
        lbl = r[pid_col].strip()
        if lbl and r[resp_col].strip() in ("Responder", "Non-responder"):
            meta_resp.setdefault(lbl, []).append(r[resp_col].strip())
    resp_of = {k: Counter(v).most_common(1)[0][0] for k, v in meta_resp.items()}
    pat = {}
    for pid in sorted(set(p for p in pid_from_col if p)):
        m = np.array([p == pid for p in pid_from_col])
        if m.sum() < 50:
            continue
        e = rc.shannon_hist(ent[m])
        resp = resp_of.get(pid, "")
        if not resp:
            continue
        pat[pid] = {"netith_e": e, "response": resp, "n_cells": int(m.sum())}
    pats = pd.DataFrame(pat).T
    print(f"[5/5] patients: {len(pats)} (R {sum(pats.response=='Responder')}, NR {sum(pats.response=='Non-responder')})", flush=True)

    # ---- responder vs non-responder ----
    r = pats.loc[pats["response"] == "Responder", "netith_e"].dropna().astype(float).values
    nr = pats.loc[pats["response"] == "Non-responder", "netith_e"].dropna().astype(float).values
    if len(r) >= 5 and len(nr) >= 5:
        c = rc.contrast(r, nr, "Responder", "Non-responder")
        d, p = c["cohen_d"], c["mw_p"]
        ci = (c["ci95"][0], c["ci95"][1])
    else:
        d, p, ci = np.nan, np.nan, (np.nan, np.nan)
    print(f"RESULT: n_R={len(r)} n_NR={len(nr)} d={d:.3f} CI={tuple(round(x,3) for x in ci)} p={p:.4f}")

    out = {"dataset": "GSE120575", "n_cells": int(n_cells), "n_genes": len(common), "n_edges": int(len(e_tf)),
           "n_patients": int(len(pats)), "n_R": int(len(r)), "n_NR": int(len(nr)),
           "cohen_d": float(d), "ci95": [float(ci[0]), float(ci[1])], "mw_p": float(p),
           "per_patient": pats.reset_index().rename(columns={"index": "patient"}).to_dict("records")}
    out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse120575_replication.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", out_json)


def run_os():
    """OS outcome replication: per-cell vN entropy -> patient/timepoint NetITH_E -> Cox + log-rank."""
    # ---- [1] per-cell vN entropy from cache (same construction as the replication stage) ----
    print("[1/4] per-cell vN entropy...", flush=True)
    arr = np.load(os.path.join(DATA_ROOT, "external", "gse120575", "focused_expr.npz"), allow_pickle=True)
    df = pd.DataFrame(arr["expr"], index=arr["genes"], columns=arr["cells"]).T.astype(np.float32)
    focused_set = rc.focused_gene_set()
    common = sorted(focused_set & set(df.columns))
    expr = df[common]
    e_tf, e_tg, e_w, n_genes = rc.build_edges(common)
    print(f"  edges: {len(e_tf)}", flush=True)
    vals = expr.values
    z = rc.cohort_zscore(vals)
    n_genes, n_cells = len(common), vals.shape[0]
    ent = rc.vn_entropy_per_cell(z, e_tf, e_tg, e_w, n_genes)
    print(f"  entropy range [{ent.min():.3f}, {ent.max():.3f}]", flush=True)

    # ---- [2] patient mapping (cell -> timepoint via pids cache, -> patient via Table S1) ----
    print("[2/4] patient mapping + OS...", flush=True)
    mapdf = pd.read_csv(os.path.join(DATA_ROOT, "external", "gse120575", "sample_to_os_mapping.tsv"), sep="\t")
    pids_arr = np.array([str(p) for p in arr["pids"]])  # per-cell timepoint label
    tp2pid = dict(zip(mapdf.geo_sample_id, mapdf.patient_id))
    tp2os = dict(zip(mapdf.geo_sample_id, zip(mapdf.os_days, mapdf.status_alive0_dead1)))
    tp2resp = dict(zip(mapdf.geo_sample_id, mapdf.geo_response))

    # patient-level: merge ALL timepoints of a patient
    pat = {}
    for pid in sorted(set(tp2pid.values())):
        tps = [tp for tp in set(pids_arr) if tp2pid.get(tp) == pid]
        m = np.array([pids_arr[i] in tps for i in range(len(pids_arr))])
        if m.sum() < 50:
            continue
        e = rc.shannon_hist(ent[m])
        if not np.isfinite(e):
            continue
        osd = [tp2os[t][0] for t in tps if t in tp2os]
        if not osd:
            continue
        pat[pid] = {"netith_e": e, "n_cells": int(m.sum()), "n_tps": len(tps),
                    "os_days": float(np.median(osd)), "status": int(tp2os[tps[0]][1])}
    pats = pd.DataFrame(pat).T
    print(f"  patients with OS: {len(pats)} (events {pats['status'].sum()})", flush=True)

    # timepoint-level (sensitivity, n=48, duplicates patient OS)
    tps_pat = []
    for t in sorted(set(pids_arr)):
        if t not in tp2os or t not in tp2pid:
            continue
        m = np.array([pids_arr[i] == t for i in range(len(pids_arr))])
        if m.sum() < 50:
            continue
        e = rc.shannon_hist(ent[m])
        if not np.isfinite(e):
            continue
        tps_pat.append({"timepoint": t, "patient": tp2pid[t], "netith_e": e,
                        "n_cells": int(m.sum()), "os_days": tp2os[t][0], "status": tp2os[t][1],
                        "response": tp2resp.get(t, "")})
    tpdf = pd.DataFrame(tps_pat)
    print(f"  timepoints with OS: {len(tpdf)} (events {tpdf['status'].sum()})", flush=True)

    # ---- [3] exact Cox + log-rank via R survival (authoritative) ----
    print("[3/4] Cox + log-rank (R survival)...", flush=True)
    lvl = pd.concat([
        pats.assign(level="patient")[["netith_e", "os_days", "status", "level"]].rename(columns={"status": "os_event"}),
        tpdf.assign(level="timepoint")[["netith_e", "os_days", "status", "level"]].rename(columns={"status": "os_event"}),
    ])
    lvl.to_csv(os.path.join(DATA_ROOT, "external", "gse120575", "os_netith_levels.csv"))
    os.system("/opt/homebrew/bin/Rscript {ROOT}/code/R/15_gse120575_os_cox.R".format(ROOT=ROOT))
    rres = json.load(open(os.path.join(DATA_ROOT, "external", "gse120575", "os_cox_results.json")))
    rp = rres["patient"]; rt = rres["timepoint"]
    res_cox_pat = {"hr": rp["hr"], "ci95": rp["ci95"], "wald_p": rp["wald_p"]}
    res_lr_pat = {"p": rp["logrank_p"]}
    res_cox_tp = {"hr": rt["hr"], "ci95": rt["ci95"], "wald_p": rt["wald_p"]}
    res_lr_tp = {"p": rt["logrank_p"]}
    print(f"  patient Cox: HR={rp['hr']:.3f} CI={tuple(round(v,2) for v in rp['ci95'])} p={rp['wald_p']:.4f}", flush=True)
    print(f"  patient log-rank: p={rp['logrank_p']:.4f}", flush=True)
    print(f"  timepoint Cox: HR={rt['hr']:.3f} p={rt['wald_p']:.4f}; log-rank p={rt['logrank_p']:.4f}", flush=True)

    # R/NR contrast for reference (timepoint level, as in the replication stage)
    r = tpdf.loc[tpdf.response == "Responder", "netith_e"].dropna().values
    nr = tpdf.loc[tpdf.response == "Non-responder", "netith_e"].dropna().values
    d_ref = (r.mean() - nr.mean()) / np.sqrt(((len(r)-1)*r.var(ddof=1) + (len(nr)-1)*nr.var(ddof=1)) / (len(r)+len(nr)-2)) if len(r) >= 3 and len(nr) >= 3 else np.nan
    p_ref = stats.mannwhitneyu(r, nr, alternative="two-sided").pvalue if len(r) >= 3 and len(nr) >= 3 else np.nan
    print(f"  R/NR reference: d={d_ref:.3f} p={p_ref:.4f}", flush=True)

    # ---- [4] output ----
    print("[4/4] output...", flush=True)
    out = {"dataset": "GSE120575", "level": "patient+timepoint",
           "n_patients": int(len(pats)), "n_patient_events": int(pats.status.sum()),
           "n_timepoints": int(len(tpdf)), "n_tp_events": int(tpdf.status.sum()),
           "cox_patient": res_cox_pat, "logrank_patient": res_lr_pat,
           "cox_timepoint": res_cox_tp, "logrank_timepoint": res_lr_tp,
           "rnr_reference": {"cohen_d": float(d_ref) if np.isfinite(d_ref) else np.nan,
                             "mw_p": float(p_ref) if np.isfinite(p_ref) else np.nan,
                             "n_R": int(len(r)), "n_NR": int(len(nr))},
           "per_patient": pats.reset_index().rename(columns={"index": "patient"}).to_dict("records"),
           "per_timepoint": tpdf.to_dict("records"),
           "note": "OS from Cell 2018 Table S1 (supplement-10); immune-cell cohort; NetITH_E construction identical to main compendium."}
    out_json = os.path.join(ROOT, "results", "scrnaseq_validation", "gse120575_os_replication.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", out_json)


if __name__ == "__main__":
    if "--os" in sys.argv:
        run_os()
    else:
        run_replication()
