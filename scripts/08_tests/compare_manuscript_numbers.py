"""
compare_manuscript_numbers.py — cross-check manuscript statistics against authoritative output files.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-05
Inputs  : results/MANUSCRIPT_DRAFT.md (manuscript draft),
          results/**/*.json (authoritative analysis outputs),
          results/NUMBER_TRUTH_TABLE.csv
Outputs : results/data_verification_report.md
Pipeline: replication stage — see repository README
"""
import json, re, glob, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD = os.path.join(ROOT, "results", "MANUSCRIPT_DRAFT.md")
OUT = os.path.join(ROOT, "results", "data_verification_report.md")


def num(text):
    """Extract the first numeric literal from a string (comma/minus normalised)."""
    t = text.replace(",", "").replace("−", "-")
    m = re.search(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    return float(m.group(0)) if m else None


# 1) collect authoritative numbers from json/csv outputs
sources = {}
for p in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
    try:
        raw = open(p).read()
        base = os.path.relpath(p, ROOT)
        # harvest numeric values under statistical key names (cohen_d, mw_p, hr, ...)
        for m in re.finditer(r'("(?:cohen_d|mw_p|hr|wald_p|or|p|p_value|rho|median_rho|fdr|ci95|logrank_p|d|r2|auc)[^"]*"\s*:\s*)(-?\d+\.?\d*(?:[eE][-+]?\d+)?)', raw):
            key = f"{m.group(2).rstrip('.')}"
            sources.setdefault(key[:10], []).append(base)
    except Exception:
        pass
# also NUMBER_TRUTH_TABLE
ntt = os.path.join(ROOT, "results", "NUMBER_TRUTH_TABLE.csv")
if os.path.exists(ntt):
    try:
        t = pd.read_csv(ntt)
        for col in t.columns:
            for v in t[col].dropna().astype(str):
                vv = num(v)
                if vv is not None:
                    sources.setdefault(f"{vv:.6g}"[:10], []).append("NUMBER_TRUTH_TABLE.csv")
    except Exception as e:
        pass

# 2) extract numbers with context from the manuscript
txt = open(MD).read()
rows = []
for m in re.finditer(r'([^\n]{0,90}?)(\d+\.\d+(?:[eE][-+]?\d+)?)([^\n]{0,90})', txt):
    ctx_before = m.group(1)[-70:]
    val = float(m.group(2))
    ctx_after = m.group(3)[:70]
    if 0.0001 < val < 1000 and (val < 0.05 or val > 1 or "0." in m.group(2)):
        matched = any(abs(val - num(k)) / val < 0.011 for k in sources if num(k) is not None) if False else None
        # substring search: number appears in a json file verbatim?
        found_in = []
        q = f"{val:.6g}"
        for src_file in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
            if q in open(src_file).read():
                found_in.append(os.path.relpath(src_file, ROOT))
        status = "OK" if found_in else "UNVERIFIED"
        rows.append((status, m.group(2), ctx_before.strip()[-60:], ctx_after.strip()[:60], "; ".join(found_in[:2])))

# 3) write report (flag only; no manuscript edits)
with open(OUT, "w") as f:
    f.write("# 文稿数据核对报告(只标不改)\n\n")
    f.write(f"核对: {MD}\n共抽取 {len(rows)} 个候选统计数字;OK=可在 results json 中逐字检索到,UNVERIFIED=未检索到(可能来自 csv/文稿汇总/圆整)。\n\n")
    f.write("| 状态 | 数字 | 前文 | 后文 | 来源文件 |\n|---|---|---|---|---|\n")
    for st, v, cb, ca, srcs in rows:
        f.write(f"| {st} | {v} | {cb} | {ca} | {srcs} |\n")
ok = sum(1 for r in rows if r[0] == "OK")
print(f"written {OUT}: {ok}/{len(rows)} verbatim-verifiable; {len(rows)-ok} flagged UNVERIFIED")
