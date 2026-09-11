#!/usr/bin/env python3
"""run_neoadjuvant_validation.py — GSE25066 neoadjuvant chemotherapy validation of NetITH.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/external/gse25066/{GSE25066_series_matrix.txt.gz, GPL96_full.txt}; data/collectri_network.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/neoadjuvant/{gse25066_netith_pcr.csv, gse25066_summary.json}
Pipeline: clinical stage — see repository README
"""
import gzip, io, re, json, time, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, rankdata

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
DATA = DATA_ROOT / "external" / "gse25066"
OUT = ROOT / "results" / "neoadjuvant"
OUT.mkdir(parents=True, exist_ok=True)


def vn_entropy(L):
    ev = np.linalg.eigvalsh(L)
    ev = ev[ev > 1e-12]
    s = ev.sum()
    if s <= 0:
        return 0.0
    rho = ev / s
    return float(-np.sum(rho * np.log2(rho)))


def parse_series_matrix(path):
    """Return (clin dict, expr_df: probes x samples)"""
    titles, accessions, char_lines = None, None, []
    with gzip.open(path, 'rt') as f:
        in_table = False
        header = None
        rows = []
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('!Sample_title'):
                titles = [p.strip().strip('"') for p in line.split('\t')[1:]]
            elif line.startswith('!Sample_geo_accession'):
                accessions = [p.strip().strip('"') for p in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics'):
                char_lines.append([p.strip().strip('"') for p in line.split('\t')[1:]])
            elif line.startswith('!series_matrix_table_begin'):
                in_table = True
            elif line.startswith('!series_matrix_table_end'):
                break
            elif in_table:
                cells = [c.strip().strip('"') for c in line.split('\t')]
                if header is None:
                    header = cells
                else:
                    rows.append(cells)
    # characteristics: one attribute per line ("key: value" format across all samples)
    clin = {'title': titles, 'accession': accessions}
    for cl in char_lines:
        first = cl[0]
        if ':' in first:
            key = first.split(':', 1)[0].strip()
            vals = [v.split(':', 1)[1].strip() if ':' in v else v for v in cl]
            clin[key] = vals
    expr = pd.DataFrame(rows)
    expr.columns = header
    expr = expr.set_index(expr.columns[0])
    # drop trailing empty columns (series-matrix files often carry extra tabs)
    expr = expr.loc[:, [c != '' for c in expr.columns]]
    # pad clinical fields to full length (some rows may lack trailing samples)
    maxlen = max((len(v) for v in clin.values()), default=0)
    for k, v in clin.items():
        if len(v) < maxlen:
            clin[k] = v + [''] * (maxlen - len(v))
    return clin, expr


def parse_gpl96_annot(path):
    """probe -> gene symbol mapping (located via the 'Gene Symbol' column)"""
    mapping = {}
    with open(path, 'r', errors='replace') as f:
        in_table = False
        header = None
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('!platform_table_begin'):
                in_table = True
                continue
            if line.startswith('!platform_table_end'):
                break
            if in_table:
                parts = line.split('\t')
                if header is None:
                    header = parts
                    sym_idx = header.index('Gene Symbol') if 'Gene Symbol' in header else 1
                    continue
                if len(parts) > sym_idx:
                    mapping[parts[0]] = parts[sym_idx]
    return mapping, header


def main():
    t0 = time.time()
    print("[1/5] parsing series matrix", flush=True)
    clin, expr = parse_series_matrix(DATA / "GSE25066_series_matrix.txt.gz")
    print(f"  clinical fields: {list(clin.keys())}", flush=True)
    print(f"  samples: {expr.shape[1]}, probes: {expr.shape[0]}", flush=True)

    print("[2/5] parsing GPL96 annotation", flush=True)
    mapping, hdr = parse_gpl96_annot(DATA / "GPL96_full.txt")
    print(f"  probe mapping: {len(mapping)} ({time.time()-t0:.0f}s)", flush=True)

    print("[3/5] gene mapping + NetITH computation", flush=True)
    expr_float = expr.apply(pd.to_numeric, errors='coerce')
    probes = expr_float.index
    symbols = [mapping.get(p, '') for p in probes]
    expr_float['symbol'] = symbols
    expr_float = expr_float[expr_float['symbol'] != '']
    # average multiple probes per gene
    gexpr = expr_float.groupby('symbol').mean()
    print(f"  genes: {gexpr.shape[0]}, samples: {gexpr.shape[1]}", flush=True)

    net = pd.read_csv(DATA_ROOT / "collectri_network.csv")
    genes_in = set(gexpr.index)
    assert len(genes_in) > 100, f"gene mapping failed: {len(genes_in)}"
    net_f = net[net['source'].isin(genes_in) & net['target'].isin(genes_in)]
    print(f"  CollecTRI edges in data: {len(net_f)}; TFs: {net_f['source'].nunique()}", flush=True)

    # align with the project bulk pipeline (run_tcga_validation.py): top-300 variable CollecTRI genes
    ct_genes = [g for g in gexpr.index if g in genes_in]
    if len(ct_genes) > 300:
        vars_ = gexpr.loc[ct_genes].var(axis=1).sort_values(ascending=False)
        edge_genes = vars_.index[:300].tolist()
    else:
        edge_genes = ct_genes
    net_f = net_f[net_f['source'].isin(edge_genes) & net_f['target'].isin(edge_genes)]
    gexpr = gexpr.loc[edge_genes]
    X = gexpr.values  # genes × samples
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-10)
    Z = np.clip(Z, -3, 3)
    gi = {g: i for i, g in enumerate(gexpr.index)}
    edges = [(gi[r['source']], gi[r['target']], abs(float(r['weight']))) for _, r in net_f.iterrows()]
    print(f"  genes involved in edges: {len(edge_genes)}", flush=True)

    n_samples = X.shape[1]
    netith = np.full(n_samples, np.nan)
    for s in range(n_samples):
        A = np.zeros((len(edge_genes), len(edge_genes)))
        for (src, dst, w) in edges:
            A[src, dst] = w * abs(Z[src, s]) * abs(Z[dst, s])
        A = A + A.T  # FIX(eigvalsh-audit 2026-08-16): symmetrize directed TF->target adjacency before Laplacian (eigvalsh silently used one triangle)
        deg = A.sum(1)
        tr = deg.sum()
        if tr < 1e-10:
            netith[s] = 0.0
            continue
        netith[s] = vn_entropy(np.diag(deg) - A)
    print(f"  NetITH: n={np.isfinite(netith).sum()}, median={np.nanmedian(netith):.3f} ({time.time()-t0:.0f}s)", flush=True)

    print("[4/5] clinical merge and statistics", flush=True)
    accessions = [str(s) for s in gexpr.columns]
    cdf = pd.DataFrame({'accession': accessions, 'title': clin.get('title', [''] * len(accessions)),
                        'NetITH': netith})
    for k, v in clin.items():
        if k in ('title', 'accession'):
            continue
        if len(v) == len(accessions):
            cdf[k] = v
    cdf = cdf.dropna(subset=['NetITH'])
    print(f"  clinical columns: {list(cdf.columns)[:25]}", flush=True)

    # pCR outcome
    pcr_map = {'pCR': 1, 'RD': 0}
    cdf['pcr_bin'] = cdf['pathologic_response_pcr_rd'].map(pcr_map) if 'pathologic_response_pcr_rd' in cdf.columns else np.nan
    cdf = cdf.dropna(subset=['pcr_bin'])
    print(f"  analysable patients: {len(cdf)} (pCR: {int(cdf['pcr_bin'].sum())})", flush=True)

    # statistics
    import statsmodels.api as sm
    cdf['netith_z'] = (cdf['NetITH'] - cdf['NetITH'].mean()) / cdf['NetITH'].std()
    # Logistic regression of pCR on z-scored NetITH: the exponentiated coefficient is the
    # odds ratio (OR) per 1-SD increase in NetITH — the within-system resistance/protection test.
    m = sm.Logit(cdf['pcr_bin'], sm.add_constant(cdf['netith_z'])).fit(disp=0)
    or_v = np.exp(m.params['netith_z'])
    lo, hi = np.exp(m.conf_int().loc['netith_z'])
    print(f"  [primary] NetITH per SD -> pCR: OR={or_v:.3f} [{lo:.3f}-{hi:.3f}] p={m.pvalues['netith_z']:.4f}", flush=True)
    r_, p_ = mannwhitneyu(cdf.loc[cdf['pcr_bin'] == 1, 'NetITH'], cdf.loc[cdf['pcr_bin'] == 0, 'NetITH'])
    print(f"  MW: p={p_:.4f}", flush=True)
    tert = pd.qcut(cdf['NetITH'], 3, labels=['low', 'mid', 'high'])
    rates = cdf.groupby(tert, observed=True)['pcr_bin'].agg(['mean', 'count'])
    print(f"  pCR rate by tertile:\n{rates.to_string()}", flush=True)

    # ER stratification
    if 'er_status_ihc_esr1_for indeterminate' in cdf.columns:
        cdf['er_pos'] = cdf['er_status_ihc_esr1_for indeterminate'].eq('P')
        for er in [True, False]:
            sub = cdf[cdf['er_pos'] == er]
            if len(sub) >= 30:
                m2 = sm.Logit(sub['pcr_bin'], sm.add_constant(sub['netith_z'])).fit(disp=0)
                print(f"  ER{'+' if er else '-'}: n={len(sub)}, OR={np.exp(m2.params['netith_z']):.3f} p={m2.pvalues['netith_z']:.4f}", flush=True)

    # source stratification (ISPY vs MDACC)
    if 'source' in cdf.columns:
        for src in sorted(cdf['source'].dropna().unique()):
            sub = cdf[cdf['source'] == src]
            if len(sub) >= 30:
                m3 = sm.Logit(sub['pcr_bin'], sm.add_constant(sub['netith_z'])).fit(disp=0)
                print(f"  source={src}: n={len(sub)}, OR={np.exp(m3.params['netith_z']):.3f} p={m3.pvalues['netith_z']:.4f}", flush=True)

    print("[5/5] saving", flush=True)
    cdf.to_csv(OUT / "gse25066_netith_pcr.csv", index=False)
    summary = {
        'cohort': 'GSE25066', 'n': int(len(cdf)), 'n_pcr': int(cdf['pcr_bin'].sum()),
        'netith_median': float(cdf['NetITH'].median()),
        'or_per_sd': float(or_v), 'ci': [float(lo), float(hi)], 'p': float(m.pvalues['netith_z']),
        'mw_p': float(p_), 'pcr_rate_high_vs_low_tertile': float(rates['mean'].iloc[-1] - rates['mean'].iloc[0]),
        'n_edges': len(edges), 'n_genes': X.shape[0],
    }
    (OUT / "gse25066_summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
