#!/usr/bin/env python3
"""run_descriptor_census.py — systematic descriptor census (four-test reporting protocol, round-4 extension #1).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-10
Pipeline: control / test stage — see repository README

Applies the manuscript's four-test protocol (Test 1 topology-vs-null; Test 2 scalar
baseline; Test 3 purity confounding; Test 4 fixed-network transfer) uniformly to a
panel of published network-state / heterogeneity descriptors, on GDSC (286 drugs;
ln(IC50) and AUC readouts; per-drug dynamic-range strata), TCGA (purity; Aran 2015
via TCGAbiolinks), and GSE25066 (neoadjuvant pCR transfer).

Descriptor panel:
  netith   NetITH (cached reference; 239-gene CollecTRI Laplacian entropy)
  sr       Signaling entropy (Teschendorff & Enver) — implementation validated
           against the manuscript's cached benchmark values (max |diff| 4e-16 on
           the 507 shared lines). Definition (reconstructed from the cached
           values): scaffold = 800 most variable GDSC genes; adjacency = |Pearson
           corr| over GDSC thresholded at the 97th percentile (diagonal 0);
           per-sample min-max-normalised sqrt-expression weights; row-normalised
           stochastic P; stationary distribution from eigh(P^T) with the
           first-|lambda-1|<0.01 eigenvector; SR = pi . H.
  dne      Differential network entropy (West et al. 2012) on the same
           correlation scaffold: p_ij ∝ x_j |corr_ij| over neighbours;
           S_i = -(1/log k_i) sum p_ij log p_ij; sample score = sum_i S_i.
  activity decoupleR-style TF-activity aggregate (per-sample variance of TF
           activity; activity = mean clipped target z per TF, CollecTRI prior)
  shannon  Per-sample Shannon entropy of expression (800-gene scaffold)
  mad800   Per-sample median absolute deviation (800-gene scaffold)
  cytotrace Bulk CytoTRACE proxy (mean of top-200 expressed genes, log space)

Test mappings: Test 1 (topology vs degree-preserving rewired null; 50 draws) for
sr/dne; regulon-membership-shuffle null (50 draws) for activity; N/A for scalar
descriptors; NetITH cites the manuscript's cached null. Test 2: per-descriptor
median rho vs the MAD scalar baseline (reported per descriptor). Test 3: TCGA
purity correlations (Aran 2015, 5 estimates). Test 4: GSE25066 neoadjuvant pCR
logistic OR per SD plus the ln(IC50)-vs-AUC readout contrast.

Inputs:
  data/gdsc/rna_expr.csv, ensg_symbol_map.csv, cell_annot.csv
  data/collectri_network.csv
  $NETITH_DATA_ROOT/gdsc_download/GDSC2_IC50_all.csv
  $NETITH_DATA_ROOT/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
  /tmp/census_aran_purity.csv (exported from TCGAbiolinks::Tumor.purity)
  data/external/gse25066/GSE25066_series_matrix.txt
  results/gdsc/gdsc_netith_cell_lines.csv, tcga/tcga_netith.csv
  results/gdsc/teschendorff/signaling_entropy_gdsc.csv (validation)
  results/neoadjuvant/gse25066_fixed_network_netith.csv (netith transfer reference)

Outputs (results/control/census/): descriptors_gdsc.csv, descriptors_tcga.csv,
descriptors_gse25066.csv, drug_assoc_<desc>.csv, null_sr_dne_activity.json,
purity_corr.json, gse25066_transfer.json.

Usage:
  NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_descriptor_census.py --stage all
  (stages: gdsc | null | tcga | gse)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.linalg import eigh
from scipy import sparse
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
CENSUS_DIR = ROOT / "results" / "control" / "census"
CENSUS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
RNG = np.random.default_rng(SEED)

# Derived caches under <repo>/data are produced by scripts/00_data/01_preprocess_data.py;
# each is overridable via its own env var so the release tree can reuse an existing cache.
GDSC_EXPR = Path(os.environ.get("NETITH_GDSC_EXPR", f"{ROOT}/data/gdsc/rna_expr.csv"))
GDSC_MAP = Path(os.environ.get("NETITH_GDSC_MAP", f"{ROOT}/data/gdsc/ensg_symbol_map.csv"))
GDSC_ANNOT = Path(os.environ.get("NETITH_GDSC_ANNOT", f"{ROOT}/data/gdsc/cell_annot.csv"))
COLLECTRI = Path(os.environ.get("NETITH_COLLECTRI_CSV", f"{ROOT}/data/collectri_network.csv"))
IC50_FILE = f"{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv"
TCGA_EXPR = f"{DATA_ROOT}/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
PURITY_CSV = "/tmp/census_aran_purity.csv"
GSE25066 = Path(os.environ.get("NETITH_GSE25066_MATRIX", f"{ROOT}/data/external/gse25066/GSE25066_series_matrix.txt"))
NETITH_GDSC = f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv"
NETITH_TCGA = f"{ROOT}/results/tcga/tcga_netith.csv"
NETITH_GSE = f"{ROOT}/results/neoadjuvant/gse25066_fixed_network_netith.csv"
SR_CACHED = f"{ROOT}/results/gdsc/teschendorff/signaling_entropy_gdsc.csv"

EPS = 1e-10
N_TOP = 800
CORR_THR_PCT = 97


def load_gdsc_expr():
    expr = pd.read_csv(GDSC_EXPR, index_col=0)
    gene_map = pd.read_csv(GDSC_MAP)
    gene_map.columns = ["ensg", "symbol"]
    ensg2sym = dict(zip(gene_map["ensg"].astype(str), gene_map["symbol"]))
    annot = pd.read_csv(GDSC_ANNOT, index_col=0)
    cell_map = {}
    for cel, row in annot.iterrows():
        cl = str(row.get("Characteristics.cell.line.", ""))
        if cl and cl not in ("nan", "NA"):
            cell_map[cel] = cl
    common = [c for c in expr.columns if c in cell_map]
    expr = expr[common].copy()
    expr.columns = [cell_map[c] for c in common]
    expr = expr.loc[:, ~expr.columns.duplicated(keep="first")]
    expr.index = expr.index.astype(str)
    matched = expr.index.isin(ensg2sym)
    expr = expr.loc[matched]
    expr.index = [ensg2sym[g] for g in expr.index]
    expr = expr[~expr.index.duplicated(keep="first")]
    return expr


def build_corr_scaffold(expr_gdsc):
    """800 most variable GDSC genes; adjacency = |Pearson corr| thresholded at
    the 97th percentile, diagonal 0 (the scaffold of the manuscript's cached SR
    benchmark; reconstructed and validated against those values)."""
    top_genes = expr_gdsc.var(axis=1).nlargest(N_TOP).index.tolist()
    corr = np.corrcoef(expr_gdsc.loc[top_genes].values)
    np.fill_diagonal(corr, 0)
    thr = np.percentile(np.abs(corr), CORR_THR_PCT)
    A = np.abs(corr)
    A[A < thr] = 0.0
    print(f"[scaffold] genes={len(top_genes)} edges={int((A > 0).sum() / 2)} "
          f"thr(|corr|)={thr:.4f}")
    return top_genes, A


def sr_census(A, x):
    """Exact cached-algorithm SR (eigh stationary distribution)."""
    n = len(x)
    xmin, xmax = x.min(), x.max()
    xn = (x - xmin) / (xmax - xmin) if xmax > xmin else np.ones(n)
    w = np.sqrt(np.maximum(xn, EPS))
    W = A * np.outer(w, w)
    rs = W.sum(axis=1)
    rs[rs < EPS] = 1.0
    P = W / rs[:, None]
    evals, evecs = eigh(P.T)
    idx = np.argmax(np.abs(evals - 1.0) < 0.01)
    if np.abs(evals[idx] - 1.0) > 0.1:
        idx = np.argmax(evals)
    pi = np.abs(evecs[:, idx])
    pi = pi / pi.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > EPS, np.log(P), 0.0)
    H = -(P * logP).sum(axis=1)
    return float(pi @ H)


def dne_census(A_sp, x, deg):
    """West 2012 differential network entropy (sparse, weighted |corr| edges)."""
    xmin, xmax = x.min(), x.max()
    xn = (x - xmin) / (xmax - xmin) if xmax > xmin else np.ones(len(x))
    xp = np.maximum(xn, EPS)
    rs = A_sp @ xp
    ok = (rs > EPS) & (deg > 1)
    rs_safe = np.where(ok, rs, 1.0)
    logx = np.log(xp)
    logrs = np.log(rs_safe)
    term = (A_sp @ (xp * logx)) - logrs * (A_sp @ xp)
    logk = np.log(np.maximum(deg, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = -(1.0 / rs_safe) * term / logk
    S[~ok] = 0.0
    return float(S.sum())


def shannon_per_sample(x):
    p = np.exp(x - x.max())
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mad_per_sample(x):
    return float(np.median(np.abs(x - np.median(x))))


def cytotrace_per_sample(x, n_top=200):
    return float(np.sort(x)[-n_top:].mean())


def activity_per_sample(Z, tf_targets):
    acts = np.array([Z[list(t)].mean() for t in tf_targets if len(t) > 0])
    return float(acts.var()) if len(acts) > 1 else np.nan


def build_tf_targets(gene_list):
    col = pd.read_csv(COLLECTRI)
    col.columns = [c.strip().lower() for c in col.columns]
    g2i = {g: i for i, g in enumerate(gene_list)}
    tf_targets = {}
    for src, tgt in zip(col["source"], col["target"]):
        if src in g2i and tgt in g2i:
            tf_targets.setdefault(g2i[src], set()).add(g2i[tgt])
    return [sorted(v) for v in tf_targets.values()]


def descriptor_matrix(expr, top_genes, A, name, tf_targets=None):
    """Per-sample descriptors. SR uses the eigh formulation on the correlation
    scaffold; DNE the sparse West formulation; scalars on the 800-gene set;
    activity (optional) on the full matrix z-scores with CollecTRI regulons."""
    in_scaff = [g for g in top_genes if g in expr.index]
    idx = [top_genes.index(g) for g in in_scaff]
    A_sub = A[np.ix_(idx, idx)]
    A_sp = sparse.csr_matrix(A_sub)
    deg = np.asarray(A_sp.sum(axis=1)).ravel()
    E = expr.loc[in_scaff].values.astype(float)
    Z_full = None
    if tf_targets is not None:
        X = expr.values.astype(float)
        mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-10
        Z_full = np.clip((X - mu) / sd, -3, 3)
    out = {}
    t0 = time.time()
    for j, cl in enumerate(expr.columns):
        x = E[:, j]
        row = {
            "sr": sr_census(A_sub, x),
            "dne": dne_census(A_sp, x, deg),
            "shannon": shannon_per_sample(x),
            "mad800": mad_per_sample(x),
            "cytotrace": cytotrace_per_sample(x),
        }
        if tf_targets is not None:
            row["activity"] = activity_per_sample(Z_full[:, j], tf_targets)
        out[cl] = row
        if (j + 1) % 1000 == 0:
            print(f"  {name} {j+1}/{len(expr.columns)} ({(time.time()-t0):.0f}s)")
    return pd.DataFrame(out).T


def drug_assoc(desc_series, ic50_mat, auc_mat):
    rows = []
    common = desc_series.index.intersection(ic50_mat.index)
    for d in ic50_mat.columns:
        y = ic50_mat[d].loc[common]
        ok = y.notna() & desc_series.loc[common].notna()
        if ok.sum() < 10:
            continue
        r_ic50, p_ic50 = spearmanr(desc_series.loc[common][ok], y[ok])
        ya = auc_mat[d].loc[common]
        oka = ya.notna() & ok
        r_auc = spearmanr(desc_series.loc[common][oka], ya[oka])[0] if oka.sum() >= 10 else np.nan
        rows.append({"drug": d, "rho_ic50": r_ic50, "p_ic50": p_ic50, "rho_auc": r_auc})
    df = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR across the per-drug Spearman p-values; the FDR<0.05
    # gate is the downstream significance / directional-consistency criterion.
    df["fdr_ic50"] = multipletests(df["p_ic50"], method="fdr_bh")[1]
    return df


def rewire_preserving(A, rng, n_attempts=8000):
    """Verified degree-preserving double-edge swap with an edge-count invariant
    (all four endpoints distinct, both cross-edges absent, edge list updated in
    lockstep). Supersedes rewire_weighted, which was found to leak edges on
    sparse graphs (573 -> 549-561 in synthetic tests; documented in
    run_control_netith_census_rewire_null.py)."""
    edges = np.argwhere(np.triu(A > 0, 1))
    m = len(edges)
    swaps_done = 0
    # Double-edge swap: pick two distinct edges (a0-a1, b0-b1), require all four
    # endpoints distinct and both cross-edges absent, then swap to (a0-b1, a1-b0).
    # This preserves every node's degree exactly (no self-loops, no parallel edges).
    for _ in range(n_attempts):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a0, a1 = edges[i]
        b0, b1 = edges[j]
        if len({a0, a1, b0, b1}) < 4:
            continue
        if A[a0, b1] or A[a1, b0]:
            continue
        A[a0, a1] = A[a1, a0] = 0
        A[b0, b1] = A[b1, b0] = 0
        A[a0, b1] = A[b1, a0] = 1
        A[a1, b0] = A[b0, a1] = 1
        edges[i] = [a0, b1]
        edges[j] = [a1, b0]
        swaps_done += 1
    assert int((A > 0).sum() / 2) == m, "edge count not preserved"
    return A, swaps_done


def rewire_weighted(A, rng, swaps=4000):
    """Degree-preserving double-edge swap on the undirected weighted graph;
    edge weights are shuffled onto the rewired topology (weight-gene
    association broken, mirroring the NetITH null's sign shuffle)."""
    edges = np.argwhere(np.triu(A > 0, 1))
    weights = np.array([A[i, j] for i, j in edges])
    m = len(edges)
    for _ in range(swaps):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b = edges[i], edges[j]
        if a[0] == b[1] or a[1] == b[0]:
            continue
        if A[a[0], b[1]] or A[a[1], b[0]]:
            continue
        A[a[0], a[1]] = A[a[1], a[0]] = 0
        A[b[0], b[1]] = A[b[1], b[0]] = 0
        A[a[0], b[1]] = A[b[1], a[0]] = 1
        A[a[1], b[0]] = A[b[0], a[1]] = 1
        edges[i] = [a[0], b[1]]
        edges[j] = [a[1], b[0]]
    rng.shuffle(weights)
    Aw = np.zeros_like(A)
    for (i, j), w in zip(edges, weights):
        Aw[i, j] = Aw[j, i] = w
    return Aw


def load_ic50():
    ic50 = pd.read_csv(IC50_FILE)
    ic50_mat = ic50.pivot_table(index="CELL_LINE_NAME", columns="DRUG_NAME",
                                values="LN_IC50", aggfunc="mean")
    auc_mat = ic50.pivot_table(index="CELL_LINE_NAME", columns="DRUG_NAME",
                               values="AUC", aggfunc="mean")
    return ic50_mat, auc_mat


# ─────────────────────────────────────────────────────────────────────────────
def stage_gdsc():
    expr = load_gdsc_expr()
    top_genes, A = build_corr_scaffold(expr)
    desc = descriptor_matrix(expr, top_genes, A, "GDSC",
                             tf_targets=build_tf_targets(expr.index))
    netith = pd.read_csv(NETITH_GDSC, index_col=0)["NetITH"]
    desc["netith"] = netith.reindex(desc.index)
    desc.to_csv(CENSUS_DIR / "descriptors_gdsc.csv")
    # validation gate: SR must reproduce the cached benchmark values
    cached_sr = pd.read_csv(SR_CACHED).set_index("cell_line")["signaling_entropy"]
    shared = desc.index.intersection(cached_sr.index)
    maxdiff = (desc.loc[shared, "sr"] - cached_sr[shared]).abs().max()
    print(f"[gate] SR vs cached (n={len(shared)}): max|diff|={maxdiff:.2e}")
    assert maxdiff < 1e-6, "SR implementation does not reproduce cached values"
    ic50_mat, auc_mat = load_ic50()
    for name in ["sr", "dne", "shannon", "mad800", "cytotrace", "activity", "netith"]:
        da = drug_assoc(desc[name], ic50_mat, auc_mat)
        da.to_csv(CENSUS_DIR / f"drug_assoc_{name}.csv", index=False)
        print(f"[gdsc] {name}: ic50 med={da['rho_ic50'].median():.4f} "
              f"(pos {(da['rho_ic50']>0).sum()}/{len(da)}, fdr05 {(da['fdr_ic50']<0.05).sum()}), "
              f"auc med={da['rho_auc'].median():.4f}")


def stage_null():
    expr = load_gdsc_expr()
    top_genes, A = build_corr_scaffold(expr)
    in_scaff = [g for g in top_genes if g in expr.index]
    idx = [top_genes.index(g) for g in in_scaff]
    A_sub = A[np.ix_(idx, idx)]
    A_sp = sparse.csr_matrix(A_sub)
    deg = np.asarray(A_sp.sum(axis=1)).ravel()
    E = expr.loc[in_scaff].values.astype(float)
    cells = list(expr.columns)
    # activity null: regulon-membership shuffle (gene labels permuted)
    X = expr.values.astype(float)
    mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-10
    Z_full = np.clip((X - mu) / sd, -3, 3)
    tf_targets = build_tf_targets(expr.index)
    gene_perm_base = np.arange(X.shape[0])
    ic50_mat, _ = load_ic50()
    N = 50
    results = {"sr": [], "dne": [], "activity": []}
    for k in range(N):
        Aw, _ = rewire_preserving(A_sub.copy(), RNG)
        Aw_sp = sparse.csr_matrix(Aw)
        desc = {}
        for j, cl in enumerate(cells):
            x = E[:, j]
            desc[cl] = {"sr": sr_census(Aw, x),
                        "dne": dne_census(Aw_sp, x, deg)}
            if tf_targets is not None:
                perm = RNG.permutation(gene_perm_base)
                tgt = [[perm[t] for t in tt] for tt in tf_targets]
                acts = np.array([Z_full[t, j].mean() for t in tgt if len(t) > 0])
                desc[cl]["activity"] = float(acts.var())
        df = pd.DataFrame(desc).T
        for nm in results:
            da = drug_assoc(df[nm], ic50_mat, ic50_mat)
            results[nm].append(float(da["rho_ic50"].median()))
        print(f"  null {k+1}/{N}: " + " ".join(f"{nm}={results[nm][-1]:.4f}"
                                              for nm in results))
    out = {nm: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "range": [float(min(v)), float(max(v))], "n": N,
                "draws": [float(x) for x in v]} for nm, v in results.items()}
    json.dump(out, open(CENSUS_DIR / "null_sr_dne_activity.json", "w"), indent=2)
    print("[null] done")


def stage_tcga():
    desc_file = CENSUS_DIR / "descriptors_tcga.csv"
    if desc_file.exists():
        desc = pd.read_csv(desc_file, index_col=0)
    else:
        tcg = pd.read_csv(TCGA_EXPR, sep="\t", compression="gzip", index_col=0)
        tcg.index = [str(i).split(".")[0] for i in tcg.index]
        tcg = tcg[~tcg.index.duplicated(keep="first")]
        expr = load_gdsc_expr()
        top_genes, A = build_corr_scaffold(expr)
        desc = descriptor_matrix(tcg, top_genes, A, "TCGA")  # no activity on TCGA
        desc["netith"] = pd.read_csv(NETITH_TCGA, index_col=0)["netith_bulk"].reindex(desc.index)
        desc.to_csv(desc_file)
    purity = pd.read_csv(PURITY_CSV, decimal=",")
    pur = purity.set_index("Sample.ID")[["ESTIMATE", "ABSOLUTE", "LUMP", "IHC", "CPE"]]
    # normalise purity IDs to 4-segment xena form (drop aliquot suffix) and
    # average across aliquots of the same tumour
    pur = pur.reset_index()
    pur["Sample.ID"] = pur["Sample.ID"].str[:15]
    pur = pur.groupby("Sample.ID", as_index=True).mean(numeric_only=True)
    corr_out = {}
    for nm in desc.columns:
        common = desc.index.intersection(pur.index)
        row = {}
        for pn in pur.columns:
            v = pur[pn].loc[common]
            ok = v.notna() & desc[nm].loc[common].notna()
            row[pn] = float(spearmanr(desc[nm].loc[common][ok], v[ok])[0]) if ok.sum() >= 100 else None
        corr_out[nm] = {**row, "n": int(len(common))}
    json.dump(corr_out, open(CENSUS_DIR / "purity_corr.json", "w"), indent=2)
    for nm, row in corr_out.items():
        print(f"[tcga] {nm}: n={row['n']} " +
              " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "n" and v is not None))


def stage_gse():
    desc_file = CENSUS_DIR / "descriptors_gse25066.csv"
    if desc_file.exists():
        desc = pd.read_csv(desc_file, index_col=0)
    else:
        import gzip, re
        path = GSE25066 if Path(GSE25066).exists() else GSE25066 + ".gz"
        opener = gzip.open if str(path).endswith(".gz") else open
        txt = opener(path, "rt").read()
        expr = load_gdsc_expr()
        top_genes, A = build_corr_scaffold(expr)
        m = re.search(r"!series_matrix_table_begin\n(.*?)!series_matrix_table_end", txt, re.S)
        assert m is not None, "series-matrix table block not found"
        rows = [l.split("\t") for l in m.group(1).strip().split("\n")]
        header = rows[0]
        data = rows[1:]
        idx = [r[0].strip('"') for r in data]
        vals = np.array([[float(v) if v not in ("", "NA") else np.nan for v in r[1:]] for r in data])
        mat = pd.DataFrame(vals, index=idx, columns=[h.strip('"') for h in header[1:]])
        # GPL96 SOFT platform: table after !platform_table_begin
        with open(os.environ.get("NETITH_GPL96_PLATFORM", f"{ROOT}/data/external/gse25066/GPL96_full.txt")) as fh:
            lines = fh.readlines()
        beg = next(i for i, l in enumerate(lines) if l.startswith("!platform_table_begin"))
        tbl = [l.rstrip("\r\n") for l in lines[beg + 1:]]
        cols = tbl[0].split("\t")
        gpl = pd.DataFrame([r.split("\t") for r in tbl[1:]], columns=cols)
        gpl = gpl[["ID", "Gene Symbol"]].dropna()
        probe2sym = dict(zip(gpl["ID"], gpl["Gene Symbol"]))
        mat = mat[mat.index.isin(probe2sym)]
        mat.index = [probe2sym[p] for p in mat.index]
        mat = mat[mat.index != "---"]
        mat = mat[~mat.index.duplicated(keep="first")]
        desc = descriptor_matrix(mat, top_genes, A, "GSE25066")
        desc.to_csv(desc_file)
    clin = pd.read_csv(NETITH_GSE, index_col=0)
    import statsmodels.api as sm
    out = {}
    for nm in desc.columns:
        df = desc[nm].to_frame("d").join(clin["pcr_bin"], how="inner").dropna()
        if len(df) < 50:
            out[nm] = {"or": None, "n": int(len(df))}
            continue
        z = (df["d"] - df["d"].mean()) / df["d"].std()
        m = sm.Logit(df["pcr_bin"], sm.add_constant(z)).fit(disp=0)
        ci = m.conf_int().loc[z.name]
        out[nm] = {"or": float(np.exp(m.params[z.name])),
                   "ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                   "p": float(m.pvalues[z.name]), "n": int(len(df))}
    # netith reference from the manuscript's fixed-network recomputation
    out["netith"] = {"or": 0.86, "ci": [0.64, 1.14], "p": 0.29,
                     "n": 306, "source": "manuscript cached (fixed 239-gene network)"}
    json.dump(out, open(CENSUS_DIR / "gse25066_transfer.json", "w"), indent=2)
    for nm, v in out.items():
        print(f"[gse] {nm}: OR={v.get('or')} p={v.get('p')} n={v.get('n')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    if args.stage in ("gdsc", "all"):
        stage_gdsc()
    if args.stage in ("null", "all"):
        stage_null()
    if args.stage in ("tcga", "all"):
        stage_tcga()
    if args.stage in ("gse", "all"):
        stage_gse()
    print("[census] stage(s) done")
