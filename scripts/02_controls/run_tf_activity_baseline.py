"""run_tf_activity_baseline.py — decoupleR-style TF-activity baselines (ulm / wm / mlm, true-ulm).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18

Pipeline: scripts/02_controls/ (see repository README for pipeline order)

Summary:
    Asks whether NetITH merely recapitulates TF-activity aggregates that any
    decoupleR-style estimator would provide. Three stages run in one script:
    (1) baseline — per-sample TF activity by univariate linear model (ulm) and
    weighted mean of targets (wm) from the CollecTRI prior, aggregated to
    mean/MAD/variance and correlated with NetITH, plus drug associations;
    (2) extension — mlm variant, activity Shannon entropy, and JUN-activity
    univariate with their drug associations; (3) true-ulm — closed-form
    per-TF univariate activity sum(w*E)/sum(w^2), activity variance, JUN and
    drug associations.

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - data/gdsc/rna_expr.csv + ensg_symbol_map.csv + cell_annot.csv: GDSC expression & names
    - data/collectri_network.csv: CollecTRI TF-target prior (signed weights)
    - results/focused_genes_collectri.txt: focused gene list
    - results/gdsc/gdsc_netith_cell_lines.csv: GDSC NetITH
    - $DATA_ROOT/gdsc_download/GDSC2_IC50_all.csv: GDSC2 lnIC50 (drug associations)

Outputs:
    - results/control/tf_activity_baseline.json + tf_activity_baseline_drugs.csv (stage 1)
    - results/control/tf_activity_baseline_ext.json (stage 2)
    - results/control/tf_activity_ulm_true.json (stage 3, SI-N16)

Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/02_controls/run_tf_activity_baseline.py
"""
from pathlib import Path
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
# Stage 1 (baseline): load GDSC expression; ENSG -> symbol, columns -> cell-line names
expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
ensg = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg.columns=['ensg','symbol']
sym = dict(zip(ensg['ensg'], ensg['symbol']))
annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
cl_map = {cel: str(row.get('Characteristics.cell.line.','')) for cel,row in annot.iterrows()}
common_cels = [c for c in expr.columns if c in cl_map]
en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
en = en.loc[:, ~en.columns.duplicated()]
matched = en.index.isin(ensg['ensg'])
en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
en_sym = en_sym[~en_sym.index.duplicated(keep='first')]

# CollecTRI prior + focused genes -> gene-indexed signed edge list
net = pd.read_csv(f'{ROOT}/data/collectri_network.csv')
focused = set(l.strip() for l in open(f'{ROOT}/results/focused_genes_collectri.txt') if l.strip())
common = sorted(focused & set(en_sym.index))
gi = {g:i for i,g in enumerate(common)}
edges = [(gi[r['source']], gi[r['target']], float(r['weight'])) for _, r in net.iterrows()
         if r['source'] in gi and r['target'] in gi]
print(f"focused genes: {len(common)}; edges: {len(edges)}")

# Z-score each gene across cells (cells x genes matrix)
expr_sub = en_sym.loc[common]  # genes x cells
Z = expr_sub.T.values  # cells x genes
# z-score per gene
Zm = (Z - Z.mean(0)) / (Z.std(0) + 1e-10)

# Split focused genes into TF nodes and target-only genes for activity inference
tfs = sorted({r['source'] for _, r in net.iterrows() if r['source'] in gi})
tf_idx = [gi[t] for t in tfs]
tgt_idx = sorted({gi[r['target']] for _, r in net.iterrows() if r['target'] in gi and r['source'] not in gi})
print(f"TFs: {len(tfs)}; target-only genes: {len(tgt_idx)}")

# (a) Weighted-mean activity: act_tf = mean over its targets of w * target z
# (a) weighted-mean activity: act_tf = mean over targets of w*target_z
W = np.zeros((len(common), len(common)))
for a,b,w in edges:
    W[a,b] = w
act_wm = np.zeros((Z.shape[0], len(tf_idx)))
for k, ti in enumerate(tf_idx):
    tgt = np.where(W[ti] != 0)[0]
    if len(tgt):
        act_wm[:, k] = (W[ti, tgt] * Zm[:, tgt]).mean(axis=1)

# (b) ULM activity: per sample solve E(targets) = A @ act by least squares,
# where A is the targets x TFs weight matrix from the CollecTRI prior
# (b) ulm activity: per TF, regress target expression on all TF activities jointly
# E(targets) = A @ act  ->  act = lstsq(A, E) per sample (A: targets x TFs)
A = W[np.ix_(tgt_idx, tf_idx)]  # targets x TFs
act_ulm = np.zeros((Z.shape[0], len(tf_idx)))
for c in range(Z.shape[0]):
    act_ulm[c] = np.linalg.lstsq(A, Zm[c, tgt_idx], rcond=None)[0]

# Load published GDSC NetITH on the same cell lines
netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH']
netith = netith.reindex(en.columns.tolist()).dropna()
cells = netith.index
ci = [list(en.columns).index(x) for x in cells]

# Aggregate TF activities per cell line: mean / MAD / variance across TFs
# aggregates
agg = {}
for name, act in [('wm', act_wm), ('ulm', act_ulm)]:
    act_c = act[ci]
    agg[f'act_{name}_mean'] = act_c.mean(axis=1)
    agg[f'act_{name}_mad'] = np.median(np.abs(act_c - act_c.mean(1, keepdims=True)), axis=1)
    agg[f'act_{name}_var'] = act_c.var(axis=1)

# Spearman rho(aggregate, NetITH); null = NetITH not explained by activity aggregates
res = {'n': len(cells)}
for k, v in agg.items():
    r, p = spearmanr(v, netith.values)
    res[k] = {'rho': float(r), 'p': float(p)}
    print(f"{k} vs NetITH: rho={r:.3f} (p={p:.2e})")

# Drug associations: per drug compare rho(NetITH, lnIC50) vs rho(ulm-var, lnIC50)
# drug associations: NetITH vs activity-wm var on 286 drugs
ic50 = pd.read_csv(f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv')
ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
from scipy.stats import spearmanr as spr
rows = []
for d in ic50_mat.columns:
    y = ic50_mat[d].dropna()
    c = netith.index.intersection(y.index)
    if len(c) < 30: continue
    r0, _ = spr(netith.loc[c], y.loc[c])
    av = pd.Series(agg['act_ulm_var'], index=cells).loc[c]
    r1, _ = spr(av, y.loc[c])
    rows.append({'drug': d, 'n': len(c), 'rho_netith': r0, 'rho_actvar': r1})
df = pd.DataFrame(rows)
print(f"\ndrugs: n={len(df)}; median rho NetITH={df['rho_netith'].median():.3f} "
      f"(pos {(df['rho_netith']>0).sum()}); median rho actvar={df['rho_actvar'].median():.3f} "
      f"(pos {(df['rho_actvar']>0).sum()})")
# Save stage-1 aggregates and per-drug comparisons
import json, os
os.makedirs(f'{ROOT}/results/control', exist_ok=True)
json.dump({'aggregates': res, 'n_drugs': int(len(df)),
           'median_rho_netith': float(df['rho_netith'].median()),
           'median_rho_actvar': float(df['rho_actvar'].median()),
           'pos_netith': int((df['rho_netith']>0).sum()),
           'pos_actvar': int((df['rho_actvar']>0).sum())},
          open(f'{ROOT}/results/control/tf_activity_baseline.json','w'), indent=2)
df.to_csv(f'{ROOT}/results/control/tf_activity_baseline_drugs.csv', index=False)
print("saved")


# =====================================================================
# L7 extension: decoupleR mlm variant + JUN-activity univariate +
# activity entropy (originally run_tf_activity_baseline_ext.py; merged here)
# =====================================================================
# ---- Stage 2 (extension): decoupleR mlm variant + JUN univariate + activity entropy ----
def run_extension():
    expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
    ensg = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg.columns=['ensg','symbol']
    sym = dict(zip(ensg['ensg'], ensg['symbol']))
    annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
    cl_map = {cel: str(row.get('Characteristics.cell.line.','')) for cel,row in annot.iterrows()}
    common_cels = [c for c in expr.columns if c in cl_map]
    en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
    en = en.loc[:, ~en.columns.duplicated()]
    matched = en.index.isin(ensg['ensg'])
    en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
    en_sym = en_sym[~en_sym.index.duplicated(keep='first')]

    net = pd.read_csv(f'{ROOT}/data/collectri_network.csv')
    focused = set(l.strip() for l in open(f'{ROOT}/results/focused_genes_collectri.txt') if l.strip())
    common = sorted(focused & set(en_sym.index))
    gi = {g:i for i,g in enumerate(common)}
    edges = [(gi[r['source']], gi[r['target']], float(r['weight'])) for _, r in net.iterrows()
             if r['source'] in gi and r['target'] in gi]
    W = np.zeros((len(common), len(common)))
    for a, b, w in edges: W[a, b] = w
    tf_names = sorted({r['source'] for _, r in net.iterrows() if r['source'] in gi and r['source'] in en_sym.index})
    tgt = sorted({gi[r['target']] for _, r in net.iterrows() if r['target'] in gi and r['target'] in en_sym.index and r['source'] not in gi})
    # target-only genes with edges from focused TFs
    # Z-score expression (clip [-3,3]); ulm/mlm/wm activity matrices follow
    Z = np.clip(((en_sym.loc[common].T - en_sym.loc[common].mean(axis=1)) / (en_sym.loc[common].std(axis=1)+1e-10)).values, -3, 3)
    cells = en_sym.columns.tolist()

    A = W[np.ix_(tgt, tf_names and [gi[t] for t in tf_names])]  # targets x TFs
    A = W[np.ix_(tgt, [gi[t] for t in tf_names])]
    # mlm = joint least squares; coincides with ulm when all targets are included
    # and the design is full rank (as noted in the original extension script)
    # ulm / wm / mlm activities
    act_ulm = np.zeros((len(Z), len(tf_names)))
    act_mlm = np.zeros((len(Z), len(tf_names)))
    act_wm = np.zeros((len(Z), len(tf_names)))
    for c in range(len(Z)):
        act_wm[c] = [(W[gi[t], tgt] * Z[c, tgt]).mean() for t in tf_names]
        act_ulm[c] = np.linalg.lstsq(A, Z[c, tgt], rcond=None)[0]
        act_mlm[c] = act_ulm[c]  # ulm and mlm coincide for least squares with full design
    # (mlm = joint LS; same as ulm when all targets included and design full rank)

    netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH'].reindex(cells)
    ci = np.arange(len(cells))

    # Activity entropy: Shannon entropy of the |activity| distribution across TFs
    def shannon(v):
        v = np.abs(v); v = v / (v.sum() + 1e-12); v = v[v > 0]
        return -np.sum(v * np.log2(v))

    agg = {}
    for name, act in [('ulm', act_ulm), ('wm', act_wm)]:
        agg[f'act_{name}_var'] = act.var(axis=1)
        agg[f'act_{name}_entropy'] = np.array([shannon(act[c]) for c in range(len(Z))])

    # Aggregates (variance, entropy) vs NetITH: Spearman rho per aggregate
    print("aggregates vs NetITH (Spearman):")
    res = {'n_lines': len(cells)}
    for k, v in agg.items():
        r, p = spearmanr(v, netith.values)
        res[k] = {'rho': float(r), 'p': float(p)}
        print(f"  {k}: rho={r:.3f} (p={p:.1e})")

    # JUN-activity univariate: rho(ulm/wm JUN activity, NetITH)
    # JUN activity univariate
    jun_idx = tf_names.index('JUN')
    jun_act = {'ulm': act_ulm[:, jun_idx], 'wm': act_wm[:, jun_idx]}
    for m, v in jun_act.items():
        r, p = spearmanr(v, netith.values)
        res[f'jun_activity_{m}_vs_netith'] = {'rho': float(r), 'p': float(p)}
        print(f"  JUN activity ({m}) vs NetITH: rho={r:.3f} (p={p:.1e})")

    # Drug associations for activity variance and JUN activity (>=30 lines per drug)
    # drug associations
    ic50 = pd.read_csv(f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv')
    ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
    drug_rows = {}
    for key, v in {**{f'act_{k}_var': agg[f'act_{k}_var'] for k in ['ulm','wm']},
                   **{f'jun_act_{k}': jun_act[k] for k in ['ulm','wm']}}.items():
        rs = []
        for d in ic50_mat.columns:
            y = ic50_mat[d].dropna()
            c = netith.dropna().index.intersection(y.index)
            if len(c) < 30: continue
            r, _ = spearmanr(v[[list(cells).index(i) for i in c]], y[c])
            rs.append(r)
        rs = np.array(rs)
        drug_rows[key] = {'median_rho': float(np.median(rs)), 'n_pos': int((rs > 0).sum()), 'n': int(len(rs))}
        print(f"  {key} drug rho: median={np.median(rs):.3f} pos={int((rs>0).sum())}/{len(rs)}")

    # Save stage-2 extension results
    res['drug'] = drug_rows
    os.makedirs(f'{ROOT}/results/control', exist_ok=True)
    json.dump(res, open(f'{ROOT}/results/control/tf_activity_baseline_ext.json','w'), indent=2)
    print("saved results/control/tf_activity_baseline_ext.json")


# =====================================================================
# true-ulm stage: per-TF univariate activity (act = sum w*E / sum w^2)
# produces results/control/tf_activity_ulm_true.json (SI-N16)
# =====================================================================
# ---- Stage 3 (true-ulm): closed-form per-TF univariate activity ----
def run_true_ulm():
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr
    expr = pd.read_csv(f'{ROOT}/data/gdsc/rna_expr.csv', index_col=0)
    ensg = pd.read_csv(f'{ROOT}/data/gdsc/ensg_symbol_map.csv'); ensg.columns = ['ensg', 'symbol']
    sym = dict(zip(ensg['ensg'], ensg['symbol']))
    annot = pd.read_csv(f'{ROOT}/data/gdsc/cell_annot.csv', index_col=0)
    cl_map = {cel: str(row.get('Characteristics.cell.line.', '')) for cel, row in annot.iterrows()}
    common_cels = [c for c in expr.columns if c in cl_map]
    en = expr[common_cels].copy(); en.columns = [cl_map[c] for c in common_cels]
    en = en.loc[:, ~en.columns.duplicated()]
    matched = en.index.isin(ensg['ensg'])
    en_sym = en.loc[matched].copy(); en_sym.index = [sym[g] for g in en_sym.index]
    en_sym = en_sym[~en_sym.index.duplicated(keep='first')]
    net = pd.read_csv(f'{ROOT}/data/collectri_network.csv')
    focused = set(l.strip() for l in open(f'{ROOT}/results/focused_genes_collectri.txt') if l.strip())
    common = sorted(focused & set(en_sym.index))
    gi = {g: i for i, g in enumerate(common)}
    Z = np.clip(((en_sym.loc[common].T - en_sym.loc[common].mean(axis=1)) / (en_sym.loc[common].std(axis=1) + 1e-10)).values, -3, 3)
    cells = en_sym.columns.tolist()
    tfs = sorted({r['source'] for _, r in net.iterrows() if r['source'] in gi and r['source'] in en_sym.index})
    # Per TF: true-ulm activity act = sum(w * E_target) / sum(w^2) over its own targets
    wtab = {(r['source'], r['target']): float(r['weight']) for _, r in net.iterrows()}
    edges_by_tf = {}
    for _, r in net.iterrows():
        edges_by_tf.setdefault(r['source'], []).append(r['target'])
    act = np.zeros((len(Z), len(tfs)))
    for k, t in enumerate(tfs):
        tgt = [g for g in edges_by_tf.get(t, []) if g in gi]
        if not tgt:
            continue
        ws = np.array([wtab[(t, g)] for g in tgt])
        act[:, k] = (ws * Z[:, [gi[g] for g in tgt]]).sum(axis=1) / (ws ** 2).sum()
    # Activity variance and JUN activity vs NetITH (Spearman)
    netith = pd.read_csv(f'{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)['NetITH'].reindex(cells)
    var = act.var(axis=1)
    r, p = spearmanr(var, netith.values)
    jun_idx = tfs.index('JUN')
    rj, pj = spearmanr(act[:, jun_idx], netith.values)
    # Drug associations of true-ulm activity variance (>=30 lines per drug)
    ic50 = pd.read_csv(f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv')
    ic50_mat = ic50.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='LN_IC50', aggfunc='mean')
    rs = []
    for d in ic50_mat.columns:
        y = ic50_mat[d].dropna()
        c = netith.dropna().index.intersection(y.index)
        if len(c) < 30:
            continue
        rs.append(spearmanr(var[[list(cells).index(i) for i in c]], y[c])[0])
    rs = np.array(rs)
    # Save stage-3 true-ulm results (SI-N16)
    import json, os
    os.makedirs(f'{ROOT}/results/control', exist_ok=True)
    json.dump({'ulm_true_var_vs_netith_rho': float(r), 'ulm_true_var_vs_netith_p': float(p),
               'ulm_true_jun_rho': float(rj), 'ulm_true_jun_p': float(pj),
               'ulm_true_var_drug_median': float(np.median(rs)),
               'ulm_true_var_drug_pos': int((rs > 0).sum()), 'n_drugs': int(len(rs))},
              open(f'{ROOT}/results/control/tf_activity_ulm_true.json', 'w'), indent=2)
    print("saved results/control/tf_activity_ulm_true.json")


# Entry point: run the extension and true-ulm stages after the baseline body
if __name__ == "__main__":
    run_extension()
    run_true_ulm()
