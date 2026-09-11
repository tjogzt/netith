#!/usr/bin/env python3
"""run_coexpr_formulation_validation.py — co-expression NetITH formulation validation on the 113-patient atlas.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/scrnaseq_reference/134d34af-cbcd-4837-9310-3d1f83ec6f18.h5ad; results/scrnaseq_validation/{patient_netith.csv, focused_genes.txt}; data/collectri_network.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/scrnaseq_validation/{coexpr_formulation_patients.csv, coexpr_formulation_per_cancer.csv, coexpr_formulation_summary.json}
Pipeline: single-cell stage — see repository README
"""
import os
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, mannwhitneyu, spearmanr
from scipy.stats import entropy as scipy_entropy
import time

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
DATA_PATH = f"{DATA_ROOT}/scrnaseq_reference/134d34af-cbcd-4837-9310-3d1f83ec6f18.h5ad"
OUT = ROOT / "results/scrnaseq_validation"
MIN_CELLS_PER_PATIENT = 10
MIN_GENES_PER_CELL = 200
MAX_MITO_FRAC = 0.25
N_CELLS_MAX = 25000
SEED = 42


def vn_entropy(L):
    ev = np.linalg.eigvalsh(L)
    ev = ev[ev > 1e-12]
    s = ev.sum()
    if s <= 0:
        return 0.0
    rho = ev / s
    return float(-np.sum(rho * np.log2(rho)))


def main():
    t0 = time.time()
    print("[1/5] Loading + preprocessing (replicating run_scrnaseq_netith_validation.py)", flush=True)
    # Load the final single-cell atlas and replicate the reference pipeline's QC/sampling
    adata = sc.read_h5ad(DATA_PATH)
    if 'feature_name' in adata.var.columns:
        adata.var['gene_symbol'] = adata.var['feature_name'].astype(str)
        dup = adata.var['gene_symbol'].duplicated()
        if dup.any():
            adata.var.loc[dup, 'gene_symbol'] = (adata.var.loc[dup, 'gene_symbol'] + '_' +
                                                 adata.var_names[dup].astype(str).str[:8])
        adata.var_names = adata.var['gene_symbol'].values
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    adata = adata[adata.obs['Combined_outcome'].isin(['Favourable', 'Unfavourable'])].copy()
    ct_counts = adata.obs['Cancer_type_update'].value_counts()
    keep_ct = ct_counts[ct_counts >= 300].index
    adata = adata[adata.obs['Cancer_type_update'].isin(keep_ct)].copy()
    donor_counts = adata.obs['donor_id'].value_counts()
    eligible = donor_counts[donor_counts >= MIN_CELLS_PER_PATIENT].index
    adata = adata[adata.obs['donor_id'].isin(eligible)].copy()
    rng = np.random.RandomState(SEED)
    if adata.n_obs > N_CELLS_MAX:
        indices = []
        cells_per_ct = max(N_CELLS_MAX // len(keep_ct), 500)
        for ct in keep_ct:
            ct_idx_all = np.where(adata.obs['Cancer_type_update'] == ct)[0]
            if len(ct_idx_all) == 0:
                continue
            n_sample = min(len(ct_idx_all), cells_per_ct)
            for out in ['Favourable', 'Unfavourable']:
                out_idx = [i for i in ct_idx_all if adata.obs.iloc[i]['Combined_outcome'] == out]
                n_out = min(len(out_idx), n_sample // 2)
                if n_out > 0:
                    indices.extend(rng.choice(out_idx, size=n_out, replace=False).tolist())
        if len(indices) > N_CELLS_MAX:
            indices = rng.choice(indices, size=N_CELLS_MAX, replace=False).tolist()
        adata = adata[indices].copy()
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    adata = adata[adata.obs.pct_counts_mt < MAX_MITO_FRAC * 100, :].copy()
    # Library-size normalisation + log1p, matching run_scrnaseq_netith_validation.py
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}, Donors: {adata.obs['donor_id'].nunique()} ({time.time()-t0:.0f}s)", flush=True)

    # verify cell set against known output
    # Verify the reconstructed cell set matches the known per-patient n_cells
    known = pd.read_csv(OUT / "patient_netith.csv")
    known_n = known.set_index('patient')['n_cells']
    obs_n = adata.obs['donor_id'].value_counts()
    common = sorted(set(known_n.index) & set(obs_n.index))
    mismatch = [p for p in common if known_n[p] != obs_n[p]]
    print(f"  [check] patients {len(common)}; n_cells mismatch: {len(mismatch)} {mismatch[:5]}")
    if len(mismatch) > 0:
        print("  WARNING: cell set differs from known output; downstream comparison is indicative only")

    print("[2/5] focused gene set and CollecTRI edges", flush=True)
    focused = [g for g in open(OUT / "focused_genes.txt").read().split() if g in adata.var_names]
    print(f"  focused genes in data: {len(focused)}")
    # Restrict CollecTRI to the focused gene set; edge weight = |strength|
    net = pd.read_csv(DATA_ROOT / "collectri_network.csv")
    net_f = net[net['source'].isin(focused) & net['target'].isin(focused)].copy()
    tfs = sorted(net_f['source'].unique())
    tgts = sorted(set(net_f['target'].unique()) - set(tfs))
    edges = [(r['source'], r['target'], abs(float(r['weight']))) for _, r in net_f.iterrows()]
    print(f"  TFs: {len(tfs)}, targets: {len(tgts)}, edges: {len(edges)}")
    gi = {g: i for i, g in enumerate(focused)}
    tf_idx = [gi[t] for t in tfs]
    tgt_idx = [gi[t] for t in tgts]

    # Extract the focused-gene expression matrix (dense) and patient metadata
    X = adata[:, focused].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    obs = adata.obs

    print("[3/5] per-patient co-expression entropy (option a1)", flush=True)
    recs = []
    for pid in sorted(obs['donor_id'].unique()):
        m = (obs['donor_id'] == pid).values
        if m.sum() < MIN_CELLS_PER_PATIENT:
            continue
        Xp = X[m]
        # Within-patient cross-cell Spearman: rank-transform, centre and unit-normalise columns
        Rr = np.apply_along_axis(rankdata, 0, Xp).astype(float)
        Rc = Rr - Rr.mean(0)
        nrm = np.sqrt((Rc ** 2).sum(0))
        nrm[nrm == 0] = 1.0
        Rn = Rc / nrm
        # Cross-cell correlation matrix C = R_n^T R_n; adjacency A = |rho| on TF-target pairs
        C = Rn.T @ Rn
        A2 = np.abs(C[np.ix_(tf_idx, tgt_idx)])
        # W = A A^T (TF co-regulation), then L = D - W and its von Neumann entropy
        W = A2 @ A2.T
        tr = W.sum()
        netith_cx = vn_entropy(np.diag(W.sum(1)) - W) if tr > 1e-10 else 0.0
        recs.append({'patient': pid,
                     'cancer_type': obs.loc[m, 'Cancer_type_update'].iloc[0],
                     'outcome': obs.loc[m, 'Combined_outcome'].iloc[0],
                     'n_cells': int(m.sum()),
                     'netith_coexpr': netith_cx})
    df = pd.DataFrame(recs)
    print(f"  patients: {len(df)} ({time.time()-t0:.0f}s)")

    print("[4/5] merge with implemented version and compare", flush=True)
    # Merge with the implemented per-cell formulation and correlate (Spearman, null: rho = 0)
    merged = df.merge(known[['patient', 'netith_e', 'mean_vn', 'median_vn']], on='patient', how='inner')
    r, p = spearmanr(merged['netith_coexpr'], merged['netith_e'])
    r2, p2 = spearmanr(merged['netith_coexpr'], merged['mean_vn'])
    print(f"  netith_coexpr vs netith_e : Spearman ρ={r:.3f} (p={p:.2e})")
    print(f"  netith_coexpr vs mean_vn  : Spearman ρ={r2:.3f} (p={p2:.2e})")

    # Mann-Whitney U: Favourable vs Unfavourable NetITH distributions (null: identical); d = pooled-SD effect size
    def outcome_stats(col):
        fav = merged[merged['outcome'] == 'Favourable'][col]
        unf = merged[merged['outcome'] == 'Unfavourable'][col]
        u, pv = mannwhitneyu(fav, unf)
        pooled = np.sqrt((fav.std(ddof=1) ** 2 + unf.std(ddof=1) ** 2) / 2)
        d = (fav.mean() - unf.mean()) / pooled if pooled > 0 else np.nan
        return fav.mean(), fav.std(), len(fav), unf.mean(), unf.std(), len(unf), pv, d

    for col in ['netith_coexpr', 'netith_e', 'mean_vn']:
        fm, fs, fn, um, us, un, pv, d = outcome_stats(col)
        print(f"  {col:15s}: Fav {fm:.3f}±{fs:.3f} (n={fn}) vs Unfav {um:.3f}±{us:.3f} (n={un}) | MW p={pv:.4f} | d={d:+.3f}")

    # cross-cancer direction consistency
    # Per-cancer-type Mann-Whitney to test direction consistency between the two formulations
    dir_rows = []
    for ct in sorted(merged['cancer_type'].unique()):
        sub = merged[merged['cancer_type'] == ct]
        if sub['outcome'].nunique() < 2 or len(sub) < 6:
            continue
        row = {'cancer': ct, 'n': len(sub)}
        for col in ['netith_coexpr', 'netith_e']:
            fav = sub[sub['outcome'] == 'Favourable'][col]
            unf = sub[sub['outcome'] == 'Unfavourable'][col]
            u, pv = mannwhitneyu(fav, unf)
            pooled = np.sqrt((fav.std(ddof=1) ** 2 + unf.std(ddof=1) ** 2) / 2)
            row[col + '_d'] = (fav.mean() - unf.mean()) / pooled if pooled > 0 else np.nan
            row[col + '_p'] = pv
        dir_rows.append(row)
    dirdf = pd.DataFrame(dir_rows)
    n_ct = len(dirdf)
    same_dir = (np.sign(dirdf['netith_coexpr_d'].fillna(0)) == np.sign(dirdf['netith_e_d'].fillna(0))).sum()
    print(f"  cross-cancer direction consistent: {same_dir}/{n_ct}")

    print("[5/5] saving output", flush=True)
    # Write per-patient, per-cancer and summary outputs
    merged.to_csv(OUT / "coexpr_formulation_patients.csv", index=False)
    dirdf.to_csv(OUT / "coexpr_formulation_per_cancer.csv", index=False)
    summary = {
        'n_patients': int(len(merged)),
        'rho_coexpr_vs_netith_e': float(r), 'p_coexpr_vs_netith_e': float(p),
        'rho_coexpr_vs_mean_vn': float(r2), 'p_coexpr_vs_mean_vn': float(p2),
        'n_tfs': len(tfs), 'n_targets': len(tgts), 'n_edges': len(edges),
        'cells_verified': len(mismatch) == 0,
    }
    import json
    (OUT / "coexpr_formulation_summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
