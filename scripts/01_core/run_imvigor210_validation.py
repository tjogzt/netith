"""
run_imvigor210_validation.py — Validate NetITH against anti-PD-L1 immunotherapy response in IMvigor210 bladder cancer.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/imvigor210/counts.csv: bulk RNA-seq counts
    - data/imvigor210/pheno.csv: clinical phenotype (response, immune phenotype, survival)
    - data/imvigor210/entrez_symbol_map.csv: Entrez -> symbol mapping (optional)
    - results/focused_genes_collectri.txt: focused CollecTRI gene set
    - <NETITH_COLLECTRI_PKL, default /tmp/collectri_net.pkl>: pickled CollecTRI network (fallback)
Outputs :
    - results/imvigor210/imvigor210_netith_results.csv
    - results/imvigor210/figures/imvigor210_{survival,comprehensive}.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os
import sys
import warnings
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal, fisher_exact
from scipy.linalg import eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Config ───
DATA_DIR = DATA_ROOT / "imvigor210"
OUTPUT_DIR = Path(f"{ROOT}/results/imvigor210")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
MIN_FOCUSED_GENES = 30  # below this, fall back to the top CollecTRI genes
COLLECTRI_PKL = Path(os.environ.get("NETITH_COLLECTRI_PKL", "/tmp/collectri_net.pkl"))

def load_collectri():
    """Return the CollecTRI network DataFrame, building the pickle from the
    shipped data/collectri_network.csv when NETITH_COLLECTRI_PKL is absent."""
    if not Path(COLLECTRI_PKL).exists():
        Path(COLLECTRI_PKL).parent.mkdir(parents=True, exist_ok=True)
        net = pd.read_csv(Path(ROOT) / "data" / "collectri_network.csv")
        with open(COLLECTRI_PKL, "wb") as fh:
            pickle.dump(net, fh)
    with open(COLLECTRI_PKL, "rb") as fh:
        return pickle.load(fh)





# ─── 1. Load & Preprocess ──────────────────────────────────
def load_data():
    """Load expression and clinical data."""
    print("[1/5] Loading data...")
    
    # Expression counts
    expr_raw = pd.read_csv(f"{DATA_DIR}/counts.csv", index_col=0)
    print(f"  Raw counts: {expr_raw.shape}")
    
    # Clinical
    pheno = pd.read_csv(f"{DATA_DIR}/pheno.csv", index_col=0)
    print(f"  Pheno: {pheno.shape}")
    
    # Map response to binary
    resp_map = {
        'CR': 'Responder', 'PR': 'Responder',
        'SD': 'Non-responder', 'PD': 'Non-responder',
        'NE': None
    }
    pheno['Response'] = pheno['Best Confirmed Overall Response'].map(resp_map)
    
    # Map immune phenotype
    pheno['Immune_Group'] = pheno['Immune phenotype'].map({
        'inflamed': 'Inflamed',
        'excluded': 'Excluded', 
        'desert': 'Desert'
    })
    
    return expr_raw, pheno


# ─── 2. Gene ID Mapping & Normalization ────────────────────
def prepare_expression(expr_raw, pheno):
    """Entrez ID → Symbol mapping, normalization, CollecTRI subset."""
    print("[2/5] Preparing expression matrix...")
    
    # Map Entrez ID to symbol (local mapping file)
    map_file = f"{DATA_DIR}/entrez_symbol_map.csv"
    if os.path.exists(map_file):
        gene_map = pd.read_csv(map_file)
        gene_map['entrez'] = gene_map['entrez'].astype(str)
        id_map_dict = dict(zip(gene_map['entrez'], gene_map['symbol']))
        print(f"  Loaded {len(id_map_dict)} Entrez→Symbol mappings (local)")
    else:
        id_map_dict = fallback_gene_map([str(g) for g in expr_raw.index])
    
    # Convert expression index to symbols
    expr_raw.index = expr_raw.index.astype(str)
    expr_sym = expr_raw.loc[expr_raw.index.isin(id_map_dict.keys())].copy()
    expr_sym.index = [id_map_dict[g] for g in expr_sym.index]
    
    # Remove duplicates (keep first)
    expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]
    
    # Match samples with pheno
    common_samples = list(set(expr_sym.columns) & set(pheno.index))
    print(f"  Common samples: {len(common_samples)}")
    expr_sym = expr_sym[common_samples]
    pheno_matched = pheno.loc[common_samples]
    
    # TPM-like normalization: CPM + log2
    lib_sizes = expr_sym.sum(axis=0)
    cpm = expr_sym / lib_sizes * 1e6
    expr_log2 = np.log2(cpm + 1)
    
    print(f"  Normalized: {expr_log2.shape}")
    
    return expr_log2, pheno_matched


def fallback_gene_map(entrez_ids):
    """Fallback: use common gene name mapping file or simple numeric mapping."""
    # Try to load from existing mapping
    map_paths = [
        f"{DATA_ROOT}/metadata/gene_map.csv",
        f"{DATA_ROOT}/gene_map.csv",
    ]
    for mp in map_paths:
        if os.path.exists(mp):
            gm = pd.read_csv(mp, index_col=0)
            d = {}
            for i, row in gm.iterrows():
                d[str(i)] = row.iloc[0] if len(row) > 0 else str(i)
            return d
    
    # Last resort: use Entrez ID as-is
    return {g: g for g in entrez_ids}


# ─── 3. CollecTRI GRN & von Neumann Entropy ────────────────
def compute_netith(expr_log2, pheno):
    """CollecTRI-based NetITH for bulk RNA-seq samples."""
    print("[3/5] Computing NetITH via CollecTRI...")
    
    # Load CollecTRI
    import decoupler as dc
    try:
        net = dc.op.collectri(organism='human')
        if 'sign_decision' in net.columns:
            net = net[~net['sign_decision'].str.startswith('default')]
    except:
        import pickle
        with open(COLLECTRI_PKL, 'rb') as f:
            net = pickle.load(f)
    
    collectri_genes = set(net['source']) | set(net['target'])
    
    # Use focused gene set for consistency with prior analyses (238 genes)
    focused_file = f"{ROOT}/results/focused_genes_collectri.txt"
    with open(focused_file) as f:
        focused_genes = set(line.strip() for line in f if line.strip())
    
    # Intersect focused genes with expression data
    common_genes = sorted(focused_genes & set(expr_log2.index))
    print(f"  Focused genes in data: {len(common_genes)}/{len(focused_genes)}")
    
    if len(common_genes) < MIN_FOCUSED_GENES:
        # Fallback: use top CollecTRI genes
        common_genes = sorted(list(collectri_genes & set(expr_log2.index)))[:239]
        print(f"  Fallback: {len(common_genes)} CollecTRI genes")
    
    # Subset expression
    expr_sub = expr_log2.loc[common_genes]
    gene_to_idx = {g: i for i, g in enumerate(common_genes)}
    n_genes = len(common_genes)
    n_samples = expr_sub.shape[1]
    print(f"  Matrix: {n_genes} genes × {n_samples} samples")
    
    # Build edge list (filtered to common genes)
    edges = []
    for _, row in net.iterrows():
        tf = row['source']; target = row['target']
        if tf in gene_to_idx and target in gene_to_idx:
            edges.append({
                'tf_idx': gene_to_idx[tf],
                'target_idx': gene_to_idx[target],
                'weight': float(row.get('weight', 1.0)),
            })
    print(f"  Edges: {len(edges)}")
    
    # Z-score normalize
    expr_vals = expr_sub.values.T  # samples × genes
    expr_mean = expr_vals.mean(axis=0)
    expr_std = expr_vals.std(axis=0) + 1e-10
    expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)
    
    # Per-sample entropy
    entropies = np.full(n_samples, np.nan)
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"    Sample {i+1}/{n_samples}")
        
        # Build adjacency from gene activity
        A = np.zeros((n_genes, n_genes))
        for e in edges:
            tf_act = abs(expr_z[i, e['tf_idx']])
            tgt_act = abs(expr_z[i, e['target_idx']])
            A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_act * tgt_act
        
        A = A + A.T  # FIX(eigvalsh-audit 2026-08-16): symmetrize directed TF->target adjacency before Laplacian (eigvalsh silently used one triangle)
        degrees = A.sum(axis=1)
        L = np.diag(degrees) - A
        trace = degrees.sum()
        
        if trace < 1e-10:
            entropies[i] = 0.0
            continue
        
        try:
            eigs = eigvalsh(L)
            eigs = np.clip(eigs, 0, None)
            rho = eigs / (trace + 1e-10)
            rho = np.clip(rho, 1e-12, 1.0)
            entropies[i] = -np.sum(rho * np.log2(rho))
        except Exception:
            entropies[i] = np.nan
    
    valid = ~np.isnan(entropies)
    print(f"  Valid: {valid.sum()}/{n_samples}")
    if valid.sum() > 0:
        print(f"  Entropy range: [{entropies[valid].min():.3f}, {entropies[valid].max():.3f}]")
    
    return entropies, expr_sub.columns.tolist()


# ─── 4. Response Analysis ──────────────────────────────────
def analyze_response(entropies, sample_ids, pheno):
    """NetITH vs immunotherapy response."""
    from scipy.stats import mannwhitneyu
    
    print("\n[4/5] Analyzing NetITH vs Response...")
    
    # Build result DataFrame
    results = pd.DataFrame({
        'sample': sample_ids,
        'NetITH': entropies,
    })
    results = results.set_index('sample')
    results = results.join(pheno[['Response', 'Immune_Group', 'binaryResponse',
                                   'Best Confirmed Overall Response',
                                   'FMOne mutation burden per MB',
                                   'Neoantigen burden per MB',
                                   'IC Level', 'TC Level',
                                   'os', 'censOS',
                                   'Lund', 'TCGA Subtype',
                                   'Immune phenotype']])
    
    results = results[~results['NetITH'].isna()].copy()
    print(f"  Analyzable samples: {len(results)}")
    
    # ── Response: CR/PR vs SD/PD ──
    resp = results[results['Response'].notna()]
    responders = resp[resp['Response'] == 'Responder']
    non_responders = resp[resp['Response'] == 'Non-responder']
    
    print(f"\n  Responders: {len(responders)}, Non-responders: {len(non_responders)}")
    
    if len(responders) > 5 and len(non_responders) > 5:
        stat, p = mannwhitneyu(responders['NetITH'], non_responders['NetITH'])
        d = (responders['NetITH'].mean() - non_responders['NetITH'].mean()) / resp['NetITH'].std()
        print(f"  Responder vs Non-responder: d={d:.3f}, p={p:.4f}")
        print(f"    R:  {responders['NetITH'].mean():.3f} +/- {responders['NetITH'].std():.3f}")
        print(f"    NR: {non_responders['NetITH'].mean():.3f} +/- {non_responders['NetITH'].std():.3f}")
    
    # ── Four-level response ──
    print(f"\n  By Response Category:")
    for cat in ['CR', 'PR', 'SD', 'PD']:
        subset = results[results['Best Confirmed Overall Response'] == cat]
        if len(subset) > 0:
            print(f"    {cat}: {subset['NetITH'].mean():.3f} +/- {subset['NetITH'].std():.3f} (n={len(subset)})")
    
    # CR+PR vs SD+PD (using binaryResponse)
    for cat in results['binaryResponse'].dropna().unique():
        subset = results[results['binaryResponse'] == cat]
        if len(subset) > 0:
            print(f"    {cat}: {subset['NetITH'].mean():.3f} +/- {subset['NetITH'].std():.3f} (n={len(subset)})")
    
    # ── Immune phenotype ──
    print(f"\n  By Immune Phenotype:")
    for ip in ['inflamed', 'excluded', 'desert']:
        subset = results[results['Immune phenotype'] == ip]
        if len(subset) > 0:
            print(f"    {ip}: {subset['NetITH'].mean():.3f} +/- {subset['NetITH'].std():.3f} (n={len(subset)})")
    
    # ── Correlation with TMB ──
    tmb = results['FMOne mutation burden per MB'].dropna()
    common_idx = results.index.intersection(tmb.index)
    if len(common_idx) > 10:
        from scipy.stats import spearmanr
        r, p = spearmanr(results.loc[common_idx, 'NetITH'], tmb.loc[common_idx])
        print(f"\n  NetITH vs TMB: rho={r:.3f}, p={p:.4f}")
    
    # ── Survival ──
    from lifelines import KaplanMeierFitter, CoxPHFitter
    surv = results[results['os'].notna() & results['censOS'].notna()].copy()
    if len(surv) > 20:
        surv['high_entropy'] = surv['NetITH'] > surv['NetITH'].median()
        
        kmf = KaplanMeierFitter()
        fig_km, ax_km = plt.subplots(figsize=(8, 6))
        for label, mask in [('High NetITH', surv['high_entropy']),
                             ('Low NetITH', ~surv['high_entropy'])]:
            kmf.fit(surv.loc[mask, 'os'], surv.loc[mask, 'censOS'], label=label)
            kmf.plot_survival_function(ax=ax_km)
        
        from lifelines.statistics import logrank_test
        lr = logrank_test(
            surv.loc[surv['high_entropy'], 'os'],
            surv.loc[~surv['high_entropy'], 'os'],
            surv.loc[surv['high_entropy'], 'censOS'],
            surv.loc[~surv['high_entropy'], 'censOS'],
        )
        ax_km.set_title(f"IMvigor210: NetITH OS (log-rank p={lr.p_value:.4f})")
        fig_km.savefig(OUTPUT_DIR / "figures" / "imvigor210_survival.png", dpi=150, bbox_inches='tight')
        print(f"\n  KM OS: log-rank p={lr.p_value:.4f}")
        
        # Cox
        cph = CoxPHFitter()
        cox_df = surv[['os', 'censOS', 'NetITH']].copy()
        try:
            cph.fit(cox_df, 'os', 'censOS')
            cph.print_summary()
            print(f"  Cox (NetITH): HR={cph.hazard_ratios_.iloc[0]:.3f}, p={cph.summary.loc['NetITH','p']:.4f}")
        except Exception as e:
            print(f"  Cox error: {e}")
    
    return results


# ─── 5. Multivariable Analysis & Figures ───────────────────
def multivariable_and_plot(results):
    """Multivariable model: NetITH + TMB + Neoantigen → Response."""
    print("\n[5/5] Multivariable analysis & figures...")
    
    # Logistic regression: NetITH + TMB + Neoantigen → binaryResponse
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    
    model_df = results[['NetITH', 'FMOne mutation burden per MB', 
                         'Neoantigen burden per MB', 'binaryResponse']].dropna()
    
    # Remove NE
    model_df = model_df[model_df['binaryResponse'].isin(['CR/PR', 'SD/PD'])]
    model_df['y'] = (model_df['binaryResponse'] == 'CR/PR').astype(int)
    
    if len(model_df) > 30 and model_df['y'].sum() > 5:
        print(f"  Logistic regression: {len(model_df)} samples, {model_df['y'].sum()} responders")
        
        X = model_df[['NetITH', 'FMOne mutation burden per MB', 'Neoantigen burden per MB']].values
        y = model_df['y'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Single feature AUCs
        for i, name in enumerate(['NetITH', 'TMB', 'Neoantigen']):
            auc = cross_val_score(LogisticRegression(C=1, max_iter=1000),
                                  X_scaled[:, i:i+1], y, cv=5, scoring='roc_auc').mean()
            print(f"    {name} AUC: {auc:.3f}")
        
        # Combined AUC
        auc_combined = cross_val_score(LogisticRegression(C=1, max_iter=1000),
                                       X_scaled, y, cv=5, scoring='roc_auc').mean()
        print(f"    Combined AUC: {auc_combined:.3f}")
        
        # Feature importance
        lr = LogisticRegression(C=1, max_iter=1000)
        lr.fit(X_scaled, y)
        for i, name in enumerate(['NetITH', 'TMB', 'Neoantigen']):
            print(f"    {name} coef: {lr.coef_[0][i]:.3f}")
    
    # ── Plot ──
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # A: Response boxplot
    ax = axes[0, 0]
    resp_order = ['CR', 'PR', 'SD', 'PD']
    colors = ['#2ecc71', '#27ae60', '#e67e22', '#e74c3c']
    for i, (cat, c) in enumerate(zip(resp_order, colors)):
        v = results[results['Best Confirmed Overall Response'] == cat]['NetITH']
        if len(v) > 0:
            bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
            bp['boxes'][0].set_facecolor(c)
            bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks(range(len(resp_order)))
    ax.set_xticklabels(resp_order)
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH by Response Category')
    
    # B: Immune phenotype
    ax = axes[0, 1]
    ip_order = ['inflamed', 'excluded', 'desert']
    ip_colors = ['#e74c3c', '#f39c12', '#3498db']
    for i, (ip, c) in enumerate(zip(ip_order, ip_colors)):
        v = results[results['Immune phenotype'] == ip]['NetITH']
        if len(v) > 0:
            bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
            bp['boxes'][0].set_facecolor(c)
            bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks(range(len(ip_order)))
    ax.set_xticklabels(ip_order)
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH by Immune Phenotype')
    
    # C: Responder vs Non-responder
    ax = axes[0, 2]
    for i, (label, mask) in enumerate([('Responder', results['Response'] == 'Responder'),
                                        ('Non-resp.', results['Response'] == 'Non-responder')]):
        v = results.loc[mask, 'NetITH']
        if len(v) > 0:
            bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
            c = '#2ecc71' if label == 'Responder' else '#e74c3c'
            bp['boxes'][0].set_facecolor(c)
            bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-responder'])
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH: Responder vs Non-responder')
    
    # D: NetITH vs TMB scatter
    ax = axes[1, 0]
    plot_df = results[['NetITH', 'FMOne mutation burden per MB', 'Response']].dropna()
    for label, c in [('Responder', '#2ecc71'), ('Non-responder', '#e74c3c')]:
        mask = plot_df['Response'] == label
        ax.scatter(plot_df.loc[mask, 'FMOne mutation burden per MB'],
                   plot_df.loc[mask, 'NetITH'],
                   c=c, alpha=0.5, s=20, label=label)
    ax.set_xlabel('TMB (mut/MB)')
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH vs TMB')
    ax.legend(fontsize=8)
    
    # E: Distribution by IC level
    ax = axes[1, 1]
    ic_order = ['IC0', 'IC1', 'IC2', 'IC2+']
    for i, ic in enumerate(ic_order):
        v = results[results['IC Level'] == ic]['NetITH']
        if len(v) > 0:
            bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
            bp['boxes'][0].set_facecolor(sns.color_palette("viridis", 4)[i])
    ax.set_xticks(range(len(ic_order)))
    ax.set_xticklabels(ic_order)
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH by IC Level (PD-L1 on immune cells)')
    
    # F: NetITH distribution
    ax = axes[1, 2]
    for label, c in [('Responder', '#2ecc71'), ('Non-responder', '#e74c3c')]:
        mask = results['Response'] == label
        v = results.loc[mask, 'NetITH']
        if len(v) > 5:
            ax.hist(v, bins=25, alpha=0.4, color=c, label=f'{label} (n={len(v)})', density=True)
    ax.set_xlabel('NetITH')
    ax.set_ylabel('Density')
    ax.set_title('NetITH Distribution')
    ax.legend(fontsize=8)
    
    plt.suptitle('IMvigor210: NetITH & Anti-PD-L1 Immunotherapy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "imvigor210_comprehensive.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: imvigor210_comprehensive.png")
    
    return results


# ─── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  IMvigor210: NetITH vs Anti-PD-L1 Immunotherapy")
    print("=" * 60)
    
    expr_raw, pheno = load_data()
    expr_log2, pheno_matched = prepare_expression(expr_raw, pheno)
    entropies, sample_ids = compute_netith(expr_log2, pheno_matched)
    results = analyze_response(entropies, sample_ids, pheno_matched)
    results = multivariable_and_plot(results)
    
    # Save
    results.to_csv(OUTPUT_DIR / "imvigor210_netith_results.csv")
    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
