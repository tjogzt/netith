"""
run_gdsc_drug_sensitivity.py — Compute NetITH in GDSC cell lines and correlate it with drug IC50 across ~300 drugs.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-23
Inputs  :
    - data/gdsc/rna_expr.csv: GDSC expression (ENSG rows x cell-line columns)
    - data/gdsc/cell_annot.csv: cell-line annotation (array-ID to cell-line-name mapping)
    - data/gdsc/ensg_symbol_map.csv: ENSG -> gene-symbol mapping
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 drug sensitivity (LN_IC50)
    - results/focused_genes_collectri.txt: focused 239-gene CollecTRI set
    - <NETITH_COLLECTRI_PKL, default /tmp/collectri_net.pkl>: pickled CollecTRI network
Outputs :
    - results/gdsc/gdsc_netith_cell_lines.csv, gdsc_drug_netith_correlations.csv
    - results/gdsc/figures/gdsc_netith_drug.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os, sys, pickle, warnings
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from scipy.linalg import eigvalsh
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = DATA_ROOT / "gdsc"
GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
OUTPUT_DIR = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
MIN_DRUG_CELLS = 30  # minimum cell lines per drug to test NetITH x IC50 correlation
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





# ─── 1. Load & Map Expression ──────────────────────────────
def load_expression():
    """Load RNA expression, map ENSG→Symbol, normalize."""
    print("[1/5] Loading expression...")
    
    # Expression (ENSG genes × CEL files)
    expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
    print(f"  Raw: {expr_raw.shape}")
    
    # Cell annotations (CEL → cell line name)
    annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
    cell_line_map = {}
    for cel, row in annot.iterrows():
        cl = str(row.get('Characteristics.cell.line.', ''))
        if cl and cl != 'nan' and cl != 'NA':
            cell_line_map[cel] = cl
    print(f"  Cell line mappings: {len(cell_line_map)}")
    
    # Rename columns from CEL to cell line name
    common_cels = [c for c in expr_raw.columns if c in cell_line_map]
    expr_named = expr_raw[common_cels].copy()
    expr_named.columns = [cell_line_map[c] for c in common_cels]
    
    # Remove duplicates (keep first)
    dup_cols = expr_named.columns.duplicated()
    if dup_cols.any():
        print(f"  Removing {dup_cols.sum()} duplicate cell lines")
        expr_named = expr_named.loc[:, ~dup_cols]
    
    # ENSG → Symbol mapping
    gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
    ensg_to_sym = dict(zip(gene_map['ensg'], gene_map['symbol']))
    
    # Map genes
    expr_named.index = expr_named.index.astype(str)
    matched = expr_named.index.isin(ensg_to_sym.keys())
    print(f"  Genes mapped: {matched.sum()}/{len(expr_named)}")
    
    expr_sym = expr_named.loc[matched].copy()
    expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
    expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]
    
    # Already log2-normalized, just ensure range check
    print(f"  Final: {expr_sym.shape}")
    print(f"  Value range: [{expr_sym.values.min():.2f}, {expr_sym.values.max():.2f}]")
    
    return expr_sym


# ─── 2. Load Drug Sensitivity ──────────────────────────────
def load_ic50():
    """Load GDSC2 IC50 data."""
    print("[2/5] Loading IC50 data...")
    
    ic50 = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    print(f"  IC50 records: {len(ic50)}")
    print(f"  Drugs: {ic50['DRUG_NAME'].nunique()}")
    print(f"  Cell lines: {ic50['CELL_LINE_NAME'].nunique()}")
    print(f"  Cancer types: {ic50['CANCER_TYPE'].nunique()}")
    
    # Pivot to cell_line × drug matrix of LN_IC50
    ic50_mat = ic50.pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME', 
        values='LN_IC50', aggfunc='mean'
    )
    print(f"  Pivoted: {ic50_mat.shape}")
    
    return ic50, ic50_mat


# ─── 3. Compute NetITH for Cell Lines ──────────────────────
def compute_netith(expr_sym):
    """NetITH via CollecTRI (focused 239-gene set for speed)."""
    print("[3/5] Computing NetITH...")
    
    # Focused genes
    focused_file = f"{ROOT}/results/focused_genes_collectri.txt"
    with open(focused_file) as f:
        focused = set(line.strip() for line in f if line.strip())
    
    common = sorted(focused & set(expr_sym.index))
    print(f"  Focused genes: {len(common)}/{len(focused)}")
    
    expr_sub = expr_sym.loc[common]
    gene_to_idx = {g: i for i, g in enumerate(common)}
    n_genes = len(common)
    n_cells = expr_sub.shape[1]
    print(f"  Matrix: {n_genes} genes × {n_cells} cell lines")
    
    # CollecTRI edges
    net = load_collectri()
    
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
    
    # Z-score
    expr_vals = expr_sub.values.T
    expr_mean = expr_vals.mean(axis=0)
    expr_std = expr_vals.std(axis=0) + 1e-10
    expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)
    
    entropies = np.full(n_cells, np.nan)
    cell_names = expr_sub.columns.tolist()
    
    for i in range(n_cells):
        if (i + 1) % 200 == 0:
            print(f"    Cell line {i+1}/{n_cells}")
        
        A = np.zeros((n_genes, n_genes))
        for e in edges:
            tf_a = abs(expr_z[i, e['tf_idx']])
            tgt_a = abs(expr_z[i, e['target_idx']])
            A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_a * tgt_a
        
        A = A + A.T  # FIX(eigvalsh-audit 2026-08-16): symmetrize directed TF->target adjacency before Laplacian (eigvalsh silently used one triangle)
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        trace = deg.sum()
        
        if trace < 1e-10:
            entropies[i] = 0.0; continue
        
        try:
            eigs = eigvalsh(L)
            eigs = np.clip(eigs, 0, None)
            rho = eigs / (trace + 1e-10)
            rho = np.clip(rho, 1e-12, 1.0)
            entropies[i] = -np.sum(rho * np.log2(rho))
        except:
            entropies[i] = np.nan
    
    valid = ~np.isnan(entropies)
    print(f"  Valid: {valid.sum()}/{n_cells}")
    print(f"  Entropy range: [{entropies[valid].min():.3f}, {entropies[valid].max():.3f}]")
    
    result = pd.DataFrame({'cell_line': cell_names, 'NetITH': entropies})
    result = result[result['NetITH'].notna()].set_index('cell_line')
    
    return result


# ─── 4. Drug Sensitivity Correlation ───────────────────────
def correlate_drugs(netith, ic50_mat, ic50_raw):
    """Correlate NetITH with drug IC50 for all drugs."""
    print("[4/5] Correlating NetITH × Drug IC50...")
    
    # Match cell lines
    common_cells = sorted(set(netith.index) & set(ic50_mat.index))
    print(f"  Common cell lines: {len(common_cells)}")
    
    netith_sub = netith.loc[common_cells]
    ic50_sub = ic50_mat.loc[common_cells]
    
    # For each drug with enough data
    results = []
    for drug in ic50_sub.columns:
        drug_data = pd.DataFrame({
            'NetITH': netith_sub['NetITH'],
            'LN_IC50': ic50_sub[drug]
        }).dropna()
        
        if len(drug_data) < MIN_DRUG_CELLS:
            continue
        
        rho, p = spearmanr(drug_data['NetITH'], drug_data['LN_IC50'])
        
        # Also compute high vs low entropy IC50 difference
        median_e = drug_data['NetITH'].median()
        high = drug_data[drug_data['NetITH'] > median_e]['LN_IC50']
        low = drug_data[drug_data['NetITH'] <= median_e]['LN_IC50']
        
        if len(high) > 5 and len(low) > 5:
            stat, p_mw = mannwhitneyu(high, low)
            d = (high.mean() - low.mean()) / drug_data['LN_IC50'].std()
        else:
            p_mw = 1.0; d = 0.0
        
        # Get drug info
        drug_info = ic50_raw[ic50_raw['DRUG_NAME'] == drug].iloc[0]
        
        results.append({
            'drug': drug,
            'target': drug_info['PUTATIVE_TARGET'],
            'pathway': drug_info['PATHWAY_NAME'],
            'n_cells': len(drug_data),
            'rho': rho,
            'p_spearman': p,
            'p_mannwhitney': p_mw,
            'cohens_d': d,
            'mean_IC50': drug_data['LN_IC50'].mean(),
        })
    
    results_df = pd.DataFrame(results)
    # FIX(review-batch-A): previously Bonferroni (fdr = p × n) mislabeled as FDR.
    # Switch to standard Benjamini-Hochberg FDR (statsmodels multipletests,
    # method='fdr_bh'), equivalent to R p.adjust(..., 'BH') used by the R
    # pipeline; yields 276/286 FDR<0.05, matching master_results.yaml and text.
    results_df['fdr'] = multipletests(
        results_df['p_spearman'].values, method='fdr_bh'
    )[1]
    results_df = results_df.sort_values('rho')
    
    print(f"\n  Tested {len(results_df)} drugs")
    print(f"  FDR < 0.05: {(results_df['fdr'] < 0.05).sum()}")
    print(f"  FDR < 0.10: {(results_df['fdr'] < 0.10).sum()}")
    
    # Top negative (high NetITH = more sensitive / lower IC50)
    print(f"\n  Top NEGATIVE rho (high NetITH → lower IC50 = more sensitive):")
    for _, r in results_df.head(8).iterrows():
        sig = '***' if r['fdr'] < 0.01 else '**' if r['fdr'] < 0.05 else '*' if r['fdr'] < 0.10 else ''
        tgt = str(r['target'])[:15] if pd.notna(r['target']) else 'N/A'
        print(f"    {str(r['drug'])[:25]:25s} [{tgt:15s}] rho={r['rho']:.3f} p={r['p_spearman']:.3f} fdr={r['fdr']:.3f} {sig}")
    
    # Top positive (high NetITH = higher IC50 = more resistant)
    print(f"\n  Top POSITIVE rho (high NetITH → higher IC50 = more resistant):")
    for _, r in results_df.tail(8).iterrows():
        sig = '***' if r['fdr'] < 0.01 else '**' if r['fdr'] < 0.05 else '*' if r['fdr'] < 0.10 else ''
        tgt = str(r['target'])[:15] if pd.notna(r['target']) else 'N/A'
        print(f"    {str(r['drug'])[:25]:25s} [{tgt:15s}] rho={r['rho']:.3f} p={r['p_spearman']:.3f} fdr={r['fdr']:.3f} {sig}")
    
    return results_df, netith_sub, ic50_sub


# ─── 5. Pathway & Cancer-Type Analysis ─────────────────────
def enrichment_and_plot(results_df, netith, ic50_raw):
    """Pathway enrichment of NetITH-associated drugs, cancer-type stratification."""
    print("[5/5] Enrichment & visualization...")
    
    # ── Pathway-level analysis ──
    print("\n  By Drug Pathway:")
    pathway_stats = results_df.groupby('pathway').agg(
        n_drugs=('drug', 'nunique'),
        mean_rho=('rho', 'mean'),
        median_rho=('rho', 'median'),
        sig_drugs=('fdr', lambda x: (x < 0.10).sum()),
        n_total=('drug', 'count'),
    ).sort_values('mean_rho')
    
    for _, row in pathway_stats.iterrows():
        if row['n_drugs'] >= 3:
            sig_str = f" [{row['sig_drugs']}/{row['n_total']} FDR<0.10]" if row['sig_drugs'] > 0 else ""
            print(f"    {row.name:30s}: mean_rho={row['mean_rho']:.3f} (n={row['n_drugs']}){sig_str}")
    
    # ── Cancer-type stratification ──
    print("\n  By Cancer Type (top 10):")
    cancer_ic50 = ic50_raw.groupby('CANCER_TYPE').agg(
        n_cell_lines=('CELL_LINE_NAME', 'nunique'),
        n_drugs=('DRUG_NAME', 'nunique'),
    ).sort_values('n_cell_lines', ascending=False)
    
    for ct in cancer_ic50.head(10).index:
        ct_cells = ic50_raw[ic50_raw['CANCER_TYPE'] == ct]['CELL_LINE_NAME'].unique()
        ct_cells_in_netith = [c for c in ct_cells if c in netith.index]
        if len(ct_cells_in_netith) >= 5:
            ct_netith = netith.loc[ct_cells_in_netith, 'NetITH']
            print(f"    {ct:35s}: NetITH={ct_netith.mean():.3f}+/-{ct_netith.std():.3f} (n={len(ct_netith)})")
    
    # ── Plot ──
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # A: Volcano plot (rho vs -log10 p)
    ax = axes[0, 0]
    results_df['neg_log10_p'] = -np.log10(results_df['p_spearman'].clip(1e-10))
    colors = ['#e74c3c' if fdr < 0.05 else '#3498db' if abs(rho) > 0.1 else '#95a5a6'
              for rho, fdr in zip(results_df['rho'], results_df['fdr'])]
    ax.scatter(results_df['rho'], results_df['neg_log10_p'], c=colors, alpha=0.6, s=15)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    # Label top hits
    for _, r in results_df.nsmallest(5, 'fdr').iterrows():
        ax.annotate(r['drug'], (r['rho'], r['neg_log10_p']), fontsize=6, alpha=0.8)
    for _, r in results_df.nlargest(5, 'fdr').iterrows():
        ax.annotate(r['drug'], (r['rho'], r['neg_log10_p']), fontsize=6, alpha=0.8)
    ax.set_xlabel("Spearman's rho (NetITH × LN_IC50)")
    ax.set_ylabel('-log10(p)')
    ax.set_title('Volcano: NetITH × Drug Sensitivity')
    
    # B: Top 15 negative rho drugs (high NetITH = sensitive)
    ax = axes[0, 1]
    top_neg = results_df.nsmallest(15, 'rho')
    bars = ax.barh(range(len(top_neg)), -top_neg['rho'].values, 
                   color=['#2ecc71' if fdr < 0.1 else '#bdc3c7' for fdr in top_neg['fdr']])
    ax.set_yticks(range(len(top_neg)))
    ax.set_yticklabels(top_neg['drug'].values, fontsize=7)
    ax.set_xlabel('-rho (high NetITH → more sensitive)')
    ax.set_title('Drugs More Effective in High-NetITH Cells')
    ax.invert_yaxis()
    
    # C: Top 15 positive rho drugs (high NetITH = resistant)
    ax = axes[0, 2]
    top_pos = results_df.nlargest(15, 'rho')
    bars = ax.barh(range(len(top_pos)), top_pos['rho'].values,
                   color=['#e74c3c' if fdr < 0.1 else '#bdc3c7' for fdr in top_pos['fdr']])
    ax.set_yticks(range(len(top_pos)))
    ax.set_yticklabels(top_pos['drug'].values, fontsize=7)
    ax.set_xlabel('rho (high NetITH → more resistant)')
    ax.set_title('Drugs More Effective in Low-NetITH Cells')
    ax.invert_yaxis()
    
    # D: Pathway-level mean rho
    ax = axes[1, 0]
    pw_sorted = pathway_stats[pathway_stats['n_drugs'] >= 3].sort_values('mean_rho')
    colors_pw = ['#2ecc71' if r < 0 else '#e74c3c' for r in pw_sorted['mean_rho']]
    ax.barh(range(len(pw_sorted)), pw_sorted['mean_rho'].values, color=colors_pw)
    ax.set_yticks(range(len(pw_sorted)))
    ax.set_yticklabels(pw_sorted.index, fontsize=7)
    ax.set_xlabel('Mean rho (NetITH × IC50)')
    ax.set_title('Drug Sensitivity by Pathway')
    ax.axvline(0, color='gray', alpha=0.5)
    ax.invert_yaxis()
    
    # E: NetITH by cancer type
    ax = axes[1, 1]
    ct_data = []
    for ct in cancer_ic50.head(12).index:
        ct_cells = [c for c in ic50_raw[ic50_raw['CANCER_TYPE'] == ct]['CELL_LINE_NAME'].unique() 
                    if c in netith.index]
        if len(ct_cells) >= 5:
            ct_data.append({'cancer': ct, 'values': netith.loc[ct_cells, 'NetITH'].values})
    for i, d in enumerate(ct_data):
        bp = ax.boxplot([d['values']], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor(sns.color_palette("tab20", len(ct_data))[i])
    ax.set_xticks(range(len(ct_data)))
    ax.set_xticklabels([d['cancer'][:20] for d in ct_data], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('NetITH')
    ax.set_title('NetITH by Cancer Type')
    
    # F: Distribution of correlations
    ax = axes[1, 2]
    ax.hist(results_df['rho'], bins=40, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='gray', linestyle='--')
    ax.axvline(results_df['rho'].mean(), color='red', linestyle='--', 
               label=f'Mean={results_df["rho"].mean():.3f}')
    ax.set_xlabel("Spearman's rho")
    ax.set_ylabel('Number of Drugs')
    ax.set_title('Distribution of NetITH-Drug Correlations')
    ax.legend(fontsize=8)
    
    plt.suptitle('GDSC2: NetITH × Drug Sensitivity (CollecTRI, 239-gene)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "gdsc_netith_drug.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: gdsc_netith_drug.png")
    
    return results_df


# ─── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GDSC2: NetITH × Drug Sensitivity")
    print("=" * 60)
    
    expr_sym = load_expression()
    ic50_raw, ic50_mat = load_ic50()
    netith = compute_netith(expr_sym)
    results_df, netith_sub, ic50_sub = correlate_drugs(netith, ic50_mat, ic50_raw)
    results_df = enrichment_and_plot(results_df, netith, ic50_raw)
    
    # Save
    netith.to_csv(OUTPUT_DIR / "gdsc_netith_cell_lines.csv")
    results_df.to_csv(OUTPUT_DIR / "gdsc_drug_netith_correlations.csv", index=False)
    
    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
