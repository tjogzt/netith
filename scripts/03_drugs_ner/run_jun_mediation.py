"""run_jun_mediation.py — AP-1/JUN mediation: NetITH's independent contribution to drug sensitivity.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/gdsc/gdsc_jun_mediation.csv; results/gdsc/figures/gdsc_jun_mediation.png
Pipeline: drug-ner stage — see repository README
"""

import numpy as np
import pandas as pd
import os, sys, warnings
from pathlib import Path
from scipy.stats import spearmanr, false_discovery_control
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = f"{DATA_ROOT}/gdsc"
GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
OUTPUT_DIR = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42


# ─── 1. Load Data ──────────────────────────────────────────
def load_data():
    """Load JUN/AP-1 expression, NetITH, IC50."""
    print("[1/4] Loading data...")
    
    # Expression
    expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
    cell_line_map = {}
    for cel, row in annot.iterrows():
        cl = str(row.get('Characteristics.cell.line.', ''))
        if cl and cl != 'nan' and cl != 'NA':
            cell_line_map[cel] = cl
    
    common_cels = [c for c in expr_raw.columns if c in cell_line_map]
    expr_named = expr_raw[common_cels].copy()
    expr_named.columns = [cell_line_map[c] for c in common_cels]
    dup_cols = expr_named.columns.duplicated()
    if dup_cols.any():
        expr_named = expr_named.loc[:, ~dup_cols]
    
    gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
    ensg_to_sym = dict(zip(gene_map['ensg'], gene_map['symbol']))
    expr_named.index = expr_named.index.astype(str)
    matched = expr_named.index.isin(ensg_to_sym.keys())
    expr_sym = expr_named.loc[matched].copy()
    expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
    expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]
    
    # AP-1 family genes
    ap1_genes = ['JUN', 'FOS', 'JUNB', 'JUND', 'FOSL1', 'FOSL2', 'FOSB', 
                 'ATF2', 'ATF3', 'ATF4', 'ATF7', 'BATF', 'BATF2', 'BATF3']
    ap1_available = [g for g in ap1_genes if g in expr_sym.index]
    print(f"  AP-1 genes available: {len(ap1_available)}/{len(ap1_genes)}")
    print(f"  AP-1: {ap1_available}")
    
    # Get AP-1 expression per cell line (z-scored across cell lines)
    ap1_expr = expr_sym.loc[ap1_available]
    ap1_z = ap1_expr.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)
    
    # Also raw expression
    ap1_raw = ap1_expr.T  # cell_line × gene
    
    # NetITH
    netith_df = pd.read_csv(f"{OUTPUT_DIR}/gdsc_netith_cell_lines.csv", index_col=0)
    netith = netith_df['NetITH']
    
    # IC50
    ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    ic50_mat = ic50_raw.pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME',
        values='LN_IC50', aggfunc='mean'
    )
    
    # Drug info
    drug_info = ic50_raw[['DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME']].drop_duplicates('DRUG_NAME')
    drug_info = drug_info.set_index('DRUG_NAME')
    
    # Common cell lines
    ap1_cells = set(ap1_raw.index)
    netith_cells = set(netith.index)
    ic50_cells = set(ic50_mat.index)
    common = sorted(ap1_cells & netith_cells & ic50_cells)
    print(f"  Common cell lines: {len(common)}")
    
    # JUN expression
    jun_expr = ap1_raw.loc[common, 'JUN'] if 'JUN' in ap1_raw.columns else None
    jun_z = (jun_expr - jun_expr.mean()) / (jun_expr.std() + 1e-10)
    
    netith_vals = netith.loc[common]
    ic50_sub = ic50_mat.loc[common]
    
    # Correlation between JUN and NetITH
    rho_jn, p_jn = spearmanr(jun_z, netith_vals)
    print(f"\n  JUN(z-score) × NetITH: rho={rho_jn:.4f}, p={p_jn:.2e}")
    
    # AP-1 composite score (mean z-score of all AP-1 members)
    ap1_composite = ap1_raw.loc[common].mean(axis=1)
    ap1_composite_z = (ap1_composite - ap1_composite.mean()) / (ap1_composite.std() + 1e-10)
    rho_ac, p_ac = spearmanr(ap1_composite_z, netith_vals)
    print(f"  AP-1 composite × NetITH: rho={rho_ac:.4f}, p={p_ac:.2e}")
    
    # All AP-1 members vs NetITH
    print(f"\n  Individual AP-1 × NetITH correlations:")
    for gene in ap1_available:
        g_expr = ap1_raw.loc[common, gene]
        rho, p = spearmanr(g_expr, netith_vals)
        print(f"    {gene:8s}: rho={rho:+.4f}, p={p:.2e}")
    
    return jun_z, ap1_composite_z, netith_vals, ic50_sub, drug_info, ap1_raw.loc[common], ap1_available


# ─── 2. Per-Drug Decomposition ─────────────────────────────
def drug_decomposition(jun_z, netith, ic50_mat, drug_info):
    """
    For each drug:
    - r(JUN × IC50)
    - r(NetITH × IC50)
    - r(JUN × IC50 | NetITH) — partial, controlling for NetITH
    - r(NetITH × IC50 | JUN) — partial, controlling for JUN
    
    Compare: does NetITH add value beyond JUN?
    """
    print("[2/4] Per-drug JUN vs NetITH decomposition...")
    
    results = []
    for drug in ic50_mat.columns:
        drug_ic50 = ic50_mat[drug].dropna()
        common = sorted(set(jun_z.index) & set(netith.index) & set(drug_ic50.index))
        
        if len(common) < 30:
            continue
        
        j = jun_z.loc[common].values
        n = netith.loc[common].values
        d = drug_ic50.loc[common].values
        
        # Simple correlations
        r_j, p_j = spearmanr(j, d)
        r_n, p_n = spearmanr(n, d)
        
        # Partial: JUN controlling for NetITH
        # Regress JUN ~ NetITH, take residuals
        from scipy.stats import linregress
        slope_jn, intercept_jn, _, _, _ = linregress(n, j)
        j_resid = j - (slope_jn * n + intercept_jn)
        r_j_resid, p_j_resid = spearmanr(j_resid, d)
        
        # Partial: NetITH controlling for JUN
        slope_nj, intercept_nj, _, _, _ = linregress(j, n)
        n_resid = n - (slope_nj * j + intercept_nj)
        r_n_resid, p_n_resid = spearmanr(n_resid, d)
        
        # Interaction: JUN × NetITH
        interaction = j * n
        r_int, p_int = spearmanr(interaction, d)
        
        # Which is stronger?
        delta = abs(r_n) - abs(r_j)  # positive = NetITH stronger
        
        # Drug info
        target = str(drug_info.loc[drug, 'PUTATIVE_TARGET']) if drug in drug_info.index else ''
        pathway = str(drug_info.loc[drug, 'PATHWAY_NAME']) if drug in drug_info.index else ''
        
        results.append({
            'drug': drug,
            'target': target,
            'pathway': pathway,
            'n': len(common),
            'r_JUN': r_j, 'p_JUN': p_j,
            'r_NetITH': r_n, 'p_NetITH': p_n,
            'r_JUN_residual': r_j_resid, 'p_JUN_residual': p_j_resid,
            'r_NetITH_residual': r_n_resid, 'p_NetITH_residual': p_n_resid,
            'r_interaction': r_int, 'p_interaction': p_int,
            'delta_r': delta,
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR
    for col in ['p_JUN', 'p_NetITH', 'p_JUN_residual', 'p_NetITH_residual', 'p_interaction']:
        results_df[f'fdr_{col}'] = false_discovery_control(results_df[col].values, method='bh')
    
    # Summaries
    n_jun = (results_df['fdr_p_JUN'] < 0.05).sum()
    n_netith = (results_df['fdr_p_NetITH'] < 0.05).sum()
    n_jun_res = (results_df['fdr_p_JUN_residual'] < 0.05).sum()
    n_netith_res = (results_df['fdr_p_NetITH_residual'] < 0.05).sum()
    n_int = (results_df['fdr_p_interaction'] < 0.05).sum()
    
    print(f"\n  FDR<0.05 drugs:")
    print(f"    JUN alone:         {n_jun}/{len(results_df)}")
    print(f"    NetITH alone:      {n_netith}/{len(results_df)}")
    print(f"    JUN (control NetITH): {n_jun_res}/{len(results_df)}")
    print(f"    NetITH (control JUN): {n_netith_res}/{len(results_df)}")
    print(f"    JUN×NetITH interaction: {n_int}/{len(results_df)}")
    
    print(f"\n  Mean |r|:")
    print(f"    JUN: {results_df['r_JUN'].abs().mean():.4f}")
    print(f"    NetITH: {results_df['r_NetITH'].abs().mean():.4f}")
    print(f"    JUN (resid): {results_df['r_JUN_residual'].abs().mean():.4f}")
    print(f"    NetITH (resid): {results_df['r_NetITH_residual'].abs().mean():.4f}")
    
    # NetITH-dominant vs JUN-dominant drugs
    netith_dominant = (results_df['r_NetITH_residual'].abs() > results_df['r_JUN_residual'].abs())
    print(f"\n  NetITH dominant (|r_resid| > JUN): {netith_dominant.sum()}/{len(results_df)}")
    print(f"  JUN dominant: {(~netith_dominant).sum()}/{len(results_df)}")
    
    # Top NetITH-dominant drugs
    netith_dom = results_df[netith_dominant].nsmallest(10, 'fdr_p_NetITH_residual')
    print(f"\n  Top NetITH-dominant drugs (independent of JUN):")
    for _, r in netith_dom.iterrows():
        print(f"    {r['drug']:30s}  r_N={r['r_NetITH']:+.3f}  r_N|J={r['r_NetITH_residual']:+.3f}  FDR={r['fdr_p_NetITH_residual']:.4f}")
    
    return results_df


# ─── 3. Pathway-Level Summary ──────────────────────────────
def pathway_summary(results_df):
    """Summarize JUN vs NetITH contribution by drug pathway."""
    print("[3/4] Pathway-level summary...")
    
    # Group by pathway
    pathway_stats = results_df.groupby('pathway').agg(
        n_drugs=('drug', 'count'),
        mean_r_JUN=('r_JUN', 'mean'),
        mean_r_NetITH=('r_NetITH', 'mean'),
        mean_r_JUN_resid=('r_JUN_residual', 'mean'),
        mean_r_NetITH_resid=('r_NetITH_residual', 'mean'),
        n_sig_JUN=('fdr_p_JUN', lambda x: (x < 0.05).sum()),
        n_sig_NetITH=('fdr_p_NetITH', lambda x: (x < 0.05).sum()),
        n_sig_NetITH_resid=('fdr_p_NetITH_residual', lambda x: (x < 0.05).sum()),
    ).sort_values('mean_r_NetITH')
    
    print(f"\n  Pathway summary (top/bottom 10 by NetITH):")
    for pathway, row in pd.concat([pathway_stats.head(10), pathway_stats.tail(10)]).iterrows():
        print(f"    {str(pathway)[:35]:35s}  n={int(row['n_drugs']):3d}  "
              f"r_JUN={row['mean_r_JUN']:+.4f}  r_N={row['mean_r_NetITH']:+.4f}  "
              f"r_N|J={row['mean_r_NetITH_resid']:+.4f}  sig_N|J={int(row['n_sig_NetITH_resid'])}")
    
    return pathway_stats


# ─── 4. Visualization ──────────────────────────────────────
def plot_jun_mediation(jun_z, netith, results_df, pathway_stats, ap1_df, ap1_genes):
    """Multi-panel JUN mediation figure."""
    print("[4/4] Plotting...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    
    # A: JUN vs NetITH scatter
    ax = axes[0, 0]
    ax.scatter(jun_z, netith, alpha=0.3, s=10, c='#e74c3c', edgecolors='none')
    rho, p = spearmanr(jun_z, netith)
    from numpy.polynomial.polynomial import polyfit
    mask = ~(np.isnan(jun_z) | np.isnan(netith))
    b, m = polyfit(jun_z[mask], netith[mask], 1)
    x_line = np.linspace(jun_z.min(), jun_z.max(), 100)
    ax.plot(x_line, b + m * x_line, 'navy', alpha=0.5, linewidth=2)
    ax.set_xlabel('JUN Expression (z-score)')
    ax.set_ylabel('NetITH (bits)')
    ax.set_title(f'JUN vs NetITH\nρ={rho:.4f}, p={p:.2e}, R²={rho**2:.2f}')
    
    # B: r(JUN) vs r(NetITH) across drugs
    ax = axes[0, 1]
    ax.scatter(results_df['r_JUN'], results_df['r_NetITH'], 
              alpha=0.4, s=12, c='#2ecc71', edgecolors='none')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.plot([-0.4, 0.4], [-0.4, 0.4], 'r--', alpha=0.3)
    
    # Highlight NetITH-dominant drugs
    nd_mask = results_df['r_NetITH_residual'].abs() > results_df['r_JUN_residual'].abs()
    ax.scatter(results_df.loc[nd_mask, 'r_JUN'], results_df.loc[nd_mask, 'r_NetITH'],
              alpha=0.7, s=20, c='#e74c3c', edgecolors='black', linewidth=0.3,
              label=f'NetITH-dominant ({nd_mask.sum()})')
    
    ax.set_xlabel('r (JUN × IC50)')
    ax.set_ylabel('r (NetITH × IC50)')
    ax.set_title('Drug Sensitivity: JUN vs NetITH')
    ax.legend(fontsize=7)
    
    # C: Residual comparison: NetITH|JUN vs JUN|NetITH
    ax = axes[0, 2]
    sig_mask = (results_df['fdr_p_NetITH_residual'] < 0.05) | (results_df['fdr_p_JUN_residual'] < 0.05)
    ax.scatter(results_df.loc[~sig_mask, 'r_JUN_residual'], 
              results_df.loc[~sig_mask, 'r_NetITH_residual'],
              alpha=0.2, s=8, c='gray')
    ax.scatter(results_df.loc[sig_mask, 'r_JUN_residual'],
              results_df.loc[sig_mask, 'r_NetITH_residual'],
              alpha=0.6, s=15, c='#3498db', edgecolors='black', linewidth=0.2)
    
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    lim = max(abs(results_df[['r_JUN_residual', 'r_NetITH_residual']].min().min()),
              abs(results_df[['r_JUN_residual', 'r_NetITH_residual']].max().max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('r (JUN × IC50 | NetITH)')
    ax.set_ylabel('r (NetITH × IC50 | JUN)')
    ax.set_title('Independent Contributions\n(After Controlling for Each Other)')
    
    # D: Pathway-level mean r comparison
    ax = axes[1, 0]
    pw_top = pathway_stats.nsmallest(15, 'mean_r_NetITH')
    x = np.arange(len(pw_top))
    w = 0.35
    ax.barh(x - w/2, pw_top['mean_r_JUN'], w, label='JUN', alpha=0.8, color='#e74c3c')
    ax.barh(x + w/2, pw_top['mean_r_NetITH_resid'], w, label='NetITH|JUN', alpha=0.8, color='#3498db')
    ax.set_yticks(x)
    ax.set_yticklabels([str(p)[:30] for p in pw_top.index], fontsize=6)
    ax.set_xlabel('Mean r')
    ax.set_title('Pathway: JUN vs NetITH|JUN')
    ax.legend(fontsize=7)
    
    # E: All AP-1 members vs NetITH heatmap-like bar
    ax = axes[1, 1]
    ap1_corrs = {}
    for gene in ap1_genes:
        if gene in ap1_df.columns:
            rho, p = spearmanr(ap1_df[gene], netith)
            ap1_corrs[gene] = rho
    
    ap1_sorted = sorted(ap1_corrs.items(), key=lambda x: x[1])
    genes = [x[0] for x in ap1_sorted]
    corrs = [x[1] for x in ap1_sorted]
    colors_ap1 = ['#e74c3c' if c > 0 else '#3498db' for c in corrs]
    ax.barh(range(len(genes)), corrs, color=colors_ap1, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)
    ax.axvline(0, color='black', alpha=0.3)
    ax.set_xlabel('ρ (Expression × NetITH)')
    ax.set_title('AP-1 Family × NetITH')
    
    # F: Venn-like: drugs sig by JUN-only, NetITH-only, both
    ax = axes[1, 2]
    jun_only = (results_df['fdr_p_JUN'] < 0.05) & (results_df['fdr_p_NetITH'] >= 0.05)
    n_only = (results_df['fdr_p_JUN'] >= 0.05) & (results_df['fdr_p_NetITH'] < 0.05)
    both = (results_df['fdr_p_JUN'] < 0.05) & (results_df['fdr_p_NetITH'] < 0.05)
    neither = (results_df['fdr_p_JUN'] >= 0.05) & (results_df['fdr_p_NetITH'] >= 0.05)
    
    categories = ['JUN only', 'Both', 'NetITH only', 'Neither']
    counts = [jun_only.sum(), both.sum(), n_only.sum(), neither.sum()]
    colors_v = ['#e74c3c', '#9b59b6', '#3498db', '#bdc3c7']
    
    ax.bar(range(4), counts, color=colors_v, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Number of Drugs (FDR<0.05)')
    
    for i, (cat, cnt) in enumerate(zip(categories, counts)):
        ax.text(i, cnt + 1, str(cnt), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'Drug Sensitivity Predictors\n(Total: {len(results_df)} drugs)')
    
    plt.tight_layout()
    figpath = f"{OUTPUT_DIR}/figures/gdsc_jun_mediation.png"
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {figpath}")
    plt.close()


def main():
    jun_z, ap1_composite, netith, ic50_mat, drug_info, ap1_df, ap1_genes = load_data()
    results_df = drug_decomposition(jun_z, netith, ic50_mat, drug_info)
    pathway_stats = pathway_summary(results_df)
    
    # Save
    results_df.to_csv(f"{OUTPUT_DIR}/gdsc_jun_mediation.csv", index=False)
    
    # Plot
    plot_jun_mediation(jun_z, netith, results_df, pathway_stats, ap1_df, ap1_genes)
    
    # Key summary stat
    n_netith_only = ((results_df['fdr_p_NetITH'] < 0.05) & (results_df['fdr_p_JUN'] >= 0.05)).sum()
    n_jun_only = ((results_df['fdr_p_JUN'] < 0.05) & (results_df['fdr_p_NetITH'] >= 0.05)).sum()
    n_both = ((results_df['fdr_p_JUN'] < 0.05) & (results_df['fdr_p_NetITH'] < 0.05)).sum()
    
    print("\n" + "=" * 60)
    print("SUMMARY: JUN vs NetITH — Independent Contributions")
    print("=" * 60)
    print(f"  JUN × NetITH: ρ={spearmanr(jun_z, netith)[0]:.4f}")
    print(f"  Drugs: JUN-only sig={n_jun_only}, NetITH-only sig={n_netith_only}, Both={n_both}")
    
    if n_netith_only > 0:
        print(f"\n  ** NetITH retains independent signal for {n_netith_only} drugs beyond JUN **")
        print(f"  Conclusion: NetITH is NOT fully explained by JUN expression")
    else:
        print(f"\n  ** NetITH signal largely captured by JUN expression **")
    
    # NetITH residual significance
    n_netith_res = (results_df['fdr_p_NetITH_residual'] < 0.05).sum()
    print(f"  NetITH|JUN FDR<0.05: {n_netith_res}/{len(results_df)}")
    
    n_jun_res = (results_df['fdr_p_JUN_residual'] < 0.05).sum()
    print(f"  JUN|NetITH FDR<0.05: {n_jun_res}/{len(results_df)}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
