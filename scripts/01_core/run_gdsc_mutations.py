"""
run_gdsc_mutations.py — Test whether COSMIC driver mutations drive NetITH (burden, per-gene, mediation).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/gdsc/mutation_matrix.csv: binarized mutation matrix (extracted via R if absent)
    - results/gdsc/gdsc_netith_cell_lines.csv: precomputed NetITH per cell line
    - results/gdsc/gdsc_drug_netith_correlations.csv: NetITH x IC50 correlations
    - <DATA_ROOT>/gdsc_download/GDSC2_molecularProfiles.rds: GDSC2 molecular profiles (R)
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 drug sensitivity (LN_IC50)
Outputs :
    - results/gdsc/gdsc_mutation_netith.csv, gdsc_mutation_mediation.csv
    - results/gdsc/figures/gdsc_mutation_analysis.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os, sys, warnings
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency, false_discovery_control
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = DATA_ROOT / "gdsc"
OUTPUT_DIR = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
MIN_DRUG_CELLS = 30  # minimum cell lines for the NetITH x IC50 mediation test


# ─── 1. Extract Mutation Data via R ────────────────────────
def extract_mutations():
    """Extract mutation matrix from GDSC2_molecularProfiles.rds via R."""
    print("[1/6] Extracting mutation matrix...")
    
    mut_csv = f"{DATA_DIR}/mutation_matrix.csv"
    
    if not os.path.exists(mut_csv):
        r_cmd = f"""
library(SummarizedExperiment)
mf <- readRDS("{DATA_ROOT}/gdsc_download/GDSC2_molecularProfiles.rds")
mut_se <- mf$mutation_exome
mat <- assays(mut_se)[[1]]
bin_mat <- matrix(0, nrow=nrow(mat), ncol=ncol(mat))
rownames(bin_mat) <- rownames(mat)
colnames(bin_mat) <- colnames(mat)
for(i in 1:nrow(mat)) {{
  for(j in 1:ncol(mat)) {{
    if(!is.na(mat[i,j]) && mat[i,j] != "wt" && mat[i,j] != "") {{
      bin_mat[i,j] <- 1
    }}
  }}
}}
write.csv(bin_mat, "{mut_csv}", quote=FALSE)
cat(sprintf("Mutation matrix: %d genes x %d cell lines\\n", nrow(bin_mat), ncol(bin_mat)))
cat(sprintf("Total mutations: %d\\n", sum(bin_mat)))
"""
        
        with open(f"{OUTPUT_DIR}/extract_mutations.R", 'w') as f:
            f.write(r_cmd)
        
        os.system(f"cd {OUTPUT_DIR} && Rscript extract_mutations.R 2>&1")
    
    mut = pd.read_csv(mut_csv, index_col=0)
    print(f"  Mutation matrix: {mut.shape}")
    print(f"  Total mutations: {mut.values.sum()}")
    print(f"  Mutated genes per cell line: mean={mut.sum(axis=0).mean():.1f}, max={mut.sum(axis=0).max()}")
    print(f"  Cell lines with ≥1 mutation: {(mut.sum(axis=0) > 0).sum()}/{mut.shape[1]}")
    
    return mut


# ─── 2. Compute Mutation Burden ─────────────────────────────
def compute_burden(mut):
    """TMB-like: number of mutated genes per cell line."""
    print("[2/6] Computing mutation burden...")
    burden = mut.sum(axis=0)
    burden.name = 'MutationBurden'
    print(f"  Burden: mean={burden.mean():.1f}, median={burden.median():.0f}, max={burden.max()}")
    return burden


# ─── 3. Load NetITH & Merge ────────────────────────────────
def load_and_merge(mut, burden):
    """Merge mutation data with NetITH and drug IC50."""
    print("[3/6] Loading NetITH and merging...")
    
    netith_df = pd.read_csv(f"{OUTPUT_DIR}/gdsc_netith_cell_lines.csv", index_col=0)
    netith = netith_df['NetITH']
    
    # Load IC50
    GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
    ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    ic50_mat = ic50_raw.pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME',
        values='LN_IC50', aggfunc='mean'
    )
    
    # Common cell lines
    mut_cells = set(mut.columns)
    netith_cells = set(netith.index)
    ic50_cells = set(ic50_mat.index)
    common = sorted(mut_cells & netith_cells & ic50_cells)
    print(f"  Common cell lines: {len(common)}")
    
    netith_sub = netith.loc[common]
    burden_sub = burden.loc[common]
    mut_sub = mut.loc[:, common]
    
    # Gene mutation frequencies for filtering
    gene_freq = mut_sub.sum(axis=1)
    genes_testable = gene_freq[(gene_freq >= 5) & (gene_freq <= len(common) - 5)]
    print(f"  Testable genes (≥5 mut, ≥5 wt): {len(genes_testable)}/{len(gene_freq)}")
    
    return netith_sub, burden_sub, mut_sub, genes_testable, ic50_mat.loc[common]


# ─── 4. Mutation Burden × NetITH ──────────────────────────
def burden_analysis(netith, burden):
    """Tier 1: Does overall mutation burden correlate with NetITH?"""
    print("[4/6] Mutation burden vs NetITH...")
    
    rho, p = spearmanr(netith, burden)
    print(f"  MutationBurden × NetITH: rho={rho:.4f}, p={p:.2e}")
    
    # Stratify by burden tertiles
    burden_tertile = pd.qcut(burden, q=3, labels=['Low', 'Medium', 'High'])
    tertile_stats = netith.groupby(burden_tertile).agg(['mean', 'std', 'count'])
    print(f"  NetITH by burden tertile:")
    for label in ['Low', 'Medium', 'High']:
        print(f"    {label}: {tertile_stats.loc[label, 'mean']:.3f} ± {tertile_stats.loc[label, 'std']:.3f} (n={int(tertile_stats.loc[label, 'count'])})")
    
    return rho, p, burden_tertile


# ─── 5. Per-Gene Driver Analysis ──────────────────────────
def per_gene_analysis(netith, mut_sub, genes_testable):
    """Tier 2: For each COSMIC gene, compare NetITH(mut) vs NetITH(wt)."""
    print(f"[5/6] Per-gene NetITH(mut vs wt) for {len(genes_testable)} genes...")
    
    results = []
    for gene in genes_testable.index:
        mut_mask = mut_sub.loc[gene] > 0
        netith_mut = netith[mut_mask]
        netith_wt = netith[~mut_mask]
        
        if len(netith_mut) < 3 or len(netith_wt) < 3:
            continue
        
        stat, p = mannwhitneyu(netith_mut, netith_wt, alternative='two-sided')
        delta = netith_mut.mean() - netith_wt.mean()
        cohens_d = delta / np.sqrt((netith_mut.var() + netith_wt.var()) / 2)
        
        results.append({
            'gene': gene,
            'n_mut': len(netith_mut),
            'n_wt': len(netith_wt),
            'mean_mut': float(netith_mut.mean()),
            'mean_wt': float(netith_wt.mean()),
            'delta': float(delta),
            'cohens_d': float(cohens_d),
            'p_value': p,
        })
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction (Benjamini-Hochberg via false_discovery_control)
    pvals = results_df['p_value'].values
    results_df['fdr'] = false_discovery_control(pvals, method='bh')
    
    n_sig = (results_df['fdr'] < 0.05).sum()
    n_nom = (results_df['p_value'] < 0.05).sum()
    print(f"  Nominal p<0.05: {n_nom}/{len(results_df)}")
    print(f"  FDR<0.05: {n_sig}/{len(results_df)}")
    
    # Top genes
    top = results_df.nsmallest(15, 'fdr')
    print(f"\n  Top 15 genes (by FDR):")
    for _, r in top.iterrows():
        direction = "↑" if r['delta'] > 0 else "↓"
        print(f"    {r['gene']:12s}  mut({r['n_mut']:3d})={r['mean_mut']:.3f}  wt({r['n_wt']:3d})={r['mean_wt']:.3f}  Δ={r['delta']:+.3f} {direction}  d={r['cohens_d']:+.3f}  FDR={r['fdr']:.4f}")
    
    # Check COSMIC functional categories
    cosmic_genes = {
        'TSG': ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'NF1', 'NF2', 
                'VHL', 'CDKN2A', 'SMAD4', 'STK11', 'CDH1', 'WT1', 'ARID1A',
                'ATM', 'ATR', 'ATRX', 'SMARCA4', 'PBRM1', 'SETD2', 'BAP1', 'KMT2D'],
        'Oncogene': ['KRAS', 'NRAS', 'HRAS', 'BRAF', 'PIK3CA', 'EGFR', 'ERBB2',
                     'FGFR1', 'FGFR2', 'FGFR3', 'KIT', 'PDGFRA', 'MET', 'RET',
                     'ALK', 'ROS1', 'IDH1', 'IDH2', 'MYC', 'MYCN', 'CCND1', 'MDM2',
                     'CTNNB1', 'GNAS', 'GNB1'],
        'Chromatin': ['ARID1A', 'ARID2', 'ARID1B', 'SMARCA4', 'SMARCB1', 'PBRM1',
                      'SETD2', 'KMT2A', 'KMT2C', 'KMT2D', 'EZH2', 'CREBBP', 'EP300'],
        'DNA_repair': ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2',
                       'RAD51', 'MLH1', 'MSH2', 'MSH6', 'PMS2'],
    }
    
    # Add category to results
    def get_category(gene):
        cats = []
        for cat, genes in cosmic_genes.items():
            if gene in genes:
                cats.append(cat)
        return ','.join(cats) if cats else 'Other'
    
    results_df['category'] = results_df['gene'].apply(get_category)
    
    return results_df


# ─── 6. Mediation: NetITH × IC50 controlling for mutation ───
def mediation_analysis(netith, mut_sub, ic50_mat, mut_genes_sig):
    """Tier 3: For top NetITH-associated genes, check if NetITH → IC50 is mediated by mutation."""
    print("[6/6] Mediation analysis (NetITH → IC50, controlling for top mutations)...")
    
    GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
    ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    
    # Load previous NetITH-drug correlations
    prev = pd.read_csv(f"{OUTPUT_DIR}/gdsc_drug_netith_correlations.csv")
    
    # Focus on drugs where NetITH is significant (FDR<0.05)
    if 'fdr' in prev.columns:
        netith_drugs = prev[prev['fdr'] < 0.05]['drug'].tolist()
    else:
        # Compute FDR
        pvals = prev['p_value'].values
        prev['fdr'] = false_discovery_control(pvals, method='bh')
        netith_drugs = prev[prev['fdr'] < 0.05]['drug'].tolist()
    
    print(f"  NetITH-significant drugs: {len(netith_drugs)}")
    
    # For top 10 mutation-associated genes, do mediation
    top_genes = mut_genes_sig.nsmallest(10, 'fdr')['gene'].tolist()
    
    mediation_results = []
    for gene in top_genes:
        mut_gene = mut_sub.loc[gene]
        common = sorted(set(netith.index) & set(mut_gene.index) & set(ic50_mat.index))
        
        netith_c = netith.loc[common]
        mut_c = mut_gene.loc[common]
        
        for drug in netith_drugs:
            ic50_c = ic50_mat.loc[common, drug].dropna()
            common_drug = sorted(set(netith_c.index) & set(ic50_c.index))
            
            if len(common_drug) < MIN_DRUG_CELLS:
                continue
            
            n_mut = mut_c[common_drug].sum()
            if n_mut < 3:
                continue
            
            # Total effect: NetITH → IC50
            r_total, p_total = spearmanr(netith_c[common_drug], ic50_c[common_drug])
            
            # Direct effect (controlling for mutation): partial Spearman
            # Simple approach: split by mutation status
            mut_idx = mut_c[common_drug] > 0
            wt_idx = ~mut_idx
            
            r_mut = np.nan
            r_wt = np.nan
            if mut_idx.sum() >= 5:
                r_mut, _ = spearmanr(netith_c[common_drug][mut_idx], ic50_c[common_drug][mut_idx])
            if wt_idx.sum() >= 5:
                r_wt, _ = spearmanr(netith_c[common_drug][wt_idx], ic50_c[common_drug][wt_idx])
            
            mediation_results.append({
                'gene': gene,
                'drug': drug,
                'n': len(common_drug),
                'n_mut': int(mut_idx.sum()),
                'r_total': r_total,
                'p_total': p_total,
                'r_mut': r_mut,
                'r_wt': r_wt,
                'delta_r': (r_mut - r_wt) if not (np.isnan(r_mut) or np.isnan(r_wt)) else np.nan,
            })
    
    mediation_df = pd.DataFrame(mediation_results)
    print(f"  Mediation tests: {len(mediation_df)}")
    if len(mediation_df) > 0:
        avg_total = mediation_df['r_total'].mean()
        avg_mut = mediation_df['r_mut'].dropna().mean()
        avg_wt = mediation_df['r_wt'].dropna().mean()
        print(f"  Mean r_total: {avg_total:.3f}, r_mut: {avg_mut:.3f}, r_wt: {avg_wt:.3f}")
    
    return mediation_df


# ─── 7. Visualization ──────────────────────────────────────
def plot_results(results_df, netith, burden, burden_tertile, mediation_df):
    """Comprehensive mutation analysis figures."""
    print("[7/7] Plotting...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    
    # A: Mutation burden vs NetITH scatter
    ax = axes[0, 0]
    ax.scatter(burden, netith, alpha=0.3, s=12, c='#3498db', edgecolors='none')
    rho, p = spearmanr(netith, burden)
    # Add trend line
    from numpy.polynomial.polynomial import polyfit
    mask = ~(np.isnan(netith) | np.isnan(burden))
    b, m = polyfit(burden[mask], netith[mask], 1)
    x_line = np.linspace(burden.min(), burden.max(), 100)
    ax.plot(x_line, b + m * x_line, 'r-', alpha=0.5, linewidth=2)
    ax.set_xlabel('Mutation Burden (n mutated genes)')
    ax.set_ylabel('NetITH (bits)')
    ax.set_title(f'Mutation Burden vs NetITH\nrho={rho:.4f}, p={p:.2e}')
    
    # B: NetITH by burden tertile boxplot
    ax = axes[0, 1]
    data_by_tertile = [netith[burden_tertile == t].values for t in ['Low', 'Medium', 'High']]
    bp = ax.boxplot(data_by_tertile, labels=['Low', 'Medium', 'High'], patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel('Mutation Burden Tertile')
    ax.set_ylabel('NetITH (bits)')
    ax.set_title('NetITH by Mutation Burden Level')
    
    # C: Volcano plot: ΔNetITH(mut-wt) vs -log10(FDR)
    ax = axes[0, 2]
    results_valid = results_df.dropna(subset=['delta', 'fdr']).copy()
    results_valid['neg_log10_fdr'] = -np.log10(results_valid['fdr'].clip(1e-30))
    
    sig_mask = results_valid['fdr'] < 0.05
    up_mask = (results_valid['delta'] > 0) & sig_mask
    down_mask = (results_valid['delta'] < 0) & sig_mask
    
    ax.scatter(results_valid.loc[~sig_mask, 'delta'], 
              results_valid.loc[~sig_mask, 'neg_log10_fdr'],
              alpha=0.3, s=8, c='gray', label='NS')
    ax.scatter(results_valid.loc[up_mask, 'delta'],
              results_valid.loc[up_mask, 'neg_log10_fdr'],
              alpha=0.7, s=30, c='#e74c3c', edgecolors='black', linewidth=0.3,
              label=f'Mut↑NetITH ({(results_valid["delta"]>0)&sig_mask.sum()})')
    ax.scatter(results_valid.loc[down_mask, 'delta'],
              results_valid.loc[down_mask, 'neg_log10_fdr'],
              alpha=0.7, s=30, c='#3498db', edgecolors='black', linewidth=0.3,
              label=f'Mut↓NetITH ({(results_valid["delta"]<0)&sig_mask.sum()})')
    
    # Label top genes
    top10 = results_valid.nsmallest(10, 'fdr')
    for _, r in top10.iterrows():
        ax.annotate(r['gene'], (r['delta'], r['neg_log10_fdr']),
                   fontsize=7, alpha=0.8,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.2)
    ax.set_xlabel('Δ NetITH (mut − wt)')
    ax.set_ylabel('−log₁₀(FDR)')
    ax.set_title('Driver Gene × NetITH Volcano')
    ax.legend(fontsize=7, loc='upper right')
    
    # D: Top 20 genes barplot
    ax = axes[1, 0]
    top20 = results_valid.nsmallest(20, 'fdr').sort_values('delta', ascending=True)
    colors_bar = ['#e74c3c' if d > 0 else '#3498db' for d in top20['delta']]
    ax.barh(range(len(top20)), top20['delta'], color=colors_bar, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['gene'], fontsize=7)
    ax.axvline(0, color='black', alpha=0.3)
    ax.set_xlabel('Δ NetITH (mut − wt)')
    ax.set_title(f'Top 20 Driver Mutations → NetITH')
    
    # Add FDR annotation
    for i, (_, r) in enumerate(top20.iterrows()):
        sig = '***' if r['fdr'] < 0.001 else '**' if r['fdr'] < 0.01 else '*' if r['fdr'] < 0.05 else ''
        if sig:
            x_pos = r['delta'] + (0.02 if r['delta'] > 0 else -0.02)
            ax.text(x_pos, i, sig, va='center', fontsize=8)
    
    # E: Category summary
    ax = axes[1, 1]
    cat_deltas = results_valid.groupby('category')['delta'].agg(['mean', 'std', 'count'])
    cat_deltas = cat_deltas[cat_deltas['count'] >= 3].sort_values('mean')
    
    colors_cat = ['#e74c3c' if m > 0 else '#3498db' for m in cat_deltas['mean']]
    ax.barh(range(len(cat_deltas)), cat_deltas['mean'], color=colors_cat, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(cat_deltas)))
    ax.set_yticklabels(cat_deltas.index, fontsize=8)
    ax.axvline(0, color='black', alpha=0.3)
    ax.set_xlabel('Mean Δ NetITH (mut − wt)')
    ax.set_title('NetITH shift by Gene Category')
    
    # F: Mediation: r_total vs r_mut/wt
    ax = axes[1, 2]
    if len(mediation_df) > 0:
        med_valid = mediation_df.dropna(subset=['r_mut', 'r_wt', 'r_total'])
        if len(med_valid) > 0:
            ax.scatter(med_valid['r_total'], med_valid['r_mut'], 
                      alpha=0.5, s=15, c='#e74c3c', label='Mut subgroup', edgecolors='none')
            ax.scatter(med_valid['r_total'], med_valid['r_wt'],
                      alpha=0.5, s=15, c='#3498db', label='WT subgroup', edgecolors='none')
            
            lim = max(abs(med_valid[['r_total', 'r_mut', 'r_wt']].min().min()),
                     abs(med_valid[['r_total', 'r_mut', 'r_wt']].max().max())) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'gray', alpha=0.3, linestyle='--')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        
    ax.set_xlabel('r (NetITH × IC50) — total')
    ax.set_ylabel('r (NetITH × IC50) — by mutation status')
    ax.set_title('Mediation: NetITH→IC50 by Mutation Status')
    ax.legend(fontsize=7)
    ax.axhline(0, color='gray', alpha=0.2)
    ax.axvline(0, color='gray', alpha=0.2)
    
    plt.tight_layout()
    figpath = f"{OUTPUT_DIR}/figures/gdsc_mutation_analysis.png"
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {figpath}")
    plt.close()


def main():
    # ─── Extract ───
    mut = extract_mutations()
    
    # ─── Burden ───
    burden = compute_burden(mut)
    
    # ─── Merge ───
    netith, burden_sub, mut_sub, genes_testable, ic50_mat = load_and_merge(mut, burden)
    
    # ─── Tier 1: Burden ───
    rho_burden, p_burden, burden_tertile = burden_analysis(netith, burden_sub)
    
    # ─── Tier 2: Per-gene ───
    results_df = per_gene_analysis(netith, mut_sub, genes_testable)
    results_df.to_csv(f"{OUTPUT_DIR}/gdsc_mutation_netith.csv", index=False)
    
    # ─── Tier 3: Mediation (only if significant genes found) ───
    sig_genes = results_df[results_df['fdr'] < 0.05]
    if len(sig_genes) >= 5:
        mediation_df = mediation_analysis(netith, mut_sub, ic50_mat, sig_genes)
        mediation_df.to_csv(f"{OUTPUT_DIR}/gdsc_mutation_mediation.csv", index=False)
    else:
        print(f"\n  Only {len(sig_genes)} FDR-significant genes — skipping mediation analysis")
        mediation_df = pd.DataFrame()
    
    # ─── Plot ───
    plot_results(results_df, netith, burden_sub, burden_tertile, mediation_df)
    
    # ─── Summary ───
    print("\n" + "=" * 60)
    print("SUMMARY: Driver Mutation Analysis")
    print("=" * 60)
    print(f"  Mutation burden (mean): {burden_sub.mean():.1f} genes")
    print(f"  Burden × NetITH: rho={rho_burden:.4f}, p={p_burden:.2e}")
    
    n_sig = (results_df['fdr'] < 0.05).sum()
    print(f"  FDR-significant driver genes: {n_sig}/{len(results_df)}")
    
    top5 = results_df.nsmallest(5, 'fdr')
    print(f"  Top 5 genes:")
    for _, r in top5.iterrows():
        direction = "↑NetITH" if r['delta'] > 0 else "↓NetITH"
        print(f"    {r['gene']}: Δ={r['delta']:+.3f} ({direction}), d={r['cohens_d']:+.3f}, FDR={r['fdr']:.4f}")
    
    print(f"\n  Key finding: Mutation burden {'IS' if p_burden < 0.05 else 'is NOT'} significantly associated with NetITH")
    print(f"  Key finding: {n_sig} driver genes {'show' if n_sig > 0 else 'show no'} significant NetITH perturbation")
    print("=" * 60)


if __name__ == "__main__":
    main()
