"""
run_gdsc_multiomics.py — Compare functional (NetITH) vs genomic (CNV) entropy contributions to GDSC drug sensitivity.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/gdsc/cnv_expr.csv: GDSC copy-number matrix
    - results/gdsc/gdsc_netith_cell_lines.csv: precomputed NetITH per cell line
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 drug sensitivity (LN_IC50)
Outputs :
    - results/gdsc/gdsc_multiomics_results.csv
    - results/gdsc/figures/gdsc_multiomics.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os, sys, warnings
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from scipy.linalg import eigvalsh
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
MIN_DRUG_CELLS = 30  # minimum cell lines per drug to test correlation


# ─── 1. Load CNV & Compute Genomic Entropy ─────────────────
def compute_cnv_entropy():
    """Shannon entropy of CNV profile per cell line — genomic instability."""
    print("[1/4] Computing CNV entropy...")
    
    cnv = pd.read_csv(f"{DATA_DIR}/cnv_expr.csv", index_col=0)
    print(f"  CNV: {cnv.shape}")
    
    # Per-cell-line Shannon entropy of CNV distribution
    # Bin CNV values into categories: deep del (<-1), shallow del (-1 to -0.3), 
    # neutral (-0.3 to 0.3), gain (0.3 to 1), amp (>1)
    bins = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]
    
    cnv_entropies = {}
    for cl in cnv.columns:
        vals = cnv[cl].dropna().values
        if len(vals) < 100:
            continue
        hist, _ = np.histogram(vals, bins=bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        cnv_entropies[cl] = -np.sum(hist * np.log2(hist))
    
    cnv_ent = pd.Series(cnv_entropies, name='CNV_entropy')
    print(f"  Cell lines: {len(cnv_ent)}")
    print(f"  CNV entropy range: [{cnv_ent.min():.3f}, {cnv_ent.max():.3f}]")
    
    # Also compute: fraction of genome altered (FGA)
    fga = {}
    for cl in cnv.columns:
        vals = cnv[cl].dropna().values
        altered = (np.abs(vals) > 0.3).mean()
        fga[cl] = altered
    fga_series = pd.Series(fga, name='FGA')
    
    return cnv_ent, fga_series


# ─── 2. Load NetITH & Merge ────────────────────────────────
def load_netith():
    """Load pre-computed NetITH from GDSC analysis."""
    print("[2/4] Loading NetITH...")
    netith = pd.read_csv(f"{OUTPUT_DIR}/gdsc_netith_cell_lines.csv", index_col=0)
    print(f"  NetITH: {len(netith)} cell lines")
    return netith['NetITH']


# ─── 3. Combined Drug Sensitivity Model ────────────────────
def combined_drug_analysis(netith, cnv_ent, fga):
    """For each drug: NetITH-only vs CNV-only vs Combined correlation."""
    print("[3/4] Combined NetITH + CNV entropy vs Drug IC50...")
    
    # Load IC50
    GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
    ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
    ic50_mat = ic50_raw.pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME',
        values='LN_IC50', aggfunc='mean'
    )
    
    # Merge all features
    common_cells = sorted(set(netith.index) & set(cnv_ent.index) & set(ic50_mat.index))
    print(f"  Common cell lines: {len(common_cells)}")
    
    netith_sub = netith.loc[common_cells]
    cnv_sub = cnv_ent.loc[common_cells]
    fga_sub = fga.loc[common_cells]
    ic50_sub = ic50_mat.loc[common_cells]
    
    # Correlation among features
    r_nc, p_nc = spearmanr(netith_sub, cnv_sub)
    r_nf, p_nf = spearmanr(netith_sub, fga_sub)
    r_cf, p_cf = spearmanr(cnv_sub, fga_sub)
    print(f"\n  Feature correlations:")
    print(f"    NetITH × CNV_entropy: rho={r_nc:.3f}, p={p_nc:.4f}")
    print(f"    NetITH × FGA: rho={r_nf:.3f}, p={p_nf:.4f}")
    print(f"    CNV_entropy × FGA: rho={r_cf:.3f}, p={p_cf:.4f}")
    
    # For each drug: compare NetITH, CNV, and NetITH+CNV residual
    results = []
    for drug in ic50_sub.columns:
        drug_data = pd.DataFrame({
            'NetITH': netith_sub,
            'CNV_entropy': cnv_sub,
            'FGA': fga_sub,
            'LN_IC50': ic50_sub[drug]
        }).dropna()
        
        if len(drug_data) < MIN_DRUG_CELLS:
            continue
        
        # NetITH alone
        r_n, p_n = spearmanr(drug_data['NetITH'], drug_data['LN_IC50'])
        
        # CNV entropy alone
        r_c, p_c = spearmanr(drug_data['CNV_entropy'], drug_data['LN_IC50'])
        
        # FGA alone
        r_f, p_f = spearmanr(drug_data['FGA'], drug_data['LN_IC50'])
        
        # NetITH residual (after controlling for CNV): partial correlation
        # Regress NetITH ~ CNV_entropy, take residuals, correlate with IC50
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(drug_data['CNV_entropy'], drug_data['NetITH'])
        netith_residual = drug_data['NetITH'] - (slope * drug_data['CNV_entropy'] + intercept)
        r_n_resid, p_n_resid = spearmanr(netith_residual, drug_data['LN_IC50'])
        
        # CNV residual (after controlling for NetITH)
        slope2, intercept2, _, _, _ = linregress(drug_data['NetITH'], drug_data['CNV_entropy'])
        cnv_residual = drug_data['CNV_entropy'] - (slope2 * drug_data['NetITH'] + intercept2)
        r_c_resid, p_c_resid = spearmanr(cnv_residual, drug_data['LN_IC50'])
        
        # Drug info
        drug_info = ic50_raw[ic50_raw['DRUG_NAME'] == drug].iloc[0]
        
        results.append({
            'drug': drug,
            'target': drug_info['PUTATIVE_TARGET'],
            'pathway': drug_info['PATHWAY_NAME'],
            'n_cells': len(drug_data),
            'r_NetITH': r_n, 'p_NetITH': p_n,
            'r_CNV': r_c, 'p_CNV': p_c,
            'r_FGA': r_f, 'p_FGA': p_f,
            'r_NetITH_residual': r_n_resid, 'p_NetITH_residual': p_n_resid,
            'r_CNV_residual': r_c_resid, 'p_CNV_residual': p_c_resid,
            'delta_r': r_n_resid - r_c_resid,  # positive = NetITH stronger
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    for col in ['p_NetITH', 'p_CNV', 'p_NetITH_residual', 'p_CNV_residual']:
        results_df[f'fdr_{col}'] = results_df[col].apply(
            lambda x: min(x * len(results_df), 1.0)
        )
    
    # Summary
    n_netith_sig = (results_df['fdr_p_NetITH'] < 0.05).sum()
    n_cnv_sig = (results_df['fdr_p_CNV'] < 0.05).sum()
    n_netith_resid_sig = (results_df['fdr_p_NetITH_residual'] < 0.05).sum()
    n_cnv_resid_sig = (results_df['fdr_p_CNV_residual'] < 0.05).sum()
    
    print(f"\n  Drugs with FDR<0.05:")
    print(f"    NetITH alone:    {n_netith_sig}/{len(results_df)}")
    print(f"    CNV alone:       {n_cnv_sig}/{len(results_df)}")
    print(f"    NetITH (resid):  {n_netith_resid_sig}/{len(results_df)} (independent of CNV)")
    print(f"    CNV (resid):     {n_cnv_resid_sig}/{len(results_df)} (independent of NetITH)")
    
    print(f"\n  Mean |r| across drugs:")
    print(f"    NetITH: {results_df['r_NetITH'].abs().mean():.3f}")
    print(f"    CNV:    {results_df['r_CNV'].abs().mean():.3f}")
    print(f"    FGA:    {results_df['r_FGA'].abs().mean():.3f}")
    print(f"    NetITH (residual): {results_df['r_NetITH_residual'].abs().mean():.3f}")
    print(f"    CNV (residual):    {results_df['r_CNV_residual'].abs().mean():.3f}")
    
    # Which is dominant?
    netith_dominant = (results_df['r_NetITH_residual'].abs() > results_df['r_CNV_residual'].abs()).sum()
    print(f"\n  NetITH dominates: {netith_dominant}/{len(results_df)} drugs")
    print(f"  CNV dominates:    {len(results_df) - netith_dominant}/{len(results_df)} drugs")
    
    return results_df, netith_sub, cnv_sub, ic50_sub


# ─── 4. Visualization ──────────────────────────────────────
def plot_multiomics(results_df, netith, cnv_ent, fga):
    """Multi-omics comparison plots."""
    print("[4/4] Plotting...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # A: NetITH vs CNV entropy scatter
    ax = axes[0, 0]
    common = sorted(set(netith.index) & set(cnv_ent.index))
    ax.scatter(cnv_ent[common], netith[common], alpha=0.3, s=10, c='#3498db')
    r, p = spearmanr(netith[common], cnv_ent[common])
    ax.set_xlabel('CNV Entropy (bits)')
    ax.set_ylabel('NetITH (bits)')
    ax.set_title(f'NetITH vs CNV Entropy (rho={r:.3f}, p={p:.2e})')
    
    # B: Scatter: NetITH r vs CNV r across drugs
    ax = axes[0, 1]
    ax.scatter(results_df['r_CNV'], results_df['r_NetITH'], alpha=0.4, s=12, c='#2ecc71')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.plot([-0.3, 0.3], [-0.3, 0.3], 'r--', alpha=0.3)
    ax.set_xlabel('r(CNV entropy × IC50)')
    ax.set_ylabel('r(NetITH × IC50)')
    ax.set_title('Drug Sensitivity: NetITH vs CNV')
    
    # C: Residual comparison
    ax = axes[0, 2]
    ax.scatter(results_df['r_CNV_residual'], results_df['r_NetITH_residual'], 
               alpha=0.4, s=12, c='#e74c3c')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('r(CNV | NetITH)')
    ax.set_ylabel('r(NetITH | CNV)')
    ax.set_title('Independent Contributions (Residual)')
    
    # D: Bar chart — mean |r| comparison
    ax = axes[1, 0]
    metrics = ['NetITH', 'CNV_entropy', 'FGA', 'NetITH\n(residual)', 'CNV\n(residual)']
    mean_abs_r = [
        results_df['r_NetITH'].abs().mean(),
        results_df['r_CNV'].abs().mean(),
        results_df['r_FGA'].abs().mean(),
        results_df['r_NetITH_residual'].abs().mean(),
        results_df['r_CNV_residual'].abs().mean(),
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#27ae60', '#c0392b']
    ax.bar(metrics, mean_abs_r, color=colors, alpha=0.7)
    ax.set_ylabel('Mean |Spearman r|')
    ax.set_title('Average Drug Sensitivity Association')
    
    # E: Pathway-level comparison
    ax = axes[1, 1]
    pw = results_df.groupby('pathway').agg(
        mean_r_NetITH=('r_NetITH', 'mean'),
        mean_r_CNV=('r_CNV', 'mean'),
        mean_r_NetITH_resid=('r_NetITH_residual', 'mean'),
        n=('drug', 'count'),
    ).query('n >= 3').sort_values('mean_r_NetITH_resid')
    
    x = np.arange(len(pw))
    w = 0.25
    ax.barh(x - w, pw['mean_r_NetITH'].values, w, label='NetITH', color='#2ecc71', alpha=0.7)
    ax.barh(x, pw['mean_r_CNV'].values, w, label='CNV', color='#e74c3c', alpha=0.7)
    ax.barh(x + w, pw['mean_r_NetITH_resid'].values, w, label='NetITH (resid)', color='#27ae60', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(pw.index, fontsize=7)
    ax.set_xlabel('Mean rho')
    ax.set_title('Pathway: NetITH vs CNV')
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    
    # F: Density of r values
    ax = axes[1, 2]
    ax.hist(results_df['r_NetITH'], bins=30, alpha=0.5, color='#2ecc71', label='NetITH', density=True)
    ax.hist(results_df['r_CNV'], bins=30, alpha=0.5, color='#e74c3c', label='CNV', density=True)
    ax.hist(results_df['r_NetITH_residual'], bins=30, alpha=0.5, color='#27ae60', 
            label='NetITH (resid)', density=True)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel('Spearman rho')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Drug Associations')
    ax.legend(fontsize=8)
    
    plt.suptitle('GDSC Multi-omics: Functional (NetITH) vs Genomic (CNV) Entropy', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "gdsc_multiomics.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: gdsc_multiomics.png")
    
    return results_df


# ─── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GDSC Multi-omics: NetITH vs CNV Entropy")
    print("=" * 60)
    
    cnv_ent, fga = compute_cnv_entropy()
    netith = load_netith()
    results_df, netith_sub, cnv_sub, ic50_sub = combined_drug_analysis(netith, cnv_ent, fga)
    plot_multiomics(results_df, netith, cnv_ent, fga)
    
    # Save
    results_df.to_csv(OUTPUT_DIR / "gdsc_multiomics_results.csv", index=False)
    
    # Summary for report
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    r_nc, p_nc = spearmanr(netith_sub, cnv_sub)
    print(f"NetITH × CNV correlation: rho={r_nc:.3f}, p={p_nc:.2e}")
    print(f"NetITH drugs FDR<0.05: {(results_df['fdr_p_NetITH']<0.05).sum()}")
    print(f"CNV drugs FDR<0.05: {(results_df['fdr_p_CNV']<0.05).sum()}")
    print(f"NetITH residual FDR<0.05: {(results_df['fdr_p_NetITH_residual']<0.05).sum()}")
    
    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
