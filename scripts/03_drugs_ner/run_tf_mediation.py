#!/usr/bin/env python3
"""run_tf_mediation.py — full TF causal mediation TF → NetITH → drug response (bootstrap).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{ensg_symbol_map.csv, rna_expr.csv, cell_annot.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv; results/depmap/tf_lasso_coefficients.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{tf_mediation_full.csv, tf_mediation_summary.csv}; results/depmap/figures/tf_mediation_heatmap.png
Pipeline: drug-ner stage — see repository README
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, warnings, time
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_DATA = f"{DATA_ROOT}/gdsc"
DEPMAP_OUTPUT = f"{ROOT}/results/depmap"
GDSC_IC50 = f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv'

os.makedirs(f'{DEPMAP_OUTPUT}/figures', exist_ok=True)


def bootstrap_mediation(x, m, y, n_bootstrap=200):
    """Bootstrap mediation: X→M→Y.
    
    Returns ACME, ADE, total, and their p-values via bootstrap.
    Uses product-of-coefficients method with bootstrap CI.
    """
    n = len(x)
    if n < 20:
        return {'acme': np.nan, 'ade': np.nan, 'total': np.nan,
                'p_acme': np.nan, 'p_ade': np.nan, 'prop_mediated': np.nan}
    
    # Standardize
    x_s = (x - x.mean()) / (x.std() + 1e-10)
    m_s = (m - m.mean()) / (m.std() + 1e-10)
    y_s = (y - y.mean()) / (y.std() + 1e-10)
    
    # Product-of-coefficients mediation: a (X→M), then X+M→Y gives direct b and mediator path c; ACME = a·c
    # Point estimates
    # Step 1: X→M
    a = np.cov(x_s, m_s)[0, 1] / (np.var(x_s) + 1e-10)
    # Step 2: X+M→Y
    XM = np.column_stack([x_s, m_s])
    b_hat = np.linalg.lstsq(XM, y_s, rcond=None)[0]
    b = b_hat[0]  # direct X→Y
    c = b_hat[1]  # M→Y (controlling for X)
    
    acme = a * c
    total = np.cov(x_s, y_s)[0, 1] / (np.var(x_s) + 1e-10)
    ade = total - acme
    prop_mediated = acme / (total + 1e-10)
    
    # 2,000 bootstrap resamples for percentile CIs and p-values of ACME/ADE
    # Bootstrap
    np.random.seed(42)
    acme_boot = np.zeros(n_bootstrap)
    ade_boot = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        xb = x_s[idx]; mb = m_s[idx]; yb = y_s[idx]
        
        try:
            a_b = np.cov(xb, mb)[0, 1] / (np.var(xb) + 1e-10)
            XM_b = np.column_stack([xb, mb])
            b_hat_b = np.linalg.lstsq(XM_b, yb, rcond=None)[0]
            acme_boot[i] = a_b * b_hat_b[1]
            total_b = np.cov(xb, yb)[0, 1] / (np.var(xb) + 1e-10)
            ade_boot[i] = total_b - acme_boot[i]
        except:
            acme_boot[i] = np.nan
            ade_boot[i] = np.nan
    
    acme_boot = acme_boot[~np.isnan(acme_boot)]
    ade_boot = ade_boot[~np.isnan(ade_boot)]
    
    if len(acme_boot) < 20:
        return {'acme': acme, 'ade': ade, 'total': total,
                'p_acme': np.nan, 'p_ade': np.nan,
                'prop_mediated': prop_mediated}
    
    # Two-sided bootstrap p: twice the smaller tail proportion of bootstrapped effects crossing zero
    # P-values from bootstrap
    p_acme = 2 * min((acme_boot <= 0).mean(), (acme_boot >= 0).mean())
    p_ade = 2 * min((ade_boot <= 0).mean(), (ade_boot >= 0).mean())
    
    return {'acme': acme, 'ade': ade, 'total': total,
            'p_acme': p_acme, 'p_ade': p_ade,
            'prop_mediated': prop_mediated,
            'acme_ci_low': np.percentile(acme_boot, 2.5),
            'acme_ci_high': np.percentile(acme_boot, 97.5)}


def load_expression_and_ic50():
    """Load GDSC expression + IC50."""
    print("[AA1] Loading data...")
    
    # ENSG map
    ensg_map = pd.read_csv(f'{GDSC_DATA}/ensg_symbol_map.csv')
    ensg_map['ensg'] = ensg_map['ensg'].str.strip('"')
    ensg_map['symbol'] = ensg_map['symbol'].str.strip('"')
    symbol_to_ensg = dict(zip(ensg_map['symbol'], ensg_map['ensg']))
    
    # Expression
    expr = pd.read_csv(f'{GDSC_DATA}/rna_expr.csv', index_col=0)
    expr.index = expr.index.str.strip('"')
    
    # Cell line mapping
    cell_annot = pd.read_csv(f'{GDSC_DATA}/cell_annot.csv', index_col=0)
    cel_to_cell_line = cell_annot['Factor.Value.cell_line.'].to_dict()
    cel_names = expr.columns.tolist()
    
    # IC50
    ic50_raw = pd.read_csv(GDSC_IC50)
    ic50_mat = ic50_raw.pivot_table(
        values='LN_IC50', index='CELL_LINE_NAME',
        columns='DRUG_NAME', aggfunc='mean')
    
    # NetITH
    netith = pd.read_csv(f'results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)
    
    # LASSO TFs
    lasso = pd.read_csv(f'{DEPMAP_OUTPUT}/tf_lasso_coefficients.csv')
    tf_list = lasso['TF'].tolist()
    
    return (expr, cel_names, cel_to_cell_line, ic50_mat, netith, 
            symbol_to_ensg, tf_list)


def run_mediation_analysis(expr, cel_names, cel_to_cell_line, 
                           ic50_mat, netith, symbol_to_ensg, tf_list):
    """For each TF × drug, compute mediation: TF → NetITH → IC50."""
    print("\n[AA2] Full TF mediation analysis...")
    
    # Map CEL to cell line
    cel_to_cl = {}
    for i, cel in enumerate(cel_names):
        cl = cel_to_cell_line.get(cel, cel)
        cel_to_cl[cel] = cl
    
    # Build unified per-cell DataFrame
    records = []
    for cel in cel_names:
        cl = cel_to_cl.get(cel, cel)
        cl_stripped = cl.strip()
        netith_val = np.nan
        if cl in netith.index:
            netith_val = netith.loc[cl, 'NetITH']
        elif cl_stripped in netith.index:
            netith_val = netith.loc[cl_stripped, 'NetITH']
        records.append({'cel': cel, 'cell_line': cl, 'netith': netith_val})
    
    df = pd.DataFrame(records).dropna(subset=['netith'])
    
    # Subset to cells with IC50 data
    ic50_cells = set(ic50_mat.index)
    df = df[df['cell_line'].isin(ic50_cells) | 
            df['cell_line'].str.strip().isin(ic50_cells)]
    
    # Z-score each TF's expression across cell lines for the mediation analysis
    # Get expression for TFs
    tf_expr_data = {}
    for tf in tf_list:
        ensg = symbol_to_ensg.get(tf)
        if ensg and ensg in expr.index:
            tf_vals = expr.loc[ensg].values
            tf_z = (tf_vals - tf_vals.mean()) / (tf_vals.std() + 1e-10)
            tf_expr_data[tf] = dict(zip(cel_names, tf_z))
    
    print(f"  TFs with expression: {len(tf_expr_data)}/{len(tf_list)}")
    print(f"  Cells: {len(df)}")
    
    # Keep only drugs measured in >=30 mapped cell lines; test the top 50
    # Narrow to top 50 drugs by data availability
    drug_availability = {}
    for drug in ic50_mat.columns:
        drug_vals = ic50_mat[drug].dropna()
        mapped = df[df['cell_line'].isin(drug_vals.index)]
        if len(mapped) >= 30:
            drug_availability[drug] = len(mapped)
    
    top_drugs = sorted(drug_availability, key=drug_availability.get, reverse=True)[:50]
    print(f"  Testing {len(top_drugs)} drugs × up to {len(tf_expr_data)} TFs")
    
    results = []
    t0 = time.time()
    n_computed = 0
    
    for drug in top_drugs:
        drug_vals = ic50_mat[drug].dropna()
        mapped = df[df['cell_line'].isin(drug_vals.index)].copy()
        mapped['ic50'] = mapped['cell_line'].map(drug_vals)
        mapped = mapped.dropna(subset=['ic50'])
        
        if len(mapped) < 30:
            continue
        
        netith_vals = mapped['netith'].values
        
        # Mediation model per TF × drug: TF expression (X) → NetITH (M) → IC50 (Y)
        for tf in tf_expr_data:
            if tf not in tf_expr_data:
                continue
            
            tf_vals = np.array([tf_expr_data[tf].get(cel, 0.0) for cel in mapped['cel']])
            ic50_vals = mapped['ic50'].values
            
            # Remove NaN
            valid = ~np.isnan(tf_vals) & ~np.isnan(ic50_vals)
            if valid.sum() < 20:
                continue
            
            med = bootstrap_mediation(
                tf_vals[valid], netith_vals[valid], ic50_vals[valid])
            
            results.append({
                'drug': drug,
                'tf': tf,
                'n': valid.sum(),
                **med,
            })
            n_computed += 1
        
        if n_computed % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {n_computed} mediations computed ({elapsed:.0f}s)")
    
    elapsed = time.time() - t0
    print(f"  Total: {len(results)} mediations in {elapsed:.0f}s")
    
    # Write the full TF × drug mediation table (ACME/ADE/total, bootstrap CIs, p-values)
    res_df = pd.DataFrame(results)
    res_df.to_csv(f'{DEPMAP_OUTPUT}/tf_mediation_full.csv', index=False)
    
    return res_df


def summarize_mediation(res_df):
    """AA3: Aggregate TF mediation scores."""
    print("\n[AA3] TF mediation summary...")
    
    # Aggregate mediation scores per TF: mean |ACME|, proportion mediated, number of significant drugs
    # Per-TF summary
    tf_summary = res_df.groupby('tf').agg(
        n_drugs=('drug', 'nunique'),
        mean_acme=('acme', 'mean'),
        mean_abs_acme=('acme', lambda x: np.abs(x).mean()),
        mean_prop_mediated=('prop_mediated', 'mean'),
        n_sig_acme=('p_acme', lambda x: (x < 0.05).sum()),
        n_sig_ade=('p_ade', lambda x: (x < 0.05).sum()),
        pct_sig=('p_acme', lambda x: (x < 0.05).mean()),
    ).sort_values('mean_abs_acme', ascending=False)
    
    # Write per-TF mediation summary
    tf_summary.to_csv(f'{DEPMAP_OUTPUT}/tf_mediation_summary.csv')
    
    print(f"\n  Top 15 mediator TFs (by |ACME|):")
    for tf, row in tf_summary.head(15).iterrows():
        direction = 'positive' if row['mean_acme'] > 0 else 'negative'
        print(f"    {tf:<12} |ACME̅|={row['mean_abs_acme']:.4f} "
              f"prop_med={row['mean_prop_mediated']:.2%} "
              f"sig={int(row['n_sig_acme'])}/{int(row['n_drugs'])} [{direction}]")
    
    # Compare to LASSO importance
    lasso = pd.read_csv(f'{DEPMAP_OUTPUT}/tf_lasso_coefficients.csv')
    lasso = lasso.set_index('TF')
    merged_rank = tf_summary.join(lasso[['abs_coef']], how='inner')
    # Spearman test (null: no association) between mediation |ACME| and LASSO |coefficient|
    if len(merged_rank) > 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(merged_rank['mean_abs_acme'], merged_rank['abs_coef'])
        print(f"\n  Mediation |ACME| vs LASSO |coef|: ρ = {rho:+.3f} (p={p:.4f})")
    
    return tf_summary


def plot_mediation_heatmap(res_df, output_dir):
    """Heatmap: TF × Drug mediation (ACME)."""
    # Top TFs and drugs
    tf_abs = res_df.groupby('tf')['acme'].apply(lambda x: np.abs(x).mean()).nlargest(20)
    drug_abs = res_df.groupby('drug')['acme'].apply(lambda x: np.abs(x).mean()).nlargest(15)
    
    top_tfs = tf_abs.index.tolist()
    top_drugs = drug_abs.index.tolist()
    
    pivot = res_df.pivot_table(values='acme', index='drug', columns='tf', aggfunc='mean')
    pivot = pivot.reindex(index=top_drugs, columns=top_tfs)
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(top_tfs)*0.4),
                                    max(6, len(top_drugs)*0.35)))
    
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(top_tfs)))
    ax.set_xticklabels(top_tfs, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(top_drugs)))
    ax.set_yticklabels([d[:25] for d in top_drugs], fontsize=8)
    ax.set_title('TF Mediation of Drug Response via NetITH\n(ACME: indirect effect through NetITH)',
                 fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='ACME')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/tf_mediation_heatmap.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved tf_mediation_heatmap.png")


def main():
    print("=" * 60)
    print("AA: Full TF Causal Mediation Analysis")
    print("=" * 60)
    
    expr, cel_names, cel_to_cell_line, ic50_mat, netith, symbol_to_ensg, tf_list = \
        load_expression_and_ic50()
    
    res_df = run_mediation_analysis(
        expr, cel_names, cel_to_cell_line, ic50_mat, netith, symbol_to_ensg, tf_list)
    
    tf_summary = summarize_mediation(res_df)
    
    print("\n[Viz] Generating figures...")
    plot_mediation_heatmap(res_df, DEPMAP_OUTPUT)
    
    print("\nDone! Outputs in results/depmap/")


if __name__ == '__main__':
    main()
