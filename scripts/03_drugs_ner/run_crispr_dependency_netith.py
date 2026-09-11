#!/usr/bin/env python3
"""run_crispr_dependency_netith.py — genomic dependency × NetITH interaction via mutation and CNV proxies.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{mutation_matrix.csv, cnv_expr.csv, rna_expr.csv, ensg_symbol_map.csv, cell_annot.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/{crispr_dependency_netith.csv, cnv_netith_interaction.csv, essentiality_netith_interaction.csv}; results/depmap/figures/genomic_netith_interaction.png
Pipeline: drug-ner stage — see repository README
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_DATA = f"{DATA_ROOT}/gdsc"
DEPMAP_OUTPUT = f"{ROOT}/results/depmap"
GDSC_IC50 = f'{DATA_ROOT}/gdsc_download/GDSC2_IC50_all.csv'

os.makedirs(f'{DEPMAP_OUTPUT}/figures', exist_ok=True)

# Known cancer essential genes (DepMap common essentials)
ESSENTIAL_GENES = [
    'MYC', 'RPL11', 'RPS6', 'PCNA', 'POLR2A', 'SF3B1', 'UBA52',
    'EEF1A1', 'ACTB', 'GAPDH', 'HSP90AA1', 'HSPA8', 'RAN',
    'CCNB1', 'CDK1', 'CDC20', 'BIRC5', 'TOP2A',
]


def load_data():
    """Load GDSC mutations, CNV, expression + NetITH."""
    print("[Z-2a] Loading genomic data...")

    # Mutations (binary matrix) — proxy for gene dependency
    # Mutations (binary matrix)
    mut_raw = pd.read_csv(f'{GDSC_DATA}/mutation_matrix.csv', index_col=0)
    print(f"  Mutation matrix: {mut_raw.shape}")

    # CNV 
    cnv_raw = pd.read_csv(f'{GDSC_DATA}/cnv_expr.csv', index_col=0)
    print(f"  CNV matrix: {cnv_raw.shape}")

    # Expression
    expr_raw = pd.read_csv(f'{GDSC_DATA}/rna_expr.csv', index_col=0)
    expr_raw.index = expr_raw.index.str.strip('"')
    print(f"  Expression: {expr_raw.shape}")

    # ENSG map
    ensg_map = pd.read_csv(f'{GDSC_DATA}/ensg_symbol_map.csv')
    ensg_map['ensg'] = ensg_map['ensg'].str.strip('"')
    ensg_map['symbol'] = ensg_map['symbol'].str.strip('"')
    symbol_to_ensg = dict(zip(ensg_map['symbol'], ensg_map['ensg']))
    ensg_to_symbol = {v: k for k, v in symbol_to_ensg.items()}

    # Cell line mapping
    cell_annot = pd.read_csv(f'{GDSC_DATA}/cell_annot.csv', index_col=0)
    cel_to_cell_line = cell_annot['Factor.Value.cell_line.'].to_dict()

    # Map CEL to cell line for expression
    cel_names = expr_raw.columns.tolist()
    cell_line_map = {}
    for cel in cel_names:
        cell_line_map[cel] = cel_to_cell_line.get(cel, cel)

    # NetITH
    netith = pd.read_csv('results/gdsc/gdsc_netith_cell_lines.csv', index_col=0)

    # IC50
    ic50_raw = pd.read_csv(GDSC_IC50)
    ic50_mat = ic50_raw.pivot_table(
        values='LN_IC50', index='CELL_LINE_NAME',
        columns='DRUG_NAME', aggfunc='mean')

    return (mut_raw, cnv_raw, expr_raw, cell_line_map, cel_names,
            netith, ic50_mat, symbol_to_ensg, ensg_to_symbol)


def analyze_mutation_interaction(mut_raw, expr_raw, cell_line_map, cel_names,
                                  netith, ic50_mat, symbol_to_ensg):
    """Z-2b: Real binary mutation × NetITH interaction (improved over ZB)."""
    print("\n[Z-2b] Mutation × NetITH interaction (real binary calls)...")

    # Match mutation cell lines to expression cell lines
    mut_cells = set(mut_raw.columns)
    expr_cell_lines = set(cell_line_map.values())

    # Focus genes
    driver_genes = ['TP53', 'KRAS', 'BRAF', 'PIK3CA', 'PTEN', 'EGFR',
                    'NRAS', 'RB1', 'APC', 'CTNNB1', 'NF1', 'ARID1A',
                    'CDKN2A', 'SMAD4', 'FBXW7', 'KEAP1', 'STK11', 'ATM',
                    'BRCA2', 'IDH1', 'KMT2D', 'NOTCH1', 'ERBB2', 'MYC']

    results = []
    ic50_cells = set(ic50_mat.index)

    # For each driver gene × drug: compare the NetITH–IC50 correlation in mutated vs WT cell lines
    for gene in driver_genes:
        if gene not in mut_raw.index:
            continue

        gene_muts = mut_raw.loc[gene]
        mut_positive = set(gene_muts[gene_muts > 0].index)

        for drug in ic50_mat.columns[:80]:
            drug_vals = ic50_mat[drug].dropna()
            drug_cells = set(drug_vals.index)

            # Find cells with both mutation AND NetITH AND drug data
            valid_cells = []
            for cl in drug_cells & ic50_cells:
                if cl in netith.index:
                    is_mut = cl in mut_positive
                    valid_cells.append({
                        'cell_line': cl,
                        'netith': netith.loc[cl, 'NetITH'],
                        'ic50': drug_vals[cl],
                        'mutated': int(is_mut),
                    })

            if len(valid_cells) < 30:
                continue

            df = pd.DataFrame(valid_cells)
            mut = df[df['mutated'] == 1]
            wt = df[df['mutated'] == 0]
            if len(mut) < 5 or len(wt) < 5:
                continue

            # Spearman NetITH–IC50 correlation within each mutation group
            rho_mut, p_mut = spearmanr(mut['netith'], mut['ic50'])
            rho_wt, p_wt = spearmanr(wt['netith'], wt['ic50'])

            # Fisher z-transformation interaction test (null: equal correlations in mutated vs WT)
            # Interaction z-test
            z_mut = np.arctanh(np.clip(rho_mut, -0.99, 0.99))
            z_wt = np.arctanh(np.clip(rho_wt, -0.99, 0.99))
            se = np.sqrt(1/(len(mut)-3) + 1/(len(wt)-3))
            z_diff = np.abs(z_mut - z_wt) / (se + 1e-10)
            p_interaction = 2 * (1 - 0.5 * (1 + np.tanh(z_diff)))

            results.append({
                'gene': gene, 'drug': drug,
                'n_mut': len(mut), 'n_wt': len(wt),
                'netith_mut': mut['netith'].mean(),
                'netith_wt': wt['netith'].mean(),
                'rho_mut': rho_mut, 'rho_wt': rho_wt,
                'rho_delta': rho_mut - rho_wt,
                'p_interaction': p_interaction,
            })

    # Write mutation × NetITH–IC50 interaction results
    res_df = pd.DataFrame(results)
    res_df.to_csv(f'{DEPMAP_OUTPUT}/crispr_dependency_netith.csv', index=False)
    print(f"  Mutation × Drug interactions: {len(res_df)}")

    # Summary
    gene_summary = res_df.groupby('gene').agg(
        n_drugs=('drug', 'nunique'),
        mean_rho_delta=('rho_delta', 'mean'),
        n_sig=('p_interaction', lambda x: (x < 0.05).sum()),
        mean_netith_delta=('netith_mut', lambda x: x.mean()),
    ).sort_values('mean_rho_delta', key=lambda x: abs(x), ascending=False)

    print("  Top genomic modifiers of NetITH→drug response:")
    for gene, row in gene_summary.head(10).iterrows():
        print(f"    {gene:<12} Δρ={row['mean_rho_delta']:+.3f} "
              f"sig={int(row['n_sig'])}/{int(row['n_drugs'])}")

    return res_df, gene_summary


def analyze_cnv_interaction(cnv_raw, netith, ic50_mat, expr_raw,
                             symbol_to_ensg, cell_line_map, cel_names):
    """Z-2c: CNV × NetITH interaction."""
    print("\n[Z-2c] CNV × NetITH interaction...")

    focus_genes = ['EGFR', 'ERBB2', 'MYC', 'CCND1', 'CDK4', 'MDM2',
                   'CDKN2A', 'PTEN', 'RB1', 'TP53']

    results = []
    for gene in focus_genes:
        ensg = symbol_to_ensg.get(gene)
        if not ensg or ensg not in cnv_raw.index:
            continue

        gene_cnv = cnv_raw.loc[ensg]
        cnv_cells = set(gene_cnv.index)

        # For each focus gene × drug: split by CNV amplification (CNV>1.5), deletion (<0.5) or normal
        for drug in ic50_mat.columns[:60]:
            drug_vals = ic50_mat[drug].dropna()
            drug_cells = set(drug_vals.index)

            valid_cells = []
            for cl in drug_cells:
                if cl in netith.index and cl in cnv_cells:
                    cnv_val = gene_cnv[cl]
                    is_amp = 1 if cnv_val > 1.5 else 0
                    is_del = 1 if cnv_val < 0.5 else 0
                    valid_cells.append({
                        'cell_line': cl,
                        'netith': netith.loc[cl, 'NetITH'],
                        'ic50': drug_vals[cl],
                        'cnv': cnv_val,
                        'amplified': is_amp,
                        'deleted': is_del,
                    })

            if len(valid_cells) < 30:
                continue

            df = pd.DataFrame(valid_cells)

            # Test amplification
            amp = df[df['amplified'] == 1]
            normal = df[(df['amplified'] == 0) & (df['deleted'] == 0)]
            if len(amp) >= 5 and len(normal) >= 5:
                rho_a, _ = spearmanr(amp['netith'], amp['ic50'])
                rho_n, _ = spearmanr(normal['netith'], normal['ic50'])
                results.append({
                    'gene': gene, 'drug': drug, 'type': 'amplification',
                    'n_alt': len(amp), 'n_ref': len(normal),
                    'rho_alt': rho_a, 'rho_ref': rho_n,
                    'rho_delta': rho_a - rho_n,
                })

    # Write CNV × NetITH–IC50 interaction results
    cnv_df = pd.DataFrame(results)
    if len(cnv_df) > 0:
        cnv_df.to_csv(f'{DEPMAP_OUTPUT}/cnv_netith_interaction.csv', index=False)
        print(f"  CNV interactions: {len(cnv_df)}")

        cnv_summary = cnv_df.groupby('gene')['rho_delta'].mean().sort_values(key=abs)
        for gene, d in cnv_summary.head(5).items():
            print(f"    {gene:<12} CNV amp Δρ={d:+.3f}")

    return cnv_df


def analyze_essentiality_proxy(expr_raw, netith, ic50_mat, symbol_to_ensg,
                                cell_line_map, cel_names):
    """Z-2d: Essential gene low-expression × NetITH."""
    print("\n[Z-2d] Essential gene expression proxy × NetITH...")

    results = []
    for gene in ESSENTIAL_GENES:
        ensg = symbol_to_ensg.get(gene)
        if not ensg or ensg not in expr_raw.index:
            continue

        # Dependency proxy: low expression (<30th percentile) of a known essential gene
        gene_expr = expr_raw.loc[ensg]
        gene_low = gene_expr.quantile(0.3)

        for drug in ic50_mat.columns[:60]:
            drug_vals = ic50_mat[drug].dropna()
            drug_cells = set(drug_vals.index)

            valid_cells = []
            for cel in cel_names:
                cl = cell_line_map.get(cel, cel)
                if cl in drug_cells and cl in netith.index:
                    valid_cells.append({
                        'cell_line': cl,
                        'netith': netith.loc[cl, 'NetITH'],
                        'ic50': drug_vals[cl],
                        'expr': gene_expr[cel],
                        'low': 1 if gene_expr[cel] < gene_low else 0,
                    })

            if len(valid_cells) < 30:
                continue

            df = pd.DataFrame(valid_cells)
            low = df[df['low'] == 1]
            normal = df[df['low'] == 0]
            if len(low) < 5 or len(normal) < 5:
                continue

            rho_l, _ = spearmanr(low['netith'], low['ic50'])
            rho_n, _ = spearmanr(normal['netith'], normal['ic50'])
            results.append({
                'gene': gene, 'drug': drug,
                'n_low': len(low), 'n_normal': len(normal),
                'rho_low': rho_l, 'rho_normal': rho_n,
                'rho_delta': rho_l - rho_n,
            })

    # Write essentiality-proxy × NetITH–IC50 interaction results
    ess_df = pd.DataFrame(results)
    if len(ess_df) > 0:
        ess_df.to_csv(f'{DEPMAP_OUTPUT}/essentiality_netith_interaction.csv', index=False)
        print(f"  Essentiality interactions: {len(ess_df)}")

        ess_summary = ess_df.groupby('gene')['rho_delta'].mean().sort_values(key=abs)
        for gene, d in ess_summary.head(5).items():
            print(f"    {gene:<12} LowExpr Δρ={d:+.3f}")

    return ess_df


def plot_genomic_interaction(mut_summary, output_dir):
    """Heatmap of genomic interaction effects."""
    if len(mut_summary) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Genomic Vulnerabilities × NetITH Interaction',
                 fontsize=13, fontweight='bold', y=0.99)

    # Panel A: Mutation interaction waterfall
    ax = axes[0]
    top = mut_summary.head(15).sort_values('mean_rho_delta')
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top['mean_rho_delta']]
    ax.barh(range(len(top)), top['mean_rho_delta'].values, color=colors, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.set_xlabel('Δρ (Mut − WT) for NetITH→IC50', fontsize=11)
    ax.set_title('A: Mutation Effect on NetITH-Drug Response',
                 fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.grid(axis='x', alpha=0.2)

    # Panel B: NetITH shift by mutation
    ax = axes[1]
    if 'mean_netith_delta' in mut_summary.columns:
        x = mut_summary.head(15)['mean_netith_delta'].values
        y = mut_summary.head(15)['mean_rho_delta'].values
        labels = mut_summary.head(15).index.tolist()
        ax.scatter(x, y, s=100, c=['#d62728' if v > 0 else '#1f77b4' for v in y],
                   alpha=0.6, edgecolors='black')
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), fontsize=8, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('ΔNetITH (Mut − WT)', fontsize=11)
        ax.set_ylabel('Δρ (Mut − WT)', fontsize=11)
        ax.set_title('B: NetITH Shift vs Drug Response Modulation',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/figures/genomic_netith_interaction.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Saved genomic_netith_interaction.png")


def main():
    print("=" * 60)
    print("Z-2: Genomic Dependency × NetITH Interaction")
    print("=" * 60)

    (mut_raw, cnv_raw, expr_raw, cell_line_map, cel_names,
     netith, ic50_mat, symbol_to_ensg, ensg_to_symbol) = load_data()

    res_df, mut_summary = analyze_mutation_interaction(
        mut_raw, expr_raw, cell_line_map, cel_names, netith, ic50_mat, symbol_to_ensg)

    cnv_df = analyze_cnv_interaction(
        cnv_raw, netith, ic50_mat, expr_raw, symbol_to_ensg, cell_line_map, cel_names)

    ess_df = analyze_essentiality_proxy(
        expr_raw, netith, ic50_mat, symbol_to_ensg, cell_line_map, cel_names)

    plot_genomic_interaction(mut_summary, DEPMAP_OUTPUT)

    print(f"\nDone! Outputs in results/depmap/")


if __name__ == '__main__':
    main()
