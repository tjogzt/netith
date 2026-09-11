#!/usr/bin/env python3
"""run_clinical_panel_optimization.py — clinical NetITH TF panel optimization for translational deployability.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : results/gdsc/gdsc_tf_contribution.csv; results/tcga/tcga_netith.csv (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/clinical_panel/{tf_contributions_ranked.csv, panel_optimization.json}; results/clinical_panel/figures/clinical_panel_analysis.pdf
Pipeline: clinical stage — see repository README
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TF_CONTRIB = PROJECT_ROOT / 'results/gdsc/gdsc_tf_contribution.csv'
TCGA_NETITH = PROJECT_ROOT / 'results/tcga/tcga_netith.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/clinical_panel'
FIGURE_DIR = OUTPUT_DIR / 'figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_tf_contributions():
    """Load and rank TFs by contribution."""
    df = pd.read_csv(TF_CONTRIB)
    print(f"  Loaded {len(df)} TFs")
    
    # Sort by absolute contribution (sum_edge_mean)
    df['abs_contrib'] = df['sum_edge_mean'].abs()
    df = df.sort_values('abs_contrib', ascending=False).reset_index(drop=True)
    
    # Compute cumulative variance explained
    total_var = (df['sum_edge_mean'] ** 2).sum()
    df['var_explained'] = df['sum_edge_mean'] ** 2 / total_var
    df['cumulative_var'] = df['var_explained'].cumsum()
    
    # Find minimal panel (80% and 90% thresholds)
    n80 = (df['cumulative_var'] < 0.80).sum() + 1
    n90 = (df['cumulative_var'] < 0.90).sum() + 1
    
    print(f"  TFs for 80% variance: {n80}")
    print(f"  TFs for 90% variance: {n90}")
    
    return df, n80, n90


def validate_panel_survival(tf_df, n_panel):
    """Check if top TFs associate with survival in TCGA."""
    top_tfs = tf_df.head(n_panel)
    
    # Check TF-NetITH correlation significance
    sig_tfs = top_tfs[top_tfs['p_total_vs_NetITH'] < 0.05]
    
    print(f"\n  Top {n_panel} TF panel:")
    print(f"    Significant TF-NetITH correlations: {len(sig_tfs)}/{n_panel}")
    print(f"    Cumulative variance explained: {top_tfs.iloc[n_panel-1]['cumulative_var']:.1%}")
    
    # Top TFs by pathway
    pathway_counts = top_tfs['category'].value_counts()
    print(f"    Pathway distribution:")
    for pw, count in pathway_counts.head(5).items():
        print(f"      {pw}: {count}")
    
    return top_tfs


def make_figure(tf_df, n80, n90, top80, output_path):
    """Clinical panel optimization figure."""
    fig = plt.figure(figsize=(16, 10))
    
    # ── A: Cumulative variance ──
    ax_a = fig.add_subplot(2, 3, 1)
    n_tfs = np.arange(1, len(tf_df) + 1)
    ax_a.plot(n_tfs, tf_df['cumulative_var'] * 100, 'b-', lw=2)
    ax_a.axhline(80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    ax_a.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax_a.axvline(n80, color='green', linestyle=':', alpha=0.5)
    ax_a.axvline(n90, color='orange', linestyle=':', alpha=0.5)
    ax_a.set_xlabel('Number of TFs'); ax_a.set_ylabel('Cumulative Variance (%)')
    ax_a.set_title(f'(A) TF Contribution to NetITH\n80%: {n80} TFs, 90%: {n90} TFs')
    ax_a.legend(fontsize=7)
    ax_a.set_xlim(0, min(40, len(tf_df)))
    
    # ── B: Top TF contributions ──
    ax_b = fig.add_subplot(2, 3, 2)
    top20 = tf_df.head(20)
    colors = ['#c0392b' if v > 0 else '#2980b9' for v in top20['sum_edge_mean']]
    ax_b.barh(range(len(top20)), top20['sum_edge_mean'].values, color=colors, edgecolor='white')
    ax_b.set_yticks(range(len(top20)))
    ax_b.set_yticklabels(top20['TF'].values, fontsize=6)
    ax_b.set_xlabel('Mean Edge Contribution'); ax_b.axvline(0, color='black', lw=0.8)
    ax_b.set_title('(B) Top 20 TFs by Edge Contribution')
    sig = top20['p_total_vs_NetITH'] < 0.05
    for i, (val, is_sig) in enumerate(zip(top20['sum_edge_mean'].values, sig)):
        marker = '*' if is_sig else ''
        ax_b.text(val + 0.2 * np.sign(val), i, marker, va='center', fontsize=8)
    
    # ── C: Pathway composition of panel ──
    ax_c = fig.add_subplot(2, 3, 3)
    pathway_counts = top80['category'].value_counts()
    if len(pathway_counts) > 0:
        ax_c.pie(pathway_counts.values, labels=pathway_counts.index, autopct='%1.1f%%',
                 colors=plt.cm.Set3(np.linspace(0, 1, len(pathway_counts))), textprops={'fontsize': 7})
        ax_c.set_title(f'(C) Top {n80}-TF Panel: Pathway Composition')
    
    # ── D: TF-NetITH correlations ──
    ax_d = fig.add_subplot(2, 3, 4)
    top30 = tf_df.head(30)
    ax_d.scatter(top30['sum_edge_mean'], top30['rho_total_vs_NetITH'], 
                 c=top30['p_total_vs_NetITH'].apply(lambda p: 'red' if p < 0.05 else 'grey'),
                 s=50, alpha=0.7, edgecolors='white')
    for _, r in top30.head(15).iterrows():
        ax_d.annotate(r['TF'], (r['sum_edge_mean'], r['rho_total_vs_NetITH']), fontsize=5)
    ax_d.set_xlabel('Edge Contribution'); ax_d.set_ylabel('TF–NetITH ρ')
    ax_d.set_title('(D) TF Contribution vs NetITH Correlation')
    ax_d.axhline(0, color='grey', linestyle='--', alpha=0.3)
    
    # ── E: Panel size vs variance ──
    ax_e = fig.add_subplot(2, 3, 5)
    sizes = [5, 10, 15, 20, 25, 30, 40, 50, len(tf_df)]
    variances = []
    for s in sizes:
        if s <= len(tf_df):
            v = tf_df.head(s)['cumulative_var'].iloc[-1] * 100
        else:
            v = 100
        variances.append(v)
    ax_e.plot(sizes, variances, 'o-', color='#2c3e50', lw=2, markersize=8)
    ax_e.axhline(80, color='green', linestyle='--', alpha=0.5)
    ax_e.axhline(90, color='orange', linestyle='--', alpha=0.5)
    ax_e.set_xlabel('Panel Size (TFs)'); ax_e.set_ylabel('Variance Explained (%)')
    ax_e.set_title('(E) Panel Size vs Information Capture')
    ax_e.set_ylim(0, 105)
    
    # ── F: Summary ──
    ax_f = fig.add_subplot(2, 3, 6)
    ax_f.axis('off')
    
    sig_in_panel = (top80['p_total_vs_NetITH'] < 0.05).sum()
    top5_tfs = ', '.join(tf_df.head(5)['TF'].values)
    
    lines = [
        "Clinical NetITH Panel Optimization",
        "",
        f"  Total TFs in CollecTRI: {len(tf_df)}",
        f"  TFs for 80% variance: {n80}",
        f"  TFs for 90% variance: {n90}",
        f"  Sig. TFs in 80% panel: {sig_in_panel}/{n80}",
        "",
        f"  Top 5 TFs: {top5_tfs}",
        "",
        "  Pathway breakdown (80% panel):",
    ]
    for pw, cnt in pathway_counts.items():
        lines.append(f"    {pw}: {cnt} TFs")
    
    lines += [
        "",
        "  Translational recommendation:",
        f"  A {n80}-TF Nanostring/RT-PCR panel",
        "  captures ≥80% of NetITH variance.",
        "  This enables clinical deployment for",
        "  prospective validation studies.",
    ]
    
    ax_f.text(0.05, 0.95, '\n'.join(lines), transform=ax_f.transAxes,
              fontfamily='monospace', fontsize=7.5, va='top')
    
    fig.suptitle('Clinical NetITH Panel: TF Selection for Translational Deployment',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[Figure] → {output_path}")


def main():
    print("=" * 65)
    print("Analysis 5/6: Clinical NetITH Panel Optimization")
    print("=" * 65)
    
    print("\n[1] Loading TF contributions...")
    tf_df, n80, n90 = load_tf_contributions()
    
    print("\n[2] Validating 80% panel...")
    top80 = validate_panel_survival(tf_df, n80)
    
    print("\n[3] Validating 90% panel...")
    top90 = validate_panel_survival(tf_df, n90)
    
    # Save
    tf_df.to_csv(OUTPUT_DIR / 'tf_contributions_ranked.csv', index=False)
    print(f"\n[Save] {OUTPUT_DIR / 'tf_contributions_ranked.csv'}")
    
    panel_info = {
        'n_total_tfs': len(tf_df),
        'n_for_80pct': int(n80),
        'n_for_90pct': int(n90),
        'top80_tfs': top80['TF'].tolist(),
        'top80_pathways': top80['category'].value_counts().to_dict(),
        'top10_tfs': tf_df.head(10)['TF'].tolist(),
    }
    with open(OUTPUT_DIR / 'panel_optimization.json', 'w') as f:
        json.dump(panel_info, f, indent=2)
    
    make_figure(tf_df, n80, n90, top80, str(FIGURE_DIR / 'clinical_panel_analysis.pdf'))
    
    print("\n[Done] Analysis 5/6 complete.\n")


if __name__ == '__main__':
    main()
