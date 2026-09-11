#!/usr/bin/env python3
"""
run_module_conservation.py — Test conservation of GDSC TF modules across TCGA cancer types via per-cancer regression.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - data/gdsc/ensg_symbol_map.csv: ENSG -> symbol mapping (reverse-mapped for the 41 module TFs)
    - <DATA_ROOT>/GraphHMM/tcga_pancan/tcga_pancan_toil_rsem_tpm.gz: TCGA pancan expression (streamed)
    - results/tcga/tcga_tri_modal_merged.csv: tri-modal NetITH per TCGA sample
    - results/depmap/tf_lasso_coefficients.csv: GDSC LASSO TF coefficients (reference)
Outputs :
    - results/tcga/module_conservation_summary.csv, module_conservation_tf_coefs.csv
    - results/tcga/figures/module_conservation_{heatmap,r2,gdsc_vs_tcga}.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""

from pathlib import Path
import os

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
import warnings
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
warnings.filterwarnings('ignore')


# ============================================================
# 1. TF MODULE DEFINITIONS (from GDSC §3.14 hierarchical clustering)
# ============================================================

# Based on LASSO coefficients + hierarchical clustering of TF-TF correlation
# Module 1: AP-1 / Stress Response (JUN-dominated, negative coefs)
MODULE_1_AP1 = ['JUN', 'ATF4', 'FOS', 'STAT1', 'FOXP3', 'PAX3',
                'SMAD7', 'HEYL', 'NFIA', 'HES1', 'ZBTB20', 'DNMT3B',
                'CREB1', 'IRF1', 'STAT3', 'CEBPB']

# Module 2: Epigenetic / Developmental
MODULE_2_EPI = ['PAX5', 'RARG', 'SFPQ', 'TEAD2', 'ESR2', 'ZNF202',
                'ZNF263', 'MEF2A', 'SOX2', 'POU5F1', 'NANOG',
                'HDAC2', 'EZH2', 'SUZ12', 'CTCF']

# Module 3: Differentiation / Lineage
MODULE_3_DIFF = ['TRIM22', 'SPI1', 'GATA3', 'FOXA1', 'MYOD1',
                 'RUNX1', 'TBX21', 'PPARG', 'CEBPA', 'MITF']

# All TFs
ALL_TFS = MODULE_1_AP1 + MODULE_2_EPI + MODULE_3_DIFF
# Remove duplicates
ALL_TFS = list(dict.fromkeys(ALL_TFS))
MODULE_MAP = {}
for tf in MODULE_1_AP1:
    MODULE_MAP[tf] = 'AP-1/Stress'
for tf in MODULE_2_EPI:
    MODULE_MAP[tf] = 'Epigenetic/Dev'
for tf in MODULE_3_DIFF:
    MODULE_MAP[tf] = 'Differentiation'


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_tf_ensg_map(ensg_map_path: str) -> dict:
    """Load ENSG → symbol mapping, return symbol → ENSG (base, no version)."""
    # Read ENSG->symbol map and keep only the module TFs (as symbol -> ENSG base)
    df = pd.read_csv(ensg_map_path)
    # Strip quotes
    df['ensg'] = df['ensg'].str.strip('"')
    df['symbol'] = df['symbol'].str.strip('"')
    # Build reverse map: symbol → ENSG base
    symbol_to_ensg = {}
    for _, row in df.iterrows():
        sym = row['symbol']
        ensg = row['ensg']
        if sym in ALL_TFS:
            if sym not in symbol_to_ensg:
                symbol_to_ensg[sym] = ensg
    return symbol_to_ensg


def extract_tf_expression(pancan_path: str, symbol_to_ensg: dict,
                           n_samples: int = None) -> pd.DataFrame:
    """Extract TF expression rows from TCGA pancan matrix (streaming)."""
    target_ensgs = set(symbol_to_ensg.values())
    print(f"  Target ENSGs: {len(target_ensgs)}")
    print(f"  TFs to extract: {len(ALL_TFS)}")

    rows = []
    sample_ids = None

    # Stream the gzipped pancan matrix line by line (never loads the full matrix into memory)
    with gzip.open(pancan_path, 'rt') as f:
        # Read header
        header_line = f.readline().strip()
        sample_ids = header_line.split('\t')[1:]  # Skip 'sample' column
        print(f"  Total samples: {len(sample_ids)}")

        # Read gene rows
        for line in f:
            fields = line.strip().split('\t')
            ensg_full = fields[0]
            # Strip version: ENSG00000259041.1 → ENSG00000259041
            ensg_base = ensg_full.split('.')[0]

            if ensg_base in target_ensgs:
                values = []
                for v in fields[1:]:
                    try:
                        values.append(float(v))
                    except (ValueError, IndexError):
                        values.append(np.nan)
                rows.append({'ensg': ensg_full, 'ensg_base': ensg_base,
                             'values': values})
                if len(rows) % 10 == 0:
                    print(f"    Found {len(rows)}/{len(target_ensgs)} TFs...")

    print(f"  Extracted {len(rows)} TF rows from {len(sample_ids)} samples")

    # Reassemble the extracted rows into a samples x TFs DataFrame (symbols)
    # Build DataFrame: samples × TFs
    # Map ENSG base → symbol
    ensg_to_symbol = {v: k for k, v in symbol_to_ensg.items()}

    data = {}
    for row in rows:
        symbol = ensg_to_symbol.get(row['ensg_base'], row['ensg_base'])
        data[symbol] = row['values']

    expr_df = pd.DataFrame(data, index=sample_ids)
    return expr_df


# ============================================================
# 3. PER-CANCER REGRESSION
# ============================================================

def run_per_cancer_regression(tri_df: pd.DataFrame, expr_df: pd.DataFrame,
                               min_samples: int = 30) -> dict:
    """For each TCGA cancer, regress NetITH on TF expression."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    results = []
    tf_coef_matrix = {}  # cancer → {tf: coef}

    # For each cancer type, regress NetITH on module-TF expression (OLS) and record R2
    cancers = sorted(tri_df['cancer'].unique())
    print(f"\n  Processing {len(cancers)} cancers...")

    for cancer in cancers:
        # Get tri-modal samples for this cancer
        tri_c = tri_df[tri_df['cancer'] == cancer].copy()
        tri_c['patient'] = tri_c['sample'].apply(
            lambda s: '-'.join(s.split('.')[:3]).upper()
        )

        # Match with expression data
        # TCGA sample IDs in pancan: TCGA-XX-XXXX-XX
        # Our patient IDs: TCGA-XX-XXXX
        matched = []
        for _, row in tri_c.iterrows():
            pid = row['patient']
            # Find exact match or prefix match in expression
            if pid in expr_df.index:
                matched.append((pid, row['netith_bulk']))
            else:
                # Try prefix match
                matches = [s for s in expr_df.index if s.startswith(pid)]
                if len(matches) == 1:
                    matched.append((matches[0], row['netith_bulk']))
                elif len(matches) > 1:
                    # Use primary tumor (01A or 01)
                    primary = [s for s in matches if '-01' in s]
                    if primary:
                        matched.append((primary[0], row['netith_bulk']))

        if len(matched) < min_samples:
            continue

        sample_ids, netith_vals = zip(*matched)
        X = expr_df.loc[list(sample_ids)]

        # Remove columns with too many NaN
        valid_tfs = [c for c in X.columns if X[c].notna().sum() >= len(X) * 0.5]
        if len(valid_tfs) < 5:
            continue

        X = X[valid_tfs].fillna(X[valid_tfs].median())
        y = np.array(netith_vals)

        n = len(y)

        # Full-model R2 plus 5-fold cross-validated R2 (honest estimate of predictive power)
        # ---- Full model (all TFs) ----
        lr = LinearRegression()
        lr.fit(X, y)
        r2_full = lr.score(X, y)
        # Cross-validated R²
        try:
            cv_scores = cross_val_score(lr, X, y, cv=min(5, n//10), scoring='r2')
            r2_cv = cv_scores.mean()
            r2_cv_std = cv_scores.std()
        except:
            r2_cv = np.nan
            r2_cv_std = np.nan

        # Per-module R2: how much NetITH variance each module alone explains
        # ---- Per-module R² ----
        module_r2 = {}
        module_coefs = {}
        for mod_name, mod_tfs in [
            ('AP-1/Stress', MODULE_1_AP1),
            ('Epigenetic/Dev', MODULE_2_EPI),
            ('Differentiation', MODULE_3_DIFF)
        ]:
            mod_tfs_in_data = [t for t in mod_tfs if t in X.columns]
            if len(mod_tfs_in_data) >= 2:
                X_mod = X[mod_tfs_in_data]
                lr_mod = LinearRegression()
                lr_mod.fit(X_mod, y)
                module_r2[mod_name] = lr_mod.score(X_mod, y)
                for tf, coef in zip(mod_tfs_in_data, lr_mod.coef_):
                    module_coefs[tf] = coef
            else:
                module_r2[mod_name] = np.nan

        # JUN-only model: baseline contribution of the AP-1 hub TF alone
        # ---- JUN-only R² ----
        jun_r2 = np.nan
        if 'JUN' in X.columns:
            lr_jun = LinearRegression()
            lr_jun.fit(X[['JUN']], y)
            jun_r2 = lr_jun.score(X[['JUN']], y)

        # Store coefficients
        full_coefs = dict(zip(valid_tfs, lr.coef_))
        tf_coef_matrix[cancer] = full_coefs

        results.append({
            'cancer': cancer,
            'n_samples': n,
            'n_tfs': len(valid_tfs),
            'r2_full': r2_full,
            'r2_cv': r2_cv,
            'r2_cv_std': r2_cv_std,
            'r2_jun': jun_r2,
            'r2_ap1_stress': module_r2.get('AP-1/Stress', np.nan),
            'r2_epigenetic_dev': module_r2.get('Epigenetic/Dev', np.nan),
            'r2_differentiation': module_r2.get('Differentiation', np.nan),
        })

        if len(results) % 5 == 0:
            print(f"    Processed {len(results)} cancers...")

    print(f"  Done: {len(results)} cancers with >= {min_samples} samples")

    return results, tf_coef_matrix


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_heatmap(tf_coef_matrix: dict, output_dir: str):
    """Heatmap: TF (row) × Cancer (column) of regression coefficients."""
    # Collect all TFs and cancers
    all_tfs_set = set()
    for coefs in tf_coef_matrix.values():
        all_tfs_set.update(coefs.keys())
    all_tfs = sorted(all_tfs_set,
                     key=lambda t: MODULE_MAP.get(t, 'Z'),
                     reverse=True)
    cancers = sorted(tf_coef_matrix.keys())

    if len(cancers) < 2 or len(all_tfs) < 3:
        print("  Not enough data for heatmap")
        return

    # TF x cancer coefficient matrix; per-cancer Z-scoring makes rows comparable for display
    # Build matrix
    n_tfs = len(all_tfs)
    n_cancers = len(cancers)
    matrix = np.full((n_tfs, n_cancers), np.nan)
    for j, cancer in enumerate(cancers):
        coefs = tf_coef_matrix[cancer]
        for i, tf in enumerate(all_tfs):
            matrix[i, j] = coefs.get(tf, np.nan)

    # Normalize per-cancer for visualization
    from sklearn.preprocessing import StandardScaler
    matrix_norm = matrix.copy()
    for j in range(n_cancers):
        col = matrix_norm[:, j]
        mask = ~np.isnan(col)
        if mask.sum() > 1:
            matrix_norm[mask, j] = (col[mask] - np.nanmean(col[mask])) / np.nanstd(col[mask])

    fig, ax = plt.subplots(figsize=(max(8, n_cancers * 1.2), max(10, n_tfs * 0.35)))

    # Color by module membership
    module_colors = {'AP-1/Stress': '#d62728', 'Epigenetic/Dev': '#1f77b4',
                     'Differentiation': '#2ca02c'}

    im = ax.imshow(matrix_norm, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2, interpolation='nearest')

    ax.set_xticks(range(n_cancers))
    ax.set_xticklabels(cancers, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_tfs))
    ax.set_yticklabels(all_tfs, fontsize=8)

    # Color y-tick labels by module
    for i, tf in enumerate(all_tfs):
        mod = MODULE_MAP.get(tf, 'Unknown')
        color = module_colors.get(mod, 'black')
        ax.get_yticklabels()[i].set_color(color)

    # Add module legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=m)
                       for m, c in module_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.15, 1), fontsize=9, title='Module')

    plt.colorbar(im, ax=ax, label='Z-scored Coefficient', shrink=0.8)
    ax.set_title('Cross-Cancer TF Module Conservation\n'
                 '(Linear Regression Coefficients for NetITH)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Transcription Factor', fontsize=12)

    plt.tight_layout()
    out = f'{output_dir}/figures/module_conservation_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


def plot_r2_summary(results: list, output_dir: str):
    """Bar chart: per-cancer R² by module."""
    df = pd.DataFrame(results)
    if len(df) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df) * 0.4)))

    # Panel A: Full R² vs JUN-only R² vs CV R²
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['r2_full'], w, color='#4472C4', alpha=0.8,
           label='Full (All TFs)')
    ax.bar(x, df['r2_jun'], w, color='#d62728', alpha=0.8,
           label='JUN only')
    ax.bar(x + w, df['r2_cv'], w, color='#2ca02c', alpha=0.8,
           label='CV (5-fold)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['cancer'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('TF Predictive Power for NetITH by Cancer',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Per-module R² decomposition
    ax = axes[1]
    module_cols = ['r2_ap1_stress', 'r2_epigenetic_dev', 'r2_differentiation']
    module_labels = ['AP-1/Stress', 'Epigenetic/Dev', 'Differentiation']
    module_colors = ['#d62728', '#1f77b4', '#2ca02c']
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, label, color in zip(module_cols, module_labels, module_colors):
        vals = df[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df['cancer'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R² (Module Contribution)', fontsize=12)
    ax.set_title('Module-Level Decomposition of NetITH Variance',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = f'{output_dir}/figures/module_conservation_r2.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


def plot_gdsc_vs_tcga_correlation(tf_coef_matrix: dict, output_dir: str):
    """Compare GDSC coefficients with per-cancer TCGA coefficients."""
    # GDSC full linear coefficients from tf_combinatorial
    # Using LASSO coefs as the reference
    # Reference: GDSC LASSO coefficients; compared with per-cancer TCGA coefficients below
    gdsc_lasso = pd.read_csv('results/depmap/tf_lasso_coefficients.csv')
    gdsc_coefs = dict(zip(gdsc_lasso['TF'], gdsc_lasso['lasso_coef']))

    cancers = sorted(tf_coef_matrix.keys())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, cancer in enumerate(cancers[:8]):
        ax = axes[idx]
        tcga_coefs = tf_coef_matrix[cancer]

        common_tfs = set(gdsc_coefs.keys()) & set(tcga_coefs.keys())
        if len(common_tfs) < 5:
            ax.text(0.5, 0.5, 'Insufficient TFs', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(cancer, fontsize=11)
            continue

        xs = [gdsc_coefs[t] for t in common_tfs]
        ys = [tcga_coefs[t] for t in common_tfs]

        # Pearson correlation of GDSC vs TCGA coefficients (null: no linear conservation)
        r, p = stats.pearsonr(xs, ys)
        ax.scatter(xs, ys, c='#4472C4', alpha=0.6, edgecolors='black',
                   linewidth=0.5)
        # Color by module
        for i, tf in enumerate(common_tfs):
            mod = MODULE_MAP.get(tf, 'Unknown')
            if mod == 'AP-1/Stress':
                ax.scatter([xs[i]], [ys[i]], c='#d62728', alpha=0.8,
                          edgecolors='black', linewidth=0.5)
            elif mod == 'Epigenetic/Dev':
                ax.scatter([xs[i]], [ys[i]], c='#1f77b4', alpha=0.8,
                          edgecolors='black', linewidth=0.5)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

        # Trend line
        if len(xs) > 3:
            z = np.polyfit(xs, ys, 1)
            pfit = np.poly1d(z)
            xline = np.linspace(min(xs), max(xs), 100)
            ax.plot(xline, pfit(xline), 'k-', linewidth=1.5)

        ax.set_xlabel('GDSC Coef', fontsize=9)
        ax.set_ylabel('TCGA Coef', fontsize=9)
        ax.set_title(f'{cancer} (ρ={r:.3f}, p={p:.3f})', fontsize=10)
        ax.grid(alpha=0.3)

    for idx in range(len(cancers), 8):
        axes[idx].set_visible(False)

    plt.suptitle('GDSC vs TCGA: TF Coefficient Conservation',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = f'{output_dir}/figures/module_conservation_gdsc_vs_tcga.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved {out}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    import os

    output_dir = 'results/tcga'
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Paths
    tri_modal_path = 'results/tcga/tcga_tri_modal_merged.csv'
    ensg_map_path = DATA_ROOT / "gdsc" / "ensg_symbol_map.csv"
    pancan_path = f'{DATA_ROOT}/GraphHMM/tcga_pancan/tcga_pancan_toil_rsem_tpm.gz'

    print("=" * 60)
    print("N: Cross-Cancer GRN Module Conservation")
    print("=" * 60)

    # 1. Load ENSG → Symbol map
    print("\n[1] Loading ENSG → Symbol mapping...")
    symbol_to_ensg = load_tf_ensg_map(ensg_map_path)
    print(f"  Mapped {len(symbol_to_ensg)}/{len(ALL_TFS)} TFs to ENSG IDs")
    missing = [t for t in ALL_TFS if t not in symbol_to_ensg]
    if missing:
        print(f"  Missing: {missing}")

    # 2. Extract TF expression from pancan matrix
    print("\n[2] Extracting TF expression from TCGA pancan matrix...")
    expr_df = extract_tf_expression(pancan_path, symbol_to_ensg)
    print(f"  Expression matrix: {expr_df.shape[0]} samples × "
          f"{expr_df.shape[1]} TFs")

    # 3. Load tri-modal data
    print("\n[3] Loading tri-modal NetITH data...")
    tri_df = pd.read_csv(tri_modal_path)
    print(f"  Tri-modal: {len(tri_df)} samples, "
          f"{tri_df['cancer'].nunique()} cancers")

    # Run the per-cancer TF -> NetITH regressions and collect coefficients
    # 4. Per-cancer regression
    print("\n[4] Running per-cancer TF → NetITH regression...")
    results, tf_coef_matrix = run_per_cancer_regression(tri_df, expr_df)

    # 5. Save results
    print("\n[5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/module_conservation_summary.csv',
                      index=False)
    print(f"  -> Saved module_conservation_summary.csv ({len(results_df)} cancers)")

    # Melt coefficients into a long TF x cancer table with module labels
    # TF × Cancer coefficient matrix
    coef_rows = []
    for cancer, coefs in tf_coef_matrix.items():
        for tf, coef in coefs.items():
            coef_rows.append({'cancer': cancer, 'tf': tf, 'coef': coef,
                              'module': MODULE_MAP.get(tf, 'Unknown')})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(f'{output_dir}/module_conservation_tf_coefs.csv',
                   index=False)
    print(f"  -> Saved module_conservation_tf_coefs.csv ({len(coef_df)} rows)")

    # 6. Visualizations
    print("\n[6] Creating visualizations...")
    plot_heatmap(tf_coef_matrix, output_dir)
    plot_r2_summary(results, output_dir)
    plot_gdsc_vs_tcga_correlation(tf_coef_matrix, output_dir)

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Module Conservation")
    print("=" * 60)

    # Per-cancer summary: GDSC vs TCGA coefficient Pearson r and AP-1 share of R2
    if len(results_df) > 0:
        # Overall conservation: correlation of GDSC vs TCGA coefs
        gdsc_lasso = pd.read_csv('results/depmap/tf_lasso_coefficients.csv')
        gdsc_coefs = dict(zip(gdsc_lasso['TF'], gdsc_lasso['lasso_coef']))

        print(f"{'Cancer':<8} {'n':>5} {'R² Full':>9} {'R² JUN':>9} "
              f"{'ρ(GDSC)':>9} {'AP-1%':>7}")
        print("-" * 55)
        for _, row in results_df.iterrows():
            cancer = row['cancer']
            tcga_coefs = tf_coef_matrix.get(cancer, {})
            common = set(gdsc_coefs.keys()) & set(tcga_coefs.keys())
            rho = np.nan
            if len(common) >= 5:
                rho, _ = stats.pearsonr(
                    [gdsc_coefs[t] for t in common],
                    [tcga_coefs[t] for t in common])
            ap1_pct = 100 * row['r2_ap1_stress'] / row['r2_full'] if row['r2_full'] > 0 else 0
            print(f"{cancer:<8} {row['n_samples']:>5} "
                  f"{row['r2_full']:>9.3f} {row['r2_jun']:>9.3f} "
                  f"{rho:>9.3f} {ap1_pct:>6.1f}%")

    print("\nDone! All outputs in results/tcga/")


if __name__ == '__main__':
    main()
