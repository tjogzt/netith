"""
run_tcga_validation.py — Compute TCGA pan-cancer NetITH and test its association with survival.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - <DATA_ROOT>/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz: TCGA expression
    - <DATA_ROOT>/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp: TCGA survival
Outputs :
    - results/tcga/tcga_netith.csv, tcga_survival_results.csv
    - results/tcga/figures/tcga_{forest_plot,pancan_km}.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os
import sys
import gzip
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.linalg
from scipy.stats import mannwhitneyu
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Config ───────────────────────────────────────────────────

TCGA_EXPR_PATH = f"{DATA_ROOT}/xena/tcgapancan/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
TCGA_SURV_PATH = f"{DATA_ROOT}/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp"
OUTPUT_DIR = Path(f"{ROOT}/results/tcga")
MIN_SAMPLES_CANCER = 30
N_TOP_GENES = 300


# ─── Load Data ────────────────────────────────────────────────

def load_tcga_expression(path: str) -> pd.DataFrame:
    """Load TCGA gene expression matrix. Returns (samples × genes)."""
    print("Loading TCGA expression...")
    df = pd.read_csv(path, sep='\t', compression='gzip', index_col=0)
    # Transpose: (genes × samples) → (samples × genes)
    df = df.T
    print(f"  Expression: {df.shape[0]} samples × {df.shape[1]} genes")
    return df


def load_tcga_survival(path: str) -> pd.DataFrame:
    """Load TCGA survival data with robust parsing.

    The file has R script preamble ending with 'head(data)' that runs
    directly into the tab-separated header line without a newline.
    """
    print("Loading TCGA survival...")
    rows = []
    header = None
    with open(path, 'r') as f:
        for line in f:
            # Find first line with tab characters (data/header)
            if '\t' in line:
                # Strip R preamble: find where "sample" starts
                idx = line.find('sample\t')
                if idx >= 0:
                    line = line[idx:]
                header = line.strip().split('\t')
                break
        for line in f:
            if '\t' in line:
                rows.append(line.strip().split('\t'))

    if header is None:
        raise ValueError("Could not find header in survival file")

    # Pad rows to match header length (some rows may have missing trailing empty fields)
    n_cols = len(header)
    rows_padded = []
    for r in rows:
        if len(r) < n_cols:
            r = r + [''] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        rows_padded.append(r)

    surv = pd.DataFrame(rows_padded, columns=header)
    print(f"  Survival: {surv.shape}")

    # Rename and convert
    surv = surv.rename(columns={'cancer type abbreviation': 'cancer_type'})
    surv['OS'] = pd.to_numeric(surv['OS'], errors='coerce')
    surv['OS.time'] = pd.to_numeric(surv['OS.time'], errors='coerce')
    surv['DSS'] = pd.to_numeric(surv['DSS'], errors='coerce')
    surv['DSS.time'] = pd.to_numeric(surv['DSS.time'], errors='coerce')
    surv['PFI'] = pd.to_numeric(surv['PFI'], errors='coerce')
    surv['PFI.time'] = pd.to_numeric(surv['PFI.time'], errors='coerce')

    surv = surv.dropna(subset=['OS', 'OS.time'])
    surv = surv[surv['OS.time'] > 0]
    print(f"  With valid OS: {len(surv)}")
    return surv


# ─── Gene ID Mapping ──────────────────────────────────────────

def map_gene_ids(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Handle mixed gene ID formats in TCGA expression data.

    The Xena PanCan dataset mostly uses gene symbols (20,501 of 20,530),
    with a few Entrez Gene ID outliers. We filter out pure numeric IDs
    and keep gene symbols.
    """
    gene_ids = expr_df.columns.tolist()

    # Separate numeric (Entrez) from symbols
    numeric_mask = [g.replace('.', '').isdigit() for g in gene_ids]
    n_numeric = sum(numeric_mask)
    n_symbol = len(gene_ids) - n_numeric

    print(f"  Gene IDs: {n_symbol} symbols, {n_numeric} numeric (Entrez)")

    if n_symbol > n_numeric:
        # Keep only gene symbols
        symbol_cols = [g for i, g in enumerate(gene_ids) if not numeric_mask[i]]
        expr_df = expr_df[symbol_cols]
        # Remove duplicate gene symbols
        expr_df = expr_df.loc[:, ~expr_df.columns.duplicated()]
        print(f"  Kept {expr_df.shape[1]} unique gene symbols")
    else:
        print("  WARNING: Mostly numeric IDs — mapping needed but skipping")

    return expr_df


# ─── NetITH from Bulk ─────────────────────────────────────────

def compute_netith_bulk(
    expr_df: pd.DataFrame,
    collectri_net: pd.DataFrame,
    n_top_genes: int = N_TOP_GENES,
) -> pd.DataFrame:
    """Compute NetITH for each TCGA sample.

    For bulk RNA-seq, the "per-cell" GRN becomes a "per-patient" GRN.
    Edge weights = |CollecTRI weight| × z(TF_expr) × z(target_expr).
    """
    from scipy.stats import zscore

    print("\n[NetITH-Bulk] Computing per-patient network entropy...")

    # Focus on genes in CollecTRI
    collectri_genes = set(collectri_net['source']) | set(collectri_net['target'])
    available_genes = [g for g in expr_df.columns if g in collectri_genes]

    if len(available_genes) < 50:
        print(f"  WARNING: Only {len(available_genes)} CollecTRI genes in expression data")
        available_genes = list(expr_df.columns)[:n_top_genes]

    # Select top variable among CollecTRI genes
    if len(available_genes) > n_top_genes:
        variances = expr_df[available_genes].var()
        available_genes = variances.nlargest(n_top_genes).index.tolist()

    print(f"  Using {len(available_genes)} genes for GRN")

    # Subset expression
    expr_sub = expr_df[available_genes].copy()
    # Z-score normalize across samples
    expr_z = expr_sub.apply(zscore, axis=0).fillna(0)

    # Map CollecTRI to available genes
    gene_to_idx = {g: i for i, g in enumerate(available_genes)}
    edges = []
    for _, row in collectri_net.iterrows():
        tf = row['source']
        target = row['target']
        weight = float(row.get('weight', 1.0))
        if tf in gene_to_idx and target in gene_to_idx:
            edges.append({
                'tf_idx': gene_to_idx[tf],
                'target_idx': gene_to_idx[target],
                'weight': abs(weight),
            })

    n_genes = len(available_genes)
    print(f"  GRN edges: {len(edges)}, genes: {n_genes}")

    # Compute per-sample entropy
    entropies = np.zeros(len(expr_sub))
    expr_vals = expr_z.values

    import time
    t_start = time.time()
    for i in range(len(expr_sub)):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(expr_sub) - i - 1)
            print(f"    Sample {i + 1}/{len(expr_sub)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        # Build adjacency: A[src,dst] = |weight| * z_src * z_dst
        A = np.zeros((n_genes, n_genes))
        for e in edges:
            tf_act = expr_vals[i, e['tf_idx']]
            target_act = expr_vals[i, e['target_idx']]
            A[e['tf_idx'], e['target_idx']] = e['weight'] * abs(tf_act) * abs(target_act)

        # Laplacian
        A = A + A.T  # FIX(eigvalsh-audit 2026-08-16): symmetrize directed TF->target adjacency before Laplacian (eigvalsh silently used one triangle)
        degrees = A.sum(axis=1)
        L = np.diag(degrees) - A

        trace_L = degrees.sum()
        if trace_L < 1e-10:
            entropies[i] = 0.0
            continue

        try:
            eigs = scipy.linalg.eigvalsh(L)
            eigs = np.clip(eigs, 0, None)
            rho = eigs / (trace_L + 1e-10)
            rho = np.clip(rho, 1e-12, 1.0)
            entropies[i] = -np.sum(rho * np.log2(rho))
        except Exception:
            entropies[i] = np.nan

    mask = ~np.isnan(entropies)
    print(f"  Valid samples: {mask.sum()}/{len(entropies)}")
    print(f"  Entropy range: [{entropies[mask].min():.3f}, {entropies[mask].max():.3f}]")

    result = pd.DataFrame({
        'sample': expr_sub.index,
        'netith_bulk': entropies,
    }, index=expr_sub.index)

    return result


# ─── Survival Analysis ───────────────────────────────────────

def survival_analysis(
    surv: pd.DataFrame,
    netith: pd.DataFrame,
    output_dir: str,
):
    """Cox PH and KM analysis per cancer type and pan-cancer."""
    print("\n" + "=" * 60)
    print("[Survival Analysis]")
    print("=" * 60)

    # Merge
    df = surv.merge(netith, left_on='sample', right_index=True, how='inner')
    df = df.dropna(subset=['netith_bulk', 'OS', 'OS.time'])
    print(f"  Merged: {len(df)} samples")

    # Binary split: high vs low NetITH
    median_netith = df['netith_bulk'].median()
    df['netith_high'] = (df['netith_bulk'] > median_netith).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # ── Pan-Cancer KM ──
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    fig, axes = None, None

    # ── Per-cancer-type analysis ──
    cancer_types = df['cancer_type'].value_counts()
    valid_cts = cancer_types[cancer_types >= MIN_SAMPLES_CANCER].index.tolist()

    print(f"\n  Cancer types with ≥{MIN_SAMPLES_CANCER} samples: {len(valid_cts)}")

    for ct in valid_cts:
        ct_df = df[df['cancer_type'] == ct]

        # KM
        high = ct_df[ct_df['netith_high'] == 1]
        low = ct_df[ct_df['netith_high'] == 0]

        if len(high) < 10 or len(low) < 10:
            continue

        lr = logrank_test(
            high['OS.time'], low['OS.time'],
            event_observed_A=high['OS'], event_observed_B=low['OS'],
        )

        # Cox PH
        try:
            cph = CoxPHFitter()
            cox_df = ct_df[['OS.time', 'OS', 'netith_bulk']].copy()
            cox_df = cox_df.rename(columns={'OS.time': 'duration', 'OS': 'event'})
            cph.fit(cox_df, duration_col='duration', event_col='event')
            hr = cph.hazard_ratios_['netith_bulk']
            cox_p = cph.summary.loc['netith_bulk', 'p']
            cox_se = cph.summary.loc['netith_bulk', 'se(coef)']  # FIX(P0-7): report true SE, not p-inferred
        except Exception:
            hr = np.nan
            cox_p = np.nan
            cox_se = np.nan

        results.append({
            'cancer_type': ct,
            'n': len(ct_df),
            'n_events': int(ct_df['OS'].sum()),
            'median_netith': ct_df['netith_bulk'].median(),
            'km_logrank_p': lr.p_value,
            'cox_hr': hr,
            'cox_p': cox_p,
            'cox_se': cox_se,
        })

        print(f"  {ct}: n={len(ct_df)}, events={int(ct_df['OS'].sum())}, "
              f"KM p={lr.p_value:.4f}, Cox HR={hr:.3f}")

    # ── Pan-Cancer Cox (stratified) ──
    try:
        cox_df_all = df[['OS.time', 'OS', 'netith_bulk', 'cancer_type']].copy()
        cox_df_all = cox_df_all.rename(columns={'OS.time': 'duration', 'OS': 'event'})
        cph_all = CoxPHFitter()
        cph_all.fit(cox_df_all, duration_col='duration', event_col='event',
                     strata=['cancer_type'])
        print(f"\n  Pan-Cancer Cox (stratified): HR={cph_all.hazard_ratios_['netith_bulk']:.3f}, "
              f"p={cph_all.summary.loc['netith_bulk', 'p']:.4f}")
    except Exception as e:
        print(f"  Pan-Cancer Cox failed: {e}")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "tcga_survival_results.csv"), index=False)
    print(f"\n  Saved: {output_dir}/tcga_survival_results.csv")

    return results_df, df


# ─── Visualize ────────────────────────────────────────────────

def visualize_tcga(results_df: pd.DataFrame, merged_df: pd.DataFrame, output_dir: str):
    """Generate TCGA figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # (A) Forest plot of Cox HR per cancer type
    df_plot = results_df.dropna(subset=['cox_hr']).sort_values('cox_hr')
    if len(df_plot) > 0:
        fig, ax = plt.subplots(figsize=(10, max(5, len(df_plot) * 0.35)))
        colors = ['#e74c3c' if hr > 1 else '#2ecc71' for hr in df_plot['cox_hr']]
        ax.barh(range(len(df_plot)), np.log2(df_plot['cox_hr']), color=colors, alpha=0.7)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels([f"{ct} (n={n})" for ct, n in zip(df_plot['cancer_type'], df_plot['n'])])
        ax.set_xlabel('log2(Hazard Ratio)')
        ax.set_title('TCGA Pan-Cancer: NetITH Hazard Ratios')
        plt.tight_layout()
        fig.savefig(fig_dir / "tcga_forest_plot.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: tcga_forest_plot.png")

    # (B) Pan-Cancer KM curve
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    high = merged_df[merged_df['netith_high'] == 1]
    low = merged_df[merged_df['netith_high'] == 0]
    kmf_h.fit(high['OS.time'], high['OS'], label=f'High NetITH (n={len(high)})')
    kmf_l.fit(low['OS.time'], low['OS'], label=f'Low NetITH (n={len(low)})')
    lr = logrank_test(high['OS.time'], low['OS.time'],
                       event_observed_A=high['OS'], event_observed_B=low['OS'])
    kmf_h.plot_survival_function(ax=ax, color='#e74c3c')
    kmf_l.plot_survival_function(ax=ax, color='#2ecc71')
    ax.set_title(f'Pan-Cancer KM: NetITH (Log-rank p={lr.p_value:.4f})')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival Probability')
    fig.savefig(fig_dir / "tcga_pancan_km.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: tcga_pancan_km.png")


# ─── Main ─────────────────────────────────────────────────────

def run_tcga_validation(
    expr_path: str = TCGA_EXPR_PATH,
    surv_path: str = TCGA_SURV_PATH,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TCGA Pan-Cancer NetITH Validation                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load CollecTRI
    import decoupler as dc
    print("Loading CollecTRI...")
    net = dc.op.collectri(organism='human')
    if 'sign_decision' in net.columns:
        net = net[~net['sign_decision'].str.startswith('default')]
    print(f"  CollecTRI: {len(net)} edges, {net['source'].nunique()} TFs")

    # Load data
    expr = load_tcga_expression(expr_path)
    surv = load_tcga_survival(surv_path)

    # Map gene IDs (fast path: most are already symbols)
    import time
    t0 = time.time()
    expr = map_gene_ids(expr)
    print(f"  Gene mapping took {time.time()-t0:.1f}s")

    # Compute NetITH
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    netith = compute_netith_bulk(expr, net)
    print(f"  NetITH computation took {time.time()-t0:.1f}s")
    netith.to_csv(os.path.join(output_dir, "tcga_netith.csv"))

    # Survival analysis
    results_df, merged_df = survival_analysis(surv, netith, output_dir)

    # Visualize
    visualize_tcga(results_df, merged_df, output_dir)

    print("\nTCGA validation complete!")
    return results_df, merged_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_tcga_validation(output_dir=args.output)
