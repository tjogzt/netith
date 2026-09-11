"""
run_tcga_methylation.py — Integrate NetITH with methylation and CNV entropy in a TCGA tri-modal survival model.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - <DATA_ROOT>/xena/tcgapancan/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted...xena.gz: 450K beta values (streamed)
    - <DATA_ROOT>/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp: TCGA survival
    - <DATA_ROOT>/xena/tcgapancan/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz: CNV (if needed)
    - results/tcga/tcga_netith.csv: per-tumor NetITH
Outputs :
    - results/tcga/tcga_methylation_entropy.csv, tcga_tri_modal_merged.csv, tcga_tri_modal_survival.csv, tcga_cnv_entropy.csv
    - results/tcga/figures/tcga_tri_modal.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""

import os

import numpy as np
import pandas as pd
import gzip, os, sys, warnings
from pathlib import Path
from io import StringIO
from scipy.stats import spearmanr, false_discovery_control
from lifelines import CoxPHFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

METH_PATH = f"{DATA_ROOT}/xena/tcgapancan/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz"
SURV_PATH = f"{DATA_ROOT}/xena/tcgapancan/Survival_SupplementalTable_S1_20171025_xena_sp"
TCGA_DIR = Path(f"{ROOT}/results/tcga")
OUTPUT_DIR = TCGA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
MIN_CANCER_SAMPLES = 30  # minimum samples per cancer for the trivariate Cox model
MIN_CANCER_EVENTS = 3    # minimum events per cancer for the trivariate Cox model


# ═══════════════════════════════════════════════════════════════
# 1. Compute Methylation Entropy (streaming)
# ═══════════════════════════════════════════════════════════════
def compute_methylation_entropy():
    """
    Stream 450K beta values and compute per-sample Shannon entropy.
    
    H(sample) = -(1/N) * Σ [β·log₂(β+ε) + (1-β)·log₂(1-β+ε)]
    
    Streams line-by-line to avoid loading 14GB into memory.
    """
    print("[1/5] Computing methylation beta-value entropy (streaming 14GB)...")
    
    eps = 1e-10
    n_processed = 0
    sample_ids = None
    H_accum = None  # per-sample entropy accumulator
    n_valid = None  # per-sample count of valid probes
    
    with gzip.open(METH_PATH, 'rt') as f:
        header = f.readline().strip().split('\t')
        sample_ids = header[1:]  # first column is 'sample'
        n_samples = len(sample_ids)
        H_accum = np.zeros(n_samples, dtype=np.float64)
        n_valid = np.zeros(n_samples, dtype=np.int32)
        
        print(f"  Samples: {n_samples}, streaming probes...")
        
        for line_no, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < n_samples + 1:
                continue
            
            probe_id = parts[0]
            values_str = parts[1:]
            
            # Parse as float array
            try:
                values = np.array([float(v) if v and v != 'NA' else np.nan 
                                   for v in values_str], dtype=np.float64)
            except (ValueError, IndexError):
                continue
            
            # Per-probe Shannon entropy of the two-state methylation readout: -[b log2 b + (1-b) log2(1-b)]
            # Compute beta entropy: -[β·log₂(β+ε) + (1-β)·log₂(1-β+ε)]
            beta = np.clip(values, eps, 1 - eps)
            H_probe = -(beta * np.log2(beta) + (1 - beta) * np.log2(1 - beta))
            
            # Accumulate (NaN-safe)
            valid_mask = ~np.isnan(H_probe)
            H_accum[valid_mask] += H_probe[valid_mask]
            n_valid[valid_mask] += 1
            
            n_processed += 1
            if n_processed % 50000 == 0:
                print(f"    Processed {n_processed} probes...")
    
    # Average per-probe entropy across valid probes per sample (NaN when no probes measured)
    # Normalize
    valid_samples = n_valid > 0
    H_avg = np.full(n_samples, np.nan)
    H_avg[valid_samples] = H_accum[valid_samples] / n_valid[valid_samples]
    
    # Build result DataFrame
    meth_entropy = pd.DataFrame({
        'sample': [s.replace('-', '.') for s in sample_ids],
        'meth_entropy': H_avg
    }).set_index('sample')
    
    meth_entropy = meth_entropy.dropna()
    
    print(f"  Total probes: {n_processed}")
    print(f"  Mean probes/sample: {n_valid[valid_samples].mean():.0f}")
    print(f"  Methylation entropy: {len(meth_entropy)} samples, "
          f"mean={meth_entropy['meth_entropy'].mean():.4f}, "
          f"std={meth_entropy['meth_entropy'].std():.4f}")
    
    return meth_entropy


# ═══════════════════════════════════════════════════════════════
# 2. Load NetITH + CNV + Survival
# ═══════════════════════════════════════════════════════════════
def load_tcga_tri_modal():
    """Load TCGA NetITH, CNV entropy, and survival data."""
    print("[2/5] Loading NetITH, CNV entropy, and survival...")
    
    # Load per-tumor NetITH and normalize sample IDs to dotted format for cross-table joining
    # NetITH
    netith = pd.read_csv(f"{TCGA_DIR}/tcga_netith.csv")
    sample_col = 'sample' if 'sample' in netith.columns else netith.columns[1]
    netith = netith.set_index(sample_col)[['netith_bulk']]
    netith.index = netith.index.str.replace('-', '.', regex=False)
    print(f"  NetITH: {netith.shape}")
    
    # CNV entropy (if available; compute if not)
    cnv_path = f"{TCGA_DIR}/tcga_cnv_entropy.csv"
    if os.path.exists(cnv_path):
        cnv = pd.read_csv(cnv_path, index_col=0)
        cnv.index = cnv.index.str.replace('-', '.', regex=False)
        print(f"  CNV entropy: {cnv.shape}")
    else:
        print("  CNV entropy not found — computing from CNV data...")
        cnv = compute_tcga_cnv_entropy()
    
    # Survival
    surv_lines = []
    with open(SURV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or not stripped or '<-' in stripped:
                continue
            if stripped.startswith('head(data)'):
                stripped = stripped[len('head(data)'):]
            surv_lines.append(stripped)
    
    surv = pd.read_csv(StringIO('\n'.join(surv_lines)), sep='\t')
    surv['sample'] = surv['sample'].str.replace('-', '.', regex=False)
    surv = surv.set_index('sample')
    
    cancer_col = 'cancer type abbreviation'
    for c in surv.columns:
        if 'cancer' in c.lower():
            cancer_col = c
            break
    
    print(f"  Survival: {surv.shape}, cancer_col='{cancer_col}'")
    
    return netith, cnv, surv, cancer_col


def compute_tcga_cnv_entropy():
    """Quick CNV entropy from GDSC-style CNV data (gene-level)."""
    # TCGA CNV from Xena
    cnv_path = f"{DATA_ROOT}/xena/tcgapancan/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
    if not os.path.exists(cnv_path):
        print("  WARNING: TCGA CNV file not found, using zeros")
        return pd.DataFrame({'cnv_entropy': 0.0}, 
                          index=pd.Index([], name='sample'))
    
    # Stream and compute CNV entropy per sample
    with gzip.open(cnv_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        samples = header[3:]  # first 3 cols: Gene Symbol, Locus ID, Cytoband
        n_samples = len(samples)
        
        # Tally GISTIC2 copy-number calls (-2..+2) per sample across genes
        # Count -2,-1,0,1,2 per sample
        counts = np.zeros((n_samples, 5), dtype=np.int32)  # -2,-1,0,1,2
        
        n_genes = 0
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3 + n_samples:
                continue
            values = parts[3:]
            for j, v in enumerate(values):
                try:
                    vi = int(float(v))
                    if -2 <= vi <= 2:
                        counts[j, vi + 2] += 1
                except (ValueError, IndexError):
                    pass
            n_genes += 1
            if n_genes % 5000 == 0:
                print(f"    CNV: {n_genes} genes...")
        
        # Per-sample Shannon entropy over the five copy-number state frequencies
        # Shannon entropy from CNV counts
        cnv_entropy = np.zeros(n_samples)
        for j in range(n_samples):
            total = counts[j].sum()
            if total > 0:
                probs = counts[j] / total
                probs = probs[probs > 0]
                cnv_entropy[j] = -np.sum(probs * np.log2(probs))
        
        sample_ids = [s.replace('-', '.') for s in samples]
        result = pd.DataFrame({
            'sample': sample_ids,
            'cnv_entropy': cnv_entropy
        }).set_index('sample')
    
    result.to_csv(f"{TCGA_DIR}/tcga_cnv_entropy.csv")
    print(f"  CNV entropy computed: {result.shape}")
    return result


# ═══════════════════════════════════════════════════════════════
# 3. Tri-Modal Merge & Analysis
# ═══════════════════════════════════════════════════════════════
def tri_modal_analysis(meth, netith, cnv, surv, cancer_col):
    """Merge all three modalities and compute correlations."""
    print("[3/5] Tri-modal merge and correlation analysis...")
    
    # Inner-join NetITH, CNV entropy, methylation entropy, and survival on shared samples
    # Merge all
    merged = (netith
              .join(cnv, how='inner')
              .join(meth, how='inner')
              .join(surv[['OS', 'OS.time', cancer_col]], how='inner'))
    merged = merged.rename(columns={cancer_col: 'cancer'})
    merged = merged.dropna(subset=['cancer', 'OS', 'OS.time'])
    
    # Clean numeric
    for col in ['netith_bulk', 'cnv_entropy', 'meth_entropy', 'OS.time', 'OS']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['netith_bulk', 'cnv_entropy', 'meth_entropy', 'OS.time'])
    merged['OS'] = merged['OS'].astype(int)
    
    print(f"  Tri-modal merged: {len(merged)} samples, "
          f"{merged['cancer'].nunique()} cancer types")
    
    # Z-score for comparison
    for col in ['netith_bulk', 'cnv_entropy', 'meth_entropy']:
        merged[f'{col}_z'] = (merged[col] - merged[col].mean()) / merged[col].std()
    
    # Pairwise Spearman correlations between the three heterogeneity measures (null: no monotonic association)
    # Pairwise correlations
    cols = ['netith_bulk', 'cnv_entropy', 'meth_entropy']
    print("\n  Pairwise Spearman correlations:")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            mask = ~(np.isnan(merged[cols[i]]) | np.isnan(merged[cols[j]]))
            r, p = spearmanr(merged.loc[mask, cols[i]], merged.loc[mask, cols[j]])
            print(f"    {cols[i]:15s} × {cols[j]:15s}: ρ={r:+.4f}, p={p:.2e}")
    
    return merged


# ═══════════════════════════════════════════════════════════════
# 4. Tri-Modal Survival Analysis
# ═══════════════════════════════════════════════════════════════
def tri_modal_survival(merged):
    """Cox models: univariate, bivariate, trivariate comparisons."""
    print("[4/5] Tri-modal survival analysis...")
    
    modalities = ['netith_bulk_z', 'cnv_entropy_z', 'meth_entropy_z']
    labels = ['NetITH', 'CNV Entropy', 'Methylation Entropy']
    
    # Univariate Cox PH model per modality (null: HR = 1, i.e., no survival association)
    # Pan-cancer: each alone
    print("\n  Pan-cancer Cox (univariate):")
    uv_results = {}
    for col, label in zip(modalities, labels):
        try:
            cph = CoxPHFitter()
            df = merged[['OS.time', 'OS', col]].dropna()
            cph.fit(df, duration_col='OS.time', event_col='OS')
            hr = cph.hazard_ratios_[col]
            p = cph.summary.loc[col, 'p']
            ci = (cph.confidence_intervals_.loc[col].values)
            uv_results[label] = {'HR': hr, 'p': p, 'CI_low': ci[0], 'CI_high': ci[1]}
            print(f"    {label:20s}: HR={hr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
        except Exception as e:
            print(f"    {label:20s}: FAILED ({e})")
    
    # Trivariate Cox model: each modality's HR adjusted for the other two modalities
    # Trivariate model
    print("\n  Pan-cancer Cox (trivariate):")
    try:
        cph3 = CoxPHFitter()
        df3 = merged[['OS.time', 'OS'] + modalities].dropna()
        cph3.fit(df3, duration_col='OS.time', event_col='OS')
        for col, label in zip(modalities, labels):
            hr = cph3.hazard_ratios_[col]
            p = cph3.summary.loc[col, 'p']
            ci = cph3.confidence_intervals_.loc[col].values
            print(f"    {label:20s}: HR={hr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
        tri_hr = {label: cph3.hazard_ratios_[col] for col, label in zip(modalities, labels)}
        tri_p = {label: cph3.summary.loc[col, 'p'] for col, label in zip(modalities, labels)}
    except Exception as e:
        print(f"    FAILED: {e}")
        tri_hr, tri_p = {}, {}
    
    # Per-cancer trivariate Cox models (top 20 cancers, >= 30 samples, >= 3 events)
    # Per-cancer
    print("\n  Per-cancer Cox (trivariate, top cancers):")
    per_cancer = []
    for cancer in merged['cancer'].value_counts().head(20).index:
        sub = merged[merged['cancer'] == cancer]
        if len(sub) < MIN_CANCER_SAMPLES:
            continue
        df = sub[['OS.time', 'OS'] + modalities].dropna()
        if len(df) < MIN_CANCER_SAMPLES or df['OS'].sum() < MIN_CANCER_EVENTS:
            continue
        
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col='OS.time', event_col='OS')
            row = {'cancer': cancer, 'n': len(df), 'n_events': int(df['OS'].sum())}
            for col, label in zip(modalities, labels):
                row[f'HR_{label}'] = cph.hazard_ratios_[col]
                row[f'p_{label}'] = cph.summary.loc[col, 'p']
            per_cancer.append(row)
        except:
            pass
    
    per_cancer_df = pd.DataFrame(per_cancer)
    
    # Benjamini-Hochberg FDR correction of per-cancer p-values per modality (scipy false_discovery_control)
    # FDR correction
    for label in labels:
        pcol = f'p_{label}'
        if pcol in per_cancer_df.columns:
            pvals = per_cancer_df[pcol].dropna().values
            if len(pvals) > 0:
                fdrs = false_discovery_control(pvals, method='bh')
                per_cancer_df.loc[per_cancer_df[pcol].notna(), f'fdr_{label}'] = fdrs
    
    for label in labels:
        fdr_col = f'fdr_{label}'
        if fdr_col in per_cancer_df.columns:
            n_sig = (per_cancer_df[fdr_col] < 0.05).sum()
            print(f"    {label}: {n_sig}/{len(per_cancer_df)} FDR<0.05")
    
    return uv_results, tri_hr, tri_p, per_cancer_df


# ═══════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════
def plot_tri_modal(merged, uv_results, tri_hr, tri_p, per_cancer_df):
    """6-panel tri-modal visualization."""
    print("[5/5] Plotting...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    
    cols = ['netith_bulk', 'cnv_entropy', 'meth_entropy']
    labels = ['NetITH (bits)', 'CNV Entropy (bits)', 'Methylation Entropy (bits)']
    short_labels = ['NetITH', 'CNV', 'Methylation']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # A-C: pairwise scatter + fit line with Spearman rho
    # A-C: Pairwise scatter plots
    pairs = [(0, 1, axes[0, 0]), (0, 2, axes[0, 1]), (1, 2, axes[0, 2])]
    
    for i, j, ax in pairs:
        mask = ~(np.isnan(merged[cols[i]]) | np.isnan(merged[cols[j]]))
        x, y = merged.loc[mask, cols[i]], merged.loc[mask, cols[j]]
        r, p = spearmanr(x, y)
        ax.scatter(x, y, alpha=0.1, s=3, c='#7f8c8d', edgecolors='none')
        # Add fit line
        from numpy.polynomial.polynomial import polyfit
        b, m = polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b + m*xs, 'navy', alpha=0.5, linewidth=2)
        ax.set_xlabel(labels[i], fontsize=9)
        ax.set_ylabel(labels[j], fontsize=9)
        ax.set_title(f'{short_labels[i]} × {short_labels[j]}: ρ={r:+.3f}, p={p:.2e}')
    
    # D: Forest plot — per-cancer HRs from trivariate Cox
    ax = axes[1, 0]
    if len(per_cancer_df) > 0:
        df_plot = per_cancer_df.sort_values('HR_NetITH')
        x = np.arange(len(df_plot))
        w = 0.25
        for k, (label, c) in enumerate(zip(short_labels, colors)):
            hr_col = f'HR_{label}'
            if hr_col in df_plot.columns:
                valid = df_plot[hr_col].notna()
                ax.barh(x[valid] + (k-1)*w, np.log2(df_plot.loc[valid, hr_col]), w,
                       label=label, alpha=0.8, color=c)
        ax.axvline(0, color='black', alpha=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(df_plot['cancer'], fontsize=5)
        ax.set_xlabel('log₂(Hazard Ratio)')
        ax.set_title(f'Per-Cancer Tri-Modal Cox\n({len(df_plot)} cancers)')
        ax.legend(fontsize=7)
    
    # E: Bar chart — pan-cancer univariate vs trivariate HRs
    ax = axes[1, 1]
    if uv_results:
        x = np.arange(3)
        w = 0.3
        uv_hrs = [uv_results.get(l, {}).get('HR', 1) for l in short_labels]
        tri_hrs_vals = [tri_hr.get(l, 1) for l in short_labels]
        
        bars1 = ax.bar(x - w/2, [np.log2(h) for h in uv_hrs], w, 
                       label='Univariate', alpha=0.8, color='#bdc3c7')
        bars2 = ax.bar(x + w/2, [np.log2(h) for h in tri_hrs_vals], w,
                       label='Trivariate', alpha=0.8, color='#2c3e50')
        ax.axhline(0, color='black', alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel('log₂(Hazard Ratio)')
        ax.set_title('Pan-Cancer: Uni- vs Tri-variate Cox')
        ax.legend(fontsize=8)
        
        # Add significance stars
        for k, label in enumerate(short_labels):
            p_uni = uv_results.get(label, {}).get('p', 1.0)
            p_tri = tri_p.get(label, 1.0)
            stars = lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            ax.text(k - w/2, np.log2(uv_hrs[k]) + 0.05, stars(p_uni), 
                   ha='center', fontsize=7)
            ax.text(k + w/2, np.log2(tri_hrs_vals[k]) + 0.05, stars(p_tri),
                   ha='center', fontsize=7)
    
    # F: logistic drop-one-AUC contributions of each modality to survival prediction
    # F: Variance decomposition (R² from linear model)
    ax = axes[1, 2]
    # Regress survival ~ NetITH + CNV + Methylation (surrogate by OS event)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    df_lr = merged[['OS', 'netith_bulk', 'cnv_entropy', 'meth_entropy']].dropna()
    if len(df_lr) > 100:
        X = df_lr[['netith_bulk', 'cnv_entropy', 'meth_entropy']].values
        y = df_lr['OS'].values
        
        # Fit logistic
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X_scaled, y)
        
        # Wald test p-values (approximate from coefficients / SE)
        # Simple approach: drop-one R²
        from sklearn.metrics import roc_auc_score
        full_auc = roc_auc_score(y, lr.predict_proba(X_scaled)[:, 1])
        
        auc_drops = []
        for k in range(3):
            X_drop = np.delete(X_scaled, k, axis=1)
            lr_drop = LogisticRegression(penalty=None, max_iter=1000)
            lr_drop.fit(X_drop, y)
            auc_drop = roc_auc_score(y, lr_drop.predict_proba(X_drop)[:, 1])
            auc_drops.append(max(0, full_auc - auc_drop))
        
        # Normalize
        total_drop = sum(auc_drops) or 1.0
        contributions = [d / total_drop for d in auc_drops]
        
        bars = ax.bar(range(3), contributions, color=colors, alpha=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel('Relative Contribution (ΔAUC)')
        ax.set_title('Survival Prediction: Modality Contributions')
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{contrib:.1%}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    figpath = f"{OUTPUT_DIR}/figures/tcga_tri_modal.png"
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {figpath}")
    plt.close()


def main():
    # Stream-compute per-sample methylation entropy and write it out
    # Compute methylation entropy
    meth = compute_methylation_entropy()
    meth.to_csv(f"{OUTPUT_DIR}/tcga_methylation_entropy.csv")
    
    # Load other data
    netith, cnv, surv, cancer_col = load_tcga_tri_modal()
    
    # Merge the three modalities with survival and write the merged table
    # Tri-modal merge
    merged = tri_modal_analysis(meth, netith, cnv, surv, cancer_col)
    merged.to_csv(f"{OUTPUT_DIR}/tcga_tri_modal_merged.csv")
    
    # Cox survival analyses (univariate, trivariate, per-cancer with FDR) and write results
    # Survival
    uv_results, tri_hr, tri_p, per_cancer = tri_modal_survival(merged)
    per_cancer.to_csv(f"{OUTPUT_DIR}/tcga_tri_modal_survival.csv", index=False)
    
    # Plot
    plot_tri_modal(merged, uv_results, tri_hr, tri_p, per_cancer)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: TCGA Tri-Modal Heterogeneity Model")
    print("=" * 60)
    print(f"  Samples: {len(merged)}")
    print(f"  Cancer types: {merged['cancer'].nunique()}")
    
    cols = ['netith_bulk', 'cnv_entropy', 'meth_entropy']
    labels = ['NetITH', 'CNV Entropy', 'Methylation Entropy']
    
    print("\n  Pairwise correlations:")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            mask = ~(np.isnan(merged[cols[i]]) | np.isnan(merged[cols[j]]))
            r, p = spearmanr(merged.loc[mask, cols[i]], merged.loc[mask, cols[j]])
            print(f"    {labels[i]} × {labels[j]}: ρ={r:+.4f}")
    
    print("\n  Pan-cancer survival (trivariate Cox):")
    for label in labels:
        if label in tri_hr and label in tri_p:
            print(f"    {label:20s}: HR={tri_hr[label]:.3f}, p={tri_p[label]:.2e}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
