"""
plots.py — Publication-ready plotting functions for NetITH.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : NetITH result DataFrames; traditional ITH metrics; survival tables
Outputs : matplotlib figures (optional: saved PNG/PDF)
Module  : src.visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Style Configuration ───────────────────────────────────────

def set_netentropy_style():
    """Set consistent publication style."""
    # Global matplotlib RC defaults for consistent publication figures
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

# Color palette for cancer types
CANCER_COLORS = {
    "LUAD": "#E64B35",
    "SKCM": "#4DBBD5",
    "GBM": "#00A087",
    "BRCA": "#F39B7F",
    "COAD": "#3C5488",
    "Normal": "#999999",
}

# ─── Figure 1: Method Overview ─────────────────────────────────

def fig1_method_overview(
    simulated_entropies: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Figure 1: Schematic overview + simulation validation.

    Args:
        simulated_entropies: Dict with keys 'low_heterogeneity' and
                            'high_heterogeneity', each an array of
                            per-cell entropy values.
        save_path: Path to save figure.
    """
    set_netentropy_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # Panel a: Schematic (placeholder - replace with illustration)
    ax = axes[0, 0]
    ax.text(0.5, 0.5, "Schematic:\nscRNA-seq → GRN → Network Entropy → NetITH",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("a) NetEntropy Framework", loc="left")
    ax.axis("off")

    # Panel b: Simulation - entropy distributions
    ax = axes[0, 1]
    for label, entropies in simulated_entropies.items():
        sns.kdeplot(entropies, label=label, ax=ax, fill=True, alpha=0.3)
    ax.set_xlabel("Per-cell Network Entropy")
    ax.set_ylabel("Density")
    ax.set_title("b) Simulated Entropy Distributions", loc="left")
    ax.legend()

    # Panel c: NetITH comparison
    ax = axes[1, 0]
    from src.network_entropy.netith import compute_netith
    netith_vals = {
        k: compute_netith(v)["netith"]
        for k, v in simulated_entropies.items()
    }
    bars = ax.bar(netith_vals.keys(), netith_vals.values(),
                  color=["#4DBBD5", "#E64B35"])
    ax.set_ylabel("NetITH Score")
    ax.set_title("c) Functional ITH Quantification", loc="left")
    for bar, val in zip(bars, netith_vals.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Panel d: Entropy type concordance (placeholder)
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Entropy Type\nConcordance",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("d) Cross-Entropy Concordance", loc="left")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Figure 2: Pan-Cancer Comparison ───────────────────────────

def fig2_pan_cancer_comparison(
    netith_df: pd.DataFrame,
    trad_ith_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Figure 2: NetITH vs traditional ITH across cancers.

    Args:
        netith_df: Columns: cancer_type, sample_id, netith, mean_entropy, etc.
        trad_ith_df: Columns: sample_id, MATH, shannon, depth, etc.
        save_path: Path to save figure.
    """
    set_netentropy_style()
    merged = netith_df.merge(trad_ith_df, on="sample_id")

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # Panel a: NetITH distribution by cancer
    ax = axes[0, 0]
    order = merged.groupby("cancer_type")["netith"].median().sort_values().index
    sns.boxplot(data=merged, x="cancer_type", y="netith", order=order,
                palette=CANCER_COLORS, ax=ax, width=0.6)
    sns.stripplot(data=merged, x="cancer_type", y="netith", order=order,
                  color="black", size=2, alpha=0.3, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("NetITH Score")
    ax.set_title("a) NetITH Across Cancers", loc="left")

    # Panel b: NetITH vs MATH
    ax = axes[0, 1]
    if "MATH" in merged.columns:
        for ct in merged["cancer_type"].unique():
            subset = merged[merged["cancer_type"] == ct]
            ax.scatter(subset["MATH"], subset["netith"],
                       c=CANCER_COLORS.get(ct, "gray"),
                       label=ct, s=15, alpha=0.7)
        ax.set_xlabel("MATH Score")
        ax.set_ylabel("NetITH Score")
        ax.set_title("b) NetITH vs MATH", loc="left")
        ax.legend(markerscale=2, frameon=False)
    else:
        ax.text(0.5, 0.5, "MATH data\nnot available",
                ha="center", va="center", transform=ax.transAxes)

    # Panel c: NetITH vs Shannon
    ax = axes[1, 0]
    if "shannon" in merged.columns:
        for ct in merged["cancer_type"].unique():
            subset = merged[merged["cancer_type"] == ct]
            ax.scatter(subset["shannon"], subset["netith"],
                       c=CANCER_COLORS.get(ct, "gray"),
                       label=ct, s=15, alpha=0.7)
        ax.set_xlabel("Shannon Diversity")
        ax.set_ylabel("NetITH Score")
        ax.set_title("c) NetITH vs Shannon Index", loc="left")
    else:
        ax.text(0.5, 0.5, "Shannon data\nnot available",
                ha="center", va="center", transform=ax.transAxes)

    # Panel d: Correlation heatmap
    ax = axes[1, 1]
    metric_cols = ["netith", "MATH", "shannon", "depth", "estimate"]
    metric_cols = [c for c in metric_cols if c in merged.columns]
    corr = merged[metric_cols].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("d) Cross-Metric Correlation", loc="left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Figure 3: Survival Analysis ───────────────────────────────

def fig3_survival_analysis(
    survival_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Figure 3: NetITH prognostic value.

    Args:
        survival_df: Columns: sample_id, cancer_type, netith, time, event,
                    netith_group ('High'/'Low'), MATH, stage, age
        save_path: Path to save figure.
    """
    set_netentropy_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # Panel a: Pan-cancer KM curve
    ax = axes[0, 0]
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    for group in ["Low", "High"]:
        mask = survival_df["netith_group"] == group
        kmf.fit(survival_df.loc[mask, "time"],
                survival_df.loc[mask, "event"],
                label=f"NetITH {group}")
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("a) Pan-Cancer Overall Survival", loc="left")

    # Panel b: Forest plot of Cox HR per cancer
    ax = axes[0, 1]
    # Placeholder - actual implementation needs Cox model results
    ax.text(0.5, 0.5, "Forest Plot\nHR (95% CI) per Cancer",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("b) Cox Regression Hazard Ratios", loc="left")
    ax.axis("off")

    # Panel c: C-index comparison
    ax = axes[1, 0]
    models = ["Clinical only", "+ MATH", "+ NetITH", "Full model"]
    cindices = [0.62, 0.65, 0.71, 0.74]  # placeholder values
    colors = ["#999999", "#4DBBD5", "#E64B35", "#00A087"]
    ax.barh(models, cindices, color=colors)
    ax.set_xlabel("C-index")
    ax.set_title("c) Incremental Prognostic Value", loc="left")
    ax.set_xlim(0.5, 0.8)

    # Panel d: Independent validation
    ax = axes[1, 1]
    # Placeholder
    ax.text(0.5, 0.5, "Independent\nTCGA Validation",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("d) External Validation (TCGA)", loc="left")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Figure 4: Drug Response ───────────────────────────────────

def fig4_drug_response(
    drug_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Figure 4: NetITH association with drug response.

    Args:
        drug_df: Columns: sample_id, cancer_type, netith, drug_name, IC50, AUC.
        save_path: Path to save figure.
    """
    set_netentropy_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    ax = axes[0, 0]
    ax.text(0.5, 0.5, "NetITH-Drug Sensitivity\nHeatmap",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("a) Drug Sensitivity Association", loc="left")

    ax = axes[0, 1]
    ax.text(0.5, 0.5, "High vs Low NetITH\nIC50 Comparison",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("b) Drug IC50 by NetITH Group", loc="left")

    ax = axes[1, 0]
    ax.text(0.5, 0.5, "Immunotherapy\nResponse",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("c) ICI Response Prediction", loc="left")

    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Immune Infiltration\nComposition",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("d) NetITH vs Immune Landscape", loc="left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Figure 5: Therapy Optimization ────────────────────────────

def fig5_therapy_optimization(
    sensitivity_df: pd.DataFrame,
    hotspot_genes: List[str],
    save_path: Optional[str] = None,
):
    """Figure 5: Entropy-based therapy optimization.

    Args:
        sensitivity_df: Gene-level entropy sensitivity values.
        hotspot_genes: Top entropy hotspot genes.
        save_path: Path to save figure.
    """
    set_netentropy_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    ax = axes[0, 0]
    ax.text(0.5, 0.5, "Entropy Sensitivity\nBar Plot",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("a) Gene Entropy Sensitivity", loc="left")

    ax = axes[0, 1]
    ax.text(0.5, 0.5, "Hotspot Gene\nNetwork Topology",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("b) Entropy Hotspot Genes", loc="left")

    ax = axes[1, 0]
    ax.text(0.5, 0.5, "CRISPR Validation\n(DepMap)",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("c) CRISPR Screening Validation", loc="left")

    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Case Study:\nTarget-Drug-Pathway",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title("d) Case Study: Druggable Targets", loc="left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
