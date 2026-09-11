#!/usr/bin/env python3
"""
01_preprocess_data.py — scRNA-seq data preprocessing pipeline.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : raw scRNA-seq (h5ad / 10x h5 / mtx / loom) from NETITH_DATA_ROOT
Outputs : processed AnnData (normalized, HVG, PCA, UMAP, Leiden)
Module  : scripts/00_data (data stage)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
ROOT = Path(__file__).resolve().parent.parent.parent
# Data root resolves from NETITH_DATA_ROOT or defaults to <repo>/data
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Configuration ────────────────────────────────────────────

# Quality control thresholds
QC_CONFIG = {
    "min_genes": 200,           # Min genes per cell
    "max_genes": 6000,          # Max genes per cell
    "min_cells": 3,             # Min cells per gene
    "max_mito_pct": 20.0,       # Max mitochondrial percentage
    "max_ribo_pct": 50.0,       # Max ribosomal percentage
}

# Highly variable gene selection
HVG_CONFIG = {
    "n_top_genes": 4000,
    "flavor": "seurat_v3",      # or 'seurat', 'cell_ranger'
    "batch_key": None,
}

# Normalization
NORM_CONFIG = {
    "target_sum": 1e4,
    "exclude_highly_expressed": True,
    "max_fraction": 0.05,
}

# Dimension reduction
PCA_CONFIG = {
    "n_comps": 50,
    "svd_solver": "arpack",
}

NEIGHBOR_CONFIG = {
    "n_neighbors": 15,
    "n_pcs": 30,
}

UMAP_CONFIG = {
    "min_dist": 0.3,
    "spread": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="scRNA-seq Preprocessing")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--cancer", type=str, default="UNKNOWN",
                        help="Cancer type label (e.g., LUAD)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "h5ad", "10x", "mtx", "loom"],
                        help="Input data format")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--n_top_genes", type=int, default=4000,
                        help="Number of highly variable genes")
    parser.add_argument("--mito_prefix", type=str, default="MT-",
                        help="Prefix for mitochondrial genes")
    parser.add_argument("--ribo_prefix", type=str, default="RPS,RPL",
                        help="Prefix for ribosomal genes (comma-separated)")
    parser.add_argument("--skip_doublet", action="store_true",
                        help="Skip doublet detection (faster)")
    parser.add_argument("--cell_type_key", type=str, default=None,
                        help="obs column for cell type annotations (if pre-annotated)")
    parser.add_argument("--sample_key", type=str, default=None,
                        help="obs column for sample/patient ID")
    parser.add_argument("--batch_key", type=str, default=None,
                        help="obs column for batch correction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


# ─── Data Loading ─────────────────────────────────────────────

def load_data(input_path: str, fmt: str = "auto") -> ad.AnnData:
    """Load scRNA-seq data from various formats.

    Args:
        input_path: Path to file or directory.
        fmt: Format hint. 'auto' tries to detect automatically.

    Returns:
        AnnData object.
    """
    input_path = os.path.abspath(input_path)

    # Auto-detect format
    if fmt == "auto":
        if os.path.isdir(input_path):
            # Check for 10x directory (has barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
            if any(f.endswith("mtx.gz") or f.endswith("mtx")
                   for f in os.listdir(input_path)[:20]):
                fmt = "10x"
            else:
                fmt = "mtx"
        elif input_path.endswith(".h5ad"):
            fmt = "h5ad"
        elif input_path.endswith(".h5"):
            fmt = "10x"
        elif input_path.endswith(".loom"):
            fmt = "loom"
        else:
            raise ValueError(f"Cannot auto-detect format for: {input_path}")

    print(f"  Loading data (format={fmt}) from: {input_path}")

    if fmt == "h5ad":
        adata = sc.read_h5ad(input_path)
    elif fmt == "10x":
        if input_path.endswith(".h5"):
            adata = sc.read_10x_h5(input_path)
        else:
            adata = sc.read_10x_mtx(input_path, var_names="gene_symbols",
                                     cache=True)
        adata.var_names_make_unique()
    elif fmt == "mtx":
        adata = sc.read_mtx(input_path)
    elif fmt == "loom":
        adata = sc.read_loom(input_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Ensure gene names are unique
    adata.var_names_make_unique()

    print(f"    Raw shape: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ─── Quality Control ──────────────────────────────────────────

def compute_qc_metrics(adata: ad.AnnData, mito_prefix: str = "MT-",
                       ribo_prefix: str = "RPS,RPL") -> None:
    """Compute QC metrics and add to adata.

    Args:
        adata: AnnData object.
        mito_prefix: Prefix for mitochondrial genes.
        ribo_prefix: Comma-separated prefixes for ribosomal genes.
    """
    # Mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)

    # Ribosomal genes
    ribo_prefixes = ribo_prefix.split(",")
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(ribo_prefixes))

    # Compute QC
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

    print(f"    Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"    Median UMIs/cell:  {adata.obs['total_counts'].median():.0f}")
    print(f"    Median %mito:      {adata.obs['pct_counts_mt'].median():.1f}%")


def filter_cells_and_genes(adata: ad.AnnData,
                           min_genes: int = 200,
                           max_genes: int = 6000,
                           min_cells: int = 3,
                           max_mito_pct: float = 20.0) -> ad.AnnData:
    """Filter low-quality cells and genes.

    Args:
        adata: AnnData with QC metrics.
        min_genes: Minimum genes per cell.
        max_genes: Maximum genes per cell.
        min_cells: Minimum cells per gene.
        max_mito_pct: Maximum mitochondrial percentage.

    Returns:
        Filtered AnnData.
    """
    n_before = adata.n_obs

    # Cell filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    adata = adata[adata.obs["pct_counts_mt"] < max_mito_pct, :].copy()

    n_after_cells = adata.n_obs
    print(f"    Cells: {n_before} → {n_after_cells} "
          f"({100*(n_before-n_after_cells)/n_before:.1f}% removed)")

    # Gene filter
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_genes_after = adata.n_vars
    print(f"    Genes: {n_genes_before} → {n_genes_after} "
          f"({100*(n_genes_before-n_genes_after)/n_genes_before:.1f}% removed)")

    return adata


# ─── Main Pipeline ────────────────────────────────────────────

def preprocess_pipeline(adata: ad.AnnData,
                        cancer_type: str,
                        n_top_genes: int = 4000,
                        mito_prefix: str = "MT-",
                        ribo_prefix: str = "RPS,RPL",
                        skip_doublet: bool = False,
                        batch_key: Optional[str] = None,
                        seed: int = 42) -> ad.AnnData:
    """Run full preprocessing pipeline.

    Steps:
    1. QC metric computation
    2. Filtering
    3. Doublet detection (optional)
    4. Normalization
    5. Log transformation
    6. Highly variable gene selection
    7. PCA
    8. Neighbors + UMAP
    9. Clustering (Leiden)

    Args:
        adata: Raw AnnData.
        cancer_type: Cancer type label.
        n_top_genes: Number of HVGs.
        mito_prefix: Mitochondrial gene prefix.
        ribo_prefix: Ribosomal gene prefix.
        skip_doublet: Skip doublet detection.
        batch_key: obs column for batch.
        seed: Random seed.

    Returns:
        Processed AnnData.
    """
    print(f"\n{'─'*50}")
    print(f"Preprocessing: {cancer_type}")
    print(f"{'─'*50}")

    # Store raw counts
    adata.raw = adata.copy()

    # 1. QC
    print("\n[1/7] Computing QC metrics...")
    compute_qc_metrics(adata, mito_prefix, ribo_prefix)

    # 2. Filtering
    print("\n[2/7] Filtering cells and genes...")
    adata = filter_cells_and_genes(adata, **QC_CONFIG)
    adata.obs["cancer_type"] = cancer_type

    # 3. Doublet detection (optional)
    if not skip_doublet and adata.n_obs < 50000:
        print("\n[3/7] Doublet detection (Scrublet)...")
        try:
            sc.external.pp.scrublet(adata, random_state=seed)
            n_doublets = adata.obs["predicted_doublet"].sum()
            print(f"    Predicted doublets: {n_doublets} "
                  f"({100*n_doublets/adata.n_obs:.1f}%)")
            # Don't remove here, just label
        except Exception as e:
            print(f"    Scrublet failed: {e}, skipping...")
    else:
        print("\n[3/7] Skipping doublet detection...")

    # 4. Normalization
    print("\n[4/7] Normalization and log transform...")
    sc.pp.normalize_total(adata, target_sum=NORM_CONFIG["target_sum"])
    sc.pp.log1p(adata)
    # Store normalized counts in .X, raw in .raw

    # 5. Highly variable genes
    print(f"\n[5/7] Selecting {n_top_genes} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                flavor=HVG_CONFIG["flavor"],
                                batch_key=batch_key)
    n_hvg = adata.var["highly_variable"].sum()
    print(f"    Selected {n_hvg} HVGs")
    adata = adata[:, adata.var["highly_variable"]].copy()

    # 6. PCA
    print("\n[6/7] PCA...")
    sc.tl.pca(adata, n_comps=PCA_CONFIG["n_comps"],
              svd_solver=PCA_CONFIG["svd_solver"],
              random_state=seed)

    # 7. Neighbors + UMAP + Clustering
    print("\n[7/7] Neighbors, UMAP, and clustering...")
    sc.pp.neighbors(adata, n_neighbors=NEIGHBOR_CONFIG["n_neighbors"],
                    n_pcs=NEIGHBOR_CONFIG["n_pcs"],
                    random_state=seed)
    sc.tl.umap(adata, min_dist=UMAP_CONFIG["min_dist"],
               spread=UMAP_CONFIG["spread"],
               random_state=seed)
    sc.tl.leiden(adata, random_state=seed)

    print(f"\n  Final: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ─── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.input:
        print("ERROR: Specify --input or use --help for usage.")
        sys.exit(1)

    # Load data
    adata = load_data(args.input, args.format)

    # Detect and preserve existing annotations
    if args.cell_type_key and args.cell_type_key in adata.obs.columns:
        print(f"  Found cell type annotations: {args.cell_type_key}")
    if args.sample_key and args.sample_key in adata.obs.columns:
        print(f"  Found sample IDs: {args.sample_key}")

    # Run pipeline
    adata = preprocess_pipeline(
        adata,
        cancer_type=args.cancer,
        n_top_genes=args.n_top_genes,
        skip_doublet=args.skip_doublet,
        batch_key=args.batch_key,
        seed=args.seed,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.cancer.lower()}_processed.h5ad"
    )
    adata.write(output_path)
    print(f"\n✅ Saved to: {output_path}")

    # Print summary stats
    print(f"\nSummary for {args.cancer}:")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    if "leiden" in adata.obs.columns:
        print(f"  Clusters: {adata.obs['leiden'].nunique()}")
    for col in ["cell_type", "Celltype", "celltype", "cluster", "sample_id",
                "patient", "tissue"]:
        if col in adata.obs.columns:
            print(f"  {col}: {adata.obs[col].nunique()} unique values")


if __name__ == "__main__":
    main()
