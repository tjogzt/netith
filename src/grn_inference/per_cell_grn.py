"""
per_cell_grn.py — Build personalized per-cell gene regulatory networks.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : pySCENIC regulon template; cell × gene expression matrix
Outputs : per-cell TF→target adjacency matrices
Module  : src.grn_inference
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from typing import Dict, List, Optional, Tuple, Union
import warnings


def build_per_cell_grn(
    regulons: Dict[str, Dict],
    cell_expr: np.ndarray,
    gene_list: List[str],
    k_neighbors: int = 30,
    neighbor_indices: Optional[np.ndarray] = None,
    neighbor_expr: Optional[np.ndarray] = None,
    weight_quantile: float = 0.9,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Build a single cell's GRN adjacency matrix.

    Uses a "template + personalization" strategy:
    - Template: Global regulons from pySCENIC
    - Personalization: Cell-specific edge weights using local kNN context

    Args:
        regulons: Dict mapping TF name -> {
            'targets': List[str] of target gene names,
            'weights': Optional[List[float]] of global importance scores
        }
        cell_expr: Expression vector for this cell (n_genes,).
        gene_list: Full list of gene names.
        k_neighbors: Number of neighbors for local correlation estimation.
        neighbor_indices: Precomputed kNN indices for this cell (k,).
        neighbor_expr: Expression matrix of all cells (n_cells, n_genes).
        weight_quantile: Quantile threshold for edge retention (0-1).
        min_weight: Absolute minimum edge weight.

    Returns:
        Adjacency matrix (n_TFs+targets, n_TFs+targets) - directed.
        Only TF→target edges are non-zero.
    """
    n_genes = len(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Collect all nodes that appear in regulons
    tf_set = set(regulons.keys())
    all_targets = set()
    for tf, info in regulons.items():
        all_targets.update(info.get("targets", []))

    # Subset to genes present in data
    tf_in_data = tf_set & set(gene_list)
    targets_in_data = all_targets & set(gene_list)

    all_nodes = sorted(tf_in_data | targets_in_data)
    n_nodes = len(all_nodes)
    node_to_idx = {g: i for i, g in enumerate(all_nodes)}

    # Initialize adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))

    # For each TF, compute cell-specific edge weights to its targets
    for tf in tf_in_data:
        if tf not in regulons:
            continue
        tf_idx_global = gene_to_idx[tf]
        tf_idx_local = node_to_idx[tf]
        targets = [t for t in regulons[tf].get("targets", []) if t in gene_list]

        for target in targets:
            target_idx_global = gene_to_idx[target]
            target_idx_local = node_to_idx[target]

            # Compute cell-specific weight
            w = _compute_edge_weight(
                tf_idx_global,
                target_idx_global,
                cell_expr,
                k_neighbors=k_neighbors,
                neighbor_indices=neighbor_indices,
                neighbor_expr=neighbor_expr,
            )

            if abs(w) >= min_weight:
                adj_matrix[tf_idx_local, target_idx_local] = w

    # Apply quantile thresholding
    if weight_quantile > 0 and adj_matrix.max() > 0:
        threshold = np.quantile(
            adj_matrix[adj_matrix > 0], 1.0 - weight_quantile
        )
        adj_matrix[adj_matrix < threshold] = 0.0

    return adj_matrix


def _compute_edge_weight(
    tf_idx: int,
    target_idx: int,
    cell_expr: np.ndarray,
    k_neighbors: int = 30,
    neighbor_indices: Optional[np.ndarray] = None,
    neighbor_expr: Optional[np.ndarray] = None,
) -> float:
    """Compute cell-specific weight for a TF→target edge.

    Strategy: Use local Pearson correlation within the cell's
    k-nearest neighbor subpopulation as a proxy for co-expression.

    If neighbor data is not available, falls back to the product
    of normalized expression values (simplified).

    Args:
        tf_idx, target_idx: Gene indices in global gene space.
        cell_expr: This cell's expression.
        k_neighbors: k for local neighborhood.
        neighbor_indices: Precomputed kNN indices.
        neighbor_expr: Full expression matrix.

    Returns:
        Edge weight (float, typically in [-1, 1] for correlation).
    """
    if neighbor_expr is not None and neighbor_indices is not None:
        # Use local kNN correlation
        local_expr = neighbor_expr[neighbor_indices[:k_neighbors], :]
        local_expr = np.vstack([cell_expr.reshape(1, -1), local_expr])

        tf_vals = local_expr[:, tf_idx]
        target_vals = local_expr[:, target_idx]

        # Pearson correlation
        tf_std = np.std(tf_vals)
        target_std = np.std(target_vals)

        # Guard against zero variance (constant expression across neighbors)
        if tf_std < 1e-10 or target_std < 1e-10:
            return 0.0

        corr = np.corrcoef(tf_vals, target_vals)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    else:
        # Fallback: expression product (simplified, less accurate)
        w = float(cell_expr[tf_idx] * cell_expr[target_idx])
        return np.tanh(w)  # Bound to [-1, 1]


def batch_build_grns(
    regulons: Dict[str, Dict],
    expr_matrix: Union[np.ndarray, csr_matrix],
    gene_list: List[str],
    cell_indices: List[int],
    k_neighbors: int = 30,
    n_jobs: int = 1,
    **kwargs,
) -> List[np.ndarray]:
    """Build GRNs for multiple cells.

    Args:
        regulons: Global regulon template.
        expr_matrix: Expression matrix (n_cells, n_genes).
        gene_list: Gene names matching columns of expr_matrix.
        cell_indices: Which cells to process.
        k_neighbors: Number of neighbors.
        n_jobs: Parallel jobs.
        **kwargs: Passed to build_per_cell_grn.

    Returns:
        List of adjacency matrices, one per cell.
    """
    if issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()

    # Precompute kNN for all cells
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    nn.fit(expr_matrix)
    all_neighbor_indices = nn.kneighbors(expr_matrix, return_distance=False)

    grns = []
    for cell_idx in cell_indices:
        neighbor_idx = all_neighbor_indices[cell_idx, 1:]  # exclude self
        cell_grn = build_per_cell_grn(
            regulons=regulons,
            cell_expr=expr_matrix[cell_idx],
            gene_list=gene_list,
            k_neighbors=k_neighbors,
            neighbor_indices=neighbor_idx,
            neighbor_expr=expr_matrix,
            **kwargs,
        )
        grns.append(cell_grn)

    return grns
