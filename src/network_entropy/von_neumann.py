"""
von_neumann.py — Von Neumann entropy of a graph from its Laplacian.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : adjacency matrix (N×N) or NetworkX graph
Outputs : von Neumann entropy in bits
Module  : src.network_entropy
"""

import numpy as np
from scipy.linalg import eigvalsh
from typing import Union, Optional
import networkx as nx


def laplacian_matrix(
    adj_matrix: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Compute the normalized Laplacian from adjacency matrix.

    Args:
        adj_matrix: Square adjacency matrix (N×N), weighted or unweighted.
        normalize: If True, use normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
                   If False, use unnormalized L = D - A.

    Returns:
        Laplacian matrix (N×N).
    """
    n = adj_matrix.shape[0]
    # Degree vector = row-sum of absolute edge weights
    degrees = np.sum(np.abs(adj_matrix), axis=1)
    D = np.diag(degrees)

    if normalize:
        # D^{-1/2}: zero degrees stay zero (avoid divide-by-zero)
        D_inv_sqrt = np.diag(
            np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        )
        L = np.eye(n) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    else:
        L = D - adj_matrix

    return L


def von_neumann_entropy(
    adj_matrix: np.ndarray,
    normalize_laplacian: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute von Neumann entropy of a graph.

    S_VN = -sum_i (λ_i * log2(λ_i))
    where λ_i are eigenvalues of ρ = L / Tr(L).

    Args:
        adj_matrix: Square adjacency matrix (N×N), weighted or unweighted.
        normalize_laplacian: Whether to normalize the Laplacian.
        eps: Small value to clip eigenvalues for numerical stability.

    Returns:
        von Neumann entropy (float, in bits).
    """
    L = laplacian_matrix(adj_matrix, normalize=normalize_laplacian)
    trace_L = np.trace(L)

    # Edge case: empty graph or all isolated nodes → zero entropy
    if trace_L < eps:
        return 0.0

    # Edge case: all nodes isolated with normalized Laplacian
    # (L = I, trace_L = n, but physically zero entropy)
    if normalize_laplacian and np.allclose(L, np.eye(L.shape[0]), atol=1e-10):
        n_isolated = np.sum(np.abs(adj_matrix).sum(axis=1) < eps)
        if n_isolated == L.shape[0]:
            return 0.0

    # Density matrix: trace-normalized Laplacian (unit trace)
    rho = L / trace_L

    # Compute eigenvalues (real symmetric matrix)
    eigenvalues = eigvalsh(rho)

    # Clip tiny negative values from numerical error
    eigenvalues = np.clip(eigenvalues, eps, 1.0)

    # Normalize eigenvalues to sum to 1
    eigenvalues = eigenvalues / np.sum(eigenvalues)

    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)


def von_neumann_entropy_from_graph(
    G: Union[nx.Graph, nx.DiGraph],
    weight_attr: Optional[str] = "weight",
    **kwargs,
) -> float:
    """Compute von Neumann entropy from a NetworkX graph.

    Args:
        G: NetworkX graph (directed or undirected).
        weight_attr: Edge attribute for weight (None for unweighted).
        **kwargs: Passed to von_neumann_entropy.

    Returns:
        von Neumann entropy.
    """
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected()

    adj = nx.to_numpy_array(G, weight=weight_attr)
    return von_neumann_entropy(adj, **kwargs)


def batch_von_neumann_entropy(
    adj_matrices: np.ndarray,
    normalize_laplacian: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute von Neumann entropy for a batch of adjacency matrices.

    Args:
        adj_matrices: Array of shape (n_cells, n_nodes, n_nodes).
        normalize_laplacian: Whether to normalize the Laplacian.
        n_jobs: Number of parallel jobs (1 = sequential, -1 = all cores).

    Returns:
        Array of entropy values, shape (n_cells,).
    """
    n_cells = adj_matrices.shape[0]
    entropies = np.zeros(n_cells)

    if n_jobs in (0, 1):
        for i in range(n_cells):
            entropies[i] = von_neumann_entropy(
                adj_matrices[i], normalize_laplacian=normalize_laplacian
            )
    else:
        from joblib import Parallel, delayed

        entropies = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(von_neumann_entropy)(
                    adj_matrices[i], normalize_laplacian=normalize_laplacian
                )
                for i in range(n_cells)
            )
        )

    return entropies


def normalized_von_neumann_entropy(
    adj_matrix: np.ndarray, **kwargs
) -> float:
    """Compute normalized von Neumann entropy (0 to 1).

    Normalized by log2(N) where N is the number of nodes.
    """
    S = von_neumann_entropy(adj_matrix, **kwargs)
    n = adj_matrix.shape[0]
    if n <= 1:
        return 0.0
    S_max = np.log2(n)
    return S / S_max if S_max > 0 else 0.0
