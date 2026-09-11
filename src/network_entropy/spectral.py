"""
spectral.py — Spectral entropy of the graph Laplacian spectrum.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : adjacency matrix (N×N)
Outputs : spectral entropy in bits
Module  : src.network_entropy
"""

import numpy as np
from scipy.linalg import eigvalsh
from .von_neumann import laplacian_matrix


def spectral_entropy(
    adj_matrix: np.ndarray,
    normalize_laplacian: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute spectral entropy of a graph.

    The spectral entropy is the Shannon entropy of the normalized
    Laplacian eigenvalues, reflecting how "flat" the spectrum is.

    Args:
        adj_matrix: Square adjacency matrix (N×N).
        normalize_laplacian: Whether to use normalized Laplacian.
        eps: Small value for numerical stability.

    Returns:
        Spectral entropy (float, in bits).
    """
    L = laplacian_matrix(adj_matrix, normalize=normalize_laplacian)
    # Diagonalize the symmetric Laplacian (real eigenvalues)
    eigenvalues = eigvalsh(L)
    # Clip tiny eigenvalues to eps so log2 stays finite
    eigenvalues = np.clip(np.abs(eigenvalues), eps, None)
    total = np.sum(eigenvalues)

    if total < eps:
        return 0.0

    # Trace-normalize the spectrum into a probability vector
    probs = eigenvalues / total
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def batch_spectral_entropy(
    adj_matrices: np.ndarray,
    normalize_laplacian: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute spectral entropy for a batch of adjacency matrices.

    Args:
        adj_matrices: Array of shape (n_cells, n_nodes, n_nodes).
        normalize_laplacian: Whether to normalize the Laplacian.
        n_jobs: Number of parallel jobs.

    Returns:
        Array of entropy values, shape (n_cells,).
    """
    n_cells = adj_matrices.shape[0]
    entropies = np.zeros(n_cells)

    if n_jobs in (0, 1):
        for i in range(n_cells):
            entropies[i] = spectral_entropy(
                adj_matrices[i], normalize_laplacian=normalize_laplacian
            )
    else:
        from joblib import Parallel, delayed

        entropies = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(spectral_entropy)(
                    adj_matrices[i], normalize_laplacian=normalize_laplacian
                )
                for i in range(n_cells)
            )
        )

    return entropies
