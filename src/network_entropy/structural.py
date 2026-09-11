"""
structural.py — Structural entropy from GRN community structure.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : adjacency matrix (N×N); optional community-detection kwargs
Outputs : structural entropy in bits
Module  : src.network_entropy
"""

import numpy as np
import networkx as nx
from typing import Optional, Union
from networkx.algorithms.community import louvain_communities


def detect_communities(
    G: nx.Graph, method: str = "louvain", **kwargs
) -> list:
    """Detect functional modules in a regulatory network.

    Args:
        G: NetworkX undirected graph.
        method: Community detection method ('louvain' or 'leiden').
        **kwargs: Passed to community detection function.

    Returns:
        List of sets, each set containing node indices in a community.
    """
    if method == "louvain":
        communities = louvain_communities(G, **kwargs)
    elif method == "leiden":
        try:
            from cdlib import algorithms
            communities = algorithms.leiden(G)
            communities = [set(c) for c in communities.communities]
        except ImportError:
            raise ImportError(
                "cdlib required for Leiden. Install: pip install cdlib"
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    return list(communities)


def structural_entropy_from_communities(
    communities: list, n_nodes: int
) -> float:
    """Compute structural entropy from community assignments.

    Args:
        communities: List of sets, each containing node indices in a community.
        n_nodes: Total number of nodes in the graph.

    Returns:
        Structural entropy (float, in bits).
    """
    if n_nodes <= 1:
        return 0.0

    entropy = 0.0
    for community in communities:
        # Community size fraction acts as the probability weight
        p = len(community) / n_nodes
        if p > 0:
            entropy -= p * np.log2(p)
    return float(entropy)


def structural_entropy(
    adj_matrix: np.ndarray,
    method: str = "louvain",
    weight_threshold: float = 0.0,
    **community_kwargs,
) -> float:
    """Compute structural entropy of a gene regulatory network.

    Args:
        adj_matrix: Square adjacency matrix (N×N), weighted.
        method: Community detection method ('louvain' or 'leiden').
        weight_threshold: Edges with |weight| below this are removed.
        **community_kwargs: Passed to detect_communities.

    Returns:
        Structural entropy (float, in bits).
    """
    # Build graph from adjacency matrix
    G = nx.Graph()
    n = adj_matrix.shape[0]

    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            w = adj_matrix[i, j]
            # Keep only edges above the (absolute) weight threshold
            if abs(w) > weight_threshold:
                G.add_edge(i, j, weight=abs(w))

    if G.number_of_edges() == 0:
        return 0.0

    communities = detect_communities(G, method=method, **community_kwargs)
    return structural_entropy_from_communities(communities, n)


def batch_structural_entropy(
    adj_matrices: np.ndarray,
    method: str = "louvain",
    weight_threshold: float = 0.0,
    n_jobs: int = 1,
    **community_kwargs,
) -> np.ndarray:
    """Compute structural entropy for a batch of adjacency matrices.

    Args:
        adj_matrices: Array of shape (n_cells, n_nodes, n_nodes).
        method: Community detection method.
        weight_threshold: Edge weight threshold.
        n_jobs: Number of parallel jobs.
        **community_kwargs: Passed to detect_communities.

    Returns:
        Array of entropy values, shape (n_cells,).
    """
    n_cells = adj_matrices.shape[0]
    entropies = np.zeros(n_cells)

    if n_jobs in (0, 1):
        for i in range(n_cells):
            entropies[i] = structural_entropy(
                adj_matrices[i],
                method=method,
                weight_threshold=weight_threshold,
                **community_kwargs,
            )
    else:
        from joblib import Parallel, delayed

        entropies = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(structural_entropy)(
                    adj_matrices[i],
                    method=method,
                    weight_threshold=weight_threshold,
                    **community_kwargs,
                )
                for i in range(n_cells)
            )
        )

    return entropies
