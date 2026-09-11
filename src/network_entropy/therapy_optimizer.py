"""
therapy_optimizer.py — Target genes whose knockdown minimizes NetITH.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : baseline per-cell GRNs; baseline entropies; gene list
Outputs : gene-level entropy sensitivities; greedy combination search
Module  : src.network_entropy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix


def entropy_sensitivity(
    baseline_grns: np.ndarray,
    baseline_entropies: np.ndarray,
    gene_list: List[str],
    perturbation_magnitude: float = 0.5,
    max_genes: int = 100,
) -> Dict[str, float]:
    """Compute entropy sensitivity for each gene.

    Entropy sensitivity ΔS(g) = how much NetITH changes when gene g
    is "knocked down" across all cells.

    Strategy: For each candidate gene, simulate its knockdown by
    removing all edges incident to that gene in all cell GRNs,
    then recompute the distribution entropy.

    Args:
        baseline_grns: Array (n_cells, n_nodes, n_nodes) of adjacency matrices.
        baseline_entropies: Array (n_cells,) of baseline per-cell entropies.
        gene_list: Gene names corresponding to GRN nodes.
        perturbation_magnitude: Fraction of edge weight to remove.
        max_genes: Maximum number of genes to test (for efficiency).

    Returns:
        Dict mapping gene_name -> ΔNetITH (positive = NetITH decreases).
    """
    n_cells, n_nodes, _ = baseline_grns.shape

    # Only test genes that actually appear in GRNs (total in/out degree)
    gene_degrees = np.sum(np.abs(baseline_grns), axis=(0, 2)) + np.sum(
        np.abs(baseline_grns), axis=(0, 1)
    )
    active_genes = np.where(gene_degrees > 0)[0]

    if len(active_genes) > max_genes:
        # Select top genes by total degree
        active_genes = active_genes[
            np.argsort(gene_degrees[active_genes])[-max_genes:]
        ]

    from .netith import compute_netith
    from .von_neumann import von_neumann_entropy

    baseline_netith = compute_netith(baseline_entropies)["netith"]

    sensitivities = {}
    for gene_idx in active_genes:
        gene_name = gene_list[gene_idx]

        # Simulate knockdown: reduce edges incident to this gene
        perturbed_grns = baseline_grns.copy()
        perturbed_grns[:, gene_idx, :] *= (1.0 - perturbation_magnitude)
        perturbed_grns[:, :, gene_idx] *= (1.0 - perturbation_magnitude)

        # Recompute per-cell entropies
        perturbed_entropies = np.array(
            [
                von_neumann_entropy(perturbed_grns[i])
                for i in range(n_cells)
            ]
        )

        # Recompute NetITH
        perturbed_netith = compute_netith(perturbed_entropies)["netith"]

        # ΔNetITH: positive means NetITH decreased (good)
        delta = baseline_netith - perturbed_netith
        sensitivities[gene_name] = float(delta)

    return sensitivities


def find_entropy_hotspots(
    sensitivities: Dict[str, float],
    top_k: int = 20,
    druggable_only: bool = False,
    druggable_genes: Optional[set] = None,
) -> List[Tuple[str, float]]:
    """Identify top "entropy hotspot" genes for therapeutic targeting.

    Args:
        sensitivities: Gene → ΔNetITH mapping from entropy_sensitivity().
        top_k: Number of top genes to return.
        druggable_only: If True, filter to known druggable targets.
        druggable_genes: Set of druggable gene symbols.

    Returns:
        List of (gene_name, delta_netith) sorted descending by delta.
    """
    if druggable_only and druggable_genes is not None:
        sensitivities = {
            g: v for g, v in sensitivities.items() if g in druggable_genes
        }

    sorted_genes = sorted(
        sensitivities.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_genes[:top_k]


def greedy_combo_search(
    sensitivities: Dict[str, float],
    baseline_grns: np.ndarray,
    baseline_entropies: np.ndarray,
    gene_list: List[str],
    max_combo_size: int = 3,
    perturbation_magnitude: float = 0.5,
) -> List[Tuple[List[str], float]]:
    """Greedy search for optimal combination of target genes.

    At each step, adds the gene that provides the largest additional
    reduction in NetITH beyond the currently selected set.

    Args:
        sensitivities: Single-gene entropy sensitivities.
        baseline_grns: Array (n_cells, n_nodes, n_nodes).
        baseline_entropies: Array (n_cells,).
        gene_list: Gene names.
        max_combo_size: Maximum combination size.
        perturbation_magnitude: Edge weight reduction fraction.

    Returns:
        List of (combo_genes, net_reduction) for each combo size.
    """
    from .netith import compute_netith
    from .von_neumann import von_neumann_entropy

    n_cells, n_nodes, _ = baseline_grns.shape
    baseline_netith = compute_netith(baseline_entropies)["netith"]

    # Pre-compute gene indices
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Start with empty set
    selected_genes = []
    selected_indices = []
    current_netith = baseline_netith
    results = []

    candidates = set(sensitivities.keys())

    for step in range(max_combo_size):
        best_gene = None
        best_delta = -float("inf")

        for gene in candidates - set(selected_genes):
            if gene not in gene_to_idx:
                continue

            test_indices = selected_indices + [gene_to_idx[gene]]

            # Simulate combined perturbation
            perturbed_grns = baseline_grns.copy()
            for gidx in test_indices:
                perturbed_grns[:, gidx, :] *= (1.0 - perturbation_magnitude)
                perturbed_grns[:, :, gidx] *= (1.0 - perturbation_magnitude)

            perturbed_entropies = np.array(
                [
                    von_neumann_entropy(perturbed_grns[i])
                    for i in range(n_cells)
                ]
            )
            new_netith = compute_netith(perturbed_entropies)["netith"]
            delta = current_netith - new_netith

            if delta > best_delta:
                best_delta = delta
                best_gene = gene

        if best_gene is None or best_delta <= 0:
            break

        selected_genes.append(best_gene)
        selected_indices.append(gene_to_idx[best_gene])
        current_netith -= best_delta
        results.append((list(selected_genes), baseline_netith - current_netith))

    return results
