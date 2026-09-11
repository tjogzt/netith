"""
netith.py — NetITH aggregation: entropy-based and distance-based heterogeneity.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : per-cell entropy values or per-cell GRN adjacency matrices
Outputs : NetITH_E / NetITH_D / NetITH_P scores per sample or cancer type
Module  : src.network_entropy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import entropy as scipy_entropy


# ─── Entropy-based NetITH (original) ──────────────────────────

def compute_netith(
    cell_entropies: np.ndarray,
    n_bins: int = 20,
    bin_range: Optional[Tuple[float, float]] = None,
    method: str = "histogram",
) -> Dict[str, float]:
    """Compute entropy-based NetITH from per-cell entropy values.

    NetITH = H(p) where p is the histogram of per-cell entropy values.
    Higher NetITH → more diverse per-cell entropy → more heterogeneous.

    Args:
        cell_entropies: Array of per-cell network entropy values.
        n_bins: Histogram bins.
        bin_range: Optional (min, max) range.
        method: 'histogram' or 'kde'.

    Returns:
        Dict with netith, netith_normalized, mean_entropy, var_entropy,
        skewness, n_cells_used.
    """
    # Keep only finite, non-NaN entropy values
    valid = cell_entropies[
        np.isfinite(cell_entropies) & ~np.isnan(cell_entropies)
    ]
    n_valid = len(valid)

    if n_valid < 10:
        return {
            "netith": np.nan, "netith_normalized": np.nan,
            "mean_entropy": np.nan, "var_entropy": np.nan,
            "skewness": np.nan, "n_cells_used": n_valid,
        }

    mean_ent = float(np.mean(valid))
    var_ent = float(np.var(valid, ddof=1))
    skew = (float(np.mean((valid - mean_ent) ** 3) / (var_ent ** 1.5))
            if var_ent > 0 else 0.0)

    if method == "histogram":
        if bin_range is None:
            bin_range = (float(np.min(valid)), float(np.max(valid)))
        if bin_range[1] <= bin_range[0]:
            netith = 0.0
        else:
            counts, _ = np.histogram(valid, bins=n_bins, range=bin_range)
            # Pseudocount avoids log2(0) for empty bins
            counts = counts + 1e-10
            probs = counts / np.sum(counts)
            netith = float(scipy_entropy(probs, base=2))
    elif method == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(valid)
        x_grid = np.linspace(
            np.min(valid) - 0.1 * np.ptp(valid),
            np.max(valid) + 0.1 * np.ptp(valid),
            n_bins * 5,
        )
        pdf = kde(x_grid)
        pdf = np.clip(pdf, 1e-15, None)
        pdf = pdf / np.sum(pdf)
        netith = float(scipy_entropy(pdf, base=2))
    else:
        raise ValueError(f"Unknown method: {method}")

    netith_norm = netith / np.log2(n_bins) if n_bins > 1 else 0.0

    return {
        "netith": netith, "netith_normalized": netith_norm,
        "mean_entropy": mean_ent, "var_entropy": var_ent,
        "skewness": skew, "n_cells_used": n_valid,
    }


# ─── Distance-based NetITH (primary) ──────────────────────────

def compute_netith_distance(
    adj_matrices: np.ndarray,
    max_cells: int = 500,
    metric: str = "frobenius",
    seed: int = 42,
) -> Dict[str, float]:
    """Compute distance-based NetITH from per-cell GRN adjacency matrices.

    For each cell, computes its GRN distance to the population consensus
    GRN. Then the distribution spread of these distances is NetITH_D.

    NetITH_D = std(distances) / mean(distances)  (coefficient of variation)
    or
    NetITH_D = entropy of distance distribution

    Args:
        adj_matrices: (n_cells, n_genes, n_genes) adjacency matrices.
        max_cells: Maximum cells to use (subsample if more).
        metric: 'frobenius' or 'cosine'.
        seed: Random seed for subsampling.

    Returns:
        Dict with:
          - netith_d: Primary NetITH_D score (CV of distances)
          - netith_d_entropy: Distribution entropy of distances
          - mean_distance: Mean GRN distance from consensus
          - std_distance: Std of GRN distances
          - consensus_distance: Frobenius norm of consensus GRN
    """
    n_cells = adj_matrices.shape[0]

    # Subsample if many cells
    if n_cells > max_cells:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        adjs = adj_matrices[idx]
    else:
        adjs = adj_matrices

    n_cells_used = adjs.shape[0]
    n_genes = adjs.shape[1]

    # Compute consensus GRN (element-wise median)
    consensus = np.median(adjs, axis=0)

    # Compute per-cell distance to consensus
    distances = np.zeros(n_cells_used)
    for i in range(n_cells_used):
        diff = adjs[i] - consensus
        if metric == "frobenius":
            distances[i] = np.linalg.norm(diff, ord="fro")
        elif metric == "cosine":
            a = adjs[i].ravel()
            b = consensus.ravel()
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 1e-10 and norm_b > 1e-10:
                cos_sim = np.dot(a, b) / (norm_a * norm_b)
                distances[i] = 1.0 - cos_sim
            else:
                distances[i] = 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # NetITH_D_raw: standard deviation of distances (raw heterogeneity)
    mean_d = float(np.mean(distances))
    std_d = float(np.std(distances, ddof=1))

    # Also compute distribution entropy of distances
    if std_d > 1e-10:
        counts, _ = np.histogram(distances, bins=20)
        # Pseudocount avoids log2(0) for empty bins
        counts = counts + 1e-10
        probs = counts / np.sum(counts)
        netith_d_entropy = float(scipy_entropy(probs, base=2))
    else:
        netith_d_entropy = 0.0

    consensus_norm = float(np.linalg.norm(consensus, ord="fro"))

    return {
        "netith_d": std_d,   # std of distances: directly measures heterogeneity spread
        "netith_d_cv": float(std_d / mean_d) if mean_d > 1e-10 else 0.0,
        "netith_d_entropy": netith_d_entropy,
        "mean_distance": mean_d,
        "std_distance": std_d,
        "consensus_norm": consensus_norm,
        "n_cells_used": n_cells_used,
    }


def compute_pairwise_netith(
    adj_matrices: np.ndarray,
    max_cells: int = 200,
    metric: str = "frobenius",
    seed: int = 42,
) -> Dict[str, float]:
    """Compute pairwise-distance-based NetITH.

    Randomly samples cell pairs, computes GRN distance for each pair,
    then uses the distribution of pairwise distances as the heterogeneity score.

    NetITH_P = mean pairwise distance / consensus norm.

    This avoids needing a consensus GRN and directly captures
    how different cells are from each other.

    Args:
        adj_matrices: (n_cells, n_genes, n_genes).
        max_cells: Max cells to use.
        metric: 'frobenius' or 'cosine'.
        seed: Random seed.

    Returns:
        Dict with netith_p, mean_pairwise_distance, etc.
    """
    n_cells = adj_matrices.shape[0]

    # Subsample
    if n_cells > max_cells:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        adjs = adj_matrices[idx]
    else:
        adjs = adj_matrices

    n_used = adjs.shape[0]
    n_pairs = min(n_used * (n_used - 1) // 2, 5000)

    # Random pairs
    rng = np.random.RandomState(seed + 1)
    pairwise_dists = np.zeros(n_pairs)

    for p in range(n_pairs):
        i, j = rng.choice(n_used, size=2, replace=False)
        diff = adjs[i] - adjs[j]
        if metric == "frobenius":
            pairwise_dists[p] = np.linalg.norm(diff, ord="fro")
        elif metric == "cosine":
            a = adjs[i].ravel()
            b = adjs[j].ravel()
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                pairwise_dists[p] = 1.0 - np.dot(a, b) / (na * nb)
            else:
                pairwise_dists[p] = 0.0

    mean_pw = float(np.mean(pairwise_dists))
    std_pw = float(np.std(pairwise_dists, ddof=1))

    # Distribution entropy of pairwise distances
    if std_pw > 1e-10:
        counts, _ = np.histogram(pairwise_dists, bins=20)
        # Pseudocount avoids log2(0) for empty bins
        counts = counts + 1e-10
        probs = counts / np.sum(counts)
        pw_entropy = float(scipy_entropy(probs, base=2))
    else:
        pw_entropy = 0.0

    netith_p = float(mean_pw)  # Raw mean pairwise distance = heterogeneity score

    return {
        "netith_p": netith_p,
        "netith_p_entropy": pw_entropy,
        "mean_pairwise_distance": mean_pw,
        "std_pairwise_distance": std_pw,
        "n_pairs": n_pairs,
        "n_cells_used": n_used,
    }


# ─── Per-sample / Pan-cancer ──────────────────────────────────

def compute_netith_per_sample(
    entropy_matrix: np.ndarray,
    sample_labels: List[str],
    n_bins: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Compute NetITH for each sample separately.

    Args:
        entropy_matrix: (n_cells,) or (n_cells, n_types).
        sample_labels: Sample IDs per cell.
        n_bins: Histogram bins.

    Returns:
        sample_id -> NetITH metrics.
    """
    if entropy_matrix.ndim == 1:
        entropy_matrix = entropy_matrix.reshape(-1, 1)

    unique_samples = sorted(set(sample_labels))
    results = {}

    for sample in unique_samples:
        mask = np.array([s == sample for s in sample_labels])
        sample_entropies = entropy_matrix[mask, 0]
        results[sample] = compute_netith(sample_entropies, n_bins=n_bins)

    return results


def compute_netith_pan_cancer(
    entropy_dict: Dict[str, np.ndarray],
    sample_metadata: Dict[str, Dict[str, str]],
    n_bins: int = 20,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute NetITH across multiple cancer types.

    Args:
        entropy_dict: cancer_type -> entropy array (n_cells,).
        sample_metadata: cancer_type -> {sample_id: ...}.
        n_bins: Histogram bins.

    Returns:
        cancer_type -> sample_id -> NetITH metrics.
    """
    results = {}
    for cancer_type, entropies in entropy_dict.items():
        metadata = sample_metadata.get(cancer_type, {})
        results[cancer_type] = compute_netith_per_sample(
            entropies, list(metadata.keys()), n_bins=n_bins
        )
    return results
