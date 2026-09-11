"""
simulate.py — Simulated GRN populations with controlled heterogeneity.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : simulation hyperparameters (n_cells, n_genes, heterogeneity level)
Outputs : per-cell GRN adjacency matrices; ground-truth heterogeneity scores
Module  : src.network_entropy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ─── GRN Archetype Generation ─────────────────────────────────

def generate_grn_archetype(
    n_genes: int,
    n_tfs: int,
    target_ratio: float,
    weight_range: Tuple[float, float] = (0.3, 1.0),
    seed: int = 42,
) -> np.ndarray:
    """Generate a single GRN archetype.

    Each TF regulates a random subset of target genes with random weights.
    Different archetypes use different seeds → different edge patterns.

    Args:
        n_genes: Total genes.
        n_tfs: Number of transcription factors.
        target_ratio: Fraction of genes each TF targets on average.
        weight_range: (min, max) edge weight.
        seed: Random seed for reproducibility.

    Returns:
        (n_genes, n_genes) adjacency matrix.
    """
    rng = np.random.RandomState(seed)
    grn = np.zeros((n_genes, n_genes))

    for tf in range(n_tfs):
        n_targets = max(2, int(n_genes * target_ratio))
        targets = rng.choice(n_genes, size=n_targets, replace=False)
        for target in targets:
            grn[tf, target] = rng.uniform(*weight_range)

    return grn


def generate_archetype_set(
    n_archetypes: int,
    n_genes: int = 80,
    n_tfs: int = 20,
    base_target_ratio: float = 0.1,
    base_seed: int = 42,
) -> List[np.ndarray]:
    """Generate a set of structurally distinct GRN archetypes.

    Each archetype varies in:
      - Which genes are TFs (cycling through gene indices)
      - Target gene ratio (0.05 to 0.25)
      - Edge weight distribution

    Args:
        n_archetypes: Number of archetypes to generate.
        n_genes: Genes per GRN.
        n_tfs: TFs per GRN.
        base_target_ratio: Baseline target ratio.
        base_seed: Master seed.

    Returns:
        List of (n_genes, n_genes) adjacency matrices.
    """
    archetypes = []

    for k in range(n_archetypes):
        # Vary target ratio: 0.5x to 2.5x of base
        ratio = base_target_ratio * (0.5 + 2.0 * k / max(n_archetypes - 1, 1))
        ratio = np.clip(ratio, 0.02, 0.30)

        # Vary TF set: cycle through genes
        tf_start = (k * (n_genes // max(n_archetypes, 1))) % max(n_genes - n_tfs, 1)
        tf_genes = list(range(tf_start, tf_start + n_tfs))

        # Build GRN with shifted TF indices
        seed = base_seed + k * 137
        rng = np.random.RandomState(seed)
        grn = np.zeros((n_genes, n_genes))

        for tf_local_idx in range(n_tfs):
            tf_gene = tf_genes[tf_local_idx]
            n_targets = max(2, int(n_genes * ratio))
            targets = rng.choice(n_genes, size=n_targets, replace=False)
            # Weight range also varies
            w_min = 0.2 + 0.2 * (k / max(n_archetypes - 1, 1))
            w_max = 0.6 + 0.4 * (k / max(n_archetypes - 1, 1))
            for target in targets:
                grn[tf_gene, target] = rng.uniform(w_min, w_max)

        archetypes.append(grn)

    return archetypes


# ─── Population Simulation ────────────────────────────────────

def simulate_grn_population(
    n_cells: int = 500,
    n_genes: int = 80,
    heterogeneity_level: float = 0.3,
    base_seed: int = 42,
    per_cell_noise: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a population of cell GRNs with controlled heterogeneity.

    Cells are assigned to GRN archetypes. The number and spread of
    archetypes is controlled by heterogeneity_level.

    H=0.0: 1 archetype, all cells share same GRN.
    H=0.5: ~5 archetypes, cells split among them.
    H=1.0: ~10 archetypes, cells uniform across them.

    Args:
        n_cells: Number of cells.
        n_genes: Genes per GRN.
        heterogeneity_level: 0.0 (identical) to 1.0 (max variation).
        base_seed: Master random seed.
        per_cell_noise: Std dev of per-cell edge weight noise.

    Returns:
        adj_matrices: (n_cells, n_genes, n_genes).
        true_heterogeneity: Per-cell ground-truth deviation score.
    """
    n_tfs = max(3, n_genes // 4)

    # Number of archetypes scales with heterogeneity
    n_archetypes = max(1, int(1 + heterogeneity_level * 9))  # 1 to 10
    n_archetypes = min(n_archetypes, n_cells)

    archetypes = generate_archetype_set(
        n_archetypes=n_archetypes,
        n_genes=n_genes,
        n_tfs=n_tfs,
        base_target_ratio=0.10,
        base_seed=base_seed,
    )

    # Cell-to-archetype assignment
    # Low H → most cells in archetype 0
    # High H → uniform distribution
    rng = np.random.RandomState(base_seed + 1000)

    if n_archetypes == 1:
        assignments = np.zeros(n_cells, dtype=int)
    else:
        # Dirichlet-distributed proportions: concentration = 1/H
        # Low H → skewed, High H → uniform
        alpha = np.ones(n_archetypes) * (1.0 / max(heterogeneity_level, 0.05))
        proportions = rng.dirichlet(alpha)
        # Adjust: make more uniform as H increases
        uniform = np.ones(n_archetypes) / n_archetypes
        proportions = (1 - heterogeneity_level) * proportions + heterogeneity_level * uniform
        proportions = proportions / proportions.sum()

        cumsum = np.cumsum(proportions)
        assignments = np.searchsorted(cumsum, rng.rand(n_cells))

    # Generate per-cell GRNs and true heterogeneity scores
    adj_matrices = np.zeros((n_cells, n_genes, n_genes))
    true_scores = np.zeros(n_cells)

    # Baseline is archetype 0 (the "normal" GRN)
    baseline = archetypes[0]
    baseline_norm = np.linalg.norm(baseline, ord="fro")
    if baseline_norm < 1e-10:
        baseline_norm = 1.0

    for c in range(n_cells):
        archetype_idx = assignments[c]
        grn = archetypes[archetype_idx].copy()

        # Per-cell noise: scale with heterogeneity
        # At H=0 (single archetype), no noise — all cells identical
        effective_noise = per_cell_noise * heterogeneity_level

        # Edge weight noise
        if effective_noise > 1e-10:
            mask = grn > 0
            noise = rng.normal(0, effective_noise, size=grn.shape)
            grn[mask] = np.clip(grn[mask] + noise[mask], 0.05, 1.0)

            # Small topological noise
            n_edge_changes = rng.poisson(effective_noise * 5)
            for _ in range(n_edge_changes):
                tf = rng.randint(0, n_tfs)
                target = rng.randint(0, n_genes)
                if grn[tf, target] > 0:
                    grn[tf, target] = 0.0
                else:
                    grn[tf, target] = rng.uniform(0.1, 0.5)

        adj_matrices[c] = grn

        # True heterogeneity: Frobenius distance from baseline
        diff = grn - baseline
        true_scores[c] = float(np.linalg.norm(diff, ord="fro") / baseline_norm)

    return adj_matrices, true_scores


# ─── Heterogeneity Gradient ───────────────────────────────────

def simulate_heterogeneity_gradient(
    n_cells: int = 500,
    n_genes: int = 80,
    heterogeneity_range: Tuple[float, float] = (0.0, 1.0),
    n_levels: int = 6,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate datasets across a gradient of heterogeneity levels.

    Args:
        n_cells: Cells per level.
        n_genes: Genes per GRN.
        heterogeneity_range: (min, max) heterogeneity.
        n_levels: Number of levels.
        seed: Random seed.

    Returns:
        Dict:
          'adj_matrices': (n_levels, n_cells, n_genes, n_genes)
          'true_levels': (n_levels,) heterogeneity level values
          'true_scores': (n_levels, n_cells) per-cell ground truth
          'labels': (n_levels,) string labels
    """
    rng = np.random.RandomState(seed)
    levels = np.linspace(heterogeneity_range[0], heterogeneity_range[1],
                         n_levels)

    all_adjs = []
    all_true_levels = []
    all_true_scores = []

    for level in levels:
        adjs, true_scores = simulate_grn_population(
            n_cells=n_cells,
            n_genes=n_genes,
            heterogeneity_level=float(level),
            base_seed=rng.randint(0, 2**31 - 1),
        )
        all_adjs.append(adjs)
        all_true_levels.append(level)
        all_true_scores.append(true_scores)

    labels = [f"H={lvl:.2f}" for lvl in levels]

    return {
        "adj_matrices": np.array(all_adjs),
        "true_levels": np.array(all_true_levels),
        "true_scores": np.array(all_true_scores),
        "labels": labels,
    }


# ─── Validation Helpers ───────────────────────────────────────

def evaluate_netith_recovery(
    true_levels: np.ndarray,
    computed_netith: np.ndarray,
) -> Dict[str, float]:
    """Evaluate how well NetITH recovers ground-truth heterogeneity.

    Args:
        true_levels: (n_levels,) ground-truth heterogeneity.
        computed_netith: (n_levels,) NetITH scores.

    Returns:
        Dict with spearman_r, pearson_r, monotonicity.
    """
    from scipy.stats import spearmanr, pearsonr

    mask = np.isfinite(computed_netith)
    true_valid = true_levels[mask]
    netith_valid = computed_netith[mask]

    if len(true_valid) < 3:
        return {"spearman_r": np.nan, "pearson_r": np.nan,
                "monotonicity": np.nan}

    spear_r, spear_p = spearmanr(true_valid, netith_valid)
    pear_r, pear_p = pearsonr(true_valid, netith_valid)
    mono = _compute_monotonicity(true_valid, netith_valid)

    return {
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "monotonicity": float(mono),
    }


def _compute_monotonicity(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of pairwise comparisons where ordering matches."""
    n = len(x)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j]:
                continue
            total += 1
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                correct += 1
    return correct / total if total > 0 else 1.0


# ─── Quick Demo ───────────────────────────────────────────────

def quick_validation_demo(output_dir: Optional[str] = None):
    """Run a quick validation of NetEntropy on simulated data.

    Uses both entropy-based NetITH_E and distance-based NetITH_D/NetITH_P.
    The distance-based metrics are expected to track heterogeneity better.
    """
    from src.network_entropy.von_neumann import batch_von_neumann_entropy
    from src.network_entropy.netith import (
        compute_netith, compute_netith_distance, compute_pairwise_netith,
    )
    from scipy.stats import spearmanr
    import time

    print("=" * 60)
    print("NetEntropy Validation with Simulated Data")
    print("=" * 60)

    # Generate data
    print("\n[1] Generating simulated GRN populations...")
    t0 = time.time()
    sim_data = simulate_heterogeneity_gradient(
        n_cells=300, n_genes=80, n_levels=6, seed=42
    )
    print(f"    Done in {time.time() - t0:.1f}s")
    print(f"    Shape: {sim_data['adj_matrices'].shape}")
    print(f"    Levels: {sim_data['true_levels']}")

    # --- Method A: Entropy-based NetITH (unnormalized Laplacian) ---
    print("\n[2a] Entropy-based NetITH_E (per-cell VN entropy → distribution entropy)...")
    results_e = []
    for lvl_idx, level in enumerate(sim_data["true_levels"]):
        adjs = sim_data["adj_matrices"][lvl_idx]
        entropies = batch_von_neumann_entropy(adjs, normalize_laplacian=False)
        netith = compute_netith(entropies)
        results_e.append({
            "H_true": level, "netith": netith["netith"],
            "mean_S": netith["mean_entropy"],
            "std_S": float(np.std(entropies)),
        })
        print(f"    H={level:.2f}: NetITH_E={netith['netith']:.4f}, "
              f"mean_S={netith['mean_entropy']:.4f}, std_S={np.std(entropies):.4f}")

    # --- Method B: Distance-based NetITH_D (per-cell to consensus) ---
    print("\n[2b] Distance-based NetITH_D (per-cell distance to consensus GRN)...")
    results_d = []
    for lvl_idx, level in enumerate(sim_data["true_levels"]):
        adjs = sim_data["adj_matrices"][lvl_idx]
        netith_d = compute_netith_distance(adjs)
        results_d.append({"H_true": level, **netith_d})
        print(f"    H={level:.2f}: NetITH_D={netith_d['netith_d']:.4f}, "
              f"mean_dist={netith_d['mean_distance']:.4f}, "
              f"std_dist={netith_d['std_distance']:.4f}")

    # --- Method C: Pairwise-distance NetITH_P ---
    print("\n[2c] Pairwise-distance NetITH_P (random cell-pair GRN distances)...")
    results_p = []
    for lvl_idx, level in enumerate(sim_data["true_levels"]):
        adjs = sim_data["adj_matrices"][lvl_idx]
        netith_p = compute_pairwise_netith(adjs)
        results_p.append({"H_true": level, **netith_p})
        print(f"    H={level:.2f}: NetITH_P={netith_p['netith_p']:.4f}, "
              f"mean_pw_dist={netith_p['mean_pairwise_distance']:.4f}")

    # --- Evaluation ---
    print("\n[3] Evaluating recovery (Spearman r vs ground-truth H)...")
    true_levels = sim_data["true_levels"]

    netith_e_vals = np.array([r["netith"] for r in results_e])
    netith_d_vals = np.array([r["netith_d"] for r in results_d])
    netith_p_vals = np.array([r["netith_p"] for r in results_p])

    spear_e = spearmanr(true_levels, netith_e_vals)
    spear_d = spearmanr(true_levels, netith_d_vals)
    spear_p = spearmanr(true_levels, netith_p_vals)

    print(f"    NetITH_E (entropy-based):        r = {spear_e[0]:.4f} (p={spear_e[1]:.4f})")
    print(f"    NetITH_D (distance-to-consensus): r = {spear_d[0]:.4f} (p={spear_d[1]:.4f})")
    print(f"    NetITH_P (pairwise-distance):     r = {spear_p[0]:.4f} (p={spear_p[1]:.4f})")

    # Find best metric
    best = max(
        ("NetITH_E", spear_e[0]),
        ("NetITH_D", spear_d[0]),
        ("NetITH_P", spear_p[0]),
        key=lambda x: x[1],
    )

    print()
    if best[1] > 0.8:
        print(f"✅ {best[0]} strongly tracks ground-truth heterogeneity (r={best[1]:.4f})!")
    elif best[1] > 0.5:
        print(f"⚠️  {best[0]} moderately tracks heterogeneity (r={best[1]:.4f}).")
    else:
        print(f"❌ No metric tracks heterogeneity well. Best: {best[0]} (r={best[1]:.4f}).")

    return {
        "entropy": results_e,
        "distance": results_d,
        "pairwise": results_p,
    }, {"spearman_e": spear_e, "spearman_d": spear_d, "spearman_p": spear_p}


if __name__ == "__main__":
    quick_validation_demo()
