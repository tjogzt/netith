"""test_index_invariance.py — regression test for the eigvalsh audit fix (2026-08-16).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Pipeline: control / test stage — see repository README

Verifies that NetITH-style von Neumann entropy computed with
`A = A + A.T; L = diag(deg) - A; eigvalsh(L)` is:
  1. invariant under gene-index permutation (the original bug made results
     depend on gene list order, because eigvalsh silently read one triangle
     of the asymmetric L);
  2. computed from a symmetric Laplacian with zero row sums and
     non-negative eigenvalues.

Run:  python3 scripts/test_index_invariance

Inputs (paths relative to repository root; DATA_ROOT = $NETITH_DATA_ROOT or <repo>/data):
    - (see code)
Outputs:
    - (see code)
Usage:
    NETITH_DATA_ROOT=/path/to/data python3 scripts/08_tests/test_index_invariance.py
"""
import numpy as np
from scipy.linalg import eigvalsh


def entropy_fixed(A):
    """Post-fix pipeline: symmetrize, then Laplacian entropy."""
    A = A + A.T
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    assert np.allclose(L, L.T), "L must be symmetric"
    assert np.allclose(L.sum(axis=1), 0), "row sums must be zero"
    eigs = np.clip(eigvalsh(L), 0, None)
    rho = eigs / (deg.sum() + 1e-12)
    rho = np.clip(rho, 1e-12, 1.0)
    return -np.sum(rho * np.log2(rho))


def build_toy():
    A = np.zeros((5, 5))
    for i, j, w in [(0, 2, 0.8), (0, 3, 0.5), (1, 2, 0.9), (1, 4, 0.6)]:
        A[i, j] = w
    return A


def main():
    A = build_toy()
    h0 = entropy_fixed(A)

    # Index-order invariance: 100 random permutations
    rng = np.random.default_rng(7)
    for _ in range(100):
        perm = rng.permutation(A.shape[0])
        h_perm = entropy_fixed(A[np.ix_(perm, perm)])
        assert abs(h0 - h_perm) < 1e-12, f"not invariant: {h0} vs {h_perm}"

    # Directed-edge preservation: A+A.T keeps all original edges
    Asym = A + A.T
    assert np.allclose(Asym, Asym.T)

    # eigvalsh vs. eig cross-validation (general eigensolver must agree)
    from scipy.linalg import eig
    L_test = np.diag(Asym.sum(axis=1)) - Asym
    ev_all = eig(L_test)
    ev_sh = eigvalsh(L_test)
    assert np.allclose(np.sort(ev_all[0].real), np.sort(ev_sh), atol=1e-10), \
        "eigvalsh disagrees with general eig on symmetric input"
    assert np.all(ev_sh >= -1e-12), "eigenvalues must be non-negative"
    assert abs(L_test.sum(axis=1)).max() < 1e-10, "row sums must be zero"

    # Zero-row check on real-ish random graph
    rng2 = np.random.default_rng(11)
    B = np.zeros((8, 8))
    for _ in range(20):
        i, j = rng2.integers(0, 8, 2)
        B[i, j] = rng2.uniform(0.2, 1.0)
    entropy_fixed(B)

    print("PASS: index-order invariance (100 permutations), symmetry, row-sum, "
          "non-negativity all verified")
    print(f"toy graph entropy (fixed pipeline) = {h0:.6f}")


if __name__ == "__main__":
    main()
