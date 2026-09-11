"""
test_von_neumann.py — Unit tests for von Neumann entropy computation.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : None (synthetic test graphs)
Outputs : pytest assertions
Module  : tests
"""

import numpy as np
import pytest
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.network_entropy.von_neumann import (
    laplacian_matrix,
    von_neumann_entropy,
    von_neumann_entropy_from_graph,
    batch_von_neumann_entropy,
    normalized_von_neumann_entropy,
)


class TestLaplacianMatrix:
    """Tests for Laplacian matrix construction."""

    def test_unnormalized_empty(self):
        adj = np.zeros((3, 3))
        L = laplacian_matrix(adj, normalize=False)
        assert L.shape == (3, 3)
        np.testing.assert_array_equal(L, np.zeros((3, 3)))

    def test_unnormalized_simple(self):
        # Simple 2-node graph with one edge
        adj = np.array([[0, 1], [1, 0]])
        L = laplacian_matrix(adj, normalize=False)
        expected = np.array([[1, -1], [-1, 1]])
        np.testing.assert_array_equal(L, expected)

    def test_unnormalized_weighted(self):
        adj = np.array([[0, 0.5], [0.5, 0]])
        L = laplacian_matrix(adj, normalize=False)
        expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_array_equal(L, expected)

    def test_normalized_simple(self):
        adj = np.array([[0, 1], [1, 0]])
        L = laplacian_matrix(adj, normalize=True)
        # Both nodes have degree 1, so D^{-1/2} = I
        expected = np.array([[1, -1], [-1, 1]])
        np.testing.assert_array_almost_equal(L, expected)

    def test_normalized_isolated_nodes(self):
        # One isolated node
        adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        L = laplacian_matrix(adj, normalize=True)
        # Node 0,1: degree 1, Node 2: degree 0
        assert np.all(np.isfinite(L))


class TestVonNeumannEntropy:
    """Tests for von Neumann entropy computation."""

    def test_empty_graph(self):
        adj = np.zeros((5, 5))
        S = von_neumann_entropy(adj)
        assert S == 0.0

    def test_complete_graph(self):
        # Complete graph K_n has von Neumann entropy = log2(n-1) for
        # unnormalized Laplacian (approximate)
        n = 4
        adj = np.ones((n, n)) - np.eye(n)
        S = von_neumann_entropy(adj, normalize_laplacian=False)
        assert S > 0.0
        # Should be less than log2(n) (maximum possible)
        assert S < np.log2(n)

    def test_single_edge(self):
        adj = np.array([[0, 1], [1, 0]])
        S = von_neumann_entropy(adj)
        assert S >= 0.0

    def test_monotonicity_sparsity(self):
        """Sparser graphs should have different entropy than denser ones."""
        n = 10
        # Sparse: ring
        adj_sparse = np.zeros((n, n))
        for i in range(n):
            adj_sparse[i, (i + 1) % n] = 1
            adj_sparse[(i + 1) % n, i] = 1

        # Dense: complete graph
        adj_dense = np.ones((n, n)) - np.eye(n)

        S_sparse = von_neumann_entropy(adj_sparse, normalize_laplacian=False)
        S_dense = von_neumann_entropy(adj_dense, normalize_laplacian=False)

        # They should differ
        assert S_sparse != S_dense

    def test_isolated_nodes_zero_entropy(self):
        """Graph with all isolated nodes should have zero entropy."""
        adj = np.zeros((10, 10))
        S = von_neumann_entropy(adj)
        assert S == 0.0

    def test_symmetry_robustness(self):
        """Von Neumann entropy should handle asymmetric adjacency."""
        # Directed graph (asymmetric)
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        S = von_neumann_entropy(adj, normalize_laplacian=False)
        assert np.isfinite(S)
        assert S >= 0.0

    def test_from_networkx_graph(self):
        G = nx.erdos_renyi_graph(20, 0.2, seed=42)
        S = von_neumann_entropy_from_graph(G)
        assert S > 0.0

    def test_directed_graph_conversion(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        S = von_neumann_entropy_from_graph(G)
        assert np.isfinite(S)


class TestBatchVonNeumann:
    """Tests for batch computation."""

    def test_batch_single(self):
        adj_batch = np.zeros((3, 5, 5))
        for i in range(3):
            adj_batch[i] = np.eye(5)  # all isolated
        entropies = batch_von_neumann_entropy(adj_batch)
        assert entropies.shape == (3,)
        np.testing.assert_array_equal(entropies, np.zeros(3))

    def test_batch_parallel(self):
        """Test parallel batch computation."""
        np.random.seed(42)
        adj_batch = np.random.rand(5, 8, 8)
        # Make symmetric
        adj_batch = (adj_batch + adj_batch.transpose(0, 2, 1)) / 2

        seq = batch_von_neumann_entropy(adj_batch, n_jobs=1)
        par = batch_von_neumann_entropy(adj_batch, n_jobs=2)

        np.testing.assert_array_almost_equal(seq, par)

    def test_batch_with_nans(self):
        """Batch should handle graphs with isolated nodes."""
        adj_batch = np.zeros((3, 10, 10))
        for i in range(3):
            adj_batch[i, 0, 1] = adj_batch[i, 1, 0] = 1.0
        entropies = batch_von_neumann_entropy(adj_batch)
        assert np.all(np.isfinite(entropies))


class TestNormalizedEntropy:
    """Tests for normalized von Neumann entropy."""

    def test_range(self):
        """Normalized entropy should be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(5, 20)
            adj = np.random.rand(n, n)
            adj = (adj + adj.T) / 2
            S_norm = normalized_von_neumann_entropy(adj)
            assert 0.0 <= S_norm <= 1.0

    def test_single_node(self):
        adj = np.zeros((1, 1))
        S_norm = normalized_von_neumann_entropy(adj)
        assert S_norm == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
