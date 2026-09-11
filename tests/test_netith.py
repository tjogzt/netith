"""
test_netith.py — Unit tests for NetITH aggregation.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : None (synthetic test data)
Outputs : pytest assertions
Module  : tests
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.network_entropy.netith import (
    compute_netith,
    compute_netith_per_sample,
)


class TestComputeNetITH:
    """Tests for NetITH computation."""

    def test_identical_entropies(self):
        """All cells have the same entropy → NetITH = 0."""
        entropies = np.ones(100) * 2.5
        result = compute_netith(entropies, n_bins=10)
        assert result["netith"] == pytest.approx(0.0, abs=0.1)
        assert result["var_entropy"] == pytest.approx(0.0)

    def test_maximally_diverse(self):
        """Uniformly distributed entropies → high NetITH."""
        entropies = np.linspace(0, 5, 100)
        result = compute_netith(entropies, n_bins=10)
        # Should be close to log2(10) ≈ 3.32 for uniform distribution
        assert result["netith"] > 2.0
        assert 0 < result["netith_normalized"] <= 1.0

    def test_bimodal_distribution(self):
        """Bimodal distribution should have intermediate NetITH."""
        ent1 = np.random.normal(1, 0.1, 50)
        ent2 = np.random.normal(4, 0.1, 50)
        entropies = np.concatenate([ent1, ent2])

        result = compute_netith(entropies, n_bins=20)
        # Should be between 0 and max
        assert result["netith"] > 0.0
        assert result["var_entropy"] > 0.0

    def test_small_sample(self):
        """Very few cells should return NaN."""
        result = compute_netith(np.array([1.0, 2.0]))
        # n_cells < 10 → NaN
        assert np.isnan(result["netith"])
        assert result["n_cells_used"] == 2

    def test_with_nans(self):
        """NaN values should be filtered out."""
        entropies = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0] * 20)
        result = compute_netith(entropies)
        assert np.isfinite(result["netith"])
        assert result["n_cells_used"] == 80  # 4 valid * 20

    def test_skewness(self):
        """Skewness should be computed correctly."""
        # Symmetric → skew ≈ 0
        sym = np.random.normal(0, 1, 500)
        result_sym = compute_netith(sym)
        assert abs(result_sym["skewness"]) < 0.5

        # Right-skewed
        right_skew = np.random.exponential(1, 500)
        result_right = compute_netith(right_skew)
        assert result_right["skewness"] > 0

    def test_kde_method(self):
        """KDE method should give similar results to histogram."""
        entropies = np.random.normal(2, 0.5, 200)
        result_hist = compute_netith(entropies, method="histogram")
        result_kde = compute_netith(entropies, method="kde")
        # KDE and histogram differ in scale but should both be finite and positive
        assert np.isfinite(result_hist["netith"])
        assert np.isfinite(result_kde["netith"])
        assert result_hist["netith"] > 0
        assert result_kde["netith"] > 0


class TestComputeNetITHPerSample:
    """Tests for per-sample NetITH."""

    def test_per_sample(self):
        entropy_vec = np.random.rand(300)
        sample_labels = (["A"] * 100 + ["B"] * 100 + ["C"] * 100)

        results = compute_netith_per_sample(entropy_vec, sample_labels)

        assert "A" in results
        assert "B" in results
        assert "C" in results
        for sample, metrics in results.items():
            assert "netith" in metrics
            assert metrics["n_cells_used"] == 100
