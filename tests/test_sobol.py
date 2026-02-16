"""Tests for Sobol sensitivity analysis (zimmerman.sobol).

Tests verify:
    1. Saltelli sampling produces the correct number of samples: N*(2D+2).
    2. On a linear model y = sum(a_i * x_i), S1 values match the
       analytical formula: S1_i = a_i^2 * Var(x_i) / Var(y).
    3. S1 values sum to approximately 1 for an additive model.
    4. ST >= S1 for all parameters (total-order includes interactions).
    5. Interaction indices (ST - S1) are near zero for an additive model.
"""

import numpy as np
import pytest

from zimmerman.sobol import saltelli_sample, rescale_samples, sobol_indices, sobol_sensitivity


class TestSaltelliSample:
    """Tests for the saltelli_sample function."""

    def test_sample_count(self):
        """Saltelli sampling produces exactly N*(D+2) samples."""
        n_base = 64
        d = 5
        rng = np.random.default_rng(42)
        samples = saltelli_sample(n_base, d, rng)
        expected_rows = n_base * (d + 2)
        assert samples.shape == (expected_rows, d)

    def test_sample_count_small(self):
        """Correct count for small N and D."""
        n_base = 8
        d = 2
        rng = np.random.default_rng(0)
        samples = saltelli_sample(n_base, d, rng)
        # N*(D+2) = 8*(2+2) = 32
        assert samples.shape == (32, 2)

    def test_samples_in_unit_cube(self):
        """All samples should be in [0, 1]^d."""
        n_base = 100
        d = 4
        rng = np.random.default_rng(123)
        samples = saltelli_sample(n_base, d, rng)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_count_single_param(self):
        """Works correctly with a single parameter (D=1)."""
        n_base = 32
        d = 1
        rng = np.random.default_rng(7)
        samples = saltelli_sample(n_base, d, rng)
        # N*(1+2) = 32*3 = 96
        assert samples.shape == (96, 1)


class TestRescaleSamples:
    """Tests for the rescale_samples function."""

    def test_rescale_identity(self):
        """Bounds [0, 1] should leave samples unchanged."""
        samples = np.array([[0.0, 0.5], [1.0, 0.25]])
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        rescaled = rescale_samples(samples, bounds)
        np.testing.assert_array_almost_equal(rescaled, samples)

    def test_rescale_shift(self):
        """Bounds [a, b] should shift and scale correctly."""
        samples = np.array([[0.0, 0.5], [1.0, 1.0]])
        bounds = np.array([[10.0, 20.0], [-5.0, 5.0]])
        rescaled = rescale_samples(samples, bounds)
        expected = np.array([[10.0, 0.0], [20.0, 5.0]])
        np.testing.assert_array_almost_equal(rescaled, expected)


class TestSobolIndices:
    """Tests for the sobol_indices function."""

    def test_constant_output_gives_zero_indices(self):
        """When output is constant, all indices should be zero."""
        n = 100
        d = 3
        y_A = np.ones(n)
        y_B = np.ones(n)
        y_C = np.ones((d, n))
        S1, ST = sobol_indices(y_A, y_B, y_C)
        np.testing.assert_array_almost_equal(S1, np.zeros(d))
        np.testing.assert_array_almost_equal(ST, np.zeros(d))

    def test_st_geq_s1(self):
        """Total-order should be >= first-order for all parameters."""
        # Use the full sobol_sensitivity pipeline with a linear model
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=3, coefficients=np.array([1.0, 2.0, 3.0]))
        result = sobol_sensitivity(sim, n_base=1024, seed=42)
        for name in sim.param_spec().keys():
            s1 = result["y"]["S1"][name]
            st = result["y"]["ST"][name]
            # ST should be >= S1 (with some numerical tolerance)
            assert st >= s1 - 0.05, (
                f"ST ({st:.4f}) < S1 ({s1:.4f}) for {name}"
            )


class TestSobolSensitivityLinear:
    """Test Sobol analysis on a linear additive model.

    For y = sum(a_i * x_i) with x_i ~ Uniform(0, 1):
        Var(x_i) = 1/12
        Var(y) = sum(a_i^2 * 1/12)
        S1_i = a_i^2 / sum(a_j^2)
        sum(S1) = 1.0  (additive model, no interactions)
        ST_i = S1_i  (no interactions)
    """

    def test_s1_matches_analytical(self):
        """S1 values should match the analytical formula for linear model."""
        from tests.conftest import LinearSimulator
        coeffs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = LinearSimulator(d=4, coefficients=coeffs)

        # Analytical S1: a_i^2 / sum(a_j^2)
        analytical_s1 = coeffs ** 2 / np.sum(coeffs ** 2)

        # Run Sobol analysis with enough samples for accuracy
        result = sobol_sensitivity(sim, n_base=1024, seed=42)

        for i, name in enumerate(sim.param_spec().keys()):
            computed_s1 = result["y"]["S1"][name]
            expected_s1 = analytical_s1[i]
            assert abs(computed_s1 - expected_s1) < 0.08, (
                f"S1 for {name}: computed {computed_s1:.4f} vs "
                f"expected {expected_s1:.4f}"
            )

    def test_s1_sum_near_one(self):
        """For a purely additive model, sum(S1) should be close to 1.0."""
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=4)
        result = sobol_sensitivity(sim, n_base=1024, seed=42)

        s1_sum = sum(result["y"]["S1"].values())
        assert abs(s1_sum - 1.0) < 0.15, (
            f"sum(S1) = {s1_sum:.4f}, expected ~1.0"
        )

    def test_interaction_near_zero(self):
        """For an additive model, interaction indices should be near zero."""
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=4)
        result = sobol_sensitivity(sim, n_base=1024, seed=42)

        for name in sim.param_spec().keys():
            interaction = result["y"]["interaction"][name]
            assert abs(interaction) < 0.1, (
                f"Interaction for {name}: {interaction:.4f}, "
                f"expected near 0 for additive model"
            )

    def test_rankings_correct(self):
        """The most influential parameter should have the largest coefficient."""
        from tests.conftest import LinearSimulator
        coeffs = np.array([1.0, 2.0, 3.0, 10.0])
        sim = LinearSimulator(d=4, coefficients=coeffs)
        result = sobol_sensitivity(sim, n_base=512, seed=42)

        rankings = result["rankings"]["y_most_influential_S1"]
        # x3 (coefficient 10) should be ranked first
        assert rankings[0] == "x3", (
            f"Expected x3 as most influential, got {rankings[0]}"
        )


class TestSobolSensitivityMultiOutput:
    """Test Sobol analysis with multiple output keys."""

    def test_auto_detect_output_keys(self):
        """Should automatically detect all numeric output keys."""
        from tests.conftest import QuadraticSimulator
        sim = QuadraticSimulator(d=3)
        result = sobol_sensitivity(sim, n_base=64, seed=42)

        # QuadraticSimulator returns both "y" and "fitness"
        assert "y" in result["output_keys"]
        assert "fitness" in result["output_keys"]

    def test_specific_output_keys(self):
        """Should analyze only specified output keys."""
        from tests.conftest import QuadraticSimulator
        sim = QuadraticSimulator(d=3)
        result = sobol_sensitivity(sim, n_base=64, seed=42, output_keys=["y"])

        assert "y" in result
        assert "fitness" not in result
        assert result["output_keys"] == ["y"]
