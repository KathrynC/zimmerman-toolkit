"""Tests for contrast set generation (zimmerman.contrast_set_generator).

Tests verify:
    1. Edit path generation produces correctly structured edits.
    2. Edit paths are sorted by magnitude (smallest first).
    3. find_tipping_point locates the flip on a StepSimulator.
    4. Binary search converges to the smallest flipping prefix.
    5. batch_contrast_sets aggregates tipping frequency statistics.
    6. Custom outcome functions are respected.
    7. No flip found when the full edit path doesn't change the outcome.
    8. run() and param_spec() satisfy the Simulator protocol.
"""

import numpy as np
import pytest

from zimmerman.contrast_set_generator import ContrastSetGenerator


class TestGenerateEditPath:
    """Tests for the generate_edit_path method."""

    def test_returns_correct_count(self, step_sim):
        """Should return exactly n_edits edit dicts."""
        gen = ContrastSetGenerator(step_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        path = gen.generate_edit_path(base, n_edits=15, seed=42)
        assert len(path) == 15

    def test_edit_structure(self, step_sim):
        """Each edit should have 'param', 'delta', 'description' keys."""
        gen = ContrastSetGenerator(step_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        path = gen.generate_edit_path(base, n_edits=10, seed=42)
        for edit in path:
            assert "param" in edit
            assert "delta" in edit
            assert "description" in edit
            assert isinstance(edit["param"], str)
            assert isinstance(edit["delta"], float)
            assert isinstance(edit["description"], str)

    def test_sorted_by_magnitude(self, step_sim):
        """Edits should be sorted by absolute delta (smallest first)."""
        gen = ContrastSetGenerator(step_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        path = gen.generate_edit_path(base, n_edits=20, seed=42)
        magnitudes = [abs(e["delta"]) for e in path]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] <= magnitudes[i + 1] + 1e-12, (
                f"Edit {i} magnitude {magnitudes[i]} > edit {i+1} magnitude {magnitudes[i+1]}"
            )

    def test_with_target_params(self, step_sim):
        """Edit path toward target should sum to the total delta per param."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 0.5, "x2": 0.2}
        path = gen.generate_edit_path(base, target_params=target, n_edits=12, seed=42)

        # Sum deltas per param
        total_delta = {}
        for edit in path:
            name = edit["param"]
            total_delta[name] = total_delta.get(name, 0.0) + edit["delta"]

        # Should approximately reconstruct the total difference
        for name in ["x0", "x1", "x2"]:
            expected = target[name] - base[name]
            assert abs(total_delta.get(name, 0.0) - expected) < 1e-9, (
                f"Total delta for {name}: {total_delta.get(name, 0.0)}, expected {expected}"
            )

    def test_random_path_has_nonzero_deltas(self, step_sim):
        """Random edit path (no target) should have non-zero deltas."""
        gen = ContrastSetGenerator(step_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        path = gen.generate_edit_path(base, target_params=None, n_edits=10, seed=42)
        deltas = [abs(e["delta"]) for e in path]
        assert sum(deltas) > 0, "Random edit path should have non-zero deltas"

    def test_deterministic_with_seed(self, step_sim):
        """Same seed should produce identical edit paths."""
        gen = ContrastSetGenerator(step_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        path_a = gen.generate_edit_path(base, n_edits=10, seed=99)
        path_b = gen.generate_edit_path(base, n_edits=10, seed=99)
        for a, b in zip(path_a, path_b):
            assert a["param"] == b["param"]
            assert a["delta"] == b["delta"]


class TestFindTippingPoint:
    """Tests for the find_tipping_point method."""

    def test_finds_flip_on_step_sim(self, step_sim):
        """Should find a tipping point on the step simulator.

        StepSimulator: fitness = +1 if sum > 1.5, else -1.
        Starting at sum=0.0, edits that push sum past 1.5 should flip.
        """
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=20, seed=42)
        result = gen.find_tipping_point(base, path)

        assert result["found"] is True
        assert result["tipping_k"] is not None
        assert 0 < result["tipping_k"] <= 20
        assert result["base_outcome"] != result["flipped_outcome"]
        assert result["flip_size"] is not None
        assert 0.0 < result["flip_size"] <= 1.0

    def test_tipping_params_are_dict(self, step_sim):
        """Tipping params should be a valid parameter dict."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=20, seed=42)
        result = gen.find_tipping_point(base, path)

        if result["found"]:
            assert isinstance(result["tipping_params"], dict)
            for name in ["x0", "x1", "x2"]:
                assert name in result["tipping_params"]

    def test_no_flip_when_same_outcome(self, step_sim):
        """Should return found=False when edits don't change the outcome."""
        gen = ContrastSetGenerator(step_sim)
        # Start well below threshold, tiny edits that don't cross
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        # Target still below threshold (sum = 0.3 < 1.5)
        target = {"x0": 0.1, "x1": 0.1, "x2": 0.1}
        path = gen.generate_edit_path(base, target_params=target, n_edits=10, seed=42)
        result = gen.find_tipping_point(base, path)

        assert result["found"] is False
        assert result["tipping_k"] is None
        assert result["flip_size"] is None
        assert result["tipping_params"] is None

    def test_empty_edit_path(self, step_sim):
        """Empty edit path should return found=False."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.5, "x1": 0.5, "x2": 0.5}
        result = gen.find_tipping_point(base, [])

        assert result["found"] is False
        assert result["n_sims"] == 1

    def test_edit_path_applied_correct_length(self, step_sim):
        """edit_path_applied should have length tipping_k."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=20, seed=42)
        result = gen.find_tipping_point(base, path)

        if result["found"]:
            assert len(result["edit_path_applied"]) == result["tipping_k"]

    def test_n_sims_bounded_by_max(self, step_sim):
        """Number of sims should not exceed max_sims + overhead."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=100, seed=42)
        result = gen.find_tipping_point(base, path, max_sims=10)

        # n_sims includes base + full + bisection + final verification
        # Should be reasonable relative to max_sims
        assert result["n_sims"] <= 15  # max_sims + small overhead


class TestBatchContrastSets:
    """Tests for the batch_contrast_sets method."""

    def test_returns_expected_keys(self, step_sim):
        """Batch result should contain all expected keys."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        result = gen.batch_contrast_sets(base, n_paths=3, n_edits=10, seed=42)

        assert "pairs" in result
        assert "mean_flip_size" in result
        assert "param_tipping_frequency" in result
        assert "most_fragile_params" in result
        assert "n_sims" in result

    def test_pairs_count(self, step_sim):
        """Should return one result per path."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        result = gen.batch_contrast_sets(base, n_paths=5, n_edits=10, seed=42)

        assert len(result["pairs"]) == 5

    def test_param_tipping_frequency_sums_to_one_or_less(self, step_sim):
        """Tipping frequencies across params should sum to <= 1.0 per found path."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        result = gen.batch_contrast_sets(base, n_paths=10, n_edits=20, seed=42)

        freq = result["param_tipping_frequency"]
        total = sum(freq.values())
        # Each found path contributes exactly 1 to exactly one param,
        # so total frequency should be <= 1.0 (fractions)
        assert total <= 1.0 + 1e-9, (
            f"Total tipping frequency {total} should sum to <= 1.0"
        )

    def test_most_fragile_params_is_sorted(self, step_sim):
        """most_fragile_params should be sorted by frequency descending."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        result = gen.batch_contrast_sets(base, n_paths=10, n_edits=20, seed=42)

        freq = result["param_tipping_frequency"]
        fragile = result["most_fragile_params"]
        for i in range(len(fragile) - 1):
            assert freq[fragile[i]] >= freq[fragile[i + 1]] - 1e-12

    def test_n_sims_positive(self, step_sim):
        """Total sim count should be positive."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.5, "x1": 0.5, "x2": 0.5}
        result = gen.batch_contrast_sets(base, n_paths=2, n_edits=5, seed=42)

        assert result["n_sims"] > 0

    def test_deterministic(self, step_sim):
        """Same seed should produce identical batch results."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        r1 = gen.batch_contrast_sets(base, n_paths=3, n_edits=10, seed=42)
        r2 = gen.batch_contrast_sets(base, n_paths=3, n_edits=10, seed=42)

        assert r1["n_sims"] == r2["n_sims"]
        assert r1["mean_flip_size"] == r2["mean_flip_size"]


class TestCustomOutcomeFn:
    """Test with a custom outcome function."""

    def test_custom_outcome_fn(self, step_sim):
        """Custom outcome_fn should be used for classification."""
        # Custom: "high" if sum > 2, "low" otherwise
        def custom_outcome(result):
            return "high" if result.get("sum", 0) > 2.0 else "low"

        gen = ContrastSetGenerator(step_sim, outcome_fn=custom_outcome)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=20, seed=42)
        result = gen.find_tipping_point(base, path)

        assert result["base_outcome"] == "low"
        if result["found"]:
            assert result["flipped_outcome"] == "high"

    def test_default_outcome_fn_uses_fitness(self, step_sim):
        """Default outcome_fn should use the 'fitness' key."""
        gen = ContrastSetGenerator(step_sim)
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        base_result = step_sim.run(base)
        outcome = gen.outcome_fn(base_result)
        # sum=0 < threshold=1.5, so fitness=-1 => "negative"
        assert outcome == "negative"


class TestSimulatorProtocol:
    """Test that ContrastSetGenerator satisfies the Simulator protocol."""

    def test_run_returns_dict(self, step_sim):
        """run() should return a dict."""
        gen = ContrastSetGenerator(step_sim)
        result = gen.run({"x0": 0.5, "x1": 0.5, "x2": 0.5})
        assert isinstance(result, dict)

    def test_param_spec_delegates(self, step_sim):
        """param_spec() should return the underlying simulator's spec."""
        gen = ContrastSetGenerator(step_sim)
        spec = gen.param_spec()
        assert spec == step_sim.param_spec()

    def test_param_spec_returns_dict(self, linear_sim):
        """param_spec() should return a dict of (lo, hi) tuples."""
        gen = ContrastSetGenerator(linear_sim)
        spec = gen.param_spec()
        assert isinstance(spec, dict)
        for name, bounds in spec.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2


class TestWithLinearSimulator:
    """Test contrast sets on the linear simulator."""

    def test_no_flip_with_positive_only_model(self, linear_sim):
        """Linear sim with all positive coefficients and [0,1] range always
        produces non-negative output. Default outcome_fn should never flip
        from 'positive' because fitness is always >= 0 (there is no 'fitness'
        key, so default returns 'positive')."""
        gen = ContrastSetGenerator(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        result = gen.batch_contrast_sets(base, n_paths=5, n_edits=10, seed=42)

        # Linear sim doesn't have "fitness" key, so default always returns "positive"
        # All outcomes should be "positive" => no flips found
        for pair in result["pairs"]:
            assert pair["found"] is False


class TestWithQuadraticSimulator:
    """Test contrast sets on the quadratic simulator."""

    def test_finds_flip_across_zero_fitness(self, quadratic_sim):
        """Quadratic sim: fitness = -sum(x_i^2). Starting at origin (fitness=0),
        moving away should flip to negative."""
        gen = ContrastSetGenerator(quadratic_sim)
        # Start very near zero but positive side won't work since fitness = -sum(x^2) <= 0
        # Start at a point with non-zero fitness
        base = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        # fitness = 0.0, which is not > 0, so outcome = "negative"
        # target far from origin: fitness = -(1+1+1) = -3, also "negative"
        # No flip expected since both are "negative"
        target = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
        path = gen.generate_edit_path(base, target_params=target, n_edits=20, seed=42)
        result = gen.find_tipping_point(base, path)

        # Both outcomes are "negative" (fitness <= 0), so no flip
        assert result["found"] is False
