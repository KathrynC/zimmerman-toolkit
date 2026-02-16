"""Tests for reversible diegeticizer (zimmerman.diegeticizer).

Tests verify:
    1. Forward pass (diegeticize) maps values to correct bin labels.
    2. Reverse pass (re_diegeticize) recovers bin midpoints.
    3. Roundtrip error is bounded by half the bin width.
    4. Batch roundtrip produces valid aggregate statistics.
    5. The Diegeticizer satisfies the Simulator protocol.
    6. Lexicon (story handles) works for forward and reverse passes.
    7. Sample mode produces values within the correct bin.
"""

import numpy as np
import pytest

from zimmerman.diegeticizer import Diegeticizer


class TestDiscretization:
    """Tests for the forward discretization pass."""

    def test_low_value_maps_to_very_low(self, linear_sim):
        """A value near the lower bound should map to 'very low'."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {f"x{i}": 0.05 for i in range(4)}
        result = dieg.diegeticize(params)

        for name in params:
            assert result["narrative"][name] == "very low", (
                f"Param {name}=0.05 in [0,1] should be 'very low', "
                f"got '{result['narrative'][name]}'"
            )

    def test_high_value_maps_to_very_high(self, linear_sim):
        """A value near the upper bound should map to 'very high'."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {f"x{i}": 0.95 for i in range(4)}
        result = dieg.diegeticize(params)

        for name in params:
            assert result["narrative"][name] == "very high"

    def test_midpoint_maps_to_medium(self, linear_sim):
        """A value at the exact midpoint should map to 'medium'."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {f"x{i}": 0.5 for i in range(4)}
        result = dieg.diegeticize(params)

        for name in params:
            assert result["narrative"][name] == "medium"

    def test_bins_used_contains_range(self, linear_sim):
        """bins_used should contain the bin range for each parameter."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {"x0": 0.15, "x1": 0.5, "x2": 0.85, "x3": 0.99}
        result = dieg.diegeticize(params)

        for name in params:
            assert name in result["bins_used"]
            bin_info = result["bins_used"][name]
            assert "label" in bin_info
            assert "range" in bin_info
            lo, hi = bin_info["range"]
            assert lo < hi, f"Bin range for {name}: lo={lo} should be < hi={hi}"

    def test_numeric_preserved(self, linear_sim):
        """The numeric field should be a copy of the original params."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {"x0": 0.1, "x1": 0.2, "x2": 0.3, "x3": 0.4}
        result = dieg.diegeticize(params)

        for name, val in params.items():
            assert result["numeric"][name] == pytest.approx(val)

    def test_three_bins(self, quadratic_sim):
        """With 3 bins, should use 'low', 'medium', 'high'."""
        dieg = Diegeticizer(quadratic_sim, n_bins=3)
        # quadratic_sim has range [-1, 1]; midpoint is 0.0.
        params = {"x0": -0.8, "x1": 0.0, "x2": 0.8}
        result = dieg.diegeticize(params)

        assert result["narrative"]["x0"] == "low"
        assert result["narrative"]["x1"] == "medium"
        assert result["narrative"]["x2"] == "high"


class TestReDiegeticize:
    """Tests for the reverse pass (narrative -> numeric)."""

    def test_deterministic_returns_midpoints(self, linear_sim):
        """Deterministic mode should return bin midpoints."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        # For [0, 1] with 5 bins: edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]
        narrative = {"x0": "very low", "x1": "low", "x2": "medium", "x3": "high"}
        result = dieg.re_diegeticize(narrative, mode="deterministic")

        assert result["params"]["x0"] == pytest.approx(0.1)
        assert result["params"]["x1"] == pytest.approx(0.3)
        assert result["params"]["x2"] == pytest.approx(0.5)
        assert result["params"]["x3"] == pytest.approx(0.7)

    def test_sample_within_bin(self, linear_sim):
        """Sample mode should return values within the correct bin."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        narrative = {"x0": "medium", "x1": "medium", "x2": "medium", "x3": "medium"}
        # "medium" bin for [0, 1] is [0.4, 0.6].
        for seed in range(10):
            result = dieg.re_diegeticize(narrative, mode="sample", seed=seed)
            for name in narrative:
                val = result["params"][name]
                assert 0.4 <= val <= 0.6, (
                    f"Sample for 'medium' should be in [0.4, 0.6], got {val}"
                )

    def test_recovery_uncertainty_equals_bin_width(self, linear_sim):
        """Recovery uncertainty should equal the bin width for each param."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        narrative = {"x0": "low", "x1": "high", "x2": "very low", "x3": "very high"}
        result = dieg.re_diegeticize(narrative)

        expected_width = 1.0 / 5.0  # range=1.0, 5 bins
        for name in narrative:
            assert result["recovery_uncertainty"][name] == pytest.approx(
                expected_width, abs=1e-9
            )

    def test_mode_field_preserved(self, linear_sim):
        """The mode field should match the input mode."""
        dieg = Diegeticizer(linear_sim)
        result_det = dieg.re_diegeticize({"x0": "low", "x1": "low", "x2": "low", "x3": "low"}, mode="deterministic")
        result_smp = dieg.re_diegeticize({"x0": "low", "x1": "low", "x2": "low", "x3": "low"}, mode="sample")

        assert result_det["mode"] == "deterministic"
        assert result_smp["mode"] == "sample"

    def test_invalid_mode_raises(self, linear_sim):
        """An invalid mode should raise ValueError."""
        dieg = Diegeticizer(linear_sim)
        with pytest.raises(ValueError, match="mode must be"):
            dieg.re_diegeticize({"x0": "low", "x1": "low", "x2": "low", "x3": "low"}, mode="invalid")

    def test_invalid_label_raises(self, linear_sim):
        """An unknown bin label should raise ValueError."""
        dieg = Diegeticizer(linear_sim)
        with pytest.raises(ValueError, match="Unknown bin label"):
            dieg.re_diegeticize({"x0": "nonexistent", "x1": "low", "x2": "low", "x3": "low"})


class TestRoundtripError:
    """Tests for roundtrip error measurement."""

    def test_midpoint_roundtrip_is_zero(self, linear_sim):
        """A value at an exact bin midpoint should have zero roundtrip error."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        # Midpoint of 'medium' bin for [0, 1] with 5 bins is 0.5.
        params = {f"x{i}": 0.5 for i in range(4)}
        rt = dieg.roundtrip_error(params)

        assert rt["total_error"] == pytest.approx(0.0, abs=1e-9)
        for name in params:
            assert rt["per_param_error"][name] == pytest.approx(0.0, abs=1e-9)

    def test_roundtrip_error_bounded(self, linear_sim):
        """Roundtrip error per param should be <= half the bin width."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        rng = np.random.default_rng(123)

        for _ in range(50):
            params = {f"x{i}": float(rng.uniform(0, 1)) for i in range(4)}
            rt = dieg.roundtrip_error(params)

            bin_width = 1.0 / 5.0
            max_allowed = bin_width / 2.0 + 1e-9
            for name in params:
                assert rt["per_param_error"][name] <= max_allowed, (
                    f"Error for {name} = {rt['per_param_error'][name]} "
                    f"exceeds max allowed {max_allowed}"
                )

    def test_roundtrip_fields(self, linear_sim):
        """Roundtrip result should contain all expected fields."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {"x0": 0.15, "x1": 0.5, "x2": 0.85, "x3": 0.33}
        rt = dieg.roundtrip_error(params)

        assert "original" in rt
        assert "recovered" in rt
        assert "per_param_error" in rt
        assert "total_error" in rt
        assert "max_error_param" in rt
        assert "unrecoverable_params" in rt

    def test_max_error_param_correct(self, linear_sim):
        """max_error_param should be the param with the largest error."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        # 0.19 is close to a bin boundary (0.2), so it has high error.
        # 0.5 is a midpoint, so zero error.
        params = {"x0": 0.19, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        rt = dieg.roundtrip_error(params)

        assert rt["max_error_param"] == "x0"


class TestBatchRoundtrip:
    """Tests for batch roundtrip error statistics."""

    def test_batch_returns_all_fields(self, linear_sim):
        """Batch roundtrip should return all expected aggregate fields."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        result = dieg.batch_roundtrip(n_samples=20, seed=42)

        assert "mean_total_error" in result
        assert "std_total_error" in result
        assert "per_param_mean_error" in result
        assert "per_param_max_error" in result
        assert "unrecoverable_fraction" in result
        assert result["n_samples"] == 20

    def test_batch_mean_error_positive(self, linear_sim):
        """Mean total error should be > 0 for random samples (not all midpoints)."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        result = dieg.batch_roundtrip(n_samples=100, seed=42)

        assert result["mean_total_error"] > 0.0

    def test_batch_per_param_keys(self, linear_sim):
        """Per-param fields should contain all parameter names."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        result = dieg.batch_roundtrip(n_samples=10, seed=42)

        for name in linear_sim.param_spec():
            assert name in result["per_param_mean_error"]
            assert name in result["per_param_max_error"]
            assert name in result["unrecoverable_fraction"]

    def test_more_bins_less_error(self, linear_sim):
        """More bins should produce lower mean roundtrip error."""
        dieg_5 = Diegeticizer(linear_sim, n_bins=5)
        dieg_20 = Diegeticizer(linear_sim, n_bins=20)

        result_5 = dieg_5.batch_roundtrip(n_samples=200, seed=42)
        result_20 = dieg_20.batch_roundtrip(n_samples=200, seed=42)

        assert result_20["mean_total_error"] < result_5["mean_total_error"], (
            "20 bins should have lower error than 5 bins"
        )

    def test_seed_reproducibility(self, linear_sim):
        """Same seed should produce identical batch results."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        r1 = dieg.batch_roundtrip(n_samples=50, seed=99)
        r2 = dieg.batch_roundtrip(n_samples=50, seed=99)

        assert r1["mean_total_error"] == pytest.approx(r2["mean_total_error"])
        assert r1["std_total_error"] == pytest.approx(r2["std_total_error"])


class TestSimulatorProtocol:
    """Tests that Diegeticizer satisfies the Simulator protocol."""

    def test_run_returns_dict(self, linear_sim):
        """run() should return a dict (the underlying simulator's output)."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        params = {"x0": 0.1, "x1": 0.2, "x2": 0.3, "x3": 0.4}
        result = dieg.run(params)

        assert isinstance(result, dict)
        assert "y" in result

    def test_run_uses_diegeticized_params(self, linear_sim):
        """run() should produce different output than raw simulator for non-midpoint values."""
        dieg = Diegeticizer(linear_sim, n_bins=5)
        # Use values that are NOT bin midpoints.
        params = {"x0": 0.15, "x1": 0.35, "x2": 0.55, "x3": 0.75}

        raw_result = linear_sim.run(params)
        dieg_result = dieg.run(params)

        # The diegeticized result should differ (midpoints != original values).
        # But for bin midpoints, they would be the same.
        # 0.15 -> midpoint 0.1, 0.35 -> 0.3, 0.55 -> 0.5, 0.75 -> 0.7
        # raw y = 1*0.15 + 2*0.35 + 3*0.55 + 4*0.75 = 0.15 + 0.70 + 1.65 + 3.00 = 5.50
        # dieg y = 1*0.1 + 2*0.3 + 3*0.5 + 4*0.7 = 0.1 + 0.6 + 1.5 + 2.8 = 5.0
        assert raw_result["y"] != pytest.approx(dieg_result["y"], abs=1e-6)

    def test_param_spec_delegates(self, linear_sim):
        """param_spec() should return the same spec as the underlying simulator."""
        dieg = Diegeticizer(linear_sim, n_bins=5)

        assert dieg.param_spec() == linear_sim.param_spec()


class TestLexicon:
    """Tests for the lexicon (story handles) feature."""

    def test_lexicon_used_in_narrative(self, linear_sim):
        """When lexicon is provided, narrative keys should use story handles."""
        lexicon = {"x0": "speed", "x1": "strength"}
        dieg = Diegeticizer(linear_sim, lexicon=lexicon, n_bins=5)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        result = dieg.diegeticize(params)

        assert "speed" in result["narrative"]
        assert "strength" in result["narrative"]
        # Params without lexicon entries use their original name.
        assert "x2" in result["narrative"]
        assert "x3" in result["narrative"]

    def test_lexicon_roundtrip(self, linear_sim):
        """Diegeticize with lexicon, then re_diegeticize, should recover params."""
        lexicon = {"x0": "speed", "x1": "strength"}
        dieg = Diegeticizer(linear_sim, lexicon=lexicon, n_bins=5)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}

        dieg_result = dieg.diegeticize(params)
        re_result = dieg.re_diegeticize(dieg_result["narrative"], mode="deterministic")

        # Midpoint of 'medium' bin is 0.5, so roundtrip should be exact.
        for name in params:
            assert re_result["params"][name] == pytest.approx(0.5, abs=1e-9)
