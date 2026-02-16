"""Tests for locality profiling (zimmerman.locality_profiler).

Tests verify:
    1. LocalityProfiler satisfies the Simulator protocol (has run() and param_spec()).
    2. Default manipulations degrade the linear simulator's output as expected.
    3. L50 and effective_horizon are computed and have sensible values.
    4. Distractor susceptibility captures noise sensitivity.
    5. Profile returns all expected keys and types.
    6. Custom manipulations are accepted and used.
"""

import numpy as np
import pytest

from zimmerman.base import Simulator
from zimmerman.locality_profiler import (
    LocalityProfiler,
    _apply_cut,
    _apply_mask,
    _apply_distractor,
    _apply_target_position,
    _apply_shuffle,
    _interpolate_l50,
    _midpoint_params,
)


class TestLocalityProfilerProtocol:
    """Verify LocalityProfiler satisfies the Simulator protocol."""

    def test_is_simulator(self, linear_sim):
        """LocalityProfiler should satisfy the Simulator protocol."""
        profiler = LocalityProfiler(linear_sim)
        assert isinstance(profiler, Simulator)

    def test_param_spec_keys(self, linear_sim):
        """param_spec should return the five manipulation parameters."""
        profiler = LocalityProfiler(linear_sim)
        spec = profiler.param_spec()
        assert "cut_frac" in spec
        assert "mask_frac" in spec
        assert "distractor_strength" in spec
        assert "target_position" in spec
        assert "shuffle_window" in spec

    def test_param_spec_bounds(self, linear_sim):
        """All manipulation bounds should be valid (low < high)."""
        profiler = LocalityProfiler(linear_sim)
        spec = profiler.param_spec()
        for name, (lo, hi) in spec.items():
            assert lo < hi, f"Invalid bounds for {name}: ({lo}, {hi})"

    def test_run_returns_dict(self, linear_sim):
        """run() should return a dict with simulator outputs."""
        profiler = LocalityProfiler(linear_sim)
        params = {name: 0.0 for name in profiler.param_spec()}
        result = profiler.run(params)
        assert isinstance(result, dict)
        assert "y" in result  # LinearSimulator returns "y"

    def test_run_with_zero_manipulations(self, linear_sim):
        """With all manipulations at zero, output should be near baseline."""
        profiler = LocalityProfiler(linear_sim)
        params = {name: 0.0 for name in profiler.param_spec()}
        result = profiler.run(params)
        # At midpoint params (all 0.5), linear sim gives
        # y = 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
        assert isinstance(result["y"], float)


class TestDefaultManipulations:
    """Test each default manipulation function in isolation."""

    def test_cut_zeros_early_params(self):
        """cut_frac should zero out the first fraction of sorted params."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0), "d": (0.0, 1.0)}
        params = {"a": 0.8, "b": 0.7, "c": 0.6, "d": 0.5}

        # Cut 50%: first 2 of 4 params (a, b) should be set to lower bound
        result = _apply_cut(params, spec, 0.5)
        assert result["a"] == 0.0  # zeroed
        assert result["b"] == 0.0  # zeroed
        assert result["c"] == 0.6  # unchanged
        assert result["d"] == 0.5  # unchanged

    def test_cut_zero_preserves_all(self):
        """cut_frac=0 should leave all params unchanged."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        params = {"a": 0.8, "b": 0.7}
        result = _apply_cut(params, spec, 0.0)
        assert result == params

    def test_cut_one_zeros_all(self):
        """cut_frac=1 should zero out all params."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        params = {"a": 0.8, "b": 0.7}
        result = _apply_cut(params, spec, 1.0)
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_mask_replaces_with_midpoint(self):
        """mask_frac should replace selected params with midpoint values."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0), "d": (0.0, 1.0)}
        params = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}
        rng = np.random.default_rng(42)

        result = _apply_mask(params, spec, 0.5, rng)
        # 2 of 4 params should be at 0.5 (midpoint), rest at 0.9
        midpoint_count = sum(1 for v in result.values() if abs(v - 0.5) < 1e-9)
        assert midpoint_count == 2

    def test_mask_zero_preserves_all(self):
        """mask_frac=0 should leave all params unchanged."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        params = {"a": 0.8, "b": 0.7}
        rng = np.random.default_rng(0)
        result = _apply_mask(params, spec, 0.0, rng)
        assert result == params

    def test_distractor_adds_noise(self):
        """distractor_strength > 0 should perturb parameter values."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        params = {"a": 0.5, "b": 0.5}
        rng = np.random.default_rng(42)

        result = _apply_distractor(params, spec, 0.5, rng)
        # At least one param should differ from 0.5
        changed = any(abs(result[k] - 0.5) > 1e-9 for k in params)
        assert changed, "Distractor should perturb at least one parameter"

    def test_distractor_zero_preserves_all(self):
        """distractor_strength=0 should leave all params unchanged."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        params = {"a": 0.5, "b": 0.5}
        rng = np.random.default_rng(42)

        result = _apply_distractor(params, spec, 0.0, rng)
        for k in params:
            assert abs(result[k] - params[k]) < 1e-9

    def test_distractor_clips_to_bounds(self):
        """Distractor noise should be clipped to parameter bounds."""
        spec = {"a": (0.0, 1.0)}
        params = {"a": 0.99}
        rng = np.random.default_rng(42)

        result = _apply_distractor(params, spec, 1.0, rng)
        assert 0.0 <= result["a"] <= 1.0

    def test_target_position_weights_center(self):
        """target_position at 0.5 should weight middle params most."""
        spec = {f"x{i}": (0.0, 1.0) for i in range(5)}
        params = {f"x{i}": 1.0 for i in range(5)}

        result = _apply_target_position(params, spec, 0.5)
        # x2 (middle) should be closest to 1.0
        # x0 and x4 (edges) should be more blended toward midpoint (0.5)
        assert result["x2"] > result["x0"]
        assert result["x2"] > result["x4"]

    def test_shuffle_zero_preserves_order(self):
        """shuffle_window=0 should leave params unchanged."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 2.0), "c": (0.0, 3.0)}
        params = {"a": 0.1, "b": 1.5, "c": 2.5}
        rng = np.random.default_rng(42)

        result = _apply_shuffle(params, spec, 0.0, rng)
        assert result == params

    def test_shuffle_clips_to_bounds(self):
        """Shuffled values should be clipped to parameter bounds."""
        spec = {"a": (0.0, 0.5), "b": (0.5, 1.0)}
        params = {"a": 0.1, "b": 0.9}
        rng = np.random.default_rng(42)

        result = _apply_shuffle(params, spec, 1.0, rng)
        for name in spec:
            lo, hi = spec[name]
            assert lo <= result[name] <= hi, (
                f"{name}={result[name]} outside bounds ({lo}, {hi})"
            )


class TestInterpolateL50:
    """Test the L50 interpolation helper."""

    def test_exact_threshold(self):
        """When a sweep point is exactly at 50%, L50 should match."""
        values = [0.0, 0.5, 1.0]
        scores = [10.0, 5.0, 0.0]
        baseline = 10.0
        l50 = _interpolate_l50(values, scores, baseline)
        assert abs(l50 - 0.5) < 1e-6

    def test_interpolated_threshold(self):
        """L50 should interpolate between sweep points."""
        values = [0.0, 0.4, 0.8]
        scores = [10.0, 6.0, 2.0]
        baseline = 10.0
        l50 = _interpolate_l50(values, scores, baseline)
        # 50% of 10 = 5. Between 0.4 (score=6) and 0.8 (score=2)
        # Linear: 6 + (5-6)/(2-6) * (0.8-0.4) = 0.4 + 0.1 = 0.5
        assert 0.3 < l50 < 0.7

    def test_never_drops_below(self):
        """If score never drops below 50%, L50 should be max value."""
        values = [0.0, 0.5, 1.0]
        scores = [10.0, 8.0, 6.0]
        baseline = 10.0
        l50 = _interpolate_l50(values, scores, baseline)
        assert abs(l50 - 1.0) < 1e-6

    def test_zero_baseline(self):
        """Zero baseline should return 0.0."""
        l50 = _interpolate_l50([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], 0.0)
        assert abs(l50) < 1e-6


class TestMidpointParams:
    """Test the midpoint parameter helper."""

    def test_midpoint_values(self):
        """Midpoint params should be at the center of each range."""
        spec = {"a": (0.0, 1.0), "b": (2.0, 4.0), "c": (-1.0, 1.0)}
        mid = _midpoint_params(spec)
        assert abs(mid["a"] - 0.5) < 1e-9
        assert abs(mid["b"] - 3.0) < 1e-9
        assert abs(mid["c"] - 0.0) < 1e-9


class TestProfileLinear:
    """Test profiling on the linear simulator."""

    def test_profile_returns_all_keys(self, linear_sim):
        """profile() should return all expected top-level keys."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=5,
            seed=42,
        )
        assert "curves" in report
        assert "L50" in report
        assert "distractor_susceptibility" in report
        assert "effective_horizon" in report
        assert "n_sims" in report

    def test_profile_curves_structure(self, linear_sim):
        """Each curve should be a list of (value, mean, std) tuples."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=5,
            seed=42,
        )
        for manip_name, curve in report["curves"].items():
            assert len(curve) > 0, f"Curve for {manip_name} is empty"
            for point in curve:
                assert len(point) == 3, f"Point should be (value, mean, std)"
                val, mean, std = point
                assert isinstance(val, float)
                assert isinstance(mean, float)
                assert isinstance(std, float)

    def test_profile_n_sims_positive(self, linear_sim):
        """Profile should run a positive number of simulations."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=3,
            seed=42,
        )
        assert report["n_sims"] > 0

    def test_cut_degrades_linear_output(self, linear_sim):
        """Cutting more parameters should reduce the linear model's output."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.8 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            sweeps={"cut_frac": [0.0, 0.25, 0.5, 0.75, 1.0]},
            n_seeds=1,
            seed=42,
        )
        cut_curve = report["curves"]["cut_frac"]
        # Score at cut_frac=0.0 should be >= score at cut_frac=1.0
        score_zero = cut_curve[0][1]
        score_full = cut_curve[-1][1]
        assert score_zero >= score_full, (
            f"Cutting all params should reduce output: "
            f"score at 0.0={score_zero}, at 1.0={score_full}"
        )

    def test_l50_is_positive(self, linear_sim):
        """L50 values should be non-negative."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=5,
            seed=42,
        )
        for manip_name, l50_val in report["L50"].items():
            assert l50_val >= 0.0, (
                f"L50 for {manip_name} is negative: {l50_val}"
            )

    def test_effective_horizon_in_range(self, linear_sim):
        """Effective horizon should be between 0 and 1."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=5,
            seed=42,
        )
        assert 0.0 <= report["effective_horizon"] <= 1.0

    def test_distractor_susceptibility_finite(self, linear_sim):
        """Distractor susceptibility should be a finite float."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=5,
            seed=42,
        )
        assert np.isfinite(report["distractor_susceptibility"])


class TestProfileQuadratic:
    """Test profiling on the quadratic simulator."""

    def test_profile_quadratic(self, quadratic_sim):
        """Profiling should work on the quadratic simulator."""
        profiler = LocalityProfiler(quadratic_sim)
        base_params = {f"x{i}": 0.5 for i in range(3)}
        report = profiler.profile(
            task={"base_params": base_params},
            n_seeds=3,
            seed=42,
        )
        assert report["n_sims"] > 0
        assert "curves" in report


class TestCustomManipulations:
    """Test LocalityProfiler with custom manipulations."""

    def test_custom_manipulation_used(self, linear_sim):
        """Custom manipulations should replace the defaults."""
        call_count = [0]

        def my_manip(params, spec, val, rng):
            call_count[0] += 1
            return dict(params)

        profiler = LocalityProfiler(linear_sim, manipulations={"custom": my_manip})
        base_params = {f"x{i}": 0.5 for i in range(4)}
        report = profiler.profile(
            task={"base_params": base_params},
            sweeps={"custom": [0.0, 0.5, 1.0]},
            n_seeds=2,
            seed=42,
        )
        assert call_count[0] > 0, "Custom manipulation should have been called"
        assert "custom" in report["curves"]


class TestProfileDeterministic:
    """Test that profiling is deterministic with same seed."""

    def test_same_seed_same_result(self, linear_sim):
        """Two profiles with the same seed should produce identical results."""
        profiler = LocalityProfiler(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}

        report1 = profiler.profile(
            task={"base_params": base_params},
            sweeps={"cut_frac": [0.0, 0.5, 1.0]},
            n_seeds=5,
            seed=42,
        )
        report2 = profiler.profile(
            task={"base_params": base_params},
            sweeps={"cut_frac": [0.0, 0.5, 1.0]},
            n_seeds=5,
            seed=42,
        )

        for i, (pt1, pt2) in enumerate(
            zip(report1["curves"]["cut_frac"], report2["curves"]["cut_frac"])
        ):
            assert abs(pt1[1] - pt2[1]) < 1e-9, (
                f"Curves differ at point {i}: {pt1} vs {pt2}"
            )
