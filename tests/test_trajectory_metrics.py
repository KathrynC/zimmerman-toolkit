"""Tests for trajectory metrics (zimmerman.trajectory_metrics).

Tests verify:
    1. trajectory_metrics returns a dict with all required keys.
    2. Linear trajectories have smoothness ~1.0 and curvature ~0.
    3. Circular trajectories are detected as periodic with low smoothness.
    4. Damped oscillators show attractor convergence.
    5. Random walks have high spectral entropy and low smoothness.
    6. Constant trajectories have zero path length and speed.
    7. Rankings correctly order variables by variability and periodicity.
    8. Results are deterministic for identical inputs.
    9. TrajectoryMetricsProfiler wraps simulators correctly.
"""

import numpy as np
import pytest

from zimmerman.trajectory_metrics import (
    trajectory_metrics,
    TrajectoryMetricsProfiler,
    _flatten_metrics,
)


# ---- Synthetic trajectory fixtures ----

def _linear_trajectory(n=100, d=3):
    """Straight-line trajectory from origin to (1,1,...,1)."""
    times = np.linspace(0, 10, n)
    states = np.column_stack([np.linspace(0, 1, n)] * d)
    return states, times


def _circular_trajectory(n=200, freq=1.0):
    """2D circular orbit at given frequency."""
    times = np.linspace(0, 2 * np.pi / freq, n, endpoint=False)
    states = np.column_stack([np.cos(freq * times), np.sin(freq * times)])
    return states, times


def _damped_oscillator(n=300, decay=0.1, freq=2.0):
    """2D damped oscillator spiraling toward the origin."""
    times = np.linspace(0, 10, n)
    envelope = np.exp(-decay * times)
    states = np.column_stack([
        envelope * np.cos(freq * times),
        envelope * np.sin(freq * times),
    ])
    return states, times


def _random_walk(n=200, d=3, seed=42):
    """Cumulative random walk in d dimensions."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 10, n)
    states = np.cumsum(rng.standard_normal((n, d)) * 0.01, axis=0)
    return states, times


def _constant_trajectory(n=100, d=3, value=0.5):
    """Constant trajectory (all states fixed at value)."""
    times = np.linspace(0, 10, n)
    states = np.full((n, d), value)
    return states, times


# ---- Required keys ----

REQUIRED_TOP_KEYS = {
    "n_steps", "n_states", "duration", "path_length", "chord_length",
    "smoothness", "mean_speed", "speed_std", "mean_curvature",
    "periodicity", "per_variable", "attractor_convergence", "rankings",
}

REQUIRED_PERIODICITY_KEYS = {"dominant_freq", "spectral_entropy", "is_periodic"}
REQUIRED_PER_VAR_KEYS = {"mean", "std", "peak", "trend_slope"}
REQUIRED_RANKING_KEYS = {"most_variable", "most_periodic"}


# ---- TestTrajectoryMetricsBasic ----

class TestTrajectoryMetricsBasic:
    """Basic structure and type checks."""

    def test_returns_dict(self):
        """trajectory_metrics should return a dict."""
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        """Result should contain all required top-level keys."""
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        for key in REQUIRED_TOP_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_correct_n_steps(self):
        """n_steps should equal the number of time points."""
        states, times = _linear_trajectory(n=75)
        result = trajectory_metrics(states, times)
        assert result["n_steps"] == 75

    def test_correct_n_states(self):
        """n_states should equal the number of state variables."""
        states, times = _linear_trajectory(n=50, d=5)
        result = trajectory_metrics(states, times)
        assert result["n_states"] == 5

    def test_correct_duration(self):
        """duration should be times[-1] - times[0]."""
        states, times = _linear_trajectory(n=100)
        result = trajectory_metrics(states, times)
        assert abs(result["duration"] - 10.0) < 1e-9

    def test_default_state_names(self):
        """Without explicit names, should use state_0, state_1, ..."""
        states, times = _linear_trajectory(n=50, d=3)
        result = trajectory_metrics(states, times)
        expected_names = {"state_0", "state_1", "state_2"}
        assert set(result["per_variable"].keys()) == expected_names
        assert set(result["periodicity"].keys()) == expected_names

    def test_custom_state_names(self):
        """Custom state names should be used as dict keys."""
        states, times = _linear_trajectory(n=50, d=2)
        result = trajectory_metrics(states, times, state_names=["x", "y"])
        assert "x" in result["per_variable"]
        assert "y" in result["per_variable"]
        assert "x" in result["periodicity"]
        assert "y" in result["periodicity"]

    def test_all_values_finite(self):
        """All scalar values in the result must be finite (no NaN, no inf)."""
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)

        # Check top-level scalars
        for key in ["n_steps", "n_states", "duration", "path_length",
                     "chord_length", "smoothness", "mean_speed", "speed_std",
                     "mean_curvature", "attractor_convergence"]:
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

        # Check per-variable scalars
        for name, stats in result["per_variable"].items():
            for k, v in stats.items():
                assert np.isfinite(v), f"per_variable[{name}][{k}] is not finite: {v}"

        # Check periodicity scalars
        for name, pdict in result["periodicity"].items():
            for k in ["dominant_freq", "spectral_entropy"]:
                assert np.isfinite(pdict[k]), f"periodicity[{name}][{k}] is not finite: {pdict[k]}"

    def test_periodicity_has_required_keys(self):
        """Each periodicity entry should have all required sub-keys."""
        states, times = _circular_trajectory()
        result = trajectory_metrics(states, times)
        for name, pdict in result["periodicity"].items():
            for key in REQUIRED_PERIODICITY_KEYS:
                assert key in pdict, f"Missing periodicity key {key} for {name}"

    def test_per_variable_has_required_keys(self):
        """Each per_variable entry should have all required sub-keys."""
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        for name, stats in result["per_variable"].items():
            for key in REQUIRED_PER_VAR_KEYS:
                assert key in stats, f"Missing per_variable key {key} for {name}"

    def test_rankings_has_required_keys(self):
        """Rankings should have most_variable and most_periodic keys."""
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        for key in REQUIRED_RANKING_KEYS:
            assert key in result["rankings"], f"Missing rankings key: {key}"


# ---- TestLinearTrajectory ----

class TestLinearTrajectory:
    """Tests on a straight-line trajectory."""

    def test_smoothness_near_one(self):
        """A straight-line trajectory should have smoothness ~1.0."""
        states, times = _linear_trajectory(n=200, d=3)
        result = trajectory_metrics(states, times)
        assert result["smoothness"] > 0.99

    def test_curvature_near_zero(self):
        """A straight-line trajectory should have near-zero curvature."""
        states, times = _linear_trajectory(n=200, d=3)
        result = trajectory_metrics(states, times)
        assert result["mean_curvature"] < 0.01

    def test_path_length_equals_chord(self):
        """For a straight line, path_length should equal chord_length."""
        states, times = _linear_trajectory(n=200, d=3)
        result = trajectory_metrics(states, times)
        assert abs(result["path_length"] - result["chord_length"]) < 1e-6

    def test_positive_speed(self):
        """A moving trajectory should have positive mean speed."""
        states, times = _linear_trajectory(n=200, d=3)
        result = trajectory_metrics(states, times)
        assert result["mean_speed"] > 0.0

    def test_low_speed_std(self):
        """A constant-velocity trajectory should have low speed std."""
        states, times = _linear_trajectory(n=200, d=3)
        result = trajectory_metrics(states, times)
        # Speed should be nearly constant for uniform time spacing
        assert result["speed_std"] < result["mean_speed"] * 0.01


# ---- TestCircularTrajectory ----

class TestCircularTrajectory:
    """Tests on a circular orbit trajectory."""

    def test_low_smoothness(self):
        """A circular orbit should have smoothness well below 1.0."""
        states, times = _circular_trajectory(n=200, freq=1.0)
        result = trajectory_metrics(states, times)
        assert result["smoothness"] < 0.5

    def test_periodicity_detected(self):
        """At least one variable should be detected as periodic."""
        states, times = _circular_trajectory(n=200, freq=1.0)
        result = trajectory_metrics(states, times)
        periodic_count = sum(
            1 for p in result["periodicity"].values() if p["is_periodic"]
        )
        assert periodic_count >= 1, "Circular trajectory should be periodic"

    def test_roughly_constant_speed(self):
        """A circular orbit should have approximately constant speed."""
        states, times = _circular_trajectory(n=500, freq=1.0)
        result = trajectory_metrics(states, times)
        # Speed std should be small relative to mean speed
        if result["mean_speed"] > 0:
            cv = result["speed_std"] / result["mean_speed"]
            assert cv < 0.1, f"Speed coefficient of variation too high: {cv}"

    def test_positive_curvature(self):
        """A circular orbit should have positive mean curvature."""
        states, times = _circular_trajectory(n=200, freq=1.0)
        result = trajectory_metrics(states, times)
        assert result["mean_curvature"] > 0.0


# ---- TestDampedOscillator ----

class TestDampedOscillator:
    """Tests on a damped oscillator trajectory."""

    def test_attractor_convergence_low(self):
        """A damped oscillator should show attractor convergence < 0.3."""
        states, times = _damped_oscillator(n=500, decay=0.3, freq=2.0)
        result = trajectory_metrics(states, times)
        assert result["attractor_convergence"] < 0.3, (
            f"attractor_convergence={result['attractor_convergence']} should be < 0.3"
        )

    def test_periodicity_detected(self):
        """A damped oscillator should still show periodic components."""
        states, times = _damped_oscillator(n=500, decay=0.05, freq=2.0)
        result = trajectory_metrics(states, times)
        periodic_count = sum(
            1 for p in result["periodicity"].values() if p["is_periodic"]
        )
        assert periodic_count >= 1, "Damped oscillator should have periodic components"

    def test_positive_path_length(self):
        """A damped oscillator should have nonzero path length."""
        states, times = _damped_oscillator()
        result = trajectory_metrics(states, times)
        assert result["path_length"] > 0.0


# ---- TestRandomWalk ----

class TestRandomWalk:
    """Tests on a random walk trajectory."""

    def test_low_smoothness(self):
        """A random walk should have low smoothness."""
        states, times = _random_walk(n=500, d=3)
        result = trajectory_metrics(states, times)
        assert result["smoothness"] < 0.8

    def test_high_spectral_entropy(self):
        """A random walk should have high spectral entropy (> 2.0 bits)."""
        states, times = _random_walk(n=500, d=3)
        result = trajectory_metrics(states, times)
        for name, pdict in result["periodicity"].items():
            assert pdict["spectral_entropy"] > 2.0, (
                f"spectral_entropy for {name} = {pdict['spectral_entropy']} should be > 2.0"
            )

    def test_white_noise_not_periodic(self):
        """White noise (non-cumulative) should not be detected as periodic."""
        rng = np.random.default_rng(42)
        times = np.linspace(0, 10, 500)
        states = rng.standard_normal((500, 3)) * 0.1
        result = trajectory_metrics(states, times)
        periodic_count = sum(
            1 for p in result["periodicity"].values() if p["is_periodic"]
        )
        # Allow at most 1 false positive out of 3
        assert periodic_count <= 1


# ---- TestConstantTrajectory ----

class TestConstantTrajectory:
    """Tests on a constant (stationary) trajectory."""

    def test_zero_path_length(self):
        """A constant trajectory should have zero path length."""
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["path_length"]) < 1e-12

    def test_zero_speed(self):
        """A constant trajectory should have zero mean speed."""
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["mean_speed"]) < 1e-12

    def test_smoothness_is_one(self):
        """A constant trajectory should have smoothness = 1.0."""
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["smoothness"] - 1.0) < 1e-9

    def test_zero_chord_length(self):
        """A constant trajectory should have zero chord length."""
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["chord_length"]) < 1e-12

    def test_all_values_finite(self):
        """Even constant trajectories should produce all-finite results."""
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        for key in ["path_length", "chord_length", "smoothness",
                     "mean_speed", "speed_std", "mean_curvature",
                     "attractor_convergence"]:
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ---- TestRankings ----

class TestRankings:
    """Tests for the rankings sub-dict."""

    def test_most_variable_correctly_ordered(self):
        """Variables with higher amplitude should rank as most variable."""
        n = 200
        times = np.linspace(0, 10, n)
        # Three variables with different amplitudes
        states = np.column_stack([
            np.sin(times) * 1.0,   # amplitude 1 (least variable)
            np.sin(times) * 5.0,   # amplitude 5 (middle)
            np.sin(times) * 10.0,  # amplitude 10 (most variable)
        ])
        result = trajectory_metrics(states, times, state_names=["low", "mid", "high"])
        ranking = result["rankings"]["most_variable"]
        assert ranking[0] == "high", f"Expected 'high' first, got {ranking}"
        assert ranking[-1] == "low", f"Expected 'low' last, got {ranking}"

    def test_most_periodic_correctly_ordered(self):
        """Periodic variables should rank higher (lower entropy) than noisy ones."""
        n = 500
        times = np.linspace(0, 10, n)
        rng = np.random.default_rng(42)
        states = np.column_stack([
            np.sin(2 * np.pi * times),          # periodic
            rng.standard_normal(n) * 0.1,        # noisy
        ])
        result = trajectory_metrics(states, times, state_names=["periodic", "noisy"])
        ranking = result["rankings"]["most_periodic"]
        assert ranking[0] == "periodic", f"Expected 'periodic' first, got {ranking}"

    def test_rankings_contain_all_names(self):
        """Rankings should contain all state variable names."""
        states, times = _linear_trajectory(n=50, d=4)
        names = ["a", "b", "c", "d"]
        result = trajectory_metrics(states, times, state_names=names)
        assert set(result["rankings"]["most_variable"]) == set(names)
        assert set(result["rankings"]["most_periodic"]) == set(names)


# ---- TestDeterminism ----

class TestDeterminism:
    """Tests that trajectory_metrics is deterministic."""

    def test_same_inputs_same_outputs(self):
        """Identical inputs should produce identical outputs."""
        states, times = _circular_trajectory(n=200, freq=1.0)
        result1 = trajectory_metrics(states, times)
        result2 = trajectory_metrics(states.copy(), times.copy())

        for key in ["path_length", "chord_length", "smoothness",
                     "mean_speed", "speed_std", "mean_curvature",
                     "attractor_convergence"]:
            assert result1[key] == result2[key], (
                f"{key} differs: {result1[key]} vs {result2[key]}"
            )

    def test_random_walk_deterministic(self):
        """Two random walks with the same seed should give identical metrics."""
        states1, times1 = _random_walk(n=200, d=3, seed=123)
        states2, times2 = _random_walk(n=200, d=3, seed=123)
        result1 = trajectory_metrics(states1, times1)
        result2 = trajectory_metrics(states2, times2)
        assert result1["path_length"] == result2["path_length"]
        assert result1["smoothness"] == result2["smoothness"]


# ---- TestTrajectoryMetricsProfiler ----

class _MockTrajectorySimulator:
    """Mock simulator with run_trajectory() for profiler tests."""

    def __init__(self, n=100, d=3):
        self.n = n
        self.d = d

    def run_trajectory(self, params):
        freq = params.get("freq", 1.0)
        times = np.linspace(0, 10, self.n)
        states = np.column_stack([
            np.sin(freq * times * (j + 1)) for j in range(self.d)
        ])
        return {
            "states": states,
            "times": times,
            "state_names": [f"var_{j}" for j in range(self.d)],
        }


class TestTrajectoryMetricsProfiler:
    """Tests for the TrajectoryMetricsProfiler class."""

    def test_profile_returns_dict(self):
        """profile() should return a dict with all required keys."""
        sim = _MockTrajectorySimulator(n=100, d=3)
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.profile({"freq": 2.0})
        assert isinstance(result, dict)
        for key in REQUIRED_TOP_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_profile_correct_dimensions(self):
        """profile() should report correct n_steps and n_states."""
        sim = _MockTrajectorySimulator(n=150, d=4)
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.profile({"freq": 1.0})
        assert result["n_steps"] == 150
        assert result["n_states"] == 4

    def test_profile_uses_state_names(self):
        """profile() should use state_names from the simulator."""
        sim = _MockTrajectorySimulator(n=50, d=2)
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.profile({"freq": 1.0})
        assert "var_0" in result["per_variable"]
        assert "var_1" in result["per_variable"]

    def test_flat_profile_returns_scalars_only(self):
        """flat_profile() should return a dict of scalar values only."""
        sim = _MockTrajectorySimulator(n=100, d=3)
        profiler = TrajectoryMetricsProfiler(sim)
        flat = profiler.flat_profile({"freq": 2.0})
        assert isinstance(flat, dict)
        for key, value in flat.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), (
                f"flat_profile key {key} has non-scalar value: {type(value)}"
            )

    def test_flat_profile_has_periodicity_keys(self):
        """flat_profile() should have periodicity_var_metric keys."""
        sim = _MockTrajectorySimulator(n=100, d=2)
        profiler = TrajectoryMetricsProfiler(sim)
        flat = profiler.flat_profile({"freq": 1.0})
        assert "periodicity_var_0_dominant_freq" in flat
        assert "periodicity_var_0_spectral_entropy" in flat
        assert "periodicity_var_0_is_periodic" in flat

    def test_flat_profile_has_per_variable_keys(self):
        """flat_profile() should have var_varname_metric keys."""
        sim = _MockTrajectorySimulator(n=100, d=2)
        profiler = TrajectoryMetricsProfiler(sim)
        flat = profiler.flat_profile({"freq": 1.0})
        assert "var_var_0_mean" in flat
        assert "var_var_0_std" in flat
        assert "var_var_0_peak" in flat
        assert "var_var_0_trend_slope" in flat

    def test_flat_profile_no_rankings(self):
        """flat_profile() should not contain rankings (non-scalar)."""
        sim = _MockTrajectorySimulator(n=100, d=3)
        profiler = TrajectoryMetricsProfiler(sim)
        flat = profiler.flat_profile({"freq": 1.0})
        for key in flat:
            assert "ranking" not in key.lower(), f"Rankings should not appear in flat profile: {key}"

    def test_flat_profile_no_nested_dicts(self):
        """flat_profile() should have no nested dict values."""
        sim = _MockTrajectorySimulator(n=100, d=3)
        profiler = TrajectoryMetricsProfiler(sim)
        flat = profiler.flat_profile({"freq": 1.0})
        for key, value in flat.items():
            assert not isinstance(value, dict), f"Nested dict at key {key}"


# ---- TestFlattenMetrics ----

class TestFlattenMetrics:
    """Tests for the _flatten_metrics helper."""

    def test_scalar_passthrough(self):
        """Top-level scalars should pass through unchanged."""
        states, times = _linear_trajectory(n=50, d=2)
        metrics = trajectory_metrics(states, times)
        flat = _flatten_metrics(metrics)
        assert flat["n_steps"] == metrics["n_steps"]
        assert flat["smoothness"] == metrics["smoothness"]

    def test_periodicity_flattened(self):
        """Periodicity entries should be flattened with underscores."""
        states, times = _linear_trajectory(n=50, d=2)
        metrics = trajectory_metrics(states, times, state_names=["x", "y"])
        flat = _flatten_metrics(metrics)
        assert "periodicity_x_dominant_freq" in flat
        assert "periodicity_y_spectral_entropy" in flat

    def test_per_variable_flattened(self):
        """Per-variable entries should be flattened with 'var_' prefix."""
        states, times = _linear_trajectory(n=50, d=2)
        metrics = trajectory_metrics(states, times, state_names=["x", "y"])
        flat = _flatten_metrics(metrics)
        assert "var_x_mean" in flat
        assert "var_y_std" in flat

    def test_rankings_dropped(self):
        """Rankings should not appear in flattened output."""
        states, times = _linear_trajectory(n=50, d=2)
        metrics = trajectory_metrics(states, times)
        flat = _flatten_metrics(metrics)
        assert "rankings" not in flat
        assert "most_variable" not in flat

    def test_bool_converted_to_int(self):
        """Boolean is_periodic values should be converted to int."""
        states, times = _circular_trajectory(n=200)
        metrics = trajectory_metrics(states, times, state_names=["x", "y"])
        flat = _flatten_metrics(metrics)
        key = "periodicity_x_is_periodic"
        assert isinstance(flat[key], int)
        assert flat[key] in (0, 1)


# ---- TestEdgeCases ----

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_trajectory(self):
        """Trajectories with < 4 steps should not crash."""
        states = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        times = np.array([0.0, 1.0, 2.0])
        result = trajectory_metrics(states, times)
        assert isinstance(result, dict)
        assert result["n_steps"] == 3

    def test_single_dimension(self):
        """A 1D trajectory should work correctly."""
        times = np.linspace(0, 10, 100)
        states = np.sin(times).reshape(-1, 1)
        result = trajectory_metrics(states, times, state_names=["x"])
        assert result["n_states"] == 1
        assert "x" in result["per_variable"]

    def test_high_dimensional(self):
        """A high-dimensional trajectory should work."""
        n, d = 100, 20
        times = np.linspace(0, 10, n)
        rng = np.random.default_rng(42)
        states = rng.standard_normal((n, d))
        result = trajectory_metrics(states, times)
        assert result["n_states"] == 20
        assert len(result["per_variable"]) == 20
        assert len(result["periodicity"]) == 20
