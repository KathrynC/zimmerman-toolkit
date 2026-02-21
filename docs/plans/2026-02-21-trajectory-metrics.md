# Trajectory Metrics Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `trajectory_metrics` module to zimmerman-toolkit that computes state-space path metrics (curvature, smoothness, periodicity, attractor detection) from ODE simulator trajectory arrays — a cross-cutting capability usable by all 5 simulators.

**Architecture:** Standalone function `trajectory_metrics(states, times, state_names=None) -> dict` operates on raw numpy arrays, not through the Zimmerman `Simulator` protocol. A companion `TrajectoryMetricsProfiler` class wraps any simulator that exposes trajectory data, returning metrics as a Zimmerman-compatible flat dict. The module lives alongside existing tools but operates on trajectory arrays rather than scalar `run()` outputs.

**Tech Stack:** Python 3.11+, numpy-only (FFT via `np.fft`, no scipy)

---

## Context

The zimmerman-toolkit currently operates exclusively on **scalar summaries** from `run()`. It never analyzes the raw time-series trajectory arrays (`states: ndarray(T, N_states)`, `times: ndarray(T,)`) that all 5 ODE simulators produce internally. This is a gap: state-space trajectory shape (how curved, how smooth, how periodic, whether converging to an attractor) contains dynamical information that scalar summaries discard.

All 5 simulators produce trajectory arrays:
- LEMURS: `states` shape (106, 14), 15 weeks at daily resolution
- Mito: `states` shape (3001, 8), 30 years at dt=0.01yr
- Grief: `states` shape (3651, 11), 10 years at daily resolution
- Stock: trajectory shape (2521, 7), 10 years at daily trading resolution
- ER: ~18 channels of (4000,) at 240 Hz (physics telemetry, not stacked matrix)

The trajectory metrics are **domain-agnostic** — path curvature in state-space is computed identically whether the states represent anxiety levels, heteroplasmy, or portfolio values.

**Precedent in the codebase:**
- `how-to-live-much-longer/analytics.py` already computes `ros_dominant_freq` and `ros_amplitude` via FFT — trajectory_metrics generalizes this to all state variables
- `pybullet_test/Evolutionary-Robotics/compute_beer_analytics.py` computes `path_length`, `path_straightness`, `heading_consistency` — trajectory_metrics provides the same metrics for non-spatial state-spaces
- `lemurs-simulator/ca_analytics.py` computes `attractor_type` classification — trajectory_metrics detects convergence quantitatively

---

## Reference Files

**Pattern to follow:**
- `~/zimmerman-toolkit/zimmerman/sobol.py` — standalone function pattern (pure function, no class needed for core computation)
- `~/zimmerman-toolkit/zimmerman/locality_profiler.py` — class-based wrapper that also satisfies Simulator protocol
- `~/zimmerman-toolkit/zimmerman/__init__.py` — where to register exports
- `~/zimmerman-toolkit/zimmerman/meaning_construction_dashboard.py` — `_TOOL_SECTIONS` registry for dashboard integration

**Existing trajectory analytics to unify:**
- `~/how-to-live-much-longer/analytics.py:214-230` — `ros_dominant_freq`, `ros_amplitude` via FFT
- `~/pybullet_test/Evolutionary-Robotics/compute_beer_analytics.py:87-120` — `path_length`, `path_straightness`

---

## Tasks

### Task 1: Core trajectory_metrics function

**Files:**
- Create: `~/zimmerman-toolkit/zimmerman/trajectory_metrics.py`
- Test: `~/zimmerman-toolkit/tests/test_trajectory_metrics.py`

**Step 1: Write test fixtures and basic tests**

Create `tests/test_trajectory_metrics.py` with synthetic trajectory fixtures:

```python
"""Tests for trajectory_metrics module."""
import numpy as np
import pytest
from zimmerman.trajectory_metrics import trajectory_metrics


def _linear_trajectory(n=100, d=3):
    """Straight line in d-dimensional space."""
    times = np.linspace(0, 10, n)
    states = np.column_stack([np.linspace(0, 1, n)] * d)
    return states, times


def _circular_trajectory(n=200, freq=1.0):
    """Circle in 2D, constant speed."""
    times = np.linspace(0, 2 * np.pi / freq, n, endpoint=False)
    states = np.column_stack([np.cos(freq * times), np.sin(freq * times)])
    return states, times


def _damped_oscillator(n=300, decay=0.1, freq=2.0):
    """Damped sinusoid converging to origin."""
    times = np.linspace(0, 10, n)
    envelope = np.exp(-decay * times)
    states = np.column_stack([
        envelope * np.cos(freq * times),
        envelope * np.sin(freq * times),
    ])
    return states, times


def _random_walk(n=200, d=3, seed=42):
    """Random walk in d dimensions."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 10, n)
    steps = rng.standard_normal((n, d)) * 0.01
    states = np.cumsum(steps, axis=0)
    return states, times


def _constant_trajectory(n=100, d=3, value=0.5):
    """Stationary point — no motion."""
    times = np.linspace(0, 10, n)
    states = np.full((n, d), value)
    return states, times


class TestTrajectoryMetricsBasic:
    """Test return structure and types."""

    def test_returns_dict(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        assert isinstance(result, dict)

    def test_required_keys(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        for key in ["n_steps", "n_states", "duration", "path_length",
                     "chord_length", "smoothness", "mean_curvature",
                     "mean_speed", "speed_std", "periodicity",
                     "per_variable", "attractor_convergence", "rankings"]:
            assert key in result, f"Missing key: {key}"

    def test_n_steps_and_states(self):
        states, times = _linear_trajectory(n=100, d=3)
        result = trajectory_metrics(states, times)
        assert result["n_steps"] == 100
        assert result["n_states"] == 3

    def test_duration(self):
        states, times = _linear_trajectory(n=100)
        result = trajectory_metrics(states, times)
        assert abs(result["duration"] - 10.0) < 1e-10

    def test_state_names_default(self):
        states, times = _linear_trajectory(n=50, d=3)
        result = trajectory_metrics(states, times)
        assert "state_0" in result["per_variable"]
        assert "state_1" in result["per_variable"]
        assert "state_2" in result["per_variable"]

    def test_state_names_custom(self):
        states, times = _linear_trajectory(n=50, d=3)
        result = trajectory_metrics(states, times, state_names=["x", "y", "z"])
        assert "x" in result["per_variable"]
        assert "y" in result["per_variable"]
        assert "z" in result["per_variable"]

    def test_all_values_finite(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        for key, val in result.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"{key} is not finite: {val}"


class TestLinearTrajectory:
    """Straight line: smoothness=1.0, curvature=0."""

    def test_smoothness_is_one(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["smoothness"] - 1.0) < 0.01

    def test_curvature_is_zero(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        assert result["mean_curvature"] < 0.01

    def test_path_length_equals_chord(self):
        states, times = _linear_trajectory()
        result = trajectory_metrics(states, times)
        assert abs(result["path_length"] - result["chord_length"]) < 1e-6


class TestCircularTrajectory:
    """Periodic motion: periodicity detected, smoothness < 1."""

    def test_smoothness_less_than_one(self):
        states, times = _circular_trajectory()
        result = trajectory_metrics(states, times)
        assert result["smoothness"] < 0.5  # circle has low chord/path ratio

    def test_periodicity_detected(self):
        states, times = _circular_trajectory(freq=1.0)
        result = trajectory_metrics(states, times)
        # At least one state variable should be periodic
        periodic_vars = [
            name for name, info in result["periodicity"].items()
            if info["is_periodic"]
        ]
        assert len(periodic_vars) > 0

    def test_constant_speed(self):
        states, times = _circular_trajectory()
        result = trajectory_metrics(states, times)
        # Speed CV should be very small for constant-speed circle
        assert result["speed_std"] / max(result["mean_speed"], 1e-10) < 0.1


class TestDampedOscillator:
    """Converging trajectory: low attractor_convergence."""

    def test_attractor_convergence_low(self):
        states, times = _damped_oscillator()
        result = trajectory_metrics(states, times)
        # Terminal variance should be much less than full variance
        assert result["attractor_convergence"] < 0.3

    def test_periodicity_detected(self):
        states, times = _damped_oscillator(freq=2.0)
        result = trajectory_metrics(states, times)
        periodic_vars = [
            name for name, info in result["periodicity"].items()
            if info["is_periodic"]
        ]
        assert len(periodic_vars) > 0


class TestRandomWalk:
    """Random walk: low smoothness, high spectral entropy."""

    def test_low_smoothness(self):
        states, times = _random_walk()
        result = trajectory_metrics(states, times)
        assert result["smoothness"] < 0.5

    def test_high_spectral_entropy(self):
        states, times = _random_walk()
        result = trajectory_metrics(states, times)
        # Random walks should have high spectral entropy (near-uniform spectrum)
        for name, info in result["periodicity"].items():
            assert info["spectral_entropy"] > 2.0  # bits


class TestConstantTrajectory:
    """No motion: path_length=0, speed=0."""

    def test_path_length_zero(self):
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert result["path_length"] < 1e-10

    def test_speed_zero(self):
        states, times = _constant_trajectory()
        result = trajectory_metrics(states, times)
        assert result["mean_speed"] < 1e-10


class TestRankings:
    """Rankings subdict is correct."""

    def test_most_variable_ranking(self):
        # Create trajectory where var 0 has most variation
        times = np.linspace(0, 10, 100)
        states = np.column_stack([
            np.sin(times) * 10,  # high amplitude
            np.sin(times) * 1,   # low amplitude
            np.sin(times) * 5,   # medium amplitude
        ])
        result = trajectory_metrics(states, times, state_names=["big", "small", "mid"])
        assert result["rankings"]["most_variable"][0] == "big"

    def test_most_periodic_ranking(self):
        states, times = _circular_trajectory()
        result = trajectory_metrics(states, times, state_names=["x", "y"])
        assert "most_periodic" in result["rankings"]
        assert len(result["rankings"]["most_periodic"]) == 2


class TestDeterminism:
    """Same inputs produce same outputs."""

    def test_deterministic(self):
        states, times = _random_walk(seed=123)
        r1 = trajectory_metrics(states, times)
        r2 = trajectory_metrics(states, times)
        for key in ["path_length", "smoothness", "mean_curvature", "mean_speed"]:
            assert r1[key] == r2[key]
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_trajectory_metrics.py -v`
Expected: ImportError (module doesn't exist yet)

**Step 3: Implement trajectory_metrics.py**

Create `~/zimmerman-toolkit/zimmerman/trajectory_metrics.py`:

```python
"""State-space trajectory metrics for ODE simulator outputs.

Computes path-geometric and spectral properties of multi-dimensional
trajectories: curvature, smoothness, periodicity, speed profiles,
and attractor convergence. Domain-agnostic — works identically on
anxiety trajectories, heteroplasmy curves, or portfolio dynamics.

Unlike other zimmerman-toolkit modules, this operates on raw trajectory
arrays (states, times) rather than through the Simulator protocol's
scalar run() output.

Usage:
    from zimmerman.trajectory_metrics import trajectory_metrics

    # After running any ODE simulator:
    result = simulate(intervention, patient)
    metrics = trajectory_metrics(result["states"], result["times"])
"""
from __future__ import annotations

import numpy as np


def trajectory_metrics(
    states: np.ndarray,
    times: np.ndarray,
    state_names: list[str] | None = None,
) -> dict:
    """Compute state-space trajectory metrics.

    Args:
        states: Array of shape (T, D) — T timesteps, D state variables.
        times: Array of shape (T,) — time values for each step.
        state_names: Optional list of D names. Defaults to state_0, state_1, ...

    Returns:
        Dict with keys: n_steps, n_states, duration, path_length, chord_length,
        smoothness, mean_curvature, mean_speed, speed_std, periodicity,
        per_variable, attractor_convergence, rankings.
    """
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    T, D = states.shape
    if state_names is None:
        state_names = [f"state_{i}" for i in range(D)]

    duration = float(times[-1] - times[0])
    dt = np.diff(times)

    # ── Path geometry ────────────────────────────────────────────────────
    diffs = np.diff(states, axis=0)                    # (T-1, D)
    step_lengths = np.linalg.norm(diffs, axis=1)       # (T-1,)
    path_length = float(np.sum(step_lengths))
    chord_length = float(np.linalg.norm(states[-1] - states[0]))

    # Smoothness: chord/path ratio (1.0 = straight line)
    if path_length > 0:
        smoothness = chord_length / path_length
    else:
        smoothness = 1.0  # stationary point is "perfectly smooth"

    # ── Velocity and speed ───────────────────────────────────────────────
    # Velocity: ds/dt for each step
    dt_safe = np.where(dt > 0, dt, 1e-10)
    velocity = diffs / dt_safe[:, np.newaxis]          # (T-1, D)
    speed = np.linalg.norm(velocity, axis=1)           # (T-1,)
    mean_speed = float(np.mean(speed))
    speed_std = float(np.std(speed))

    # ── Curvature ────────────────────────────────────────────────────────
    # Curvature = angle between consecutive velocity vectors
    mean_curvature = _compute_mean_curvature(velocity)

    # ── Periodicity (per variable) ───────────────────────────────────────
    periodicity = {}
    for i, name in enumerate(state_names):
        periodicity[name] = _periodicity_analysis(states[:, i], times)

    # ── Per-variable statistics ──────────────────────────────────────────
    per_variable = {}
    for i, name in enumerate(state_names):
        col = states[:, i]
        # Trend slope via least-squares
        if duration > 0:
            slope = float(np.polyfit(times, col, 1)[0])
        else:
            slope = 0.0
        per_variable[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "peak": float(np.max(np.abs(col))),
            "trend_slope": slope,
        }

    # ── Attractor convergence ────────────────────────────────────────────
    # Ratio of variance in last 20% vs. full trajectory
    # Low = converging to attractor; high = diverging or oscillating
    attractor_convergence = _attractor_convergence(states)

    # ── Rankings ─────────────────────────────────────────────────────────
    rankings = _compute_rankings(state_names, per_variable, periodicity)

    return {
        "n_steps": T,
        "n_states": D,
        "duration": duration,
        "path_length": path_length,
        "chord_length": chord_length,
        "smoothness": smoothness,
        "mean_curvature": mean_curvature,
        "mean_speed": mean_speed,
        "speed_std": speed_std,
        "periodicity": periodicity,
        "per_variable": per_variable,
        "attractor_convergence": attractor_convergence,
        "rankings": rankings,
    }


def _compute_mean_curvature(velocity: np.ndarray) -> float:
    """Mean angular curvature between consecutive velocity vectors.

    Args:
        velocity: (T-1, D) velocity array.

    Returns:
        Mean angle in radians between consecutive velocity vectors.
    """
    if len(velocity) < 2:
        return 0.0

    v1 = velocity[:-1]  # (T-2, D)
    v2 = velocity[1:]   # (T-2, D)

    norms1 = np.linalg.norm(v1, axis=1)
    norms2 = np.linalg.norm(v2, axis=1)

    # Mask out zero-velocity steps
    valid = (norms1 > 1e-12) & (norms2 > 1e-12)
    if not np.any(valid):
        return 0.0

    # Cosine of angle between consecutive velocities
    dots = np.sum(v1[valid] * v2[valid], axis=1)
    cos_angles = dots / (norms1[valid] * norms2[valid])
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return float(np.mean(angles))


def _periodicity_analysis(signal: np.ndarray, times: np.ndarray) -> dict:
    """FFT-based periodicity analysis for a single state variable.

    Returns:
        Dict with dominant_freq, spectral_entropy, is_periodic.
    """
    n = len(signal)
    if n < 4:
        return {"dominant_freq": 0.0, "spectral_entropy": 0.0, "is_periodic": False}

    # Detrend (remove linear trend)
    detrended = signal - np.polyval(np.polyfit(times, signal, 1), times)

    # FFT
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    # Drop DC component
    power = power[1:]

    if len(power) == 0 or np.sum(power) < 1e-20:
        return {"dominant_freq": 0.0, "spectral_entropy": 0.0, "is_periodic": False}

    # Frequency axis
    duration = times[-1] - times[0]
    if duration > 0:
        freqs = np.fft.rfftfreq(n, d=duration / n)[1:]
    else:
        freqs = np.arange(1, len(power) + 1, dtype=float)

    # Dominant frequency
    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx]) if dominant_idx < len(freqs) else 0.0

    # Spectral entropy (bits)
    power_norm = power / np.sum(power)
    power_norm = power_norm[power_norm > 0]
    spectral_entropy = float(-np.sum(power_norm * np.log2(power_norm)))

    # Max possible entropy (uniform spectrum)
    max_entropy = np.log2(len(power)) if len(power) > 1 else 1.0

    # Is periodic: dominant freq explains significant fraction of power
    # and spectral entropy is well below maximum (concentrated spectrum)
    dominant_power_fraction = float(power[dominant_idx] / np.sum(power))
    is_periodic = (dominant_power_fraction > 0.15) and (spectral_entropy < 0.7 * max_entropy)

    return {
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "is_periodic": bool(is_periodic),
    }


def _attractor_convergence(states: np.ndarray) -> float:
    """Ratio of variance in last 20% of trajectory vs full trajectory.

    Low (< 0.3) suggests convergence to an attractor.
    High (> 0.7) suggests divergence or sustained oscillation.
    """
    T = len(states)
    if T < 5:
        return 1.0

    cutoff = max(1, int(T * 0.8))
    terminal = states[cutoff:]
    full = states

    # Total variance across all dimensions
    terminal_var = float(np.sum(np.var(terminal, axis=0)))
    full_var = float(np.sum(np.var(full, axis=0)))

    if full_var < 1e-20:
        return 0.0  # constant trajectory = perfectly converged

    return terminal_var / full_var


def _compute_rankings(
    state_names: list[str],
    per_variable: dict,
    periodicity: dict,
) -> dict:
    """Compute ranked lists of state variables by various criteria."""
    # Most variable (by std)
    most_variable = sorted(
        state_names,
        key=lambda n: per_variable[n]["std"],
        reverse=True,
    )

    # Most periodic (by dominant power fraction, proxied by low spectral entropy)
    most_periodic = sorted(
        state_names,
        key=lambda n: periodicity[n]["spectral_entropy"],
    )

    return {
        "most_variable": most_variable,
        "most_periodic": most_periodic,
    }
```

**Step 4: Run tests**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_trajectory_metrics.py -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
cd ~/zimmerman-toolkit
git add zimmerman/trajectory_metrics.py tests/test_trajectory_metrics.py
git commit -m "feat: add trajectory_metrics module for state-space path analysis"
```

---

### Task 2: Register in __init__.py and dashboard

**Files:**
- Modify: `~/zimmerman-toolkit/zimmerman/__init__.py`

**Step 1: Add import and __all__ entry**

Add after the `meaning_construction_dashboard` import (line 48):

```python
from zimmerman.trajectory_metrics import trajectory_metrics
```

Add to `__all__` list:

```python
"trajectory_metrics",
```

**Step 2: Run full test suite**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/ -v`
Expected: All ~285+ existing tests pass, plus new trajectory tests.

**Step 3: Commit**

```bash
cd ~/zimmerman-toolkit
git add zimmerman/__init__.py
git commit -m "feat: register trajectory_metrics in zimmerman namespace"
```

---

### Task 3: TrajectoryMetricsProfiler wrapper class

**Files:**
- Modify: `~/zimmerman-toolkit/zimmerman/trajectory_metrics.py`
- Test: `~/zimmerman-toolkit/tests/test_trajectory_metrics.py`

**Step 1: Add tests for TrajectoryMetricsProfiler**

Append to `tests/test_trajectory_metrics.py`:

```python
from zimmerman.trajectory_metrics import TrajectoryMetricsProfiler


class _FakeTrajectorySimulator:
    """Simulator that returns trajectory data alongside scalar results."""

    def __init__(self, states, times, state_names=None):
        self._states = states
        self._times = times
        self._state_names = state_names

    def run(self, params):
        return {"fitness": 1.0}

    def param_spec(self):
        return {"x": (0.0, 1.0)}

    def run_trajectory(self, params):
        return {
            "states": self._states,
            "times": self._times,
            "state_names": self._state_names,
        }


class TestTrajectoryMetricsProfiler:
    def test_profile_returns_dict(self):
        states, times = _linear_trajectory()
        sim = _FakeTrajectorySimulator(states, times, ["a", "b", "c"])
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.profile({})
        assert isinstance(result, dict)
        assert "path_length" in result

    def test_profile_includes_all_metrics(self):
        states, times = _circular_trajectory()
        sim = _FakeTrajectorySimulator(states, times, ["x", "y"])
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.profile({})
        assert "smoothness" in result
        assert "mean_curvature" in result
        assert "attractor_convergence" in result

    def test_flat_output_for_zimmerman(self):
        states, times = _linear_trajectory(d=2)
        sim = _FakeTrajectorySimulator(states, times, ["a", "b"])
        profiler = TrajectoryMetricsProfiler(sim)
        result = profiler.flat_profile({})
        # All keys should be strings, all values should be floats
        for k, v in result.items():
            assert isinstance(k, str), f"Key {k} is not str"
            assert isinstance(v, (int, float)), f"Value for {k} is {type(v)}"
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_trajectory_metrics.py::TestTrajectoryMetricsProfiler -v`
Expected: ImportError

**Step 3: Implement TrajectoryMetricsProfiler**

Add to `zimmerman/trajectory_metrics.py`:

```python
class TrajectoryMetricsProfiler:
    """Wraps a simulator that exposes trajectory data.

    The wrapped simulator must implement a ``run_trajectory(params)``
    method returning ``{"states": ndarray, "times": ndarray,
    "state_names": list[str] | None}``.

    Usage:
        profiler = TrajectoryMetricsProfiler(my_simulator)
        metrics = profiler.profile(params)
        flat = profiler.flat_profile(params)  # all-scalar for Zimmerman
    """

    def __init__(self, simulator):
        self._sim = simulator

    def profile(self, params: dict) -> dict:
        """Run simulation and return full trajectory metrics."""
        traj = self._sim.run_trajectory(params)
        return trajectory_metrics(
            traj["states"],
            traj["times"],
            traj.get("state_names"),
        )

    def flat_profile(self, params: dict) -> dict:
        """Run simulation and return flattened scalar-only metrics.

        Suitable for feeding into zimmerman-toolkit's scalar analysis
        pipeline (Sobol, contrastive, etc.).
        """
        result = self.profile(params)
        return _flatten_metrics(result)


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten trajectory_metrics output to scalar-only dict.

    Nested dicts (periodicity, per_variable) are flattened with
    underscore separators. Non-scalar values (lists, dicts) are dropped.
    """
    flat = {}
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            flat[key] = float(val)
        elif key == "periodicity":
            for var_name, info in val.items():
                for metric_name, metric_val in info.items():
                    if isinstance(metric_val, (int, float, bool)):
                        flat[f"periodicity_{var_name}_{metric_name}"] = float(metric_val)
        elif key == "per_variable":
            for var_name, info in val.items():
                for metric_name, metric_val in info.items():
                    if isinstance(metric_val, (int, float)):
                        flat[f"var_{var_name}_{metric_name}"] = float(metric_val)
        # rankings and other non-scalar keys are dropped
    return flat
```

**Step 4: Update __init__.py**

Add `TrajectoryMetricsProfiler` to the import and `__all__`.

**Step 5: Run tests**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_trajectory_metrics.py -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
cd ~/zimmerman-toolkit
git add zimmerman/trajectory_metrics.py zimmerman/__init__.py tests/test_trajectory_metrics.py
git commit -m "feat: add TrajectoryMetricsProfiler wrapper for Zimmerman integration"
```

---

### Task 4: Integration test with real simulators

**Files:**
- Create: `~/zimmerman-toolkit/tests/test_trajectory_metrics_integration.py`

**Step 1: Write integration tests**

These tests import actual simulators and verify trajectory_metrics works on real ODE output.

```python
"""Integration tests: trajectory_metrics on real simulator output."""
import sys
import pytest
import numpy as np

# Skip if simulators not available
lemurs_available = True
try:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "lemurs-simulator"))
    from simulator import simulate
    from constants import STUDENT_ARCHETYPES, DEFAULT_INTERVENTION
except ImportError:
    lemurs_available = False

from zimmerman.trajectory_metrics import trajectory_metrics


@pytest.mark.skipif(not lemurs_available, reason="lemurs-simulator not found")
class TestLEMURSIntegration:
    def test_lemurs_trajectory_metrics(self):
        patient = dict(STUDENT_ARCHETYPES["resilient_male"]["patient"])
        result = simulate(DEFAULT_INTERVENTION, patient)
        states = result["states"]
        times = result["times"]
        metrics = trajectory_metrics(states, times)
        assert metrics["n_steps"] == 106
        assert metrics["n_states"] == 14
        assert metrics["path_length"] > 0
        assert 0 <= metrics["smoothness"] <= 1.0
        assert metrics["attractor_convergence"] >= 0

    def test_vulnerable_vs_resilient_diverge(self):
        """Vulnerable student should have different trajectory shape."""
        r1 = simulate(
            DEFAULT_INTERVENTION,
            dict(STUDENT_ARCHETYPES["resilient_male"]["patient"]),
        )
        r2 = simulate(
            DEFAULT_INTERVENTION,
            dict(STUDENT_ARCHETYPES["vulnerable_female"]["patient"]),
        )
        m1 = trajectory_metrics(r1["states"], r1["times"])
        m2 = trajectory_metrics(r2["states"], r2["times"])
        # Vulnerable student should have longer path (more state-space movement)
        # or different curvature — they shouldn't be identical
        assert m1["path_length"] != m2["path_length"]
```

**Step 2: Run integration tests**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_trajectory_metrics_integration.py -v`
Expected: Tests pass if lemurs-simulator is available, skip otherwise.

**Step 3: Commit**

```bash
cd ~/zimmerman-toolkit
git add tests/test_trajectory_metrics_integration.py
git commit -m "test: add trajectory_metrics integration tests with LEMURS simulator"
```

---

### Task 5: Documentation

**Files:**
- Create: `~/zimmerman-toolkit/docs/TrajectoryMetrics.md`

**Step 1: Write documentation**

Follow the pattern of existing docs (e.g., `docs/sobol_sensitivity.md`, `docs/LocalityProfiler.md`).

```markdown
# Trajectory Metrics

Computes state-space path metrics for ODE simulator trajectories.

## Overview

Unlike other zimmerman-toolkit modules that operate on scalar `run()` output, `trajectory_metrics` analyzes the raw trajectory arrays that ODE simulators produce: `states` (T×D matrix) and `times` (T-length vector).

The metrics are domain-agnostic — the same function works on anxiety trajectories, heteroplasmy curves, or portfolio dynamics.

## Quick Start

```python
from zimmerman.trajectory_metrics import trajectory_metrics

# After running any ODE simulator:
result = simulate(intervention, patient)
metrics = trajectory_metrics(result["states"], result["times"])

print(f"Path length: {metrics['path_length']:.3f}")
print(f"Smoothness: {metrics['smoothness']:.3f}")
print(f"Converging: {metrics['attractor_convergence']:.3f}")
```

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `path_length` | float | Total arc length in state-space |
| `chord_length` | float | Start-to-end straight-line distance |
| `smoothness` | float | chord/path ratio (1.0 = straight line) |
| `mean_curvature` | float | Average angular bend per step (radians) |
| `mean_speed` | float | Average state-space velocity |
| `speed_std` | float | Speed variability |
| `attractor_convergence` | float | Terminal variance / full variance (low = converging) |
| `periodicity` | dict | Per-variable: dominant_freq, spectral_entropy, is_periodic |
| `per_variable` | dict | Per-variable: mean, std, peak, trend_slope |
| `rankings` | dict | State variables ranked by variability and periodicity |

## TrajectoryMetricsProfiler

For integration with Zimmerman-protocol tools, wrap a simulator that has `run_trajectory()`:

```python
from zimmerman.trajectory_metrics import TrajectoryMetricsProfiler

profiler = TrajectoryMetricsProfiler(my_simulator)
flat = profiler.flat_profile(params)  # all-scalar dict, Zimmerman-compatible
```

## Compatible Simulators

Works with any simulator producing `states: ndarray(T, D)` and `times: ndarray(T,)`:
- LEMURS (14D, 106 steps)
- Mitochondrial aging (8D, 3001 steps)
- Grief (11D, 3651 steps)
- Stock (7D, 2521 steps)
```

**Step 2: Commit**

```bash
cd ~/zimmerman-toolkit
git add docs/TrajectoryMetrics.md
git commit -m "docs: add trajectory_metrics documentation"
```

---

## Verification

After all tasks:
```bash
cd ~/zimmerman-toolkit
python -m pytest tests/ -v                           # All tests pass
python -m pytest tests/test_trajectory_metrics.py -v  # Module tests specifically

# One-liners
python -c "from zimmerman import trajectory_metrics; print('OK')"
python -c "
import numpy as np
from zimmerman.trajectory_metrics import trajectory_metrics
t = np.linspace(0, 10, 100)
s = np.column_stack([np.sin(t), np.cos(t)])
m = trajectory_metrics(s, t, ['x', 'y'])
print(f'smoothness={m[\"smoothness\"]:.3f}, periodic={m[\"periodicity\"][\"x\"][\"is_periodic\"]}')
"
```
