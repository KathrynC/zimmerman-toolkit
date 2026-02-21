"""State-space trajectory metrics for ODE simulator outputs.

Computes path geometry, velocity, curvature, periodicity, and attractor
convergence metrics from time-series state matrices. Domain-agnostic --
works on any (T, D) state matrix produced by an ODE simulator.

The core function ``trajectory_metrics()`` takes a state matrix and time
vector and returns a comprehensive dictionary of scalar and per-variable
metrics. The ``TrajectoryMetricsProfiler`` class wraps a simulator that
exposes ``run_trajectory()`` to make trajectory analysis composable with
other Zimmerman tools.

Metrics fall into six categories:

    Path geometry:
        Arc length, chord length, and smoothness (chord/arc ratio).
        A smoothness of 1.0 indicates a straight-line trajectory; values
        near 0 indicate a highly curved or oscillatory path.

    Velocity:
        Mean and standard deviation of state-space speed (norm of the
        velocity vector at each time step).

    Curvature:
        Mean angular deflection (radians) between consecutive velocity
        vectors. Zero for straight lines, pi for full reversals.

    Periodicity:
        Per-variable dominant frequency, spectral entropy, and a boolean
        periodicity flag. Uses FFT on linearly detrended signals.

    Per-variable statistics:
        Mean, standard deviation, peak absolute value, and linear trend
        slope for each state variable.

    Attractor convergence:
        Ratio of variance in the last 20% of the trajectory to full
        variance. Low values indicate convergence to a fixed point or
        limit cycle.

Pure numpy implementation (no scipy dependency).
"""

from __future__ import annotations

import numpy as np


def trajectory_metrics(
    states: np.ndarray,
    times: np.ndarray,
    state_names: list[str] | None = None,
) -> dict:
    """Compute state-space trajectory metrics from an ODE state matrix.

    Args:
        states: State matrix of shape (T, D) where T is the number of
            time steps and D is the number of state variables.
        times: Time vector of shape (T,) with monotonically increasing
            time values.
        state_names: Optional list of D state variable names. If None,
            defaults to ["state_0", "state_1", ...].

    Returns:
        Dictionary with keys: n_steps, n_states, duration, path_length,
        chord_length, smoothness, mean_speed, speed_std, mean_curvature,
        periodicity, per_variable, attractor_convergence, rankings.
        All scalar values are finite floats (no NaN, no inf).
    """
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    T, D = states.shape
    if state_names is None:
        state_names = [f"state_{i}" for i in range(D)]

    # -- Path geometry --
    diffs = np.diff(states, axis=0)  # (T-1, D)
    step_lengths = np.linalg.norm(diffs, axis=1)  # (T-1,)
    path_length = float(np.sum(step_lengths))
    chord = states[-1] - states[0]
    chord_length = float(np.linalg.norm(chord))

    if path_length > 0.0:
        smoothness = chord_length / path_length
    else:
        smoothness = 1.0  # constant trajectory is trivially smooth

    # -- Velocity --
    dt = np.diff(times)  # (T-1,)
    # Guard against zero dt
    dt_safe = np.where(dt > 0.0, dt, 1.0)
    velocities = diffs / dt_safe[:, np.newaxis]  # (T-1, D)
    speeds = np.linalg.norm(velocities, axis=1)  # (T-1,)

    if len(speeds) > 0:
        mean_speed = float(np.mean(speeds))
        speed_std = float(np.std(speeds))
    else:
        mean_speed = 0.0
        speed_std = 0.0

    # -- Curvature --
    mean_curvature = _compute_mean_curvature(velocities)

    # -- Periodicity (per variable) --
    periodicity = {}
    for j, name in enumerate(state_names):
        periodicity[name] = _compute_periodicity(states[:, j], times)

    # -- Per-variable stats --
    per_variable = {}
    for j, name in enumerate(state_names):
        col = states[:, j]
        per_variable[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "peak": float(np.max(np.abs(col))),
            "trend_slope": _linear_slope(times, col),
        }

    # -- Attractor convergence --
    attractor_convergence = _compute_attractor_convergence(states)

    # -- Rankings --
    stds = [(name, per_variable[name]["std"]) for name in state_names]
    most_variable = [name for name, _ in sorted(stds, key=lambda x: x[1], reverse=True)]

    entropies = [(name, periodicity[name]["spectral_entropy"]) for name in state_names]
    most_periodic = [name for name, _ in sorted(entropies, key=lambda x: x[1])]

    return {
        "n_steps": T,
        "n_states": D,
        "duration": float(times[-1] - times[0]),
        "path_length": path_length,
        "chord_length": chord_length,
        "smoothness": smoothness,
        "mean_speed": mean_speed,
        "speed_std": speed_std,
        "mean_curvature": mean_curvature,
        "periodicity": periodicity,
        "per_variable": per_variable,
        "attractor_convergence": attractor_convergence,
        "rankings": {
            "most_variable": most_variable,
            "most_periodic": most_periodic,
        },
    }


def _compute_mean_curvature(velocities: np.ndarray) -> float:
    """Mean angle (radians) between consecutive velocity vectors.

    Handles zero-velocity steps by skipping them. Returns 0.0 if fewer
    than two non-zero velocity vectors exist.
    """
    if len(velocities) < 2:
        return 0.0

    norms = np.linalg.norm(velocities, axis=1)
    angles = []
    for i in range(len(velocities) - 1):
        n1 = norms[i]
        n2 = norms[i + 1]
        if n1 < 1e-15 or n2 < 1e-15:
            continue
        cos_angle = np.dot(velocities[i], velocities[i + 1]) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.arccos(cos_angle))

    if len(angles) == 0:
        return 0.0
    return float(np.mean(angles))


def _compute_periodicity(signal: np.ndarray, times: np.ndarray) -> dict:
    """Compute periodicity metrics for a single state variable.

    Detrends the signal (removes linear fit), computes FFT, and extracts
    dominant frequency, spectral entropy, and a boolean periodicity flag.

    Returns dict with keys: dominant_freq, spectral_entropy, is_periodic.
    """
    n = len(signal)
    if n < 4:
        return {
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "is_periodic": False,
        }

    # Detrend: remove linear fit
    slope = _linear_slope(times, signal)
    intercept = np.mean(signal) - slope * np.mean(times)
    detrended = signal - (slope * times + intercept)

    # FFT
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2

    # Drop DC component (index 0)
    if len(power) > 1:
        power = power[1:]
    else:
        return {
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "is_periodic": False,
        }

    total_power = np.sum(power)
    if total_power < 1e-30:
        return {
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "is_periodic": False,
        }

    # Frequency axis (drop DC)
    duration = times[-1] - times[0]
    if duration <= 0.0:
        duration = 1.0
    freqs = np.fft.rfftfreq(n, d=duration / n)
    if len(freqs) > 1:
        freqs = freqs[1:]
    else:
        freqs = np.array([0.0])

    # Dominant frequency
    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx]) if dominant_idx < len(freqs) else 0.0

    # Dominant power fraction
    dominant_power_fraction = float(power[dominant_idx] / total_power)

    # Spectral entropy (in bits)
    p = power / total_power
    # Avoid log2(0)
    p_safe = np.where(p > 1e-30, p, 1e-30)
    spectral_entropy = float(-np.sum(p * np.log2(p_safe)))
    # Clamp to non-negative
    spectral_entropy = max(spectral_entropy, 0.0)

    # Max entropy for this many frequency bins
    n_bins = len(power)
    max_entropy = float(np.log2(n_bins)) if n_bins > 1 else 1.0

    # is_periodic: dominant power fraction > 0.15 AND entropy < 0.7 * max_entropy
    is_periodic = (dominant_power_fraction > 0.15) and (spectral_entropy < 0.7 * max_entropy)

    return {
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "is_periodic": bool(is_periodic),
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the slope of a linear fit using numpy polyfit.

    Returns 0.0 for constant or degenerate inputs.
    """
    if len(x) < 2:
        return 0.0
    try:
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        if not np.isfinite(slope):
            return 0.0
        return slope
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _compute_attractor_convergence(states: np.ndarray) -> float:
    """Ratio of variance in the last 20% to full trajectory variance.

    Low values indicate convergence to an attractor (fixed point or
    limit cycle). Returns 0.0 for constant trajectories.
    """
    T = states.shape[0]
    if T < 5:
        return 1.0

    tail_start = max(1, int(T * 0.8))
    tail = states[tail_start:]
    full_var = np.sum(np.var(states, axis=0))
    tail_var = np.sum(np.var(tail, axis=0))

    if full_var < 1e-30:
        return 0.0  # constant trajectory has converged trivially
    ratio = tail_var / full_var
    return float(np.clip(ratio, 0.0, 1.0))


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten a trajectory_metrics result dict to scalar-only keys.

    Nested dicts are flattened with underscores:
        - periodicity.{var}.{metric} -> periodicity_{var}_{metric}
        - per_variable.{var}.{metric} -> var_{var}_{metric}
        - rankings are dropped (not scalar)
        - All other top-level keys pass through unchanged.

    Args:
        metrics: Result dict from trajectory_metrics().

    Returns:
        Flat dict mapping string keys to scalar (float, int, bool) values.
    """
    flat = {}

    for key, value in metrics.items():
        if key == "periodicity":
            for var_name, var_dict in value.items():
                for metric_name, metric_value in var_dict.items():
                    flat_key = f"periodicity_{var_name}_{metric_name}"
                    if isinstance(metric_value, bool):
                        flat[flat_key] = int(metric_value)
                    else:
                        flat[flat_key] = metric_value
        elif key == "per_variable":
            for var_name, var_dict in value.items():
                for metric_name, metric_value in var_dict.items():
                    flat[f"var_{var_name}_{metric_name}"] = metric_value
        elif key == "rankings":
            continue  # drop non-scalar rankings
        elif isinstance(value, (int, float, bool)):
            flat[key] = value

    return flat


class TrajectoryMetricsProfiler:
    """Wraps a simulator with run_trajectory() for trajectory analysis.

    The wrapped simulator must provide a ``run_trajectory(params)`` method
    that returns a dict with at least ``"states"`` (ndarray of shape
    (T, D)) and ``"times"`` (ndarray of shape (T,)). An optional
    ``"state_names"`` key (list of D strings) provides variable names.

    This class does NOT satisfy the Simulator protocol (it does not have
    param_spec). It provides ``profile()`` and ``flat_profile()`` as
    convenience methods for trajectory analysis.

    Example:
        >>> profiler = TrajectoryMetricsProfiler(my_simulator)
        >>> report = profiler.profile({"dose": 0.5, "age": 65})
        >>> report["smoothness"]
        0.87
        >>> flat = profiler.flat_profile({"dose": 0.5, "age": 65})
        >>> all(isinstance(v, (int, float)) for v in flat.values())
        True
    """

    def __init__(self, simulator):
        """Initialize with a simulator that has run_trajectory().

        Args:
            simulator: Any object with a ``run_trajectory(params) -> dict``
                method. The returned dict must contain ``"states"`` and
                ``"times"`` arrays, and optionally ``"state_names"``.
        """
        self._simulator = simulator

    def profile(self, params: dict) -> dict:
        """Run the simulator and compute trajectory metrics.

        Args:
            params: Parameter dict to pass to simulator.run_trajectory().

        Returns:
            Full trajectory_metrics result dict with all nested structure.
        """
        result = self._simulator.run_trajectory(params)
        states = np.asarray(result["states"], dtype=np.float64)
        times = np.asarray(result["times"], dtype=np.float64)
        state_names = result.get("state_names", None)
        return trajectory_metrics(states, times, state_names)

    def flat_profile(self, params: dict) -> dict:
        """Run the simulator and return flattened scalar-only metrics.

        Like ``profile()``, but flattens nested dicts and drops non-scalar
        values (rankings) for Zimmerman tool compatibility.

        Args:
            params: Parameter dict to pass to simulator.run_trajectory().

        Returns:
            Flat dict mapping string keys to scalar values.
        """
        metrics = self.profile(params)
        return _flatten_metrics(metrics)
