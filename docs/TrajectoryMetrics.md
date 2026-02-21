# trajectory_metrics

computes state-space path metrics for ODE simulator trajectory arrays

---

## Calling Patterns

```
trajectory_metrics(states, times)                       default state names (state_0, state_1, ...)
trajectory_metrics(states, times, state_names=names)    custom variable names
```

```
TrajectoryMetricsProfiler(simulator)                    wrap a simulator with run_trajectory()
profiler.profile(params)                                full nested metrics dict
profiler.flat_profile(params)                           scalar-only flattened dict
```

```
_flatten_metrics(metrics)                               flatten a metrics dict to scalar keys
```

---

## Details and Options

### trajectory_metrics

- `states` is a numpy array of shape `(T, D)` where `T` is the number of time steps and `D` is the number of state variables.
- `times` is a numpy array of shape `(T,)` with monotonically increasing time values.
- `state_names` is an optional list of `D` strings. If `None`, defaults to `["state_0", "state_1", ...]`.
- Returns a dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `n_steps` | int | Number of time steps (T) |
| `n_states` | int | Number of state variables (D) |
| `duration` | float | `times[-1] - times[0]` |
| `path_length` | float | Total arc length: sum of norm(diff(states)) |
| `chord_length` | float | Straight-line distance: norm(states[-1] - states[0]) |
| `smoothness` | float | `chord_length / path_length` (1.0 = straight line) |
| `mean_speed` | float | Mean of state-space velocity magnitudes |
| `speed_std` | float | Std of velocity magnitudes |
| `mean_curvature` | float | Mean angle (radians) between consecutive velocity vectors |
| `periodicity` | dict | Per-variable: `{name: {dominant_freq, spectral_entropy, is_periodic}}` |
| `per_variable` | dict | Per-variable: `{name: {mean, std, peak, trend_slope}}` |
| `attractor_convergence` | float | `var(last 20%) / var(full)` -- low = converging |
| `rankings` | dict | `{most_variable: [...], most_periodic: [...]}` |

- All scalar values are guaranteed finite (no NaN, no inf).
- Smoothness is 1.0 for constant trajectories (zero path length).
- Periodicity detection uses FFT on linearly detrended signals. `is_periodic` is True when dominant power fraction > 0.15 AND spectral entropy < 0.7 * max entropy.
- `attractor_convergence` is clamped to [0.0, 1.0]. Values near 0 indicate convergence to a fixed point; values near 1 indicate sustained dynamics.
- Mean curvature handles zero-velocity steps by skipping them.

### TrajectoryMetricsProfiler

- `simulator` must have a `run_trajectory(params) -> dict` method returning `{"states": ndarray, "times": ndarray}` and optionally `"state_names"`.
- `profile(params)` returns the full nested metrics dict.
- `flat_profile(params)` returns a scalar-only dict suitable for Zimmerman tool compatibility (Sobol, Falsifier, etc.).

### _flatten_metrics

- Periodicity entries become `periodicity_{var}_{metric}` keys.
- Per-variable entries become `var_{var}_{metric}` keys.
- Boolean `is_periodic` values are converted to int (0 or 1).
- Rankings are dropped (not scalar).

---

## Basic Examples

Compute metrics for a circular orbit:

```python
>>> import numpy as np
>>> from zimmerman.trajectory_metrics import trajectory_metrics

>>> times = np.linspace(0, 2*np.pi, 200, endpoint=False)
>>> states = np.column_stack([np.cos(times), np.sin(times)])
>>> result = trajectory_metrics(states, times, state_names=["x", "y"])

>>> result["n_steps"]
200
>>> result["smoothness"]  # circular, not straight
0.003  # close to 0

>>> result["periodicity"]["x"]["is_periodic"]
True

>>> result["attractor_convergence"]  # sustained orbit, not converging
0.97
```

Analyze a damped oscillator:

```python
>>> times = np.linspace(0, 10, 300)
>>> envelope = np.exp(-0.3 * times)
>>> states = np.column_stack([envelope*np.cos(2*times), envelope*np.sin(2*times)])
>>> result = trajectory_metrics(states, times)

>>> result["attractor_convergence"]  # converging to origin
0.001  # very low -- strong convergence

>>> result["per_variable"]["state_0"]["trend_slope"]
-0.08  # negative trend from decay
```

---

## Scope

Works with any dimensionality:

```python
>>> # 20-dimensional random trajectory
>>> rng = np.random.default_rng(42)
>>> states = rng.standard_normal((100, 20))
>>> times = np.linspace(0, 10, 100)
>>> result = trajectory_metrics(states, times)
>>> result["n_states"]
20
>>> len(result["per_variable"])
20
```

Handles edge cases gracefully:

```python
>>> # Constant trajectory
>>> states = np.full((100, 3), 0.5)
>>> times = np.linspace(0, 10, 100)
>>> result = trajectory_metrics(states, times)
>>> result["smoothness"]
1.0
>>> result["path_length"]
0.0

>>> # Very short trajectory (3 steps)
>>> states = np.array([[0, 0], [1, 1], [2, 2]])
>>> times = np.array([0, 1, 2])
>>> result = trajectory_metrics(states, times)
>>> result["n_steps"]
3
```

Use `TrajectoryMetricsProfiler` with any simulator that exposes trajectory data:

```python
>>> from zimmerman.trajectory_metrics import TrajectoryMetricsProfiler

>>> class MyODE:
...     def run_trajectory(self, params):
...         times = np.linspace(0, 10, 200)
...         x = np.exp(-params["decay"] * times) * np.cos(params["freq"] * times)
...         return {"states": x.reshape(-1, 1), "times": times, "state_names": ["x"]}

>>> profiler = TrajectoryMetricsProfiler(MyODE())
>>> report = profiler.profile({"decay": 0.1, "freq": 3.0})
>>> report["periodicity"]["x"]["is_periodic"]
True
```

Flatten for Zimmerman tool compatibility:

```python
>>> flat = profiler.flat_profile({"decay": 0.1, "freq": 3.0})
>>> all(isinstance(v, (int, float)) for v in flat.values())
True
>>> "periodicity_x_dominant_freq" in flat
True
>>> "var_x_mean" in flat
True
```

---

## Applications

**LEMURS simulator: characterizing student trajectories over a 15-week semester.** The 14-state LEMURS ODE produces time-series of sleep, stress, anxiety, GPA, and nature exposure. Trajectory metrics reveal whether a student archetype converges to a healthy equilibrium, oscillates, or diverges:

```python
import sys
sys.path.insert(0, "/path/to/lemurs-simulator")
from simulator import LEMURSSimulator
from zimmerman.trajectory_metrics import trajectory_metrics

sim = LEMURSSimulator()
result = sim.run_trajectory({"nature_dose": 0.7, "sleep_hygiene": 0.8})
metrics = trajectory_metrics(
    result["states"], result["times"], result["state_names"]
)

# Does the student reach a steady state?
print(f"Attractor convergence: {metrics['attractor_convergence']:.3f}")

# Which variables oscillate vs. trend?
for name in ["stress", "sleep_quality", "anxiety"]:
    p = metrics["periodicity"][name]
    v = metrics["per_variable"][name]
    print(f"  {name}: periodic={p['is_periodic']}, trend={v['trend_slope']:.3f}")
```

**Grief simulator: identifying PGD bifurcation.** The 11-state grief ODE can bifurcate into prolonged grief disorder when avoidance exceeds a threshold. Trajectory metrics detect this via attractor convergence and smoothness:

```python
import sys
sys.path.insert(0, "/path/to/grief-simulator")
from grief_simulator import GriefSimulator
from zimmerman.trajectory_metrics import trajectory_metrics

sim = GriefSimulator()
# High avoidance -> PGD bifurcation
result = sim.run_trajectory({"avoidance": 0.9, "social_support": 0.2})
metrics = trajectory_metrics(result["states"], result["times"])

# PGD trajectories have high attractor convergence (stuck in grief state)
# vs. normal grief which converges to recovery
print(f"Convergence: {metrics['attractor_convergence']:.3f}")
print(f"Most variable: {metrics['rankings']['most_variable'][:3]}")
```

**Mitochondrial aging model: cliff characterization.** Near the heteroplasmy cliff at ~70%, trajectories change character sharply. Trajectory metrics quantify this:

```python
import sys
sys.path.insert(0, "/path/to/how-to-live-much-longer")
from zimmerman_bridge import MitoSimulator
from zimmerman.trajectory_metrics import trajectory_metrics

sim = MitoSimulator()

# Below cliff
result_safe = sim.run_trajectory({"baseline_heteroplasmy": 0.5})
metrics_safe = trajectory_metrics(result_safe["states"], result_safe["times"])

# Above cliff
result_cliff = sim.run_trajectory({"baseline_heteroplasmy": 0.75})
metrics_cliff = trajectory_metrics(result_cliff["states"], result_cliff["times"])

# The cliff trajectory has higher curvature and lower smoothness
# as the system undergoes rapid ATP collapse
print(f"Safe smoothness: {metrics_safe['smoothness']:.3f}")
print(f"Cliff smoothness: {metrics_cliff['smoothness']:.3f}")
```

---

## Properties & Relations

- **Smoothness is bounded to [0, 1].** A smoothness of 1.0 means the trajectory is a straight line (chord equals arc). Smoothness near 0 indicates the trajectory wanders far from a direct path. For closed orbits, smoothness approaches 0 as the start and end points coincide.
- **Spectral entropy is measured in bits.** For `n` frequency bins, the maximum entropy is `log2(n)` (uniform power spectrum = white noise). Low entropy indicates power concentration at few frequencies (periodic signal).
- **Attractor convergence generalizes the concept of equilibrium detection.** Fixed-point attractors yield convergence near 0. Limit cycles yield convergence near the ratio of cycle variance to transient variance. Chaotic attractors yield convergence near 1 (sustained variance).
- **Connection to Sobol analysis.** Using `TrajectoryMetricsProfiler.flat_profile()` as the simulator's `run()` method enables Sobol analysis over trajectory characteristics -- answering "which parameters control periodicity?" or "which parameters drive convergence?"
- **Connection to Cramer-toolkit.** Flattened metrics can be used as outputs for scenario-based resilience analysis, asking "how does trajectory smoothness degrade under environmental stress?"
- **Deterministic computation.** Identical inputs always produce identical outputs (no random state). This satisfies the reproducibility requirement for Sobol and Falsifier analysis.

---

## Possible Issues

- **FFT periodicity detection assumes uniform time spacing.** Non-uniform time vectors will produce incorrect frequency estimates. All five ODE simulators in the workspace use uniform `np.linspace` time grids, so this is not an issue in practice.
- **Cumulative random walks appear periodic.** Integrated noise has a 1/f^2 power spectrum that concentrates power at low frequencies. The periodicity detector may flag such signals as periodic. This is physically correct (the signal does have dominant low-frequency components) but may be surprising.
- **Short trajectories (< 4 steps) limit periodicity analysis.** The periodicity computation returns zeros for signals shorter than 4 points. Path geometry and velocity metrics are still computed.
- **Attractor convergence uses a fixed 80/20 split.** This works well for trajectories where transients occupy less than 80% of the time series. For very slow convergence, the last 20% may still contain significant transient dynamics, yielding an overestimate.
- **State variables with different scales.** Path length, chord length, and velocity metrics combine all state variables equally via Euclidean norm. If variables have very different scales (e.g., cortisol in ng/mL vs. body temperature in degrees), the high-magnitude variable will dominate. Normalize states before calling `trajectory_metrics()` if scale-invariant path metrics are needed.

---

## See Also

`sobol_sensitivity` | `LocalityProfiler` | `Falsifier` | `Simulator` | `_flatten_metrics`

---

## References

- Strogatz, S.H. (2015). *Nonlinear Dynamics and Chaos.* 2nd ed. Westview Press. (Attractor convergence, limit cycles)
- Kantz, H. & Schreiber, T. (2004). *Nonlinear Time Series Analysis.* 2nd ed. Cambridge University Press. (Spectral entropy, periodicity detection)
