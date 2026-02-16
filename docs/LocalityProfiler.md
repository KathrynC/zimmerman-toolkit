# LocalityProfiler

measures how local a system's behavior is via controlled manipulation sweeps

---

## Calling Patterns

```
LocalityProfiler(simulator)                                   default 5 manipulations
LocalityProfiler(simulator, manipulations=custom_dict)         custom manipulation functions
```

```
profiler.profile(task)                                         default sweeps, 30 seeds
profiler.profile(task, sweeps=custom_sweeps)                   custom sweep schedule
profiler.profile(task, n_seeds=100, seed=0)                    more seeds, different RNG
```

```
profiler.run(params)                                           Simulator protocol (manipulation intensities as params)
profiler.param_spec()                                          returns manipulation parameter bounds
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol (`run(params) -> dict` + `param_spec() -> dict`).
- `manipulations` is a dict mapping manipulation names to callables with signature `(params, spec, value, rng) -> modified_params`. If `None`, uses five default manipulations (see below).

### profile

- `task` is a dict with key `"base_params"` containing the baseline parameter values to perturb. These should be a meaningful operating point, not just the midpoint.
- `sweeps` is a dict mapping manipulation names to lists of float values to sweep. Default: `{"cut_frac": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8], "target_position": [0.1, 0.3, 0.5, 0.7, 0.9]}`.
- `n_seeds` (default 30): number of random seeds per sweep point. Higher values reduce noise in stochastic manipulations but increase cost linearly.
- `seed` (default 42): base random seed for reproducibility.
- Returns a dict with:
  - `"curves"`: `{manipulation_name: [(value, mean_score, std_score), ...]}` -- raw locality decay curves.
  - `"L50"`: `{manipulation_name: float}` -- the manipulation intensity where performance drops to 50% of baseline.
  - `"distractor_susceptibility"`: float -- linear slope of score vs. distractor strength. Negative = normal degradation; near-zero = robust; positive = possible stochastic resonance.
  - `"effective_horizon"`: float -- the largest prefix fraction (`cut_frac`) that can be removed while retaining >90% of baseline performance.
  - `"n_sims"`: int -- total simulator evaluations performed.

### run (Simulator protocol)

- `params` is a dict mapping manipulation names to float intensities.
- Base parameters are always the midpoint of the inner simulator's spec (the "uninformative default").
- Manipulations are applied sequentially in iteration order. When multiple manipulations have non-zero values, they compose.
- Returns the inner simulator's result dict, augmented with a `"manipulations_applied"` key.

### param_spec (Simulator protocol)

- Returns `{"cut_frac": (0.0, 1.0), "mask_frac": (0.0, 1.0), "distractor_strength": (0.0, 1.0), "target_position": (0.1, 0.9), "shuffle_window": (0.0, 1.0)}`.
- `target_position` is bounded to `[0.1, 0.9]` to avoid degenerate edge cases.

### The five default manipulations

1. **cut_frac** (prefix ablation): Sets the first `cut_frac` fraction of parameters (sorted alphabetically) to their lower bound. Simulates removing early context. Deterministic.

2. **mask_frac** (random masking): Replaces a randomly selected `mask_frac` fraction of parameters with their midpoint (uninformative default). Simulates information dropout. Stochastic.

3. **distractor_strength** (additive noise): Adds uniform noise scaled by `distractor_strength * (hi - lo)` to all parameters, clipped to bounds. Simulates irrelevant context. Stochastic.

4. **target_position** (positional emphasis): Applies a Gaussian attention window (sigma=0.3) centered at `target_position` along the parameter index, blending parameters outside the window toward midpoint. Deterministic.

5. **shuffle_window** (local reordering): Permutes parameter values within non-overlapping windows of size `shuffle_window * n_params`, clipping shuffled values to bounds. Stochastic.

### Score extraction

- The profiler extracts a scalar score from simulation results using priority lookup: `"fitness"` > `"score"` > `"y"` > mean of all finite numeric values.

---

## Basic Examples

Create a profiler and run a locality analysis:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.locality_profiler import LocalityProfiler

>>> def weighted_sum(p):
...     return {"score": 3*p["a"] + 2*p["b"] + 1*p["c"]}

>>> sim = SimulatorWrapper(weighted_sum, {"a": (0, 1), "b": (0, 1), "c": (0, 1)})
>>> profiler = LocalityProfiler(sim)

>>> report = profiler.profile(
...     task={"base_params": {"a": 0.8, "b": 0.5, "c": 0.3}},
...     n_seeds=10,
... )

>>> report["effective_horizon"]
0.0  # all params contribute, removing even 10% degrades performance

>>> report["L50"]["cut_frac"]
0.35  # cutting ~35% of prefix drops score to half baseline

>>> report["distractor_susceptibility"]
-2.8  # negative slope: score degrades with noise (normal)
```

Interpret the L50 values:

```python
>>> for manip, l50 in sorted(report["L50"].items()):
...     print(f"{manip}: L50 = {l50:.2f}")
cut_frac: L50 = 0.35
distractor_strength: L50 = 0.52
target_position: L50 = 0.90  # robust to positional emphasis
```

---

## Scope

Works with high-dimensional parameter spaces:

```python
>>> spec = {f"x{i}": (0, 1) for i in range(20)}
>>> sim = SimulatorWrapper(lambda p: {"score": sum(p.values())}, spec)
>>> profiler = LocalityProfiler(sim)
>>> report = profiler.profile(
...     task={"base_params": {f"x{i}": 0.7 for i in range(20)}},
...     n_seeds=5,
... )
>>> report["n_sims"]  # cost depends on sweep schedule and n_seeds
250
```

Custom manipulations can replace or extend the defaults:

```python
>>> def reverse_params(params, spec, value, rng):
...     """Reverse the parameter ordering (custom manipulation)."""
...     names = sorted(params.keys())
...     vals = [params[n] for n in names]
...     n_reverse = int(value * len(names))
...     vals[:n_reverse] = vals[:n_reverse][::-1]
...     return {n: v for n, v in zip(names, vals)}

>>> profiler = LocalityProfiler(sim, manipulations={
...     "reverse_frac": reverse_params,
... })
```

Wrapping other zimmerman tools -- profile a `Falsifier` or `sobol_sensitivity` indirectly:

```python
>>> # Treat the profiler itself as a Simulator for Sobol analysis
>>> from zimmerman.sobol import sobol_sensitivity
>>> result = sobol_sensitivity(profiler, n_base=64)
>>> result["ST"]  # which manipulation has the highest total-order index?
```

---

## Applications

**ER robot: which weights are local vs global.** Profile a 12D robot simulator to determine which synaptic weights have local effects (small perturbation degrades performance) vs global effects (system tolerates large perturbation):

```python
from zimmerman.locality_profiler import LocalityProfiler

profiler = LocalityProfiler(robot_sim)
report = profiler.profile(
    task={"base_params": known_good_gait_weights},
    n_seeds=30,
)
# A low L50 for cut_frac means the alphabetically-early weights
# (e.g., w0, w1) are critical -- the system is "front-loaded."
# A high L50 means early weights can be removed without much loss.
print(f"Effective horizon: {report['effective_horizon']}")
print(f"Distractor susceptibility: {report['distractor_susceptibility']:.3f}")

# Compare L50 across manipulations to see which kind of context
# disruption the robot is most sensitive to:
for manip, l50 in report["L50"].items():
    print(f"  {manip}: L50 = {l50:.3f}")
```

**JGC mitochondrial simulator: which interventions have local vs systemic effects.** The 12D mitochondrial model has 6 intervention parameters and 6 patient parameters. Profiling reveals whether interventions act locally (affecting only nearby state variables) or systemically (affecting the entire trajectory):

```python
profiler = LocalityProfiler(mito_sim)
report = profiler.profile(
    task={"base_params": healthy_patient_baseline},
    sweeps={
        "mask_frac": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
        "distractor_strength": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    },
    n_seeds=50,
)
# A high mask_frac L50 means the system is robust to losing individual
# parameters -- interventions have redundant, systemic effects.
# A low mask_frac L50 means specific parameters are critical --
# interventions act locally.
```

---

## Properties & Relations

- **L50 is analogous to IC50 in pharmacology.** IC50 is the drug concentration that inhibits a biological response by 50%. L50 is the manipulation intensity that degrades simulator performance by 50%. Both quantify sensitivity to a perturbation along a continuous dose-response curve.
- **Connection to Sobol analysis.** Because `LocalityProfiler` satisfies the `Simulator` protocol, running `sobol_sensitivity(profiler)` performs a second-order analysis: Sobol indices over manipulation intensities. This reveals which *kind* of context disruption matters most and whether manipulations interact (e.g., whether cutting + distracting is worse than either alone).
- **Grounded in Zimmerman Ch. 2-3.** The five default manipulations operationalize the five failure modes of locality identified in Zimmerman (2025) Ch. 2: ablation (section 2.3), masking (section 2.4), distraction (section 2.5), positional emphasis (section 2.3), and local reordering (section 2.2.3). The effective_horizon metric directly measures the "effective context window" of section 2.3. The distractor_susceptibility metric quantifies the robustness of the "salioscape" from section 2.5.
- **Sequential composition of manipulations.** When `run()` is called with multiple non-zero manipulation values, they apply in iteration order. This means the order of manipulations in the dict matters -- an important subtlety when interpreting interaction effects in a Sobol-over-manipulations analysis.
- **Deterministic RNG seeding in run().** The `run()` method derives its RNG from a hash of the parameter values, ensuring identical inputs always produce identical outputs (satisfying the deterministic simulation invariant required by Sobol analysis).

---

## Possible Issues

- **Computational cost.** Total simulations = `n_seeds * (1 + sum(len(sweep_values) for each manipulation))`, plus separate distractor and cut sweeps if not included in the explicit schedule. For default settings with `n_seeds=30`, this is approximately 360-600 simulations. Set `n_seeds=5` for exploratory analysis and `n_seeds=100+` for publication-quality results.
- **Stochastic manipulations need n_seeds > 1.** The `mask_frac`, `distractor_strength`, and `shuffle_window` manipulations are stochastic. With `n_seeds=1`, the decay curves will be noisy and L50 estimates unreliable. Use `n_seeds >= 10` for meaningful statistics on stochastic manipulations. Deterministic manipulations (`cut_frac`, `target_position`) give identical results across seeds.
- **Midpoint baseline assumption.** The `run()` method (Simulator protocol) always uses midpoint base parameters. The `profile()` method uses user-specified `task["base_params"]`. These serve different purposes: `run()` is for composability (feeding to Sobol, Falsifier, etc.), while `profile()` is for direct analysis at a meaningful operating point. Results from the two methods are not directly comparable.
- **Alphabetical parameter ordering.** The `cut_frac` and `target_position` manipulations depend on alphabetical sorting of parameter names. Systems where parameter ordering carries no semantic meaning may find these manipulations less interpretable. Consider renaming parameters to encode a meaningful ordering (e.g., `a_weight0`, `b_weight1`, ...) or using custom manipulations.
- **Clipping artifacts.** Both `distractor_strength` and `shuffle_window` clip perturbed values to parameter bounds. For parameters near their bounds, clipping biases the perturbation distribution, potentially understating the effect of noise. This bias is small for parameters near midpoint but can be significant at extremes.

---

## Neat Examples

**Recursive analysis: profiling a profiler.** Because `LocalityProfiler` satisfies the Simulator protocol, you can wrap a profiler in another profiler to ask "how sensitive is the locality analysis itself to the manipulation parameters?":

```python
from zimmerman.locality_profiler import LocalityProfiler
from zimmerman.base import SimulatorWrapper

# Inner simulator
sim = SimulatorWrapper(
    lambda p: {"score": p["x"]**2 + p["y"]**0.5},
    {"x": (0, 1), "y": (0, 1)},
)

# First-order profiler (the system under study)
profiler1 = LocalityProfiler(sim)

# Second-order profiler (profiling the profiler itself)
profiler2 = LocalityProfiler(profiler1)

# The meta-profile reveals which manipulation is most sensitive
# to *its own* locality disruption -- a self-referential analysis
# in the spirit of Zimmerman's TALOT/OTTITT (§4.7.6)
meta_report = profiler2.profile(
    task={"base_params": {
        "cut_frac": 0.0,
        "mask_frac": 0.0,
        "distractor_strength": 0.0,
        "target_position": 0.5,
        "shuffle_window": 0.0,
    }},
    n_seeds=10,
)
print(meta_report["L50"])
```

---

## See Also

`sobol_sensitivity` | `PromptReceptiveField` | `Falsifier` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Ch. 2-3.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.* Wiley.
