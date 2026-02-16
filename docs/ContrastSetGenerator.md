# ContrastSetGenerator

edit-path contrast sets with binary-search tipping point identification

---

## Calling Patterns

```
ContrastSetGenerator(simulator)                                          default outcome: fitness > 0
ContrastSetGenerator(simulator, outcome_fn=my_classifier)                custom outcome classifier

gen.generate_edit_path(base_params, target_params)                       targeted interpolation path
gen.generate_edit_path(base_params)                                      random perturbation path
gen.generate_edit_path(base_params, n_edits=50, seed=0)                  finer resolution, different seed

gen.find_tipping_point(base_params, edit_path)                           binary search for flip
gen.find_tipping_point(base_params, edit_path, max_sims=200)             with simulation budget

gen.batch_contrast_sets(base_params)                                     default: 10 paths, 20 edits each
gen.batch_contrast_sets(base_params, n_paths=50, n_edits=40, seed=0)     larger campaign

gen.run(params)                                                          Simulator protocol (calls batch_contrast_sets)
gen.param_spec()                                                         delegates to underlying simulator
```

---

## Details and Options

- `simulator` must satisfy the `Simulator` protocol: `run(params) -> dict` and `param_spec() -> dict[str, (float, float)]`.
- `outcome_fn` (default `None`) is a callable taking a result dict and returning a hashable outcome class. When `None`, the default classifier returns `"positive"` if `result.get("fitness", 0) > 0` else `"negative"`. This function defines the qualitative classification that the generator tries to flip -- it is the "Q" in Lipton's "Why P rather than Q?" formulation.

### generate_edit_path

- `base_params` is the starting parameter dict.
- `target_params` (default `None`) is the target parameter dict. When provided, the path distributes the total difference `(target - base)` across `n_edits` steps via round-robin assignment to parameters (targeted interpolation mode). When `None`, random perturbations of 1-20% of each parameter's range are generated (random perturbation mode).
- `n_edits` (default 20) is the number of micro-edits in the path. More edits give finer resolution for tipping point localization.
- `seed` (default 42) controls the random number generator.
- Returns a list of edit dicts, sorted by absolute magnitude (smallest first). Each edit dict contains:
  - `"param"`: str -- name of the parameter being adjusted.
  - `"delta"`: float -- signed change to apply.
  - `"description"`: str -- human-readable summary (e.g., `"Adjust w24 by +0.015000 (3.0% of range)"`).

### find_tipping_point

- `base_params` is the starting parameter dict.
- `edit_path` is the ordered list of edit dicts from `generate_edit_path`. Must be sorted by magnitude for the binary search to be valid.
- `max_sims` (default 100) is the maximum number of simulator calls allowed. If the budget is exhausted before convergence, the best-known tipping point is returned.
- Returns a dict with:
  - `"found"`: bool -- whether a tipping point (outcome flip) was found.
  - `"base_outcome"`: hashable -- outcome at `base_params` (the "P" in "Why P rather than Q?").
  - `"flipped_outcome"`: hashable or `None` -- outcome after flip (the "Q"), `None` if no flip found.
  - `"tipping_k"`: int or `None` -- smallest k such that applying `edit_path[0:k]` flips the outcome.
  - `"flip_size"`: float or `None` -- `tipping_k / len(edit_path)`, normalized to [0, 1]. Small `flip_size` indicates a fragile configuration (close to a decision boundary).
  - `"tipping_params"`: dict or `None` -- the parameter configuration at the flip point.
  - `"edit_path_applied"`: list -- `edit_path[0:tipping_k]`, the edits that were applied to reach the tipping point.
  - `"n_sims"`: int -- total simulator calls made.

### batch_contrast_sets

- `base_params` is the starting (reference) parameter dict.
- `n_paths` (default 10) is the number of distinct random edit paths to generate and test. More paths give better coverage and more robust tipping frequency estimates.
- `n_edits` (default 20) is the number of edits per path.
- `seed` (default 42) is the base random seed. Path i uses seed `(seed + i)`.
- Returns a dict with:
  - `"pairs"`: list of `find_tipping_point` result dicts (one per path).
  - `"mean_flip_size"`: float or `None` -- average `flip_size` across paths where a flip was found. Small values indicate the base configuration is generally fragile. `None` if no flips were found on any path.
  - `"param_tipping_frequency"`: `{param_name: float}` -- fraction of found flips where each parameter was the tipping edit (the "last straw"). Frequencies sum to 1.0 across all parameters.
  - `"most_fragile_params"`: list of param names sorted by tipping frequency (descending). The first element is the parameter most often responsible for outcome flips.
  - `"n_sims"`: int -- total simulator calls across all paths.

### run / param_spec

- `run(params)` calls `batch_contrast_sets(params)` with default settings, making `ContrastSetGenerator` itself satisfy the `Simulator` protocol for meta-analysis composability.
- `param_spec()` delegates to the underlying simulator's `param_spec()`.

---

## Basic Examples

Create a generator and find a tipping point on a simple step model:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.contrast_set_generator import ContrastSetGenerator

>>> def step_model(p):
...     return {"fitness": 1.0 if p["x"] + p["y"] > 1.0 else -1.0}

>>> sim = SimulatorWrapper(step_model, {"x": (0.0, 1.0), "y": (0.0, 1.0)})
>>> gen = ContrastSetGenerator(sim)

>>> base = {"x": 0.3, "y": 0.3}
>>> target = {"x": 0.8, "y": 0.8}
>>> path = gen.generate_edit_path(base, target, n_edits=20)
>>> len(path)
20

>>> result = gen.find_tipping_point(base, path)
>>> result["found"]
True
>>> result["base_outcome"]
"negative"
>>> result["flipped_outcome"]
"positive"
>>> result["flip_size"]  # fraction of path needed to flip
0.55  # approximately
```

Run a batch analysis across multiple random paths:

```python
>>> batch = gen.batch_contrast_sets(base, n_paths=20, n_edits=30)
>>> batch["mean_flip_size"]
0.42  # approximately -- the base is moderately fragile
>>> batch["most_fragile_params"][:2]
["x", "y"]  # both parameters equally contribute to flips (symmetric model)
>>> batch["n_sims"]
180  # approximately -- O(n_paths * log2(n_edits)) total cost
```

---

## Scope

Custom outcome functions enable multi-class contrast analysis:

```python
>>> def gait_classifier(result):
...     d = result.get("distance", 0)
...     if d > 2.0: return "walker"
...     elif d > 0.5: return "shuffler"
...     else: return "faller"

>>> gen = ContrastSetGenerator(sim, outcome_fn=gait_classifier)
>>> result = gen.find_tipping_point(walker_params, path)
>>> result["base_outcome"]
"walker"
>>> result["flipped_outcome"]
"shuffler"  # first regime transition encountered along the path
```

Targeted edit paths reconstruct the exact target when fully applied:

```python
>>> path = gen.generate_edit_path(base, target, n_edits=10)
>>> # Applying all edits reconstructs target (modulo clipping)
>>> all_applied = gen._apply_edits(base, path)
>>> abs(all_applied["x"] - target["x"]) < 1e-10
True
```

Different `n_paths` values trade coverage for speed:

```python
>>> quick = gen.batch_contrast_sets(base, n_paths=5)    # fast survey
>>> thorough = gen.batch_contrast_sets(base, n_paths=100) # robust statistics
>>> thorough["n_sims"]  # ~100 * (2 + log2(20)) evaluations
850  # approximately
```

---

## Applications

**ER: Which weight change flips a gait from walking to falling?** Identify the most fragile synaptic weights for a champion robot gait:

```python
gen = ContrastSetGenerator(robot_sim, outcome_fn=lambda r: "walker" if r["distance"] > 2.0 else "faller")

# Analyze gait 5_pelton -- a known antifragile walker
champion_weights = zoo["5_pelton"]["weights"]
batch = gen.batch_contrast_sets(champion_weights, n_paths=50, n_edits=30)

# Which weights are most often the "last straw"?
print(batch["most_fragile_params"])
# ["w24", "w14", "w04"] -- hip joint weights dominate
print(batch["mean_flip_size"])
# 0.68 -- antifragile gait requires large perturbation to break

# Compare with a knife-edge gait
knife_weights = zoo["86_breton_nadja"]["weights"]
batch_knife = gen.batch_contrast_sets(knife_weights, n_paths=50)
print(batch_knife["mean_flip_size"])
# 0.15 -- knife-edge gait is fragile, near many decision boundaries
```

**JGC: Which intervention change crosses the heteroplasmy cliff?** The mitochondrial aging simulator has a critical nonlinearity at ~70% heteroplasmy. Edit-path analysis locates which parameter is most often the "last straw" pushing a patient past the cliff:

```python
from zimmerman.contrast_set_generator import ContrastSetGenerator
from zimmerman_bridge import MitoSimulator

sim = MitoSimulator()  # full 12D

def cliff_outcome(result):
    het = result.get("final_heteroplasmy", 0.5)
    return "collapsed" if het >= 0.70 else "healthy"

gen = ContrastSetGenerator(sim, outcome_fn=cliff_outcome)

base_params = {"rapamycin_dose": 0.0, "nad_supplement": 0.0, "senolytic_dose": 0.0,
               "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.0,
               "baseline_age": 70.0, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6,
               "genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.25}

result = gen.batch_contrast_sets(base_params, n_paths=10, n_edits=20, seed=42)
# result["mean_flip_size"] → 0.375; most fragile param: baseline_heteroplasmy
# 10 tipping points found, mean flip size=0.375
# baseline_heteroplasmy is most often the decisive edit crossing the cliff
```

---

## Properties & Relations

- **TALOT/OTTITT (Zimmerman, 2025, S4.7.6).** `ContrastSetGenerator` operationalizes OTTITT ("One Thing Turns Into The Other Thing"): the edit path traces how one behavioral regime transforms into another through incremental parameter changes. The tipping point is where the transformation becomes qualitative. TALOT ("Things Are Like Other Things") is the complementary perspective addressed by `ContrastiveGenerator`.
- **Contrastive explanation (Lipton, 1990).** The generator answers "Why P rather than Q?" by identifying the minimal edit sequence that changes the outcome from P to Q. The tipping edit is the *difference-maker* -- the minimal contrastive factor between the last non-flipped and first flipped configuration.
- **Catastrophe theory (Thom, 1975).** The tipping point is a catastrophe fold: a smooth change in control parameters (the edit path) produces a discontinuous jump in system behavior (the outcome class flip). The binary search is a numerical method for locating these fold points along one-dimensional paths through high-dimensional parameter space.
- **Binary search efficiency.** `find_tipping_point` requires O(log2(n_edits)) simulator evaluations, compared to O(n_edits) for linear search. For `n_edits=20`, this is approximately 7 evaluations (including base, full, and verification).
- **Complementary to Sobol analysis.** `param_tipping_frequency` from `batch_contrast_sets` measures boundary-proximity importance: how often a parameter is the decisive factor at a decision boundary. This is complementary to Sobol indices from `sobol_sensitivity`, which measure variance contribution. A parameter with low Sobol S1 but high tipping frequency is one that rarely matters -- except at critical thresholds.
- **Simulator protocol composability.** Because `ContrastSetGenerator` satisfies the `Simulator` protocol (`run` + `param_spec`), it can be wrapped in another Zimmerman tool. For example, running `sobol_sensitivity` on a `ContrastSetGenerator` measures how much each parameter's *tipping frequency* varies across the parameter space.

---

## Possible Issues

- **Binary search assumes monotonic tipping.** The search assumes that if `edits[0:lo]` produces the base outcome and `edits[0:hi]` produces a flipped outcome, then the tipping point lies in `[lo, hi]`. This can fail if the outcome oscillates along the edit path (e.g., the system flips, then flips back). In such cases, the binary search may locate one of several tipping points, not necessarily the first.
- **Edit path order matters.** Edits are sorted by absolute magnitude (smallest first). This means the path always applies the smallest perturbations first. A different sorting criterion (e.g., by parameter importance) would produce a different path and potentially a different tipping point.
- **`flip_size` is `None` when no flip is found.** If the full edit path does not change the outcome class, `find_tipping_point` returns `"found": False` with `flip_size=None`. Check for `None` before aggregating.
- **`mean_flip_size` is `None` when no paths flip.** If no edit path in a batch produces a flip, `batch_contrast_sets` returns `mean_flip_size=None`. This indicates the base configuration is robustly within a single outcome region in all sampled directions.
- **Parameter clipping at bounds.** Cumulative edit application clips parameter values to `[lo, hi]` from `param_spec()`. Near parameter bounds, edits may be absorbed by clipping, effectively reducing the perturbation magnitude and potentially masking tipping points.
- **Random mode perturbation range.** In random mode, each edit perturbs a single parameter by 1-20% of its range. For simulators with very different sensitivities across parameters, this uniform range may be suboptimal -- highly sensitive parameters may over-flip while insensitive ones under-flip.
- **Computational cost.** `batch_contrast_sets` with `n_paths=50` and `n_edits=20` requires approximately `50 * (2 + log2(20)) ~ 350` simulator evaluations. Budget accordingly for expensive simulators.

---

## Neat Examples

**Mapping the fragility boundary of a champion gait.** Sweep the number of edit paths to build a convergence curve for tipping frequency, revealing when the estimate stabilizes:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.contrast_set_generator import ContrastSetGenerator

>>> gen = ContrastSetGenerator(robot_sim, outcome_fn=gait_classifier)
>>> champion = zoo["5_pelton"]["weights"]

>>> # Convergence analysis: how many paths are needed for stable estimates?
>>> for n in [5, 10, 25, 50, 100]:
...     batch = gen.batch_contrast_sets(champion, n_paths=n, seed=0)
...     top = batch["most_fragile_params"][0]
...     freq = batch["param_tipping_frequency"][top]
...     print(f"n_paths={n:3d}  top_param={top}  freq={freq:.3f}  "
...           f"mean_flip={batch['mean_flip_size']:.3f}")
n_paths=  5  top_param=w24  freq=0.600  mean_flip=0.631
n_paths= 10  top_param=w24  freq=0.500  mean_flip=0.645
n_paths= 25  top_param=w24  freq=0.440  mean_flip=0.658
n_paths= 50  top_param=w24  freq=0.420  mean_flip=0.661
n_paths=100  top_param=w24  freq=0.415  mean_flip=0.663
# w24 (hip joint weight) is consistently the most fragile parameter
# mean_flip_size converges near 0.66 -- this gait is antifragile
```

**Comparing fragility across the entire gait zoo.** Batch analysis reveals the spectrum from antifragile to knife-edge:

```python
>>> for name, data in sorted(zoo.items()):
...     batch = gen.batch_contrast_sets(data["weights"], n_paths=20)
...     if batch["mean_flip_size"] is not None:
...         label = "antifragile" if batch["mean_flip_size"] > 0.5 else "knife-edge"
...         print(f"{name:30s}  flip={batch['mean_flip_size']:.2f}  {label}")
5_pelton                        flip=0.66  antifragile
12_cosette                      flip=0.58  antifragile
86_breton_nadja                 flip=0.14  knife-edge
91_wraith                       flip=0.08  knife-edge
# Small flip_size = close to catastrophe fold = fragile gait
```

---

## See Also

`ContrastiveGenerator` | `Falsifier` | `sobol_sensitivity` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Section 4.7.6 (TALOT/OTTITT).
- Lipton, P. (1990). "Contrastive Explanation." *Royal Institute of Philosophy Supplement*, 27, 247-266.
- Thom, R. (1975). *Structural Stability and Morphogenesis.* W.A. Benjamin.
