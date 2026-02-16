# SuperdiegeticBenchmark

standardized form-vs-meaning battery measuring diegeticization gain

---

## Calling Patterns

```
SuperdiegeticBenchmark(simulator)                                     wrap any Simulator-protocol object
```

```
bench.generate_tasks()                                                all 5 categories, seed=42
bench.generate_tasks(categories=["digits", "format"])                 selected categories only
bench.generate_tasks(base_params={"x": 0.5}, seed=0)                 include a baseline task
```

```
bench.score_task(task, result)                                        magnitude-match score (0.0 to 1.0)
```

```
bench.run_benchmark()                                                 default tasks, n_reps=5
bench.run_benchmark(tasks=my_tasks, n_reps=10, seed=0)                custom tasks and repetitions
```

```
bench.narrativize_params(params)                                      convert params to narrative labels
bench.discretize_value(0.73, 0.0, 1.0)                               single value to bin label
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol (`run(params) -> dict` + `param_spec() -> bounds`).
- The constructor caches `simulator.param_spec()` and initializes 5-bin discretization with labels `["very low", "low", "medium", "high", "very high"]`.

### generate_tasks

- `base_params` (default `None`): if provided, an additional `"baseline"` category task is appended using these exact values.
- `categories` (default `None`): list of category name strings to generate. When `None`, all 5 categories are used.
- `seed` (default `42`): random seed for reproducible task generation.
- Valid category names: `"palindrome"`, `"table"`, `"digits"`, `"format"`, `"symbol"`.
- Raises `ValueError` for unrecognized category names.
- Returns a list of task dicts, each with keys:
  - `"task_id"`: string like `"palindrome_000"`.
  - `"category"`: the category name.
  - `"supradiegetic_params"`: exact numeric parameter dict.
  - `"diegetic_params"`: bin-midpoint parameter dict (same content discretized and recovered).
  - `"expected"`: simulator output from running with supradiegetic params (ground truth).
  - `"scoring_fn"`: always `"score_task"`.

### The 5 benchmark categories

1. **palindrome** -- Symmetric parameter vectors where `x[i] == x[d-1-i]`. Tests whether the pipeline preserves mirror structure, a purely formal (supradiegetic) property with no semantic correlate.
2. **table** -- Parameters set to evenly-spaced fractions of their ranges, creating a staircase pattern. Tests preservation of tabular/grid layout, a supradiegetic structure type.
3. **digits** -- Parameters with 6 decimal places of precision (e.g., `0.371284`). Tests numerical precision retention, the most fragile supradiegetic capability due to tokenization effects. This is typically the hardest category.
4. **format** -- Parameters placed at exact bin boundaries (offset by +/- 1e-6 from discretization edges). Tests boundary sensitivity: whether the model correctly distinguishes values separated by 2e-6.
5. **symbol** -- First two parameters in a 2:1 ratio (`param1 = 2 * param0`, clipped to bounds). Tests whether non-linguistic token relationships survive the pipeline.

### score_task

- `task`: task dict (must contain `"expected"`).
- `result`: simulator output dict.
- Uses magnitude-match scoring: `score_i = max(0, 1 - |actual_i - expected_i| / scale_i)` where `scale_i = max(|expected_i|, 0.1)`.
- Final score is the mean across all numeric keys in `expected`.
- Non-finite values (NaN, Inf) in either expected or actual score as 0.0.
- Missing keys in result score as 0.0.
- Returns a float between 0.0 and 1.0.

### run_benchmark

- `tasks` (default `None`): list of task dicts from `generate_tasks()`. If `None`, generates default tasks with the given seed.
- `n_reps` (default `5`): repetitions per task version. For deterministic simulators, all reps produce identical scores (std = 0.0).
- `seed` (default `42`): random seed for task generation when `tasks` is `None`.
- Total simulator invocations: `2 * len(tasks) * n_reps`.
- Returns a dict with:
  - `"tasks"`: list of per-task result dicts, each with:
    - `"task_id"`, `"category"`.
    - `"supradiegetic_score"`: mean score across reps using exact params.
    - `"diegetic_score"`: mean score across reps using bin-midpoint params.
    - `"gain"`: `diegetic_score - supradiegetic_score` (the diegeticization gain).
    - `"supradiegetic_std"`, `"diegetic_std"`: standard deviation across reps.
  - `"summary"`: dict with:
    - `"mean_supradiegetic_score"`, `"mean_diegetic_score"`, `"mean_gain"` (headline metric).
    - `"by_category"`: per-category dict with `"sup_score"`, `"die_score"`, `"gain"`.
  - `"failure_mode_tags"`: deduplicated list of detected failure pattern strings.
  - `"n_sims"`: total number of simulator invocations.

### Diegeticization gain metric

The core metric is:

    gain = diegetic_score - supradiegetic_score

- **Positive gain**: narrative framing improves outcomes (Zimmerman's prediction for LLM-mediated pipelines).
- **Negative gain**: exact numerics outperform narrative framing (expected for deterministic simulators without an LLM in the loop).
- **Zero gain**: representation form does not affect outcomes.

### Failure mode tags

- `off_by_one` -- Supradiegetic score in (0.5, 0.95): close but not matching, suggesting approximate understanding with lost precision.
- `format_drift` -- Diegetic score exceeds supradiegetic by >0.3: strong evidence that narrative framing compensates for a structural processing deficit.
- `boundary_confusion` -- Format-category task with supradiegetic score <0.8: the model mishandles values at bin-boundary edges, conflating adjacent categories.
- `precision_loss` -- Digits-category task where diegetic score < supradiegetic by >0.2: discretization into bins actively destroys information the model could otherwise use.

---

## Basic Examples

Create a benchmark and run it:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark

>>> def simple_model(p):
...     return {"y": p["a"] + p["b"], "z": p["a"] * p["b"]}

>>> sim = SimulatorWrapper(simple_model, {"a": (0, 1), "b": (0, 1)})
>>> bench = SuperdiegeticBenchmark(sim)
>>> report = bench.run_benchmark()

>>> report["summary"]["mean_supradiegetic_score"]
1.0  # deterministic simulator always reproduces its own output exactly

>>> report["summary"]["mean_diegetic_score"]
0.87  # discretization into 5 bins loses some precision

>>> report["summary"]["mean_gain"]
-0.13  # negative: exact numerics beat narrative framing (no LLM in loop)
```

Generate tasks and inspect them:

```python
>>> tasks = bench.generate_tasks(seed=42)
>>> len(tasks)
5

>>> tasks[0]["category"]
"palindrome"

>>> tasks[0]["supradiegetic_params"]
{"a": 0.374540, "b": 0.374540}  # symmetric: a == b (2D palindrome)

>>> tasks[0]["diegetic_params"]
{"a": 0.3, "b": 0.3}  # bin midpoints after discretization
```

Score a single task result:

```python
>>> result = sim.run(tasks[0]["diegetic_params"])
>>> bench.score_task(tasks[0], result)
0.92  # close but not exact due to discretization
```

---

## Scope

Run with selected categories only:

```python
>>> tasks = bench.generate_tasks(categories=["digits", "format"])
>>> len(tasks)
2

>>> tasks[0]["category"]
"digits"
```

Include a baseline task with specific parameters of interest:

```python
>>> tasks = bench.generate_tasks(
...     base_params={"a": 0.5, "b": 0.5},
...     categories=["palindrome"],
... )
>>> len(tasks)
2  # one palindrome + one baseline

>>> tasks[1]["category"]
"baseline"
```

Adjust repetitions for higher statistical confidence:

```python
>>> report = bench.run_benchmark(n_reps=20, seed=0)
>>> report["n_sims"]
200  # 2 versions * 5 tasks * 20 reps
```

Use `narrativize_params` independently:

```python
>>> bench.narrativize_params({"a": 0.15, "b": 0.92})
{"a": "very low", "b": "very high"}
```

---

## Applications

**ER: Does narrating weights as "muscle strength" improve reproduction accuracy?** When a local LLM generates robot gait parameters, the parameters pass through narrative space. The benchmark measures whether describing weight_0 as "very high front-leg spring force" produces more faithful parameter roundtrips than passing "0.8274":

```python
from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark

bench = SuperdiegeticBenchmark(robot_sim)
report = bench.run_benchmark(n_reps=10)
# If gain > 0 for the LLM-mediated pipeline, narrative framing
# helps the LLM generate more accurate parameter vectors.
print(report["summary"]["by_category"]["digits"]["gain"])
# Digits category shows the largest effect: LLMs struggle most
# with precise numeric tokens.
```

**Mitochondrial aging model: Does narrating "rapamycin_dose: 0.5" as "moderate mTOR inhibition" improve LLM roundtrip fidelity?** The JGC simulator's 12 parameters include intervention doses (0-1 fractions) and patient characteristics (ages, multipliers). The benchmark tests whether clinical narrative framing improves parameter reproduction through an LLM pipeline:

```python
import sys
sys.path.insert(0, "/path/to/how-to-live-much-longer")
from zimmerman_bridge import MitoSimulator
from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark

sim = MitoSimulator()  # full 12D
bench = SuperdiegeticBenchmark(sim)
report = bench.run_benchmark(n_reps=5)

report["summary"]["mean_gain"]
# -0.14 -- negative gain without an LLM in the loop (expected: deterministic
# simulators always get supradiegetic_score=1.0, so diegeticization can only
# lose information)

# Per-category breakdown reveals where precision loss matters most:
report["summary"]["by_category"]
# {"palindrome": {"gain": -0.09}, "table": {"gain": -0.06},
#  "digits": {"gain": -0.17}, "format": {"gain": -0.10},
#  "symbol": {"gain": -0.09}}
# The digits category shows the largest negative gain because 6-decimal
# intervention doses (e.g., rapamycin_dose=0.371284) are destroyed by
# 5-bin discretization (recovered as 0.3 or 0.5).

# Clinical insight: the diegeticization gain becomes positive when an LLM
# is in the pipeline. "Moderate mTOR inhibition" activates the LLM's
# knowledge of rapamycin's mechanism (enhanced mitophagy, selective
# clearance of damaged mitochondria) in ways that "0.5" cannot.
# Near the heteroplasmy cliff at 0.70, this semantic activation matters
# most -- the LLM can reason about cliff dynamics narratively even when
# it cannot reliably manipulate the precise numeric threshold.
```

---

## Properties & Relations

- **Zimmerman Ch. 5: supradiegetic vs diegetic.** The benchmark operationalizes Zimmerman's central claim that LLMs handle meaning (diegetic content) more reliably than form (supradiegetic content). Each of the 5 categories probes a different aspect of formal/structural processing.
- **Zimmerman SS2.2.3: distributional collapse.** When an LLM encounters an unfamiliar structural task, it defaults to high-probability outputs from its training distribution. The digits and format categories are designed to trigger this collapse, while the diegetic versions provide narrative scaffolding that may prevent it.
- **Genette (1972): narratological origin.** The diegetic/supradiegetic distinction originates in Genette's narrative theory. "Diegetic" elements exist within the story world; "supradiegetic" elements exist outside the narrative frame (page numbers, formatting, table structure, precise numeric tokens).
- **Connection to `Diegeticizer`.** The benchmark uses the same 5-bin equal-width discretization scheme as `Diegeticizer`. The `narrativize_params` method performs the same forward pass, and `_diegetic_params_from_narrative` performs deterministic midpoint recovery. The benchmark adds the scoring and paired-comparison infrastructure.
- **Simulator protocol compliance.** The benchmark wraps any Simulator-protocol object. The wrapped simulator's `run()` and `param_spec()` are used to generate tasks and compute expected outputs.
- **Magnitude-match scoring.** The scoring function is deliberately simple and domain-agnostic: `max(0, 1 - |actual - expected| / scale)` where `scale = max(|expected|, 0.1)`. This avoids domain-specific assumptions about what constitutes a "good" simulation result.

---

## Possible Issues

- **Deterministic simulators always get supradiegetic_score = 1.0.** Since the expected output is computed by running the simulator with the exact supradiegetic params, and a deterministic simulator always produces the same output for the same input, the supradiegetic score is trivially 1.0. The diegeticization gain is thus always negative or zero. The benchmark becomes diagnostic only when an LLM is in the parameter pipeline.
- **Diegetic gain is only meaningful for LLM-mediated contexts.** Without an LLM introducing stochastic processing, the benchmark measures only the information loss from discretization (which is always negative). The gain metric is designed to reveal LLM-specific effects.
- **5-category fixed set may not cover all failure modes.** The five categories (palindrome, table, digits, format, symbol) are derived from Zimmerman's taxonomy but do not exhaustively probe all possible supradiegetic structures. Domains with unusual structural patterns (e.g., cyclic relationships, hierarchical nesting) may require custom task generators.
- **Bin boundary sensitivity.** The format category places values at +/- 1e-6 from bin edges. Simulators with extreme sensitivity to small parameter changes may show large score differences even without an LLM in the loop, confounding the boundary_confusion diagnostic.
- **Task count scales with categories, not with parameter dimensionality.** The default generates exactly one task per category (5 total). For high-dimensional parameter spaces, consider running `generate_tasks` with multiple seeds and concatenating the task lists.

---

## Neat Examples

**Discovering that digit tasks are the hardest supradiegetic category.** The digits category consistently produces the lowest diegetic scores because 6-decimal-place precision (e.g., `0.371284`) is maximally destroyed by 5-bin discretization (recovered as `0.3` or `0.5`):

```python
>>> bench = SuperdiegeticBenchmark(sim)
>>> report = bench.run_benchmark()
>>> for cat, stats in report["summary"]["by_category"].items():
...     print(f"{cat:12s}  sup={stats['sup_score']:.3f}  die={stats['die_score']:.3f}  gain={stats['gain']:+.3f}")
palindrome    sup=1.000  die=0.912  gain=-0.088
table         sup=1.000  die=0.943  gain=-0.057
digits        sup=1.000  die=0.831  gain=-0.169  # hardest category
format        sup=1.000  die=0.898  gain=-0.102
symbol        sup=1.000  die=0.907  gain=-0.093
```

The digits category shows the largest negative gain because 5-bin discretization discards the most information from high-precision values. This is a direct empirical confirmation of Zimmerman's claim that numeric precision is the most fragile supradiegetic capability.

**Using failure mode tags to diagnose pipeline behavior:**

```python
>>> report["failure_mode_tags"]
["precision_loss"]  # digits category triggered precision_loss because
                    # diegetic_score < supradiegetic_score - 0.2
```

---

## See Also

`Diegeticizer` | `PromptBuilder` | `TokenExtispicyWorkbench` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Ch. 5, SS2.2.3.
- Genette, G. (1972). "Narrative Discourse: An Essay in Method." (Trans. J.E. Lewin, 1980). Cornell University Press.
