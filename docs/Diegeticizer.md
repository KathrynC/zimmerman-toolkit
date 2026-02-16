# Diegeticizer

reversible translation between parameter vectors and narrative descriptions

---

## Calling Patterns

```
Diegeticizer(simulator)                                               default: no lexicon, 5 bins
Diegeticizer(simulator, n_bins=3)                                     coarse 3-bin discretization
Diegeticizer(simulator, n_bins=7)                                     fine 7-bin discretization
Diegeticizer(simulator, lexicon={"x0": "sprint speed"})               with domain story handles
Diegeticizer(simulator, lexicon=lex, n_bins=5)                        lexicon + explicit bin count
```

```
dieg.diegeticize(params)                                              forward pass: floats to narrative
```

```
dieg.re_diegeticize(narrative)                                        reverse pass: deterministic midpoints
dieg.re_diegeticize(narrative, mode="sample", seed=0)                 reverse pass: uniform sample within bin
```

```
dieg.roundtrip_error(params)                                          measure single-vector information loss
```

```
dieg.batch_roundtrip()                                                100 random samples, seed=42
dieg.batch_roundtrip(n_samples=500, seed=0)                           custom sample count
```

```
dieg.run(params)                                                      Simulator protocol: diegeticize + run
dieg.param_spec()                                                     Simulator protocol: delegates to wrapped sim
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol (`run(params) -> dict` + `param_spec() -> bounds`).
- `lexicon` (default `None`): dict mapping parameter names to narrative "story handles" (e.g., `{"spring_constant": "spring stiffness"}`). If `None`, parameter names are used directly. The lexicon must be invertible (no duplicate values) to support the reverse mapping.
- `n_bins` (default `5`): number of equal-width discretization bins per parameter.
- The constructor precomputes bin edges, midpoints, and widths for each parameter from `param_spec()`.

### Bin system

The bin system divides each parameter's range `[lo, hi]` into `n_bins` equal-width bins. Each bin is labeled with a narrative term.

**3 bins** -- `["low", "medium", "high"]`. Very coarse. Large roundtrip error but maximally stable labels. Best when LLM consistency matters more than precision.

**5 bins** (default) -- `["very low", "low", "medium", "high", "very high"]`. Moderate precision with well-known "Likert scale" semantics that LLMs encounter frequently in training data. The recommended balance between granularity and narrative stability.

**7 bins** -- `["very low", "low", "moderately low", "medium", "moderately high", "high", "very high"]`. Finer precision, but the "moderately low" and "moderately high" labels are less semantically distinct, approaching the distributional collapse threshold where LLMs confuse adjacent categories.

**Other bin counts** -- Generates generic numbered labels `["bin_0", "bin_1", ...]`. These lack semantic richness but maintain the structural protocol.

Bin assignment uses left-closed, right-open intervals for all bins except the last, which is closed on both sides:
- `bin_0: [edge_0, edge_1)`
- `bin_1: [edge_1, edge_2)`
- `bin_{n-1}: [edge_{n-1}, edge_n]`

### Lexicon system

The lexicon provides domain-specific story handles that activate an LLM's domain knowledge:

- Without lexicon: `{"x0": "high", "x1": "low"}`
- With lexicon `{"x0": "sprint speed", "x1": "armor thickness"}`: `{"sprint speed": "high", "armor thickness": "low"}`

Zimmerman SS3.5.3 argues these handles are not cosmetic: "high sprint speed" and "low armor" together activate the LLM's knowledge of fast, fragile agents -- semantic associations invisible in `{"x0": 0.82, "x1": 0.15}`.

The reverse lexicon is built lazily on first call to `re_diegeticize` and cached for subsequent lookups.

### diegeticize

- `params`: dict mapping parameter names to float values.
- Returns a dict with:
  - `"narrative"`: `{handle: bin_label, ...}` -- the narrative representation.
  - `"numeric"`: dict (copy of original params).
  - `"bins_used"`: `{param_name: {"label": str, "range": (bin_lo, bin_hi)}, ...}` -- diagnostic info showing which bin each parameter fell into.

### re_diegeticize

- `narrative`: dict mapping handles (or parameter names) to bin labels.
- `mode` (default `"deterministic"`): recovery mode.
  - `"deterministic"`: returns the bin midpoint for each label. Minimizes expected L2 error under a uniform prior. Produces the same output every time.
  - `"sample"`: draws uniformly at random from within each bin. Produces diverse parameter vectors consistent with the narrative description.
- `seed` (default `42`): random seed, used only when `mode="sample"`.
- Raises `ValueError` for unrecognized mode or unknown bin labels.
- Returns a dict with:
  - `"params"`: dict of recovered numeric values.
  - `"narrative"`: dict (copy of input narrative).
  - `"mode"`: str.
  - `"recovery_uncertainty"`: `{param_name: bin_width, ...}` -- the fundamental resolution limit of the narrative representation for each parameter.

### roundtrip_error

- `params`: dict mapping parameter names to float values.
- Performs the full roundtrip: `params -> diegeticize -> re_diegeticize(deterministic) -> params'`.
- Returns a dict with:
  - `"original"`: dict (copy of input params).
  - `"recovered"`: dict (deterministic midpoint recovery).
  - `"per_param_error"`: `{param_name: abs(original - recovered), ...}`.
  - `"total_error"`: float (L2 norm of range-normalized per-parameter errors).
  - `"max_error_param"`: str (parameter with the largest absolute error).
  - `"unrecoverable_params"`: list of parameter names where error exceeds half the bin width (should be empty for correct implementations).

### batch_roundtrip

- `n_samples` (default `100`): number of uniformly random parameter vectors to evaluate.
- `seed` (default `42`): random seed for reproducibility.
- Returns a dict with:
  - `"mean_total_error"`: float -- average information loss across the parameter space.
  - `"std_total_error"`: float -- variability of information loss.
  - `"per_param_mean_error"`: `{param_name: float, ...}`.
  - `"per_param_max_error"`: `{param_name: float, ...}` -- approaches half the bin width as n_samples increases.
  - `"unrecoverable_fraction"`: `{param_name: float, ...}` -- should be 0.0 for a correct implementation.
  - `"n_samples"`: int.

### run (Simulator protocol)

- `params`: dict mapping parameter names to float values.
- Pipeline: diegeticize params -> re_diegeticize to midpoints -> run underlying simulator on midpoint params.
- Returns the underlying simulator's output dict.
- Enables recursive interrogation: the Diegeticizer can be analyzed by Sobol sensitivity, Falsifier, ContrastiveGenerator, POSIWID, etc.

### param_spec (Simulator protocol)

- Delegates to the underlying simulator's `param_spec()`.
- The parameter bounds are identical -- diegeticization does not change the valid range, only the internal representation.

---

## Basic Examples

Create a diegeticizer and run the forward pass:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.diegeticizer import Diegeticizer

>>> def model(p):
...     return {"y": p["speed"] + p["armor"]}

>>> sim = SimulatorWrapper(model, {"speed": (0, 1), "armor": (0, 1)})
>>> dieg = Diegeticizer(sim, n_bins=5)

>>> result = dieg.diegeticize({"speed": 0.82, "armor": 0.15})
>>> result["narrative"]
{"speed": "very high", "armor": "very low"}

>>> result["bins_used"]["speed"]
{"label": "very high", "range": (0.8, 1.0)}
```

Run the reverse pass:

```python
>>> recovered = dieg.re_diegeticize(result["narrative"])
>>> recovered["params"]
{"speed": 0.9, "armor": 0.1}  # bin midpoints

>>> recovered["recovery_uncertainty"]
{"speed": 0.2, "armor": 0.2}  # bin width = 1.0 / 5 = 0.2
```

Measure roundtrip error:

```python
>>> rt = dieg.roundtrip_error({"speed": 0.82, "armor": 0.15})
>>> rt["per_param_error"]
{"speed": 0.08, "armor": 0.05}  # |0.82 - 0.9| and |0.15 - 0.1|

>>> rt["total_error"]
0.094  # L2 norm of range-normalized errors

>>> rt["max_error_param"]
"speed"
```

---

## Scope

Use a custom lexicon for domain-specific narrative handles:

```python
>>> lex = {"speed": "sprint velocity", "armor": "shield thickness"}
>>> dieg = Diegeticizer(sim, lexicon=lex, n_bins=5)
>>> result = dieg.diegeticize({"speed": 0.82, "armor": 0.15})
>>> result["narrative"]
{"sprint velocity": "very high", "shield thickness": "very low"}

>>> recovered = dieg.re_diegeticize(result["narrative"])
>>> recovered["params"]
{"speed": 0.9, "armor": 0.1}  # reverse lexicon maps handles back to param names
```

Compare 3-bin vs 5-bin vs 7-bin precision:

```python
>>> for n in [3, 5, 7]:
...     d = Diegeticizer(sim, n_bins=n)
...     stats = d.batch_roundtrip(n_samples=200)
...     print(f"n_bins={n}  mean_error={stats['mean_total_error']:.4f}")
n_bins=3  mean_error=0.1361
n_bins=5  mean_error=0.0816
n_bins=7  mean_error=0.0583
```

Use sample recovery mode for narrative-consistent diversity:

```python
>>> recovered1 = dieg.re_diegeticize({"speed": "high"}, mode="sample", seed=0)
>>> recovered2 = dieg.re_diegeticize({"speed": "high"}, mode="sample", seed=1)
>>> recovered1["params"]["speed"]
0.674  # random value within "high" bin [0.6, 0.8)

>>> recovered2["params"]["speed"]
0.718  # different random value, same bin
```

Batch roundtrip statistics:

```python
>>> stats = dieg.batch_roundtrip(n_samples=500, seed=0)
>>> stats["per_param_max_error"]
{"speed": 0.099, "armor": 0.099}  # approaches 0.1 (half bin width for 5 bins)

>>> stats["unrecoverable_fraction"]
{"speed": 0.0, "armor": 0.0}  # no errors exceed half bin width
```

---

## Applications

**ER: Narrating 6 weights as body-part descriptions.** The evolutionary robotics simulator uses 6 synapse weights that control the 3-link robot's gait. Raw weights like `0.82` mean nothing to an LLM, but "very high front-leg spring force" activates biomechanical knowledge:

```python
from zimmerman.diegeticizer import Diegeticizer

robot_lexicon = {
    "w0": "front-leg spring force",
    "w1": "front-leg damping",
    "w2": "mid-joint torque limit",
    "w3": "mid-joint flexibility",
    "w4": "rear-leg spring force",
    "w5": "rear-leg damping",
}
dieg = Diegeticizer(robot_sim, lexicon=robot_lexicon, n_bins=5)

# Forward: weights -> narrative
result = dieg.diegeticize({"w0": 0.82, "w1": 0.15, "w2": 0.97,
                           "w3": 0.45, "w4": 0.33, "w5": 0.68})
# result["narrative"] == {
#   "front-leg spring force": "very high",
#   "front-leg damping": "very low",
#   "mid-joint torque limit": "very high",
#   "mid-joint flexibility": "medium",
#   "rear-leg spring force": "low",
#   "rear-leg damping": "high",
# }

# Measure diegeticization cost for the 6-weight space
stats = dieg.batch_roundtrip(n_samples=500)
print(f"Mean roundtrip error across 6D weight space: {stats['mean_total_error']:.4f}")
```

**JGC: Narrating intervention protocols as clinical language.** The mitochondrial aging simulator's intervention parameters translate naturally into medical vocabulary:

```python
mito_lexicon = {
    "rapamycin_dose": "mTOR inhibition intensity",
    "nad_supplement": "NAD+ precursor dosage",
    "exercise_level": "daily exercise intensity",
    "caloric_restriction": "caloric deficit severity",
    "metformin_dose": "AMPK activation level",
    "antioxidant_dose": "ROS scavenging capacity",
}
dieg = Diegeticizer(mito_sim, lexicon=mito_lexicon, n_bins=5)

result = dieg.diegeticize({"rapamycin_dose": 0.5, "nad_supplement": 0.8,
                           "exercise_level": 0.3, "caloric_restriction": 0.6,
                           "metformin_dose": 0.1, "antioxidant_dose": 0.9})
# "mTOR inhibition intensity": "medium"
# "NAD+ precursor dosage": "very high"
# etc.

# Roundtrip: how much precision is lost?
rt = dieg.roundtrip_error({"rapamycin_dose": 0.5, "nad_supplement": 0.8,
                           "exercise_level": 0.3, "caloric_restriction": 0.6,
                           "metformin_dose": 0.1, "antioxidant_dose": 0.9})
print(f"Total roundtrip error: {rt['total_error']:.4f}")
```

---

## Properties & Relations

- **Zimmerman SS3.5.3: narrative flattening.** Diegeticization systematically quantifies the information loss described in SS3.5.3. A parameter vector like `[0.82, 0.15, 0.97]` contains exact positional and magnitude information; its narrative equivalent "high speed, very low armor, very high damage" preserves ordinal relationships but discards cardinal precision. The `roundtrip_error` method measures this loss precisely.
- **Roundtrip fidelity as information theory.** The roundtrip `params -> narrative -> params'` is a lossy compression-decompression cycle. The `total_error` is the distortion, and the bin count controls the rate-distortion tradeoff. Fewer bins = higher distortion but more robust narrative labels. More bins = lower distortion but weaker semantic discriminability.
- **Connection to `SuperdiegeticBenchmark`.** The benchmark uses the same discretization scheme as the Diegeticizer and adds paired scoring infrastructure. The Diegeticizer measures information loss in isolation; the benchmark measures whether that loss is compensated by improved LLM processing of narrative content.
- **Simulator protocol enables measuring "diegeticization cost."** Because `Diegeticizer` satisfies the Simulator protocol via `run()` and `param_spec()`, you can pass it directly to `sobol_sensitivity()`, `Falsifier`, `ContrastiveGenerator`, or `POSIWIDAuditor`. This enables studying how diegeticization affects the simulator's behavior surface -- for example, whether diegeticization smooths out behavioral cliffs by snapping nearby parameter values to the same bin midpoint.
- **Equal-width binning by design.** The bins are equal-width (not equal-frequency) because the goal is maximal narrative contrast between adjacent labels, even though equal-frequency bins would be more statistically efficient.

---

## Possible Issues

- **Bin width bounds max recovery precision.** Each parameter's roundtrip error is at most half the bin width: `(hi - lo) / (2 * n_bins)`. For 5 bins on `[0, 1]`, this is 0.1. No recovery method (deterministic or sample) can reduce error below this fundamental resolution limit.
- **Lexicon must be invertible.** If two parameters map to the same story handle (e.g., both `"x0"` and `"x1"` map to `"speed"`), the reverse lexicon will silently shadow one mapping. The constructor does not check for this. Ensure all lexicon values are unique.
- **7 bins approaches raw numeric territory, reducing diegetic advantage.** With 7 bins, labels like "moderately low" and "moderately high" are less semantically distinct than the 5-bin labels. LLMs may confuse adjacent 7-bin labels more often than 5-bin labels, partially negating the precision gain. Zimmerman predicts a diminishing-returns curve where increasing bins past 5-7 degrades diegetic performance.
- **Parameters at exact bin boundaries.** A value exactly on a bin edge is assigned to the lower bin (left-closed, right-open intervals), except for the final bin. Values at `hi` are assigned to the last bin. This deterministic rule prevents ambiguity but means two values separated by 1e-15 can be assigned to different bins.
- **`re_diegeticize` raises `ValueError` for unknown labels.** If the narrative dict contains labels not in `self._labels` (e.g., from a different `n_bins` setting or a corrupted narrative), the method raises rather than silently defaulting. Ensure the same `n_bins` is used for both forward and reverse passes.
- **Sample mode is not reproducible across different narrative dicts.** The RNG state advances differently depending on how many parameters are in the narrative dict, so `seed=42` with 3 parameters and `seed=42` with 6 parameters produce different samples for the first parameter.

---

## Neat Examples

**Measuring exactly how much precision is lost when "rapamycin_dose: 0.73" becomes "high":**

```python
>>> sim = SimulatorWrapper(
...     lambda p: {"effect": p["rapamycin_dose"] ** 2},
...     {"rapamycin_dose": (0.0, 1.0)}
... )
>>> dieg = Diegeticizer(sim, n_bins=5)

>>> # Forward: 0.73 falls in bin [0.6, 0.8) -> "high"
>>> result = dieg.diegeticize({"rapamycin_dose": 0.73})
>>> result["narrative"]
{"rapamycin_dose": "high"}

>>> # Reverse: "high" -> midpoint 0.7
>>> recovered = dieg.re_diegeticize(result["narrative"])
>>> recovered["params"]
{"rapamycin_dose": 0.7}

>>> # Precision lost: |0.73 - 0.7| = 0.03
>>> rt = dieg.roundtrip_error({"rapamycin_dose": 0.73})
>>> rt["per_param_error"]["rapamycin_dose"]
0.03

>>> # Simulation effect: 0.73^2 = 0.5329, but 0.7^2 = 0.49
>>> # Diegeticization cost: |0.5329 - 0.49| = 0.0429
>>> sim.run({"rapamycin_dose": 0.73})["effect"]
0.5329

>>> dieg.run({"rapamycin_dose": 0.73})["effect"]
0.49  # the price of translating through narrative space
```

**Visualizing how bin count controls the precision-stability tradeoff:**

```python
>>> for n in [3, 5, 7]:
...     d = Diegeticizer(sim, n_bins=n)
...     fwd = d.diegeticize({"rapamycin_dose": 0.73})
...     rt = d.roundtrip_error({"rapamycin_dose": 0.73})
...     print(f"n_bins={n}  label={fwd['narrative']['rapamycin_dose']:>16s}"
...           f"  recovered={rt['recovered']['rapamycin_dose']:.4f}"
...           f"  error={rt['per_param_error']['rapamycin_dose']:.4f}")
n_bins=3  label=            high  recovered=0.8333  error=0.1033
n_bins=5  label=            high  recovered=0.7000  error=0.0300
n_bins=7  label=  moderately high  recovered=0.7143  error=0.0157
```

---

## See Also

`SuperdiegeticBenchmark` | `PromptBuilder` | `TokenExtispicyWorkbench` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. SS3.5.3, SS2.2.3.
