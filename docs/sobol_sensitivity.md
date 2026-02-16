# sobol_sensitivity

global sensitivity analysis via Saltelli sampling with Saltelli/Jansen estimators

---

## Calling Patterns

```
sobol_sensitivity(simulator)                                    default: N=256, all outputs
sobol_sensitivity(simulator, n_base=1024)                       higher accuracy
sobol_sensitivity(simulator, seed=0)                            different random seed
sobol_sensitivity(simulator, output_keys=["fitness", "energy"]) analyze specific outputs only
```

**Low-level functions** (used internally, available for custom pipelines):

```
saltelli_sample(n_base, d, rng)             generate N*(D+2) samples in [0,1]^d
rescale_samples(samples_01, bounds)         rescale from [0,1]^d to parameter ranges
sobol_indices(y_A, y_B, y_C)               compute S1 and ST from cross-matrix outputs
```

---

## Details and Options

- `simulator` must satisfy the `Simulator` protocol: `run(params) -> dict` and `param_spec() -> dict[str, (float, float)]`.
- `n_base` (default 256) is the base sample count N. Total simulations = N * (D + 2), where D is the number of parameters. Larger N gives more accurate indices at proportionally higher cost.
- `seed` (default 42) controls the random number generator for reproducibility.
- `output_keys` (default `None`) specifies which output keys to analyze. When `None`, all numeric keys from the first simulation result are used.
- Returns a dict with structure:
  - `"n_base"`: the base sample count used.
  - `"n_total_sims"`: total simulations run (N * (D + 2)).
  - `"parameter_names"`: list of parameter names from `param_spec()`.
  - `"output_keys"`: list of output keys analyzed.
  - `"<output_key>"`: for each output key, a dict with:
    - `"S1"`: dict of first-order Sobol indices `{param_name: float}`.
    - `"ST"`: dict of total-order Sobol indices `{param_name: float}`.
    - `"interaction"`: dict of interaction contributions `{param_name: float}` (ST - S1).
  - `"rankings"`: dict with sorted parameter lists:
    - `"<key>_most_influential_S1"`: parameters sorted by S1 (descending).
    - `"<key>_most_interactive"`: parameters sorted by interaction (descending).

### saltelli_sample

- Produces N * (D + 2) samples using the Saltelli (2010) scheme.
- Layout: rows 0..N-1 = matrix A, rows N..2N-1 = matrix B, rows 2N..end = D cross-matrices C^(i) where C^(i) = A with column i replaced from B.
- All values are in [0, 1]^d (unit hypercube).

### rescale_samples

- Linear rescaling: `value = lo + sample_01 * (hi - lo)`.
- `bounds` is shape (D, 2) with `[lo, hi]` for each parameter.

### sobol_indices

- Uses the Saltelli (2010) S1 estimator: `V_i = E[f(B) * (f(C_i) - f(A))]`.
- Uses the Jansen (1999) ST estimator: `VT_i = 0.5 * E[(f(A) - f(C_i))^2]`.
- Variance denominator is computed from the pooled A and B samples: `Var(concat(y_A, y_B))`.
- Returns zeros when total variance < 1e-12 (constant output).

---

## Basic Examples

Analyze a simple 2-parameter model:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.sobol import sobol_sensitivity

>>> def model(p):
...     return {"y": 3 * p["a"] + p["b"] ** 2}

>>> sim = SimulatorWrapper(model, {"a": (0.0, 1.0), "b": (0.0, 1.0)})
>>> result = sobol_sensitivity(sim, n_base=512)

>>> result["n_total_sims"]
2048

>>> result["y"]["S1"]["a"]    # a has strong linear effect
0.68  # approximately

>>> result["y"]["S1"]["b"]    # b has weaker but nonlinear effect
0.25  # approximately

>>> result["rankings"]["y_most_influential_S1"]
["a", "b"]
```

---

## Scope

Works with any number of parameters:

```python
>>> spec = {f"x{i}": (0.0, 1.0) for i in range(10)}
>>> def sum_model(p):
...     return {"total": sum(p.values())}
>>> sim = SimulatorWrapper(sum_model, spec)
>>> result = sobol_sensitivity(sim, n_base=128)
>>> result["n_total_sims"]
1536  # 128 * (10 + 2)
```

Handles multiple output keys simultaneously:

```python
>>> def multi_output(p):
...     return {"sum": p["a"] + p["b"], "product": p["a"] * p["b"]}
>>> sim = SimulatorWrapper(multi_output, {"a": (0, 1), "b": (0, 1)})
>>> result = sobol_sensitivity(sim)
>>> list(result["sum"]["S1"].keys())
["a", "b"]
>>> list(result["product"]["S1"].keys())
["a", "b"]
```

Analyze only specific outputs:

```python
>>> result = sobol_sensitivity(sim, output_keys=["sum"])
>>> result["output_keys"]
["sum"]
```

---

## Applications

**Full-Zoo Sobol Sensitivity Atlas.** Analyze all 116 robot gaits in the ER project to identify which synaptic weights drive each gait's behavior:

```python
for gait_name, gait_data in zoo.items():
    weights = gait_data["weights"]
    spec = {name: (val - 0.3, val + 0.3) for name, val in weights.items()}
    sim = SimulatorWrapper(run_trial, spec)
    result = sobol_sensitivity(sim, n_base=256)
    # result reveals which weights are critical for this gait
```

This campaign (328,192 simulations, ~7 hours) produces a sensitivity atlas revealing whether antifragile gaits have distributed S1 profiles (no single dominant weight) while knife-edge gaits concentrate sensitivity on one or two weights.

**Mitochondrial aging model.** Identify which patient parameters most strongly influence heteroplasmy trajectory:

```python
result = sobol_sensitivity(mito_sim, n_base=512)
# Expect: genetic_vulnerability has highest ST for het_final
# Expect: metabolic_demand has highest S1 for atp_final
```

---

## Properties & Relations

- **S1 interpretation.** S1_i measures the fraction of output variance due to parameter i alone (main effect). For a purely additive model (no interactions), `sum(S1) == 1.0`.
- **ST interpretation.** ST_i measures the fraction of output variance due to parameter i AND all its interactions. Always `ST_i >= S1_i`.
- **Interaction contribution.** `ST_i - S1_i` is the fraction due to interactions only. When `ST_i >> S1_i`, parameter i is primarily influential through interactions with other parameters.
- **Additivity check.** If `sum(S1) ≈ 1.0` and `ST ≈ S1` for all parameters, the model is approximately additive.
- **Saltelli sampling efficiency.** The scheme requires N * (D + 2) evaluations, not N * (2D + 2), because both S1 and ST are estimated from the same set of cross-matrices.
- `sobol_sensitivity` uses `sobol_indices` internally, which uses `saltelli_sample` and `rescale_samples`. These can be called directly for custom workflows (e.g., parallel execution, custom output extraction).
- Related to `ContrastiveGenerator.sensitivity_from_contrastives()`, which provides a complementary local sensitivity measure at specific parameter points.

---

## Possible Issues

- **Negative S1 values.** With small N, sampling noise can produce slightly negative S1 estimates. This is a known property of the Saltelli (2010) S1 estimator; increase N to resolve.
- **Computational cost.** Total simulations scale as N * (D + 2). For D = 10 and N = 256, that is 3,072 simulations per analysis. Budget accordingly.
- **Constant outputs.** If an output key has zero variance (identical across all samples), both S1 and ST are returned as zero vectors.
- **Non-numeric outputs.** Output keys with non-numeric values (strings, lists, None) are silently skipped.
- **Exception handling.** If `simulator.run()` raises an exception during any of the N * (D + 2) calls, `sobol_sensitivity` does not catch it. Use `Falsifier` first to verify that the simulator is stable across the parameter range.
- **Local vs. global.** Standard Sobol analysis is global: it sweeps the full parameter range defined by `param_spec()`. For local sensitivity around a specific operating point, construct a `SimulatorWrapper` with narrow bounds centered on the point of interest.

---

## Neat Examples

**Detecting hidden interactions.** A model where two parameters interact multiplicatively:

```python
>>> def interaction_model(p):
...     return {"y": p["a"] * p["b"]}  # pure interaction, no main effects
>>> sim = SimulatorWrapper(interaction_model, {"a": (0, 1), "b": (0, 1)})
>>> r = sobol_sensitivity(sim, n_base=1024)
>>> r["y"]["S1"]   # both near zero (no main effects)
{"a": 0.01, "b": 0.01}
>>> r["y"]["ST"]   # both large (strong interaction)
{"a": 0.50, "b": 0.50}
```

**Identifying a "master weight."** In robot gaits, some weights have S1 > 0.2 while others are near zero, revealing the control hierarchy:

```python
# Gait 5_pelton (antifragile): w24=0.283, w14=0.275, w04=0.168
# Gait 86_breton_nadja:         w24=0.325, w44=0.170, w03=0.158
# Hip joint weights (w*4) consistently dominate for forward walkers
```

---

## See Also

`Simulator` | `SimulatorWrapper` | `Falsifier` | `ContrastiveGenerator`

---

## References

- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model output." *Computer Physics Communications*, 181(2), 259-270.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
