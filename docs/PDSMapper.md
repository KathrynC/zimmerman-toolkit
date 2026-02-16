# PDSMapper

maps abstract dimensions (Power, Danger, Structure) to concrete simulator parameters

---

## Calling Patterns

```
PDSMapper(simulator, dimension_names, dimension_to_param_mapping)   create a mapper
```

```
mapper.map_dimensions_to_params(dimension_values)               convert abstract → concrete
```

```
mapper.audit_mapping(n_samples=100, seed=42)                    audit how well dimensions predict outputs
```

```
mapper.sensitivity_per_dimension(n_samples=100, seed=42)        rank dimensions by influence
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol.
- `dimension_names`: list of abstract dimension names (e.g., `["power", "danger", "structure"]`).
- `dimension_to_param_mapping`: dict mapping each dimension name to a dict of `{param_name: weight}`. Weights determine how strongly each dimension influences each parameter.
- Validates that all mapped parameter names exist in `simulator.param_spec()`. Raises `ValueError` for unknown parameters.
- A single parameter can be influenced by multiple dimensions.
- Dimensions not mapped to any parameter are valid but have no effect.

### map_dimensions_to_params

- `dimension_values`: dict mapping dimension names to float values. Typical range is [-1, +1] but not enforced.
- The mapping for each simulator parameter is:
  ```
  value = midpoint + sum(dim_value * weight * half_range)
  ```
  where `midpoint = (lo + hi) / 2` and `half_range = (hi - lo) / 2`.
- Parameters not referenced by any dimension are set to their midpoint.
- Missing dimensions in `dimension_values` are treated as 0.0.
- All values are clipped to `[lo, hi]` bounds.
- Returns a complete parameter dict with all `param_spec()` keys.

### audit_mapping

- `n_samples` (default 100): number of random dimension settings to test.
- `seed` (default 42): random seed.
- Generates uniform random dimension values in [-1, +1], maps to parameters, runs the simulator, and measures correlations.
- Returns a dict with:
  - `"dimension_output_correlations"`: `{dim_name: {output_key: pearson_r}}`.
  - `"variance_explained"`: `{output_key: R^2}` from OLS linear fit of all dimensions to each output.
  - `"n_samples"`: number of samples tested.
  - `"dimension_stats"`: `{dim_name: {"mean": float, "std": float}}` of generated values.
  - `"output_keys"`: list of numeric output keys found.
- R^2 is computed via least-squares with intercept (`X_aug = [X | 1]`). Clipped to minimum 0.0.
- NaN values in outputs are excluded from correlation/regression computations.

### sensitivity_per_dimension

- Calls `audit_mapping()` internally.
- Returns `{dim_name: {output_key: abs_correlation, "overall": mean_abs_correlation}}`.
- "overall" is the mean absolute correlation across all output keys — a single-number importance score per dimension.

---

## Basic Examples

Create a PDS mapping for a robot simulator:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.pds import PDSMapper

>>> def robot(p):
...     return {"speed": p["motor"] * 2, "stability": 1 - p["height"] * 0.5}

>>> sim = SimulatorWrapper(robot, {"motor": (0.0, 1.0), "height": (0.0, 1.0)})

>>> mapper = PDSMapper(sim,
...     dimension_names=["aggression", "caution"],
...     dimension_to_param_mapping={
...         "aggression": {"motor": 0.5, "height": -0.3},
...         "caution":    {"motor": -0.3, "height": 0.4},
...     }
... )

>>> params = mapper.map_dimensions_to_params({"aggression": 0.8, "caution": -0.2})
>>> params["motor"]   # midpoint + 0.8*0.5*0.5 + (-0.2)*(-0.3)*0.5
0.73
>>> params["height"]  # midpoint + 0.8*(-0.3)*0.5 + (-0.2)*0.4*0.5
0.34
```

---

## Scope

Audit the mapping to verify dimensions actually predict simulator behavior:

```python
>>> audit = mapper.audit_mapping(n_samples=200)
>>> audit["dimension_output_correlations"]["aggression"]["speed"]
0.85  # strong positive: aggression → speed

>>> audit["variance_explained"]["speed"]
0.78  # dimensions explain 78% of speed variance

>>> audit["variance_explained"]["stability"]
0.42  # dimensions explain less stability variance → unmapped factors exist
```

Rank dimensions by overall influence:

```python
>>> sens = mapper.sensitivity_per_dimension(n_samples=200)
>>> sens["aggression"]["overall"]
0.72
>>> sens["caution"]["overall"]
0.45
# aggression is the more influential dimension
```

Parameters not mapped to any dimension get midpoint values:

```python
>>> sim2 = SimulatorWrapper(model, {"a": (0, 10), "b": (0, 10), "c": (0, 10)})
>>> mapper2 = PDSMapper(sim2, ["power"], {"power": {"a": 0.5}})
>>> params = mapper2.map_dimensions_to_params({"power": 1.0})
>>> params["a"]  # influenced by power
7.5
>>> params["b"]  # not mapped → midpoint
5.0
>>> params["c"]  # not mapped → midpoint
5.0
```

---

## Applications

**JGC mitochondrial simulator.** Map Zimmerman's (2025) PDS framework to patient parameters:

```python
mapper = PDSMapper(mito_sim,
    dimension_names=["power", "danger", "structure"],
    dimension_to_param_mapping={
        "power":     {"metabolic_demand": 0.4, "genetic_vulnerability": -0.2},
        "danger":    {"inflammation_level": 0.25, "genetic_vulnerability": 0.3},
        "structure": {"repair_capacity": 0.5, "membrane_integrity": 0.3},
    }
)

# A "high power, low danger" patient:
params = mapper.map_dimensions_to_params({"power": 0.9, "danger": -0.5, "structure": 0.3})
result = mito_sim.run(params)
```

**ER robot gait design.** Explore behavioral space through semantic dimensions:

```python
mapper = PDSMapper(robot_sim,
    dimension_names=["aggression", "stability", "symmetry"],
    dimension_to_param_mapping={
        "aggression": {"w03": 0.5, "w14": 0.4, "w24": 0.3},
        "stability":  {"w13": 0.4, "w23": -0.2},
        "symmetry":   {"w03": 0.3, "w13": -0.3, "w23": 0.3},
    }
)

# Sweep the aggression axis to find behavioral transitions:
for agg in np.linspace(-1, 1, 20):
    params = mapper.map_dimensions_to_params({"aggression": agg})
    result = robot_sim.run(params)
    # result["dx"] should increase with aggression
```

---

## Properties & Relations

- **PDS framework.** Power, Danger, Structure are the three most significant axes discovered through ousiometric analysis of word meanings (Dodds et al. 2023). Power runs from weak to powerful (nothing-something, *ex nihilo*). Danger runs from safe to dangerous (angels to demons). Structure runs from structured to unstructured (traditionalists to adventurers, stasis to mutation). These dimensions also align with Zimmerman's character-space analysis (§4.6.4): Fool↔Hero maps to Power, Angel↔Demon maps to Danger, Traditionalist↔Adventurer maps to Structure. The alignment is *emergent* — none of the PDS axis words appear in the character space rating data. The toolkit applies PDS to simulation parameter spaces, exploiting the fact that domain experts naturally reason in these terms.
- **Linear mapping.** The dimension-to-parameter mapping is linear: `value = midpoint + sum(dim * weight * half_range)`. Nonlinear mappings require extending the class.
- **Dimensionality reduction.** PDS maps a low-dimensional abstract space (typically 2-5 dimensions) to a high-dimensional parameter space (6-20 parameters). This is the reverse of typical dimensionality reduction: it *expands* meaningful axes into concrete parameters.
- **Relationship to Sobol.** `sensitivity_per_dimension` uses correlation, not Sobol indices. For linear mappings, correlation captures the relationship well. For nonlinear simulator responses, Sobol analysis on the PDS dimensions (by wrapping the mapper as a new simulator) would be more accurate.
- `audit_mapping` provides R^2 via OLS. When R^2 is low, it means either (a) the mapping is weak (weights too small), (b) the simulator is nonlinear in ways the linear mapping can't capture, or (c) the chosen dimensions don't align with the simulator's actual axes of variation.

---

## Possible Issues

- **Weight scale ambiguity.** Weights are multiplied by `half_range`, so a weight of 0.5 on a parameter with range [0, 10] produces a ±2.5 offset, while the same weight on a [0, 1] parameter produces only ±0.25. This is by design (dimensionless weights), but can be surprising.
- **Clipping.** When dimension values are large (e.g., 2.0) and weights are large, the mapped parameter may hit the bounds and be clipped. This flattens the response surface near extremes.
- **Unknown parameter validation.** The constructor validates that mapped parameter names exist in `param_spec()`. However, it does not validate dimension names — dimensions referenced in `dimension_to_param_mapping` but not in `dimension_names` are silently used.
- **OLS numerical stability.** `audit_mapping` uses `np.linalg.lstsq`. For constant outputs (variance < 1e-12), it returns R^2 = 1.0 (trivially explained). For collinear dimensions that cause `LinAlgError`, it returns R^2 = 0.0.

---

## Neat Examples

**Visualizing the PDS landscape.** Sweep two dimensions to create a behavioral phase map:

```python
import numpy as np
grid = np.zeros((20, 20))
for i, power in enumerate(np.linspace(-1, 1, 20)):
    for j, danger in enumerate(np.linspace(-1, 1, 20)):
        params = mapper.map_dimensions_to_params({"power": power, "danger": danger})
        result = sim.run(params)
        grid[i, j] = result["fitness"]

# grid is now a 20x20 fitness landscape in PDS coordinates
# Contour plot reveals the behavioral regions
```

---

## See Also

`Simulator` | `POSIWIDAuditor` | `PromptBuilder` | `sobol_sensitivity`

---

## References

- Dodds, P.S., Alshaabi, T., Fudolig, M.I., Zimmerman, J.W., et al. (2023). "Ousiometrics and telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework."
- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. §4.6.4.
