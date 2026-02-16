# ContrastiveGenerator

finds minimal parameter perturbations that flip simulation outcomes

---

## Calling Patterns

```
ContrastiveGenerator(simulator)                               default outcome: sign of "fitness"
ContrastiveGenerator(simulator, outcome_fn=fn)                custom outcome classifier
```

```
gen.find_contrastive(base_params)                             find smallest flip from one point
gen.find_contrastive(base_params, n_attempts=200)             more random directions
gen.find_contrastive(base_params, max_delta_frac=0.2)         search up to 20% perturbation
gen.find_contrastive(base_params, seed=0)                     different random seed
```

```
gen.contrastive_pairs(params_list)                            batch mode: one flip per starting point
gen.contrastive_pairs(params_list, n_per_point=10)            more attempts per point
gen.contrastive_pairs(params_list, seed=0)                    different base seed
```

```
gen.sensitivity_from_contrastives(pairs)                      analyze which params drive flips
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol.
- `outcome_fn` takes a result dict and returns a categorical value (typically `+1`/`-1` or `True`/`False`). Default: `1 if result.get("fitness", 0) >= 0 else -1`.
- Outcome values are compared with `==`. Any hashable return type works.

### find_contrastive

- `base_params`: starting parameter dict.
- `n_attempts` (default 100): number of random directions to sample. More attempts increase the chance of finding a flip and finding the *smallest* flip.
- `max_delta_frac` (default 0.1): maximum perturbation as a fraction of each parameter's range. At 0.1, no parameter moves more than 10% of its range.
- `seed` (default 42): random seed.
- **Algorithm:** For each random direction in parameter space:
  1. Perturb `base_params` by `max_delta_frac` along that direction.
  2. If the outcome flips, bisect (15 iterations, ~1e-5 precision) to find the minimal perturbation magnitude.
  3. Track the smallest flip found across all directions.
- Parameters are clipped to bounds after perturbation.
- Returns a dict with:
  - `"found"`: `True` if a contrastive example was found.
  - `"original_params"`: the starting parameters.
  - `"contrastive_params"`: the flipped parameters (or `None`).
  - `"delta"`: dict of per-parameter deltas `{param_name: contrastive - original}` (or `None`).
  - `"outcome_original"`: outcome at `base_params`.
  - `"outcome_flipped"`: outcome at the contrastive point (or `None`).
  - `"perturbation_magnitude"`: the minimal magnitude found (fraction of range).
  - `"n_sims"`: total simulations run during the search.

### contrastive_pairs

- `params_list`: list of starting parameter dicts.
- `n_per_point` (default 5): attempts per starting point (different seeds). Stops after first success per point.
- `seed` (default 42): base seed. Actual seed for each attempt = `seed + idx * n_per_point + attempt`.
- Each point uses `n_attempts=20` and `max_delta_frac=0.15` internally.
- Returns a list of successful contrastive pair dicts (one per point that had a flip).

### sensitivity_from_contrastives

- `pairs`: list of contrastive pair dicts (output of `find_contrastive` or `contrastive_pairs`). Only pairs with `"found" == True` are analyzed.
- Returns a dict with:
  - `"param_importance"`: `{param_name: mean_abs_delta}` — average absolute change at the decision boundary.
  - `"param_flip_frequency"`: `{param_name: fraction}` — how often this parameter had the largest absolute delta in a flip.
  - `"n_pairs"`: count of successful pairs analyzed.
  - `"rankings"`: parameter names sorted by importance (descending).

---

## Basic Examples

Find the minimal change that makes a model's output flip sign:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.contrastive import ContrastiveGenerator

>>> def model(p):
...     return {"fitness": p["a"] - 0.5}  # flips at a=0.5

>>> sim = SimulatorWrapper(model, {"a": (0.0, 1.0), "b": (0.0, 1.0)})
>>> gen = ContrastiveGenerator(sim)
>>> result = gen.find_contrastive({"a": 0.3, "b": 0.5})

>>> result["found"]
True
>>> result["delta"]["a"]    # a must increase ~0.2 to cross 0.5
0.20
>>> result["delta"]["b"]    # b is irrelevant to the flip
0.01  # near zero
```

---

## Scope

Custom outcome functions enable classification of any behavior:

```python
>>> gen = ContrastiveGenerator(sim, outcome_fn=lambda r: "fast" if r["speed"] > 1.0 else "slow")
>>> result = gen.find_contrastive(slow_params)
>>> result["outcome_original"]
"slow"
>>> result["outcome_flipped"]
"fast"
```

Batch analysis across multiple starting points:

```python
>>> pairs = gen.contrastive_pairs(list_of_100_params)
>>> len(pairs)  # number of points where a flip was found
67

>>> sens = gen.sensitivity_from_contrastives(pairs)
>>> sens["rankings"]
["weight_3", "weight_1", "weight_5", ...]
```

When no flip exists within `max_delta_frac`:

```python
>>> result = gen.find_contrastive(very_stable_params, max_delta_frac=0.05)
>>> result["found"]
False
>>> result["n_sims"]
101  # 1 base + 100 max-perturbation checks, no bisections triggered
```

---

## Applications

**ER robot behavioral cliffs.** Find the smallest weight change that makes an antifragile gait collapse:

```python
gen = ContrastiveGenerator(
    robot_sim,
    outcome_fn=lambda r: "walk" if r["dx"] > 0.5 else "fall"
)
result = gen.find_contrastive(haraway_weights, n_attempts=200)
# result["delta"] reveals which weight is the "cliff edge"
# result["perturbation_magnitude"] quantifies fragility
```

**Mapping the antifragile/knife-edge boundary.** Combine with batch analysis:

```python
all_gaits = load_zoo()
pairs = gen.contrastive_pairs([g["weights"] for g in all_gaits])
sens = gen.sensitivity_from_contrastives(pairs)
# sens["param_flip_frequency"]["w14"] = 0.45 means w14 is the dominant
# cliff-driver in 45% of gaits
```

---

## Properties & Relations

- **Bisection precision.** 15 bisection steps give approximately 3e-5 precision in the perturbation magnitude (2^-15 ≈ 3.05e-5).
- **Computational cost.** Per starting point: 1 base + `n_attempts` * (1 max-check + 15 bisection + 1 final confirmation) = worst case 1 + 100 * 17 = 1,701 sims for `n_attempts=100`. Average is much lower since most directions don't flip.
- **Directionality.** Perturbation directions are sampled uniformly on the unit sphere. The magnitude is scaled by each parameter's range, so the search is isotropic in normalized parameter space.
- **Complementary to Sobol.** Sobol analysis measures *global* sensitivity across the full parameter range. Contrastive generation measures *local* sensitivity at specific operating points and specific behavioral boundaries. Together they give a complete picture.
- `sensitivity_from_contrastives()` provides a different kind of importance ranking than Sobol S1: it identifies parameters that are dominant *at decision boundaries*, not across the whole space.

---

## Possible Issues

- **Outcome function granularity.** If `outcome_fn` returns continuous values, `==` comparison may fail due to floating-point noise. Always discretize outcomes into categories.
- **Boundary clipping.** When `base_params` is near a parameter bound, perturbations in that direction are clipped, reducing the effective search radius. The algorithm doesn't compensate for this.
- **No flip within range.** For very stable regions of parameter space, `find_contrastive` returns `found=False`. Increase `max_delta_frac` or change `outcome_fn` to a finer-grained classification.
- **Seed sensitivity in batch mode.** `contrastive_pairs` uses deterministic seed = `base_seed + idx * n_per_point + attempt`. Reordering the params_list changes results.

---

## Neat Examples

**Measuring fragility as a scalar.** The perturbation magnitude is a direct fragility metric:

```python
gaits = load_zoo()
fragilities = {}
for name, gait in gaits.items():
    result = gen.find_contrastive(gait["weights"])
    fragilities[name] = result["perturbation_magnitude"] if result["found"] else float("inf")

# Sort: smallest magnitude = most fragile (knife-edge)
most_fragile = sorted(fragilities, key=fragilities.get)
# ['32_carry_trade', '43_hidden_cpg_champion', ...]
```

---

## See Also

`Simulator` | `sobol_sensitivity` | `Falsifier` | `POSIWIDAuditor`
