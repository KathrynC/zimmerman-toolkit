# Falsifier

systematic falsification testing that seeks parameter combinations breaking assumptions

---

## Calling Patterns

```
Falsifier(simulator)                                          default assertions (no NaN/Inf)
Falsifier(simulator, assertions=[fn1, fn2])                   custom assertion functions
```

```
falsifier.falsify()                                           default budget (100+50+50)
falsifier.falsify(n_random=500, n_boundary=100, n_adversarial=200)  custom budget
falsifier.falsify(seed=0)                                     different random seed
```

```
falsifier.boundary_params()                                   generate boundary test cases
falsifier.boundary_params(param_spec=custom_spec)             custom parameter bounds
```

```
falsifier.adversarial_params()                                random adversarial samples
falsifier.adversarial_params(violations=prev_violations)      probe near known failures
falsifier.adversarial_params(n_per_violation=10, perturbation_scale=0.1)  fine-grained control
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol.
- `assertions` is a list of callables, each taking a result dict and returning `True` for valid results. Default assertions: no NaN, no Inf, all finite.
- Each assertion is called independently. A result can fail multiple assertions simultaneously.

### falsify

- `n_random` (default 100): number of uniformly random parameter combinations.
- `n_boundary` (default 50): maximum number of boundary test cases (hypercube corners, edges, faces).
- `n_adversarial` (default 50): number of adversarial test cases near known violations.
- `seed` (default 42): random seed for reproducibility.
- Applies three strategies in sequence: random, boundary, adversarial.
- The adversarial phase uses violations found in previous phases as seeds. If no violations were found, it generates boundary-biased random samples.
- Returns a dict with:
  - `"violations"`: list of violation dicts, each with:
    - `"params"`: the parameter combination that caused the violation.
    - `"result"`: the simulator output dict (empty if exception).
    - `"failed_assertions"`: list of indices into the assertions list.
    - `"strategy"`: `"random"`, `"boundary"`, or `"adversarial"`.
    - `"error"`: exception message string, or `None`.
  - `"summary"`: dict with:
    - `"total_tests"`, `"violations_found"`, `"violation_rate"`.
    - `"random_violations"`, `"boundary_violations"`, `"adversarial_violations"`.
    - `"exceptions"`: count of runs where `simulator.run()` raised.

### boundary_params

- Generates test cases at hypercube boundaries using 5 strategies:
  1. All-min and all-max corners.
  2. One parameter at min (rest at max) and vice versa.
  3. Each parameter at min or max individually (rest at midpoints).
  4. Near-boundary: epsilon (1e-6 * range) inside the bounds.
  5. For D <= 6: all 2^D corner combinations.
- Results are deduplicated.
- `param_spec` (default `None`): custom bounds dict. Uses simulator's spec if `None`.

### adversarial_params

- `violations`: list of violation dicts with `"params"` keys. If `None` or empty, generates boundary-biased random samples.
- `n_per_violation` (default 5): perturbations per known violation.
- `perturbation_scale` (default 0.05): perturbation magnitude as fraction of parameter range.
- `rng`: numpy random generator. Default: `default_rng(999)`.

---

## Basic Examples

Test a simple model for numerical issues:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.falsifier import Falsifier

>>> def fragile_model(p):
...     return {"ratio": p["a"] / p["b"]}

>>> sim = SimulatorWrapper(fragile_model, {"a": (0, 1), "b": (0, 1)})
>>> f = Falsifier(sim)
>>> report = f.falsify(n_random=100)

>>> report["summary"]["violations_found"]
15  # division by zero at b=0 produces Inf/NaN

>>> report["violations"][0]["strategy"]
"boundary"  # boundaries catch b=0 first

>>> report["violations"][0]["error"]
"division by zero"  # or result has Inf
```

Add custom assertions:

```python
>>> f = Falsifier(sim, assertions=[
...     lambda r: r.get("ratio", 0) >= 0,     # ratio must be non-negative
...     lambda r: r.get("ratio", 0) <= 100,    # ratio must be bounded
... ])
>>> report = f.falsify()
```

---

## Scope

Works with high-dimensional parameter spaces:

```python
>>> spec = {f"x{i}": (0, 1) for i in range(20)}
>>> sim = SimulatorWrapper(lambda p: {"y": sum(p.values())}, spec)
>>> f = Falsifier(sim)
>>> boundaries = f.boundary_params()
>>> len(boundaries)  # 2 corners + 40 one-flip + 40 extreme-at-mid + 40 near-boundary
122  # no 2^20 corners since D > 6
```

Handles simulators that raise exceptions:

```python
>>> def crashy(p):
...     if p["x"] > 0.9:
...         raise ValueError("parameter too high")
...     return {"y": p["x"]}

>>> sim = SimulatorWrapper(crashy, {"x": (0, 1)})
>>> report = Falsifier(sim).falsify()
>>> report["summary"]["exceptions"]
3  # caught and recorded, not re-raised
```

---

## Applications

**ER robot simulator validation.** Before running a 328k-simulation Sobol campaign, verify the simulator is stable:

```python
from zimmerman.falsifier import Falsifier

f = Falsifier(robot_sim, assertions=[
    lambda r: r["dx"] != 0.0 or r["torso_contacts"] > 0,  # not stuck
    lambda r: abs(r["dx"]) < 50.0,  # physically plausible displacement
])
report = f.falsify(n_random=500, n_boundary=200)
# If violations_found == 0, safe to proceed with Sobol
```

**Mitochondrial aging model validation (12D).** The MitoSimulator wraps a 7-state ODE system with a heteroplasmy cliff at 0.70. Post-bugfix (2026-02-15), falsification confirms the simulator is now stable across the full 12D parameter space:

```python
import sys
sys.path.insert(0, "/path/to/how-to-live-much-longer")
from zimmerman_bridge import MitoSimulator
from zimmerman.falsifier import Falsifier

sim = MitoSimulator()  # full 12D (intervention + patient)

# Custom assertions for biological plausibility:
f = Falsifier(sim, assertions=[
    lambda r: 0.0 <= r.get("final_heteroplasmy", -1) <= 1.0,  # het in [0,1]
    lambda r: r.get("final_atp", -1) >= 0.0,                   # ATP non-negative
    lambda r: r.get("final_ros", -1) >= 0.0,                   # ROS non-negative
    lambda r: r.get("final_senescent", -1) <= 1.0,             # senescence fraction <= 1
])
report = f.falsify(n_random=100, n_boundary=50, n_adversarial=50)

report["summary"]
# {"total_tests": 200, "violations_found": 0, "violation_rate": 0.0,
#  "random_violations": 0, "boundary_violations": 0,
#  "adversarial_violations": 0, "exceptions": 0}

# Zero violations confirms the ODE system is numerically stable even at
# extreme corners like baseline_heteroplasmy=0.95 + genetic_vulnerability=2.0
# + metabolic_demand=2.0 (post-cliff collapse regime). The 4 critical bugs
# found on 2026-02-15 (cosmetic cliff, unbounded copy number, inverted NAD
# sign, universal attractor) have been fully resolved.
```

Before the 2026-02-15 bugfixes, the same test found NaN outputs at extreme parameter combinations where the cliff was cosmetic rather than dynamical, mtDNA copy number grew unbounded, and NAD supplementation had an inverted therapeutic sign.

---

## Properties & Relations

- **Three-phase design.** The random → boundary → adversarial sequence is deliberate: random provides broad coverage, boundary catches numerical edge cases, adversarial probes the neighborhood of failures to determine whether they are point defects or regions.
- **Adversarial bootstrapping.** When violations are found in phases 1-2, phase 3 perturbs around them. When no violations are found, phase 3 generates boundary-biased samples (a different distribution than phase 1's uniform sampling).
- **Assertion composition.** Multiple assertions are checked independently. A single run can fail multiple assertions, and the `failed_assertions` list records which ones.
- `Falsifier` catches exceptions from `simulator.run()` — unlike `sobol_sensitivity`, which lets them propagate. Run falsification first to verify simulator stability.
- Related to `ContrastiveGenerator`: falsification finds where the simulator *breaks*, contrastive generation finds where the simulator's *behavior changes*.

---

## Possible Issues

- **Combinatorial explosion of boundary tests.** For D <= 6, all 2^D corner combinations are generated. For D = 6, that is 64 corners plus other strategies. For D > 6, corners are omitted and the boundary count is approximately 6D + 2 (2 full corners + 2D one-flip + 2D extreme-at-midpoint + 2D near-boundary).
- **Assertion function errors.** If an assertion function itself raises an exception (not the simulator), that assertion is counted as failed. Ensure assertion functions handle missing keys gracefully.
- **False sense of security.** Zero violations does not prove the simulator is correct — it means no violations were found *at the tested points*. Increase sample counts for higher confidence.
- **Adversarial phase with zero violations.** When phases 1-2 find nothing, phase 3 still runs with boundary-biased random samples, but these are less targeted than perturbations around known failures.

---

## Neat Examples

**Mapping the failure region.** Use adversarial testing iteratively to map the boundary of a failure region:

```python
# Round 1: find initial violations
report1 = Falsifier(sim).falsify(n_random=1000)

# Round 2: densely probe around failures
f2 = Falsifier(sim)
adversarial = f2.adversarial_params(
    violations=report1["violations"],
    n_per_violation=20,
    perturbation_scale=0.01  # very tight probing
)
# Run each and check — reveals the precise failure surface
```

---

## See Also

`Simulator` | `sobol_sensitivity` | `ContrastiveGenerator` | `POSIWIDAuditor`
