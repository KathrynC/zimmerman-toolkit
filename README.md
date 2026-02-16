# Zimmerman Toolkit

A generalized library for interrogating black-box simulators. Given any simulator that satisfies a simple protocol -- `run(params) -> dict` and `param_spec() -> bounds` -- the toolkit answers the questions that matter: what inputs drive outcomes, where are the tipping points, does the system do what you think it does, and where does it break.

Based on Julia Zimmerman's (she/her) 2025 PhD dissertation at the University of Vermont: *"Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)."*

Extracted and generalized from the [how-to-live-much-longer](../how-to-live-much-longer) mitochondrial aging simulator project.

## Installation

Pure Python + numpy. No scipy, no SALib, no external dependencies beyond numpy.

```bash
pip install numpy
```

## The Simulator Protocol

Any object that provides two methods can be analyzed by the full toolkit:

```python
class MySimulator:
    def run(self, params: dict) -> dict:
        """Execute simulation. Returns named numeric outputs."""
        x = params["x"]
        return {"y": x ** 2, "z": x + 1}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds: {name: (low, high)}."""
        return {"x": (0.0, 10.0)}
```

For standalone functions, use the `SimulatorWrapper`:

```python
from zimmerman import SimulatorWrapper

def my_model(params):
    return {"output": params["a"] + params["b"]}

sim = SimulatorWrapper(my_model, {"a": (0.0, 1.0), "b": (-1.0, 1.0)})
```

## Modules

### `sobol` -- What inputs matter?

Global sensitivity analysis via Saltelli sampling and Sobol indices. Reveals which parameters actually drive outcomes vs. which are noise.

- **S1 (first-order):** fraction of output variance explained by each parameter alone
- **ST (total-order):** fraction due to each parameter and all its interactions
- **ST - S1:** the interaction contribution

Pure numpy implementation of the Saltelli (2002) sampling scheme with Jansen (1999) estimators. No SALib dependency.

```python
from zimmerman import sobol_sensitivity

result = sobol_sensitivity(my_sim, n_base=256, seed=42)

# Which parameter drives displacement the most?
print(result["dx"]["S1"])      # {"w03": 0.42, "w04": 0.31, ...}
print(result["dx"]["ST"])      # Total-order (includes interactions)
print(result["rankings"])      # Sorted by influence
```

Total simulations: `N * (D + 2)` where N = `n_base`, D = number of parameters.

### `falsifier` -- Where does it break?

Systematically probes the parameter space looking for violations: NaN outputs, infinities, and broken assumptions. Three testing strategies applied in sequence:

1. **Random sampling** -- uniformly random parameter combinations
2. **Boundary testing** -- corners, edges, and faces of the parameter hypercube (where numerical issues like overflow and division-by-zero most often occur)
3. **Adversarial testing** -- perturbed variants near already-found violations, probing whether failures are isolated points or entire regions

This module found 4 real bugs in the JGC mitochondrial ODE equations.

```python
from zimmerman import Falsifier

# Default assertions: no NaN, no Inf, all finite
falsifier = Falsifier(my_sim)

# Custom assertions
falsifier = Falsifier(my_sim, assertions=[
    lambda r: r["energy"] >= 0,       # energy must be non-negative
    lambda r: r["health"] <= 1.0,     # health capped at 1.0
    lambda r: abs(r["dx"]) < 100.0,   # robot shouldn't teleport
])

report = falsifier.falsify(n_random=200, n_boundary=50, n_adversarial=50)
print(f"Found {report['summary']['violations_found']} violations")
print(f"Violation rate: {report['summary']['violation_rate']:.1%}")
```

### `contrastive` -- Where are the tipping points?

Finds the smallest parameter change that flips the outcome from one category to another. Uses bisection along random perturbation directions to locate decision boundaries in parameter space.

Useful for understanding fragility, generating adversarial examples, and finding the exact point where a robot stops walking and starts falling.

```python
from zimmerman import ContrastiveGenerator

gen = ContrastiveGenerator(
    my_sim,
    outcome_fn=lambda r: "forward" if r["dx"] > 0 else "backward"
)

pair = gen.find_contrastive(
    base_params={"w03": 0.855, "w04": -0.659, ...},
    n_attempts=100,
    max_delta_frac=0.1,  # search within 10% of parameter ranges
)

if pair["found"]:
    print(f"Flip magnitude: {pair['perturbation_magnitude']:.4f}")
    print(f"Delta: {pair['delta']}")  # which params changed most

# Batch analysis: which parameters are most involved in flips?
pairs = gen.contrastive_pairs(list_of_weight_dicts, n_per_point=5)
sensitivity = gen.sensitivity_from_contrastives(pairs)
print(sensitivity["rankings"])  # params sorted by flip involvement
```

### `posiwid` -- Does it do what you think?

POSIWID alignment auditor: "The Purpose Of a System Is What It Does" (Stafford Beer, 1974). Measures the gap between intended outcomes and actual simulation results.

Given intended outcomes (from an LLM, a human designer, or an optimization target) and actual parameters, how well do the actual results match the intention?

```python
from zimmerman import POSIWIDAuditor

auditor = POSIWIDAuditor(my_sim)

# Single audit
result = auditor.audit(
    intended_outcomes={"dx": 5.0, "speed": 0.8},
    params={"w03": 0.5, "w04": -0.3, ...},
)
print(result["alignment"]["overall"])         # 0.0 to 1.0
print(result["alignment"]["per_key"]["dx"])   # direction + magnitude match

# Batch audit across multiple scenarios
report = auditor.batch_audit([
    {"intended": {"dx": 5.0}, "params": weights_a},
    {"intended": {"dx": -2.0}, "params": weights_b},
])
print(report["aggregate"]["mean_overall"])
```

Alignment scoring per output key:
- **direction_match:** did the value move in the expected direction? (binary)
- **magnitude_match:** how close is actual to intended? (continuous 0-1)
- **combined:** 50/50 weighted average

### `pds` -- What do abstract concepts map to?

Parameter-Design-State dimension mapper. Maps abstract high-level dimensions (Power, Danger, Stability) to weighted combinations of concrete simulator parameters. Lets domain experts reason about simulations in meaningful terms rather than raw parameter names.

From ousiometric analysis of word meanings (Dodds et al. 2023); applied to simulation via Zimmerman (2025) §4.6.4.

```python
from zimmerman import PDSMapper

mapper = PDSMapper(
    simulator=my_sim,
    dimension_names=["aggression", "stability"],
    dimension_to_param_mapping={
        "aggression": {"w03": 0.5, "w04": 0.3},
        "stability": {"w13": -0.4, "w23": 0.2},
    },
)

# Convert abstract dimensions to concrete params
params = mapper.map_dimensions_to_params({"aggression": 0.8, "stability": -0.3})

# Audit how well the abstract dimensions predict outcomes
audit = mapper.audit_mapping(n_samples=100)
print(audit["variance_explained"])                # R^2 per output
print(audit["dimension_output_correlations"])      # correlation matrix

# Which dimensions matter most?
sensitivity = mapper.sensitivity_per_dimension(n_samples=100)
print(sensitivity["aggression"]["overall"])
```

### `prompts` -- How should an LLM talk to it?

Diegetic prompt builder for LLM-mediated parameter generation. Three prompt styles based on Zimmerman's finding (§2.2.3) that LLMs handle diegetic content (meaning) better than supradiegetic content (form/numbers), compounded by tokenization-induced flattening of numeric content (§3.5.3):

```python
from zimmerman import PromptBuilder

builder = PromptBuilder(my_sim, context={"domain": "3-link walking robot"})

# Style 1: Numeric -- straightforward parameter request with bounds
prompt = builder.build_numeric("Design a fast robot that walks forward")

# Style 2: Diegetic -- parameters embedded in a domain narrative
prompt = builder.build_diegetic(
    "Design a fast robot that walks forward",
    state_description="Currently the robot flips after 2 seconds",
)

# Style 3: Contrastive -- two opposing agents bracket the problem
# Exploits TALOT/OTTITT meaning-from-contrast (Zimmerman §4.7.6)
prompt = builder.build_contrastive(
    "Design a walking robot",
    agent_a="cautious engineer",
    agent_b="aggressive optimizer",
)
```

## Projects Using the Toolkit

Three simulators currently satisfy the Zimmerman protocol:

| Project | Simulator | Parameters | Outputs |
|---------|-----------|------------|---------|
| [Evolutionary Robotics](../pybullet_test/Evolutionary-Robotics) | `SimulatorWrapper` around `run_trial_inmemory()` | 6 synapse weights (w03..w24), each in [-1, 1] | dx, speed, efficiency, work |
| [how-to-live-much-longer](../how-to-live-much-longer) | JGC mitochondrial aging ODE | 6 patient/treatment params | het_final, atp_final, lifespan |
| [stock-simulator](../stock-simulator) | `StockSimulator` (7D financial ODE) | 6 strategy params | portfolio_value, sharpe, max_drawdown |

The ER integration is in `zimmerman_analysis.py` at the Evolutionary Robotics project root, which provides convenience wrappers:

```python
from zimmerman_analysis import run_sobol_analysis, run_falsification, run_contrastive

# Sobol: which of the 6 synapse weights matters most for displacement?
sobol = run_sobol_analysis(n_base=64)

# Falsifier: find weight combos that produce NaN or teleportation
report = run_falsification(n_random=50)

# Contrastive: smallest weight change that reverses walking direction
champion = {"w03": 0.855, "w04": -0.659, "w13": 0.204,
            "w14": -0.911, "w23": 0.478, "w24": -0.738}
pair = run_contrastive(champion, n_attempts=50)
```

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont.
- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
- Beer, S. (1974). "Designing Freedom." CBC Massey Lectures.
