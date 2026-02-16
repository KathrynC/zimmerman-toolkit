# PromptReceptiveField

feature attribution for simulator inputs via Sobol analysis over segment inclusion weights

---

## Calling Patterns

```
PromptReceptiveField(simulator)                                default segmenter and scorer
PromptReceptiveField(simulator, segmenter=fn)                  custom segment grouping
PromptReceptiveField(simulator, scorer=fn)                     custom score extraction
PromptReceptiveField(simulator, segmenter=fn, scorer=fn)       fully custom
```

```
prf.analyze(base_params)                                       default n_base=64, seed=42
prf.analyze(base_params, n_base=256)                           higher accuracy
prf.analyze(base_params, n_base=64, seed=0)                    different random seed
```

```
prf.run(params)                                                Simulator protocol (segment weights as params)
prf.param_spec()                                               returns segment weight bounds
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol (`run(params) -> dict` + `param_spec() -> dict`).
- `segmenter` is a callable taking `param_spec` dict and returning a list of segment dicts, each with `"name"` (str) and `"params"` (list of parameter names in that segment). If `None`, each parameter is its own segment (finest-grained decomposition).
- `scorer` is a callable taking a result dict and returning a float. Default: priority lookup of `"fitness"` > `"score"` > `"y"`, falling back to `0.0`.

### analyze

- `base_params` is a dict of baseline parameter values -- the operating point around which segment inclusion/exclusion is tested. Should be a meaningful configuration, not just midpoint.
- `n_base` (default 64): base sample count for Saltelli sampling (N in Saltelli 2010). Total simulations = `n_base * (n_segments + 2)`. For 6 segments and `n_base=64`, that is 512 simulations. Use 64 for exploratory analysis, 256+ for publication quality.
- `seed` (default 42): random seed for reproducibility.
- Returns a dict with:
  - `"segment_names"`: `[str, ...]` -- ordered list of segment names.
  - `"S1"`: `{segment_name: float, ...}` -- first-order (main effect) Sobol indices.
  - `"ST"`: `{segment_name: float, ...}` -- total-order (main + interaction) Sobol indices.
  - `"interaction"`: `{segment_name: float, ...}` -- interaction term per segment (`ST - S1`).
  - `"rankings"`: `[str, ...]` -- segment names sorted by descending `ST`.
  - `"n_sims"`: int -- total simulator evaluations performed.

### The three-level weight scheme

Each segment is assigned a continuous weight in `[0, 1]`. The weight determines how the segment's parameters are treated:

- **Drop** (`weight < 0.33`): all parameters in the segment are set to their midpoint (uninformative default). The segment is absent.
- **Compress** (`0.33 <= weight < 0.66`): parameters are linearly interpolated between midpoint (at `weight=0.33`) and their original value (at `weight=0.66`). The blend factor is `(weight - 0.33) / 0.33`.
- **Include** (`weight >= 0.66`): parameters retain their original values unchanged. The segment is fully present.

The thresholds `0.33` and `0.66` divide `[0, 1]` into three equal bands. The compress region provides a smooth transition that helps Sobol analysis estimate derivatives rather than only step functions.

### run (Simulator protocol)

- `params` is a dict mapping segment names to weight values in `[0, 1]`.
- Base parameters are the midpoint of the inner simulator's spec (not user-specified -- use `analyze()` for a specific operating point).
- Returns the inner simulator's result dict, augmented with a `"segment_weights"` key.

### param_spec (Simulator protocol)

- Returns `{segment_name: (0.0, 1.0)}` for each segment. The full `[0, 1]` range spans all three weight regimes (drop/compress/include).

---

## Basic Examples

Create a receptive field analyzer and identify which segments matter:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.prompt_receptive_field import PromptReceptiveField

>>> def model(p):
...     return {"score": 5*p["a"] + 3*p["b"] + 0.1*p["c"]}

>>> sim = SimulatorWrapper(model, {"a": (0, 1), "b": (0, 1), "c": (0, 1)})
>>> prf = PromptReceptiveField(sim)

>>> report = prf.analyze(
...     base_params={"a": 0.8, "b": 0.5, "c": 0.3},
...     n_base=64,
... )

>>> report["rankings"]
['a', 'b', 'c']  # a has highest ST, c is nearly irrelevant

>>> report["S1"]["a"]
0.72  # a's main effect explains ~72% of output variance

>>> report["ST"]["a"]
0.74  # total-order ~ S1 => a's effect is mostly standalone

>>> report["interaction"]["c"]
0.01  # c contributes almost nothing, even through interactions
```

Read S1 vs ST to distinguish inherent vs relational meaning:

```python
>>> for name in report["rankings"]:
...     s1 = report["S1"][name]
...     st = report["ST"][name]
...     inter = report["interaction"][name]
...     print(f"{name}: S1={s1:.3f}  ST={st:.3f}  interaction={inter:.3f}")
a: S1=0.720  ST=0.740  interaction=0.020   # inherent meaning (standalone)
b: S1=0.250  ST=0.270  interaction=0.020   # inherent meaning (weaker)
c: S1=0.005  ST=0.015  interaction=0.010   # negligible
```

---

## Scope

Custom segmenters group parameters into meaningful blocks:

```python
>>> def brain_body_segmenter(spec):
...     brain = [n for n in spec if n.startswith("w")]
...     body = [n for n in spec if n.startswith("p")]
...     return [
...         {"name": "brain", "params": brain},
...         {"name": "body", "params": body},
...     ]

>>> prf = PromptReceptiveField(robot_sim, segmenter=brain_body_segmenter)
>>> report = prf.analyze(base_params=gait_weights, n_base=128)
>>> report["rankings"]
['brain', 'body']  # or vice versa
```

Custom scorers extract domain-specific metrics:

```python
>>> def distance_scorer(result):
...     return abs(result.get("dx", 0.0))

>>> prf = PromptReceptiveField(robot_sim, scorer=distance_scorer)
```

Different `n_base` values trade accuracy for speed:

```python
>>> report_fast = prf.analyze(base_params, n_base=32)   # 32*(d+2) sims, exploratory
>>> report_full = prf.analyze(base_params, n_base=256)  # 256*(d+2) sims, publication
>>> report_fast["n_sims"], report_full["n_sims"]
(256, 2048)  # for d=6 segments
```

---

## Applications

**ER robot: which body segments drive locomotion.** Segment the 12D parameter space into brain (6 synaptic weights) and body (6 physics parameters) to test whether locomotion is brain-driven or body-driven:

```python
from zimmerman.prompt_receptive_field import PromptReceptiveField

def er_segmenter(spec):
    weights = [n for n in sorted(spec) if "weight" in n or n.startswith("w")]
    physics = [n for n in sorted(spec) if n not in weights]
    return [
        {"name": "synaptic_weights", "params": weights},
        {"name": "physics_params", "params": physics},
    ]

prf = PromptReceptiveField(robot_sim, segmenter=er_segmenter)
report = prf.analyze(base_params=gait_116_weights, n_base=128)

# If ST("synaptic_weights") >> ST("physics_params"), the robot's
# locomotion is brain-driven: the specific weight configuration matters
# more than the physical parameters. High interaction for either segment
# means brain-body coupling is important (relational meaning, §3.1).
print(f"Brain ST:  {report['ST']['synaptic_weights']:.3f}")
print(f"Body ST:   {report['ST']['physics_params']:.3f}")
print(f"Brain interaction: {report['interaction']['synaptic_weights']:.3f}")
```

**JGC mitochondrial model: which intervention categories matter.** Segment the 12D parameter space into intervention parameters (what the patient does) and patient parameters (who the patient is):

```python
def mito_segmenter(spec):
    interventions = ["exercise", "nad_supplement", "antioxidant",
                     "caloric_restriction", "sleep_quality", "stress_mgmt"]
    patient = ["age", "genetic_vulnerability", "metabolic_demand",
               "baseline_heteroplasmy", "mitophagy_rate", "biogenesis_rate"]
    return [
        {"name": "interventions", "params": [p for p in interventions if p in spec]},
        {"name": "patient", "params": [p for p in patient if p in spec]},
    ]

prf = PromptReceptiveField(mito_sim, segmenter=mito_segmenter)
report = prf.analyze(base_params=healthy_baseline, n_base=256)

# High interaction for "interventions" means the effectiveness of
# interventions depends on who the patient is -- a key clinical insight.
print(f"Intervention interaction: {report['interaction']['interventions']:.3f}")
```

---

## Properties & Relations

- **S1 as inherent meaning, ST - S1 as relational meaning.** Per Zimmerman section 3.1, some tokens/parameters carry meaning on their own (high S1), while others contribute only through their combination with other tokens/parameters (high ST - S1). The PromptReceptiveField makes this distinction empirically measurable. A segment with high ST but low S1 is one that "only matters in context" -- a direct operationalization of relational meaning construction.
- **Connection to Sobol sensitivity analysis.** PromptReceptiveField uses the same Saltelli (2002, 2010) sampling scheme and Jansen (1999) total-order estimator as `sobol_sensitivity`. The difference is that `sobol_sensitivity` operates over the simulator's native parameters, while `PromptReceptiveField` operates over segment inclusion weights. The former asks "which parameter matters?" and the latter asks "which group of parameters matters?"
- **Hubel and Wiesel receptive field analogy.** In visual neuroscience, a neuron's receptive field is the region of the visual field that modulates the neuron's firing rate (Hubel & Wiesel, 1962). By analogy, a simulator's "prompt receptive field" is the region of its input space that modulates its output. Segments with high ST are "inside the receptive field"; segments with low ST are "outside" -- the system is blind to them.
- **Composability.** Because `PromptReceptiveField` satisfies the Simulator protocol, it can be wrapped by `LocalityProfiler` (to test how the receptive field degrades under locality manipulations), analyzed by `sobol_sensitivity` (Sobol-on-Sobol: which segment weights interact?), or probed by `Falsifier` (which weight configurations cause failures?). This follows Zimmerman's TALOT/OTTITT principle (section 4.7.6).
- **Rankings use ST, not S1.** Segments are ranked by total-order index (descending) because ST captures both standalone and relational contributions. Ranking by S1 alone would miss segments that are important only through interaction -- precisely the relational meaning that Zimmerman section 3.1 argues is central.
- **For additive models, S1 values sum to approximately 1.0** and interaction terms are near zero. Departures from this indicate nonlinearity or interaction effects in the simulator.

---

## Possible Issues

- **Computational cost = n_base * (d + 2).** For `d=6` segments and `n_base=64`, that is 512 simulations. For `d=12` and `n_base=256`, that is 3584 simulations. Cost scales linearly with both `n_base` and `d`. Grouping parameters into fewer segments (via a custom segmenter) reduces `d` and thus cost.
- **Segment grouping affects results.** The choice of segmenter determines what the analysis can detect. Too fine-grained (each parameter is its own segment) may miss group-level interactions; too coarse (all parameters in one segment) trivially yields ST=1.0. Choose a segmenter that reflects meaningful domain boundaries.
- **Drop/compress/include thresholds at 0.33/0.66.** These are fixed design choices, not configurable parameters. The equal-thirds partition is reasonable for most cases, but systems with sharp behavioral cliffs may benefit from different thresholds. Adjusting thresholds requires modifying `_apply_segment_weights()`.
- **Negative Sobol indices.** With small `n_base`, numerical noise can produce slightly negative S1 or ST values (typically in the range -0.05 to 0). These should be interpreted as "approximately zero." Increase `n_base` if negative indices appear.
- **Scorer must return finite floats.** If the scorer returns NaN or Inf for some weight configurations (e.g., because the simulator crashes when a critical segment is dropped), the Sobol index computation will produce NaN. Use `Falsifier` first to verify simulator stability, or wrap the scorer with a NaN guard.
- **run() vs analyze() use different baselines.** The `run()` method (Simulator protocol) uses midpoint base parameters for composability. The `analyze()` method uses user-specified `base_params`. Do not compare results from the two methods directly.

---

## Neat Examples

**Analyzing which prompt sections an LLM attends to.** Wrap an LLM-based scorer as a simulator with segmented prompt sections, then use `PromptReceptiveField` to determine which sections drive the LLM's output:

```python
from zimmerman.base import SimulatorWrapper
from zimmerman.prompt_receptive_field import PromptReceptiveField

# Each "parameter" is a weight controlling whether a prompt section
# is included. The simulator calls the LLM with the composed prompt.
def llm_simulator(params):
    """Score the LLM's response quality given prompt section weights."""
    sections = compose_prompt(params)  # build prompt from weighted sections
    response = call_llm(sections)
    return {"score": evaluate_response(response)}

spec = {
    "system_prompt": (0, 1),
    "few_shot_examples": (0, 1),
    "task_description": (0, 1),
    "output_format": (0, 1),
    "chain_of_thought": (0, 1),
}

sim = SimulatorWrapper(llm_simulator, spec)

# Group into supradiegetic (structural) vs diegetic (content) per §5.2
def prompt_segmenter(spec):
    return [
        {"name": "structural", "params": ["system_prompt", "output_format"]},
        {"name": "examples", "params": ["few_shot_examples"]},
        {"name": "task", "params": ["task_description", "chain_of_thought"]},
    ]

prf = PromptReceptiveField(sim, segmenter=prompt_segmenter)
report = prf.analyze(
    base_params={k: 1.0 for k in spec},  # all sections fully present
    n_base=64,
)

# Which matters more: the structural scaffolding (supradiegetic)
# or the task content (diegetic)?
for name in report["rankings"]:
    print(f"{name}: ST={report['ST'][name]:.3f}, "
          f"interaction={report['interaction'][name]:.3f}")
```

---

## See Also

`sobol_sensitivity` | `LocalityProfiler` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Section 3.1.
- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model output." *Computer Physics Communications*, 181(2), 259-270.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
- Hubel, D.H. & Wiesel, T.N. (1962). "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." *Journal of Physiology*, 160(1), 106-154.
