# MeaningConstructionDashboard

unified aggregator for all zimmerman toolkit analyses across 6 dimensions

---

## Calling Patterns

```
MeaningConstructionDashboard(simulator)                         create dashboard for a simulator

dashboard.compile(reports={"sobol": sobol_report})              compile with one tool report
dashboard.compile(reports={"sobol": r1, "falsifier": r2, ...})  compile with multiple tool reports
dashboard.compile()                                             compile with no reports (baseline)

dashboard.generate_recommendations(sections)                    generate cross-section recommendations from compiled sections
dashboard.to_markdown(result)                                   render compiled dashboard as markdown string

dashboard.run(params)                                           Simulator protocol: delegates to underlying simulator
dashboard.param_spec()                                          Simulator protocol: delegates to underlying simulator
```

---

## Details and Options

- `simulator` must satisfy the `Simulator` protocol: `run(params) -> dict` and `param_spec() -> dict[str, (float, float)]`.

### Six analytical sections

The dashboard organizes all 12 zimmerman toolkit tools into six sections, each addressing a distinct dimension of simulator behavior:

| Section | Key | Tools | Question answered |
|---------|-----|-------|-------------------|
| **Sensitivity** | `"sensitivity"` | `sobol`, `receptive_field` | Which parameters matter most? |
| **Robustness** | `"robustness"` | `falsifier`, `contrast_sets`, `contrastive` | Where does the simulator break? |
| **Locality** | `"locality"` | `locality` | How far do perturbation effects propagate? |
| **Alignment** | `"alignment"` | `posiwid` | Does the simulator do what it intends? |
| **Representation** | `"representation"` | `benchmark`, `diegeticizer`, `token_extispicy` | How well is meaning preserved through the LLM? |
| **Structure** | `"structure"` | `relation_graph`, `pds` | What relational patterns emerge? |

### All 12 tool keys

The `reports` argument to `compile()` accepts a dict with any subset of these keys:

- `"sobol"` -> sensitivity section: Sobol S1/ST indices from `sobol_sensitivity()`.
- `"receptive_field"` -> sensitivity section: prompt receptive field analysis.
- `"falsifier"` -> robustness section: constraint violation detection at boundaries.
- `"contrast_sets"` -> robustness section: minimal contrast set identification.
- `"contrastive"` -> robustness section: contrastive sensitivity (which params flip outcomes).
- `"locality"` -> locality section: L50 and effective horizon from `locality_profiler`.
- `"posiwid"` -> alignment section: intention-outcome gap from POSIWID audit.
- `"benchmark"` -> representation section: supradiegetic benchmark roundtrip error.
- `"diegeticizer"` -> representation section: diegeticization gain (narrative vs numeric).
- `"token_extispicy"` -> representation section: fragmentation-output correlation.
- `"relation_graph"` -> structure section: causal dependency graph.
- `"pds"` -> structure section: Parameter Dimension Structure (latent dimensions).

### compile

- `reports` (default `None`): dict mapping tool keys to their result dicts. When `None`, treated as an empty dict.
- Returns a dict with keys:
  - `"simulator_info"`: dict with `"n_params"`, `"param_names"`, `"param_ranges"`.
  - `"sections"`: dict of six section dicts, each containing:
    - `"available"`: bool -- whether any data was found for this section.
    - Section-specific metrics (see below).
  - `"recommendations"`: list of actionable suggestion strings.
  - `"tools_used"`: list of tool keys that contributed data.
  - `"tools_missing"`: list of tool keys that were not provided.

#### Section-specific metrics:

- **Sensitivity**: `"top_params"` (list, ranked by S1), `"interaction_strength"` (float, mean |ST - S1|).
- **Robustness**: `"violation_rate"` (float, fraction of boundary violations), `"mean_flip_size"` (float, mean parameter change to flip outcomes), `"most_fragile_params"` (list).
- **Locality**: `"L50"` (float, perturbation radius for 50% variance), `"effective_horizon"` (float, radius of negligible effect).
- **Alignment**: `"overall_alignment"` (float in [0, 1]), `"worst_aligned_keys"` (list, worst-aligned output keys first).
- **Representation**: `"diegeticization_gain"` (float, positive = narrative helps), `"roundtrip_error"` (float, numeric distortion), `"fragmentation_correlation"` (float, mean |r| between fragmentation and output).
- **Structure**: `"most_causal_params"` (list, ranked by causal effect), `"variance_explained"` (float, mean PDS variance explained).

### generate_recommendations

- Accepts the `"sections"` dict from `compile()`.
- Applies threshold-based heuristics to produce cross-section recommendations:
  - `violation_rate > 0.05`: flag significant failure region.
  - `interaction_strength > 0.1`: recommend joint parameter prompts.
  - `overall_alignment < 0.5`: flag intention-outcome gap.
  - `fragmentation_correlation > 0.3`: recommend diegetic prompts.
  - `roundtrip_error > 0.1`: flag numeric encoding degradation.
  - `diegeticization_gain > 0.1`: confirm narrative prompts outperform numeric.
  - `effective_horizon < 0.3`: flag high locality, suggest reordering.
  - `variance_explained < 0.3`: flag incomplete PDS coverage.
- Returns `["No specific recommendations -- run more tools for deeper analysis."]` when no thresholds are triggered.

### to_markdown

- Accepts the full result dict from `compile()`.
- Returns a formatted markdown string with six sections, recommendations, and tool coverage.
- Sections with no data are marked "*No data available for this section.*"
- Numeric values are formatted to 4 decimal places.

### Graceful degradation

Missing tools are simply omitted. Each section builder returns `{"available": False, ...}` when none of its relevant tool reports are present. The dashboard always produces useful output regardless of which subset of tools was run.

### run / param_spec (Simulator protocol)

- `run(params)` delegates directly to the underlying simulator (unmodified result).
- `param_spec()` delegates directly to the underlying simulator.
- The dashboard itself satisfies the Simulator protocol, enabling recursive meta-analysis: e.g., `sobol_sensitivity(dashboard)` reveals which parameters most affect the dashboard's own output metrics.

---

## Basic Examples

Create a dashboard and compile with Sobol and falsifier reports:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.sobol import sobol_sensitivity
>>> from zimmerman.falsifier import Falsifier
>>> from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard

>>> def model(p):
...     return {"y": 3 * p["a"] + p["b"] ** 2}

>>> sim = SimulatorWrapper(model, {"a": (0.0, 1.0), "b": (0.0, 1.0)})

>>> sobol_report = sobol_sensitivity(sim, n_base=256)
>>> falsifier_report = Falsifier(sim).falsify()

>>> dash = MeaningConstructionDashboard(sim)
>>> result = dash.compile(reports={
...     "sobol": sobol_report,
...     "falsifier": falsifier_report,
... })

>>> result["sections"]["sensitivity"]["available"]
True

>>> result["sections"]["sensitivity"]["top_params"]
["a", "b"]

>>> result["tools_used"]
["sobol", "falsifier"]

>>> result["tools_missing"]
["receptive_field", "contrast_sets", "contrastive", "locality", "posiwid",
 "benchmark", "diegeticizer", "token_extispicy", "relation_graph", "pds"]
```

Generate recommendations:

```python
>>> result["recommendations"]
["Most influential parameters: a, b. Focus LLM prompts on these for maximum impact."]
```

Render as markdown:

```python
>>> md = dash.to_markdown(result)
>>> print(md[:200])
# Meaning Construction Dashboard

## Simulator Info
- **Parameters**: 2
- **Names**: a, b

## Sensitivity
- **top_params**: a, b
- **interaction_strength**: 0.0012
```

---

## Scope

Partial reports -- compile with any subset of tools:

```python
>>> result = dash.compile(reports={"sobol": sobol_report})
>>> result["sections"]["sensitivity"]["available"]
True
>>> result["sections"]["robustness"]["available"]
False
```

Adding reports incrementally:

```python
>>> # Start with just Sobol
>>> reports = {"sobol": sobol_report}
>>> result = dash.compile(reports=reports)
>>> len(result["tools_used"])
1

>>> # Add falsifier later
>>> reports["falsifier"] = Falsifier(sim).falsify()
>>> result = dash.compile(reports=reports)
>>> len(result["tools_used"])
2

>>> # Add token extispicy
>>> from zimmerman.token_extispicy import TokenExtispicyWorkbench
>>> wb = TokenExtispicyWorkbench(sim)
>>> reports["token_extispicy"] = wb.analyze(n_samples=100)
>>> result = dash.compile(reports=reports)
>>> result["sections"]["representation"]["available"]
True
>>> result["sections"]["representation"]["fragmentation_correlation"]
0.15  # approximately
```

Compile with no reports (baseline):

```python
>>> result = dash.compile()
>>> all(not s["available"] for s in result["sections"].values())
True
>>> result["recommendations"]
["No specific recommendations -- run more tools for deeper analysis."]
```

All 12 tool keys:

```python
>>> from zimmerman.meaning_construction_dashboard import ALL_TOOL_KEYS
>>> ALL_TOOL_KEYS
["sobol", "receptive_field", "falsifier", "contrast_sets", "contrastive",
 "locality", "posiwid", "benchmark", "diegeticizer", "token_extispicy",
 "relation_graph", "pds"]
>>> len(ALL_TOOL_KEYS)
12
```

---

## Applications

**ER: Comprehensive behavioral audit of 116 gaits.** Run the full zimmerman toolkit on the robot gait simulator and compile a unified report:

```python
from zimmerman.sobol import sobol_sensitivity
from zimmerman.falsifier import Falsifier
from zimmerman.token_extispicy import TokenExtispicyWorkbench
from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard

robot_sim = SimulatorWrapper(run_trial, weight_spec)

# Run individual analyses
sobol_report = sobol_sensitivity(robot_sim, n_base=256)
falsifier_report = Falsifier(robot_sim).falsify()
extispicy_report = TokenExtispicyWorkbench(robot_sim).analyze(n_samples=200)

# Compile unified dashboard
dash = MeaningConstructionDashboard(robot_sim)
result = dash.compile(reports={
    "sobol": sobol_report,
    "falsifier": falsifier_report,
    "token_extispicy": extispicy_report,
})

# The dashboard reveals cross-section insights:
# - sensitivity tells you which weights matter (w24, w14)
# - robustness tells you where gaits break (violation_rate ~ 12%)
# - representation tells you if fragmentation predicts failure
print(dash.to_markdown(result))
```

**JGC: Full mitochondrial simulator characterization.** The mitochondrial aging simulator has 12D input (6 intervention + 6 patient), ~40 output metrics across 4 health pillars (energy, damage, dynamics, intervention), and a heteroplasmy cliff at 0.70. Combine all available analyses into a unified report:

```python
from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard
from zimmerman_bridge import MitoSimulator

sim = MitoSimulator()  # full 12D
dashboard = MeaningConstructionDashboard(sim)

# Compile from existing tool reports (run individually first)
compiled = dashboard.compile(reports)
# compiled["coverage"]["tools_present"] → 12 (out of 12 total)
# compiled["recommendations"] → actionable findings list

# The dashboard reveals cross-section insights for the mito simulator:
# - Sobol sensitivity: genetic_vulnerability and transplant_rate are most influential
# - Contrastive: baseline_heteroplasmy near 0.65-0.70 is the cliff boundary
# - POSIWID: clinical intentions often overestimate achievable heteroplasmy reduction
# - Locality: the heteroplasmy cliff creates highly nonlinear local effects
print(dashboard.to_markdown(compiled))
```

---

## Properties & Relations

- **Zimmerman Ch. 6 synthesis.** The dashboard implements the synthesis layer argued for in Chapter 6 of the Zimmerman dissertation: understanding a simulator's behavior requires integrating multiple complementary analytical perspectives. No single tool captures the full picture.
- **Beer POSIWID as overarching philosophy.** Beer (2024): "the Purpose Of a System Is What It Does." The dashboard operationalizes POSIWID by combining six empirical lenses on the simulator's actual behavior, rather than relying on the designer's stated intentions. Each section contributes evidence about what the system *actually does*.
- **Saltelli for sensitivity section.** The sensitivity section draws on Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer" for the variance decomposition methodology. S1 measures independent effects; ST includes interactions; ST - S1 quantifies non-additive coupling.
- **Cross-section recommendations.** The recommendation engine's key strength is combining findings across sections. For example: high fragmentation correlation (representation) + low POSIWID alignment (alignment) together suggest diegetic prompts would help -- a cross-section insight neither tool alone could produce. This follows Zimmerman's argument that *meaning construction* requires relating findings across analytical dimensions.
- **Graceful degradation by design.** The dashboard always produces useful output, even with a single tool report. This mirrors the epistemological principle that partial knowledge is still knowledge -- better to have an incomplete map than no map at all.
- **Simulator protocol composability.** The dashboard can be fed to any zimmerman tool. Running `Falsifier(dashboard)` stress-tests the simulator through the dashboard lens; running `sobol_sensitivity(dashboard)` identifies which parameters most affect the dashboard's combined analytical output.

---

## Possible Issues

- **Recommendation rules are heuristic, not proven.** The threshold values (e.g., `violation_rate > 0.05`, `fragmentation_correlation > 0.3`, `alignment < 0.5`) are based on practical experience with the ER and mitochondrial simulators, not on formal statistical criteria. Treat them as starting points for investigation, not definitive diagnoses.
- **Missing tools reduce insight.** The dashboard gracefully degrades, but recommendations that require cross-section combinations (e.g., fragmentation + alignment) can only trigger when both contributing tools are present. Running more tools yields richer recommendations.
- **Tool key names must match exactly.** The `reports` dict keys must be one of the 12 recognized tool keys (`ALL_TOOL_KEYS`). A typo like `"sobol_sensitivity"` instead of `"sobol"` will cause that report to be silently ignored. Check against `ALL_TOOL_KEYS` if a report seems to be missing.
- **Section builders extract specific dict keys.** Each section builder expects the tool report to have specific keys (e.g., `"rankings"`, `"summary"`, `"aggregate"`). If a tool's output format changes, the corresponding section builder may fail to extract data and will report `"available": False`.
- **Recommendations accumulate, not deduplicate.** If multiple sections trigger similar suggestions, they will appear as separate recommendations. The list is ordered by section, not by priority.
- **to_markdown truncates lists to 5 items.** Long parameter lists or key lists are shortened for readability. Inspect the raw `compile()` result for complete data.

---

## Neat Examples

**Discovering that high token fragmentation + low POSIWID alignment together suggest diegetic prompts would help:**

```python
>>> dash = MeaningConstructionDashboard(sim)

>>> # Scenario: high fragmentation correlation and low alignment
>>> result = dash.compile(reports={
...     "token_extispicy": {
...         "fragmentation_output_correlation": {"fitness": -0.42, "energy": 0.35}
...     },
...     "posiwid": {
...         "alignment": {"overall": 0.38, "per_key": {
...             "fitness": {"combined": 0.32},
...             "energy": {"combined": 0.44},
...         }}
...     },
... })

>>> result["sections"]["representation"]["fragmentation_correlation"]
0.385  # mean |r| of |-0.42| and |0.35|

>>> result["sections"]["alignment"]["overall_alignment"]
0.38

>>> result["recommendations"]
["POSIWID alignment is low (0.38) -- intention-outcome gap needs investigation.",
 "Worst-aligned output keys: fitness, energy. These diverge most from intended outcomes.",
 "High fragmentation correlates with output degradation (|r| = 0.39) -- consider "
 "diegetic prompts to reduce numeric tokenization flattening."]

>>> # Neither tool alone would have suggested diegetic prompts --
>>> # it is the COMBINATION of low alignment and high fragmentation
>>> # that triggers the specific "diegetic prompts" recommendation
```

**Tracking analytical coverage over an iterative investigation:**

```python
>>> reports = {}
>>> result = dash.compile(reports)
>>> len(result["tools_used"]), len(result["tools_missing"])
(0, 12)

>>> reports["sobol"] = sobol_sensitivity(sim)
>>> result = dash.compile(reports)
>>> len(result["tools_used"]), len(result["tools_missing"])
(1, 11)

>>> reports["falsifier"] = Falsifier(sim).falsify()
>>> reports["token_extispicy"] = TokenExtispicyWorkbench(sim).analyze()
>>> result = dash.compile(reports)
>>> len(result["tools_used"]), len(result["tools_missing"])
(3, 9)

>>> # Each compile() adds more sections and richer cross-section recommendations
>>> len(result["recommendations"])
3  # grows as more tools contribute data
```

---

## See Also

`sobol_sensitivity` | `Falsifier` | `POSIWIDAuditor` | `LocalityProfiler` | `Diegeticizer` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Chapter 6 (synthesis and dashboard architecture).
- Beer, S. (2024). "The Purpose Of a System Is What It Does (POSIWID)." (Overarching philosophy: the dashboard reveals what a simulator actually does, not what it is supposed to do.)
- Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., & Tarantola, S. (2008). "Global Sensitivity Analysis: The Primer." Wiley.
