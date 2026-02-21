# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Zimmerman Toolkit is a generalized library for interrogating black-box simulators through sixteen complementary analytical lenses. Given any simulator satisfying the simple `Simulator` protocol (`run(params) -> dict`, `param_spec() -> bounds`), the toolkit answers critical questions: what inputs drive outcomes, where are the tipping points, does the system behave as expected, and where does it fail.

Based on Julia Zimmerman's 2025 PhD dissertation at the University of Vermont: *"Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)."* Eleven of the sixteen modules operationalize specific dissertation findings; the remaining five implement classical analysis techniques (Sobol, Falsifier, ContrastiveGenerator, TrajectoryMetrics, OutputSchema) that serve the thesis-derived tools.

Extracted and generalized from the [how-to-live-much-longer](../how-to-live-much-longer/) mitochondrial aging simulator project.

All numerical operations use **numpy only** (no scipy, no SALib).

## Commands

```bash
# Run the full test suite (385 tests across 16 modules)
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_sobol.py -v
pytest tests/test_trajectory_metrics.py -v
pytest tests/test_output_schema.py -v

# Run a single test
pytest tests/test_sobol.py::TestSobol::test_example -v

# Import and explore interactively
python -c "from zimmerman import sobol_sensitivity; help(sobol_sensitivity)"
python -c "from zimmerman import TrajectoryMetricsProfiler; help(TrajectoryMetricsProfiler)"
```

## Architecture

### The Sixteen Modules

#### Tier 1 -- Classical Analysis

**`sobol`** -- Saltelli sampling + Sobol global sensitivity indices. Reveals which parameters drive outcomes vs. noise.
- S1 (first-order): fraction of output variance per parameter alone
- ST (total-order): fraction including all interactions
- Pure numpy implementation, no SALib dependency

**`falsifier`** -- Systematic falsification: random sampling → boundary testing → adversarial perturbations. Probes for NaN, Inf, violated assumptions. Found 4 real bugs in the JGC mitochondrial ODE.

**`contrastive`** -- Contrastive scenario generation: minimal parameter flips to change outcomes. Reveals system sensitivity via smallest-perturbation paths.

**`contrast_set_generator`** -- Structured edit-space contrast sets (TALOT/OTTITT harness): systematic combinations of minimal edits to explore outcome shifts.

#### Tier 2 -- Zimmerman-Derived (Dissertation §2-3: Locality & Relation)

**`relation_graph_extractor`** -- Meaning-from-relations: extracts multigraph of semantic dependencies from simulator inputs → outputs. Reveals causal structure as LLM would infer it.

**`locality_profiler`** -- Locality profiling via manipulation sweeps (Zimmerman §3.5, §4.6): identifies input-output localities (tight vs. loose couplings). Tests whether locality assumptions hold.

**`prompt_receptive_field`** -- Feature attribution over input segments via Sobol-based ablation (Zimmerman §4.6, §4.7): which segments of input text/parameters influence which outputs?

#### Tier 3 -- Zimmerman-Derived (Dissertation §4.7: Diegesis & Meaning)

**`prompts`** -- Diegetic prompt builder for LLM-mediated design (Zimmerman §2.2.3, §3.5.3, §4.7.6): constructs coherent narrative prompts grounding parameter generation.

**`diegeticizer`** -- Reversible translation between parameter vectors and narrative descriptions. Encodes/decodes parameter semantics ↔ natural language.

**`token_extispicy`** -- Token fragmentation hazard surface analysis (Zimmerman §3.5.3): maps where LLM tokenization may break semantic integrity.

**`supradiegetic_benchmark`** -- Standardized form-vs-meaning battery. Measures diegeticization gain (how much semantic structure survives param ↔ narrative round-trip).

#### Tier 4 -- Framework & Dimensional Analysis (Dissertation §4.6)

**`pds`** -- Power-Danger-Structure dimension mapper (Zimmerman §4.6.4; Dodds et al. 2023): places outcomes in 3D meaning space for structural analysis.

**`posiwid`** -- POSIWID alignment auditor: compares intended vs. actual simulator behavior (Beer 1974; Zimmerman §3.5.2). Reveals design-reality gaps.

#### Tier 5 -- Integration & Aggregation

**`meaning_construction_dashboard`** -- Unified aggregator: synthesizes reports from all 11 Zimmerman-derived modules into coherent narrative of system meaning structure.

#### Tier 6 -- Trajectory & Output Interoperability (NEW)

**`trajectory_metrics`** -- State-space path metrics for ODE simulator trajectories. Computes curvature, smoothness, periodicity, attractor convergence for each variable. `TrajectoryMetricsProfiler` wraps any simulator with `run_trajectory()` method.

**`output_schema`** -- Shared JSON envelope (`SimulatorOutput`) for cross-simulator output interoperability. `validate_output()` and `compare_outputs()` utilities. Each ODE simulator (LEMURS, mito, grief) has a `to_standard_output()` adapter.

### Module Dependency Graph

```
base.py                              ← Simulator protocol (no dependencies)
    ↓
sobol.py                            ← Saltelli sampling + Sobol indices
falsifier.py                        ← Systematic falsification
contrastive.py                      ← Contrastive scenario generation
contrast_set_generator.py           ← Structured contrast sets
    ↓
relation_graph_extractor.py         ← Meaning-from-relations (imports base)
locality_profiler.py                ← Locality profiling (imports base)
prompt_receptive_field.py           ← Receptive field analysis (imports sobol)
pds.py                              ← PDS dimension mapping
posiwid.py                          ← POSIWID alignment
prompts.py                          ← Diegetic prompt builder
diegeticizer.py                     ← Param ↔ narrative translation
token_extispicy.py                  ← Token fragmentation analysis
    ↓
supradiegetic_benchmark.py          ← Form-vs-meaning battery
meaning_construction_dashboard.py   ← Multi-tool aggregator
trajectory_metrics.py               ← State-space metrics for ODE trajectories
output_schema.py                    ← Shared output envelope
```

### Protocol-Based Design

All simulators implement the `Simulator` protocol (duck-typed, runtime-checkable):

```python
class Simulator:
    def run(self, params: dict) -> dict:
        """Execute simulation. Returns named numeric outputs."""

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds: {name: (low, high)}."""
```

Simulators satisfying this protocol are interchangeable targets for any of the 16 modules.

## Usage Examples

### Basic sensitivity analysis

```python
from zimmerman import sobol_sensitivity

result = sobol_sensitivity(my_sim, n_base=256, seed=42)

# First-order indices: which parameters matter most?
print(result["output_name"]["S1"])   # {"param1": 0.42, "param2": 0.31, ...}

# Total-order indices: including interactions
print(result["output_name"]["ST"])

# Ranked by total-order influence
print(result["rankings"])
```

### Falsification & robustness

```python
from zimmerman import Falsifier

falsifier = Falsifier(my_sim, assertions=[
    lambda r: r["energy"] >= 0,      # Custom assertions
    lambda r: r["health"] <= 1.0,
])

report = falsifier.falsify(n_random=200, n_boundary=50, n_adversarial=50)
print(f"Violations found: {report['summary']['violations_found']}")
```

### Trajectory metrics (for ODE simulators)

```python
from zimmerman import trajectory_metrics, TrajectoryMetricsProfiler

# Metrics on raw trajectory data
metrics = trajectory_metrics(states, times, state_names=["x", "y", "z"])
# Returns: path_length, smoothness, curvature, periodicity per variable, attractor_convergence

# Profile a simulator that returns full trajectories
profiler = TrajectoryMetricsProfiler(my_ode_sim)
trajectory_analysis = profiler.run_trajectory(params=my_params)
```

### Shared output schema (cross-simulator interoperability)

```python
from zimmerman import SimulatorOutput, validate_output, compare_outputs

# Each simulator has to_standard_output():
standard = my_sim.to_standard_output(params={})
# Returns: {"schema_version": "1.0", "simulator": {...},
#           "trajectory": {...}, "analytics": {...}, "parameters": {...}}

# Validate conformance
validate_output(standard)

# Compare outputs from two simulators
diff = compare_outputs(output_1, output_2)
```

### Relation graph extraction (meaning structure)

```python
from zimmerman import RelationGraphExtractor

extractor = RelationGraphExtractor(my_sim)
graph = extractor.extract(n_samples=100)

# Reveals causal/relational dependencies
print(graph["edges"])          # Input-output relations
print(graph["transitivity"])   # Indirect influence paths
```

### Diegeticization (parameter ↔ narrative)

```python
from zimmerman import Diegeticizer

diegeticizer = Diegeticizer(my_sim)

# Convert params to narrative
narrative = diegeticizer.encode({"dosage": 50, "frequency": 3})
# "Patient receives 50 mg three times daily..."

# Convert narrative back to params
recovered = diegeticizer.decode(narrative)
# {"dosage": 50, "frequency": 3}

# Measure round-trip fidelity
gain = diegeticizer.round_trip_fidelity(my_samples)
```

## Conventions

- All parameter vectors are plain `dict[str, float]` — no numpy arrays in public APIs
- Parameter vectors sorted by key for consistency in internal operations
- All randomness flows through `np.random.Generator` for reproducibility
- Results use plain dicts for JSON serialization; numpy types wrapped with `NumpyEncoder`
- Sensitivity indices: S1 + ST - S1 partitions total variance into main + interaction effects
- Budget in Sobol means total function evaluations: `N * (D + 2)` where N = `n_base`, D = params
- Bounds clamping applied before simulation execution
- Trajectory metrics assume finite time windows; periodicity detection via FFT frequency peaks

## Relationship to Parent Projects

| Component | how-to-live-much-longer Source | This Toolkit |
|-----------|---|---|
| ODE state-space analysis | `cliff_mapping.py`, `visualize.py` | `trajectory_metrics.py` + `TrajectoryMetricsProfiler` |
| Falsification | `falsifier_report_2026-02-15.md` findings | `falsifier.py` + systematic test harness |
| Contrastive scenarios | Early design iteration | `contrastive.py` + `contrast_set_generator.py` |
| Parameter semantics | Patient narrative integration | `diegeticizer.py`, `prompts.py` |
| Multi-tool reports | Tier 7 integration layer | `meaning_construction_dashboard.py` |
| Output interoperability | Mito/LEMURS/Grief adapters | `output_schema.py` + `SimulatorOutput` protocol |

The how-to-live-much-longer project uses this toolkit for Tiers 5-7 (Zimmerman bridge, Cramer scenarios, integration).

## Cross-Project Usage

The toolkit is used by:
- **how-to-live-much-longer** (`zimmerman_bridge.py`): Tiers 5-6 analysis
- **lemurs-simulator** (`zimmerman_analysis.py`): Sobol + all 16 tools
- **ea-toolkit** (Zimmerman bridge tests): landscape sensitivity analysis
- **scenario-forward-investing-lab**: Sobol sensitivity + interaction mapping
- **stock-simulator**: causal structure via `relation_graph_extractor`
- **rosetta-motion** (via cramer-toolkit): motion robustness analysis

All integrate via the `Simulator` protocol — any compatible simulator can be analyzed with any module.
