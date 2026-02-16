# RelationGraphExtractor

three-frame relation graph construction (causal, similarity, contrast)

---

## Calling Patterns

```
RelationGraphExtractor(simulator)                                        auto-detect all numeric outputs
RelationGraphExtractor(simulator, output_keys=["fitness", "energy"])      analyze specific outputs only

ext.causal_frame(base_params, rng)                                       gradient-based causal edges
ext.similarity_frame(base_params, n_probes, rng)                         local neighborhood probing
ext.contrast_frame(base_params, n_probes, rng)                           global dissimilarity probing

ext.extract(base_params)                                                 default: n_probes=50, seed=42
ext.extract(base_params, n_probes=200, seed=0)                           higher coverage, different seed

ext.run(params)                                                          Simulator protocol (calls extract)
ext.param_spec()                                                         delegates to underlying simulator
```

---

## Details and Options

- `simulator` must satisfy the `Simulator` protocol: `run(params) -> dict` and `param_spec() -> dict[str, (float, float)]`.
- `output_keys` (default `None`) specifies which output keys to include in the relation graph. When `None`, all keys with finite numeric values are auto-detected from a baseline run. Specifying `output_keys` focuses the analysis and reduces computational cost.

### causal_frame

- `base_params` is the parameter dict at which gradients are estimated.
- `rng` is a `numpy.random.Generator` (reserved for future stochastic gradient estimation; currently unused since central differences are deterministic).
- Uses central finite differences with epsilon = 1% of each parameter's range: `d(output_j)/d(p_i) ~ [f(p_i + eps) - f(p_i - eps)] / (2 * eps)`.
- Cost: 2 simulator evaluations per parameter (forward + backward perturbation).
- Returns a list of edge dicts, each with:
  - `"from"`: str -- parameter name (cause).
  - `"to"`: str -- output key (effect).
  - `"weight"`: float -- absolute gradient magnitude (always >= 0).
  - `"sign"`: int -- `+1` if increasing the parameter increases the output, `-1` otherwise.

### similarity_frame

- `base_params` is the center of the local neighborhood search.
- `n_probes` is the number of random perturbations to try.
- `rng` is a `numpy.random.Generator`.
- Perturbations are +/- 5% of each parameter's range (local neighborhood).
- Retains probes with cosine similarity > 0.9 to the base output vector.
- Returns a list of dicts, each with:
  - `"params"`: dict -- the perturbed parameter configuration.
  - `"output"`: dict -- the full simulation result.
  - `"similarity"`: float -- cosine similarity to base output (always > 0.9).

### contrast_frame

- `base_params` is the reference point.
- `n_probes` is the number of random configurations to try.
- `rng` is a `numpy.random.Generator`.
- Configurations are sampled uniformly across the full parameter range (global search).
- Retains probes with cosine similarity < 0.5 to the base output vector.
- Returns a list of dicts, each with:
  - `"params"`: dict -- the contrasting parameter configuration.
  - `"output"`: dict -- the full simulation result.
  - `"similarity"`: float -- cosine similarity to base output (always < 0.5).

### extract

- `base_params` is the target parameter dict around which the graph is built.
- `n_probes` (default 50) is the number of random probes for similarity and contrast frames. More probes give better coverage at higher cost.
- `seed` (default 42) controls the random number generator.
- Computational cost: approximately `1 + 2*D + 2*n_probes + 4*D` simulator evaluations, where D is the number of parameters. The `4*D` term comes from the stability assessment (two additional causal frames).
- Returns a dict with:
  - `"nodes"`: dict with:
    - `"params"`: `{param_name: base_value, ...}` -- parameter node attributes.
    - `"outputs"`: `{output_key: base_value, ...}` -- output node attributes.
  - `"edges"`: dict with:
    - `"causal"`: list of `{from, to, weight, sign}` dicts -- directed parameter -> output edges.
    - `"param_correlation"`: list of `{param_a, param_b, correlation}` dicts -- undirected parameter <-> parameter edges (Pearson correlation across probe set).
    - `"output_correlation"`: list of `{output_a, output_b, correlation}` dicts -- undirected output <-> output edges (Pearson correlation across probe set).
  - `"stability"`: dict with:
    - `"jaccard_overlap"`: float in [0, 1] -- Jaccard index of top-5 causal edges between two independent runs.
    - `"edge_survival_rate"`: float in [0, 1] -- fraction of edges from run A that persist in run B.
  - `"rankings"`: dict with:
    - `"most_causal_params"`: list of param names sorted by total causal weight (sum of edge weights across all outputs), descending.
    - `"most_connected_outputs"`: list of output keys sorted by causal in-degree (number of parameter -> output edges), descending.
  - `"n_sims"`: int -- total simulator evaluations performed.

### run / param_spec

- `run(params)` calls `extract(params)` with default settings, making `RelationGraphExtractor` itself satisfy the `Simulator` protocol for meta-analysis composability.
- `param_spec()` delegates to the underlying simulator's `param_spec()`.

---

## Basic Examples

Create an extractor and build the relation graph for a 2-parameter model:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.relation_graph_extractor import RelationGraphExtractor

>>> def model(p):
...     return {"y": 3 * p["a"] + p["b"] ** 2, "z": p["a"] * p["b"]}

>>> sim = SimulatorWrapper(model, {"a": (0.0, 1.0), "b": (0.0, 1.0)})
>>> ext = RelationGraphExtractor(sim)
>>> graph = ext.extract({"a": 0.5, "b": 0.5}, n_probes=50)

>>> graph["nodes"]["params"]
{"a": 0.5, "b": 0.5}

>>> graph["nodes"]["outputs"]
{"y": 1.75, "z": 0.25}
```

Inspect causal edges to see which parameters drive which outputs:

```python
>>> for edge in sorted(graph["edges"]["causal"], key=lambda e: e["weight"], reverse=True):
...     print(f"{edge['from']} -> {edge['to']}  weight={edge['weight']:.2f}  sign={edge['sign']:+d}")
a -> y  weight=3.00  sign=+1
b -> y  weight=1.00  sign=+1
b -> z  weight=0.50  sign=+1
a -> z  weight=0.50  sign=+1

>>> graph["rankings"]["most_causal_params"]
["a", "b"]

>>> graph["rankings"]["most_connected_outputs"]
["y", "z"]
```

Check stability of the extracted graph:

```python
>>> graph["stability"]["jaccard_overlap"]
1.0  # deterministic simulator: perfect agreement between independent runs

>>> graph["stability"]["edge_survival_rate"]
1.0
```

---

## Scope

Custom output keys focus the analysis on specific outputs:

```python
>>> ext = RelationGraphExtractor(sim, output_keys=["y"])
>>> graph = ext.extract({"a": 0.5, "b": 0.5})
>>> [e["to"] for e in graph["edges"]["causal"]]
["y", "y"]  # only causal edges to "y", "z" is ignored
```

Works with any number of parameters:

```python
>>> spec = {f"x{i}": (0.0, 1.0) for i in range(10)}
>>> def high_d(p):
...     return {"sum": sum(p.values()), "max": max(p.values())}
>>> sim = SimulatorWrapper(high_d, spec)
>>> ext = RelationGraphExtractor(sim)
>>> graph = ext.extract({f"x{i}": 0.5 for i in range(10)}, n_probes=30)
>>> len(graph["edges"]["causal"])
20  # 10 params * 2 outputs
>>> graph["n_sims"]  # 1 + 2*10 + 2*30 + 4*10 = 141
141
```

Analyzing non-standard simulators with string or list outputs (non-numeric keys are silently skipped):

```python
>>> def mixed_model(p):
...     return {"fitness": p["a"] * 2, "label": "category_A", "history": [1, 2, 3]}
>>> sim = SimulatorWrapper(mixed_model, {"a": (0, 1)})
>>> ext = RelationGraphExtractor(sim)
>>> graph = ext.extract({"a": 0.5})
>>> list(graph["nodes"]["outputs"].keys())
["fitness"]  # only numeric keys are included
```

---

## Applications

**ER: Causal map of weight -> gait metrics.** Build the causal relation graph for a specific robot gait to understand which synaptic weights control which behavioral metrics:

```python
ext = RelationGraphExtractor(robot_sim, output_keys=["distance", "energy", "stability", "complexity"])
graph = ext.extract(zoo["5_pelton"]["weights"], n_probes=100)

# Which weights most strongly control the gait?
print(graph["rankings"]["most_causal_params"])
# ["w24", "w14", "w04"] -- hip joint weights dominate

# Are distance and energy correlated or anti-correlated?
for edge in graph["edges"]["output_correlation"]:
    print(f"{edge['output_a']} <-> {edge['output_b']}  corr={edge['correlation']:.3f}")
# distance <-> energy  corr=-0.72  -- faster gaits use less energy (efficient)
# stability <-> complexity  corr=0.45  -- complex gaits tend to be more stable

# Is the graph stable?
print(graph["stability"]["jaccard_overlap"])
# 1.0 -- deterministic simulator, perfectly reproducible causal structure
```

**JGC: Which interventions causally drive which health pillars?** Map the causal structure of the mitochondrial aging simulator:

```python
ext = RelationGraphExtractor(mito_sim, output_keys=["het_final", "atp_final", "ros_level", "cell_health"])
graph = ext.extract(patient_params, n_probes=80)

# Which interventions have the broadest causal influence?
print(graph["rankings"]["most_causal_params"][:3])
# ["exercise", "caloric_restriction", "antioxidant"]

# Which health pillars are most connected (influenced by the most interventions)?
print(graph["rankings"]["most_connected_outputs"])
# ["het_final", "atp_final", "ros_level", "cell_health"]

# Discover hidden coupling: do health pillars move together?
for edge in graph["edges"]["output_correlation"]:
    if abs(edge["correlation"]) > 0.7:
        print(f"{edge['output_a']} <-> {edge['output_b']}  corr={edge['correlation']:.3f}")
# het_final <-> cell_health  corr=-0.91  -- heteroplasmy directly degrades cell health
# atp_final <-> ros_level   corr=-0.78  -- ATP and ROS are antagonistic
```

---

## Properties & Relations

- **Meaning-from-relations (Zimmerman, 2025, Ch. 3).** The relation graph operationalizes the thesis that a parameter configuration's meaning is constituted by its web of relations: what it causes (causal frame), what it resembles (similarity frame), and what it differs from (contrast frame). The graph *is* the meaning.
- **POSIWID for the causal frame (Beer, 2024).** The causal frame implements Beer's "The Purpose Of a System Is What It Does": a parameter's purpose is inferred from its observable causal effects (gradients), not from any a priori label. A parameter with high causal weight on an output *is* a controller of that output, regardless of its name.
- **Jaccard stability (Jaccard, 1912).** The split-half stability assessment uses the Jaccard index `J(A, B) = |A intersect B| / |A union B|` to measure agreement between two independent causal graph extractions. For deterministic simulators, `J = 1.0`. For stochastic simulators, J quantifies the signal-to-noise ratio of the graph structure.
- **Network science (Newman, 2010).** The multigraph structure (typed nodes, typed weighted edges) follows standard network science conventions. `most_causal_params` is a strength-centrality ranking; `most_connected_outputs` is an in-degree ranking in the bipartite causal subgraph. The full graph can be exported for analysis with tools like NetworkX.
- **Central difference accuracy.** The causal frame uses central differences with error O(eps^2), which is more accurate than forward differences (error O(eps)). The epsilon of 1% of parameter range balances numerical accuracy against floating-point stability.
- **Cosine similarity for qualitative comparison.** Similarity and contrast frames use cosine similarity rather than Euclidean distance. This captures qualitative similarity (same pattern of relative output values) regardless of quantitative scale, aligning with the relational semantics perspective.
- **Simulator protocol composability.** `RelationGraphExtractor` satisfies the `Simulator` protocol (`run` + `param_spec`), enabling meta-analysis. For example, running `sobol_sensitivity` on a `RelationGraphExtractor` measures how the causal structure itself varies across the parameter space.

---

## Possible Issues

- **Central difference epsilon tuning.** The default epsilon (1% of parameter range) works well for smooth simulators. For highly non-linear simulators with sharp gradients, this epsilon may be too large (smoothing over important structure) or too small (amplifying numerical noise). Adaptive step-size selection is not implemented.
- **Cosine similarity threshold.** The similarity frame uses a hardcoded threshold of 0.9; the contrast frame uses 0.5. These thresholds are design choices, not tunable parameters. For simulators with very uniform outputs, the similarity frame may return nearly all probes; for simulators with very diverse outputs, the contrast frame may return nearly all probes.
- **O(n_params^2) correlation computation.** The `_compute_correlations` method computes pairwise Pearson correlations for all parameter pairs and all output pairs. For D parameters and K outputs, this is O(D^2 + K^2) correlation computations, each over the full probe set. For high-dimensional simulators (D > 50), this can become a bottleneck.
- **Pearson correlation limitations.** Only linear relationships are captured. Non-linear dependencies between outputs (e.g., one output is the square of another) will have low Pearson correlation despite strong functional coupling. Spearman rank correlation or mutual information would be more robust but are not implemented (numpy-only constraint).
- **Minimum probe count for correlations.** If fewer than 3 probes pass the similarity or contrast threshold (combined), correlation computation returns empty lists. Increase `n_probes` to ensure sufficient data.
- **Computational cost.** For D parameters and n_probes probes, the total cost is `1 + 6*D + 2*n_probes` simulator evaluations. For D=12 and n_probes=50, that is 173 evaluations. The `4*D` stability overhead from the two additional causal frames is the largest per-parameter cost.
- **Non-finite outputs.** Parameters or outputs producing NaN or inf values are handled gracefully (NaN propagation, finite-check filtering), but excessive non-finite outputs degrade correlation estimates and similarity/contrast frame coverage.

---

## Neat Examples

**Discovering hidden output correlations revealing internal simulator coupling.** Even when outputs appear unrelated by name, the relation graph can reveal deep structural coupling:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.relation_graph_extractor import RelationGraphExtractor

>>> # A simulator with hidden internal coupling: both outputs depend on
>>> # the same latent variable (a + b), but through different transformations
>>> def coupled_model(p):
...     latent = p["a"] + p["b"]
...     return {
...         "alpha": np.sin(latent * 3.14),
...         "beta": latent ** 2,
...         "gamma": p["c"] * 0.5,  # independent of the latent variable
...     }

>>> sim = SimulatorWrapper(coupled_model, {"a": (0, 1), "b": (0, 1), "c": (0, 1)})
>>> ext = RelationGraphExtractor(sim)
>>> graph = ext.extract({"a": 0.3, "b": 0.3, "c": 0.5}, n_probes=100)

>>> # Output correlations reveal the hidden coupling
>>> for edge in graph["edges"]["output_correlation"]:
...     print(f"{edge['output_a']} <-> {edge['output_b']}  corr={edge['correlation']:+.3f}")
alpha <-> beta   corr=+0.85   # both driven by (a + b) -- coupled!
alpha <-> gamma  corr=-0.02   # independent -- gamma depends only on c
beta  <-> gamma  corr=+0.01   # independent

>>> # Causal edges confirm: a and b both drive alpha and beta, c only drives gamma
>>> for edge in sorted(graph["edges"]["causal"], key=lambda e: e["weight"], reverse=True)[:4]:
...     print(f"{edge['from']} -> {edge['to']}  weight={edge['weight']:.2f}")
a -> beta   weight=1.20
b -> beta   weight=1.20
a -> alpha  weight=0.95
b -> alpha  weight=0.95
# c -> gamma is the only edge involving c or gamma
```

**Comparing causal structures across behavioral regimes.** Extract graphs at two different operating points to see how the causal structure changes:

```python
>>> # Near the heteroplasmy cliff vs. far from it
>>> safe_graph = ext.extract({"exercise": 0.8, "antioxidant": 0.5, ...})
>>> risky_graph = ext.extract({"exercise": 0.2, "antioxidant": 0.1, ...})

>>> # Compare which parameters are most causal in each regime
>>> print("Safe regime:", safe_graph["rankings"]["most_causal_params"][:3])
Safe regime: ["exercise", "caloric_restriction", "antioxidant"]
>>> print("Risk regime:", risky_graph["rankings"]["most_causal_params"][:3])
Risk regime: ["genetic_vulnerability", "metabolic_demand", "exercise"]
# Near the cliff, genetic_vulnerability dominates -- the causal structure shifts
```

---

## See Also

`sobol_sensitivity` | `ContrastSetGenerator` | `LocalityProfiler` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Chapters 2-3 (relational semantics, meaning-from-relations).
- Beer, S. (2024). *The Heart of Enterprise.* Revised edition. (Original work published 1979.) Chapter 7 (POSIWID).
- Newman, M.E.J. (2010). *Networks: An Introduction.* Oxford University Press.
- Jaccard, P. (1912). "The Distribution of the Flora in the Alpine Zone." *New Phytologist*, 11(2), 37-50.
