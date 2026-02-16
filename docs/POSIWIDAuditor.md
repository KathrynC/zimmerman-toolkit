# POSIWIDAuditor

measures the gap between intended outcomes and actual simulation outcomes

---

## Calling Patterns

```
POSIWIDAuditor(simulator)                                     create an auditor
```

```
auditor.audit(intended_outcomes, params)                      single audit
```

```
auditor.score_alignment(intended, actual)                     score without running simulator
```

```
auditor.batch_audit(scenarios)                                audit multiple scenarios with aggregation
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol.

### audit

- `intended_outcomes`: dict mapping output key names to their intended (expected) float values. Example: `{"fitness": 0.8, "energy": 0.5}`.
- `params`: dict mapping parameter names to float values to pass to the simulator.
- Runs `simulator.run(params)`, then calls `score_alignment(intended_outcomes, actual_result)`.
- Returns a dict with:
  - `"intended"`: copy of intended_outcomes.
  - `"actual"`: the simulator output dict.
  - `"params"`: copy of params.
  - `"alignment"`: alignment scores from `score_alignment()`.

### score_alignment

- For each key in `intended`:
  - **direction_match** (binary 0 or 1): Does the actual value have the same sign as the intended value? When intended is 0.0, direction_match penalizes by magnitude: `max(0, 1 - |actual| * 5)`. When actual is 0.0 and intended is nonzero, direction_match is 0.
  - **magnitude_match** (continuous 0 to 1): `max(0, 1 - |actual - intended| / scale)` where `scale = max(|intended|, 0.1)`. The 0.1 floor prevents division-by-zero for small intended values.
  - **combined**: `0.5 * direction_match + 0.5 * magnitude_match`.
- Keys in `intended` that are missing from `actual` (or non-numeric) are counted as `n_keys_missing` and excluded from scores.
- Non-finite values (NaN, Inf) get all-zero scores.
- Returns a dict with:
  - `"per_key"`: per-key scores.
  - `"overall"`: mean of all combined scores.
  - `"n_keys_matched"`: count of keys found and scored.
  - `"n_keys_missing"`: count of keys not found or non-numeric.

### batch_audit

- `scenarios`: list of dicts, each with `"intended"` and `"params"` keys (optionally `"label"`).
- Runs `audit()` for each scenario and computes aggregate statistics.
- Returns a dict with:
  - `"individual_results"`: list of audit results.
  - `"aggregate"`: dict with:
    - `"mean_overall"`, `"std_overall"`, `"min_overall"`, `"max_overall"`.
    - `"mean_direction_accuracy"`, `"mean_magnitude_accuracy"`.
    - `"n_scenarios"`.
    - `"per_key_mean"`: mean scores per output key across all scenarios.

---

## Basic Examples

Audit whether a simulator achieves the intended outcome:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.posiwid import POSIWIDAuditor

>>> def model(p):
...     return {"fitness": p["x"] * 2, "energy": 1 - p["x"]}

>>> sim = SimulatorWrapper(model, {"x": (0.0, 1.0)})
>>> auditor = POSIWIDAuditor(sim)

>>> result = auditor.audit(
...     intended_outcomes={"fitness": 1.0, "energy": 0.5},
...     params={"x": 0.5}
... )

>>> result["actual"]
{"fitness": 1.0, "energy": 0.5}

>>> result["alignment"]["overall"]
1.0  # perfect alignment

>>> result["alignment"]["per_key"]["fitness"]
{"direction_match": 1.0, "magnitude_match": 1.0, "combined": 1.0}
```

Poor alignment example:

```python
>>> result = auditor.audit(
...     intended_outcomes={"fitness": 1.0, "energy": 0.5},
...     params={"x": 0.1}
... )
>>> result["actual"]
{"fitness": 0.2, "energy": 0.9}

>>> result["alignment"]["per_key"]["fitness"]["magnitude_match"]
0.2  # actual=0.2 is far from intended=1.0

>>> result["alignment"]["per_key"]["energy"]["direction_match"]
1.0  # same sign (positive), direction correct
```

---

## Scope

Score alignment without running the simulator:

```python
>>> intended = {"a": 1.0, "b": -0.5}
>>> actual = {"a": 0.8, "b": -0.3}
>>> auditor.score_alignment(intended, actual)
{"per_key": {"a": {...}, "b": {...}}, "overall": 0.8, ...}
```

Batch audit across multiple scenarios:

```python
>>> scenarios = [
...     {"intended": {"fitness": 0.8}, "params": {"x": 0.4}},
...     {"intended": {"fitness": 0.8}, "params": {"x": 0.5}},
...     {"intended": {"fitness": 0.8}, "params": {"x": 0.6}},
... ]
>>> batch = auditor.batch_audit(scenarios)
>>> batch["aggregate"]["mean_overall"]
0.73
>>> batch["aggregate"]["std_overall"]
0.15
```

Handles missing output keys gracefully:

```python
>>> result = auditor.audit(
...     intended_outcomes={"fitness": 1.0, "nonexistent_key": 0.5},
...     params={"x": 0.5}
... )
>>> result["alignment"]["n_keys_missing"]
1
>>> result["alignment"]["n_keys_matched"]
1
```

---

## Applications

**TIQM offer/confirmation wave auditing.** In the Transactional Interpretation framework, the offer wave (LLM generating parameters) has an intended outcome. The confirmation wave (VLM analyzing results) measures actual outcomes. POSIWID quantifies the gap:

```python
# Offer wave: LLM intended the robot to walk forward with fitness 0.8
intended = {"dx": 2.0, "efficiency": 0.8, "stability": 0.9}
# Actual: run the simulator with the LLM-generated weights
result = auditor.audit(intended, llm_generated_weights)
# result["alignment"]["overall"] = 0.45 → LLM's intention poorly realized
```

**LLM calibration across models.** Compare how well different LLMs translate intentions into parameters:

```python
scenarios_by_model = {
    "qwen3-coder": [...],
    "deepseek-r1": [...],
    "llama3.1": [...],
}
for model, scenarios in scenarios_by_model.items():
    batch = auditor.batch_audit(scenarios)
    print(f"{model}: mean alignment = {batch['aggregate']['mean_overall']:.3f}")
```

---

## Properties & Relations

- **POSIWID principle.** "The Purpose Of a System Is What It Does" — Stafford Beer (1974). In Zimmerman's thesis (§3.5.2), POSIWID is used to analyze objective function misalignment — e.g., the tokenizer's purpose (compress orthographically) is not aligned with the model's purpose (learn semantics). The toolkit applies POSIWID to simulation: the auditor compares the *intended* purpose (the offer wave) against what the system *actually does* (the confirmation wave).
- **Scoring decomposition.** The combined score is an equal-weighted average of direction and magnitude. Direction captures whether the system moved in the right direction; magnitude captures how close it got. For binary outcomes, direction dominates. For continuous optimization, magnitude dominates.
- **Scale normalization.** The magnitude score uses `max(|intended|, 0.1)` as the denominator. This prevents small intended values (e.g., 0.01) from creating outsized error ratios, but means that near-zero intended values have a "dead zone" of tolerance.
- `score_alignment()` is a pure function — it can be used without a simulator, making it useful for post-hoc analysis of pre-existing simulation data.
- Complementary to `Falsifier`: POSIWID measures *alignment* (does it do what we want?), falsification measures *robustness* (does it break?).

---

## Possible Issues

- **Direction scoring for zero intended values.** When `intended_val == 0.0`, the direction score is `max(0, 1 - |actual| * 5)`. This means actual values within ±0.2 of zero get partial credit, but the scoring is not symmetric with the nonzero case.
- **Equal weighting.** The 50/50 split between direction and magnitude is hardcoded. For applications where one matters more (e.g., drug dosing where direction is critical), the scores should be reweighted manually from `per_key` data.
- **Non-numeric outputs.** Output values that are not `int`, `float`, or numpy numeric types are counted as missing, not as errors.
- **Batch aggregation.** `per_key_mean` averages across all scenarios where that key appears. If a key is missing in some scenarios, the mean only reflects scenarios where it was present.

---

## Neat Examples

**Measuring the "intention gap" across a parameter sweep:**

```python
# How does alignment change as we sweep one parameter?
alignments = []
for x in np.linspace(0, 1, 100):
    result = auditor.audit({"fitness": 0.8}, {"x": x})
    alignments.append(result["alignment"]["overall"])

# Plot: alignment vs x reveals the optimal operating region
# where intention best matches reality
```

---

## See Also

`Simulator` | `Falsifier` | `PDSMapper` | `PromptBuilder`

---

## References

- Beer, Stafford (1974). "Designing Freedom." CBC Massey Lectures.
- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. §3.5.2.
