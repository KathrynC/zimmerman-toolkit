# PromptBuilder

builds structured prompts for LLM-mediated parameter generation in three styles

---

## Calling Patterns

```
PromptBuilder(simulator)                                      no extra context
PromptBuilder(simulator, context={"domain": "robotics"})      with context dict
```

```
builder.build_numeric(scenario)                               straightforward parameter request
```

```
builder.build_diegetic(scenario)                              narrative/contextual prompt
builder.build_diegetic(scenario, state_description="...")      with current state
```

```
builder.build_contrastive(scenario)                           two agents: "cautious" vs "aggressive"
builder.build_contrastive(scenario, agent_a="conservative", agent_b="radical")  custom agents
```

---

## Details and Options

### Constructor

- `simulator` must satisfy the `Simulator` protocol (only `param_spec()` is used; `run()` is never called).
- `context`: optional dict of additional context. Keys become section headers, values become section content. Appended to all prompt types.

### build_numeric

- `scenario`: string describing the design goal (e.g., "Design a fast robot").
- Generates a prompt that:
  1. Sets the role: "simulation design specialist."
  2. Presents the scenario.
  3. Lists all parameters with their `(min, max)` ranges.
  4. Includes context sections if provided.
  5. Asks for brief reasoning then a JSON object with all parameter keys.
- Returns the complete prompt string.

### build_diegetic

- `scenario`: string describing the design goal.
- `state_description` (default `None`): description of the current simulation state.
- Based on Zimmerman (2025) Ch. 2 (§2.2.3) and Ch. 3 (§3.5.3): LLMs handle diegetic content (meaning, semantics) better than supradiegetic content (form, structure). Numbers are particularly vulnerable to tokenization-induced *flattening* — their mathematical structure is destroyed when parsed into arbitrary token boundaries.
- Generates a prompt that:
  1. Frames the scenario as a real situation.
  2. Optionally includes the current state.
  3. Presents each parameter as a judgment call: "How strongly should this be set?" with min, max, and midpoint.
  4. Asks the LLM to consider tradeoffs and interactions.
  5. Requests reasoning followed by a JSON object.
- Returns the complete prompt string.

### build_contrastive

- `scenario`: string describing the design goal.
- `agent_a` (default `"cautious"`): name/description of the conservative agent.
- `agent_b` (default `"aggressive"`): name/description of the aggressive agent.
- Exploits TALOT/OTTITT meaning-from-contrast (Zimmerman §4.7.6): "Things Are Like Other Things" vs. "Only The Thing Is The Thing." Meaning emerges from the tension between identity and difference.
- Generates a prompt requesting TWO parameter sets: one from each agent's perspective.
- Expert A "prioritizes safety, stability, and minimal intervention."
- Expert B "believes the situation requires maximum response."
- Output format: a JSON object with two keys whose names are the *values* of `agent_a` and `agent_b` (e.g., `"cautious"` and `"aggressive"` by default), each containing a full parameter dict.
- Returns the complete prompt string.

---

## Basic Examples

Build a numeric prompt:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.prompts import PromptBuilder

>>> sim = SimulatorWrapper(lambda p: {}, {"speed": (0, 10), "armor": (0, 5)})
>>> builder = PromptBuilder(sim)
>>> prompt = builder.build_numeric("Design a fast but resilient robot")

>>> print(prompt)
You are a simulation design specialist. Given the scenario below,
choose parameter values that best achieve the described goal.

SCENARIO:
Design a fast but resilient robot

PARAMETERS (name: (min, max)):
  speed: (0, 10)
  armor: (0, 5)

Choose a value for each parameter within its valid range.

Output a JSON object with ALL parameter keys:
{"speed": _, "armor": _}

Brief reasoning (1-2 sentences), then ONLY the JSON object.
```

Build a diegetic prompt with state:

```python
>>> prompt = builder.build_diegetic(
...     "Optimize the robot for rough terrain",
...     state_description="Currently falling over after 2 seconds on gravel"
... )
# Prompt frames each parameter as a judgment call with narrative context
```

Build a contrastive prompt with custom agents:

```python
>>> prompt = builder.build_contrastive(
...     "Navigate a minefield",
...     agent_a="cautious engineer",
...     agent_b="bold explorer"
... )
# Requests TWO parameter sets from opposing perspectives
```

---

## Scope

Add context to enrich prompts:

```python
>>> builder = PromptBuilder(sim, context={
...     "domain": "3-link walking robot, 4000 timesteps at 240 Hz",
...     "previous_result": "Last attempt: dx=0.3m, fell at step 2800",
...     "constraint": "Must not fall (torso_contacts == 0)",
... })
>>> prompt = builder.build_numeric("Improve the gait")
# Context sections appear after parameters
```

Works with any parameter count:

```python
>>> spec = {f"w{i}": (-1, 1) for i in range(20)}
>>> sim = SimulatorWrapper(lambda p: {}, spec)
>>> prompt = PromptBuilder(sim).build_numeric("High-dimensional design")
# All 20 parameters listed with ranges
```

---

## Applications

**TIQM offer wave generation.** In the Transactional Interpretation pipeline, `PromptBuilder` generates the offer wave prompt:

```python
builder = PromptBuilder(robot_sim, context={
    "domain": "3-link PyBullet robot, 6 synaptic weights",
    "gait_class": "antifragile walker",
})

# Offer wave via diegetic prompt (Zimmerman recommends over numeric)
prompt = builder.build_diegetic(
    "Design a robot that walks forward reliably even with perturbations",
    state_description="Previous gait achieved 2.1m but fell at step 3500"
)

# Send to LLM (qwen3-coder:30b for offer wave)
response = ollama.generate("qwen3-coder:30b", prompt)
weights = parse_json(response)

# Run simulation, then audit alignment with POSIWIDAuditor
```

**Contrastive bracket exploration.** Generate opposing parameter sets to bracket the solution space:

```python
prompt = builder.build_contrastive(
    "Design a robot gait for a 3-link walker",
    agent_a="stability-focused engineer",
    agent_b="speed-maximizing optimizer"
)
# LLM returns two parameter sets
# Use both as starting points for ContrastiveGenerator to find the boundary
```

**Model comparison.** Same prompt to different LLMs reveals how each constructs meaning from the scenario:

```python
prompt = builder.build_diegetic("Design an aggressive forward walker")
for model in ["qwen3-coder:30b", "deepseek-r1:8b", "llama3.1"]:
    response = ollama.generate(model, prompt)
    # Compare: do different LLMs produce different weight patterns?
    # Hypothesis from Zimmerman (2025): since all LLMs share diegetic strengths,
    # diegetic prompts should yield more consistent cross-model behavior
```

---

## Properties & Relations

- **Three prompt styles map to three cognitive strategies:**
  - *Numeric*: direct parameter selection. The supradiegetic baseline — numbers are the content LLMs process worst due to tokenization-induced flattening (§3.5.3). Best only when the LLM has strong domain knowledge.
  - *Diegetic*: narrative framing. Aligns with Zimmerman (2025) §2.2.3: LLMs excel at diegetic content (distributional semantics, meaning) because their diegetic realm is "basically, [their] entire world." Produces more thoughtful parameter choices.
  - *Contrastive*: opposing perspectives. Exploits TALOT/OTTITT meaning-from-contrast (§4.7.6). Forces the LLM to navigate the identity-difference spectrum explicitly, reasoning about tradeoffs through the tension between opposing perspectives.
- **No simulation execution.** `PromptBuilder` only reads `param_spec()`. It never calls `run()`. The prompts are strings meant to be sent to an external LLM.
- **JSON template generation.** The `_format_param_json_template()` method creates a template like `{"speed": _, "armor": _}` to guide the LLM's output format.
- Complementary to `POSIWIDAuditor`: `PromptBuilder` generates the *intention* (offer wave), `POSIWIDAuditor` measures how well the intention was *realized* (confirmation wave).

---

## Possible Issues

- **LLM output parsing.** `PromptBuilder` generates prompts but does not parse responses. The caller must handle JSON extraction, markdown fence stripping, `<think>` tag removal, and grid snapping of numeric outputs.
- **Prompt length.** For simulators with many parameters (D > 20), prompts become long. Some LLMs may truncate or lose attention. Consider grouping parameters or using `output_keys` to focus.
- **Context ordering.** Context sections are rendered in dict iteration order (insertion order in Python 3.7+). Order may matter for LLM attention.
- **Diegetic midpoint display.** The diegetic prompt shows midpoints with 2 decimal places (`:.2f`). For parameters with very small ranges (e.g., 0.001 to 0.002), the midpoint display may be misleading.

---

## Neat Examples

**Iterative refinement loop.** Use prompt output as feedback for the next iteration:

```python
result = None
for iteration in range(5):
    state = f"Iteration {iteration}: dx={result['dx']:.2f}m" if result else None
    prompt = builder.build_diegetic("Maximize forward distance", state_description=state)
    weights = send_to_llm(prompt)
    result = sim.run(weights)
    # Each iteration, the LLM sees the previous result and adjusts
```

---

## See Also

`Simulator` | `POSIWIDAuditor` | `PDSMapper` | `ContrastiveGenerator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. §2.2.3, §3.5.3, §4.7.6.
- Zimmerman, J.W., Hudon, D., Cramer, K., et al. (2024). "A blind spot for large language models: Supradiegetic linguistic information." *Plutonics*, 17, 107-156.
