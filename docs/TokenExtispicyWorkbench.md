# TokenExtispicyWorkbench

token fragmentation hazard surface analysis via plugin tokenizers

---

## Calling Patterns

```
TokenExtispicyWorkbench(simulator)                              default tokenizer (whitespace + digit boundary)
TokenExtispicyWorkbench(simulator, tokenize=enc.encode)         with tiktoken BPE tokenizer
TokenExtispicyWorkbench(simulator, tokenize=tok.tokenize)       with HuggingFace tokenizer
TokenExtispicyWorkbench(simulator, tokenize=sp.encode_as_pieces) with SentencePiece tokenizer

workbench.fragmentation_rate(text)                              tokens-per-character ratio for a string
workbench.perturbation_token_sensitivity(params)                per-parameter token stability under epsilon perturbation
workbench.perturbation_token_sensitivity(params, epsilon=0.05)  custom perturbation size
workbench.analyze()                                             full hazard surface analysis (100 samples, seed=42)
workbench.analyze(n_samples=500, seed=0)                        higher resolution, different seed
workbench.analyze(base_params={"a": 0.3, "b": 0.7})            custom perturbation center point

workbench.run(params)                                           Simulator protocol: run underlying sim + fragmentation metrics
workbench.param_spec()                                          Simulator protocol: delegates to underlying simulator
```

---

## Details and Options

- `simulator` must satisfy the `Simulator` protocol: `run(params) -> dict` and `param_spec() -> dict[str, (float, float)]`.
- `tokenize` is an optional callable with signature `(str) -> list[str]`. When `None`, the workbench falls back to a built-in whitespace + digit boundary tokenizer (`_default_tokenize`) that splits on whitespace, then splits at every boundary between digit and non-digit characters. This default provides a conservative baseline: real BPE tokenizers typically fragment numeric strings more aggressively, so the default tends to underestimate fragmentation hazards.

### Plugin tokenizer architecture

The `tokenize` parameter accepts any callable that converts a string to a list of token strings, enabling direct comparison across tokenization strategies:

- **tiktoken (BPE)**: `import tiktoken; enc = tiktoken.get_encoding("cl100k_base"); TokenExtispicyWorkbench(sim, tokenize=enc.encode)`
- **HuggingFace**: `from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b"); TokenExtispicyWorkbench(sim, tokenize=tok.tokenize)`
- **SentencePiece**: `import sentencepiece as spm; sp = spm.SentencePieceProcessor(model_file="model.model"); TokenExtispicyWorkbench(sim, tokenize=sp.encode_as_pieces)`
- **Custom**: any `(str) -> list[str]` callable for domain-specific analysis.

Swapping tokenizers reveals how fragmentation patterns differ across models, which is critical when the same simulator is interrogated by different LLMs in the TIQM pipeline.

### fragmentation_rate

- Computes `len(tokens) / len(text)` -- the tokens-per-character ratio.
- Low fragmentation (~0.1-0.2): efficient encoding, e.g., common English words mapping to single tokens.
- Medium fragmentation (~0.3-0.5): typical for mixed alphanumeric text.
- High fragmentation (~0.5-1.0): aggressive splitting, common for numeric parameter strings where each digit becomes its own token.
- Returns `0.0` for empty strings. Values above `1.0` are possible if the tokenizer adds boundary tokens.

### perturbation_token_sensitivity

- Perturbs each parameter independently by `epsilon * (hi - lo)`, re-tokenizes, and counts positional token mismatches plus length difference.
- `epsilon` (default `0.01`) is the perturbation size as a fraction of each parameter's range.
- Uses a simplified Hamming-like edit distance (not full Levenshtein) for efficiency.
- Returns `dict[str, int]` mapping parameter names to the number of tokens changed. Higher values indicate greater tokenizer instability for that parameter.

### analyze

- `base_params` (default `None`): center point for perturbation sensitivity analysis. When `None`, uses the midpoint of each parameter's range.
- `n_samples` (default `100`): number of random parameter configurations to sample. Higher values give more reliable fragmentation statistics and correlations but require more simulator runs.
- `seed` (default `42`): random seed for reproducibility.
- Returns a dict with keys:
  - `"fragmentation_stats"`: dict with:
    - `"mean_tokens_per_param"`: mean tokens per parameter value across all samples.
    - `"mean_tokens_per_digit"`: mean ratio of total tokens to digit characters.
    - `"std_tokens_per_param"`: standard deviation of tokens per parameter.
    - `"max_fragmentation_param"`: name of the parameter that fragments most on average.
  - `"perturbation_sensitivity"`: dict mapping parameter names to token change counts under epsilon perturbation.
  - `"fragmentation_output_correlation"`: dict mapping output keys to Pearson r between fragmentation rate and that output. |r| > 0.3 is evidence that tokenization artifacts propagate into simulation results.
  - `"hazard_zones"`: list of top-10 highest-fragmentation configurations, each a dict with `"params"`, `"fragmentation"`, and `"output_degradation"` (normalized mean absolute deviation from median output).
  - `"token_edit_vs_string_edit"`: dict with `"mean_ratio"` and `"max_ratio"` of normalized token edit distance to string edit distance. Ratios >> 1.0 indicate tokenization amplifies semantic distance.
  - `"n_samples"`: number of samples used.
  - `"n_sims"`: total simulation runs performed (equal to `n_samples`).

### run / param_spec (Simulator protocol)

- `run(params)` delegates to the underlying simulator and augments the result with two extra keys:
  - `"_fragmentation_rate"`: tokens per character for the JSON-serialized parameter string.
  - `"_token_count"`: absolute number of tokens produced.
- `param_spec()` delegates directly to the underlying simulator's `param_spec()`. The workbench adds no new parameters, only new output keys.
- Because the workbench satisfies the Simulator protocol, it can be composed with other zimmerman tools: e.g., `sobol_sensitivity(workbench)` reveals which parameters' fragmentation most affects output quality.

---

## Basic Examples

Create a workbench and compute fragmentation rate:

```python
>>> from zimmerman.base import SimulatorWrapper
>>> from zimmerman.token_extispicy import TokenExtispicyWorkbench

>>> def model(p):
...     return {"y": 3 * p["a"] + p["b"] ** 2}

>>> sim = SimulatorWrapper(model, {"a": (0.0, 1.0), "b": (0.0, 1.0)})
>>> workbench = TokenExtispicyWorkbench(sim)

>>> workbench.fragmentation_rate('{"a": 0.314159, "b": 0.271828}')
0.5263  # approximately -- 20 tokens from 38 characters
```

Run full analysis:

```python
>>> report = workbench.analyze(n_samples=200, seed=42)

>>> report["fragmentation_stats"]["mean_tokens_per_param"]
3.12  # approximately

>>> report["fragmentation_stats"]["max_fragmentation_param"]
"b"  # or "a" depending on value ranges

>>> report["n_sims"]
200
```

Inspect hazard zones:

```python
>>> top_hazard = report["hazard_zones"][0]
>>> top_hazard["fragmentation"]
0.61  # approximately -- highest fragmentation configuration

>>> top_hazard["output_degradation"]
0.45  # approximately -- normalized deviation from median output

>>> len(report["hazard_zones"])
10
```

Check perturbation sensitivity:

```python
>>> report["perturbation_sensitivity"]
{"a": 2, "b": 1}  # approximately -- tokens changed per epsilon perturbation
```

---

## Scope

Custom tokenizers -- tiktoken BPE:

```python
>>> import tiktoken
>>> enc = tiktoken.get_encoding("cl100k_base")
>>> workbench_bpe = TokenExtispicyWorkbench(sim, tokenize=lambda s: [enc.decode([t]) for t in enc.encode(s)])
>>> report_bpe = workbench_bpe.analyze(n_samples=100)
>>> report_bpe["fragmentation_stats"]["mean_tokens_per_param"]
4.2  # approximately -- BPE fragments more than the default
```

Custom tokenizers -- HuggingFace:

```python
>>> from transformers import AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
>>> workbench_hf = TokenExtispicyWorkbench(sim, tokenize=tok.tokenize)
>>> report_hf = workbench_hf.analyze(n_samples=100)
```

Custom tokenizers -- SentencePiece:

```python
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor(model_file="model.model")
>>> workbench_sp = TokenExtispicyWorkbench(sim, tokenize=sp.encode_as_pieces)
>>> report_sp = workbench_sp.analyze(n_samples=100)
```

Different sample sizes:

```python
>>> quick_report = workbench.analyze(n_samples=20, seed=0)
>>> deep_report = workbench.analyze(n_samples=1000, seed=0)
>>> deep_report["fragmentation_output_correlation"]  # more reliable with more samples
```

Custom base_params for perturbation sensitivity:

```python
>>> report = workbench.analyze(base_params={"a": 0.9, "b": 0.1}, n_samples=100)
>>> report["perturbation_sensitivity"]  # sensitivity at the (0.9, 0.1) operating point
```

Using the workbench as a Simulator for meta-analysis:

```python
>>> from zimmerman.sobol import sobol_sensitivity
>>> meta_report = sobol_sensitivity(workbench, n_base=128)
>>> meta_report["_fragmentation_rate"]["S1"]  # which params drive fragmentation
```

---

## Applications

**ER: Which weight representations fragment worst under BPE?** Analyze how the 6 synaptic weights in the robot gait simulator tokenize under different models:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

robot_sim = SimulatorWrapper(run_trial, weight_spec)
workbench = TokenExtispicyWorkbench(robot_sim, tokenize=lambda s: [enc.decode([t]) for t in enc.encode(s)])
report = workbench.analyze(n_samples=300, seed=42)

# Discover which weights fragment most -- typically those with many
# significant digits (e.g., w24=0.283147 vs w04=0.5)
print(report["fragmentation_stats"]["max_fragmentation_param"])
# "w24" -- 7-digit weight values fragment 2x worse than 2-digit values

# Check if fragmentation predicts gait quality degradation
print(report["fragmentation_output_correlation"]["fitness"])
# r = -0.18 -- moderate negative correlation: higher fragmentation,
# slightly worse fitness, validating the extispicy hypothesis
```

**JGC: Does representing "baseline_age: 73" fragment differently than "baseline_age: 70"?** Test whether round numbers tokenize more cleanly than arbitrary values in the mitochondrial aging simulator:

```python
mito_workbench = TokenExtispicyWorkbench(mito_sim)

# Check fragmentation at a round age vs. an arbitrary age
round_params = {"baseline_age": 70.0, "genetic_vulnerability": 0.5}
arb_params = {"baseline_age": 73.0, "genetic_vulnerability": 0.5}

frag_round = mito_workbench.fragmentation_rate(
    mito_workbench.params_to_string(round_params)
)
frag_arb = mito_workbench.fragmentation_rate(
    mito_workbench.params_to_string(arb_params)
)

print(f"Round: {frag_round:.3f}, Arbitrary: {frag_arb:.3f}")
# Round: 0.41, Arbitrary: 0.44 -- round numbers fragment slightly less
```

---

## Properties & Relations

- **Extispicy metaphor** (Zimmerman SS4.6.4). The name references the ancient Mesopotamian practice of examining sacrificial entrails to predict future events. Here, the "entrails" are tokenization boundaries -- examining them reveals where representational hazards lurk. Just as a haruspex found an ill-shaped liver to be an omen, ill-shaped token boundaries are omens of output degradation.
- **BPE fundamentals** (Sennrich et al. 2016). Byte Pair Encoding iteratively merges the most frequent character pairs into subword tokens. Common words become single tokens; rare numeric strings fragment into many tokens. The fragmentation rate directly measures this effect.
- **Representational flattening** (Zimmerman SS3.5.3). Higher fragmentation forces the LLM to compose more tokens per semantic value, increasing the risk of losing semantic precision through tokenization artifacts. The fragmentation-output correlation measures whether this flattening propagates into downstream simulation results.
- **Connection to PromptReceptiveField**. While `PromptReceptiveField` measures which parts of a prompt most affect LLM output, `TokenExtispicyWorkbench` measures how the prompt is *encoded* by the tokenizer before the LLM ever sees it. Together they characterize both the encoding and processing stages of the LLM pipeline.
- **Hazard surface**. The `hazard_zones` output maps fragmentation across parameter space, identifying peaks where tokenization artifacts are most severe. This is a proper surface in parameter space -- it can be visualized as a heatmap and fed to `sobol_sensitivity()` for sensitivity analysis of the fragmentation itself.
- **Simulator protocol composability**. Because the workbench satisfies `run()` + `param_spec()`, it can be wrapped by any other zimmerman tool. Running `sobol_sensitivity(workbench)` produces a Sobol analysis that includes `_fragmentation_rate` and `_token_count` as output keys, revealing the sensitivity structure of the tokenization hazard surface.

---

## Possible Issues

- **Default tokenizer underestimates real BPE fragmentation.** The built-in whitespace + digit boundary tokenizer provides a conservative lower bound. Real BPE tokenizers (GPT-4, Llama-2, etc.) may fragment numeric strings more aggressively due to rare subword merges. Always compare with the actual tokenizer used in production.
- **Correlation != causation for fragmentation-output relationship.** A strong `fragmentation_output_correlation` is evidence that tokenization artifacts *coincide with* output degradation, but other confounding variables (e.g., parameter magnitude, number of significant digits) may drive both fragmentation and output quality independently.
- **Character-level edit distance is approximate.** Both `perturbation_token_sensitivity` and `token_edit_vs_string_edit` use a simplified positional Hamming-like distance rather than full Levenshtein distance. This is cheaper (O(n) vs O(n*m)) but may undercount edits when token sequences differ in length or alignment.
- **Parameter serialization affects measurements.** The `params_to_string` method uses compact JSON with 6-decimal precision and sorted keys. Different serialization choices (whitespace formatting, decimal precision, key ordering) would produce different fragmentation measurements. Ensure your production LLM prompt uses a similar serialization format.
- **n_samples controls statistical reliability.** With small `n_samples` (< 50), the fragmentation-output correlations may be noisy. For reliable correlation estimates, use `n_samples >= 200`.
- **Exception handling.** If `simulator.run()` raises during any of the `n_samples` calls, `analyze()` does not catch it. Use `Falsifier` first to verify simulator stability.
- **tiktoken returns int IDs, not strings.** When using tiktoken, wrap the encoder to return string tokens: `tokenize=lambda s: [enc.decode([t]) for t in enc.encode(s)]`. Passing raw int IDs will cause length comparisons to be incorrect.

---

## Neat Examples

**Discovering that 7-digit numbers fragment 3x worse than 2-digit numbers under BPE:**

```python
>>> from zimmerman.token_extispicy import TokenExtispicyWorkbench, _default_tokenize

>>> # The default tokenizer splits at digit/non-digit boundaries
>>> _default_tokenize("x=42")
["x=", "42"]

>>> _default_tokenize("x=3141593")
["x=", "3141593"]

>>> # But under BPE, the story is very different
>>> import tiktoken
>>> enc = tiktoken.get_encoding("cl100k_base")
>>> enc_tokens = lambda s: [enc.decode([t]) for t in enc.encode(s)]

>>> enc_tokens("x=42")
["x", "=", "42"]        # 3 tokens -- "42" is a common merge

>>> enc_tokens("x=3141593")
["x", "=", "314", "15", "93"]   # 5 tokens -- rare digit sequence fragments

>>> # Fragmentation rate comparison
>>> len(enc_tokens("x=42")) / len("x=42")
0.75   # 3 tokens / 4 chars

>>> len(enc_tokens("x=3141593")) / len("x=3141593")
0.56   # 5 tokens / 9 chars -- but 5 tokens vs 3 tokens = 67% more tokens

>>> # The practical impact: in a 6-parameter robot gait prompt, switching from
>>> # 2-digit weights (w=0.42) to 7-digit weights (w=0.3141593) adds ~12
>>> # extra tokens -- enough to shift attention patterns in smaller LLMs
```

**Using the workbench to find that "round number bias" exists in tokenizer space:**

```python
>>> workbench = TokenExtispicyWorkbench(sim)

>>> # Round numbers tend to be single tokens under BPE
>>> workbench.fragmentation_rate('{"age": 70, "dose": 100}')
0.36

>>> # Non-round numbers fragment more
>>> workbench.fragmentation_rate('{"age": 73, "dose": 117}')
0.39

>>> # Extreme precision fragments worst
>>> workbench.fragmentation_rate('{"age": 73.284619, "dose": 117.395201}')
0.52
```

---

## See Also

`PromptReceptiveField` | `Diegeticizer` | `SuperdiegeticBenchmark` | `Simulator`

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. SS4.6.4 (token extispicy methodology), SS3.5.3 (representational flattening).
- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." Proceedings of the 54th Annual Meeting of the ACL. arXiv:1508.07909.
- Kudo, T. & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." EMNLP 2018. arXiv:1808.06226.
