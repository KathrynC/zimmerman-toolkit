# Zimmerman Simulation Toolkit — Reference Guide

black-box simulator interrogation through six complementary lenses

---

## From Thesis to Toolkit

This toolkit operationalizes key findings from Julia Witte Zimmerman's 2025 PhD dissertation at the University of Vermont: *"Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)."* (Specializing in Complex Systems and Data Science; advised by Peter Sheridan Dodds and Christopher M. Danforth.)

Zimmerman's dissertation investigates how LLMs construct meaning — and where that meaning-making breaks down. Its concepts generate three toolkit components and inform the design of all six:

**Chapter 2 (§2.2.3): Diegetic vs. Supradiegetic Linguistic Information.** Zimmerman distinguishes two kinds of information carried by language: *diegetic* information — the inside of the word, its meaning, its semantic and propositional content ("imagine a word minus any letters or sounds") — and *supradiegetic* information — the exterior of the word, the arbitrary physical form through which meaning is packaged (letter shapes, syllable sounds). LLMs have "extremely curtailed access — essentially no access at all" to supradiegetic information, because by the time text reaches the model, it has already been converted to token numbers. For an LLM, the diegetic realm is essentially its entire world.

**Chapter 3 (§3.5.3): Flattening.** Tokenization treats all content identically at the linear-algebraic level — a phenomenon Zimmerman calls *flattening*. Numbers like "9,342" get tokenized destructively (e.g., "9" + "," + "342"), obscuring their mathematical structure. "Imagine trying to do math when you cannot see that those tokenized sequences involve the same numbers." This directly explains why narrative (diegetic) prompts outperform bare numeric requests when asking an LLM to generate simulation parameters: numbers are precisely the kind of supradiegetic-adjacent content that tokenization handles worst.

These findings generate **`PromptBuilder`**, which implements three prompt styles: *numeric* (the supradiegetic baseline — raw parameter values the LLM processes poorly), *diegetic* (narrative-embedded parameters that exploit the LLM's semantic strengths), and *contrastive* (opposing perspectives that force differentiation).

**Chapter 4 (§4.6.4): The PDS Framework.** PDS — Power, Danger, Structure — are the three most significant axes discovered through ousiometric analysis of large sets of word meanings (Dodds et al. 2023). Power runs from weak to powerful (the nothing–something continuum, *ex nihilo*). Danger runs from safe to dangerous (angels to demons, givers to takers). Structure runs from structured to unstructured (traditionalists to adventurers, stasis to mutation). The alignment between these lexical dimensions and Zimmerman's character-space dimensions is *emergent*: none of the words "powerful," "weak," "dangerous," "safe," "structured," or "unstructured" explicitly appear in the 464 bipolar adjective pairs used to construct character space. Power and Danger are roughly equal in significance; Structure's contribution is smaller.

This generates **`PDSMapper`**, which maps abstract PDS dimension values (typically in [-1, +1]) to concrete simulator parameters, audits how well those dimensions predict simulation outcomes, and ranks dimensions by influence. The toolkit's application of PDS to simulator parameter spaces is an *extension* of the framework — using dimensions discovered in lexical semantics as a navigation tool for high-dimensional parameter spaces, exploiting the fact that domain experts naturally reason in terms like "high power, low danger."

**Chapter 3 (§3.5.2): POSIWID.** Stafford Beer's (1974) POSIWID principle — "The Purpose Of a System Is What It Does" — appears in the thesis as a framework for analyzing objective function misalignment: the tokenizer's purpose (compress orthographically) is not aligned with the model's purpose (learn semantics). The toolkit applies POSIWID to simulation: when an LLM generates parameters intending a fast, stable walker, and the simulator produces a robot that falls on its face, that gap between intended and actual is quantifiable.

**Chapter 4 (§4.7.6): TALOT vs. OTTITT — Meaning-from-Contrast.** TALOT ("Things Are Like Other Things") and OTTITT ("Only The Thing Is The Thing") define the identity-difference spectrum through which meaning is constructed. Categorization is "the process by which detectably different (i.e., non-identical) stimuli come to be represented as identical, in some respect" (Lupyan). Meaning emerges from the tension between things being alike (TALOT) and unique (OTTITT). Presenting two opposing perspectives forces an LLM to navigate this spectrum explicitly.

These generate **`POSIWIDAuditor`** (alignment scoring between intended and actual outcomes) and the *contrastive* prompt style in `PromptBuilder` (two agents — cautious vs. aggressive — bracketing the solution space).

**The classical analysis tools** — `sobol_sensitivity`, `Falsifier`, and `ContrastiveGenerator` — are not thesis-derived. They implement standard simulation analysis techniques (Saltelli 2002, Jansen 1999) generalized to work with any simulator satisfying the same protocol. But they serve the thesis-derived tools: you need to know which parameters matter (Sobol) before you can build meaningful PDS mappings, you need to know where the simulator breaks (Falsifier) before you can trust POSIWID alignment scores, and you need to know where behavioral boundaries lie (ContrastiveGenerator) before you can design prompts that navigate them.

The toolkit was created by Kathryn Cramer, using Claude as utilities for working on research with Zimmerman and as an enhancement to the aging simulator she created for John G. Cramer. It was extracted and generalized from domain-specific code in the [how-to-live-much-longer](../how-to-live-much-longer) mitochondrial aging simulator project (based on John G. Cramer's forthcoming book *How to Live Much Longer*), where each module first proved its value on a concrete research problem before being lifted into a simulator-agnostic library.

---

## Overview

The Zimmerman Toolkit analyzes any simulator satisfying a two-method protocol (`run` + `param_spec`). It provides six tools that ask fundamentally different questions about the simulator's behavior:

| Tool | Question | Method |
|------|----------|--------|
| `sobol_sensitivity` | Which parameters matter most? | Saltelli sampling + Saltelli/Jansen estimators |
| `Falsifier` | Where does the simulator break? | Random + boundary + adversarial testing |
| `ContrastiveGenerator` | What's the smallest change that flips the outcome? | Bisection along random directions |
| `POSIWIDAuditor` | Does it do what we intended? | Alignment scoring (direction + magnitude) |
| `PDSMapper` | Can we control it through abstract dimensions? | Linear dimension-to-parameter mapping |
| `PromptBuilder` | How do we ask an LLM to design parameters? | Three prompt styles (numeric, diegetic, contrastive) |

---

## The Simulator Protocol

Every tool takes a `simulator` argument satisfying:

```python
class Simulator(Protocol):
    def run(self, params: dict) -> dict: ...
    def param_spec(self) -> dict[str, tuple[float, float]]: ...
```

Wrap any function with `SimulatorWrapper(fn, spec)`. See `Simulator` for details.

---

## Function Pages

### Analysis

- **[`sobol_sensitivity`](sobol_sensitivity.md)** — Global sensitivity via Saltelli sampling. Decomposes output variance into per-parameter main effects (S1) and interaction effects (ST).
- **[`Falsifier`](Falsifier.md)** — Three-phase falsification: random sampling, boundary testing, adversarial probing. Finds parameter combinations that violate assertions.
- **[`ContrastiveGenerator`](ContrastiveGenerator.md)** — Minimal-perturbation search via bisection. Finds the smallest parameter change that flips a categorical outcome.

### Alignment & Design

- **[`POSIWIDAuditor`](POSIWIDAuditor.md)** — "The Purpose Of a System Is What It Does." Quantifies the gap between intended and actual outcomes.
- **[`PDSMapper`](PDSMapper.md)** — Maps abstract semantic dimensions (Power, Danger, Structure) to concrete parameter values. Enables navigation of parameter space through meaningful axes.
- **[`PromptBuilder`](PromptBuilder.md)** — Generates prompts in three styles (numeric, diegetic, contrastive) for LLM-mediated parameter generation.

### Infrastructure

- **[`Simulator`](Simulator.md)** — The protocol and `SimulatorWrapper` convenience class.

---

## Typical Workflow

```
1. Define simulator         Simulator / SimulatorWrapper
         │
2. Verify stability         Falsifier.falsify()
         │
3. Map sensitivity          sobol_sensitivity()
         │                  ContrastiveGenerator.find_contrastive()
         │
4. Design parameters        PromptBuilder.build_diegetic()  →  LLM
         │                  PDSMapper.map_dimensions_to_params()
         │
5. Audit alignment          POSIWIDAuditor.audit()
         │
6. Iterate                  Update prompts, mappings, or assertions
```

---

## Cross-Cutting Concepts

### TIQM Pipeline

The toolkit implements Cramer's Transactional Interpretation of Quantum Mechanics (TIQM) as a simulation design metaphor:

- **Offer wave**: `PromptBuilder` → LLM → parameter vector
- **Confirmation wave**: `POSIWIDAuditor` → alignment score
- **Transaction**: High alignment = the offer was confirmed

Using different LLM models for offer vs. confirmation prevents self-confirmation bias.

### Complementary Sensitivity Measures

- `sobol_sensitivity` gives **global** sensitivity across the full parameter range.
- `ContrastiveGenerator.sensitivity_from_contrastives()` gives **local** sensitivity at specific behavioral boundaries.
- `PDSMapper.sensitivity_per_dimension()` gives **abstract-dimension** sensitivity through the PDS mapping.

Three views of the same underlying question: "what drives this simulator?"

### Numpy-Only

The entire toolkit is pure numpy — no scipy, no SALib, no sklearn. This is deliberate: it eliminates dependency management and ensures the mathematical implementations are transparent and auditable.

---

## Installation

```bash
# The toolkit is a pure Python package with numpy as its only dependency
cd zimmerman-toolkit
pip install -e .
# or just add to PYTHONPATH:
export PYTHONPATH="/path/to/zimmerman-toolkit:$PYTHONPATH"
```

---

## References

- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
- Beer, Stafford (1974). "Designing Freedom." CBC Massey Lectures.
- Dodds, P.S., Alshaabi, T., Fudolig, M.I., Zimmerman, J.W., Lovato, J., Beaulieu, S., Minot, J.R., Arnold, M.V., Reagan, A.J., and Danforth, C.M. (2023). "Ousiometrics and telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework." *arXiv*.
- Zimmerman, J.W., Hudon, D., Cramer, K., St-Onge, J., Fudolig, M., Trujillo, M.Z., Danforth, C.M., and Dodds, P.S. (2024). "A blind spot for large language models: Supradiegetic linguistic information." *Plutonics*, 17, 107-156.
- Cramer, John G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag.
- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Graduate College Dissertations and Theses, 2082.
