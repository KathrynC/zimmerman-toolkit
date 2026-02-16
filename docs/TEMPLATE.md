# TEMPLATE.md — Wolfram-Style Documentation Template

This template adapts the Wolfram Research documentation house style for
Python functions and classes. Each public API entity gets its own page.

---

## Page Structure

```markdown
# FunctionName

one-line summary in plain English

---

## Calling Patterns

    FunctionName(arg1, arg2)               basic form
    FunctionName(arg1, arg2, option=val)   with options
    FunctionName(arg1, ..., seed=42)       reproducible variant

---

## Details and Options

- `arg1` is a ... that satisfies ...
- The default value of `option` is `val`.
- `FunctionName` returns a dict with keys ...
- When `arg2` is None, `FunctionName` uses ...
- The computation uses the Foo (Year) estimator.

---

## Basic Examples

Minimal working examples. Show input and output literally.

```python
>>> result = function_name(simple_input)
>>> result["key"]
0.42
```

---

## Scope

Extended examples showing the breadth of valid inputs,
edge cases that work correctly, different argument forms.

```python
>>> # Works with custom objects
>>> result = function_name(MyCustomClass())
```

---

## Applications

Real-world use cases drawn from the ER or JGC projects.
Show the function solving an actual research problem.

```python
>>> # Sobol analysis of a 6-weight robot gait
>>> sim = SimulatorWrapper(run_robot, weight_spec)
>>> result = sobol_sensitivity(sim, n_base=256)
```

---

## Properties & Relations

- Mathematical properties (e.g., "S1 values sum to 1.0 for additive models")
- Connections to other toolkit functions
- Theoretical grounding (cite papers)

---

## Possible Issues

- Known limitations, gotchas, performance considerations
- Parameter ranges that cause problems
- Common mistakes

---

## Neat Examples

Surprising, elegant, or particularly illuminating uses.

---

## See Also

`RelatedFunction1` | `RelatedFunction2` | `RelatedFunction3`
```

---

## Conventions

1. **One page per major public entity** (function or class).
   Internal helpers (prefixed with `_`) are not documented separately.

2. **Calling Patterns** use Python signature syntax with brief
   right-aligned descriptions. Show all valid calling forms.

3. **Details and Options** uses bullet points. Every default value,
   every key in a returned dict, every behavioral nuance.

4. **Basic Examples** must be copy-pasteable. Use `>>>` prompts.
   Show actual output values, not placeholders.

5. **Applications** reference the actual research projects
   (ER robot gaits, JGC mitochondrial simulator, stock simulator).

6. **Properties & Relations** includes citations where relevant
   (Saltelli 2002, Jansen 1999, Stafford Beer 1974, Zimmerman 2025).

7. **See Also** links to related pages using backtick function names.

8. Pages are pure Markdown. No build system required.
