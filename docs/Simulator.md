# Simulator

protocol that any simulator must satisfy for toolkit analysis

---

## Calling Patterns

```
class MySimulator:
    def run(self, params: dict) -> dict          execute one simulation
    def param_spec(self) -> dict[str, (lo, hi)]  declare parameter bounds
```

```
SimulatorWrapper(run_fn, spec)                   wrap a function as a Simulator
SimulatorWrapper(run_fn, spec).run(params)        execute the wrapped function
SimulatorWrapper(run_fn, spec).param_spec()       return the stored spec
```

---

## Details and Options

- `Simulator` is a `typing.Protocol` decorated with `@runtime_checkable`. Any object with `run()` and `param_spec()` methods of the correct signature satisfies it automatically (structural subtyping).
- `run(params)` takes a `dict[str, float]` mapping parameter names to values. All keys from `param_spec()` should be present.
- `run(params)` returns a `dict` mapping output names to numeric values (`float`, `int`, or numpy scalar). Non-numeric values are silently ignored by all analysis tools.
- `param_spec()` returns a `dict[str, tuple[float, float]]` mapping parameter names to `(lower_bound, upper_bound)` tuples.
- `isinstance(obj, Simulator)` works at runtime due to `@runtime_checkable`.
- `SimulatorWrapper` is a convenience class that wraps a bare function `(dict -> dict)` and a parameter spec dict into a `Simulator`-compatible object.
- `SimulatorWrapper.param_spec()` returns a copy of the stored spec (defensive copy via `dict()`).

---

## Basic Examples

Define a minimal simulator as a class:

```python
>>> class Quadratic:
...     def run(self, params):
...         x = params["x"]
...         return {"y": x ** 2, "z": x + 1}
...     def param_spec(self):
...         return {"x": (0.0, 10.0)}

>>> sim = Quadratic()
>>> sim.run({"x": 3.0})
{"y": 9.0, "z": 4.0}

>>> sim.param_spec()
{"x": (0.0, 10.0)}
```

Check protocol conformance at runtime:

```python
>>> from zimmerman.base import Simulator
>>> isinstance(sim, Simulator)
True
```

Wrap a standalone function using `SimulatorWrapper`:

```python
>>> from zimmerman.base import SimulatorWrapper

>>> def my_model(params):
...     return {"output": params["a"] + params["b"]}

>>> spec = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
>>> sim = SimulatorWrapper(my_model, spec)
>>> sim.run({"a": 0.5, "b": 0.3})
{"output": 0.8}
```

---

## Scope

The protocol accepts any number of parameters with any names:

```python
>>> class HighDim:
...     def run(self, params):
...         return {"sum": sum(params.values())}
...     def param_spec(self):
...         return {f"x{i}": (0.0, 1.0) for i in range(100)}

>>> isinstance(HighDim(), Simulator)
True
```

Output dicts may contain non-numeric values; they are simply ignored:

```python
>>> def mixed_output(params):
...     return {"fitness": 0.5, "label": "good", "array": [1, 2, 3]}

>>> sim = SimulatorWrapper(mixed_output, {"x": (0.0, 1.0)})
>>> # sobol_sensitivity will only analyze the "fitness" key
```

---

## Applications

**ER robot gait analysis.** The 3-link PyBullet robot simulator is wrapped as:

```python
spec = {name: (center - radius, center + radius) for name, center in zip(weight_names, weight_values)}
sim = SimulatorWrapper(lambda p: run_trial_inmemory(list(p.values())), spec)
```

**JGC mitochondrial aging simulator.** The 7-state ODE system is wrapped as:

```python
spec = {"metabolic_demand": (0.5, 2.0), "genetic_vulnerability": (0.0, 1.0), ...}
sim = SimulatorWrapper(run_mito_simulation, spec)
```

Both can then be passed to `sobol_sensitivity()`, `Falsifier`, `ContrastiveGenerator`, etc. without any adapter code.

---

## Properties & Relations

- The protocol follows the "duck typing with verification" pattern: structural subtyping via `Protocol`, runtime checkable via `@runtime_checkable`.
- `SimulatorWrapper` is the idiomatic way to adapt a legacy function. It performs no validation on `run_fn` — the function is called as-is.
- Every other class in the toolkit (`Falsifier`, `ContrastiveGenerator`, `POSIWIDAuditor`, `PDSMapper`, `PromptBuilder`) takes a `simulator` argument that must satisfy this protocol.
- The separation of `run()` and `param_spec()` ensures that analysis tools can inspect the parameter space without running any simulations.

---

## Possible Issues

- `SimulatorWrapper.param_spec()` returns a shallow copy. If bounds contain mutable objects (unusual), mutations may propagate.
- The protocol does not enforce that `run()` uses only keys from `param_spec()`. Passing extra keys or missing keys is the caller's responsibility.
- Simulators that raise exceptions during `run()` may cause issues with some tools. `Falsifier` handles exceptions gracefully; `sobol_sensitivity()` does not.

---

## See Also

`sobol_sensitivity` | `Falsifier` | `ContrastiveGenerator` | `POSIWIDAuditor` | `PDSMapper` | `PromptBuilder`
