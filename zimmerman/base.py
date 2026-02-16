"""Base types and protocols for the Zimmerman Simulation Toolkit.

Defines the Simulator protocol that any simulator must satisfy to be
analyzed by the toolkit's sensitivity, auditing, and falsification tools.

The protocol requires two methods:
    run(params: dict) -> dict
        Execute the simulation with the given parameter values.
        Returns a result dictionary with named numeric outputs.

    param_spec() -> dict[str, tuple[float, float]]
        Return the parameter space specification as a mapping from
        parameter name to (lower_bound, upper_bound) tuple.

Any object that provides these two methods -- whether a class instance,
a wrapped function, or a full simulation engine -- can be used with
sobol_sensitivity(), POSIWIDAuditor, Falsifier, ContrastiveGenerator, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Simulator(Protocol):
    """Protocol for any simulator that can be analyzed by the toolkit.

    Any object providing run() and param_spec() methods with the correct
    signatures satisfies this protocol. Both the JGC mitochondrial simulator
    and ER's robot simulator satisfy this pattern.

    Example:
        class MySimulator:
            def run(self, params: dict) -> dict:
                x = params["x"]
                return {"y": x ** 2, "z": x + 1}

            def param_spec(self) -> dict[str, tuple[float, float]]:
                return {"x": (0.0, 10.0)}

        sim = MySimulator()
        assert isinstance(sim, Simulator)  # True at runtime
    """

    def run(self, params: dict) -> dict:
        """Execute the simulation with the given parameters.

        Args:
            params: Dictionary mapping parameter names to float values.
                All keys from param_spec() should be present.

        Returns:
            Dictionary mapping output names to numeric values (float, int,
            or numpy scalar). Non-numeric values are ignored by the
            analysis tools.
        """
        ...

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return the parameter space specification.

        Returns:
            Dictionary mapping parameter names to (low, high) bounds.
            Each bound is a float specifying the valid range for that
            parameter.
        """
        ...


class SimulatorWrapper:
    """Wraps a callable and a parameter spec into a Simulator-compatible object.

    This is a convenience class for cases where you have a standalone
    function rather than a class with run/param_spec methods.

    Example:
        def my_model(params):
            return {"output": params["a"] + params["b"]}

        spec = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
        sim = SimulatorWrapper(my_model, spec)
        result = sim.run({"a": 0.5, "b": 0.3})
        # result == {"output": 0.8}
    """

    def __init__(self, run_fn: callable, spec: dict[str, tuple[float, float]]):
        """Initialize the wrapper.

        Args:
            run_fn: A callable that takes a dict of parameter values and
                returns a dict of output values.
            spec: Parameter space specification mapping parameter names
                to (low, high) bound tuples.
        """
        self._run = run_fn
        self._spec = spec

    def run(self, params: dict) -> dict:
        """Execute the wrapped function with the given parameters."""
        return self._run(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return the stored parameter specification."""
        return dict(self._spec)
