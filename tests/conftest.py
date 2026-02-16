"""Shared test fixtures for the Zimmerman toolkit test suite.

Provides four toy simulators for testing the toolkit's analysis tools:

    LinearSimulator: y = sum(a_i * x_i)
        Purely additive model. For Sobol testing: S1 values should match
        analytical expectations, sum(S1) ~ 1.0, interaction ~ 0.

    QuadraticSimulator: y = sum(x_i^2)
        Has a known optimum at the origin. Useful for testing
        optimization and sensitivity around non-linear models.

    BrokenSimulator: Returns NaN when any parameter > 0.9
        Known failure mode for falsifier testing. Behaves normally
        for parameters in [0, 0.9] and breaks above.

    StepSimulator: Returns +1 if sum(params) > threshold, else -1
        Sharp decision boundary. Ideal for contrastive testing:
        small changes near the threshold flip the outcome.
"""

import numpy as np
import pytest


class LinearSimulator:
    """Linear additive model: y = sum(a_i * x_i).

    Each parameter x_i has range [0, 1]. The coefficients a_i determine
    the sensitivity of the output to each parameter.

    Args:
        d: Number of parameters.
        coefficients: Optional array of coefficients. If None, uses
            a_i = i + 1 (so a_0=1, a_1=2, ..., a_{d-1}=d).
    """

    def __init__(self, d: int = 4, coefficients: np.ndarray | None = None):
        self.d = d
        if coefficients is not None:
            self.coefficients = np.asarray(coefficients, dtype=float)
        else:
            self.coefficients = np.arange(1, d + 1, dtype=float)

    def run(self, params: dict) -> dict:
        total = 0.0
        for i in range(self.d):
            name = f"x{i}"
            total += self.coefficients[i] * params[name]
        return {"y": float(total)}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {f"x{i}": (0.0, 1.0) for i in range(self.d)}


class QuadraticSimulator:
    """Quadratic model: y = sum(x_i^2).

    All parameters have range [-1, 1]. Minimum at origin (y=0).
    Useful for testing optimization-oriented tools.

    Args:
        d: Number of parameters.
    """

    def __init__(self, d: int = 3):
        self.d = d

    def run(self, params: dict) -> dict:
        total = 0.0
        for i in range(self.d):
            name = f"x{i}"
            total += params[name] ** 2
        return {"y": float(total), "fitness": float(-total)}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {f"x{i}": (-1.0, 1.0) for i in range(self.d)}


class BrokenSimulator:
    """Simulator with known failure: returns NaN when any param > 0.9.

    For parameters all in [0, 0.9]: returns y = sum(x_i).
    When any parameter exceeds 0.9: returns y = NaN.

    This predictable failure mode lets us test that the falsifier
    reliably discovers the NaN region.

    Args:
        d: Number of parameters.
        threshold: Value above which NaN is returned. Default 0.9.
    """

    def __init__(self, d: int = 3, threshold: float = 0.9):
        self.d = d
        self.threshold = threshold

    def run(self, params: dict) -> dict:
        for i in range(self.d):
            name = f"x{i}"
            if params[name] > self.threshold:
                return {"y": float("nan"), "status": "broken"}
        total = sum(params[f"x{i}"] for i in range(self.d))
        return {"y": float(total), "status": "ok"}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {f"x{i}": (0.0, 1.0) for i in range(self.d)}


class StepSimulator:
    """Step function: returns +1 if sum(params) > threshold, else -1.

    Creates a sharp decision boundary in parameter space. Ideal for
    testing contrastive generation: near the boundary, small parameter
    changes flip the outcome.

    Args:
        d: Number of parameters.
        threshold: Sum threshold for the step. Default is d * 0.5
            (midpoint of the parameter space).
    """

    def __init__(self, d: int = 3, threshold: float | None = None):
        self.d = d
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = d * 0.5

    def run(self, params: dict) -> dict:
        total = sum(params[f"x{i}"] for i in range(self.d))
        outcome = 1 if total > self.threshold else -1
        return {
            "fitness": float(outcome),
            "sum": float(total),
            "threshold": float(self.threshold),
        }

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {f"x{i}": (0.0, 1.0) for i in range(self.d)}


# ---- Pytest fixtures ----

@pytest.fixture
def linear_sim():
    """4-parameter linear simulator with coefficients [1, 2, 3, 4]."""
    return LinearSimulator(d=4)


@pytest.fixture
def quadratic_sim():
    """3-parameter quadratic simulator."""
    return QuadraticSimulator(d=3)


@pytest.fixture
def broken_sim():
    """3-parameter broken simulator (NaN when any param > 0.9)."""
    return BrokenSimulator(d=3)


@pytest.fixture
def step_sim():
    """3-parameter step simulator (threshold = 1.5)."""
    return StepSimulator(d=3, threshold=1.5)
