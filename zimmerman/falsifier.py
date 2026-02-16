"""Systematic falsification: seeks parameter combinations that break assumptions.

Inspired by the falsifier agent from the how-to-live-much-longer project,
which found 4 critical bugs in the ODE equations (2026-02-15). This module
generalizes that approach to work with any Simulator.

Three testing strategies:
    1. Random sampling: Uniformly random parameter combinations to test
       the simulation across a broad range of inputs.
    2. Boundary testing: Parameter combinations at corners, edges, and
       faces of the parameter hypercube. These are where numerical issues
       (overflow, division by zero, discontinuities) most often occur.
    3. Adversarial testing: Parameter combinations near known violations,
       probing whether issues are isolated or systemic.

Default assertions (checked for every run):
    - No NaN values in any numeric output
    - No Inf values in any numeric output
    - All numeric values are finite

Users can add custom assertions (e.g., "output X must be positive",
"output Y must be less than 100", etc.).
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _default_assertions() -> list[callable]:
    """Return default assertion functions: no NaN, no Inf, all finite."""

    def no_nan(result: dict) -> bool:
        for key, val in result.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isnan(float(val)):
                    return False
        return True

    def no_inf(result: dict) -> bool:
        for key, val in result.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isinf(float(val)):
                    return False
        return True

    def all_finite(result: dict) -> bool:
        for key, val in result.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if not np.isfinite(float(val)):
                    return False
        return True

    return [no_nan, no_inf, all_finite]


class Falsifier:
    """Systematically tests a simulator for assumption violations.

    Applies three testing strategies (random, boundary, adversarial) to
    find parameter combinations that violate user-defined or default
    assertions about the simulation output.

    Args:
        simulator: Any Simulator-compatible object.
        assertions: List of callable(result_dict) -> bool. Each should
            return True for valid results. If None, uses default
            assertions (no NaN, no Inf, all finite).

    Example:
        falsifier = Falsifier(my_sim, assertions=[
            lambda r: r["energy"] >= 0,  # energy must be non-negative
            lambda r: r["health"] <= 1.0,  # health must be <= 1.0
        ])
        report = falsifier.falsify(n_random=200)
        print(f"Found {report['summary']['violations_found']} violations")
    """

    def __init__(self, simulator, assertions: list[callable] | None = None):
        self.simulator = simulator
        self._spec = simulator.param_spec()
        if assertions is None:
            self.assertions = _default_assertions()
        else:
            self.assertions = list(assertions)

    def _check_assertions(self, result: dict) -> list[int]:
        """Check all assertions, return indices of failed ones."""
        failed = []
        for i, assertion in enumerate(self.assertions):
            try:
                if not assertion(result):
                    failed.append(i)
            except Exception:
                failed.append(i)
        return failed

    def _random_params(self, rng: np.random.Generator) -> dict:
        """Generate a random parameter combination within bounds."""
        params = {}
        for name, (lo, hi) in self._spec.items():
            params[name] = float(rng.uniform(lo, hi))
        return params

    def boundary_params(self, param_spec: dict | None = None) -> list[dict]:
        """Generate parameter combinations at hypercube boundaries.

        Generates combinations at corners (all min/max), edges (one param
        varies, rest at extremes), and faces (combinations of min/mid/max).

        For high-dimensional spaces, limits the number of combinations to
        avoid combinatorial explosion.

        Args:
            param_spec: Parameter spec dict. If None, uses simulator's spec.

        Returns:
            List of parameter dicts at boundary locations.
        """
        if param_spec is None:
            param_spec = self._spec

        param_names = list(param_spec.keys())
        d = len(param_names)
        bounds = {name: param_spec[name] for name in param_names}

        boundary_combos = []

        # Strategy 1: All-min and all-max corners
        all_min = {name: bounds[name][0] for name in param_names}
        all_max = {name: bounds[name][1] for name in param_names}
        boundary_combos.extend([all_min, all_max])

        # Strategy 2: One param at min, rest at max (and vice versa)
        for i in range(d):
            # One at min, rest at max
            combo = dict(all_max)
            combo[param_names[i]] = bounds[param_names[i]][0]
            boundary_combos.append(combo)

            # One at max, rest at min
            combo = dict(all_min)
            combo[param_names[i]] = bounds[param_names[i]][1]
            boundary_combos.append(combo)

        # Strategy 3: Each param at min, mid, max individually (rest at mid)
        midpoints = {
            name: (bounds[name][0] + bounds[name][1]) / 2.0
            for name in param_names
        }
        for i in range(d):
            for extreme_val in [bounds[param_names[i]][0], bounds[param_names[i]][1]]:
                combo = dict(midpoints)
                combo[param_names[i]] = extreme_val
                boundary_combos.append(combo)

        # Strategy 4: Near-boundary (epsilon inside the bounds)
        epsilon = 1e-6
        for i in range(d):
            lo, hi = bounds[param_names[i]]
            rng_width = hi - lo
            if rng_width > 0:
                near_lo = lo + epsilon * rng_width
                near_hi = hi - epsilon * rng_width
                combo_lo = dict(midpoints)
                combo_lo[param_names[i]] = near_lo
                combo_hi = dict(midpoints)
                combo_hi[param_names[i]] = near_hi
                boundary_combos.extend([combo_lo, combo_hi])

        # Strategy 5: For low dimensions, add some corner combinations
        if d <= 6:
            for bits in product([0, 1], repeat=d):
                combo = {}
                for j, name in enumerate(param_names):
                    combo[name] = bounds[name][bits[j]]
                boundary_combos.append(combo)

        # Deduplicate (approximate: convert to tuples)
        seen = set()
        unique = []
        for combo in boundary_combos:
            key = tuple(sorted(combo.items()))
            if key not in seen:
                seen.add(key)
                unique.append(combo)

        return unique

    def adversarial_params(
        self,
        param_spec: dict | None = None,
        violations: list[dict] | None = None,
        n_per_violation: int = 5,
        perturbation_scale: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Generate parameter combinations near known violations.

        For each known violation, generates perturbed variants to probe
        whether the issue is a point failure or a region.

        Args:
            param_spec: Parameter spec dict. If None, uses simulator's spec.
            violations: List of violation dicts (each with "params" key).
                If None or empty, generates random adversarial samples.
            n_per_violation: Number of perturbations per known violation.
            perturbation_scale: Scale of perturbation as fraction of range.
            rng: Random number generator.

        Returns:
            List of parameter dicts near known failure points.
        """
        if param_spec is None:
            param_spec = self._spec
        if rng is None:
            rng = np.random.default_rng(999)

        param_names = list(param_spec.keys())
        adversarial = []

        if violations:
            for violation in violations:
                base_params = violation.get("params", {})
                for _ in range(n_per_violation):
                    perturbed = dict(base_params)
                    for name in param_names:
                        if name in perturbed:
                            lo, hi = param_spec[name]
                            delta = rng.normal(0, perturbation_scale * (hi - lo))
                            perturbed[name] = float(np.clip(
                                perturbed[name] + delta, lo, hi
                            ))
                    adversarial.append(perturbed)
        else:
            # No known violations: generate adversarial samples by
            # concentrating near boundaries with noise
            for _ in range(n_per_violation * 3):
                params = {}
                for name, (lo, hi) in param_spec.items():
                    # Bias toward extremes
                    if rng.random() < 0.5:
                        base = lo if rng.random() < 0.5 else hi
                    else:
                        base = rng.uniform(lo, hi)
                    noise = rng.normal(0, perturbation_scale * (hi - lo))
                    params[name] = float(np.clip(base + noise, lo, hi))
                adversarial.append(params)

        return adversarial

    def falsify(
        self,
        n_random: int = 100,
        n_boundary: int = 50,
        n_adversarial: int = 50,
        seed: int = 42,
    ) -> dict:
        """Run systematic falsification testing.

        Applies three strategies in sequence:
        1. Random: n_random uniformly random parameter combinations.
        2. Boundary: up to n_boundary combinations at parameter bounds.
        3. Adversarial: n_adversarial combinations near found violations.

        Args:
            n_random: Number of random test cases.
            n_boundary: Maximum number of boundary test cases.
            n_adversarial: Number of adversarial test cases.
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                "violations": list of {
                    "params": dict, "result": dict,
                    "failed_assertions": list[int],
                    "strategy": str ("random"|"boundary"|"adversarial"),
                    "error": str or None (if simulator raised an exception),
                },
                "summary": {
                    "total_tests": int,
                    "violations_found": int,
                    "violation_rate": float,
                    "random_violations": int,
                    "boundary_violations": int,
                    "adversarial_violations": int,
                    "exceptions": int,
                },
        """
        rng = np.random.default_rng(seed)
        violations = []
        total_tests = 0
        exceptions = 0
        random_violations = 0
        boundary_violations = 0
        adversarial_violations = 0

        # --- Phase 1: Random sampling ---
        for _ in range(n_random):
            params = self._random_params(rng)
            total_tests += 1
            try:
                result = self.simulator.run(params)
                failed = self._check_assertions(result)
                if failed:
                    violations.append({
                        "params": params,
                        "result": result,
                        "failed_assertions": failed,
                        "strategy": "random",
                        "error": None,
                    })
                    random_violations += 1
            except Exception as e:
                violations.append({
                    "params": params,
                    "result": {},
                    "failed_assertions": list(range(len(self.assertions))),
                    "strategy": "random",
                    "error": str(e),
                })
                random_violations += 1
                exceptions += 1

        # --- Phase 2: Boundary testing ---
        boundary_combos = self.boundary_params()
        # Limit to n_boundary
        if len(boundary_combos) > n_boundary:
            indices = rng.choice(len(boundary_combos), size=n_boundary, replace=False)
            boundary_combos = [boundary_combos[i] for i in indices]

        for params in boundary_combos:
            total_tests += 1
            try:
                result = self.simulator.run(params)
                failed = self._check_assertions(result)
                if failed:
                    violations.append({
                        "params": params,
                        "result": result,
                        "failed_assertions": failed,
                        "strategy": "boundary",
                        "error": None,
                    })
                    boundary_violations += 1
            except Exception as e:
                violations.append({
                    "params": params,
                    "result": {},
                    "failed_assertions": list(range(len(self.assertions))),
                    "strategy": "boundary",
                    "error": str(e),
                })
                boundary_violations += 1
                exceptions += 1

        # --- Phase 3: Adversarial testing ---
        adversarial_combos = self.adversarial_params(
            violations=violations if violations else None,
            n_per_violation=max(1, n_adversarial // max(len(violations), 1)),
            rng=rng,
        )
        # Limit to n_adversarial
        if len(adversarial_combos) > n_adversarial:
            adversarial_combos = adversarial_combos[:n_adversarial]

        for params in adversarial_combos:
            total_tests += 1
            try:
                result = self.simulator.run(params)
                failed = self._check_assertions(result)
                if failed:
                    violations.append({
                        "params": params,
                        "result": result,
                        "failed_assertions": failed,
                        "strategy": "adversarial",
                        "error": None,
                    })
                    adversarial_violations += 1
            except Exception as e:
                violations.append({
                    "params": params,
                    "result": {},
                    "failed_assertions": list(range(len(self.assertions))),
                    "strategy": "adversarial",
                    "error": str(e),
                })
                adversarial_violations += 1
                exceptions += 1

        violation_rate = len(violations) / max(total_tests, 1)

        return {
            "violations": violations,
            "summary": {
                "total_tests": total_tests,
                "violations_found": len(violations),
                "violation_rate": float(violation_rate),
                "random_violations": random_violations,
                "boundary_violations": boundary_violations,
                "adversarial_violations": adversarial_violations,
                "exceptions": exceptions,
            },
        }
