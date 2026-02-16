"""Contrastive scenario generation: find minimal changes that flip outcomes.

Given a simulator and a base parameter set, finds the smallest perturbation
that flips the outcome from one category to another (e.g., from "survive"
to "collapse", or from "positive fitness" to "negative fitness").

This is useful for:
- Understanding decision boundaries in parameter space
- Identifying which parameters are most fragile/sensitive
- Generating adversarial examples for robustness testing
- Finding the "tipping points" in a simulation

The approach uses bisection along random perturbation directions:
1. Sample a random direction in parameter space
2. Perturb the base parameters along that direction
3. If the outcome flips, bisect to find the minimal perturbation
4. Track which parameters are most often involved in flips

This is a generalized version: works with any Simulator and any
user-defined outcome function.
"""

from __future__ import annotations

import numpy as np


class ContrastiveGenerator:
    """Finds minimal parameter perturbations that flip simulation outcomes.

    Args:
        simulator: Any Simulator-compatible object.
        outcome_fn: Callable that takes a result dict from simulator.run()
            and returns a categorical/binary outcome (typically +1/-1 or
            True/False). Default: sign of result.get("fitness", 0).

    Example:
        gen = ContrastiveGenerator(my_sim, outcome_fn=lambda r: r["alive"])
        pair = gen.find_contrastive({"speed": 0.5, "armor": 0.3})
        print(pair["delta"])  # shows which params changed
    """

    def __init__(self, simulator, outcome_fn: callable | None = None):
        self.simulator = simulator
        self._spec = simulator.param_spec()
        if outcome_fn is None:
            self.outcome_fn = self._default_outcome_fn
        else:
            self.outcome_fn = outcome_fn

    @staticmethod
    def _default_outcome_fn(result: dict) -> int:
        """Default outcome: sign of 'fitness' key, or 1 if missing."""
        fitness = result.get("fitness", 0.0)
        if isinstance(fitness, (int, float, np.integer, np.floating)):
            return 1 if float(fitness) >= 0 else -1
        return 1

    def _clip_params(self, params: dict) -> dict:
        """Clip parameter values to their bounds."""
        clipped = {}
        for name, val in params.items():
            if name in self._spec:
                lo, hi = self._spec[name]
                clipped[name] = float(np.clip(val, lo, hi))
            else:
                clipped[name] = val
        return clipped

    def _perturb_params(
        self,
        base_params: dict,
        direction: np.ndarray,
        magnitude: float,
        param_names: list[str],
        ranges: np.ndarray,
    ) -> dict:
        """Apply a perturbation to base parameters along a direction."""
        perturbed = dict(base_params)
        for i, name in enumerate(param_names):
            delta = direction[i] * magnitude * ranges[i]
            perturbed[name] = base_params[name] + delta
        return self._clip_params(perturbed)

    def find_contrastive(
        self,
        base_params: dict,
        n_attempts: int = 100,
        max_delta_frac: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Find the smallest perturbation that flips the outcome.

        Starting from base_params, samples random directions in parameter
        space and uses bisection to find the minimal perturbation along
        each direction that changes the outcome.

        Args:
            base_params: Starting parameter dict.
            n_attempts: Number of random directions to try.
            max_delta_frac: Maximum perturbation as a fraction of each
                parameter's range. Default 0.1 = 10%.
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                "found": bool -- whether a contrastive example was found,
                "original_params": dict,
                "contrastive_params": dict (or None if not found),
                "delta": dict of per-param deltas (or None),
                "outcome_original": outcome value,
                "outcome_flipped": flipped outcome (or None),
                "perturbation_magnitude": float (L2 norm of delta / ranges),
                "n_sims": int (total simulations run),
        """
        rng = np.random.default_rng(seed)
        param_names = list(self._spec.keys())
        d = len(param_names)
        ranges = np.array([self._spec[name][1] - self._spec[name][0] for name in param_names])

        # Get base outcome
        base_result = self.simulator.run(base_params)
        base_outcome = self.outcome_fn(base_result)
        n_sims = 1

        best_contrastive = None
        best_magnitude = float("inf")

        for attempt in range(n_attempts):
            # Random unit direction in parameter space
            direction = rng.standard_normal(d)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm

            # Check if this direction can flip the outcome at max magnitude
            max_params = self._perturb_params(
                base_params, direction, max_delta_frac, param_names, ranges
            )
            max_result = self.simulator.run(max_params)
            max_outcome = self.outcome_fn(max_result)
            n_sims += 1

            if max_outcome == base_outcome:
                # This direction doesn't flip at max perturbation; skip
                continue

            # Bisection to find minimal flip magnitude
            lo_mag = 0.0
            hi_mag = max_delta_frac
            n_bisect = 15  # ~15 steps gives ~1e-5 precision

            for _ in range(n_bisect):
                mid_mag = (lo_mag + hi_mag) / 2.0
                mid_params = self._perturb_params(
                    base_params, direction, mid_mag, param_names, ranges
                )
                mid_result = self.simulator.run(mid_params)
                mid_outcome = self.outcome_fn(mid_result)
                n_sims += 1

                if mid_outcome == base_outcome:
                    lo_mag = mid_mag
                else:
                    hi_mag = mid_mag

            # The contrastive point is at hi_mag (just past the flip)
            contrastive_params = self._perturb_params(
                base_params, direction, hi_mag, param_names, ranges
            )
            contrastive_result = self.simulator.run(contrastive_params)
            contrastive_outcome = self.outcome_fn(contrastive_result)
            n_sims += 1

            if contrastive_outcome != base_outcome and hi_mag < best_magnitude:
                best_magnitude = hi_mag
                delta = {}
                for i, name in enumerate(param_names):
                    delta[name] = contrastive_params[name] - base_params[name]
                best_contrastive = {
                    "contrastive_params": contrastive_params,
                    "delta": delta,
                    "outcome_flipped": contrastive_outcome,
                    "perturbation_magnitude": float(hi_mag),
                }

        if best_contrastive is None:
            return {
                "found": False,
                "original_params": dict(base_params),
                "contrastive_params": None,
                "delta": None,
                "outcome_original": base_outcome,
                "outcome_flipped": None,
                "perturbation_magnitude": None,
                "n_sims": n_sims,
            }

        return {
            "found": True,
            "original_params": dict(base_params),
            "contrastive_params": best_contrastive["contrastive_params"],
            "delta": best_contrastive["delta"],
            "outcome_original": base_outcome,
            "outcome_flipped": best_contrastive["outcome_flipped"],
            "perturbation_magnitude": best_contrastive["perturbation_magnitude"],
            "n_sims": n_sims,
        }

    def contrastive_pairs(
        self,
        params_list: list[dict],
        n_per_point: int = 5,
        seed: int = 42,
    ) -> list[dict]:
        """Find contrastive pairs for multiple starting points.

        Args:
            params_list: List of parameter dicts to find contrastive
                examples for.
            n_per_point: Number of contrastive attempts per starting point.
                Uses different random seeds for each.
            seed: Base random seed.

        Returns:
            List of contrastive pair dicts (one per starting point that
            had a successful flip). Each dict is the output of
            find_contrastive().
        """
        pairs = []
        for idx, base_params in enumerate(params_list):
            for attempt in range(n_per_point):
                pair_seed = seed + idx * n_per_point + attempt
                result = self.find_contrastive(
                    base_params,
                    n_attempts=20,
                    max_delta_frac=0.15,
                    seed=pair_seed,
                )
                if result["found"]:
                    pairs.append(result)
                    break  # One success per starting point is enough
        return pairs

    def sensitivity_from_contrastives(self, pairs: list[dict]) -> dict:
        """Analyze which parameters are most involved in outcome flips.

        Given a list of contrastive pairs (from find_contrastive or
        contrastive_pairs), computes statistics about which parameters
        change the most at decision boundaries.

        Args:
            pairs: List of contrastive pair dicts. Each must have
                "found" == True and a "delta" dict.

        Returns:
            Dict with:
                "param_importance": {param_name: mean_abs_delta},
                "param_flip_frequency": {param_name: fraction of pairs
                    where this param had the largest absolute delta},
                "n_pairs": int,
                "rankings": list of param names sorted by importance,
        """
        successful = [p for p in pairs if p.get("found", False) and p.get("delta")]
        if not successful:
            return {
                "param_importance": {},
                "param_flip_frequency": {},
                "n_pairs": 0,
                "rankings": [],
            }

        # Collect all parameter names
        all_params = set()
        for pair in successful:
            all_params.update(pair["delta"].keys())
        param_names = sorted(all_params)

        # Compute mean absolute delta per parameter
        abs_deltas = {name: [] for name in param_names}
        dominant_counts = {name: 0 for name in param_names}

        for pair in successful:
            delta = pair["delta"]
            # Find which param had the largest absolute change
            max_param = None
            max_abs = 0.0
            for name in param_names:
                d = abs(delta.get(name, 0.0))
                abs_deltas[name].append(d)
                if d > max_abs:
                    max_abs = d
                    max_param = name
            if max_param is not None:
                dominant_counts[max_param] += 1

        n_pairs = len(successful)
        param_importance = {
            name: float(np.mean(abs_deltas[name])) if abs_deltas[name] else 0.0
            for name in param_names
        }
        param_flip_frequency = {
            name: dominant_counts[name] / n_pairs
            for name in param_names
        }

        rankings = sorted(param_names, key=lambda n: param_importance[n], reverse=True)

        return {
            "param_importance": param_importance,
            "param_flip_frequency": param_flip_frequency,
            "n_pairs": n_pairs,
            "rankings": rankings,
        }
