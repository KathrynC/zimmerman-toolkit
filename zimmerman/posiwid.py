"""POSIWID alignment auditor: "The Purpose Of a System Is What It Does."

Generalized from how-to-live-much-longer/posiwid_audit.py. Measures the
gap between intended outcomes and actual simulation outcomes.

The POSIWID principle (Stafford Beer, 1974) states that the purpose of a
system is not what it claims to intend, but what it actually does. This
module quantifies that gap for any simulator: given intended outcomes
(from an LLM, a human designer, or an optimization target) and actual
parameters, how well do the actual results match the intention?

This is simulator-agnostic: it works with any object satisfying the
Simulator protocol.

Reference:
    Stafford Beer (1974). "Designing Freedom." CBC Massey Lectures.
    Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. Ch. 5.
"""

from __future__ import annotations

import numpy as np


class POSIWIDAuditor:
    """Audits the alignment between intended and actual simulation outcomes.

    Given a simulator, compares what was intended (e.g., "reduce fitness by 10%")
    against what actually happens when the simulator is run with specific parameters.

    Args:
        simulator: Any Simulator-compatible object with run() and param_spec().

    Example:
        auditor = POSIWIDAuditor(my_simulator)
        result = auditor.audit(
            intended_outcomes={"fitness": 0.8, "energy": 0.5},
            params={"speed": 0.3, "strength": 0.7},
        )
        print(result["alignment"]["overall"])
    """

    def __init__(self, simulator):
        self.simulator = simulator

    def audit(
        self,
        intended_outcomes: dict[str, float],
        params: dict[str, float],
    ) -> dict:
        """Run simulator with params, compare against intended outcomes.

        Args:
            intended_outcomes: Dict mapping output key names to their
                intended/expected numeric values.
            params: Dict mapping parameter names to float values to
                pass to the simulator.

        Returns:
            Dict with:
                "intended": the intended_outcomes dict,
                "actual": the actual simulator output dict,
                "params": the params dict,
                "alignment": alignment scores from score_alignment(),
        """
        actual = self.simulator.run(params)
        alignment = self.score_alignment(intended_outcomes, actual)
        return {
            "intended": dict(intended_outcomes),
            "actual": dict(actual),
            "params": dict(params),
            "alignment": alignment,
        }

    def score_alignment(
        self,
        intended: dict[str, float],
        actual: dict[str, float],
    ) -> dict:
        """Score how well actual outcomes match intended outcomes.

        For each key in intended:
        - direction_match: Did the value move in the expected direction
          relative to the midpoint of the intended value and 0? Computed
          as: does actual have the same sign-relative-to-zero as intended?
          Binary 0 or 1.
        - magnitude_match: How close is the actual value to the intended
          value? Computed as max(0, 1 - |actual - intended| / scale),
          where scale = max(|intended|, 0.1). Continuous 0 to 1.
        - combined: 0.5 * direction_match + 0.5 * magnitude_match.

        Args:
            intended: Dict mapping output keys to intended float values.
            actual: Dict mapping output keys to actual float values.

        Returns:
            Dict with:
                "per_key": {key: {"direction_match": float, "magnitude_match": float, "combined": float}},
                "overall": float (mean of all combined scores),
                "n_keys_matched": int (number of intended keys found in actual),
                "n_keys_missing": int (intended keys not found in actual),
        """
        per_key = {}
        matched_count = 0
        missing_count = 0

        for key, intended_val in intended.items():
            if key not in actual:
                missing_count += 1
                continue

            actual_val = actual[key]
            if not isinstance(actual_val, (int, float, np.integer, np.floating)):
                missing_count += 1
                continue
            actual_val = float(actual_val)
            intended_val = float(intended_val)

            if not (np.isfinite(actual_val) and np.isfinite(intended_val)):
                per_key[key] = {
                    "direction_match": 0.0,
                    "magnitude_match": 0.0,
                    "combined": 0.0,
                }
                matched_count += 1
                continue

            # Direction match: same sign relative to zero
            if intended_val == 0.0:
                # Intended zero: penalize by magnitude of deviation
                direction_match = max(0.0, 1.0 - abs(actual_val) * 5.0)
            elif actual_val == 0.0:
                direction_match = 0.0
            else:
                direction_match = 1.0 if (intended_val * actual_val > 0) else 0.0

            # Magnitude match: closeness of values
            scale = max(abs(intended_val), 0.1)
            error = abs(actual_val - intended_val)
            magnitude_match = float(max(0.0, 1.0 - error / scale))

            combined = 0.5 * direction_match + 0.5 * magnitude_match

            per_key[key] = {
                "direction_match": float(direction_match),
                "magnitude_match": float(magnitude_match),
                "combined": float(combined),
            }
            matched_count += 1

        # Overall score
        if per_key:
            overall = float(np.mean([v["combined"] for v in per_key.values()]))
        else:
            overall = 0.0

        return {
            "per_key": per_key,
            "overall": overall,
            "n_keys_matched": matched_count,
            "n_keys_missing": missing_count,
        }

    def batch_audit(
        self,
        scenarios: list[dict],
    ) -> dict:
        """Audit multiple scenarios and return aggregate statistics.

        Args:
            scenarios: List of dicts, each with:
                "intended": dict of intended outcome values,
                "params": dict of parameter values.
                Optionally "label": str for identification.

        Returns:
            Dict with:
                "individual_results": list of audit results,
                "aggregate": {
                    "mean_overall": float,
                    "std_overall": float,
                    "min_overall": float,
                    "max_overall": float,
                    "mean_direction_accuracy": float,
                    "mean_magnitude_accuracy": float,
                    "n_scenarios": int,
                    "per_key_mean": {key: {"direction": float, "magnitude": float, "combined": float}},
                },
        """
        individual_results = []
        overall_scores = []
        all_direction_scores = []
        all_magnitude_scores = []
        per_key_accum = {}

        for scenario in scenarios:
            intended = scenario["intended"]
            params = scenario["params"]
            result = self.audit(intended, params)
            individual_results.append(result)

            alignment = result["alignment"]
            overall_scores.append(alignment["overall"])

            for key, scores in alignment["per_key"].items():
                all_direction_scores.append(scores["direction_match"])
                all_magnitude_scores.append(scores["magnitude_match"])

                if key not in per_key_accum:
                    per_key_accum[key] = {
                        "direction": [],
                        "magnitude": [],
                        "combined": [],
                    }
                per_key_accum[key]["direction"].append(scores["direction_match"])
                per_key_accum[key]["magnitude"].append(scores["magnitude_match"])
                per_key_accum[key]["combined"].append(scores["combined"])

        # Aggregate per-key means
        per_key_mean = {}
        for key, accum in per_key_accum.items():
            per_key_mean[key] = {
                "direction": float(np.mean(accum["direction"])) if accum["direction"] else 0.0,
                "magnitude": float(np.mean(accum["magnitude"])) if accum["magnitude"] else 0.0,
                "combined": float(np.mean(accum["combined"])) if accum["combined"] else 0.0,
            }

        overall_arr = np.array(overall_scores) if overall_scores else np.array([0.0])

        aggregate = {
            "mean_overall": float(np.mean(overall_arr)),
            "std_overall": float(np.std(overall_arr)),
            "min_overall": float(np.min(overall_arr)),
            "max_overall": float(np.max(overall_arr)),
            "mean_direction_accuracy": float(np.mean(all_direction_scores)) if all_direction_scores else 0.0,
            "mean_magnitude_accuracy": float(np.mean(all_magnitude_scores)) if all_magnitude_scores else 0.0,
            "n_scenarios": len(scenarios),
            "per_key_mean": per_key_mean,
        }

        return {
            "individual_results": individual_results,
            "aggregate": aggregate,
        }
