"""Tests for POSIWID alignment auditor (zimmerman.posiwid).

Tests verify:
    1. Perfect alignment (intended == actual) scores 1.0.
    2. Opposite direction scores 0.0 on the direction component.
    3. Magnitude scoring: close predictions score higher than far ones.
    4. Batch audit returns proper aggregate statistics.
"""

import numpy as np
import pytest

from zimmerman.posiwid import POSIWIDAuditor


class _IdentitySimulator:
    """Toy simulator that returns params as outputs (perfect identity)."""

    def run(self, params: dict) -> dict:
        return dict(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {"a": (0.0, 1.0), "b": (-1.0, 1.0)}


class _OffsetSimulator:
    """Simulator that returns params with a fixed offset."""

    def __init__(self, offset: float = 0.1):
        self.offset = offset

    def run(self, params: dict) -> dict:
        return {k: v + self.offset for k, v in params.items()}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {"a": (0.0, 2.0), "b": (0.0, 2.0)}


class _NegatingSimulator:
    """Simulator that returns the negation of each parameter."""

    def run(self, params: dict) -> dict:
        return {k: -v for k, v in params.items()}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}


class TestScoreAlignment:
    """Tests for the score_alignment method."""

    def test_perfect_alignment_scores_one(self):
        """When intended exactly matches actual, overall should be 1.0."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        intended = {"a": 0.5, "b": 0.3}
        actual = {"a": 0.5, "b": 0.3}
        scores = auditor.score_alignment(intended, actual)

        assert scores["overall"] == pytest.approx(1.0, abs=0.01)
        for key in intended:
            assert scores["per_key"][key]["direction_match"] == pytest.approx(1.0)
            assert scores["per_key"][key]["magnitude_match"] == pytest.approx(1.0)
            assert scores["per_key"][key]["combined"] == pytest.approx(1.0)

    def test_opposite_direction_scores_zero_direction(self):
        """When actual is opposite sign from intended, direction should be 0."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        intended = {"a": 0.5}  # positive
        actual = {"a": -0.5}  # negative
        scores = auditor.score_alignment(intended, actual)

        assert scores["per_key"]["a"]["direction_match"] == pytest.approx(0.0)

    def test_same_direction_scores_one_direction(self):
        """When actual has same sign as intended, direction should be 1."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        intended = {"a": 0.5}
        actual = {"a": 0.3}
        scores = auditor.score_alignment(intended, actual)

        assert scores["per_key"]["a"]["direction_match"] == pytest.approx(1.0)

    def test_magnitude_close_vs_far(self):
        """Close prediction should have higher magnitude score than far."""
        auditor = POSIWIDAuditor(_IdentitySimulator())

        # Close prediction
        close_scores = auditor.score_alignment({"a": 1.0}, {"a": 0.95})
        # Far prediction
        far_scores = auditor.score_alignment({"a": 1.0}, {"a": 0.2})

        close_mag = close_scores["per_key"]["a"]["magnitude_match"]
        far_mag = far_scores["per_key"]["a"]["magnitude_match"]

        assert close_mag > far_mag, (
            f"Close magnitude ({close_mag}) should exceed far ({far_mag})"
        )

    def test_missing_key_counted(self):
        """Keys in intended but not in actual should be counted as missing."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        intended = {"a": 0.5, "nonexistent": 0.3}
        actual = {"a": 0.5}
        scores = auditor.score_alignment(intended, actual)

        assert scores["n_keys_missing"] == 1
        assert scores["n_keys_matched"] == 1

    def test_nan_in_actual_scores_zero(self):
        """NaN in actual output should score zero for that key."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        intended = {"a": 0.5}
        actual = {"a": float("nan")}
        scores = auditor.score_alignment(intended, actual)

        assert scores["per_key"]["a"]["combined"] == pytest.approx(0.0)


class TestAudit:
    """Tests for the audit method (runs simulator + scores)."""

    def test_identity_simulator_perfect_audit(self):
        """Identity simulator with intended == params should score ~1.0."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        result = auditor.audit(
            intended_outcomes={"a": 0.7, "b": 0.3},
            params={"a": 0.7, "b": 0.3},
        )

        assert result["alignment"]["overall"] == pytest.approx(1.0, abs=0.01)
        assert result["actual"]["a"] == pytest.approx(0.7)
        assert result["actual"]["b"] == pytest.approx(0.3)

    def test_negating_simulator_low_alignment(self):
        """Negating simulator should have low alignment for positive intended."""
        auditor = POSIWIDAuditor(_NegatingSimulator())
        result = auditor.audit(
            intended_outcomes={"a": 0.5, "b": 0.5},
            params={"a": 0.5, "b": 0.5},
        )

        # Actual is -0.5 when intended is 0.5: opposite direction
        assert result["alignment"]["overall"] < 0.3


class TestBatchAudit:
    """Tests for the batch_audit method."""

    def test_batch_returns_aggregate_stats(self):
        """Batch audit should return aggregate statistics."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        scenarios = [
            {"intended": {"a": 0.5, "b": 0.3}, "params": {"a": 0.5, "b": 0.3}},
            {"intended": {"a": 0.8, "b": 0.1}, "params": {"a": 0.8, "b": 0.1}},
        ]
        result = auditor.batch_audit(scenarios)

        agg = result["aggregate"]
        assert agg["n_scenarios"] == 2
        assert "mean_overall" in agg
        assert "std_overall" in agg
        assert "min_overall" in agg
        assert "max_overall" in agg
        assert "mean_direction_accuracy" in agg
        assert "mean_magnitude_accuracy" in agg

    def test_batch_perfect_scenarios(self):
        """All-perfect scenarios should give mean_overall = 1.0."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        scenarios = [
            {"intended": {"a": 0.5}, "params": {"a": 0.5}},
            {"intended": {"a": 0.8}, "params": {"a": 0.8}},
            {"intended": {"a": 0.2}, "params": {"a": 0.2}},
        ]
        result = auditor.batch_audit(scenarios)

        assert result["aggregate"]["mean_overall"] == pytest.approx(1.0, abs=0.01)

    def test_batch_mixed_scenarios(self):
        """Mixed perfect and bad scenarios should give intermediate mean."""
        auditor = POSIWIDAuditor(_NegatingSimulator())
        scenarios = [
            # This one will have low alignment (actual = -params)
            {"intended": {"a": 0.5}, "params": {"a": 0.5}},
            {"intended": {"a": 0.8}, "params": {"a": 0.8}},
        ]
        result = auditor.batch_audit(scenarios)

        agg = result["aggregate"]
        assert agg["n_scenarios"] == 2
        assert len(result["individual_results"]) == 2
        # Overall should be low because negating simulator opposes intended
        assert agg["mean_overall"] < 0.5

    def test_batch_per_key_mean(self):
        """Batch audit should compute per-key mean scores."""
        auditor = POSIWIDAuditor(_IdentitySimulator())
        scenarios = [
            {"intended": {"a": 0.5, "b": 0.3}, "params": {"a": 0.5, "b": 0.3}},
            {"intended": {"a": 0.8, "b": 0.1}, "params": {"a": 0.8, "b": 0.1}},
        ]
        result = auditor.batch_audit(scenarios)

        per_key = result["aggregate"]["per_key_mean"]
        assert "a" in per_key
        assert "b" in per_key
        assert per_key["a"]["combined"] == pytest.approx(1.0, abs=0.01)
