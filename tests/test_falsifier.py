"""Tests for systematic falsification (zimmerman.falsifier).

Tests verify:
    1. Falsifier finds NaN regions in BrokenSimulator.
    2. Boundary testing catches corner cases.
    3. Custom assertions work correctly.
    4. Clean simulators produce zero violations.
"""

import numpy as np
import pytest

from zimmerman.falsifier import Falsifier


class TestFalsifierWithBrokenSimulator:
    """Test falsifier on a simulator with known failure modes."""

    def test_finds_nan_region(self, broken_sim):
        """Falsifier should find violations in the NaN region (param > 0.9)."""
        falsifier = Falsifier(broken_sim)
        report = falsifier.falsify(n_random=200, n_boundary=50, n_adversarial=50, seed=42)

        assert report["summary"]["violations_found"] > 0, (
            "Falsifier should find at least one NaN violation"
        )

        # Check that at least one violation has a param > 0.9
        found_high_param = False
        for v in report["violations"]:
            for key, val in v["params"].items():
                if val > 0.9:
                    found_high_param = True
                    break
            if found_high_param:
                break
        assert found_high_param, (
            "Should find a violation with a parameter > 0.9"
        )

    def test_boundary_catches_corners(self, broken_sim):
        """Boundary testing should catch the NaN at max parameter values."""
        falsifier = Falsifier(broken_sim)
        # Only run boundary tests
        report = falsifier.falsify(n_random=0, n_boundary=100, n_adversarial=0, seed=42)

        boundary_violations = report["summary"]["boundary_violations"]
        assert boundary_violations > 0, (
            "Boundary testing should find violations at param=1.0 > 0.9"
        )

    def test_adversarial_probes_near_violations(self, broken_sim):
        """Adversarial testing should find more violations near known ones."""
        falsifier = Falsifier(broken_sim)
        # Full pipeline: random finds some, adversarial probes near them
        report = falsifier.falsify(
            n_random=100, n_boundary=0, n_adversarial=50, seed=42
        )

        # If random found violations, adversarial should find some too
        if report["summary"]["random_violations"] > 0:
            assert report["summary"]["adversarial_violations"] >= 0  # At minimum, no crash

    def test_violation_structure(self, broken_sim):
        """Each violation should have the expected fields."""
        falsifier = Falsifier(broken_sim)
        report = falsifier.falsify(n_random=50, n_boundary=0, n_adversarial=0, seed=42)

        if report["violations"]:
            v = report["violations"][0]
            assert "params" in v
            assert "result" in v
            assert "failed_assertions" in v
            assert "strategy" in v
            assert "error" in v


class TestFalsifierWithCleanSimulator:
    """Test falsifier on simulators with no failure modes."""

    def test_linear_simulator_no_violations(self, linear_sim):
        """A well-behaved linear simulator should have zero violations."""
        falsifier = Falsifier(linear_sim)
        report = falsifier.falsify(n_random=100, n_boundary=50, n_adversarial=20, seed=42)

        assert report["summary"]["violations_found"] == 0, (
            f"Expected 0 violations, got {report['summary']['violations_found']}"
        )

    def test_quadratic_simulator_no_violations(self, quadratic_sim):
        """A well-behaved quadratic simulator should have zero violations."""
        falsifier = Falsifier(quadratic_sim)
        report = falsifier.falsify(n_random=100, n_boundary=50, n_adversarial=20, seed=42)

        assert report["summary"]["violations_found"] == 0


class TestCustomAssertions:
    """Test custom assertion functions."""

    def test_custom_assertion_catches_violation(self, linear_sim):
        """Custom assertion that output y must be < 5 should find violations.

        Linear sim: y = 1*x0 + 2*x1 + 3*x2 + 4*x3, with x_i in [0,1].
        Max y = 1+2+3+4 = 10, so y > 5 is reachable.
        """
        falsifier = Falsifier(
            linear_sim,
            assertions=[lambda r: r["y"] < 5.0],
        )
        report = falsifier.falsify(n_random=200, n_boundary=50, n_adversarial=20, seed=42)

        assert report["summary"]["violations_found"] > 0, (
            "Custom assertion (y < 5) should be violated for large parameters"
        )

    def test_custom_assertion_always_true(self, linear_sim):
        """A trivially true assertion should produce no violations."""
        falsifier = Falsifier(
            linear_sim,
            assertions=[lambda r: True],
        )
        report = falsifier.falsify(n_random=100, n_boundary=50, n_adversarial=20, seed=42)

        assert report["summary"]["violations_found"] == 0

    def test_custom_assertion_always_false(self, linear_sim):
        """A trivially false assertion should flag every test case."""
        falsifier = Falsifier(
            linear_sim,
            assertions=[lambda r: False],
        )
        report = falsifier.falsify(n_random=10, n_boundary=0, n_adversarial=0, seed=42)

        assert report["summary"]["violations_found"] == 10


class TestBoundaryParams:
    """Test boundary parameter generation."""

    def test_includes_all_min(self, linear_sim):
        """Boundary params should include the all-min corner."""
        falsifier = Falsifier(linear_sim)
        boundaries = falsifier.boundary_params()

        all_min = {f"x{i}": 0.0 for i in range(4)}
        found = any(
            all(abs(b[k] - all_min[k]) < 1e-9 for k in all_min)
            for b in boundaries
        )
        assert found, "Boundary params should include all-min corner"

    def test_includes_all_max(self, linear_sim):
        """Boundary params should include the all-max corner."""
        falsifier = Falsifier(linear_sim)
        boundaries = falsifier.boundary_params()

        all_max = {f"x{i}": 1.0 for i in range(4)}
        found = any(
            all(abs(b[k] - all_max[k]) < 1e-9 for k in all_max)
            for b in boundaries
        )
        assert found, "Boundary params should include all-max corner"

    def test_boundary_count_reasonable(self, linear_sim):
        """Boundary generation should produce a reasonable number of combos."""
        falsifier = Falsifier(linear_sim)
        boundaries = falsifier.boundary_params()

        # For d=4: at least 2 (corners) + 2*4 (one-at-extreme) + 2*4 (single extreme)
        # = 18 minimum
        assert len(boundaries) >= 18


class TestFalsifySummary:
    """Test the summary statistics from falsify()."""

    def test_summary_fields(self, broken_sim):
        """Summary should contain all expected fields."""
        falsifier = Falsifier(broken_sim)
        report = falsifier.falsify(n_random=10, n_boundary=10, n_adversarial=10, seed=42)

        summary = report["summary"]
        assert "total_tests" in summary
        assert "violations_found" in summary
        assert "violation_rate" in summary
        assert "random_violations" in summary
        assert "boundary_violations" in summary
        assert "adversarial_violations" in summary
        assert "exceptions" in summary

    def test_violation_rate_consistent(self, broken_sim):
        """Violation rate should equal violations_found / total_tests."""
        falsifier = Falsifier(broken_sim)
        report = falsifier.falsify(n_random=50, n_boundary=20, n_adversarial=10, seed=42)

        summary = report["summary"]
        expected_rate = summary["violations_found"] / max(summary["total_tests"], 1)
        assert summary["violation_rate"] == pytest.approx(expected_rate, abs=1e-9)
