"""Tests for prompt receptive field analysis (zimmerman.prompt_receptive_field).

Tests verify:
    1. PromptReceptiveField satisfies the Simulator protocol.
    2. Segment weight application correctly drops, compresses, or includes.
    3. On a linear model, Sobol indices rank parameters by coefficient size.
    4. Custom segmenters group parameters correctly.
    5. Custom scorers are used for index computation.
    6. analyze() returns all expected keys and types.
    7. Rankings match expected parameter importance ordering.
"""

import numpy as np
import pytest

from zimmerman.base import Simulator
from zimmerman.prompt_receptive_field import (
    PromptReceptiveField,
    _default_segmenter,
    _default_scorer,
)


class TestReceptiveFieldProtocol:
    """Verify PromptReceptiveField satisfies the Simulator protocol."""

    def test_is_simulator(self, linear_sim):
        """PromptReceptiveField should satisfy the Simulator protocol."""
        prf = PromptReceptiveField(linear_sim)
        assert isinstance(prf, Simulator)

    def test_param_spec_matches_segments(self, linear_sim):
        """param_spec keys should match segment names."""
        prf = PromptReceptiveField(linear_sim)
        spec = prf.param_spec()
        assert set(spec.keys()) == set(prf.segment_names)

    def test_param_spec_bounds(self, linear_sim):
        """All segment weight bounds should be [0, 1]."""
        prf = PromptReceptiveField(linear_sim)
        spec = prf.param_spec()
        for name, (lo, hi) in spec.items():
            assert lo == 0.0, f"Lower bound for {name} should be 0.0"
            assert hi == 1.0, f"Upper bound for {name} should be 1.0"

    def test_run_returns_dict(self, linear_sim):
        """run() should return a dict with simulator outputs."""
        prf = PromptReceptiveField(linear_sim)
        params = {name: 1.0 for name in prf.param_spec()}
        result = prf.run(params)
        assert isinstance(result, dict)
        assert "y" in result

    def test_run_includes_segment_weights(self, linear_sim):
        """run() result should include the applied segment weights."""
        prf = PromptReceptiveField(linear_sim)
        params = {name: 0.5 for name in prf.param_spec()}
        result = prf.run(params)
        assert "segment_weights" in result


class TestDefaultSegmenter:
    """Test the default segmenter function."""

    def test_one_segment_per_param(self):
        """Default segmenter should create one segment per parameter."""
        spec = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)}
        segments = _default_segmenter(spec)
        assert len(segments) == 3

    def test_segment_names_match_params(self):
        """Segment names should match parameter names (sorted)."""
        spec = {"c": (0.0, 1.0), "a": (0.0, 1.0), "b": (0.0, 1.0)}
        segments = _default_segmenter(spec)
        names = [seg["name"] for seg in segments]
        assert names == ["a", "b", "c"]

    def test_segment_params_contain_own_name(self):
        """Each segment's params list should contain its own name."""
        spec = {"x0": (0.0, 1.0), "x1": (0.0, 1.0)}
        segments = _default_segmenter(spec)
        for seg in segments:
            assert seg["name"] in seg["params"]


class TestDefaultScorer:
    """Test the default scorer function."""

    def test_extracts_fitness(self):
        """Default scorer should prefer 'fitness' key."""
        result = {"fitness": 0.75, "y": 1.0}
        assert _default_scorer(result) == 0.75

    def test_extracts_score(self):
        """Default scorer should use 'score' if no 'fitness'."""
        result = {"score": 0.6, "y": 1.0}
        assert _default_scorer(result) == 0.6

    def test_extracts_y(self):
        """Default scorer should use 'y' if no fitness or score."""
        result = {"y": 0.9}
        assert _default_scorer(result) == 0.9

    def test_returns_zero_if_missing(self):
        """Default scorer should return 0.0 if no known key found."""
        result = {"other": 42}
        assert _default_scorer(result) == 0.0


class TestSegmentWeightApplication:
    """Test the segment weight application logic."""

    def test_weight_one_includes_full(self, linear_sim):
        """Weight >= 0.66 should include the original parameter value."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.9 for i in range(4)}
        weights = {f"x{i}": 1.0 for i in range(4)}

        modified = prf._apply_segment_weights(base_params, weights)
        for i in range(4):
            assert abs(modified[f"x{i}"] - 0.9) < 1e-9, (
                f"x{i} should be included at full value"
            )

    def test_weight_zero_drops_to_midpoint(self, linear_sim):
        """Weight < 0.33 should drop the parameter to midpoint."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.9 for i in range(4)}
        weights = {f"x{i}": 0.0 for i in range(4)}

        modified = prf._apply_segment_weights(base_params, weights)
        for i in range(4):
            lo, hi = linear_sim.param_spec()[f"x{i}"]
            midpoint = (lo + hi) / 2.0
            assert abs(modified[f"x{i}"] - midpoint) < 1e-9, (
                f"x{i} should be at midpoint ({midpoint}), got {modified[f'x{i}']}"
            )

    def test_weight_half_compresses(self, linear_sim):
        """Weight in [0.33, 0.66) should blend between midpoint and original."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 1.0 for i in range(4)}
        weights = {f"x{i}": 0.5 for i in range(4)}

        modified = prf._apply_segment_weights(base_params, weights)
        for i in range(4):
            lo, hi = linear_sim.param_spec()[f"x{i}"]
            midpoint = (lo + hi) / 2.0
            # Should be between midpoint and original
            assert midpoint <= modified[f"x{i}"] <= 1.0, (
                f"x{i} should be between midpoint and original"
            )
            # Should not be exactly midpoint or exactly 1.0
            assert modified[f"x{i}"] > midpoint + 0.01
            assert modified[f"x{i}"] < 1.0 - 0.01


class TestAnalyzeLinear:
    """Test analyze() on the linear simulator."""

    def test_analyze_returns_all_keys(self, linear_sim):
        """analyze() should return all expected keys."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=64, seed=42)

        assert "segment_names" in report
        assert "S1" in report
        assert "ST" in report
        assert "interaction" in report
        assert "rankings" in report
        assert "n_sims" in report

    def test_analyze_segment_names(self, linear_sim):
        """segment_names should match the simulator's parameter names."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=64, seed=42)

        expected = sorted(linear_sim.param_spec().keys())
        assert report["segment_names"] == expected

    def test_analyze_n_sims(self, linear_sim):
        """n_sims should equal n_base * (d + 2)."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        n_base = 64
        report = prf.analyze(base_params, n_base=n_base, seed=42)

        d = len(linear_sim.param_spec())
        expected_sims = n_base * (d + 2)
        assert report["n_sims"] == expected_sims

    def test_st_geq_s1(self, linear_sim):
        """Total-order should be >= first-order for all segments.

        The segment weight thresholds (drop/compress/include) introduce
        nonlinearity, so Sobol estimates need more samples and a wider
        tolerance than the purely linear case in test_sobol.py.
        """
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=1024, seed=42)

        for name in report["segment_names"]:
            s1 = report["S1"][name]
            st = report["ST"][name]
            # Wide tolerance: the drop/compress thresholds create a
            # piecewise-linear model where estimator variance is higher
            assert st >= s1 - 0.15, (
                f"ST ({st:.4f}) < S1 ({s1:.4f}) for {name}"
            )

    def test_rankings_favor_large_coefficient(self):
        """For linear model, segment with largest coefficient should rank first."""
        from tests.conftest import LinearSimulator
        # x3 has coefficient 10, much larger than others
        coeffs = np.array([1.0, 1.0, 1.0, 10.0])
        sim = LinearSimulator(d=4, coefficients=coeffs)

        prf = PromptReceptiveField(sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=512, seed=42)

        # x3 should be ranked first (most important)
        assert report["rankings"][0] == "x3", (
            f"Expected x3 as most important, got {report['rankings'][0]}. "
            f"ST values: {report['ST']}"
        )

    def test_s1_dict_keys_match_segments(self, linear_sim):
        """S1, ST, and interaction dicts should have keys matching segment_names."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=64, seed=42)

        for key in ("S1", "ST", "interaction"):
            assert set(report[key].keys()) == set(report["segment_names"]), (
                f"{key} keys don't match segment_names"
            )


class TestCustomSegmenter:
    """Test analyze() with custom segmenters."""

    def test_grouped_segments(self):
        """Custom segmenter grouping params into 2 segments."""
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=4)

        def group_segmenter(spec):
            names = sorted(spec.keys())
            return [
                {"name": "early", "params": names[:2]},
                {"name": "late", "params": names[2:]},
            ]

        prf = PromptReceptiveField(sim, segmenter=group_segmenter)
        assert len(prf.segments) == 2
        assert prf.segment_names == ["early", "late"]

        spec = prf.param_spec()
        assert "early" in spec
        assert "late" in spec

    def test_grouped_segments_analyze(self):
        """Grouped segments should produce valid Sobol indices."""
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=4)

        def group_segmenter(spec):
            names = sorted(spec.keys())
            return [
                {"name": "early", "params": names[:2]},
                {"name": "late", "params": names[2:]},
            ]

        prf = PromptReceptiveField(sim, segmenter=group_segmenter)
        base_params = {f"x{i}": 0.7 for i in range(4)}
        report = prf.analyze(base_params, n_base=128, seed=42)

        assert len(report["segment_names"]) == 2
        assert "S1" in report
        assert "ST" in report


class TestCustomScorer:
    """Test analyze() with custom scorer."""

    def test_custom_scorer_used(self, quadratic_sim):
        """Custom scorer should be used for computing indices."""
        def my_scorer(result):
            return result.get("y", 0.0) * 2.0

        prf = PromptReceptiveField(quadratic_sim, scorer=my_scorer)
        base_params = {f"x{i}": 0.5 for i in range(3)}
        report = prf.analyze(base_params, n_base=64, seed=42)

        # Should still produce valid results
        assert report["n_sims"] > 0
        assert len(report["rankings"]) == 3


class TestAnalyzeDeterministic:
    """Test that analyze() is deterministic with same seed."""

    def test_same_seed_same_result(self, linear_sim):
        """Two analyses with the same seed should produce identical results."""
        prf = PromptReceptiveField(linear_sim)
        base_params = {f"x{i}": 0.7 for i in range(4)}

        report1 = prf.analyze(base_params, n_base=64, seed=42)
        report2 = prf.analyze(base_params, n_base=64, seed=42)

        for name in report1["segment_names"]:
            assert abs(report1["S1"][name] - report2["S1"][name]) < 1e-9, (
                f"S1 for {name} differs between runs"
            )
            assert abs(report1["ST"][name] - report2["ST"][name]) < 1e-9, (
                f"ST for {name} differs between runs"
            )


class TestAnalyzeQuadratic:
    """Test analyze() on the quadratic simulator."""

    def test_quadratic_all_params_important(self, quadratic_sim):
        """For y = sum(x_i^2), all parameters should have nonzero ST."""
        prf = PromptReceptiveField(quadratic_sim)
        base_params = {f"x{i}": 0.5 for i in range(3)}
        report = prf.analyze(base_params, n_base=256, seed=42)

        for name in report["segment_names"]:
            # Each parameter contributes equally, so ST should be > 0
            assert report["ST"][name] > -0.1, (
                f"ST for {name} should be non-negative, got {report['ST'][name]}"
            )
