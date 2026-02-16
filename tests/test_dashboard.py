"""Tests for the meaning construction dashboard (zimmerman.meaning_construction_dashboard).

Tests verify:
    1. compile() with no reports returns a valid empty dashboard.
    2. compile() with sobol report populates the sensitivity section.
    3. compile() with falsifier report populates the robustness section.
    4. compile() with posiwid report populates the alignment section.
    5. compile() with token_extispicy report populates the representation section.
    6. compile() with pds report populates the structure section.
    7. tools_used and tools_missing are correctly tracked.
    8. generate_recommendations produces actionable strings.
    9. to_markdown renders without errors.
    10. run() and param_spec() satisfy the Simulator protocol.
"""

import numpy as np
import pytest

from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard


class TestCompileEmpty:
    """Tests for compile() with no reports."""

    def test_empty_reports(self, linear_sim):
        """compile() with no reports should return a valid dashboard."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile()

        assert "simulator_info" in result
        assert "sections" in result
        assert "recommendations" in result
        assert "tools_used" in result
        assert "tools_missing" in result

    def test_simulator_info(self, linear_sim):
        """simulator_info should reflect the underlying simulator."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile()
        info = result["simulator_info"]

        assert info["n_params"] == 4
        assert len(info["param_names"]) == 4
        assert "x0" in info["param_names"]
        assert "x1" in info["param_names"]
        assert "param_ranges" in info

    def test_all_sections_unavailable(self, linear_sim):
        """With no reports, all sections should be unavailable."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile()
        sections = result["sections"]

        for key in ["sensitivity", "robustness", "locality", "alignment",
                     "representation", "structure"]:
            assert key in sections
            assert sections[key]["available"] is False

    def test_all_tools_missing(self, linear_sim):
        """With no reports, all tools should be in tools_missing."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile()

        assert len(result["tools_used"]) == 0
        assert len(result["tools_missing"]) > 0

    def test_has_fallback_recommendation(self, linear_sim):
        """With no data, should still produce at least one recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile()
        assert len(result["recommendations"]) >= 1


class TestCompileWithSobol:
    """Tests for compile() with a sobol report."""

    def _make_sobol_report(self):
        """Create a mock sobol_sensitivity result dict."""
        return {
            "n_base": 256,
            "n_total_sims": 1536,
            "parameter_names": ["x0", "x1", "x2", "x3"],
            "output_keys": ["y"],
            "y": {
                "S1": {"x0": 0.03, "x1": 0.13, "x2": 0.30, "x3": 0.53},
                "ST": {"x0": 0.04, "x1": 0.14, "x2": 0.31, "x3": 0.54},
                "interaction": {"x0": 0.01, "x1": 0.01, "x2": 0.01, "x3": 0.01},
            },
            "rankings": {
                "y_most_influential_S1": ["x3", "x2", "x1", "x0"],
                "y_most_interactive": ["x0", "x1", "x2", "x3"],
            },
        }

    def test_sensitivity_section_available(self, linear_sim):
        """Sensitivity section should be available when sobol report given."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"sobol": self._make_sobol_report()})
        section = result["sections"]["sensitivity"]

        assert section["available"] is True

    def test_sensitivity_top_params(self, linear_sim):
        """Top params should come from sobol rankings."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"sobol": self._make_sobol_report()})
        section = result["sections"]["sensitivity"]

        assert section["top_params"] == ["x3", "x2", "x1", "x0"]

    def test_interaction_strength(self, linear_sim):
        """Interaction strength should be computed from sobol data."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"sobol": self._make_sobol_report()})
        section = result["sections"]["sensitivity"]

        assert section["interaction_strength"] is not None
        assert section["interaction_strength"] == pytest.approx(0.01, abs=0.001)

    def test_sobol_in_tools_used(self, linear_sim):
        """sobol should appear in tools_used."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"sobol": self._make_sobol_report()})
        assert "sobol" in result["tools_used"]
        assert "sobol" not in result["tools_missing"]


class TestCompileWithFalsifier:
    """Tests for compile() with a falsifier report."""

    def _make_falsifier_report(self, violation_rate=0.15):
        """Create a mock Falsifier.falsify result dict."""
        return {
            "violations": [
                {"params": {"x0": 0.95}, "result": {"y": float("nan")},
                 "failed_assertions": [0], "strategy": "random", "error": None}
            ],
            "summary": {
                "total_tests": 100,
                "violations_found": 15,
                "violation_rate": violation_rate,
                "random_violations": 10,
                "boundary_violations": 3,
                "adversarial_violations": 2,
                "exceptions": 0,
            },
        }

    def test_robustness_section_available(self, linear_sim):
        """Robustness section should be available when falsifier report given."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"falsifier": self._make_falsifier_report()})
        section = result["sections"]["robustness"]

        assert section["available"] is True
        assert section["violation_rate"] == pytest.approx(0.15)

    def test_high_violation_rate_recommendation(self, linear_sim):
        """High violation rate should produce a warning recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"falsifier": self._make_falsifier_report(0.15)})
        recs = result["recommendations"]

        assert any("violation" in r.lower() or "violation" in r.lower() for r in recs)


class TestCompileWithPosiwid:
    """Tests for compile() with a POSIWID report."""

    def _make_posiwid_report_single(self, overall=0.3):
        """Create a mock single-audit POSIWID result dict."""
        return {
            "intended": {"y": 5.0},
            "actual": {"y": 2.0},
            "params": {"x0": 0.5},
            "alignment": {
                "per_key": {
                    "y": {"direction_match": 1.0, "magnitude_match": 0.4, "combined": 0.7},
                },
                "overall": overall,
                "n_keys_matched": 1,
                "n_keys_missing": 0,
            },
        }

    def _make_posiwid_report_batch(self, mean_overall=0.3):
        """Create a mock batch_audit POSIWID result dict."""
        return {
            "individual_results": [],
            "aggregate": {
                "mean_overall": mean_overall,
                "std_overall": 0.1,
                "min_overall": 0.1,
                "max_overall": 0.5,
                "mean_direction_accuracy": 0.5,
                "mean_magnitude_accuracy": 0.2,
                "n_scenarios": 10,
                "per_key_mean": {
                    "y": {"direction": 0.5, "magnitude": 0.2, "combined": 0.35},
                    "fitness": {"direction": 0.3, "magnitude": 0.1, "combined": 0.2},
                },
            },
        }

    def test_alignment_from_single_audit(self, linear_sim):
        """Should extract alignment from a single audit result."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"posiwid": self._make_posiwid_report_single(0.3)})
        section = result["sections"]["alignment"]

        assert section["available"] is True
        assert section["overall_alignment"] == pytest.approx(0.3)

    def test_alignment_from_batch_audit(self, linear_sim):
        """Should extract alignment from a batch audit result."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"posiwid": self._make_posiwid_report_batch(0.3)})
        section = result["sections"]["alignment"]

        assert section["available"] is True
        assert section["overall_alignment"] == pytest.approx(0.3)

    def test_worst_aligned_keys(self, linear_sim):
        """Should identify worst-aligned keys from batch audit."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"posiwid": self._make_posiwid_report_batch(0.3)})
        section = result["sections"]["alignment"]

        assert section["worst_aligned_keys"] is not None
        # "fitness" has lower combined (0.2) than "y" (0.35), so it should be first
        assert section["worst_aligned_keys"][0] == "fitness"

    def test_low_alignment_recommendation(self, linear_sim):
        """Low alignment should produce a POSIWID warning recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"posiwid": self._make_posiwid_report_single(0.3)})
        recs = result["recommendations"]

        assert any("posiwid" in r.lower() or "alignment" in r.lower() for r in recs)


class TestCompileWithTokenExtispicy:
    """Tests for compile() with a token_extispicy report."""

    def _make_token_ext_report(self, mean_corr=0.5):
        """Create a mock TokenExtispicyWorkbench.analyze result dict."""
        return {
            "fragmentation_stats": {
                "mean_tokens_per_param": 3.5,
                "mean_tokens_per_digit": 1.2,
                "std_tokens_per_param": 0.8,
                "max_fragmentation_param": "x3",
            },
            "perturbation_sensitivity": {"x0": 2, "x1": 3, "x2": 1, "x3": 4},
            "fragmentation_output_correlation": {
                "y": mean_corr,
                "fitness": -0.1,
            },
            "hazard_zones": [],
            "token_edit_vs_string_edit": {"mean_ratio": 1.2, "max_ratio": 2.0},
            "n_samples": 100,
            "n_sims": 100,
        }

    def test_representation_section_available(self, linear_sim):
        """Representation section should be available with token_extispicy."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"token_extispicy": self._make_token_ext_report()})
        section = result["sections"]["representation"]

        assert section["available"] is True

    def test_fragmentation_correlation_computed(self, linear_sim):
        """fragmentation_correlation should be mean |corr| across outputs."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"token_extispicy": self._make_token_ext_report(0.5)})
        section = result["sections"]["representation"]

        # Mean of |0.5| and |-0.1| = 0.3
        assert section["fragmentation_correlation"] is not None
        assert section["fragmentation_correlation"] == pytest.approx(0.3, abs=0.01)

    def test_high_fragmentation_recommendation(self, linear_sim):
        """High fragmentation correlation should trigger a recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"token_extispicy": self._make_token_ext_report(0.8)})
        recs = result["recommendations"]

        # |0.8| and |0.1| -> mean 0.45 > 0.3 threshold
        assert any("fragmentation" in r.lower() or "diegetic" in r.lower() for r in recs)


class TestCompileWithPDS:
    """Tests for compile() with a PDS report."""

    def _make_pds_report(self, ve_y=0.2):
        """Create a mock PDSMapper.audit_mapping result dict."""
        return {
            "dimension_output_correlations": {
                "power": {"y": 0.7},
                "danger": {"y": -0.3},
            },
            "variance_explained": {"y": ve_y},
            "n_samples": 100,
            "dimension_stats": {
                "power": {"mean": 0.01, "std": 0.58},
                "danger": {"mean": -0.02, "std": 0.57},
            },
            "output_keys": ["y"],
        }

    def test_structure_section_available(self, linear_sim):
        """Structure section should be available with pds report."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"pds": self._make_pds_report()})
        section = result["sections"]["structure"]

        assert section["available"] is True

    def test_variance_explained_extracted(self, linear_sim):
        """variance_explained should be extracted from pds report."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"pds": self._make_pds_report(0.2)})
        section = result["sections"]["structure"]

        assert section["variance_explained"] == pytest.approx(0.2)

    def test_low_variance_recommendation(self, linear_sim):
        """Low variance explained should trigger mapping revision recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"pds": self._make_pds_report(0.15)})
        recs = result["recommendations"]

        assert any("variance" in r.lower() or "mapping" in r.lower() for r in recs)


class TestCompileMultipleReports:
    """Tests for compile() with multiple reports."""

    def test_multiple_tools_used(self, linear_sim):
        """Multiple reports should all appear in tools_used."""
        sobol_report = {
            "n_base": 64, "n_total_sims": 384,
            "parameter_names": ["x0", "x1", "x2", "x3"],
            "output_keys": ["y"],
            "y": {"S1": {"x0": 0.1}, "ST": {"x0": 0.1}, "interaction": {"x0": 0.0}},
            "rankings": {"y_most_influential_S1": ["x0"]},
        }
        falsifier_report = {
            "violations": [],
            "summary": {"total_tests": 50, "violations_found": 0,
                        "violation_rate": 0.0, "random_violations": 0,
                        "boundary_violations": 0, "adversarial_violations": 0,
                        "exceptions": 0},
        }

        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={
            "sobol": sobol_report,
            "falsifier": falsifier_report,
        })

        assert "sobol" in result["tools_used"]
        assert "falsifier" in result["tools_used"]
        assert "sobol" not in result["tools_missing"]
        assert "falsifier" not in result["tools_missing"]

    def test_mixed_available_sections(self, linear_sim):
        """Only sections with matching reports should be available."""
        sobol_report = {
            "n_base": 64, "n_total_sims": 384,
            "parameter_names": ["x0", "x1", "x2", "x3"],
            "output_keys": ["y"],
            "y": {"S1": {"x0": 0.1}, "ST": {"x0": 0.1}, "interaction": {"x0": 0.0}},
            "rankings": {"y_most_influential_S1": ["x0"]},
        }

        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"sobol": sobol_report})
        sections = result["sections"]

        assert sections["sensitivity"]["available"] is True
        assert sections["robustness"]["available"] is False
        assert sections["locality"]["available"] is False
        assert sections["alignment"]["available"] is False
        assert sections["representation"]["available"] is False
        assert sections["structure"]["available"] is False


class TestToMarkdown:
    """Tests for the to_markdown rendering method."""

    def test_renders_without_error(self, linear_sim):
        """to_markdown should produce a string without raising."""
        dash = MeaningConstructionDashboard(linear_sim)
        dashboard = dash.compile()
        md = dash.to_markdown(dashboard)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_contains_header(self, linear_sim):
        """Markdown should contain the main header."""
        dash = MeaningConstructionDashboard(linear_sim)
        dashboard = dash.compile()
        md = dash.to_markdown(dashboard)
        assert "# Meaning Construction Dashboard" in md

    def test_contains_sections(self, linear_sim):
        """Markdown should contain section headers."""
        dash = MeaningConstructionDashboard(linear_sim)
        dashboard = dash.compile()
        md = dash.to_markdown(dashboard)
        assert "## Sensitivity" in md
        assert "## Robustness" in md
        assert "## Recommendations" in md
        assert "## Tool Coverage" in md

    def test_renders_with_data(self, linear_sim):
        """Markdown should include data when reports are provided."""
        sobol_report = {
            "n_base": 64, "n_total_sims": 384,
            "parameter_names": ["x0", "x1", "x2", "x3"],
            "output_keys": ["y"],
            "y": {"S1": {"x0": 0.1}, "ST": {"x0": 0.1}, "interaction": {"x0": 0.0}},
            "rankings": {"y_most_influential_S1": ["x0"]},
        }

        dash = MeaningConstructionDashboard(linear_sim)
        dashboard = dash.compile(reports={"sobol": sobol_report})
        md = dash.to_markdown(dashboard)
        assert "x0" in md


class TestGenerateRecommendations:
    """Tests for the generate_recommendations method."""

    def test_returns_list_of_strings(self, linear_sim):
        """Should always return a list of strings."""
        dash = MeaningConstructionDashboard(linear_sim)
        sections = {
            "sensitivity": {"available": False},
            "robustness": {"available": False},
            "locality": {"available": False},
            "alignment": {"available": False},
            "representation": {"available": False},
            "structure": {"available": False},
        }
        recs = dash.generate_recommendations(sections)
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)

    def test_never_empty(self, linear_sim):
        """Should always produce at least one recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        sections = {
            "sensitivity": {"available": False},
            "robustness": {"available": False},
            "locality": {"available": False},
            "alignment": {"available": False},
            "representation": {"available": False},
            "structure": {"available": False},
        }
        recs = dash.generate_recommendations(sections)
        assert len(recs) >= 1

    def test_sensitivity_recommendation(self, linear_sim):
        """Available sensitivity data should produce a recommendation."""
        dash = MeaningConstructionDashboard(linear_sim)
        sections = {
            "sensitivity": {"available": True, "top_params": ["x3", "x2"], "interaction_strength": 0.01},
            "robustness": {"available": False},
            "locality": {"available": False},
            "alignment": {"available": False},
            "representation": {"available": False},
            "structure": {"available": False},
        }
        recs = dash.generate_recommendations(sections)
        assert any("influential" in r.lower() or "x3" in r for r in recs)


class TestSimulatorProtocol:
    """Tests that MeaningConstructionDashboard satisfies the Simulator protocol."""

    def test_run_delegates(self, linear_sim):
        """run() should return same result as underlying simulator."""
        dash = MeaningConstructionDashboard(linear_sim)
        params = {"x0": 0.5, "x1": 0.3, "x2": 0.7, "x3": 0.1}
        direct = linear_sim.run(params)
        wrapped = dash.run(params)

        assert direct["y"] == pytest.approx(wrapped["y"])

    def test_param_spec_delegates(self, linear_sim):
        """param_spec() should match underlying simulator."""
        dash = MeaningConstructionDashboard(linear_sim)
        assert dash.param_spec() == linear_sim.param_spec()

    def test_works_with_quadratic(self, quadratic_sim):
        """Should work with a different simulator type."""
        dash = MeaningConstructionDashboard(quadratic_sim)
        result = dash.compile()
        assert result["simulator_info"]["n_params"] == 3


class TestCompileWithLocality:
    """Tests for compile() with a locality profiler report."""

    def test_locality_section_available(self, linear_sim):
        """Locality section should be available when locality report given."""
        locality_report = {
            "L50": {"cut_frac": 0.3, "mask_frac": 0.4},
            "effective_horizon": 0.25,
        }
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"locality": locality_report})
        section = result["sections"]["locality"]

        assert section["available"] is True
        assert section["effective_horizon"] == pytest.approx(0.25)
        assert section["L50"] is not None

    def test_short_horizon_recommendation(self, linear_sim):
        """Short effective horizon should trigger a recommendation."""
        locality_report = {
            "L50": {"cut_frac": 0.1},
            "effective_horizon": 0.15,
        }
        dash = MeaningConstructionDashboard(linear_sim)
        result = dash.compile(reports={"locality": locality_report})
        recs = result["recommendations"]

        assert any("horizon" in r.lower() or "local" in r.lower() for r in recs)
