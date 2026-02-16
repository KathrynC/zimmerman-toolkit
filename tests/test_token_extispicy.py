"""Tests for token extispicy workbench (zimmerman.token_extispicy).

Tests verify:
    1. Default tokenizer splits on digit/non-digit boundaries correctly.
    2. fragmentation_rate produces sensible values (tokens/chars).
    3. perturbation_token_sensitivity detects token changes from epsilon perturbations.
    4. analyze() returns all expected keys with correct types.
    5. Custom tokenizer callback is used when provided.
    6. run() and param_spec() satisfy the Simulator protocol.
    7. Hazard zones are sorted by fragmentation (descending).
    8. Deterministic output with fixed seed.
"""

import numpy as np
import pytest

from zimmerman.token_extispicy import TokenExtispicyWorkbench, _default_tokenize


class TestDefaultTokenize:
    """Tests for the default whitespace + digit boundary tokenizer."""

    def test_simple_assignment(self):
        """'x=3.14' should split at digit/non-digit boundaries.

        The regex splits at digit<->non-digit boundaries:
        'x=' is all non-digit, '3' is digit, '.' is non-digit, '14' is digit.
        """
        tokens = _default_tokenize("x=3.14")
        assert tokens == ["x=", "3", ".", "14"]

    def test_pure_digits(self):
        """'12345' should stay as one token (no boundaries within digits)."""
        tokens = _default_tokenize("12345")
        assert tokens == ["12345"]

    def test_pure_alpha(self):
        """'hello' should stay as one token."""
        tokens = _default_tokenize("hello")
        assert tokens == ["hello"]

    def test_whitespace_splitting(self):
        """'a 1 b' should split on whitespace first."""
        tokens = _default_tokenize("a 1 b")
        assert tokens == ["a", "1", "b"]

    def test_decimal_number(self):
        """'0.001' should split at digit/non-digit boundaries."""
        tokens = _default_tokenize("0.001")
        assert tokens == ["0", ".", "001"]

    def test_empty_string(self):
        """Empty string should return empty list."""
        tokens = _default_tokenize("")
        assert tokens == []

    def test_mixed_content(self):
        """JSON-like string should produce multiple tokens."""
        tokens = _default_tokenize('{"x0": 0.5}')
        assert len(tokens) > 1
        # Should split digits from non-digits
        assert any(t.isdigit() for t in tokens)


class TestFragmentationRate:
    """Tests for the fragmentation_rate method."""

    def test_empty_string_returns_zero(self, linear_sim):
        """Empty string should have fragmentation rate 0."""
        wb = TokenExtispicyWorkbench(linear_sim)
        assert wb.fragmentation_rate("") == 0.0

    def test_positive_rate(self, linear_sim):
        """Non-empty string should have positive fragmentation rate."""
        wb = TokenExtispicyWorkbench(linear_sim)
        rate = wb.fragmentation_rate("x=3.14")
        assert rate > 0.0

    def test_higher_fragmentation_for_numbers(self, linear_sim):
        """Numeric strings should fragment more than pure alpha strings."""
        wb = TokenExtispicyWorkbench(linear_sim)
        alpha_rate = wb.fragmentation_rate("abcdefghij")
        numeric_rate = wb.fragmentation_rate("0.123456789")
        # Numeric strings split at digit/non-digit boundaries (the decimal point)
        assert numeric_rate >= alpha_rate


class TestParamsToString:
    """Tests for the params_to_string method."""

    def test_deterministic(self, linear_sim):
        """Same params should produce same string."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.3, "x2": 0.7, "x3": 0.1}
        s1 = wb.params_to_string(params)
        s2 = wb.params_to_string(params)
        assert s1 == s2

    def test_sorted_keys(self, linear_sim):
        """Output should have sorted keys regardless of input order."""
        wb = TokenExtispicyWorkbench(linear_sim)
        p1 = {"x3": 0.1, "x0": 0.5, "x2": 0.7, "x1": 0.3}
        p2 = {"x0": 0.5, "x1": 0.3, "x2": 0.7, "x3": 0.1}
        assert wb.params_to_string(p1) == wb.params_to_string(p2)


class TestPerturbationSensitivity:
    """Tests for perturbation_token_sensitivity."""

    def test_returns_all_params(self, linear_sim):
        """Should return a sensitivity value for each parameter."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        sensitivity = wb.perturbation_token_sensitivity(params, epsilon=0.01)

        for name in linear_sim.param_spec():
            assert name in sensitivity

    def test_non_negative_values(self, linear_sim):
        """All sensitivity values should be non-negative."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        sensitivity = wb.perturbation_token_sensitivity(params, epsilon=0.01)

        for name, val in sensitivity.items():
            assert val >= 0, f"Sensitivity for {name} is negative: {val}"

    def test_larger_epsilon_more_sensitive(self, linear_sim):
        """Larger epsilon should generally cause at least as many token changes."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        small = wb.perturbation_token_sensitivity(params, epsilon=0.001)
        large = wb.perturbation_token_sensitivity(params, epsilon=0.1)

        total_small = sum(small.values())
        total_large = sum(large.values())
        # Larger perturbation should cause at least as many total changes
        # (with tolerance for edge cases)
        assert total_large >= total_small - 1


class TestAnalyze:
    """Tests for the full analyze() method."""

    def test_returns_all_keys(self, linear_sim):
        """Analyze should return all expected top-level keys."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=20, seed=42)

        assert "fragmentation_stats" in result
        assert "perturbation_sensitivity" in result
        assert "fragmentation_output_correlation" in result
        assert "hazard_zones" in result
        assert "token_edit_vs_string_edit" in result
        assert "n_samples" in result
        assert "n_sims" in result

    def test_fragmentation_stats_keys(self, linear_sim):
        """Fragmentation stats should have all expected sub-keys."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=20, seed=42)
        stats = result["fragmentation_stats"]

        assert "mean_tokens_per_param" in stats
        assert "mean_tokens_per_digit" in stats
        assert "std_tokens_per_param" in stats
        assert "max_fragmentation_param" in stats

    def test_n_samples_matches(self, linear_sim):
        """n_samples in result should match input."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=30, seed=42)
        assert result["n_samples"] == 30

    def test_n_sims_at_least_n_samples(self, linear_sim):
        """Should run at least n_samples simulations."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=25, seed=42)
        assert result["n_sims"] >= 25

    def test_hazard_zones_sorted(self, linear_sim):
        """Hazard zones should be sorted by fragmentation descending."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=50, seed=42)
        zones = result["hazard_zones"]

        if len(zones) >= 2:
            for i in range(len(zones) - 1):
                assert zones[i]["fragmentation"] >= zones[i + 1]["fragmentation"], (
                    f"Hazard zones not sorted: {zones[i]['fragmentation']} < "
                    f"{zones[i + 1]['fragmentation']}"
                )

    def test_hazard_zones_max_ten(self, linear_sim):
        """Hazard zones should have at most 10 entries."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=50, seed=42)
        assert len(result["hazard_zones"]) <= 10

    def test_token_edit_vs_string_edit_keys(self, linear_sim):
        """Token edit vs string edit should have mean_ratio and max_ratio."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=20, seed=42)
        te = result["token_edit_vs_string_edit"]
        assert "mean_ratio" in te
        assert "max_ratio" in te
        assert te["mean_ratio"] > 0
        assert te["max_ratio"] > 0

    def test_deterministic_with_seed(self, linear_sim):
        """Same seed should produce identical results."""
        wb = TokenExtispicyWorkbench(linear_sim)
        r1 = wb.analyze(n_samples=20, seed=42)
        r2 = wb.analyze(n_samples=20, seed=42)
        assert r1["fragmentation_stats"] == r2["fragmentation_stats"]
        assert r1["n_sims"] == r2["n_sims"]

    def test_with_quadratic_sim(self, quadratic_sim):
        """Should work with a different simulator."""
        wb = TokenExtispicyWorkbench(quadratic_sim)
        result = wb.analyze(n_samples=20, seed=42)
        assert result["n_samples"] == 20
        # Quadratic sim has "y" and "fitness" outputs
        assert "y" in result["fragmentation_output_correlation"]


class TestCustomTokenizer:
    """Tests with a custom tokenizer callback."""

    def test_custom_tokenizer_used(self, linear_sim):
        """Custom tokenizer should be called instead of default."""
        call_count = {"n": 0}

        def counting_tokenizer(text):
            call_count["n"] += 1
            return text.split()

        wb = TokenExtispicyWorkbench(linear_sim, tokenize=counting_tokenizer)
        wb.analyze(n_samples=5, seed=42)
        assert call_count["n"] > 0

    def test_custom_tokenizer_affects_results(self, linear_sim):
        """Different tokenizers should produce different fragmentation stats."""
        # Aggressive tokenizer: one token per character
        def char_tokenizer(text):
            return list(text)

        # Coarse tokenizer: one token per word
        def word_tokenizer(text):
            return text.split()

        wb_char = TokenExtispicyWorkbench(linear_sim, tokenize=char_tokenizer)
        wb_word = TokenExtispicyWorkbench(linear_sim, tokenize=word_tokenizer)

        r_char = wb_char.analyze(n_samples=10, seed=42)
        r_word = wb_word.analyze(n_samples=10, seed=42)

        # Character tokenizer should show higher fragmentation
        assert (
            r_char["fragmentation_stats"]["mean_tokens_per_param"]
            != r_word["fragmentation_stats"]["mean_tokens_per_param"]
        )


class TestSimulatorProtocol:
    """Tests that TokenExtispicyWorkbench satisfies the Simulator protocol."""

    def test_run_delegates(self, linear_sim):
        """run() should return the simulator's output plus fragmentation fields."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        result = wb.run(params)

        # Should have original output
        assert "y" in result
        # Should have fragmentation annotations
        assert "_fragmentation_rate" in result
        assert "_token_count" in result

    def test_run_output_matches_underlying(self, linear_sim):
        """run() output for original keys should match simulator directly."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.3, "x2": 0.7, "x3": 0.1}
        direct = linear_sim.run(params)
        wrapped = wb.run(params)

        assert direct["y"] == pytest.approx(wrapped["y"])

    def test_param_spec_delegates(self, linear_sim):
        """param_spec() should match underlying simulator."""
        wb = TokenExtispicyWorkbench(linear_sim)
        assert wb.param_spec() == linear_sim.param_spec()

    def test_fragmentation_rate_positive(self, linear_sim):
        """_fragmentation_rate in run output should be positive."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        result = wb.run(params)
        assert result["_fragmentation_rate"] > 0

    def test_token_count_positive(self, linear_sim):
        """_token_count in run output should be positive."""
        wb = TokenExtispicyWorkbench(linear_sim)
        params = {"x0": 0.5, "x1": 0.5, "x2": 0.5, "x3": 0.5}
        result = wb.run(params)
        assert result["_token_count"] > 0


class TestAnalyzeWithBaseParams:
    """Tests for analyze() with explicit base_params."""

    def test_custom_base_params(self, linear_sim):
        """analyze() should accept custom base_params without error."""
        wb = TokenExtispicyWorkbench(linear_sim)
        base = {"x0": 0.1, "x1": 0.9, "x2": 0.5, "x3": 0.3}
        result = wb.analyze(base_params=base, n_samples=10, seed=42)
        assert result["n_samples"] == 10

    def test_default_base_params(self, linear_sim):
        """analyze() with no base_params should use midpoints."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=10, seed=42)
        # Should not raise, and perturbation sensitivity should be computed
        assert "perturbation_sensitivity" in result
        for name in linear_sim.param_spec():
            assert name in result["perturbation_sensitivity"]


class TestFragmentationOutputCorrelation:
    """Tests for fragmentation-output correlation computation."""

    def test_correlation_keys_match_outputs(self, quadratic_sim):
        """Correlation dict should have entries for each numeric output key."""
        wb = TokenExtispicyWorkbench(quadratic_sim)
        result = wb.analyze(n_samples=30, seed=42)
        corr = result["fragmentation_output_correlation"]

        # QuadraticSimulator returns "y" and "fitness"
        assert "y" in corr
        assert "fitness" in corr

    def test_correlation_values_bounded(self, linear_sim):
        """Correlation values should be in [-1, 1]."""
        wb = TokenExtispicyWorkbench(linear_sim)
        result = wb.analyze(n_samples=50, seed=42)
        for key, val in result["fragmentation_output_correlation"].items():
            assert -1.0 <= val <= 1.0, (
                f"Correlation for {key} out of bounds: {val}"
            )
