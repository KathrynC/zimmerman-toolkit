"""Tests for supradiegetic benchmark (zimmerman.supradiegetic_benchmark).

Tests verify:
    1. Task generation produces valid paired tasks.
    2. Scoring returns values in [0, 1].
    3. Benchmark runner returns correct structure.
    4. Discretization maps values to expected labels.
    5. Narrativization and diegetic params are consistent.
    6. Failure mode tagging produces valid tags.
    7. Deterministic simulators produce zero-std results.
"""

import numpy as np
import pytest

from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark


class TestDiscretizeValue:
    """Tests for the discretize_value helper."""

    def test_low_end(self):
        """Value near low bound should map to 'very low'."""
        bench = SuperdiegeticBenchmark.__new__(SuperdiegeticBenchmark)
        bench._n_bins = 5
        bench._labels = ["very low", "low", "medium", "high", "very high"]
        assert bench.discretize_value(0.05, 0.0, 1.0, n_bins=5) == "very low"

    def test_high_end(self):
        """Value near high bound should map to 'very high'."""
        bench = SuperdiegeticBenchmark.__new__(SuperdiegeticBenchmark)
        bench._n_bins = 5
        bench._labels = ["very low", "low", "medium", "high", "very high"]
        assert bench.discretize_value(0.95, 0.0, 1.0, n_bins=5) == "very high"

    def test_midpoint(self):
        """Value at midpoint should map to 'medium'."""
        bench = SuperdiegeticBenchmark.__new__(SuperdiegeticBenchmark)
        bench._n_bins = 5
        bench._labels = ["very low", "low", "medium", "high", "very high"]
        assert bench.discretize_value(0.5, 0.0, 1.0, n_bins=5) == "medium"

    def test_exact_low_bound(self):
        """Value at exact low bound should map to 'very low'."""
        bench = SuperdiegeticBenchmark.__new__(SuperdiegeticBenchmark)
        bench._n_bins = 5
        bench._labels = ["very low", "low", "medium", "high", "very high"]
        assert bench.discretize_value(0.0, 0.0, 1.0, n_bins=5) == "very low"

    def test_exact_high_bound(self):
        """Value at exact high bound should map to 'very high'."""
        bench = SuperdiegeticBenchmark.__new__(SuperdiegeticBenchmark)
        bench._n_bins = 5
        bench._labels = ["very low", "low", "medium", "high", "very high"]
        assert bench.discretize_value(1.0, 0.0, 1.0, n_bins=5) == "very high"


class TestNarrativizeParams:
    """Tests for narrativize_params."""

    def test_all_params_narrativized(self, linear_sim):
        """All parameters should be present in the narrative."""
        bench = SuperdiegeticBenchmark(linear_sim)
        params = {f"x{i}": 0.5 for i in range(4)}
        narrative = bench.narrativize_params(params)

        for name in params:
            assert name in narrative
            assert isinstance(narrative[name], str)

    def test_narrativized_are_labels(self, linear_sim):
        """All narrative values should be valid bin labels."""
        bench = SuperdiegeticBenchmark(linear_sim)
        valid_labels = {"very low", "low", "medium", "high", "very high"}
        rng = np.random.default_rng(42)

        for _ in range(20):
            params = {f"x{i}": float(rng.uniform(0, 1)) for i in range(4)}
            narrative = bench.narrativize_params(params)
            for name, label in narrative.items():
                assert label in valid_labels, (
                    f"Label '{label}' for {name} not in valid labels"
                )


class TestGenerateTasks:
    """Tests for task generation."""

    def test_default_generates_all_categories(self, linear_sim):
        """Default generation should produce one task per category."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)

        categories = {t["category"] for t in tasks}
        assert "palindrome" in categories
        assert "table" in categories
        assert "digits" in categories
        assert "format" in categories
        assert "symbol" in categories

    def test_task_has_required_fields(self, linear_sim):
        """Each task should have all required fields."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)

        for task in tasks:
            assert "task_id" in task
            assert "category" in task
            assert "supradiegetic_params" in task
            assert "diegetic_params" in task
            assert "expected" in task
            assert "scoring_fn" in task

    def test_supradiegetic_params_within_bounds(self, linear_sim):
        """Supradiegetic params should be within the simulator's bounds."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)
        spec = linear_sim.param_spec()

        for task in tasks:
            for name, val in task["supradiegetic_params"].items():
                lo, hi = spec[name]
                assert lo <= val <= hi, (
                    f"Param {name}={val} out of bounds [{lo}, {hi}] "
                    f"in task {task['task_id']}"
                )

    def test_diegetic_params_within_bounds(self, linear_sim):
        """Diegetic params (midpoints) should be within bounds."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)
        spec = linear_sim.param_spec()

        for task in tasks:
            for name, val in task["diegetic_params"].items():
                lo, hi = spec[name]
                assert lo <= val <= hi, (
                    f"Diegetic param {name}={val} out of bounds [{lo}, {hi}] "
                    f"in task {task['task_id']}"
                )

    def test_selected_categories(self, linear_sim):
        """Specifying categories should only generate those categories."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(categories=["palindrome", "digits"], seed=42)

        categories = {t["category"] for t in tasks}
        assert categories == {"palindrome", "digits"}

    def test_base_params_appended(self, linear_sim):
        """Providing base_params should add a 'baseline' task."""
        bench = SuperdiegeticBenchmark(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        tasks = bench.generate_tasks(base_params=base, seed=42)

        baseline_tasks = [t for t in tasks if t["category"] == "baseline"]
        assert len(baseline_tasks) == 1
        assert baseline_tasks[0]["supradiegetic_params"] == base

    def test_invalid_category_raises(self, linear_sim):
        """An unknown category name should raise ValueError."""
        bench = SuperdiegeticBenchmark(linear_sim)
        with pytest.raises(ValueError, match="Unknown category"):
            bench.generate_tasks(categories=["nonexistent"])

    def test_seed_reproducibility(self, linear_sim):
        """Same seed should produce identical tasks."""
        bench = SuperdiegeticBenchmark(linear_sim)
        t1 = bench.generate_tasks(seed=99)
        t2 = bench.generate_tasks(seed=99)

        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a["task_id"] == b["task_id"]
            for name in a["supradiegetic_params"]:
                assert a["supradiegetic_params"][name] == pytest.approx(
                    b["supradiegetic_params"][name]
                )

    def test_expected_is_simulator_output(self, linear_sim):
        """The expected field should match running the simulator with supradiegetic params."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)

        for task in tasks:
            result = linear_sim.run(task["supradiegetic_params"])
            for key in result:
                assert task["expected"][key] == pytest.approx(result[key])


class TestScoreTask:
    """Tests for task scoring."""

    def test_perfect_score(self, linear_sim):
        """Running supradiegetic params should score 1.0 against expected."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)

        for task in tasks:
            result = linear_sim.run(task["supradiegetic_params"])
            score = bench.score_task(task, result)
            assert score == pytest.approx(1.0, abs=1e-9), (
                f"Task {task['task_id']} should score 1.0 for exact params"
            )

    def test_diegetic_score_less_than_or_equal_one(self, linear_sim):
        """Diegetic params should score <= 1.0."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)

        for task in tasks:
            result = linear_sim.run(task["diegetic_params"])
            score = bench.score_task(task, result)
            assert 0.0 <= score <= 1.0 + 1e-9

    def test_empty_result_scores_zero(self, linear_sim):
        """An empty result dict should score 0.0."""
        bench = SuperdiegeticBenchmark(linear_sim)
        task = bench.generate_tasks(seed=42)[0]
        score = bench.score_task(task, {})

        assert score == pytest.approx(0.0)

    def test_nan_result_scores_zero(self, linear_sim):
        """A result with NaN values should score 0.0 for those keys."""
        bench = SuperdiegeticBenchmark(linear_sim)
        task = bench.generate_tasks(seed=42)[0]
        score = bench.score_task(task, {"y": float("nan")})

        assert score == pytest.approx(0.0)


class TestRunBenchmark:
    """Tests for the full benchmark runner."""

    def test_returns_correct_structure(self, linear_sim):
        """Benchmark result should have tasks, summary, failure_mode_tags, n_sims."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        assert "tasks" in report
        assert "summary" in report
        assert "failure_mode_tags" in report
        assert "n_sims" in report

    def test_task_results_have_fields(self, linear_sim):
        """Each task result should have all expected fields."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        for tr in report["tasks"]:
            assert "task_id" in tr
            assert "category" in tr
            assert "supradiegetic_score" in tr
            assert "diegetic_score" in tr
            assert "gain" in tr
            assert "supradiegetic_std" in tr
            assert "diegetic_std" in tr

    def test_summary_has_fields(self, linear_sim):
        """Summary should contain mean scores, gain, and by_category."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        summary = report["summary"]
        assert "mean_supradiegetic_score" in summary
        assert "mean_diegetic_score" in summary
        assert "mean_gain" in summary
        assert "by_category" in summary

    def test_n_sims_correct(self, linear_sim):
        """n_sims should equal 2 * n_tasks * n_reps."""
        bench = SuperdiegeticBenchmark(linear_sim)
        tasks = bench.generate_tasks(seed=42)
        n_reps = 3
        report = bench.run_benchmark(tasks=tasks, n_reps=n_reps, seed=42)

        expected_sims = 2 * len(tasks) * n_reps
        assert report["n_sims"] == expected_sims

    def test_supradiegetic_score_is_one_for_deterministic(self, linear_sim):
        """For a deterministic sim, supradiegetic score should be exactly 1.0."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=3, seed=42)

        for tr in report["tasks"]:
            assert tr["supradiegetic_score"] == pytest.approx(1.0, abs=1e-9), (
                f"Task {tr['task_id']} supradiegetic score should be 1.0"
            )

    def test_deterministic_sim_zero_std(self, linear_sim):
        """For a deterministic sim, std across reps should be 0."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=5, seed=42)

        for tr in report["tasks"]:
            assert tr["supradiegetic_std"] == pytest.approx(0.0, abs=1e-12)
            assert tr["diegetic_std"] == pytest.approx(0.0, abs=1e-12)

    def test_gain_is_difference(self, linear_sim):
        """Gain should equal diegetic_score - supradiegetic_score."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        for tr in report["tasks"]:
            expected_gain = tr["diegetic_score"] - tr["supradiegetic_score"]
            assert tr["gain"] == pytest.approx(expected_gain, abs=1e-12)

    def test_by_category_present(self, linear_sim):
        """by_category should have entries for all generated categories."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        by_cat = report["summary"]["by_category"]
        for tr in report["tasks"]:
            assert tr["category"] in by_cat

    def test_failure_mode_tags_are_strings(self, linear_sim):
        """Failure mode tags should be a list of strings."""
        bench = SuperdiegeticBenchmark(linear_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        assert isinstance(report["failure_mode_tags"], list)
        for tag in report["failure_mode_tags"]:
            assert isinstance(tag, str)


class TestWithQuadraticSimulator:
    """Tests using the quadratic simulator to verify non-linear behavior."""

    def test_quadratic_sim_runs(self, quadratic_sim):
        """Benchmark should work with the quadratic simulator."""
        bench = SuperdiegeticBenchmark(quadratic_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        assert report["n_sims"] > 0
        assert len(report["tasks"]) > 0

    def test_diegetic_gain_can_be_nonzero(self, quadratic_sim):
        """For non-linear sims, diegetic gain can be non-zero."""
        bench = SuperdiegeticBenchmark(quadratic_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        # At least some tasks should show a non-zero gain
        # (diegetic uses midpoints which may differ from exact values).
        gains = [tr["gain"] for tr in report["tasks"]]
        # We just verify the structure is valid; the gain direction
        # depends on the specific tasks and simulator.
        assert all(isinstance(g, float) for g in gains)


class TestWithStepSimulator:
    """Tests using the step simulator to verify boundary behavior."""

    def test_step_sim_runs(self, step_sim):
        """Benchmark should work with the step simulator."""
        bench = SuperdiegeticBenchmark(step_sim)
        report = bench.run_benchmark(n_reps=2, seed=42)

        assert report["n_sims"] > 0
        assert len(report["tasks"]) > 0
