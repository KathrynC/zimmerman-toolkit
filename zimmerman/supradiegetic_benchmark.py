"""Supradiegetic benchmark: standardized form-vs-meaning battery for LLM-mediated simulators.

Generates paired simulation tasks: one supradiegetic (exact numeric
parameters) and one diegetic (same content encoded as narrative bin labels).
Measures the "diegeticization gain" -- how much simulation outcomes improve
when parameters are presented narratively rather than as raw numbers.

This implements a key prediction from Zimmerman (2025) Ch. 5 and SS2.2.3:
LLMs handle diegetic content (meaning, semantics) more reliably than
supradiegetic content (form, exact numbers). The supradiegetic/diegetic
distinction originates in narratology (Genette, 1972): "diegetic" elements
exist within the story world (characters, events, descriptions), while
"supradiegetic" elements exist outside the narrative frame (page numbers,
formatting, table structure, precise numeric tokens). LLMs are trained
predominantly on diegetic content -- natural language expressing meaning --
and consequently struggle with supradiegetic tasks that require exact
formal/structural manipulation rather than semantic understanding.

Zimmerman SS2.2.3 describes "distributional collapse": when an LLM
encounters unfamiliar structural tasks (e.g., reversing digit sequences,
preserving exact numeric precision), it defaults to high-probability outputs
from its training distribution rather than performing the requested formal
operation. This manifests as rounding to "nice" numbers, reordering elements
into more common patterns, or silently dropping precision -- all symptoms of
the model retreating to its distributional prior when the task lacks
narrative scaffolding.

Task categories (5 benchmark probes):
    palindrome  -- Symmetric parameter vectors (e.g., [0.1, 0.5, 0.5, 0.1]).
                   Tests whether the model preserves mirror structure -- a
                   purely formal property with no semantic correlate.
    table       -- Parameters that form a structured evenly-spaced pattern.
                   Tests extraction and preservation of tabular layout, a
                   supradiegetic structure type.
    digits      -- Parameters with many significant digits (6 decimal places).
                   Tests numerical precision retention, the most fragile
                   supradiegetic capability due to tokenization effects.
    format      -- Parameters at exact bin boundaries (e.g., 0.20001 vs 0.19999).
                   Tests boundary sensitivity -- whether the model correctly
                   handles values at discretization edges.
    symbol      -- Parameters encoding symbolic relationships (ratios, sums).
                   Tests whether non-linguistic token relationships (e.g.,
                   param1 = 2 * param0) survive the processing pipeline.

Each category probes a different aspect of form sensitivity. The
supradiegetic version uses exact floats; the diegetic version discretizes
them into narrative bins ("very low", "low", ..., "very high"). The
simulator is run with both, and the score comparison reveals whether
narrative encoding helps, hurts, or is neutral.

Failure mode tagging classifies observed degradation patterns:
    off_by_one       -- Score in (0.5, 0.95): close but not matching,
                        suggesting the model approximately understood the task
                        but lost precision in execution.
    format_drift     -- Diegetic score exceeds supradiegetic by >0.3: the model
                        performs substantially better with narrative framing,
                        indicating strong dependence on diegetic scaffolding.
    boundary_confusion -- Format-category task with supradiegetic score <0.8:
                        the model mishandles bin-boundary values, conflating
                        adjacent categories at discretization edges.
    precision_loss   -- Digits-category task where diegetic score < supradiegetic
                        by >0.2: discretization into bins actively destroys
                        information that the model could otherwise use.

These failure signatures map directly to Zimmerman's taxonomy of LLM
structural failure modes and provide diagnostic evidence for which aspects
of supradiegetic processing are weakest in a given model.

Simulator protocol satisfaction: this benchmark class itself satisfies the
Zimmerman toolkit's Simulator protocol by wrapping any simulator that
implements ``run(params) -> dict`` and ``param_spec() -> bounds``. This
enables recursive analysis -- the benchmark can be interrogated by other
Zimmerman tools (Sobol sensitivity, falsifier, contrastive pairs, POSIWID)
to study the structure of the diegeticization gain surface itself.

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont.

    Genette, G. (1972). "Narrative Discourse: An Essay in Method."
    (Trans. J.E. Lewin, 1980). Cornell University Press.
"""

from __future__ import annotations

import numpy as np


# Bin labels used for the 5-level discretization scheme.
# These must match the convention in diegeticizer.py so that the two modules
# produce consistent narrative representations.  The 5-bin scheme is the
# default because it balances granularity (enough bins to capture variation)
# against narrative stability (few enough that each label carries clear
# semantic weight for an LLM).
_BIN_LABELS_5 = ["very low", "low", "medium", "high", "very high"]


class SuperdiegeticBenchmark:
    """Standardized form-vs-meaning benchmark for a black-box simulator.

    Wraps a simulator to test how simulation outcomes differ when the
    same parameter content is presented as exact numeric values
    (supradiegetic) versus narrative bin labels that are then converted
    back to midpoint values (diegetic).

    The core metric is the "diegeticization gain": for each task,

        gain = diegetic_score - supradiegetic_score

    A positive gain means narrative framing improves the simulation outcome
    relative to raw numeric input. Zimmerman (2025) predicts this should be
    positive for LLM-mediated parameter pipelines, because the LLM's
    distributional semantics operates more reliably on words like "high" or
    "very low" than on tokens like "0.7832".

    For deterministic simulators (identical inputs always produce identical
    outputs), the supradiegetic score is always 1.0, and the diegetic score
    reflects how much information was lost by discretization. The gain is
    thus negative or zero.  The benchmark becomes diagnostic in the presence
    of an LLM in the pipeline (e.g., when the simulator's parameters are
    generated or interpreted by a language model).

    This class satisfies the Simulator protocol (run + param_spec), enabling
    recursive self-analysis with other Zimmerman toolkit modules.

    Args:
        simulator: Any Simulator-compatible object with run() and
            param_spec() methods.

    Example:
        bench = SuperdiegeticBenchmark(my_sim)
        tasks = bench.generate_tasks(seed=42)
        report = bench.run_benchmark(tasks, n_reps=5)
        print(report["summary"]["mean_gain"])
    """

    def __init__(self, simulator):
        self.simulator = simulator
        # Cache the parameter specification for repeated use in task generation
        # and scoring.  The spec maps parameter names to (low, high) bounds.
        self._spec = simulator.param_spec()
        self._param_names = list(self._spec.keys())
        self._n_bins = 5
        self._labels = list(_BIN_LABELS_5)

    # ------------------------------------------------------------------
    # Discretization helpers
    # ------------------------------------------------------------------

    def discretize_value(
        self,
        value: float,
        low: float,
        high: float,
        n_bins: int = 5,
    ) -> str:
        """Convert a numeric value to a narrative bin label.

        Divides [low, high] into *n_bins* equal-width bins and returns the
        label for the bin containing *value*.  This is the core
        supradiegetic-to-diegetic translation: replacing a precise numeric
        token with a semantically rich narrative label.

        The equal-width binning strategy is deliberate.  Zimmerman SS3.5.3
        notes that equal-width bins produce maximal narrative contrast
        between adjacent labels, even though equal-frequency bins would be
        more statistically efficient.  The goal is not optimal information
        retention but maximal semantic discriminability for the LLM.

        Uses ``np.searchsorted`` with ``side='right'`` so that bin edges
        are assigned to the lower bin (left-closed, right-open intervals),
        except for the final bin which is closed on both sides.

        Args:
            value: The numeric value to discretize.
            low: Lower bound of the parameter range.
            high: Upper bound of the parameter range.
            n_bins: Number of bins. Default 5.

        Returns:
            Bin label string (e.g., "medium").
        """
        if n_bins == 5:
            labels = _BIN_LABELS_5
        else:
            # Fallback to generic numbered labels when a non-standard bin
            # count is requested.  These lack semantic richness but
            # maintain the structural protocol.
            labels = [f"bin_{i}" for i in range(n_bins)]

        # Compute n_bins + 1 edges spanning [low, high].
        edges = np.linspace(low, high, n_bins + 1)
        # searchsorted on interior edges: finds which bin the value falls into.
        # side='right' means values exactly on an edge go to the lower bin.
        idx = int(np.searchsorted(edges[1:], value, side="right"))
        # Clamp to valid bin range to handle floating-point edge cases where
        # value == high (would otherwise index past the last bin).
        idx = int(np.clip(idx, 0, n_bins - 1))
        return labels[idx]

    def narrativize_params(self, params: dict, param_spec: dict | None = None) -> dict:
        """Convert all params from numeric to narrative bin labels.

        This is the full diegeticization pass: every parameter in the dict
        is translated from its exact numeric value to a narrative label.
        Parameters not found in param_spec are converted to their string
        representation as a fallback (preserving supradiegetic form when no
        narrative mapping is available).

        Args:
            params: Dict mapping parameter names to float values.
            param_spec: Parameter specification. If None, uses the
                simulator's param_spec.

        Returns:
            Dict mapping parameter names to narrative label strings.
        """
        if param_spec is None:
            param_spec = self._spec
        narrative = {}
        for name, value in params.items():
            if name in param_spec:
                lo, hi = param_spec[name]
                # Translate: numeric value -> narrative bin label.
                narrative[name] = self.discretize_value(float(value), lo, hi, self._n_bins)
            else:
                # Unknown parameter: fall back to string representation.
                # This preserves the supradiegetic form directly, which is
                # the conservative choice when no range information exists.
                narrative[name] = str(value)
        return narrative

    def _midpoint_from_label(self, label: str, low: float, high: float) -> float:
        """Return the bin midpoint for a given narrative label.

        The midpoint is the deterministic inverse of discretization: it maps
        each label back to the single most representative value for that bin.
        This introduces quantization error bounded by half the bin width --
        the fundamental information loss of diegeticization (see
        ``Diegeticizer.roundtrip_error()`` for systematic measurement).

        Args:
            label: Narrative bin label (e.g., "high").
            low: Lower bound of the parameter range.
            high: Upper bound of the parameter range.

        Returns:
            Midpoint of the corresponding bin.
        """
        idx = self._labels.index(label)
        edges = np.linspace(low, high, self._n_bins + 1)
        # Midpoint = average of the bin's left and right edges.
        return float((edges[idx] + edges[idx + 1]) / 2.0)

    def _diegetic_params_from_narrative(self, narrative: dict) -> dict:
        """Convert narrative labels back to numeric midpoint values.

        This is the re-diegeticization step: translating the narrative
        representation back into numbers for the simulator.  The recovered
        values are bin midpoints, so they are "snapped" to a discrete grid.
        The difference between the original exact values and these midpoints
        is the diegeticization error -- the cost of translating through
        narrative space.

        Args:
            narrative: Dict mapping parameter names to bin labels.

        Returns:
            Dict mapping parameter names to bin midpoint floats.
        """
        recovered = {}
        for name, label in narrative.items():
            if name in self._spec:
                lo, hi = self._spec[name]
                recovered[name] = self._midpoint_from_label(label, lo, hi)
            else:
                # Unknown parameter: default to 0.0 as a safe neutral value.
                recovered[name] = 0.0
        return recovered

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def _generate_palindrome_task(self, rng: np.random.Generator, task_idx: int) -> dict:
        """Generate a palindrome task: symmetric parameter vector.

        Palindrome tasks probe whether the pipeline preserves mirror
        symmetry -- a purely structural (supradiegetic) property.  The
        parameter vector is constructed so that x[i] == x[d-1-i].  An LLM
        in the pipeline might break this symmetry by independently
        processing each parameter token, since "0.3 appearing first" and
        "0.3 appearing last" are distributionally identical to the model
        but structurally distinct.

        This is one of the clearest tests of Zimmerman's claim that LLMs
        optimize for meaning (diegetic) over form (supradiegetic): mirror
        structure has no semantic content, only formal structure.
        """
        d = len(self._param_names)
        half = d // 2
        # Generate the first half of the parameter vector randomly.
        half_values = [float(rng.uniform(*self._spec[self._param_names[i]])) for i in range(half)]
        # Mirror: first half forward, second half reversed.
        # If odd dimensionality, insert a random middle value.
        if d % 2 == 1:
            mid_val = float(rng.uniform(*self._spec[self._param_names[half]]))
            values = half_values + [mid_val] + list(reversed(half_values))
        else:
            values = half_values + list(reversed(half_values))
        # Clip to per-parameter bounds.  The reversed half may need different
        # bounds than the first half (parameter ranges are not necessarily
        # identical), so clipping ensures validity.
        params = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self._spec[name]
            params[name] = float(np.clip(values[i], lo, hi))
        return self._make_task(params, "palindrome", task_idx)

    def _generate_table_task(self, rng: np.random.Generator, task_idx: int) -> dict:
        """Generate a table task: structured evenly-spaced pattern.

        Table tasks probe whether the pipeline preserves tabular/grid
        structure.  Parameters are set to evenly-spaced fractions of their
        ranges, creating a regular staircase pattern.  This is a
        supradiegetic layout property: the "tableness" of the data exists
        in its structure, not in any individual parameter's meaning.

        Zimmerman Ch. 5 argues that LLMs process table content through
        narrative linearization, losing the 2D relational structure.  This
        task quantifies that loss.
        """
        d = len(self._param_names)
        params = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self._spec[name]
            # Place parameter i at fraction (i+1)/(d+1) of its range,
            # creating an evenly-spaced ascending pattern.
            frac = (i + 1) / (d + 1)
            params[name] = float(lo + frac * (hi - lo))
        return self._make_task(params, "table", task_idx)

    def _generate_digits_task(self, rng: np.random.Generator, task_idx: int) -> dict:
        """Generate a digits task: parameters with many significant digits.

        Digits tasks are the most direct probe of numerical precision --
        the supradiegetic capability most vulnerable to tokenization effects.
        Values are rounded to 6 decimal places (e.g., 0.371284), creating
        "difficult" precise numbers that stress the token-level
        representation.

        Zimmerman SS2.2.3 shows that LLM tokenizers split numbers into
        digit-level tokens (e.g., "0", ".", "3", "7", "1", "2", "8", "4"),
        destroying the positional encoding that gives digits their
        magnitude.  The model must reconstruct numeric meaning from a
        sequence of character tokens -- a fundamentally supradiegetic task.
        Distributional collapse manifests here as rounding to "common"
        numbers (0.5, 0.25, etc.) or truncating trailing digits.
        """
        params = {}
        for name in self._param_names:
            lo, hi = self._spec[name]
            # Generate a value with many significant digits.
            raw = rng.uniform(lo, hi)
            # Round to 6 decimal places to create precise but not
            # pathologically long numbers.
            params[name] = float(np.round(raw, 6))
        return self._make_task(params, "digits", task_idx)

    def _generate_format_task(self, rng: np.random.Generator, task_idx: int) -> dict:
        """Generate a format task: values at exact bin boundaries.

        Format tasks probe boundary sensitivity by placing values at (or
        infinitesimally near) the edges between discretization bins.  A
        value of 0.200001 should fall in the "low" bin, while 0.199999
        should fall in "very low" -- but the difference is 2e-6, far below
        the precision that LLM tokenization typically preserves.

        This tests the "boundary_confusion" failure mode: the model may
        collapse both values to the same bin, or inconsistently assign them
        across repeated runs.  Boundary confusion is a specific instance of
        Zimmerman's distributional collapse (SS2.2.3), where the model's
        uncertainty about exact numeric values causes it to default to the
        higher-probability bin.
        """
        params = {}
        for name in self._param_names:
            lo, hi = self._spec[name]
            edges = np.linspace(lo, hi, self._n_bins + 1)
            # Pick a random interior edge (boundary between two bins).
            edge_idx = rng.integers(1, len(edges) - 1)
            # Offset by +/- 1e-6 to place the value just on one side of
            # the boundary.  This tiny offset is the crux of the test:
            # can the pipeline distinguish values separated by 2e-6?
            offset = rng.choice([-1e-6, 1e-6])
            params[name] = float(np.clip(edges[edge_idx] + offset, lo, hi))
        return self._make_task(params, "format", task_idx)

    def _generate_symbol_task(self, rng: np.random.Generator, task_idx: int) -> dict:
        """Generate a symbol task: parameters encoding a ratio relationship.

        Symbol tasks probe whether non-linguistic token relationships
        survive the pipeline.  The first two parameters are set so that
        param1 = 2 * param0 (clipped to bounds).  This ratio is a
        mathematical relationship -- a supradiegetic structure that exists
        in the formal domain, not in narrative meaning.

        An LLM processing these parameters independently will not "see"
        the 2:1 ratio unless it has been explicitly told about it in
        narrative form.  This is a direct test of Zimmerman's claim that
        LLMs lack compositional understanding of formal relationships
        between tokens.
        """
        d = len(self._param_names)
        params = {}
        if d >= 2:
            # Set the first two params in a 2:1 ratio.
            lo0, hi0 = self._spec[self._param_names[0]]
            lo1, hi1 = self._spec[self._param_names[1]]
            val0 = float(rng.uniform(lo0, hi0))
            # param1 = 2 * param0, clipped to its own bounds.
            val1 = float(np.clip(2.0 * val0, lo1, hi1))
            params[self._param_names[0]] = val0
            params[self._param_names[1]] = val1
        # Fill remaining parameters with random values (no special structure).
        for i in range(min(2, d), d):
            name = self._param_names[i]
            lo, hi = self._spec[name]
            params[name] = float(rng.uniform(lo, hi))
        return self._make_task(params, "symbol", task_idx)

    def _make_task(self, supradiegetic_params: dict, category: str, task_idx: int) -> dict:
        """Construct a paired task dict from a set of supradiegetic (exact) params.

        Each task contains both representations of the same parameter content:
        1. supradiegetic_params: the exact numeric values (form-level).
        2. diegetic_params: the same values discretized to narrative labels
           and then recovered as bin midpoints (meaning-level).

        The "expected" output is computed by running the simulator with the
        exact supradiegetic params.  Scoring compares any result against this
        ground truth, so:
        - Supradiegetic runs should score ~1.0 (same params as expected).
        - Diegetic runs score lower to the extent that discretization
          changes the output (the diegeticization cost).

        Args:
            supradiegetic_params: Exact numeric parameter values.
            category: Task category string (palindrome, table, etc.).
            task_idx: Integer index for task_id.

        Returns:
            Task dict with supradiegetic_params, diegetic_params, expected
            output, and scoring function name.
        """
        # Forward pass: exact values -> narrative labels.
        narrative = self.narrativize_params(supradiegetic_params)
        # Reverse pass: narrative labels -> bin midpoint values.
        diegetic_params = self._diegetic_params_from_narrative(narrative)

        # Run the simulator with the supradiegetic (exact) params to establish
        # the ground-truth expected output for scoring.
        expected = self.simulator.run(supradiegetic_params)

        return {
            "task_id": f"{category}_{task_idx:03d}",
            "category": category,
            "supradiegetic_params": supradiegetic_params,
            "diegetic_params": diegetic_params,
            "expected": expected,
            "scoring_fn": "score_task",
        }

    # Registry mapping category names to their generator method names.
    # This dispatch pattern allows generate_tasks() to accept category
    # names as strings and dynamically resolve the appropriate generator.
    _CATEGORY_GENERATORS = {
        "palindrome": "_generate_palindrome_task",
        "table": "_generate_table_task",
        "digits": "_generate_digits_task",
        "format": "_generate_format_task",
        "symbol": "_generate_symbol_task",
    }

    def generate_tasks(
        self,
        base_params: dict | None = None,
        categories: list[str] | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Generate paired task instances across benchmark categories.

        Each task contains both a supradiegetic (exact numeric) and a
        diegetic (narrative bin label -> midpoint) version of the same
        parameter set.  The pairing is essential: it ensures that any
        score difference between the two versions is attributable solely
        to the form of representation (numeric vs narrative), not to
        different parameter content.

        The seed-based RNG ensures full reproducibility: the same seed
        always generates the same tasks, enabling longitudinal comparison
        across different models or pipeline configurations.

        Args:
            base_params: Optional base parameter dict. If provided, one
                additional task is generated using these exact values
                (category="baseline"). Useful for benchmarking a specific
                parameter configuration of interest alongside the
                standardized probes.
            categories: List of category names to generate. If None, all
                categories are used: palindrome, table, digits, format,
                symbol. See module docstring for category descriptions.
            seed: Random seed for reproducibility.

        Returns:
            List of task dicts. Each has keys: task_id, category,
            supradiegetic_params, diegetic_params, expected, scoring_fn.
        """
        rng = np.random.default_rng(seed)

        if categories is None:
            categories = list(self._CATEGORY_GENERATORS.keys())

        tasks = []
        task_idx = 0

        # Generate one task per requested category.
        for cat in categories:
            if cat not in self._CATEGORY_GENERATORS:
                raise ValueError(
                    f"Unknown category '{cat}'. "
                    f"Valid: {list(self._CATEGORY_GENERATORS.keys())}"
                )
            gen_method = getattr(self, self._CATEGORY_GENERATORS[cat])
            task = gen_method(rng, task_idx)
            tasks.append(task)
            task_idx += 1

        # Optional baseline task from user-specified base_params.
        # This bypasses the random generation to benchmark a specific
        # parameter configuration alongside the standardized probes.
        if base_params is not None:
            task = self._make_task(base_params, "baseline", task_idx)
            tasks.append(task)

        return tasks

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_task(self, task: dict, result: dict) -> float:
        """Score a simulation result against the task's expected output.

        Uses magnitude-match scoring: for each numeric key in expected,
        computes::

            score_i = max(0, 1 - |actual_i - expected_i| / scale_i)

        where ``scale_i = max(|expected_i|, 0.1)`` prevents division by
        zero and ensures small expected values are not over-penalized.
        The final score is the mean across all numeric keys.

        This scoring function is deliberately simple and domain-agnostic.
        It measures how close the simulation output is to the ground truth
        on a 0-to-1 scale, where 1.0 means perfect match and 0.0 means
        the result is off by at least 100% of the expected magnitude.

        Non-finite values (NaN, inf) in either expected or actual are
        scored as 0.0, which correctly penalizes pipeline failures that
        produce degenerate outputs -- a common symptom of distributional
        collapse (Zimmerman SS2.2.3).

        Args:
            task: Task dict (must contain "expected").
            result: Simulation result dict from simulator.run().

        Returns:
            Score between 0.0 and 1.0.
        """
        expected = task["expected"]
        scores = []

        for key, exp_val in expected.items():
            # Skip non-numeric keys (e.g., string status fields).
            if not isinstance(exp_val, (int, float, np.integer, np.floating)):
                continue
            exp_val = float(exp_val)
            # Non-finite expected values cannot be meaningfully scored.
            if not np.isfinite(exp_val):
                continue

            act_val = result.get(key)
            if act_val is None:
                # Missing key in result: score as complete failure.
                scores.append(0.0)
                continue
            if not isinstance(act_val, (int, float, np.integer, np.floating)):
                # Non-numeric actual value: type mismatch failure.
                scores.append(0.0)
                continue
            act_val = float(act_val)
            if not np.isfinite(act_val):
                # NaN or inf in result: degenerate output failure.
                scores.append(0.0)
                continue

            # Magnitude-match scoring with floor at 0.1 to prevent
            # division by zero for near-zero expected values.
            scale = max(abs(exp_val), 0.1)
            score = max(0.0, 1.0 - abs(act_val - exp_val) / scale)
            scores.append(score)

        if not scores:
            return 0.0
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Failure mode detection
    # ------------------------------------------------------------------

    def _detect_failure_modes(self, task: dict, sup_score: float, die_score: float) -> list[str]:
        """Detect failure mode tags for a single task.

        Tags characterize *how* the pipeline failed, not just whether it
        failed.  This diagnostic taxonomy is derived from Zimmerman (2025)
        Ch. 5's analysis of LLM structural failure patterns:

        - off_by_one: The model approximately solved the task but lost
          precision.  Supradiegetic score is in (0.5, 0.95), meaning the
          output is close but not matching -- analogous to being "one bin
          off" in the discretization.

        - format_drift: The model performs substantially better with
          narrative framing (diegetic score exceeds supradiegetic by >0.3).
          This is direct evidence of Zimmerman's prediction: the same
          information is more reliably processed when presented as meaning
          (diegetic) than as form (supradiegetic).

        - boundary_confusion: Format-category tasks with low supradiegetic
          scores (<0.8).  The model mishandles values at bin boundaries,
          suggesting it cannot reliably distinguish values separated by
          small numeric differences -- a tokenization-level limitation.

        - precision_loss: Digits-category tasks where discretization
          actively hurts (diegetic score < supradiegetic by >0.2).  The
          bin-midpoint approximation loses more information than the model
          gains from narrative framing, indicating that for high-precision
          tasks, the supradiegetic representation is actually superior.

        Args:
            task: The task dict.
            sup_score: Mean supradiegetic score.
            die_score: Mean diegetic score.

        Returns:
            List of failure mode tag strings.
        """
        tags = []

        # off_by_one: supradiegetic score is close but not matching.
        # The (0.5, 0.95) range captures "approximately correct" results
        # that lost precision in the formal manipulation.
        if 0.5 < sup_score < 0.95:
            tags.append("off_by_one")

        # format_drift: supradiegetic dramatically worse than diegetic.
        # A gap > 0.3 is strong evidence that narrative framing is
        # compensating for a structural processing deficit.
        if die_score - sup_score > 0.3:
            tags.append("format_drift")

        # boundary_confusion: format-category tasks with low scores.
        # These tasks specifically place values at bin edges, so low
        # scores indicate the model cannot handle boundary precision.
        if task["category"] == "format" and sup_score < 0.8:
            tags.append("boundary_confusion")

        # precision_loss: digits-category tasks where diegetic is much worse.
        # When discretization destroys more information than it adds in
        # narrative scaffolding, the diegetic score drops below supradiegetic.
        if task["category"] == "digits" and die_score < sup_score - 0.2:
            tags.append("precision_loss")

        return tags

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        tasks: list[dict] | None = None,
        n_reps: int = 5,
        seed: int = 42,
    ) -> dict:
        """Run both supradiegetic and diegetic versions of each task.

        For each task, runs the simulator n_reps times with the
        supradiegetic params and n_reps times with the diegetic params,
        scoring each run against the expected output.

        The repetition structure exists to support stochastic simulators
        or future extensions with noise injection.  For deterministic
        simulators (the common case), all reps produce identical scores,
        so std will be 0.0.  This is verified by the test suite
        (``test_deterministic_sim_zero_std``).

        The total number of simulator invocations is:
            n_sims = 2 * len(tasks) * n_reps
        (one supradiegetic and one diegetic run per task per repetition).

        The output report includes:
        - Per-task scores, gains, and standard deviations.
        - Per-category aggregate statistics.
        - The overall mean diegeticization gain (the headline metric).
        - Failure mode tags for diagnostic analysis.

        Args:
            tasks: List of task dicts from generate_tasks(). If None,
                generates default tasks with the given seed.
            n_reps: Number of repetitions per task version. Default 5.
            seed: Random seed (used for task generation if tasks is None).

        Returns:
            Dict with:
                "tasks": list of per-task result dicts,
                "summary": aggregate statistics including mean_gain
                    (the diegeticization gain averaged across all tasks),
                "failure_mode_tags": deduplicated list of detected failure
                    pattern strings (see _detect_failure_modes),
                "n_sims": total number of simulations run,
        """
        if tasks is None:
            tasks = self.generate_tasks(seed=seed)

        task_results = []
        all_sup_scores = []
        all_die_scores = []
        all_failure_tags = []
        by_category: dict[str, dict] = {}
        n_sims = 0

        for task in tasks:
            sup_scores_reps = []
            die_scores_reps = []

            for _ in range(n_reps):
                # --- Supradiegetic run ---
                # Uses exact numeric parameters (the "form" representation).
                sup_result = self.simulator.run(task["supradiegetic_params"])
                n_sims += 1
                sup_score = self.score_task(task, sup_result)
                sup_scores_reps.append(sup_score)

                # --- Diegetic run ---
                # Uses bin-midpoint parameters recovered from narrative labels
                # (the "meaning" representation).
                die_result = self.simulator.run(task["diegetic_params"])
                n_sims += 1
                die_score = self.score_task(task, die_result)
                die_scores_reps.append(die_score)

            mean_sup = float(np.mean(sup_scores_reps))
            mean_die = float(np.mean(die_scores_reps))
            # Diegeticization gain: positive means narrative framing helped.
            # This is the core metric of the benchmark (Zimmerman Ch. 5).
            gain = mean_die - mean_sup

            task_entry = {
                "task_id": task["task_id"],
                "category": task["category"],
                "supradiegetic_score": mean_sup,
                "diegetic_score": mean_die,
                "gain": gain,
                "supradiegetic_std": float(np.std(sup_scores_reps)),
                "diegetic_std": float(np.std(die_scores_reps)),
            }
            task_results.append(task_entry)

            all_sup_scores.append(mean_sup)
            all_die_scores.append(mean_die)

            # Failure mode detection: classify observed degradation patterns.
            tags = self._detect_failure_modes(task, mean_sup, mean_die)
            all_failure_tags.extend(tags)

            # Accumulate per-category statistics for the summary breakdown.
            cat = task["category"]
            if cat not in by_category:
                by_category[cat] = {"sup_scores": [], "die_scores": []}
            by_category[cat]["sup_scores"].append(mean_sup)
            by_category[cat]["die_scores"].append(mean_die)

        # Build per-category summary: mean scores and gain for each
        # benchmark category, enabling analysis of which structural
        # dimensions show the largest diegeticization effect.
        category_summary = {}
        for cat, data in by_category.items():
            cat_sup = float(np.mean(data["sup_scores"]))
            cat_die = float(np.mean(data["die_scores"]))
            category_summary[cat] = {
                "sup_score": cat_sup,
                "die_score": cat_die,
                "gain": cat_die - cat_sup,
            }

        summary = {
            "mean_supradiegetic_score": float(np.mean(all_sup_scores)) if all_sup_scores else 0.0,
            "mean_diegetic_score": float(np.mean(all_die_scores)) if all_die_scores else 0.0,
            # The headline metric: mean diegeticization gain across all tasks.
            # Positive gain = narrative framing improves outcomes.
            # Negative gain = exact numerics are superior (expected for
            # deterministic simulators without an LLM in the loop).
            "mean_gain": (
                float(np.mean(all_die_scores)) - float(np.mean(all_sup_scores))
                if all_sup_scores else 0.0
            ),
            "by_category": category_summary,
        }

        # Deduplicate failure tags while preserving detection order.
        # Order preservation matters for diagnostic reporting: the first
        # tag detected is often the most informative.
        seen_tags = set()
        unique_tags = []
        for tag in all_failure_tags:
            if tag not in seen_tags:
                seen_tags.add(tag)
                unique_tags.append(tag)

        return {
            "tasks": task_results,
            "summary": summary,
            "failure_mode_tags": unique_tags,
            "n_sims": n_sims,
        }
