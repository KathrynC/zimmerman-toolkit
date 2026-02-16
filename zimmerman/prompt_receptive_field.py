"""Prompt receptive field analysis: feature attribution over input segments.

This module implements "receptive field analysis" for black-box simulators,
borrowing the concept from neuroscience and applying it to parameterized
systems. Just as a neuron's receptive field defines the region of sensory
space that influences its firing (Hubel & Wiesel, 1962), a simulator's
"prompt receptive field" defines which segments of its input space
influence its output -- and how strongly.

The key insight is to apply Sobol global sensitivity analysis
(Saltelli, 2002; Saltelli et al., 2010) over *segment inclusion weights*
rather than the simulator's native numeric parameters. Each segment of
the input (a parameter group, a prompt section, a control block) gets a
continuous weight in [0, 1] that controls whether it is included (weight
>= 0.66), compressed (0.33 <= weight < 0.66), or dropped (weight < 0.33).
Saltelli sampling over these weights yields first-order (S1) and total-order
(ST) Sobol indices that rank which segments matter most and which interact.

Theoretical motivation (Zimmerman 2025)
---------------------------------------
Zimmerman §5.2 distinguishes between *supradiegetic* content (tables,
formatting, structural cues, system prompts) and *diegetic* content
(narrative, descriptions, the "story" of the input). These two types
of content may contribute very differently to a system's output. The
PromptReceptiveField makes this distinction empirically testable:
segment the input into supradiegetic vs. diegetic blocks and compare
their Sobol indices.

More broadly, this tool operationalizes the question "what parts of the
input actually matter?" -- which is central to Zimmerman's Chapter 3 on
relation and meaning construction. The first-order index S1 measures
the *main effect* of a segment: does it influence output on its own?
The interaction term (ST - S1) measures whether a segment's influence
depends on the presence of other segments -- i.e., whether it
participates in *relational* meaning construction (§3.1). A segment
with high ST but low S1 is one that "only matters in context" -- it
has no standalone effect but becomes important through interaction
with other segments. This is precisely Zimmerman's claim about
relational meaning: some tokens/parameters carry meaning only through
their relations to other tokens/parameters.

The three-level weight scheme (drop/compress/include) is designed to
capture both binary presence effects (drop vs. include) and graded
contribution effects (compress interpolates between the two). This is
more informative than a simple binary ablation study, which can only
detect whether a segment matters at all, not *how much* it matters
along a continuum. The compress region (0.33--0.66) provides a smooth
transition that helps Sobol analysis estimate derivatives rather than
only step functions.

Composability
-------------
The PromptReceptiveField satisfies the Simulator protocol (run() +
param_spec()), so it can be:
    - Analyzed by sobol_sensitivity() for a second-order Sobol-on-Sobol
      analysis (which segment weights interact with which?).
    - Wrapped by LocalityProfiler to test how the receptive field
      degrades under locality manipulations.
    - Fed to Falsifier to find segment weight configurations that
      cause the system to fail.
    - Fed to ContrastiveGenerator (§4.7.6 TALOT/OTTITT) to find pairs
      of segment configurations that produce maximally different outputs.

This follows Zimmerman's TALOT/OTTITT principle (§4.7.6): "meaning
emerges from contrast," and the tools should generate their own
contrasts for analysis.

Segment weight interpretation
-----------------------------
    s_i < 0.33:           drop segment (set params to midpoint)
    0.33 <= s_i < 0.66:   compress segment (linearly blend toward midpoint)
    s_i >= 0.66:          include full segment (keep original values)

The thresholds 0.33 and 0.66 divide the [0, 1] weight range into three
equal bands. This is a design choice, not a deep theoretical commitment
-- the exact thresholds could be adjusted for specific applications.

References
----------
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
        in Language, as Implemented in Humans and Large Language Models
        (LLMs)." PhD dissertation, University of Vermont.
        - §3.1: Relational meaning construction
        - §4.6.4: Power-Danger-Structure (PDS) dimensions
        - §4.7.6: TALOT/OTTITT -- meaning from contrast
        - §5.2: Supradiegetic vs. diegetic information
        - Chapter 3: Relation and meaning construction (general)

    Saltelli, A. (2002). "Making best use of model evaluations to compute
        sensitivity indices." Computer Physics Communications, 145(2),
        280-297. (Foundation for Sobol index computation.)

    Saltelli, A. et al. (2010). "Variance based sensitivity analysis of
        model output." Computer Physics Communications, 181(2), 259-270.
        (Saltelli sampling scheme used here.)

    Jansen, M.J.W. (1999). "Analysis of variance designs for model output."
        Computer Physics Communications, 117(1-2), 35-43.
        (Jansen estimator for total-order indices.)

    Hubel, D.H. & Wiesel, T.N. (1962). "Receptive fields, binocular
        interaction and functional architecture in the cat's visual cortex."
        Journal of Physiology, 160(1), 106-154. (Original receptive field
        concept in neuroscience, borrowed here as a metaphor for which
        input segments "activate" system output.)

    Beer, R.D. (2003). "The Dynamics of Active Categorical Perception in
        an Evolved Model Agent." Adaptive Behavior, 11(4), 209-243.
        (Inspiration for systematic behavioral characterization of agents.)
"""

from __future__ import annotations

import numpy as np

from zimmerman.sobol import saltelli_sample, rescale_samples, sobol_indices


def _default_segmenter(spec: dict[str, tuple[float, float]]) -> list[dict]:
    """Default segmenter: each parameter is its own segment.

    When no custom segmenter is provided, each parameter is treated as
    an independent segment. This is the finest-grained decomposition
    possible, equivalent to running standard Sobol analysis but through
    the segment-weight mechanism (with the drop/compress/include
    thresholds adding interpretive structure).

    For richer analysis, users should provide a custom segmenter that
    groups related parameters into meaningful segments. For example,
    in the Evolutionary Robotics simulator (12D), one might group the
    6 synaptic weights into a "brain" segment and the 6 physics
    parameters into a "body" segment, testing whether brain or body
    contributes more to fitness. In Zimmerman's framework (§5.2), this
    would correspond to segmenting supradiegetic (structural/control)
    parameters from diegetic (narrative/behavioral) parameters.

    Args:
        spec: Parameter spec from the simulator.

    Returns:
        List of segment dicts, each with 'name' (param name) and
        'params' (list containing just that param name).
    """
    return [
        {"name": name, "params": [name]}
        for name in sorted(spec.keys())
    ]


def _default_scorer(result: dict) -> float:
    """Default scorer: extract a scalar performance metric from the result.

    Uses the same priority-ordered key lookup as LocalityProfiler._get_score:
    'fitness' > 'score' > 'y'. Falls back to 0.0 if no recognized key is
    found (unlike LocalityProfiler, which falls back to the mean of all
    numeric values -- here we prefer a clean zero to avoid confounding
    the Sobol analysis with diagnostic values).

    For simulators with non-standard output keys, users should provide
    a custom scorer function to the PromptReceptiveField constructor.

    Args:
        result: Simulation result dict.

    Returns:
        Float score value. Returns 0.0 if no recognized key is found.
    """
    for key in ("fitness", "score", "y"):
        if key in result:
            val = result[key]
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
    return 0.0


class PromptReceptiveField:
    """Feature attribution for simulator inputs via Sobol over segment weights.

    Given a simulator, divides its parameters into segments and runs
    Sobol global sensitivity analysis over segment inclusion weights.
    This reveals which parts of the input are most important (high ST),
    which are redundant (low S1 and low ST), and which interact
    (high ST - S1).

    Interpreting the indices (Zimmerman §3.1, §4.7.6)
    --------------------------------------------------
    The Sobol indices have a direct interpretation in Zimmerman's
    meaning-construction framework:

        S1 (first-order / main effect):
            "Does this segment matter on its own?" A high S1 means the
            segment has a direct, standalone influence on the system's
            output. In Zimmerman's terms, it carries *inherent* meaning
            that does not depend on relational context (§3.1).

        ST (total-order):
            "Does this segment matter at all, including through
            interactions?" A high ST means the segment is important,
            whether through its own effect or through its combination
            with other segments.

        ST - S1 (interaction):
            "Does this segment only matter in combination with others?"
            A high interaction term means the segment participates in
            *relational* meaning construction (§3.1) -- it has no
            standalone effect but becomes significant when combined
            with other segments. This is Zimmerman's core claim about
            language: most meaning is relational, not inherent.

        Rankings (sorted by ST):
            The segments are ranked by total-order importance, giving a
            prioritized map of the system's "receptive field" -- which
            input regions the system is most sensitive to.

    The three-level weight scheme
    -----------------------------
    Rather than binary ablation (present/absent), the weight scheme
    provides three regimes:
        - Drop (s < 0.33): complete information removal, equivalent to
          setting the segment's parameters to their uninformative default
          (midpoint). Tests whether the segment is necessary at all.
        - Compress (0.33 <= s < 0.66): partial information, linearly
          interpolating between midpoint and full value. Tests whether
          the segment's influence is graded or all-or-nothing.
        - Include (s >= 0.66): full information retained. The segment
          contributes its original values unchanged.

    This three-level scheme is more informative than binary ablation
    because it allows the Sobol analysis to detect *graded* sensitivity
    (the compress region acts as a smooth transition), which helps
    distinguish parameters whose effects are approximately linear from
    those with sharp thresholds (behavioral cliffs).

    Composability (Simulator protocol)
    -----------------------------------
    The PromptReceptiveField satisfies the Simulator protocol (run() +
    param_spec()), so it can be analyzed by any zimmerman-toolkit tool.
    See the module docstring for examples of recursive analysis.

    Args:
        simulator: Any Simulator-compatible object. Must implement
            run(params: dict) -> dict and
            param_spec() -> dict[str, tuple[float, float]].
        segmenter: Callable that takes the param_spec dict and returns a
            list of segment dicts, each with 'name' (str) and 'params'
            (list of param names in this segment). If None, each
            parameter is its own segment (finest-grained decomposition).
        scorer: Callable that takes a result dict and returns a float
            score. Default: priority lookup of 'fitness', 'score', 'y'.

    Example:
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=6)
        prf = PromptReceptiveField(sim)
        report = prf.analyze(
            base_params={f"x{i}": 0.7 for i in range(6)},
            n_base=64,
        )
        print(report["rankings"])

    References:
        Zimmerman (2025), §3.1 (relational meaning), §4.7.6 (TALOT/OTTITT),
            §5.2 (supradiegetic vs. diegetic).
        Saltelli (2002), Saltelli et al. (2010) -- Sobol sensitivity analysis.
        Jansen (1999) -- total-order index estimator.
        Hubel & Wiesel (1962) -- receptive field concept (neuroscience origin).
    """

    def __init__(
        self,
        simulator,
        segmenter: callable | None = None,
        scorer: callable | None = None,
    ):
        self.simulator = simulator
        # Cache the inner simulator's parameter specification.
        # This defines the ground-truth parameter space that segment
        # weights will modulate.
        self._spec = simulator.param_spec()

        # Apply the segmenter to divide the parameter space into semantic groups.
        # The segmenter is called once at construction time; the resulting
        # segment list is fixed for the lifetime of this PromptReceptiveField.
        if segmenter is not None:
            self.segments = segmenter(self._spec)
        else:
            # Default: one segment per parameter (finest-grained decomposition)
            self.segments = _default_segmenter(self._spec)

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = _default_scorer

        # Pre-extract segment names for fast lookup during analysis.
        # The order of segment_names determines the column ordering in
        # the Saltelli sample matrix, which must be consistent throughout.
        self.segment_names = [seg["name"] for seg in self.segments]

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return segment weight spec: each segment weight in [0, 1].

        This makes PromptReceptiveField a valid Simulator for use with
        sobol_sensitivity(), Falsifier, ContrastiveGenerator, etc.
        The "parameters" of this meta-simulator are the segment inclusion
        weights, each bounded to [0, 1].

        The full weight range [0, 1] spans all three regimes:
            [0.0, 0.33): drop     -- segment is absent
            [0.33, 0.66): compress -- segment is partially present
            [0.66, 1.0]: include   -- segment is fully present

        Returns:
            Dict mapping segment names to (0.0, 1.0) bounds.
        """
        return {seg["name"]: (0.0, 1.0) for seg in self.segments}

    def _apply_segment_weights(
        self,
        base_params: dict[str, float],
        segment_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply segment weights to base parameters using the 3-level scheme.

        This is the core mapping from continuous segment weights to
        parameter modifications. The three-level scheme provides a richer
        perturbation landscape than binary ablation:

        For each segment with weight s_i:
            s_i < 0.33 (DROP):
                All parameters in this segment are set to their midpoint
                (uninformative default). The segment is fully absent from
                the system's input. This tests whether the segment is
                *necessary* for the system's behavior.

            0.33 <= s_i < 0.66 (COMPRESS):
                Parameters are linearly interpolated between midpoint
                (at s_i = 0.33) and their original value (at s_i = 0.66).
                The blend factor is (s_i - 0.33) / 0.33, which maps
                [0.33, 0.66) to [0, 1). This region tests *graded*
                sensitivity: does performance degrade smoothly as the
                segment's information content decreases, or is there a
                sharp threshold (behavioral cliff)?

            s_i >= 0.66 (INCLUDE):
                Parameters retain their original values unchanged. The
                segment is fully present. This is the "control" condition.

        The midpoint serves as the "zero-information" anchor (Zimmerman
        §2.4): a parameter at its midpoint conveys no information about
        whether the system should behave differently from its default.

        Design note: the compress region's linear interpolation means
        that the mapping from weight to parameter value is piecewise
        linear with two breakpoints at 0.33 and 0.66. This is C0
        continuous (no discontinuity in value) but not C1 continuous
        (there are slope discontinuities at the breakpoints). The Sobol
        analysis handles this fine -- it is a variance-based method that
        does not require smoothness.

        Args:
            base_params: Original parameter values (the "full information"
                configuration).
            segment_weights: Dict mapping segment names to weight values
                in [0, 1].

        Returns:
            Modified parameter dict with segment weights applied.
        """
        modified = dict(base_params)

        for seg in self.segments:
            seg_name = seg["name"]
            seg_params = seg["params"]
            # Default weight = 1.0 (include) if not specified, so
            # unmentioned segments are fully present
            weight = segment_weights.get(seg_name, 1.0)

            for param_name in seg_params:
                if param_name not in self._spec:
                    continue
                lo, hi = self._spec[param_name]
                # Midpoint = uninformative default (center of feasible range)
                midpoint = (lo + hi) / 2.0
                original = base_params.get(param_name, midpoint)

                if weight < 0.33:
                    # DROP regime: replace with midpoint (zero information)
                    modified[param_name] = midpoint
                elif weight < 0.66:
                    # COMPRESS regime: linear interpolation from midpoint
                    # (at weight=0.33) to original (at weight=0.66).
                    # blend=0 at weight=0.33, blend~1 at weight~0.66
                    blend = (weight - 0.33) / 0.33
                    modified[param_name] = float(
                        midpoint + blend * (original - midpoint)
                    )
                else:
                    # INCLUDE regime: keep original value unchanged
                    modified[param_name] = original

        return modified

    def run(self, params: dict) -> dict:
        """Simulator protocol: params are segment weights.

        When called via the Simulator protocol (e.g., by
        sobol_sensitivity(PromptReceptiveField(sim))), the "parameters"
        are segment inclusion weights and the "output" is the inner
        simulator's result under those segment configurations.

        The base parameters are the midpoint of the inner simulator's
        spec -- the uninformative default. This means the run() method
        tests segment influence starting from a neutral baseline, not
        from any specific operating point. For analysis relative to a
        specific operating point, use the analyze() method instead,
        which accepts explicit base_params.

        This design choice means run() and analyze() serve different
        purposes:
            - run() is for composability (Simulator protocol compliance).
            - analyze() is for direct receptive field analysis with a
              user-specified baseline.

        Args:
            params: Dict mapping segment names to weight values in [0, 1].

        Returns:
            The inner simulator's result dict, augmented with a
            "segment_weights" key recording the applied weights.
        """
        # Use midpoint as base params for the Simulator protocol run.
        # This ensures that the "default" state (all segments included)
        # corresponds to the center of the parameter space, providing
        # a symmetric baseline for the Sobol analysis.
        base_params = {
            name: (lo + hi) / 2.0
            for name, (lo, hi) in self._spec.items()
        }
        modified = self._apply_segment_weights(base_params, params)
        result = self.simulator.run(modified)
        # Annotate the result with the segment weights for traceability
        result["segment_weights"] = dict(params)
        return result

    def analyze(
        self,
        base_params: dict[str, float],
        n_base: int = 64,
        seed: int = 42,
    ) -> dict:
        """Run receptive field analysis using Sobol over segment inclusion weights.

        This is the main analysis entry point. It generates Saltelli
        samples over segment weights (Saltelli, 2002; 2010), applies
        each weight configuration to the provided base parameters, runs
        the inner simulator, scores each result, and computes Sobol
        first-order (S1) and total-order (ST) indices using the Jansen
        (1999) estimator (via zimmerman.sobol.sobol_indices).

        The algorithm has five steps:

        Step 1 -- Saltelli sampling:
            Generate n_base * (d + 2) quasi-random weight vectors in
            [0, 1]^d using the Saltelli (2010) cross-matrix scheme,
            where d = number of segments. The layout is:
                rows [0, n_base): matrix A
                rows [n_base, 2*n_base): matrix B
                rows [2*n_base, ...): d cross-matrices C^(i), each of
                    size n_base, where C^(i) = A with column i from B.

        Step 2 -- Simulate:
            For each of the n_total weight vectors, apply the segment
            weights to the base parameters via _apply_segment_weights(),
            then run the inner simulator and score the result.

        Step 3 -- Extract Saltelli sub-arrays:
            Partition the score vector into y_A, y_B, and y_C matching
            the Saltelli layout.

        Step 4 -- Compute Sobol indices:
            Call zimmerman.sobol.sobol_indices(y_A, y_B, y_C) to get
            S1 and ST for each segment. The interaction term is ST - S1.

        Step 5 -- Rank and return:
            Sort segments by descending ST to produce a priority ranking
            of which segments the system is most sensitive to.

        Interpreting the results (Zimmerman §3.1):
            - High S1: segment has standalone influence (inherent meaning).
            - High ST - S1: segment's influence depends on other segments
              (relational meaning, §3.1).
            - Low ST: segment is redundant / the system ignores it.
            - Rankings: ordered by total importance (ST), not just main
              effect (S1), because relational contributions matter.

        Computational cost:
            n_base * (d + 2) simulator evaluations, where d is the number
            of segments. For 6 segments and n_base=64, this is 512 runs.
            The cost scales linearly with n_base and linearly with d.

        Args:
            base_params: Baseline parameter values to perturb. This is the
                operating point around which segment inclusion/exclusion
                is tested. It should be a meaningful configuration (e.g.,
                a known good gait, a healthy patient, a specific scenario),
                not just the midpoint.
            n_base: Base sample count for Saltelli sampling (N in the
                notation of Saltelli 2010). Total simulations =
                n_base * (n_segments + 2). Larger values give more
                accurate Sobol indices but increase cost linearly.
                64 is a reasonable minimum; 256+ for publication quality.
            seed: Random seed for reproducibility. Passed to the numpy
                RNG that generates the Saltelli sample matrices.

        Returns:
            Dict with:
                "segment_names": [str, ...] -- ordered list of segment names,
                "S1": {segment_name: float, ...} -- first-order Sobol indices,
                "ST": {segment_name: float, ...} -- total-order Sobol indices,
                "interaction": {segment_name: float, ...} -- ST - S1 per segment,
                "rankings": [segment_names sorted by descending ST],
                "n_sims": int -- total simulator evaluations performed,
        """
        d = len(self.segments)  # Number of segments = dimensionality of the weight space
        rng = np.random.default_rng(seed)

        # =================================================================
        # Step 1: Generate Saltelli samples over segment weights in [0,1]^d
        # -----------------------------------------------------------------
        # Uses the Saltelli (2010) cross-matrix scheme via zimmerman.sobol.
        # Produces n_base * (d + 2) sample vectors, laid out as:
        #   [0, n_base):           matrix A (base random samples)
        #   [n_base, 2*n_base):    matrix B (independent random samples)
        #   [2*n_base, ...):       d cross-matrices C^(i), each n_base rows,
        #                          where C^(i) = A with column i from B.
        # =================================================================
        n_total = n_base * (d + 2)
        samples_01 = saltelli_sample(n_base, d, rng)

        # Segment weights are inherently in [0, 1], so rescaling is a no-op.
        # We still call rescale_samples for API consistency with the Sobol
        # pipeline, which expects bounds to be explicitly provided.
        bounds = np.array([[0.0, 1.0]] * d)
        samples = rescale_samples(samples_01, bounds)

        # Sanity check: the Saltelli scheme should produce exactly n_total rows
        assert samples.shape[0] == n_total, (
            f"Expected {n_total} samples, got {samples.shape[0]}"
        )

        # =================================================================
        # Step 2: Run all simulations
        # -----------------------------------------------------------------
        # For each sample vector, construct segment weights, apply the
        # 3-level scheme (drop/compress/include) to the base parameters,
        # run the inner simulator, and score the result.
        # =================================================================
        scores = np.zeros(n_total)
        for idx in range(n_total):
            # Build segment weight dict from the sample row.
            # Column j of the sample matrix corresponds to segment j.
            segment_weights = {
                self.segment_names[j]: float(samples[idx, j])
                for j in range(d)
            }
            # Apply the 3-level weight scheme and run the inner simulator
            modified = self._apply_segment_weights(base_params, segment_weights)
            result = self.simulator.run(modified)
            scores[idx] = self.scorer(result)

        # =================================================================
        # Step 3: Extract sub-arrays matching the Saltelli layout
        # -----------------------------------------------------------------
        # The Sobol index estimators need the scores partitioned into:
        #   y_A: scores from matrix A (shape: n_base)
        #   y_B: scores from matrix B (shape: n_base)
        #   y_C: scores from cross-matrices (shape: d x n_base)
        # The cross-matrix y_C[i] corresponds to "A with column i from B,"
        # which isolates the effect of segment i.
        # =================================================================
        y_A = scores[:n_base]
        y_B = scores[n_base:2 * n_base]
        y_C = scores[2 * n_base:].reshape(d, n_base)

        # =================================================================
        # Step 4: Compute Sobol indices (Saltelli 2010 + Jansen 1999)
        # -----------------------------------------------------------------
        # S1[i]: first-order index -- fraction of output variance explained
        #        by segment i alone (main effect).
        # ST[i]: total-order index -- fraction of output variance involving
        #        segment i in any combination (main + all interactions).
        # interaction[i] = ST[i] - S1[i]: the purely relational contribution
        #        of segment i (Zimmerman §3.1).
        # =================================================================
        S1, ST = sobol_indices(y_A, y_B, y_C)
        interaction = ST - S1  # Relational/interaction contribution per segment

        # =================================================================
        # Step 5: Build result dictionaries and rank segments
        # =================================================================
        s1_dict = {self.segment_names[i]: float(S1[i]) for i in range(d)}
        st_dict = {self.segment_names[i]: float(ST[i]) for i in range(d)}
        interaction_dict = {self.segment_names[i]: float(interaction[i]) for i in range(d)}

        # Rankings: sorted by total-order (ST) in descending order.
        # ST is preferred over S1 for ranking because it captures both
        # standalone and relational contributions. A segment with low S1
        # but high ST is one that "only matters in context" (§3.1) -- it
        # would be missed by a ranking based on S1 alone.
        rankings = sorted(
            self.segment_names,
            key=lambda name: st_dict[name],
            reverse=True,
        )

        return {
            "segment_names": list(self.segment_names),  # Ordered segment names
            "S1": s1_dict,              # First-order (main effect) indices
            "ST": st_dict,              # Total-order (main + interaction) indices
            "interaction": interaction_dict,  # Relational contribution (ST - S1)
            "rankings": rankings,       # Segments ranked by total importance
            "n_sims": n_total,          # Total simulator evaluations performed
        }
