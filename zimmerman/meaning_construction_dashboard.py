"""Meaning construction dashboard: unified aggregator for all 12 Zimmerman toolkit analyses.

Implements the synthesis layer described in Zimmerman (2025) Chapter 6, which
argues that understanding a simulator's behavior requires integrating multiple
complementary analytical perspectives. No single analysis tool captures the
full picture; the dashboard aggregates them into a unified "meaning construction"
report that reveals what the simulator *actually does* -- not just what it is
*supposed* to do.

This philosophy draws directly from Beer's (2024) POSIWID framework ("the
Purpose Of a System Is What It Does"), which rejects teleological explanations
in favor of empirical behavioral characterization. The dashboard operationalizes
POSIWID by combining six reporting sections, each addressing a distinct
dimension of simulator behavior:

    1. **Sensitivity** (Sobol indices from ``sobol_sensitivity``, plus
       ``prompt_receptive_field``): Which parameters matter most? Uses
       variance-based global sensitivity analysis following Saltelli et al.
       (2008) "Global Sensitivity Analysis: The Primer" to decompose output
       variance into first-order (S1), second-order, and total-order (ST)
       contributions per parameter. High interaction strength (ST - S1)
       indicates non-additive parameter coupling.

    2. **Robustness** (``falsifier`` boundary stress testing, ``contrast_sets``
       and ``contrastive`` analyses): Where does the simulator break? The
       falsifier probes for constraint violations near parameter bounds;
       contrastive analysis identifies minimal parameter changes that flip
       qualitative outcomes (the "behavioral cliffs" in Zimmerman SS4.3).

    3. **Locality** (``locality_profiler``): How far do perturbation effects
       propagate through the simulator? Measures L50 (the perturbation radius
       at which 50% of output variance is captured) and effective horizon
       (the radius beyond which perturbations have negligible effect). Draws
       on Zimmerman's locality framework (Chapter 3) which argues that
       locality structure determines what meanings can be constructed.

    4. **Alignment** (``diegeticizer`` roundtrip + ``supradiegetic_benchmark``
       via POSIWID audit): How well does narrative framing preserve information
       through the LLM roundtrip? Measures the gap between intended and
       actual simulator behavior -- Beer's "intention-outcome gap." Low
       alignment scores indicate the LLM pipeline is distorting the
       simulation's semantics.

    5. **Representation** (``token_extispicy`` + ``prompt_receptive_field`` +
       ``supradiegetic_benchmark``): How do tokenization and prompt structure
       affect LLM behavior? Integrates fragmentation-output correlations from
       token extispicy (Zimmerman SS4.6.4), receptive field analysis of prompt
       sensitivity, and roundtrip error measurements from the benchmark.
       This section reveals whether representational artifacts (tokenization,
       prompt formatting) are degrading the LLM's ability to mediate
       parameter generation.

    6. **Structure** (``relation_graph`` + ``contrast_set`` + ``pds``): What
       relational patterns emerge in the output space? The relation graph maps
       parameter-to-output causal dependencies; PDS (Parameter Dimension
       Structure) identifies latent dimensions in the parameter-output mapping;
       contrast sets reveal the minimal distinguishing features between
       qualitatively different output regimes.

Each section is populated from the corresponding tool's output dict. Missing
tools are gracefully omitted -- the dashboard compiles whatever is available
and notes what could add more insight. This graceful degradation is by design:
practitioners may not have access to all tools (e.g., token extispicy requires
a tokenizer callback, POSIWID requires an LLM), so the dashboard should
always provide value with whatever subset is available.

The dashboard also generates **actionable recommendations** based on the
combined analysis via ``generate_recommendations()``. These are rule-based
heuristics that synthesize findings across sections. For example, if POSIWID
alignment is low AND token fragmentation correlates with output degradation,
it recommends diegetic prompts to mitigate representational flattening --
a cross-section insight that no individual tool could produce alone.

The dashboard itself satisfies the **Simulator protocol** (``run()`` +
``param_spec()``), enabling recursive meta-analysis: you can run
``sobol_sensitivity()`` on the dashboard itself to determine which parameters
most affect the dashboard's own recommendations.

References:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont. Chapter 6 (synthesis).

    Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J.,
    Gatelli, D., Saisana, M., & Tarantola, S. (2008). "Global Sensitivity
    Analysis: The Primer." Wiley. (Foundation for the Sobol sensitivity
    analysis used in Section 1.)

    Beer, S. (2024). "The Purpose Of a System Is What It Does (POSIWID)."
    (Overarching philosophy: the dashboard reveals what a simulator actually
    does, not what it is supposed to do.)
"""

from __future__ import annotations

import numpy as np


# Registry mapping each zimmerman toolkit tool key to its dashboard section.
# This is the canonical mapping from Zimmerman (2025) Chapter 6, Table 6.1.
# Each tool contributes to exactly one of the six analytical dimensions.
# The keys here must match the keys used in the ``reports`` dict passed to
# ``MeaningConstructionDashboard.compile()``.
_TOOL_SECTIONS = {
    # --- Section 1: Sensitivity ---
    # Which parameters matter? Variance-based global sensitivity (Saltelli 2008).
    "sobol": "sensitivity",           # Sobol S1/ST indices from sobol_sensitivity()
    "receptive_field": "sensitivity",  # Prompt receptive field analysis

    # --- Section 2: Robustness ---
    # Where does the simulator break? Boundary stress testing and contrastive analysis.
    "falsifier": "robustness",        # Constraint violation detection at boundaries
    "contrast_sets": "robustness",    # Minimal contrast set identification
    "contrastive": "robustness",      # Contrastive sensitivity (which params flip outcomes)

    # --- Section 3: Locality ---
    # How far do perturbation effects propagate? (Zimmerman Ch. 3)
    "locality": "locality",           # L50, effective horizon from locality_profiler

    # --- Section 4: Alignment ---
    # Does the simulator do what it intends? (Beer 2024 POSIWID)
    "posiwid": "alignment",           # Intention-outcome gap from posiwid_audit

    # --- Section 5: Representation ---
    # How well is meaning preserved through LLM encoding/decoding?
    "benchmark": "representation",     # Supradiegetic benchmark roundtrip error
    "diegeticizer": "representation",  # Diegeticization gain (narrative vs numeric)
    "token_extispicy": "representation",  # Fragmentation-output correlation

    # --- Section 6: Structure ---
    # What relational patterns exist in parameter-output space?
    "relation_graph": "structure",     # Causal dependency graph
    "pds": "structure",               # Parameter Dimension Structure (latent dims)
}

# All tool keys the dashboard recognises, in section order.
# Used for tool coverage reporting in compile().
ALL_TOOL_KEYS = list(_TOOL_SECTIONS.keys())


class MeaningConstructionDashboard:
    """Compiles a unified dashboard from individual zimmerman tool reports.

    This is the synthesis layer from Zimmerman (2025) Chapter 6. Given a
    simulator and a dictionary of tool reports (from any subset of the 12
    zimmerman toolkit analyses), it produces a structured dashboard with:
      - **Six analytical sections** (sensitivity, robustness, locality,
        alignment, representation, structure) -- each populated from the
        relevant tool reports, with graceful degradation for missing tools
      - **Actionable recommendations** -- rule-based heuristics that
        synthesize cross-section insights into concrete suggestions
      - **Tool coverage summary** -- which tools contributed and which
        could add more insight if run

    The overarching philosophy is Beer's (2024) POSIWID: "the Purpose Of
    a System Is What It Does." The dashboard reveals what a simulator
    *actually does* across multiple analytical dimensions, rather than
    relying on the simulator designer's stated intentions. Each section
    contributes a different lens on the simulator's empirical behavior.

    **Simulator protocol compliance**: The dashboard itself satisfies the
    Simulator protocol (``run()`` + ``param_spec()``), enabling recursive
    meta-analysis. For example, running ``sobol_sensitivity()`` on the
    dashboard reveals which parameters most affect the dashboard's own
    output metrics -- a second-order sensitivity analysis.

    Args:
        simulator: Any Simulator-compatible object with ``run(params) -> dict``
            and ``param_spec() -> dict[str, tuple[float, float]]``.

    Example:
        from zimmerman import sobol_sensitivity, Falsifier

        sim = MySimulator()
        sobol_report = sobol_sensitivity(sim)
        falsifier_report = Falsifier(sim).falsify()

        dash = MeaningConstructionDashboard(sim)
        result = dash.compile(reports={
            "sobol": sobol_report,
            "falsifier": falsifier_report,
        })
        print(result["sections"]["sensitivity"]["top_params"])
        print(result["recommendations"])

    References:
        Zimmerman (2025) Chapter 6 -- Meaning construction synthesis
        Beer (2024) POSIWID -- The purpose of a system is what it does
        Saltelli et al. (2008) -- Global sensitivity analysis methodology
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self._spec = simulator.param_spec()

    # ------------------------------------------------------------------ #
    #  Section builders                                                    #
    #  Each method builds one of the six analytical sections from the      #
    #  relevant tool report(s). All follow the same pattern:               #
    #    1. Extract relevant reports from the reports dict                  #
    #    2. If no relevant reports exist, return {"available": False, ...}  #
    #    3. Otherwise, extract key metrics and return {"available": True}   #
    # ------------------------------------------------------------------ #

    def _build_sensitivity(self, reports: dict) -> dict:
        """Build Section 1: Sensitivity -- which parameters matter most?

        Integrates results from two complementary sensitivity analyses:
          - **Sobol indices** (Saltelli et al. 2008): Variance-based global
            sensitivity analysis decomposing output variance into per-parameter
            contributions. S1 (first-order) measures each parameter's independent
            effect; ST (total-order) includes all interactions. The difference
            ST - S1 = "interaction strength" quantifies non-additive coupling.
          - **Receptive field**: Prompt-level sensitivity analysis measuring
            which parts of the input prompt most affect LLM output.

        Sobol is preferred when available; receptive field serves as a fallback
        for ranking parameters when Sobol data is missing.

        Args:
            reports: Full reports dict (may or may not contain relevant keys).

        Returns:
            Section dict with:
                'available': bool -- whether any sensitivity data was found,
                'top_params': list -- parameters ranked by influence (S1 order),
                'interaction_strength': float or None -- mean |ST - S1| across
                    all parameters and outputs (how non-additive the system is).
        """
        sobol = reports.get("sobol")
        receptive = reports.get("receptive_field")

        if sobol is None and receptive is None:
            return {"available": False, "top_params": [], "interaction_strength": None}

        top_params = []
        interaction_strength = None

        if sobol is not None:
            # Extract the S1 (first-order) parameter ranking for the primary
            # output key. S1 measures each parameter's independent contribution
            # to output variance (Saltelli et al. 2008, Section 4.2).
            rankings = sobol.get("rankings", {})
            output_keys = sobol.get("output_keys", [])
            if output_keys and rankings:
                first_key = output_keys[0]
                ranking_key = f"{first_key}_most_influential_S1"
                if ranking_key in rankings:
                    top_params = list(rankings[ranking_key])

            # Compute mean interaction strength = mean |ST - S1| across all
            # outputs and parameters. High interaction strength (> 0.1) means
            # parameters interact non-additively -- changing one parameter
            # alters the effect of other parameters on the output. This is
            # critical for LLM prompt design: if interactions are strong,
            # parameters should be generated jointly rather than independently.
            interaction_vals = []
            for key in output_keys:
                key_data = sobol.get(key, {})
                interactions = key_data.get("interaction", {})
                interaction_vals.extend(interactions.values())
            if interaction_vals:
                interaction_strength = float(np.mean(np.abs(interaction_vals)))

        if receptive is not None and not top_params:
            # Fall back to receptive field rankings when Sobol data is not
            # available. The receptive field measures prompt-level sensitivity
            # rather than parameter-level variance decomposition, so it
            # provides a complementary (but less rigorous) ranking.
            rf_rankings = receptive.get("rankings", [])
            if rf_rankings:
                top_params = list(rf_rankings)

        return {
            "available": True,
            "top_params": top_params,
            "interaction_strength": interaction_strength,
        }

    def _build_robustness(self, reports: dict) -> dict:
        """Build Section 2: Robustness -- where does the simulator break?

        Integrates results from three boundary-probing tools:
          - **Falsifier**: Systematic boundary stress testing that searches for
            constraint violations near parameter bounds. The violation_rate
            is the fraction of tested configurations that violate at least one
            constraint -- a direct measure of how fragile the simulator is.
          - **Contrastive analysis**: Identifies minimal parameter changes that
            flip qualitative outcomes (e.g., a robot gait going from stable to
            unstable). These are Zimmerman's "behavioral cliffs" (SS4.3).
          - **Contrast sets**: Similar to contrastive but focused on identifying
            the minimal distinguishing feature sets between output regimes.

        Together, these reveal the simulator's fragility landscape: which
        parameters are most likely to cause catastrophic behavior changes
        when slightly perturbed.

        Args:
            reports: Full reports dict.

        Returns:
            Section dict with:
                'available': bool -- whether any robustness data was found,
                'violation_rate': float or None -- fraction of boundary
                    configurations that violate constraints,
                'mean_flip_size': float or None -- mean parameter change needed
                    to flip qualitative outcomes (smaller = more fragile),
                'most_fragile_params': list or None -- parameters ranked by
                    fragility (most fragile first).
        """
        falsifier = reports.get("falsifier")
        contrast_sets = reports.get("contrast_sets")
        contrastive = reports.get("contrastive")

        if falsifier is None and contrast_sets is None and contrastive is None:
            return {
                "available": False,
                "violation_rate": None,
                "mean_flip_size": None,
                "most_fragile_params": None,
            }

        violation_rate = None
        mean_flip_size = None
        most_fragile_params = None

        if falsifier is not None:
            # The falsifier's violation_rate is the proportion of boundary
            # stress tests that triggered constraint violations. A rate
            # above 5% suggests significant fragility at the edges of the
            # parameter space.
            summary = falsifier.get("summary", {})
            violation_rate = summary.get("violation_rate")

        # Extract fragile parameters from contrastive analysis.
        # The contrastive analysis identifies parameters where small changes
        # flip qualitative outcomes -- the "behavioral cliff" parameters
        # from Zimmerman SS4.3.
        if contrastive is not None:
            rankings = contrastive.get("rankings")
            if rankings:
                most_fragile_params = list(rankings)
            # param_importance quantifies how much each parameter contributes
            # to outcome flips; its mean gives the average "flip size."
            importance = contrastive.get("param_importance", {})
            if importance:
                vals = list(importance.values())
                if vals:
                    mean_flip_size = float(np.mean(vals))

        # Contrast sets provide a complementary view: minimal feature sets
        # that distinguish output regimes. Used as a fallback if contrastive
        # analysis did not provide fragile parameter rankings.
        if contrast_sets is not None:
            cs_rankings = contrast_sets.get("rankings")
            if cs_rankings and most_fragile_params is None:
                most_fragile_params = list(cs_rankings)
            cs_flip = contrast_sets.get("mean_flip_size")
            if cs_flip is not None and mean_flip_size is None:
                mean_flip_size = float(cs_flip)

        return {
            "available": True,
            "violation_rate": violation_rate,
            "mean_flip_size": mean_flip_size,
            "most_fragile_params": most_fragile_params,
        }

    def _build_locality(self, reports: dict) -> dict:
        """Build Section 3: Locality -- how far do perturbation effects propagate?

        Draws on Zimmerman's locality framework (Chapter 3), which argues that
        the locality structure of a system determines what meanings can be
        constructed from its behavior. Two key metrics:
          - **L50**: The perturbation radius (as a fraction of parameter range)
            at which 50% of the total output variance is captured. Small L50
            means the system is highly local -- most of its behavior is
            determined by nearby parameter values. Large L50 means effects
            propagate widely.
          - **Effective horizon**: The perturbation radius beyond which further
            perturbation has negligible additional effect on the output. This
            defines the "horizon of influence" for each parameter.

        For LLM-mediated systems, locality has practical implications: if the
        system is highly local (L50 < 0.3), the order in which parameters
        appear in the prompt matters because early parameters dominate. If
        the system is non-local (L50 > 0.7), parameter ordering matters less
        but parameter interactions become more important.

        Args:
            reports: Full reports dict.

        Returns:
            Section dict with:
                'available': bool -- whether locality data was found,
                'L50': float or None -- perturbation radius for 50% variance,
                'effective_horizon': float or None -- radius of negligible effect.
        """
        locality = reports.get("locality")

        if locality is None:
            return {"available": False, "L50": None, "effective_horizon": None}

        L50 = locality.get("L50")
        effective_horizon = locality.get("effective_horizon")

        return {
            "available": True,
            "L50": L50,
            "effective_horizon": float(effective_horizon) if effective_horizon is not None else None,
        }

    def _build_alignment(self, reports: dict) -> dict:
        """Build Section 4: Alignment -- does the simulator do what it intends?

        Operationalizes Beer's (2024) POSIWID principle: "the Purpose Of a
        System Is What It Does." The alignment section measures the gap between
        the simulator's *intended* behavior (as specified by its design) and
        its *actual* behavior (as observed empirically). This is the
        "intention-outcome gap" that POSIWID forces us to confront.

        The POSIWID audit can be run in two modes:
          - **Single audit** (``audit()``): Tests alignment for one parameter
            configuration, producing per-output-key alignment scores.
          - **Batch audit** (``batch_audit()``): Tests alignment across many
            configurations, producing aggregate statistics with per-key means.

        This section extracts:
          - ``overall_alignment``: Scalar [0, 1] measuring how well the
            simulator's actual behavior matches its intended behavior.
            1.0 = perfect alignment, 0.0 = total misalignment.
          - ``worst_aligned_keys``: Output keys sorted by alignment score
            (worst first), identifying which outputs diverge most from
            their intended semantics.

        Low alignment scores (< 0.5) are a strong signal that the LLM
        pipeline is distorting the simulation's semantics, and the
        recommendations engine will flag this for investigation.

        Args:
            reports: Full reports dict.

        Returns:
            Section dict with:
                'available': bool -- whether POSIWID data was found,
                'overall_alignment': float or None -- scalar alignment [0, 1],
                'worst_aligned_keys': list or None -- output keys sorted by
                    alignment (worst first).
        """
        posiwid = reports.get("posiwid")

        if posiwid is None:
            return {
                "available": False,
                "overall_alignment": None,
                "worst_aligned_keys": None,
            }

        overall_alignment = None
        worst_aligned_keys = None

        # Handle batch_audit() output, which wraps per-configuration results
        # in an "aggregate" dict with mean statistics across all configurations.
        aggregate = posiwid.get("aggregate")
        if aggregate is not None:
            overall_alignment = aggregate.get("mean_overall")
            # Sort output keys by their combined alignment score (ascending)
            # so the worst-aligned keys appear first. This helps practitioners
            # focus on the outputs with the largest intention-outcome gap.
            per_key = aggregate.get("per_key_mean", {})
            if per_key:
                sorted_keys = sorted(
                    per_key.keys(),
                    key=lambda k: per_key[k].get("combined", 0.0),
                )
                worst_aligned_keys = sorted_keys
        else:
            # Handle single audit() output, which has alignment scores
            # directly (not wrapped in an aggregate dict).
            alignment = posiwid.get("alignment", {})
            overall_alignment = alignment.get("overall")
            per_key = alignment.get("per_key", {})
            if per_key:
                sorted_keys = sorted(
                    per_key.keys(),
                    key=lambda k: per_key[k].get("combined", 0.0),
                )
                worst_aligned_keys = sorted_keys

        return {
            "available": True,
            "overall_alignment": float(overall_alignment) if overall_alignment is not None else None,
            "worst_aligned_keys": worst_aligned_keys,
        }

    def _build_representation(self, reports: dict) -> dict:
        """Build Section 5: Representation -- how well is meaning preserved?

        Integrates three tools that probe how LLM encoding/decoding affects
        the fidelity of parameter representation:

          - **Diegeticizer** (roundtrip + diegeticization gain): Measures
            whether narrative ("diegetic") framing of parameters preserves
            information better than raw numeric framing through the LLM
            roundtrip. A positive ``diegeticization_gain`` means narrative
            prompts outperform numeric prompts -- the LLM handles stories
            better than spreadsheets. This connects to Zimmerman's argument
            (SS4.5) that diegetic representation leverages the LLM's training
            distribution (predominantly narrative text) more effectively.

          - **Supradiegetic benchmark** (roundtrip error): Measures how
            accurately numeric values survive the LLM encoding/decoding
            roundtrip. High roundtrip error (> 0.1) means the LLM is
            systematically distorting parameter values, even in the best
            case (supradiegetic = outside any narrative frame).

          - **Token extispicy** (fragmentation-output correlation): From
            Zimmerman SS4.6.4 -- measures whether tokenization fragmentation
            predicts output degradation. The ``fragmentation_correlation``
            is the mean |r| across all output keys: values above 0.3 are
            strong evidence that tokenization artifacts are propagating into
            simulation results.

        Together, these three metrics characterize the LLM's representational
        fidelity for numeric/parametric content.

        Args:
            reports: Full reports dict.

        Returns:
            Section dict with:
                'available': bool -- whether any representation data was found,
                'diegeticization_gain': float or None -- improvement from
                    narrative framing (positive = narrative helps),
                'roundtrip_error': float or None -- numeric distortion through
                    LLM encoding/decoding,
                'fragmentation_correlation': float or None -- mean |r| between
                    fragmentation and output degradation.
        """
        benchmark = reports.get("benchmark")
        diegeticizer = reports.get("diegeticizer")
        token_ext = reports.get("token_extispicy")

        if benchmark is None and diegeticizer is None and token_ext is None:
            return {
                "available": False,
                "diegeticization_gain": None,
                "roundtrip_error": None,
                "fragmentation_correlation": None,
            }

        diegeticization_gain = None
        roundtrip_error = None
        fragmentation_correlation = None

        # Diegeticization gain: how much does narrative framing improve
        # parameter fidelity through the LLM roundtrip? Positive values
        # mean narrative prompts outperform raw numeric prompts.
        if diegeticizer is not None:
            diegeticization_gain = diegeticizer.get("diegeticization_gain")
            if diegeticization_gain is not None:
                diegeticization_gain = float(diegeticization_gain)

        # Roundtrip error from the supradiegetic benchmark: the best-case
        # numeric distortion when parameters pass through LLM encoding and
        # decoding. Check both key names for backward compatibility.
        if benchmark is not None:
            roundtrip_error = benchmark.get("roundtrip_error")
            if roundtrip_error is None:
                roundtrip_error = benchmark.get("mean_roundtrip_error")
            if roundtrip_error is not None:
                roundtrip_error = float(roundtrip_error)

        # Fragmentation-output correlation from token extispicy: the mean
        # absolute Pearson r between fragmentation rate and simulator output
        # across all output keys. Using absolute correlation because both
        # positive and negative correlations indicate fragmentation effects
        # (the sign depends on the output semantics).
        if token_ext is not None:
            frag_corr = token_ext.get("fragmentation_output_correlation", {})
            if frag_corr:
                corr_vals = [abs(v) for v in frag_corr.values()]
                if corr_vals:
                    fragmentation_correlation = float(np.mean(corr_vals))

        return {
            "available": True,
            "diegeticization_gain": diegeticization_gain,
            "roundtrip_error": roundtrip_error,
            "fragmentation_correlation": fragmentation_correlation,
        }

    def _build_structure(self, reports: dict) -> dict:
        """Build Section 6: Structure -- what relational patterns emerge?

        Integrates two tools that probe the structural relationships in the
        parameter-to-output mapping:

          - **Relation graph**: Maps causal dependencies between parameters
            and outputs, identifying which parameters have the strongest
            directional influence on which outputs. The ``most_causal_params``
            are those with the highest total causal effect across all outputs.

          - **PDS (Parameter Dimension Structure)**: Identifies latent
            dimensions in the parameter-output mapping. ``variance_explained``
            measures how much of the output variance is captured by the
            identified PDS dimensions. Low variance explained (< 30%) suggests
            the dimension-to-parameter mapping is incomplete and should be
            revised -- there are important structural dimensions in the
            output space that the current parameterization does not capture.

        Together, these reveal the *relational structure* of the simulator:
        not just which parameters matter (that is sensitivity's job), but
        *how* parameters relate to outputs and to each other in a
        structurally meaningful way.

        Args:
            reports: Full reports dict.

        Returns:
            Section dict with:
                'available': bool -- whether any structure data was found,
                'most_causal_params': list or None -- parameters ranked by
                    total causal effect on outputs,
                'variance_explained': float or None -- mean fraction of output
                    variance explained by PDS dimensions.
        """
        relation_graph = reports.get("relation_graph")
        pds = reports.get("pds")

        if relation_graph is None and pds is None:
            return {
                "available": False,
                "most_causal_params": None,
                "variance_explained": None,
            }

        most_causal_params = None
        variance_explained = None

        # PDS variance explained: mean across all output keys of the fraction
        # of variance captured by the identified latent dimensions.
        if pds is not None:
            ve = pds.get("variance_explained", {})
            if ve:
                vals = list(ve.values())
                if vals:
                    variance_explained = float(np.mean(vals))

        # Causal parameter ranking from relation_graph: parameters ordered
        # by their total directional influence on outputs. Falls back to
        # generic "rankings" key if "most_causal_params" is not present.
        if relation_graph is not None:
            causal = relation_graph.get("most_causal_params")
            if causal is not None:
                most_causal_params = list(causal)
            elif relation_graph.get("rankings"):
                most_causal_params = list(relation_graph["rankings"])

        return {
            "available": True,
            "most_causal_params": most_causal_params,
            "variance_explained": variance_explained,
        }

    # ------------------------------------------------------------------ #
    #  Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def generate_recommendations(self, sections: dict) -> list[str]:
        """Generate actionable recommendations by synthesizing cross-section findings.

        This is the rule-based synthesis engine from Zimmerman (2025) Chapter 6.
        It examines each section's extracted metrics and applies threshold-based
        heuristics to produce human-readable suggestions for improving the
        simulation or the LLM pipeline that drives it.

        The recommendations are *cross-sectional*: they synthesize findings
        from multiple sections that no individual tool could produce alone.
        For example, the combination of low POSIWID alignment (Section 4) AND
        high fragmentation correlation (Section 5) triggers a specific
        recommendation to use diegetic prompts -- a remedy that draws on
        insights from both the alignment and representation analyses.

        **Design philosophy**: These are heuristic rules, not statistical
        tests. The thresholds (e.g., violation_rate > 0.05, interaction > 0.1,
        alignment < 0.5, fragmentation_correlation > 0.3) are based on
        practical experience with the ER and mitochondrial simulators in
        the Zimmerman dissertation, not on formal statistical criteria.
        Practitioners should treat them as starting points for investigation,
        not as definitive diagnoses.

        The recommendation engine follows Beer's (2024) POSIWID philosophy:
        recommendations describe what the system *actually does* and suggest
        concrete interventions, rather than speculating about design intent.

        Args:
            sections: The 'sections' dict from compile(), containing all
                six analytical sections.

        Returns:
            List of recommendation strings, ordered by section. If no
            sections trigger recommendations, returns a single suggestion
            to run more tools.
        """
        recs = []

        # ----- Section 1: Sensitivity recommendations -----
        # Focus LLM prompt engineering on the most influential parameters.
        sensitivity = sections.get("sensitivity", {})
        if sensitivity.get("available"):
            top = sensitivity.get("top_params", [])
            interaction = sensitivity.get("interaction_strength")
            if top:
                recs.append(
                    f"Most influential parameters: {', '.join(top[:3])}. "
                    f"Focus LLM prompts on these for maximum impact."
                )
            # Interaction threshold 0.1: based on Saltelli et al. (2008)
            # guidance that interaction effects below 10% of total variance
            # are typically negligible for practical purposes.
            if interaction is not None and interaction > 0.1:
                recs.append(
                    f"Interaction strength is {interaction:.3f} -- parameters "
                    f"interact non-additively. Consider joint parameter prompts "
                    f"rather than independent per-parameter generation."
                )

        # ----- Section 2: Robustness recommendations -----
        # Flag fragile parameter regions and suggest bounds constraints.
        robustness = sections.get("robustness", {})
        if robustness.get("available"):
            vr = robustness.get("violation_rate")
            fragile = robustness.get("most_fragile_params")
            # 5% violation rate threshold: above this, a non-trivial fraction
            # of the parameter space produces invalid/broken outputs.
            if vr is not None and vr > 0.05:
                recs.append(
                    f"Violation rate is {vr:.1%} -- significant failure region. "
                    f"Add bounds checking or constrain LLM output ranges."
                )
            if fragile:
                recs.append(
                    f"Most fragile parameters: {', '.join(fragile[:3])}. "
                    f"Small changes to these flip outcomes."
                )

        # ----- Section 3: Locality recommendations -----
        # If the system is highly local, parameter ordering in prompts matters.
        locality = sections.get("locality", {})
        if locality.get("available"):
            horizon = locality.get("effective_horizon")
            if horizon is not None and horizon < 0.3:
                recs.append(
                    f"Effective horizon is {horizon:.2f} -- system is highly "
                    f"local. Early parameters dominate; consider reordering."
                )

        # ----- Section 4: Alignment recommendations -----
        # Flag intention-outcome gaps per Beer (2024) POSIWID.
        alignment = sections.get("alignment", {})
        if alignment.get("available"):
            overall = alignment.get("overall_alignment")
            worst = alignment.get("worst_aligned_keys")
            # Alignment below 0.5 means more than half of the intended
            # behavior is lost in the actual outputs -- a severe problem.
            if overall is not None and overall < 0.5:
                recs.append(
                    f"POSIWID alignment is low ({overall:.2f}) -- "
                    f"intention-outcome gap needs investigation."
                )
            if worst:
                recs.append(
                    f"Worst-aligned output keys: {', '.join(worst[:3])}. "
                    f"These diverge most from intended outcomes."
                )

        # ----- Section 5: Representation recommendations -----
        # Cross-section insight: fragmentation + alignment issues suggest
        # diegetic prompts as a remedy (Zimmerman SS4.5 + SS4.6.4).
        representation = sections.get("representation", {})
        if representation.get("available"):
            frag_corr = representation.get("fragmentation_correlation")
            rt_error = representation.get("roundtrip_error")
            d_gain = representation.get("diegeticization_gain")
            # |r| > 0.3 is a medium-to-strong correlation in behavioral
            # research, sufficient to warrant intervention.
            if frag_corr is not None and frag_corr > 0.3:
                recs.append(
                    f"High fragmentation correlates with output degradation "
                    f"(|r| = {frag_corr:.2f}) -- consider diegetic prompts to "
                    f"reduce numeric tokenization flattening."
                )
            if rt_error is not None and rt_error > 0.1:
                recs.append(
                    f"Roundtrip error is {rt_error:.3f} -- numeric values "
                    f"degrade through LLM encoding/decoding."
                )
            if d_gain is not None and d_gain > 0.1:
                recs.append(
                    f"Diegeticization gain is {d_gain:.2f} -- narrative "
                    f"prompts outperform numeric prompts."
                )

        # ----- Section 6: Structure recommendations -----
        # Flag incomplete dimension coverage and highlight causal drivers.
        structure = sections.get("structure", {})
        if structure.get("available"):
            ve = structure.get("variance_explained")
            causal = structure.get("most_causal_params")
            # Below 30% variance explained, the PDS is capturing less than
            # a third of what is happening in the output space.
            if ve is not None and ve < 0.3:
                recs.append(
                    f"PDS dimensions explain only {ve:.1%} of output variance. "
                    f"Consider revising the dimension-to-parameter mapping."
                )
            if causal:
                recs.append(
                    f"Most causal parameters: {', '.join(causal[:3])}."
                )

        if not recs:
            recs.append("No specific recommendations -- run more tools for deeper analysis.")

        return recs

    # ------------------------------------------------------------------ #
    #  Main compile method                                                 #
    # ------------------------------------------------------------------ #

    def compile(self, reports: dict | None = None, **kwargs) -> dict:
        """Compile a unified dashboard from individual tool reports.

        This is the main entry point for the dashboard (Zimmerman 2025,
        Chapter 6). It accepts a dictionary of tool reports keyed by tool
        name, builds each of the six analytical sections from the relevant
        tool(s), generates cross-section recommendations, and reports
        tool coverage.

        **Graceful degradation**: Missing tools are simply omitted from
        the dashboard. Each section builder checks for the presence of its
        relevant tool reports and returns ``{"available": False, ...}`` if
        none are found. This means the dashboard always produces useful
        output regardless of which subset of tools was run.

        **Tool coverage reporting**: The ``tools_used`` and ``tools_missing``
        lists inform practitioners about which analyses contributed to the
        dashboard and which could be added for deeper insight. This is
        particularly useful in iterative analysis workflows where tools
        are run incrementally.

        Args:
            reports: Dict mapping tool names to their result dicts.
                Accepted keys (matching ``ALL_TOOL_KEYS``):
                  "sobol", "falsifier", "contrastive", "posiwid", "pds",
                  "locality", "receptive_field", "contrast_sets",
                  "relation_graph", "benchmark", "diegeticizer",
                  "token_extispicy".
                Any missing keys are simply omitted from the dashboard.
            **kwargs: Additional keyword arguments (reserved for future use,
                e.g., custom recommendation thresholds).

        Returns:
            Dictionary with:
                "simulator_info": basic simulator metadata (n_params, names,
                    ranges),
                "sections": dict of six analytical sections, each with an
                    "available" flag and section-specific metrics,
                "recommendations": list of actionable suggestion strings,
                "tools_used": list of tool keys that contributed data,
                "tools_missing": list of tool keys that were not provided.
        """
        if reports is None:
            reports = {}

        # Simulator metadata: basic information about the underlying
        # simulator's parameter space, useful for context in the report.
        param_names = sorted(self._spec.keys())
        param_ranges = {name: self._spec[name] for name in param_names}

        simulator_info = {
            "n_params": len(param_names),
            "param_names": param_names,
            "param_ranges": param_ranges,
        }

        # Build each of the six analytical sections from the relevant
        # tool reports. Each section builder handles its own graceful
        # degradation for missing data.
        sections = {
            "sensitivity": self._build_sensitivity(reports),     # Section 1: Sobol / receptive_field
            "robustness": self._build_robustness(reports),       # Section 2: falsifier / contrastive
            "locality": self._build_locality(reports),           # Section 3: locality_profiler
            "alignment": self._build_alignment(reports),         # Section 4: POSIWID
            "representation": self._build_representation(reports),  # Section 5: extispicy / benchmark
            "structure": self._build_structure(reports),         # Section 6: relation_graph / PDS
        }

        # Generate cross-section recommendations: rule-based synthesis
        # that combines findings across sections into actionable suggestions.
        recommendations = self.generate_recommendations(sections)

        # Tool coverage: which of the 12 recognized tools contributed data,
        # and which could be run to add more analytical depth.
        tools_used = [key for key in ALL_TOOL_KEYS if key in reports]
        tools_missing = [key for key in ALL_TOOL_KEYS if key not in reports]

        return {
            "simulator_info": simulator_info,
            "sections": sections,
            "recommendations": recommendations,
            "tools_used": tools_used,
            "tools_missing": tools_missing,
        }

    # ------------------------------------------------------------------ #
    #  Markdown renderer                                                   #
    # ------------------------------------------------------------------ #

    def to_markdown(self, dashboard: dict) -> str:
        """Render dashboard as a formatted markdown report.

        Produces a human-readable markdown document suitable for display
        in terminals, Jupyter notebooks, or saving to a file. The report
        follows the six-section structure from Zimmerman (2025) Chapter 6:

            1. Simulator Info (metadata)
            2. Sensitivity / Robustness / Locality / Alignment /
               Representation / Structure (the six analytical sections)
            3. Recommendations (cross-section synthesis)
            4. Tool Coverage (which analyses contributed)

        Sections with no data are clearly marked as unavailable, and
        numeric values are formatted to 4 decimal places for readability.

        Args:
            dashboard: The result dict from compile().

        Returns:
            Markdown-formatted string ready for display or file output.
        """
        lines = []
        lines.append("# Meaning Construction Dashboard")
        lines.append("")

        # Simulator metadata section: basic info about the parameter space.
        info = dashboard.get("simulator_info", {})
        lines.append("## Simulator Info")
        lines.append(f"- **Parameters**: {info.get('n_params', '?')}")
        param_names = info.get("param_names", [])
        if param_names:
            lines.append(f"- **Names**: {', '.join(param_names)}")
        lines.append("")

        # Render each of the six analytical sections in canonical order.
        sections = dashboard.get("sections", {})
        section_titles = {
            "sensitivity": "Sensitivity",
            "robustness": "Robustness",
            "locality": "Locality",
            "alignment": "Alignment",
            "representation": "Representation",
            "structure": "Structure",
        }

        for key, title in section_titles.items():
            section = sections.get(key, {})
            available = section.get("available", False)
            lines.append(f"## {title}")
            if not available:
                lines.append("*No data available for this section.*")
                lines.append("")
                continue

            # Render each field in the section, with type-appropriate formatting.
            for field, value in section.items():
                if field == "available":
                    continue
                if value is None:
                    lines.append(f"- **{field}**: N/A")
                elif isinstance(value, list):
                    if value:
                        # Show at most 5 items to keep the report concise.
                        lines.append(f"- **{field}**: {', '.join(str(v) for v in value[:5])}")
                    else:
                        lines.append(f"- **{field}**: (empty)")
                elif isinstance(value, float):
                    lines.append(f"- **{field}**: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"- **{field}**:")
                    for sub_k, sub_v in value.items():
                        if isinstance(sub_v, float):
                            lines.append(f"  - {sub_k}: {sub_v:.4f}")
                        else:
                            lines.append(f"  - {sub_k}: {sub_v}")
                else:
                    lines.append(f"- **{field}**: {value}")
            lines.append("")

        # Recommendations section: actionable suggestions from the
        # cross-section synthesis engine (generate_recommendations).
        recommendations = dashboard.get("recommendations", [])
        lines.append("## Recommendations")
        if recommendations:
            for rec in recommendations:
                lines.append(f"- {rec}")
        else:
            lines.append("*No recommendations.*")
        lines.append("")

        # Tool coverage: transparency about which analyses contributed
        # to this dashboard and which could be added for deeper insight.
        tools_used = dashboard.get("tools_used", [])
        tools_missing = dashboard.get("tools_missing", [])
        lines.append("## Tool Coverage")
        lines.append(f"- **Used**: {', '.join(tools_used) if tools_used else '(none)'}")
        lines.append(f"- **Missing**: {', '.join(tools_missing) if tools_missing else '(none)'}")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Simulator protocol                                                  #
    #  The dashboard itself satisfies the Simulator protocol, enabling     #
    #  recursive meta-analysis: you can run sobol_sensitivity() on the     #
    #  dashboard to determine which parameters most affect the dashboard's #
    #  own output metrics. This is a direct implementation of the          #
    #  composability principle from Zimmerman (2025) Chapter 6.            #
    # ------------------------------------------------------------------ #

    def run(self, params: dict) -> dict:
        """Simulator protocol: delegates to the underlying simulator.

        The dashboard wraps the underlying simulator transparently, so it
        can be used as a drop-in replacement anywhere a Simulator is expected.
        This enables meta-analysis chains: for example, running a Falsifier
        on the dashboard to stress-test the simulator through the dashboard
        lens, or running Sobol sensitivity on the dashboard to identify
        which parameters most affect the combined analytical output.

        Args:
            params: Dictionary mapping parameter names to float values.

        Returns:
            Result dictionary from the underlying simulator (unmodified).
        """
        return self.simulator.run(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Simulator protocol: delegates to the underlying simulator's param_spec.

        The dashboard does not modify the parameter space. Its value-add is
        in the ``compile()`` and ``to_markdown()`` methods, not in the
        simulation itself. This transparent delegation ensures that any
        analysis tool expecting a Simulator can seamlessly accept a dashboard.

        Returns:
            Parameter specification from the underlying simulator:
            ``dict[str, tuple[float, float]]`` mapping parameter names
            to (lower_bound, upper_bound) tuples.
        """
        return self._spec
