"""Locality profiler: measures how local a system's behavior is.

Operationalizes Chapter 2 of Zimmerman (2025) on locality. The core claim
of the dissertation is that "meaning is built from locality + relation"
(§2.1): an agent's interpretation of any input depends on what nearby
context is available, how that context is arranged, and how far the
agent's effective context window extends. This module quantifies those
dependencies for arbitrary black-box simulators.

The profiler wraps an existing Simulator with *locality manipulations* --
controlled perturbations that simulate removing, masking, reordering, or
distracting from parts of the input space. By sweeping each manipulation
intensity and measuring how the output degrades, we produce *locality
decay curves* that characterize the system's "effective horizon" and its
sensitivity to information locality.

Theoretical motivation
----------------------
Zimmerman §2.3 ("Effective Context Windows") argues that not all context
contributes equally to meaning construction. In LLMs, attention is
bounded by the context window, but *effective* attention decays well before
the token limit. The same principle applies to any parameterized simulator:
some inputs carry more weight depending on their position, and removing or
scrambling them degrades output quality at different rates. The locality
decay curve is the empirical signature of this phenomenon.

The five default manipulations correspond to different kinds of context
disruption identified in §2.3--§2.5:

    cut_frac (prefix ablation):
        Removes early context entirely, analogous to truncating the beginning
        of a prompt. Maps to §2.3's observation that LLMs privilege recency
        over primacy for long contexts. See also §3.5.3 on "flattening" --
        when qualitative distinctions in early context are collapsed, the
        system may lose access to foundational meaning.

    mask_frac (random masking):
        Replaces randomly selected parameters with their midpoint (the
        "uninformative default"), simulating information loss within an
        otherwise intact context. Measures the system's robustness to
        partial information dropout -- a form of the noise resilience
        discussed in §2.4 ("Locality Under Degradation").

    distractor_strength (additive noise):
        Adds calibrated noise to all parameters, simulating the presence
        of irrelevant or misleading context. This directly tests the
        system's ability to extract signal from noise -- what Zimmerman
        §2.5 calls "salience within the gnogeography" (the sphere of
        accessible meaning). A system with high distractor susceptibility
        has a fragile salioscape.

    target_position (positional emphasis):
        Applies a Gaussian attention window centered at a specified
        position, blending parameters outside the window toward their
        midpoint. Tests whether the system is sensitive to *where* in
        the input space information appears -- the positional locality
        that Zimmerman §2.3 identifies as a key limitation of
        transformer architectures.

    shuffle_window (local reordering):
        Permutes parameter values within a sliding window, preserving
        global structure while disrupting local ordering. Tests whether
        the system depends on fine-grained arrangement of inputs -- a
        computational analog of the word-order sensitivity discussed
        in §2.2.3 ("Tokenization-Induced Distributional Collapse").

Key metrics
-----------
    L50:
        The manipulation intensity where performance drops to 50% of
        baseline, analogous to a half-life in decay kinetics. Lower L50
        means the system is more sensitive to that manipulation. Named
        by analogy with IC50 in pharmacology.

    effective_horizon:
        The largest prefix fraction that can be removed while retaining
        >90% of baseline performance. Directly measures the "effective
        context window" of §2.3 -- the portion of the input space the
        system actually attends to.

    distractor_susceptibility:
        Linear slope of score vs. distractor strength. Negative slope
        means performance degrades with noise (normal); near-zero means
        the system is robust to distractors; positive would indicate
        stochastic resonance.

Composability (the Simulator protocol)
--------------------------------------
The LocalityProfiler is itself a Simulator -- it implements run() and
param_spec() -- so it can be fed to sobol_sensitivity(), Falsifier,
ContrastiveGenerator, or even another LocalityProfiler. This recursive
composability enables second-order questions such as "which locality
manipulation has the largest Sobol total-order index?" (i.e., which
kind of context disruption matters most for this system). This design
follows the "tools that eat their own output" principle articulated in
Zimmerman §4.7.6 (TALOT/OTTITT).

References
----------
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
        in Language, as Implemented in Humans and Large Language Models
        (LLMs)." PhD dissertation, University of Vermont.
        - §2.1: Meaning = locality + relation (foundational claim)
        - §2.2.3: Tokenization-induced distributional collapse
        - §2.3: Effective context windows
        - §2.4: Locality under degradation
        - §2.5: Salience and the gnogeography/salioscape
        - §3.5.3: Flattening -- tokenization collapses qualitative distinctions
        - §4.6.4: Power-Danger-Structure (PDS) dimensions
        - §4.7.6: TALOT/OTTITT -- tools that analyze themselves

    Beer, R.D. (2003). "The Dynamics of Active Categorical Perception in
        an Evolved Model Agent." Adaptive Behavior, 11(4), 209-243.
        (Inspiration for sweep-based behavioral characterization of
        dynamical systems.)

    Saltelli, A. et al. (2010). "Variance based sensitivity analysis of
        model output." Computer Physics Communications, 181(2), 259-270.
        (Sobol analysis can be applied to the manipulation parameters
        themselves when the profiler is treated as a Simulator.)
"""

from __future__ import annotations

import numpy as np


def _midpoint_params(spec: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Return midpoint values for all parameters in the spec.

    The midpoint serves as the "uninformative default" -- the value a
    parameter takes when no information about it is available. This is
    the center of the parameter's feasible range, representing maximum
    ignorance under a uniform prior. Used as the replacement value in
    masking, dropping, and blending operations.

    In Zimmerman's framework (§2.4), the midpoint represents the
    "zero-information" state: the point to which a parameter relaxes
    when its context is removed from the system's effective horizon.
    """
    return {
        name: (lo + hi) / 2.0
        for name, (lo, hi) in spec.items()
    }


def _apply_cut(
    params: dict[str, float],
    spec: dict[str, tuple[float, float]],
    cut_frac: float,
) -> dict[str, float]:
    """Zero out the first cut_frac fraction of parameters (prefix ablation).

    Simulates removing early context: parameters are sorted alphabetically
    (establishing a canonical ordering) and the first cut_frac fraction
    are set to their lower bound, effectively ablating the "prefix" of
    the parameter vector.

    This is the primary manipulation for measuring the "effective context
    window" (Zimmerman §2.3). In NLP terms, it is analogous to truncating
    the beginning of a prompt and observing how model output degrades.
    The decay curve of score vs. cut_frac directly yields the effective
    horizon: the smallest cut_frac that still preserves >90% of baseline.

    The choice to set cut parameters to the lower bound (rather than
    midpoint) represents a harder ablation -- complete removal rather
    than replacement with a neutral value. This design choice makes
    the cut manipulation complementary to mask (which uses midpoint).

    Algorithm:
        1. Sort parameter names alphabetically to establish position.
        2. Compute n_cut = floor(cut_frac * n_params).
        3. Set the first n_cut parameters to their lower bound.

    Args:
        params: Base parameter values.
        spec: Parameter spec with bounds.
        cut_frac: Fraction of parameters to zero out, in [0, 1].

    Returns:
        Modified parameter dict with early parameters set to lower bound.
    """
    names = sorted(params.keys())
    # floor() ensures we only cut whole parameters; partial cuts are not meaningful
    n_cut = int(np.floor(cut_frac * len(names)))
    result = dict(params)
    for i in range(n_cut):
        lo, _hi = spec[names[i]]
        # Set to lower bound (hard ablation, not midpoint replacement)
        result[names[i]] = lo
    return result


def _apply_mask(
    params: dict[str, float],
    spec: dict[str, tuple[float, float]],
    mask_frac: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Replace a random fraction of parameters with midpoint values.

    Selects mask_frac fraction of parameters uniformly at random (without
    replacement) and replaces them with the midpoint of their range --
    the "uninformative default" (see _midpoint_params). This simulates
    *information dropout*: the system receives most of its input intact
    but loses access to specific parameters.

    Unlike prefix ablation (_apply_cut), masking is position-independent:
    it tests whether the system is robust to losing *any* subset of its
    inputs, regardless of where they fall in the parameter ordering.
    This distinction is important for Zimmerman's locality framework
    (§2.3): a system might tolerate losing its prefix (early context)
    but fail when random interior parameters are masked, or vice versa.

    The use of midpoint replacement (rather than lower bound) makes
    masking a softer perturbation than cutting. A masked parameter
    provides the system with a "plausible but uninformative" value,
    whereas a cut parameter provides a boundary value that may be
    clearly anomalous. This design choice follows the principle that
    masking should simulate *ambiguity* (partial information loss),
    not *absence* (complete removal).

    Algorithm:
        1. Sort parameter names alphabetically.
        2. Compute n_mask = floor(mask_frac * n_params).
        3. Randomly select n_mask indices without replacement.
        4. Replace selected parameters with their midpoint values.

    Args:
        params: Base parameter values.
        spec: Parameter spec with bounds.
        mask_frac: Fraction of parameters to mask, in [0, 1].
        rng: Random generator for selecting which parameters to mask.

    Returns:
        Modified parameter dict with masked parameters at midpoint.
    """
    names = sorted(params.keys())
    n_mask = int(np.floor(mask_frac * len(names)))
    if n_mask == 0:
        return dict(params)
    # Without-replacement sampling ensures each parameter is masked at most once
    indices = rng.choice(len(names), size=n_mask, replace=False)
    result = dict(params)
    for idx in indices:
        name = names[idx]
        lo, hi = spec[name]
        # Midpoint = "uninformative default" -- the center of the feasible range
        result[name] = (lo + hi) / 2.0
    return result


def _apply_distractor(
    params: dict[str, float],
    spec: dict[str, tuple[float, float]],
    distractor_strength: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Add uniform noise scaled by distractor_strength to all parameters.

    Simulates the presence of *distractors* -- irrelevant information
    that pollutes the signal. In Zimmerman's framework (§2.5), this
    tests the system's ability to maintain salience within the
    "gnogeography" (the sphere of accessible meaning). A system with
    a robust salioscape will tolerate noise without significant
    performance degradation; a fragile one will degrade rapidly.

    The noise model is uniform on [-strength, +strength] scaled by each
    parameter's range, meaning distractor_strength=1.0 can displace a
    parameter by up to its full range in either direction. Values are
    clipped to stay within bounds -- this clipping introduces a subtle
    asymmetry for parameters near their bounds, but preserves the
    invariant that all perturbed parameters remain physically meaningful.

    The slope of score vs. distractor_strength (the "distractor
    susceptibility") is computed by the profiler as a summary statistic.
    A near-zero slope indicates a system that is *robust to distractors*
    -- it can pick out the signal even in noisy conditions. A steeply
    negative slope indicates fragility.

    Algorithm:
        1. For each parameter, draw noise ~ Uniform(-1, 1).
        2. Scale noise by distractor_strength * (hi - lo).
        3. Add noise to the parameter value.
        4. Clip to [lo, hi] to maintain feasibility.

    Args:
        params: Base parameter values.
        spec: Parameter spec with bounds.
        distractor_strength: Noise scale in [0, 1].
        rng: Random generator for noise.

    Returns:
        Modified parameter dict with noise added, clipped to bounds.
    """
    result = {}
    for name, val in params.items():
        lo, hi = spec[name]
        # Noise is symmetric and proportional to the parameter's range
        noise = rng.uniform(-1.0, 1.0) * distractor_strength * (hi - lo)
        # Clip to bounds to maintain physical meaningfulness
        result[name] = float(np.clip(val + noise, lo, hi))
    return result


def _apply_target_position(
    params: dict[str, float],
    spec: dict[str, tuple[float, float]],
    target_position: float,
) -> dict[str, float]:
    """Apply a Gaussian attention window centered at target_position.

    This is the positional locality manipulation. Parameters sorted by
    name are weighted according to a Gaussian kernel centered at
    target_position along the [0, 1] parameter index. Parameters near
    the center of the Gaussian retain their full original value; those
    in the tails are progressively blended toward the midpoint
    (uninformative default).

    This manipulation directly tests Zimmerman's §2.3 claim about
    positional locality: in transformer-based LLMs, attention decays
    with distance from the query position, creating an effective window
    within which information contributes to output. By sweeping
    target_position from 0.1 (attend to early parameters) to 0.9
    (attend to late parameters), we can map how the system's behavior
    changes when different regions of the input are emphasized.

    The Gaussian kernel with sigma=0.3 was chosen to be broad enough
    to avoid harsh boundaries (which would make results sensitive to
    the exact alphabetical ordering of parameter names) while narrow
    enough to distinguish early vs. middle vs. late emphasis. With
    sigma=0.3, a parameter at distance 0.6 from the target retains
    only ~13% of its original information (weight ~ 0.13).

    Algorithm:
        1. Sort parameter names alphabetically.
        2. Assign each parameter a normalized position pos = i / (n - 1).
        3. Compute Gaussian weight: w = exp(-0.5 * ((pos - target) / sigma)^2).
        4. Blend: param = w * original + (1 - w) * midpoint.

    Args:
        params: Base parameter values.
        spec: Parameter spec with bounds.
        target_position: Position of emphasis in [0.1, 0.9].

    Returns:
        Modified parameter dict with positional weighting applied.
    """
    names = sorted(params.keys())
    n = len(names)
    if n == 0:
        return dict(params)

    result = dict(params)
    for i, name in enumerate(names):
        # Normalized position of this parameter in [0, 1]
        pos = i / max(n - 1, 1)
        # Gaussian attention kernel centered at target_position
        # sigma=0.3 gives a smooth window covering roughly 60% of the range
        sigma = 0.3
        weight = np.exp(-0.5 * ((pos - target_position) / sigma) ** 2)
        # Blend between original value (full information) and midpoint (zero information)
        lo, hi = spec[name]
        midpoint = (lo + hi) / 2.0
        result[name] = float(weight * params[name] + (1.0 - weight) * midpoint)
    return result


def _apply_shuffle(
    params: dict[str, float],
    spec: dict[str, tuple[float, float]],
    shuffle_window: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Randomly permute parameter values within a sliding window.

    Simulates *local reordering* of information while preserving global
    structure. The window size is shuffle_window * n_params; within each
    non-overlapping window, parameter *values* are randomly permuted
    among the *names* in that window. The parameter names retain their
    positions, but the values assigned to them are scrambled locally.

    This manipulation tests whether the system depends on the precise
    assignment of values to parameter names within a local neighborhood.
    It is the computational analog of testing word-order sensitivity in
    NLP: can the system extract the same meaning from a locally
    scrambled version of its input? Zimmerman §2.2.3 discusses how
    tokenization-induced distributional collapse can make LLMs
    insensitive to word order within short spans; this manipulation
    tests the same phenomenon in simulator parameter spaces.

    At shuffle_window=0, no shuffling occurs (identity). At
    shuffle_window=1.0, the entire parameter vector is globally
    permuted. Intermediate values test the transition from local
    to global disorder.

    Note: values are clipped to bounds after shuffling because a value
    that is valid for one parameter may be out of range for the
    parameter it gets shuffled into.

    Algorithm:
        1. Sort parameter names alphabetically and extract their values.
        2. Compute window_size = ceil(shuffle_window * n_params).
        3. Partition values into non-overlapping windows of that size.
        4. Shuffle values within each window independently.
        5. Reassign shuffled values to parameter names; clip to bounds.

    Args:
        params: Base parameter values.
        spec: Parameter spec with bounds.
        shuffle_window: Window size as fraction of total params, in [0, 1].
        rng: Random generator for permutation.

    Returns:
        Modified parameter dict with locally shuffled values.
    """
    names = sorted(params.keys())
    n = len(names)
    if n <= 1 or shuffle_window < 1e-9:
        return dict(params)

    values = [params[name] for name in names]
    window_size = max(1, int(np.ceil(shuffle_window * n)))

    # Non-overlapping windows: step by window_size to avoid double-shuffling
    for start in range(0, n, max(1, window_size)):
        end = min(start + window_size, n)
        chunk = values[start:end]
        rng.shuffle(chunk)  # in-place Fisher-Yates shuffle
        values[start:end] = chunk

    result = {}
    for i, name in enumerate(names):
        lo, hi = spec[name]
        # Clip is necessary because a shuffled value may come from a parameter
        # with a different range than the one it is now assigned to
        result[name] = float(np.clip(values[i], lo, hi))
    return result


def _interpolate_l50(values: list[float], scores: list[float], baseline: float) -> float:
    """Find the manipulation value where score drops below 50% of baseline.

    L50 is the manipulation intensity at which the system's performance
    falls to half its unperturbed level -- a "half-life" for behavioral
    integrity under that manipulation. The name follows the convention
    of pharmacological dose-response curves (IC50, EC50), adapted here
    to characterize information-processing systems.

    Lower L50 means the system is more sensitive to the manipulation:
    even mild perturbation causes significant degradation. Higher L50
    means the system is robust: it can tolerate strong perturbation
    before losing half its performance.

    Uses linear interpolation between adjacent sweep points to estimate
    the crossing point. If the score never drops below 50%, returns the
    maximum sweep value as a conservative lower bound on L50.

    Algorithm:
        1. Compute threshold = 0.5 * baseline.
        2. Walk the sweep curve looking for a crossing (score drops
           below threshold between consecutive points).
        3. Linearly interpolate within that interval.
        4. If no crossing found, return the last sweep value.

    Args:
        values: Sorted manipulation values (ascending).
        scores: Corresponding mean scores at each sweep value.
        baseline: Baseline score at manipulation value 0 (no perturbation).

    Returns:
        L50 value (float), or the maximum sweep value if score never
        drops below 50%.
    """
    # Guard against zero or near-zero baseline (undefined ratio)
    if baseline < 1e-12:
        return 0.0

    threshold = 0.5 * baseline

    for i in range(len(values) - 1):
        if scores[i] >= threshold and scores[i + 1] < threshold:
            # Linear interpolation to find exact crossing point
            # frac is the fractional position within the interval [values[i], values[i+1]]
            frac = (threshold - scores[i]) / (scores[i + 1] - scores[i])
            return float(values[i] + frac * (values[i + 1] - values[i]))

    # Score never dropped below 50% -- the system is robust to this manipulation
    # across the entire sweep range. Return the max sweep value as a lower bound.
    return float(values[-1]) if values else 1.0


class LocalityProfiler:
    """Measures how local a system's behavior is via manipulation sweeps.

    The LocalityProfiler is the primary tool for operationalizing
    Zimmerman Chapter 2's theory of locality. It wraps any black-box
    simulator and systematically probes how different kinds of context
    disruption affect the system's output. The result is a set of
    *locality decay curves* -- one per manipulation -- that together
    characterize the system's locality profile.

    Theoretical grounding (Zimmerman §2.1--§2.5)
    ---------------------------------------------
    Zimmerman argues that meaning is constructed from two primitives:
    *locality* (what context is available) and *relation* (how that
    context is composed). This class isolates the locality dimension
    by holding relation fixed (the simulator's internal logic) and
    systematically degrading locality (the input context).

    The five default manipulations correspond to five distinct failure
    modes of locality:
        - Ablation (cut): early context is simply absent.
        - Masking (mask): random context is replaced with neutral values.
        - Distraction (distractor): all context is perturbed by noise.
        - Positional emphasis (target_position): attention is focused
          on a specific region of the input, letting the rest fade.
        - Local reordering (shuffle): fine-grained ordering is disrupted
          while global structure is preserved.

    Composability (Simulator protocol)
    -----------------------------------
    The profiler satisfies the Simulator protocol (run() + param_spec()),
    where the "parameters" are the manipulation intensities. This means
    it can be analyzed by any zimmerman-toolkit tool:
        - sobol_sensitivity(LocalityProfiler(sim)) reveals which
          manipulation has the highest total-order index.
        - Falsifier(LocalityProfiler(sim)) searches for manipulation
          combinations that cause catastrophic failure.
        - ContrastiveGenerator(LocalityProfiler(sim)) finds pairs of
          manipulation settings that produce maximally different outputs.

    This recursive composability embodies Zimmerman's TALOT/OTTITT
    principle (§4.7.6): the tools are designed to analyze themselves.

    Args:
        simulator: Any Simulator-compatible object (the system under test).
            Must implement run(params: dict) -> dict and
            param_spec() -> dict[str, tuple[float, float]].
        manipulations: Dict of manipulation names to callables. Each callable
            takes (params, spec, value, rng) and returns modified params.
            If None, uses the five default manipulations described above.

    Example:
        from tests.conftest import LinearSimulator
        sim = LinearSimulator(d=6)
        profiler = LocalityProfiler(sim)
        report = profiler.profile(
            task={"base_params": {f"x{i}": 0.7 for i in range(6)}},
            n_seeds=10,
        )
        print(report["effective_horizon"])

    References:
        Zimmerman (2025), Chapter 2: Locality and context windows.
        Beer, R.D. (2003). Sweep-based characterization of dynamical systems.
    """

    def __init__(self, simulator, manipulations: dict | None = None):
        self.simulator = simulator
        # Cache the inner simulator's param_spec for use in manipulations.
        # This is the "ground truth" parameter space that the manipulations
        # will perturb.
        self._spec = simulator.param_spec()

        if manipulations is not None:
            self.manipulations = dict(manipulations)
        else:
            # Use the five default manipulations described in the module docstring.
            # Each maps a manipulation intensity (float) to a parameter perturbation.
            self.manipulations = self._default_manipulations()

    @staticmethod
    def _default_manipulations() -> dict:
        """Return the five default locality manipulation functions.

        Each function has a uniform signature:
            (params, spec, value, rng) -> modified_params

        The uniform signature allows the profiler to iterate over all
        manipulations generically, even though some (cut, target_position)
        are deterministic and ignore the rng argument. The rng argument
        is always passed for interface consistency.

        The five manipulations are ordered by "severity" in a conceptual
        sense: cut is the harshest (complete removal), followed by mask
        (neutral replacement), shuffle (local disorder), distractor
        (global noise), and target_position (attention reweighting).
        However, the actual severity depends on the specific system
        being profiled -- that is precisely what the profiler measures.
        """
        return {
            "cut_frac": lambda params, spec, val, rng: _apply_cut(params, spec, val),
            "mask_frac": lambda params, spec, val, rng: _apply_mask(params, spec, val, rng),
            "distractor_strength": lambda params, spec, val, rng: _apply_distractor(params, spec, val, rng),
            "target_position": lambda params, spec, val, rng: _apply_target_position(params, spec, val),
            "shuffle_window": lambda params, spec, val, rng: _apply_shuffle(params, spec, val, rng),
        }

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return the manipulation parameter spec (Simulator protocol).

        Each manipulation parameter is in its natural range. This makes
        the LocalityProfiler a valid Simulator for use with
        sobol_sensitivity(), Falsifier, ContrastiveGenerator, etc.

        Note that target_position is bounded to [0.1, 0.9] rather than
        [0, 1] to avoid degenerate edge cases where the Gaussian
        attention window is centered entirely outside the parameter
        range, which would collapse all parameters to midpoint.

        Returns:
            Dict mapping manipulation names to (low, high) bounds.
        """
        return {
            "cut_frac": (0.0, 1.0),            # Fraction of prefix to ablate
            "mask_frac": (0.0, 1.0),            # Fraction of params to mask
            "distractor_strength": (0.0, 1.0),  # Noise amplitude (relative to range)
            "target_position": (0.1, 0.9),      # Center of Gaussian attention window
            "shuffle_window": (0.0, 1.0),       # Shuffle window as fraction of params
        }

    def run(self, params: dict) -> dict:
        """Simulator protocol: apply manipulations and run the inner simulator.

        This is the core of the profiler's composability. When called via
        the Simulator protocol (e.g., by sobol_sensitivity()), the
        "parameters" are manipulation intensities, and the "output" is
        the inner simulator's result under those manipulations.

        The base parameters are always the midpoint of the inner simulator's
        spec -- the "uninformative default." This ensures that the profiler
        is measuring the *effect of manipulations* starting from a neutral
        baseline, not conflating manipulation effects with parameter choice.

        Manipulations are applied sequentially in iteration order. When
        multiple manipulations have non-zero values, they compose: e.g.,
        cut_frac=0.3 followed by distractor_strength=0.5 first ablates
        30% of the parameters, then adds noise to the remaining ones.
        This sequential composition means manipulation order matters --
        an important subtlety when interpreting Sobol interaction terms.

        Algorithm:
            1. Derive a deterministic RNG from the parameter values (for
               reproducibility: identical inputs always produce identical
               outputs, satisfying the deterministic simulation invariant).
            2. Start from midpoint base parameters.
            3. Apply each manipulation with non-zero intensity in sequence.
            4. Run the inner simulator on the modified parameters.
            5. Annotate the result with which manipulations were active.

        Args:
            params: Dict mapping manipulation names to float values.
                Keys should match param_spec() names.

        Returns:
            The inner simulator's result dict, augmented with a
            "manipulations_applied" key listing which manipulations
            had non-zero values.
        """
        # Deterministic RNG seeded from the parameter values themselves.
        # This ensures reproducibility: the same manipulation intensities
        # always produce the same stochastic perturbations (mask selection,
        # distractor noise, shuffle permutation).
        rng = np.random.default_rng(hash(tuple(sorted(params.items()))) % (2**31))
        # Start from the midpoint -- the "zero-information" baseline (§2.4)
        base_params = _midpoint_params(self._spec)

        modified = dict(base_params)
        manipulations_applied = []

        for manip_name, manip_fn in self.manipulations.items():
            val = params.get(manip_name, 0.0)
            # Skip manipulations with effectively zero intensity (< 1e-9)
            # to avoid unnecessary computation and floating-point noise
            if abs(val) > 1e-9:
                modified = manip_fn(modified, self._spec, val, rng)
                manipulations_applied.append(manip_name)

        result = self.simulator.run(modified)
        result["manipulations_applied"] = manipulations_applied
        return result

    def _get_score(self, result: dict) -> float:
        """Extract a scalar performance score from a simulation result.

        The locality decay curve requires a single scalar metric that
        summarizes "how well the system is performing." This method
        implements a priority-based extraction strategy:

            1. 'fitness' -- the standard key for evolutionary robotics
               simulators (e.g., the ER project's distance-traveled metric).
            2. 'score' -- a generic performance key.
            3. 'y' -- used by simple function-wrapping simulators.
            4. Fallback: mean of all finite numeric values in the result,
               providing a crude but universal summary.

        The fallback is deliberately conservative: by averaging all
        numeric outputs, it avoids silently returning 0.0 for simulators
        that use non-standard output keys. However, it may produce
        misleading results if the result dict contains both performance
        and diagnostic values (e.g., 'fitness' and 'n_steps'). Users
        with non-standard output keys should override this method or
        provide a custom scorer.

        Args:
            result: Simulation result dict.

        Returns:
            Float score value.
        """
        # Priority-ordered key lookup
        for key in ("fitness", "score", "y"):
            if key in result:
                val = result[key]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    return float(val)

        # Fallback: mean of all finite numeric values in the result dict
        vals = []
        for key, val in result.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                fval = float(val)
                if np.isfinite(fval):
                    vals.append(fval)
        return float(np.mean(vals)) if vals else 0.0

    def profile(
        self,
        task: dict,
        sweeps: dict[str, list[float]] | None = None,
        n_seeds: int = 30,
        seed: int = 42,
    ) -> dict:
        """Run locality profiling over manipulation sweeps.

        This is the main analysis entry point. For each manipulation in
        the sweep schedule, it evaluates the simulator at multiple
        intensities, each repeated over n_seeds random seeds (for
        stochastic manipulations like mask and distractor). The result
        is a comprehensive locality report.

        The profiling algorithm has four phases:

        Phase 1 -- Baseline:
            Run the simulator n_seeds times with the unperturbed base
            parameters to establish the baseline score. This is the
            reference against which all degradation is measured.

        Phase 2 -- Sweep each manipulation:
            For each (manipulation, intensity) pair, apply the
            manipulation to the base parameters and run the simulator
            n_seeds times. Record the mean and standard deviation of
            scores to produce the locality decay curve.

        Phase 3 -- Distractor susceptibility:
            If distractor_strength was not included in the explicit
            sweep schedule, run a separate sweep to compute the linear
            slope of score vs. distractor strength. This slope is a
            single-number summary of the system's noise robustness
            (Zimmerman §2.5).

        Phase 4 -- Effective horizon:
            From the cut_frac curve, find the largest fraction that
            can be ablated while retaining >90% of baseline. This
            directly measures the "effective context window" of
            Zimmerman §2.3.

        Computational cost:
            Total simulations = n_seeds * (1 + sum(len(sweep_values)
            for each manipulation)). For the default sweeps and
            n_seeds=30, this is approximately 30 * (1 + 6 + 5) = 360
            simulations plus distractor and cut sweeps if not included.

        Args:
            task: Dict with 'base_params' (baseline parameters to perturb).
                These should be a meaningful operating point, not just
                the midpoint -- the profiler measures locality relative
                to *this specific* input configuration.
            sweeps: Dict of manipulation_name to list of values to sweep.
                Default: {"cut_frac": [0, .1, .2, .4, .6, .8],
                         "target_position": [.1, .3, .5, .7, .9]}.
            n_seeds: Number of random seeds per sweep point. Higher values
                reduce noise in the decay curves but increase cost linearly.
            seed: Base random seed for reproducibility.

        Returns:
            Dict with:
                "curves": {manipulation_name: [(value, mean_score, std_score), ...]},
                "L50": {manipulation_name: float},
                "distractor_susceptibility": float (slope of score vs. distractor_strength),
                "effective_horizon": float (largest cut_frac retaining >90% baseline),
                "n_sims": int (total simulator evaluations performed),
        """
        base_params = task["base_params"]

        if sweeps is None:
            # Default sweep schedule: cut_frac tests ablation sensitivity
            # at 6 levels; target_position tests positional emphasis at 5
            # evenly spaced positions across the parameter vector.
            sweeps = {
                "cut_frac": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
                "target_position": [0.1, 0.3, 0.5, 0.7, 0.9],
            }

        rng = np.random.default_rng(seed)
        n_sims = 0

        # ===================================================================
        # Phase 1: Compute baseline score (no manipulation applied)
        # -------------------------------------------------------------------
        # The baseline is the unperturbed simulator output, averaged over
        # n_seeds runs. For deterministic simulators, all runs will produce
        # identical results (the averaging is for stochastic simulators).
        # All L50 and effective_horizon computations are relative to this.
        # ===================================================================
        baseline_scores = []
        for s in range(n_seeds):
            result = self.simulator.run(dict(base_params))
            baseline_scores.append(self._get_score(result))
            n_sims += 1
        baseline_mean = float(np.mean(baseline_scores))

        # ===================================================================
        # Phase 2: Sweep each manipulation in the schedule
        # -------------------------------------------------------------------
        # For each (manipulation, intensity) pair, apply the manipulation
        # n_seeds times with different random seeds and record the mean
        # and std of scores. This produces the locality decay curve for
        # each manipulation.
        # ===================================================================
        curves = {}
        l50 = {}

        for manip_name, sweep_values in sweeps.items():
            if manip_name not in self.manipulations:
                continue

            manip_fn = self.manipulations[manip_name]
            curve_points = []

            for val in sweep_values:
                scores = []
                for s in range(n_seeds):
                    # Fresh RNG per seed to ensure independence across seeds
                    # while maintaining reproducibility within the profile() call
                    seed_rng = np.random.default_rng(rng.integers(0, 2**31))
                    modified = manip_fn(dict(base_params), self._spec, val, seed_rng)
                    result = self.simulator.run(modified)
                    scores.append(self._get_score(result))
                    n_sims += 1

                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
                curve_points.append((float(val), mean_score, std_score))

            curves[manip_name] = curve_points

            # Compute L50 for this manipulation: the intensity at which
            # performance drops to 50% of baseline (see _interpolate_l50)
            sweep_vals = [pt[0] for pt in curve_points]
            sweep_means = [pt[1] for pt in curve_points]
            l50[manip_name] = _interpolate_l50(sweep_vals, sweep_means, baseline_mean)

        # ===================================================================
        # Phase 3: Compute distractor susceptibility (§2.5)
        # -------------------------------------------------------------------
        # If distractor_strength was not in the explicit sweep schedule,
        # run a dedicated sweep. Then compute the linear slope of score
        # vs. distractor_strength using numpy's polyfit. The slope
        # summarizes how rapidly performance degrades with noise.
        #
        # A negative slope = normal degradation (more noise = worse).
        # A near-zero slope = robust to distractors.
        # A positive slope = possible stochastic resonance (rare).
        # ===================================================================
        distractor_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        if "distractor_strength" in sweeps:
            # Reuse the curve already computed in Phase 2
            distractor_values = sweeps["distractor_strength"]
            distractor_scores = [pt[1] for pt in curves.get("distractor_strength", [])]
        else:
            # Run a separate distractor sweep
            distractor_scores = []
            manip_fn = self.manipulations.get("distractor_strength")
            if manip_fn is not None:
                curve_points = []
                for val in distractor_values:
                    scores = []
                    for s in range(n_seeds):
                        seed_rng = np.random.default_rng(rng.integers(0, 2**31))
                        modified = manip_fn(dict(base_params), self._spec, val, seed_rng)
                        result = self.simulator.run(modified)
                        scores.append(self._get_score(result))
                        n_sims += 1
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    distractor_scores.append(mean_score)
                    curve_points.append((float(val), mean_score, std_score))
                curves["distractor_strength"] = curve_points
                l50["distractor_strength"] = _interpolate_l50(
                    distractor_values, distractor_scores, baseline_mean
                )

        # Distractor susceptibility: slope of the best-fit line (score vs. strength).
        # Uses np.polyfit (degree 1) rather than manual slope calculation for
        # numerical stability. The [0] index extracts the slope coefficient.
        distractor_susceptibility = 0.0
        if len(distractor_values) >= 2 and len(distractor_scores) >= 2:
            x = np.array(distractor_values[:len(distractor_scores)])
            y = np.array(distractor_scores)
            # Guard against degenerate case where all x values are identical
            if len(x) > 1 and np.std(x) > 1e-12:
                slope = float(np.polyfit(x, y, 1)[0])
                distractor_susceptibility = slope

        # ===================================================================
        # Phase 4: Compute effective horizon (§2.3)
        # -------------------------------------------------------------------
        # The effective horizon is the largest cut_frac that still preserves
        # >90% of baseline performance. It directly measures how much of the
        # input space the system actually attends to -- the "effective
        # context window" of Zimmerman §2.3.
        #
        # A low effective_horizon (e.g., 0.0) means the system uses its
        # entire input -- removing even 10% causes significant degradation.
        # A high effective_horizon (e.g., 0.6) means 60% of the input can
        # be ablated with minimal impact -- the system only "reads" the
        # last 40% of its parameters.
        # ===================================================================
        effective_horizon = 1.0
        if "cut_frac" in sweeps:
            cut_values = sweeps["cut_frac"]
        else:
            cut_values = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

        if "cut_frac" in curves:
            # Reuse the cut curve already computed in Phase 2
            cut_curve = curves["cut_frac"]
            threshold_90 = 0.9 * baseline_mean
            # Walk the curve until performance drops below 90% of baseline.
            # The last value above threshold is the effective horizon.
            for val, mean_score, _std in cut_curve:
                if mean_score >= threshold_90:
                    effective_horizon = float(val)
                else:
                    break
        else:
            # Run a separate cut sweep if not already done
            manip_fn = self.manipulations.get("cut_frac")
            if manip_fn is not None:
                cut_curve = []
                for val in cut_values:
                    scores = []
                    for s in range(n_seeds):
                        seed_rng = np.random.default_rng(rng.integers(0, 2**31))
                        modified = manip_fn(dict(base_params), self._spec, val, seed_rng)
                        result = self.simulator.run(modified)
                        scores.append(self._get_score(result))
                        n_sims += 1
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    cut_curve.append((float(val), mean_score, std_score))

                curves["cut_frac"] = cut_curve
                l50["cut_frac"] = _interpolate_l50(
                    cut_values, [pt[1] for pt in cut_curve], baseline_mean
                )

                threshold_90 = 0.9 * baseline_mean
                for val, mean_score, _std in cut_curve:
                    if mean_score >= threshold_90:
                        effective_horizon = float(val)
                    else:
                        break

        return {
            "curves": curves,                                          # Raw decay curves per manipulation
            "L50": l50,                                                # Half-performance intensity per manipulation
            "distractor_susceptibility": float(distractor_susceptibility),  # Slope of score vs. noise
            "effective_horizon": float(effective_horizon),             # Effective context window (§2.3)
            "n_sims": n_sims,                                          # Total simulator evaluations
        }
