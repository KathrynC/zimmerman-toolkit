"""Reversible translation between parameter vectors and narrative descriptions.

Converts numeric simulation parameters into discretized narrative labels
("very low", "low", "medium", "high", "very high") and back. This implements
the diegetic/supradiegetic distinction from Zimmerman (2025) SS2.2.3: LLMs
construct meaning from diegetic content (distributional semantics), not from
supradiegetic form. Numbers are particularly vulnerable to tokenization-induced
flattening (SS3.5.3).

Zimmerman SS3.5.3 on "narrative flattening": when parameters are translated
into natural language, structural information is lost.  A parameter vector
like [0.82, 0.15, 0.97] contains exact positional and magnitude information.
Its narrative equivalent -- "high speed, very low armor, very high damage" --
preserves ordinal relationships but discards cardinal precision.  This module
systematically quantifies that information loss.

The core design principle is the **reversibility principle**: a good
diegeticization should be invertible.  The roundtrip

    params -> narrative -> params'

should preserve meaning, with ``params'`` being a faithful (if approximate)
reconstruction of ``params``.  The L2 distance between ``params`` and ``params'``
(the "roundtrip error") quantifies information loss.  Perfect reversibility
(roundtrip error = 0) occurs only when every parameter value is exactly at a
bin midpoint.  For all other values, the roundtrip error is bounded by the
bin geometry: each parameter's error is at most half the bin width.

Zimmerman argues that narrative framing changes LLM behavior: the same
information presented as a story ("the robot moves with high speed and low
caution") vs as raw numbers ("speed=0.82, caution=0.15") produces different
outputs from the same LLM.  This module enables controlled experiments to
measure that effect by providing matched numeric and narrative representations
of identical parameter content.

The diegeticizer creates equal-width bins over each parameter's range and maps
numeric values to bin labels. The reverse mapping uses bin midpoints
(deterministic mode) or uniform sampling within the bin (sample mode).

The roundtrip error (params -> narrative -> params') quantifies the information
loss inherent in discretization. This is a fundamental tradeoff: fewer bins
lose more precision but produce more stable narrative labels; more bins
preserve precision but approach raw numeric representation (defeating the
purpose of diegeticization).

This module also satisfies the Simulator protocol itself: calling run()
diegeticizes the input, re-diegeticizes to recovered params, and runs the
underlying simulator on those recovered params. This lets you measure how
much discretization degrades simulation outputs, and enables recursive
interrogation of the diegeticizer by other Zimmerman tools (Sobol
sensitivity, falsifier, contrastive pairs, POSIWID, PDS).

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont.
"""

from __future__ import annotations

import numpy as np


# Default bin labels for common bin counts.
# These are the "narrative vocabulary" -- the set of natural-language terms
# that replace numeric values.  The labels are chosen for maximal semantic
# discriminability: an LLM can clearly distinguish "very low" from "high"
# in a way that it cannot reliably distinguish 0.15 from 0.73.
#
# The 3-bin, 5-bin, and 7-bin schemes correspond to different granularity
# levels.  5 bins is the default because it provides a good balance:
# - 3 bins: very coarse, large roundtrip error, but maximally stable labels.
# - 5 bins: moderate precision, well-known "Likert scale" semantics that
#   LLMs encounter frequently in training data.
# - 7 bins: finer precision, but the "moderately low" and "moderately high"
#   labels are less semantically distinct, approaching the distributional
#   collapse threshold where LLMs confuse adjacent categories.
_DEFAULT_LABELS = {
    3: ["low", "medium", "high"],
    5: ["very low", "low", "medium", "high", "very high"],
    7: ["very low", "low", "moderately low", "medium",
        "moderately high", "high", "very high"],
}


def _make_labels(n_bins: int) -> list[str]:
    """Return bin labels for the given number of bins.

    Uses predefined semantically rich labels for common counts (3, 5, 7).
    For other counts, generates numbered labels ("bin_0", "bin_1", ...),
    which lack semantic richness but maintain the structural protocol.

    The predefined labels are preferred because they leverage the LLM's
    distributional knowledge of natural-language quantity terms.  "High"
    carries more meaning than "bin_3" to a language model trained on
    natural text (Zimmerman SS2.2.3).

    Args:
        n_bins: Number of bins. Must be >= 2.

    Returns:
        List of string labels, one per bin.
    """
    if n_bins in _DEFAULT_LABELS:
        return list(_DEFAULT_LABELS[n_bins])
    return [f"bin_{i}" for i in range(n_bins)]


class Diegeticizer:
    """Reversible translation between parameter vectors and narrative labels.

    For each parameter in the simulator's param_spec, divides the range
    [lo, hi] into *n_bins* equal-width bins labeled with narrative terms.
    The forward pass (diegeticize) maps a float to its bin label; the
    reverse pass (re_diegeticize) maps the label back to the bin midpoint
    (deterministic) or a uniform sample within the bin (sample mode).

    The **lexicon** system provides domain-specific "story handles" that
    enrich the narrative representation.  Without a lexicon, parameter
    names are used directly (e.g., ``{"x0": "high"}``).  With a lexicon,
    parameters receive semantically meaningful names that provide context
    to an LLM:

        lexicon = {"x0": "sprint speed", "x1": "armor thickness"}
        -> {"sprint speed": "high", "armor thickness": "low"}

    Zimmerman (2025) SS3.5.3 argues that these "story handles" are not
    merely cosmetic: they activate the LLM's domain-specific distributional
    knowledge.  An LLM knows that "high sprint speed" and "low armor"
    together suggest a fast, fragile agent -- semantic associations that
    are invisible in the raw numeric representation {"x0": 0.82, "x1": 0.15}.

    The lexicon must be invertible (no duplicate values) to support the
    reverse mapping from narrative handles back to parameter names.

    Args:
        simulator: Any Simulator-compatible object with run() and
            param_spec() methods.
        lexicon: Optional dict mapping parameter names to narrative
            "story handles" (e.g., {"spring_constant": "spring stiffness"}).
            If None, parameter names are used directly.
        n_bins: Number of discretization bins. Default 5.

    Example:
        dieg = Diegeticizer(my_sim, n_bins=5)
        result = dieg.diegeticize({"speed": 0.73, "armor": 0.2})
        # result["narrative"] == {"speed": "high", "armor": "low"}
        recovered = dieg.re_diegeticize(result["narrative"])
        # recovered["params"] == {"speed": 0.7, "armor": 0.2} (midpoints)
    """

    def __init__(
        self,
        simulator,
        lexicon: dict[str, str] | None = None,
        n_bins: int = 5,
    ):
        self.simulator = simulator
        # Cache the parameter specification: maps param names to (lo, hi) bounds.
        self._spec = simulator.param_spec()
        # The lexicon maps internal parameter names to semantically rich
        # "story handles" for the narrative representation.
        self.lexicon = lexicon or {}
        self.n_bins = n_bins
        self._labels = _make_labels(n_bins)

        # Precompute bin edges and midpoints for each parameter.
        # This avoids redundant np.linspace calls during the hot path of
        # diegeticize/re_diegeticize, which matters for batch operations.
        #
        # For a parameter with range [lo, hi] and n_bins=5:
        #   edges = [lo, lo+w, lo+2w, lo+3w, lo+4w, hi]  (6 edges for 5 bins)
        #   midpoints = [lo+w/2, lo+3w/2, lo+5w/2, lo+7w/2, lo+9w/2]
        #   width = w = (hi - lo) / n_bins
        self._bins: dict[str, dict] = {}
        for name, (lo, hi) in self._spec.items():
            edges = np.linspace(lo, hi, n_bins + 1)
            midpoints = (edges[:-1] + edges[1:]) / 2.0
            self._bins[name] = {
                "edges": edges,
                "midpoints": midpoints,
                "lo": lo,
                "hi": hi,
                "width": (hi - lo) / n_bins,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle(self, param_name: str) -> str:
        """Return the narrative handle for a parameter name.

        If a lexicon is provided and contains this parameter, returns the
        story handle (e.g., "sprint speed" instead of "x0").  Otherwise
        returns the raw parameter name.  This is the forward direction of
        the lexicon lookup.
        """
        return self.lexicon.get(param_name, param_name)

    def _param_name_from_handle(self, handle: str) -> str:
        """Reverse-lookup: narrative handle -> parameter name.

        Given a story handle (e.g., "sprint speed"), returns the
        corresponding internal parameter name (e.g., "x0").  If the handle
        is not in the lexicon, assumes it is already a raw parameter name.

        The reverse lexicon is built lazily on first call and cached for
        subsequent lookups.  The lexicon must be invertible (no duplicate
        values) for this to work correctly -- duplicate story handles would
        cause one mapping to silently shadow the other.
        """
        # Build reverse map lazily if lexicon is non-trivial.
        if not hasattr(self, "_reverse_lexicon"):
            self._reverse_lexicon = {v: k for k, v in self.lexicon.items()}
        return self._reverse_lexicon.get(handle, handle)

    def _value_to_bin_index(self, value: float, param_name: str) -> int:
        """Map a numeric value to its bin index for the given parameter.

        Uses ``np.searchsorted`` with ``side='right'`` on the interior
        edges to find which bin contains the value.  The result is clamped
        to [0, n_bins - 1] to handle floating-point edge cases where the
        value equals the upper bound exactly.

        The bin assignment uses left-closed, right-open intervals for all
        bins except the last, which is closed on both sides:
            bin_0: [edge_0, edge_1)
            bin_1: [edge_1, edge_2)
            ...
            bin_{n-1}: [edge_{n-1}, edge_n]
        """
        info = self._bins[param_name]
        # np.searchsorted with side='right' gives the first edge > value.
        idx = int(np.searchsorted(info["edges"][1:], value, side="right"))
        # Clamp to valid range [0, n_bins - 1].
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _label_to_bin_index(self, label: str) -> int:
        """Map a narrative label back to its bin index.

        This is the inverse of indexing into self._labels.  Raises
        ValueError if the label is not recognized, which typically
        indicates a mismatch between the n_bins used for diegeticization
        and re-diegeticization, or a corrupted narrative representation.
        """
        try:
            return self._labels.index(label)
        except ValueError:
            raise ValueError(
                f"Unknown bin label '{label}'. Valid labels: {self._labels}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diegeticize(self, params: dict) -> dict:
        """Convert numeric params to narrative representation (forward pass).

        This is the core diegeticization operation: translating a parameter
        vector from supradiegetic form (exact floats) into diegetic form
        (narrative labels).  For example:

            {"exercise_level": 0.82} -> {"exercise_level": "very high"}

        or with a lexicon:

            {"x0": 0.82} -> {"daily exercise": "very high"}

        The operation is lossy: all values within a bin map to the same
        label.  The information lost is bounded by the bin width and can be
        precisely measured via ``roundtrip_error()``.

        The returned dict includes:
        - "narrative": the narrative representation (handles -> labels).
        - "numeric": a copy of the original params (for reference).
        - "bins_used": diagnostic info showing which bin each param fell
          into, including the bin's numeric range.

        Args:
            params: Dict mapping parameter names to float values.

        Returns:
            Dict with:
                "narrative": {handle: bin_label, ...},
                "numeric": dict (copy of original params),
                "bins_used": {param_name: {"label": str, "range": (lo, hi)}, ...},
        """
        narrative = {}
        bins_used = {}

        for name in self._spec:
            value = float(params[name])
            # Determine which bin this value falls into.
            idx = self._value_to_bin_index(value, name)
            # Map bin index to narrative label.
            label = self._labels[idx]
            # Apply lexicon to get the story handle for this parameter.
            handle = self._handle(name)
            narrative[handle] = label

            # Record diagnostic info: which bin was used and its numeric range.
            info = self._bins[name]
            bin_lo = float(info["edges"][idx])
            bin_hi = float(info["edges"][idx + 1])
            bins_used[name] = {
                "label": label,
                "range": (bin_lo, bin_hi),
            }

        return {
            "narrative": narrative,
            "numeric": dict(params),
            "bins_used": bins_used,
        }

    def re_diegeticize(
        self,
        narrative: dict,
        mode: str = "deterministic",
        seed: int = 42,
    ) -> dict:
        """Convert narrative back to numeric params (reverse/inverse pass).

        This is the inverse of diegeticize(): translating narrative labels
        back into numeric parameter values.  For example:

            {"daily exercise": "very high"} -> {"x0": 0.9}

        Two recovery modes are available:

        - **deterministic** (default): Returns the bin midpoint for each label.
          This is the maximum-likelihood estimate under a uniform prior within
          each bin, and produces the minimum expected roundtrip error.

        - **sample**: Draws uniformly at random from within each bin.  This
          mode is useful for generating diverse parameter vectors that are
          all consistent with the narrative description -- a form of
          "narrative-consistent sampling" that respects the information
          constraints of the diegetic representation.

        The ``recovery_uncertainty`` field in the output reports the bin width
        for each parameter, which is the fundamental resolution limit of the
        narrative representation.  No re-diegeticization method can recover
        information below this resolution.

        Args:
            narrative: Dict mapping handles (or param names) to bin labels.
            mode: "deterministic" uses bin midpoints; "sample" draws
                uniformly within each bin.
            seed: Random seed (used only when mode="sample").

        Returns:
            Dict with:
                "params": dict of recovered numeric values,
                "narrative": dict (copy of input narrative),
                "mode": str,
                "recovery_uncertainty": {param_name: bin_width, ...},
        """
        if mode not in ("deterministic", "sample"):
            raise ValueError(f"mode must be 'deterministic' or 'sample', got '{mode}'")

        rng = np.random.default_rng(seed) if mode == "sample" else None
        recovered = {}
        uncertainty = {}

        for handle, label in narrative.items():
            # Reverse-lookup: story handle -> internal parameter name.
            param_name = self._param_name_from_handle(handle)
            # Map narrative label -> bin index.
            idx = self._label_to_bin_index(label)
            info = self._bins[param_name]

            if mode == "deterministic":
                # Midpoint recovery: the single most representative value
                # for this bin.  Minimizes expected L2 error under a uniform
                # prior within the bin.
                recovered[param_name] = float(info["midpoints"][idx])
            else:
                # Sample recovery: draw uniformly from within the bin.
                # Produces diverse parameter vectors that are all consistent
                # with the narrative description.
                bin_lo = float(info["edges"][idx])
                bin_hi = float(info["edges"][idx + 1])
                recovered[param_name] = float(rng.uniform(bin_lo, bin_hi))

            # Recovery uncertainty = bin width: the fundamental resolution
            # limit of the narrative representation for this parameter.
            uncertainty[param_name] = float(info["width"])

        return {
            "params": recovered,
            "narrative": dict(narrative),
            "mode": mode,
            "recovery_uncertainty": uncertainty,
        }

    def roundtrip_error(self, params: dict) -> dict:
        """Measure information loss of the diegeticization roundtrip.

        Performs the full roundtrip: params -> narrative -> params', then
        computes the per-parameter and aggregate error between the original
        and recovered parameter vectors.

        This implements the **roundtrip fidelity** measurement described in
        Zimmerman (2025) SS3.5.3: the L2 distance between ``params`` and
        ``params'`` quantifies how much information is lost when numeric
        parameters are translated through narrative space and back.

        The total_error is the L2 norm of per-parameter errors, each
        normalized by the parameter's range.  Normalization ensures that
        parameters with different scales contribute equally to the aggregate
        metric.  For a D-dimensional parameter space with n_bins per
        dimension, the expected total_error under uniform sampling is:

            E[total_error] = sqrt(D) * bin_width / (2 * sqrt(3) * range)

        which is the standard deviation of a uniform distribution over
        half the bin width, summed in quadrature across dimensions.

        The "unrecoverable_params" list identifies parameters where the
        roundtrip error exceeds half the bin width (plus a small epsilon
        for floating-point tolerance).  This should never happen for the
        deterministic midpoint recovery mode -- if it does, it indicates a
        bug in the discretization logic.  In practice, errors are always
        bounded by half the bin width because the midpoint is the center
        of each bin.

        Args:
            params: Dict mapping parameter names to float values.

        Returns:
            Dict with:
                "original": dict (copy of input params),
                "recovered": dict (deterministic midpoint recovery),
                "per_param_error": {param_name: abs(original - recovered), ...},
                "total_error": float (L2 norm of range-normalized errors),
                "max_error_param": str (param with largest absolute error),
                "unrecoverable_params": [str, ...] (params where error >
                    half the bin width, i.e., where the value was near a
                    bin boundary -- should be empty for correct implementations),
        """
        # Forward pass: params -> narrative labels.
        dieg_result = self.diegeticize(params)
        # Reverse pass: narrative labels -> recovered params (deterministic midpoints).
        re_result = self.re_diegeticize(dieg_result["narrative"], mode="deterministic")
        recovered = re_result["params"]

        per_param_error = {}
        errors_normalized = []
        max_error = -1.0
        max_error_param = None
        unrecoverable = []

        for name in self._spec:
            orig_val = float(params[name])
            rec_val = float(recovered[name])
            # Absolute error: the raw information loss for this parameter.
            error = abs(orig_val - rec_val)
            per_param_error[name] = float(error)

            # Normalize by parameter range for the L2 computation.
            # This ensures parameters with different scales (e.g., [0, 1] vs
            # [-100, 100]) contribute equally to the aggregate error metric.
            lo, hi = self._spec[name]
            rng_width = hi - lo
            if rng_width > 0:
                errors_normalized.append(error / rng_width)
            else:
                # Degenerate case: parameter has zero range.
                errors_normalized.append(0.0)

            if error > max_error:
                max_error = error
                max_error_param = name

            # Flag parameters where error exceeds half the bin width.
            # The 1e-12 epsilon accounts for floating-point arithmetic.
            # In theory, midpoint recovery should always stay within this
            # bound; violations indicate a discretization edge case.
            bin_width = self._bins[name]["width"]
            threshold = bin_width / 2.0
            if error > threshold + 1e-12:
                unrecoverable.append(name)

        # L2 norm of normalized errors: a single scalar summarizing the
        # total information loss across all parameters.
        total_error = float(np.linalg.norm(errors_normalized))

        return {
            "original": dict(params),
            "recovered": recovered,
            "per_param_error": per_param_error,
            "total_error": total_error,
            "max_error_param": max_error_param,
            "unrecoverable_params": unrecoverable,
        }

    def batch_roundtrip(self, n_samples: int = 100, seed: int = 42) -> dict:
        """Measure roundtrip error statistics over random parameter samples.

        Systematically measures diegeticization fidelity across the full
        parameter space by generating ``n_samples`` uniformly random
        parameter vectors and computing roundtrip error for each.

        This provides a comprehensive characterization of information loss:
        - **mean_total_error**: average information loss across the parameter
          space.  Decreases as n_bins increases (more bins = less loss).
        - **std_total_error**: variability of information loss.  High std
          indicates that some regions of parameter space are much more
          lossy than others (typically near bin boundaries).
        - **per_param_mean_error**: identifies which parameters are most
          affected by discretization.  Parameters with narrow ranges
          relative to their bin width suffer more.
        - **per_param_max_error**: worst-case error for each parameter.
          Approaches half the bin width as n_samples increases.
        - **unrecoverable_fraction**: fraction of samples where each
          parameter's error exceeds half the bin width.  Should be 0.0
          for a correct implementation.

        This is the batch version of the reversibility principle test:
        rather than checking a single roundtrip, it characterizes the
        fidelity surface across the entire parameter space.

        Args:
            n_samples: Number of random samples to evaluate.  Higher values
                give more accurate statistics at the cost of more computation.
                100 is sufficient for stable estimates in most cases.
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                "mean_total_error": float,
                "std_total_error": float,
                "per_param_mean_error": {param_name: float, ...},
                "per_param_max_error": {param_name: float, ...},
                "unrecoverable_fraction": {param_name: float, ...},
                "n_samples": int,
        """
        rng = np.random.default_rng(seed)
        param_names = list(self._spec.keys())

        total_errors = []
        per_param_errors = {name: [] for name in param_names}
        unrecoverable_counts = {name: 0 for name in param_names}

        for _ in range(n_samples):
            # Generate a random parameter vector uniformly within bounds.
            params = {}
            for name, (lo, hi) in self._spec.items():
                params[name] = float(rng.uniform(lo, hi))

            # Compute roundtrip error for this sample.
            rt = self.roundtrip_error(params)
            total_errors.append(rt["total_error"])

            # Accumulate per-parameter error statistics.
            for name in param_names:
                per_param_errors[name].append(rt["per_param_error"][name])

            # Count unrecoverable parameters (error > half bin width).
            for name in rt["unrecoverable_params"]:
                unrecoverable_counts[name] += 1

        total_arr = np.array(total_errors)

        return {
            "mean_total_error": float(np.mean(total_arr)),
            "std_total_error": float(np.std(total_arr)),
            "per_param_mean_error": {
                name: float(np.mean(per_param_errors[name]))
                for name in param_names
            },
            "per_param_max_error": {
                name: float(np.max(per_param_errors[name]))
                for name in param_names
            },
            "unrecoverable_fraction": {
                name: float(unrecoverable_counts[name] / n_samples)
                for name in param_names
            },
            "n_samples": n_samples,
        }

    def run(self, params: dict) -> dict:
        """Simulator protocol: diegeticize, re-diegeticize, run on recovered params.

        This method makes the Diegeticizer itself satisfy the Simulator
        protocol (``run(params) -> dict`` + ``param_spec() -> bounds``),
        enabling direct comparison of "pristine" vs "diegeticized" simulation
        runs.  It also enables recursive analysis: the diegeticizer can be
        interrogated by other Zimmerman toolkit tools (Sobol sensitivity
        analysis, falsifier, contrastive pair generation, POSIWID) to study
        how diegeticization affects the simulator's behavior surface.

        The pipeline is:
            1. Diegeticize: params -> narrative labels (lossy discretization).
            2. Re-diegeticize: narrative labels -> midpoint params (deterministic).
            3. Run: midpoint params -> simulator output.

        The output differs from the raw simulator's output to the extent
        that discretization changes the parameters.  The difference is the
        "diegeticization cost" -- the price of translating through narrative
        space.

        Args:
            params: Dict mapping parameter names to float values.

        Returns:
            The underlying simulator's output dict, run with the
            deterministically recovered (midpoint) parameter values.
        """
        # Step 1: Forward diegeticization (params -> narrative).
        dieg_result = self.diegeticize(params)
        # Step 2: Inverse diegeticization (narrative -> midpoint params).
        re_result = self.re_diegeticize(dieg_result["narrative"], mode="deterministic")
        # Step 3: Run the underlying simulator with the recovered params.
        return self.simulator.run(re_result["params"])

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Delegates to the underlying simulator's param_spec.

        This delegation is required for the Simulator protocol: any tool
        that interrogates the Diegeticizer as a black-box simulator needs
        to know the parameter bounds.  The bounds are identical to the
        underlying simulator's -- diegeticization does not change the
        valid parameter range, only the internal representation.

        Returns:
            Parameter specification dict from the wrapped simulator.
        """
        return self.simulator.param_spec()
