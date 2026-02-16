"""Contrast set generation: find minimal *structured edits* that flip outcomes.

This module implements the contrast set generator, a core component of the
Zimmerman toolkit for black-box simulator interrogation. It operationalizes
two key principles from Zimmerman's dissertation:

    **TALOT** (Things Are Like Other Things): Meaning is constructed by
    identifying similarities and differences between configurations. A
    parameter vector's significance is revealed by what it is *near* in
    outcome space, not just where it sits in parameter space.

    **OTTITT** (One Thing Turns Into The Other Thing): Understanding
    transitions -- how one behavioral regime morphs into another -- reveals
    the causal structure of the system. The edit path formalism captures
    this: an ordered sequence of micro-edits traces a trajectory through
    parameter space, and the point at which qualitative behavior flips
    exposes the system's structural sensitivity.

These principles are developed in Zimmerman (2025) Section 4.7.6, where
TALOT/OTTITT are introduced as complementary heuristics for exploring
black-box simulators. TALOT asks "what else produces similar behavior?"
while OTTITT asks "what minimal change flips the behavior?"

The approach builds on Lipton's (1990) framework of *contrastive
explanation*, which argues that good explanations answer "Why P rather
than Q?" instead of "Why P?" Categorical explanations enumerate causes;
contrastive explanations identify the *difference-makers* -- the minimal
changes that separate one outcome class from another. This module
automates the search for such difference-makers in parameter space.

The tipping point search connects to catastrophe theory (Thom, 1975):
smooth changes in control parameters can produce discontinuous jumps in
system behavior at critical thresholds (catastrophe points, or "folds").
The binary search for the smallest flipping prefix is a numerical method
for locating these fold points along a one-dimensional path through a
potentially high-dimensional parameter space.

The connection to sensitivity analysis (Saltelli et al., 2008) is also
important: the ``param_tipping_frequency`` aggregation across many random
edit paths identifies which parameters are most often the "last straw"
at tipping points. This is complementary to variance-based sensitivity
(Sobol indices), which measures how much of the output *variance* each
parameter explains. Parameters with high tipping frequency may not have
high Sobol indices if their effect is localized near decision boundaries.

Use cases:
    - Understanding which parameter *changes* (not values) drive outcomes
    - Finding fragile edit sequences where tiny additions flip behavior
    - Generating "minimal repair" recipes: what's the least you must change?
    - Robustness analysis: how many edits before the system breaks?
    - Mapping decision boundaries in high-dimensional parameter spaces

References:
    Lipton, P. (1990). "Contrastive Explanation." Royal Institute of
        Philosophy Supplement, 27, 247-266.
    Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J.,
        Gatelli, D., Saisana, M., & Tarantola, S. (2008). *Global
        Sensitivity Analysis: The Primer.* John Wiley & Sons.
    Thom, R. (1975). *Structural Stability and Morphogenesis.* W.A. Benjamin.
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
        in Language, as Implemented in Humans and Large Language Models
        (LLMs)." PhD dissertation, University of Vermont. Ch.3-4, esp. S4.7.6.
"""

from __future__ import annotations

import numpy as np


class ContrastSetGenerator:
    """Finds minimal structured edit sequences that flip simulation outcomes.

    Instead of perturbing in continuous parameter space (like a simple
    sensitivity analysis), this works in discrete *edit space*: ordered
    sequences of named parameter changes. Binary search along the edit
    path finds the tipping point -- the smallest prefix of edits that
    flips the outcome.

    This is the computational realization of Lipton's (1990) contrastive
    explanation: rather than asking "why does configuration A produce
    outcome X?", we ask "why does A produce X rather than Y, and what
    is the minimal change from A that would produce Y instead?"

    The edit path formalism (Zimmerman, 2025, S4.7.6) provides structure
    to this search. An edit path is an ordered sequence of micro-edits
    (e1, e2, ..., en), each adjusting one parameter by a small delta.
    Applying edits[0:k] yields an intermediate configuration. By varying
    k and observing when the outcome class changes, we locate the
    tipping point -- the catastrophe fold (Thom, 1975) along this path.

    Args:
        simulator: Any Simulator-compatible object with run() and param_spec().
            The simulator must implement the black-box protocol: run(params)
            returns a result dict, and param_spec() returns {name: (lo, hi)}.
        outcome_fn: Callable that takes a result dict from simulator.run()
            and returns a hashable outcome class. Default: "positive" if
            result.get("fitness", 0) > 0 else "negative". This function
            defines the qualitative classification that the contrast set
            generator tries to flip -- it is the "Q" in Lipton's
            "Why P rather than Q?" formulation.

    Example:
        gen = ContrastSetGenerator(my_sim)
        path = gen.generate_edit_path(base_params, target_params)
        result = gen.find_tipping_point(base_params, path)
        print(result["tipping_k"], result["flip_size"])
    """

    def __init__(self, simulator, outcome_fn: callable | None = None):
        self.simulator = simulator
        # Cache the parameter specification (names and bounds) from the
        # underlying simulator. This defines the valid region of parameter
        # space within which all edit paths are constrained.
        self._spec = simulator.param_spec()
        if outcome_fn is None:
            self.outcome_fn = self._default_outcome_fn
        else:
            self.outcome_fn = outcome_fn

    @staticmethod
    def _default_outcome_fn(result: dict) -> str:
        """Default binary outcome classifier based on the 'fitness' key.

        This implements the simplest possible contrastive partition:
        "positive" vs "negative" fitness. For more nuanced contrastive
        analysis (e.g., multi-class outcomes, behavioral regimes), users
        should provide a custom outcome_fn.

        Args:
            result: Simulation result dict.

        Returns:
            "positive" if fitness > 0, "negative" otherwise.
        """
        fitness = result.get("fitness", 0.0)
        if isinstance(fitness, (int, float)):
            return "positive" if float(fitness) > 0 else "negative"
        try:
            import numpy as _np
            if isinstance(fitness, (_np.integer, _np.floating)):
                return "positive" if float(fitness) > 0 else "negative"
        except Exception:
            pass
        # Fallback for non-numeric fitness values: treat as positive
        # (conservative default to avoid false negatives in flip detection)
        return "positive"

    def _clip_params(self, params: dict) -> dict:
        """Clip parameter values to their specification bounds.

        Ensures that cumulative edit application never pushes parameters
        outside the valid region defined by param_spec(). This is
        important because edit paths can accumulate deltas that exceed
        the parameter range, especially when many edits target the same
        parameter.

        Args:
            params: Parameter dict, possibly with out-of-bounds values.

        Returns:
            New parameter dict with all values clipped to [lo, hi].
        """
        clipped = {}
        for name, val in params.items():
            if name in self._spec:
                lo, hi = self._spec[name]
                clipped[name] = float(np.clip(val, lo, hi))
            else:
                # Pass through parameters not in the spec unchanged.
                # This supports simulators with auxiliary (non-optimizable)
                # parameters.
                clipped[name] = val
        return clipped

    def _apply_edits(self, base_params: dict, edits: list[dict]) -> dict:
        """Apply a list of edits cumulatively to base_params.

        Each edit is a dict with at least "param" and "delta" keys.
        Edits are applied in order: each delta is added to the running
        parameter value. This means the result depends on the edit
        ordering -- which is by design, since edit paths are ordered
        sequences (smallest perturbation first).

        The cumulative application models the OTTITT principle: one
        configuration gradually transforms into another through a
        sequence of incremental changes.

        Args:
            base_params: Starting parameter dict.
            edits: List of edit dicts, each with "param" and "delta".

        Returns:
            New parameter dict with all edits applied and clipped to bounds.
        """
        params = dict(base_params)
        for edit in edits:
            name = edit["param"]
            delta = edit["delta"]
            # Cumulative addition: if multiple edits target the same
            # parameter, their deltas stack.
            params[name] = params.get(name, 0.0) + delta
        return self._clip_params(params)

    def generate_edit_path(
        self,
        base_params: dict,
        target_params: dict | None = None,
        n_edits: int = 20,
        seed: int = 42,
    ) -> list[dict]:
        """Generate an ordered edit path from base_params toward target_params.

        This constructs the one-dimensional trajectory through parameter
        space along which the tipping point search will operate. The path
        is a sequence of micro-edits, ordered by magnitude (smallest first),
        so that applying edits[0:k] for increasing k traces a monotonically
        growing perturbation.

        Two modes of path generation:

        **Targeted mode** (target_params provided): Distributes the total
        difference (target - base) across n_edits steps via round-robin
        assignment to parameters. This creates a path that, if fully
        applied, exactly reconstructs the target configuration. The
        interpolation between base and target is a discrete analog of
        the linear homotopy used in continuation methods for bifurcation
        analysis.

        **Random mode** (target_params is None): Generates random
        perturbations within parameter bounds. Each edit perturbs a
        single parameter by a random fraction (1-20%) of its range.
        This explores diverse directions in parameter space, useful
        when no specific target is known.

        In both modes, edits are sorted by absolute magnitude (smallest
        first). This ordering is crucial for the tipping point search:
        it ensures that the binary search over prefix length k corresponds
        to a monotonically increasing perturbation magnitude.

        Args:
            base_params: Starting parameter dict.
            target_params: Target parameter dict. If None, random perturbations
                are generated within parameter bounds.
            n_edits: Number of micro-edits to generate. More edits give finer
                resolution for tipping point localization (analogous to
                increasing the grid density in a bisection search).
            seed: Random seed for reproducibility. Deterministic edit paths
                ensure that tipping point analyses are replicable.

        Returns:
            List of edit dicts, ordered by magnitude (smallest first).
            Each edit dict contains:
                "param": str -- name of the parameter being adjusted,
                "delta": float -- signed change to apply,
                "description": str -- human-readable summary with magnitude
                    expressed as a fraction of the parameter's total range.
        """
        rng = np.random.default_rng(seed)
        param_names = list(self._spec.keys())
        edits = []

        if target_params is not None:
            # --- Targeted interpolation mode ---
            # Compute the total delta needed to reach the target for each
            # parameter. This is the "displacement vector" in parameter space.
            total_deltas = {}
            for name in param_names:
                base_val = base_params.get(name, 0.0)
                target_val = target_params.get(name, base_val)
                total_deltas[name] = target_val - base_val

            # Distribute n_edits across parameters via round-robin.
            # Each parameter receives ceil(n_edits / n_params) or
            # floor(n_edits / n_params) edits. Each edit for a given
            # parameter carries an equal fraction of that parameter's
            # total delta.
            for i in range(n_edits):
                # Round-robin parameter assignment: edit i is assigned to
                # parameter (i mod n_params).
                name = param_names[i % len(param_names)]
                # Count how many edits target this parameter in total,
                # so we can split its total delta evenly among them.
                n_edits_for_param = sum(
                    1 for j in range(n_edits) if param_names[j % len(param_names)] == name
                )
                if n_edits_for_param > 0:
                    delta = total_deltas[name] / n_edits_for_param
                else:
                    delta = 0.0
                lo, hi = self._spec[name]
                param_range = hi - lo if hi > lo else 1.0
                # Express the delta as a fraction of the parameter's range
                # for interpretability in the description string.
                frac = abs(delta) / param_range if param_range > 0 else 0.0
                edits.append({
                    "param": name,
                    "delta": float(delta),
                    "description": f"Adjust {name} by {delta:+.6f} ({frac:.1%} of range)",
                })
        else:
            # --- Random perturbation mode ---
            # Generate random micro-edits with magnitudes between 1% and 20%
            # of each parameter's range. The sign is chosen uniformly at random.
            for i in range(n_edits):
                name = param_names[i % len(param_names)]
                lo, hi = self._spec[name]
                param_range = hi - lo
                # Scale drawn from U(0.01, 0.2) * range -- this keeps edits
                # small enough to be "micro" but large enough to eventually
                # accumulate into meaningful perturbations.
                scale = param_range * rng.uniform(0.01, 0.2)
                delta = float(rng.choice([-1, 1]) * scale)
                frac = abs(delta) / param_range if param_range > 0 else 0.0
                edits.append({
                    "param": name,
                    "delta": delta,
                    "description": f"Perturb {name} by {delta:+.6f} ({frac:.1%} of range)",
                })

        # Sort by absolute magnitude (smallest first). This ordering
        # ensures that the prefix edits[0:k] represents the k smallest
        # individual perturbations, so increasing k monotonically increases
        # the total perturbation magnitude. This is the key invariant that
        # makes binary search over k well-defined: if edits[0:k] doesn't
        # flip the outcome but edits[0:n] does, then the tipping point
        # lies in [k, n].
        edits.sort(key=lambda e: abs(e["delta"]))
        return edits

    def find_tipping_point(
        self,
        base_params: dict,
        edit_path: list[dict],
        max_sims: int = 100,
    ) -> dict:
        """Binary search along edit_path to find the smallest flip prefix.

        This is the core algorithmic contribution: a bisection search over
        the discrete edit path to locate the tipping point -- the smallest
        k such that applying edits[0:k] changes the outcome class relative
        to the base configuration.

        The algorithm proceeds as follows:
            1. Evaluate the base outcome (k=0) and full-path outcome (k=n).
            2. If the full path doesn't flip the outcome, return immediately
               (no tipping point exists on this path).
            3. Otherwise, binary search: maintain lo (last k with base outcome)
               and hi (first k with flipped outcome). At each step, evaluate
               mid = (lo + hi) // 2 and narrow the interval.
            4. Converge to hi = the smallest flipping k.

        This is a discrete analog of the bisection method for root-finding,
        applied to the outcome classification function along the edit path.
        The "root" is the decision boundary -- the catastrophe fold (Thom,
        1975) where the system's qualitative behavior changes.

        Computational cost: O(log2(n_edits)) simulator evaluations, plus
        a constant overhead of 3 (base, full, verification). This is
        dramatically more efficient than linear search, which would require
        O(n_edits) evaluations.

        Args:
            base_params: Starting parameter dict.
            edit_path: Ordered list of edit dicts (from generate_edit_path).
                Must be sorted by magnitude for the binary search to be valid.
            max_sims: Maximum number of simulator calls allowed. Provides a
                computational budget constraint. If the budget is exhausted
                before convergence, the best-known tipping point is returned.

        Returns:
            Dict with:
                "found": bool -- whether a tipping point (outcome flip) was found,
                "base_outcome": hashable -- outcome at base_params (the "P" in
                    "Why P rather than Q?"),
                "flipped_outcome": hashable or None -- outcome after flip (the "Q"),
                "tipping_k": int or None -- smallest k that flips the outcome,
                "flip_size": float or None -- k / len(edit_path), normalized
                    measure in [0, 1] of how much of the edit path is needed
                    to flip. Small flip_size indicates a fragile configuration
                    (close to a decision boundary).
                "tipping_params": dict or None -- the parameter configuration at
                    the flip point (the actual tipping point in parameter space),
                "edit_path_applied": list -- edits[0:k] that were applied to
                    reach the tipping point,
                "n_sims": int -- total simulator calls made (for cost tracking),
        """
        n_edits = len(edit_path)
        if n_edits == 0:
            # Degenerate case: no edits to apply. Evaluate the base to
            # populate the outcome field, but there's nothing to search.
            base_result = self.simulator.run(base_params)
            return {
                "found": False,
                "base_outcome": self.outcome_fn(base_result),
                "flipped_outcome": None,
                "tipping_k": None,
                "flip_size": None,
                "tipping_params": None,
                "edit_path_applied": [],
                "n_sims": 1,
            }

        # Step 1: Evaluate the base outcome (no edits applied).
        # This establishes the "fact" side of the contrastive pair:
        # "configuration A produces outcome P."
        base_result = self.simulator.run(base_params)
        base_outcome = self.outcome_fn(base_result)
        n_sims = 1

        # Step 2: Evaluate the full edit path to check if a flip exists.
        # If the full path doesn't change the outcome, no tipping point
        # can exist along this path and we short-circuit.
        full_params = self._apply_edits(base_params, edit_path)
        full_result = self.simulator.run(full_params)
        full_outcome = self.outcome_fn(full_result)
        n_sims += 1

        if full_outcome == base_outcome:
            # Full path doesn't flip -- no tipping point on this path.
            # This is common when the edit path doesn't cross a decision
            # boundary, i.e., both endpoints lie in the same outcome region.
            return {
                "found": False,
                "base_outcome": base_outcome,
                "flipped_outcome": None,
                "tipping_k": None,
                "flip_size": None,
                "tipping_params": None,
                "edit_path_applied": [],
                "n_sims": n_sims,
            }

        # Step 3: Binary search for the tipping point.
        # Invariant: edits[0:lo] produces base_outcome,
        #            edits[0:hi] produces a different outcome.
        # We seek the smallest hi such that hi - lo == 1, giving the
        # exact tipping edit.
        lo = 0       # k=0 means no edits applied => base_outcome
        hi = n_edits  # k=n_edits applies all edits => flipped outcome

        while hi - lo > 1 and n_sims < max_sims:
            mid = (lo + hi) // 2
            # Apply the first 'mid' edits (the smallest 'mid' perturbations)
            # and classify the resulting configuration.
            mid_params = self._apply_edits(base_params, edit_path[:mid])
            mid_result = self.simulator.run(mid_params)
            mid_outcome = self.outcome_fn(mid_result)
            n_sims += 1

            if mid_outcome == base_outcome:
                # Midpoint still in the base outcome region.
                # Tipping point must be in [mid, hi].
                lo = mid
            else:
                # Midpoint has already flipped.
                # Tipping point must be in [lo, mid].
                hi = mid

        # Step 4: hi is the tipping point -- the smallest k where the
        # outcome flips. Verify by evaluating the tipping configuration.
        tipping_k = hi
        tipping_params = self._apply_edits(base_params, edit_path[:tipping_k])
        tipping_result = self.simulator.run(tipping_params)
        flipped_outcome = self.outcome_fn(tipping_result)
        n_sims += 1

        return {
            "found": True,
            "base_outcome": base_outcome,
            "flipped_outcome": flipped_outcome,
            "tipping_k": tipping_k,
            # flip_size normalizes k to [0, 1]. A small flip_size means
            # the system is fragile -- it sits close to a decision boundary
            # and a small perturbation flips the outcome. This is analogous
            # to the "distance to instability" in dynamical systems theory.
            "flip_size": float(tipping_k) / float(n_edits),
            "tipping_params": tipping_params,
            "edit_path_applied": list(edit_path[:tipping_k]),
            "n_sims": n_sims,
        }

    def batch_contrast_sets(
        self,
        base_params: dict,
        n_paths: int = 10,
        n_edits: int = 20,
        seed: int = 42,
    ) -> dict:
        """Generate multiple edit paths and find tipping points for each.

        This is the systematic enumeration component: instead of searching
        a single edit path, it generates n_paths different random edit paths
        (each exploring a different direction in parameter space) and runs
        find_tipping_point on each. The aggregated statistics reveal which
        parameters are structurally important for the system's qualitative
        behavior, complementing variance-based sensitivity analysis
        (Saltelli et al., 2008) with a *boundary-focused* perspective.

        The key insight (Zimmerman, 2025, S4.7.6) is that the "most
        important" parameters may differ depending on whether you measure
        importance by variance contribution (Sobol indices) or by
        boundary proximity (tipping frequency). A parameter with low
        variance contribution but high tipping frequency is one that
        rarely matters -- except when it does, at which point it is
        the decisive factor.

        Args:
            base_params: Starting parameter dict. This is the "reference
                configuration" from which all edit paths depart.
            n_paths: Number of distinct edit paths to generate and test.
                More paths give better coverage of the parameter space
                and more robust tipping frequency estimates.
            n_edits: Number of edits per path. Controls the resolution
                of each individual tipping point search.
            seed: Base random seed. Path i uses seed (seed + i) for
                reproducibility.

        Returns:
            Dict with:
                "pairs": list of find_tipping_point results (one per path),
                "mean_flip_size": float or None -- average flip_size across
                    paths where a flip was found. Small values indicate the
                    base configuration is generally fragile (near many
                    decision boundaries).
                "param_tipping_frequency": {param_name: float} -- fraction
                    of found flips where each parameter was the tipping edit
                    (the "last straw" that caused the flip). This identifies
                    which parameters sit at critical thresholds. Frequencies
                    sum to 1.0 across all parameters (each flip attributes
                    to exactly one parameter).
                "most_fragile_params": [param_names sorted by tipping
                    frequency, descending] -- parameters most often
                    responsible for outcome flips, i.e., the system's
                    most sensitive control dimensions near the base
                    configuration.
                "n_sims": int -- total simulator calls across all paths
                    (for computational cost tracking).
        """
        pairs = []
        total_sims = 0
        # Initialize per-parameter tipping counters. Each parameter starts
        # with zero counts; we increment when that parameter is the
        # "tipping edit" (the edit at index tipping_k - 1 that causes
        # the outcome flip).
        param_tipping_counts = {name: 0 for name in self._spec.keys()}
        n_found = 0

        for i in range(n_paths):
            # Each path uses a different seed to explore a different
            # random direction in parameter space. The diversity of
            # directions is what gives the aggregated tipping frequency
            # its statistical meaning.
            path_seed = seed + i
            edit_path = self.generate_edit_path(
                base_params,
                target_params=None,  # Random perturbation mode
                n_edits=n_edits,
                seed=path_seed,
            )
            result = self.find_tipping_point(base_params, edit_path)
            pairs.append(result)
            total_sims += result["n_sims"]

            if result["found"] and result["tipping_k"] is not None:
                n_found += 1
                # Identify the tipping edit: the edit at index (tipping_k - 1)
                # is the one whose application caused the outcome to flip.
                # This is the "difference-maker" in Lipton's (1990) sense --
                # the minimal contrastive factor between the last non-flipped
                # and first flipped configuration.
                tipping_idx = result["tipping_k"] - 1
                if 0 <= tipping_idx < len(edit_path):
                    tipping_edit = edit_path[tipping_idx]
                    param_name = tipping_edit["param"]
                    if param_name in param_tipping_counts:
                        param_tipping_counts[param_name] += 1

        # Aggregate statistics across all paths.
        if n_found > 0:
            # Normalize tipping counts to frequencies. Each found path
            # contributes exactly 1 to the total count (one tipping edit
            # per path), so frequencies sum to 1.0 across parameters.
            param_tipping_frequency = {
                name: count / n_found
                for name, count in param_tipping_counts.items()
            }
            # Compute mean flip_size across paths where a flip was found.
            # This gives a scalar summary of how fragile the base
            # configuration is: small mean_flip_size means the base is
            # close to decision boundaries in many directions.
            flip_sizes = [
                p["flip_size"] for p in pairs
                if p["found"] and p["flip_size"] is not None
            ]
            mean_flip_size = float(np.mean(flip_sizes)) if flip_sizes else None
        else:
            # No flips found on any path. The base configuration is
            # robustly within a single outcome region -- it is far from
            # any decision boundary in the directions sampled.
            param_tipping_frequency = {name: 0.0 for name in self._spec.keys()}
            mean_flip_size = None

        # Sort parameters by tipping frequency (descending). The most
        # fragile parameters -- those most often at the tipping point --
        # appear first. This ranking is the primary output for identifying
        # structurally critical parameters near the base configuration.
        most_fragile_params = sorted(
            param_tipping_frequency.keys(),
            key=lambda n: param_tipping_frequency[n],
            reverse=True,
        )

        return {
            "pairs": pairs,
            "mean_flip_size": mean_flip_size,
            "param_tipping_frequency": param_tipping_frequency,
            "most_fragile_params": most_fragile_params,
            "n_sims": total_sims,
        }

    def run(self, params: dict) -> dict:
        """Simulator protocol wrapper for composability.

        Makes ContrastSetGenerator itself satisfy the Simulator protocol
        (run() + param_spec()), enabling meta-analysis: you can wrap a
        ContrastSetGenerator in another Zimmerman tool (e.g., Sobol
        sensitivity on the tipping frequency outputs).

        Runs batch_contrast_sets with default settings, using the given
        params as the base configuration.

        Args:
            params: Base parameter dict.

        Returns:
            Dict with batch_contrast_sets results.
        """
        return self.batch_contrast_sets(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Delegates to underlying simulator's parameter specification.

        Returns:
            Parameter specification from the wrapped simulator:
            {param_name: (lower_bound, upper_bound)}.
        """
        return self.simulator.param_spec()
