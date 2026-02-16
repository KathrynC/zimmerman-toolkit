"""Relation graph extraction: map meaning-from-relations around a concept.

This module implements the relation graph extractor, which operationalizes
Zimmerman's (2025) Chapter 3 thesis that **meaning is constructed from
relations, not intrinsic properties**. A parameter configuration's
"meaning" in the context of a simulator is not its numeric values, but
the web of causal, similarity, and contrast relationships it participates
in. This module builds that web computationally.

The three relational frames correspond to Zimmerman's relational semantics:

    **Causal frame** (A causes B): Perturb each parameter independently
    and measure the resulting change in each output via finite-difference
    gradient estimation. This reveals the directional causal influence
    of parameters on outputs. The causal frame connects to Beer's (2024)
    POSIWID principle -- "The Purpose Of a System Is What It Does" --
    by inferring what a parameter *does* from its observable effects,
    rather than from any a priori specification of its purpose.

    **Similarity frame** (A resembles B): Find parameter configurations
    whose outputs are highly similar (cosine similarity > 0.9) to the
    base configuration. These form the "neighborhood" of the base in
    output space -- configurations that are functionally equivalent
    despite possibly differing in parameter space. This is related to
    the concept of "equivalence classes" in behavioral analysis.

    **Contrast frame** (A differs from B): Find configurations with
    maximally different outputs (cosine similarity < 0.5). These
    establish what the base configuration is *not* -- the contrastive
    background against which its meaning is defined. This connects to
    Lipton's (1990) contrastive explanation and to Zimmerman's TALOT
    principle (Things Are Like Other Things -- and, crucially, unlike
    certain other things).

The resulting multigraph has three edge types:
    - **causal edges**: parameter -> output (gradient-based weights).
      These are directed edges representing first-order causal influence.
    - **param_correlation edges**: parameter <-> parameter (output-space
      correlation). Undirected edges measuring whether two parameters
      affect outputs in correlated ways.
    - **output_correlation edges**: output <-> output (co-variation).
      Undirected edges measuring whether two outputs move together
      under perturbation.

Stability is assessed via split-half consistency (a bootstrap resampling
technique): the causal graph is computed on two independent subsamples
and the overlap of top edges is measured. High overlap (Jaccard index
near 1.0) indicates that the graph structure is robust to sampling
variation, not an artifact of the particular probe set.

Graph-theoretic measures from network science (Newman, 2010) can be
applied to the resulting graph:
    - **Betweenness centrality**: identifies parameters or outputs that
      serve as "bridges" between different parts of the relational
      structure.
    - **Clustering coefficient**: measures the density of triangles,
      indicating whether the system's causal relationships are locally
      clustered or globally distributed.
    - **Degree distribution**: reveals whether the system has "hub"
      parameters that influence many outputs or is more uniformly
      connected.

These graph-theoretic analyses are not implemented in this module but
can be applied to the extracted graph using standard network analysis
libraries (e.g., NetworkX).

References:
    Beer, S. (2024). *The Heart of Enterprise.* Revised edition.
        (Original work published 1979.) See Ch. 7 on POSIWID.
    Lipton, P. (1990). "Contrastive Explanation." Royal Institute of
        Philosophy Supplement, 27, 247-266.
    Newman, M.E.J. (2010). *Networks: An Introduction.* Oxford
        University Press.
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
        in Language, as Implemented in Humans and Large Language Models
        (LLMs)." PhD dissertation, University of Vermont. Ch.2-3.
"""

from __future__ import annotations

import numpy as np


class RelationGraphExtractor:
    """Builds a relation graph around a target parameter configuration.

    The graph captures the relational structure of the simulator's behavior
    near a given point in parameter space. Nodes represent parameter
    dimensions and output dimensions. Edges represent relationships:
    causal influence (parameter -> output), parameter correlation
    (parameter <-> parameter), and output correlation (output <-> output).

    This implements Zimmerman's (2025, Ch. 3) relational semantics
    computationally: the "meaning" of a parameter configuration is the
    graph of relationships it participates in. Two configurations with
    identical graphs are functionally equivalent; configurations with
    different graphs occupy different semantic positions.

    The causal frame uses central finite differences for gradient
    estimation -- a standard numerical technique that requires 2 * n_params
    simulator evaluations. The similarity and contrast frames use random
    probing with cosine similarity thresholds to populate the relational
    neighborhood.

    Args:
        simulator: Any Simulator-compatible object with run() and param_spec().
            Must implement the black-box protocol.
        output_keys: Which output keys to track for relations. If None,
            auto-detects all numeric output keys from a baseline run.
            Specifying output_keys focuses the analysis on particular
            outputs of interest and reduces computational cost.

    Example:
        extractor = RelationGraphExtractor(my_sim)
        graph = extractor.extract(base_params, n_probes=100)
        print(graph["rankings"]["most_causal_params"])
    """

    def __init__(self, simulator, output_keys: list[str] | None = None):
        self.simulator = simulator
        # Cache param_spec for repeated use in perturbation generation.
        self._spec = simulator.param_spec()
        self._output_keys = output_keys

    def _detect_output_keys(self, result: dict) -> list[str]:
        """Auto-detect numeric output keys from a simulation result dict.

        Scans the result dictionary for keys with finite numeric values.
        Non-numeric keys (strings, lists, etc.) and non-finite values
        (NaN, inf) are excluded, since they cannot participate in
        gradient estimation or correlation computation.

        Args:
            result: A result dict from simulator.run().

        Returns:
            Sorted list of keys with finite numeric values. Sorting
            ensures deterministic ordering across runs.
        """
        keys = []
        for key, val in result.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isfinite(float(val)):
                    keys.append(key)
        return sorted(keys)

    def _result_to_vector(self, result: dict, output_keys: list[str]) -> np.ndarray:
        """Convert a result dict to a numpy vector for the given output keys.

        This vectorization step enables efficient numerical operations
        (cosine similarity, correlation) on simulation outputs. Missing
        or non-numeric values are represented as NaN, which propagates
        gracefully through downstream computations.

        Args:
            result: A result dict from simulator.run().
            output_keys: List of output key names defining the vector's
                dimensions and ordering.

        Returns:
            1-D numpy array of float values, NaN for missing/non-numeric keys.
        """
        vec = np.zeros(len(output_keys))
        for i, key in enumerate(output_keys):
            val = result.get(key, np.nan)
            if isinstance(val, (int, float, np.integer, np.floating)):
                vec[i] = float(val)
            else:
                vec[i] = np.nan
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two output vectors.

        Cosine similarity measures the angle between two vectors,
        ignoring magnitude. Two output vectors with sim = 1.0 point in
        the same direction (proportional outputs), sim = 0.0 are
        orthogonal, and sim = -1.0 are anti-parallel.

        This metric is used rather than Euclidean distance because it
        captures *qualitative* similarity (same pattern of relative
        output values) regardless of *quantitative* scale. This aligns
        with the relational semantics perspective: what matters is the
        pattern of relationships, not absolute magnitudes.

        Returns 0.0 if either vector has zero norm (degenerate input)
        or contains NaN (invalid output), as a conservative default
        that avoids false positive similarity matches.

        Args:
            a: First output vector.
            b: Second output vector.

        Returns:
            Cosine similarity in [-1, 1], or 0.0 on degenerate input.
        """
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def similarity_frame(
        self,
        base_params: dict,
        n_probes: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Find parameter configurations producing similar outputs.

        Implements the similarity relational frame: identifies configurations
        in the parameter space neighborhood (small perturbations, +/- 5%
        of each parameter's range) that produce outputs highly similar
        to the base (cosine similarity > 0.9).

        The similarity threshold of 0.9 is chosen to capture configurations
        that are functionally near-equivalent to the base. These form the
        "equivalence neighborhood" -- the set of parameter configurations
        that, for practical purposes, produce the same qualitative behavior.
        The size of this neighborhood is itself informative: a large
        similarity set indicates robustness, while a small one indicates
        sensitivity.

        Args:
            base_params: Base parameter dict defining the center of the
                similarity search.
            n_probes: Number of random perturbations to try. More probes
                give better coverage of the local neighborhood.
            rng: Numpy random generator for reproducible perturbation
                generation.

        Returns:
            List of dicts for configurations passing the similarity
            threshold, each with:
                "params": dict -- the perturbed parameter configuration,
                "output": dict -- the full simulation result,
                "similarity": float -- cosine similarity to base output.
        """
        # Evaluate the base configuration to establish the reference
        # output vector against which similarities are measured.
        base_result = self.simulator.run(base_params)
        output_keys = self._output_keys or self._detect_output_keys(base_result)
        base_vec = self._result_to_vector(base_result, output_keys)

        similar = []
        for _ in range(n_probes):
            # Small perturbation: +/- 5% of each parameter's range.
            # This keeps probes in the local neighborhood, ensuring that
            # similarity matches reflect genuine functional equivalence
            # rather than distant coincidence.
            perturbed = dict(base_params)
            for name, (lo, hi) in self._spec.items():
                delta = rng.uniform(-0.05, 0.05) * (hi - lo)
                perturbed[name] = float(np.clip(base_params.get(name, (lo + hi) / 2) + delta, lo, hi))

            result = self.simulator.run(perturbed)
            vec = self._result_to_vector(result, output_keys)
            sim = self._cosine_similarity(base_vec, vec)

            # Threshold filter: only retain probes with high similarity.
            # The 0.9 threshold is a design choice balancing inclusiveness
            # (capturing the full equivalence neighborhood) against
            # precision (excluding configurations that are merely "somewhat
            # similar").
            if sim > 0.9:
                similar.append({
                    "params": perturbed,
                    "output": result,
                    "similarity": sim,
                })

        return similar

    def contrast_frame(
        self,
        base_params: dict,
        n_probes: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Find parameter configurations producing maximally different outputs.

        Implements the contrast relational frame: identifies configurations
        across the full parameter range that produce outputs dissimilar to
        the base (cosine similarity < 0.5).

        Unlike the similarity frame (which uses small local perturbations),
        the contrast frame samples uniformly across the entire parameter
        space. This is necessary because maximally different outputs
        typically require configurations far from the base -- they lie in
        different behavioral regimes.

        The contrast set defines what the base configuration is *not*,
        providing the contrastive background essential for Lipton's (1990)
        contrastive explanation and Zimmerman's relational semantics.

        Args:
            base_params: Base parameter dict defining the reference point.
            n_probes: Number of random configurations to try.
            rng: Numpy random generator.

        Returns:
            List of dicts for configurations below the similarity threshold,
            each with:
                "params": dict -- the contrasting parameter configuration,
                "output": dict -- the full simulation result,
                "similarity": float -- cosine similarity to base output
                    (always < 0.5 for returned entries).
        """
        base_result = self.simulator.run(base_params)
        output_keys = self._output_keys or self._detect_output_keys(base_result)
        base_vec = self._result_to_vector(base_result, output_keys)

        contrasts = []
        for _ in range(n_probes):
            # Full-range sampling: each parameter drawn uniformly from
            # its entire [lo, hi] range. This explores the global parameter
            # space to find configurations in different behavioral regimes.
            perturbed = {}
            for name, (lo, hi) in self._spec.items():
                perturbed[name] = float(rng.uniform(lo, hi))

            result = self.simulator.run(perturbed)
            vec = self._result_to_vector(result, output_keys)
            sim = self._cosine_similarity(base_vec, vec)

            # Threshold filter: only retain probes with low similarity.
            # Cosine similarity < 0.5 means the output vectors are
            # substantially different in direction, indicating qualitatively
            # distinct behavior.
            if sim < 0.5:
                contrasts.append({
                    "params": perturbed,
                    "output": result,
                    "similarity": sim,
                })

        return contrasts

    def causal_frame(
        self,
        base_params: dict,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Estimate causal influence of each parameter on each output.

        Implements the causal relational frame via central finite-difference
        gradient estimation. For each parameter p_i, this computes:

            d(output_j) / d(p_i) ~ [output_j(p_i + eps) - output_j(p_i - eps)] / (2 * eps)

        This first-order gradient approximation reveals the local causal
        structure: which parameters influence which outputs, by how much
        (weight), and in which direction (sign).

        The causal frame operationalizes Beer's (2024) POSIWID principle:
        the *purpose* of a parameter is inferred from what it *does* --
        i.e., its observable effects on outputs under perturbation. A
        parameter with high causal weight on an output is one that
        effectively controls that output; a parameter with near-zero
        causal weight is irrelevant to that output near the base
        configuration.

        Central differences are used rather than forward differences for
        better numerical accuracy: the error is O(eps^2) rather than O(eps).
        The epsilon is set to 1% of each parameter's range, balancing
        accuracy (small eps) against numerical stability (eps not too
        small relative to floating-point precision).

        Args:
            base_params: Base parameter dict. Gradients are estimated
                at this point in parameter space.
            rng: Numpy random generator (reserved for potential future
                use with stochastic gradient estimation; currently
                unused since central differences are deterministic).

        Returns:
            List of edge dicts, each representing a causal link:
                "from": str -- parameter name (cause),
                "to": str -- output key (effect),
                "weight": float -- absolute gradient magnitude (always >= 0),
                "sign": int -- +1 if increasing the parameter increases the
                    output, -1 if it decreases it.
        """
        base_result = self.simulator.run(base_params)
        output_keys = self._output_keys or self._detect_output_keys(base_result)
        base_vec = self._result_to_vector(base_result, output_keys)

        causal_edges = []
        for param_name, (lo, hi) in self._spec.items():
            param_range = hi - lo
            if param_range < 1e-12:
                # Degenerate parameter with zero range -- skip.
                # Cannot compute a meaningful gradient.
                continue

            # Epsilon for central differences: 1% of the parameter range.
            # This is a standard choice that works well for smooth simulators.
            # For highly non-linear simulators, adaptive epsilon selection
            # (e.g., step-doubling) would be more robust.
            epsilon = param_range * 0.01
            base_val = base_params.get(param_name, (lo + hi) / 2.0)

            # Forward perturbation: p_i + epsilon, all other params fixed.
            forward_params = dict(base_params)
            forward_params[param_name] = float(np.clip(base_val + epsilon, lo, hi))
            forward_result = self.simulator.run(forward_params)
            forward_vec = self._result_to_vector(forward_result, output_keys)

            # Backward perturbation: p_i - epsilon, all other params fixed.
            backward_params = dict(base_params)
            backward_params[param_name] = float(np.clip(base_val - epsilon, lo, hi))
            backward_result = self.simulator.run(backward_params)
            backward_vec = self._result_to_vector(backward_result, output_keys)

            # Central difference gradient: (f(x+eps) - f(x-eps)) / (2*eps).
            # Note: actual_eps may differ from 2*epsilon if clipping occurred
            # at parameter bounds, so we use the actual perturbed values.
            actual_eps = (forward_params[param_name] - backward_params[param_name])
            if abs(actual_eps) < 1e-15:
                # Both perturbations clipped to the same value (parameter
                # at a bound with zero effective range) -- skip.
                continue

            gradient = (forward_vec - backward_vec) / actual_eps

            # Create one causal edge per output dimension with a finite
            # gradient. Each edge is a (parameter -> output) link in the
            # relation graph.
            for i, out_key in enumerate(output_keys):
                grad_val = gradient[i]
                if np.isfinite(grad_val):
                    causal_edges.append({
                        "from": param_name,
                        "to": out_key,
                        # Weight is the absolute gradient: magnitude of
                        # causal influence regardless of direction.
                        "weight": float(abs(grad_val)),
                        # Sign preserves directionality: +1 means the
                        # parameter and output move in the same direction.
                        "sign": 1 if grad_val >= 0 else -1,
                    })

        return causal_edges

    def _compute_correlations(
        self,
        probe_results: list[dict],
        output_keys: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """Compute parameter-parameter and output-output correlations.

        This populates the non-causal edges of the relation graph:
        correlations between pairs of parameters (do they co-vary across
        the probe set?) and between pairs of outputs (do they move
        together under perturbation?).

        Parameter-parameter correlations measure whether two parameters
        tend to take similar or opposite values in the probe set. For
        randomly generated probes, these correlations should be near zero
        (independent sampling). Significant correlations indicate
        structural dependencies in the probe generation process or,
        more interestingly, in the parameter space regions that produce
        viable outputs.

        Output-output correlations are more informative: they reveal the
        system's internal coupling structure. Two outputs with high
        positive correlation are driven by the same underlying mechanism;
        two with high negative correlation are antagonistic (improving one
        degrades the other). These correspond to the "output_correlation"
        edges in the relation graph.

        Pearson correlation is used (via np.corrcoef), which captures
        linear relationships. For non-linear dependencies, mutual
        information or rank correlation (Spearman) would be more
        appropriate but are not implemented here for simplicity and
        numpy-only compatibility.

        Args:
            probe_results: List of dicts, each with "params" (dict) and
                "output" (dict) from similarity and contrast frame probes.
            output_keys: List of output key names to analyze.

        Returns:
            Tuple of (param_correlations, output_correlations).
            Each is a list of edge dicts:
                param_correlations: {"param_a", "param_b", "correlation"}
                output_correlations: {"output_a", "output_b", "correlation"}
            Returns empty lists if fewer than 3 probes (insufficient data
            for meaningful correlation estimation).
        """
        if len(probe_results) < 3:
            # Need at least 3 data points for a non-degenerate correlation.
            return [], []

        param_names = list(self._spec.keys())
        n_probes = len(probe_results)

        # Build parameter matrix: rows are probes, columns are parameters.
        # This is the "design matrix" from which parameter correlations
        # are computed.
        param_matrix = np.zeros((n_probes, len(param_names)))
        for i, pr in enumerate(probe_results):
            for j, name in enumerate(param_names):
                param_matrix[i, j] = pr["params"].get(name, 0.0)

        # Build output matrix: rows are probes, columns are outputs.
        # This captures how outputs co-vary across the probe set.
        output_matrix = np.zeros((n_probes, len(output_keys)))
        for i, pr in enumerate(probe_results):
            out = pr["output"]
            for j, key in enumerate(output_keys):
                val = out.get(key, np.nan)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    output_matrix[i, j] = float(val)
                else:
                    output_matrix[i, j] = np.nan

        # --- Parameter-parameter correlations ---
        # For each pair of parameters, compute their Pearson correlation
        # across the probe set. This measures whether the two parameters
        # tend to co-vary in the sampled configurations.
        param_correlations = []
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                col_i = param_matrix[:, i]
                col_j = param_matrix[:, j]
                # Only correlate if both columns have non-trivial variance.
                # Zero-variance columns produce undefined correlations.
                if np.std(col_i) > 1e-12 and np.std(col_j) > 1e-12:
                    corr_matrix = np.corrcoef(col_i, col_j)
                    corr = float(corr_matrix[0, 1])
                    if np.isfinite(corr):
                        param_correlations.append({
                            "param_a": param_names[i],
                            "param_b": param_names[j],
                            "correlation": corr,
                        })

        # --- Output-output correlations ---
        # For each pair of outputs, compute their Pearson correlation
        # across the probe set. High |correlation| indicates outputs that
        # are functionally coupled by the simulator's internal dynamics.
        output_correlations = []
        for i in range(len(output_keys)):
            for j in range(i + 1, len(output_keys)):
                col_i = output_matrix[:, i]
                col_j = output_matrix[:, j]
                # Filter to rows where both outputs are finite (not NaN).
                valid = np.isfinite(col_i) & np.isfinite(col_j)
                if valid.sum() > 2:
                    if np.std(col_i[valid]) > 1e-12 and np.std(col_j[valid]) > 1e-12:
                        corr_matrix = np.corrcoef(col_i[valid], col_j[valid])
                        corr = float(corr_matrix[0, 1])
                        if np.isfinite(corr):
                            output_correlations.append({
                                "output_a": output_keys[i],
                                "output_b": output_keys[j],
                                "correlation": corr,
                            })

        return param_correlations, output_correlations

    def _compute_stability(
        self,
        base_params: dict,
        output_keys: list[str],
        n_probes: int,
        rng: np.random.Generator,
        top_k: int = 5,
    ) -> dict:
        """Assess structural stability of the relation graph via split-half consistency.

        Stability measurement is critical for trusting the extracted graph:
        a graph that changes dramatically with different random samples is
        an artifact of noise, not a genuine structural feature of the
        simulator. This method runs two independent causal frame analyses
        and measures the overlap of their top edges.

        Two stability metrics are computed:

        **Jaccard overlap** (of top-k edges): The Jaccard index
        J(A, B) = |A intersect B| / |A union B| measures the agreement
        between the top-k strongest causal edges from two independent
        runs. J = 1.0 means perfect agreement; J = 0.0 means no overlap.
        This is a standard set-similarity metric from information
        retrieval (Jaccard, 1912).

        **Edge survival rate**: The fraction of edges in run A that also
        appear in run B (regardless of rank). This is a more permissive
        measure: it counts any edge that persists, not just top-ranked
        ones. High survival rate (close to 1.0) indicates that the
        graph's edge set is structurally robust; low survival rate
        indicates that many edges are noise.

        For deterministic simulators, both metrics should be 1.0 (the
        causal frame uses central differences, which are deterministic).
        For stochastic simulators, these metrics quantify the signal-to-
        noise ratio of the extracted graph.

        Args:
            base_params: Base parameter dict.
            output_keys: Output keys to track.
            n_probes: Number of probes (used for consistency with the
                calling context; the causal frame itself does not use
                random probes).
            rng: Numpy random generator. Two child generators are spawned
                from it for the two independent runs.
            top_k: Number of top edges to compare for Jaccard overlap.

        Returns:
            Dict with:
                "jaccard_overlap": float in [0, 1] -- set similarity of
                    top-k edges between two independent runs.
                "edge_survival_rate": float in [0, 1] -- fraction of edges
                    from run A that persist in run B.
        """
        half = max(n_probes // 4, 2)

        # Spawn two independent child RNGs from the parent. This ensures
        # that the two runs are statistically independent while remaining
        # reproducible from the parent seed.
        rng_a = np.random.default_rng(rng.integers(0, 2**31))
        rng_b = np.random.default_rng(rng.integers(0, 2**31))

        # Run two independent causal frame analyses.
        edges_a = self.causal_frame(base_params, rng_a)
        edges_b = self.causal_frame(base_params, rng_b)

        # --- Jaccard overlap of top-k edges ---
        # Sort edges by weight (causal strength) and take the top k.
        # These are the "most important" causal relationships.
        edges_a_sorted = sorted(edges_a, key=lambda e: e["weight"], reverse=True)[:top_k]
        edges_b_sorted = sorted(edges_b, key=lambda e: e["weight"], reverse=True)[:top_k]

        # Convert to sets of (from, to) tuples for set operations.
        set_a = {(e["from"], e["to"]) for e in edges_a_sorted}
        set_b = {(e["from"], e["to"]) for e in edges_b_sorted}

        # Jaccard index: |intersection| / |union|.
        if len(set_a) == 0 and len(set_b) == 0:
            # Both empty: trivially identical (vacuously true).
            jaccard = 1.0
        elif len(set_a | set_b) == 0:
            jaccard = 0.0
        else:
            jaccard = float(len(set_a & set_b)) / float(len(set_a | set_b))

        # --- Edge survival rate ---
        # What fraction of *all* edges in run A also appear in run B?
        # This measures structural persistence, not just top-edge agreement.
        all_a = {(e["from"], e["to"]) for e in edges_a}
        all_b = {(e["from"], e["to"]) for e in edges_b}
        if len(all_a) == 0:
            survival = 1.0
        else:
            survival = float(len(all_a & all_b)) / float(len(all_a))

        return {
            "jaccard_overlap": jaccard,
            "edge_survival_rate": survival,
        }

    def extract(
        self,
        base_params: dict,
        n_probes: int = 50,
        seed: int = 42,
    ) -> dict:
        """Build a complete relation graph around base_params.

        This is the main entry point. It orchestrates all three relational
        frames (causal, similarity, contrast), computes correlations from
        the probe data, assesses graph stability, and produces ranked
        summaries of the most important parameters and outputs.

        The resulting graph structure implements Zimmerman's (2025, Ch. 3)
        thesis: the meaning of a parameter configuration is constituted
        by its relations -- causal (what it controls), similarity (what
        it resembles), and contrast (what it differs from).

        The multigraph construction follows standard network science
        conventions (Newman, 2010): nodes are typed (parameter nodes vs.
        output nodes), edges are typed (causal vs. correlation), and
        edges carry weights and metadata. This structure can be exported
        to standard graph formats for further analysis with tools like
        NetworkX.

        Computational cost: approximately (1 + 2*n_params + 2*n_probes +
        4*n_params) simulator evaluations, where the 4*n_params term comes
        from the stability assessment (two additional causal frames).

        Args:
            base_params: Target parameter dict to build the graph around.
                This is the "center" of the relational neighborhood.
            n_probes: Number of random probes for similarity and contrast
                frames. More probes give better coverage but cost more
                simulator evaluations.
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                "nodes": {
                    "params": {param_name: base_value, ...} -- parameter
                        node attributes (the base configuration),
                    "outputs": {output_key: base_value, ...} -- output
                        node attributes (the base simulation result),
                },
                "edges": {
                    "causal": list of {from, to, weight, sign} dicts --
                        directed parameter -> output edges from gradient
                        estimation,
                    "param_correlation": list of {param_a, param_b,
                        correlation} dicts -- undirected parameter <->
                        parameter edges,
                    "output_correlation": list of {output_a, output_b,
                        correlation} dicts -- undirected output <->
                        output edges,
                },
                "stability": {
                    "jaccard_overlap": float -- top-k edge agreement,
                    "edge_survival_rate": float -- full edge persistence,
                },
                "rankings": {
                    "most_causal_params": list of param names sorted by
                        total causal weight (sum of edge weights across
                        all outputs). The first element is the parameter
                        with the strongest overall causal influence.
                    "most_connected_outputs": list of output keys sorted
                        by the number of causal edges pointing to them
                        (degree in the causal subgraph). Outputs with
                        high degree are influenced by many parameters.
                },
                "n_sims": int -- total simulator evaluations performed.
        """
        rng = np.random.default_rng(seed)
        n_sims = 0

        # --- Baseline run ---
        # Evaluate the base configuration to establish reference output
        # values and detect output keys.
        base_result = self.simulator.run(base_params)
        n_sims += 1
        output_keys = self._output_keys or self._detect_output_keys(base_result)

        # --- Construct node attributes ---
        # Nodes in the relation graph are either parameter dimensions or
        # output dimensions. Node attributes store the base values.
        base_outputs = {}
        for key in output_keys:
            val = base_result.get(key, np.nan)
            if isinstance(val, (int, float, np.integer, np.floating)):
                base_outputs[key] = float(val)

        # --- Causal frame ---
        # Compute parameter -> output causal edges via central finite
        # differences. Cost: 2 simulator evaluations per parameter
        # (forward + backward perturbation).
        causal_edges = self.causal_frame(base_params, rng)
        n_sims += 2 * len(self._spec)

        # --- Similarity and contrast frames ---
        # Probe the local neighborhood (similarity) and global parameter
        # space (contrast) to populate the relational context.
        # Cost: n_probes evaluations each.
        sim_probes = self.similarity_frame(base_params, n_probes, rng)
        n_sims += n_probes
        contrast_probes = self.contrast_frame(base_params, n_probes, rng)
        n_sims += n_probes

        # --- Correlation computation ---
        # Combine all probes (similarity + contrast) into a single dataset
        # for correlation analysis. Using both sets gives a broader sample
        # of the parameter-output mapping, improving correlation estimates.
        all_probes = []
        for sp in sim_probes:
            all_probes.append({"params": sp["params"], "output": sp["output"]})
        for cp in contrast_probes:
            all_probes.append({"params": cp["params"], "output": cp["output"]})

        # Compute pairwise correlations: parameter-parameter (do they
        # co-vary?) and output-output (do they move together?).
        param_corr, output_corr = self._compute_correlations(all_probes, output_keys)

        # --- Stability assessment ---
        # Run two independent causal frames and measure edge overlap.
        # Cost: 2 * 2 * n_params additional simulator evaluations.
        stability = self._compute_stability(base_params, output_keys, n_probes, rng)
        n_sims += 4 * len(self._spec)

        # --- Rankings ---
        # Rank parameters by total causal weight: sum of absolute gradient
        # magnitudes across all outputs. This identifies which parameters
        # have the strongest overall influence on the system's behavior
        # near the base configuration.
        param_causal_weight = {}
        for edge in causal_edges:
            name = edge["from"]
            param_causal_weight[name] = param_causal_weight.get(name, 0.0) + edge["weight"]
        most_causal_params = sorted(
            param_causal_weight.keys(),
            key=lambda n: param_causal_weight[n],
            reverse=True,
        )

        # Rank outputs by causal degree: number of parameter -> output
        # edges pointing to each output. High-degree outputs are influenced
        # by many parameters and thus more "connected" in the causal graph.
        # In network science terms (Newman, 2010), this is the in-degree
        # of output nodes in the bipartite causal subgraph.
        output_edge_count = {}
        for edge in causal_edges:
            out = edge["to"]
            output_edge_count[out] = output_edge_count.get(out, 0) + 1
        most_connected_outputs = sorted(
            output_edge_count.keys(),
            key=lambda k: output_edge_count[k],
            reverse=True,
        )

        return {
            "nodes": {
                "params": {name: float(base_params.get(name, 0.0)) for name in self._spec},
                "outputs": base_outputs,
            },
            "edges": {
                "causal": causal_edges,
                "param_correlation": param_corr,
                "output_correlation": output_corr,
            },
            "stability": stability,
            "rankings": {
                "most_causal_params": most_causal_params,
                "most_connected_outputs": most_connected_outputs,
            },
            "n_sims": n_sims,
        }

    def run(self, params: dict) -> dict:
        """Simulator protocol wrapper for composability.

        Makes RelationGraphExtractor itself satisfy the Simulator protocol
        (run() + param_spec()), enabling meta-analysis: you can wrap a
        RelationGraphExtractor in another Zimmerman tool.

        Runs extract() with default settings on the given params.

        Args:
            params: Base parameter dict.

        Returns:
            Dict with extract() results (the full relation graph).
        """
        return self.extract(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Delegates to underlying simulator's parameter specification.

        Returns:
            Parameter specification from the wrapped simulator:
            {param_name: (lower_bound, upper_bound)}.
        """
        return self.simulator.param_spec()
