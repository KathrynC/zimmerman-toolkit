"""Token extispicy workbench: quantifies tokenization-induced flattening as a hazard surface.

Implements the "token extispicy" analysis described in Zimmerman (2025) SS4.6.4.
The name is a deliberate metaphor: in the ancient Mesopotamian practice of
*extispicy*, diviners examined the entrails of sacrificial animals to predict
future events. Here, we examine the "entrails" of language models -- their
tokenization boundaries -- to predict where representational hazards lurk.
Just as a haruspex might find an ill-shaped liver to be an omen of disaster,
we find ill-shaped token boundaries to be omens of output degradation.

The core insight is that tokenization is not semantically neutral. When an LLM
mediates parameter generation (the "offer wave" in the TIQM framework), the way
numbers are tokenized affects what the model "sees." A parameter value like
3.14159 may become ["3", ".", "14", "159"] or ["3.14", "159"] depending on the
tokenizer. Higher fragmentation means the model must compose more tokens to
represent a single semantic value, increasing the risk of *representational
flattening* -- the loss of semantic precision through tokenization artifacts
(Zimmerman 2025, SS3.5.3).

**Plugin-based tokenizer architecture**: The workbench accepts any tokenizer
callback with signature ``(str) -> list[str]``, enabling comparison across
different tokenization strategies:
    - BPE (Byte Pair Encoding): as in GPT-family models. See Sennrich et al.
      (2016) "Neural Machine Translation of Rare Words with Subword Units"
      (arXiv:1508.07909) for the foundational BPE algorithm for NMT.
    - SentencePiece: Kudo & Richardson (2018) "SentencePiece: A simple and
      language independent subword tokenizer and detokenizer for Neural Text
      Processing" (arXiv:1808.06226). Unigram language model approach.
    - Whitespace fallback: the built-in default; splits on whitespace and
      digit/non-digit boundaries as a tokenizer-agnostic baseline.

Swapping tokenizers reveals how *fragmentation patterns differ* across models,
which is critical when the same simulator is interrogated by different LLMs.

This module measures five key quantities:
    - **Fragmentation rate**: ratio of tokens to characters. Higher values
      mean the tokenizer splits input more finely, potentially losing semantic
      coherence at token boundaries. A fragmentation rate of 0.5 means roughly
      one token per two characters -- typical for natural language. Numeric
      strings often fragment at 0.7+ due to digit-by-digit splitting.
    - **Perturbation token sensitivity**: how much the tokenization changes
      when input is slightly perturbed (by epsilon). This measures tokenizer
      *stability* -- an unstable tokenizer produces different token sequences
      for semantically similar inputs, breaking the LLM's ability to generalize
      across nearby parameter values.
    - **Fragmentation-output correlation**: the Pearson correlation between
      per-sample fragmentation rate and downstream simulator output quality.
      A strong correlation (|r| > 0.3) is evidence that tokenization artifacts
      propagate through the LLM into the simulator, validating the extispicy
      hypothesis: fragmentation *predicts* failure.
    - **Hazard zones**: parameter configurations with the highest fragmentation,
      forming Zimmerman's "hazard surface" -- a mapping of fragmentation across
      parameter space that identifies regions where tokenization artifacts are
      most likely to cause representational flattening and degraded outputs.
    - **Token edit vs string edit ratio**: how much larger token-level edits
      are compared to character-level edits, revealing the amplification factor
      of tokenization on semantic distance.

The workbench itself satisfies the Simulator protocol (``run()`` + ``param_spec()``),
so it can be fed to ``sobol_sensitivity()``, ``Falsifier``, etc. for meta-analysis
of tokenization effects as a first-class simulation dimension.

References:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont. SS3.5.3, SS4.6.4, SS4.7.6.

    Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation
    of Rare Words with Subword Units." Proceedings of the 54th Annual Meeting
    of the ACL. arXiv:1508.07909.

    Kudo, T. & Richardson, J. (2018). "SentencePiece: A simple and language
    independent subword tokenizer and detokenizer for Neural Text Processing."
    EMNLP 2018. arXiv:1808.06226.
"""

from __future__ import annotations

import json
import re

import numpy as np


def _default_tokenize(text: str) -> list[str]:
    """Default fallback tokenizer: whitespace + digit boundary splitting.

    This is the baseline tokenizer used when no external tokenizer callback
    is provided. It provides a tokenizer-agnostic approximation of how BPE
    and SentencePiece tokenizers handle numeric strings, without requiring
    any external dependencies (tiktoken, sentencepiece, etc.).

    The strategy is deliberately simple:
      1. Split on whitespace (approximates whitespace pre-tokenization in GPT-2/3)
      2. Within each whitespace-delimited chunk, split at every boundary between
         digit and non-digit characters (captures the key numeric fragmentation
         pattern that real tokenizers exhibit)

    This baseline is useful for two reasons:
      - It provides a lower bound on fragmentation: real BPE tokenizers may
        fragment *more* aggressively on rare numeric patterns.
      - It enables analysis even when the actual tokenizer is proprietary or
        unavailable (e.g., closed-source API models).

    Compare with real tokenizers by passing a ``tokenize`` callback to
    ``TokenExtispicyWorkbench.__init__()``.

    Examples:
        "x=3.14"  -> ["x=", "3", ".", "14"]   (4 tokens from 6 chars)
        "0.001"   -> ["0", ".", "001"]          (3 tokens from 5 chars)
        "hello"   -> ["hello"]                  (1 token, no fragmentation)
        "a 1 b"   -> ["a", "1", "b"]           (3 tokens from 5 chars)

    Args:
        text: Input string to tokenize.

    Returns:
        List of token strings. Empty strings are filtered out.
    """
    # Phase 1: Split on whitespace. This mirrors the pre-tokenization step
    # used by most BPE implementations (Sennrich et al. 2016, Section 3.2).
    parts = text.split()
    tokens = []
    for part in parts:
        # Phase 2: Split at boundaries between digit and non-digit characters.
        # Uses regex zero-width lookahead/lookbehind assertions to split without
        # consuming characters. This captures the key numeric fragmentation
        # pattern: "3.14" -> ["3", ".", "14"] rather than keeping "3.14" as
        # one token (which only sophisticated BPE merges would achieve).
        sub_tokens = re.split(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', part)
        tokens.extend(t for t in sub_tokens if t)
    return tokens


class TokenExtispicyWorkbench:
    """Analyzes how token fragmentation correlates with simulator behavior.

    This is the main class implementing Zimmerman's token extispicy analysis
    (SS4.6.4). Given a simulator and an optional tokenizer callback, it:
      1. Samples the parameter space using Latin hypercube-style random sampling
      2. Converts each parameter configuration to its string representation
      3. Tokenizes the string using the pluggable tokenizer
      4. Runs the simulator to obtain output metrics
      5. Correlates fragmentation patterns with output quality
      6. Identifies "hazard zones" -- regions of parameter space where
         tokenization artifacts are most severe

    **Plugin architecture**: The ``tokenize`` parameter accepts any callable
    with signature ``(str) -> list[str]``. This allows direct comparison of:
      - OpenAI's tiktoken (BPE): ``import tiktoken; enc = tiktoken.get_encoding("cl100k_base"); workbench = TokenExtispicyWorkbench(sim, tokenize=enc.encode)``
      - HuggingFace tokenizers: ``from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b"); workbench = TokenExtispicyWorkbench(sim, tokenize=tok.tokenize)``
      - SentencePiece: ``import sentencepiece as spm; sp = spm.SentencePieceProcessor(model_file='model.model'); workbench = TokenExtispicyWorkbench(sim, tokenize=sp.encode_as_pieces)``
      - Custom tokenizers for domain-specific analysis

    **Simulator protocol compliance**: The workbench itself satisfies the
    Simulator protocol (``run()`` + ``param_spec()``), so it can be composed
    with other zimmerman tools. For example, running ``sobol_sensitivity()``
    on a ``TokenExtispicyWorkbench`` reveals which parameters' tokenization
    fragmentation most affects output quality -- a meta-analysis of the
    tokenization hazard surface itself.

    Args:
        simulator: Any Simulator-compatible object with ``run(params) -> dict``
            and ``param_spec() -> dict[str, tuple[float, float]]``.
        tokenize: Callable ``(str) -> list[str]``. Converts a text string into
            a list of token strings. If None, falls back to the built-in
            whitespace + digit boundary tokenizer (``_default_tokenize``),
            which provides a tokenizer-agnostic baseline.

    Example:
        workbench = TokenExtispicyWorkbench(my_sim)
        report = workbench.analyze(n_samples=200, seed=42)
        print(report["fragmentation_stats"]["mean_tokens_per_param"])
        print(report["hazard_zones"][:3])

    References:
        Zimmerman (2025) SS4.6.4 -- Token extispicy methodology
        Sennrich et al. (2016) -- BPE tokenization fundamentals
    """

    def __init__(self, simulator, tokenize: callable | None = None):
        self.simulator = simulator
        self._spec = simulator.param_spec()
        # Plugin-based tokenizer architecture: accept any callable that
        # converts a string to a list of token strings. This enables
        # comparison across different tokenizer implementations (BPE,
        # SentencePiece, whitespace fallback) to reveal how fragmentation
        # patterns differ across models and tokenization strategies.
        if tokenize is not None:
            self._tokenize = tokenize
        else:
            # Fall back to the built-in whitespace + digit boundary tokenizer.
            # This default provides a conservative baseline: real BPE tokenizers
            # typically fragment numeric strings *more* aggressively, so the
            # default tends to underestimate fragmentation hazards.
            self._tokenize = _default_tokenize

    def default_tokenize(self, text: str) -> list[str]:
        """Public accessor for the built-in whitespace + digit boundary tokenizer.

        Exposes the default fallback tokenizer as an instance method, useful
        for comparing the baseline tokenization against a custom tokenizer
        passed to the constructor. For example, to measure how much more
        a BPE tokenizer fragments compared to the baseline:

            baseline_tokens = workbench.default_tokenize("3.14159")
            custom_tokens = workbench._tokenize("3.14159")
            fragmentation_amplification = len(custom_tokens) / len(baseline_tokens)

        Splits on whitespace, then further splits digits from non-digits.
        e.g., "x=3.14" -> ["x=", "3", ".", "14"]

        Args:
            text: Input string to tokenize.

        Returns:
            List of token strings.
        """
        return _default_tokenize(text)

    def params_to_string(self, params: dict) -> str:
        """Convert params dict to a deterministic JSON string representation.

        This is the critical serialization step in the extispicy pipeline:
        it determines *exactly what string the tokenizer sees*. Design choices
        here directly affect fragmentation measurements:
          - Sorted keys ensure deterministic ordering (same params always
            produce the same string, enabling meaningful token comparisons)
          - 6-decimal-place rounding balances precision against token count
            (more decimals = more digit tokens = higher fragmentation)
          - Compact JSON (no extra whitespace) avoids inflating character
            count, which would deflate the fragmentation rate

        In an LLM-mediated pipeline, this string (or something structurally
        similar) is what appears in the prompt when parameters are passed to
        the model. The tokenizer's treatment of this string determines how
        the LLM "perceives" the parameter values.

        Args:
            params: Dictionary mapping parameter names to numeric values.

        Returns:
            Compact JSON string with sorted keys and 6-decimal float precision.
        """
        formatted = {}
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, (int, np.integer)):
                formatted[key] = int(val)
            else:
                # Round to 6 decimal places -- a pragmatic choice that
                # preserves float64-level precision for most simulator
                # applications while keeping token count bounded.
                formatted[key] = round(float(val), 6)
        return json.dumps(formatted, sort_keys=True)

    def fragmentation_rate(self, text: str) -> float:
        """Compute tokens-per-character ratio: the core fragmentation metric.

        This is the fundamental measurement in token extispicy (Zimmerman 2025,
        SS4.6.4). The fragmentation rate quantifies how finely the tokenizer
        splits the input:
          - Low fragmentation (~0.1-0.2): efficient encoding, e.g. common
            English words that map to single tokens
          - Medium fragmentation (~0.3-0.5): typical for mixed alphanumeric text
          - High fragmentation (~0.5-1.0): aggressive splitting, common for
            numeric parameter strings where each digit becomes its own token

        Higher fragmentation means the tokenizer needs more tokens to represent
        the same text, which has two downstream effects:
          1. The LLM's attention mechanism must attend across more token
             positions to compose a single semantic value, increasing the
             risk of representational flattening (Zimmerman 2025, SS3.5.3)
          2. The parameter representation consumes more of the model's finite
             context window, potentially crowding out other useful information

        Args:
            text: Input string to measure fragmentation for.

        Returns:
            Ratio of token count to character count (float in [0, 1+]).
            Returns 0.0 for empty strings. Values above 1.0 are theoretically
            possible if the tokenizer generates more tokens than characters
            (e.g., by adding special boundary tokens).
        """
        if not text:
            return 0.0
        tokens = self._tokenize(text)
        # The ratio len(tokens)/len(text) directly measures fragmentation:
        # a ratio of 1.0 means every character is its own token (maximum
        # fragmentation for character-level tokenizers).
        return len(tokens) / len(text)

    def perturbation_token_sensitivity(self, params: dict, epsilon: float = 0.01) -> dict:
        """Measure tokenizer stability: how many tokens change per epsilon perturbation.

        This metric tests whether the tokenizer is *stable* with respect to
        small parameter changes (Zimmerman 2025, SS4.6.4). In a well-behaved
        tokenization, a tiny parameter change (e.g., 3.14 -> 3.15) should
        produce a small token change (one or two token substitutions). In a
        poorly-behaved tokenization, the same tiny change can cascade into
        a completely different token sequence -- the tokenizer is "unstable"
        in that region of parameter space.

        Tokenizer instability is problematic for LLM-mediated generation
        because it means the model cannot form consistent internal
        representations across nearby parameter values. If "3.14" and "3.15"
        tokenize into radically different sequences, the model effectively
        sees them as unrelated inputs, breaking its ability to interpolate
        or generalize across the parameter space.

        The method perturbs each parameter independently by epsilon * (hi - lo),
        re-tokenizes, and counts positional token mismatches. This is a
        simplified edit distance (Hamming-like, with length penalty) rather
        than full Levenshtein distance, chosen for computational efficiency
        when analyzing large parameter spaces.

        Args:
            params: Base parameter dictionary to perturb from.
            epsilon: Perturbation size as a fraction of each parameter's range.
                Default 0.01 (1% of range) is small enough to test local
                stability without crossing into qualitatively different regions.

        Returns:
            Dictionary mapping parameter names to the number of tokens that
            changed when that parameter was perturbed by epsilon. Higher
            values indicate greater tokenizer instability for that parameter.
        """
        base_str = self.params_to_string(params)
        base_tokens = self._tokenize(base_str)
        sensitivity = {}

        for name in sorted(self._spec.keys()):
            lo, hi = self._spec[name]
            # Perturbation magnitude scales with the parameter's range,
            # so epsilon has a consistent meaning across parameters with
            # different scales (e.g., [0,1] vs [0,1000]).
            delta = epsilon * (hi - lo)

            perturbed = dict(params)
            # Clip to bounds to avoid out-of-range values that would
            # produce invalid parameter strings.
            perturbed[name] = float(np.clip(params[name] + delta, lo, hi))

            perturbed_str = self.params_to_string(perturbed)
            perturbed_tokens = self._tokenize(perturbed_str)

            # Simplified edit distance: count positional mismatches plus
            # length difference. This is cheaper than full Levenshtein
            # (O(n) vs O(n*m)) and sufficient for the sensitivity analysis
            # since we expect token sequences of similar length.
            # A full LCS-based distance would be more accurate but the
            # positional alignment works well when perturbations are small.
            max_len = max(len(base_tokens), len(perturbed_tokens))
            n_changes = abs(len(base_tokens) - len(perturbed_tokens))

            min_len = min(len(base_tokens), len(perturbed_tokens))
            for i in range(min_len):
                if base_tokens[i] != perturbed_tokens[i]:
                    n_changes += 1

            sensitivity[name] = n_changes

        return sensitivity

    def _sample_params(self, rng: np.random.Generator) -> dict:
        """Generate a random parameter combination uniformly within bounds.

        Samples each parameter independently from a uniform distribution
        over its specified range [lo, hi]. This provides space-filling
        coverage of the parameter space for fragmentation analysis.

        Note: Unlike the Sobol sensitivity analysis (which uses Saltelli's
        quasi-random sampling for variance decomposition), this uses simple
        uniform random sampling since we need only aggregate statistics
        over the fragmentation surface, not variance-based indices.

        Args:
            rng: numpy random generator for reproducibility.

        Returns:
            Dictionary mapping parameter names to random float values
            within their specified bounds.
        """
        params = {}
        for name, (lo, hi) in self._spec.items():
            params[name] = float(rng.uniform(lo, hi))
        return params

    def analyze(
        self,
        base_params: dict | None = None,
        n_samples: int = 100,
        seed: int = 42,
    ) -> dict:
        """Full token extispicy analysis: map the fragmentation hazard surface.

        This is the main entry point for the extispicy analysis (Zimmerman 2025,
        SS4.6.4). It performs a five-phase analysis:

        Phase 1 -- **Sampling & fragmentation measurement**: For each of
        ``n_samples`` random parameter configurations, convert params to their
        string representation, tokenize, and measure fragmentation rate. Also
        run the simulator to capture output metrics for correlation analysis.

        Phase 2 -- **Fragmentation statistics**: Aggregate per-parameter token
        counts to identify which parameters fragment most severely. The
        ``max_fragmentation_param`` identifies the parameter most vulnerable
        to tokenization artifacts.

        Phase 3 -- **Perturbation sensitivity**: Test tokenizer stability at
        ``base_params`` by perturbing each parameter by epsilon and counting
        token changes. High sensitivity = tokenizer instability = LLM hazard.

        Phase 4 -- **Fragmentation-output correlation**: Compute Pearson
        correlation between fragmentation rate and each simulator output key.
        This is the key test of the extispicy hypothesis: does fragmentation
        *predict* output degradation? A significant correlation (|r| > 0.3)
        validates that tokenization artifacts propagate into simulation results.

        Phase 5 -- **Hazard zone identification & token edit analysis**: Find
        the top-10 highest-fragmentation parameter configurations (the "hazard
        surface" peaks) and measure the token-edit-to-string-edit amplification
        ratio across random parameter pairs.

        Args:
            base_params: Optional base parameter dict used as the center point
                for perturbation sensitivity analysis (Phase 3). If None, uses
                the midpoint of each parameter's range -- a natural "neutral"
                point in the parameter space.
            n_samples: Number of random parameter configurations to sample.
                Higher values give more reliable fragmentation statistics and
                correlations but require more simulator runs.
            seed: Random seed for reproducibility. Deterministic sampling
                is essential for comparing fragmentation across different
                tokenizers on the same parameter configurations.

        Returns:
            Dictionary with:
                "fragmentation_stats": Aggregate fragmentation metrics
                    (mean/std tokens per param, most-fragmented parameter),
                "perturbation_sensitivity": Per-parameter token change counts
                    when perturbed by epsilon (tokenizer stability map),
                "fragmentation_output_correlation": Pearson r between
                    fragmentation rate and each output key (the extispicy test),
                "hazard_zones": Top-10 highest-fragmentation configurations
                    with associated output degradation (hazard surface peaks),
                "token_edit_vs_string_edit": Mean and max ratio of normalized
                    token edit distance to string edit distance (amplification),
                "n_samples": Number of samples used,
                "n_sims": Total simulation runs performed.
        """
        rng = np.random.default_rng(seed)
        param_names = sorted(self._spec.keys())
        d = len(param_names)

        # Default base_params to midpoints -- the center of the parameter
        # hypercube, which serves as a natural reference point for
        # perturbation sensitivity analysis.
        if base_params is None:
            base_params = {
                name: (lo + hi) / 2.0
                for name, (lo, hi) in self._spec.items()
            }

        # ==================================================================
        # Phase 1: Sample parameter space, tokenize, and run simulations.
        # This is the most computationally expensive phase since it requires
        # n_samples simulator runs. The fragmentation measurements themselves
        # are cheap (string operations only).
        # ==================================================================
        all_params = []
        all_fragmentations = []
        per_param_token_counts = {name: [] for name in param_names}
        all_results = []
        n_sims = 0

        for _ in range(n_samples):
            params = self._sample_params(rng)
            all_params.append(params)

            # Tokenize the full parameter string to get the overall
            # fragmentation rate for this configuration.
            param_str = self.params_to_string(params)
            tokens = self._tokenize(param_str)
            frag_rate = self.fragmentation_rate(param_str)
            all_fragmentations.append(frag_rate)

            # Per-parameter token counts: tokenize each parameter value
            # individually to identify which parameters fragment most.
            # This isolates per-parameter fragmentation from the overhead
            # of JSON structure tokens (braces, colons, commas).
            for name in param_names:
                val_str = str(round(float(params[name]), 6))
                val_tokens = self._tokenize(val_str)
                per_param_token_counts[name].append(len(val_tokens))

            # Run the underlying simulator to obtain output metrics.
            # These outputs will be correlated with fragmentation in Phase 4.
            result = self.simulator.run(params)
            all_results.append(result)
            n_sims += 1

        frag_array = np.array(all_fragmentations)

        # ==================================================================
        # Phase 2: Fragmentation statistics.
        # Aggregate token counts to characterize the fragmentation landscape.
        # ==================================================================

        # tokens_per_digit measures how efficiently the tokenizer handles
        # numeric content specifically (ignoring JSON structure characters).
        # A ratio near 1.0 means each digit gets its own token (worst case);
        # ratios below 0.5 indicate the tokenizer merges digit sequences.
        tokens_per_digit_list = []
        for params in all_params:
            param_str = self.params_to_string(params)
            tokens = self._tokenize(param_str)
            digit_count = sum(1 for c in param_str if c.isdigit())
            if digit_count > 0:
                tokens_per_digit_list.append(len(tokens) / digit_count)

        # Mean tokens per parameter: averaged across all samples, identifies
        # which parameters are consistently most fragmented.
        mean_tokens_per_param = {
            name: float(np.mean(per_param_token_counts[name]))
            for name in param_names
        }
        overall_mean_tpp = float(np.mean(
            [mean_tokens_per_param[n] for n in param_names]
        ))
        overall_std_tpp = float(np.std(
            [mean_tokens_per_param[n] for n in param_names]
        ))

        # Identify the parameter that fragments most on average -- this is
        # the parameter most vulnerable to tokenization-induced representational
        # flattening (Zimmerman 2025, SS3.5.3).
        max_frag_param = max(param_names, key=lambda n: mean_tokens_per_param[n])

        fragmentation_stats = {
            "mean_tokens_per_param": overall_mean_tpp,
            "mean_tokens_per_digit": float(np.mean(tokens_per_digit_list)) if tokens_per_digit_list else 0.0,
            "std_tokens_per_param": overall_std_tpp,
            "max_fragmentation_param": max_frag_param,
        }

        # ==================================================================
        # Phase 3: Perturbation sensitivity at the base point.
        # Tests tokenizer stability (see perturbation_token_sensitivity docs).
        # ==================================================================
        perturbation_sensitivity = self.perturbation_token_sensitivity(
            base_params, epsilon=0.01
        )

        # ==================================================================
        # Phase 4: Fragmentation-output correlation.
        # This is the central test of the extispicy hypothesis: does token
        # fragmentation predict simulator output quality? We compute Pearson
        # correlation between the fragmentation rate vector and each numeric
        # output key across all n_samples configurations.
        #
        # A significant positive correlation means higher fragmentation
        # coincides with higher output values (which may be good or bad
        # depending on the output semantics). A significant negative
        # correlation means higher fragmentation coincides with lower values.
        # The absolute correlation |r| is what matters for hazard assessment.
        # ==================================================================

        # Determine numeric output keys from the first result dict.
        output_keys = []
        if all_results:
            for key, val in all_results[0].items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    if np.isfinite(val):
                        output_keys.append(key)

        frag_output_correlation = {}
        for key in output_keys:
            output_vals = np.zeros(n_samples)
            for i, result in enumerate(all_results):
                val = result.get(key, np.nan)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    output_vals[i] = float(val)
                else:
                    output_vals[i] = np.nan

            # Require at least 3 valid data points for a meaningful correlation.
            valid = np.isfinite(output_vals)
            if valid.sum() > 2:
                corr_matrix = np.corrcoef(frag_array[valid], output_vals[valid])
                corr = float(corr_matrix[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                frag_output_correlation[key] = corr
            else:
                frag_output_correlation[key] = 0.0

        # ==================================================================
        # Phase 5a: Hazard zone identification.
        # Map Zimmerman's "hazard surface" (SS4.6.4) by finding parameter
        # configurations where fragmentation is highest. These are the peaks
        # of the hazard surface -- regions of parameter space where
        # tokenization artifacts are most likely to cause problems.
        #
        # For each configuration, we also compute "output degradation" as
        # the mean absolute deviation from median output values (normalized
        # by scale). This pairs fragmentation with its downstream effect,
        # allowing practitioners to see whether high-fragmentation zones
        # actually produce degraded outputs.
        # ==================================================================

        # Compute output medians as reference points for degradation measurement.
        output_medians = {}
        for key in output_keys:
            vals = []
            for result in all_results:
                val = result.get(key, np.nan)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    v = float(val)
                    if np.isfinite(v):
                        vals.append(v)
            output_medians[key] = float(np.median(vals)) if vals else 0.0

        hazard_data = []
        for i in range(n_samples):
            # Output degradation: normalized mean absolute deviation from
            # median across all output keys. The scale factor (max of
            # |median| and 0.1) prevents division-by-zero for outputs
            # centered near zero while preserving relative magnitude
            # for larger outputs.
            degradation = 0.0
            n_keys = 0
            for key in output_keys:
                val = all_results[i].get(key, np.nan)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    v = float(val)
                    if np.isfinite(v):
                        median_val = output_medians[key]
                        scale = max(abs(median_val), 0.1)
                        degradation += abs(v - median_val) / scale
                        n_keys += 1
            if n_keys > 0:
                degradation /= n_keys
            hazard_data.append({
                "params": dict(all_params[i]),
                "fragmentation": float(frag_array[i]),
                "output_degradation": float(degradation),
            })

        # Sort by fragmentation descending to surface the hazard surface peaks.
        # The top 10 highest-fragmentation configurations are returned as
        # the "hazard zones" -- the parameter regions most at risk.
        hazard_data.sort(key=lambda h: h["fragmentation"], reverse=True)
        hazard_zones = hazard_data[:10]

        # ==================================================================
        # Phase 5b: Token edit vs string edit ratio.
        # Measures the "amplification factor" of tokenization on edit
        # distance. If a small string edit (one digit change) produces a
        # large token edit (many tokens rearranged), the tokenizer amplifies
        # distance in representation space. This amplification makes it
        # harder for LLMs to learn smooth mappings from parameters to outputs,
        # since nearby inputs in parameter space become distant in token space.
        #
        # A ratio of ~1.0 means tokenization preserves edit structure.
        # Ratios >> 1.0 indicate amplification (token space is "rougher"
        # than string space). Ratios << 1.0 indicate compression (rare,
        # but possible with very aggressive BPE merging of common patterns).
        # ==================================================================
        edit_ratios = []
        for _ in range(min(n_samples, 50)):
            params_a = self._sample_params(rng)
            params_b = self._sample_params(rng)

            str_a = self.params_to_string(params_a)
            str_b = self.params_to_string(params_b)

            tokens_a = self._tokenize(str_a)
            tokens_b = self._tokenize(str_b)

            # Character-level edit distance (simplified positional Hamming).
            max_str_len = max(len(str_a), len(str_b))
            min_str_len = min(len(str_a), len(str_b))
            str_diffs = abs(len(str_a) - len(str_b))
            for j in range(min_str_len):
                if str_a[j] != str_b[j]:
                    str_diffs += 1

            # Token-level edit distance (same simplified metric).
            max_tok_len = max(len(tokens_a), len(tokens_b))
            min_tok_len = min(len(tokens_a), len(tokens_b))
            tok_diffs = abs(len(tokens_a) - len(tokens_b))
            for j in range(min_tok_len):
                if tokens_a[j] != tokens_b[j]:
                    tok_diffs += 1

            # Normalize both edit distances by their respective sequence
            # lengths, then take the ratio. This gives a scale-independent
            # measure of how much tokenization amplifies edit distance.
            if str_diffs > 0 and max_str_len > 0 and max_tok_len > 0:
                str_edit_frac = str_diffs / max_str_len
                tok_edit_frac = tok_diffs / max_tok_len
                if str_edit_frac > 1e-12:
                    ratio = tok_edit_frac / str_edit_frac
                    edit_ratios.append(ratio)

        token_edit_vs_string_edit = {
            "mean_ratio": float(np.mean(edit_ratios)) if edit_ratios else 1.0,
            "max_ratio": float(np.max(edit_ratios)) if edit_ratios else 1.0,
        }

        return {
            "fragmentation_stats": fragmentation_stats,
            "perturbation_sensitivity": perturbation_sensitivity,
            "fragmentation_output_correlation": frag_output_correlation,
            "hazard_zones": hazard_zones,
            "token_edit_vs_string_edit": token_edit_vs_string_edit,
            "n_samples": n_samples,
            "n_sims": n_sims,
        }

    def run(self, params: dict) -> dict:
        """Simulator protocol: run underlying sim, augmented with fragmentation metrics.

        This method makes the workbench itself a valid Simulator, enabling
        composition with other zimmerman tools. For example:

            workbench = TokenExtispicyWorkbench(my_sim)
            # Now use the workbench AS a simulator with sobol_sensitivity:
            sobol_report = sobol_sensitivity(workbench, n=256)
            # The Sobol analysis will include _fragmentation_rate and
            # _token_count as output keys, revealing which parameters'
            # fragmentation most affects overall output quality.

        The augmented output keys (prefixed with ``_``) are:
          - ``_fragmentation_rate``: tokens per character for this parameter
            configuration (the extispicy metric)
          - ``_token_count``: absolute number of tokens produced by the
            tokenizer for the JSON-serialized parameter string

        Args:
            params: Dictionary mapping parameter names to float values.

        Returns:
            Dictionary with all original simulator outputs plus the two
            fragmentation metrics described above.
        """
        result = self.simulator.run(params)
        param_str = self.params_to_string(params)
        tokens = self._tokenize(param_str)
        # Augment the simulator output with fragmentation metrics so they
        # can be analyzed by downstream tools (Sobol, falsifier, etc.)
        result["_fragmentation_rate"] = self.fragmentation_rate(param_str)
        result["_token_count"] = len(tokens)
        return result

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Simulator protocol: delegates to the underlying simulator's param_spec.

        The workbench does not add any new parameters -- it only adds new
        *output* keys. The parameter space is identical to the underlying
        simulator's, ensuring that any analysis tool can seamlessly swap
        between the raw simulator and the extispicy-augmented workbench.

        Returns:
            Parameter specification from the underlying simulator:
            ``dict[str, tuple[float, float]]`` mapping parameter names
            to (lower_bound, upper_bound) tuples.
        """
        return self._spec
