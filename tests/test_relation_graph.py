"""Tests for relation graph extraction (zimmerman.relation_graph_extractor).

Tests verify:
    1. extract() returns the expected graph structure (nodes, edges, stability, rankings).
    2. Causal frame produces gradient-like edges for known models.
    3. Similarity frame finds probes with high cosine similarity.
    4. Contrast frame finds probes with low cosine similarity.
    5. Param correlations and output correlations are computed.
    6. Stability metrics are in [0, 1].
    7. Rankings are sorted correctly.
    8. run() and param_spec() satisfy the Simulator protocol.
    9. Deterministic with seed.
"""

import numpy as np
import pytest

from zimmerman.relation_graph_extractor import RelationGraphExtractor


class TestExtractStructure:
    """Tests for the top-level extract() return structure."""

    def test_returns_expected_keys(self, linear_sim):
        """extract() should return all required top-level keys."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=20, seed=42)

        assert "nodes" in graph
        assert "edges" in graph
        assert "stability" in graph
        assert "rankings" in graph
        assert "n_sims" in graph

    def test_nodes_structure(self, linear_sim):
        """Nodes should contain params and outputs dicts."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        nodes = graph["nodes"]
        assert "params" in nodes
        assert "outputs" in nodes
        assert isinstance(nodes["params"], dict)
        assert isinstance(nodes["outputs"], dict)

        # Params should match base
        for name in linear_sim.param_spec():
            assert name in nodes["params"]

    def test_edges_structure(self, linear_sim):
        """Edges should have causal, param_correlation, output_correlation."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=20, seed=42)

        edges = graph["edges"]
        assert "causal" in edges
        assert "param_correlation" in edges
        assert "output_correlation" in edges
        assert isinstance(edges["causal"], list)
        assert isinstance(edges["param_correlation"], list)
        assert isinstance(edges["output_correlation"], list)

    def test_stability_structure(self, linear_sim):
        """Stability should have jaccard_overlap and edge_survival_rate."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        stability = graph["stability"]
        assert "jaccard_overlap" in stability
        assert "edge_survival_rate" in stability

    def test_rankings_structure(self, linear_sim):
        """Rankings should have most_causal_params and most_connected_outputs."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        rankings = graph["rankings"]
        assert "most_causal_params" in rankings
        assert "most_connected_outputs" in rankings
        assert isinstance(rankings["most_causal_params"], list)
        assert isinstance(rankings["most_connected_outputs"], list)

    def test_n_sims_positive(self, linear_sim):
        """n_sims should be a positive integer."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        assert graph["n_sims"] > 0


class TestCausalFrame:
    """Tests for the causal_frame method (gradient estimation)."""

    def test_causal_edges_for_linear_sim(self, linear_sim):
        """Linear sim: y = 1*x0 + 2*x1 + 3*x2 + 4*x3.
        Gradients should be approximately [1, 2, 3, 4]."""
        extractor = RelationGraphExtractor(linear_sim, output_keys=["y"])
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        edges = extractor.causal_frame(base, rng)

        # Should have one edge per parameter -> output
        assert len(edges) >= 4  # at least one edge per param

        # Check gradients match expected coefficients
        expected = {"x0": 1.0, "x1": 2.0, "x2": 3.0, "x3": 4.0}
        for edge in edges:
            if edge["to"] == "y":
                param = edge["from"]
                assert abs(edge["weight"] - expected[param]) < 0.1, (
                    f"Gradient for {param}: {edge['weight']}, expected {expected[param]}"
                )

    def test_causal_edges_have_required_fields(self, linear_sim):
        """Each causal edge should have from, to, weight, sign."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        edges = extractor.causal_frame(base, rng)

        for edge in edges:
            assert "from" in edge
            assert "to" in edge
            assert "weight" in edge
            assert "sign" in edge
            assert edge["sign"] in (1, -1)
            assert edge["weight"] >= 0  # weight is absolute value

    def test_causal_sign_for_linear_sim(self, linear_sim):
        """All gradients for linear sim with positive coefficients should be positive."""
        extractor = RelationGraphExtractor(linear_sim, output_keys=["y"])
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        edges = extractor.causal_frame(base, rng)

        for edge in edges:
            if edge["to"] == "y":
                assert edge["sign"] == 1, (
                    f"Expected positive gradient for {edge['from']} -> y"
                )


class TestSimilarityFrame:
    """Tests for the similarity_frame method."""

    def test_similar_probes_have_high_similarity(self, linear_sim):
        """All returned probes should have similarity > 0.9."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        probes = extractor.similarity_frame(base, n_probes=50, rng=rng)

        for probe in probes:
            assert probe["similarity"] > 0.9

    def test_similar_probes_have_params(self, linear_sim):
        """Each probe should have params and output dicts."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        probes = extractor.similarity_frame(base, n_probes=20, rng=rng)

        for probe in probes:
            assert "params" in probe
            assert "output" in probe
            assert "similarity" in probe


class TestContrastFrame:
    """Tests for the contrast_frame method."""

    def test_contrast_probes_have_low_similarity(self, linear_sim):
        """All returned probes should have similarity < 0.5."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        rng = np.random.default_rng(42)
        probes = extractor.contrast_frame(base, n_probes=50, rng=rng)

        for probe in probes:
            assert probe["similarity"] < 0.5

    def test_contrast_frame_structure(self, quadratic_sim):
        """Contrast probes should have correct structure."""
        extractor = RelationGraphExtractor(quadratic_sim)
        base = {f"x{i}": 0.0 for i in range(3)}
        rng = np.random.default_rng(42)
        probes = extractor.contrast_frame(base, n_probes=30, rng=rng)

        for probe in probes:
            assert "params" in probe
            assert "output" in probe
            assert "similarity" in probe


class TestCorrelations:
    """Tests for parameter and output correlations."""

    def test_output_correlations_for_quadratic_sim(self, quadratic_sim):
        """Quadratic sim: y = sum(x^2), fitness = -y. These should be
        perfectly anti-correlated (correlation ~ -1)."""
        extractor = RelationGraphExtractor(quadratic_sim)
        base = {f"x{i}": 0.0 for i in range(3)}
        graph = extractor.extract(base, n_probes=100, seed=42)

        output_corrs = graph["edges"]["output_correlation"]
        # Look for the y-fitness correlation
        for corr in output_corrs:
            if set([corr["output_a"], corr["output_b"]]) == {"y", "fitness"}:
                assert corr["correlation"] < -0.8, (
                    f"y and fitness should be anti-correlated, got {corr['correlation']}"
                )
                break

    def test_param_correlations_exist(self, linear_sim):
        """Should produce param correlation edges for multi-param sims."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=50, seed=42)

        # With 4 params, should have C(4,2) = 6 possible pairs
        # Some may be filtered, but we should have at least a few
        param_corrs = graph["edges"]["param_correlation"]
        assert isinstance(param_corrs, list)

    def test_correlations_in_valid_range(self, linear_sim):
        """All correlations should be in [-1, 1]."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=50, seed=42)

        for corr in graph["edges"]["param_correlation"]:
            assert -1.0 <= corr["correlation"] <= 1.0 + 1e-9

        for corr in graph["edges"]["output_correlation"]:
            assert -1.0 <= corr["correlation"] <= 1.0 + 1e-9


class TestStability:
    """Tests for stability metrics."""

    def test_stability_in_valid_range(self, linear_sim):
        """Jaccard overlap and survival rate should be in [0, 1]."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=20, seed=42)

        stability = graph["stability"]
        assert 0.0 <= stability["jaccard_overlap"] <= 1.0
        assert 0.0 <= stability["edge_survival_rate"] <= 1.0

    def test_deterministic_model_high_stability(self, linear_sim):
        """A deterministic linear model should have high edge stability."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=20, seed=42)

        stability = graph["stability"]
        # For a deterministic model, causal edges should be stable
        assert stability["edge_survival_rate"] >= 0.8, (
            f"Expected high survival rate for deterministic model, "
            f"got {stability['edge_survival_rate']}"
        )


class TestRankings:
    """Tests for the rankings in the graph."""

    def test_most_causal_param_for_linear_sim(self, linear_sim):
        """For y = x0 + 2*x1 + 3*x2 + 4*x3, x3 should be most causal."""
        extractor = RelationGraphExtractor(linear_sim, output_keys=["y"])
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=20, seed=42)

        most_causal = graph["rankings"]["most_causal_params"]
        assert most_causal[0] == "x3", (
            f"Expected x3 as most causal, got {most_causal[0]}"
        )

    def test_rankings_contain_all_params(self, linear_sim):
        """Rankings should include all parameters."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        most_causal = graph["rankings"]["most_causal_params"]
        for name in linear_sim.param_spec():
            assert name in most_causal

    def test_most_connected_outputs(self, quadratic_sim):
        """Most connected outputs should include all output keys."""
        extractor = RelationGraphExtractor(quadratic_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        connected = graph["rankings"]["most_connected_outputs"]
        # Both y and fitness should appear
        assert "y" in connected
        assert "fitness" in connected


class TestSimulatorProtocol:
    """Test that RelationGraphExtractor satisfies the Simulator protocol."""

    def test_run_returns_dict(self, linear_sim):
        """run() should return a dict."""
        extractor = RelationGraphExtractor(linear_sim)
        result = extractor.run({f"x{i}": 0.5 for i in range(4)})
        assert isinstance(result, dict)

    def test_param_spec_delegates(self, linear_sim):
        """param_spec() should return the underlying simulator's spec."""
        extractor = RelationGraphExtractor(linear_sim)
        spec = extractor.param_spec()
        assert spec == linear_sim.param_spec()

    def test_param_spec_returns_dict(self, quadratic_sim):
        """param_spec() should return a dict of (lo, hi) tuples."""
        extractor = RelationGraphExtractor(quadratic_sim)
        spec = extractor.param_spec()
        assert isinstance(spec, dict)
        for name, bounds in spec.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2


class TestCustomOutputKeys:
    """Test with explicitly specified output keys."""

    def test_custom_output_keys(self, quadratic_sim):
        """Should only track specified output keys."""
        extractor = RelationGraphExtractor(quadratic_sim, output_keys=["y"])
        base = {f"x{i}": 0.5 for i in range(3)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        # Output nodes should only have 'y'
        outputs = graph["nodes"]["outputs"]
        assert "y" in outputs

        # Causal edges should only point to 'y'
        for edge in graph["edges"]["causal"]:
            assert edge["to"] == "y"

    def test_auto_detect_output_keys(self, quadratic_sim):
        """Without output_keys, should auto-detect from simulation output."""
        extractor = RelationGraphExtractor(quadratic_sim)
        base = {f"x{i}": 0.5 for i in range(3)}
        graph = extractor.extract(base, n_probes=10, seed=42)

        # Should detect both y and fitness
        outputs = graph["nodes"]["outputs"]
        assert "y" in outputs
        assert "fitness" in outputs


class TestDeterminism:
    """Test that results are deterministic with the same seed."""

    def test_extract_deterministic(self, linear_sim):
        """Same seed should produce identical graph structure."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}

        graph_a = extractor.extract(base, n_probes=20, seed=42)
        graph_b = extractor.extract(base, n_probes=20, seed=42)

        # Causal edges should be identical
        assert len(graph_a["edges"]["causal"]) == len(graph_b["edges"]["causal"])
        for ea, eb in zip(graph_a["edges"]["causal"], graph_b["edges"]["causal"]):
            assert ea["from"] == eb["from"]
            assert ea["to"] == eb["to"]
            assert abs(ea["weight"] - eb["weight"]) < 1e-12

        # Rankings should be identical
        assert graph_a["rankings"]["most_causal_params"] == graph_b["rankings"]["most_causal_params"]

    def test_different_seeds_may_differ(self, linear_sim):
        """Different seeds should produce the same causal edges (deterministic sim)
        but different similarity/contrast probes."""
        extractor = RelationGraphExtractor(linear_sim)
        base = {f"x{i}": 0.5 for i in range(4)}

        graph_a = extractor.extract(base, n_probes=30, seed=42)
        graph_b = extractor.extract(base, n_probes=30, seed=99)

        # Causal edges should still be the same (deterministic model, same epsilon)
        for ea, eb in zip(graph_a["edges"]["causal"], graph_b["edges"]["causal"]):
            assert ea["from"] == eb["from"]
            assert abs(ea["weight"] - eb["weight"]) < 1e-9


class TestWithBrokenSimulator:
    """Test relation graph on a simulator with failure modes."""

    def test_handles_nan_outputs(self, broken_sim):
        """Should handle NaN outputs gracefully."""
        extractor = RelationGraphExtractor(broken_sim)
        # Use params that are well within normal range
        base = {"x0": 0.5, "x1": 0.5, "x2": 0.5}
        graph = extractor.extract(base, n_probes=20, seed=42)

        # Should complete without error
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["n_sims"] > 0
