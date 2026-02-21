"""Tests for shared simulator output schema."""
import json

import numpy as np
import pytest

from zimmerman.output_schema import (
    NumpyEncoder,
    SimulatorOutput,
    compare_outputs,
    validate_output,
)


def _make_example_output(name="test", d=3, n=101):
    """Create a valid SimulatorOutput for testing."""
    rng = np.random.default_rng(42)
    return SimulatorOutput(
        simulator_name=name,
        simulator_description=f"Test simulator ({d}D)",
        state_dim=d,
        param_dim=2,
        state_names=[f"s{i}" for i in range(d)],
        time_unit="years",
        time_horizon=10.0,
        times=np.linspace(0, 10, n),
        states=rng.random((n, d)),
        input_params={"a": 1.0, "b": 2.0},
        param_bounds={"a": (0.0, 5.0), "b": (0.0, 10.0)},
        pillars={
            "outcome": {"metric_a": 1.5, "metric_b": 2.3},
            "risk": {"metric_c": 0.1},
        },
    )


class TestSimulatorOutput:
    """Test SimulatorOutput construction and serialization."""

    def test_construction(self):
        out = _make_example_output()
        assert out.simulator_name == "test"
        assert out.state_dim == 3

    def test_to_dict_structure(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["simulator"]["name"] == "test"
        assert d["simulator"]["state_dim"] == 3
        assert d["simulator"]["param_dim"] == 2
        assert d["trajectory"]["n_steps"] == 101
        assert len(d["trajectory"]["times"]) == 101
        assert len(d["trajectory"]["states"]) == 101
        assert len(d["trajectory"]["states"][0]) == 3
        assert "outcome" in d["analytics"]["pillars"]
        assert d["analytics"]["flat"]["outcome_metric_a"] == 1.5
        assert d["parameters"]["input"]["a"] == 1.0

    def test_json_serializable(self):
        out = _make_example_output()
        d = out.to_dict()
        s = json.dumps(d, cls=NumpyEncoder)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["schema_version"] == "1.0"

    def test_to_json_method(self):
        out = _make_example_output()
        s = out.to_json()
        parsed = json.loads(s)
        assert parsed["simulator"]["name"] == "test"

    def test_pillar_names_preserved(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["analytics"]["pillar_names"] == ["outcome", "risk"]

    def test_flat_keys_correct(self):
        out = _make_example_output()
        d = out.to_dict()
        assert "outcome_metric_a" in d["analytics"]["flat"]
        assert "outcome_metric_b" in d["analytics"]["flat"]
        assert "risk_metric_c" in d["analytics"]["flat"]
        assert len(d["analytics"]["flat"]) == 3

    def test_extra_arrays(self):
        out = _make_example_output()
        out.extra_arrays = {"heteroplasmy": np.linspace(0, 0.5, 101)}
        d = out.to_dict()
        assert "heteroplasmy" in d["trajectory"]["extra"]
        assert len(d["trajectory"]["extra"]["heteroplasmy"]) == 101

    def test_bounds_format(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["parameters"]["bounds"]["a"] == [0.0, 5.0]
        assert d["parameters"]["bounds"]["b"] == [0.0, 10.0]

    def test_dt_computed(self):
        out = _make_example_output()
        d = out.to_dict()
        assert abs(d["trajectory"]["dt"] - 0.1) < 0.001  # 10.0 / 100 steps

    def test_state_names_in_metadata(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["simulator"]["state_names"] == ["s0", "s1", "s2"]

    def test_empty_extra_arrays(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["trajectory"]["extra"] == {}

    def test_no_trajectory(self):
        out = SimulatorOutput(
            simulator_name="empty",
            pillars={"p": {"m": 1.0}},
        )
        d = out.to_dict()
        assert d["trajectory"]["n_steps"] == 0
        assert d["trajectory"]["times"] == []
        assert d["trajectory"]["states"] == []


class TestValidation:
    """Test validate_output checks."""

    def test_valid_output_passes(self):
        out = _make_example_output()
        d = out.to_dict()
        errors = validate_output(d)
        assert len(errors) == 0

    def test_missing_schema_version(self):
        out = _make_example_output()
        d = out.to_dict()
        del d["schema_version"]
        errors = validate_output(d)
        assert any("schema_version" in e for e in errors)

    def test_missing_section(self):
        out = _make_example_output()
        d = out.to_dict()
        del d["trajectory"]
        errors = validate_output(d)
        assert any("trajectory" in e for e in errors)

    def test_wrong_state_dim(self):
        out = _make_example_output()
        d = out.to_dict()
        d["simulator"]["state_dim"] = 5  # should be 3
        errors = validate_output(d)
        assert any("state_dim" in e for e in errors)

    def test_mismatched_times_states(self):
        out = _make_example_output()
        d = out.to_dict()
        d["trajectory"]["times"] = d["trajectory"]["times"][:50]
        errors = validate_output(d)
        assert any("mismatch" in e.lower() or "steps" in e.lower() for e in errors)

    def test_missing_pillars(self):
        out = _make_example_output()
        d = out.to_dict()
        del d["analytics"]["pillars"]
        errors = validate_output(d)
        assert any("pillars" in e for e in errors)

    def test_missing_flat(self):
        out = _make_example_output()
        d = out.to_dict()
        del d["analytics"]["flat"]
        errors = validate_output(d)
        assert any("flat" in e for e in errors)

    def test_flat_pillar_consistency(self):
        out = _make_example_output()
        d = out.to_dict()
        # Add extra flat key not in pillars
        d["analytics"]["flat"]["phantom_key"] = 999.0
        errors = validate_output(d)
        assert any("phantom_key" in str(e) for e in errors)


class TestCompareOutputs:
    """Test compare_outputs utility."""

    def test_compare_two(self):
        o1 = _make_example_output("sim_a")
        o2 = _make_example_output("sim_b")
        # Give them different analytics
        o2.pillars = {
            "outcome": {"metric_a": 3.0, "metric_b": 1.0},
            "risk": {"metric_c": 0.5},
        }
        result = compare_outputs(o1.to_dict(), o2.to_dict())
        assert result["simulators"] == ["sim_a", "sim_b"]
        assert len(result["shared_metrics"]) == 3
        assert result["per_metric"]["outcome_metric_a"]["range"] == 1.5

    def test_compare_needs_two(self):
        o1 = _make_example_output()
        result = compare_outputs(o1.to_dict())
        assert "error" in result

    def test_unique_metrics_tracked(self):
        o1 = _make_example_output("a")
        o2 = SimulatorOutput(
            simulator_name="b",
            pillars={"outcome": {"metric_a": 1.0}, "special": {"unique_m": 9.0}},
        )
        result = compare_outputs(o1.to_dict(), o2.to_dict())
        assert "outcome_metric_a" in result["shared_metrics"]
        # outcome_metric_b and risk_metric_c only in a
        assert "outcome_metric_b" in result["unique_metrics"]["a"]
        # special_unique_m only in b
        assert "special_unique_m" in result["unique_metrics"]["b"]

    def test_compare_three(self):
        outputs = [_make_example_output(f"sim_{i}") for i in range(3)]
        result = compare_outputs(*[o.to_dict() for o in outputs])
        assert len(result["simulators"]) == 3


class TestNumpyEncoder:
    """Test NumpyEncoder handles all numpy types."""

    def test_ndarray(self):
        d = {"arr": np.array([1.0, 2.0, 3.0])}
        s = json.dumps(d, cls=NumpyEncoder)
        assert json.loads(s) == {"arr": [1.0, 2.0, 3.0]}

    def test_integer(self):
        d = {"n": np.int64(42)}
        s = json.dumps(d, cls=NumpyEncoder)
        assert json.loads(s) == {"n": 42}

    def test_floating(self):
        d = {"x": np.float64(3.14)}
        s = json.dumps(d, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert abs(parsed["x"] - 3.14) < 1e-10

    def test_bool(self):
        d = {"b": np.bool_(True)}
        s = json.dumps(d, cls=NumpyEncoder)
        assert json.loads(s) == {"b": True}
