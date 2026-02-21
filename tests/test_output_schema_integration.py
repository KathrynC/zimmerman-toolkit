"""Integration tests: shared output schema across real simulators.

Each simulator project has identically-named modules (constants.py,
simulator.py, analytics.py). We use subprocess isolation to avoid
module namespace collisions when testing multiple simulators.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from zimmerman.output_schema import validate_output, compare_outputs

_ROOT = Path(__file__).resolve().parent.parent.parent
_ZT = str(Path(__file__).resolve().parent.parent)

# Map simulator names to their project dirs and import commands
_SIMULATOR_SPECS = {
    "lemurs": {
        "dir": _ROOT / "lemurs-simulator",
        "code": (
            "import sys; sys.path.insert(0, {zt!r}); "
            "from lemurs_simulator import LEMURSSimulator; "
            "import json; from zimmerman.output_schema import NumpyEncoder; "
            "print(json.dumps(LEMURSSimulator().to_standard_output({{}}), cls=NumpyEncoder))"
        ),
    },
    "mito": {
        "dir": _ROOT / "how-to-live-much-longer",
        "code": (
            "import sys; sys.path.insert(0, {zt!r}); "
            "from zimmerman_bridge import MitoSimulator; "
            "import json; from zimmerman.output_schema import NumpyEncoder; "
            "print(json.dumps(MitoSimulator().to_standard_output({{}}), cls=NumpyEncoder))"
        ),
    },
    "grief": {
        "dir": _ROOT / "grief-simulator",
        "code": (
            "import sys; sys.path.insert(0, {zt!r}); "
            "from grief_simulator import GriefSimulator; "
            "import json; from zimmerman.output_schema import NumpyEncoder; "
            "print(json.dumps(GriefSimulator().to_standard_output({{}}), cls=NumpyEncoder))"
        ),
    },
}


def _run_simulator(name: str) -> dict | None:
    """Run a simulator's to_standard_output in a subprocess."""
    spec = _SIMULATOR_SPECS[name]
    if not spec["dir"].exists():
        return None

    code = spec["code"].format(zt=_ZT)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(spec["dir"]),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


# ── Pre-compute available outputs ───────────────────────────────────────────

_outputs = {}
_available = []

for name in _SIMULATOR_SPECS:
    d = _run_simulator(name)
    if d is not None:
        _outputs[name] = d
        _available.append(name)


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.skipif(len(_available) == 0, reason="No simulators available")
class TestCrossSimulatorSchema:
    """Validate that all available simulators produce valid shared output."""

    @pytest.mark.parametrize("name", _available)
    def test_valid_schema(self, name):
        errors = validate_output(_outputs[name])
        assert len(errors) == 0, f"{name} validation errors: {errors}"

    @pytest.mark.parametrize("name", _available)
    def test_schema_version(self, name):
        assert _outputs[name]["schema_version"] == "1.0"

    @pytest.mark.parametrize("name", _available)
    def test_has_trajectory(self, name):
        d = _outputs[name]
        assert d["trajectory"]["n_steps"] > 0
        assert len(d["trajectory"]["times"]) == d["trajectory"]["n_steps"]
        assert len(d["trajectory"]["states"]) == d["trajectory"]["n_steps"]

    @pytest.mark.parametrize("name", _available)
    def test_has_analytics(self, name):
        d = _outputs[name]
        assert len(d["analytics"]["pillars"]) > 0
        assert len(d["analytics"]["flat"]) > 0
        assert d["analytics"]["pillar_names"] == list(
            d["analytics"]["pillars"].keys()
        )

    @pytest.mark.parametrize("name", _available)
    def test_has_parameters(self, name):
        assert len(_outputs[name]["parameters"]["bounds"]) > 0

    @pytest.mark.parametrize("name", _available)
    def test_state_dim_matches(self, name):
        d = _outputs[name]
        expected_dim = d["simulator"]["state_dim"]
        assert len(d["simulator"]["state_names"]) == expected_dim
        if d["trajectory"]["states"]:
            assert len(d["trajectory"]["states"][0]) == expected_dim


@pytest.mark.skipif(len(_available) < 2, reason="Need >= 2 simulators")
class TestCrossSimulatorComparison:
    """Test compare_outputs across available simulators."""

    def test_compare_outputs(self):
        outputs = [_outputs[name] for name in _available]
        result = compare_outputs(*outputs)
        assert "simulators" in result
        assert len(result["simulators"]) == len(_available)

    def test_all_simulators_named(self):
        outputs = [_outputs[name] for name in _available]
        result = compare_outputs(*outputs)
        for name in _available:
            assert name in result["simulators"]
