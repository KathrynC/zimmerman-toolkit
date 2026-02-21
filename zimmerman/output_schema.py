"""Shared output schema for ODE simulator results.

Defines a common JSON envelope that any ODE simulator can produce,
enabling cross-simulator analysis, visualization, and comparison
without per-simulator special-casing.

Schema structure::

    {
        "schema_version": "1.0",
        "simulator": {name, description, state_dim, param_dim, state_names, time_unit, time_horizon},
        "trajectory": {times, states, n_steps, dt, extra},
        "analytics": {pillars, pillar_names, flat},
        "parameters": {input, bounds},
    }

Usage::

    from zimmerman.output_schema import SimulatorOutput, validate_output

    output = SimulatorOutput(
        simulator_name="lemurs",
        state_dim=14,
        state_names=[...],
        time_unit="weeks",
        time_horizon=15.0,
        times=result["times"],
        states=result["states"],
        pillars=analytics_dict,
        input_params=params,
        param_bounds=sim.param_spec(),
    )
    d = output.to_dict()
    errors = validate_output(d)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


@dataclass
class SimulatorOutput:
    """Standardized simulator output envelope.

    Wraps trajectory data, analytics, and parameters into a common
    format that cross-simulator tools can consume.
    """

    # Metadata
    simulator_name: str
    simulator_description: str = ""
    state_dim: int = 0
    param_dim: int = 0
    state_names: list[str] = field(default_factory=list)
    time_unit: str = "years"
    time_horizon: float = 0.0

    # Trajectory
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)

    # Analytics
    pillars: dict[str, dict[str, float]] = field(default_factory=dict)

    # Parameters
    input_params: dict = field(default_factory=dict)
    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict following the shared schema."""
        times_list = self.times.tolist() if self.times is not None else []
        states_list = self.states.tolist() if self.states is not None else []
        n_steps = len(times_list)

        # Compute mean dt
        if self.times is not None and len(self.times) > 1:
            dt = float(np.mean(np.diff(self.times)))
        else:
            dt = 0.0

        # Flatten analytics: pillar_metric -> value
        flat = {}
        for pillar_name, metrics in self.pillars.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    flat[f"{pillar_name}_{metric_name}"] = float(value)

        # Extra arrays to lists
        extra = {}
        for name, arr in self.extra_arrays.items():
            extra[name] = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)

        # Bounds as lists (JSON has no tuples)
        bounds = {k: list(v) for k, v in self.param_bounds.items()}

        return {
            "schema_version": "1.0",
            "simulator": {
                "name": self.simulator_name,
                "description": self.simulator_description,
                "state_dim": self.state_dim,
                "param_dim": self.param_dim,
                "state_names": list(self.state_names),
                "time_unit": self.time_unit,
                "time_horizon": self.time_horizon,
            },
            "trajectory": {
                "times": times_list,
                "states": states_list,
                "n_steps": n_steps,
                "dt": dt,
                "extra": extra,
            },
            "analytics": {
                "pillars": dict(self.pillars),
                "pillar_names": list(self.pillars.keys()),
                "flat": flat,
            },
            "parameters": {
                "input": dict(self.input_params),
                "bounds": bounds,
            },
        }

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), cls=NumpyEncoder, **kwargs)


def validate_output(d: dict) -> list[str]:
    """Validate a dict against the shared output schema.

    Returns a list of error messages. Empty list = valid.
    """
    errors = []

    # Top-level keys
    if "schema_version" not in d:
        errors.append("Missing required key: schema_version")
    for section in ["simulator", "trajectory", "analytics", "parameters"]:
        if section not in d:
            errors.append(f"Missing required section: {section}")

    if errors:
        return errors  # can't validate further

    # Simulator metadata
    sim = d["simulator"]
    if "name" not in sim:
        errors.append("Missing simulator.name")
    if "state_dim" not in sim:
        errors.append("Missing simulator.state_dim")

    # Trajectory consistency
    traj = d["trajectory"]
    times = traj.get("times", [])
    states = traj.get("states", [])

    if len(times) != len(states):
        errors.append(
            f"Trajectory length mismatch: times has {len(times)} steps, "
            f"states has {len(states)} steps"
        )

    if states and sim.get("state_dim"):
        actual_dim = len(states[0]) if states[0] else 0
        if actual_dim != sim["state_dim"]:
            errors.append(
                f"state_dim mismatch: simulator says {sim['state_dim']}, "
                f"trajectory has {actual_dim} columns"
            )

    # Analytics
    analytics = d["analytics"]
    if "pillars" not in analytics:
        errors.append("Missing analytics.pillars")
    if "flat" not in analytics:
        errors.append("Missing analytics.flat")

    # Verify flat keys match pillars
    if "pillars" in analytics and "flat" in analytics:
        expected_flat = set()
        for pillar_name, metrics in analytics["pillars"].items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    expected_flat.add(f"{pillar_name}_{metric_name}")
        actual_flat = set(analytics["flat"].keys())
        missing = expected_flat - actual_flat
        extra = actual_flat - expected_flat
        if missing:
            errors.append(f"Flat keys missing from pillars: {missing}")
        if extra:
            errors.append(f"Flat keys not in pillars: {extra}")

    return errors


def compare_outputs(*outputs: dict) -> dict:
    """Compare multiple simulator outputs sharing the same schema.

    Args:
        *outputs: SimulatorOutput.to_dict() results.

    Returns:
        Comparison dict with per-metric deltas and rankings.
    """
    if len(outputs) < 2:
        return {"error": "Need at least 2 outputs to compare"}

    names = [o["simulator"]["name"] for o in outputs]

    # Find shared flat metric keys
    all_keys = [set(o["analytics"]["flat"].keys()) for o in outputs]
    shared_keys = set(all_keys[0])  # copy to avoid mutating all_keys[0]
    for ks in all_keys[1:]:
        shared_keys &= ks

    comparison = {
        "simulators": names,
        "shared_metrics": sorted(shared_keys),
        "unique_metrics": {
            name: sorted(ks - shared_keys)
            for name, ks in zip(names, all_keys)
        },
        "per_metric": {},
    }

    for key in sorted(shared_keys):
        values = [o["analytics"]["flat"][key] for o in outputs]
        comparison["per_metric"][key] = {
            "values": dict(zip(names, values)),
            "range": float(max(values) - min(values)),
        }

    return comparison
