"""Sobol global sensitivity analysis via Saltelli sampling.

Generalized from how-to-live-much-longer/sobol_sensitivity.py into a
simulator-agnostic implementation. Works with any object satisfying the
Simulator protocol (run + param_spec).

Method:
    1. Generate N*(2D+2) parameter samples using Saltelli's cross-matrix
       scheme (D = number of parameters, N = base sample count).
    2. Run the simulator for each sample.
    3. Compute first-order (S1) and total-order (ST) Sobol indices for
       every numeric output key using the Jansen (1999) estimator.

Pure numpy implementation (no scipy/SALib dependency).

Reference:
    Saltelli, A. (2002). "Making best use of model evaluations to compute
    sensitivity indices." Computer Physics Communications, 145(2), 280-297.

    Jansen, M.J.W. (1999). "Analysis of variance designs for model output."
    Computer Physics Communications, 117(1-2), 35-43.
"""

from __future__ import annotations

import numpy as np


def saltelli_sample(n_base: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Saltelli's sampling matrices for Sobol analysis.

    Produces N*(D+2) samples by constructing matrices A, B, and D
    cross-matrices C^(i) where C^(i) = A with column i replaced from B.

    This is the standard Saltelli (2010) scheme where both S1 and ST
    are estimated from the same set of cross-matrices, requiring only
    N*(D+2) model evaluations (not N*(2D+2)).

    Args:
        n_base: Number of base samples (N). Larger N gives more accurate
            indices but requires more simulation runs.
        d: Number of parameters (D).
        rng: numpy random generator instance.

    Returns:
        np.ndarray of shape (N*(D+2), D) with all parameter samples
        in [0, 1]^d. The layout is:
            rows 0..N-1:           matrix A
            rows N..2N-1:          matrix B
            rows 2N..2N+N*D-1:    C cross-matrices (D blocks of N rows)
                                  C^(i) = A with col i from B
    """
    A = rng.random((n_base, d))
    B = rng.random((n_base, d))

    samples = [A, B]

    # Cross-matrices: C^(i) = A with column i replaced from B
    # Used for both S1 and ST estimation
    for i in range(d):
        C_i = A.copy()
        C_i[:, i] = B[:, i]
        samples.append(C_i)

    return np.vstack(samples)


def rescale_samples(
    samples_01: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Rescale samples from [0,1]^d to actual parameter ranges.

    Args:
        samples_01: Array of shape (N, D) with values in [0, 1].
        bounds: Array of shape (D, 2) with [low, high] for each parameter.

    Returns:
        Array of shape (N, D) with values in [low_i, high_i] for each
        parameter dimension i.
    """
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + samples_01 * (hi - lo)


def sobol_indices(
    y_A: np.ndarray,
    y_B: np.ndarray,
    y_C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order (S1) and total-order (ST) Sobol indices.

    Uses the Saltelli (2010) sampling scheme with Jansen (1999) estimators.

    Given cross-matrices C^(i) = A with column i replaced from B:

        S1_i = E[f(B) * (f(C_i) - f(A))] / Var(Y)
        ST_i = 0.5 * E[(f(A) - f(C_i))^2] / Var(Y)

    S1 measures the fraction of output variance explained by parameter i
    alone (main effect). ST measures the fraction due to parameter i AND
    all its interactions with other parameters. The difference ST - S1
    is the interaction contribution.

    For a purely additive model (no interactions):
        ST == S1 for all parameters
        sum(S1) == 1.0

    Args:
        y_A: Model output for matrix A, shape (N,).
        y_B: Model output for matrix B, shape (N,).
        y_C: Model output for C cross-matrices (A with col i from B),
            shape (D, N).

    Returns:
        Tuple (S1, ST) where each is an array of shape (D,).
        S1[i] is the first-order index for parameter i.
        ST[i] is the total-order index for parameter i.
    """
    d = y_C.shape[0]
    var_total = np.var(np.concatenate([y_A, y_B]))

    if var_total < 1e-12:
        return np.zeros(d), np.zeros(d)

    S1 = np.zeros(d)
    ST = np.zeros(d)

    for i in range(d):
        # First-order: Saltelli (2010) estimator
        # C[i] = A with col i from B. So C[i] and B share column i,
        # while C[i] and A share all OTHER columns. This isolates the
        # effect of parameter i.
        V_i = np.mean(y_B * (y_C[i] - y_A))
        S1[i] = V_i / var_total

        # Total-order: Jansen (1999) estimator
        # Measures variance due to everything involving parameter i.
        VT_i = 0.5 * np.mean((y_A - y_C[i]) ** 2)
        ST[i] = VT_i / var_total

    return S1, ST


def sobol_sensitivity(
    simulator,
    n_base: int = 256,
    seed: int = 42,
    output_keys: list[str] | None = None,
) -> dict:
    """Run Sobol sensitivity analysis on any Simulator-compatible object.

    This is the main entry point. It generates Saltelli samples, runs the
    simulator for each sample, and computes Sobol indices for all (or
    specified) numeric output keys.

    The key generalization from the JGC version: this works with ANY
    simulator that has run() and param_spec(), and analyzes ALL numeric
    outputs, not just hardcoded het_final/atp_final.

    Args:
        simulator: Any object with run(params) -> dict and
            param_spec() -> dict[str, (float, float)] methods.
        n_base: Base sample count. Total simulations = N * (2D + 2).
            Default 256. Larger values give more accurate indices.
        seed: Random seed for reproducibility.
        output_keys: List of output keys to analyze. If None, all numeric
            keys from the first simulation result are used.

    Returns:
        Dictionary with structure:
            {
                "n_base": int,
                "n_total_sims": int,
                "parameter_names": list[str],
                "output_keys": list[str],
                "<output_key>": {
                    "S1": {param_name: float, ...},
                    "ST": {param_name: float, ...},
                    "interaction": {param_name: float, ...},
                },
                ...
                "rankings": {
                    "<output_key>_most_influential_S1": [param_names sorted],
                    "<output_key>_most_interactive": [param_names sorted],
                    ...
                },
            }
    """
    spec = simulator.param_spec()
    param_names = list(spec.keys())
    d = len(param_names)
    bounds = np.array([spec[name] for name in param_names])

    rng = np.random.default_rng(seed)

    # Generate Saltelli samples: N*(D+2) total
    n_total = n_base * (d + 2)
    samples_01 = saltelli_sample(n_base, d, rng)
    samples = rescale_samples(samples_01, bounds)

    assert samples.shape[0] == n_total, (
        f"Expected {n_total} samples, got {samples.shape[0]}"
    )

    # Run all simulations
    results_list = []
    for idx in range(n_total):
        params = {name: float(samples[idx, j]) for j, name in enumerate(param_names)}
        result = simulator.run(params)
        results_list.append(result)

    # Determine output keys (all numeric keys from first result)
    if output_keys is None:
        output_keys = []
        first = results_list[0]
        for key, val in first.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isfinite(val):
                    output_keys.append(key)

    # Extract output arrays for each key
    output_arrays = {}
    for key in output_keys:
        arr = np.zeros(n_total)
        for idx, result in enumerate(results_list):
            val = result.get(key, np.nan)
            if isinstance(val, (int, float, np.integer, np.floating)):
                arr[idx] = float(val)
            else:
                arr[idx] = np.nan
        output_arrays[key] = arr

    # Compute Sobol indices for each output key
    analysis = {
        "n_base": n_base,
        "n_total_sims": n_total,
        "parameter_names": param_names,
        "output_keys": list(output_keys),
        "rankings": {},
    }

    for key in output_keys:
        y = output_arrays[key]

        # Extract sub-arrays matching Saltelli layout:
        # [0..N): A, [N..2N): B, [2N..2N+D*N): C cross-matrices
        y_A = y[:n_base]
        y_B = y[n_base:2 * n_base]
        y_C = y[2 * n_base:].reshape(d, n_base)

        S1, ST = sobol_indices(y_A, y_B, y_C)
        interaction = ST - S1

        analysis[key] = {
            "S1": {name: float(S1[i]) for i, name in enumerate(param_names)},
            "ST": {name: float(ST[i]) for i, name in enumerate(param_names)},
            "interaction": {name: float(interaction[i]) for i, name in enumerate(param_names)},
        }

        # Rankings
        analysis["rankings"][f"{key}_most_influential_S1"] = sorted(
            param_names, key=lambda n: S1[param_names.index(n)], reverse=True
        )
        analysis["rankings"][f"{key}_most_interactive"] = sorted(
            param_names, key=lambda n: interaction[param_names.index(n)], reverse=True
        )

    return analysis
