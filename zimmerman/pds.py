"""Parameter-Design-State (PDS) dimension mapper.

Generalized from how-to-live-much-longer/pds_mapping.py. Maps abstract
high-level dimensions (e.g., Power, Danger, Structure from Zimmerman 2025)
to concrete simulator parameter values.

The PDS concept allows domain experts to reason about a simulation in
terms of meaningful abstract dimensions rather than raw parameter values.
Each abstract dimension maps to a weighted combination of concrete
parameters.

Example:
    In the JGC mitochondrial simulator:
    - "Power" maps to metabolic_demand (weight +0.4) and
      genetic_vulnerability (weight -0.2)
    - "Danger" maps to inflammation_level (weight +0.25) and
      genetic_vulnerability (weight +0.3)

    In the ER robot simulator:
    - "Aggression" might map to motor_force (weight +0.5) and
      speed_limit (weight +0.3)
    - "Stability" might map to leg_spread (weight +0.4) and
      center_of_mass_height (weight -0.3)

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont. §4.6.4.
"""

from __future__ import annotations

import numpy as np


class PDSMapper:
    """Maps abstract dimensions to concrete simulator parameters.

    Each dimension (e.g., Power, Danger, Structure) is defined by a
    weighted mapping to one or more simulator parameters. The mapper
    converts dimension values (typically in [-1, +1]) to concrete
    parameter values within the simulator's bounds.

    The mapping is:
        param_value = base_value + sum(dimension_value_i * weight_i)
    where base_value is the midpoint of the parameter's range.

    Args:
        simulator: Any Simulator-compatible object.
        dimension_names: List of abstract dimension names.
        dimension_to_param_mapping: Dict mapping each dimension name to
            a dict of {param_name: weight}. Weights determine how strongly
            each dimension influences each parameter.

    Example:
        mapper = PDSMapper(
            simulator=my_sim,
            dimension_names=["power", "danger"],
            dimension_to_param_mapping={
                "power": {"motor_force": 0.3, "speed": 0.2},
                "danger": {"risk_tolerance": 0.5},
            },
        )
        params = mapper.map_dimensions_to_params({"power": 0.8, "danger": -0.3})
    """

    def __init__(
        self,
        simulator,
        dimension_names: list[str],
        dimension_to_param_mapping: dict[str, dict[str, float]],
    ):
        self.simulator = simulator
        self.dimension_names = list(dimension_names)
        self.mapping = dict(dimension_to_param_mapping)
        self._spec = simulator.param_spec()

        # Validate that all mapped parameters exist in the simulator
        all_mapped_params = set()
        for dim_name, param_weights in self.mapping.items():
            all_mapped_params.update(param_weights.keys())
        unknown = all_mapped_params - set(self._spec.keys())
        if unknown:
            raise ValueError(
                f"Mapped parameters not found in simulator param_spec: {unknown}"
            )

    def map_dimensions_to_params(self, dimension_values: dict[str, float]) -> dict[str, float]:
        """Convert abstract dimension values to a concrete parameter dict.

        For each simulator parameter, the value is computed as:
            value = midpoint + sum(dim_value * weight for each dimension that maps to this param)
        The result is clipped to the parameter's [low, high] bounds.

        Parameters not referenced by any dimension are set to their midpoint.

        Args:
            dimension_values: Dict mapping dimension names to float values.
                Typical range is [-1, +1] but not enforced.

        Returns:
            Dict mapping simulator parameter names to float values,
            all within the bounds defined by param_spec().
        """
        params = {}
        for param_name, (lo, hi) in self._spec.items():
            midpoint = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            offset = 0.0
            for dim_name, param_weights in self.mapping.items():
                if param_name in param_weights:
                    dim_val = dimension_values.get(dim_name, 0.0)
                    weight = param_weights[param_name]
                    offset += dim_val * weight * half_range
            value = midpoint + offset
            params[param_name] = float(np.clip(value, lo, hi))
        return params

    def audit_mapping(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> dict:
        """Audit how well abstract dimensions predict simulation outcomes.

        Generates random dimension settings, maps them to parameters,
        runs the simulator, and measures correlations between dimension
        values and output values.

        Args:
            n_samples: Number of random dimension settings to test.
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                "dimension_output_correlations": {dim_name: {output_key: corr}},
                "variance_explained": {output_key: R^2 from linear fit},
                "n_samples": int,
                "dimension_stats": {dim_name: {"mean": float, "std": float}},
        """
        rng = np.random.default_rng(seed)
        n_dims = len(self.dimension_names)

        # Generate random dimension values in [-1, +1]
        dim_matrix = rng.uniform(-1.0, 1.0, (n_samples, n_dims))

        # Run simulations
        results_list = []
        for i in range(n_samples):
            dim_vals = {
                name: float(dim_matrix[i, j])
                for j, name in enumerate(self.dimension_names)
            }
            params = self.map_dimensions_to_params(dim_vals)
            result = self.simulator.run(params)
            results_list.append(result)

        # Determine numeric output keys
        output_keys = []
        if results_list:
            for key, val in results_list[0].items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    if np.isfinite(val):
                        output_keys.append(key)

        # Build output matrix
        output_matrix = np.zeros((n_samples, len(output_keys)))
        for i, result in enumerate(results_list):
            for j, key in enumerate(output_keys):
                val = result.get(key, np.nan)
                output_matrix[i, j] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else np.nan

        # Compute correlations between dimensions and outputs
        correlations = {}
        for j_dim, dim_name in enumerate(self.dimension_names):
            correlations[dim_name] = {}
            dim_col = dim_matrix[:, j_dim]
            for j_out, out_key in enumerate(output_keys):
                out_col = output_matrix[:, j_out]
                # Handle NaN
                valid = np.isfinite(out_col)
                if valid.sum() > 2:
                    corr_matrix = np.corrcoef(dim_col[valid], out_col[valid])
                    corr = float(corr_matrix[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    correlations[dim_name][out_key] = corr
                else:
                    correlations[dim_name][out_key] = 0.0

        # Variance explained: R^2 from multivariate linear regression
        # y = X @ beta + residual, R^2 = 1 - Var(residual) / Var(y)
        variance_explained = {}
        for j_out, out_key in enumerate(output_keys):
            out_col = output_matrix[:, j_out]
            valid = np.isfinite(out_col)
            if valid.sum() > n_dims + 1:
                X = dim_matrix[valid]
                y = out_col[valid]
                y_var = np.var(y)
                if y_var < 1e-12:
                    variance_explained[out_key] = 1.0
                else:
                    # OLS: beta = (X^T X)^{-1} X^T y
                    X_aug = np.column_stack([X, np.ones(X.shape[0])])
                    try:
                        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                        y_pred = X_aug @ beta
                        residual_var = np.var(y - y_pred)
                        r2 = 1.0 - residual_var / y_var
                        variance_explained[out_key] = float(max(r2, 0.0))
                    except np.linalg.LinAlgError:
                        variance_explained[out_key] = 0.0
            else:
                variance_explained[out_key] = 0.0

        # Dimension stats
        dim_stats = {}
        for j, name in enumerate(self.dimension_names):
            col = dim_matrix[:, j]
            dim_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }

        return {
            "dimension_output_correlations": correlations,
            "variance_explained": variance_explained,
            "n_samples": n_samples,
            "dimension_stats": dim_stats,
            "output_keys": output_keys,
        }

    def sensitivity_per_dimension(self, n_samples: int = 100, seed: int = 42) -> dict[str, dict[str, float]]:
        """Measure which dimensions matter most for each output.

        For each dimension, computes the absolute correlation with each
        output. Higher absolute correlation means the dimension has more
        influence on that output.

        Args:
            n_samples: Number of random dimension settings.
            seed: Random seed.

        Returns:
            Dict mapping dimension names to dicts of
            {output_key: abs_correlation}. Also includes an "overall"
            key with the mean absolute correlation across all outputs.
        """
        audit = self.audit_mapping(n_samples=n_samples, seed=seed)
        correlations = audit["dimension_output_correlations"]

        sensitivity = {}
        for dim_name in self.dimension_names:
            dim_corrs = correlations.get(dim_name, {})
            abs_corrs = {key: abs(val) for key, val in dim_corrs.items()}
            if abs_corrs:
                overall = float(np.mean(list(abs_corrs.values())))
            else:
                overall = 0.0
            sensitivity[dim_name] = {**abs_corrs, "overall": overall}

        return sensitivity
