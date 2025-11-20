"""
DesignSpaceExplorer: Design space exploration and analysis tools.

This module implements methods for exploring the PV system design space
through parameter sweeps, sensitivity analysis, Monte Carlo simulation,
and Pareto frontier analysis.
"""

from typing import Callable, List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.stats import qmc, spearmanr
import concurrent.futures
from dataclasses import dataclass

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    DesignPoint,
    SensitivityResult,
    ParetoSolution,
)


@dataclass
class ParameterRange:
    """Parameter range for exploration."""
    name: str
    min_value: float
    max_value: float
    num_samples: int = 20


class DesignSpaceExplorer:
    """
    Design space exploration and analysis.

    This class provides comprehensive tools for exploring the PV system
    design space including parameter sweeps, sensitivity analysis,
    Monte Carlo simulation, and Pareto frontier analysis.

    Attributes:
        parameters: Base PV system parameters
        constraints: Optimization constraints
        evaluator: Function to evaluate design performance
    """

    def __init__(
        self,
        parameters: PVSystemParameters,
        constraints: OptimizationConstraints,
        evaluator: Optional[Callable[[PVSystemParameters], DesignPoint]] = None,
    ):
        """
        Initialize design space explorer.

        Args:
            parameters: Base PV system parameters
            constraints: Optimization constraints
            evaluator: Function to evaluate design points
        """
        self.parameters = parameters
        self.constraints = constraints
        self.evaluator = evaluator

    def parameter_sweep(
        self,
        param_ranges: List[ParameterRange],
        output_metric: str = "lcoe",
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform parameter sweep over specified ranges.

        This method systematically varies parameters across their ranges
        and evaluates the output metric at each point.

        Args:
            param_ranges: List of parameter ranges to sweep
            output_metric: Output metric to track ('lcoe', 'energy', 'npv', etc.)
            parallel: Whether to use parallel processing

        Returns:
            Dictionary with sweep results including parameter values and outputs
        """
        # Generate sample points for each parameter
        if len(param_ranges) == 1:
            # 1D sweep
            param_values = np.linspace(
                param_ranges[0].min_value,
                param_ranges[0].max_value,
                param_ranges[0].num_samples,
            )
            results = []

            def evaluate_point(value: float) -> float:
                """Evaluate single point."""
                params = self.parameters.model_copy(deep=True)
                setattr(params, param_ranges[0].name, value)
                design = self._evaluate_design(params)
                return getattr(design, output_metric)

            if parallel:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(evaluate_point, param_values))
            else:
                results = [evaluate_point(v) for v in param_values]

            return {
                "parameter": param_ranges[0].name,
                "values": param_values.tolist(),
                "output_metric": output_metric,
                "results": results,
                "dimension": 1,
            }

        elif len(param_ranges) == 2:
            # 2D sweep
            param1_values = np.linspace(
                param_ranges[0].min_value,
                param_ranges[0].max_value,
                param_ranges[0].num_samples,
            )
            param2_values = np.linspace(
                param_ranges[1].min_value,
                param_ranges[1].max_value,
                param_ranges[1].num_samples,
            )

            X, Y = np.meshgrid(param1_values, param2_values)
            results = np.zeros_like(X)

            def evaluate_point(i: int, j: int) -> Tuple[int, int, float]:
                """Evaluate grid point."""
                params = self.parameters.model_copy(deep=True)
                setattr(params, param_ranges[0].name, X[i, j])
                setattr(params, param_ranges[1].name, Y[i, j])
                design = self._evaluate_design(params)
                return i, j, getattr(design, output_metric)

            points = [(i, j) for i in range(len(param2_values)) for j in range(len(param1_values))]

            if parallel:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for i, j, result in executor.map(lambda p: evaluate_point(*p), points):
                        results[i, j] = result
            else:
                for i, j in points:
                    _, _, results[i, j] = evaluate_point(i, j)

            return {
                "parameters": [param_ranges[0].name, param_ranges[1].name],
                "param1_values": param1_values.tolist(),
                "param2_values": param2_values.tolist(),
                "output_metric": output_metric,
                "results": results.tolist(),
                "dimension": 2,
            }

        else:
            # Multi-dimensional sweep using Latin Hypercube Sampling
            n_samples = param_ranges[0].num_samples
            n_dims = len(param_ranges)

            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=n_dims)
            samples = sampler.random(n=n_samples)

            # Scale to parameter ranges
            param_values = np.zeros((n_samples, n_dims))
            for i, param_range in enumerate(param_ranges):
                param_values[:, i] = (
                    samples[:, i] * (param_range.max_value - param_range.min_value)
                    + param_range.min_value
                )

            def evaluate_sample(sample_idx: int) -> float:
                """Evaluate single sample."""
                params = self.parameters.model_copy(deep=True)
                for i, param_range in enumerate(param_ranges):
                    setattr(params, param_range.name, param_values[sample_idx, i])
                design = self._evaluate_design(params)
                return getattr(design, output_metric)

            if parallel:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(evaluate_sample, range(n_samples)))
            else:
                results = [evaluate_sample(i) for i in range(n_samples)]

            return {
                "parameters": [pr.name for pr in param_ranges],
                "parameter_values": param_values.tolist(),
                "output_metric": output_metric,
                "results": results,
                "dimension": n_dims,
                "sampling_method": "latin_hypercube",
            }

    def sensitivity_analysis(
        self,
        parameters_to_vary: List[str],
        variation_percent: float = 10.0,
        output_metrics: List[str] = ["lcoe", "annual_energy_kwh", "npv"],
    ) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis using one-at-a-time (OAT) method.

        This method varies each parameter individually while keeping others
        constant to assess the impact on output metrics.

        Args:
            parameters_to_vary: List of parameter names to vary
            variation_percent: Percentage variation (+/- from base)
            output_metrics: List of output metrics to analyze

        Returns:
            List of SensitivityResult objects for each parameter
        """
        results = []

        for param_name in parameters_to_vary:
            # Get base value
            base_value = getattr(self.parameters, param_name)

            # Create variation range
            variation = base_value * variation_percent / 100
            param_values = np.linspace(
                base_value - variation,
                base_value + variation,
                11,  # -10%, -8%, ..., 0%, ..., +10%
            )

            # Evaluate at each point
            output_data = {metric: [] for metric in output_metrics}

            for value in param_values:
                params = self.parameters.model_copy(deep=True)
                setattr(params, param_name, float(value))
                design = self._evaluate_design(params)

                for metric in output_metrics:
                    output_data[metric].append(getattr(design, metric))

            # Calculate sensitivity indices for each output metric
            for metric in output_metrics:
                # Normalize output values
                base_output = output_data[metric][5]  # Middle value (0% variation)
                normalized_outputs = [
                    (out - base_output) / base_output if base_output != 0 else 0
                    for out in output_data[metric]
                ]

                # Sensitivity index: (ΔOutput/Output) / (ΔInput/Input)
                output_range = max(normalized_outputs) - min(normalized_outputs)
                input_range = 2 * variation_percent / 100  # ±variation_percent
                sensitivity_index = output_range / input_range if input_range != 0 else 0

                # Correlation
                correlation = np.corrcoef(param_values, output_data[metric])[0, 1]

                results.append(
                    SensitivityResult(
                        parameter_name=f"{param_name}_{metric}",
                        parameter_values=param_values.tolist(),
                        output_values=output_data[metric],
                        sensitivity_index=abs(sensitivity_index),
                        correlation=correlation,
                    )
                )

        return results

    def monte_carlo_simulation(
        self,
        num_samples: int = 1000,
        uncertainty_ranges: Dict[str, Tuple[float, float]] = None,
        output_metrics: List[str] = ["lcoe", "npv"],
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation with parameter uncertainties.

        This method samples parameters from specified uncertainty distributions
        to assess output variability and risk.

        Args:
            num_samples: Number of Monte Carlo samples
            uncertainty_ranges: Dictionary of parameter: (min, max) ranges
            output_metrics: Output metrics to track
            parallel: Whether to use parallel processing

        Returns:
            Dictionary with simulation results and statistics
        """
        if uncertainty_ranges is None:
            # Default uncertainty ranges (±10% for key parameters)
            uncertainty_ranges = {
                "module_efficiency": (
                    self.parameters.module_efficiency * 0.95,
                    self.parameters.module_efficiency * 1.05,
                ),
                "module_cost": (
                    self.parameters.module_cost * 0.9,
                    self.parameters.module_cost * 1.1,
                ),
                "discount_rate": (
                    self.parameters.discount_rate * 0.8,
                    self.parameters.discount_rate * 1.2,
                ),
                "degradation_rate": (
                    self.parameters.degradation_rate * 0.8,
                    self.parameters.degradation_rate * 1.2,
                ),
            }

        # Generate samples using Sobol sequence for better coverage
        n_params = len(uncertainty_ranges)
        sampler = qmc.Sobol(d=n_params, scramble=True)
        samples = sampler.random(n=num_samples)

        # Scale to uncertainty ranges
        param_names = list(uncertainty_ranges.keys())
        param_samples = np.zeros((num_samples, n_params))

        for i, param_name in enumerate(param_names):
            min_val, max_val = uncertainty_ranges[param_name]
            param_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val

        def evaluate_sample(sample_idx: int) -> Dict[str, float]:
            """Evaluate single Monte Carlo sample."""
            params = self.parameters.model_copy(deep=True)
            for i, param_name in enumerate(param_names):
                setattr(params, param_name, float(param_samples[sample_idx, i]))

            design = self._evaluate_design(params)
            return {metric: getattr(design, metric) for metric in output_metrics}

        # Run simulation
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_sample, range(num_samples)))
        else:
            results = [evaluate_sample(i) for i in range(num_samples)]

        # Aggregate results
        output_data = {metric: [] for metric in output_metrics}
        for result in results:
            for metric in output_metrics:
                output_data[metric].append(result[metric])

        # Calculate statistics
        statistics = {}
        for metric in output_metrics:
            values = np.array(output_data[metric])
            statistics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }

        return {
            "num_samples": num_samples,
            "parameter_names": param_names,
            "parameter_samples": param_samples.tolist(),
            "output_metrics": output_metrics,
            "output_data": output_data,
            "statistics": statistics,
        }

    def pareto_frontier_analysis(
        self,
        objective1: str = "lcoe",
        objective2: str = "annual_energy_kwh",
        num_points: int = 50,
        minimize_obj1: bool = True,
        minimize_obj2: bool = False,
    ) -> List[ParetoSolution]:
        """
        Generate Pareto frontier for two competing objectives.

        This method finds the Pareto-optimal trade-off between two
        objectives through multi-objective optimization.

        Args:
            objective1: First objective (e.g., 'lcoe')
            objective2: Second objective (e.g., 'annual_energy_kwh')
            num_points: Number of points to sample
            minimize_obj1: Whether to minimize objective 1
            minimize_obj2: Whether to minimize objective 2

        Returns:
            List of ParetoSolution objects on the frontier
        """
        # Sample design space using Latin Hypercube
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=num_points)

        # Scale to constraint ranges
        designs = []
        for sample in samples:
            gcr = (
                sample[0] * (self.constraints.max_gcr - self.constraints.min_gcr)
                + self.constraints.min_gcr
            )
            dc_ac = (
                sample[1] * (self.constraints.max_dc_ac_ratio - self.constraints.min_dc_ac_ratio)
                + self.constraints.min_dc_ac_ratio
            )
            tilt = (
                sample[2] * (self.constraints.max_tilt - self.constraints.min_tilt)
                + self.constraints.min_tilt
            )

            params = self.parameters.model_copy(deep=True)
            params.gcr = float(gcr)
            params.dc_ac_ratio = float(dc_ac)
            params.tilt_angle = float(tilt)

            design = self._evaluate_design(params)
            designs.append(design)

        # Extract objectives
        obj1_values = np.array([getattr(d, objective1) for d in designs])
        obj2_values = np.array([getattr(d, objective2) for d in designs])

        # Normalize objectives for dominance check
        obj1_normalized = obj1_values if minimize_obj1 else -obj1_values
        obj2_normalized = obj2_values if minimize_obj2 else -obj2_values

        # Find Pareto front using non-dominated sorting
        pareto_front_indices = self._non_dominated_sort(
            np.column_stack([obj1_normalized, obj2_normalized])
        )

        # Create Pareto solutions
        pareto_solutions = []
        for idx in pareto_front_indices:
            pareto_solutions.append(
                ParetoSolution(
                    design=designs[idx],
                    objectives={
                        objective1: obj1_values[idx],
                        objective2: obj2_values[idx],
                    },
                    rank=0,
                    crowding_distance=0.0,
                )
            )

        # Calculate crowding distances
        if len(pareto_solutions) > 2:
            crowding_distances = self._calculate_crowding_distance(
                np.array([[s.objectives[objective1], s.objectives[objective2]]
                         for s in pareto_solutions])
            )
            for i, solution in enumerate(pareto_solutions):
                solution.crowding_distance = crowding_distances[i]

        return pareto_solutions

    def constraint_handling(
        self,
        design: DesignPoint,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a design satisfies all constraints.

        Args:
            design: Design point to check

        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []

        # GCR constraints
        if design.gcr < self.constraints.min_gcr:
            violations.append(f"GCR {design.gcr:.3f} < minimum {self.constraints.min_gcr}")
        if design.gcr > self.constraints.max_gcr:
            violations.append(f"GCR {design.gcr:.3f} > maximum {self.constraints.max_gcr}")

        # DC/AC ratio constraints
        if design.dc_ac_ratio < self.constraints.min_dc_ac_ratio:
            violations.append(
                f"DC/AC ratio {design.dc_ac_ratio:.3f} < minimum {self.constraints.min_dc_ac_ratio}"
            )
        if design.dc_ac_ratio > self.constraints.max_dc_ac_ratio:
            violations.append(
                f"DC/AC ratio {design.dc_ac_ratio:.3f} > maximum {self.constraints.max_dc_ac_ratio}"
            )

        # Tilt constraints
        if design.tilt_angle < self.constraints.min_tilt:
            violations.append(
                f"Tilt {design.tilt_angle:.1f}° < minimum {self.constraints.min_tilt}°"
            )
        if design.tilt_angle > self.constraints.max_tilt:
            violations.append(
                f"Tilt {design.tilt_angle:.1f}° > maximum {self.constraints.max_tilt}°"
            )

        # Land use constraint
        if self.constraints.max_land_use_acres and design.land_use_acres > self.constraints.max_land_use_acres:
            violations.append(
                f"Land use {design.land_use_acres:.1f} acres > maximum {self.constraints.max_land_use_acres}"
            )

        # Capacity constraints
        if self.constraints.min_capacity_mw and design.capacity_mw < self.constraints.min_capacity_mw:
            violations.append(
                f"Capacity {design.capacity_mw:.2f} MW < minimum {self.constraints.min_capacity_mw}"
            )
        if self.constraints.max_capacity_mw and design.capacity_mw > self.constraints.max_capacity_mw:
            violations.append(
                f"Capacity {design.capacity_mw:.2f} MW > maximum {self.constraints.max_capacity_mw}"
            )

        # Shading loss constraint
        if design.shading_loss > self.constraints.max_shading_loss:
            violations.append(
                f"Shading loss {design.shading_loss:.3f} > maximum {self.constraints.max_shading_loss}"
            )

        return len(violations) == 0, violations

    def _evaluate_design(self, params: PVSystemParameters) -> DesignPoint:
        """Evaluate design using provided evaluator or simple model."""
        if self.evaluator is not None:
            return self.evaluator(params)
        else:
            # Simple evaluation model
            return self._simple_evaluation(params)

    def _simple_evaluation(self, params: PVSystemParameters) -> DesignPoint:
        """Simple analytical evaluation."""
        # This is a simplified version - in production, use full simulation
        capacity_mw = params.num_modules * params.module_power / 1e6

        # Energy
        psh = 5.0 - 0.02 * abs(params.latitude)
        shading_loss = max(0, (params.gcr - 0.35) * 0.2)
        annual_energy_kwh = (
            capacity_mw * 1000 * psh * 365
            * params.inverter_efficiency * (1 - shading_loss) * 0.95
        )

        # Economics
        module_capex = params.num_modules * params.module_cost
        inverter_capacity_kw = capacity_mw * 1000 / params.dc_ac_ratio
        inverter_capex = inverter_capacity_kw * params.inverter_cost_per_kw
        module_area_total = params.num_modules * params.module_area
        land_area_acres = (module_area_total / params.gcr) / 4046.86
        land_capex = land_area_acres * params.land_cost_per_acre
        total_capex = module_capex + inverter_capex + land_capex + capacity_mw * 200000

        total_energy = sum(
            annual_energy_kwh * (1 - params.degradation_rate) ** year
            for year in range(params.project_lifetime)
        )

        annual_om = params.om_cost_per_kw_year * capacity_mw * 1000
        om_pv = sum(
            annual_om / (1 + params.discount_rate) ** year
            for year in range(1, params.project_lifetime + 1)
        )

        lcoe = (total_capex + om_pv) / total_energy if total_energy > 0 else 999

        revenue_pv = sum(
            annual_energy_kwh * (1 - params.degradation_rate) ** year * 0.05
            / (1 + params.discount_rate) ** year
            for year in range(1, params.project_lifetime + 1)
        )
        npv = revenue_pv - total_capex - om_pv

        return DesignPoint(
            gcr=params.gcr,
            dc_ac_ratio=params.dc_ac_ratio,
            tilt_angle=params.tilt_angle,
            num_modules=params.num_modules,
            row_spacing=params.row_spacing,
            annual_energy_kwh=annual_energy_kwh,
            capacity_mw=capacity_mw,
            lcoe=lcoe,
            npv=npv,
            land_use_acres=land_area_acres,
            shading_loss=shading_loss,
            bifacial_gain=0.0,
            capex=total_capex,
        )

    @staticmethod
    def _non_dominated_sort(objectives: np.ndarray) -> List[int]:
        """Find non-dominated solutions (Pareto front)."""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                i_dominates_j = all(objectives[i] <= objectives[j]) and any(
                    objectives[i] < objectives[j]
                )
                # Check if j dominates i
                j_dominates_i = all(objectives[j] <= objectives[i]) and any(
                    objectives[j] < objectives[i]
                )

                if i_dominates_j:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dominates_i:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Return indices of non-dominated solutions
        return [i for i in range(n) if domination_count[i] == 0]

    @staticmethod
    def _calculate_crowding_distance(front: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for solutions on Pareto front."""
        n = len(front)
        if n <= 2:
            return np.full(n, float('inf'))

        distances = np.zeros(n)
        n_objectives = front.shape[1]

        for m in range(n_objectives):
            # Sort by objective m
            sorted_indices = np.argsort(front[:, m])

            # Infinite distance for boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Normalize objective range
            obj_range = front[sorted_indices[-1], m] - front[sorted_indices[0], m]
            if obj_range == 0:
                continue

            # Calculate crowding distance
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    front[sorted_indices[i + 1], m] - front[sorted_indices[i - 1], m]
                ) / obj_range

        return distances
