"""
Sensitivity Analysis Engine for PV Financial Models.

This module provides comprehensive sensitivity analysis capabilities including
one-way, two-way, tornado diagrams, and Monte Carlo simulation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from ..models.financial_models import (
    CostStructure,
    RevenueStream,
    CircularityMetrics,
    SensitivityParameter,
)
from .lcoe_calculator import LCOECalculator


class SensitivityMetric(Enum):
    """Financial metrics available for sensitivity analysis."""

    LCOE = "lcoe"
    NPV = "npv"
    IRR = "irr"
    PAYBACK_PERIOD = "payback_period"
    ROI = "roi"


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.

    Attributes:
        parameter_name: Name of the varied parameter
        parameter_values: Array of parameter values tested
        metric_name: Name of the output metric
        metric_values: Array of metric values corresponding to parameter values
        base_parameter_value: Base case parameter value
        base_metric_value: Base case metric value
        elasticity: Sensitivity elasticity (% change in metric / % change in parameter)
    """

    parameter_name: str
    parameter_values: np.ndarray
    metric_name: str
    metric_values: np.ndarray
    base_parameter_value: float
    base_metric_value: float
    elasticity: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame({
            'parameter': self.parameter_name,
            'parameter_value': self.parameter_values,
            'metric': self.metric_name,
            'metric_value': self.metric_values,
        })

    def get_range_impact(self) -> Tuple[float, float]:
        """
        Get the range of metric values across parameter range.

        Returns:
            Tuple of (min_metric, max_metric)
        """
        return (np.min(self.metric_values), np.max(self.metric_values))

    def get_percentage_change(self) -> np.ndarray:
        """Get percentage change in metric from base case."""
        if self.base_metric_value == 0:
            return np.zeros_like(self.metric_values)
        return ((self.metric_values - self.base_metric_value) /
                abs(self.base_metric_value) * 100)


@dataclass
class TornadoData:
    """
    Data for tornado diagram visualization.

    Attributes:
        parameter_names: List of parameter names
        low_values: Metric values at low parameter values
        high_values: Metric values at high parameter values
        base_value: Base case metric value
        sorted_by_impact: Whether data is sorted by impact magnitude
    """

    parameter_names: List[str]
    low_values: np.ndarray
    high_values: np.ndarray
    base_value: float
    sorted_by_impact: bool = True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for plotting."""
        return pd.DataFrame({
            'parameter': self.parameter_names,
            'low': self.low_values,
            'high': self.high_values,
            'impact': np.abs(self.high_values - self.low_values),
        })


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for PV financial models.

    Supports:
    - One-way sensitivity analysis
    - Two-way (2D) sensitivity analysis
    - Tornado diagrams
    - Monte Carlo simulation
    - Elasticity calculations
    """

    def __init__(
        self,
        base_cost_structure: CostStructure,
        base_revenue_stream: RevenueStream,
        base_circularity_metrics: CircularityMetrics,
        lifetime_years: int = 25,
        discount_rate: float = 0.06,
        inflation_rate: float = 0.025,
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            base_cost_structure: Base case cost structure
            base_revenue_stream: Base case revenue stream
            base_circularity_metrics: Base case circularity metrics
            lifetime_years: System lifetime in years
            discount_rate: Discount rate
            inflation_rate: Inflation rate
        """
        self.base_cost_structure = base_cost_structure
        self.base_revenue_stream = base_revenue_stream
        self.base_circularity_metrics = base_circularity_metrics
        self.lifetime_years = lifetime_years
        self.discount_rate = discount_rate
        self.inflation_rate = inflation_rate

        # Create base calculator
        self.base_calculator = LCOECalculator(
            cost_structure=base_cost_structure,
            revenue_stream=base_revenue_stream,
            circularity_metrics=base_circularity_metrics,
            lifetime_years=lifetime_years,
            discount_rate=discount_rate,
            inflation_rate=inflation_rate,
        )

    def one_way_sensitivity(
        self,
        parameter: SensitivityParameter,
        metric: SensitivityMetric = SensitivityMetric.LCOE,
        include_circularity: bool = True,
    ) -> SensitivityResult:
        """
        Perform one-way sensitivity analysis.

        Args:
            parameter: Parameter to vary
            metric: Output metric to analyze
            include_circularity: Include circular economy benefits

        Returns:
            SensitivityResult with analysis results
        """
        parameter_values = parameter.get_range()
        metric_values = []

        for value in parameter_values:
            calc = self._create_modified_calculator(parameter.name, value)
            metric_val = self._calculate_metric(calc, metric, include_circularity)
            metric_values.append(metric_val)

        metric_values = np.array(metric_values)

        # Calculate base case values
        base_metric_value = self._calculate_metric(
            self.base_calculator,
            metric,
            include_circularity
        )

        # Calculate elasticity
        elasticity = self._calculate_elasticity(
            parameter_values,
            metric_values,
            parameter.base_value,
            base_metric_value,
        )

        return SensitivityResult(
            parameter_name=parameter.name,
            parameter_values=parameter_values,
            metric_name=metric.value,
            metric_values=metric_values,
            base_parameter_value=parameter.base_value,
            base_metric_value=base_metric_value,
            elasticity=elasticity,
        )

    def two_way_sensitivity(
        self,
        parameter1: SensitivityParameter,
        parameter2: SensitivityParameter,
        metric: SensitivityMetric = SensitivityMetric.LCOE,
        include_circularity: bool = True,
    ) -> pd.DataFrame:
        """
        Perform two-way (2D) sensitivity analysis.

        Args:
            parameter1: First parameter to vary
            parameter2: Second parameter to vary
            metric: Output metric to analyze
            include_circularity: Include circular economy benefits

        Returns:
            DataFrame with 2D grid of metric values
        """
        param1_values = parameter1.get_range()
        param2_values = parameter2.get_range()

        # Create meshgrid
        results = []

        for p1_val in param1_values:
            for p2_val in param2_values:
                # Create calculator with both parameters modified
                calc = self._create_modified_calculator(parameter1.name, p1_val)
                calc = self._modify_calculator(calc, parameter2.name, p2_val)

                metric_val = self._calculate_metric(calc, metric, include_circularity)

                results.append({
                    parameter1.name: p1_val,
                    parameter2.name: p2_val,
                    metric.value: metric_val,
                })

        return pd.DataFrame(results)

    def tornado_analysis(
        self,
        parameters: List[SensitivityParameter],
        metric: SensitivityMetric = SensitivityMetric.LCOE,
        include_circularity: bool = True,
        variation_percent: float = 20.0,
    ) -> TornadoData:
        """
        Perform tornado diagram analysis.

        Varies each parameter by a fixed percentage and shows the impact
        on the output metric, sorted by impact magnitude.

        Args:
            parameters: List of parameters to analyze
            metric: Output metric to analyze
            include_circularity: Include circular economy benefits
            variation_percent: Percentage to vary each parameter (+/- %)

        Returns:
            TornadoData for visualization
        """
        results = []

        base_metric = self._calculate_metric(
            self.base_calculator,
            metric,
            include_circularity
        )

        for param in parameters:
            # Calculate low value (base - variation%)
            low_value = param.base_value * (1 - variation_percent / 100)
            low_value = max(low_value, param.min_value)

            # Calculate high value (base + variation%)
            high_value = param.base_value * (1 + variation_percent / 100)
            high_value = min(high_value, param.max_value)

            # Calculate metrics at low and high values
            calc_low = self._create_modified_calculator(param.name, low_value)
            metric_low = self._calculate_metric(calc_low, metric, include_circularity)

            calc_high = self._create_modified_calculator(param.name, high_value)
            metric_high = self._calculate_metric(calc_high, metric, include_circularity)

            results.append({
                'parameter': param.name,
                'low': metric_low,
                'high': metric_high,
                'impact': abs(metric_high - metric_low),
            })

        # Sort by impact magnitude
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('impact', ascending=False)

        return TornadoData(
            parameter_names=results_df['parameter'].tolist(),
            low_values=results_df['low'].values,
            high_values=results_df['high'].values,
            base_value=base_metric,
            sorted_by_impact=True,
        )

    def monte_carlo_simulation(
        self,
        parameter_distributions: Dict[str, Tuple[stats.rv_continuous, Dict]],
        metric: SensitivityMetric = SensitivityMetric.LCOE,
        n_simulations: int = 10000,
        include_circularity: bool = True,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform Monte Carlo simulation for risk analysis.

        Args:
            parameter_distributions: Dict mapping parameter names to
                (distribution, params) tuples. Example:
                {'equipment_cost': (stats.norm, {'loc': 100000, 'scale': 10000})}
            metric: Output metric to analyze
            n_simulations: Number of Monte Carlo iterations
            include_circularity: Include circular economy benefits
            random_seed: Random seed for reproducibility

        Returns:
            DataFrame with simulation results and statistics
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        results = []

        for i in range(n_simulations):
            # Sample parameters from distributions
            sampled_params = {}
            for param_name, (distribution, dist_params) in parameter_distributions.items():
                sampled_value = distribution.rvs(**dist_params)
                sampled_params[param_name] = sampled_value

            # Create calculator with sampled parameters
            calc = self.base_calculator
            for param_name, value in sampled_params.items():
                calc = self._create_modified_calculator(param_name, value)

            # Calculate metric
            metric_value = self._calculate_metric(calc, metric, include_circularity)

            result_row = {'simulation': i, metric.value: metric_value}
            result_row.update(sampled_params)
            results.append(result_row)

        df = pd.DataFrame(results)

        # Add statistics
        df['percentile'] = df[metric.value].rank(pct=True) * 100

        return df

    def calculate_correlation_matrix(
        self,
        parameters: List[SensitivityParameter],
        metric: SensitivityMetric = SensitivityMetric.LCOE,
        n_samples: int = 1000,
        include_circularity: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between parameters and metric.

        Uses Latin Hypercube Sampling for efficient parameter space coverage.

        Args:
            parameters: List of parameters to analyze
            metric: Output metric to analyze
            n_samples: Number of samples for correlation analysis
            include_circularity: Include circular economy benefits

        Returns:
            DataFrame with correlation coefficients
        """
        # Latin Hypercube Sampling
        n_params = len(parameters)
        samples = self._latin_hypercube_sample(n_samples, n_params)

        # Scale samples to parameter ranges
        param_samples = {}
        for i, param in enumerate(parameters):
            scaled_samples = (param.min_value +
                            samples[:, i] * (param.max_value - param.min_value))
            param_samples[param.name] = scaled_samples

        # Calculate metrics for each sample
        metric_values = []
        for sample_idx in range(n_samples):
            calc = self.base_calculator
            for param_name, samples_array in param_samples.items():
                value = samples_array[sample_idx]
                calc = self._create_modified_calculator(param_name, value)

            metric_val = self._calculate_metric(calc, metric, include_circularity)
            metric_values.append(metric_val)

        # Create DataFrame and calculate correlations
        data = param_samples.copy()
        data[metric.value] = metric_values
        df = pd.DataFrame(data)

        return df.corr()

    def _calculate_metric(
        self,
        calculator: LCOECalculator,
        metric: SensitivityMetric,
        include_circularity: bool,
    ) -> float:
        """Calculate specified financial metric."""
        if metric == SensitivityMetric.LCOE:
            result = calculator.calculate_lcoe(include_circularity=include_circularity)
            return result.lcoe
        elif metric == SensitivityMetric.NPV:
            from ..models.financial_models import CashFlowModel
            cf_model = CashFlowModel(
                cost_structure=calculator.cost_structure,
                revenue_stream=calculator.revenue_stream,
                circularity_metrics=calculator.circularity_metrics,
                lifetime_years=calculator.lifetime_years,
                discount_rate=calculator.discount_rate,
            )
            return cf_model.calculate_npv()
        elif metric == SensitivityMetric.IRR:
            from ..models.financial_models import CashFlowModel
            cf_model = CashFlowModel(
                cost_structure=calculator.cost_structure,
                revenue_stream=calculator.revenue_stream,
                circularity_metrics=calculator.circularity_metrics,
                lifetime_years=calculator.lifetime_years,
                discount_rate=calculator.discount_rate,
            )
            return cf_model.calculate_irr()
        elif metric == SensitivityMetric.PAYBACK_PERIOD:
            from ..models.financial_models import CashFlowModel
            cf_model = CashFlowModel(
                cost_structure=calculator.cost_structure,
                revenue_stream=calculator.revenue_stream,
                circularity_metrics=calculator.circularity_metrics,
                lifetime_years=calculator.lifetime_years,
                discount_rate=calculator.discount_rate,
            )
            return cf_model.calculate_payback_period()
        elif metric == SensitivityMetric.ROI:
            from ..models.financial_models import CashFlowModel
            cf_model = CashFlowModel(
                cost_structure=calculator.cost_structure,
                revenue_stream=calculator.revenue_stream,
                circularity_metrics=calculator.circularity_metrics,
                lifetime_years=calculator.lifetime_years,
                discount_rate=calculator.discount_rate,
            )
            return cf_model.calculate_roi()

        return 0.0

    def _create_modified_calculator(
        self,
        parameter_name: str,
        value: float,
    ) -> LCOECalculator:
        """Create calculator with modified parameter."""
        import copy

        cost_structure = copy.deepcopy(self.base_cost_structure)
        revenue_stream = copy.deepcopy(self.base_revenue_stream)
        circularity_metrics = copy.deepcopy(self.base_circularity_metrics)

        # Modify parameter
        if hasattr(cost_structure, parameter_name):
            setattr(cost_structure, parameter_name, value)
        elif hasattr(revenue_stream, parameter_name):
            setattr(revenue_stream, parameter_name, value)
        elif hasattr(circularity_metrics, parameter_name):
            setattr(circularity_metrics, parameter_name, value)

        return LCOECalculator(
            cost_structure=cost_structure,
            revenue_stream=revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=self.lifetime_years,
            discount_rate=self.discount_rate,
            inflation_rate=self.inflation_rate,
        )

    def _modify_calculator(
        self,
        calculator: LCOECalculator,
        parameter_name: str,
        value: float,
    ) -> LCOECalculator:
        """Modify an existing calculator with new parameter value."""
        import copy

        cost_structure = copy.deepcopy(calculator.cost_structure)
        revenue_stream = copy.deepcopy(calculator.revenue_stream)
        circularity_metrics = copy.deepcopy(calculator.circularity_metrics)

        if hasattr(cost_structure, parameter_name):
            setattr(cost_structure, parameter_name, value)
        elif hasattr(revenue_stream, parameter_name):
            setattr(revenue_stream, parameter_name, value)
        elif hasattr(circularity_metrics, parameter_name):
            setattr(circularity_metrics, parameter_name, value)

        return LCOECalculator(
            cost_structure=cost_structure,
            revenue_stream=revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=calculator.lifetime_years,
            discount_rate=calculator.discount_rate,
            inflation_rate=calculator.inflation_rate,
        )

    def _calculate_elasticity(
        self,
        param_values: np.ndarray,
        metric_values: np.ndarray,
        base_param: float,
        base_metric: float,
    ) -> float:
        """
        Calculate sensitivity elasticity.

        Elasticity = (% change in metric) / (% change in parameter)
        """
        if base_param == 0 or base_metric == 0:
            return 0.0

        # Find values closest to +/- 1% change in parameter
        target_low = base_param * 0.99
        target_high = base_param * 1.01

        idx_low = np.argmin(np.abs(param_values - target_low))
        idx_high = np.argmin(np.abs(param_values - target_high))

        param_pct_change = ((param_values[idx_high] - param_values[idx_low]) /
                           base_param * 100)
        metric_pct_change = ((metric_values[idx_high] - metric_values[idx_low]) /
                            base_metric * 100)

        if param_pct_change == 0:
            return 0.0

        return metric_pct_change / param_pct_change

    def _latin_hypercube_sample(
        self,
        n_samples: int,
        n_dimensions: int,
    ) -> np.ndarray:
        """
        Generate Latin Hypercube samples.

        Args:
            n_samples: Number of samples
            n_dimensions: Number of dimensions (parameters)

        Returns:
            Array of shape (n_samples, n_dimensions) with values in [0, 1]
        """
        samples = np.zeros((n_samples, n_dimensions))

        for dim in range(n_dimensions):
            # Divide [0, 1] into n_samples intervals
            intervals = np.linspace(0, 1, n_samples + 1)

            # Random sample within each interval
            for i in range(n_samples):
                samples[i, dim] = np.random.uniform(intervals[i], intervals[i + 1])

            # Shuffle to break correlations
            np.random.shuffle(samples[:, dim])

        return samples
