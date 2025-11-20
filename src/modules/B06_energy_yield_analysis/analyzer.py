"""Energy Yield Analysis Engine.

This module implements comprehensive energy yield analysis for PV systems including
performance analysis, loss breakdown, financial metrics, and probabilistic analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ...models.eya_models import (
    ProjectInfo,
    SystemConfiguration,
    EnergyOutput,
    LossBreakdown,
    FinancialMetrics,
    SensitivityAnalysis,
    ProbabilisticAnalysis,
    PerformanceMetrics,
)


class EnergyYieldAnalyzer:
    """Energy Yield Analysis engine for PV systems.

    This class provides comprehensive energy yield analysis capabilities including:
    - Detailed loss analysis
    - Financial metrics calculation (LCOE, NPV, IRR)
    - Sensitivity analysis
    - Probabilistic analysis (P50, P90, P99)

    Attributes:
        project_info: Project information
        system_config: System configuration
        loss_breakdown: System loss breakdown
    """

    def __init__(
        self,
        project_info: ProjectInfo,
        system_config: SystemConfiguration,
        loss_breakdown: Optional[LossBreakdown] = None,
    ):
        """Initialize the energy yield analyzer.

        Args:
            project_info: Project information and metadata
            system_config: System configuration parameters
            loss_breakdown: Custom loss breakdown (uses defaults if None)
        """
        self.project_info = project_info
        self.system_config = system_config
        self.loss_breakdown = loss_breakdown or LossBreakdown()
        self.loss_breakdown.total_loss = self.loss_breakdown.calculate_total_loss()

    def analyze_losses(self) -> Dict[str, float]:
        """Analyze system losses in detail.

        Returns:
            Dictionary with detailed loss analysis
        """
        losses = self.loss_breakdown.model_dump()

        # Calculate loss categories
        optical_losses = (
            losses["soiling_loss"] + losses["shading_loss"] + losses["snow_loss"]
        )

        electrical_losses = (
            losses["mismatch_loss"]
            + losses["wiring_loss"]
            + losses["connection_loss"]
            + losses["inverter_loss"]
            + losses["transformer_loss"]
        )

        degradation_losses = (
            losses["lid_loss"] + losses["nameplate_loss"] + losses["age_loss"]
        )

        environmental_losses = losses["temperature_loss"]

        system_losses = losses["availability_loss"]

        return {
            **losses,
            "optical_losses": optical_losses,
            "electrical_losses": electrical_losses,
            "degradation_losses": degradation_losses,
            "environmental_losses": environmental_losses,
            "system_losses": system_losses,
        }

    def calculate_financial_metrics(
        self, annual_energy: float, financial_params: FinancialMetrics
    ) -> FinancialMetrics:
        """Calculate comprehensive financial metrics.

        Args:
            annual_energy: Annual energy production in kWh/year
            financial_params: Financial parameters (CAPEX, OPEX, etc.)

        Returns:
            Financial metrics with calculated LCOE, NPV, IRR, and payback
        """
        lifetime = self.project_info.project_lifetime
        degradation = financial_params.degradation_rate
        discount_rate = financial_params.discount_rate
        energy_price = financial_params.energy_price

        # Calculate annual energy production with degradation
        annual_energies = [
            annual_energy * (1 - degradation) ** year for year in range(lifetime)
        ]

        # Calculate annual revenues
        annual_revenues = [energy * energy_price for energy in annual_energies]

        # Calculate annual costs (OPEX)
        annual_costs = [financial_params.opex_annual] * lifetime

        # Calculate annual cash flows
        cash_flows = [revenue - cost for revenue, cost in zip(annual_revenues, annual_costs)]

        # Initial investment (negative)
        cash_flows.insert(0, -financial_params.capex)

        # Calculate NPV
        npv = sum(
            cf / (1 + discount_rate) ** year for year, cf in enumerate(cash_flows)
        )

        # Calculate IRR using numpy
        try:
            irr = np.irr(cash_flows)
        except:
            irr = None

        # Calculate LCOE
        total_energy = sum(
            energy / (1 + discount_rate) ** (year + 1)
            for year, energy in enumerate(annual_energies)
        )
        total_costs = financial_params.capex + sum(
            financial_params.opex_annual / (1 + discount_rate) ** (year + 1)
            for year in range(lifetime)
        )
        lcoe = total_costs / total_energy if total_energy > 0 else 0

        # Calculate simple payback period
        cumulative_cash_flow = 0
        payback_period = None
        for year, cf in enumerate(cash_flows[1:], start=1):  # Skip initial investment
            cumulative_cash_flow += cf
            if cumulative_cash_flow >= financial_params.capex:
                payback_period = year
                break

        # Update financial metrics
        financial_params.lcoe = lcoe
        financial_params.npv = npv
        financial_params.irr = irr if irr is not None else 0.0
        financial_params.payback_period = payback_period

        return financial_params

    def perform_sensitivity_analysis(
        self,
        base_annual_energy: float,
        parameter_name: str,
        base_value: float,
        variation_pct: float = 20,
        num_points: int = 11,
    ) -> SensitivityAnalysis:
        """Perform sensitivity analysis on a parameter.

        Args:
            base_annual_energy: Base case annual energy in kWh/year
            parameter_name: Name of parameter to vary
            base_value: Base value of parameter
            variation_pct: Percentage variation range (default: ±20%)
            num_points: Number of points to calculate (default: 11)

        Returns:
            Sensitivity analysis results
        """
        # Calculate variation range
        variation = base_value * variation_pct / 100
        min_value = base_value - variation
        max_value = base_value + variation

        # Generate parameter values
        param_values = np.linspace(min_value, max_value, num_points)

        # Calculate energy output for each parameter value
        results = {}
        for param_value in param_values:
            # Simplified linear sensitivity
            change_factor = param_value / base_value
            energy = base_annual_energy * change_factor
            results[float(param_value)] = energy

        return SensitivityAnalysis(
            parameter_name=parameter_name,
            base_value=base_value,
            variation_range=(min_value, max_value),
            results=results,
        )

    def calculate_probabilistic_yield(
        self,
        base_annual_energy: float,
        uncertainty_pct: float = 10.0,
        num_simulations: int = 10000,
    ) -> ProbabilisticAnalysis:
        """Calculate probabilistic energy yield (P50, P90, P99).

        Args:
            base_annual_energy: Base case annual energy in kWh/year
            uncertainty_pct: Total uncertainty in % (default: 10%)
            num_simulations: Number of Monte Carlo simulations (default: 10000)

        Returns:
            Probabilistic analysis results
        """
        # Standard deviation from uncertainty
        std_dev = base_annual_energy * uncertainty_pct / 100

        # Generate random samples (assuming normal distribution)
        samples = np.random.normal(base_annual_energy, std_dev, num_simulations)

        # Calculate exceedance probabilities
        p99 = np.percentile(samples, 1)  # 99% probability of exceeding
        p90 = np.percentile(samples, 10)  # 90% probability of exceeding
        p75 = np.percentile(samples, 25)  # 75% probability of exceeding
        p50 = np.percentile(samples, 50)  # 50% probability (median)

        # Calculate statistics
        mean = np.mean(samples)
        std_dev_actual = np.std(samples)

        # Confidence intervals
        confidence_intervals = {
            "90%": (np.percentile(samples, 5), np.percentile(samples, 95)),
            "95%": (np.percentile(samples, 2.5), np.percentile(samples, 97.5)),
            "99%": (np.percentile(samples, 0.5), np.percentile(samples, 99.5)),
        }

        return ProbabilisticAnalysis(
            p99=p99,
            p90=p90,
            p75=p75,
            p50=p50,
            mean=mean,
            std_dev=std_dev_actual,
            confidence_intervals=confidence_intervals,
        )

    def calculate_monthly_performance(
        self, energy_outputs: List[EnergyOutput]
    ) -> pd.DataFrame:
        """Calculate monthly performance metrics.

        Args:
            energy_outputs: List of energy outputs

        Returns:
            DataFrame with monthly performance metrics
        """
        # Convert to DataFrame
        df = pd.DataFrame([output.model_dump() for output in energy_outputs])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by month
        monthly = df.groupby(df["timestamp"].dt.to_period("M")).agg(
            {
                "dc_energy": "sum",
                "ac_energy": "sum",
                "exported_energy": "sum",
                "specific_yield": "sum",
                "capacity_factor": "mean",
            }
        )

        monthly["month"] = monthly.index.astype(str)
        monthly.reset_index(drop=True, inplace=True)

        return monthly

    def calculate_degradation_impact(
        self, annual_energy: float, years: int = 25
    ) -> pd.DataFrame:
        """Calculate degradation impact over project lifetime.

        Args:
            annual_energy: First year annual energy in kWh
            years: Number of years to project (default: 25)

        Returns:
            DataFrame with annual energy projections
        """
        degradation_rate = 0.005  # Default 0.5%/year

        data = []
        for year in range(1, years + 1):
            degraded_energy = annual_energy * (1 - degradation_rate) ** (year - 1)
            cumulative_energy = sum(
                annual_energy * (1 - degradation_rate) ** y for y in range(year)
            )

            data.append(
                {
                    "year": year,
                    "annual_energy_kwh": degraded_energy,
                    "cumulative_energy_kwh": cumulative_energy,
                    "degradation_factor": (1 - degradation_rate) ** (year - 1),
                }
            )

        return pd.DataFrame(data)

    def generate_performance_summary(
        self,
        annual_energy: float,
        performance_metrics: PerformanceMetrics,
        financial_metrics: FinancialMetrics,
    ) -> Dict:
        """Generate comprehensive performance summary.

        Args:
            annual_energy: Annual energy production in kWh/year
            performance_metrics: Performance metrics
            financial_metrics: Financial metrics

        Returns:
            Dictionary with comprehensive performance summary
        """
        summary = {
            "project": {
                "name": self.project_info.project_name,
                "location": self.project_info.location,
                "capacity_dc_kwp": self.system_config.capacity_dc,
                "capacity_ac_kwac": self.system_config.capacity_ac,
                "commissioning_date": self.project_info.commissioning_date.isoformat(),
            },
            "energy_production": {
                "annual_energy_kwh": annual_energy,
                "specific_yield_kwh_kwp": annual_energy / self.system_config.capacity_dc,
                "capacity_factor": (annual_energy / (self.system_config.capacity_ac * 8760)),
            },
            "performance": {
                "performance_ratio": performance_metrics.performance_ratio,
                "reference_yield": performance_metrics.reference_yield,
                "final_yield": performance_metrics.final_yield,
            },
            "losses": self.analyze_losses(),
            "financial": {
                "lcoe_usd_kwh": financial_metrics.lcoe,
                "npv_usd": financial_metrics.npv,
                "irr": financial_metrics.irr,
                "payback_period_years": financial_metrics.payback_period,
            },
        }

        return summary
