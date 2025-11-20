"""
Levelized Cost of Energy (LCOE) Calculator for PV Systems.

This module provides comprehensive LCOE calculations incorporating system costs,
energy production, degradation, circularity benefits, and various financial parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.financial_models import (
    CostStructure,
    RevenueStream,
    CircularityMetrics,
    CashFlowModel,
)


@dataclass
class LCOEResult:
    """
    Result container for LCOE calculations.

    Attributes:
        lcoe: Levelized Cost of Energy (USD/kWh)
        lcoe_real: Real LCOE accounting for inflation (USD/kWh)
        total_lifetime_cost: Total lifecycle costs (USD)
        total_lifetime_energy: Total energy produced over lifetime (kWh)
        cost_breakdown: Breakdown of costs by category
        with_circularity: LCOE including circular economy benefits
        without_circularity: LCOE without circular economy benefits
        circularity_benefit: Reduction in LCOE from circularity (USD/kWh)
    """

    lcoe: float
    lcoe_real: float
    total_lifetime_cost: float
    total_lifetime_energy: float
    cost_breakdown: Dict[str, float]
    with_circularity: float
    without_circularity: float
    circularity_benefit: float

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            'lcoe': self.lcoe,
            'lcoe_real': self.lcoe_real,
            'total_lifetime_cost': self.total_lifetime_cost,
            'total_lifetime_energy': self.total_lifetime_energy,
            'cost_breakdown': self.cost_breakdown,
            'with_circularity': self.with_circularity,
            'without_circularity': self.without_circularity,
            'circularity_benefit': self.circularity_benefit,
            'circularity_benefit_percent': (
                (self.circularity_benefit / self.without_circularity * 100)
                if self.without_circularity > 0 else 0.0
            ),
        }

    def get_summary_text(self) -> str:
        """Generate human-readable summary of LCOE results."""
        return f"""
LCOE Analysis Summary
{'=' * 50}
Levelized Cost of Energy (Nominal): ${self.lcoe:.4f}/kWh
Levelized Cost of Energy (Real): ${self.lcoe_real:.4f}/kWh

Total Lifetime Cost: ${self.total_lifetime_cost:,.2f}
Total Lifetime Energy: {self.total_lifetime_energy:,.0f} kWh

Circularity Impact:
  LCOE with Circularity: ${self.with_circularity:.4f}/kWh
  LCOE without Circularity: ${self.without_circularity:.4f}/kWh
  Circularity Benefit: ${self.circularity_benefit:.4f}/kWh ({self.circularity_benefit / self.without_circularity * 100:.2f}%)

Cost Breakdown:
  {chr(10).join(f'  {k}: ${v:,.2f}' for k, v in self.cost_breakdown.items())}
"""


class LCOECalculator:
    """
    Comprehensive LCOE calculator for PV systems.

    The LCOE represents the per-unit cost of energy production over the
    system's lifetime, accounting for all costs and energy production.

    Formula:
        LCOE = Sum(Ct / (1+r)^t) / Sum(Et / (1+r)^t)

    Where:
        Ct = Total costs in year t
        Et = Energy produced in year t
        r = Discount rate
        t = Year
    """

    def __init__(
        self,
        cost_structure: CostStructure,
        revenue_stream: RevenueStream,
        circularity_metrics: Optional[CircularityMetrics] = None,
        lifetime_years: int = 25,
        discount_rate: float = 0.06,
        inflation_rate: float = 0.025,
    ):
        """
        Initialize LCOE calculator.

        Args:
            cost_structure: System cost structure
            revenue_stream: Revenue and energy production parameters
            circularity_metrics: Circular economy metrics (optional)
            lifetime_years: System operational lifetime in years
            discount_rate: Discount rate for present value calculations
            inflation_rate: Annual inflation rate
        """
        self.cost_structure = cost_structure
        self.revenue_stream = revenue_stream
        self.circularity_metrics = circularity_metrics or CircularityMetrics()
        self.lifetime_years = lifetime_years
        self.discount_rate = discount_rate
        self.inflation_rate = inflation_rate

    def calculate_lcoe(
        self,
        include_circularity: bool = True,
        use_real_discount_rate: bool = False,
    ) -> LCOEResult:
        """
        Calculate Levelized Cost of Energy.

        Args:
            include_circularity: Whether to include circular economy benefits
            use_real_discount_rate: Use inflation-adjusted discount rate

        Returns:
            LCOEResult with comprehensive LCOE analysis
        """
        # Calculate with and without circularity for comparison
        lcoe_with = self._calculate_lcoe_value(
            include_circularity=True,
            use_real_discount_rate=use_real_discount_rate
        )

        lcoe_without = self._calculate_lcoe_value(
            include_circularity=False,
            use_real_discount_rate=use_real_discount_rate
        )

        # Choose primary LCOE based on parameter
        primary_lcoe = lcoe_with if include_circularity else lcoe_without

        # Calculate real LCOE (inflation-adjusted)
        lcoe_real = self._calculate_lcoe_value(
            include_circularity=include_circularity,
            use_real_discount_rate=True
        )

        # Get cost breakdown and totals
        breakdown = self._get_cost_breakdown(include_circularity)
        total_cost = sum(breakdown.values())
        total_energy = self._calculate_total_lifetime_energy()

        return LCOEResult(
            lcoe=primary_lcoe,
            lcoe_real=lcoe_real,
            total_lifetime_cost=total_cost,
            total_lifetime_energy=total_energy,
            cost_breakdown=breakdown,
            with_circularity=lcoe_with,
            without_circularity=lcoe_without,
            circularity_benefit=lcoe_without - lcoe_with,
        )

    def _calculate_lcoe_value(
        self,
        include_circularity: bool,
        use_real_discount_rate: bool,
    ) -> float:
        """
        Calculate single LCOE value.

        Args:
            include_circularity: Include circular economy benefits
            use_real_discount_rate: Use inflation-adjusted discount rate

        Returns:
            LCOE in USD/kWh
        """
        # Determine discount rate
        if use_real_discount_rate:
            # Fisher equation: (1 + nominal) = (1 + real) * (1 + inflation)
            discount_rate = ((1 + self.discount_rate) /
                           (1 + self.inflation_rate)) - 1
        else:
            discount_rate = self.discount_rate

        # Calculate present value of costs
        pv_costs = self._calculate_pv_costs(discount_rate, include_circularity)

        # Calculate present value of energy production
        pv_energy = self._calculate_pv_energy(discount_rate)

        # LCOE = PV(Costs) / PV(Energy)
        return pv_costs / pv_energy if pv_energy > 0 else float('inf')

    def _calculate_pv_costs(
        self,
        discount_rate: float,
        include_circularity: bool,
    ) -> float:
        """
        Calculate present value of all lifetime costs.

        Args:
            discount_rate: Discount rate to use
            include_circularity: Include end-of-life recovery value

        Returns:
            Present value of costs in USD
        """
        pv_total = 0.0

        # Year 0: Initial CAPEX
        pv_total += self.cost_structure.get_total_capex()

        # Operational years: OPEX
        annual_opex = self.cost_structure.get_total_annual_opex()
        for year in range(1, self.lifetime_years + 1):
            # Apply inflation to OPEX
            inflated_opex = annual_opex * (1 + self.inflation_rate) ** year

            # Discount to present value
            discount_factor = 1 / (1 + discount_rate) ** year
            pv_total += inflated_opex * discount_factor

            # Add replacement costs if scheduled
            if year in self.cost_structure.replacement_costs:
                replacement_cost = self.cost_structure.replacement_costs[year]
                pv_total += replacement_cost * discount_factor

        # Final year: Decommissioning and disposal (or recovery)
        final_year = self.lifetime_years
        discount_factor = 1 / (1 + discount_rate) ** final_year

        if include_circularity:
            # Subtract recovery value (negative cost)
            eol_recovery = self.circularity_metrics.get_eol_recovery_value(
                self.cost_structure.equipment_cost
            )
            pv_total -= eol_recovery * discount_factor
        else:
            # Add disposal costs
            pv_total += self.cost_structure.disposal_cost * discount_factor

        pv_total += self.cost_structure.decommissioning_cost * discount_factor

        return pv_total

    def _calculate_pv_energy(self, discount_rate: float) -> float:
        """
        Calculate present value of lifetime energy production.

        Args:
            discount_rate: Discount rate to use

        Returns:
            Present value of energy production in kWh
        """
        pv_energy = 0.0

        for year in range(self.lifetime_years):
            # Apply degradation
            annual_production = (
                self.revenue_stream.annual_energy_production *
                (1 - self.revenue_stream.degradation_rate) ** year
            )

            # Discount to present value
            discount_factor = 1 / (1 + discount_rate) ** year
            pv_energy += annual_production * discount_factor

        return pv_energy

    def _calculate_total_lifetime_energy(self) -> float:
        """Calculate total energy production over lifetime (not discounted)."""
        total = 0.0
        for year in range(self.lifetime_years):
            annual_production = (
                self.revenue_stream.annual_energy_production *
                (1 - self.revenue_stream.degradation_rate) ** year
            )
            total += annual_production
        return total

    def _get_cost_breakdown(self, include_circularity: bool) -> Dict[str, float]:
        """
        Get detailed breakdown of lifetime costs.

        Args:
            include_circularity: Include circular economy benefits

        Returns:
            Dictionary of cost categories and their present values
        """
        breakdown = {}

        # CAPEX components
        breakdown['Equipment'] = self.cost_structure.equipment_cost
        breakdown['Installation'] = self.cost_structure.installation_cost
        breakdown['Soft Costs'] = self.cost_structure.soft_costs

        # OPEX components (present value)
        opex_items = {
            'Maintenance': self.cost_structure.maintenance_cost,
            'Insurance': self.cost_structure.insurance_cost,
            'Land Lease': self.cost_structure.land_lease_cost,
        }

        for name, annual_cost in opex_items.items():
            pv_cost = sum(
                annual_cost * (1 + self.inflation_rate) ** year /
                (1 + self.discount_rate) ** year
                for year in range(1, self.lifetime_years + 1)
            )
            breakdown[name] = pv_cost

        # Replacement costs
        if self.cost_structure.replacement_costs:
            total_replacements = sum(
                cost / (1 + self.discount_rate) ** year
                for year, cost in self.cost_structure.replacement_costs.items()
            )
            breakdown['Replacements'] = total_replacements

        # End-of-life costs/benefits
        discount_factor = 1 / (1 + self.discount_rate) ** self.lifetime_years

        if include_circularity:
            eol_recovery = self.circularity_metrics.get_eol_recovery_value(
                self.cost_structure.equipment_cost
            )
            breakdown['EOL Recovery'] = -eol_recovery * discount_factor
        else:
            breakdown['Disposal'] = (
                self.cost_structure.disposal_cost * discount_factor
            )

        breakdown['Decommissioning'] = (
            self.cost_structure.decommissioning_cost * discount_factor
        )

        return breakdown

    def calculate_lcoe_by_year(
        self,
        include_circularity: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate cumulative LCOE for each year of operation.

        Useful for understanding how LCOE evolves over the project lifetime.

        Args:
            include_circularity: Include circular economy benefits

        Returns:
            DataFrame with yearly LCOE calculations
        """
        data = []

        for year_count in range(1, self.lifetime_years + 1):
            # Create temporary calculator with reduced lifetime
            temp_calc = LCOECalculator(
                cost_structure=self.cost_structure,
                revenue_stream=self.revenue_stream,
                circularity_metrics=self.circularity_metrics,
                lifetime_years=year_count,
                discount_rate=self.discount_rate,
                inflation_rate=self.inflation_rate,
            )

            lcoe = temp_calc._calculate_lcoe_value(
                include_circularity=include_circularity,
                use_real_discount_rate=False,
            )

            total_energy = temp_calc._calculate_total_lifetime_energy()

            data.append({
                'year': year_count,
                'lcoe': lcoe,
                'cumulative_energy_kwh': total_energy,
            })

        return pd.DataFrame(data)

    def compare_scenarios(
        self,
        scenarios: Dict[str, 'LCOECalculator'],
    ) -> pd.DataFrame:
        """
        Compare LCOE across multiple scenarios.

        Args:
            scenarios: Dictionary mapping scenario names to LCOECalculator instances

        Returns:
            DataFrame comparing key metrics across scenarios
        """
        results = []

        for scenario_name, calculator in scenarios.items():
            result = calculator.calculate_lcoe()
            results.append({
                'Scenario': scenario_name,
                'LCOE ($/kWh)': result.lcoe,
                'Real LCOE ($/kWh)': result.lcoe_real,
                'Total Cost ($)': result.total_lifetime_cost,
                'Total Energy (kWh)': result.total_lifetime_energy,
                'Circularity Benefit ($/kWh)': result.circularity_benefit,
            })

        return pd.DataFrame(results)

    def sensitivity_analysis(
        self,
        parameter_name: str,
        parameter_range: np.ndarray,
        include_circularity: bool = True,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on LCOE for a given parameter.

        Args:
            parameter_name: Name of parameter to vary
            parameter_range: Array of parameter values to test
            include_circularity: Include circular economy benefits

        Returns:
            DataFrame with parameter values and corresponding LCOE
        """
        results = []

        for value in parameter_range:
            # Create modified calculator
            calc = self._create_modified_calculator(parameter_name, value)

            # Calculate LCOE
            lcoe = calc._calculate_lcoe_value(
                include_circularity=include_circularity,
                use_real_discount_rate=False,
            )

            results.append({
                'parameter': parameter_name,
                'value': value,
                'lcoe': lcoe,
            })

        return pd.DataFrame(results)

    def _create_modified_calculator(
        self,
        parameter_name: str,
        value: float,
    ) -> 'LCOECalculator':
        """
        Create a new calculator with a modified parameter value.

        Args:
            parameter_name: Parameter to modify
            value: New parameter value

        Returns:
            New LCOECalculator instance
        """
        # Deep copy current objects
        import copy
        cost_structure = copy.deepcopy(self.cost_structure)
        revenue_stream = copy.deepcopy(self.revenue_stream)
        circularity_metrics = copy.deepcopy(self.circularity_metrics)

        # Modify the appropriate parameter
        if hasattr(cost_structure, parameter_name):
            setattr(cost_structure, parameter_name, value)
        elif hasattr(revenue_stream, parameter_name):
            setattr(revenue_stream, parameter_name, value)
        elif hasattr(circularity_metrics, parameter_name):
            setattr(circularity_metrics, parameter_name, value)
        elif parameter_name == 'discount_rate':
            return LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=self.lifetime_years,
                discount_rate=value,
                inflation_rate=self.inflation_rate,
            )
        elif parameter_name == 'lifetime_years':
            return LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=int(value),
                discount_rate=self.discount_rate,
                inflation_rate=self.inflation_rate,
            )

        return LCOECalculator(
            cost_structure=cost_structure,
            revenue_stream=revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=self.lifetime_years,
            discount_rate=self.discount_rate,
            inflation_rate=self.inflation_rate,
        )
