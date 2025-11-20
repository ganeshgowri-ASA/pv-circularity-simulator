"""
B13-S01: LCOE Calculations
Production-ready Levelized Cost of Energy calculator with sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.optimize import fsolve

from ..core.data_models import ProjectFinancials, LCOEResult
from ..core.utilities import present_value, real_to_nominal


class LCOECalculator:
    """
    Comprehensive LCOE calculator with sensitivity and scenario analysis.
    """

    def __init__(self, project: ProjectFinancials):
        """
        Initialize LCOE calculator.

        Args:
            project: Project financial parameters
        """
        self.project = project

    def levelized_costs(self,
                       annual_generation_kwh: float,
                       include_inflation: bool = True) -> LCOEResult:
        """
        Calculate levelized cost of energy.

        Args:
            annual_generation_kwh: First year energy generation
            include_inflation: Whether to include inflation

        Returns:
            LCOE calculation results
        """
        # Calculate lifetime costs and generation
        total_cost_pv = self.project.capex_usd
        total_generation_pv = 0

        for year in range(1, self.project.project_lifetime_years + 1):
            # Generation with degradation
            generation = annual_generation_kwh * (1 - self.project.degradation_rate) ** (year - 1)

            # O&M costs with inflation
            if include_inflation:
                opex = self.project.opex_annual_usd * (1 + self.project.inflation_rate) ** (year - 1)
            else:
                opex = self.project.opex_annual_usd

            # Present value
            opex_pv = present_value(opex, self.project.discount_rate, year)
            generation_pv = present_value(generation, self.project.discount_rate, year)

            total_cost_pv += opex_pv
            total_generation_pv += generation_pv

        # LCOE
        lcoe = total_cost_pv / total_generation_pv if total_generation_pv > 0 else 0

        # Real vs Nominal
        if include_inflation:
            nominal_lcoe = lcoe
            real_discount_rate = (self.project.discount_rate - self.project.inflation_rate) / \
                                (1 + self.project.inflation_rate)

            # Recalculate with real rate
            total_cost_pv_real = self.project.capex_usd
            total_generation_pv_real = 0

            for year in range(1, self.project.project_lifetime_years + 1):
                generation = annual_generation_kwh * (1 - self.project.degradation_rate) ** (year - 1)
                opex = self.project.opex_annual_usd

                opex_pv = present_value(opex, real_discount_rate, year)
                generation_pv = present_value(generation, real_discount_rate, year)

                total_cost_pv_real += opex_pv
                total_generation_pv_real += generation_pv

            real_lcoe = total_cost_pv_real / total_generation_pv_real if total_generation_pv_real > 0 else 0
        else:
            nominal_lcoe = lcoe
            real_lcoe = lcoe

        # Total lifetime values
        total_lifetime_energy = sum(
            annual_generation_kwh * (1 - self.project.degradation_rate) ** (year - 1)
            for year in range(1, self.project.project_lifetime_years + 1)
        )

        total_lifetime_cost = self.project.capex_usd + \
                             sum(self.project.opex_annual_usd *
                                 (1 + self.project.inflation_rate) ** (year - 1)
                                 for year in range(1, self.project.project_lifetime_years + 1))

        return LCOEResult(
            lcoe_usd_per_kwh=lcoe,
            total_lifetime_cost=total_lifetime_cost,
            total_lifetime_energy_kwh=total_lifetime_energy,
            real_lcoe=real_lcoe,
            nominal_lcoe=nominal_lcoe,
            sensitivity_range=None
        )

    def sensitivity_analysis(self,
                            annual_generation_kwh: float,
                            parameters: Optional[List[str]] = None,
                            variation_pct: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on LCOE.

        Args:
            annual_generation_kwh: Annual generation
            parameters: Parameters to vary (if None, use defaults)
            variation_pct: Variation percentage (e.g., 0.2 for ±20%)

        Returns:
            Sensitivity results for each parameter
        """
        if parameters is None:
            parameters = ['capex_usd', 'opex_annual_usd', 'discount_rate',
                         'degradation_rate', 'project_lifetime_years']

        base_lcoe = self.levelized_costs(annual_generation_kwh).lcoe_usd_per_kwh

        results = {}

        for param in parameters:
            base_value = getattr(self.project, param)

            # Test lower bound
            setattr(self.project, param, base_value * (1 - variation_pct))
            lcoe_low = self.levelized_costs(annual_generation_kwh).lcoe_usd_per_kwh

            # Test upper bound
            setattr(self.project, param, base_value * (1 + variation_pct))
            lcoe_high = self.levelized_costs(annual_generation_kwh).lcoe_usd_per_kwh

            # Restore base value
            setattr(self.project, param, base_value)

            # Calculate sensitivity
            sensitivity = (lcoe_high - lcoe_low) / (2 * variation_pct * base_lcoe) if base_lcoe > 0 else 0

            results[param] = {
                'base_lcoe': base_lcoe,
                'lcoe_low': lcoe_low,
                'lcoe_high': lcoe_high,
                'sensitivity_coefficient': sensitivity,
                'variation_pct': variation_pct
            }

        return results

    def scenario_comparison(self,
                           scenarios: Dict[str, Dict[str, float]],
                           annual_generation_kwh: float) -> pd.DataFrame:
        """
        Compare LCOE across multiple scenarios.

        Args:
            scenarios: Dictionary of scenario parameters
            annual_generation_kwh: Annual generation

        Returns:
            DataFrame comparing scenarios
        """
        results = []

        for scenario_name, params in scenarios.items():
            # Create modified project
            modified_project = ProjectFinancials(
                project_name=f"{self.project.project_name}_{scenario_name}",
                capacity_kw=self.project.capacity_kw,
                capex_usd=params.get('capex_usd', self.project.capex_usd),
                opex_annual_usd=params.get('opex_annual_usd', self.project.opex_annual_usd),
                project_lifetime_years=params.get('project_lifetime_years',
                                                  self.project.project_lifetime_years),
                discount_rate=params.get('discount_rate', self.project.discount_rate),
                inflation_rate=params.get('inflation_rate', self.project.inflation_rate),
                tax_rate=params.get('tax_rate', self.project.tax_rate),
                degradation_rate=params.get('degradation_rate', self.project.degradation_rate)
            )

            calc = LCOECalculator(modified_project)
            lcoe_result = calc.levelized_costs(annual_generation_kwh)

            results.append({
                'scenario': scenario_name,
                'lcoe_usd_per_kwh': lcoe_result.lcoe_usd_per_kwh,
                'real_lcoe': lcoe_result.real_lcoe,
                'total_lifetime_cost': lcoe_result.total_lifetime_cost,
                'capex': modified_project.capex_usd,
                'annual_opex': modified_project.opex_annual_usd,
                'discount_rate': modified_project.discount_rate
            })

        return pd.DataFrame(results)


__all__ = ["LCOECalculator"]
