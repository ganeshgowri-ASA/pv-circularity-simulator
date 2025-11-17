"""
Financial Data Models for PV Circularity Simulator.

This module defines the core financial data structures including cost models,
revenue streams, cash flow projections, and circular economy metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd


class CostCategory(Enum):
    """Categories of costs in PV system lifecycle."""

    CAPEX_EQUIPMENT = "capex_equipment"
    CAPEX_INSTALLATION = "capex_installation"
    CAPEX_SOFT_COSTS = "capex_soft_costs"
    OPEX_MAINTENANCE = "opex_maintenance"
    OPEX_OPERATIONS = "opex_operations"
    OPEX_INSURANCE = "opex_insurance"
    OPEX_LAND_LEASE = "opex_land_lease"
    EOL_DECOMMISSIONING = "eol_decommissioning"
    EOL_DISPOSAL = "eol_disposal"


class RevenueCategory(Enum):
    """Categories of revenue streams."""

    ENERGY_SALES = "energy_sales"
    FEED_IN_TARIFF = "feed_in_tariff"
    TAX_CREDITS = "tax_credits"
    SUBSIDIES = "subsidies"
    REC_SALES = "rec_sales"  # Renewable Energy Certificates
    CIRCULARITY_RECOVERY = "circularity_recovery"


@dataclass
class CostStructure:
    """
    Comprehensive cost structure for PV system lifecycle.

    Attributes:
        initial_capex: Initial capital expenditure (USD)
        equipment_cost: Cost of PV modules, inverters, mounting (USD)
        installation_cost: Labor and installation costs (USD)
        soft_costs: Permits, design, project management (USD)
        annual_opex: Annual operating expenditure (USD/year)
        maintenance_cost: Annual maintenance costs (USD/year)
        insurance_cost: Annual insurance costs (USD/year)
        land_lease_cost: Annual land/roof lease (USD/year)
        decommissioning_cost: End-of-life decommissioning (USD)
        disposal_cost: Waste disposal costs (USD)
        replacement_costs: Schedule of component replacements {year: cost}
    """

    initial_capex: float
    equipment_cost: float
    installation_cost: float
    soft_costs: float
    annual_opex: float
    maintenance_cost: float
    insurance_cost: float = 0.0
    land_lease_cost: float = 0.0
    decommissioning_cost: float = 0.0
    disposal_cost: float = 0.0
    replacement_costs: Dict[int, float] = field(default_factory=dict)

    def get_total_capex(self) -> float:
        """Calculate total initial capital expenditure."""
        return self.equipment_cost + self.installation_cost + self.soft_costs

    def get_total_annual_opex(self) -> float:
        """Calculate total annual operating expenditure."""
        return (self.maintenance_cost + self.insurance_cost +
                self.land_lease_cost)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown by category."""
        return {
            "Equipment": self.equipment_cost,
            "Installation": self.installation_cost,
            "Soft Costs": self.soft_costs,
            "Annual O&M": self.maintenance_cost,
            "Insurance": self.insurance_cost,
            "Land Lease": self.land_lease_cost,
            "Decommissioning": self.decommissioning_cost,
            "Disposal": self.disposal_cost,
        }


@dataclass
class RevenueStream:
    """
    Revenue streams from PV system operation and circularity.

    Attributes:
        annual_energy_production: Expected annual energy (kWh/year)
        energy_price: Price per kWh (USD/kWh)
        feed_in_tariff: Feed-in tariff rate (USD/kWh)
        tariff_duration: Duration of feed-in tariff (years)
        tax_credits: Total tax credits available (USD)
        subsidies: Total subsidies received (USD)
        rec_value: Annual REC sales value (USD/year)
        degradation_rate: Annual performance degradation (% per year)
        escalation_rate: Annual energy price escalation (% per year)
    """

    annual_energy_production: float
    energy_price: float
    feed_in_tariff: float = 0.0
    tariff_duration: int = 0
    tax_credits: float = 0.0
    subsidies: float = 0.0
    rec_value: float = 0.0
    degradation_rate: float = 0.005  # 0.5% per year default
    escalation_rate: float = 0.02  # 2% per year default

    def get_annual_revenue(self, year: int) -> float:
        """
        Calculate annual revenue for a specific year.

        Args:
            year: Year number (0-indexed from start)

        Returns:
            Annual revenue in USD
        """
        # Apply degradation to energy production
        degraded_production = (self.annual_energy_production *
                              (1 - self.degradation_rate) ** year)

        # Apply escalation to energy price
        escalated_price = self.energy_price * (1 + self.escalation_rate) ** year

        # Calculate energy revenue
        energy_revenue = degraded_production * escalated_price

        # Add feed-in tariff if applicable
        if year < self.tariff_duration:
            energy_revenue += degraded_production * self.feed_in_tariff

        # Add REC value
        rec_revenue = self.rec_value * (1 - self.degradation_rate) ** year

        return energy_revenue + rec_revenue

    def get_lifetime_revenue(self, lifetime_years: int) -> float:
        """Calculate total revenue over system lifetime."""
        return sum(self.get_annual_revenue(year) for year in range(lifetime_years))


@dataclass
class CircularityMetrics:
    """
    Circular economy metrics for 3R approach (Reduce, Reuse, Recycle).

    Attributes:
        material_recovery_rate: Percentage of materials recoverable (0-1)
        recovered_material_value: Value of recovered materials (USD/kg)
        system_weight: Total system weight (kg)
        refurbishment_potential: Percentage reusable as-is (0-1)
        refurbishment_value: Value retention after refurbishment (0-1)
        recycling_cost: Cost to recycle materials (USD/kg)
        recycling_revenue: Revenue from recycled materials (USD/kg)
        avoided_disposal_cost: Cost avoided by recycling vs disposal (USD/kg)
    """

    material_recovery_rate: float = 0.90  # 90% default
    recovered_material_value: float = 15.0  # USD/kg
    system_weight: float = 1000.0  # kg
    refurbishment_potential: float = 0.30  # 30% can be refurbished
    refurbishment_value: float = 0.40  # Retains 40% of original value
    recycling_cost: float = 5.0  # USD/kg
    recycling_revenue: float = 12.0  # USD/kg
    avoided_disposal_cost: float = 8.0  # USD/kg

    def get_eol_recovery_value(self, original_system_value: float) -> float:
        """
        Calculate total end-of-life value recovery.

        Args:
            original_system_value: Original system purchase value (USD)

        Returns:
            Total recovery value (USD)
        """
        # Refurbishment value
        refurb_value = (original_system_value * self.refurbishment_potential *
                       self.refurbishment_value)

        # Recycling value (for non-refurbished portion)
        recyclable_weight = self.system_weight * (1 - self.refurbishment_potential)
        recovered_weight = recyclable_weight * self.material_recovery_rate
        recycling_value = (recovered_weight *
                          (self.recycling_revenue - self.recycling_cost))

        # Avoided disposal costs
        avoided_costs = recovered_weight * self.avoided_disposal_cost

        return refurb_value + recycling_value + avoided_costs

    def get_circularity_score(self) -> float:
        """
        Calculate overall circularity score (0-100).

        Combines material recovery, refurbishment potential, and economic value.
        """
        material_score = self.material_recovery_rate * 40
        refurb_score = self.refurbishment_potential * 30
        economic_score = min((self.recycling_revenue /
                             max(self.recycling_cost, 0.01)) * 10, 30)

        return material_score + refurb_score + economic_score


@dataclass
class CashFlowModel:
    """
    Complete cash flow model for PV system financial analysis.

    Attributes:
        cost_structure: System cost structure
        revenue_stream: Revenue stream configuration
        circularity_metrics: Circular economy metrics
        lifetime_years: System operational lifetime (years)
        discount_rate: Discount rate for NPV calculations (decimal)
        tax_rate: Corporate tax rate (decimal)
        inflation_rate: General inflation rate (decimal)
        start_date: Project start date
    """

    cost_structure: CostStructure
    revenue_stream: RevenueStream
    circularity_metrics: CircularityMetrics
    lifetime_years: int = 25
    discount_rate: float = 0.06  # 6% default
    tax_rate: float = 0.21  # 21% default
    inflation_rate: float = 0.025  # 2.5% default
    start_date: datetime = field(default_factory=datetime.now)

    def generate_cash_flow_series(self) -> pd.DataFrame:
        """
        Generate complete cash flow series over system lifetime.

        Returns:
            DataFrame with columns: year, revenue, costs, net_cash_flow,
            discounted_cash_flow, cumulative_cash_flow
        """
        years = range(self.lifetime_years + 1)
        data = []

        cumulative_cf = 0.0

        for year in years:
            if year == 0:
                # Initial year - CAPEX only
                revenue = (self.revenue_stream.tax_credits +
                          self.revenue_stream.subsidies)
                costs = self.cost_structure.get_total_capex()
                net_cf = revenue - costs
            elif year == self.lifetime_years:
                # Final year - include EOL recovery
                revenue = self.revenue_stream.get_annual_revenue(year - 1)
                eol_recovery = self.circularity_metrics.get_eol_recovery_value(
                    self.cost_structure.equipment_cost
                )
                revenue += eol_recovery

                costs = self.cost_structure.get_total_annual_opex()
                costs += self.cost_structure.decommissioning_cost
                costs += self.replacement_costs.get(year, 0.0)

                net_cf = revenue - costs
            else:
                # Operational years
                revenue = self.revenue_stream.get_annual_revenue(year - 1)
                costs = self.cost_structure.get_total_annual_opex()
                costs += self.cost_structure.replacement_costs.get(year, 0.0)

                net_cf = revenue - costs

            # Apply discount factor
            discount_factor = 1 / (1 + self.discount_rate) ** year
            discounted_cf = net_cf * discount_factor

            cumulative_cf += net_cf

            data.append({
                'year': year,
                'revenue': revenue,
                'costs': costs,
                'net_cash_flow': net_cf,
                'discount_factor': discount_factor,
                'discounted_cash_flow': discounted_cf,
                'cumulative_cash_flow': cumulative_cf,
            })

        return pd.DataFrame(data)

    def calculate_npv(self) -> float:
        """Calculate Net Present Value."""
        df = self.generate_cash_flow_series()
        return df['discounted_cash_flow'].sum()

    def calculate_irr(self) -> float:
        """Calculate Internal Rate of Return."""
        df = self.generate_cash_flow_series()
        cash_flows = df['net_cash_flow'].values

        # Use numpy's IRR calculation
        try:
            return np.irr(cash_flows)
        except:
            # Fallback to manual calculation if numpy.irr fails
            return self._calculate_irr_manual(cash_flows)

    def _calculate_irr_manual(self, cash_flows: np.ndarray) -> float:
        """Manual IRR calculation using Newton-Raphson method."""

        def npv_at_rate(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

        # Newton-Raphson iteration
        rate = 0.1  # Initial guess
        for _ in range(100):
            npv = npv_at_rate(rate)
            if abs(npv) < 1e-6:
                return rate

            # Derivative approximation
            delta = 0.0001
            derivative = (npv_at_rate(rate + delta) - npv) / delta

            if abs(derivative) < 1e-10:
                break

            rate = rate - npv / derivative

        return rate

    def calculate_payback_period(self) -> float:
        """
        Calculate simple payback period in years.

        Returns:
            Payback period in years, or np.inf if never pays back
        """
        df = self.generate_cash_flow_series()

        for idx, row in df.iterrows():
            if row['cumulative_cash_flow'] >= 0:
                if idx == 0:
                    return 0.0

                # Linear interpolation for fractional year
                prev_cumulative = df.iloc[idx - 1]['cumulative_cash_flow']
                curr_cumulative = row['cumulative_cash_flow']
                years_into_period = abs(prev_cumulative) / (curr_cumulative - prev_cumulative)

                return idx - 1 + years_into_period

        return np.inf

    def calculate_roi(self) -> float:
        """Calculate Return on Investment as percentage."""
        total_investment = self.cost_structure.get_total_capex()
        lifetime_revenue = self.revenue_stream.get_lifetime_revenue(self.lifetime_years)
        lifetime_opex = (self.cost_structure.get_total_annual_opex() *
                        self.lifetime_years)

        total_return = lifetime_revenue - lifetime_opex

        return (total_return / total_investment) * 100 if total_investment > 0 else 0.0


@dataclass
class SensitivityParameter:
    """
    Parameter definition for sensitivity analysis.

    Attributes:
        name: Parameter name
        base_value: Base case value
        min_value: Minimum value for sensitivity range
        max_value: Maximum value for sensitivity range
        step: Step size for parameter variation
        unit: Unit of measurement
    """

    name: str
    base_value: float
    min_value: float
    max_value: float
    step: float
    unit: str = ""

    def get_range(self) -> np.ndarray:
        """Get array of values across sensitivity range."""
        return np.arange(self.min_value, self.max_value + self.step, self.step)

    def get_variation_percentages(self) -> np.ndarray:
        """Get percentage variations from base value."""
        if self.base_value == 0:
            return np.zeros_like(self.get_range())
        return ((self.get_range() - self.base_value) / self.base_value) * 100
