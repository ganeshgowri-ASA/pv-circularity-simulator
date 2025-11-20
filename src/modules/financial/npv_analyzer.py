"""
B13-S02: NPV Analysis
Production-ready Net Present Value analyzer with cash flow projections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import numpy_financial as npf

from ..core.data_models import ProjectFinancials, NPVResult, CashFlowProjection
from ..core.utilities import present_value


class NPVAnalyzer:
    """
    Comprehensive NPV analyzer with cash flow modeling and project valuation.
    """

    def __init__(self, project: ProjectFinancials):
        """
        Initialize NPV analyzer.

        Args:
            project: Project financial parameters
        """
        self.project = project

    def cash_flow_projection(self,
                             annual_revenue: float,
                             include_depreciation: bool = True,
                             depreciation_method: str = "straight_line") -> List[CashFlowProjection]:
        """
        Project annual cash flows over project lifetime.

        Args:
            annual_revenue: First year revenue
            include_depreciation: Whether to include depreciation
            depreciation_method: Depreciation method

        Returns:
            List of annual cash flow projections
        """
        cash_flows = []
        cumulative_cash_flow = -self.project.capex_usd  # Initial investment

        for year in range(1, self.project.project_lifetime_years + 1):
            # Revenue with degradation
            revenue = annual_revenue * (1 - self.project.degradation_rate) ** (year - 1)

            # O&M costs with inflation
            opex = self.project.opex_annual_usd * (1 + self.project.inflation_rate) ** (year - 1)

            # Depreciation
            if include_depreciation:
                if depreciation_method == "straight_line":
                    depreciation = self.project.capex_usd / self.project.project_lifetime_years
                elif depreciation_method == "macrs":
                    # Simplified MACRS (5-year property for solar)
                    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
                    if year <= len(macrs_schedule):
                        depreciation = self.project.capex_usd * macrs_schedule[year - 1]
                    else:
                        depreciation = 0
                else:
                    depreciation = 0
            else:
                depreciation = 0

            # Taxable income
            taxable_income = revenue - opex - depreciation

            # Taxes
            taxes = max(0, taxable_income * self.project.tax_rate)

            # Net income
            net_income = taxable_income - taxes

            # Free cash flow (add back depreciation, non-cash expense)
            free_cash_flow = net_income + depreciation

            # Cumulative
            cumulative_cash_flow += free_cash_flow

            cash_flows.append(CashFlowProjection(
                year=year,
                revenue=revenue,
                operating_expenses=opex,
                depreciation=depreciation,
                taxable_income=taxable_income,
                taxes=taxes,
                net_income=net_income,
                free_cash_flow=free_cash_flow,
                cumulative_cash_flow=cumulative_cash_flow
            ))

        return cash_flows

    def project_valuation(self,
                         annual_revenue: float,
                         terminal_value_multiple: Optional[float] = None) -> NPVResult:
        """
        Calculate project NPV and related metrics.

        Args:
            annual_revenue: First year revenue
            terminal_value_multiple: Terminal value as multiple of final year revenue

        Returns:
            NPV analysis results
        """
        # Generate cash flows
        cash_flows = self.cash_flow_projection(annual_revenue)

        # Calculate NPV
        npv = -self.project.capex_usd
        for cf in cash_flows:
            pv = present_value(cf.free_cash_flow, self.project.discount_rate, cf.year)
            npv += pv

        # Add terminal value if specified
        if terminal_value_multiple:
            final_revenue = cash_flows[-1].revenue
            terminal_value = final_revenue * terminal_value_multiple
            terminal_pv = present_value(terminal_value, self.project.discount_rate,
                                       self.project.project_lifetime_years)
            npv += terminal_pv

        # Benefit-cost ratio
        total_benefits_pv = sum(
            present_value(cf.revenue, self.project.discount_rate, cf.year)
            for cf in cash_flows
        )
        bcr = total_benefits_pv / self.project.capex_usd if self.project.capex_usd > 0 else 0

        # Simple payback period
        cumulative = -self.project.capex_usd
        simple_payback = None
        for cf in cash_flows:
            cumulative += cf.free_cash_flow
            if cumulative > 0 and simple_payback is None:
                simple_payback = cf.year

        # Discounted payback period
        cumulative_discounted = -self.project.capex_usd
        discounted_payback = None
        for cf in cash_flows:
            pv = present_value(cf.free_cash_flow, self.project.discount_rate, cf.year)
            cumulative_discounted += pv
            if cumulative_discounted > 0 and discounted_payback is None:
                discounted_payback = cf.year

        # Profitability index
        pv_future_cash_flows = sum(
            present_value(cf.free_cash_flow, self.project.discount_rate, cf.year)
            for cf in cash_flows
        )
        profitability_index = pv_future_cash_flows / self.project.capex_usd if self.project.capex_usd > 0 else 0

        return NPVResult(
            npv_usd=npv,
            benefit_cost_ratio=bcr,
            payback_period_years=simple_payback,
            discounted_payback_years=discounted_payback,
            profitability_index=profitability_index,
            annual_cash_flows=cash_flows
        )

    def discount_rate_analysis(self,
                               annual_revenue: float,
                               discount_rates: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Analyze NPV sensitivity to discount rate.

        Args:
            annual_revenue: Annual revenue
            discount_rates: List of discount rates to test

        Returns:
            DataFrame with NPV for each discount rate
        """
        if discount_rates is None:
            discount_rates = [0.04, 0.06, 0.08, 0.10, 0.12]

        original_rate = self.project.discount_rate
        results = []

        for rate in discount_rates:
            self.project.discount_rate = rate
            npv_result = self.project_valuation(annual_revenue)

            results.append({
                'discount_rate': rate,
                'npv': npv_result.npv_usd,
                'profitability_index': npv_result.profitability_index,
                'payback_period': npv_result.payback_period_years
            })

        # Restore original rate
        self.project.discount_rate = original_rate

        return pd.DataFrame(results)


__all__ = ["NPVAnalyzer"]
