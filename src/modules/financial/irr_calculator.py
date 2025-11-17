"""
B13-S03: IRR Modeling
Production-ready Internal Rate of Return calculator with MIRR and hurdle rate analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import numpy_financial as npf
from scipy.optimize import brentq

from ..core.data_models import ProjectFinancials, IRRResult, FinancingStructure
from .npv_analyzer import NPVAnalyzer


class IRRCalculator:
    """
    Comprehensive IRR calculator with modified IRR and equity IRR.
    """

    def __init__(self, project: ProjectFinancials):
        """
        Initialize IRR calculator.

        Args:
            project: Project financial parameters
        """
        self.project = project

    def internal_rate_of_return(self,
                                annual_revenue: float) -> IRRResult:
        """
        Calculate project IRR and related metrics.

        Args:
            annual_revenue: First year revenue

        Returns:
            IRR calculation results
        """
        # Generate cash flows
        npv_analyzer = NPVAnalyzer(self.project)
        cash_flows_list = npv_analyzer.cash_flow_projection(annual_revenue)

        # Create cash flow array (including initial investment)
        cash_flows = [-self.project.capex_usd]
        cash_flows.extend([cf.free_cash_flow for cf in cash_flows_list])

        # Calculate IRR
        try:
            irr = npf.irr(cash_flows)
            irr_percent = irr * 100 if not np.isnan(irr) else 0
        except:
            irr_percent = 0

        # Hurdle rate (typically WACC)
        hurdle_rate_percent = self.project.discount_rate * 100

        # Check if IRR exceeds hurdle rate
        exceeds_hurdle = irr_percent > hurdle_rate_percent

        return IRRResult(
            irr_percent=irr_percent,
            mirr_percent=None,  # Will be calculated separately
            hurdle_rate_percent=hurdle_rate_percent,
            exceeds_hurdle=exceeds_hurdle,
            equity_irr=None,  # Will be calculated separately
            project_irr=irr_percent
        )

    def modified_irr(self,
                    annual_revenue: float,
                    finance_rate: Optional[float] = None,
                    reinvestment_rate: Optional[float] = None) -> float:
        """
        Calculate Modified Internal Rate of Return (MIRR).

        Args:
            annual_revenue: First year revenue
            finance_rate: Financing rate for negative cash flows
            reinvestment_rate: Reinvestment rate for positive cash flows

        Returns:
            MIRR as percentage
        """
        if finance_rate is None:
            finance_rate = self.project.discount_rate
        if reinvestment_rate is None:
            reinvestment_rate = self.project.discount_rate

        # Generate cash flows
        npv_analyzer = NPVAnalyzer(self.project)
        cash_flows_list = npv_analyzer.cash_flow_projection(annual_revenue)

        cash_flows = [-self.project.capex_usd]
        cash_flows.extend([cf.free_cash_flow for cf in cash_flows_list])

        try:
            mirr = npf.mirr(cash_flows, finance_rate, reinvestment_rate)
            return mirr * 100 if not np.isnan(mirr) else 0
        except:
            return 0

    def hurdle_rate_comparison(self,
                               annual_revenue: float,
                               hurdle_rates: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Compare IRR against multiple hurdle rates.

        Args:
            annual_revenue: Annual revenue
            hurdle_rates: List of hurdle rates to compare

        Returns:
            DataFrame comparing IRR to hurdle rates
        """
        if hurdle_rates is None:
            hurdle_rates = [0.06, 0.08, 0.10, 0.12, 0.15]

        irr_result = self.internal_rate_of_return(annual_revenue)
        project_irr = irr_result.irr_percent / 100

        results = []
        for hurdle in hurdle_rates:
            exceeds = project_irr > hurdle
            spread = (project_irr - hurdle) * 100  # basis points

            results.append({
                'hurdle_rate_percent': hurdle * 100,
                'project_irr_percent': project_irr * 100,
                'exceeds_hurdle': exceeds,
                'spread_bps': spread * 100,
                'decision': 'Accept' if exceeds else 'Reject'
            })

        return pd.DataFrame(results)

    def equity_irr_analysis(self,
                           annual_revenue: float,
                           financing: FinancingStructure) -> Dict[str, float]:
        """
        Calculate equity IRR for leveraged project.

        Args:
            annual_revenue: Annual revenue
            financing: Financing structure

        Returns:
            Equity IRR analysis
        """
        # Generate unleveraged cash flows
        npv_analyzer = NPVAnalyzer(self.project)
        cash_flows_list = npv_analyzer.cash_flow_projection(annual_revenue)

        # Equity contribution (initial)
        equity_cash_flows = [-financing.equity_amount]

        # Annual debt service
        annual_debt_service = npf.pmt(
            financing.debt_interest_rate,
            financing.debt_term_years,
            -financing.debt_amount
        )

        # Calculate equity cash flows
        for cf in cash_flows_list:
            # Equity CF = Project CF - Debt Service
            if cf.year <= financing.debt_term_years:
                equity_cf = cf.free_cash_flow - annual_debt_service
            else:
                equity_cf = cf.free_cash_flow

            equity_cash_flows.append(equity_cf)

        # Calculate equity IRR
        try:
            equity_irr = npf.irr(equity_cash_flows)
            equity_irr_percent = equity_irr * 100 if not np.isnan(equity_irr) else 0
        except:
            equity_irr_percent = 0

        # Calculate project IRR (unleveraged)
        project_cash_flows = [-self.project.capex_usd]
        project_cash_flows.extend([cf.free_cash_flow for cf in cash_flows_list])

        try:
            project_irr = npf.irr(project_cash_flows)
            project_irr_percent = project_irr * 100 if not np.isnan(project_irr) else 0
        except:
            project_irr_percent = 0

        # Leverage multiple
        leverage_multiple = equity_irr_percent / project_irr_percent if project_irr_percent > 0 else 0

        return {
            'equity_irr_percent': equity_irr_percent,
            'project_irr_percent': project_irr_percent,
            'leverage_multiple': leverage_multiple,
            'exceeds_equity_target': equity_irr_percent > (financing.equity_return_target * 100),
            'debt_to_equity_ratio': financing.debt_amount / financing.equity_amount if financing.equity_amount > 0 else 0
        }


__all__ = ["IRRCalculator"]
