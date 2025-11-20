"""
B13-S04: Bankability Assessment
Production-ready bankability analyzer with risk assessment and credit rating.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ..core.data_models import (
    ProjectFinancials,
    BankabilityAssessment,
    DebtServiceCoverage,
    FinancingStructure,
    RiskLevel
)
from .npv_analyzer import NPVAnalyzer


class BankabilityAnalyzer:
    """
    Comprehensive bankability assessment for renewable energy projects.
    """

    def __init__(self, project: ProjectFinancials, financing: FinancingStructure):
        """
        Initialize bankability analyzer.

        Args:
            project: Project financial parameters
            financing: Financing structure
        """
        self.project = project
        self.financing = financing

    def risk_assessment(self, annual_revenue: float) -> Dict[str, any]:
        """
        Assess project risks.

        Args:
            annual_revenue: Expected annual revenue

        Returns:
            Risk assessment results
        """
        risk_factors = []
        risk_scores = {}

        # Technology risk
        if self.project.capacity_kw < 1000:
            tech_risk = "LOW"
            tech_score = 20
        elif self.project.capacity_kw < 10000:
            tech_risk = "MEDIUM"
            tech_score = 50
        else:
            tech_risk = "LOW"
            tech_score = 30

        risk_scores['technology_risk'] = tech_score
        if tech_risk != "LOW":
            risk_factors.append(f"Technology risk: {tech_risk}")

        # Market risk (based on revenue volatility)
        revenue_to_capex_ratio = (annual_revenue * self.project.project_lifetime_years) / self.project.capex_usd
        if revenue_to_capex_ratio < 1.5:
            market_risk = "HIGH"
            market_score = 80
            risk_factors.append("Low revenue-to-CAPEX ratio")
        elif revenue_to_capex_ratio < 2.5:
            market_risk = "MEDIUM"
            market_score = 50
        else:
            market_risk = "LOW"
            market_score = 20

        risk_scores['market_risk'] = market_score

        # Financial risk (leverage ratio)
        leverage_ratio = self.financing.debt_amount / self.financing.total_project_cost
        if leverage_ratio > 0.8:
            financial_risk = "HIGH"
            financial_score = 80
            risk_factors.append("High leverage ratio (>80%)")
        elif leverage_ratio > 0.6:
            financial_risk = "MEDIUM"
            financial_score = 50
        else:
            financial_risk = "LOW"
            financial_score = 20

        risk_scores['financial_risk'] = financial_score

        # Construction risk
        if self.project.capex_usd > 100_000_000:
            construction_risk = "MEDIUM"
            construction_score = 50
            risk_factors.append("Large project construction risk")
        else:
            construction_risk = "LOW"
            construction_score = 20

        risk_scores['construction_risk'] = construction_score

        # Overall risk score (weighted average)
        overall_score = (
            tech_score * 0.25 +
            market_score * 0.35 +
            financial_score * 0.30 +
            construction_score * 0.10
        )

        if overall_score < 30:
            overall_risk = RiskLevel.LOW
        elif overall_score < 50:
            overall_risk = RiskLevel.MEDIUM
        elif overall_score < 70:
            overall_risk = RiskLevel.HIGH
        else:
            overall_risk = RiskLevel.CRITICAL

        return {
            'overall_risk_level': overall_risk,
            'overall_risk_score': overall_score,
            'risk_components': risk_scores,
            'risk_factors': risk_factors
        }

    def debt_service_coverage(self, annual_revenue: float) -> List[DebtServiceCoverage]:
        """
        Calculate Debt Service Coverage Ratio over loan term.

        Args:
            annual_revenue: First year revenue

        Returns:
            Annual DSCR values
        """
        import numpy_financial as npf

        # Generate cash flows
        npv_analyzer = NPVAnalyzer(self.project)
        cash_flows = npv_analyzer.cash_flow_projection(annual_revenue)

        # Annual debt service
        if self.financing.debt_amount > 0:
            annual_debt_service = abs(npf.pmt(
                self.financing.debt_interest_rate,
                self.financing.debt_term_years,
                -self.financing.debt_amount
            ))
        else:
            annual_debt_service = 0

        dscr_results = []

        for cf in cash_flows:
            if cf.year <= self.financing.debt_term_years:
                # EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization
                ebitda = cf.net_income + cf.taxes + cf.depreciation
                # Add back interest expense
                interest_expense = self.financing.debt_amount * self.financing.debt_interest_rate
                ebitda += interest_expense

                # DSCR
                if annual_debt_service > 0:
                    dscr = ebitda / annual_debt_service
                else:
                    dscr = float('inf')

                meets_covenant = dscr >= 1.2  # Typical minimum DSCR requirement

                dscr_results.append(DebtServiceCoverage(
                    year=cf.year,
                    ebitda=ebitda,
                    debt_service=annual_debt_service,
                    dscr=dscr,
                    minimum_dscr=1.2,
                    meets_covenant=meets_covenant
                ))

        return dscr_results

    def credit_rating(self, annual_revenue: float) -> BankabilityAssessment:
        """
        Determine project bankability and credit rating.

        Args:
            annual_revenue: Annual revenue

        Returns:
            Comprehensive bankability assessment
        """
        # Risk assessment
        risk_assessment = self.risk_assessment(annual_revenue)

        # DSCR analysis
        dscr_values = self.debt_service_coverage(annual_revenue)
        if dscr_values:
            average_dscr = np.mean([d.dscr for d in dscr_values])
            minimum_dscr = np.min([d.dscr for d in dscr_values])
        else:
            average_dscr = 0
            minimum_dscr = 0

        # Credit score calculation (0-100 scale)
        credit_score = 100

        # DSCR component (40 points max)
        if minimum_dscr >= 1.5:
            dscr_points = 40
        elif minimum_dscr >= 1.3:
            dscr_points = 30
        elif minimum_dscr >= 1.2:
            dscr_points = 20
        elif minimum_dscr >= 1.0:
            dscr_points = 10
        else:
            dscr_points = 0

        credit_score = dscr_points

        # Risk score component (30 points max)
        risk_points = max(0, 30 - (risk_assessment['overall_risk_score'] * 0.3))
        credit_score += risk_points

        # LTV ratio component (20 points max)
        ltv_ratio = self.financing.debt_amount / self.financing.total_project_cost if self.financing.total_project_cost > 0 else 0
        if ltv_ratio <= 0.6:
            ltv_points = 20
        elif ltv_ratio <= 0.7:
            ltv_points = 15
        elif ltv_ratio <= 0.8:
            ltv_points = 10
        else:
            ltv_points = 5

        credit_score += ltv_points

        # Project size and track record (10 points max)
        if self.project.capacity_kw > 10000:
            size_points = 10
        elif self.project.capacity_kw > 1000:
            size_points = 5
        else:
            size_points = 2

        credit_score += size_points

        # Determine overall rating
        if credit_score >= 80:
            overall_rating = RiskLevel.LOW
        elif credit_score >= 60:
            overall_rating = RiskLevel.MEDIUM
        elif credit_score >= 40:
            overall_rating = RiskLevel.HIGH
        else:
            overall_rating = RiskLevel.CRITICAL

        # Bankability decision
        bankable = (
            minimum_dscr >= 1.2 and
            average_dscr >= 1.3 and
            overall_rating in [RiskLevel.LOW, RiskLevel.MEDIUM] and
            ltv_ratio <= 0.8
        )

        # Recommended debt capacity
        if bankable:
            # Conservative estimate: debt that maintains DSCR > 1.3
            recommended_debt = min(
                self.financing.debt_amount,
                self.financing.total_project_cost * 0.7
            )
        else:
            recommended_debt = self.financing.total_project_cost * 0.5  # Lower leverage

        # Debt to equity ratio
        debt_to_equity = self.financing.debt_amount / self.financing.equity_amount if self.financing.equity_amount > 0 else float('inf')

        # Mitigation strategies
        mitigation_strategies = []
        if ltv_ratio > 0.7:
            mitigation_strategies.append("Increase equity contribution")
        if minimum_dscr < 1.3:
            mitigation_strategies.append("Improve revenue projections or reduce costs")
        if overall_rating == RiskLevel.HIGH:
            mitigation_strategies.append("Implement comprehensive risk management plan")
        if debt_to_equity > 2.0:
            mitigation_strategies.append("Reduce leverage ratio")

        return BankabilityAssessment(
            overall_rating=overall_rating,
            credit_score=credit_score,
            minimum_dscr=minimum_dscr,
            average_dscr=average_dscr,
            loan_to_value_ratio=ltv_ratio,
            debt_to_equity_ratio=debt_to_equity,
            risk_factors=risk_assessment['risk_factors'],
            mitigation_strategies=mitigation_strategies,
            bankable=bankable,
            recommended_debt_capacity=recommended_debt
        )


__all__ = ["BankabilityAnalyzer"]
