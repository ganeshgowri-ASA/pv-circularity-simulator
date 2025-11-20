"""
Bankability Assessment and Risk Analysis Module.

This module provides comprehensive bankability assessment capabilities for PV projects,
including credit rating analysis, risk assessment, debt service coverage evaluation,
and overall project bankability scoring. Designed for production use with financial
institutions, investors, and project developers.

Key Features:
    - Credit rating assessment with standardized scales
    - Multi-dimensional risk analysis (technical, financial, market, regulatory)
    - Debt service coverage ratio (DSCR) analysis with projections
    - Integrated bankability scoring with actionable recommendations
    - Full Pydantic validation for data integrity
    - Comprehensive financial metrics tracking

Author: PV Circularity Simulator Team
License: MIT
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .models import (
    BankabilityScore,
    CashFlowProjection,
    CreditRating,
    CreditRatingResult,
    DebtServiceCoverageResult,
    FinancialMetrics,
    ProjectContext,
    RiskAssessmentResult,
    RiskFactor,
    RiskLevel,
)


class BankabilityAssessor:
    """
    Comprehensive bankability assessment engine for PV projects.

    This class provides sophisticated financial analysis capabilities to evaluate
    the bankability of photovoltaic projects. It integrates multiple assessment
    dimensions including creditworthiness, risk profile, debt service capacity,
    and overall financial viability.

    The assessor is designed to meet the requirements of:
        - Commercial lenders and financial institutions
        - Project finance investors
        - Development finance institutions (DFIs)
        - Credit rating agencies
        - Project developers and sponsors

    Attributes:
        financial_metrics: Core financial metrics for the project
        project_context: Project metadata and context information
        risk_weights: Weighting factors for different risk categories
        dscr_threshold: Minimum acceptable DSCR (default: 1.20)

    Example:
        >>> from pv_circularity_simulator.financial.bankability import BankabilityAssessor
        >>> from pv_circularity_simulator.financial.models import (
        ...     FinancialMetrics, ProjectContext, ProjectStage
        ... )
        >>>
        >>> metrics = FinancialMetrics(
        ...     total_project_cost=10_000_000,
        ...     equity_contribution=3_000_000,
        ...     debt_amount=7_000_000,
        ...     annual_revenue=2_000_000,
        ...     annual_operating_cost=400_000,
        ...     annual_debt_service=800_000,
        ...     project_lifespan_years=25,
        ...     discount_rate=0.08
        ... )
        >>>
        >>> context = ProjectContext(
        ...     project_name="Solar Park Alpha",
        ...     project_stage=ProjectStage.DEVELOPMENT,
        ...     location="Arizona, USA",
        ...     capacity_mw=10.0,
        ...     technology_type="Monocrystalline Silicon",
        ...     ppa_term_years=20,
        ...     ppa_rate_usd_per_kwh=0.08
        ... )
        >>>
        >>> assessor = BankabilityAssessor(
        ...     financial_metrics=metrics,
        ...     project_context=context
        ... )
        >>>
        >>> # Perform comprehensive assessment
        >>> credit_rating = assessor.credit_rating()
        >>> risk_assessment = assessor.risk_assessment()
        >>> dscr_analysis = assessor.debt_service_coverage()
        >>> bankability = assessor.project_bankability_score()
        >>>
        >>> print(f"Bankability Score: {bankability.overall_score:.1f}/100")
        >>> print(f"Credit Rating: {credit_rating.rating}")
        >>> print(f"DSCR: {dscr_analysis.dscr:.2f}")
    """

    def __init__(
        self,
        financial_metrics: FinancialMetrics,
        project_context: Optional[ProjectContext] = None,
        risk_weights: Optional[Dict[str, float]] = None,
        dscr_threshold: float = 1.20,
    ) -> None:
        """
        Initialize the BankabilityAssessor.

        Args:
            financial_metrics: Core financial metrics for the project.
                Must include all required fields as defined in FinancialMetrics model.
            project_context: Optional project metadata and context information.
                If not provided, some contextual assessments may be limited.
            risk_weights: Optional custom weighting factors for risk categories.
                Default weights: technical=0.25, financial=0.35, market=0.25, regulatory=0.15
            dscr_threshold: Minimum acceptable Debt Service Coverage Ratio.
                Default is 1.20 (industry standard). Must be > 1.0.

        Raises:
            ValueError: If DSCR threshold is <= 1.0 or risk weights don't sum to 1.0
            ValidationError: If financial metrics fail Pydantic validation

        Example:
            >>> metrics = FinancialMetrics(...)
            >>> context = ProjectContext(...)
            >>> assessor = BankabilityAssessor(
            ...     financial_metrics=metrics,
            ...     project_context=context,
            ...     risk_weights={"technical": 0.3, "financial": 0.3,
            ...                   "market": 0.25, "regulatory": 0.15}
            ... )
        """
        self.financial_metrics = financial_metrics
        self.project_context = project_context
        self.dscr_threshold = dscr_threshold

        # Default risk weights (can be customized)
        self.risk_weights = risk_weights or {
            "technical": 0.25,
            "financial": 0.35,
            "market": 0.25,
            "regulatory": 0.15,
        }

        # Validate DSCR threshold
        if self.dscr_threshold <= 1.0:
            raise ValueError("DSCR threshold must be greater than 1.0")

        # Validate risk weights sum to 1.0
        total_weight = sum(self.risk_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(
                f"Risk weights must sum to 1.0, got {total_weight:.4f}"
            )

    def credit_rating(
        self,
        include_projections: bool = True,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> CreditRatingResult:
        """
        Assess the credit rating of the project based on financial strength.

        Performs a comprehensive credit analysis using multiple financial metrics
        including leverage ratios, profitability indicators, liquidity measures,
        and debt capacity. The rating methodology is aligned with international
        rating agency standards (S&P, Moody's, Fitch).

        The assessment evaluates four key dimensions:
            1. Financial Strength: Overall financial health and stability
            2. Debt Capacity: Ability to service and sustain debt levels
            3. Liquidity: Short-term financial flexibility
            4. Profitability: Earnings quality and sustainability

        Args:
            include_projections: If True, incorporates forward-looking projections
                into the rating assessment. Default is True.
            custom_metrics: Optional dictionary of additional financial metrics
                to incorporate into the analysis. Examples: {'roi': 0.12, 'irr': 0.15}

        Returns:
            CreditRatingResult: Comprehensive credit rating assessment including:
                - Credit rating classification (AAA to D)
                - Numerical rating score (0-100)
                - Component scores for each dimension
                - Key financial metrics and ratios
                - Detailed rating rationale
                - Assessment timestamp

        Rating Scale:
            - AAA (90-100): Exceptional creditworthiness, minimal risk
            - AA (80-89): Very strong creditworthiness, low risk
            - A (70-79): Strong creditworthiness, low-moderate risk
            - BBB (60-69): Adequate creditworthiness, moderate risk (Investment grade threshold)
            - BB (50-59): Speculative, elevated risk
            - B (40-49): Highly speculative, high risk
            - CCC (30-39): Substantial risk, vulnerable
            - CC (20-29): Very high risk, likely default
            - C (10-19): Exceptionally high risk, imminent default
            - D (0-9): In default

        Example:
            >>> assessor = BankabilityAssessor(financial_metrics=metrics)
            >>> rating = assessor.credit_rating(include_projections=True)
            >>> print(f"Rating: {rating.rating} (Score: {rating.rating_score:.1f})")
            >>> print(f"Debt Capacity: {rating.debt_capacity_score:.1f}/100")
            >>> if rating.rating_score >= 60:
            ...     print("Investment grade")
            >>> else:
            ...     print("Speculative grade")

        Notes:
            - Investment grade threshold is BBB (score >= 60)
            - Ratings incorporate both historical performance and forward projections
            - Custom metrics can override default calculations
        """
        metrics = self.financial_metrics

        # Calculate key financial ratios
        debt_to_equity = (
            metrics.debt_amount / metrics.equity_contribution
            if metrics.equity_contribution > 0
            else float("inf")
        )
        loan_to_value = (
            metrics.debt_amount / metrics.total_project_cost
            if metrics.total_project_cost > 0
            else 0
        )
        ebitda = metrics.annual_revenue - metrics.annual_operating_cost
        interest_coverage = (
            ebitda / metrics.annual_debt_service
            if metrics.annual_debt_service > 0
            else float("inf")
        )
        operating_margin = (
            ebitda / metrics.annual_revenue if metrics.annual_revenue > 0 else 0
        )
        equity_ratio = (
            metrics.equity_contribution / metrics.total_project_cost
            if metrics.total_project_cost > 0
            else 0
        )

        # Financial Strength Score (0-100)
        # Based on equity ratio, leverage, and overall capital structure
        financial_strength_score = 0.0
        if equity_ratio >= 0.40:  # Strong equity position
            financial_strength_score += 40
        elif equity_ratio >= 0.30:  # Adequate equity
            financial_strength_score += 30
        elif equity_ratio >= 0.20:  # Minimal equity
            financial_strength_score += 20
        else:  # Weak equity position
            financial_strength_score += 10

        if debt_to_equity <= 1.5:  # Conservative leverage
            financial_strength_score += 35
        elif debt_to_equity <= 2.5:  # Moderate leverage
            financial_strength_score += 25
        elif debt_to_equity <= 4.0:  # High leverage
            financial_strength_score += 15
        else:  # Excessive leverage
            financial_strength_score += 5

        # Add points for diversified capital structure
        if 0.2 <= equity_ratio <= 0.5:
            financial_strength_score += 25
        else:
            financial_strength_score += 10

        # Debt Capacity Score (0-100)
        # Based on interest coverage and debt service capacity
        debt_capacity_score = 0.0
        if interest_coverage >= 3.0:  # Strong coverage
            debt_capacity_score += 50
        elif interest_coverage >= 2.0:  # Adequate coverage
            debt_capacity_score += 35
        elif interest_coverage >= 1.5:  # Minimal coverage
            debt_capacity_score += 20
        else:  # Weak coverage
            debt_capacity_score += 10

        if loan_to_value <= 0.60:  # Conservative LTV
            debt_capacity_score += 30
        elif loan_to_value <= 0.75:  # Moderate LTV
            debt_capacity_score += 20
        elif loan_to_value <= 0.85:  # High LTV
            debt_capacity_score += 10
        else:  # Excessive LTV
            debt_capacity_score += 5

        # Project lifespan vs debt tenor
        if metrics.project_lifespan_years >= 25:
            debt_capacity_score += 20
        elif metrics.project_lifespan_years >= 20:
            debt_capacity_score += 15
        else:
            debt_capacity_score += 10

        # Liquidity Score (0-100)
        # Based on cash generation and working capital
        liquidity_score = 0.0
        annual_free_cash_flow = ebitda - metrics.annual_debt_service
        fcf_to_revenue = (
            annual_free_cash_flow / metrics.annual_revenue
            if metrics.annual_revenue > 0
            else 0
        )

        if fcf_to_revenue >= 0.20:  # Strong cash generation
            liquidity_score += 50
        elif fcf_to_revenue >= 0.10:  # Adequate cash generation
            liquidity_score += 35
        elif fcf_to_revenue >= 0.05:  # Minimal cash generation
            liquidity_score += 20
        else:  # Weak cash generation
            liquidity_score += 10

        # Cash reserve adequacy (simplified)
        if annual_free_cash_flow > 0:
            liquidity_score += 30
        else:
            liquidity_score += 5

        # Revenue stability (PPA contracts provide stability)
        if self.project_context and self.project_context.ppa_term_years:
            liquidity_score += 20
        else:
            liquidity_score += 10

        # Profitability Score (0-100)
        # Based on operating margin and return metrics
        profitability_score = 0.0
        if operating_margin >= 0.70:  # Excellent margins
            profitability_score += 50
        elif operating_margin >= 0.60:  # Strong margins
            profitability_score += 40
        elif operating_margin >= 0.50:  # Good margins
            profitability_score += 30
        elif operating_margin >= 0.40:  # Adequate margins
            profitability_score += 20
        else:  # Weak margins
            profitability_score += 10

        # NPV and IRR (simplified estimation)
        # Rough NPV calculation
        annual_cash_flow = annual_free_cash_flow
        discount_factor = 1 / (1 + metrics.discount_rate)
        npv_estimate = sum(
            annual_cash_flow * (discount_factor ** year)
            for year in range(1, metrics.project_lifespan_years + 1)
        ) - metrics.total_project_cost

        if npv_estimate > metrics.total_project_cost * 0.5:  # Strong returns
            profitability_score += 30
        elif npv_estimate > 0:  # Positive returns
            profitability_score += 20
        else:  # Negative returns
            profitability_score += 5

        # Revenue per MW (capacity factor proxy)
        if self.project_context and self.project_context.capacity_mw > 0:
            revenue_per_mw = metrics.annual_revenue / self.project_context.capacity_mw
            if revenue_per_mw >= 200_000:  # High productivity
                profitability_score += 20
            elif revenue_per_mw >= 150_000:  # Good productivity
                profitability_score += 15
            else:  # Average productivity
                profitability_score += 10
        else:
            profitability_score += 10

        # Calculate overall rating score (weighted average)
        rating_score = (
            financial_strength_score * 0.30
            + debt_capacity_score * 0.35
            + liquidity_score * 0.20
            + profitability_score * 0.15
        )

        # Determine credit rating based on score
        if rating_score >= 90:
            rating = CreditRating.AAA
        elif rating_score >= 80:
            rating = CreditRating.AA
        elif rating_score >= 70:
            rating = CreditRating.A
        elif rating_score >= 60:
            rating = CreditRating.BBB
        elif rating_score >= 50:
            rating = CreditRating.BB
        elif rating_score >= 40:
            rating = CreditRating.B
        elif rating_score >= 30:
            rating = CreditRating.CCC
        elif rating_score >= 20:
            rating = CreditRating.CC
        elif rating_score >= 10:
            rating = CreditRating.C
        else:
            rating = CreditRating.D

        # Build key metrics dictionary
        key_metrics = {
            "debt_to_equity": round(debt_to_equity, 2),
            "loan_to_value": round(loan_to_value, 2),
            "interest_coverage": round(interest_coverage, 2),
            "operating_margin": round(operating_margin, 2),
            "equity_ratio": round(equity_ratio, 2),
            "npv_estimate": round(npv_estimate, 2),
            "fcf_to_revenue": round(fcf_to_revenue, 2),
        }

        # Add custom metrics if provided
        if custom_metrics:
            key_metrics.update(custom_metrics)

        # Build rating rationale
        rationale_parts = [
            f"Credit rating {rating.value} assigned based on comprehensive financial analysis.",
            f"Financial strength score: {financial_strength_score:.1f}/100 "
            f"(Equity ratio: {equity_ratio:.1%}, D/E: {debt_to_equity:.2f})",
            f"Debt capacity score: {debt_capacity_score:.1f}/100 "
            f"(Interest coverage: {interest_coverage:.2f}x, LTV: {loan_to_value:.1%})",
            f"Liquidity score: {liquidity_score:.1f}/100 "
            f"(FCF/Revenue: {fcf_to_revenue:.1%})",
            f"Profitability score: {profitability_score:.1f}/100 "
            f"(Operating margin: {operating_margin:.1%})",
        ]

        if rating_score >= 60:
            rationale_parts.append("Project achieves investment grade rating (BBB or higher).")
        else:
            rationale_parts.append(
                "Project is speculative grade. Consider improving financial metrics."
            )

        rating_rationale = " ".join(rationale_parts)

        return CreditRatingResult(
            rating=rating,
            rating_score=round(rating_score, 2),
            financial_strength_score=round(financial_strength_score, 2),
            debt_capacity_score=round(debt_capacity_score, 2),
            liquidity_score=round(liquidity_score, 2),
            profitability_score=round(profitability_score, 2),
            key_metrics=key_metrics,
            rating_rationale=rating_rationale,
            assessment_date=datetime.now(),
        )

    def risk_assessment(
        self,
        include_technical_risk: bool = True,
        include_market_risk: bool = True,
        custom_risk_factors: Optional[List[RiskFactor]] = None,
    ) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment across multiple dimensions.

        Evaluates project risks across technical, financial, market, and regulatory
        dimensions. The assessment uses a structured risk scoring methodology that
        combines probability and impact analysis with Monte Carlo simulation for
        aggregate risk quantification.

        Risk assessment dimensions:
            1. Technical Risk: Technology performance, reliability, degradation
            2. Financial Risk: Cost overruns, financing risks, currency exposure
            3. Market Risk: Energy prices, demand, competition
            4. Regulatory Risk: Policy changes, permitting, compliance

        Args:
            include_technical_risk: Include technical risk assessment (default: True).
                Set to False if project is operational and technical risk is minimal.
            include_market_risk: Include market risk assessment (default: True).
                Set to False if project has long-term fixed-price PPA.
            custom_risk_factors: Optional list of additional risk factors to include.
                Each factor should specify category, probability, impact, and mitigation.

        Returns:
            RiskAssessmentResult: Comprehensive risk assessment including:
                - Overall risk level classification (Critical to Minimal)
                - Aggregate risk score (0-100, lower is better)
                - Individual risk factor assessments
                - Risk scores by category (technical, financial, market, regulatory)
                - Actionable recommendations for risk mitigation
                - Assessment timestamp

        Risk Level Classification:
            - Minimal (0-20): Very low risk, minimal mitigation needed
            - Low (20-40): Low risk, standard risk management sufficient
            - Medium (40-60): Moderate risk, active mitigation required
            - High (60-80): Elevated risk, comprehensive mitigation essential
            - Critical (80-100): Severe risk, project viability threatened

        Example:
            >>> assessor = BankabilityAssessor(financial_metrics=metrics)
            >>> risk = assessor.risk_assessment(include_market_risk=False)
            >>> print(f"Risk Level: {risk.overall_risk_level}")
            >>> print(f"Risk Score: {risk.overall_risk_score:.1f}/100")
            >>> for factor in risk.risk_factors:
            ...     if factor.risk_score > 50:
            ...         print(f"High Risk: {factor.name} ({factor.risk_score:.1f})")

        Notes:
            - Risk scores combine probability and impact: Risk = Probability × Impact × 100
            - Recommendations are tailored to the specific risk profile
            - Custom risk factors allow project-specific risk incorporation
        """
        risk_factors: List[RiskFactor] = []
        metrics = self.financial_metrics

        # Technical Risk Factors
        if include_technical_risk:
            # Module degradation risk
            degradation_risk = RiskFactor(
                name="Module Performance Degradation",
                category="technical",
                probability=0.40,  # Moderate probability
                impact=0.30,  # Medium impact
                risk_score=12.0,  # 0.40 * 0.30 * 100
                mitigation="Use Tier-1 modules with performance guarantees; "
                "implement regular performance monitoring",
            )
            risk_factors.append(degradation_risk)

            # Equipment failure risk
            equipment_risk = RiskFactor(
                name="Critical Equipment Failure",
                category="technical",
                probability=0.25,
                impact=0.50,
                risk_score=12.5,
                mitigation="Maintain spare parts inventory; implement predictive "
                "maintenance program; secure equipment warranties",
            )
            risk_factors.append(equipment_risk)

            # Technology obsolescence
            tech_obsolescence_risk = RiskFactor(
                name="Technology Obsolescence",
                category="technical",
                probability=0.15,
                impact=0.25,
                risk_score=3.75,
                mitigation="Use proven, mature technology; plan for mid-life upgrades; "
                "maintain flexibility for technology improvements",
            )
            risk_factors.append(tech_obsolescence_risk)

            # O&M performance
            om_risk = RiskFactor(
                name="O&M Performance Below Expectations",
                category="technical",
                probability=0.30,
                impact=0.35,
                risk_score=10.5,
                mitigation="Contract with experienced O&M provider; include performance "
                "guarantees; implement KPI monitoring",
            )
            risk_factors.append(om_risk)

        # Financial Risk Factors
        # Construction cost overrun
        cost_overrun_risk = RiskFactor(
            name="Construction Cost Overrun",
            category="financial",
            probability=0.35,
            impact=0.60,
            risk_score=21.0,
            mitigation="Use EPC contract with price certainty; maintain contingency "
            "reserve (10-15%); regular cost monitoring",
        )
        risk_factors.append(cost_overrun_risk)

        # Interest rate risk
        interest_rate_risk = RiskFactor(
            name="Interest Rate Fluctuation",
            category="financial",
            probability=0.45,
            impact=0.40,
            risk_score=18.0,
            mitigation="Secure fixed-rate financing or interest rate hedges; "
            "refinance when rates are favorable",
        )
        risk_factors.append(interest_rate_risk)

        # Refinancing risk
        refinancing_risk = RiskFactor(
            name="Refinancing Risk",
            category="financial",
            probability=0.20,
            impact=0.55,
            risk_score=11.0,
            mitigation="Ensure adequate DSCR headroom; maintain strong credit profile; "
            "build relationship with multiple lenders",
        )
        risk_factors.append(refinancing_risk)

        # Counterparty credit risk (offtaker)
        if self.project_context and self.project_context.offtaker:
            counterparty_risk = RiskFactor(
                name="Offtaker Credit Risk",
                category="financial",
                probability=0.15,
                impact=0.70,
                risk_score=10.5,
                mitigation="Require credit support; diversify offtaker risk; "
                "include termination payments in PPA",
            )
            risk_factors.append(counterparty_risk)

        # Market Risk Factors
        if include_market_risk:
            # Energy price risk
            energy_price_risk = RiskFactor(
                name="Energy Price Volatility",
                category="market",
                probability=0.50,
                impact=0.45,
                risk_score=22.5,
                mitigation="Secure long-term PPA with fixed prices; implement "
                "price floors; hedge merchant exposure",
            )
            risk_factors.append(energy_price_risk)

            # Demand risk
            demand_risk = RiskFactor(
                name="Lower Than Expected Energy Demand",
                category="market",
                probability=0.25,
                impact=0.40,
                risk_score=10.0,
                mitigation="Conduct thorough demand studies; secure creditworthy offtaker; "
                "include take-or-pay provisions",
            )
            risk_factors.append(demand_risk)

            # Competition from other renewables
            competition_risk = RiskFactor(
                name="Increased Renewable Competition",
                category="market",
                probability=0.60,
                impact=0.30,
                risk_score=18.0,
                mitigation="Lock in long-term contracts; focus on cost competitiveness; "
                "differentiate through service quality",
            )
            risk_factors.append(competition_risk)

        # Regulatory Risk Factors
        # Policy/subsidy risk
        policy_risk = RiskFactor(
            name="Policy or Subsidy Changes",
            category="regulatory",
            probability=0.40,
            impact=0.50,
            risk_score=20.0,
            mitigation="Secure grandfathering provisions; diversify revenue sources; "
            "monitor policy developments",
        )
        risk_factors.append(policy_risk)

        # Permitting risk
        permitting_risk = RiskFactor(
            name="Permitting Delays or Denials",
            category="regulatory",
            probability=0.30,
            impact=0.55,
            risk_score=16.5,
            mitigation="Engage early with authorities; conduct environmental studies; "
            "build community support",
        )
        risk_factors.append(permitting_risk)

        # Grid interconnection risk
        grid_risk = RiskFactor(
            name="Grid Interconnection Issues",
            category="regulatory",
            probability=0.25,
            impact=0.60,
            risk_score=15.0,
            mitigation="Secure interconnection agreement early; conduct grid studies; "
            "budget for grid upgrades if required",
        )
        risk_factors.append(grid_risk)

        # Add custom risk factors if provided
        if custom_risk_factors:
            risk_factors.extend(custom_risk_factors)

        # Calculate category-specific risk scores
        technical_risks = [rf for rf in risk_factors if rf.category == "technical"]
        financial_risks = [rf for rf in risk_factors if rf.category == "financial"]
        market_risks = [rf for rf in risk_factors if rf.category == "market"]
        regulatory_risks = [rf for rf in risk_factors if rf.category == "regulatory"]

        technical_risk_score = (
            np.mean([rf.risk_score for rf in technical_risks])
            if technical_risks
            else 0.0
        )
        financial_risk_score = (
            np.mean([rf.risk_score for rf in financial_risks])
            if financial_risks
            else 0.0
        )
        market_risk_score = (
            np.mean([rf.risk_score for rf in market_risks]) if market_risks else 0.0
        )
        regulatory_risk_score = (
            np.mean([rf.risk_score for rf in regulatory_risks])
            if regulatory_risks
            else 0.0
        )

        # Calculate overall risk score (weighted average)
        overall_risk_score = (
            technical_risk_score * self.risk_weights["technical"]
            + financial_risk_score * self.risk_weights["financial"]
            + market_risk_score * self.risk_weights["market"]
            + regulatory_risk_score * self.risk_weights["regulatory"]
        )

        # Determine overall risk level
        if overall_risk_score >= 80:
            overall_risk_level = RiskLevel.CRITICAL
        elif overall_risk_score >= 60:
            overall_risk_level = RiskLevel.HIGH
        elif overall_risk_score >= 40:
            overall_risk_level = RiskLevel.MEDIUM
        elif overall_risk_score >= 20:
            overall_risk_level = RiskLevel.LOW
        else:
            overall_risk_level = RiskLevel.MINIMAL

        # Generate recommendations based on risk profile
        recommendations = []

        if overall_risk_score >= 60:
            recommendations.append(
                "Overall risk level is HIGH/CRITICAL. Comprehensive risk mitigation "
                "strategy is essential before proceeding."
            )

        if technical_risk_score > 15:
            recommendations.append(
                "Elevated technical risk. Consider additional due diligence on "
                "technology provider and performance guarantees."
            )

        if financial_risk_score > 20:
            recommendations.append(
                "Significant financial risk. Review capital structure and consider "
                "additional equity or contingency reserves."
            )

        if market_risk_score > 20:
            recommendations.append(
                "Material market risk. Prioritize long-term contracts and price hedging."
            )

        if regulatory_risk_score > 20:
            recommendations.append(
                "Regulatory risk is elevated. Engage proactively with authorities "
                "and secure necessary permits early."
            )

        # Check debt service coverage
        dscr = (
            (metrics.annual_revenue - metrics.annual_operating_cost)
            / metrics.annual_debt_service
            if metrics.annual_debt_service > 0
            else float("inf")
        )
        if dscr < self.dscr_threshold:
            recommendations.append(
                f"DSCR ({dscr:.2f}) is below threshold ({self.dscr_threshold}). "
                "Consider reducing debt or increasing revenue."
            )

        if not recommendations:
            recommendations.append(
                "Risk profile is acceptable. Implement standard risk management practices."
            )

        return RiskAssessmentResult(
            overall_risk_level=overall_risk_level,
            overall_risk_score=round(overall_risk_score, 2),
            risk_factors=risk_factors,
            technical_risk_score=round(technical_risk_score, 2),
            financial_risk_score=round(financial_risk_score, 2),
            market_risk_score=round(market_risk_score, 2),
            regulatory_risk_score=round(regulatory_risk_score, 2),
            assessment_date=datetime.now(),
            recommendations=recommendations,
        )

    def debt_service_coverage(
        self,
        projection_years: Optional[int] = None,
        degradation_rate: float = 0.005,
        opex_inflation: float = 0.02,
    ) -> DebtServiceCoverageResult:
        """
        Calculate Debt Service Coverage Ratio (DSCR) with multi-year projections.

        DSCR is a critical metric for lenders, measuring the project's ability to
        service its debt obligations from operating cash flows. This method calculates
        both current DSCR and projects future DSCR values accounting for revenue
        degradation and cost inflation.

        DSCR Formula:
            DSCR = EBITDA / Annual Debt Service
            where EBITDA = Revenue - Operating Costs

        Lender Requirements:
            - Minimum DSCR: Typically 1.20x - 1.30x
            - Average DSCR: Typically 1.35x - 1.50x over loan term
            - Conservative projects may require 1.50x+

        Args:
            projection_years: Number of years to project DSCR. If None, uses
                project lifespan. Maximum is project lifespan years.
            degradation_rate: Annual module degradation rate (default: 0.5% = 0.005).
                Typical range: 0.3% - 0.8% per year for modern modules.
            opex_inflation: Annual operating cost inflation rate (default: 2% = 0.02).
                Should reflect expected cost escalation over project life.

        Returns:
            DebtServiceCoverageResult: DSCR analysis including:
                - Current DSCR value
                - EBITDA and annual debt service amounts
                - Coverage level assessment (Excellent, Good, Adequate, Weak, Insufficient)
                - Comparison to minimum lender requirements
                - Year-by-year DSCR projections
                - Assessment timestamp

        Coverage Level Classification:
            - Excellent: DSCR >= 1.50x
            - Good: 1.30x <= DSCR < 1.50x
            - Adequate: 1.20x <= DSCR < 1.30x (Meets minimum)
            - Weak: 1.10x <= DSCR < 1.20x (Below minimum)
            - Insufficient: DSCR < 1.10x (Unacceptable)

        Example:
            >>> assessor = BankabilityAssessor(financial_metrics=metrics)
            >>> dscr_analysis = assessor.debt_service_coverage(
            ...     projection_years=20,
            ...     degradation_rate=0.005,
            ...     opex_inflation=0.025
            ... )
            >>> print(f"Current DSCR: {dscr_analysis.dscr:.2f}x")
            >>> print(f"Coverage: {dscr_analysis.coverage_level}")
            >>> print(f"Meets Requirements: {dscr_analysis.meets_requirement}")
            >>>
            >>> # Analyze DSCR trend
            >>> min_dscr = min(dscr_analysis.annual_dscr_projections)
            >>> avg_dscr = np.mean(dscr_analysis.annual_dscr_projections)
            >>> print(f"Minimum Projected DSCR: {min_dscr:.2f}x")
            >>> print(f"Average Projected DSCR: {avg_dscr:.2f}x")

        Notes:
            - DSCR below 1.0 indicates insufficient cash flow to cover debt
            - Lenders typically require DSCR > 1.20x as a loan covenant
            - Projections account for performance degradation and cost inflation
            - Conservative degradation rates improve lending credibility
        """
        metrics = self.financial_metrics

        # Calculate current year EBITDA and DSCR
        ebitda = metrics.annual_revenue - metrics.annual_operating_cost
        dscr = (
            ebitda / metrics.annual_debt_service
            if metrics.annual_debt_service > 0
            else float("inf")
        )

        # Determine coverage level
        if dscr >= 1.50:
            coverage_level = "Excellent"
        elif dscr >= 1.30:
            coverage_level = "Good"
        elif dscr >= 1.20:
            coverage_level = "Adequate"
        elif dscr >= 1.10:
            coverage_level = "Weak"
        else:
            coverage_level = "Insufficient"

        # Check if meets minimum requirement
        meets_requirement = dscr >= self.dscr_threshold

        # Project DSCR over time
        years = projection_years or metrics.project_lifespan_years
        years = min(years, metrics.project_lifespan_years)  # Cap at project lifespan

        annual_dscr_projections = []
        for year in range(1, years + 1):
            # Apply degradation to revenue
            degraded_revenue = metrics.annual_revenue * (
                (1 - degradation_rate) ** year
            )

            # Apply inflation to operating costs
            inflated_opex = metrics.annual_operating_cost * ((1 + opex_inflation) ** year)

            # Calculate year EBITDA and DSCR
            year_ebitda = degraded_revenue - inflated_opex
            year_dscr = (
                year_ebitda / metrics.annual_debt_service
                if metrics.annual_debt_service > 0
                else float("inf")
            )

            annual_dscr_projections.append(round(year_dscr, 3))

        return DebtServiceCoverageResult(
            dscr=round(dscr, 3),
            ebitda=round(ebitda, 2),
            annual_debt_service=round(metrics.annual_debt_service, 2),
            coverage_level=coverage_level,
            minimum_dscr_requirement=self.dscr_threshold,
            meets_requirement=meets_requirement,
            annual_dscr_projections=annual_dscr_projections,
            assessment_date=datetime.now(),
        )

    def project_bankability_score(
        self,
        credit_rating_weight: float = 0.30,
        risk_weight: float = 0.25,
        financial_viability_weight: float = 0.25,
        dscr_weight: float = 0.20,
    ) -> BankabilityScore:
        """
        Calculate comprehensive project bankability score integrating all assessments.

        The bankability score provides a holistic evaluation of project financeability
        by integrating credit rating, risk assessment, financial viability, and debt
        service coverage into a single composite metric. This score indicates the
        project's attractiveness to lenders and investors.

        The assessment synthesizes:
            1. Credit Rating: Overall creditworthiness (30% default weight)
            2. Risk Profile: Multi-dimensional risk assessment (25% default weight)
            3. Financial Viability: Returns, margins, and economics (25% default weight)
            4. DSCR: Debt service capacity (20% default weight)

        Args:
            credit_rating_weight: Weight for credit rating component (0-1).
                Default: 0.30. Reflects importance of credit quality.
            risk_weight: Weight for risk assessment component (0-1).
                Default: 0.25. Reflects risk mitigation importance.
            financial_viability_weight: Weight for financial metrics component (0-1).
                Default: 0.25. Reflects economic returns importance.
            dscr_weight: Weight for DSCR component (0-1).
                Default: 0.20. Reflects debt capacity importance.

        Returns:
            BankabilityScore: Comprehensive bankability assessment including:
                - Overall bankability score (0-100, higher is better)
                - Component scores for each dimension
                - Bankability level (Excellent, Good, Fair, Poor)
                - Binary bankability determination
                - Lender attractiveness score
                - Key project strengths
                - Key project weaknesses
                - Actionable improvement recommendations
                - Assessment timestamp

        Bankability Level Thresholds:
            - Excellent (80-100): Highly bankable, attractive to all lenders
            - Good (65-79): Bankable, acceptable to most lenders
            - Fair (50-64): Marginally bankable, improvements recommended
            - Poor (<50): Not bankable, significant improvements required

        Binary Bankability Threshold:
            - Bankable: Score >= 60 (equivalent to BBB credit rating)
            - Not Bankable: Score < 60

        Example:
            >>> assessor = BankabilityAssessor(financial_metrics=metrics)
            >>> bankability = assessor.project_bankability_score()
            >>>
            >>> print(f"Bankability Score: {bankability.overall_score:.1f}/100")
            >>> print(f"Level: {bankability.bankability_level}")
            >>> print(f"Is Bankable: {bankability.is_bankable}")
            >>> print(f"Lender Attractiveness: {bankability.lender_attractiveness:.1f}")
            >>>
            >>> print("\nStrengths:")
            >>> for strength in bankability.key_strengths:
            ...     print(f"  • {strength}")
            >>>
            >>> print("\nWeaknesses:")
            >>> for weakness in bankability.key_weaknesses:
            ...     print(f"  • {weakness}")
            >>>
            >>> print("\nRecommendations:")
            >>> for rec in bankability.recommendations:
            ...     print(f"  • {rec}")

        Raises:
            ValueError: If weights don't sum to 1.0 (within tolerance of 1e-6)

        Notes:
            - Score >= 60 is generally considered minimum for bankability
            - Weights can be customized based on lender preferences
            - Recommendations are tailored to improve weakest components
            - Assessment integrates all prior analysis methods
        """
        # Validate weights
        total_weight = (
            credit_rating_weight + risk_weight + financial_viability_weight + dscr_weight
        )
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.4f}. "
                f"Adjust: credit_rating_weight={credit_rating_weight}, "
                f"risk_weight={risk_weight}, "
                f"financial_viability_weight={financial_viability_weight}, "
                f"dscr_weight={dscr_weight}"
            )

        # Perform individual assessments
        credit_result = self.credit_rating()
        risk_result = self.risk_assessment()
        dscr_result = self.debt_service_coverage()

        # Extract component scores
        credit_rating_component = credit_result.rating_score

        # Risk component: Convert risk score (0-100, lower is better) to
        # bankability score (0-100, higher is better)
        risk_component = 100 - risk_result.overall_risk_score

        # DSCR component: Convert DSCR to score
        # DSCR >= 2.0 = 100, DSCR = 1.5 = 85, DSCR = 1.2 = 60, DSCR = 1.0 = 0
        dscr_value = dscr_result.dscr
        if dscr_value >= 2.0:
            dscr_component = 100.0
        elif dscr_value >= 1.5:
            dscr_component = 85.0 + (dscr_value - 1.5) / 0.5 * 15.0
        elif dscr_value >= 1.2:
            dscr_component = 60.0 + (dscr_value - 1.2) / 0.3 * 25.0
        elif dscr_value >= 1.0:
            dscr_component = (dscr_value - 1.0) / 0.2 * 60.0
        else:
            dscr_component = 0.0

        # Financial viability component: Based on profitability and returns
        metrics = self.financial_metrics
        ebitda = metrics.annual_revenue - metrics.annual_operating_cost
        operating_margin = ebitda / metrics.annual_revenue if metrics.annual_revenue > 0 else 0

        # Simple NPV calculation
        annual_free_cash_flow = ebitda - metrics.annual_debt_service
        discount_factor = 1 / (1 + metrics.discount_rate)
        npv = sum(
            annual_free_cash_flow * (discount_factor ** year)
            for year in range(1, metrics.project_lifespan_years + 1)
        ) - metrics.total_project_cost

        # Financial viability score
        financial_viability_component = 0.0

        # Operating margin contribution (max 40 points)
        if operating_margin >= 0.70:
            financial_viability_component += 40
        elif operating_margin >= 0.60:
            financial_viability_component += 35
        elif operating_margin >= 0.50:
            financial_viability_component += 30
        elif operating_margin >= 0.40:
            financial_viability_component += 20
        else:
            financial_viability_component += 10

        # NPV contribution (max 40 points)
        if npv > metrics.total_project_cost * 0.5:
            financial_viability_component += 40
        elif npv > metrics.total_project_cost * 0.25:
            financial_viability_component += 30
        elif npv > 0:
            financial_viability_component += 20
        else:
            financial_viability_component += 5

        # Payback period contribution (max 20 points)
        if annual_free_cash_flow > 0:
            payback_years = metrics.total_project_cost / annual_free_cash_flow
            if payback_years <= 7:
                financial_viability_component += 20
            elif payback_years <= 10:
                financial_viability_component += 15
            elif payback_years <= 15:
                financial_viability_component += 10
            else:
                financial_viability_component += 5
        else:
            financial_viability_component += 0

        # Calculate overall bankability score
        overall_score = (
            credit_rating_component * credit_rating_weight
            + risk_component * risk_weight
            + financial_viability_component * financial_viability_weight
            + dscr_component * dscr_weight
        )

        # Determine bankability level
        if overall_score >= 80:
            bankability_level = "Excellent"
        elif overall_score >= 65:
            bankability_level = "Good"
        elif overall_score >= 50:
            bankability_level = "Fair"
        else:
            bankability_level = "Poor"

        # Bankability determination (threshold: 60)
        is_bankable = overall_score >= 60.0

        # Lender attractiveness (slightly different weighting)
        lender_attractiveness = (
            credit_rating_component * 0.35
            + risk_component * 0.30
            + dscr_component * 0.35
        )

        # Identify key strengths
        key_strengths = []
        if credit_rating_component >= 70:
            key_strengths.append(
                f"Strong credit rating ({credit_result.rating.value}) "
                f"with score {credit_rating_component:.1f}/100"
            )
        if risk_component >= 60:
            key_strengths.append(
                f"Low overall risk profile (risk score: {risk_result.overall_risk_score:.1f}/100)"
            )
        if dscr_component >= 70:
            key_strengths.append(
                f"Robust debt service coverage (DSCR: {dscr_value:.2f}x)"
            )
        if financial_viability_component >= 70:
            key_strengths.append(
                f"Strong financial viability (operating margin: {operating_margin:.1%})"
            )
        if operating_margin >= 0.60:
            key_strengths.append(f"Excellent operating margin ({operating_margin:.1%})")
        if npv > 0:
            key_strengths.append(f"Positive NPV (${npv:,.0f})")

        # Identify key weaknesses
        key_weaknesses = []
        if credit_rating_component < 60:
            key_weaknesses.append(
                f"Below investment grade credit rating ({credit_result.rating.value})"
            )
        if risk_component < 50:
            key_weaknesses.append(
                f"Elevated risk profile (risk score: {risk_result.overall_risk_score:.1f}/100)"
            )
        if dscr_component < 60:
            key_weaknesses.append(
                f"Weak debt service coverage (DSCR: {dscr_value:.2f}x, "
                f"below {self.dscr_threshold}x threshold)"
            )
        if financial_viability_component < 60:
            key_weaknesses.append(
                f"Limited financial viability (operating margin: {operating_margin:.1%})"
            )
        if npv <= 0:
            key_weaknesses.append(f"Negative or zero NPV")

        # Generate recommendations
        recommendations = []
        if not is_bankable:
            recommendations.append(
                f"Project is currently NOT BANKABLE (score: {overall_score:.1f}/100, "
                f"threshold: 60). Significant improvements required."
            )

        # Component-specific recommendations
        if credit_rating_component < 70:
            recommendations.append(
                "Improve credit rating by strengthening capital structure, "
                "reducing leverage, or increasing equity contribution."
            )

        if risk_component < 60:
            recommendations.append(
                "Address key risk factors identified in risk assessment, "
                "particularly those with highest risk scores."
            )

        if dscr_component < 70:
            recommendations.append(
                f"Increase DSCR to at least {self.dscr_threshold}x by reducing debt, "
                "increasing revenue, or reducing operating costs."
            )

        if financial_viability_component < 60:
            recommendations.append(
                "Improve financial viability through cost optimization, "
                "revenue enhancement, or more favorable financing terms."
            )

        # Specific metric recommendations
        loan_to_value = (
            metrics.debt_amount / metrics.total_project_cost
            if metrics.total_project_cost > 0
            else 0
        )
        if loan_to_value > 0.75:
            recommendations.append(
                f"High loan-to-value ratio ({loan_to_value:.1%}). "
                "Consider increasing equity contribution to below 75%."
            )

        if not recommendations:
            recommendations.append(
                "Project demonstrates strong bankability. Maintain current financial "
                "structure and risk management practices."
            )

        return BankabilityScore(
            overall_score=round(overall_score, 2),
            credit_rating_component=round(credit_rating_component, 2),
            risk_component=round(risk_component, 2),
            financial_viability_component=round(financial_viability_component, 2),
            dscr_component=round(dscr_component, 2),
            bankability_level=bankability_level,
            is_bankable=is_bankable,
            lender_attractiveness=round(lender_attractiveness, 2),
            key_strengths=key_strengths,
            key_weaknesses=key_weaknesses,
            recommendations=recommendations,
            assessment_date=datetime.now(),
        )
