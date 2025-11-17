"""
Pydantic models for financial assessment and bankability analysis.

This module provides comprehensive data models for financial risk assessment,
credit rating, and bankability evaluation of PV projects.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class CreditRating(str, Enum):
    """Credit rating categories based on standardized rating scales."""

    AAA = "AAA"  # Exceptional creditworthiness
    AA = "AA"  # Very strong creditworthiness
    A = "A"  # Strong creditworthiness
    BBB = "BBB"  # Adequate creditworthiness (Investment grade threshold)
    BB = "BB"  # Speculative
    B = "B"  # Highly speculative
    CCC = "CCC"  # Substantial risk
    CC = "CC"  # Very high risk
    C = "C"  # Exceptionally high risk
    D = "D"  # Default


class RiskLevel(str, Enum):
    """Risk level categories for project assessment."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class ProjectStage(str, Enum):
    """Project development stages."""

    PRE_FEASIBILITY = "pre_feasibility"
    FEASIBILITY = "feasibility"
    DEVELOPMENT = "development"
    CONSTRUCTION = "construction"
    OPERATIONAL = "operational"
    DECOMMISSIONING = "decommissioning"


class FinancialMetrics(BaseModel):
    """Core financial metrics for project evaluation.

    Attributes:
        total_project_cost: Total capital expenditure for the project (USD)
        equity_contribution: Equity investment amount (USD)
        debt_amount: Total debt financing (USD)
        annual_revenue: Projected annual revenue (USD)
        annual_operating_cost: Annual operating and maintenance costs (USD)
        annual_debt_service: Annual debt service payment (principal + interest) (USD)
        project_lifespan_years: Expected operational lifespan (years)
        discount_rate: Discount rate for NPV calculations (decimal, e.g., 0.08 for 8%)
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    total_project_cost: float = Field(
        ..., gt=0, description="Total capital expenditure (USD)"
    )
    equity_contribution: float = Field(..., ge=0, description="Equity investment (USD)")
    debt_amount: float = Field(..., ge=0, description="Debt financing (USD)")
    annual_revenue: float = Field(..., gt=0, description="Projected annual revenue (USD)")
    annual_operating_cost: float = Field(
        ..., ge=0, description="Annual operating costs (USD)"
    )
    annual_debt_service: float = Field(
        ..., ge=0, description="Annual debt service payment (USD)"
    )
    project_lifespan_years: int = Field(
        ..., gt=0, le=50, description="Project operational lifespan (years)"
    )
    discount_rate: float = Field(
        ..., gt=0, le=1, description="Discount rate (decimal)"
    )

    @field_validator("equity_contribution", "debt_amount")
    @classmethod
    def validate_financing_structure(cls, v: float, info) -> float:
        """Validate that equity and debt are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v


class CashFlowProjection(BaseModel):
    """Cash flow projection for a specific year.

    Attributes:
        year: Project year (1-indexed)
        revenue: Revenue for the year (USD)
        operating_cost: Operating costs for the year (USD)
        capital_expenditure: Capital expenditure for the year (USD)
        debt_service: Debt service payment for the year (USD)
        net_cash_flow: Net cash flow after all expenses (USD)
    """

    model_config = ConfigDict(frozen=False)

    year: int = Field(..., gt=0, description="Project year (1-indexed)")
    revenue: float = Field(..., ge=0, description="Annual revenue (USD)")
    operating_cost: float = Field(..., ge=0, description="Operating costs (USD)")
    capital_expenditure: float = Field(..., ge=0, description="Capital expenditure (USD)")
    debt_service: float = Field(..., ge=0, description="Debt service (USD)")
    net_cash_flow: float = Field(..., description="Net cash flow (USD)")


class RiskFactor(BaseModel):
    """Individual risk factor assessment.

    Attributes:
        name: Risk factor name
        category: Risk category (technical, financial, market, regulatory, etc.)
        probability: Probability of occurrence (0.0 to 1.0)
        impact: Impact severity if occurs (0.0 to 1.0)
        risk_score: Combined risk score (probability × impact × 100)
        mitigation: Proposed mitigation strategies
    """

    model_config = ConfigDict(frozen=False)

    name: str = Field(..., min_length=1, description="Risk factor name")
    category: str = Field(..., min_length=1, description="Risk category")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of occurrence (0-1)"
    )
    impact: float = Field(
        ..., ge=0.0, le=1.0, description="Impact severity (0-1)"
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Combined risk score"
    )
    mitigation: Optional[str] = Field(
        None, description="Mitigation strategies"
    )

    @field_validator("risk_score")
    @classmethod
    def validate_risk_score(cls, v: float, info) -> float:
        """Ensure risk score is consistent with probability and impact."""
        if v < 0 or v > 100:
            raise ValueError("Risk score must be between 0 and 100")
        return v


class RiskAssessmentResult(BaseModel):
    """Comprehensive risk assessment results.

    Attributes:
        overall_risk_level: Overall project risk classification
        overall_risk_score: Aggregate risk score (0-100)
        risk_factors: List of individual risk factor assessments
        technical_risk_score: Technical risk component (0-100)
        financial_risk_score: Financial risk component (0-100)
        market_risk_score: Market risk component (0-100)
        regulatory_risk_score: Regulatory risk component (0-100)
        assessment_date: Timestamp of assessment
        recommendations: List of recommended actions
    """

    model_config = ConfigDict(frozen=False)

    overall_risk_level: RiskLevel = Field(..., description="Overall risk classification")
    overall_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Aggregate risk score (0-100)"
    )
    risk_factors: List[RiskFactor] = Field(
        default_factory=list, description="Individual risk factors"
    )
    technical_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Technical risk (0-100)"
    )
    financial_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Financial risk (0-100)"
    )
    market_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Market risk (0-100)"
    )
    regulatory_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Regulatory risk (0-100)"
    )
    assessment_date: datetime = Field(
        default_factory=datetime.now, description="Assessment timestamp"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )


class CreditRatingResult(BaseModel):
    """Credit rating assessment results.

    Attributes:
        rating: Credit rating classification
        rating_score: Numerical rating score (0-100, higher is better)
        financial_strength_score: Financial strength component (0-100)
        debt_capacity_score: Debt capacity component (0-100)
        liquidity_score: Liquidity component (0-100)
        profitability_score: Profitability component (0-100)
        key_metrics: Dictionary of key financial ratios
        rating_rationale: Explanation of rating determination
        assessment_date: Timestamp of assessment
    """

    model_config = ConfigDict(frozen=False)

    rating: CreditRating = Field(..., description="Credit rating")
    rating_score: float = Field(
        ..., ge=0.0, le=100.0, description="Numerical rating score (0-100)"
    )
    financial_strength_score: float = Field(
        ..., ge=0.0, le=100.0, description="Financial strength (0-100)"
    )
    debt_capacity_score: float = Field(
        ..., ge=0.0, le=100.0, description="Debt capacity (0-100)"
    )
    liquidity_score: float = Field(
        ..., ge=0.0, le=100.0, description="Liquidity (0-100)"
    )
    profitability_score: float = Field(
        ..., ge=0.0, le=100.0, description="Profitability (0-100)"
    )
    key_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Key financial ratios"
    )
    rating_rationale: str = Field(..., min_length=1, description="Rating explanation")
    assessment_date: datetime = Field(
        default_factory=datetime.now, description="Assessment timestamp"
    )


class DebtServiceCoverageResult(BaseModel):
    """Debt service coverage ratio analysis results.

    Attributes:
        dscr: Debt Service Coverage Ratio (EBITDA / Annual Debt Service)
        ebitda: Earnings Before Interest, Taxes, Depreciation, Amortization (USD)
        annual_debt_service: Annual debt service payment (USD)
        coverage_level: Assessment of coverage adequacy
        minimum_dscr_requirement: Typical lender requirement (e.g., 1.20)
        meets_requirement: Whether DSCR meets typical lender standards
        annual_dscr_projections: Year-by-year DSCR projections
        assessment_date: Timestamp of assessment
    """

    model_config = ConfigDict(frozen=False)

    dscr: float = Field(..., ge=0.0, description="Debt Service Coverage Ratio")
    ebitda: float = Field(..., description="EBITDA (USD)")
    annual_debt_service: float = Field(..., ge=0.0, description="Annual debt service (USD)")
    coverage_level: str = Field(..., description="Coverage adequacy assessment")
    minimum_dscr_requirement: float = Field(
        default=1.20, description="Minimum DSCR requirement"
    )
    meets_requirement: bool = Field(..., description="Meets lender requirements")
    annual_dscr_projections: List[float] = Field(
        default_factory=list, description="Year-by-year DSCR"
    )
    assessment_date: datetime = Field(
        default_factory=datetime.now, description="Assessment timestamp"
    )


class BankabilityScore(BaseModel):
    """Comprehensive bankability assessment score.

    Attributes:
        overall_score: Overall bankability score (0-100, higher is better)
        credit_rating_component: Credit rating contribution (0-100)
        risk_component: Risk assessment contribution (0-100)
        financial_viability_component: Financial viability contribution (0-100)
        dscr_component: DSCR contribution (0-100)
        bankability_level: Qualitative assessment (Excellent, Good, Fair, Poor)
        is_bankable: Whether project is considered bankable
        lender_attractiveness: Attractiveness to lenders (0-100)
        key_strengths: List of project strengths
        key_weaknesses: List of project weaknesses
        recommendations: Recommendations for improving bankability
        assessment_date: Timestamp of assessment
    """

    model_config = ConfigDict(frozen=False)

    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall bankability score (0-100)"
    )
    credit_rating_component: float = Field(
        ..., ge=0.0, le=100.0, description="Credit rating contribution"
    )
    risk_component: float = Field(
        ..., ge=0.0, le=100.0, description="Risk assessment contribution"
    )
    financial_viability_component: float = Field(
        ..., ge=0.0, le=100.0, description="Financial viability contribution"
    )
    dscr_component: float = Field(
        ..., ge=0.0, le=100.0, description="DSCR contribution"
    )
    bankability_level: str = Field(..., description="Qualitative assessment")
    is_bankable: bool = Field(..., description="Is project bankable")
    lender_attractiveness: float = Field(
        ..., ge=0.0, le=100.0, description="Lender attractiveness (0-100)"
    )
    key_strengths: List[str] = Field(
        default_factory=list, description="Project strengths"
    )
    key_weaknesses: List[str] = Field(
        default_factory=list, description="Project weaknesses"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    assessment_date: datetime = Field(
        default_factory=datetime.now, description="Assessment timestamp"
    )


class ProjectContext(BaseModel):
    """Context and metadata for project being assessed.

    Attributes:
        project_name: Project identifier
        project_stage: Current development stage
        location: Geographic location
        capacity_mw: System capacity in megawatts
        technology_type: PV technology type (e.g., monocrystalline, thin-film)
        offtaker: Power purchase agreement offtaker
        ppa_term_years: PPA term length in years
        ppa_rate_usd_per_kwh: PPA rate in USD per kWh
    """

    model_config = ConfigDict(frozen=False)

    project_name: str = Field(..., min_length=1, description="Project identifier")
    project_stage: ProjectStage = Field(..., description="Development stage")
    location: str = Field(..., min_length=1, description="Geographic location")
    capacity_mw: float = Field(..., gt=0, description="System capacity (MW)")
    technology_type: str = Field(..., min_length=1, description="PV technology type")
    offtaker: Optional[str] = Field(None, description="PPA offtaker")
    ppa_term_years: Optional[int] = Field(None, gt=0, description="PPA term (years)")
    ppa_rate_usd_per_kwh: Optional[float] = Field(
        None, gt=0, description="PPA rate (USD/kWh)"
    )
