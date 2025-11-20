"""
Unit tests for financial assessment modules.

Tests cover bankability assessment, credit rating, risk analysis,
and debt service coverage calculations.
"""

import pytest
from datetime import datetime
from typing import Dict

from pv_circularity_simulator.financial.bankability import BankabilityAssessor
from pv_circularity_simulator.financial.models import (
    BankabilityScore,
    CreditRating,
    CreditRatingResult,
    DebtServiceCoverageResult,
    FinancialMetrics,
    ProjectContext,
    ProjectStage,
    RiskAssessmentResult,
    RiskFactor,
    RiskLevel,
)


class TestFinancialMetrics:
    """Test suite for FinancialMetrics Pydantic model."""

    def test_valid_financial_metrics(self):
        """Test creation of valid FinancialMetrics instance."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=3_000_000,
            debt_amount=7_000_000,
            annual_revenue=2_000_000,
            annual_operating_cost=400_000,
            annual_debt_service=800_000,
            project_lifespan_years=25,
            discount_rate=0.08,
        )
        assert metrics.total_project_cost == 10_000_000
        assert metrics.equity_contribution == 3_000_000
        assert metrics.debt_amount == 7_000_000
        assert metrics.project_lifespan_years == 25

    def test_invalid_negative_cost(self):
        """Test that negative project cost raises ValidationError."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            FinancialMetrics(
                total_project_cost=-1_000_000,  # Invalid: negative
                equity_contribution=3_000_000,
                debt_amount=7_000_000,
                annual_revenue=2_000_000,
                annual_operating_cost=400_000,
                annual_debt_service=800_000,
                project_lifespan_years=25,
                discount_rate=0.08,
            )

    def test_invalid_discount_rate(self):
        """Test that invalid discount rate raises ValidationError."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            FinancialMetrics(
                total_project_cost=10_000_000,
                equity_contribution=3_000_000,
                debt_amount=7_000_000,
                annual_revenue=2_000_000,
                annual_operating_cost=400_000,
                annual_debt_service=800_000,
                project_lifespan_years=25,
                discount_rate=1.5,  # Invalid: > 1.0
            )


class TestProjectContext:
    """Test suite for ProjectContext Pydantic model."""

    def test_valid_project_context(self):
        """Test creation of valid ProjectContext instance."""
        context = ProjectContext(
            project_name="Solar Park Alpha",
            project_stage=ProjectStage.DEVELOPMENT,
            location="Arizona, USA",
            capacity_mw=10.0,
            technology_type="Monocrystalline Silicon",
            ppa_term_years=20,
            ppa_rate_usd_per_kwh=0.08,
        )
        assert context.project_name == "Solar Park Alpha"
        assert context.project_stage == ProjectStage.DEVELOPMENT
        assert context.capacity_mw == 10.0


class TestBankabilityAssessor:
    """Test suite for BankabilityAssessor class."""

    @pytest.fixture
    def sample_metrics(self) -> FinancialMetrics:
        """Fixture providing sample financial metrics."""
        return FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=3_000_000,
            debt_amount=7_000_000,
            annual_revenue=2_000_000,
            annual_operating_cost=400_000,
            annual_debt_service=800_000,
            project_lifespan_years=25,
            discount_rate=0.08,
        )

    @pytest.fixture
    def sample_context(self) -> ProjectContext:
        """Fixture providing sample project context."""
        return ProjectContext(
            project_name="Solar Park Alpha",
            project_stage=ProjectStage.DEVELOPMENT,
            location="Arizona, USA",
            capacity_mw=10.0,
            technology_type="Monocrystalline Silicon",
            offtaker="Utility Company A",
            ppa_term_years=20,
            ppa_rate_usd_per_kwh=0.08,
        )

    @pytest.fixture
    def assessor(
        self, sample_metrics: FinancialMetrics, sample_context: ProjectContext
    ) -> BankabilityAssessor:
        """Fixture providing BankabilityAssessor instance."""
        return BankabilityAssessor(
            financial_metrics=sample_metrics, project_context=sample_context
        )

    def test_initialization(self, sample_metrics, sample_context):
        """Test BankabilityAssessor initialization."""
        assessor = BankabilityAssessor(
            financial_metrics=sample_metrics, project_context=sample_context
        )
        assert assessor.financial_metrics == sample_metrics
        assert assessor.project_context == sample_context
        assert assessor.dscr_threshold == 1.20

    def test_initialization_custom_threshold(self, sample_metrics):
        """Test initialization with custom DSCR threshold."""
        assessor = BankabilityAssessor(
            financial_metrics=sample_metrics, dscr_threshold=1.35
        )
        assert assessor.dscr_threshold == 1.35

    def test_invalid_dscr_threshold(self, sample_metrics):
        """Test that invalid DSCR threshold raises ValueError."""
        with pytest.raises(ValueError, match="DSCR threshold must be greater than 1.0"):
            BankabilityAssessor(financial_metrics=sample_metrics, dscr_threshold=0.9)

    def test_invalid_risk_weights(self, sample_metrics):
        """Test that invalid risk weights raise ValueError."""
        with pytest.raises(ValueError, match="Risk weights must sum to 1.0"):
            BankabilityAssessor(
                financial_metrics=sample_metrics,
                risk_weights={
                    "technical": 0.25,
                    "financial": 0.25,
                    "market": 0.25,
                    "regulatory": 0.10,  # Sum = 0.85, not 1.0
                },
            )

    def test_credit_rating_basic(self, assessor):
        """Test basic credit rating assessment."""
        result = assessor.credit_rating()

        assert isinstance(result, CreditRatingResult)
        assert isinstance(result.rating, CreditRating)
        assert 0 <= result.rating_score <= 100
        assert 0 <= result.financial_strength_score <= 100
        assert 0 <= result.debt_capacity_score <= 100
        assert 0 <= result.liquidity_score <= 100
        assert 0 <= result.profitability_score <= 100
        assert len(result.rating_rationale) > 0
        assert isinstance(result.assessment_date, datetime)

    def test_credit_rating_key_metrics(self, assessor):
        """Test that credit rating includes key financial metrics."""
        result = assessor.credit_rating()

        assert "debt_to_equity" in result.key_metrics
        assert "loan_to_value" in result.key_metrics
        assert "interest_coverage" in result.key_metrics
        assert "operating_margin" in result.key_metrics
        assert "equity_ratio" in result.key_metrics

        # Validate metric calculations
        assert result.key_metrics["equity_ratio"] == pytest.approx(0.30, rel=0.01)
        assert result.key_metrics["loan_to_value"] == pytest.approx(0.70, rel=0.01)

    def test_credit_rating_high_quality(self):
        """Test credit rating for high-quality project."""
        # Strong financial metrics
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=5_000_000,  # 50% equity
            debt_amount=5_000_000,
            annual_revenue=3_000_000,
            annual_operating_cost=500_000,
            annual_debt_service=500_000,
            project_lifespan_years=25,
            discount_rate=0.06,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)
        result = assessor.credit_rating()

        # Should achieve high rating
        assert result.rating_score >= 70  # At least A rating
        assert result.rating in [CreditRating.AAA, CreditRating.AA, CreditRating.A]

    def test_credit_rating_poor_quality(self):
        """Test credit rating for poor-quality project."""
        # Weak financial metrics
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=500_000,  # Only 5% equity
            debt_amount=9_500_000,
            annual_revenue=1_200_000,
            annual_operating_cost=600_000,
            annual_debt_service=1_000_000,
            project_lifespan_years=20,
            discount_rate=0.12,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)
        result = assessor.credit_rating()

        # Should achieve low rating
        assert result.rating_score < 60  # Below investment grade
        assert result.rating in [
            CreditRating.BB,
            CreditRating.B,
            CreditRating.CCC,
            CreditRating.CC,
            CreditRating.C,
            CreditRating.D,
        ]

    def test_risk_assessment_basic(self, assessor):
        """Test basic risk assessment."""
        result = assessor.risk_assessment()

        assert isinstance(result, RiskAssessmentResult)
        assert isinstance(result.overall_risk_level, RiskLevel)
        assert 0 <= result.overall_risk_score <= 100
        assert 0 <= result.technical_risk_score <= 100
        assert 0 <= result.financial_risk_score <= 100
        assert 0 <= result.market_risk_score <= 100
        assert 0 <= result.regulatory_risk_score <= 100
        assert len(result.risk_factors) > 0
        assert len(result.recommendations) > 0

    def test_risk_assessment_risk_factors(self, assessor):
        """Test that risk assessment includes various risk factors."""
        result = assessor.risk_assessment()

        # Check for different risk categories
        categories = {rf.category for rf in result.risk_factors}
        assert "technical" in categories
        assert "financial" in categories
        assert "market" in categories
        assert "regulatory" in categories

        # Validate individual risk factors
        for factor in result.risk_factors:
            assert isinstance(factor, RiskFactor)
            assert 0 <= factor.probability <= 1
            assert 0 <= factor.impact <= 1
            assert 0 <= factor.risk_score <= 100

    def test_risk_assessment_exclude_technical(self, assessor):
        """Test risk assessment with technical risk excluded."""
        result = assessor.risk_assessment(include_technical_risk=False)

        # Should not have technical risk factors
        technical_factors = [
            rf for rf in result.risk_factors if rf.category == "technical"
        ]
        assert len(technical_factors) == 0
        assert result.technical_risk_score == 0.0

    def test_risk_assessment_custom_factors(self, assessor):
        """Test risk assessment with custom risk factors."""
        custom_factors = [
            RiskFactor(
                name="Custom Risk",
                category="custom",
                probability=0.5,
                impact=0.6,
                risk_score=30.0,
                mitigation="Custom mitigation strategy",
            )
        ]

        result = assessor.risk_assessment(custom_risk_factors=custom_factors)

        # Should include custom factor
        assert any(rf.name == "Custom Risk" for rf in result.risk_factors)

    def test_debt_service_coverage_basic(self, assessor):
        """Test basic debt service coverage calculation."""
        result = assessor.debt_service_coverage()

        assert isinstance(result, DebtServiceCoverageResult)
        assert result.dscr > 0
        assert result.ebitda == 1_600_000  # 2M revenue - 400K opex
        assert result.annual_debt_service == 800_000
        assert result.dscr == pytest.approx(2.0, rel=0.01)  # 1.6M / 800K
        assert result.coverage_level == "Excellent"
        assert result.meets_requirement is True

    def test_debt_service_coverage_projections(self, assessor):
        """Test DSCR projections over time."""
        result = assessor.debt_service_coverage(projection_years=10)

        assert len(result.annual_dscr_projections) == 10

        # DSCR should decline over time due to degradation and inflation
        # First year should be close to base DSCR
        assert result.annual_dscr_projections[0] == pytest.approx(
            result.dscr, rel=0.05
        )

        # DSCR should generally decrease over time
        assert result.annual_dscr_projections[-1] < result.annual_dscr_projections[0]

    def test_debt_service_coverage_degradation_impact(self, assessor):
        """Test impact of degradation rate on DSCR projections."""
        # Low degradation
        result_low = assessor.debt_service_coverage(
            projection_years=10, degradation_rate=0.003
        )
        # High degradation
        result_high = assessor.debt_service_coverage(
            projection_years=10, degradation_rate=0.008
        )

        # Higher degradation should result in lower final DSCR
        assert result_high.annual_dscr_projections[-1] < result_low.annual_dscr_projections[-1]

    def test_debt_service_coverage_weak(self):
        """Test DSCR assessment for weak coverage."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=2_000_000,
            debt_amount=8_000_000,
            annual_revenue=1_500_000,
            annual_operating_cost=600_000,
            annual_debt_service=800_000,
            project_lifespan_years=20,
            discount_rate=0.10,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)
        result = assessor.debt_service_coverage()

        # DSCR = (1.5M - 600K) / 800K = 1.125
        assert result.dscr == pytest.approx(1.125, rel=0.01)
        assert result.coverage_level == "Weak"
        assert result.meets_requirement is False  # Below 1.20 threshold

    def test_project_bankability_score_basic(self, assessor):
        """Test basic bankability score calculation."""
        result = assessor.project_bankability_score()

        assert isinstance(result, BankabilityScore)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.credit_rating_component <= 100
        assert 0 <= result.risk_component <= 100
        assert 0 <= result.financial_viability_component <= 100
        assert 0 <= result.dscr_component <= 100
        assert result.bankability_level in ["Excellent", "Good", "Fair", "Poor"]
        assert isinstance(result.is_bankable, bool)
        assert 0 <= result.lender_attractiveness <= 100

    def test_project_bankability_score_components(self, assessor):
        """Test that bankability score includes all components."""
        result = assessor.project_bankability_score()

        assert len(result.key_strengths) >= 0
        assert len(result.key_weaknesses) >= 0
        assert len(result.recommendations) > 0
        assert isinstance(result.assessment_date, datetime)

    def test_project_bankability_score_custom_weights(self, assessor):
        """Test bankability score with custom weights."""
        result = assessor.project_bankability_score(
            credit_rating_weight=0.40,
            risk_weight=0.30,
            financial_viability_weight=0.20,
            dscr_weight=0.10,
        )

        # Should complete without error
        assert 0 <= result.overall_score <= 100

    def test_project_bankability_score_invalid_weights(self, assessor):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            assessor.project_bankability_score(
                credit_rating_weight=0.30,
                risk_weight=0.30,
                financial_viability_weight=0.30,
                dscr_weight=0.30,  # Sum = 1.20, not 1.0
            )

    def test_bankability_strong_project(self):
        """Test bankability assessment for strong project."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=4_000_000,
            debt_amount=6_000_000,
            annual_revenue=3_000_000,
            annual_operating_cost=600_000,
            annual_debt_service=600_000,
            project_lifespan_years=25,
            discount_rate=0.07,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)
        result = assessor.project_bankability_score()

        # Strong project should be bankable
        assert result.is_bankable is True
        assert result.overall_score >= 65
        assert result.bankability_level in ["Excellent", "Good"]
        assert len(result.key_strengths) > 0

    def test_bankability_weak_project(self):
        """Test bankability assessment for weak project."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=500_000,
            debt_amount=9_500_000,
            annual_revenue=1_200_000,
            annual_operating_cost=700_000,
            annual_debt_service=1_000_000,
            project_lifespan_years=20,
            discount_rate=0.12,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)
        result = assessor.project_bankability_score()

        # Weak project should not be bankable
        assert result.is_bankable is False
        assert result.overall_score < 60
        assert result.bankability_level in ["Fair", "Poor"]
        assert len(result.key_weaknesses) > 0

    def test_integration_all_assessments(self, assessor):
        """Test integration of all assessment methods."""
        # Run all assessments
        credit_rating = assessor.credit_rating()
        risk_assessment = assessor.risk_assessment()
        dscr_analysis = assessor.debt_service_coverage()
        bankability = assessor.project_bankability_score()

        # Verify all return valid results
        assert isinstance(credit_rating, CreditRatingResult)
        assert isinstance(risk_assessment, RiskAssessmentResult)
        assert isinstance(dscr_analysis, DebtServiceCoverageResult)
        assert isinstance(bankability, BankabilityScore)

        # Verify consistency across assessments
        # Bankability score should reflect individual assessments
        if credit_rating.rating_score >= 70 and dscr_analysis.dscr >= 1.5:
            assert bankability.overall_score >= 60  # Should be bankable


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_zero_debt_scenario(self):
        """Test assessment with zero debt (100% equity)."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=10_000_000,
            debt_amount=0,
            annual_revenue=2_000_000,
            annual_operating_cost=400_000,
            annual_debt_service=0,
            project_lifespan_years=25,
            discount_rate=0.08,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)

        # Should handle zero debt gracefully
        credit_rating = assessor.credit_rating()
        dscr = assessor.debt_service_coverage()

        assert credit_rating.rating_score > 0
        # DSCR will be infinite with zero debt service
        assert dscr.dscr == float("inf") or dscr.dscr > 100

    def test_minimal_project_lifespan(self):
        """Test assessment with minimum project lifespan."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=3_000_000,
            debt_amount=7_000_000,
            annual_revenue=2_000_000,
            annual_operating_cost=400_000,
            annual_debt_service=800_000,
            project_lifespan_years=1,  # Minimum
            discount_rate=0.08,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)

        # Should complete without error
        result = assessor.project_bankability_score()
        assert result.overall_score >= 0

    def test_maximum_project_lifespan(self):
        """Test assessment with maximum project lifespan."""
        metrics = FinancialMetrics(
            total_project_cost=10_000_000,
            equity_contribution=3_000_000,
            debt_amount=7_000_000,
            annual_revenue=2_000_000,
            annual_operating_cost=400_000,
            annual_debt_service=800_000,
            project_lifespan_years=50,  # Maximum
            discount_rate=0.08,
        )
        assessor = BankabilityAssessor(financial_metrics=metrics)

        # Should complete without error
        result = assessor.project_bankability_score()
        assert result.overall_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
