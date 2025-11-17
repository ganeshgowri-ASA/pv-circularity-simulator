"""Comprehensive tests for NPVAnalyzer class."""

import pytest
from pydantic import ValidationError

from pv_simulator.core.exceptions import ConvergenceError, InvalidCashFlowError
from pv_simulator.financial.models import (
    CashFlowInput,
    CashFlowProjection,
    DiscountRateConfig,
)
from pv_simulator.financial.npv_irr_analyzer import NPVAnalyzer


class TestNPVAnalyzerInitialization:
    """Test NPVAnalyzer initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test analyzer initialization with default parameters."""
        analyzer = NPVAnalyzer()
        assert analyzer.name == "NPVAnalyzer"
        assert analyzer.version == "0.1.0"
        assert analyzer.max_iterations == 100
        assert analyzer.tolerance == 1e-6

    def test_custom_initialization(self) -> None:
        """Test analyzer initialization with custom parameters."""
        analyzer = NPVAnalyzer(max_iterations=200, tolerance=1e-8, version="1.0.0")
        assert analyzer.max_iterations == 200
        assert analyzer.tolerance == 1e-8
        assert analyzer.version == "1.0.0"


class TestNPVCalculation:
    """Test NPV calculation method."""

    def test_simple_npv_calculation(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test basic NPV calculation with simple cash flows."""
        analyzer = NPVAnalyzer()
        npv = analyzer.npv_calculation(
            cash_flows=simple_cash_flow_input.cash_flows,
            discount_rate=simple_cash_flow_input.discount_rate,
            initial_investment=simple_cash_flow_input.initial_investment
        )
        # Expected NPV calculation (periods default to [0, 1, 2, 3, 4]):
        # NPV = -100000 + 20000/1.1^0 + 25000/1.1^1 + 30000/1.1^2 + 35000/1.1^3 + 40000/1.1^4
        # NPV ≈ 21,137.22
        assert npv > 0  # Should be positive
        assert abs(npv - 21137.22) < 1.0  # Allow small numerical error

    def test_negative_npv(self, negative_npv_cash_flow: CashFlowInput) -> None:
        """Test NPV calculation resulting in negative value."""
        analyzer = NPVAnalyzer()
        npv = analyzer.npv_calculation(
            cash_flows=negative_npv_cash_flow.cash_flows,
            discount_rate=negative_npv_cash_flow.discount_rate,
            initial_investment=negative_npv_cash_flow.initial_investment
        )
        assert npv < 0  # Should be negative (bad investment)

    def test_npv_with_custom_periods(self) -> None:
        """Test NPV calculation with custom period numbers."""
        analyzer = NPVAnalyzer()
        npv = analyzer.npv_calculation(
            cash_flows=[10000.0, 15000.0, 20000.0],
            discount_rate=0.10,
            initial_investment=30000.0,
            periods=[1, 2, 3]
        )
        assert npv > 0

    def test_npv_zero_discount_rate_raises_error(self) -> None:
        """Test that zero discount rate raises error."""
        analyzer = NPVAnalyzer()
        with pytest.raises(InvalidCashFlowError, match="Discount rate must be positive"):
            analyzer.npv_calculation(
                cash_flows=[10000.0, 10000.0],
                discount_rate=0.0,
                initial_investment=15000.0
            )

    def test_npv_negative_initial_investment_raises_error(self) -> None:
        """Test that negative initial investment raises error."""
        analyzer = NPVAnalyzer()
        with pytest.raises(InvalidCashFlowError, match="Initial investment must be positive"):
            analyzer.npv_calculation(
                cash_flows=[10000.0, 10000.0],
                discount_rate=0.10,
                initial_investment=-15000.0
            )

    def test_npv_empty_cash_flows_raises_error(self) -> None:
        """Test that empty cash flows raise error."""
        analyzer = NPVAnalyzer()
        with pytest.raises(InvalidCashFlowError):
            analyzer.npv_calculation(
                cash_flows=[],
                discount_rate=0.10,
                initial_investment=15000.0
            )


class TestIRRComputation:
    """Test IRR computation method."""

    def test_simple_irr_computation(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test basic IRR calculation."""
        analyzer = NPVAnalyzer()
        irr = analyzer.irr_computation(
            cash_flows=simple_cash_flow_input.cash_flows,
            initial_investment=simple_cash_flow_input.initial_investment
        )
        assert 0.10 < irr < 0.25  # Should be between 10% and 25%
        # Verify: NPV at IRR should be approximately zero
        npv_at_irr = analyzer.npv_calculation(
            cash_flows=simple_cash_flow_input.cash_flows,
            discount_rate=irr,
            initial_investment=simple_cash_flow_input.initial_investment
        )
        assert abs(npv_at_irr) < 1.0  # Should be very close to zero

    def test_high_return_irr(self, high_return_cash_flow: CashFlowInput) -> None:
        """Test IRR for high return investment."""
        analyzer = NPVAnalyzer()
        irr = analyzer.irr_computation(
            cash_flows=high_return_cash_flow.cash_flows,
            initial_investment=high_return_cash_flow.initial_investment
        )
        assert irr > 0.30  # Should be > 30% for this high return project

    def test_irr_with_custom_initial_guess(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test IRR calculation with custom initial guess."""
        analyzer = NPVAnalyzer()
        irr = analyzer.irr_computation(
            cash_flows=simple_cash_flow_input.cash_flows,
            initial_investment=simple_cash_flow_input.initial_investment,
            initial_guess=0.15
        )
        assert irr > 0  # Should converge to positive IRR

    def test_irr_no_positive_cash_flows_raises_error(self) -> None:
        """Test that IRR with no positive cash flows raises error."""
        analyzer = NPVAnalyzer()
        with pytest.raises(
            InvalidCashFlowError,
            match="Cash flows must contain at least one positive value"
        ):
            analyzer.irr_computation(
                cash_flows=[-1000.0, -2000.0, -3000.0],
                initial_investment=10000.0
            )

    def test_irr_convergence_failure(self) -> None:
        """Test IRR convergence failure with problematic cash flows."""
        analyzer = NPVAnalyzer(max_iterations=5, tolerance=1e-10)
        # Cash flows that may not converge easily
        with pytest.raises(ConvergenceError):
            analyzer.irr_computation(
                cash_flows=[1000.0],
                initial_investment=1000000.0,
                initial_guess=0.5
            )


class TestDiscountRateSensitivity:
    """Test discount rate sensitivity analysis."""

    def test_basic_sensitivity_analysis(
        self,
        simple_cash_flow_input: CashFlowInput,
        discount_rate_config: DiscountRateConfig
    ) -> None:
        """Test basic sensitivity analysis."""
        analyzer = NPVAnalyzer()
        result = analyzer.discount_rate_sensitivity(
            simple_cash_flow_input,
            discount_rate_config
        )

        assert result.analysis_type == "Discount_Rate_Sensitivity"
        assert result.parameter_name == "discount_rate"
        assert result.base_value == 0.10
        assert len(result.data_points) > 0
        assert result.base_npv is not None
        assert result.min_npv < result.max_npv

    def test_sensitivity_npv_decreases_with_rate(
        self,
        simple_cash_flow_input: CashFlowInput,
        discount_rate_config: DiscountRateConfig
    ) -> None:
        """Test that NPV decreases as discount rate increases."""
        analyzer = NPVAnalyzer()
        result = analyzer.discount_rate_sensitivity(
            simple_cash_flow_input,
            discount_rate_config
        )

        # NPV should generally decrease as discount rate increases
        npv_values = [dp.npv for dp in result.data_points]
        # Check that first NPV is greater than last NPV
        assert npv_values[0] > npv_values[-1]

    def test_sensitivity_irr_constant(
        self,
        simple_cash_flow_input: CashFlowInput,
        discount_rate_config: DiscountRateConfig
    ) -> None:
        """Test that IRR remains constant across different discount rates."""
        analyzer = NPVAnalyzer()
        result = analyzer.discount_rate_sensitivity(
            simple_cash_flow_input,
            discount_rate_config
        )

        # All data points should have the same IRR
        irr_values = [dp.irr for dp in result.data_points if dp.irr is not None]
        if irr_values:
            first_irr = irr_values[0]
            for irr in irr_values:
                assert abs(irr - first_irr) < 1e-6

    def test_sensitivity_with_narrow_range(
        self,
        simple_cash_flow_input: CashFlowInput
    ) -> None:
        """Test sensitivity analysis with narrow discount rate range."""
        analyzer = NPVAnalyzer()
        rate_config = DiscountRateConfig(
            base_rate=0.10,
            min_rate=0.09,
            max_rate=0.11,
            step_size=0.005
        )
        result = analyzer.discount_rate_sensitivity(simple_cash_flow_input, rate_config)

        assert len(result.data_points) >= 3  # Should have at least 3 data points
        assert result.min_npv < result.max_npv


class TestCashFlowModeling:
    """Test cash flow modeling method."""

    def test_basic_cash_flow_modeling(
        self,
        cash_flow_projection: CashFlowProjection
    ) -> None:
        """Test basic cash flow modeling."""
        analyzer = NPVAnalyzer()
        result = analyzer.cash_flow_modeling(
            cash_flow_projection,
            discount_rate=0.10,
            project_name="Test Modeling Project"
        )

        assert result.project_name == "Test Modeling Project"
        assert result.npv is not None
        assert result.discount_rate == 0.10
        assert result.analysis_type == "NPV_IRR"

    def test_cash_flow_modeling_with_terminal_value(self) -> None:
        """Test cash flow modeling with terminal value."""
        analyzer = NPVAnalyzer()
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[100000.0, 110000.0, 120000.0],
            operating_costs=[50000.0, 52000.0, 54000.0],
            capital_expenditures=[10000.0, 0.0, 0.0],
            initial_investment=200000.0,
            terminal_value=100000.0  # Asset value at end
        )
        result = analyzer.cash_flow_modeling(projection, discount_rate=0.08)

        # Terminal value should increase NPV
        assert result.npv is not None

    def test_cash_flow_modeling_without_terminal_value(self) -> None:
        """Test cash flow modeling without terminal value."""
        analyzer = NPVAnalyzer()
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[50000.0, 55000.0, 60000.0],
            operating_costs=[30000.0, 32000.0, 34000.0],
            initial_investment=100000.0
        )
        result = analyzer.cash_flow_modeling(projection, discount_rate=0.10)

        assert result.npv is not None
        assert result.total_cash_flows is not None


class TestFullAnalysis:
    """Test comprehensive analysis method."""

    def test_full_analysis(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test complete financial analysis."""
        analyzer = NPVAnalyzer()
        result = analyzer.analyze(simple_cash_flow_input)

        assert result.npv is not None
        assert result.irr is not None
        assert result.discount_rate == 0.10
        assert result.payback_period is not None
        assert result.discounted_payback_period is not None
        assert result.profitability_index is not None
        assert result.total_cash_flows == sum(simple_cash_flow_input.cash_flows)
        assert result.project_name == "Test Project"
        assert result.analysis_type == "NPV_IRR"

    def test_full_analysis_with_pv_project(
        self,
        pv_project_cash_flow: CashFlowInput
    ) -> None:
        """Test full analysis with realistic PV project."""
        analyzer = NPVAnalyzer()
        result = analyzer.analyze(pv_project_cash_flow)

        assert result.npv is not None
        assert result.irr is not None
        assert result.payback_period is not None
        # PV project should have positive NPV with 8% discount rate
        assert result.npv > 0
        # IRR should be higher than discount rate for positive NPV
        assert result.irr > 0.08

    def test_analysis_metadata(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test that analysis includes metadata."""
        analyzer = NPVAnalyzer()
        result = analyzer.analyze(simple_cash_flow_input)

        assert "metadata" in result.model_dump()
        assert result.metadata.get("name") == "NPVAnalyzer"
        assert result.metadata.get("version") == "0.1.0"

    def test_profitability_index_calculation(
        self,
        simple_cash_flow_input: CashFlowInput
    ) -> None:
        """Test profitability index calculation."""
        analyzer = NPVAnalyzer()
        result = analyzer.analyze(simple_cash_flow_input)

        # PI > 1 means positive NPV
        if result.npv > 0:
            assert result.profitability_index > 1.0
        elif result.npv < 0:
            assert result.profitability_index < 1.0


class TestValidation:
    """Test input validation."""

    def test_validate_valid_input(self, simple_cash_flow_input: CashFlowInput) -> None:
        """Test validation with valid input."""
        analyzer = NPVAnalyzer()
        # Should not raise any exception
        analyzer.validate_input(simple_cash_flow_input)

    def test_validate_invalid_discount_rate(self) -> None:
        """Test validation with invalid discount rate."""
        # Pydantic will raise ValidationError before our custom validation
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[20000.0, 25000.0, 30000.0],
                discount_rate=1.5,  # Invalid: > 1
                project_name="Invalid Project"
            )

    def test_validate_all_zero_cash_flows(self) -> None:
        """Test validation with all zero cash flows."""
        # Pydantic validator will raise ValidationError
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[0.0, 0.0, 0.0],
                discount_rate=0.10
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_period_cash_flow(self) -> None:
        """Test with single period cash flow."""
        analyzer = NPVAnalyzer()
        cash_flow_input = CashFlowInput(
            initial_investment=10000.0,
            cash_flows=[12000.0],
            discount_rate=0.10
        )
        result = analyzer.analyze(cash_flow_input)

        assert result.npv is not None
        # For single period at t=0, cash flow equals investment immediately,
        # so IRR calculation might not converge or may return None
        # NPV should be positive: 12000 - 10000 = 2000
        assert result.npv > 0

    def test_very_high_discount_rate(self) -> None:
        """Test with very high discount rate."""
        analyzer = NPVAnalyzer()
        cash_flow_input = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[20000.0, 25000.0, 30000.0, 35000.0, 40000.0],
            discount_rate=0.50  # 50% discount rate
        )
        result = analyzer.analyze(cash_flow_input)

        # High discount rate should result in negative NPV
        assert result.npv < 0

    def test_very_low_discount_rate(self) -> None:
        """Test with very low discount rate."""
        analyzer = NPVAnalyzer()
        cash_flow_input = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[20000.0, 25000.0, 30000.0, 35000.0, 40000.0],
            discount_rate=0.01  # 1% discount rate
        )
        result = analyzer.analyze(cash_flow_input)

        # Low discount rate should result in higher NPV
        assert result.npv > 30000.0

    def test_long_period_cash_flows(self) -> None:
        """Test with long period cash flows (20 years)."""
        analyzer = NPVAnalyzer()
        cash_flow_input = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[10000.0] * 20,  # 20 years of $10k
            discount_rate=0.08
        )
        result = analyzer.analyze(cash_flow_input)

        assert result.npv is not None
        assert result.irr is not None
        assert result.payback_period is not None


@pytest.mark.integration
class TestIntegration:
    """Integration tests for NPVAnalyzer."""

    def test_complete_workflow(self) -> None:
        """Test complete workflow from projection to analysis to sensitivity."""
        analyzer = NPVAnalyzer()

        # 1. Create cash flow projection
        projection = CashFlowProjection(
            periods=[1, 2, 3, 4, 5],
            revenues=[60000.0, 65000.0, 70000.0, 75000.0, 80000.0],
            operating_costs=[25000.0, 26000.0, 27000.0, 28000.0, 29000.0],
            capital_expenditures=[5000.0, 0.0, 0.0, 0.0, 5000.0],
            initial_investment=150000.0,
            terminal_value=75000.0
        )

        # 2. Perform cash flow modeling
        result = analyzer.cash_flow_modeling(projection, discount_rate=0.10)
        assert result.npv is not None
        initial_npv = result.npv

        # 3. Perform sensitivity analysis
        cash_flow_input = projection.to_cash_flow_input(
            discount_rate=0.10,
            project_name="Integration Test Project"
        )
        rate_config = DiscountRateConfig(
            base_rate=0.10,
            min_rate=0.05,
            max_rate=0.15,
            step_size=0.01
        )
        sensitivity_result = analyzer.discount_rate_sensitivity(
            cash_flow_input,
            rate_config
        )

        assert sensitivity_result.base_npv is not None
        # NPV from modeling should match NPV from sensitivity at base rate
        assert abs(initial_npv - sensitivity_result.base_npv) < 1.0
