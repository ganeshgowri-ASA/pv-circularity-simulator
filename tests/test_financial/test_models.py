"""Tests for Pydantic models in financial module."""

import pytest
from pydantic import ValidationError

from pv_simulator.financial.models import (
    CashFlowInput,
    CashFlowProjection,
    DiscountRateConfig,
    FinancialMetrics,
    SensitivityAnalysisResult,
    SensitivityDataPoint,
)


class TestCashFlowInput:
    """Test CashFlowInput model."""

    def test_valid_cash_flow_input(self) -> None:
        """Test creating valid cash flow input."""
        cash_flow = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[20000.0, 25000.0, 30000.0],
            discount_rate=0.10,
            project_name="Test Project"
        )
        assert cash_flow.initial_investment == 100000.0
        assert len(cash_flow.cash_flows) == 3
        assert cash_flow.discount_rate == 0.10
        assert cash_flow.periods == [0, 1, 2]

    def test_default_periods(self) -> None:
        """Test that periods are auto-generated if not provided."""
        cash_flow = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[10000.0, 15000.0, 20000.0, 25000.0],
            discount_rate=0.08
        )
        assert cash_flow.periods == [0, 1, 2, 3]

    def test_custom_periods(self) -> None:
        """Test cash flow with custom periods."""
        cash_flow = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[10000.0, 15000.0, 20000.0],
            discount_rate=0.08,
            periods=[1, 2, 3]
        )
        assert cash_flow.periods == [1, 2, 3]

    def test_invalid_discount_rate_too_high(self) -> None:
        """Test that discount rate > 1 raises error."""
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[10000.0, 15000.0],
                discount_rate=1.5
            )

    def test_invalid_discount_rate_zero(self) -> None:
        """Test that zero discount rate raises error."""
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[10000.0, 15000.0],
                discount_rate=0.0
            )

    def test_invalid_initial_investment(self) -> None:
        """Test that negative initial investment raises error."""
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=-100000.0,
                cash_flows=[10000.0, 15000.0],
                discount_rate=0.10
            )

    def test_empty_cash_flows(self) -> None:
        """Test that empty cash flows raise error."""
        with pytest.raises(ValidationError):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[],
                discount_rate=0.10
            )

    def test_all_zero_cash_flows(self) -> None:
        """Test that all zero cash flows raise error."""
        with pytest.raises(ValidationError, match="Cash flows cannot all be zero"):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[0.0, 0.0, 0.0],
                discount_rate=0.10
            )

    def test_mismatched_periods_length(self) -> None:
        """Test that mismatched periods and cash flows raise error."""
        with pytest.raises(ValidationError, match="Number of periods.*must match"):
            CashFlowInput(
                initial_investment=100000.0,
                cash_flows=[10000.0, 15000.0, 20000.0],
                discount_rate=0.10,
                periods=[1, 2]  # Only 2 periods for 3 cash flows
            )

    def test_to_numpy(self) -> None:
        """Test conversion to numpy arrays."""
        cash_flow = CashFlowInput(
            initial_investment=100000.0,
            cash_flows=[10000.0, 15000.0, 20000.0],
            discount_rate=0.10
        )
        periods_array, cash_flows_array = cash_flow.to_numpy()
        assert len(periods_array) == 3
        assert len(cash_flows_array) == 3


class TestDiscountRateConfig:
    """Test DiscountRateConfig model."""

    def test_valid_config(self) -> None:
        """Test creating valid discount rate config."""
        config = DiscountRateConfig(
            base_rate=0.10,
            min_rate=0.05,
            max_rate=0.20,
            step_size=0.01
        )
        assert config.base_rate == 0.10
        assert config.min_rate == 0.05
        assert config.max_rate == 0.20
        assert config.step_size == 0.01

    def test_invalid_rate_order(self) -> None:
        """Test that invalid rate order raises error."""
        with pytest.raises(ValidationError, match="min_rate.*base_rate.*max_rate"):
            DiscountRateConfig(
                base_rate=0.25,  # Base > max
                min_rate=0.05,
                max_rate=0.20,
                step_size=0.01
            )

    def test_base_rate_below_min(self) -> None:
        """Test that base_rate < min_rate raises error."""
        with pytest.raises(ValidationError, match="min_rate.*base_rate.*max_rate"):
            DiscountRateConfig(
                base_rate=0.03,
                min_rate=0.05,
                max_rate=0.20,
                step_size=0.01
            )


class TestCashFlowProjection:
    """Test CashFlowProjection model."""

    def test_valid_projection(self) -> None:
        """Test creating valid cash flow projection."""
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[50000.0, 55000.0, 60000.0],
            operating_costs=[20000.0, 22000.0, 24000.0],
            capital_expenditures=[5000.0, 0.0, 0.0],
            initial_investment=100000.0
        )
        assert len(projection.periods) == 3
        assert len(projection.net_cash_flows) == 3
        # Net = Revenue - OpCosts - CapEx
        assert projection.net_cash_flows[0] == 50000.0 - 20000.0 - 5000.0

    def test_auto_compute_net_cash_flows(self) -> None:
        """Test that net cash flows are auto-computed."""
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[100000.0, 110000.0, 120000.0],
            operating_costs=[50000.0, 52000.0, 54000.0]
        )
        # Should auto-fill capex with zeros and compute net
        assert len(projection.capital_expenditures) == 3
        assert all(capex == 0.0 for capex in projection.capital_expenditures)
        assert projection.net_cash_flows[0] == 100000.0 - 50000.0

    def test_projection_with_terminal_value(self) -> None:
        """Test projection with terminal value."""
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[50000.0, 55000.0, 60000.0],
            operating_costs=[20000.0, 22000.0, 24000.0],
            initial_investment=100000.0,
            terminal_value=80000.0
        )
        assert projection.terminal_value == 80000.0

    def test_to_cash_flow_input(self) -> None:
        """Test conversion to CashFlowInput."""
        projection = CashFlowProjection(
            periods=[1, 2, 3],
            revenues=[50000.0, 55000.0, 60000.0],
            operating_costs=[20000.0, 22000.0, 24000.0],
            initial_investment=100000.0,
            terminal_value=50000.0
        )
        cash_flow_input = projection.to_cash_flow_input(
            discount_rate=0.10,
            project_name="Test Project"
        )
        assert cash_flow_input.discount_rate == 0.10
        assert cash_flow_input.project_name == "Test Project"
        assert cash_flow_input.initial_investment == 100000.0
        # Last cash flow should include terminal value
        assert cash_flow_input.cash_flows[-1] == projection.net_cash_flows[-1] + 50000.0

    def test_mismatched_lengths(self) -> None:
        """Test that mismatched array lengths raise error."""
        with pytest.raises(ValidationError, match="length must match"):
            CashFlowProjection(
                periods=[1, 2, 3],
                revenues=[50000.0, 55000.0],  # Only 2 values
                operating_costs=[20000.0, 22000.0, 24000.0]
            )


class TestFinancialMetrics:
    """Test FinancialMetrics model."""

    def test_valid_metrics(self) -> None:
        """Test creating valid financial metrics."""
        metrics = FinancialMetrics(
            analysis_type="NPV_IRR",
            npv=50000.0,
            irr=0.18,
            discount_rate=0.10,
            payback_period=3.5,
            discounted_payback_period=4.2,
            profitability_index=1.5,
            total_cash_flows=200000.0,
            project_name="Test Project"
        )
        assert metrics.npv == 50000.0
        assert metrics.irr == 0.18
        assert metrics.discount_rate == 0.10

    def test_metrics_with_none_values(self) -> None:
        """Test metrics with optional None values."""
        metrics = FinancialMetrics(
            analysis_type="NPV_IRR",
            npv=50000.0,
            irr=None,  # IRR may not converge
            discount_rate=0.10,
            total_cash_flows=200000.0
        )
        assert metrics.irr is None
        assert metrics.payback_period is None


class TestSensitivityAnalysis:
    """Test sensitivity analysis models."""

    def test_sensitivity_data_point(self) -> None:
        """Test creating sensitivity data point."""
        data_point = SensitivityDataPoint(
            parameter_value=0.10,
            npv=50000.0,
            irr=0.18
        )
        assert data_point.parameter_value == 0.10
        assert data_point.npv == 50000.0
        assert data_point.irr == 0.18

    def test_sensitivity_analysis_result(self) -> None:
        """Test creating sensitivity analysis result."""
        data_points = [
            SensitivityDataPoint(parameter_value=0.05, npv=70000.0, irr=0.18),
            SensitivityDataPoint(parameter_value=0.10, npv=50000.0, irr=0.18),
            SensitivityDataPoint(parameter_value=0.15, npv=35000.0, irr=0.18),
        ]
        result = SensitivityAnalysisResult(
            analysis_type="Sensitivity",
            parameter_name="discount_rate",
            base_value=0.10,
            data_points=data_points,
            base_npv=50000.0,
            base_irr=0.18,
            min_npv=35000.0,
            max_npv=70000.0
        )
        assert result.parameter_name == "discount_rate"
        assert len(result.data_points) == 3
        assert result.min_npv == 35000.0
        assert result.max_npv == 70000.0
