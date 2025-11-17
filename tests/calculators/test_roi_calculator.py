"""
Comprehensive test suite for ROICalculator.

This module provides extensive testing of all ROI calculator functionality
including ROI calculations, NPV, IRR, payback period, and sensitivity analysis.
"""

import pytest
import numpy as np
from typing import List

from src.pv_simulator.calculators.roi_calculator import ROICalculator
from src.pv_simulator.core.models import (
    InvestmentInput,
    CashFlow,
    SensitivityInput,
    ROIResult,
)
from src.pv_simulator.core.enums import CurrencyType, SensitivityParameter
from src.pv_simulator.exceptions import (
    ValidationError,
    CalculationError,
    ConvergenceError,
)


class TestROICalculatorValidation:
    """Test suite for input validation."""

    def test_validate_valid_inputs(self, basic_investment_input):
        """Test validation passes with valid inputs."""
        calculator = ROICalculator()
        assert calculator.validate(basic_investment_input) is True

    def test_validate_negative_initial_investment(self, basic_investment_input):
        """Test validation fails with negative initial investment."""
        basic_investment_input.initial_investment = -100000
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Initial investment must be positive"):
            calculator.validate(basic_investment_input)

    def test_validate_negative_revenue(self, basic_investment_input):
        """Test validation fails with negative revenue."""
        basic_investment_input.annual_revenue = -25000
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Annual revenue cannot be negative"):
            calculator.validate(basic_investment_input)

    def test_validate_negative_costs(self, basic_investment_input):
        """Test validation fails with negative costs."""
        basic_investment_input.annual_costs = -5000
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Annual costs cannot be negative"):
            calculator.validate(basic_investment_input)

    def test_validate_invalid_discount_rate_negative(self, basic_investment_input):
        """Test validation fails with negative discount rate."""
        basic_investment_input.discount_rate = -0.1
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Discount rate must be between 0 and 1"):
            calculator.validate(basic_investment_input)

    def test_validate_invalid_discount_rate_too_high(self, basic_investment_input):
        """Test validation fails with discount rate > 1."""
        basic_investment_input.discount_rate = 1.5
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Discount rate must be between 0 and 1"):
            calculator.validate(basic_investment_input)

    def test_validate_invalid_project_lifetime_too_short(self, basic_investment_input):
        """Test validation fails with project lifetime < 1."""
        basic_investment_input.project_lifetime = 0
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Project lifetime must be between 1 and 50"):
            calculator.validate(basic_investment_input)

    def test_validate_invalid_project_lifetime_too_long(self, basic_investment_input):
        """Test validation fails with project lifetime > 50."""
        basic_investment_input.project_lifetime = 51
        calculator = ROICalculator()
        with pytest.raises(ValidationError, match="Project lifetime must be between 1 and 50"):
            calculator.validate(basic_investment_input)


class TestROICalculation:
    """Test suite for ROI calculation method."""

    def test_roi_calculation_basic(self, basic_investment_input):
        """Test basic ROI calculation without tax or salvage value."""
        calculator = ROICalculator()
        roi = calculator.roi_calculation(basic_investment_input)

        # Expected: (25000-5000)*25 - 100000 = 400000
        # ROI = (400000 / 100000) * 100 = 400%
        assert roi == pytest.approx(400.0, rel=1e-2)

    def test_roi_calculation_with_tax(self):
        """Test ROI calculation with tax considerations."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.10,
            project_lifetime=25,
            tax_rate=0.25,  # 25% tax
        )

        calculator = ROICalculator()
        roi = calculator.roi_calculation(inputs)

        # Net annual = (25000-5000) * (1-0.25) = 15000
        # Total net = 15000 * 25 = 375000
        # Net gain = 375000 - 100000 = 275000
        # ROI = (275000 / 100000) * 100 = 275%
        assert roi == pytest.approx(275.0, rel=1e-2)

    def test_roi_calculation_with_salvage_value(self):
        """Test ROI calculation with salvage value."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.10,
            project_lifetime=25,
            salvage_value=20000,
        )

        calculator = ROICalculator()
        roi = calculator.roi_calculation(inputs)

        # Net revenue = (25000-5000)*25 + 20000 = 520000
        # Net gain = 520000 - 100000 = 420000
        # ROI = (420000 / 100000) * 100 = 420%
        assert roi == pytest.approx(420.0, rel=1e-2)

    def test_roi_calculation_negative_returns(self):
        """Test ROI calculation with negative returns."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=5000,  # Low revenue
            annual_costs=8000,   # High costs
            discount_rate=0.10,
            project_lifetime=10,
        )

        calculator = ROICalculator()
        roi = calculator.roi_calculation(inputs)

        # Net annual = 5000-8000 = -3000
        # Total = -3000 * 10 = -30000
        # Net gain = -30000 - 100000 = -130000
        # ROI = (-130000 / 100000) * 100 = -130%
        assert roi == pytest.approx(-130.0, rel=1e-2)


class TestPaybackPeriod:
    """Test suite for payback period calculations."""

    def test_payback_period_simple(self, simple_cash_flows):
        """Test simple payback period calculation."""
        calculator = ROICalculator()
        payback = calculator.payback_period(simple_cash_flows, discounted=False)

        # Should break even at year 5
        assert payback == pytest.approx(5.0, rel=1e-2)

    def test_payback_period_discounted(self, simple_cash_flows):
        """Test discounted payback period calculation."""
        calculator = ROICalculator()
        payback = calculator.payback_period(simple_cash_flows, discounted=True)

        # Discounted payback should be longer than simple payback if it occurs
        # In this case, with small flows, it may never recover
        if payback is not None:
            assert payback > 5.0

    def test_payback_period_never_recovered(self):
        """Test payback period when investment is never recovered."""
        cash_flows = [
            CashFlow(
                year=0, inflow=0, outflow=100000, net_flow=-100000,
                cumulative_flow=-100000, discounted_flow=-100000
            ),
            CashFlow(
                year=1, inflow=1000, outflow=2000, net_flow=-1000,
                cumulative_flow=-101000, discounted_flow=-909.09
            ),
        ]

        calculator = ROICalculator()
        payback = calculator.payback_period(cash_flows, discounted=False)

        assert payback is None

    def test_payback_period_empty_cash_flows(self):
        """Test payback period with empty cash flows."""
        calculator = ROICalculator()
        with pytest.raises(CalculationError, match="Cash flows list is empty"):
            calculator.payback_period([], discounted=False)

    def test_payback_period_fractional(self):
        """Test payback period with fractional year result."""
        cash_flows = [
            CashFlow(
                year=0, inflow=0, outflow=100000, net_flow=-100000,
                cumulative_flow=-100000, discounted_flow=-100000
            ),
            CashFlow(
                year=1, inflow=30000, outflow=0, net_flow=30000,
                cumulative_flow=-70000, discounted_flow=27272.73
            ),
            CashFlow(
                year=2, inflow=30000, outflow=0, net_flow=30000,
                cumulative_flow=-40000, discounted_flow=24793.39
            ),
            CashFlow(
                year=3, inflow=30000, outflow=0, net_flow=30000,
                cumulative_flow=-10000, discounted_flow=22539.44
            ),
            CashFlow(
                year=4, inflow=30000, outflow=0, net_flow=30000,
                cumulative_flow=20000, discounted_flow=20490.40
            ),
        ]

        calculator = ROICalculator()
        payback = calculator.payback_period(cash_flows, discounted=False)

        # Should be between year 3 and 4
        # Year 3 cumulative: -10000
        # Year 4 net flow: 30000
        # Fraction: 10000 / 30000 = 0.333
        # Payback: 3 + 0.333 = 3.333 years
        assert payback == pytest.approx(3.333, rel=1e-2)


class TestIRRCalculation:
    """Test suite for IRR calculations."""

    def test_irr_calculation_basic(self, simple_cash_flows):
        """Test basic IRR calculation."""
        calculator = ROICalculator()
        irr = calculator.irr_calculation(simple_cash_flows)

        # IRR should be positive for profitable project
        assert irr is not None
        assert irr > 0

    def test_irr_calculation_high_return(self):
        """Test IRR calculation for high return project."""
        cash_flows = [
            CashFlow(
                year=0, inflow=0, outflow=100000, net_flow=-100000,
                cumulative_flow=-100000, discounted_flow=-100000
            ),
            CashFlow(
                year=1, inflow=50000, outflow=0, net_flow=50000,
                cumulative_flow=-50000, discounted_flow=45454.55
            ),
            CashFlow(
                year=2, inflow=80000, outflow=0, net_flow=80000,
                cumulative_flow=30000, discounted_flow=66115.70
            ),
        ]

        calculator = ROICalculator()
        irr = calculator.irr_calculation(cash_flows)

        # High positive IRR expected (around 17-18%)
        assert irr is not None
        assert irr > 15  # At least 15% return

    def test_irr_calculation_negative_return(self):
        """Test IRR calculation for negative return project."""
        cash_flows = [
            CashFlow(
                year=0, inflow=0, outflow=100000, net_flow=-100000,
                cumulative_flow=-100000, discounted_flow=-100000
            ),
            CashFlow(
                year=1, inflow=10000, outflow=0, net_flow=10000,
                cumulative_flow=-90000, discounted_flow=9090.91
            ),
            CashFlow(
                year=2, inflow=10000, outflow=0, net_flow=10000,
                cumulative_flow=-80000, discounted_flow=8264.46
            ),
        ]

        calculator = ROICalculator()

        # Should either return negative IRR or raise ConvergenceError
        try:
            irr = calculator.irr_calculation(cash_flows)
            if irr is not None:
                assert irr < 0  # Negative return
        except ConvergenceError:
            # Acceptable if IRR cannot be calculated
            pass

    def test_irr_calculation_empty_cash_flows(self):
        """Test IRR with empty cash flows."""
        calculator = ROICalculator()
        with pytest.raises(CalculationError, match="Cash flows list is empty"):
            calculator.irr_calculation([])

    def test_irr_calculation_all_zero_flows(self):
        """Test IRR with all zero cash flows."""
        cash_flows = [
            CashFlow(year=0, inflow=0, outflow=0, net_flow=0,
                    cumulative_flow=0, discounted_flow=0),
            CashFlow(year=1, inflow=0, outflow=0, net_flow=0,
                    cumulative_flow=0, discounted_flow=0),
        ]

        calculator = ROICalculator()
        with pytest.raises(CalculationError, match="All cash flows are zero"):
            calculator.irr_calculation(cash_flows)


class TestComprehensiveCalculation:
    """Test suite for comprehensive calculate() method."""

    def test_calculate_basic(self, basic_investment_input):
        """Test comprehensive calculation with basic inputs."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        # Verify result type
        assert isinstance(result, ROIResult)

        # Verify ROI
        assert result.roi_percentage == pytest.approx(400.0, rel=1e-2)

        # Verify NPV is calculated
        assert result.net_present_value is not None

        # Verify IRR is calculated
        assert result.internal_rate_of_return is not None

        # Verify payback period
        assert result.payback_period_years is not None
        assert result.payback_period_years > 0

        # Verify cash flows generated
        assert len(result.cash_flows) == basic_investment_input.project_lifetime + 1

    def test_calculate_complex(self, complex_investment_input):
        """Test comprehensive calculation with complex inputs."""
        calculator = ROICalculator()
        result = calculator.calculate(complex_investment_input)

        # Verify result is profitable
        assert result.is_profitable

        # Verify all metrics are calculated
        assert result.roi_percentage is not None
        assert result.net_present_value is not None
        assert result.internal_rate_of_return is not None
        assert result.payback_period_years is not None

        # Verify currency is preserved
        assert result.currency == complex_investment_input.currency

    def test_calculate_profitability_metrics(self, basic_investment_input):
        """Test profitability index and annual ROI calculations."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        # Verify profitability index
        assert result.profitability_index > 0

        # Verify annual ROI
        assert result.annual_roi == pytest.approx(
            result.roi_percentage / basic_investment_input.project_lifetime,
            rel=1e-2
        )

    def test_calculate_total_metrics(self, basic_investment_input):
        """Test total revenue, costs, and profit calculations."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        # Verify totals
        expected_total_revenue = (
            basic_investment_input.annual_revenue * basic_investment_input.project_lifetime
        )
        expected_total_costs = (
            basic_investment_input.annual_costs * basic_investment_input.project_lifetime
        )

        assert result.total_revenue == pytest.approx(expected_total_revenue, rel=1e-2)
        assert result.total_costs == pytest.approx(expected_total_costs, rel=1e-2)

    def test_calculate_meets_hurdle_rate(self, basic_investment_input):
        """Test hurdle rate evaluation."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        # For profitable project, IRR should exceed discount rate
        if result.internal_rate_of_return is not None:
            if result.is_profitable:
                assert result.meets_hurdle_rate


class TestSensitivityAnalysis:
    """Test suite for sensitivity analysis."""

    def test_sensitivity_analysis_discount_rate(
        self, basic_investment_input, sensitivity_input_discount_rate
    ):
        """Test sensitivity analysis for discount rate."""
        calculator = ROICalculator()
        results = calculator.sensitivity_analysis(
            basic_investment_input, [sensitivity_input_discount_rate]
        )

        assert len(results) == 1
        sens_result = results[0]

        # Verify parameter
        assert sens_result.parameter == SensitivityParameter.DISCOUNT_RATE

        # Verify base case is calculated
        assert sens_result.base_case is not None

        # Verify results for each variation
        assert len(sens_result.results) == 5  # 5 variations

        # Verify ranges
        assert len(sens_result.roi_range) == 2
        assert len(sens_result.npv_range) == 2

        # Verify NPV decreases as discount rate increases
        npv_values = [r.net_present_value for r in sens_result.results]
        # Generally, higher discount rate = lower NPV
        assert npv_values[0] > npv_values[-1]

    def test_sensitivity_analysis_multiple_parameters(
        self, basic_investment_input, sensitivity_input_discount_rate, sensitivity_input_revenue
    ):
        """Test sensitivity analysis for multiple parameters."""
        calculator = ROICalculator()
        results = calculator.sensitivity_analysis(
            basic_investment_input,
            [sensitivity_input_discount_rate, sensitivity_input_revenue]
        )

        assert len(results) == 2

        # Verify both parameters analyzed
        params = [r.parameter for r in results]
        assert SensitivityParameter.DISCOUNT_RATE in params
        assert SensitivityParameter.ANNUAL_REVENUE in params

    def test_sensitivity_analysis_revenue_variation(
        self, basic_investment_input, sensitivity_input_revenue
    ):
        """Test sensitivity to revenue changes."""
        calculator = ROICalculator()
        results = calculator.sensitivity_analysis(
            basic_investment_input, [sensitivity_input_revenue]
        )

        sens_result = results[0]

        # ROI should increase with revenue
        roi_values = [r.roi_percentage for r in sens_result.results]
        # First variation is -30%, last is +30%
        assert roi_values[-1] > roi_values[0]

    def test_sensitivity_analysis_volatility(
        self, basic_investment_input, sensitivity_input_discount_rate
    ):
        """Test volatility calculations in sensitivity analysis."""
        calculator = ROICalculator()
        results = calculator.sensitivity_analysis(
            basic_investment_input, [sensitivity_input_discount_rate]
        )

        sens_result = results[0]

        # Verify volatility properties
        assert sens_result.roi_volatility >= 0
        assert sens_result.npv_volatility >= 0

    def test_sensitivity_analysis_with_explicit_values(self, basic_investment_input):
        """Test sensitivity analysis with explicit test values."""
        sensitivity = SensitivityInput(
            parameter=SensitivityParameter.DISCOUNT_RATE,
            base_value=0.10,
            variation_values=[0.05, 0.08, 0.10, 0.12, 0.15],
        )

        calculator = ROICalculator()
        results = calculator.sensitivity_analysis(basic_investment_input, [sensitivity])

        assert len(results) == 1
        assert len(results[0].results) == 5


class TestCashFlowGeneration:
    """Test suite for cash flow generation."""

    def test_cash_flow_count(self, basic_investment_input):
        """Test correct number of cash flows generated."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        # Should have N+1 cash flows (year 0 + N years)
        expected_count = basic_investment_input.project_lifetime + 1
        assert len(result.cash_flows) == expected_count

    def test_cash_flow_year_zero(self, basic_investment_input):
        """Test year 0 cash flow is initial investment."""
        calculator = ROICalculator()
        result = calculator.calculate(basic_investment_input)

        year_zero = result.cash_flows[0]
        assert year_zero.year == 0
        assert year_zero.outflow == basic_investment_input.initial_investment
        assert year_zero.net_flow == -basic_investment_input.initial_investment

    def test_cash_flow_with_inflation(self):
        """Test cash flows with inflation applied."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.10,
            project_lifetime=5,
            inflation_rate=0.03,  # 3% inflation
        )

        calculator = ROICalculator()
        result = calculator.calculate(inputs)

        # Verify revenue increases with inflation
        year1_revenue = result.cash_flows[1].inflow
        year5_revenue = result.cash_flows[5].inflow

        # Year 5 revenue should be higher due to inflation
        assert year5_revenue > year1_revenue

    def test_cash_flow_salvage_value(self):
        """Test salvage value appears in final year."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.10,
            project_lifetime=5,
            salvage_value=15000,
        )

        calculator = ROICalculator()
        result = calculator.calculate(inputs)

        # Final year should include salvage value
        final_year = result.cash_flows[-1]
        assert final_year.inflow > inputs.annual_revenue


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_very_short_project(self):
        """Test calculation with 1-year project."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=150000,
            annual_costs=10000,
            discount_rate=0.10,
            project_lifetime=1,
        )

        calculator = ROICalculator()
        result = calculator.calculate(inputs)

        assert result is not None
        assert result.payback_period_years is not None

    def test_zero_discount_rate(self):
        """Test calculation with zero discount rate."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.0,  # Zero discount rate
            project_lifetime=10,
        )

        calculator = ROICalculator()
        result = calculator.calculate(inputs)

        # NPV should equal simple sum of cash flows
        assert result.net_present_value is not None

    def test_very_high_discount_rate(self):
        """Test calculation with high discount rate."""
        inputs = InvestmentInput(
            initial_investment=100000,
            annual_revenue=25000,
            annual_costs=5000,
            discount_rate=0.30,  # 30% discount rate
            project_lifetime=25,
        )

        calculator = ROICalculator()
        result = calculator.calculate(inputs)

        # High discount rate should reduce NPV significantly
        assert result.net_present_value < result.total_revenue - result.total_costs

    def test_calculator_run_method(self, basic_investment_input):
        """Test calculator run() convenience method."""
        calculator = ROICalculator()
        result = calculator.run(basic_investment_input, validate_first=True)

        assert isinstance(result, ROIResult)
        assert result.roi_percentage is not None

    def test_calculator_run_without_validation(self, basic_investment_input):
        """Test calculator run() without validation."""
        calculator = ROICalculator()
        result = calculator.run(basic_investment_input, validate_first=False)

        assert isinstance(result, ROIResult)


class TestCalculatorConfiguration:
    """Test suite for calculator configuration."""

    def test_custom_irr_iterations(self, basic_investment_input):
        """Test calculator with custom IRR iterations."""
        calculator = ROICalculator(max_irr_iterations=200)
        assert calculator.max_irr_iterations == 200

        result = calculator.calculate(basic_investment_input)
        assert result is not None

    def test_custom_irr_tolerance(self, basic_investment_input):
        """Test calculator with custom IRR tolerance."""
        calculator = ROICalculator(irr_tolerance=1e-8)
        assert calculator.irr_tolerance == 1e-8

        result = calculator.calculate(basic_investment_input)
        assert result is not None

    def test_calculator_name(self):
        """Test calculator name configuration."""
        calculator = ROICalculator(name="TestCalculator")
        assert calculator.name == "TestCalculator"
        assert "TestCalculator" in str(calculator)

    def test_calculator_repr(self):
        """Test calculator string representation."""
        calculator = ROICalculator(name="MyCalculator")
        repr_str = repr(calculator)
        assert "ROICalculator" in repr_str
        assert "MyCalculator" in repr_str


@pytest.mark.parametrize(
    "initial_investment,annual_revenue,annual_costs,expected_roi",
    [
        (100000, 30000, 5000, 525.0),  # High profit
        (100000, 15000, 5000, 150.0),  # Medium profit
        (100000, 8000, 5000, -25.0),   # Small loss (8000-5000)*25-100000 = -25000
        (100000, 5000, 8000, -175.0),  # Larger loss (5000-8000)*25-100000 = -175000
    ],
)
def test_roi_parametrized(initial_investment, annual_revenue, annual_costs, expected_roi):
    """Parametrized test for various ROI scenarios."""
    inputs = InvestmentInput(
        initial_investment=initial_investment,
        annual_revenue=annual_revenue,
        annual_costs=annual_costs,
        discount_rate=0.10,
        project_lifetime=25,
    )

    calculator = ROICalculator()
    roi = calculator.roi_calculation(inputs)

    assert roi == pytest.approx(expected_roi, rel=1e-2)
