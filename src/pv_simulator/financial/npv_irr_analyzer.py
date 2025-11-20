"""NPV and IRR Analysis Engine for financial evaluation of PV projects.

This module provides the NPVAnalyzer class for comprehensive financial analysis
including Net Present Value (NPV), Internal Rate of Return (IRR), sensitivity
analysis, and cash flow modeling.
"""

from typing import Optional

import numpy as np
from scipy.optimize import newton

from pv_simulator.core.base import BaseAnalyzer
from pv_simulator.core.exceptions import (
    ConvergenceError,
    FinancialAnalysisError,
    InvalidCashFlowError,
)
from pv_simulator.financial.cash_flow import (
    calculate_discounted_payback_period,
    calculate_payback_period,
    calculate_present_value,
    calculate_profitability_index,
    validate_cash_flow_arrays,
)
from pv_simulator.financial.models import (
    CashFlowInput,
    CashFlowProjection,
    DiscountRateConfig,
    FinancialMetrics,
    SensitivityAnalysisResult,
    SensitivityDataPoint,
)


class NPVAnalyzer(BaseAnalyzer[CashFlowInput, FinancialMetrics]):
    """NPV and IRR Analysis Engine for financial evaluation.

    This class provides comprehensive financial analysis capabilities including:
    - Net Present Value (NPV) calculation
    - Internal Rate of Return (IRR) computation
    - Discount rate sensitivity analysis
    - Cash flow modeling and projection

    The analyzer uses time value of money (TVM) principles to evaluate
    investment opportunities and supports various sensitivity analyses.

    Attributes:
        name: Name of the analyzer ("NPVAnalyzer").
        version: Version of the analyzer implementation.
        max_iterations: Maximum iterations for IRR convergence (default: 100).
        tolerance: Convergence tolerance for IRR calculation (default: 1e-6).

    Example:
        >>> analyzer = NPVAnalyzer()
        >>> cash_flow_input = CashFlowInput(
        ...     initial_investment=100000,
        ...     cash_flows=[20000, 25000, 30000, 35000, 40000],
        ...     discount_rate=0.10
        ... )
        >>> result = analyzer.analyze(cash_flow_input)
        >>> print(f"NPV: ${result.npv:,.2f}")
        >>> print(f"IRR: {result.irr:.2%}")
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        version: str = "0.1.0"
    ) -> None:
        """Initialize the NPVAnalyzer.

        Args:
            max_iterations: Maximum iterations for IRR convergence.
            tolerance: Convergence tolerance for IRR calculation.
            version: Version string for this analyzer.
        """
        super().__init__(name="NPVAnalyzer", version=version)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def analyze(self, input_data: CashFlowInput) -> FinancialMetrics:
        """Perform comprehensive financial analysis.

        This method orchestrates the complete financial analysis including
        NPV, IRR, payback periods, and profitability index calculations.

        Args:
            input_data: Validated cash flow input data.

        Returns:
            FinancialMetrics containing NPV, IRR, and other financial metrics.

        Raises:
            InvalidCashFlowError: If input data is invalid.
            FinancialAnalysisError: If analysis fails.
        """
        # Validate input
        self.validate_input(input_data)

        # Calculate NPV
        npv = self.npv_calculation(
            cash_flows=input_data.cash_flows,
            discount_rate=input_data.discount_rate,
            initial_investment=input_data.initial_investment,
            periods=input_data.periods
        )

        # Calculate IRR
        try:
            irr = self.irr_computation(
                cash_flows=input_data.cash_flows,
                initial_investment=input_data.initial_investment,
                periods=input_data.periods
            )
        except ConvergenceError:
            irr = None  # IRR may not exist for all cash flow patterns

        # Convert to numpy arrays for calculations
        periods_array, cash_flows_array = input_data.to_numpy()

        # Calculate payback periods
        payback = calculate_payback_period(
            cash_flows_array,
            periods_array,
            input_data.initial_investment
        )

        discounted_payback = calculate_discounted_payback_period(
            cash_flows_array,
            periods_array,
            input_data.discount_rate,
            input_data.initial_investment
        )

        # Calculate present value of cash flows for profitability index
        pv_cash_flows = sum(
            calculate_present_value(cf, int(period), input_data.discount_rate)
            for cf, period in zip(input_data.cash_flows, input_data.periods)
        )

        profitability_index = calculate_profitability_index(
            pv_cash_flows,
            input_data.initial_investment
        )

        # Total undiscounted cash flows
        total_cash_flows = sum(input_data.cash_flows)

        return FinancialMetrics(
            analysis_type="NPV_IRR",
            npv=npv,
            irr=irr,
            discount_rate=input_data.discount_rate,
            payback_period=payback,
            discounted_payback_period=discounted_payback,
            profitability_index=profitability_index,
            total_cash_flows=total_cash_flows,
            project_name=input_data.project_name,
            metadata=self.get_metadata()
        )

    def validate_input(self, input_data: CashFlowInput) -> None:
        """Validate input data for financial analysis.

        Args:
            input_data: Cash flow input to validate.

        Raises:
            InvalidCashFlowError: If input data is invalid.
        """
        periods_array, cash_flows_array = input_data.to_numpy()
        validate_cash_flow_arrays(cash_flows_array, periods_array)

        if input_data.initial_investment <= 0:
            raise InvalidCashFlowError(
                "Initial investment must be positive",
                details={"initial_investment": input_data.initial_investment}
            )

        if input_data.discount_rate <= 0 or input_data.discount_rate > 1:
            raise InvalidCashFlowError(
                "Discount rate must be between 0 and 1",
                details={"discount_rate": input_data.discount_rate}
            )

    def npv_calculation(
        self,
        cash_flows: list[float],
        discount_rate: float,
        initial_investment: float,
        periods: Optional[list[int]] = None
    ) -> float:
        """Calculate Net Present Value (NPV) of cash flows.

        NPV is calculated using the formula:
        NPV = -Initial_Investment + Σ(CF_t / (1 + r)^t)

        where:
        - CF_t is the cash flow at time t
        - r is the discount rate
        - t is the time period

        Args:
            cash_flows: List of periodic cash flows.
            discount_rate: Discount rate (e.g., 0.10 for 10%).
            initial_investment: Initial investment amount (positive value).
            periods: Optional list of period numbers (defaults to 0, 1, 2, ...).

        Returns:
            Net Present Value (NPV) of the investment.

        Raises:
            InvalidCashFlowError: If inputs are invalid.

        Example:
            >>> analyzer = NPVAnalyzer()
            >>> npv = analyzer.npv_calculation(
            ...     cash_flows=[20000, 25000, 30000],
            ...     discount_rate=0.10,
            ...     initial_investment=60000
            ... )
            >>> print(f"NPV: ${npv:,.2f}")
        """
        # Set default periods if not provided
        if periods is None:
            periods = list(range(len(cash_flows)))

        # Validate inputs
        cash_flows_array = np.array(cash_flows, dtype=float)
        periods_array = np.array(periods, dtype=float)
        validate_cash_flow_arrays(cash_flows_array, periods_array)

        if discount_rate <= 0:
            raise InvalidCashFlowError(
                "Discount rate must be positive",
                details={"discount_rate": discount_rate}
            )

        if initial_investment <= 0:
            raise InvalidCashFlowError(
                "Initial investment must be positive",
                details={"initial_investment": initial_investment}
            )

        # Calculate present value of each cash flow
        pv_sum = 0.0
        for cf, period in zip(cash_flows, periods):
            pv = calculate_present_value(cf, int(period), discount_rate)
            pv_sum += pv

        # NPV = PV of cash flows - Initial investment
        npv = pv_sum - initial_investment

        return float(npv)

    def irr_computation(
        self,
        cash_flows: list[float],
        initial_investment: float,
        periods: Optional[list[int]] = None,
        initial_guess: float = 0.1
    ) -> float:
        """Calculate Internal Rate of Return (IRR) using Newton-Raphson method.

        IRR is the discount rate that makes NPV equal to zero:
        0 = -Initial_Investment + Σ(CF_t / (1 + IRR)^t)

        This method uses the Newton-Raphson iterative algorithm to find the IRR.
        It may fail to converge for certain cash flow patterns (e.g., multiple
        sign changes, no positive cash flows).

        Args:
            cash_flows: List of periodic cash flows.
            initial_investment: Initial investment amount (positive value).
            periods: Optional list of period numbers (defaults to 0, 1, 2, ...).
            initial_guess: Initial guess for IRR (default: 0.1 = 10%).

        Returns:
            Internal Rate of Return (IRR) as a decimal (e.g., 0.15 for 15%).

        Raises:
            InvalidCashFlowError: If inputs are invalid.
            ConvergenceError: If IRR calculation fails to converge.

        Example:
            >>> analyzer = NPVAnalyzer()
            >>> irr = analyzer.irr_computation(
            ...     cash_flows=[20000, 25000, 30000],
            ...     initial_investment=60000
            ... )
            >>> print(f"IRR: {irr:.2%}")
        """
        # Set default periods if not provided
        if periods is None:
            periods = list(range(len(cash_flows)))

        # Validate inputs
        cash_flows_array = np.array(cash_flows, dtype=float)
        periods_array = np.array(periods, dtype=float)
        validate_cash_flow_arrays(cash_flows_array, periods_array)

        if initial_investment <= 0:
            raise InvalidCashFlowError(
                "Initial investment must be positive",
                details={"initial_investment": initial_investment}
            )

        # Check if there are any positive cash flows
        if np.all(cash_flows_array <= 0):
            raise InvalidCashFlowError(
                "Cash flows must contain at least one positive value for IRR calculation"
            )

        # Define NPV function for a given rate
        def npv_function(rate: float) -> float:
            """Calculate NPV for a given discount rate."""
            if rate <= -1:
                # Prevent division by zero or negative denominator
                return float('inf')
            pv_sum = sum(
                cf / ((1 + rate) ** period)
                for cf, period in zip(cash_flows, periods)
            )
            return pv_sum - initial_investment

        # Define derivative of NPV function (for Newton-Raphson)
        def npv_derivative(rate: float) -> float:
            """Calculate derivative of NPV with respect to rate."""
            if rate <= -1:
                return float('inf')
            deriv_sum = sum(
                -period * cf / ((1 + rate) ** (period + 1))
                for cf, period in zip(cash_flows, periods)
            )
            return deriv_sum

        try:
            # Use Newton-Raphson method to find IRR
            irr = newton(
                npv_function,
                x0=initial_guess,
                fprime=npv_derivative,
                maxiter=self.max_iterations,
                tol=self.tolerance
            )

            # Validate the result
            if not np.isfinite(irr):
                raise ConvergenceError(
                    "IRR calculation produced non-finite result",
                    details={"irr": irr}
                )

            # Check if NPV is actually close to zero at this IRR
            final_npv = npv_function(irr)
            if abs(final_npv) > self.tolerance * 100:  # Allow some numerical error
                raise ConvergenceError(
                    "IRR found but NPV is not sufficiently close to zero",
                    details={"irr": irr, "npv_at_irr": final_npv}
                )

            return float(irr)

        except RuntimeError as e:
            # Newton-Raphson failed to converge
            raise ConvergenceError(
                f"IRR calculation failed to converge: {str(e)}",
                iterations=self.max_iterations,
                details={"initial_guess": initial_guess}
            )
        except (ValueError, ZeroDivisionError) as e:
            raise ConvergenceError(
                f"IRR calculation encountered numerical error: {str(e)}",
                details={"initial_guess": initial_guess}
            )

    def discount_rate_sensitivity(
        self,
        cash_flow_input: CashFlowInput,
        rate_config: DiscountRateConfig
    ) -> SensitivityAnalysisResult:
        """Perform discount rate sensitivity analysis.

        This method calculates NPV and IRR across a range of discount rates
        to show how sensitive the investment metrics are to changes in the
        discount rate assumption.

        Args:
            cash_flow_input: Base cash flow input data.
            rate_config: Configuration for discount rate range and steps.

        Returns:
            SensitivityAnalysisResult with NPV/IRR for each discount rate.

        Raises:
            InvalidCashFlowError: If inputs are invalid.
            FinancialAnalysisError: If sensitivity analysis fails.

        Example:
            >>> analyzer = NPVAnalyzer()
            >>> cash_flow_input = CashFlowInput(
            ...     initial_investment=100000,
            ...     cash_flows=[20000, 25000, 30000, 35000, 40000],
            ...     discount_rate=0.10
            ... )
            >>> rate_config = DiscountRateConfig(
            ...     base_rate=0.10,
            ...     min_rate=0.05,
            ...     max_rate=0.20,
            ...     step_size=0.01
            ... )
            >>> result = analyzer.discount_rate_sensitivity(cash_flow_input, rate_config)
            >>> print(f"NPV range: ${result.min_npv:,.2f} to ${result.max_npv:,.2f}")
        """
        # Validate input
        self.validate_input(cash_flow_input)

        # Generate discount rates
        discount_rates = np.arange(
            rate_config.min_rate,
            rate_config.max_rate + rate_config.step_size,
            rate_config.step_size
        )

        # Calculate IRR once (it doesn't depend on discount rate)
        try:
            base_irr = self.irr_computation(
                cash_flows=cash_flow_input.cash_flows,
                initial_investment=cash_flow_input.initial_investment,
                periods=cash_flow_input.periods
            )
        except ConvergenceError:
            base_irr = None

        # Calculate NPV for each discount rate
        data_points: list[SensitivityDataPoint] = []
        base_npv: Optional[float] = None

        for rate in discount_rates:
            npv = self.npv_calculation(
                cash_flows=cash_flow_input.cash_flows,
                discount_rate=float(rate),
                initial_investment=cash_flow_input.initial_investment,
                periods=cash_flow_input.periods
            )

            data_point = SensitivityDataPoint(
                parameter_value=float(rate),
                npv=npv,
                irr=base_irr  # IRR is constant across discount rates
            )
            data_points.append(data_point)

            # Store NPV at base rate
            if abs(rate - rate_config.base_rate) < rate_config.step_size / 2:
                base_npv = npv

        # If we didn't find exact base rate, calculate it
        if base_npv is None:
            base_npv = self.npv_calculation(
                cash_flows=cash_flow_input.cash_flows,
                discount_rate=rate_config.base_rate,
                initial_investment=cash_flow_input.initial_investment,
                periods=cash_flow_input.periods
            )

        # Find min and max NPV
        npv_values = [dp.npv for dp in data_points]
        min_npv = min(npv_values)
        max_npv = max(npv_values)

        return SensitivityAnalysisResult(
            analysis_type="Discount_Rate_Sensitivity",
            parameter_name="discount_rate",
            base_value=rate_config.base_rate,
            data_points=data_points,
            base_npv=base_npv,
            base_irr=base_irr,
            min_npv=min_npv,
            max_npv=max_npv,
            metadata=self.get_metadata()
        )

    def cash_flow_modeling(
        self,
        projection: CashFlowProjection,
        discount_rate: float,
        project_name: Optional[str] = None
    ) -> FinancialMetrics:
        """Model and analyze cash flow projections.

        This method takes a complete cash flow projection (with revenues,
        costs, and capital expenditures) and performs comprehensive financial
        analysis including NPV and IRR calculations.

        Args:
            projection: Cash flow projection with revenues, costs, and capex.
            discount_rate: Discount rate for NPV calculation.
            project_name: Optional name for the project.

        Returns:
            FinancialMetrics with complete financial analysis results.

        Raises:
            InvalidCashFlowError: If projection data is invalid.
            FinancialAnalysisError: If analysis fails.

        Example:
            >>> analyzer = NPVAnalyzer()
            >>> projection = CashFlowProjection(
            ...     periods=[1, 2, 3, 4, 5],
            ...     revenues=[50000, 55000, 60000, 65000, 70000],
            ...     operating_costs=[20000, 21000, 22000, 23000, 24000],
            ...     capital_expenditures=[5000, 0, 0, 0, 10000],
            ...     initial_investment=100000,
            ...     terminal_value=50000
            ... )
            >>> result = analyzer.cash_flow_modeling(projection, discount_rate=0.10)
            >>> print(f"NPV: ${result.npv:,.2f}")
        """
        # Convert projection to CashFlowInput
        if project_name is None:
            project_name = "Cash Flow Projection"

        cash_flow_input = projection.to_cash_flow_input(
            discount_rate=discount_rate,
            project_name=project_name
        )

        # Perform analysis using the standard analyze method
        return self.analyze(cash_flow_input)
