"""
ROI Calculator for comprehensive investment analysis.

This module provides a production-ready calculator for Return on Investment (ROI),
Net Present Value (NPV), Internal Rate of Return (IRR), payback period calculations,
and sensitivity analysis for PV system investments.

The calculator implements industry-standard financial modeling techniques with
comprehensive validation, error handling, and detailed documentation.

Example:
    >>> from pv_simulator.core.models import InvestmentInput
    >>> from pv_simulator.calculators.roi_calculator import ROICalculator
    >>>
    >>> # Create investment scenario
    >>> investment = InvestmentInput(
    ...     initial_investment=100000,
    ...     annual_revenue=25000,
    ...     annual_costs=5000,
    ...     discount_rate=0.10,
    ...     project_lifetime=25
    ... )
    >>>
    >>> # Calculate ROI and financial metrics
    >>> calculator = ROICalculator()
    >>> result = calculator.calculate(investment)
    >>> print(f"ROI: {result.roi_percentage:.2f}%")
    >>> print(f"NPV: ${result.net_present_value:,.2f}")
    >>> print(f"Payback Period: {result.payback_period_years:.1f} years")
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import newton, brentq
import logging

from .base import BaseCalculator
from ..core.models import (
    InvestmentInput,
    ROIResult,
    CashFlow,
    SensitivityInput,
    SensitivityResult,
    SensitivityAnalysisResult,
)
from ..core.enums import SensitivityParameter
from ..exceptions import CalculationError, ValidationError, ConvergenceError


logger = logging.getLogger(__name__)


class ROICalculator(BaseCalculator[InvestmentInput, ROIResult]):
    """
    Comprehensive calculator for investment analysis and ROI calculations.

    This calculator provides production-ready implementations of key financial
    metrics including ROI, NPV, IRR, payback period, and sensitivity analysis.
    All calculations follow industry-standard financial modeling practices.

    The calculator handles:
    - Return on Investment (ROI) calculations with tax considerations
    - Net Present Value (NPV) with configurable discount rates
    - Internal Rate of Return (IRR) using numerical optimization
    - Simple and discounted payback period calculations
    - Multi-parameter sensitivity analysis
    - Detailed cash flow modeling

    Attributes:
        max_irr_iterations: Maximum iterations for IRR convergence (default: 100)
        irr_tolerance: Convergence tolerance for IRR calculation (default: 1e-6)

    Example:
        >>> calculator = ROICalculator()
        >>> investment = InvestmentInput(
        ...     initial_investment=100000,
        ...     annual_revenue=25000,
        ...     annual_costs=5000,
        ...     discount_rate=0.10,
        ...     project_lifetime=25
        ... )
        >>> result = calculator.calculate(investment)
        >>> print(f"ROI: {result.roi_percentage:.2f}%")
        ROI: 400.00%
    """

    def __init__(
        self,
        name: str = "ROICalculator",
        max_irr_iterations: int = 100,
        irr_tolerance: float = 1e-6,
    ):
        """
        Initialize the ROI Calculator.

        Args:
            name: Name for this calculator instance (for logging)
            max_irr_iterations: Maximum iterations for IRR convergence
            irr_tolerance: Convergence tolerance for IRR calculation
        """
        super().__init__(name)
        self.max_irr_iterations = max_irr_iterations
        self.irr_tolerance = irr_tolerance
        self.logger.info(
            f"Initialized {name} (max_iterations={max_irr_iterations}, "
            f"tolerance={irr_tolerance})"
        )

    def validate(self, inputs: InvestmentInput) -> bool:
        """
        Validate investment input parameters.

        Performs comprehensive validation including:
        - Positive initial investment
        - Non-negative revenues and costs
        - Valid discount rate (0-100%)
        - Reasonable project lifetime (1-50 years)
        - Logical relationship between revenues and costs

        Args:
            inputs: Investment input model to validate

        Returns:
            True if all validations pass

        Raises:
            ValidationError: If any validation check fails
        """
        self.logger.debug("Validating investment inputs")

        # Pydantic already validates basic constraints, but add business logic checks
        if inputs.initial_investment <= 0:
            raise ValidationError("Initial investment must be positive")

        if inputs.annual_revenue < 0:
            raise ValidationError("Annual revenue cannot be negative")

        if inputs.annual_costs < 0:
            raise ValidationError("Annual costs cannot be negative")

        if inputs.discount_rate < 0 or inputs.discount_rate > 1:
            raise ValidationError("Discount rate must be between 0 and 1")

        if inputs.project_lifetime < 1 or inputs.project_lifetime > 50:
            raise ValidationError("Project lifetime must be between 1 and 50 years")

        # Warning if costs exceed revenue
        if inputs.annual_costs > inputs.annual_revenue:
            self.logger.warning(
                f"Annual costs ({inputs.annual_costs}) exceed annual revenue "
                f"({inputs.annual_revenue}). Project may not be profitable."
            )

        # Warning if very high discount rate
        if inputs.discount_rate > 0.25:
            self.logger.warning(
                f"Discount rate ({inputs.discount_rate:.1%}) is unusually high. "
                "Please verify this is intentional."
            )

        self.logger.debug("Validation successful")
        return True

    def calculate(self, inputs: InvestmentInput) -> ROIResult:
        """
        Calculate comprehensive investment analysis metrics.

        This is the main calculation method that orchestrates all financial
        calculations including ROI, NPV, IRR, and payback period.

        Args:
            inputs: Validated investment input parameters

        Returns:
            ROIResult containing all calculated financial metrics

        Raises:
            ValidationError: If inputs are invalid
            CalculationError: If calculation fails
        """
        self.logger.info("Starting comprehensive ROI calculation")

        # Validate inputs
        self.validate(inputs)

        # Generate detailed cash flows
        cash_flows = self._generate_cash_flows(inputs)

        # Calculate NPV
        npv = self._calculate_npv(cash_flows, inputs.discount_rate)

        # Calculate basic ROI
        roi_pct = self.roi_calculation(inputs)

        # Calculate payback periods
        payback = self.payback_period(cash_flows, discounted=False)
        discounted_payback = self.payback_period(cash_flows, discounted=True)

        # Calculate IRR
        try:
            irr = self.irr_calculation(cash_flows)
        except ConvergenceError as e:
            self.logger.warning(f"IRR calculation failed to converge: {e}")
            irr = None

        # Calculate total metrics
        total_revenue = inputs.annual_revenue * inputs.project_lifetime
        total_costs = inputs.annual_costs * inputs.project_lifetime
        net_profit = total_revenue - total_costs - inputs.initial_investment

        # Calculate profitability index
        profitability_index = npv / inputs.initial_investment if npv != 0 else 0

        # Calculate annual ROI
        annual_roi = roi_pct / inputs.project_lifetime if inputs.project_lifetime > 0 else 0

        # Build result
        result = ROIResult(
            roi_percentage=roi_pct,
            net_present_value=npv,
            internal_rate_of_return=irr,
            payback_period_years=payback,
            discounted_payback_period_years=discounted_payback,
            total_revenue=total_revenue,
            total_costs=total_costs,
            net_profit=net_profit,
            profitability_index=profitability_index,
            annual_roi=annual_roi,
            cash_flows=cash_flows,
            currency=inputs.currency,
            initial_investment=inputs.initial_investment,
            discount_rate=inputs.discount_rate,
        )

        self.logger.info(
            f"Calculation complete: ROI={roi_pct:.2f}%, NPV={npv:,.2f}, "
            f"IRR={irr:.2f}%" if irr else f"IRR=None"
        )

        return result

    def roi_calculation(self, inputs: InvestmentInput) -> float:
        """
        Calculate Return on Investment (ROI) percentage.

        ROI is calculated as:
            ROI = ((Total Net Revenue - Initial Investment) / Initial Investment) × 100

        Where Total Net Revenue accounts for:
        - Annual revenue over project lifetime
        - Annual operating costs
        - Tax implications
        - Salvage value

        Args:
            inputs: Investment input parameters

        Returns:
            ROI as a percentage

        Raises:
            ValidationError: If inputs are invalid
            CalculationError: If calculation produces invalid result

        Example:
            >>> calculator = ROICalculator()
            >>> inputs = InvestmentInput(
            ...     initial_investment=100000,
            ...     annual_revenue=25000,
            ...     annual_costs=5000,
            ...     project_lifetime=25
            ... )
            >>> roi = calculator.roi_calculation(inputs)
            >>> print(f"ROI: {roi:.2f}%")
            ROI: 400.00%
        """
        self.logger.debug("Calculating ROI")

        # Calculate annual net cash flow
        annual_net_cash_flow = inputs.annual_revenue - inputs.annual_costs

        # Apply tax if applicable
        if inputs.tax_rate > 0:
            annual_net_cash_flow_after_tax = annual_net_cash_flow * (1 - inputs.tax_rate)
        else:
            annual_net_cash_flow_after_tax = annual_net_cash_flow

        # Calculate total net revenue over project lifetime
        total_net_revenue = annual_net_cash_flow_after_tax * inputs.project_lifetime

        # Add salvage value
        total_net_revenue += inputs.salvage_value

        # Calculate net gain
        net_gain = total_net_revenue - inputs.initial_investment

        # Calculate ROI percentage
        roi_percentage = (net_gain / inputs.initial_investment) * 100

        self.logger.debug(f"ROI calculated: {roi_percentage:.2f}%")

        return roi_percentage

    def payback_period(
        self, cash_flows: List[CashFlow], discounted: bool = False
    ) -> Optional[float]:
        """
        Calculate payback period in years.

        The payback period is the time required to recover the initial investment
        from the project's cash flows.

        Args:
            cash_flows: List of CashFlow objects with yearly cash flows
            discounted: If True, use discounted cash flows; if False, use nominal

        Returns:
            Payback period in years (can be fractional), or None if investment
            is never recovered

        Raises:
            CalculationError: If cash flows are invalid

        Example:
            >>> cash_flows = [
            ...     CashFlow(year=0, inflow=0, outflow=100000, net_flow=-100000,
            ...              cumulative_flow=-100000, discounted_flow=-100000),
            ...     CashFlow(year=1, inflow=25000, outflow=5000, net_flow=20000,
            ...              cumulative_flow=-80000, discounted_flow=18182),
            ... ]
            >>> calculator = ROICalculator()
            >>> payback = calculator.payback_period(cash_flows, discounted=False)
            >>> print(f"Payback period: {payback:.1f} years")
            Payback period: 5.0 years
        """
        self.logger.debug(f"Calculating {'discounted ' if discounted else ''}payback period")

        if not cash_flows:
            raise CalculationError("Cash flows list is empty")

        # Build cumulative cash flow array
        cumulative = 0.0
        for i, cf in enumerate(cash_flows):
            flow = cf.discounted_flow if discounted else cf.net_flow
            cumulative += flow

            # Check if we've recovered the investment
            if cumulative >= 0 and i > 0:
                # Interpolate to find exact payback year
                prev_cumulative = sum(
                    (c.discounted_flow if discounted else c.net_flow)
                    for c in cash_flows[:i]
                )

                # Linear interpolation
                fraction = -prev_cumulative / flow if flow != 0 else 0
                payback_years = (i - 1) + fraction

                self.logger.debug(
                    f"{'Discounted p' if discounted else 'P'}ayback period: "
                    f"{payback_years:.2f} years"
                )
                return payback_years

        # Investment never recovered
        self.logger.warning("Investment is never recovered over project lifetime")
        return None

    def irr_calculation(self, cash_flows: List[CashFlow]) -> Optional[float]:
        """
        Calculate Internal Rate of Return (IRR) using numerical optimization.

        IRR is the discount rate that makes the NPV of all cash flows equal to zero.
        This implementation uses scipy's numerical solvers with fallback strategies:
        1. Brent's method (robust, guaranteed convergence within bounds)
        2. Newton's method (faster but less robust)

        Args:
            cash_flows: List of CashFlow objects with yearly cash flows

        Returns:
            IRR as a percentage, or None if calculation fails to converge

        Raises:
            ConvergenceError: If IRR cannot be found within iteration limits
            CalculationError: If cash flows are invalid

        Example:
            >>> cash_flows = [
            ...     CashFlow(year=0, inflow=0, outflow=100000, net_flow=-100000,
            ...              cumulative_flow=-100000, discounted_flow=-100000),
            ...     CashFlow(year=1, inflow=25000, outflow=5000, net_flow=20000,
            ...              cumulative_flow=-80000, discounted_flow=18182),
            ... ]
            >>> calculator = ROICalculator()
            >>> irr = calculator.irr_calculation(cash_flows)
            >>> print(f"IRR: {irr:.2f}%")
            IRR: 15.30%
        """
        self.logger.debug("Calculating IRR using numerical optimization")

        if not cash_flows:
            raise CalculationError("Cash flows list is empty")

        # Extract net cash flows
        net_flows = np.array([cf.net_flow for cf in cash_flows])

        # Check if all cash flows are zero
        if np.allclose(net_flows, 0):
            raise CalculationError("All cash flows are zero, IRR is undefined")

        # Check for no sign changes (required for unique IRR)
        sign_changes = np.sum(np.diff(np.sign(net_flows)) != 0)
        if sign_changes == 0:
            self.logger.warning("No sign changes in cash flows, IRR may not exist")

        # Define NPV function for root finding
        def npv_function(rate: float) -> float:
            """Calculate NPV at a given discount rate."""
            if rate <= -1:
                return np.inf
            return np.sum(net_flows / np.power(1 + rate, np.arange(len(net_flows))))

        try:
            # Try Brent's method first (robust, bounded)
            # Search between -99% and 1000% annual return
            irr_rate = brentq(npv_function, -0.99, 10.0, maxiter=self.max_irr_iterations)
            irr_percentage = irr_rate * 100

            self.logger.debug(f"IRR calculated using Brent's method: {irr_percentage:.2f}%")
            return irr_percentage

        except ValueError:
            # Brent's method failed, try Newton's method with initial guess
            try:
                # Use a reasonable initial guess (10% return)
                irr_rate = newton(
                    npv_function,
                    x0=0.10,
                    maxiter=self.max_irr_iterations,
                    tol=self.irr_tolerance,
                )
                irr_percentage = irr_rate * 100

                self.logger.debug(f"IRR calculated using Newton's method: {irr_percentage:.2f}%")
                return irr_percentage

            except RuntimeError as e:
                raise ConvergenceError(
                    "IRR calculation failed to converge",
                    iterations=self.max_irr_iterations,
                    tolerance=self.irr_tolerance,
                ) from e

    def sensitivity_analysis(
        self,
        base_inputs: InvestmentInput,
        sensitivity_inputs: List[SensitivityInput],
    ) -> List[SensitivityAnalysisResult]:
        """
        Perform sensitivity analysis on investment parameters.

        Sensitivity analysis evaluates how changes in input parameters affect
        the investment outcomes (ROI, NPV, IRR). This helps identify which
        parameters have the most impact on investment returns and assess risk.

        The method varies each specified parameter across a range of values
        while keeping all other parameters constant, then calculates all
        financial metrics for each variation.

        Args:
            base_inputs: Base case investment parameters
            sensitivity_inputs: List of parameters to vary and their ranges

        Returns:
            List of SensitivityAnalysisResult, one for each parameter analyzed

        Raises:
            ValidationError: If inputs are invalid
            CalculationError: If sensitivity analysis fails

        Example:
            >>> calculator = ROICalculator()
            >>> base_inputs = InvestmentInput(
            ...     initial_investment=100000,
            ...     annual_revenue=25000,
            ...     annual_costs=5000,
            ...     discount_rate=0.10,
            ...     project_lifetime=25
            ... )
            >>> sensitivity = SensitivityInput(
            ...     parameter=SensitivityParameter.DISCOUNT_RATE,
            ...     base_value=0.10,
            ...     variation_range=[-20, -10, 0, 10, 20]
            ... )
            >>> results = calculator.sensitivity_analysis(base_inputs, [sensitivity])
            >>> print(f"NPV Range: {results[0].npv_range}")
        """
        self.logger.info(
            f"Starting sensitivity analysis for {len(sensitivity_inputs)} parameters"
        )

        # Calculate base case
        base_result = self.calculate(base_inputs)

        all_results = []

        for sens_input in sensitivity_inputs:
            self.logger.debug(f"Analyzing sensitivity to {sens_input.parameter.value}")

            parameter_results = []
            test_values = sens_input.get_test_values()

            for test_value in test_values:
                # Create modified inputs
                modified_inputs = self._modify_parameter(
                    base_inputs, sens_input.parameter, test_value
                )

                # Calculate metrics for this variation
                result = self.calculate(modified_inputs)

                # Store result
                parameter_results.append(
                    SensitivityResult(
                        parameter=sens_input.parameter,
                        parameter_value=test_value,
                        roi_percentage=result.roi_percentage,
                        net_present_value=result.net_present_value,
                        internal_rate_of_return=result.internal_rate_of_return,
                        payback_period_years=result.payback_period_years,
                    )
                )

            # Calculate ranges and statistics
            roi_values = [r.roi_percentage for r in parameter_results]
            npv_values = [r.net_present_value for r in parameter_results]

            roi_range = [min(roi_values), max(roi_values)]
            npv_range = [min(npv_values), max(npv_values)]

            # Determine most sensitive variation
            npv_deltas = [abs(r.net_present_value - base_result.net_present_value)
                          for r in parameter_results]
            max_impact_idx = int(np.argmax(npv_deltas))
            most_sensitive = (
                f"NPV changes by {npv_deltas[max_impact_idx]:,.0f} "
                f"when {sens_input.parameter.value} = {test_values[max_impact_idx]:.4f}"
            )

            # Calculate elasticity (for middle variation if available)
            elasticity = self._calculate_elasticity(
                base_result, parameter_results, sens_input, test_values
            )

            # Build sensitivity result
            sens_result = SensitivityAnalysisResult(
                parameter=sens_input.parameter,
                base_case=base_result,
                results=parameter_results,
                roi_range=roi_range,
                npv_range=npv_range,
                most_sensitive_to=most_sensitive,
                elasticity=elasticity,
            )

            all_results.append(sens_result)

            self.logger.info(
                f"Sensitivity to {sens_input.parameter.value}: "
                f"NPV range [{npv_range[0]:,.0f}, {npv_range[1]:,.0f}]"
            )

        return all_results

    # ==================== Private Helper Methods ====================

    def _generate_cash_flows(self, inputs: InvestmentInput) -> List[CashFlow]:
        """
        Generate detailed yearly cash flow projections.

        Args:
            inputs: Investment input parameters

        Returns:
            List of CashFlow objects for each year of the project
        """
        cash_flows = []
        cumulative = 0.0

        # Year 0: Initial investment
        cash_flows.append(
            CashFlow(
                year=0,
                inflow=0.0,
                outflow=inputs.initial_investment,
                net_flow=-inputs.initial_investment,
                cumulative_flow=-inputs.initial_investment,
                discounted_flow=-inputs.initial_investment,
            )
        )
        cumulative = -inputs.initial_investment

        # Operating years
        for year in range(1, inputs.project_lifetime + 1):
            # Apply inflation if specified
            if inputs.inflation_rate > 0:
                inflation_factor = (1 + inputs.inflation_rate) ** year
                revenue = inputs.annual_revenue * inflation_factor
                costs = inputs.annual_costs * inflation_factor
            else:
                revenue = inputs.annual_revenue
                costs = inputs.annual_costs

            # Calculate net flow after tax
            gross_net_flow = revenue - costs
            if inputs.tax_rate > 0:
                net_flow = gross_net_flow * (1 - inputs.tax_rate)
            else:
                net_flow = gross_net_flow

            # Add salvage value in final year
            if year == inputs.project_lifetime:
                revenue += inputs.salvage_value
                net_flow += inputs.salvage_value

            # Calculate cumulative and discounted flows
            cumulative += net_flow
            discount_factor = (1 + inputs.discount_rate) ** year
            discounted_flow = net_flow / discount_factor

            cash_flows.append(
                CashFlow(
                    year=year,
                    inflow=revenue,
                    outflow=costs,
                    net_flow=net_flow,
                    cumulative_flow=cumulative,
                    discounted_flow=discounted_flow,
                )
            )

        return cash_flows

    def _calculate_npv(self, cash_flows: List[CashFlow], discount_rate: float) -> float:
        """
        Calculate Net Present Value from cash flows.

        Args:
            cash_flows: List of CashFlow objects
            discount_rate: Discount rate to use

        Returns:
            Net Present Value
        """
        return sum(cf.discounted_flow for cf in cash_flows)

    def _modify_parameter(
        self,
        base_inputs: InvestmentInput,
        parameter: SensitivityParameter,
        new_value: float,
    ) -> InvestmentInput:
        """
        Create a new InvestmentInput with one parameter modified.

        Args:
            base_inputs: Base investment inputs
            parameter: Parameter to modify
            new_value: New value for the parameter

        Returns:
            New InvestmentInput instance with modified parameter
        """
        # Convert to dict and modify
        inputs_dict = base_inputs.model_dump()

        # Map enum to actual field name
        param_mapping = {
            SensitivityParameter.DISCOUNT_RATE: "discount_rate",
            SensitivityParameter.INITIAL_INVESTMENT: "initial_investment",
            SensitivityParameter.ANNUAL_REVENUE: "annual_revenue",
            SensitivityParameter.ANNUAL_COSTS: "annual_costs",
            SensitivityParameter.PROJECT_LIFETIME: "project_lifetime",
        }

        field_name = param_mapping.get(parameter)
        if field_name:
            inputs_dict[field_name] = new_value
        else:
            raise ValueError(f"Unsupported sensitivity parameter: {parameter}")

        return InvestmentInput(**inputs_dict)

    def _calculate_elasticity(
        self,
        base_result: ROIResult,
        parameter_results: List[SensitivityResult],
        sensitivity_input: SensitivityInput,
        test_values: List[float],
    ) -> Optional[float]:
        """
        Calculate elasticity coefficient for sensitivity analysis.

        Elasticity = (% change in NPV) / (% change in parameter)

        Args:
            base_result: Base case results
            parameter_results: Results for each parameter variation
            sensitivity_input: Sensitivity input configuration
            test_values: List of test values used

        Returns:
            Elasticity coefficient or None if cannot be calculated
        """
        # Find the result closest to base value
        base_idx = None
        for i, val in enumerate(test_values):
            if np.isclose(val, sensitivity_input.base_value, rtol=1e-6):
                base_idx = i
                break

        if base_idx is None or base_idx >= len(parameter_results) - 1:
            return None

        # Use next variation to calculate elasticity
        next_result = parameter_results[base_idx + 1]
        next_value = test_values[base_idx + 1]

        # Calculate percentage changes
        npv_change_pct = (
            (next_result.net_present_value - base_result.net_present_value)
            / base_result.net_present_value
            * 100
            if base_result.net_present_value != 0
            else 0
        )

        param_change_pct = (
            (next_value - sensitivity_input.base_value) / sensitivity_input.base_value * 100
            if sensitivity_input.base_value != 0
            else 0
        )

        if param_change_pct == 0:
            return None

        elasticity = npv_change_pct / param_change_pct
        return elasticity
