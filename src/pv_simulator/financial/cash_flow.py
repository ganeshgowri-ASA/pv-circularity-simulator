"""Cash flow modeling and calculations for financial analysis.

This module provides utility functions for working with cash flows,
including present value calculations, cumulative cash flow analysis,
and payback period computations.
"""

from typing import Optional

import numpy as np

from pv_simulator.core.exceptions import InvalidCashFlowError


def calculate_present_value(
    cash_flow: float,
    period: int,
    discount_rate: float
) -> float:
    """Calculate the present value of a future cash flow.

    Uses the formula: PV = CF / (1 + r)^t

    Args:
        cash_flow: Future cash flow amount.
        period: Time period (0 = present, 1 = one period in future, etc.).
        discount_rate: Discount rate (e.g., 0.10 for 10%).

    Returns:
        Present value of the cash flow.

    Raises:
        InvalidCashFlowError: If discount_rate is invalid.
    """
    if discount_rate <= 0:
        raise InvalidCashFlowError(
            "Discount rate must be positive",
            details={"discount_rate": discount_rate}
        )

    if period == 0:
        return cash_flow

    discount_factor = 1.0 / ((1.0 + discount_rate) ** period)
    return cash_flow * discount_factor


def calculate_cumulative_cash_flows(
    cash_flows: np.ndarray,
    initial_investment: float
) -> np.ndarray:
    """Calculate cumulative cash flows over time.

    Args:
        cash_flows: Array of periodic cash flows.
        initial_investment: Initial investment (treated as negative cash flow at t=0).

    Returns:
        Array of cumulative cash flows including initial investment.
    """
    # Start with negative initial investment
    all_cash_flows = np.concatenate([[-initial_investment], cash_flows])
    return np.cumsum(all_cash_flows)


def calculate_discounted_cumulative_cash_flows(
    cash_flows: np.ndarray,
    periods: np.ndarray,
    discount_rate: float,
    initial_investment: float
) -> np.ndarray:
    """Calculate cumulative discounted cash flows over time.

    Args:
        cash_flows: Array of periodic cash flows.
        periods: Array of time periods.
        discount_rate: Discount rate for present value calculations.
        initial_investment: Initial investment (treated as negative cash flow at t=0).

    Returns:
        Array of cumulative discounted cash flows.

    Raises:
        InvalidCashFlowError: If inputs are invalid.
    """
    if len(cash_flows) != len(periods):
        raise InvalidCashFlowError(
            "Cash flows and periods must have the same length",
            details={"cash_flows_len": len(cash_flows), "periods_len": len(periods)}
        )

    # Calculate present value for each cash flow
    pv_cash_flows = np.array([
        calculate_present_value(cf, int(period), discount_rate)
        for cf, period in zip(cash_flows, periods)
    ])

    # Calculate cumulative with initial investment
    all_pv_cash_flows = np.concatenate([[-initial_investment], pv_cash_flows])
    return np.cumsum(all_pv_cash_flows)


def calculate_payback_period(
    cash_flows: np.ndarray,
    periods: np.ndarray,
    initial_investment: float
) -> Optional[float]:
    """Calculate the simple payback period.

    The payback period is the time it takes for cumulative cash flows
    to equal the initial investment.

    Args:
        cash_flows: Array of periodic cash flows.
        periods: Array of time periods.
        initial_investment: Initial investment amount.

    Returns:
        Payback period in years, or None if payback never occurs.
    """
    cumulative = calculate_cumulative_cash_flows(cash_flows, initial_investment)

    # Find the first period where cumulative cash flow becomes positive
    positive_indices = np.where(cumulative >= 0)[0]

    if len(positive_indices) == 0:
        return None  # Payback never occurs

    first_positive_idx = positive_indices[0]

    if first_positive_idx == 0:
        return 0.0  # Immediate payback

    # Interpolate to find exact payback period
    cf_before = cumulative[first_positive_idx - 1]
    cf_after = cumulative[first_positive_idx]

    # Linear interpolation between the two periods
    fraction = -cf_before / (cf_after - cf_before)

    # periods array includes period 0 (initial investment) in cumulative calc
    # So we need to adjust the index
    if first_positive_idx == 0:
        payback = 0.0
    else:
        period_before = periods[first_positive_idx - 1] if first_positive_idx <= len(periods) else first_positive_idx - 1
        period_after = periods[first_positive_idx] if first_positive_idx < len(periods) else first_positive_idx

        payback = period_before + fraction * (period_after - period_before)

    return float(payback)


def calculate_discounted_payback_period(
    cash_flows: np.ndarray,
    periods: np.ndarray,
    discount_rate: float,
    initial_investment: float
) -> Optional[float]:
    """Calculate the discounted payback period.

    The discounted payback period is the time it takes for cumulative
    discounted cash flows to equal the initial investment.

    Args:
        cash_flows: Array of periodic cash flows.
        periods: Array of time periods.
        discount_rate: Discount rate for present value calculations.
        initial_investment: Initial investment amount.

    Returns:
        Discounted payback period in years, or None if payback never occurs.
    """
    cumulative = calculate_discounted_cumulative_cash_flows(
        cash_flows, periods, discount_rate, initial_investment
    )

    # Find the first period where cumulative discounted cash flow becomes positive
    positive_indices = np.where(cumulative >= 0)[0]

    if len(positive_indices) == 0:
        return None  # Payback never occurs

    first_positive_idx = positive_indices[0]

    if first_positive_idx == 0:
        return 0.0  # Immediate payback

    # Interpolate to find exact payback period
    cf_before = cumulative[first_positive_idx - 1]
    cf_after = cumulative[first_positive_idx]

    # Linear interpolation
    fraction = -cf_before / (cf_after - cf_before)

    # Adjust for initial investment period
    if first_positive_idx == 0:
        payback = 0.0
    else:
        period_before = periods[first_positive_idx - 1] if first_positive_idx <= len(periods) else first_positive_idx - 1
        period_after = periods[first_positive_idx] if first_positive_idx < len(periods) else first_positive_idx

        payback = period_before + fraction * (period_after - period_before)

    return float(payback)


def calculate_profitability_index(
    pv_cash_flows: float,
    initial_investment: float
) -> float:
    """Calculate the profitability index.

    Profitability Index = PV(future cash flows) / Initial Investment

    Args:
        pv_cash_flows: Present value of all future cash flows.
        initial_investment: Initial investment amount.

    Returns:
        Profitability index.

    Raises:
        InvalidCashFlowError: If initial investment is zero or negative.
    """
    if initial_investment <= 0:
        raise InvalidCashFlowError(
            "Initial investment must be positive",
            details={"initial_investment": initial_investment}
        )

    return pv_cash_flows / initial_investment


def validate_cash_flow_arrays(
    cash_flows: np.ndarray,
    periods: np.ndarray
) -> None:
    """Validate cash flow arrays for consistency.

    Args:
        cash_flows: Array of cash flows.
        periods: Array of periods.

    Raises:
        InvalidCashFlowError: If arrays are invalid or inconsistent.
    """
    if len(cash_flows) == 0:
        raise InvalidCashFlowError("Cash flows array cannot be empty")

    if len(periods) == 0:
        raise InvalidCashFlowError("Periods array cannot be empty")

    if len(cash_flows) != len(periods):
        raise InvalidCashFlowError(
            "Cash flows and periods must have the same length",
            details={"cash_flows_len": len(cash_flows), "periods_len": len(periods)}
        )

    if np.all(cash_flows == 0):
        raise InvalidCashFlowError("Cash flows cannot all be zero")

    # Check for non-finite values
    if not np.all(np.isfinite(cash_flows)):
        raise InvalidCashFlowError("Cash flows contain non-finite values (NaN or Inf)")

    if not np.all(np.isfinite(periods)):
        raise InvalidCashFlowError("Periods contain non-finite values (NaN or Inf)")
