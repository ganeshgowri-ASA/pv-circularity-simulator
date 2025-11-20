"""Pydantic models for financial analysis inputs and outputs."""

from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from pv_simulator.core.base import AnalysisResult


class CashFlowType(str, Enum):
    """Type of cash flow."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NET = "net"


class DiscountRateConfig(BaseModel):
    """Configuration for discount rate sensitivity analysis.

    This model defines the range and step size for discount rate sensitivity
    analysis. It's used to test how NPV changes with different discount rates.

    Attributes:
        base_rate: Base discount rate (e.g., 0.10 for 10%).
        min_rate: Minimum discount rate for sensitivity analysis.
        max_rate: Maximum discount rate for sensitivity analysis.
        step_size: Step size for rate increments in sensitivity analysis.
    """

    base_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Base discount rate (e.g., 0.10 for 10%)"
    )
    min_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Minimum rate for sensitivity analysis"
    )
    max_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Maximum rate for sensitivity analysis"
    )
    step_size: float = Field(
        default=0.01,
        gt=0.0,
        le=0.1,
        description="Step size for rate increments"
    )

    @model_validator(mode="after")
    def validate_rate_ranges(self) -> "DiscountRateConfig":
        """Validate that min_rate <= base_rate <= max_rate."""
        if not (self.min_rate <= self.base_rate <= self.max_rate):
            raise ValueError(
                f"Discount rates must satisfy: min_rate ({self.min_rate}) <= "
                f"base_rate ({self.base_rate}) <= max_rate ({self.max_rate})"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "base_rate": 0.10,
                "min_rate": 0.05,
                "max_rate": 0.20,
                "step_size": 0.01
            }
        }


class CashFlowInput(BaseModel):
    """Input data for cash flow analysis.

    This model represents the cash flows for a financial analysis, including
    initial investment, periodic cash flows, and discount rate.

    Attributes:
        initial_investment: Initial investment amount (positive value).
        cash_flows: List of periodic cash flows (can be positive or negative).
        discount_rate: Discount rate for NPV calculation (e.g., 0.10 for 10%).
        periods: Optional list of period numbers (defaults to 0, 1, 2, ...).
        project_name: Optional name for the project or investment.
    """

    initial_investment: float = Field(
        ...,
        gt=0.0,
        description="Initial investment amount (must be positive)"
    )
    cash_flows: list[float] = Field(
        ...,
        min_length=1,
        description="Periodic cash flows (positive = inflow, negative = outflow)"
    )
    discount_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Discount rate (e.g., 0.10 for 10%)"
    )
    periods: Optional[list[int]] = Field(
        default=None,
        description="Period numbers (defaults to 0, 1, 2, ...)"
    )
    project_name: str = Field(
        default="Unnamed Project",
        description="Name of the project or investment"
    )

    @field_validator("cash_flows")
    @classmethod
    def validate_cash_flows(cls, v: list[float]) -> list[float]:
        """Validate that cash flows are not all zero."""
        if all(cf == 0.0 for cf in v):
            raise ValueError("Cash flows cannot all be zero")
        return v

    @model_validator(mode="after")
    def set_default_periods(self) -> "CashFlowInput":
        """Set default periods if not provided."""
        if self.periods is None:
            self.periods = list(range(len(self.cash_flows)))
        elif len(self.periods) != len(self.cash_flows):
            raise ValueError(
                f"Number of periods ({len(self.periods)}) must match "
                f"number of cash flows ({len(self.cash_flows)})"
            )
        return self

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert cash flows to numpy arrays.

        Returns:
            Tuple of (periods_array, cash_flows_array) as numpy arrays.
        """
        periods_array = np.array(self.periods, dtype=float)
        cash_flows_array = np.array(self.cash_flows, dtype=float)
        return periods_array, cash_flows_array

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "initial_investment": 100000.0,
                "cash_flows": [20000.0, 25000.0, 30000.0, 35000.0, 40000.0],
                "discount_rate": 0.10,
                "project_name": "PV Installation Project"
            }
        }


class CashFlowProjection(BaseModel):
    """Cash flow projection model for modeling future cash flows.

    This model represents a complete cash flow projection including
    revenue, costs, and net cash flows over time.

    Attributes:
        periods: List of time periods (years or months).
        revenues: Revenue for each period.
        operating_costs: Operating costs for each period.
        capital_expenditures: Capital expenditures for each period.
        net_cash_flows: Net cash flows (revenues - costs - capex).
        initial_investment: Initial investment at period 0.
        terminal_value: Optional terminal value at the end of the projection.
    """

    periods: list[int] = Field(
        ...,
        min_length=1,
        description="Time periods for the projection"
    )
    revenues: list[float] = Field(
        ...,
        min_length=1,
        description="Revenue for each period"
    )
    operating_costs: list[float] = Field(
        ...,
        min_length=1,
        description="Operating costs for each period"
    )
    capital_expenditures: list[float] = Field(
        default_factory=list,
        description="Capital expenditures for each period"
    )
    net_cash_flows: list[float] = Field(
        default_factory=list,
        description="Net cash flows (computed automatically)"
    )
    initial_investment: float = Field(
        default=0.0,
        ge=0.0,
        description="Initial investment at period 0"
    )
    terminal_value: Optional[float] = Field(
        default=None,
        description="Terminal value at the end of the projection"
    )

    @model_validator(mode="after")
    def compute_net_cash_flows(self) -> "CashFlowProjection":
        """Compute net cash flows if not provided."""
        # Ensure all lists have the same length
        n_periods = len(self.periods)
        if len(self.revenues) != n_periods:
            raise ValueError("Revenues length must match periods length")
        if len(self.operating_costs) != n_periods:
            raise ValueError("Operating costs length must match periods length")

        # Set default capex if not provided
        if not self.capital_expenditures:
            self.capital_expenditures = [0.0] * n_periods
        elif len(self.capital_expenditures) != n_periods:
            raise ValueError("Capital expenditures length must match periods length")

        # Compute net cash flows if not provided
        if not self.net_cash_flows:
            self.net_cash_flows = [
                rev - op_cost - capex
                for rev, op_cost, capex in zip(
                    self.revenues, self.operating_costs, self.capital_expenditures
                )
            ]

        return self

    def to_cash_flow_input(self, discount_rate: float, project_name: str = "Projection") -> CashFlowInput:
        """Convert to CashFlowInput for analysis.

        Args:
            discount_rate: Discount rate to use.
            project_name: Name for the project.

        Returns:
            CashFlowInput instance ready for analysis.
        """
        # Add terminal value to the last cash flow if present
        cash_flows = self.net_cash_flows.copy()
        if self.terminal_value is not None:
            cash_flows[-1] += self.terminal_value

        return CashFlowInput(
            initial_investment=self.initial_investment,
            cash_flows=cash_flows,
            discount_rate=discount_rate,
            periods=self.periods,
            project_name=project_name
        )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "periods": [1, 2, 3, 4, 5],
                "revenues": [50000, 55000, 60000, 65000, 70000],
                "operating_costs": [20000, 21000, 22000, 23000, 24000],
                "capital_expenditures": [5000, 0, 0, 0, 10000],
                "initial_investment": 100000,
                "terminal_value": 50000
            }
        }


class FinancialMetrics(AnalysisResult):
    """Financial analysis results including NPV, IRR, and related metrics.

    This model contains the results of a comprehensive financial analysis,
    including net present value, internal rate of return, payback period,
    and other key financial metrics.

    Attributes:
        npv: Net Present Value of the investment.
        irr: Internal Rate of Return (as decimal, e.g., 0.15 for 15%).
        discount_rate: Discount rate used for NPV calculation.
        payback_period: Simple payback period in years.
        discounted_payback_period: Discounted payback period in years.
        profitability_index: Profitability index (PV of future cash flows / Initial investment).
        total_cash_flows: Sum of all cash flows (undiscounted).
        project_name: Name of the analyzed project.
    """

    npv: float = Field(
        ...,
        description="Net Present Value of the investment"
    )
    irr: Optional[float] = Field(
        default=None,
        description="Internal Rate of Return (decimal, e.g., 0.15 for 15%)"
    )
    discount_rate: float = Field(
        ...,
        description="Discount rate used for NPV calculation"
    )
    payback_period: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Simple payback period in years"
    )
    discounted_payback_period: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Discounted payback period in years"
    )
    profitability_index: Optional[float] = Field(
        default=None,
        description="Profitability index (PV of cash flows / Initial investment)"
    )
    total_cash_flows: float = Field(
        ...,
        description="Sum of all cash flows (undiscounted)"
    )
    project_name: str = Field(
        default="Unnamed Project",
        description="Name of the analyzed project"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "analysis_type": "NPV_IRR",
                "npv": 45123.45,
                "irr": 0.18,
                "discount_rate": 0.10,
                "payback_period": 3.5,
                "discounted_payback_period": 4.2,
                "profitability_index": 1.45,
                "total_cash_flows": 150000.0,
                "project_name": "PV Installation Project",
                "timestamp": "2025-11-17T10:30:00"
            }
        }


class SensitivityDataPoint(BaseModel):
    """Single data point in a sensitivity analysis.

    Attributes:
        parameter_value: Value of the parameter being varied.
        npv: Resulting NPV at this parameter value.
        irr: Resulting IRR at this parameter value (if computable).
    """

    parameter_value: float = Field(
        ...,
        description="Value of the parameter being varied"
    )
    npv: float = Field(
        ...,
        description="Resulting NPV at this parameter value"
    )
    irr: Optional[float] = Field(
        default=None,
        description="Resulting IRR at this parameter value"
    )


class SensitivityAnalysisResult(AnalysisResult):
    """Results of a sensitivity analysis.

    This model contains the results of a sensitivity analysis showing how
    financial metrics change as a parameter (e.g., discount rate) varies.

    Attributes:
        parameter_name: Name of the parameter being varied.
        base_value: Base value of the parameter.
        data_points: List of sensitivity data points.
        base_npv: NPV at the base parameter value.
        base_irr: IRR at the base parameter value.
        min_npv: Minimum NPV across all data points.
        max_npv: Maximum NPV across all data points.
    """

    parameter_name: str = Field(
        ...,
        description="Name of the parameter being varied"
    )
    base_value: float = Field(
        ...,
        description="Base value of the parameter"
    )
    data_points: list[SensitivityDataPoint] = Field(
        ...,
        min_length=1,
        description="Sensitivity analysis data points"
    )
    base_npv: float = Field(
        ...,
        description="NPV at the base parameter value"
    )
    base_irr: Optional[float] = Field(
        default=None,
        description="IRR at the base parameter value"
    )
    min_npv: float = Field(
        ...,
        description="Minimum NPV across all data points"
    )
    max_npv: float = Field(
        ...,
        description="Maximum NPV across all data points"
    )

    @model_validator(mode="after")
    def compute_min_max_npv(self) -> "SensitivityAnalysisResult":
        """Compute min and max NPV from data points."""
        if self.data_points:
            npv_values = [dp.npv for dp in self.data_points]
            self.min_npv = min(npv_values)
            self.max_npv = max(npv_values)
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "analysis_type": "Sensitivity",
                "parameter_name": "discount_rate",
                "base_value": 0.10,
                "base_npv": 45123.45,
                "base_irr": 0.18,
                "min_npv": 20000.0,
                "max_npv": 70000.0,
                "data_points": [
                    {"parameter_value": 0.05, "npv": 70000.0, "irr": 0.18},
                    {"parameter_value": 0.10, "npv": 45123.45, "irr": 0.18},
                    {"parameter_value": 0.15, "npv": 25000.0, "irr": 0.18}
                ],
                "timestamp": "2025-11-17T10:30:00"
            }
        }
