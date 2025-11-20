"""Pydantic models for tax incentives and financial calculations."""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from pv_simulator.models.base import FinancialBase, SimulationBase


class DepreciationMethod(str, Enum):
    """Supported depreciation methods for solar assets.

    Attributes:
        MACRS_5: Modified Accelerated Cost Recovery System - 5 year schedule.
        MACRS_7: Modified Accelerated Cost Recovery System - 7 year schedule.
        STRAIGHT_LINE: Straight-line depreciation method.
        DECLINING_BALANCE: Double declining balance depreciation.
    """

    MACRS_5 = "MACRS_5"
    MACRS_7 = "MACRS_7"
    STRAIGHT_LINE = "STRAIGHT_LINE"
    DECLINING_BALANCE = "DECLINING_BALANCE"


class SystemConfiguration(SimulationBase):
    """Configuration for a PV system.

    Represents the physical and financial characteristics of a photovoltaic system.

    Attributes:
        system_size_kw: System capacity in kilowatts (DC).
        installation_cost_total: Total installation cost in USD.
        installation_date: Date when the system is or was installed.
        location_state: US state code (e.g., 'CA', 'TX') for regional incentives.
        expected_annual_production_kwh: Expected annual energy production in kWh.
        system_lifetime_years: Expected operational lifetime in years.
        module_efficiency: Module efficiency as a decimal (0-1).
        inverter_efficiency: Inverter efficiency as a decimal (0-1).
    """

    system_size_kw: float = Field(
        ...,
        gt=0,
        description="System capacity in kilowatts (DC)",
    )
    installation_cost_total: float = Field(
        ...,
        gt=0,
        description="Total installation cost in USD",
    )
    installation_date: date = Field(
        ...,
        description="Date when the system is/was installed",
    )
    location_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="US state code (e.g., 'CA', 'TX')",
    )
    expected_annual_production_kwh: float = Field(
        ...,
        gt=0,
        description="Expected annual energy production in kWh",
    )
    system_lifetime_years: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Expected operational lifetime in years",
    )
    module_efficiency: float = Field(
        default=0.20,
        ge=0.05,
        le=0.30,
        description="Module efficiency as a decimal (0-1)",
    )
    inverter_efficiency: float = Field(
        default=0.96,
        ge=0.90,
        le=1.0,
        description="Inverter efficiency as a decimal (0-1)",
    )

    @field_validator("location_state")
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Validate and uppercase state code."""
        return v.upper()


class ITCConfiguration(SimulationBase):
    """Configuration for Investment Tax Credit (ITC) calculation.

    The Investment Tax Credit provides a federal tax credit for solar installations.

    Attributes:
        system_config: Configuration of the PV system.
        itc_rate: ITC rate as a decimal (e.g., 0.30 for 30%).
        bonus_rate: Additional bonus credit rate for domestic content or other criteria.
        apply_bonus: Whether to apply bonus credit.
        basis_reduction_for_grants: Reduction in ITC basis if grants received.
        state_grants_received: Amount of state grants received that reduce ITC basis.
        is_commercial: Whether this is a commercial (vs residential) installation.
        meets_domestic_content: Whether system meets domestic content requirements.
        is_energy_community: Whether project is in an energy community.
    """

    system_config: SystemConfiguration = Field(
        ...,
        description="Configuration of the PV system",
    )
    itc_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="ITC rate as a decimal (e.g., 0.30 for 30%)",
    )
    bonus_rate: float = Field(
        default=0.10,
        ge=0.0,
        le=0.50,
        description="Additional bonus credit rate",
    )
    apply_bonus: bool = Field(
        default=False,
        description="Whether to apply bonus credit",
    )
    basis_reduction_for_grants: bool = Field(
        default=True,
        description="Whether to reduce basis for received grants",
    )
    state_grants_received: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount of state grants received",
    )
    is_commercial: bool = Field(
        default=True,
        description="Whether this is a commercial installation",
    )
    meets_domestic_content: bool = Field(
        default=False,
        description="Whether system meets domestic content requirements",
    )
    is_energy_community: bool = Field(
        default=False,
        description="Whether project is in an energy community",
    )


class ITCResult(FinancialBase):
    """Result of Investment Tax Credit calculation.

    Attributes:
        total_itc_amount: Total ITC credit amount in USD.
        base_itc: Base ITC amount before bonuses.
        bonus_itc: Additional bonus credit amount.
        eligible_basis: Cost basis eligible for ITC.
        effective_rate: Effective ITC rate after all adjustments.
        basis_reduction: Amount by which basis was reduced for grants.
        recapture_period_years: Period during which ITC can be recaptured.
        calculation_details: Detailed breakdown of calculation steps.
    """

    total_itc_amount: float = Field(
        ...,
        ge=0,
        description="Total ITC credit amount in USD",
    )
    base_itc: float = Field(
        ...,
        ge=0,
        description="Base ITC amount before bonuses",
    )
    bonus_itc: float = Field(
        default=0.0,
        ge=0,
        description="Additional bonus credit amount",
    )
    eligible_basis: float = Field(
        ...,
        gt=0,
        description="Cost basis eligible for ITC",
    )
    effective_rate: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Effective ITC rate after all adjustments",
    )
    basis_reduction: float = Field(
        default=0.0,
        ge=0,
        description="Amount by which basis was reduced",
    )
    recapture_period_years: int = Field(
        default=5,
        ge=0,
        description="Period during which ITC can be recaptured",
    )
    calculation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of calculation steps",
    )


class PTCConfiguration(SimulationBase):
    """Configuration for Production Tax Credit (PTC) calculation.

    The Production Tax Credit provides a per-kWh credit for electricity production.

    Attributes:
        system_config: Configuration of the PV system.
        ptc_rate_per_kwh: Base PTC rate in USD per kWh.
        credit_period_years: Number of years PTC applies (typically 10).
        inflation_adjustment: Whether to adjust for inflation annually.
        inflation_rate: Expected annual inflation rate as decimal.
        production_degradation_rate: Annual degradation rate for production.
        apply_bonus: Whether to apply bonus credit multiplier.
        bonus_multiplier: Multiplier for bonus credit (e.g., 5 for 5x).
    """

    system_config: SystemConfiguration = Field(
        ...,
        description="Configuration of the PV system",
    )
    ptc_rate_per_kwh: float = Field(
        default=0.0275,
        gt=0,
        description="Base PTC rate in USD per kWh",
    )
    credit_period_years: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of years PTC applies",
    )
    inflation_adjustment: bool = Field(
        default=True,
        description="Whether to adjust for inflation annually",
    )
    inflation_rate: float = Field(
        default=0.025,
        ge=0.0,
        le=0.10,
        description="Expected annual inflation rate as decimal",
    )
    production_degradation_rate: float = Field(
        default=0.005,
        ge=0.0,
        le=0.02,
        description="Annual degradation rate for production",
    )
    apply_bonus: bool = Field(
        default=False,
        description="Whether to apply bonus credit multiplier",
    )
    bonus_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        le=5.0,
        description="Multiplier for bonus credit",
    )


class PTCResult(FinancialBase):
    """Result of Production Tax Credit calculation.

    Attributes:
        total_ptc_lifetime: Total PTC credits over entire credit period.
        annual_credits: List of annual credit amounts for each year.
        annual_production: List of annual production values (kWh) for each year.
        present_value_ptc: Net present value of all PTC credits.
        discount_rate: Discount rate used for NPV calculation.
        credit_period_years: Number of years included in calculation.
        first_year_credit: Credit amount in first year.
        last_year_credit: Credit amount in final year.
        calculation_details: Detailed breakdown of calculation.
    """

    total_ptc_lifetime: float = Field(
        ...,
        ge=0,
        description="Total PTC credits over entire credit period",
    )
    annual_credits: list[float] = Field(
        ...,
        min_length=1,
        description="List of annual credit amounts",
    )
    annual_production: list[float] = Field(
        ...,
        min_length=1,
        description="List of annual production values (kWh)",
    )
    present_value_ptc: float = Field(
        ...,
        ge=0,
        description="Net present value of all PTC credits",
    )
    discount_rate: float = Field(
        default=0.06,
        ge=0.0,
        le=0.20,
        description="Discount rate used for NPV calculation",
    )
    credit_period_years: int = Field(
        ...,
        ge=1,
        description="Number of years included in calculation",
    )
    first_year_credit: float = Field(
        ...,
        ge=0,
        description="Credit amount in first year",
    )
    last_year_credit: float = Field(
        ...,
        ge=0,
        description="Credit amount in final year",
    )
    calculation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of calculation",
    )

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "PTCResult":
        """Ensure annual_credits and annual_production have matching lengths."""
        if len(self.annual_credits) != len(self.annual_production):
            raise ValueError(
                "annual_credits and annual_production must have the same length"
            )
        if len(self.annual_credits) != self.credit_period_years:
            raise ValueError(
                "Length of annual lists must equal credit_period_years"
            )
        return self


class DepreciationScheduleResult(FinancialBase):
    """Result of depreciation schedule calculation.

    Attributes:
        method: Depreciation method used.
        asset_basis: Initial asset basis for depreciation.
        schedule_years: Number of years in depreciation schedule.
        annual_depreciation: List of depreciation amounts by year.
        cumulative_depreciation: List of cumulative depreciation by year.
        remaining_basis: List of remaining basis values by year.
        total_depreciation: Total depreciation over schedule period.
        macrs_convention: Convention used for MACRS (half-year, mid-quarter).
        bonus_depreciation_rate: Bonus depreciation rate applied in year 1.
        bonus_depreciation_amount: Amount of bonus depreciation taken.
        calculation_details: Detailed breakdown of calculation.
    """

    method: DepreciationMethod = Field(
        ...,
        description="Depreciation method used",
    )
    asset_basis: float = Field(
        ...,
        gt=0,
        description="Initial asset basis for depreciation",
    )
    schedule_years: int = Field(
        ...,
        ge=1,
        description="Number of years in depreciation schedule",
    )
    annual_depreciation: list[float] = Field(
        ...,
        min_length=1,
        description="List of depreciation amounts by year",
    )
    cumulative_depreciation: list[float] = Field(
        ...,
        min_length=1,
        description="List of cumulative depreciation by year",
    )
    remaining_basis: list[float] = Field(
        ...,
        min_length=1,
        description="List of remaining basis values by year",
    )
    total_depreciation: float = Field(
        ...,
        ge=0,
        description="Total depreciation over schedule period",
    )
    macrs_convention: str | None = Field(
        default=None,
        description="Convention used for MACRS",
    )
    bonus_depreciation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Bonus depreciation rate applied",
    )
    bonus_depreciation_amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount of bonus depreciation taken",
    )
    calculation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of calculation",
    )

    @model_validator(mode="after")
    def validate_schedule_consistency(self) -> "DepreciationScheduleResult":
        """Validate consistency of depreciation schedules."""
        list_len = len(self.annual_depreciation)
        if (
            len(self.cumulative_depreciation) != list_len
            or len(self.remaining_basis) != list_len
        ):
            raise ValueError("All schedule lists must have the same length")
        if list_len != self.schedule_years:
            raise ValueError("Schedule length must equal schedule_years")
        return self


class TaxEquityConfiguration(SimulationBase):
    """Configuration for tax equity structure modeling.

    Models partnership flip and other tax equity structures common in solar finance.

    Attributes:
        system_config: Configuration of the PV system.
        investor_equity_percentage: Investor's equity stake as decimal.
        target_flip_irr: Target IRR for investor before flip (decimal).
        post_flip_investor_percentage: Investor's share after flip.
        tax_rate: Combined federal and state tax rate.
        required_investor_return: Minimum required return for investor.
        project_lifetime_years: Total project lifetime for modeling.
        discount_rate: Discount rate for NPV calculations.
        include_itc: Whether to include ITC in the structure.
        include_ptc: Whether to include PTC in the structure.
        include_depreciation: Whether to include depreciation benefits.
    """

    system_config: SystemConfiguration = Field(
        ...,
        description="Configuration of the PV system",
    )
    investor_equity_percentage: float = Field(
        default=0.99,
        ge=0.01,
        le=1.0,
        description="Investor's equity stake as decimal",
    )
    target_flip_irr: float = Field(
        default=0.08,
        ge=0.0,
        le=0.30,
        description="Target IRR for investor before flip",
    )
    post_flip_investor_percentage: float = Field(
        default=0.05,
        ge=0.0,
        le=0.50,
        description="Investor's share after flip",
    )
    tax_rate: float = Field(
        default=0.40,
        ge=0.0,
        le=0.60,
        description="Combined federal and state tax rate",
    )
    required_investor_return: float = Field(
        default=0.06,
        ge=0.0,
        le=0.30,
        description="Minimum required return for investor",
    )
    project_lifetime_years: int = Field(
        default=25,
        ge=10,
        le=40,
        description="Total project lifetime for modeling",
    )
    discount_rate: float = Field(
        default=0.06,
        ge=0.0,
        le=0.20,
        description="Discount rate for NPV calculations",
    )
    include_itc: bool = Field(
        default=True,
        description="Whether to include ITC in the structure",
    )
    include_ptc: bool = Field(
        default=False,
        description="Whether to include PTC in the structure",
    )
    include_depreciation: bool = Field(
        default=True,
        description="Whether to include depreciation benefits",
    )


class TaxEquityResult(FinancialBase):
    """Result of tax equity modeling calculation.

    Attributes:
        flip_year: Year when partnership flip occurs.
        investor_irr: Calculated investor IRR.
        sponsor_irr: Calculated sponsor/developer IRR.
        investor_npv: Net present value to investor.
        sponsor_npv: Net present value to sponsor.
        total_investor_benefit: Total benefit to tax equity investor.
        total_sponsor_benefit: Total benefit to sponsor/developer.
        annual_cash_flows_investor: Annual cash flows to investor.
        annual_cash_flows_sponsor: Annual cash flows to sponsor.
        annual_tax_benefits_investor: Annual tax benefits to investor.
        annual_tax_benefits_sponsor: Annual tax benefits to sponsor.
        total_tax_benefits: Total tax benefits across all parties.
        pre_flip_years: Number of years in pre-flip period.
        post_flip_years: Number of years in post-flip period.
        calculation_details: Detailed breakdown of calculation.
    """

    flip_year: int = Field(
        ...,
        ge=0,
        description="Year when partnership flip occurs",
    )
    investor_irr: float = Field(
        ...,
        ge=-1.0,
        le=5.0,
        description="Calculated investor IRR",
    )
    sponsor_irr: float = Field(
        ...,
        ge=-1.0,
        le=5.0,
        description="Calculated sponsor/developer IRR",
    )
    investor_npv: float = Field(
        ...,
        description="Net present value to investor",
    )
    sponsor_npv: float = Field(
        ...,
        description="Net present value to sponsor",
    )
    total_investor_benefit: float = Field(
        ...,
        ge=0,
        description="Total benefit to tax equity investor",
    )
    total_sponsor_benefit: float = Field(
        ...,
        ge=0,
        description="Total benefit to sponsor/developer",
    )
    annual_cash_flows_investor: list[float] = Field(
        ...,
        min_length=1,
        description="Annual cash flows to investor",
    )
    annual_cash_flows_sponsor: list[float] = Field(
        ...,
        min_length=1,
        description="Annual cash flows to sponsor",
    )
    annual_tax_benefits_investor: list[float] = Field(
        ...,
        min_length=1,
        description="Annual tax benefits to investor",
    )
    annual_tax_benefits_sponsor: list[float] = Field(
        ...,
        min_length=1,
        description="Annual tax benefits to sponsor",
    )
    total_tax_benefits: float = Field(
        ...,
        ge=0,
        description="Total tax benefits across all parties",
    )
    pre_flip_years: int = Field(
        ...,
        ge=0,
        description="Number of years in pre-flip period",
    )
    post_flip_years: int = Field(
        ...,
        ge=0,
        description="Number of years in post-flip period",
    )
    calculation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of calculation",
    )

    @model_validator(mode="after")
    def validate_annual_flows(self) -> "TaxEquityResult":
        """Validate consistency of annual flow arrays."""
        list_len = len(self.annual_cash_flows_investor)
        if (
            len(self.annual_cash_flows_sponsor) != list_len
            or len(self.annual_tax_benefits_investor) != list_len
            or len(self.annual_tax_benefits_sponsor) != list_len
        ):
            raise ValueError("All annual flow lists must have the same length")
        if list_len != (self.pre_flip_years + self.post_flip_years):
            raise ValueError(
                "Total list length must equal pre_flip_years + post_flip_years"
            )
        return self
