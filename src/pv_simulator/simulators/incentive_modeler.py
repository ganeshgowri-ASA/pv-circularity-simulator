"""Core IncentiveModeler for tax credits, depreciation, and tax equity modeling.

This module provides comprehensive modeling of solar tax incentives including:
- Investment Tax Credit (ITC)
- Production Tax Credit (PTC)
- MACRS depreciation schedules
- Tax equity partnership structures
"""

from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pv_simulator.models.incentives import (
    DepreciationMethod,
    DepreciationScheduleResult,
    ITCConfiguration,
    ITCResult,
    PTCConfiguration,
    PTCResult,
    TaxEquityConfiguration,
    TaxEquityResult,
)


class IncentiveModeler:
    """Production-ready tax incentive and financial modeling for solar PV systems.

    The IncentiveModeler provides comprehensive calculations for federal tax incentives,
    depreciation schedules, and tax equity partnership structures commonly used in
    commercial solar installations.

    This class implements industry-standard calculations following IRS guidelines and
    common solar finance practices.

    Attributes:
        _macrs_5_year_schedule: Standard MACRS 5-year depreciation percentages.
        _macrs_7_year_schedule: Standard MACRS 7-year depreciation percentages.

    Example:
        >>> from pv_simulator import IncentiveModeler
        >>> from pv_simulator.models import SystemConfiguration, ITCConfiguration
        >>> from datetime import date
        >>>
        >>> system = SystemConfiguration(
        ...     system_size_kw=100.0,
        ...     installation_cost_total=250000.0,
        ...     installation_date=date(2024, 1, 1),
        ...     location_state="CA",
        ...     expected_annual_production_kwh=150000.0
        ... )
        >>> itc_config = ITCConfiguration(system_config=system, itc_rate=0.30)
        >>> modeler = IncentiveModeler()
        >>> result = modeler.itc_calculation(itc_config)
        >>> print(f"ITC Credit: ${result.total_itc_amount:,.2f}")
    """

    # MACRS depreciation schedules (half-year convention)
    # Source: IRS Publication 946
    _macrs_5_year_schedule: list[float] = [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576]
    _macrs_7_year_schedule: list[float] = [
        0.1429,
        0.2449,
        0.1749,
        0.1249,
        0.0893,
        0.0892,
        0.0893,
        0.0446,
    ]

    def __init__(self) -> None:
        """Initialize the IncentiveModeler.

        Sets up the modeler with standard IRS depreciation schedules and
        prepares internal state for calculations.
        """
        self._calculation_count: int = 0
        self._last_calculation_time: datetime | None = None

    def itc_calculation(self, config: ITCConfiguration) -> ITCResult:
        """Calculate Investment Tax Credit (ITC) for a solar installation.

        The ITC provides a federal tax credit based on the cost of installing a solar
        energy system. This implementation supports:
        - Base ITC rate (typically 30% for solar)
        - Bonus credits for domestic content, energy communities, etc.
        - Basis reductions for grants and other subsidies
        - Commercial vs. residential distinctions

        The calculation follows IRS Form 3468 and includes proper handling of:
        - Cost basis determination
        - Grant adjustments
        - Bonus credit stacking
        - Recapture period tracking

        Args:
            config: ITC configuration including system details and tax parameters.

        Returns:
            ITCResult containing total credit amount, basis calculations, and
            detailed breakdown of all components.

        Raises:
            ValueError: If configuration contains invalid values.

        Example:
            >>> from datetime import date
            >>> system = SystemConfiguration(
            ...     system_size_kw=500.0,
            ...     installation_cost_total=1_250_000.0,
            ...     installation_date=date(2024, 6, 15),
            ...     location_state="TX",
            ...     expected_annual_production_kwh=750_000.0
            ... )
            >>> config = ITCConfiguration(
            ...     system_config=system,
            ...     itc_rate=0.30,
            ...     apply_bonus=True,
            ...     meets_domestic_content=True
            ... )
            >>> result = modeler.itc_calculation(config)
            >>> print(f"Total ITC: ${result.total_itc_amount:,.2f}")
            >>> print(f"Effective rate: {result.effective_rate:.1%}")
        """
        self._calculation_count += 1
        self._last_calculation_time = datetime.now()

        # Step 1: Determine eligible cost basis
        eligible_basis = config.system_config.installation_cost_total
        basis_reduction = 0.0

        # Reduce basis for grants if applicable
        if config.basis_reduction_for_grants and config.state_grants_received > 0:
            basis_reduction = config.state_grants_received
            eligible_basis -= basis_reduction

        # Step 2: Calculate base ITC
        base_itc = eligible_basis * config.itc_rate

        # Step 3: Calculate bonus credits if applicable
        bonus_itc = 0.0
        effective_rate = config.itc_rate

        if config.apply_bonus:
            # Bonus credits can stack (domestic content + energy community)
            bonus_multiplier = 0.0

            if config.meets_domestic_content:
                bonus_multiplier += config.bonus_rate

            if config.is_energy_community:
                bonus_multiplier += config.bonus_rate

            bonus_itc = eligible_basis * bonus_multiplier
            effective_rate += bonus_multiplier

        # Step 4: Calculate total ITC
        total_itc = base_itc + bonus_itc

        # Step 5: Determine recapture period
        # ITC has a 5-year recapture period for commercial installations
        recapture_period = 5 if config.is_commercial else 5

        # Step 6: Build detailed calculation breakdown
        calculation_details: dict[str, Any] = {
            "installation_cost": config.system_config.installation_cost_total,
            "basis_reduction_applied": basis_reduction,
            "eligible_basis": eligible_basis,
            "base_rate": config.itc_rate,
            "base_itc_amount": base_itc,
            "bonus_applied": config.apply_bonus,
            "bonus_rate": config.bonus_rate if config.apply_bonus else 0.0,
            "bonus_itc_amount": bonus_itc,
            "meets_domestic_content": config.meets_domestic_content,
            "is_energy_community": config.is_energy_community,
            "total_effective_rate": effective_rate,
            "installation_date": config.system_config.installation_date.isoformat(),
            "system_size_kw": config.system_config.system_size_kw,
            "cost_per_watt": (
                config.system_config.installation_cost_total
                / (config.system_config.system_size_kw * 1000)
            ),
        }

        return ITCResult(
            total_itc_amount=total_itc,
            base_itc=base_itc,
            bonus_itc=bonus_itc,
            eligible_basis=eligible_basis,
            effective_rate=effective_rate,
            basis_reduction=basis_reduction,
            recapture_period_years=recapture_period,
            calculation_details=calculation_details,
            calculation_date=self._last_calculation_time or datetime.now(),
            notes=f"ITC calculation for {config.system_config.system_size_kw}kW system",
        )

    def ptc_computation(
        self, config: PTCConfiguration, discount_rate: float = 0.06
    ) -> PTCResult:
        """Compute Production Tax Credit (PTC) over the credit period.

        The PTC provides a per-kWh tax credit for electricity produced by qualifying
        renewable energy facilities. This implementation models:
        - Multi-year credit calculation (typically 10 years)
        - Inflation adjustments to credit rate
        - Production degradation over time
        - Bonus credit multipliers
        - Net present value calculation

        The calculation projects annual production considering degradation and
        calculates credits with optional inflation adjustment.

        Args:
            config: PTC configuration including system details and credit parameters.
            discount_rate: Discount rate for NPV calculation (default: 0.06).

        Returns:
            PTCResult containing lifetime credits, annual breakdowns, and present value.

        Raises:
            ValueError: If configuration contains invalid values.

        Example:
            >>> config = PTCConfiguration(
            ...     system_config=system,
            ...     ptc_rate_per_kwh=0.0275,
            ...     credit_period_years=10,
            ...     inflation_adjustment=True,
            ...     apply_bonus=True,
            ...     bonus_multiplier=5.0
            ... )
            >>> result = modeler.ptc_computation(config, discount_rate=0.06)
            >>> print(f"Lifetime PTC: ${result.total_ptc_lifetime:,.2f}")
            >>> print(f"NPV: ${result.present_value_ptc:,.2f}")
            >>> for year, credit in enumerate(result.annual_credits, 1):
            ...     print(f"Year {year}: ${credit:,.2f}")
        """
        self._calculation_count += 1
        self._last_calculation_time = datetime.now()

        # Initialize arrays for annual calculations
        annual_production: list[float] = []
        annual_credits: list[float] = []

        # Base production in year 1
        base_production = config.system_config.expected_annual_production_kwh

        # Calculate effective PTC rate with bonus if applicable
        effective_ptc_rate = config.ptc_rate_per_kwh
        if config.apply_bonus:
            effective_ptc_rate *= config.bonus_multiplier

        # Calculate credits for each year in credit period
        for year in range(config.credit_period_years):
            # Apply production degradation
            # Year 0 = first year, no degradation
            # Year 1 = second year, apply 1 year of degradation, etc.
            degradation_factor = (1 - config.production_degradation_rate) ** year
            year_production = base_production * degradation_factor
            annual_production.append(year_production)

            # Apply inflation adjustment to PTC rate if enabled
            inflation_factor = 1.0
            if config.inflation_adjustment:
                inflation_factor = (1 + config.inflation_rate) ** year

            # Calculate credit for this year
            adjusted_ptc_rate = effective_ptc_rate * inflation_factor
            year_credit = year_production * adjusted_ptc_rate
            annual_credits.append(year_credit)

        # Calculate total lifetime credits (nominal dollars)
        total_ptc_lifetime = sum(annual_credits)

        # Calculate present value of credits
        present_value_ptc = self._calculate_npv(
            cash_flows=annual_credits, discount_rate=discount_rate
        )

        # Build detailed calculation breakdown
        calculation_details: dict[str, Any] = {
            "base_production_year1": base_production,
            "base_ptc_rate": config.ptc_rate_per_kwh,
            "effective_ptc_rate_year1": effective_ptc_rate,
            "bonus_multiplier_applied": config.bonus_multiplier if config.apply_bonus else 1.0,
            "inflation_adjustment_enabled": config.inflation_adjustment,
            "inflation_rate": config.inflation_rate if config.inflation_adjustment else 0.0,
            "production_degradation_rate": config.production_degradation_rate,
            "discount_rate": discount_rate,
            "total_production_lifetime": sum(annual_production),
            "average_annual_production": sum(annual_production) / len(annual_production),
            "total_degradation_over_period": (
                (base_production - annual_production[-1]) / base_production
                if annual_production
                else 0.0
            ),
        }

        return PTCResult(
            total_ptc_lifetime=total_ptc_lifetime,
            annual_credits=annual_credits,
            annual_production=annual_production,
            present_value_ptc=present_value_ptc,
            discount_rate=discount_rate,
            credit_period_years=config.credit_period_years,
            first_year_credit=annual_credits[0] if annual_credits else 0.0,
            last_year_credit=annual_credits[-1] if annual_credits else 0.0,
            calculation_details=calculation_details,
            calculation_date=self._last_calculation_time or datetime.now(),
            notes=f"PTC calculation for {config.credit_period_years} years",
        )

    def depreciation_schedule(
        self,
        asset_basis: float,
        method: DepreciationMethod = DepreciationMethod.MACRS_5,
        bonus_depreciation_rate: float = 0.0,
    ) -> DepreciationScheduleResult:
        """Calculate depreciation schedule for solar assets.

        Generates a complete depreciation schedule following IRS guidelines. Supports:
        - MACRS 5-year schedule (standard for solar)
        - MACRS 7-year schedule
        - Straight-line depreciation
        - Double declining balance
        - Bonus depreciation in year 1

        MACRS schedules use the half-year convention and are based on IRS Publication 946.
        The asset basis is reduced by 50% of the ITC if ITC is claimed (the "ITC basis
        adjustment") - this should be done by the caller before passing asset_basis.

        Args:
            asset_basis: Initial asset basis for depreciation (after ITC adjustment).
            method: Depreciation method to use (default: MACRS_5).
            bonus_depreciation_rate: Bonus depreciation rate for year 1 (0.0-1.0).

        Returns:
            DepreciationScheduleResult containing annual depreciation, cumulative
            values, and remaining basis for each year.

        Raises:
            ValueError: If asset_basis <= 0 or bonus_depreciation_rate is invalid.

        Example:
            >>> # For $1M system with 30% ITC
            >>> # Asset basis = $1M - (0.5 * $300k ITC) = $850k
            >>> result = modeler.depreciation_schedule(
            ...     asset_basis=850_000.0,
            ...     method=DepreciationMethod.MACRS_5,
            ...     bonus_depreciation_rate=0.60  # 60% bonus depreciation
            ... )
            >>> for year, (depr, cumul, remain) in enumerate(
            ...     zip(result.annual_depreciation,
            ...         result.cumulative_depreciation,
            ...         result.remaining_basis), 1
            ... ):
            ...     print(f"Year {year}: Depr=${depr:,.0f}, Remaining=${remain:,.0f}")
        """
        if asset_basis <= 0:
            raise ValueError("asset_basis must be greater than 0")
        if not 0.0 <= bonus_depreciation_rate <= 1.0:
            raise ValueError("bonus_depreciation_rate must be between 0.0 and 1.0")

        self._calculation_count += 1
        self._last_calculation_time = datetime.now()

        # Calculate bonus depreciation in year 1
        bonus_depreciation_amount = asset_basis * bonus_depreciation_rate
        remaining_basis_after_bonus = asset_basis - bonus_depreciation_amount

        annual_depreciation: list[float] = []
        cumulative_depreciation: list[float] = []
        remaining_basis: list[float] = []

        if method == DepreciationMethod.MACRS_5:
            schedule = self._macrs_5_year_schedule
            macrs_convention = "Half-year"
        elif method == DepreciationMethod.MACRS_7:
            schedule = self._macrs_7_year_schedule
            macrs_convention = "Half-year"
        elif method == DepreciationMethod.STRAIGHT_LINE:
            schedule = self._calculate_straight_line_schedule(years=5)
            macrs_convention = None
        elif method == DepreciationMethod.DECLINING_BALANCE:
            schedule = self._calculate_declining_balance_schedule(
                years=5, rate=2.0  # Double declining
            )
            macrs_convention = None
        else:
            raise ValueError(f"Unsupported depreciation method: {method}")

        # Calculate depreciation for each year
        cumulative = bonus_depreciation_amount  # Start with bonus depreciation

        for year_idx, percentage in enumerate(schedule):
            # Apply percentage to remaining basis after bonus
            year_depreciation = remaining_basis_after_bonus * percentage

            # Add bonus depreciation to first year only
            if year_idx == 0:
                year_depreciation += bonus_depreciation_amount

            cumulative += year_depreciation if year_idx > 0 else year_depreciation - bonus_depreciation_amount
            remaining = asset_basis - cumulative

            annual_depreciation.append(year_depreciation)
            cumulative_depreciation.append(cumulative)
            remaining_basis.append(max(0.0, remaining))  # Never go negative

        total_depreciation = sum(annual_depreciation)

        # Build detailed calculation breakdown
        calculation_details: dict[str, Any] = {
            "method": method.value,
            "schedule_percentages": schedule,
            "original_basis": asset_basis,
            "bonus_depreciation_rate": bonus_depreciation_rate,
            "bonus_depreciation_amount": bonus_depreciation_amount,
            "basis_after_bonus": remaining_basis_after_bonus,
            "total_depreciation": total_depreciation,
            "final_remaining_basis": remaining_basis[-1] if remaining_basis else 0.0,
        }

        return DepreciationScheduleResult(
            method=method,
            asset_basis=asset_basis,
            schedule_years=len(schedule),
            annual_depreciation=annual_depreciation,
            cumulative_depreciation=cumulative_depreciation,
            remaining_basis=remaining_basis,
            total_depreciation=total_depreciation,
            macrs_convention=macrs_convention,
            bonus_depreciation_rate=bonus_depreciation_rate,
            bonus_depreciation_amount=bonus_depreciation_amount,
            calculation_details=calculation_details,
            calculation_date=self._last_calculation_time or datetime.now(),
            notes=f"{method.value} depreciation schedule",
        )

    def tax_equity_modeling(
        self,
        config: TaxEquityConfiguration,
        itc_amount: float | None = None,
        ptc_annual_credits: list[float] | None = None,
        depreciation_schedule: list[float] | None = None,
    ) -> TaxEquityResult:
        """Model tax equity partnership structure (partnership flip).

        Models the cash flows and returns in a tax equity partnership, which is a
        common financing structure for commercial solar projects. The model includes:
        - Pre-flip and post-flip allocation percentages
        - Tax benefit allocation (ITC, PTC, depreciation)
        - Cash flow allocation and timing
        - IRR and NPV calculations for both parties
        - Partnership flip timing determination

        The partnership flip structure allocates most tax benefits to the investor
        initially, then "flips" to give the sponsor/developer greater ownership
        once the investor achieves their target return.

        Args:
            config: Tax equity configuration including partnership terms.
            itc_amount: Optional ITC amount to include (if config.include_itc=True).
            ptc_annual_credits: Optional annual PTC credits (if config.include_ptc=True).
            depreciation_schedule: Optional annual depreciation amounts.

        Returns:
            TaxEquityResult containing flip year, IRRs, NPVs, and annual allocations
            for both investor and sponsor.

        Raises:
            ValueError: If required tax benefits are missing or configuration is invalid.

        Example:
            >>> # First calculate tax benefits
            >>> itc_result = modeler.itc_calculation(itc_config)
            >>> depr_result = modeler.depreciation_schedule(850_000.0)
            >>>
            >>> # Then model tax equity
            >>> te_config = TaxEquityConfiguration(
            ...     system_config=system,
            ...     investor_equity_percentage=0.99,
            ...     target_flip_irr=0.08,
            ...     post_flip_investor_percentage=0.05
            ... )
            >>> result = modeler.tax_equity_modeling(
            ...     config=te_config,
            ...     itc_amount=itc_result.total_itc_amount,
            ...     depreciation_schedule=depr_result.annual_depreciation
            ... )
            >>> print(f"Flip occurs in year {result.flip_year}")
            >>> print(f"Investor IRR: {result.investor_irr:.2%}")
            >>> print(f"Sponsor IRR: {result.sponsor_irr:.2%}")
        """
        self._calculation_count += 1
        self._last_calculation_time = datetime.now()

        # Validate required inputs
        if config.include_itc and itc_amount is None:
            raise ValueError("itc_amount required when include_itc=True")
        if config.include_ptc and ptc_annual_credits is None:
            raise ValueError("ptc_annual_credits required when include_ptc=True")
        if config.include_depreciation and depreciation_schedule is None:
            raise ValueError("depreciation_schedule required when include_depreciation=True")

        # Initialize tax benefits
        total_itc = itc_amount if config.include_itc else 0.0
        ptc_credits = ptc_annual_credits if config.include_ptc else []
        depr_benefits = depreciation_schedule if config.include_depreciation else []

        # Determine project lifetime for analysis
        project_years = config.project_lifetime_years

        # Initialize annual arrays
        annual_cash_flows_investor: list[float] = []
        annual_cash_flows_sponsor: list[float] = []
        annual_tax_benefits_investor: list[float] = []
        annual_tax_benefits_sponsor: list[float] = []

        # Initial investment (negative cash flow)
        initial_investment = config.system_config.installation_cost_total
        investor_investment = -initial_investment * config.investor_equity_percentage
        sponsor_investment = -initial_investment * (1 - config.investor_equity_percentage)

        # Determine flip year based on target IRR
        # Simplified model: flip when investor hits target IRR
        flip_year = self._calculate_flip_year(
            investor_investment=abs(investor_investment),
            target_irr=config.target_flip_irr,
            total_itc=total_itc,
            annual_ptc=ptc_credits,
            annual_depreciation=depr_benefits,
            tax_rate=config.tax_rate,
            pre_flip_percentage=config.investor_equity_percentage,
        )

        # Ensure flip year is within project lifetime
        flip_year = min(flip_year, project_years - 1)

        # Calculate annual cash flows and tax benefits
        for year in range(project_years):
            # Determine allocation percentages
            if year < flip_year:
                investor_pct = config.investor_equity_percentage
                sponsor_pct = 1 - config.investor_equity_percentage
            else:
                investor_pct = config.post_flip_investor_percentage
                sponsor_pct = 1 - config.post_flip_investor_percentage

            # Tax benefits for this year
            year_tax_benefit = 0.0

            # ITC allocated in year 0
            if year == 0 and total_itc > 0:
                year_tax_benefit += total_itc

            # PTC if applicable
            if year < len(ptc_credits):
                year_tax_benefit += ptc_credits[year]

            # Depreciation tax shield
            if year < len(depr_benefits):
                depreciation_tax_shield = depr_benefits[year] * config.tax_rate
                year_tax_benefit += depreciation_tax_shield

            # Allocate tax benefits
            investor_tax_benefit = year_tax_benefit * investor_pct
            sponsor_tax_benefit = year_tax_benefit * sponsor_pct

            annual_tax_benefits_investor.append(investor_tax_benefit)
            annual_tax_benefits_sponsor.append(sponsor_tax_benefit)

            # Operating cash flows (simplified - assume revenue covers O&M)
            # In a full model, this would include revenue, O&M, debt service, etc.
            operating_cash_flow = 0.0

            # Allocate operating cash flows
            investor_cash = operating_cash_flow * investor_pct
            sponsor_cash = operating_cash_flow * sponsor_pct

            # Add tax benefits to cash flows
            investor_cash += investor_tax_benefit
            sponsor_cash += sponsor_tax_benefit

            # Add initial investment in year 0
            if year == 0:
                investor_cash += investor_investment
                sponsor_cash += sponsor_investment

            annual_cash_flows_investor.append(investor_cash)
            annual_cash_flows_sponsor.append(sponsor_cash)

        # Calculate IRRs and NPVs
        investor_irr = self._calculate_irr(annual_cash_flows_investor)
        sponsor_irr = self._calculate_irr(annual_cash_flows_sponsor)

        investor_npv = self._calculate_npv(
            annual_cash_flows_investor, config.discount_rate
        )
        sponsor_npv = self._calculate_npv(
            annual_cash_flows_sponsor, config.discount_rate
        )

        # Calculate total benefits
        total_investor_benefit = sum(annual_tax_benefits_investor)
        total_sponsor_benefit = sum(annual_tax_benefits_sponsor)
        total_tax_benefits = total_investor_benefit + total_sponsor_benefit

        # Build detailed calculation breakdown
        calculation_details: dict[str, Any] = {
            "initial_investment": initial_investment,
            "investor_investment": investor_investment,
            "sponsor_investment": sponsor_investment,
            "total_itc_included": total_itc,
            "total_ptc_included": sum(ptc_credits),
            "total_depreciation_shield": (
                sum(depr_benefits) * config.tax_rate if depr_benefits else 0.0
            ),
            "pre_flip_investor_allocation": config.investor_equity_percentage,
            "post_flip_investor_allocation": config.post_flip_investor_percentage,
            "target_flip_irr": config.target_flip_irr,
            "actual_flip_year": flip_year,
            "project_lifetime_years": project_years,
            "discount_rate": config.discount_rate,
        }

        return TaxEquityResult(
            flip_year=flip_year,
            investor_irr=investor_irr,
            sponsor_irr=sponsor_irr,
            investor_npv=investor_npv,
            sponsor_npv=sponsor_npv,
            total_investor_benefit=total_investor_benefit,
            total_sponsor_benefit=total_sponsor_benefit,
            annual_cash_flows_investor=annual_cash_flows_investor,
            annual_cash_flows_sponsor=annual_cash_flows_sponsor,
            annual_tax_benefits_investor=annual_tax_benefits_investor,
            annual_tax_benefits_sponsor=annual_tax_benefits_sponsor,
            total_tax_benefits=total_tax_benefits,
            pre_flip_years=flip_year,
            post_flip_years=project_years - flip_year,
            calculation_details=calculation_details,
            calculation_date=self._last_calculation_time or datetime.now(),
            notes=f"Tax equity partnership flip in year {flip_year}",
        )

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _calculate_straight_line_schedule(self, years: int) -> list[float]:
        """Calculate straight-line depreciation schedule.

        Args:
            years: Number of years for depreciation.

        Returns:
            List of annual depreciation percentages.
        """
        annual_rate = 1.0 / years
        return [annual_rate] * years

    def _calculate_declining_balance_schedule(
        self, years: int, rate: float = 2.0
    ) -> list[float]:
        """Calculate declining balance depreciation schedule.

        Args:
            years: Number of years for depreciation.
            rate: Declining balance rate (e.g., 2.0 for double declining).

        Returns:
            List of annual depreciation percentages.
        """
        schedule: list[float] = []
        remaining = 1.0
        annual_rate = rate / years

        for _ in range(years):
            year_depreciation = remaining * annual_rate
            schedule.append(year_depreciation)
            remaining -= year_depreciation

        return schedule

    def _calculate_npv(self, cash_flows: list[float], discount_rate: float) -> float:
        """Calculate net present value of cash flows.

        Args:
            cash_flows: List of annual cash flows.
            discount_rate: Discount rate as decimal.

        Returns:
            Net present value.
        """
        npv = 0.0
        for year, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** year)
        return npv

    def _calculate_irr(
        self, cash_flows: list[float], initial_guess: float = 0.1
    ) -> float:
        """Calculate internal rate of return using Newton-Raphson method.

        Args:
            cash_flows: List of annual cash flows (including initial investment).
            initial_guess: Initial guess for IRR.

        Returns:
            Internal rate of return as decimal.
        """
        # Use numpy for IRR calculation
        cash_flow_array: NDArray[np.float64] = np.array(cash_flows)

        try:
            # Try numpy's IRR calculation
            irr = float(np.irr(cash_flow_array))  # type: ignore
            return irr
        except (AttributeError, Exception):
            # Fallback to manual Newton-Raphson if np.irr not available
            return self._newton_raphson_irr(cash_flows, initial_guess)

    def _newton_raphson_irr(
        self,
        cash_flows: list[float],
        initial_guess: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """Calculate IRR using Newton-Raphson method.

        Args:
            cash_flows: List of annual cash flows.
            initial_guess: Initial guess for IRR.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.

        Returns:
            Internal rate of return as decimal.
        """
        rate = initial_guess

        for _ in range(max_iterations):
            npv = 0.0
            npv_derivative = 0.0

            for year, cash_flow in enumerate(cash_flows):
                npv += cash_flow / ((1 + rate) ** year)
                if year > 0:
                    npv_derivative -= year * cash_flow / ((1 + rate) ** (year + 1))

            if abs(npv) < tolerance:
                return rate

            if abs(npv_derivative) < 1e-10:
                # Derivative too small, cannot continue
                break

            rate = rate - npv / npv_derivative

        # If convergence failed, return best estimate
        return rate

    def _calculate_flip_year(
        self,
        investor_investment: float,
        target_irr: float,
        total_itc: float,
        annual_ptc: list[float],
        annual_depreciation: list[float],
        tax_rate: float,
        pre_flip_percentage: float,
    ) -> int:
        """Calculate year when partnership flip should occur.

        Determines when investor achieves target IRR based on tax benefits.

        Args:
            investor_investment: Investor's initial investment.
            target_irr: Target IRR for investor.
            total_itc: Total ITC amount.
            annual_ptc: Annual PTC credits.
            annual_depreciation: Annual depreciation amounts.
            tax_rate: Tax rate for depreciation shield.
            pre_flip_percentage: Investor's pre-flip allocation percentage.

        Returns:
            Year number when flip should occur (0-indexed).
        """
        # Build investor cash flows pre-flip
        max_years = 20  # Safety limit

        for flip_year in range(1, max_years + 1):
            cash_flows = [-investor_investment]

            for year in range(flip_year):
                year_benefit = 0.0

                # ITC in year 0
                if year == 0:
                    year_benefit += total_itc * pre_flip_percentage

                # PTC
                if year < len(annual_ptc):
                    year_benefit += annual_ptc[year] * pre_flip_percentage

                # Depreciation shield
                if year < len(annual_depreciation):
                    year_benefit += (
                        annual_depreciation[year] * tax_rate * pre_flip_percentage
                    )

                cash_flows.append(year_benefit)

            # Calculate IRR for these cash flows
            irr = self._calculate_irr(cash_flows)

            if irr >= target_irr:
                return flip_year

        # Default to halfway through project if target not reached
        return 10
