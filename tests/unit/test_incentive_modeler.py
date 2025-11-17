"""Comprehensive unit tests for IncentiveModeler class."""

from datetime import date

import pytest

from pv_simulator.models.incentives import (
    DepreciationMethod,
    ITCConfiguration,
    PTCConfiguration,
    SystemConfiguration,
    TaxEquityConfiguration,
)
from pv_simulator.simulators.incentive_modeler import IncentiveModeler


class TestITCCalculation:
    """Test suite for ITC calculation functionality."""

    def test_basic_itc_calculation(
        self, incentive_modeler: IncentiveModeler, itc_config_basic: ITCConfiguration
    ) -> None:
        """Test basic ITC calculation with standard 30% rate."""
        result = incentive_modeler.itc_calculation(itc_config_basic)

        # With $250k installation cost and 30% rate
        expected_itc = 250_000 * 0.30
        assert result.total_itc_amount == pytest.approx(expected_itc)
        assert result.base_itc == pytest.approx(expected_itc)
        assert result.bonus_itc == 0.0
        assert result.effective_rate == pytest.approx(0.30)
        assert result.eligible_basis == pytest.approx(250_000.0)
        assert result.recapture_period_years == 5

    def test_itc_with_bonus_credits(
        self, incentive_modeler: IncentiveModeler, itc_config_with_bonus: ITCConfiguration
    ) -> None:
        """Test ITC calculation with bonus credits for domestic content."""
        result = incentive_modeler.itc_calculation(itc_config_with_bonus)

        # Base: $250k * 0.30 = $75k
        # Bonus: $250k * 0.10 = $25k
        # Total: $100k
        expected_base = 250_000 * 0.30
        expected_bonus = 250_000 * 0.10
        expected_total = expected_base + expected_bonus

        assert result.base_itc == pytest.approx(expected_base)
        assert result.bonus_itc == pytest.approx(expected_bonus)
        assert result.total_itc_amount == pytest.approx(expected_total)
        assert result.effective_rate == pytest.approx(0.40)  # 30% + 10%

    def test_itc_with_grant_reduction(
        self, incentive_modeler: IncentiveModeler, sample_system_config: SystemConfiguration
    ) -> None:
        """Test ITC calculation with basis reduction for state grants."""
        config = ITCConfiguration(
            system_config=sample_system_config,
            itc_rate=0.30,
            basis_reduction_for_grants=True,
            state_grants_received=50_000.0,
        )

        result = incentive_modeler.itc_calculation(config)

        # Eligible basis: $250k - $50k = $200k
        # ITC: $200k * 0.30 = $60k
        expected_basis = 200_000.0
        expected_itc = expected_basis * 0.30

        assert result.eligible_basis == pytest.approx(expected_basis)
        assert result.basis_reduction == pytest.approx(50_000.0)
        assert result.total_itc_amount == pytest.approx(expected_itc)

    def test_itc_with_energy_community_bonus(
        self, incentive_modeler: IncentiveModeler, sample_system_config: SystemConfiguration
    ) -> None:
        """Test ITC with energy community bonus stacking."""
        config = ITCConfiguration(
            system_config=sample_system_config,
            itc_rate=0.30,
            apply_bonus=True,
            bonus_rate=0.10,
            meets_domestic_content=True,
            is_energy_community=True,
        )

        result = incentive_modeler.itc_calculation(config)

        # Base: 30%
        # Domestic content: +10%
        # Energy community: +10%
        # Total: 50%
        expected_rate = 0.30 + 0.10 + 0.10
        expected_itc = 250_000 * expected_rate

        assert result.effective_rate == pytest.approx(expected_rate)
        assert result.total_itc_amount == pytest.approx(expected_itc)

    def test_itc_result_has_calculation_details(
        self, incentive_modeler: IncentiveModeler, itc_config_basic: ITCConfiguration
    ) -> None:
        """Test that ITC result includes detailed calculation breakdown."""
        result = incentive_modeler.itc_calculation(itc_config_basic)

        assert "installation_cost" in result.calculation_details
        assert "eligible_basis" in result.calculation_details
        assert "base_itc_amount" in result.calculation_details
        assert "total_effective_rate" in result.calculation_details
        assert "system_size_kw" in result.calculation_details
        assert result.calculation_date is not None
        assert result.notes is not None


class TestPTCComputation:
    """Test suite for PTC computation functionality."""

    def test_basic_ptc_computation(
        self, incentive_modeler: IncentiveModeler, ptc_config_basic: PTCConfiguration
    ) -> None:
        """Test basic PTC computation over 10 years."""
        result = incentive_modeler.ptc_computation(ptc_config_basic)

        assert result.credit_period_years == 10
        assert len(result.annual_credits) == 10
        assert len(result.annual_production) == 10
        assert result.total_ptc_lifetime > 0
        assert result.present_value_ptc > 0
        assert result.present_value_ptc < result.total_ptc_lifetime  # Due to discounting
        assert result.first_year_credit > result.last_year_credit  # Due to degradation

    def test_ptc_production_degradation(
        self, incentive_modeler: IncentiveModeler, ptc_config_basic: PTCConfiguration
    ) -> None:
        """Test that PTC correctly applies production degradation."""
        result = incentive_modeler.ptc_computation(ptc_config_basic)

        # Production should decrease each year due to 0.5% degradation
        for i in range(len(result.annual_production) - 1):
            assert result.annual_production[i] > result.annual_production[i + 1]

        # First year should be at expected production
        assert result.annual_production[0] == pytest.approx(150_000.0)

        # Last year should be reduced by cumulative degradation
        # After 9 years of degradation: (1-0.005)^9 ≈ 0.956
        expected_last_year = 150_000.0 * (0.995**9)
        assert result.annual_production[-1] == pytest.approx(expected_last_year)

    def test_ptc_inflation_adjustment(
        self, incentive_modeler: IncentiveModeler, ptc_config_basic: PTCConfiguration
    ) -> None:
        """Test that PTC correctly applies inflation adjustment."""
        result = incentive_modeler.ptc_computation(ptc_config_basic)

        # Credit per kWh should increase with inflation
        # Year 0: production * base_rate
        # Year 1: production * base_rate * (1+inflation)
        base_rate = 0.0275
        year_0_credit_per_kwh = result.annual_credits[0] / result.annual_production[0]
        year_1_credit_per_kwh = result.annual_credits[1] / result.annual_production[1]

        assert year_0_credit_per_kwh == pytest.approx(base_rate, rel=1e-6)
        expected_year_1_rate = base_rate * 1.025
        assert year_1_credit_per_kwh == pytest.approx(expected_year_1_rate, rel=1e-3)

    def test_ptc_with_bonus_multiplier(
        self, incentive_modeler: IncentiveModeler, ptc_config_with_bonus: PTCConfiguration
    ) -> None:
        """Test PTC computation with 5x bonus multiplier."""
        # Calculate without bonus
        config_no_bonus = PTCConfiguration(
            system_config=ptc_config_with_bonus.system_config,
            ptc_rate_per_kwh=0.0275,
            credit_period_years=10,
            inflation_adjustment=False,
            production_degradation_rate=0.0,
            apply_bonus=False,
        )
        result_no_bonus = incentive_modeler.ptc_computation(config_no_bonus)

        # Calculate with 5x bonus
        config_with_bonus = PTCConfiguration(
            system_config=ptc_config_with_bonus.system_config,
            ptc_rate_per_kwh=0.0275,
            credit_period_years=10,
            inflation_adjustment=False,
            production_degradation_rate=0.0,
            apply_bonus=True,
            bonus_multiplier=5.0,
        )
        result_with_bonus = incentive_modeler.ptc_computation(config_with_bonus)

        # Total PTC with bonus should be 5x the base
        assert result_with_bonus.total_ptc_lifetime == pytest.approx(
            result_no_bonus.total_ptc_lifetime * 5.0
        )

    def test_ptc_npv_calculation(
        self, incentive_modeler: IncentiveModeler, ptc_config_basic: PTCConfiguration
    ) -> None:
        """Test that NPV is correctly calculated with discount rate."""
        discount_rate = 0.08
        result = incentive_modeler.ptc_computation(ptc_config_basic, discount_rate=discount_rate)

        # Manually calculate NPV
        manual_npv = sum(
            credit / ((1 + discount_rate) ** year)
            for year, credit in enumerate(result.annual_credits)
        )

        assert result.present_value_ptc == pytest.approx(manual_npv)
        assert result.discount_rate == discount_rate


class TestDepreciationSchedule:
    """Test suite for depreciation schedule functionality."""

    def test_macrs_5_year_schedule(self, incentive_modeler: IncentiveModeler) -> None:
        """Test MACRS 5-year depreciation schedule."""
        asset_basis = 1_000_000.0
        result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_5,
            bonus_depreciation_rate=0.0,
        )

        assert result.method == DepreciationMethod.MACRS_5
        assert result.asset_basis == asset_basis
        assert result.schedule_years == 6  # MACRS 5-year has 6 periods
        assert len(result.annual_depreciation) == 6
        assert result.macrs_convention == "Half-year"

        # Verify MACRS percentages sum to 100%
        total_percentage = sum(result.annual_depreciation) / asset_basis
        assert total_percentage == pytest.approx(1.0, rel=1e-6)

        # Verify cumulative depreciation
        assert result.cumulative_depreciation[-1] == pytest.approx(
            result.total_depreciation
        )

        # Verify remaining basis
        assert result.remaining_basis[0] > result.remaining_basis[-1]
        assert result.remaining_basis[-1] == pytest.approx(0.0, abs=1.0)

    def test_macrs_7_year_schedule(self, incentive_modeler: IncentiveModeler) -> None:
        """Test MACRS 7-year depreciation schedule."""
        asset_basis = 500_000.0
        result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_7,
            bonus_depreciation_rate=0.0,
        )

        assert result.method == DepreciationMethod.MACRS_7
        assert result.schedule_years == 8  # MACRS 7-year has 8 periods

        # Verify total depreciation equals asset basis
        assert result.total_depreciation == pytest.approx(asset_basis)

    def test_bonus_depreciation(self, incentive_modeler: IncentiveModeler) -> None:
        """Test depreciation with bonus depreciation in year 1."""
        asset_basis = 1_000_000.0
        bonus_rate = 0.60  # 60% bonus depreciation

        result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_5,
            bonus_depreciation_rate=bonus_rate,
        )

        expected_bonus = asset_basis * bonus_rate
        assert result.bonus_depreciation_amount == pytest.approx(expected_bonus)

        # First year should include bonus depreciation
        # First year = bonus + (remaining basis * first MACRS percentage)
        remaining_after_bonus = asset_basis - expected_bonus
        expected_first_year = expected_bonus + (remaining_after_bonus * 0.2000)
        assert result.annual_depreciation[0] == pytest.approx(expected_first_year)

        # Total depreciation should still equal asset basis
        assert result.total_depreciation == pytest.approx(asset_basis)

    def test_straight_line_depreciation(self, incentive_modeler: IncentiveModeler) -> None:
        """Test straight-line depreciation method."""
        asset_basis = 100_000.0
        result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.STRAIGHT_LINE,
            bonus_depreciation_rate=0.0,
        )

        assert result.method == DepreciationMethod.STRAIGHT_LINE
        assert result.macrs_convention is None

        # Each year should have equal depreciation (except potential rounding)
        expected_annual = asset_basis / result.schedule_years
        for annual_depr in result.annual_depreciation:
            assert annual_depr == pytest.approx(expected_annual, rel=1e-6)

    def test_declining_balance_depreciation(self, incentive_modeler: IncentiveModeler) -> None:
        """Test declining balance depreciation method."""
        asset_basis = 100_000.0
        result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.DECLINING_BALANCE,
            bonus_depreciation_rate=0.0,
        )

        assert result.method == DepreciationMethod.DECLINING_BALANCE

        # Depreciation should decrease each year
        for i in range(len(result.annual_depreciation) - 1):
            assert result.annual_depreciation[i] > result.annual_depreciation[i + 1]

    def test_depreciation_invalid_basis(self, incentive_modeler: IncentiveModeler) -> None:
        """Test that invalid asset basis raises ValueError."""
        with pytest.raises(ValueError, match="asset_basis must be greater than 0"):
            incentive_modeler.depreciation_schedule(
                asset_basis=0.0,
                method=DepreciationMethod.MACRS_5,
            )

        with pytest.raises(ValueError, match="asset_basis must be greater than 0"):
            incentive_modeler.depreciation_schedule(
                asset_basis=-1000.0,
                method=DepreciationMethod.MACRS_5,
            )

    def test_depreciation_invalid_bonus_rate(self, incentive_modeler: IncentiveModeler) -> None:
        """Test that invalid bonus depreciation rate raises ValueError."""
        with pytest.raises(ValueError, match="bonus_depreciation_rate must be between"):
            incentive_modeler.depreciation_schedule(
                asset_basis=100_000.0,
                method=DepreciationMethod.MACRS_5,
                bonus_depreciation_rate=1.5,  # Invalid: > 1.0
            )


class TestTaxEquityModeling:
    """Test suite for tax equity modeling functionality."""

    def test_basic_tax_equity_structure(
        self,
        incentive_modeler: IncentiveModeler,
        tax_equity_config: TaxEquityConfiguration,
    ) -> None:
        """Test basic tax equity partnership flip structure."""
        # Calculate ITC for the system
        itc_config = ITCConfiguration(
            system_config=tax_equity_config.system_config,
            itc_rate=0.30,
        )
        itc_result = incentive_modeler.itc_calculation(itc_config)

        # Calculate depreciation (with ITC basis adjustment)
        # Asset basis = installation cost - 50% of ITC
        asset_basis = (
            tax_equity_config.system_config.installation_cost_total
            - 0.5 * itc_result.total_itc_amount
        )
        depr_result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_5,
        )

        # Model tax equity
        result = incentive_modeler.tax_equity_modeling(
            config=tax_equity_config,
            itc_amount=itc_result.total_itc_amount,
            depreciation_schedule=depr_result.annual_depreciation,
        )

        # Verify basic structure
        assert result.flip_year > 0
        assert result.flip_year < tax_equity_config.project_lifetime_years
        assert result.pre_flip_years == result.flip_year
        assert (
            result.post_flip_years
            == tax_equity_config.project_lifetime_years - result.flip_year
        )

        # Verify cash flow arrays
        assert len(result.annual_cash_flows_investor) == tax_equity_config.project_lifetime_years
        assert len(result.annual_cash_flows_sponsor) == tax_equity_config.project_lifetime_years
        assert len(result.annual_tax_benefits_investor) == tax_equity_config.project_lifetime_years
        assert len(result.annual_tax_benefits_sponsor) == tax_equity_config.project_lifetime_years

        # Verify total benefits
        assert result.total_investor_benefit > 0
        assert result.total_sponsor_benefit > 0
        assert result.total_tax_benefits == pytest.approx(
            result.total_investor_benefit + result.total_sponsor_benefit
        )

    def test_tax_equity_investor_gets_most_benefits_pre_flip(
        self,
        incentive_modeler: IncentiveModeler,
        tax_equity_config: TaxEquityConfiguration,
    ) -> None:
        """Test that investor receives most tax benefits pre-flip."""
        itc_config = ITCConfiguration(
            system_config=tax_equity_config.system_config,
            itc_rate=0.30,
        )
        itc_result = incentive_modeler.itc_calculation(itc_config)

        asset_basis = (
            tax_equity_config.system_config.installation_cost_total
            - 0.5 * itc_result.total_itc_amount
        )
        depr_result = incentive_modeler.depreciation_schedule(asset_basis=asset_basis)

        result = incentive_modeler.tax_equity_modeling(
            config=tax_equity_config,
            itc_amount=itc_result.total_itc_amount,
            depreciation_schedule=depr_result.annual_depreciation,
        )

        # In pre-flip period, investor should get 99% of benefits
        for year in range(result.pre_flip_years):
            total_year_benefit = (
                result.annual_tax_benefits_investor[year]
                + result.annual_tax_benefits_sponsor[year]
            )
            if total_year_benefit > 0:  # Avoid division by zero
                investor_share = result.annual_tax_benefits_investor[year] / total_year_benefit
                assert investor_share == pytest.approx(0.99, rel=0.01)

    def test_tax_equity_allocation_flips(
        self,
        incentive_modeler: IncentiveModeler,
        tax_equity_config: TaxEquityConfiguration,
    ) -> None:
        """Test that allocation percentages flip at the flip year."""
        itc_config = ITCConfiguration(
            system_config=tax_equity_config.system_config,
            itc_rate=0.30,
        )
        itc_result = incentive_modeler.itc_calculation(itc_config)

        asset_basis = (
            tax_equity_config.system_config.installation_cost_total
            - 0.5 * itc_result.total_itc_amount
        )
        depr_result = incentive_modeler.depreciation_schedule(asset_basis=asset_basis)

        result = incentive_modeler.tax_equity_modeling(
            config=tax_equity_config,
            itc_amount=itc_result.total_itc_amount,
            depreciation_schedule=depr_result.annual_depreciation,
        )

        # Post-flip, investor should get 5% of benefits
        flip_year = result.flip_year
        if flip_year < len(result.annual_tax_benefits_investor) - 1:
            post_flip_year = flip_year + 1
            if post_flip_year < len(result.annual_tax_benefits_investor):
                total_benefit = (
                    result.annual_tax_benefits_investor[post_flip_year]
                    + result.annual_tax_benefits_sponsor[post_flip_year]
                )
                if total_benefit > 0:
                    investor_share = (
                        result.annual_tax_benefits_investor[post_flip_year] / total_benefit
                    )
                    assert investor_share == pytest.approx(0.05, rel=0.01)

    def test_tax_equity_missing_itc_raises_error(
        self,
        incentive_modeler: IncentiveModeler,
        tax_equity_config: TaxEquityConfiguration,
    ) -> None:
        """Test that missing ITC when required raises ValueError."""
        with pytest.raises(ValueError, match="itc_amount required when include_itc=True"):
            incentive_modeler.tax_equity_modeling(
                config=tax_equity_config,
                itc_amount=None,  # Missing ITC
            )

    def test_tax_equity_with_ptc(
        self,
        incentive_modeler: IncentiveModeler,
        sample_system_config: SystemConfiguration,
    ) -> None:
        """Test tax equity modeling with PTC instead of ITC."""
        # Configure for PTC
        ptc_config = PTCConfiguration(
            system_config=sample_system_config,
            ptc_rate_per_kwh=0.0275,
            credit_period_years=10,
        )
        ptc_result = incentive_modeler.ptc_computation(ptc_config)

        asset_basis = sample_system_config.installation_cost_total
        depr_result = incentive_modeler.depreciation_schedule(asset_basis=asset_basis)

        te_config = TaxEquityConfiguration(
            system_config=sample_system_config,
            investor_equity_percentage=0.99,
            target_flip_irr=0.08,
            include_itc=False,
            include_ptc=True,
            include_depreciation=True,
        )

        result = incentive_modeler.tax_equity_modeling(
            config=te_config,
            ptc_annual_credits=ptc_result.annual_credits,
            depreciation_schedule=depr_result.annual_depreciation,
        )

        assert result.flip_year >= 0
        assert result.total_tax_benefits > 0
        assert len(result.annual_cash_flows_investor) == te_config.project_lifetime_years


class TestIncentiveModelerIntegration:
    """Integration tests for complete workflows."""

    def test_complete_itc_depreciation_workflow(
        self,
        incentive_modeler: IncentiveModeler,
        large_system_config: SystemConfiguration,
    ) -> None:
        """Test complete workflow: ITC + Depreciation for large system."""
        # Step 1: Calculate ITC with bonuses
        itc_config = ITCConfiguration(
            system_config=large_system_config,
            itc_rate=0.30,
            apply_bonus=True,
            meets_domestic_content=True,
            is_energy_community=True,
            bonus_rate=0.10,
        )
        itc_result = incentive_modeler.itc_calculation(itc_config)

        # Verify ITC
        assert itc_result.total_itc_amount > 0
        assert itc_result.effective_rate == pytest.approx(0.50)  # 30% + 10% + 10%

        # Step 2: Calculate depreciation with ITC basis adjustment
        asset_basis = (
            large_system_config.installation_cost_total - 0.5 * itc_result.total_itc_amount
        )
        depr_result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_5,
            bonus_depreciation_rate=0.80,  # 80% bonus depreciation
        )

        # Verify depreciation
        assert depr_result.total_depreciation == pytest.approx(asset_basis)
        assert depr_result.bonus_depreciation_amount > 0

        # Step 3: Verify total tax benefits
        total_tax_benefit = itc_result.total_itc_amount + depr_result.total_depreciation
        assert total_tax_benefit > itc_result.total_itc_amount

    def test_complete_ptc_workflow(
        self,
        incentive_modeler: IncentiveModeler,
        large_system_config: SystemConfiguration,
    ) -> None:
        """Test complete PTC workflow with bonus multiplier."""
        ptc_config = PTCConfiguration(
            system_config=large_system_config,
            ptc_rate_per_kwh=0.0275,
            credit_period_years=10,
            inflation_adjustment=True,
            inflation_rate=0.025,
            production_degradation_rate=0.005,
            apply_bonus=True,
            bonus_multiplier=5.0,
        )

        result = incentive_modeler.ptc_computation(ptc_config, discount_rate=0.06)

        assert result.total_ptc_lifetime > 0
        assert result.present_value_ptc < result.total_ptc_lifetime
        assert len(result.annual_credits) == 10
        assert result.first_year_credit > result.last_year_credit

    def test_complete_tax_equity_workflow(
        self,
        incentive_modeler: IncentiveModeler,
        large_system_config: SystemConfiguration,
    ) -> None:
        """Test complete tax equity workflow with all components."""
        # Calculate ITC
        itc_config = ITCConfiguration(
            system_config=large_system_config,
            itc_rate=0.30,
        )
        itc_result = incentive_modeler.itc_calculation(itc_config)

        # Calculate depreciation
        asset_basis = (
            large_system_config.installation_cost_total - 0.5 * itc_result.total_itc_amount
        )
        depr_result = incentive_modeler.depreciation_schedule(
            asset_basis=asset_basis,
            method=DepreciationMethod.MACRS_5,
            bonus_depreciation_rate=0.60,
        )

        # Model tax equity
        te_config = TaxEquityConfiguration(
            system_config=large_system_config,
            investor_equity_percentage=0.99,
            target_flip_irr=0.08,
            post_flip_investor_percentage=0.05,
            tax_rate=0.40,
            project_lifetime_years=25,
            include_itc=True,
            include_depreciation=True,
        )

        result = incentive_modeler.tax_equity_modeling(
            config=te_config,
            itc_amount=itc_result.total_itc_amount,
            depreciation_schedule=depr_result.annual_depreciation,
        )

        # Verify complete structure
        assert result.flip_year > 0
        assert result.investor_irr > 0
        assert result.sponsor_irr != 0  # Could be positive or negative
        assert result.total_tax_benefits > 0
        assert abs(result.investor_npv) > 0
        assert abs(result.sponsor_npv) > 0
