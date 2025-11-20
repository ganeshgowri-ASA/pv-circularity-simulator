"""
Tests for calculation helpers.
"""

import pytest
from src.pv_simulator.utils.calculations import (
    calculate_mean,
    calculate_median,
    calculate_standard_deviation,
    calculate_variance,
    calculate_percentile,
    calculate_weighted_average,
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_lcoe,
    calculate_panel_efficiency,
    calculate_temperature_derating,
    calculate_performance_ratio,
    calculate_capacity_factor,
    calculate_degradation_factor,
    calculate_material_recovery_rate,
    calculate_circular_economy_score,
    calculate_carbon_footprint_reduction,
    clamp,
    linear_interpolation,
    round_to_significant_figures,
)


class TestStatisticalCalculations:
    """Tests for statistical calculation functions."""

    def test_calculate_mean(self):
        """Test mean calculation."""
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        assert calculate_mean([10, 20, 30]) == 20.0

    def test_calculate_mean_empty(self):
        """Test mean of empty list raises error."""
        with pytest.raises(ValueError, match="Cannot calculate mean of empty list"):
            calculate_mean([])

    def test_calculate_median_odd(self):
        """Test median with odd number of values."""
        assert calculate_median([1, 2, 3, 4, 5]) == 3

    def test_calculate_median_even(self):
        """Test median with even number of values."""
        assert calculate_median([1, 2, 3, 4]) == 2.5

    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        result = calculate_standard_deviation(values)
        assert abs(result - 2.138) < 0.01

    def test_calculate_standard_deviation_insufficient_data(self):
        """Test standard deviation with insufficient data."""
        with pytest.raises(ValueError, match="at least 2 values"):
            calculate_standard_deviation([1])

    def test_calculate_variance(self):
        """Test variance calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        result = calculate_variance(values)
        assert abs(result - 4.571) < 0.01

    def test_calculate_percentile(self):
        """Test percentile calculation."""
        values = list(range(1, 11))  # 1 to 10

        assert calculate_percentile(values, 50) == 5.5
        assert calculate_percentile(values, 25) == 3.25
        assert calculate_percentile(values, 75) == 7.75

    def test_calculate_percentile_out_of_range(self):
        """Test percentile out of range raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            calculate_percentile([1, 2, 3], 150)

    def test_calculate_weighted_average(self):
        """Test weighted average calculation."""
        values = [10, 20, 30]
        weights = [1, 2, 3]
        result = calculate_weighted_average(values, weights)
        assert abs(result - 23.333) < 0.01

    def test_calculate_weighted_average_different_lengths(self):
        """Test weighted average with mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            calculate_weighted_average([1, 2, 3], [1, 2])


class TestFinancialCalculations:
    """Tests for financial calculation functions."""

    def test_calculate_npv(self):
        """Test NPV calculation."""
        cash_flows = [100, 100, 100]
        discount_rate = 0.1
        initial_investment = 250

        npv = calculate_npv(cash_flows, discount_rate, initial_investment)
        assert abs(npv - (-1.3)) < 0.1

    def test_calculate_npv_positive(self):
        """Test NPV with positive result."""
        cash_flows = [1000, 2000, 3000]
        discount_rate = 0.05

        npv = calculate_npv(cash_flows, discount_rate)
        assert npv > 5000  # Should be positive

    def test_calculate_irr(self):
        """Test IRR calculation."""
        cash_flows = [100, 100, 100]
        initial_investment = 250

        irr = calculate_irr(cash_flows, initial_investment)

        assert irr is not None
        assert 0.09 < irr < 0.10

    def test_calculate_irr_negative_case(self):
        """Test IRR with case that has no solution."""
        cash_flows = [-100, -100, -100]  # All negative
        initial_investment = 100

        irr = calculate_irr(cash_flows, initial_investment, max_iterations=100)
        # May return None if no solution found
        assert irr is None or irr < -0.99

    def test_calculate_payback_period(self):
        """Test payback period calculation."""
        cash_flows = [100, 100, 100]
        initial_investment = 250

        payback = calculate_payback_period(cash_flows, initial_investment)
        assert payback == 2.5  # 2.5 periods to recover investment

    def test_calculate_payback_period_not_recovered(self):
        """Test payback period when investment not recovered."""
        cash_flows = [50, 50, 50]
        initial_investment = 200

        payback = calculate_payback_period(cash_flows, initial_investment)
        assert payback is None

    def test_calculate_lcoe(self):
        """Test LCOE calculation."""
        total_costs = 10000
        total_energy = 50000  # kWh
        discount_rate = 0.05
        lifetime = 25

        lcoe = calculate_lcoe(total_costs, total_energy, discount_rate, lifetime)

        assert lcoe > 0
        assert lcoe < 1  # Should be reasonable cost per kWh

    def test_calculate_lcoe_zero_energy(self):
        """Test LCOE with zero energy raises error."""
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_lcoe(10000, 0, 0.05, 25)


class TestPVSystemCalculations:
    """Tests for PV system technical calculations."""

    def test_calculate_panel_efficiency(self):
        """Test panel efficiency calculation."""
        efficiency = calculate_panel_efficiency(
            power_output_w=300, area_m2=1.6, irradiance_w_m2=1000
        )
        assert abs(efficiency - 0.1875) < 0.001

    def test_calculate_panel_efficiency_zero_area(self):
        """Test efficiency with zero area raises error."""
        with pytest.raises(ValueError, match="Area cannot be zero"):
            calculate_panel_efficiency(300, 0, 1000)

    def test_calculate_temperature_derating(self):
        """Test temperature derating calculation."""
        derating = calculate_temperature_derating(
            module_temp_c=45, stc_temp_c=25, temp_coeff_percent_per_c=-0.4
        )
        assert abs(derating - 0.92) < 0.01

    def test_calculate_temperature_derating_at_stc(self):
        """Test derating at STC temperature."""
        derating = calculate_temperature_derating(25, 25, -0.4)
        assert derating == 1.0

    def test_calculate_performance_ratio(self):
        """Test performance ratio calculation."""
        pr = calculate_performance_ratio(actual_energy_kwh=8500, theoretical_energy_kwh=10000)
        assert pr == 0.85

    def test_calculate_performance_ratio_zero_theoretical(self):
        """Test PR with zero theoretical energy raises error."""
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_performance_ratio(8500, 0)

    def test_calculate_capacity_factor(self):
        """Test capacity factor calculation."""
        cf = calculate_capacity_factor(
            actual_energy_kwh=1000, rated_power_kw=5, period_hours=8760
        )
        assert abs(cf - 0.0228) < 0.001

    def test_calculate_capacity_factor_zero_power(self):
        """Test capacity factor with zero power raises error."""
        with pytest.raises(ValueError, match="Rated power cannot be zero"):
            calculate_capacity_factor(1000, 0, 8760)

    def test_calculate_degradation_factor(self):
        """Test degradation factor calculation."""
        # 0.5% per year for 10 years
        factor = calculate_degradation_factor(0.5, 10)
        assert abs(factor - 0.951) < 0.01

        # 1% per year for 25 years
        factor = calculate_degradation_factor(1.0, 25)
        assert abs(factor - 0.778) < 0.01

    def test_calculate_degradation_factor_zero_years(self):
        """Test degradation with zero years."""
        factor = calculate_degradation_factor(0.5, 0)
        assert factor == 1.0  # No degradation


class TestCircularEconomyCalculations:
    """Tests for circular economy calculations."""

    def test_calculate_material_recovery_rate(self):
        """Test material recovery rate calculation."""
        rate = calculate_material_recovery_rate(recovered_mass_kg=85, total_mass_kg=100)
        assert rate == 0.85

    def test_calculate_material_recovery_rate_zero_total(self):
        """Test recovery rate with zero total mass raises error."""
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_material_recovery_rate(50, 0)

    def test_calculate_circular_economy_score(self):
        """Test circular economy score calculation."""
        score = calculate_circular_economy_score(
            recyclability=0.8, reusability=0.6, renewable_content=0.4
        )
        # 0.8*0.4 + 0.6*0.3 + 0.4*0.3 = 0.32 + 0.18 + 0.12 = 0.62 * 100 = 62.0
        assert score == 62.0

    def test_calculate_circular_economy_score_custom_weights(self):
        """Test CE score with custom weights."""
        score = calculate_circular_economy_score(
            recyclability=0.9, reusability=0.7, renewable_content=0.5, weights=(0.5, 0.3, 0.2)
        )
        # 0.9*0.5 + 0.7*0.3 + 0.5*0.2 = 0.45 + 0.21 + 0.10 = 0.76 * 100 = 76.0
        assert score == 76.0

    def test_calculate_carbon_footprint_reduction(self):
        """Test carbon footprint reduction calculation."""
        reduction = calculate_carbon_footprint_reduction(
            virgin_material_emissions_kg_co2=100,
            recycled_material_emissions_kg_co2=20,
            recycling_rate=0.5,
        )
        assert reduction == 40.0

    def test_calculate_carbon_footprint_full_recycling(self):
        """Test carbon reduction with 100% recycling."""
        reduction = calculate_carbon_footprint_reduction(
            virgin_material_emissions_kg_co2=500,
            recycled_material_emissions_kg_co2=100,
            recycling_rate=1.0,
        )
        assert reduction == 400.0


class TestMathUtilities:
    """Tests for general math utility functions."""

    def test_clamp_within_range(self):
        """Test clamp with value within range."""
        assert clamp(5, 0, 10) == 5

    def test_clamp_below_minimum(self):
        """Test clamp with value below minimum."""
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_maximum(self):
        """Test clamp with value above maximum."""
        assert clamp(15, 0, 10) == 10

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        # Interpolate at x=5 between (0,0) and (10,100)
        result = linear_interpolation(5, 0, 0, 10, 100)
        assert result == 50.0

        # Interpolate at x=7.5 between (5,20) and (10,40)
        result = linear_interpolation(7.5, 5, 20, 10, 40)
        assert result == 30.0

    def test_linear_interpolation_same_x(self):
        """Test interpolation with same x values raises error."""
        with pytest.raises(ValueError, match="cannot be equal"):
            linear_interpolation(5, 10, 20, 10, 30)

    def test_round_to_significant_figures(self):
        """Test rounding to significant figures."""
        assert round_to_significant_figures(12345, 3) == 12300.0
        assert round_to_significant_figures(0.0012345, 2) == 0.0012

    def test_round_to_significant_figures_zero(self):
        """Test rounding zero."""
        assert round_to_significant_figures(0, 3) == 0.0
