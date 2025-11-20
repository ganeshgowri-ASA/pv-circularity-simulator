"""
Comprehensive tests for LCOE calculator.

Tests cover:
- Basic LCOE calculations
- Circularity impact
- Cost breakdown
- Edge cases
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from financial.models import CostStructure, RevenueStream, CircularityMetrics
from financial.calculators import LCOECalculator


class TestLCOECalculator:
    """Test suite for LCOE Calculator."""

    @pytest.fixture
    def basic_cost_structure(self):
        """Create basic cost structure for testing."""
        return CostStructure(
            initial_capex=200000.0,
            equipment_cost=150000.0,
            installation_cost=30000.0,
            soft_costs=20000.0,
            annual_opex=2000.0,
            maintenance_cost=1500.0,
            insurance_cost=500.0,
        )

    @pytest.fixture
    def basic_revenue_stream(self):
        """Create basic revenue stream for testing."""
        return RevenueStream(
            annual_energy_production=150000.0,  # 150 MWh
            energy_price=0.12,  # $0.12/kWh
            degradation_rate=0.005,  # 0.5% per year
        )

    @pytest.fixture
    def circularity_metrics(self):
        """Create circularity metrics for testing."""
        return CircularityMetrics(
            material_recovery_rate=0.90,
            system_weight=1000.0,
            refurbishment_potential=0.30,
            refurbishment_value=0.40,
        )

    def test_basic_lcoe_calculation(
        self,
        basic_cost_structure,
        basic_revenue_stream,
        circularity_metrics
    ):
        """Test basic LCOE calculation."""
        calculator = LCOECalculator(
            cost_structure=basic_cost_structure,
            revenue_stream=basic_revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=25,
            discount_rate=0.06,
        )

        result = calculator.calculate_lcoe()

        # LCOE should be positive
        assert result.lcoe > 0
        assert result.lcoe_real > 0

        # LCOE should be reasonable (typically $0.05-0.30/kWh for solar)
        assert 0.02 < result.lcoe < 0.50

        # With circularity should be less than without
        assert result.with_circularity < result.without_circularity

        # Circularity benefit should be positive
        assert result.circularity_benefit > 0

    def test_cost_breakdown(
        self,
        basic_cost_structure,
        basic_revenue_stream,
        circularity_metrics
    ):
        """Test cost breakdown components."""
        calculator = LCOECalculator(
            cost_structure=basic_cost_structure,
            revenue_stream=basic_revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=25,
        )

        result = calculator.calculate_lcoe()

        # All cost categories should be present
        assert 'Equipment' in result.cost_breakdown
        assert 'Installation' in result.cost_breakdown
        assert 'Soft Costs' in result.cost_breakdown
        assert 'Maintenance' in result.cost_breakdown

        # Total should match sum
        total_positive = sum(v for v in result.cost_breakdown.values() if v > 0)
        assert total_positive > 0

    def test_no_circularity(
        self,
        basic_cost_structure,
        basic_revenue_stream
    ):
        """Test LCOE without circularity benefits."""
        no_circ = CircularityMetrics(
            material_recovery_rate=0.0,
            refurbishment_potential=0.0,
        )

        calculator = LCOECalculator(
            cost_structure=basic_cost_structure,
            revenue_stream=basic_revenue_stream,
            circularity_metrics=no_circ,
            lifetime_years=25,
        )

        result = calculator.calculate_lcoe(include_circularity=False)

        # Without circularity, benefit should be minimal
        assert result.circularity_benefit < 0.001

    def test_varying_lifetime(
        self,
        basic_cost_structure,
        basic_revenue_stream,
        circularity_metrics
    ):
        """Test LCOE with different system lifetimes."""
        lifetimes = [10, 25, 40]
        lcoe_values = []

        for lifetime in lifetimes:
            calculator = LCOECalculator(
                cost_structure=basic_cost_structure,
                revenue_stream=basic_revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=lifetime,
            )

            result = calculator.calculate_lcoe()
            lcoe_values.append(result.lcoe)

        # Longer lifetime generally means lower LCOE
        # (more energy production to amortize costs)
        assert lcoe_values[0] > lcoe_values[1]  # 10yr > 25yr

    def test_discount_rate_impact(
        self,
        basic_cost_structure,
        basic_revenue_stream,
        circularity_metrics
    ):
        """Test impact of discount rate on LCOE."""
        discount_rates = [0.03, 0.06, 0.10]
        lcoe_values = []

        for rate in discount_rates:
            calculator = LCOECalculator(
                cost_structure=basic_cost_structure,
                revenue_stream=basic_revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=25,
                discount_rate=rate,
            )

            result = calculator.calculate_lcoe()
            lcoe_values.append(result.lcoe)

        # Higher discount rate generally means higher LCOE
        assert lcoe_values[0] < lcoe_values[2]  # 3% < 10%

    def test_lcoe_by_year(
        self,
        basic_cost_structure,
        basic_revenue_stream,
        circularity_metrics
    ):
        """Test cumulative LCOE calculation by year."""
        calculator = LCOECalculator(
            cost_structure=basic_cost_structure,
            revenue_stream=basic_revenue_stream,
            circularity_metrics=circularity_metrics,
            lifetime_years=25,
        )

        df = calculator.calculate_lcoe_by_year()

        # Should have 25 years
        assert len(df) == 25

        # LCOE should generally decrease over time
        # (fixed costs amortized over more energy)
        assert df.iloc[0]['lcoe'] > df.iloc[-1]['lcoe']

        # Energy should be cumulative and increasing
        assert df['cumulative_energy_kwh'].is_monotonic_increasing

    def test_zero_cost_handling(self, basic_revenue_stream, circularity_metrics):
        """Test handling of zero costs."""
        zero_cost = CostStructure(
            initial_capex=1.0,  # Minimal to avoid division by zero
            equipment_cost=0.5,
            installation_cost=0.5,
            soft_costs=0.0,
            annual_opex=0.0,
            maintenance_cost=0.0,
        )

        calculator = LCOECalculator(
            cost_structure=zero_cost,
            revenue_stream=basic_revenue_stream,
            circularity_metrics=circularity_metrics,
        )

        result = calculator.calculate_lcoe()

        # Should handle gracefully
        assert result.lcoe >= 0
        assert not np.isnan(result.lcoe)
        assert not np.isinf(result.lcoe)


class TestCircularityImpact:
    """Test circularity impact calculations."""

    def test_high_circularity_benefit(self):
        """Test system with high circularity benefits."""
        cost_structure = CostStructure(
            initial_capex=200000.0,
            equipment_cost=150000.0,
            installation_cost=30000.0,
            soft_costs=20000.0,
            annual_opex=2000.0,
            maintenance_cost=1500.0,
            disposal_cost=5000.0,
        )

        revenue_stream = RevenueStream(
            annual_energy_production=150000.0,
            energy_price=0.12,
        )

        high_circ = CircularityMetrics(
            material_recovery_rate=0.95,
            system_weight=1000.0,
            refurbishment_potential=0.50,
            refurbishment_value=0.60,
            recycling_revenue=20.0,
            recycling_cost=5.0,
        )

        calculator = LCOECalculator(
            cost_structure=cost_structure,
            revenue_stream=revenue_stream,
            circularity_metrics=high_circ,
            lifetime_years=25,
        )

        result = calculator.calculate_lcoe()

        # High circularity should provide significant benefit
        benefit_percent = (result.circularity_benefit /
                          result.without_circularity * 100)
        assert benefit_percent > 1.0  # At least 1% benefit

    def test_circularity_score_impact(self):
        """Test relationship between circularity score and LCOE benefit."""
        cost_structure = CostStructure(
            initial_capex=200000.0,
            equipment_cost=150000.0,
            installation_cost=30000.0,
            soft_costs=20000.0,
            annual_opex=2000.0,
            maintenance_cost=1500.0,
        )

        revenue_stream = RevenueStream(
            annual_energy_production=150000.0,
            energy_price=0.12,
        )

        # Test different circularity levels
        recovery_rates = [0.50, 0.75, 0.95]
        benefits = []

        for rate in recovery_rates:
            circ = CircularityMetrics(
                material_recovery_rate=rate,
                system_weight=1000.0,
            )

            calculator = LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circ,
                lifetime_years=25,
            )

            result = calculator.calculate_lcoe()
            benefits.append(result.circularity_benefit)

        # Higher recovery rate should provide more benefit
        assert benefits[0] < benefits[2]  # 50% < 95%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
