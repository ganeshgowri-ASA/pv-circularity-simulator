"""
Comprehensive tests for Recycling Economics module.

Tests cover all major functionality including:
- Material composition and validation
- Cost calculations
- Recovery rates
- Revenue calculations
- Environmental credits
- Integration tests
"""

import pytest
from pydantic import ValidationError

from pv_circularity_simulator.recycling import (
    RecyclingEconomics,
    MaterialComposition,
    MaterialExtractionCosts,
    RecoveryRates,
    RecyclingRevenue,
    EnvironmentalCredits,
    PVMaterialType,
    RecyclingTechnology,
)


class TestMaterialComposition:
    """Tests for MaterialComposition model."""

    def test_valid_material_composition(self):
        """Test creating valid material composition."""
        material = MaterialComposition(
            material_type=PVMaterialType.SILICON,
            mass_kg=5.0,
            purity_percent=99.0,
            market_value_per_kg=15.0,
        )
        assert material.material_type == PVMaterialType.SILICON
        assert material.mass_kg == 5.0
        assert material.purity_percent == 99.0
        assert material.market_value_per_kg == 15.0

    def test_invalid_mass(self):
        """Test that negative mass raises validation error."""
        with pytest.raises(ValidationError):
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=-1.0,
                purity_percent=99.0,
                market_value_per_kg=15.0,
            )

    def test_invalid_purity(self):
        """Test that purity outside 0-100 raises validation error."""
        with pytest.raises(ValidationError):
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=5.0,
                purity_percent=101.0,
                market_value_per_kg=15.0,
            )

    def test_material_composition_immutable(self):
        """Test that MaterialComposition is immutable."""
        material = MaterialComposition(
            material_type=PVMaterialType.SILICON,
            mass_kg=5.0,
            purity_percent=99.0,
            market_value_per_kg=15.0,
        )
        with pytest.raises(ValidationError):
            material.mass_kg = 10.0


class TestMaterialExtractionCosts:
    """Tests for MaterialExtractionCosts model."""

    def test_valid_extraction_costs(self):
        """Test creating valid extraction costs."""
        costs = MaterialExtractionCosts(
            technology=RecyclingTechnology.HYBRID,
            collection_cost_per_panel=5.0,
            preprocessing_cost_per_kg=2.0,
            processing_cost_per_kg=3.0,
            purification_cost_per_kg=1.5,
            labor_cost_per_hour=25.0,
            processing_time_hours=0.5,
            energy_cost_per_kwh=0.12,
            energy_consumption_kwh_per_kg=2.0,
            disposal_cost_per_kg=0.5,
            equipment_depreciation_per_panel=2.0,
        )
        assert costs.technology == RecyclingTechnology.HYBRID
        assert costs.collection_cost_per_panel == 5.0

    def test_total_fixed_cost(self):
        """Test total fixed cost calculation."""
        costs = MaterialExtractionCosts(
            technology=RecyclingTechnology.MECHANICAL,
            collection_cost_per_panel=5.0,
            preprocessing_cost_per_kg=2.0,
            processing_cost_per_kg=3.0,
            purification_cost_per_kg=1.5,
            labor_cost_per_hour=25.0,
            processing_time_hours=2.0,
            energy_cost_per_kwh=0.12,
            energy_consumption_kwh_per_kg=2.0,
            disposal_cost_per_kg=0.5,
            equipment_depreciation_per_panel=2.0,
        )
        # Fixed cost = collection (5) + depreciation (2) + labor (25*2)
        assert costs.total_fixed_cost == 57.0

    def test_calculate_total_cost(self):
        """Test total cost calculation."""
        costs = MaterialExtractionCosts(
            technology=RecyclingTechnology.MECHANICAL,
            collection_cost_per_panel=5.0,
            preprocessing_cost_per_kg=2.0,
            processing_cost_per_kg=3.0,
            purification_cost_per_kg=1.5,
            labor_cost_per_hour=20.0,
            processing_time_hours=1.0,
            energy_cost_per_kwh=0.10,
            energy_consumption_kwh_per_kg=2.0,
            disposal_cost_per_kg=0.5,
            equipment_depreciation_per_panel=2.0,
            overhead_multiplier=1.2,
        )

        total_cost = costs.calculate_total_cost(
            total_mass_kg=20.0,
            non_recoverable_mass_kg=2.0
        )

        # Fixed: 5 + 2 + 20 = 27
        # Variable: (2+3+1.5)*20 + 0.1*2*20 + 0.5*2 = 130 + 4 + 1 = 135
        # Total before overhead: 162
        # With 1.2 overhead: 162 * 1.2 = 194.4
        assert abs(total_cost - 194.4) < 0.01


class TestRecoveryRates:
    """Tests for RecoveryRates model."""

    def test_valid_recovery_rates(self):
        """Test creating valid recovery rates."""
        rates = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={
                PVMaterialType.SILICON: 95.0,
                PVMaterialType.SILVER: 90.0,
                PVMaterialType.GLASS: 98.0,
            },
            technology_efficiency=0.85,
            quality_grade="B",
        )
        assert rates.technology == RecyclingTechnology.HYBRID
        assert rates.material_recovery_rates[PVMaterialType.SILICON] == 95.0

    def test_invalid_recovery_rate(self):
        """Test that recovery rate > 100 raises validation error."""
        with pytest.raises(ValidationError):
            RecoveryRates(
                technology=RecyclingTechnology.HYBRID,
                material_recovery_rates={
                    PVMaterialType.SILICON: 105.0,
                },
            )

    def test_effective_recovery_rate(self):
        """Test effective recovery rate calculation."""
        rates = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={
                PVMaterialType.SILICON: 90.0,
            },
            technology_efficiency=0.8,
        )
        effective = rates.get_effective_recovery_rate(PVMaterialType.SILICON)
        # 90.0 * 0.8 = 72.0
        assert effective == 72.0

    def test_quality_multiplier(self):
        """Test quality multiplier for different grades."""
        rates_a = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={},
            quality_grade="A",
        )
        rates_b = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={},
            quality_grade="B",
        )
        rates_c = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={},
            quality_grade="C",
        )

        assert rates_a.get_quality_multiplier() == 1.0
        assert rates_b.get_quality_multiplier() == 0.8
        assert rates_c.get_quality_multiplier() == 0.6


class TestRecyclingRevenue:
    """Tests for RecyclingRevenue model."""

    def test_gross_revenue_calculation(self):
        """Test gross revenue calculation."""
        materials = [
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=5.0,
                purity_percent=95.0,
                market_value_per_kg=15.0,
            ),
            MaterialComposition(
                material_type=PVMaterialType.SILVER,
                mass_kg=0.01,
                purity_percent=90.0,
                market_value_per_kg=600.0,
            ),
        ]

        revenue = RecyclingRevenue(
            recovered_materials=materials,
            quality_discount=1.0,
            transportation_cost_per_kg=0.0,
            sales_commission_percent=0.0,
        )

        # Gross = 5*15 + 0.01*600 = 75 + 6 = 81
        assert abs(revenue.gross_revenue - 81.0) < 0.01

    def test_net_revenue_calculation(self):
        """Test net revenue with costs."""
        materials = [
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=5.0,
                purity_percent=95.0,
                market_value_per_kg=20.0,
            ),
        ]

        revenue = RecyclingRevenue(
            recovered_materials=materials,
            quality_discount=1.0,
            transportation_cost_per_kg=0.5,
            sales_commission_percent=10.0,
        )

        # Gross = 5 * 20 = 100
        # Transport = 5 * 0.5 = 2.5
        # Commission = 100 * 0.1 = 10
        # Net = 100 - 2.5 - 10 = 87.5
        assert abs(revenue.net_revenue - 87.5) < 0.01

    def test_revenue_by_material(self):
        """Test revenue breakdown by material."""
        materials = [
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=5.0,
                purity_percent=95.0,
                market_value_per_kg=15.0,
            ),
            MaterialComposition(
                material_type=PVMaterialType.ALUMINUM,
                mass_kg=2.0,
                purity_percent=98.0,
                market_value_per_kg=2.5,
            ),
        ]

        revenue = RecyclingRevenue(
            recovered_materials=materials,
            quality_discount=1.0,
        )

        breakdown = revenue.get_revenue_by_material()
        assert abs(breakdown[PVMaterialType.SILICON] - 75.0) < 0.01
        assert abs(breakdown[PVMaterialType.ALUMINUM] - 5.0) < 0.01


class TestEnvironmentalCredits:
    """Tests for EnvironmentalCredits model."""

    def test_carbon_value_calculation(self):
        """Test carbon value calculation."""
        credits = EnvironmentalCredits(
            avoided_emissions_kg_co2=1000.0,  # 1 ton
            energy_savings_kwh=500.0,
            carbon_price_per_ton_co2=50.0,
        )

        # 1000 kg / 1000 * 50 = 50
        assert credits.total_carbon_value == 50.0

    def test_total_environmental_value(self):
        """Test total environmental value."""
        credits = EnvironmentalCredits(
            avoided_emissions_kg_co2=1000.0,
            energy_savings_kwh=500.0,
            landfill_diversion_credits=10.0,
            epr_credit_value=15.0,
            carbon_price_per_ton_co2=50.0,
        )

        # Carbon: 50, Landfill: 10, EPR: 15 = 75
        assert credits.total_environmental_value == 75.0

    def test_lca_metrics(self):
        """Test LCA metrics output."""
        credits = EnvironmentalCredits(
            avoided_emissions_kg_co2=1000.0,
            energy_savings_kwh=500.0,
            water_savings_liters=2000.0,
            carbon_price_per_ton_co2=50.0,
        )

        metrics = credits.get_lca_metrics()
        assert metrics["global_warming_potential_kg_co2_eq"] == 1000.0
        assert metrics["primary_energy_demand_kwh"] == 500.0
        assert metrics["water_consumption_liters"] == 2000.0


class TestRecyclingEconomics:
    """Integration tests for RecyclingEconomics class."""

    @pytest.fixture
    def sample_panel_composition(self):
        """Sample panel composition for testing."""
        return [
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=5.0,
                purity_percent=99.0,
                market_value_per_kg=15.0,
            ),
            MaterialComposition(
                material_type=PVMaterialType.SILVER,
                mass_kg=0.015,
                purity_percent=95.0,
                market_value_per_kg=600.0,
            ),
            MaterialComposition(
                material_type=PVMaterialType.ALUMINUM,
                mass_kg=3.0,
                purity_percent=98.0,
                market_value_per_kg=2.5,
            ),
            MaterialComposition(
                material_type=PVMaterialType.GLASS,
                mass_kg=10.0,
                purity_percent=99.0,
                market_value_per_kg=0.1,
            ),
        ]

    @pytest.fixture
    def sample_extraction_costs(self):
        """Sample extraction costs for testing."""
        return MaterialExtractionCosts(
            technology=RecyclingTechnology.HYBRID,
            collection_cost_per_panel=5.0,
            preprocessing_cost_per_kg=2.0,
            processing_cost_per_kg=3.0,
            purification_cost_per_kg=1.5,
            labor_cost_per_hour=25.0,
            processing_time_hours=0.5,
            energy_cost_per_kwh=0.12,
            energy_consumption_kwh_per_kg=2.0,
            disposal_cost_per_kg=0.5,
            equipment_depreciation_per_panel=2.0,
        )

    @pytest.fixture
    def sample_recovery_rates(self):
        """Sample recovery rates for testing."""
        return RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={
                PVMaterialType.SILICON: 95.0,
                PVMaterialType.SILVER: 90.0,
                PVMaterialType.GLASS: 98.0,
                PVMaterialType.ALUMINUM: 97.0,
            },
            technology_efficiency=0.85,
            quality_grade="B",
        )

    def test_initialization(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test RecyclingEconomics initialization."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        assert economics.panel_mass_kg == 20.0
        assert len(economics.panel_composition) == 4

    def test_invalid_panel_mass(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test that negative panel mass raises error."""
        with pytest.raises(ValueError, match="Panel mass must be positive"):
            RecyclingEconomics(
                panel_composition=sample_panel_composition,
                extraction_costs=sample_extraction_costs,
                recovery_rates_model=sample_recovery_rates,
                panel_mass_kg=-1.0,
            )

    def test_technology_mismatch(
        self,
        sample_panel_composition,
        sample_extraction_costs,
    ):
        """Test that mismatched technologies raise error."""
        different_recovery = RecoveryRates(
            technology=RecyclingTechnology.MECHANICAL,  # Different from HYBRID
            material_recovery_rates={},
        )

        with pytest.raises(ValueError, match="same technology"):
            RecyclingEconomics(
                panel_composition=sample_panel_composition,
                extraction_costs=sample_extraction_costs,
                recovery_rates_model=different_recovery,
                panel_mass_kg=20.0,
            )

    def test_material_extraction_costs(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test material extraction costs calculation."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        costs = economics.material_extraction_costs()

        assert "total_cost" in costs
        assert "cost_per_kg" in costs
        assert "fixed_costs" in costs
        assert "variable_costs" in costs
        assert "overhead_costs" in costs
        assert "cost_breakdown" in costs

        assert costs["total_cost"] > 0
        assert costs["cost_per_kg"] == costs["total_cost"] / 20.0

    def test_recovery_rates(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test recovery rates calculation."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        rates = economics.recovery_rates()

        assert "material_recovery_rates" in rates
        assert "recovered_masses" in rates
        assert "total_recovery_rate" in rates
        assert "total_recovered_mass" in rates

        # Check effective rates include efficiency factor
        silicon_rate = rates["material_recovery_rates"][PVMaterialType.SILICON]
        # 95.0 * 0.85 = 80.75
        assert abs(silicon_rate - 80.75) < 0.01

        # Check recovered mass
        silicon_mass = rates["recovered_masses"][PVMaterialType.SILICON]
        # 5.0 kg * 0.8075 = 4.0375 kg
        assert abs(silicon_mass - 4.0375) < 0.001

    def test_recycling_revenue_calculation(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test recycling revenue calculation."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        revenue = economics.recycling_revenue_calculation()

        assert "gross_revenue" in revenue
        assert "net_revenue" in revenue
        assert "revenue_by_material" in revenue
        assert "total_recovered_mass" in revenue

        assert revenue["gross_revenue"] > 0
        assert revenue["net_revenue"] <= revenue["gross_revenue"]

    def test_environmental_credits(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test environmental credits calculation."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        credits = economics.environmental_credits(carbon_price_per_ton_co2=75.0)

        assert "total_environmental_value" in credits
        assert "carbon_value" in credits
        assert "avoided_emissions_kg_co2" in credits
        assert "energy_savings_kwh" in credits
        assert "lca_metrics" in credits

        assert credits["avoided_emissions_kg_co2"] > 0
        assert credits["energy_savings_kwh"] > 0
        assert credits["total_environmental_value"] > 0

    def test_net_economic_value(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test net economic value calculation."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        net_value = economics.net_economic_value(
            carbon_price_per_ton_co2=100.0,
            include_environmental_credits=True,
        )

        assert "net_value" in net_value
        assert "total_revenue" in net_value
        assert "total_costs" in net_value
        assert "environmental_value" in net_value
        assert "roi_percent" in net_value
        assert "breakeven_carbon_price" in net_value

        # Verify calculation
        expected_net = (
            net_value["total_revenue"] +
            net_value["environmental_value"] -
            net_value["total_costs"]
        )
        assert abs(net_value["net_value"] - expected_net) < 0.01

    def test_net_economic_value_without_credits(
        self,
        sample_panel_composition,
        sample_extraction_costs,
        sample_recovery_rates,
    ):
        """Test net economic value without environmental credits."""
        economics = RecyclingEconomics(
            panel_composition=sample_panel_composition,
            extraction_costs=sample_extraction_costs,
            recovery_rates_model=sample_recovery_rates,
            panel_mass_kg=20.0,
        )

        net_value = economics.net_economic_value(
            include_environmental_credits=False,
        )

        assert net_value["environmental_value"] == 0.0

    def test_realistic_scenario(self):
        """Test a realistic complete recycling scenario."""
        # Typical silicon panel composition
        composition = [
            MaterialComposition(
                material_type=PVMaterialType.GLASS,
                mass_kg=12.0,
                purity_percent=99.0,
                market_value_per_kg=0.08,
            ),
            MaterialComposition(
                material_type=PVMaterialType.ALUMINUM,
                mass_kg=3.5,
                purity_percent=98.0,
                market_value_per_kg=2.2,
            ),
            MaterialComposition(
                material_type=PVMaterialType.SILICON,
                mass_kg=3.0,
                purity_percent=99.0,
                market_value_per_kg=18.0,
            ),
            MaterialComposition(
                material_type=PVMaterialType.COPPER,
                mass_kg=0.8,
                purity_percent=99.5,
                market_value_per_kg=8.5,
            ),
            MaterialComposition(
                material_type=PVMaterialType.SILVER,
                mass_kg=0.012,
                purity_percent=99.0,
                market_value_per_kg=650.0,
            ),
        ]

        costs = MaterialExtractionCosts(
            technology=RecyclingTechnology.HYBRID,
            collection_cost_per_panel=8.0,
            preprocessing_cost_per_kg=1.5,
            processing_cost_per_kg=2.5,
            purification_cost_per_kg=2.0,
            labor_cost_per_hour=30.0,
            processing_time_hours=1.0,
            energy_cost_per_kwh=0.15,
            energy_consumption_kwh_per_kg=3.0,
            disposal_cost_per_kg=0.3,
            equipment_depreciation_per_panel=3.0,
            overhead_multiplier=1.25,
        )

        recovery = RecoveryRates(
            technology=RecyclingTechnology.HYBRID,
            material_recovery_rates={
                PVMaterialType.GLASS: 98.0,
                PVMaterialType.ALUMINUM: 95.0,
                PVMaterialType.SILICON: 92.0,
                PVMaterialType.COPPER: 96.0,
                PVMaterialType.SILVER: 85.0,
            },
            technology_efficiency=0.88,
            quality_grade="B",
        )

        economics = RecyclingEconomics(
            panel_composition=composition,
            extraction_costs=costs,
            recovery_rates_model=recovery,
            panel_mass_kg=22.0,
        )

        # Get all metrics
        extraction_costs = economics.material_extraction_costs()
        recovery_rates = economics.recovery_rates()
        revenue = economics.recycling_revenue_calculation()
        env_credits = economics.environmental_credits(carbon_price_per_ton_co2=60.0)
        net_value = economics.net_economic_value(
            carbon_price_per_ton_co2=60.0,
            include_environmental_credits=True,
        )

        # Verify all calculations complete without errors
        assert extraction_costs["total_cost"] > 0
        assert recovery_rates["total_recovery_rate"] > 0
        assert revenue["net_revenue"] >= 0
        assert env_credits["total_environmental_value"] > 0

        # Check if recycling is economically viable
        print(f"\nRealistic Scenario Results:")
        print(f"Total Costs: ${extraction_costs['total_cost']:.2f}")
        print(f"Revenue: ${revenue['net_revenue']:.2f}")
        print(f"Environmental Credits: ${env_credits['total_environmental_value']:.2f}")
        print(f"Net Value: ${net_value['net_value']:.2f}")
        print(f"ROI: {net_value['roi_percent']:.1f}%")

        # Basic sanity checks
        assert net_value["total_costs"] == extraction_costs["total_cost"]
        assert net_value["total_revenue"] == revenue["net_revenue"]
