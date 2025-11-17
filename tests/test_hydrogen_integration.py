"""
Unit tests for Hydrogen System Integration module.

Tests cover all core functionality: electrolyzer modeling, storage design,
fuel cell integration, and power-to-X analysis.
"""

import pytest
import numpy as np
from pv_circularity_simulator.hydrogen import (
    HydrogenIntegrator,
    ElectrolyzerConfig,
    ElectrolyzerType,
    StorageConfig,
    StorageType,
    FuelCellConfig,
    FuelCellType,
    PowerToXConfig,
    PowerToXPathway,
)


class TestElectrolyzerModeling:
    """Test suite for electrolyzer modeling functionality."""

    def test_basic_electrolyzer_operation(self):
        """Test basic electrolyzer operation with constant power."""
        integrator = HydrogenIntegrator()
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
        )

        # 100 hours at 80% load
        power_profile = [800.0] * 100
        results = integrator.electrolyzer_modeling(
            config=config,
            power_input_profile=power_profile,
            timestep_hours=1.0,
        )

        # Verify results
        assert results.h2_production_kg > 0
        assert results.energy_consumption_kwh == pytest.approx(80000.0, rel=0.01)
        assert 0 < results.average_efficiency <= config.efficiency
        assert 0 < results.capacity_factor <= 1.0

    def test_variable_power_profile(self):
        """Test electrolyzer with variable renewable power."""
        integrator = HydrogenIntegrator()
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.ALKALINE,
            rated_power_kw=5000.0,
            efficiency=0.65,
            min_load_fraction=0.2,
        )

        # Variable power: 24 hours with sine wave
        power_profile = [
            5000.0 * max(0, np.sin(i * np.pi / 12)) for i in range(24)
        ]
        results = integrator.electrolyzer_modeling(
            config=config,
            power_input_profile=power_profile,
            timestep_hours=1.0,
        )

        assert results.h2_production_kg > 0
        assert results.capacity_factor < 1.0  # Variable operation
        assert results.operating_hours <= 24

    def test_below_minimum_load(self):
        """Test that electrolyzer doesn't operate below minimum load."""
        integrator = HydrogenIntegrator()
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
            min_load_fraction=0.1,
        )

        # Power below minimum load (5% of rated)
        power_profile = [50.0] * 10
        results = integrator.electrolyzer_modeling(
            config=config,
            power_input_profile=power_profile,
            timestep_hours=1.0,
        )

        # Should produce no hydrogen
        assert results.h2_production_kg == 0
        assert results.operating_hours == 0

    def test_invalid_power_profile(self):
        """Test validation of power input."""
        integrator = HydrogenIntegrator()
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
        )

        # Empty profile
        with pytest.raises(ValueError):
            integrator.electrolyzer_modeling(config, [], 1.0)

        # Negative values
        with pytest.raises(ValueError):
            integrator.electrolyzer_modeling(config, [-100.0, 500.0], 1.0)


class TestStorageDesign:
    """Test suite for hydrogen storage design."""

    def test_basic_storage_operation(self):
        """Test basic storage charge/discharge cycle."""
        integrator = HydrogenIntegrator()
        config = StorageConfig(
            storage_type=StorageType.COMPRESSED_GAS,
            capacity_kg=1000.0,
            charging_rate_kg_h=50.0,
            discharging_rate_kg_h=50.0,
            round_trip_efficiency=0.90,
        )

        # Charge for 10 hours, discharge for 10 hours
        charge_profile = [50.0] * 10 + [0.0] * 10
        discharge_profile = [0.0] * 10 + [40.0] * 10

        results = integrator.h2_storage_design(
            config=config,
            charge_profile=charge_profile,
            discharge_profile=discharge_profile,
            timestep_hours=1.0,
            initial_soc_fraction=0.5,
        )

        assert results.total_charged_kg > 0
        assert results.total_discharged_kg > 0
        assert results.total_discharged_kg < results.total_charged_kg  # Losses
        assert 0 <= results.average_soc <= 1.0

    def test_storage_efficiency(self):
        """Test round-trip efficiency is applied correctly."""
        integrator = HydrogenIntegrator()
        config = StorageConfig(
            storage_type=StorageType.COMPRESSED_GAS,
            capacity_kg=10000.0,
            charging_rate_kg_h=100.0,
            discharging_rate_kg_h=100.0,
            round_trip_efficiency=0.80,
            self_discharge_rate_per_day=0.0,  # Disable for this test
        )

        # Charge 100 kg, then discharge all
        charge_profile = [100.0] * 1 + [0.0] * 10
        discharge_profile = [0.0] * 1 + [100.0] * 10

        results = integrator.h2_storage_design(
            config=config,
            charge_profile=charge_profile,
            discharge_profile=discharge_profile,
            timestep_hours=1.0,
            initial_soc_fraction=0.5,
        )

        # Check efficiency: discharged should be ~80% of charged
        efficiency = results.total_discharged_kg / results.total_charged_kg
        assert efficiency == pytest.approx(config.round_trip_efficiency, rel=0.05)

    def test_soc_limits(self):
        """Test that SOC respects min/max limits."""
        integrator = HydrogenIntegrator()
        config = StorageConfig(
            storage_type=StorageType.COMPRESSED_GAS,
            capacity_kg=1000.0,
            charging_rate_kg_h=100.0,
            discharging_rate_kg_h=100.0,
            min_soc_fraction=0.2,
            max_soc_fraction=0.9,
        )

        # Try to fully charge
        charge_profile = [100.0] * 20
        discharge_profile = [0.0] * 20

        results = integrator.h2_storage_design(
            config=config,
            charge_profile=charge_profile,
            discharge_profile=discharge_profile,
            timestep_hours=1.0,
            initial_soc_fraction=0.2,
        )

        # Verify SOC didn't exceed max
        assert results.average_soc <= config.max_soc_fraction


class TestFuelCellIntegration:
    """Test suite for fuel cell integration."""

    def test_basic_fuel_cell_operation(self):
        """Test basic fuel cell operation."""
        integrator = HydrogenIntegrator()
        config = FuelCellConfig(
            fuel_cell_type=FuelCellType.PEMFC,
            rated_power_kw=500.0,
            efficiency=0.55,
        )

        # Constant 300 kW demand for 10 hours
        power_demand = [300.0] * 10
        results = integrator.fuel_cell_integration(
            config=config,
            power_demand_profile=power_demand,
            timestep_hours=1.0,
        )

        assert results.electrical_output_kwh == pytest.approx(3000.0, rel=0.01)
        assert results.h2_consumed_kg > 0
        assert results.average_efficiency > 0
        assert results.capacity_factor > 0

    def test_fuel_cell_chp(self):
        """Test combined heat and power operation."""
        integrator = HydrogenIntegrator()
        config = FuelCellConfig(
            fuel_cell_type=FuelCellType.PEMFC,
            rated_power_kw=1000.0,
            efficiency=0.55,
            heat_recovery_fraction=0.35,
            cogeneration_enabled=True,
        )

        power_demand = [800.0] * 100
        heat_demand = [300.0] * 100

        results = integrator.fuel_cell_integration(
            config=config,
            power_demand_profile=power_demand,
            timestep_hours=1.0,
            heat_demand_profile=heat_demand,
        )

        assert results.electrical_output_kwh > 0
        assert results.thermal_output_kwh > 0  # CHP enabled
        assert results.cogeneration_efficiency is not None
        assert results.cogeneration_efficiency > results.average_efficiency

    def test_h2_supply_limitation(self):
        """Test fuel cell operation with limited H2 supply."""
        integrator = HydrogenIntegrator()
        config = FuelCellConfig(
            fuel_cell_type=FuelCellType.PEMFC,
            rated_power_kw=1000.0,
            efficiency=0.55,
        )

        power_demand = [800.0] * 10
        # Limited H2 supply (enough for ~50% operation)
        h2_lhv = 33.33
        required_h2_kg_h = 800.0 / (0.55 * h2_lhv)
        h2_supply = [required_h2_kg_h * 0.5] * 10  # 50% of requirement

        results = integrator.fuel_cell_integration(
            config=config,
            power_demand_profile=power_demand,
            h2_supply_profile=h2_supply,
            timestep_hours=1.0,
        )

        # Output should be limited by H2 supply
        assert results.electrical_output_kwh < 8000.0  # Less than full demand


class TestPowerToXAnalysis:
    """Test suite for Power-to-X analysis."""

    def test_power_to_h2_pathway(self):
        """Test direct Power-to-H2 pathway."""
        integrator = HydrogenIntegrator()

        electrolyzer_config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
        )

        ptx_config = PowerToXConfig(
            pathway=PowerToXPathway.POWER_TO_H2,
            electrolyzer_config=electrolyzer_config,
            conversion_efficiency=1.0,  # Direct H2, no conversion
        )

        power_profile = [800.0] * 100
        results = integrator.power_to_x_analysis(
            config=ptx_config,
            power_input_profile=power_profile,
            timestep_hours=1.0,
        )

        assert results.product_output_kg > 0
        assert results.product_output_kg == results.h2_intermediate_kg
        assert results.co2_consumed_kg == 0.0  # No CO2 needed for H2

    def test_power_to_methanol_pathway(self):
        """Test Power-to-Methanol pathway."""
        integrator = HydrogenIntegrator()

        electrolyzer_config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=5000.0,
            efficiency=0.70,
        )

        ptx_config = PowerToXConfig(
            pathway=PowerToXPathway.POWER_TO_METHANOL,
            electrolyzer_config=electrolyzer_config,
            conversion_efficiency=0.75,
            co2_source="DAC",
        )

        power_profile = [4000.0] * 100
        co2_profile = [200.0] * 100  # kg/h

        results = integrator.power_to_x_analysis(
            config=ptx_config,
            power_input_profile=power_profile,
            timestep_hours=1.0,
            co2_availability_profile=co2_profile,
        )

        assert results.product_output_kg > 0
        assert results.h2_intermediate_kg > 0
        assert results.co2_consumed_kg > 0
        assert results.overall_efficiency > 0
        assert results.levelized_cost_product > 0

    def test_missing_co2_for_methane(self):
        """Test that error is raised if CO2 not provided for methane pathway."""
        integrator = HydrogenIntegrator()

        electrolyzer_config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
        )

        ptx_config = PowerToXConfig(
            pathway=PowerToXPathway.POWER_TO_METHANE,
            electrolyzer_config=electrolyzer_config,
            conversion_efficiency=0.80,
        )

        power_profile = [800.0] * 10

        with pytest.raises(ValueError, match="requires CO2"):
            integrator.power_to_x_analysis(
                config=ptx_config,
                power_input_profile=power_profile,
                timestep_hours=1.0,
                # No CO2 profile provided
            )


class TestPydanticValidation:
    """Test Pydantic model validation."""

    def test_electrolyzer_config_validation(self):
        """Test electrolyzer config validation."""
        # Valid config
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
        )
        assert config.rated_power_kw == 1000.0

        # Invalid: negative power
        with pytest.raises(ValueError):
            ElectrolyzerConfig(
                electrolyzer_type=ElectrolyzerType.PEM,
                rated_power_kw=-100.0,
            )

        # Invalid: efficiency out of bounds
        with pytest.raises(ValueError):
            ElectrolyzerConfig(
                electrolyzer_type=ElectrolyzerType.PEM,
                rated_power_kw=1000.0,
                efficiency=1.5,  # > 1.0
            )

    def test_storage_config_validation(self):
        """Test storage config validation."""
        # Valid config
        config = StorageConfig(
            storage_type=StorageType.COMPRESSED_GAS,
            capacity_kg=1000.0,
            charging_rate_kg_h=50.0,
            discharging_rate_kg_h=50.0,
        )
        assert config.capacity_kg == 1000.0

        # Invalid: max_soc <= min_soc
        with pytest.raises(ValueError):
            StorageConfig(
                storage_type=StorageType.COMPRESSED_GAS,
                capacity_kg=1000.0,
                charging_rate_kg_h=50.0,
                discharging_rate_kg_h=50.0,
                min_soc_fraction=0.8,
                max_soc_fraction=0.5,  # Less than min!
            )

    def test_computed_fields(self):
        """Test computed fields in models."""
        # Electrolyzer computed field
        config = ElectrolyzerConfig(
            electrolyzer_type=ElectrolyzerType.PEM,
            rated_power_kw=1000.0,
            efficiency=0.68,
        )
        h2_lhv = 33.33
        expected_h2_rate = (1000.0 * 0.68) / h2_lhv
        assert config.h2_production_rate_kg_h == pytest.approx(expected_h2_rate)

        # Storage computed field
        storage_config = StorageConfig(
            storage_type=StorageType.COMPRESSED_GAS,
            capacity_kg=1000.0,
            charging_rate_kg_h=50.0,
            discharging_rate_kg_h=50.0,
            min_soc_fraction=0.1,
            max_soc_fraction=0.9,
        )
        expected_usable = 1000.0 * (0.9 - 0.1)
        assert storage_config.usable_capacity_kg == pytest.approx(expected_usable)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
