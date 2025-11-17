"""
Comprehensive test suite for PV circularity simulator models.

Tests all Pydantic models including validation, serialization, and business logic.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from pv_circularity_simulator.models import (
    # Core models
    BaseModel,
    TimestampedModel,
    UUIDModel,
    # Material models
    MaterialModel,
    MaterialType,
    MaterialProperties,
    SiliconMaterial,
    PassivationMaterial,
    ContactMaterial,
    CrystalType,
    # Cell models
    CellModel,
    CellType,
    CellArchitecture,
    CellGeometry,
    CellElectricalCharacteristics,
    CellDesign,
    # Module models
    ModuleModel,
    ModuleConfiguration,
    ElectricalParameters,
    MechanicalProperties,
    ThermalProperties,
    # System models
    SystemModel,
    SystemConfiguration,
    LocationCoordinates,
    Orientation,
    InverterConfiguration,
    MountingStructure,
    ElectricalProtection,
    # Performance models
    PerformanceModel,
    PerformanceMetrics,
    TemperatureCoefficients,
    DegradationModel,
    LossAnalysis,
    # Financial models
    FinancialModel,
    CapitalCost,
    OperatingCost,
    FinancialAnalysis,
    EndOfLifeCost,
    EndOfLifeScenario,
    MaterialRecoveryData,
    CircularityMetrics,
)


class TestMaterialModels:
    """Test suite for material models."""

    def test_material_properties_creation(self):
        """Test creating material properties."""
        props = MaterialProperties(
            name="Silicon Properties",
            density_kg_m3=2330.0,
            thermal_conductivity_w_mk=148.0,
            specific_heat_j_kgk=700.0,
            band_gap_ev=1.12,
            refractive_index=3.5,
            recyclability_percentage=95.0,
            environmental_impact_kg_co2_eq=50.0,
        )
        assert props.density_kg_m3 == 2330.0
        assert props.band_gap_ev == 1.12

    def test_material_properties_validation(self):
        """Test material properties validation."""
        with pytest.raises(ValidationError):
            # Density too high
            MaterialProperties(
                name="Invalid",
                density_kg_m3=50000.0,  # Exceeds maximum
                thermal_conductivity_w_mk=100.0,
                specific_heat_j_kgk=700.0,
            )

    def test_silicon_material_creation(self):
        """Test creating silicon material."""
        props = MaterialProperties(
            name="Mono-Si",
            density_kg_m3=2330.0,
            thermal_conductivity_w_mk=148.0,
            specific_heat_j_kgk=700.0,
            band_gap_ev=1.12,
        )

        silicon = SiliconMaterial(
            name="Monocrystalline Silicon",
            material_type=MaterialType.SILICON,
            properties=props,
            crystal_type=CrystalType.MONOCRYSTALLINE,
            purity_percentage=99.9999,
            doping_type="N-type",
            doping_concentration_cm3=1e15,
            minority_carrier_lifetime_us=100.0,
            wafer_thickness_um=180.0,
        )

        assert silicon.crystal_type == CrystalType.MONOCRYSTALLINE
        assert silicon.purity_percentage == 99.9999

    def test_silicon_material_validation(self):
        """Test silicon material validation."""
        props = MaterialProperties(
            name="Si",
            density_kg_m3=2330.0,
            thermal_conductivity_w_mk=148.0,
            specific_heat_j_kgk=700.0,
            band_gap_ev=1.5,  # Wrong band gap for silicon
        )

        with pytest.raises(ValidationError):
            SiliconMaterial(
                name="Bad Silicon",
                material_type=MaterialType.SILICON,
                properties=props,
                crystal_type=CrystalType.MONOCRYSTALLINE,
                purity_percentage=99.9999,
                doping_type="N-type",
                doping_concentration_cm3=1e15,
                minority_carrier_lifetime_us=100.0,
            )


class TestCellModels:
    """Test suite for cell models."""

    def test_cell_geometry_creation(self):
        """Test creating cell geometry."""
        geometry = CellGeometry(
            name="M6 Cell",
            width_mm=166.0,
            height_mm=166.0,
            thickness_um=180.0,
            busbar_count=9,
        )

        assert geometry.width_mm == 166.0
        assert geometry.area_cm2 is not None  # Auto-calculated

    def test_cell_electrical_characteristics(self):
        """Test cell electrical characteristics validation."""
        # Valid characteristics
        electrical = CellElectricalCharacteristics(
            name="Cell STC",
            voc_v=0.68,
            isc_a=9.5,
            vmpp_v=0.58,
            impp_a=9.0,
            pmpp_w=5.22,
            fill_factor=0.80,
            efficiency_percentage=20.0,
        )

        assert electrical.fill_factor == 0.80

        # Invalid: Vmpp >= Voc should fail
        with pytest.raises(ValidationError):
            CellElectricalCharacteristics(
                name="Bad Cell",
                voc_v=0.68,
                isc_a=9.5,
                vmpp_v=0.70,  # Greater than Voc
                impp_a=9.0,
                pmpp_w=6.3,
                fill_factor=0.80,
                efficiency_percentage=20.0,
            )

    def test_cell_model_integration(self):
        """Test complete cell model with all components."""
        # Create materials
        si_props = MaterialProperties(
            name="Si",
            density_kg_m3=2330.0,
            thermal_conductivity_w_mk=148.0,
            specific_heat_j_kgk=700.0,
            band_gap_ev=1.12,
        )

        substrate = SiliconMaterial(
            name="Mono-Si",
            material_type=MaterialType.SILICON,
            properties=si_props,
            crystal_type=CrystalType.MONOCRYSTALLINE,
            purity_percentage=99.9999,
            doping_type="P-type",
            doping_concentration_cm3=1e16,
            minority_carrier_lifetime_us=500.0,
        )

        contact_props = MaterialProperties(
            name="Silver",
            density_kg_m3=10490.0,
            thermal_conductivity_w_mk=429.0,
            specific_heat_j_kgk=235.0,
        )

        contact = ContactMaterial(
            name="Silver Contact",
            material_type=MaterialType.CONTACT,
            properties=contact_props,
            conductivity_s_m=6.3e7,
            contact_resistance_ohm_cm2=0.001,
            metal_type="Ag",
            layer_thickness_um=20.0,
        )

        geometry = CellGeometry(
            name="M6",
            width_mm=166.0,
            height_mm=166.0,
        )

        electrical = CellElectricalCharacteristics(
            name="STC",
            voc_v=0.68,
            isc_a=9.5,
            vmpp_v=0.58,
            impp_a=9.0,
            pmpp_w=5.22,
            fill_factor=0.80,
            efficiency_percentage=20.0,
        )

        design = CellDesign(
            name="PERC",
            architecture=CellArchitecture.PERC,
        )

        cell = CellModel(
            name="High-Efficiency PERC Cell",
            cell_type=CellType.PERC,
            substrate_material=substrate,
            contact_front=contact,
            contact_rear=contact,
            geometry=geometry,
            electrical=electrical,
            design=design,
        )

        assert cell.name == "High-Efficiency PERC Cell"
        assert cell.calculate_power_density() > 0


class TestModuleModels:
    """Test suite for module models."""

    def test_module_configuration(self):
        """Test module configuration validation."""
        config = ModuleConfiguration(
            name="60-cell config",
            cells_in_series=60,
            cells_in_parallel=1,
            total_cells=60,
            bypass_diodes=3,
            cells_per_bypass_diode=20,
        )

        assert config.total_cells == 60

        # Invalid: total cells mismatch
        with pytest.raises(ValidationError):
            ModuleConfiguration(
                name="Bad config",
                cells_in_series=60,
                cells_in_parallel=1,
                total_cells=72,  # Mismatch
                bypass_diodes=3,
                cells_per_bypass_diode=20,
            )

    def test_electrical_parameters(self):
        """Test module electrical parameters."""
        params = ElectricalParameters(
            name="Module STC",
            pmax_w=400.0,
            voc_v=48.0,
            isc_a=9.5,
            vmpp_v=40.0,
            impp_a=10.0,
            efficiency_percentage=20.0,
        )

        assert params.pmax_w == 400.0


class TestSystemModels:
    """Test suite for system models."""

    def test_location_coordinates(self):
        """Test location coordinates validation."""
        location = LocationCoordinates(
            name="San Francisco",
            latitude=37.7749,
            longitude=-122.4194,
            altitude_m=16.0,
            timezone="America/Los_Angeles",
            location_name="San Francisco, CA, USA",
        )

        assert location.latitude == 37.7749

        # Invalid latitude
        with pytest.raises(ValidationError):
            LocationCoordinates(
                name="Invalid",
                latitude=100.0,  # Out of range
                longitude=0.0,
                timezone="UTC",
                location_name="Invalid",
            )

    def test_system_configuration(self):
        """Test system configuration validation."""
        config = SystemConfiguration(
            name="10kW System",
            modules_per_string=20,
            strings_in_parallel=2,
            total_modules=40,
            dc_capacity_w=16000.0,
            ac_capacity_w=15000.0,
            dc_ac_ratio=1.067,
            string_voltage_voc=960.0,
            string_voltage_vmpp=800.0,
            string_current_isc=10.0,
        )

        assert config.total_modules == 40


class TestPerformanceModels:
    """Test suite for performance models."""

    def test_performance_metrics(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            power_output_w=380.0,
            voltage_v=38.0,
            current_a=10.0,
            irradiance_w_m2=1000.0,
            cell_temperature_c=45.0,
            ambient_temperature_c=25.0,
        )

        assert metrics.power_output_w == 380.0

    def test_degradation_model(self):
        """Test degradation model calculations."""
        degradation = DegradationModel(
            name="Standard degradation",
            initial_degradation_percentage=2.0,
            annual_degradation_rate_percentage=0.5,
            lifetime_years=25,
        )

        # Test power retention calculation
        retention_year_0 = degradation.calculate_power_retention(0.0)
        assert retention_year_0 == 1.0

        retention_year_25 = degradation.calculate_power_retention(25.0)
        assert 0.7 < retention_year_25 < 0.9  # Should have some degradation

    def test_loss_analysis(self):
        """Test loss analysis and performance ratio."""
        losses = LossAnalysis(
            name="System losses",
            soiling_loss_percentage=2.0,
            shading_loss_percentage=1.0,
            temperature_loss_percentage=5.0,
            inverter_loss_percentage=3.0,
        )

        pr = losses.calculate_performance_ratio()
        assert 0.0 < pr < 1.0  # PR should be between 0 and 1


class TestFinancialModels:
    """Test suite for financial models."""

    def test_capital_cost(self):
        """Test capital cost model."""
        capex = CapitalCost(
            name="10kW System CAPEX",
            module_cost_usd_per_wp=0.25,
            inverter_cost_usd=2000.0,
            installation_labor_cost_usd=3000.0,
            total_capex_usd=8000.0,
        )

        assert capex.total_capex_usd == 8000.0

        # Test cost per watt calculation
        cost_per_watt = capex.calculate_cost_per_watt(10000.0)
        assert cost_per_watt == 0.8

    def test_operating_cost(self):
        """Test operating cost model."""
        opex = OperatingCost(
            name="Annual O&M",
            maintenance_annual_usd=200.0,
            insurance_annual_usd=100.0,
            total_opex_annual_usd=300.0,
        )

        # Test escalation
        opex_year_5 = opex.calculate_opex_at_year(5)
        assert opex_year_5 > 300.0  # Should be higher due to escalation

    def test_end_of_life_scenario(self):
        """Test end-of-life scenario validation."""
        scenario = EndOfLifeScenario(
            name="Best case",
            recycling_percentage=80.0,
            refurbishment_percentage=15.0,
            reuse_percentage=5.0,
            landfill_percentage=0.0,
        )

        assert scenario.recycling_percentage == 80.0

        # Invalid: doesn't sum to 100%
        with pytest.raises(ValidationError):
            EndOfLifeScenario(
                name="Bad scenario",
                recycling_percentage=50.0,
                refurbishment_percentage=20.0,
                reuse_percentage=10.0,
                landfill_percentage=10.0,  # Sum = 90%
            )

    def test_material_recovery_data(self):
        """Test material recovery calculations."""
        recovery = MaterialRecoveryData(
            name="Silicon recovery",
            material_type="silicon",
            total_mass_kg=100.0,
            recovery_rate_percentage=95.0,
            recovered_mass_kg=95.0,
            market_value_usd_per_kg=5.0,
            recovery_cost_usd_per_kg=2.0,
        )

        net_value = recovery.calculate_net_value()
        assert net_value == 95.0 * (5.0 - 2.0)  # (market - cost) * mass

    def test_circularity_metrics(self):
        """Test circularity metrics."""
        metrics = CircularityMetrics(
            name="System circularity",
            material_circularity_indicator=0.75,
            recyclability_rate_percentage=90.0,
            total_waste_kg=1000.0,
            waste_diverted_from_landfill_kg=900.0,
        )

        diversion_rate = metrics.calculate_waste_diversion_rate()
        assert diversion_rate == 90.0


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_model_dump(self):
        """Test model serialization to dict."""
        props = MaterialProperties(
            name="Test",
            density_kg_m3=2000.0,
            thermal_conductivity_w_mk=100.0,
            specific_heat_j_kgk=700.0,
        )

        data = props.model_dump()
        assert isinstance(data, dict)
        assert data["density_kg_m3"] == 2000.0

    def test_model_dump_json(self):
        """Test model serialization to JSON."""
        props = MaterialProperties(
            name="Test",
            density_kg_m3=2000.0,
            thermal_conductivity_w_mk=100.0,
            specific_heat_j_kgk=700.0,
        )

        json_data = props.model_dump_json()
        assert isinstance(json_data, str)
        assert "2000" in json_data

    def test_model_parse(self):
        """Test model deserialization from dict."""
        data = {
            "name": "Test",
            "density_kg_m3": 2000.0,
            "thermal_conductivity_w_mk": 100.0,
            "specific_heat_j_kgk": 700.0,
        }

        props = MaterialProperties(**data)
        assert props.density_kg_m3 == 2000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
