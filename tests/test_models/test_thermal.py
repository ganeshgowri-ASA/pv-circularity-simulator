"""Tests for thermal data models."""

import pytest
from datetime import datetime
from pv_simulator.models.thermal import (
    TemperatureConditions,
    ThermalParameters,
    MountingConfiguration,
    TemperatureCoefficients,
    HeatTransferCoefficients,
    ThermalModelOutput,
)


class TestTemperatureConditions:
    """Test TemperatureConditions model."""

    def test_create_basic_conditions(self):
        """Test creating basic temperature conditions."""
        conditions = TemperatureConditions(
            ambient_temp=25.0,
            irradiance=1000.0,
            wind_speed=3.0,
        )
        assert conditions.ambient_temp == 25.0
        assert conditions.irradiance == 1000.0
        assert conditions.wind_speed == 3.0

    def test_sky_temperature_calculation(self):
        """Test automatic sky temperature calculation."""
        conditions = TemperatureConditions(
            ambient_temp=25.0,
            irradiance=1000.0,
            wind_speed=3.0,
        )
        # Sky temp should be ambient - 10°C
        assert conditions.sky_temperature == 15.0

    def test_validation_ranges(self):
        """Test validation of input ranges."""
        # Valid ranges
        conditions = TemperatureConditions(
            ambient_temp=-10.0,
            irradiance=0.0,
            wind_speed=0.0,
        )
        assert conditions.ambient_temp == -10.0

        # Invalid temperature (too low)
        with pytest.raises(Exception):
            TemperatureConditions(
                ambient_temp=-60.0,
                irradiance=1000.0,
                wind_speed=3.0,
            )


class TestMountingConfiguration:
    """Test MountingConfiguration model."""

    def test_create_mounting_config(self):
        """Test creating mounting configuration."""
        mounting = MountingConfiguration(
            mounting_type="open_rack",
            tilt_angle=30.0,
        )
        assert mounting.mounting_type == "open_rack"
        assert mounting.tilt_angle == 30.0

    def test_mounting_types(self):
        """Test all mounting type options."""
        types = ["open_rack", "roof_mounted", "ground_mounted", "building_integrated"]
        for mount_type in types:
            mounting = MountingConfiguration(mounting_type=mount_type)
            assert mounting.mounting_type == mount_type


class TestThermalParameters:
    """Test ThermalParameters model."""

    def test_create_thermal_params(self):
        """Test creating thermal parameters."""
        params = ThermalParameters(
            heat_capacity=11000.0,
            absorptivity=0.9,
            emissivity=0.85,
        )
        assert params.heat_capacity == 11000.0
        assert params.absorptivity == 0.9
        assert params.emissivity == 0.85

    def test_module_area_calculation(self):
        """Test automatic module area calculation."""
        params = ThermalParameters(
            module_length=1.65,
            module_width=1.0,
        )
        assert abs(params.module_area - 1.65) < 0.01


class TestTemperatureCoefficients:
    """Test TemperatureCoefficients model."""

    def test_create_temp_coefficients(self):
        """Test creating temperature coefficients."""
        coeffs = TemperatureCoefficients(
            power=-0.0040,
            voc=-0.0030,
            isc=0.0005,
        )
        assert coeffs.power == -0.0040
        assert coeffs.voc == -0.0030
        assert coeffs.isc == 0.0005


class TestHeatTransferCoefficients:
    """Test HeatTransferCoefficients model."""

    def test_create_heat_transfer_coeffs(self):
        """Test creating heat transfer coefficients."""
        coeffs = HeatTransferCoefficients(
            convective_front=30.0,
            convective_back=20.0,
            radiative_front=5.0,
            radiative_back=5.0,
        )
        assert coeffs.convective_front == 30.0
        assert coeffs.total_front == 35.0  # Auto-calculated
        assert coeffs.total_back == 25.0  # Auto-calculated


class TestThermalModelOutput:
    """Test ThermalModelOutput model."""

    def test_create_model_output(self, standard_conditions, mounting_config):
        """Test creating thermal model output."""
        output = ThermalModelOutput(
            cell_temperature=50.0,
            module_temperature=48.0,
            model_name="TestModel",
            conditions=standard_conditions,
            mounting=mounting_config,
        )
        assert output.cell_temperature == 50.0
        assert output.module_temperature == 48.0
        assert output.model_name == "TestModel"
