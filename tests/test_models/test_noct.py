"""Tests for NOCT data models."""

import pytest
from pv_simulator.models.noct import (
    NOCTTestConditions,
    NOCTSpecification,
    ModuleNOCTData,
)


class TestNOCTTestConditions:
    """Test NOCTTestConditions model."""

    def test_create_standard_noct_conditions(self):
        """Test creating standard NOCT test conditions."""
        conditions = NOCTTestConditions()
        assert conditions.irradiance == 800.0
        assert conditions.ambient_temp == 20.0
        assert conditions.wind_speed == 1.0
        assert conditions.mounting_type == "open_rack"

    def test_custom_noct_conditions(self):
        """Test creating custom NOCT test conditions."""
        conditions = NOCTTestConditions(
            irradiance=850.0,
            ambient_temp=22.0,
            wind_speed=1.5,
        )
        assert conditions.irradiance == 850.0
        assert conditions.ambient_temp == 22.0


class TestNOCTSpecification:
    """Test NOCTSpecification model."""

    def test_create_noct_spec(self):
        """Test creating NOCT specification."""
        spec = NOCTSpecification(
            noct_celsius=45.0,
        )
        assert spec.noct_celsius == 45.0
        assert spec.test_conditions.irradiance == 800.0

    def test_noct_with_test_conditions(self):
        """Test NOCT spec with custom test conditions."""
        test_conds = NOCTTestConditions(irradiance=850.0)
        spec = NOCTSpecification(
            noct_celsius=46.0,
            test_conditions=test_conds,
        )
        assert spec.noct_celsius == 46.0
        assert spec.test_conditions.irradiance == 850.0


class TestModuleNOCTData:
    """Test ModuleNOCTData model."""

    def test_create_module_noct_data(self, noct_specification):
        """Test creating module NOCT data."""
        module = ModuleNOCTData(
            module_id="B03-00001",
            manufacturer="TestMfg",
            model_name="TestModel",
            technology="mono_si",
            noct_spec=noct_specification,
            temp_coeff_power=-0.0040,
            temp_coeff_voc=-0.0030,
            rated_power_stc=400.0,
            efficiency_stc=20.0,
            module_area=2.0,
        )
        assert module.module_id == "B03-00001"
        assert module.manufacturer == "TestMfg"
        assert module.noct_spec.noct_celsius == 45.0

    def test_b03_module_id_validation(self, noct_specification):
        """Test B03 module ID format validation."""
        # Valid B03 ID
        module = ModuleNOCTData(
            module_id="B03-00001",
            manufacturer="TestMfg",
            model_name="TestModel",
            technology="mono_si",
            noct_spec=noct_specification,
            temp_coeff_power=-0.0040,
            temp_coeff_voc=-0.0030,
            rated_power_stc=400.0,
            efficiency_stc=20.0,
            module_area=2.0,
        )
        assert module.module_id == "B03-00001"

        # Invalid B03 ID format should raise error
        with pytest.raises(Exception):
            ModuleNOCTData(
                module_id="B03-ABC",
                manufacturer="TestMfg",
                model_name="TestModel",
                technology="mono_si",
                noct_spec=noct_specification,
                temp_coeff_power=-0.0040,
                temp_coeff_voc=-0.0030,
                rated_power_stc=400.0,
                efficiency_stc=20.0,
                module_area=2.0,
            )

    def test_get_thermal_parameters(self, sample_module_noct_data):
        """Test extracting thermal parameters."""
        params = sample_module_noct_data.get_thermal_parameters()
        assert "heat_capacity" in params
        assert "absorptivity" in params
        assert "emissivity" in params
        assert "noct" in params

    def test_get_temperature_coefficients(self, sample_module_noct_data):
        """Test extracting temperature coefficients."""
        coeffs = sample_module_noct_data.get_temperature_coefficients()
        assert "power" in coeffs
        assert "voc" in coeffs
        assert "isc" in coeffs

    def test_estimate_cell_temperature(self, sample_module_noct_data):
        """Test simple cell temperature estimation."""
        cell_temp = sample_module_noct_data.estimate_cell_temperature(
            ambient_temp=25.0,
            irradiance=1000.0,
        )
        assert cell_temp > 25.0  # Should be higher than ambient
        assert cell_temp < 80.0  # Should be reasonable
