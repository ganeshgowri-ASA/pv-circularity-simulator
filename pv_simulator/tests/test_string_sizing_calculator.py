"""Unit tests for StringSizingCalculator."""

import pytest
import numpy as np

from pv_simulator.system_design.models import ModuleParameters, InverterParameters, InverterType
from pv_simulator.system_design.string_sizing_calculator import StringSizingCalculator


@pytest.fixture
def sample_module():
    """Create sample module parameters."""
    return ModuleParameters(
        manufacturer="Test Solar",
        model="TS400",
        technology="mtSiMono",
        pmax=400.0,
        voc=48.5,
        isc=10.5,
        vmp=40.8,
        imp=9.8,
        temp_coeff_pmax=-0.37,
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        length=1.776,
        width=1.052,
        thickness=0.035,
        weight=21.5,
        cells_in_series=72,
        efficiency=21.4,
    )


@pytest.fixture
def sample_inverter():
    """Create sample inverter parameters."""
    return InverterParameters(
        manufacturer="Test Inverter",
        model="TI100K",
        inverter_type=InverterType.STRING,
        pac_max=100000,
        vac_nom=480,
        iac_max=150,
        pdc_max=150000,
        vdc_max=1000,
        vdc_nom=600,
        vdc_min=200,
        idc_max=200,
        num_mppt=6,
        mppt_vmin=200,
        mppt_vmax=850,
        strings_per_mppt=3,
        max_efficiency=98.5,
        weight=65,
    )


class TestStringSizingCalculator:
    """Test suite for StringSizingCalculator."""

    def test_initialization(self, sample_module, sample_inverter):
        """Test calculator initialization."""
        calc = StringSizingCalculator(
            module=sample_module,
            inverter=sample_inverter,
            site_temp_min=-10.0,
            site_temp_max=70.0,
        )
        assert calc.module == sample_module
        assert calc.inverter == sample_inverter
        assert calc.site_temp_min == -10.0
        assert calc.site_temp_max == 70.0

    def test_temperature_voltage_correction(self, sample_module, sample_inverter):
        """Test temperature voltage correction."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        # At 25°C (STC), voltage should remain unchanged
        v_stc = calc.temperature_voltage_correction(48.5, 25.0, -0.28)
        assert abs(v_stc - 48.5) < 0.01

        # At cold temperature, voltage should increase
        v_cold = calc.temperature_voltage_correction(48.5, -10.0, -0.28)
        assert v_cold > 48.5

        # At hot temperature, voltage should decrease
        v_hot = calc.temperature_voltage_correction(48.5, 70.0, -0.28)
        assert v_hot < 48.5

    def test_calculate_max_string_length(self, sample_module, sample_inverter):
        """Test maximum string length calculation."""
        calc = StringSizingCalculator(sample_module, sample_inverter, site_temp_min=-10.0)

        max_modules = calc.calculate_max_string_length()
        assert max_modules > 0
        assert isinstance(max_modules, int)

        # Verify it respects inverter Vdc_max
        voc_cold = calc.temperature_voltage_correction(
            sample_module.voc, -10.0, sample_module.temp_coeff_voc
        )
        assert max_modules * voc_cold <= sample_inverter.vdc_max

    def test_calculate_min_string_length(self, sample_module, sample_inverter):
        """Test minimum string length calculation."""
        calc = StringSizingCalculator(sample_module, sample_inverter, site_temp_max=70.0)

        min_modules = calc.calculate_min_string_length()
        assert min_modules > 0
        assert isinstance(min_modules, int)

        # Verify it respects MPPT minimum
        vmp_hot = calc.temperature_voltage_correction(
            sample_module.vmp, 70.0, sample_module.temp_coeff_voc
        )
        assert min_modules * vmp_hot >= sample_inverter.mppt_vmin

    def test_calculate_max_strings_per_mppt(self, sample_module, sample_inverter):
        """Test maximum strings per MPPT calculation."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        max_strings = calc.calculate_max_strings_per_mppt()
        assert max_strings > 0
        assert isinstance(max_strings, int)
        assert max_strings <= sample_inverter.strings_per_mppt

    def test_validate_string_configuration_valid(self, sample_module, sample_inverter):
        """Test validation of valid string configuration."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        modules_per_string = 15
        strings_per_mppt = 2

        is_valid, message = calc.validate_string_configuration(
            modules_per_string, strings_per_mppt
        )
        assert is_valid
        assert "valid" in message.lower()

    def test_validate_string_configuration_invalid(self, sample_module, sample_inverter):
        """Test validation of invalid string configuration."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        # Too many modules per string
        modules_per_string = 100
        strings_per_mppt = 2

        is_valid, message = calc.validate_string_configuration(
            modules_per_string, strings_per_mppt
        )
        assert not is_valid

    def test_design_optimal_string(self, sample_module, sample_inverter):
        """Test optimal string design."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        string_config = calc.design_optimal_string()

        assert string_config.modules_per_string > 0
        assert string_config.strings_per_mppt > 0
        assert string_config.voc_stc is not None
        assert string_config.vmp_stc is not None

    def test_fuse_sizing(self, sample_module, sample_inverter):
        """Test fuse sizing calculation."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        fuse_info = calc.fuse_sizing(strings_per_mppt=2)

        assert fuse_info["min_fuse_rating"] > 0
        assert fuse_info["recommended_fuse_rating"] >= fuse_info["min_fuse_rating"]
        assert fuse_info["string_isc"] == sample_module.isc

    def test_string_mismatch_analysis(self, sample_module, sample_inverter):
        """Test string mismatch analysis."""
        calc = StringSizingCalculator(sample_module, sample_inverter)

        # No mismatch
        mismatch_base = calc.string_mismatch_analysis(
            azimuth_diff=0.0, tilt_diff=0.0, shading_factor=0.0
        )
        assert mismatch_base >= 1.0  # Base mismatch

        # With orientation mismatch
        mismatch_azimuth = calc.string_mismatch_analysis(
            azimuth_diff=30.0, tilt_diff=0.0, shading_factor=0.0
        )
        assert mismatch_azimuth > mismatch_base

        # With shading
        mismatch_shading = calc.string_mismatch_analysis(
            azimuth_diff=0.0, tilt_diff=0.0, shading_factor=0.2
        )
        assert mismatch_shading > mismatch_base
