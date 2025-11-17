"""Pytest configuration and shared fixtures for testing."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pv_simulator.models.thermal import (
    TemperatureConditions,
    ThermalParameters,
    MountingConfiguration,
    TemperatureCoefficients,
)
from pv_simulator.models.noct import (
    NOCTSpecification,
    NOCTTestConditions,
    ModuleNOCTData,
)


@pytest.fixture
def standard_conditions():
    """Standard test conditions fixture."""
    return TemperatureConditions(
        ambient_temp=25.0,
        irradiance=1000.0,
        wind_speed=3.0,
    )


@pytest.fixture
def thermal_parameters():
    """Standard thermal parameters fixture."""
    return ThermalParameters()


@pytest.fixture
def mounting_config():
    """Standard mounting configuration fixture."""
    return MountingConfiguration(mounting_type="open_rack")


@pytest.fixture
def temp_coefficients():
    """Standard temperature coefficients fixture."""
    return TemperatureCoefficients(
        power=-0.0040,
        voc=-0.0030,
        isc=0.0005,
    )


@pytest.fixture
def noct_specification():
    """Standard NOCT specification fixture."""
    return NOCTSpecification(
        noct_celsius=45.0,
        test_conditions=NOCTTestConditions(),
    )


@pytest.fixture
def sample_module_noct_data(noct_specification):
    """Sample B03 module NOCT data fixture."""
    return ModuleNOCTData(
        module_id="B03-00001",
        manufacturer="TestManufacturer",
        model_name="TestModule",
        technology="mono_si",
        noct_spec=noct_specification,
        temp_coeff_power=-0.0040,
        temp_coeff_voc=-0.0030,
        temp_coeff_isc=0.0005,
        rated_power_stc=400.0,
        efficiency_stc=20.0,
        module_area=2.0,
        cell_count=72,
        b03_verified=True,
        data_source="test_fixture",
    )
