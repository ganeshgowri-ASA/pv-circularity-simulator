"""
Pytest configuration and shared fixtures.
"""

from datetime import datetime

import numpy as np
import pytest

from pv_circularity_simulator.core.models import (
    ElectricalParameters,
    IVCurveData,
    ThermalImageData,
    ThermalImageMetadata,
)


@pytest.fixture
def sample_thermal_metadata():
    """Create sample thermal image metadata."""
    return ThermalImageMetadata(
        timestamp=datetime.now(),
        camera_model="FLIR E95",
        ambient_temp=25.0,
        measurement_distance=5.0,
        emissivity=0.90,
        irradiance=1000.0,
        module_id="TEST-001",
    )


@pytest.fixture
def sample_thermal_image(sample_thermal_metadata):
    """Create sample thermal image with synthetic hotspot."""
    # Create base temperature field
    temps = np.random.normal(40.0, 2.0, (100, 100))

    # Add a hotspot
    temps[20:30, 20:30] += 25.0  # 25°C hotter than median

    return ThermalImageData(
        temperature_matrix=temps,
        metadata=sample_thermal_metadata,
        width=100,
        height=100,
    )


@pytest.fixture
def sample_iv_curve_data():
    """Create sample IV curve data."""
    # Generate realistic IV curve
    v = np.linspace(0, 36, 100)
    isc = 9.0
    voc = 36.0

    # Single diode model approximation
    i = isc * (1 - (v / voc) ** 3)

    return IVCurveData(
        voltage=v,
        current=i,
        temperature=25.0,
        irradiance=1000.0,
        timestamp=datetime.now(),
        module_id="TEST-001",
        num_cells=60,
    )


@pytest.fixture
def sample_electrical_params():
    """Create sample electrical parameters."""
    return ElectricalParameters(
        voc=36.0,
        isc=9.0,
        vmp=30.0,
        imp=8.5,
        pmp=255.0,
        fill_factor=0.78,
        efficiency=0.18,
        rs=0.5,
        rsh=800.0,
        ideality_factor=1.2,
    )
