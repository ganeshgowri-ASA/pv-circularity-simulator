"""
Pytest configuration and shared fixtures for PV Circularity Simulator tests.

This module provides common test fixtures and configuration for all test suites.
"""

import pytest
from datetime import datetime
from typing import List

import numpy as np

from pv_circularity_simulator.core.iec63202.models import (
    CTMTestConfig,
    CellProperties,
    ModuleConfiguration,
    ReferenceDeviceData,
    FlashSimulatorData,
    IVCurveData,
    CellTechnology,
    FlashSimulatorType,
)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def perc_cell_properties() -> CellProperties:
    """Standard PERC cell properties for testing."""
    return CellProperties(
        technology=CellTechnology.PERC,
        area=244.3,
        efficiency=22.8,
        voc=0.682,
        isc=8.52,
        vmp=0.580,
        imp=8.24,
        pmax=5.22,
        temperature_coefficient_pmax=-0.39,
        temperature_coefficient_voc=-0.0029,
        temperature_coefficient_isc=0.0005,
        manufacturer="Test Manufacturer",
        batch_number="BATCH-2025-001",
    )


@pytest.fixture
def hjt_cell_properties() -> CellProperties:
    """HJT cell properties for testing."""
    return CellProperties(
        technology=CellTechnology.HJT,
        area=244.3,
        efficiency=24.5,
        voc=0.730,
        isc=8.45,
        vmp=0.640,
        imp=8.20,
        pmax=5.65,
        temperature_coefficient_pmax=-0.30,
        temperature_coefficient_voc=-0.0024,
        temperature_coefficient_isc=0.0003,
    )


@pytest.fixture
def standard_module_config() -> ModuleConfiguration:
    """Standard 60-cell module configuration."""
    return ModuleConfiguration(
        num_cells_series=60,
        num_strings_parallel=1,
        cell_spacing=2.0,
        encapsulant_type="EVA",
        glass_type="3.2mm AR-coated",
        backsheet_type="White TPT",
        bypass_diodes=3,
    )


@pytest.fixture
def bifacial_module_config() -> ModuleConfiguration:
    """Bifacial module configuration."""
    return ModuleConfiguration(
        num_cells_series=72,
        num_strings_parallel=1,
        cell_spacing=2.5,
        encapsulant_type="POE",
        glass_type="2.0mm AR-coated dual glass",
        backsheet_type="Glass",
        bypass_diodes=3,
    )


@pytest.fixture
def calibrated_reference_device() -> ReferenceDeviceData:
    """Calibrated reference device for testing."""
    return ReferenceDeviceData(
        device_id="NREL-REF-2025-001",
        calibration_date=datetime(2025, 1, 15),
        calibration_lab="NREL PV Performance Lab",
        calibration_certificate="NREL-CAL-2025-0123",
        short_circuit_current=8.520,
        responsivity=0.00852,
        temperature_coefficient=0.00050,
        spectral_response={
            300: 0.10,
            400: 0.50,
            500: 0.75,
            600: 0.88,
            700: 0.95,
            800: 0.98,
            900: 0.94,
            1000: 0.80,
            1100: 0.60,
            1200: 0.35,
        },
        uncertainty_isc=1.2,
        uncertainty_temperature=0.15,
        traceability_chain="WPVS → NREL Secondary → Working Standard",
        next_calibration_due=datetime(2026, 1, 15),
    )


@pytest.fixture
def aaa_flash_simulator() -> FlashSimulatorData:
    """Class AAA flash simulator."""
    return FlashSimulatorData(
        simulator_type=FlashSimulatorType.LED,
        spectral_distribution={
            350: 0.35,
            400: 0.95,
            450: 1.45,
            500: 1.75,
            550: 1.80,
            600: 1.65,
            650: 1.50,
            700: 1.35,
            750: 1.20,
            800: 1.05,
            850: 0.90,
            900: 0.75,
            950: 0.60,
            1000: 0.50,
        },
        spatial_uniformity=99.0,
        temporal_stability=99.5,
        flash_duration=8.0,
        class_rating="AAA",
    )


def generate_iv_curve(
    voc: float,
    isc: float,
    ff: float = 0.80,
    num_points: int = 50,
    temperature: float = 25.0,
    irradiance: float = 1000.0
) -> IVCurveData:
    """
    Generate realistic IV curve using single-diode model approximation.

    Args:
        voc: Open-circuit voltage (V)
        isc: Short-circuit current (A)
        ff: Fill factor (dimensionless)
        num_points: Number of IV points
        temperature: Cell temperature (°C)
        irradiance: Irradiance (W/m²)

    Returns:
        IVCurveData instance with generated IV curve
    """
    # Generate voltage points
    voltage = np.linspace(0, voc, num_points)

    # Simple single-diode approximation
    # I = Isc * (1 - C1*(exp(V/(C2*Voc)) - 1))
    # where C1 and C2 are chosen to match FF
    c2 = 0.85  # Typical for c-Si
    c1 = (1 - isc * c2 * voc / (isc * voc * ff)) / (np.exp(1/c2) - 1)

    current = isc * (1 - c1 * (np.exp(voltage / (c2 * voc)) - 1))
    current = np.maximum(current, 0)  # Ensure non-negative

    return IVCurveData(
        voltage=voltage.tolist(),
        current=current.tolist(),
        temperature=temperature,
        irradiance=irradiance,
    )


@pytest.fixture
def sample_cell_iv_curves(perc_cell_properties) -> List[IVCurveData]:
    """Generate sample cell IV curves with realistic variation."""
    curves = []
    np.random.seed(42)  # For reproducibility

    for _ in range(5):
        # Add small random variation (±2%)
        voc = perc_cell_properties.voc * (1 + np.random.normal(0, 0.01))
        isc = perc_cell_properties.isc * (1 + np.random.normal(0, 0.01))

        curves.append(generate_iv_curve(
            voc=voc,
            isc=isc,
            ff=perc_cell_properties.fill_factor,
        ))

    return curves


@pytest.fixture
def sample_module_iv_curves(
    perc_cell_properties,
    standard_module_config
) -> List[IVCurveData]:
    """Generate sample module IV curves."""
    curves = []
    np.random.seed(42)

    n_cells = standard_module_config.total_cells
    module_voc = perc_cell_properties.voc * n_cells
    module_isc = perc_cell_properties.isc

    for _ in range(3):
        # Add small variation
        voc = module_voc * (1 + np.random.normal(0, 0.005))
        isc = module_isc * (1 + np.random.normal(0, 0.005))
        ff = perc_cell_properties.fill_factor * 0.98  # Slight reduction due to CTM losses

        curves.append(generate_iv_curve(
            voc=voc,
            isc=isc,
            ff=ff,
        ))

    return curves
