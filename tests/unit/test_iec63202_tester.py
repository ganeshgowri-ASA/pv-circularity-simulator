"""
Unit tests for IEC 63202 CTM Tester.

Tests the core IEC63202CTMTester class including:
- Reference cell measurements
- Module flash testing
- CTM ratio calculation
- Loss analysis
- Validation against IEC 63202
- Certificate generation
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
from pv_circularity_simulator.core.iec63202.tester import IEC63202CTMTester
from pv_circularity_simulator.core.iec63202.loss_analyzer import CTMPowerLossAnalyzer


@pytest.fixture
def sample_cell_properties() -> CellProperties:
    """Create sample cell properties for testing."""
    return CellProperties(
        technology=CellTechnology.PERC,
        area=244.0,
        efficiency=22.5,
        voc=0.68,
        isc=8.5,
        vmp=0.58,
        imp=8.2,
        pmax=5.2,
        temperature_coefficient_pmax=-0.39,
        temperature_coefficient_voc=-0.0029,
        temperature_coefficient_isc=0.0005,
    )


@pytest.fixture
def sample_module_config() -> ModuleConfiguration:
    """Create sample module configuration for testing."""
    return ModuleConfiguration(
        num_cells_series=60,
        num_strings_parallel=1,
        bypass_diodes=3,
    )


@pytest.fixture
def sample_reference_device() -> ReferenceDeviceData:
    """Create sample reference device data for testing."""
    return ReferenceDeviceData(
        device_id="TEST-REF-001",
        calibration_date=datetime(2025, 1, 1),
        calibration_lab="Test Lab",
        calibration_certificate="CAL-2025-001",
        short_circuit_current=8.5,
        responsivity=0.0085,
        temperature_coefficient=0.0005,
        uncertainty_isc=1.5,
        uncertainty_temperature=0.2,
        next_calibration_due=datetime(2026, 1, 1),
    )


@pytest.fixture
def sample_flash_simulator() -> FlashSimulatorData:
    """Create sample flash simulator data for testing."""
    return FlashSimulatorData(
        simulator_type=FlashSimulatorType.LED,
        spatial_uniformity=98.5,
        temporal_stability=99.2,
    )


@pytest.fixture
def sample_test_config(
    sample_cell_properties,
    sample_module_config,
    sample_reference_device,
    sample_flash_simulator
) -> CTMTestConfig:
    """Create sample CTM test configuration."""
    return CTMTestConfig(
        test_id="TEST-001",
        laboratory="Test Laboratory",
        operator="Test Operator",
        cell_properties=sample_cell_properties,
        module_config=sample_module_config,
        reference_device=sample_reference_device,
        flash_simulator=sample_flash_simulator,
    )


@pytest.fixture
def sample_iv_curve() -> IVCurveData:
    """Create sample IV curve data."""
    voltage = np.linspace(0, 0.68, 20).tolist()
    current = (8.5 * (1 - (np.array(voltage) / 0.68) ** 3)).tolist()

    return IVCurveData(
        voltage=voltage,
        current=current,
        temperature=25.0,
        irradiance=1000.0,
    )


class TestIEC63202CTMTester:
    """Test suite for IEC63202CTMTester class."""

    def test_initialization(self, sample_test_config):
        """Test tester initialization."""
        tester = IEC63202CTMTester(config=sample_test_config)

        assert tester.config == sample_test_config
        assert tester.test_result is None

    def test_reference_cell_measurement(self, sample_test_config, sample_iv_curve):
        """Test reference cell measurement under STC."""
        tester = IEC63202CTMTester(config=sample_test_config)

        result_iv = tester.reference_cell_measurement(
            voltage=sample_iv_curve.voltage,
            current=sample_iv_curve.current,
            temperature=25.0,
            irradiance=1000.0,
        )

        assert result_iv is not None
        assert result_iv.pmax > 0
        assert result_iv.voc > 0
        assert result_iv.isc > 0

    def test_module_flash_test(self, sample_test_config):
        """Test module flash testing."""
        tester = IEC63202CTMTester(config=sample_test_config)

        # Generate module IV curve
        voltage = np.linspace(0, 40.8, 30).tolist()
        current = (8.5 * (1 - (np.array(voltage) / 40.8) ** 3)).tolist()

        result_iv = tester.module_flash_test(
            voltage=voltage,
            current=current,
            temperature=25.0,
            irradiance=1000.0,
        )

        assert result_iv is not None
        assert result_iv.pmax > 0

    def test_ctm_power_ratio_test(self, sample_test_config):
        """Test CTM power ratio calculation."""
        tester = IEC63202CTMTester(config=sample_test_config)

        # Create cell measurements
        cell_measurements = []
        for _ in range(5):
            voltage = np.linspace(0, 0.68, 20).tolist()
            current = (8.5 * (1 - (np.array(voltage) / 0.68) ** 3)).tolist()
            cell_measurements.append(IVCurveData(
                voltage=voltage,
                current=current,
                temperature=25.0,
                irradiance=1000.0,
            ))

        # Create module measurements
        module_measurements = []
        for _ in range(3):
            voltage = np.linspace(0, 40.8, 30).tolist()
            current = (8.5 * (1 - (np.array(voltage) / 40.8) ** 3)).tolist()
            module_measurements.append(IVCurveData(
                voltage=voltage,
                current=current,
                temperature=25.0,
                irradiance=1000.0,
            ))

        result = tester.ctm_power_ratio_test(
            cell_measurements=cell_measurements,
            module_measurements=module_measurements,
        )

        assert result is not None
        assert result.ctm_ratio > 0
        assert 80 <= result.ctm_ratio <= 110
        assert result.ctm_ratio_uncertainty >= 0

    def test_validate_ctm_ratio_pass(self, sample_test_config):
        """Test CTM ratio validation - passing case."""
        tester = IEC63202CTMTester(config=sample_test_config)

        is_valid = tester.validate_ctm_ratio(ctm_ratio=97.5, uncertainty=1.8)

        assert is_valid is True

    def test_validate_ctm_ratio_fail(self, sample_test_config):
        """Test CTM ratio validation - failing case."""
        tester = IEC63202CTMTester(config=sample_test_config)

        is_valid = tester.validate_ctm_ratio(ctm_ratio=92.0, uncertainty=1.5)

        assert is_valid is False

    def test_generate_ctm_certificate(self, sample_test_config):
        """Test CTM certificate generation."""
        tester = IEC63202CTMTester(config=sample_test_config)

        # First run a test
        cell_measurements = [
            IVCurveData(
                voltage=np.linspace(0, 0.68, 20).tolist(),
                current=(8.5 * (1 - (np.linspace(0, 0.68, 20) / 0.68) ** 3)).tolist(),
                temperature=25.0,
                irradiance=1000.0,
            )
            for _ in range(5)
        ]

        module_measurements = [
            IVCurveData(
                voltage=np.linspace(0, 40.8, 30).tolist(),
                current=(8.5 * (1 - (np.linspace(0, 40.8, 30) / 40.8) ** 3)).tolist(),
                temperature=25.0,
                irradiance=1000.0,
            )
            for _ in range(3)
        ]

        tester.ctm_power_ratio_test(
            cell_measurements=cell_measurements,
            module_measurements=module_measurements,
        )

        # Generate certificate
        certificate = tester.generate_ctm_certificate()

        assert certificate is not None
        assert certificate.certificate_number.startswith("IEC63202-")
        assert certificate.test_result is not None

    def test_insufficient_cell_measurements(self, sample_test_config):
        """Test error handling for insufficient cell measurements."""
        tester = IEC63202CTMTester(config=sample_test_config)

        cell_measurements = [
            IVCurveData(
                voltage=[0.0, 0.68],
                current=[8.5, 0.0],
                temperature=25.0,
                irradiance=1000.0,
            )
        ]  # Only 1 cell, need minimum 3

        module_measurements = [
            IVCurveData(
                voltage=[0.0, 40.8],
                current=[8.5, 0.0],
                temperature=25.0,
                irradiance=1000.0,
            )
        ]

        with pytest.raises(ValueError, match="Minimum 3 cell measurements required"):
            tester.ctm_power_ratio_test(
                cell_measurements=cell_measurements,
                module_measurements=module_measurements,
            )
