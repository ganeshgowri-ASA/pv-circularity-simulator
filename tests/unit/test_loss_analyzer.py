"""
Unit tests for CTM Power Loss Analyzer.

Tests the CTMPowerLossAnalyzer class including:
- Optical losses calculation
- Electrical losses calculation
- Thermal losses calculation
- Spatial non-uniformity analysis
- Spectral mismatch factor calculation
- Total CTM loss budget
"""

import pytest
from typing import Dict

from pv_circularity_simulator.core.iec63202.loss_analyzer import CTMPowerLossAnalyzer
from pv_circularity_simulator.core.iec63202.models import (
    CellProperties,
    ModuleConfiguration,
    FlashSimulatorData,
    CellTechnology,
    FlashSimulatorType,
)


@pytest.fixture
def loss_analyzer() -> CTMPowerLossAnalyzer:
    """Create CTM power loss analyzer instance."""
    return CTMPowerLossAnalyzer()


@pytest.fixture
def sample_cell_properties() -> CellProperties:
    """Create sample cell properties."""
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
    """Create sample module configuration."""
    return ModuleConfiguration(
        num_cells_series=60,
        num_strings_parallel=1,
        bypass_diodes=3,
    )


@pytest.fixture
def sample_flash_simulator() -> FlashSimulatorData:
    """Create sample flash simulator data."""
    return FlashSimulatorData(
        simulator_type=FlashSimulatorType.LED,
        spatial_uniformity=98.5,
        temporal_stability=99.2,
        spectral_distribution={
            400: 0.8,
            500: 1.5,
            600: 1.8,
            700: 1.4,
            800: 1.0,
            900: 0.7,
        }
    )


class TestCTMPowerLossAnalyzer:
    """Test suite for CTMPowerLossAnalyzer class."""

    def test_initialization(self, loss_analyzer):
        """Test analyzer initialization."""
        assert loss_analyzer is not None
        assert loss_analyzer.am15_spectrum is not None

    def test_optical_losses(self, loss_analyzer):
        """Test optical losses calculation."""
        losses = loss_analyzer.optical_losses(
            glass_transmission=0.96,
            encapsulant_absorption=0.015,
            grid_coverage_ratio=0.025,
            num_busbars=5,
        )

        assert "reflection" in losses
        assert "absorption" in losses
        assert "grid_shading" in losses
        assert "total" in losses

        # Losses should be positive
        assert losses["total"] > 0
        assert losses["total"] == sum([
            losses["reflection"],
            losses["absorption"],
            losses["grid_shading"]
        ])

    def test_electrical_losses(self, loss_analyzer, sample_cell_properties, sample_module_config):
        """Test electrical losses calculation."""
        losses = loss_analyzer.electrical_losses(
            cell_properties=sample_cell_properties,
            module_config=sample_module_config,
        )

        assert "series_resistance" in losses
        assert "mismatch" in losses
        assert "total" in losses

        # Losses should be positive
        assert losses["series_resistance"] >= 0
        assert losses["mismatch"] >= 0
        assert losses["total"] > 0

    def test_thermal_losses(self, loss_analyzer, sample_cell_properties):
        """Test thermal losses calculation."""
        losses = loss_analyzer.thermal_losses(
            cell_properties=sample_cell_properties,
            lamination_temp=150.0,
            cooldown_rate=2.0,
            assembly_time=15.0,
        )

        assert "thermal_stress" in losses
        assert "total" in losses

        # Thermal losses should be small but positive
        assert 0 <= losses["total"] <= 2.0

    def test_spatial_non_uniformity(self, loss_analyzer, sample_flash_simulator):
        """Test spatial non-uniformity calculation."""
        loss = loss_analyzer.spatial_non_uniformity(
            flash_simulator=sample_flash_simulator,
            module_area=1.6,
        )

        # Loss should be related to non-uniformity
        # 98.5% uniformity = 1.5% loss
        assert 1.0 <= loss <= 3.0

    def test_spectral_mismatch_factor(self, loss_analyzer, sample_flash_simulator):
        """Test spectral mismatch factor calculation."""
        factor = loss_analyzer.spectral_mismatch_factor(
            simulator_spectrum=sample_flash_simulator.spectral_distribution,
        )

        # Factor should be close to 1.0 (typically 0.95-1.05)
        assert 0.90 <= factor <= 1.10

    def test_spectral_mismatch_custom_response(self, loss_analyzer):
        """Test spectral mismatch with custom spectral response."""
        simulator_spectrum = {
            400: 0.8,
            500: 1.5,
            600: 1.8,
            700: 1.4,
            800: 1.0,
        }

        cell_spectral_response = {
            400: 0.5,
            500: 0.75,
            600: 0.88,
            700: 0.95,
            800: 0.98,
        }

        factor = loss_analyzer.spectral_mismatch_factor(
            simulator_spectrum=simulator_spectrum,
            cell_spectral_response=cell_spectral_response,
        )

        assert 0.90 <= factor <= 1.10

    def test_total_ctm_loss_budget(
        self,
        loss_analyzer,
        sample_cell_properties,
        sample_module_config,
        sample_flash_simulator
    ):
        """Test total CTM loss budget calculation."""
        loss_components = loss_analyzer.total_ctm_loss_budget(
            cell_properties=sample_cell_properties,
            module_config=sample_module_config,
            flash_simulator=sample_flash_simulator,
        )

        # Verify all loss components are present
        assert loss_components.optical_reflection > 0
        assert loss_components.optical_absorption > 0
        assert loss_components.optical_shading > 0
        assert loss_components.electrical_series_resistance >= 0
        assert loss_components.electrical_mismatch >= 0

        # Total loss should match sum of components
        expected_total = (
            loss_components.total_optical_loss +
            loss_components.total_electrical_loss +
            loss_components.thermal_assembly +
            loss_components.spatial_non_uniformity +
            loss_components.spectral_mismatch
        )

        assert abs(loss_components.total_loss - expected_total) < 0.01

    def test_visualize_loss_waterfall(self, loss_analyzer, sample_cell_properties, sample_module_config):
        """Test loss waterfall visualization data."""
        loss_components = loss_analyzer.total_ctm_loss_budget(
            cell_properties=sample_cell_properties,
            module_config=sample_module_config,
        )

        waterfall = loss_analyzer.visualize_loss_waterfall(
            loss_components=loss_components,
            initial_power=100.0
        )

        # Verify waterfall stages
        assert "Initial (Cell Power × N)" in waterfall
        assert "Final (Module Power)" in waterfall

        # Initial should be 100%
        assert waterfall["Initial (Cell Power × N)"] == 100.0

        # Final should be less than initial
        assert waterfall["Final (Module Power)"] < 100.0

        # Final should equal initial minus total loss
        expected_final = 100.0 - loss_components.total_loss
        assert abs(waterfall["Final (Module Power)"] - expected_final) < 0.01

    def test_default_spectral_response(self, loss_analyzer):
        """Test default crystalline silicon spectral response."""
        spectral_response = loss_analyzer._default_spectral_response()

        # Should have wavelength range from 300-1200 nm
        wavelengths = list(spectral_response.keys())
        assert min(wavelengths) == 300
        assert max(wavelengths) == 1200

        # Peak response should be around 800-900 nm for c-Si
        peak_wavelength = max(spectral_response, key=spectral_response.get)
        assert 750 <= peak_wavelength <= 950
