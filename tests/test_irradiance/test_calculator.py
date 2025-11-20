"""Tests for IrradianceCalculator."""

import pandas as pd
import pytest
from datetime import datetime
import numpy as np

from src.irradiance.calculator import IrradianceCalculator
from src.irradiance.models import LocationConfig, SurfaceConfig


@pytest.fixture
def location():
    """Test location (Golden, CO)."""
    return LocationConfig(
        latitude=39.7555,
        longitude=-105.2211,
        altitude=1829,
        timezone="America/Denver",
        name="Golden, CO",
    )


@pytest.fixture
def surface():
    """Test surface configuration."""
    return SurfaceConfig(tilt=30.0, azimuth=180.0, albedo=0.2)


@pytest.fixture
def test_times():
    """Test time range."""
    return pd.date_range(
        start="2024-06-21 06:00",
        end="2024-06-21 18:00",
        freq="h",
        tz="America/Denver",
    )


class TestIrradianceCalculator:
    """Test suite for IrradianceCalculator."""

    def test_initialization(self, location):
        """Test calculator initialization."""
        calc = IrradianceCalculator(location)
        assert calc.location.latitude == 39.7555
        assert calc.pvlib_location.latitude == 39.7555

    def test_solar_position(self, location, test_times):
        """Test solar position calculation."""
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        # Check that we get reasonable values
        assert len(solar_pos.zenith) == len(test_times)
        assert solar_pos.zenith.min() >= 0
        assert solar_pos.zenith.max() <= 90  # Sun should be above horizon during test period
        assert all(solar_pos.elevation >= 0)  # All values should be positive (daytime)

    def test_ghi_dni_dhi_decomposition(self, location, test_times):
        """Test GHI decomposition into DNI and DHI."""
        calc = IrradianceCalculator(location)

        # Create synthetic GHI data
        ghi = pd.Series(
            [0, 100, 400, 700, 900, 950, 900, 700, 400, 100, 0, 0, 0],
            index=test_times,
        )

        # Test different decomposition models
        for model in ["dirint", "disc", "erbs"]:
            components = calc.ghi_dni_dhi_decomposition(ghi, times=test_times, model=model)

            # Check that components are returned
            assert len(components.ghi) == len(ghi)
            assert len(components.dni) == len(ghi)
            assert len(components.dhi) == len(ghi)

            # Check physical constraints
            assert all(components.dni >= 0)
            assert all(components.dhi >= 0)

            # GHI should approximately equal DNI*cos(zenith) + DHI
            solar_pos = calc.get_solar_position(test_times)
            reconstructed_ghi = components.dni * np.cos(
                np.radians(solar_pos.zenith)
            ) + components.dhi
            # Allow some tolerance due to decomposition models
            np.testing.assert_allclose(components.ghi, reconstructed_ghi, rtol=0.1)

    def test_isotropic_sky(self, location, surface, test_times):
        """Test isotropic sky model."""
        calc = IrradianceCalculator(location)

        dhi = pd.Series([100, 200, 300, 200, 100], index=test_times[:5])
        sky_diffuse = calc.isotropic_sky(surface, dhi)

        # Check that output length matches input
        assert len(sky_diffuse) == len(dhi)

        # Check that values are positive
        assert all(sky_diffuse >= 0)

        # For 30° tilt, isotropic model should give DHI * (1 + cos(30))/2
        expected_factor = (1 + np.cos(np.radians(30))) / 2
        expected = dhi * expected_factor
        np.testing.assert_allclose(sky_diffuse, expected, rtol=0.01)

    def test_perez_transposition(self, location, surface, test_times):
        """Test Perez transposition model."""
        calc = IrradianceCalculator(location)

        dni = pd.Series([600, 700, 800, 700, 600], index=test_times[:5])
        dhi = pd.Series([100, 150, 200, 150, 100], index=test_times[:5])

        solar_pos = calc.get_solar_position(test_times[:5])
        sky_diffuse = calc.perez_transposition(surface, solar_pos, dni, dhi, times=test_times[:5])

        # Check output
        assert len(sky_diffuse) == len(dni)
        assert all(sky_diffuse >= 0)

    def test_hay_davies_model(self, location, surface, test_times):
        """Test Hay-Davies model."""
        calc = IrradianceCalculator(location)

        dni = pd.Series([600, 700, 800, 700, 600], index=test_times[:5])
        dhi = pd.Series([100, 150, 200, 150, 100], index=test_times[:5])

        solar_pos = calc.get_solar_position(test_times[:5])
        sky_diffuse = calc.hay_davies_model(surface, solar_pos, dni, dhi, times=test_times[:5])

        # Check output
        assert len(sky_diffuse) == len(dni)
        assert all(sky_diffuse >= 0)

    def test_anisotropic_corrections(self, location, surface, test_times):
        """Test anisotropic correction calculations."""
        calc = IrradianceCalculator(location)

        dni = pd.Series([700, 800, 900, 800, 700], index=test_times[:5])
        dhi = pd.Series([150, 180, 200, 180, 150], index=test_times[:5])

        solar_pos = calc.get_solar_position(test_times[:5])
        corrections = calc.anisotropic_corrections(
            surface, solar_pos, dni, dhi, model="perez", times=test_times[:5]
        )

        # Check that all keys are present
        assert "isotropic" in corrections
        assert "anisotropic" in corrections
        assert "correction_factor" in corrections
        assert "absolute_correction" in corrections

        # Check lengths
        assert len(corrections["isotropic"]) == len(dni)
        assert len(corrections["anisotropic"]) == len(dni)

    def test_clearness_index(self, location, test_times):
        """Test clearness index calculation."""
        calc = IrradianceCalculator(location)

        ghi = pd.Series([0, 200, 500, 800, 900, 800, 500, 200, 0, 0, 0, 0, 0], index=test_times)

        kt = calc.calculate_clearness_index(ghi)

        # Check output
        assert len(kt) == len(ghi)

        # Clearness index should be between 0 and 1
        assert all((kt >= 0) & (kt <= 1))

        # During peak hours, kt should be reasonably high for clear sky
        assert kt.max() > 0.3  # At least some moderate clearness
