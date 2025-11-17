"""Tests for POAIrradianceModel."""

import pandas as pd
import pytest
import numpy as np

from src.irradiance.calculator import IrradianceCalculator
from src.irradiance.poa_model import POAIrradianceModel
from src.irradiance.models import IrradianceComponents, LocationConfig, SurfaceConfig


@pytest.fixture
def location():
    """Test location."""
    return LocationConfig(
        latitude=39.7555, longitude=-105.2211, altitude=1829, timezone="America/Denver"
    )


@pytest.fixture
def surface():
    """Test surface."""
    return SurfaceConfig(tilt=30.0, azimuth=180.0, albedo=0.2)


@pytest.fixture
def test_times():
    """Test time range."""
    return pd.date_range(start="2024-06-21 06:00", end="2024-06-21 18:00", freq="h", tz="America/Denver")


@pytest.fixture
def irradiance_data(test_times):
    """Sample irradiance data."""
    ghi = pd.Series([0, 100, 400, 700, 900, 950, 900, 700, 400, 100, 0, 0, 0], index=test_times)
    dni = pd.Series([0, 300, 600, 800, 850, 900, 850, 800, 600, 300, 0, 0, 0], index=test_times)
    dhi = ghi - dni * 0.7  # Approximate DHI
    dhi = dhi.clip(lower=0)

    return IrradianceComponents(ghi=ghi, dni=dni, dhi=dhi)


class TestPOAIrradianceModel:
    """Test suite for POAIrradianceModel."""

    def test_initialization(self, location, surface):
        """Test model initialization."""
        poa_model = POAIrradianceModel(location, surface)
        assert poa_model.location.latitude == 39.7555
        assert poa_model.surface.tilt == 30.0

    def test_direct_beam(self, location, surface, test_times, irradiance_data):
        """Test direct beam calculation."""
        poa_model = POAIrradianceModel(location, surface)
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        poa_direct = poa_model.direct_beam(irradiance_data.dni, solar_pos)

        # Check output
        assert len(poa_direct) == len(irradiance_data.dni)
        assert all(poa_direct >= 0)

        # POA direct should be less than or equal to DNI
        assert all(poa_direct <= irradiance_data.dni)

    def test_sky_diffuse(self, location, surface, test_times, irradiance_data):
        """Test sky diffuse calculation."""
        poa_model = POAIrradianceModel(location, surface)
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        # Test different models
        for model in ["perez", "haydavies", "isotropic"]:
            sky_diff = poa_model.sky_diffuse(
                irradiance_data.dni,
                irradiance_data.dhi,
                solar_pos,
                model=model,
                times=test_times,
            )

            assert len(sky_diff) == len(irradiance_data.dhi)
            assert all(sky_diff >= 0)

    def test_ground_reflected(self, location, surface, irradiance_data):
        """Test ground reflected irradiance."""
        poa_model = POAIrradianceModel(location, surface)

        poa_ground = poa_model.ground_reflected(irradiance_data.ghi)

        # Check output
        assert len(poa_ground) == len(irradiance_data.ghi)
        assert all(poa_ground >= 0)

        # Ground reflected should be small compared to GHI
        assert all(poa_ground < irradiance_data.ghi * 0.5)

    def test_aoi_losses(self, location, surface, test_times):
        """Test AOI loss calculation."""
        poa_model = POAIrradianceModel(location, surface)
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        aoi_factor = poa_model.aoi_losses(solar_pos)

        # Check output
        assert len(aoi_factor) == len(test_times)

        # AOI factor should be between 0 and 1
        assert all((aoi_factor >= 0) & (aoi_factor <= 1))

    def test_calculate_poa_components(self, location, surface, test_times, irradiance_data):
        """Test complete POA component calculation."""
        poa_model = POAIrradianceModel(location, surface)
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        # Without loss factors
        components = poa_model.calculate_poa_components(
            irradiance_data,
            solar_pos,
            transposition_model="perez",
            include_spectral=False,
            include_aoi=False,
            times=test_times,
        )

        # Check components
        assert len(components.poa_global) == len(test_times)
        assert len(components.poa_direct) == len(test_times)
        assert len(components.poa_diffuse) == len(test_times)
        assert len(components.poa_ground) == len(test_times)

        # Check that global = direct + diffuse + ground (approximately)
        calculated_global = (
            components.poa_direct + components.poa_diffuse + components.poa_ground
        )
        np.testing.assert_allclose(components.poa_global, calculated_global, rtol=0.01)

    def test_calculate_effective_irradiance(self, location, surface, test_times, irradiance_data):
        """Test effective irradiance calculation with loss factors."""
        poa_model = POAIrradianceModel(location, surface)
        calc = IrradianceCalculator(location)
        solar_pos = calc.get_solar_position(test_times)

        # Calculate with all loss factors
        eff_irradiance = poa_model.calculate_effective_irradiance(
            irradiance_data, solar_pos, transposition_model="perez", times=test_times
        )

        # Check output
        assert len(eff_irradiance) == len(test_times)
        assert all(eff_irradiance >= 0)

        # Effective irradiance should be less than GHI due to losses
        # (though sometimes POA can exceed GHI due to reflection and tilt)
