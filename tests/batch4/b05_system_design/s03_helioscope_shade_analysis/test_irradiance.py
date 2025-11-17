"""
Unit tests for IrradianceOnSurface.
"""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.irradiance import (
    IrradianceOnSurface,
)
from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.models import (
    ArrayGeometry,
    IrradianceComponents,
    Location,
    SunPosition,
    TranspositionModel,
    AOIModel,
)


class TestIrradianceOnSurface(unittest.TestCase):
    """Test cases for IrradianceOnSurface."""

    def setUp(self):
        """Set up test fixtures."""
        self.location = Location(latitude=37.7749, longitude=-122.4194)
        self.array_geometry = ArrayGeometry(
            tilt=20.0,
            azimuth=180.0,
            gcr=0.4,
            module_width=1.0,
            module_height=2.0,
            modules_per_string=20,
            row_spacing=5.0
        )
        self.irradiance_calc = IrradianceOnSurface(
            self.location,
            self.array_geometry
        )

    def test_initialization(self):
        """Test initialization."""
        self.assertIsNotNone(self.irradiance_calc)
        self.assertEqual(self.irradiance_calc.transposition_model, TranspositionModel.PEREZ)

    def test_poa_irradiance_calculation(self):
        """Test POA irradiance calculation."""
        sun_position = SunPosition(
            timestamp=datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
            azimuth=180.0,
            elevation=70.0,
            zenith=20.0,
            declination=23.45,
            hour_angle=0.0,
            equation_of_time=0.0
        )

        irradiance = IrradianceComponents(
            timestamp=datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
            ghi=1000.0,
            dni=900.0,
            dhi=100.0
        )

        result = self.irradiance_calc.poa_irradiance(irradiance, sun_position)

        self.assertGreater(result.poa_global, 0)
        self.assertGreater(result.poa_direct, 0)
        self.assertGreater(result.poa_diffuse, 0)

    def test_transposition_models(self):
        """Test different transposition models."""
        models = [
            TranspositionModel.ISOTROPIC,
            TranspositionModel.PEREZ,
            TranspositionModel.HAY_DAVIES
        ]

        sun_position = SunPosition(
            timestamp=datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
            azimuth=180.0,
            elevation=70.0,
            zenith=20.0,
            declination=23.45,
            hour_angle=0.0,
            equation_of_time=0.0
        )

        for model in models:
            poa_direct, poa_diffuse, poa_ground = self.irradiance_calc.transposition_model(
                ghi=1000.0,
                dni=900.0,
                dhi=100.0,
                sun_position=sun_position,
                surface_tilt=20.0,
                surface_azimuth=180.0,
                model=model
            )

            self.assertGreaterEqual(poa_direct, 0)
            self.assertGreaterEqual(poa_diffuse, 0)
            self.assertGreaterEqual(poa_ground, 0)

    def test_aoi_correction(self):
        """Test AOI correction models."""
        aoi_values = [0, 30, 60, 80]

        for aoi in aoi_values:
            modifier = self.irradiance_calc.aoi_correction(aoi)
            self.assertGreaterEqual(modifier, 0)
            self.assertLessEqual(modifier, 1)

    def test_soiling_model(self):
        """Test soiling loss calculation."""
        soiling_factor = self.irradiance_calc.soiling_model(
            base_soiling_rate=0.001,
            days_since_cleaning=30
        )

        self.assertGreater(soiling_factor, 0)
        self.assertLessEqual(soiling_factor, 1)

    def test_spectral_correction(self):
        """Test spectral correction."""
        air_mass = 1.5
        correction = self.irradiance_calc.spectral_correction(air_mass)

        self.assertGreater(correction, 0.9)
        self.assertLess(correction, 1.1)


if __name__ == '__main__':
    unittest.main()
