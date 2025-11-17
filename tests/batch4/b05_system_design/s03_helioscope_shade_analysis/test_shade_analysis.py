"""
Unit tests for ShadeAnalysisEngine.
"""

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.shade_analysis import (
    ShadeAnalysisEngine,
)
from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.models import (
    ArrayGeometry,
    Location,
    ShadeAnalysisConfig,
    SiteModel,
    SunPosition,
)


class TestShadeAnalysisEngine(unittest.TestCase):
    """Test cases for ShadeAnalysisEngine."""

    def setUp(self):
        """Set up test fixtures."""
        location = Location(latitude=37.7749, longitude=-122.4194)
        self.site_model = SiteModel(location=location, albedo=0.2)

        self.array_geometry = ArrayGeometry(
            tilt=20.0,
            azimuth=180.0,
            gcr=0.4,
            module_width=1.0,
            module_height=2.0,
            modules_per_string=20,
            row_spacing=5.0
        )

        self.config = ShadeAnalysisConfig(
            start_date=datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")),
            end_date=datetime(2024, 12, 31, tzinfo=ZoneInfo("UTC"))
        )

        self.engine = ShadeAnalysisEngine(
            self.site_model,
            self.array_geometry,
            self.config
        )

    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.sun_calculator)
        self.assertIsNotNone(self.engine.irradiance_calculator)

    def test_near_shading_analysis(self):
        """Test near shading calculation."""
        sun_position = SunPosition(
            timestamp=datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
            azimuth=180.0,
            elevation=70.0,
            zenith=20.0,
            declination=23.45,
            hour_angle=0.0,
            equation_of_time=0.0
        )

        shading_loss = self.engine.near_shading_analysis(sun_position, row_index=5, total_rows=10)

        self.assertGreaterEqual(shading_loss, 0)
        self.assertLessEqual(shading_loss, 1)

    def test_far_shading_analysis(self):
        """Test far shading calculation."""
        sun_position = SunPosition(
            timestamp=datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
            azimuth=180.0,
            elevation=70.0,
            zenith=20.0,
            declination=23.45,
            hour_angle=0.0,
            equation_of_time=0.0
        )

        shading_loss = self.engine.far_shading_analysis(sun_position)

        self.assertGreaterEqual(shading_loss, 0)
        self.assertLessEqual(shading_loss, 1)

    def test_backtracking_optimization(self):
        """Test backtracking algorithm."""
        sun_position = SunPosition(
            timestamp=datetime(2024, 6, 21, 8, 0, 0, tzinfo=ZoneInfo("UTC")),
            azimuth=90.0,
            elevation=30.0,
            zenith=60.0,
            declination=23.45,
            hour_angle=-60.0,
            equation_of_time=0.0
        )

        tracker_angle = self.engine.backtracking_optimization(sun_position)

        self.assertIsInstance(tracker_angle, float)
        self.assertLessEqual(abs(tracker_angle), self.array_geometry.tracker_max_angle)

    def test_irradiance_loss_calculation(self):
        """Test irradiance loss calculation."""
        timestamp = datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

        shading_loss, irradiance = self.engine.irradiance_loss_calculation(
            timestamp=timestamp,
            ghi=1000.0,
            dni=900.0,
            dhi=100.0,
            row_index=5,
            total_rows=10
        )

        self.assertGreaterEqual(shading_loss, 0)
        self.assertLessEqual(shading_loss, 1)
        self.assertIsNotNone(irradiance)


if __name__ == '__main__':
    unittest.main()
