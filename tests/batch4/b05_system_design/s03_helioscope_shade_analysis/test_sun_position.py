"""
Unit tests for SunPositionCalculator.
"""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.sun_position import (
    SunPositionCalculator,
)
from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.models import Location


class TestSunPositionCalculator(unittest.TestCase):
    """Test cases for SunPositionCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.location = Location(
            latitude=37.7749,
            longitude=-122.4194,
            elevation=0.0,
            timezone="America/Los_Angeles"
        )
        self.calculator = SunPositionCalculator(self.location)

    def test_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.location.latitude, 37.7749)

    def test_solar_position_algorithm(self):
        """Test NREL SPA algorithm calculation."""
        timestamp = datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        sun_position = self.calculator.solar_position_algorithm(timestamp)

        self.assertIsNotNone(sun_position)
        self.assertGreater(sun_position.elevation, 0)
        self.assertGreaterEqual(sun_position.azimuth, 0)
        self.assertLessEqual(sun_position.azimuth, 360)

    def test_sun_azimuth_elevation(self):
        """Test azimuth and elevation calculation."""
        timestamp = datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        azimuth, elevation = self.calculator.sun_azimuth_elevation(timestamp)

        self.assertIsInstance(azimuth, float)
        self.assertIsInstance(elevation, float)
        self.assertGreater(elevation, 0)

    def test_sunrise_sunset_times(self):
        """Test sunrise and sunset calculation."""
        date = datetime(2024, 6, 21, tzinfo=ZoneInfo("America/Los_Angeles"))
        sunrise, sunset = self.calculator.sunrise_sunset_times(date)

        self.assertIsNotNone(sunrise)
        self.assertIsNotNone(sunset)
        self.assertLess(sunrise, sunset)

    def test_day_length(self):
        """Test day length calculation."""
        date = datetime(2024, 6, 21, tzinfo=ZoneInfo("America/Los_Angeles"))
        day_length = self.calculator.day_length(date)

        self.assertGreater(day_length, 12)  # Summer solstice, longer than 12 hours
        self.assertLess(day_length, 24)

    def test_solar_noon(self):
        """Test solar noon calculation."""
        date = datetime(2024, 6, 21, tzinfo=ZoneInfo("America/Los_Angeles"))
        solar_noon_time = self.calculator.solar_noon(date)

        self.assertIsNotNone(solar_noon_time)
        self.assertEqual(solar_noon_time.date(), date.date())

    def test_equation_of_time(self):
        """Test equation of time calculation."""
        timestamp = datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        eot = self.calculator.equation_of_time(timestamp)

        self.assertIsInstance(eot, float)
        self.assertGreater(abs(eot), 0)

    def test_sun_path_3d(self):
        """Test 3D sun path generation."""
        date = datetime(2024, 6, 21, tzinfo=ZoneInfo("America/Los_Angeles"))
        sun_path = self.calculator.sun_path_3d(date, time_step_minutes=60)

        self.assertIsNotNone(sun_path)
        self.assertGreater(len(sun_path), 0)
        self.assertEqual(len(sun_path), 24)


class TestSunPositionEdgeCases(unittest.TestCase):
    """Test edge cases for sun position calculations."""

    def test_polar_region(self):
        """Test calculations in polar regions."""
        location = Location(latitude=80.0, longitude=0.0)
        calculator = SunPositionCalculator(location)

        timestamp = datetime(2024, 6, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        sun_position = calculator.solar_position_algorithm(timestamp)

        self.assertIsNotNone(sun_position)

    def test_equator(self):
        """Test calculations at the equator."""
        location = Location(latitude=0.0, longitude=0.0)
        calculator = SunPositionCalculator(location)

        timestamp = datetime(2024, 3, 21, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        sun_position = calculator.solar_position_algorithm(timestamp)

        self.assertIsNotNone(sun_position)
        self.assertGreater(sun_position.elevation, 85)  # Near zenith at noon on equinox


if __name__ == '__main__':
    unittest.main()
