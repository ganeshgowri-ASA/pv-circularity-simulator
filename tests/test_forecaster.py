"""Unit tests for Energy Forecaster module."""

import pytest
from datetime import datetime
from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType
from src.modules.B05_energy_forecasting.forecaster import EnergyForecaster


class TestEnergyForecaster:
    """Tests for EnergyForecaster class."""

    @pytest.fixture
    def project_info(self):
        """Create test project info."""
        return ProjectInfo(
            project_name="Test Project",
            location="San Francisco, CA",
            latitude=37.7749,
            longitude=-122.4194,
            commissioning_date=datetime(2024, 1, 1),
        )

    @pytest.fixture
    def system_config(self):
        """Create test system configuration."""
        return SystemConfiguration(
            capacity_dc=1000.0,
            capacity_ac=850.0,
            module_type=ModuleType.MONO_SI,
            module_efficiency=0.20,
            module_count=5000,
            tilt_angle=30.0,
            azimuth_angle=180.0,
        )

    @pytest.fixture
    def forecaster(self, project_info, system_config):
        """Create forecaster instance."""
        return EnergyForecaster(project_info, system_config)

    def test_forecaster_initialization(self, forecaster):
        """Test forecaster initializes correctly."""
        assert forecaster is not None
        assert forecaster.project_info.project_name == "Test Project"
        assert forecaster.system_config.capacity_dc == 1000.0

    def test_synthetic_weather_generation(self, forecaster):
        """Test synthetic weather data generation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)

        weather_data = forecaster.generate_synthetic_weather_data(start_date, end_date)

        assert len(weather_data) > 0
        assert all(w.ghi >= 0 for w in weather_data)
        assert all(w.timestamp >= start_date for w in weather_data)
        assert all(w.timestamp <= end_date for w in weather_data)

    def test_energy_forecast(self, forecaster):
        """Test energy forecasting."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Generate weather data
        weather_data = forecaster.generate_synthetic_weather_data(start_date, end_date)
        forecaster.load_weather_data(weather_data)

        # Forecast energy
        energy_outputs = forecaster.forecast_energy(start_date, end_date)

        assert len(energy_outputs) > 0
        assert all(e.ac_energy >= 0 for e in energy_outputs)
        assert all(e.dc_energy >= e.ac_energy for e in energy_outputs)

    def test_annual_energy_calculation(self, forecaster):
        """Test annual energy calculation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 2, 1)

        weather_data = forecaster.generate_synthetic_weather_data(start_date, end_date)
        forecaster.load_weather_data(weather_data)
        energy_outputs = forecaster.forecast_energy(start_date, end_date)

        annual_energy = forecaster.calculate_annual_energy(energy_outputs)

        assert annual_energy > 0
        assert isinstance(annual_energy, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
