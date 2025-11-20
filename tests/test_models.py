"""Unit tests for Pydantic data models."""

import pytest
from datetime import datetime
from src.models.eya_models import (
    ProjectInfo,
    SystemConfiguration,
    ModuleType,
    MountingType,
    WeatherData,
    EnergyOutput,
    LossBreakdown,
    PerformanceMetrics,
    FinancialMetrics,
    ProbabilisticAnalysis,
)


class TestProjectInfo:
    """Tests for ProjectInfo model."""

    def test_valid_project_info(self):
        """Test creating a valid project info object."""
        project = ProjectInfo(
            project_name="Test Project",
            location="Test Location",
            latitude=37.7749,
            longitude=-122.4194,
            commissioning_date=datetime(2024, 1, 1),
        )
        assert project.project_name == "Test Project"
        assert project.latitude == 37.7749
        assert project.project_lifetime == 25  # default value

    def test_invalid_latitude(self):
        """Test that invalid latitude raises error."""
        with pytest.raises(Exception):
            ProjectInfo(
                project_name="Test",
                location="Test",
                latitude=100.0,  # Invalid: > 90
                longitude=0.0,
                commissioning_date=datetime.now(),
            )


class TestSystemConfiguration:
    """Tests for SystemConfiguration model."""

    def test_valid_system_config(self):
        """Test creating a valid system configuration."""
        config = SystemConfiguration(
            capacity_dc=1000.0,
            capacity_ac=850.0,
            module_type=ModuleType.MONO_SI,
            module_efficiency=0.20,
            module_count=5000,
            tilt_angle=30.0,
            azimuth_angle=180.0,
        )
        assert config.capacity_dc == 1000.0
        assert config.module_type == ModuleType.MONO_SI
        assert config.dc_ac_ratio == 1.2  # default value

    def test_ac_capacity_validation(self):
        """Test that AC capacity cannot exceed DC capacity."""
        with pytest.raises(Exception):
            SystemConfiguration(
                capacity_dc=850.0,
                capacity_ac=1000.0,  # Invalid: AC > DC
                module_type=ModuleType.MONO_SI,
                module_efficiency=0.20,
                module_count=5000,
                tilt_angle=30.0,
                azimuth_angle=180.0,
            )


class TestLossBreakdown:
    """Tests for LossBreakdown model."""

    def test_loss_calculation(self):
        """Test total loss calculation."""
        losses = LossBreakdown(
            soiling_loss=2.0,
            shading_loss=3.0,
            temperature_loss=5.0,
        )
        total = losses.calculate_total_loss()
        assert total > 0
        assert total < 100  # Total loss should be less than 100%


class TestFinancialMetrics:
    """Tests for FinancialMetrics model."""

    def test_valid_financial_metrics(self):
        """Test creating valid financial metrics."""
        financial = FinancialMetrics(
            capex=1000000.0,
            opex_annual=15000.0,
            energy_price=0.12,
        )
        assert financial.capex == 1000000.0
        assert financial.degradation_rate == 0.005  # default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
