"""
Unit tests for TMY Data Manager.
"""

import pytest
from pathlib import Path

from pv_simulator.models.weather import TMYData, TMYFormat
from pv_simulator.services.tmy_manager import TMYDataManager


class TestTMYDataManager:
    """Tests for TMYDataManager."""

    @pytest.fixture
    def tmy_manager(self, temp_data_dir: Path) -> TMYDataManager:
        """Create TMY manager with temp directory."""
        return TMYDataManager(cache_dir=temp_data_dir)

    def test_manager_initialization(self, tmy_manager: TMYDataManager) -> None:
        """Test TMY manager initialization."""
        assert tmy_manager.cache_dir.exists()

    def test_load_csv_format(self, tmy_manager: TMYDataManager, sample_csv_tmy: Path) -> None:
        """Test loading CSV format TMY file."""
        tmy_data = tmy_manager.load_tmy_data(sample_csv_tmy, format_type=TMYFormat.CSV)

        assert isinstance(tmy_data, TMYData)
        assert len(tmy_data.hourly_data) > 0

    def test_validate_tmy_completeness(
        self, tmy_manager: TMYDataManager, sample_tmy_data: TMYData
    ) -> None:
        """Test TMY completeness validation."""
        is_valid, metrics = tmy_manager.validate_tmy_completeness(sample_tmy_data)

        assert is_valid is True
        assert metrics["completeness"] == 100.0
        assert metrics["total_points"] == 8760

    def test_interpolate_missing_data(
        self, tmy_manager: TMYDataManager, sample_tmy_data: TMYData
    ) -> None:
        """Test data interpolation."""
        # Set some values to None to test interpolation
        sample_tmy_data.hourly_data[100].temperature = None
        sample_tmy_data.hourly_data[101].temperature = None

        interpolated = tmy_manager.interpolate_missing_data(sample_tmy_data)

        # Values should be interpolated
        assert interpolated.hourly_data[100].temperature is not None
        assert interpolated.hourly_data[101].temperature is not None

    def test_detect_format(self, tmy_manager: TMYDataManager) -> None:
        """Test format auto-detection."""
        # Test EPW detection
        fmt = tmy_manager._detect_format(Path("test.epw"))
        assert fmt == TMYFormat.EPW

        # Test CSV detection
        fmt = tmy_manager._detect_format(Path("test.csv"))
        assert fmt == TMYFormat.CSV
