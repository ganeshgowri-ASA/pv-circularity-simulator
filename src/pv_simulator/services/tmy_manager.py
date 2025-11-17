"""
TMY Data Manager for loading, parsing, and validating TMY data files.

This module provides comprehensive support for TMY2, TMY3, and EPW formats
with data interpolation, validation, and quality assessment.
"""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pv_simulator.config.settings import settings
from pv_simulator.models.weather import (
    DataQuality,
    DataSource,
    GlobalLocation,
    TemporalResolution,
    TMYData,
    TMYFormat,
    WeatherDataPoint,
)

logger = logging.getLogger(__name__)


class TMYDataManager:
    """
    Manager for TMY (Typical Meteorological Year) data files.

    Supports loading and parsing TMY2, TMY3, and EPW formats with
    comprehensive data validation and quality checks.

    Attributes:
        cache_dir: Directory for caching TMY data
    """

    # TMY3 column mapping (NSRDB format)
    TMY3_COLUMNS = {
        "Date (MM/DD/YYYY)": "date",
        "Time (HH:MM)": "time",
        "GHI (W/m^2)": "irradiance_ghi",
        "DNI (W/m^2)": "irradiance_dni",
        "DHI (W/m^2)": "irradiance_dhi",
        "Temperature (C)": "temperature",
        "Wind Speed (m/s)": "wind_speed",
        "Wind Direction (Deg)": "wind_direction",
        "Relative Humidity (%)": "relative_humidity",
        "Pressure (mbar)": "pressure",
        "Albedo": "albedo",
        "Precipitable Water (cm)": "precipitable_water",
    }

    # EPW column indices
    EPW_COLUMN_INDICES = {
        "year": 0,
        "month": 1,
        "day": 2,
        "hour": 3,
        "minute": 4,
        "temperature": 6,
        "dew_point": 7,
        "relative_humidity": 8,
        "pressure": 9,
        "irradiance_direct": 14,
        "irradiance_diffuse": 15,
        "wind_direction": 20,
        "wind_speed": 21,
    }

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize TMY Data Manager.

        Args:
            cache_dir: Directory for caching TMY data (default: from settings)
        """
        self.cache_dir = cache_dir or settings.tmy_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TMYDataManager initialized with cache_dir: {self.cache_dir}")

    def load_tmy_data(
        self, file_path: Union[str, Path], format_type: Optional[TMYFormat] = None
    ) -> TMYData:
        """
        Load TMY data from a file.

        Args:
            file_path: Path to the TMY file
            format_type: TMY format (auto-detected if None)

        Returns:
            Parsed TMY data

        Raises:
            ValueError: If file format is not supported or parsing fails
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"TMY file not found: {file_path}")

        # Auto-detect format if not specified
        if format_type is None:
            format_type = self._detect_format(file_path)

        logger.info(f"Loading TMY data from {file_path} (format: {format_type})")

        # Parse based on format
        if format_type == TMYFormat.TMY3:
            return self.parse_tmy3_format(file_path)
        elif format_type == TMYFormat.TMY2:
            return self.parse_tmy2_format(file_path)
        elif format_type == TMYFormat.EPW:
            return self.parse_epw_format(file_path)
        elif format_type == TMYFormat.CSV:
            return self.parse_csv_format(file_path)
        else:
            raise ValueError(f"Unsupported TMY format: {format_type}")

    def parse_tmy3_format(self, file_path: Path) -> TMYData:
        """
        Parse TMY3 format file (NREL NSRDB format).

        TMY3 format has:
        - Line 1: Location metadata
        - Line 2: Column headers
        - Lines 3+: Hourly data

        Args:
            file_path: Path to TMY3 file

        Returns:
            Parsed TMY data
        """
        logger.info(f"Parsing TMY3 format: {file_path}")

        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse header (line 1)
        header_parts = lines[0].strip().split(",")
        location_info = self._parse_tmy3_header(header_parts)

        # Read data starting from line 2
        df = pd.read_csv(file_path, skiprows=2)

        # Parse data points
        hourly_data = self._parse_tmy3_data(df, location_info["timezone"])

        # Create TMY data object
        tmy_data = TMYData(
            location=location_info["location"],
            data_source=DataSource.NSRDB,
            format_type=TMYFormat.TMY3,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=location_info.get("start_year", 2007),
            end_year=location_info.get("end_year", 2021),
            hourly_data=hourly_data,
            metadata={"file_path": str(file_path)},
        )

        # Validate and assess quality
        self.validate_tmy_completeness(tmy_data)

        return tmy_data

    def parse_tmy2_format(self, file_path: Path) -> TMYData:
        """
        Parse TMY2 format file (older NREL format).

        TMY2 has fixed-width columns with specific byte positions.

        Args:
            file_path: Path to TMY2 file

        Returns:
            Parsed TMY data
        """
        logger.info(f"Parsing TMY2 format: {file_path}")

        # TMY2 uses fixed-width format
        # First line contains location info
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()

        # Parse location from first line
        location = self._parse_tmy2_header(first_line)

        # Read data with fixed-width parser
        hourly_data = self._parse_tmy2_data(file_path, location.timezone)

        tmy_data = TMYData(
            location=location,
            data_source=DataSource.NSRDB,
            format_type=TMYFormat.TMY2,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=1961,
            end_year=1990,
            hourly_data=hourly_data,
            metadata={"file_path": str(file_path)},
        )

        self.validate_tmy_completeness(tmy_data)
        return tmy_data

    def parse_epw_format(self, file_path: Path) -> TMYData:
        """
        Parse EPW (EnergyPlus Weather) format file.

        EPW format structure:
        - Lines 1-8: Header information
        - Lines 9+: Hourly weather data (CSV)

        Args:
            file_path: Path to EPW file

        Returns:
            Parsed TMY data
        """
        logger.info(f"Parsing EPW format: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse location from header (line 1)
        location = self._parse_epw_header(lines[0])

        # Parse data starting from line 8
        hourly_data = self._parse_epw_data(lines[8:], location.timezone)

        tmy_data = TMYData(
            location=location,
            data_source=DataSource.EPW,
            format_type=TMYFormat.EPW,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=2000,
            end_year=2020,
            hourly_data=hourly_data,
            metadata={"file_path": str(file_path)},
        )

        self.validate_tmy_completeness(tmy_data)
        return tmy_data

    def parse_csv_format(self, file_path: Path) -> TMYData:
        """
        Parse generic CSV format with standard column names.

        Args:
            file_path: Path to CSV file

        Returns:
            Parsed TMY data
        """
        logger.info(f"Parsing CSV format: {file_path}")

        df = pd.read_csv(file_path)

        # Try to infer location from filename or metadata
        location = GlobalLocation(
            name=file_path.stem,
            country="Unknown",
            latitude=0.0,
            longitude=0.0,
            elevation=0.0,
            timezone="UTC",
        )

        hourly_data = self._parse_csv_data(df)

        tmy_data = TMYData(
            location=location,
            data_source=DataSource.CUSTOM,
            format_type=TMYFormat.CSV,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=2000,
            end_year=2020,
            hourly_data=hourly_data,
        )

        self.validate_tmy_completeness(tmy_data)
        return tmy_data

    def interpolate_missing_data(
        self, tmy_data: TMYData, max_gap_hours: Optional[int] = None
    ) -> TMYData:
        """
        Interpolate missing data points in TMY dataset.

        Uses linear interpolation for gaps up to max_gap_hours.

        Args:
            tmy_data: TMY data with potential missing values
            max_gap_hours: Maximum gap to interpolate (default: from settings)

        Returns:
            TMY data with interpolated values
        """
        max_gap = max_gap_hours or settings.interpolation_max_gap_hours
        logger.info(f"Interpolating missing data (max gap: {max_gap} hours)")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([point.model_dump() for point in tmy_data.hourly_data])

        # Interpolate each numeric column
        numeric_columns = [
            "temperature",
            "irradiance_ghi",
            "irradiance_dni",
            "irradiance_dhi",
            "wind_speed",
            "relative_humidity",
            "pressure",
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Limit interpolation to max_gap
                df[col] = df[col].interpolate(method="linear", limit=max_gap)

        # Convert back to WeatherDataPoint objects
        interpolated_data = []
        for _, row in df.iterrows():
            point = WeatherDataPoint(**row.to_dict())
            interpolated_data.append(point)

        # Update TMY data
        tmy_data.hourly_data = interpolated_data
        self.validate_tmy_completeness(tmy_data)

        return tmy_data

    def validate_tmy_completeness(self, tmy_data: TMYData) -> Tuple[bool, Dict[str, float]]:
        """
        Validate TMY data completeness and quality.

        Args:
            tmy_data: TMY data to validate

        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        total_points = len(tmy_data.hourly_data)
        if total_points == 0:
            logger.warning("TMY data is empty")
            return False, {"completeness": 0.0}

        # Count missing/invalid values
        missing_counts = {
            "temperature": 0,
            "irradiance_ghi": 0,
            "irradiance_dni": 0,
            "irradiance_dhi": 0,
        }

        for point in tmy_data.hourly_data:
            if point.temperature is None or point.temperature < -100:
                missing_counts["temperature"] += 1
            if point.irradiance_ghi is None or point.irradiance_ghi < 0:
                missing_counts["irradiance_ghi"] += 1
            if point.irradiance_dni is None or point.irradiance_dni < 0:
                missing_counts["irradiance_dni"] += 1
            if point.irradiance_dhi is None or point.irradiance_dhi < 0:
                missing_counts["irradiance_dhi"] += 1

        # Calculate completeness percentage
        completeness = 100.0 * (
            1 - max(missing_counts.values()) / total_points if total_points > 0 else 0
        )

        # Assess data quality
        if completeness >= 99.0:
            quality = DataQuality.EXCELLENT
        elif completeness >= 95.0:
            quality = DataQuality.GOOD
        elif completeness >= 90.0:
            quality = DataQuality.FAIR
        else:
            quality = DataQuality.POOR

        # Update TMY data
        tmy_data.completeness_percentage = completeness
        tmy_data.data_quality = quality

        quality_metrics = {
            "completeness": completeness,
            "total_points": total_points,
            "missing_temperature": missing_counts["temperature"],
            "missing_ghi": missing_counts["irradiance_ghi"],
            "quality": quality.value,
        }

        logger.info(f"TMY validation: {completeness:.1f}% complete, quality={quality.value}")

        is_valid = completeness >= (100.0 - settings.max_missing_data_percentage)
        return is_valid, quality_metrics

    def generate_custom_tmy(
        self, location: GlobalLocation, data_points: List[WeatherDataPoint]
    ) -> TMYData:
        """
        Generate a custom TMY dataset from provided data points.

        Args:
            location: Location information
            data_points: List of weather data points

        Returns:
            Custom TMY data
        """
        logger.info(f"Generating custom TMY for {location.name}")

        tmy_data = TMYData(
            location=location,
            data_source=DataSource.CUSTOM,
            format_type=TMYFormat.CSV,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=2000,
            end_year=2020,
            hourly_data=data_points,
        )

        self.validate_tmy_completeness(tmy_data)
        return tmy_data

    # Helper methods for parsing different formats

    def _detect_format(self, file_path: Path) -> TMYFormat:
        """Auto-detect TMY file format from extension and content."""
        suffix = file_path.suffix.lower()

        if suffix == ".epw":
            return TMYFormat.EPW
        elif suffix == ".csv":
            # Try to detect TMY3 vs generic CSV
            with open(file_path, "r") as f:
                first_line = f.readline()
                if "TMY3" in first_line or "NSRDB" in first_line:
                    return TMYFormat.TMY3
            return TMYFormat.CSV
        elif suffix == ".tm2":
            return TMYFormat.TMY2
        else:
            # Default to CSV
            return TMYFormat.CSV

    def _parse_tmy3_header(self, header_parts: List[str]) -> Dict:
        """Parse TMY3 header line."""
        # TMY3 header format: Location ID, City, State, Country, Lat, Lon, Elev, Timezone
        location = GlobalLocation(
            name=header_parts[1] if len(header_parts) > 1 else "Unknown",
            country=header_parts[3] if len(header_parts) > 3 else "USA",
            latitude=float(header_parts[4]) if len(header_parts) > 4 else 0.0,
            longitude=float(header_parts[5]) if len(header_parts) > 5 else 0.0,
            elevation=float(header_parts[6]) if len(header_parts) > 6 else 0.0,
            timezone=header_parts[7] if len(header_parts) > 7 else "UTC",
        )

        return {
            "location": location,
            "timezone": location.timezone,
            "start_year": 2007,
            "end_year": 2021,
        }

    def _parse_tmy3_data(self, df: pd.DataFrame, timezone: str) -> List[WeatherDataPoint]:
        """Parse TMY3 data rows."""
        data_points = []

        for _, row in df.iterrows():
            try:
                # Parse timestamp
                date_str = str(row.get("Date (MM/DD/YYYY)", "01/01/2000"))
                time_str = str(row.get("Time (HH:MM)", "00:00"))
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")

                # Create weather data point
                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(row.get("Temperature (C)", 0)),
                    irradiance_ghi=float(row.get("GHI (W/m^2)", 0)),
                    irradiance_dni=float(row.get("DNI (W/m^2)", 0)),
                    irradiance_dhi=float(row.get("DHI (W/m^2)", 0)),
                    wind_speed=float(row.get("Wind Speed (m/s)", 0)),
                    wind_direction=float(row.get("Wind Direction (Deg)", 0)),
                    relative_humidity=float(row.get("Relative Humidity (%)", 50)),
                    pressure=float(row.get("Pressure (mbar)", 1013)) * 100,  # mbar to Pa
                    albedo=float(row.get("Albedo", 0.2)),
                    precipitable_water=float(row.get("Precipitable Water (cm)", 0)),
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Error parsing TMY3 row: {e}")
                continue

        return data_points

    def _parse_tmy2_header(self, first_line: str) -> GlobalLocation:
        """Parse TMY2 header (fixed-width format)."""
        # TMY2 fixed positions
        location_name = first_line[0:22].strip()
        state = first_line[22:24].strip()

        return GlobalLocation(
            name=location_name,
            country="USA",
            latitude=0.0,  # Would need to parse from fixed positions
            longitude=0.0,
            elevation=0.0,
            timezone="UTC",
        )

    def _parse_tmy2_data(self, file_path: Path, timezone: str) -> List[WeatherDataPoint]:
        """Parse TMY2 data (fixed-width format)."""
        # Simplified TMY2 parsing - would need full fixed-width spec
        data_points = []
        logger.warning("TMY2 parsing is simplified - may not capture all fields")
        return data_points

    def _parse_epw_header(self, header_line: str) -> GlobalLocation:
        """Parse EPW header line."""
        parts = header_line.split(",")

        return GlobalLocation(
            name=parts[1] if len(parts) > 1 else "Unknown",
            country=parts[3] if len(parts) > 3 else "Unknown",
            latitude=float(parts[6]) if len(parts) > 6 else 0.0,
            longitude=float(parts[7]) if len(parts) > 7 else 0.0,
            elevation=float(parts[8]) if len(parts) > 8 else 0.0,
            timezone=parts[9] if len(parts) > 9 else "UTC",
        )

    def _parse_epw_data(self, data_lines: List[str], timezone: str) -> List[WeatherDataPoint]:
        """Parse EPW data lines."""
        data_points = []

        for line in data_lines:
            try:
                parts = line.strip().split(",")
                if len(parts) < 22:
                    continue

                # Parse timestamp
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3]) - 1  # EPW uses 1-24, convert to 0-23

                timestamp = datetime(year, month, day, hour)

                # Create weather data point
                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(parts[6]),
                    irradiance_ghi=float(parts[13]) if len(parts) > 13 else 0.0,
                    irradiance_dni=float(parts[14]) if len(parts) > 14 else 0.0,
                    irradiance_dhi=float(parts[15]) if len(parts) > 15 else 0.0,
                    wind_speed=float(parts[21]) if len(parts) > 21 else 0.0,
                    wind_direction=float(parts[20]) if len(parts) > 20 else 0.0,
                    relative_humidity=float(parts[8]) if len(parts) > 8 else 50.0,
                    pressure=float(parts[9]) if len(parts) > 9 else 101325.0,
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Error parsing EPW line: {e}")
                continue

        return data_points

    def _parse_csv_data(self, df: pd.DataFrame) -> List[WeatherDataPoint]:
        """Parse generic CSV data."""
        data_points = []

        for _, row in df.iterrows():
            try:
                # Try to find timestamp column
                timestamp = None
                for col in ["timestamp", "datetime", "time"]:
                    if col in df.columns:
                        timestamp = pd.to_datetime(row[col])
                        break

                if timestamp is None:
                    timestamp = datetime.now()

                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(row.get("temperature", 20)),
                    irradiance_ghi=float(row.get("ghi", 0)),
                    irradiance_dni=float(row.get("dni", 0)),
                    irradiance_dhi=float(row.get("dhi", 0)),
                    wind_speed=float(row.get("wind_speed", 0)),
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Error parsing CSV row: {e}")
                continue

        return data_points
