"""
Weather Data Validator for quality checks and preprocessing.

This module provides comprehensive data validation, outlier detection,
gap filling, unit conversions, and timestamp synchronization for weather data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from pv_simulator.config import Settings, get_settings
from pv_simulator.models.weather import (
    DataQualityMetrics,
    HistoricalWeather,
    WeatherDataPoint,
)

logger = logging.getLogger(__name__)


class WeatherDataValidator:
    """
    Validator for weather data quality and preprocessing.

    Provides data quality checks, outlier detection, gap filling,
    unit conversions, and timestamp synchronization.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Weather Data Validator.

        Args:
            settings: Application settings (loads from environment if not provided)
        """
        self.settings = settings or get_settings()
        logger.info("Initialized Weather Data Validator")

    def data_quality_checks(
        self, weather_data: list[WeatherDataPoint]
    ) -> DataQualityMetrics:
        """
        Perform comprehensive data quality checks on weather data.

        Args:
            weather_data: List of weather data points

        Returns:
            Data quality metrics
        """
        if not weather_data:
            raise ValueError("Weather data cannot be empty")

        total_points = len(weather_data)
        valid_points = 0
        missing_points = 0

        # Check field completeness
        temp_count = 0
        solar_count = 0
        wind_count = 0

        for point in weather_data:
            # Count valid points (at least one field present)
            has_data = any([
                point.temperature is not None,
                point.ghi is not None,
                point.wind_speed is not None,
            ])

            if has_data:
                valid_points += 1
            else:
                missing_points += 1

            # Field-specific completeness
            if point.temperature is not None:
                temp_count += 1
            if point.ghi is not None or point.dni is not None:
                solar_count += 1
            if point.wind_speed is not None:
                wind_count += 1

        # Calculate completeness ratios
        temp_completeness = temp_count / total_points if total_points > 0 else 0.0
        solar_completeness = solar_count / total_points if total_points > 0 else 0.0
        wind_completeness = wind_count / total_points if total_points > 0 else 0.0

        # Calculate overall quality score (weighted average)
        quality_score = (
            0.4 * temp_completeness +
            0.4 * solar_completeness +
            0.2 * wind_completeness
        )

        # Get time range
        timestamps = [p.timestamp for p in weather_data]
        start_time = min(timestamps)
        end_time = max(timestamps)

        metrics = DataQualityMetrics(
            total_points=total_points,
            valid_points=valid_points,
            missing_points=missing_points,
            temperature_completeness=temp_completeness,
            solar_completeness=solar_completeness,
            wind_completeness=wind_completeness,
            start_time=start_time,
            end_time=end_time,
            quality_score=quality_score,
        )

        logger.info(
            f"Data quality check: {valid_points}/{total_points} valid points, "
            f"quality score: {quality_score:.2f}"
        )

        return metrics

    def outlier_detection(
        self,
        weather_data: list[WeatherDataPoint],
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> tuple[list[WeatherDataPoint], int]:
        """
        Detect outliers in weather data using statistical methods.

        Args:
            weather_data: List of weather data points
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or Z-score)

        Returns:
            Tuple of (data with outliers marked, number of outliers detected)
        """
        if not weather_data:
            return weather_data, 0

        outliers_detected = 0

        # Extract numeric fields for analysis
        temp_values = [p.temperature for p in weather_data if p.temperature is not None]
        wind_values = [p.wind_speed for p in weather_data if p.wind_speed is not None]
        ghi_values = [p.ghi for p in weather_data if p.ghi is not None]

        # Detect outliers for each field
        temp_outliers = self._detect_outliers_in_series(temp_values, method, threshold)
        wind_outliers = self._detect_outliers_in_series(wind_values, method, threshold)
        ghi_outliers = self._detect_outliers_in_series(ghi_values, method, threshold)

        # Mark outliers in data points
        result_data = []
        temp_idx = 0
        wind_idx = 0
        ghi_idx = 0

        for point in weather_data:
            is_outlier = False

            if point.temperature is not None:
                if temp_outliers[temp_idx]:
                    is_outlier = True
                temp_idx += 1

            if point.wind_speed is not None:
                if wind_outliers[wind_idx]:
                    is_outlier = True
                wind_idx += 1

            if point.ghi is not None:
                if ghi_outliers[ghi_idx]:
                    is_outlier = True
                ghi_idx += 1

            if is_outlier:
                outliers_detected += 1
                # Mark in quality score
                if point.quality_score is None:
                    point.quality_score = 0.5
                else:
                    point.quality_score *= 0.5

            result_data.append(point)

        logger.info(f"Detected {outliers_detected} outliers using {method} method")

        return result_data, outliers_detected

    def _detect_outliers_in_series(
        self, values: list[float], method: str, threshold: float
    ) -> list[bool]:
        """
        Detect outliers in a numeric series.

        Args:
            values: List of numeric values
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            List of boolean flags indicating outliers
        """
        if not values:
            return []

        arr = np.array(values)

        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (arr < lower_bound) | (arr > upper_bound)

        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            z_scores = np.abs((arr - mean) / std) if std > 0 else np.zeros_like(arr)
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers.tolist()

    def gap_filling(
        self,
        weather_data: list[WeatherDataPoint],
        method: str = "linear",
        max_gap_hours: Optional[int] = None,
    ) -> tuple[list[WeatherDataPoint], int]:
        """
        Fill gaps in weather data using interpolation.

        Args:
            weather_data: List of weather data points
            method: Interpolation method ('linear', 'forward_fill', 'backward_fill')
            max_gap_hours: Maximum gap size to fill (uses setting if not specified)

        Returns:
            Tuple of (data with gaps filled, number of points interpolated)
        """
        if not weather_data:
            return weather_data, 0

        if max_gap_hours is None:
            max_gap_hours = self.settings.max_gap_hours

        if not self.settings.gap_filling_enabled:
            logger.info("Gap filling is disabled in settings")
            return weather_data, 0

        # Convert to DataFrame for easier manipulation
        df = self._to_dataframe(weather_data)

        if df.empty:
            return weather_data, 0

        # Identify gaps
        df = df.sort_values("timestamp")
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600  # hours

        interpolated_count = 0

        # Interpolate each numeric column
        numeric_columns = [
            "temperature", "humidity", "pressure", "wind_speed",
            "ghi", "dni", "dhi", "precipitation"
        ]

        for col in numeric_columns:
            if col not in df.columns:
                continue

            # Only interpolate gaps smaller than max_gap_hours
            mask = df["time_diff"] <= max_gap_hours

            if method == "linear":
                df.loc[mask, col] = df.loc[mask, col].interpolate(method="linear")
            elif method == "forward_fill":
                df.loc[mask, col] = df.loc[mask, col].ffill()
            elif method == "backward_fill":
                df.loc[mask, col] = df.loc[mask, col].bfill()
            else:
                raise ValueError(f"Unknown gap filling method: {method}")

            # Count interpolated values
            interpolated_count += df.loc[mask, col].isna().sum()

        # Convert back to WeatherDataPoint list
        filled_data = self._from_dataframe(df, weather_data[0])

        logger.info(f"Filled {interpolated_count} data gaps using {method} interpolation")

        return filled_data, interpolated_count

    def unit_conversions(
        self,
        weather_data: list[WeatherDataPoint],
        temp_unit: Optional[str] = None,
        wind_unit: Optional[str] = None,
        irradiance_unit: Optional[str] = None,
    ) -> list[WeatherDataPoint]:
        """
        Convert weather data to specified units.

        Args:
            weather_data: List of weather data points
            temp_unit: Target temperature unit ('celsius', 'fahrenheit', 'kelvin')
            wind_unit: Target wind speed unit ('m_per_s', 'km_per_h', 'mph')
            irradiance_unit: Target irradiance unit ('w_per_m2', 'kw_per_m2')

        Returns:
            Weather data with converted units
        """
        if not weather_data:
            return weather_data

        # Use settings defaults if not specified
        temp_unit = temp_unit or self.settings.temperature_unit
        wind_unit = wind_unit or self.settings.wind_speed_unit
        irradiance_unit = irradiance_unit or self.settings.irradiance_unit

        converted_data = []

        for point in weather_data:
            # Temperature conversion (assume input is Celsius)
            if point.temperature is not None:
                point.temperature = self._convert_temperature(
                    point.temperature, "celsius", temp_unit
                )
            if point.feels_like is not None:
                point.feels_like = self._convert_temperature(
                    point.feels_like, "celsius", temp_unit
                )
            if point.dew_point is not None:
                point.dew_point = self._convert_temperature(
                    point.dew_point, "celsius", temp_unit
                )

            # Wind speed conversion (assume input is m/s)
            if point.wind_speed is not None:
                point.wind_speed = self._convert_wind_speed(
                    point.wind_speed, "m_per_s", wind_unit
                )
            if point.wind_gust is not None:
                point.wind_gust = self._convert_wind_speed(
                    point.wind_gust, "m_per_s", wind_unit
                )

            # Irradiance conversion (assume input is W/m²)
            if point.ghi is not None:
                point.ghi = self._convert_irradiance(
                    point.ghi, "w_per_m2", irradiance_unit
                )
            if point.dni is not None:
                point.dni = self._convert_irradiance(
                    point.dni, "w_per_m2", irradiance_unit
                )
            if point.dhi is not None:
                point.dhi = self._convert_irradiance(
                    point.dhi, "w_per_m2", irradiance_unit
                )

            converted_data.append(point)

        logger.info(
            f"Converted units: temp={temp_unit}, wind={wind_unit}, irradiance={irradiance_unit}"
        )

        return converted_data

    def timestamp_synchronization(
        self,
        weather_data: list[WeatherDataPoint],
        target_timezone: Optional[str] = None,
        resample_frequency: Optional[str] = None,
    ) -> list[WeatherDataPoint]:
        """
        Synchronize and resample weather data timestamps.

        Args:
            weather_data: List of weather data points
            target_timezone: Target timezone (uses setting if not specified)
            resample_frequency: Resampling frequency ('1H', '15T', etc.)

        Returns:
            Weather data with synchronized timestamps
        """
        if not weather_data:
            return weather_data

        target_timezone = target_timezone or self.settings.default_timezone

        # Convert to DataFrame
        df = self._to_dataframe(weather_data)

        if df.empty:
            return weather_data

        # Ensure UTC timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Convert to target timezone
        if target_timezone != "UTC":
            tz = pytz.timezone(target_timezone)
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

        # Resample if requested
        if resample_frequency:
            df = df.set_index("timestamp")
            df = df.resample(resample_frequency).mean()
            df = df.reset_index()

        # Convert back to WeatherDataPoint list
        synchronized_data = self._from_dataframe(df, weather_data[0])

        logger.info(
            f"Synchronized timestamps to {target_timezone}, "
            f"resample={resample_frequency or 'none'}"
        )

        return synchronized_data

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between units."""
        if from_unit == to_unit:
            return value

        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        else:
            celsius = value

        # Convert from Celsius to target
        if to_unit == "fahrenheit":
            return celsius * 9 / 5 + 32
        elif to_unit == "kelvin":
            return celsius + 273.15
        else:
            return celsius

    def _convert_wind_speed(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert wind speed between units."""
        if from_unit == to_unit:
            return value

        # Convert to m/s first
        if from_unit == "km_per_h":
            ms = value / 3.6
        elif from_unit == "mph":
            ms = value * 0.44704
        else:
            ms = value

        # Convert from m/s to target
        if to_unit == "km_per_h":
            return ms * 3.6
        elif to_unit == "mph":
            return ms / 0.44704
        else:
            return ms

    def _convert_irradiance(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert irradiance between units."""
        if from_unit == to_unit:
            return value

        # Convert to W/m² first
        if from_unit == "kw_per_m2":
            wm2 = value * 1000
        else:
            wm2 = value

        # Convert from W/m² to target
        if to_unit == "kw_per_m2":
            return wm2 / 1000
        else:
            return wm2

    def _to_dataframe(self, weather_data: list[WeatherDataPoint]) -> pd.DataFrame:
        """Convert weather data points to pandas DataFrame."""
        data_dicts = [point.model_dump() for point in weather_data]
        return pd.DataFrame(data_dicts)

    def _from_dataframe(
        self, df: pd.DataFrame, template: WeatherDataPoint
    ) -> list[WeatherDataPoint]:
        """Convert pandas DataFrame back to weather data points."""
        data_points = []

        for _, row in df.iterrows():
            point_dict = row.to_dict()

            # Use template for location and provider
            point_dict["location"] = template.location
            point_dict["provider"] = template.provider

            # Handle NaN values
            point_dict = {k: (None if pd.isna(v) else v) for k, v in point_dict.items()}

            try:
                point = WeatherDataPoint(**point_dict)
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Failed to convert row to WeatherDataPoint: {e}")

        return data_points
