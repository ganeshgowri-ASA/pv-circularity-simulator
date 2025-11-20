"""
Weather data models for PV Circularity Simulator.

This module provides Pydantic models for weather data validation,
serialization, and type safety across different weather API providers.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WeatherProvider(str, Enum):
    """Enumeration of supported weather API providers."""

    OPENWEATHERMAP = "openweathermap"
    VISUALCROSSING = "visualcrossing"
    METEOMATICS = "meteomatics"
    TOMORROW_IO = "tomorrow_io"
    NREL_PSM = "nrel_psm"


class GeoLocation(BaseModel):
    """Geographic location coordinates."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    elevation: Optional[float] = Field(
        None, description="Elevation above sea level in meters"
    )
    city: Optional[str] = Field(None, description="City name")
    country: Optional[str] = Field(None, description="Country name or code")
    timezone: Optional[str] = Field(None, description="IANA timezone identifier")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within valid range."""
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within valid range."""
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v

    def __str__(self) -> str:
        """Return string representation of location."""
        if self.city and self.country:
            return f"{self.city}, {self.country} ({self.latitude:.4f}, {self.longitude:.4f})"
        return f"({self.latitude:.4f}, {self.longitude:.4f})"


class WeatherDataPoint(BaseModel):
    """
    Core weather data point with all relevant meteorological parameters.

    This model represents a single weather observation at a specific time and location.
    All fields use SI units for consistency.
    """

    timestamp: datetime = Field(..., description="UTC timestamp of the observation")
    location: GeoLocation = Field(..., description="Geographic location of observation")
    provider: WeatherProvider = Field(..., description="Weather data provider")

    # Temperature (Celsius)
    temperature: Optional[float] = Field(
        None, ge=-100, le=60, description="Air temperature in Celsius"
    )
    feels_like: Optional[float] = Field(
        None, ge=-100, le=60, description="Apparent temperature in Celsius"
    )
    dew_point: Optional[float] = Field(
        None, ge=-100, le=50, description="Dew point temperature in Celsius"
    )

    # Humidity (%)
    humidity: Optional[float] = Field(
        None, ge=0, le=100, description="Relative humidity percentage"
    )

    # Pressure (hPa/mbar)
    pressure: Optional[float] = Field(
        None, ge=800, le=1100, description="Atmospheric pressure in hPa"
    )
    pressure_sea_level: Optional[float] = Field(
        None, ge=800, le=1100, description="Sea level pressure in hPa"
    )

    # Wind (m/s and degrees)
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed in m/s")
    wind_gust: Optional[float] = Field(None, ge=0, description="Wind gust speed in m/s")
    wind_direction: Optional[float] = Field(
        None, ge=0, le=360, description="Wind direction in degrees"
    )

    # Clouds and visibility
    cloud_cover: Optional[float] = Field(
        None, ge=0, le=100, description="Cloud cover percentage"
    )
    visibility: Optional[float] = Field(None, ge=0, description="Visibility in meters")

    # Precipitation
    precipitation: Optional[float] = Field(
        None, ge=0, description="Precipitation amount in mm"
    )
    precipitation_probability: Optional[float] = Field(
        None, ge=0, le=100, description="Precipitation probability percentage"
    )
    rain: Optional[float] = Field(None, ge=0, description="Rain amount in mm")
    snow: Optional[float] = Field(None, ge=0, description="Snow amount in mm")

    # Solar radiation (critical for PV simulation)
    ghi: Optional[float] = Field(
        None, ge=0, description="Global Horizontal Irradiance in W/m²"
    )
    dni: Optional[float] = Field(
        None, ge=0, description="Direct Normal Irradiance in W/m²"
    )
    dhi: Optional[float] = Field(
        None, ge=0, description="Diffuse Horizontal Irradiance in W/m²"
    )
    solar_elevation: Optional[float] = Field(
        None, ge=-90, le=90, description="Solar elevation angle in degrees"
    )
    solar_azimuth: Optional[float] = Field(
        None, ge=0, le=360, description="Solar azimuth angle in degrees"
    )

    # UV index
    uv_index: Optional[float] = Field(None, ge=0, description="UV index")

    # Weather condition
    condition: Optional[str] = Field(None, description="Weather condition description")
    condition_code: Optional[int] = Field(None, description="Weather condition code")

    # Data quality indicators
    quality_score: Optional[float] = Field(
        None, ge=0, le=1, description="Data quality score (0-1)"
    )
    is_interpolated: bool = Field(
        default=False, description="Whether data was interpolated"
    )
    is_forecast: bool = Field(default=False, description="Whether data is forecast")

    class Config:
        """Pydantic model configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @field_validator("wind_direction")
    @classmethod
    def validate_wind_direction(cls, v: Optional[float]) -> Optional[float]:
        """Normalize wind direction to 0-360 range."""
        if v is not None:
            return v % 360
        return v

    def has_solar_data(self) -> bool:
        """Check if this data point contains solar irradiance data."""
        return any([self.ghi is not None, self.dni is not None, self.dhi is not None])

    def has_temperature_data(self) -> bool:
        """Check if this data point contains temperature data."""
        return self.temperature is not None

    def has_wind_data(self) -> bool:
        """Check if this data point contains wind data."""
        return self.wind_speed is not None


class CurrentWeather(BaseModel):
    """Current weather conditions at a location."""

    data: WeatherDataPoint = Field(..., description="Current weather data")
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when data was fetched"
    )
    cache_key: Optional[str] = Field(None, description="Cache key for this data")

    def is_stale(self, max_age_seconds: int = 600) -> bool:
        """
        Check if the current weather data is stale.

        Args:
            max_age_seconds: Maximum age in seconds (default: 10 minutes)

        Returns:
            True if data is older than max_age_seconds
        """
        age = (datetime.utcnow() - self.fetched_at).total_seconds()
        return age > max_age_seconds


class ForecastWeather(BaseModel):
    """Weather forecast for a location."""

    location: GeoLocation = Field(..., description="Forecast location")
    provider: WeatherProvider = Field(..., description="Weather data provider")
    forecast_data: list[WeatherDataPoint] = Field(
        ..., description="List of forecasted weather data points"
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when forecast was fetched"
    )
    forecast_start: datetime = Field(..., description="Start of forecast period")
    forecast_end: datetime = Field(..., description="End of forecast period")

    @field_validator("forecast_data")
    @classmethod
    def validate_forecast_data(cls, v: list[WeatherDataPoint]) -> list[WeatherDataPoint]:
        """Validate forecast data is not empty and sorted by timestamp."""
        if not v:
            raise ValueError("Forecast data cannot be empty")

        # Check if sorted
        timestamps = [dp.timestamp for dp in v]
        if timestamps != sorted(timestamps):
            raise ValueError("Forecast data must be sorted by timestamp")

        return v

    def get_hourly_forecast(self) -> list[WeatherDataPoint]:
        """Get hourly forecast data points."""
        return self.forecast_data

    def get_daily_summary(self) -> dict[str, WeatherDataPoint]:
        """Get daily summary of forecast (simplified - returns first point per day)."""
        daily = {}
        for dp in self.forecast_data:
            date_key = dp.timestamp.date().isoformat()
            if date_key not in daily:
                daily[date_key] = dp
        return daily


class HistoricalWeather(BaseModel):
    """Historical weather data for a location."""

    location: GeoLocation = Field(..., description="Location")
    provider: WeatherProvider = Field(..., description="Weather data provider")
    historical_data: list[WeatherDataPoint] = Field(
        ..., description="List of historical weather data points"
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when data was fetched"
    )
    period_start: datetime = Field(..., description="Start of historical period")
    period_end: datetime = Field(..., description="End of historical period")

    @field_validator("historical_data")
    @classmethod
    def validate_historical_data(cls, v: list[WeatherDataPoint]) -> list[WeatherDataPoint]:
        """Validate historical data is not empty and sorted by timestamp."""
        if not v:
            raise ValueError("Historical data cannot be empty")

        # Check if sorted
        timestamps = [dp.timestamp for dp in v]
        if timestamps != sorted(timestamps):
            raise ValueError("Historical data must be sorted by timestamp")

        return v

    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get the actual date range of the data."""
        timestamps = [dp.timestamp for dp in self.historical_data]
        return min(timestamps), max(timestamps)

    def count_data_points(self) -> int:
        """Get the number of data points."""
        return len(self.historical_data)


class DataQualityMetrics(BaseModel):
    """Metrics for assessing weather data quality."""

    total_points: int = Field(..., ge=0, description="Total number of data points")
    valid_points: int = Field(..., ge=0, description="Number of valid data points")
    missing_points: int = Field(..., ge=0, description="Number of missing data points")
    interpolated_points: int = Field(
        default=0, ge=0, description="Number of interpolated points"
    )
    outliers_detected: int = Field(default=0, ge=0, description="Number of outliers detected")
    outliers_corrected: int = Field(
        default=0, ge=0, description="Number of outliers corrected"
    )

    # Field-specific completeness
    temperature_completeness: float = Field(
        default=0.0, ge=0, le=1, description="Temperature data completeness (0-1)"
    )
    solar_completeness: float = Field(
        default=0.0, ge=0, le=1, description="Solar irradiance data completeness (0-1)"
    )
    wind_completeness: float = Field(
        default=0.0, ge=0, le=1, description="Wind data completeness (0-1)"
    )

    # Time range
    start_time: datetime = Field(..., description="Start of data period")
    end_time: datetime = Field(..., description="End of data period")

    # Overall quality score
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")

    @field_validator("valid_points")
    @classmethod
    def validate_valid_points(cls, v: int, info: dict) -> int:
        """Validate that valid_points <= total_points."""
        if "total_points" in info.data and v > info.data["total_points"]:
            raise ValueError("valid_points cannot exceed total_points")
        return v

    @field_validator("outliers_corrected")
    @classmethod
    def validate_outliers_corrected(cls, v: int, info: dict) -> int:
        """Validate that outliers_corrected <= outliers_detected."""
        if "outliers_detected" in info.data and v > info.data["outliers_detected"]:
            raise ValueError("outliers_corrected cannot exceed outliers_detected")
        return v

    def completeness_ratio(self) -> float:
        """Calculate overall data completeness ratio."""
        if self.total_points == 0:
            return 0.0
        return self.valid_points / self.total_points

    def has_high_quality(self, threshold: float = 0.8) -> bool:
        """
        Check if data meets high quality threshold.

        Args:
            threshold: Minimum quality score (default: 0.8)

        Returns:
            True if quality_score >= threshold
        """
        return self.quality_score >= threshold
