"""
Pydantic models for weather and TMY data.

This module defines all data models for weather data, TMY files,
locations, and related metadata with full type hints and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DataSource(str, Enum):
    """Supported weather data sources."""

    NSRDB = "NSRDB"  # National Solar Radiation Database
    PVGIS = "PVGIS"  # Photovoltaic Geographical Information System
    METEONORM = "Meteonorm"
    EPW = "EPW"  # EnergyPlus Weather
    LOCAL_STATION = "Local Station"
    SATELLITE = "Satellite"
    CUSTOM = "Custom"


class TemporalResolution(str, Enum):
    """Temporal resolution of weather data."""

    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    ANNUAL = "annual"


class TMYFormat(str, Enum):
    """Supported TMY file formats."""

    TMY2 = "TMY2"  # TMY2 format
    TMY3 = "TMY3"  # TMY3 format
    EPW = "EPW"  # EnergyPlus Weather format
    CSV = "CSV"  # Generic CSV format
    JSON = "JSON"  # JSON format
    NETCDF = "NetCDF"  # NetCDF format


class DataQuality(str, Enum):
    """Data quality flags."""

    EXCELLENT = "excellent"  # < 1% missing data
    GOOD = "good"  # 1-5% missing data
    FAIR = "fair"  # 5-10% missing data
    POOR = "poor"  # > 10% missing data
    UNKNOWN = "unknown"


class WeatherDataPoint(BaseModel):
    """
    Single weather data point with all meteorological variables.

    Attributes:
        timestamp: UTC timestamp of the measurement
        temperature: Ambient temperature in Celsius
        irradiance_ghi: Global Horizontal Irradiance in W/m²
        irradiance_dni: Direct Normal Irradiance in W/m²
        irradiance_dhi: Diffuse Horizontal Irradiance in W/m²
        wind_speed: Wind speed in m/s
        wind_direction: Wind direction in degrees (0-360)
        relative_humidity: Relative humidity in %
        pressure: Atmospheric pressure in Pa
        albedo: Surface albedo (0-1)
        precipitable_water: Precipitable water in cm
        cloud_cover: Cloud cover fraction (0-1)
        data_quality: Quality flag for this data point
    """

    timestamp: datetime = Field(..., description="UTC timestamp")
    temperature: float = Field(..., description="Ambient temperature (°C)", ge=-100, le=60)
    irradiance_ghi: float = Field(
        ..., description="Global Horizontal Irradiance (W/m²)", ge=0, le=1500
    )
    irradiance_dni: float = Field(
        ..., description="Direct Normal Irradiance (W/m²)", ge=0, le=1500
    )
    irradiance_dhi: float = Field(
        ..., description="Diffuse Horizontal Irradiance (W/m²)", ge=0, le=1000
    )
    wind_speed: float = Field(..., description="Wind speed (m/s)", ge=0, le=100)
    wind_direction: Optional[float] = Field(
        None, description="Wind direction (degrees)", ge=0, le=360
    )
    relative_humidity: Optional[float] = Field(
        None, description="Relative humidity (%)", ge=0, le=100
    )
    pressure: Optional[float] = Field(
        None, description="Atmospheric pressure (Pa)", ge=50000, le=110000
    )
    albedo: Optional[float] = Field(None, description="Surface albedo", ge=0, le=1)
    precipitable_water: Optional[float] = Field(
        None, description="Precipitable water (cm)", ge=0
    )
    cloud_cover: Optional[float] = Field(None, description="Cloud cover fraction", ge=0, le=1)
    data_quality: DataQuality = Field(default=DataQuality.UNKNOWN, description="Data quality flag")

    @field_validator("irradiance_ghi", "irradiance_dni", "irradiance_dhi")
    @classmethod
    def validate_irradiance(cls, v: float) -> float:
        """Validate irradiance values are non-negative."""
        if v < 0:
            raise ValueError("Irradiance values must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_irradiance_consistency(self) -> "WeatherDataPoint":
        """Validate that GHI >= DHI (physical consistency)."""
        if self.irradiance_ghi < self.irradiance_dhi:
            raise ValueError("GHI must be greater than or equal to DHI")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2023-07-15T12:00:00Z",
                "temperature": 28.5,
                "irradiance_ghi": 950.0,
                "irradiance_dni": 850.0,
                "irradiance_dhi": 150.0,
                "wind_speed": 3.5,
                "wind_direction": 180.0,
                "relative_humidity": 45.0,
                "pressure": 101325.0,
                "albedo": 0.2,
                "data_quality": "excellent",
            }
        }
    }


class GlobalLocation(BaseModel):
    """
    Global location with geographic metadata.

    Attributes:
        name: Location name (city, station name, etc.)
        country: Country name or code
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        elevation: Elevation above sea level in meters
        timezone: IANA timezone identifier
        climate_zone: Köppen climate classification
        population: Population (optional)
        metadata: Additional metadata
    """

    name: str = Field(..., description="Location name", min_length=1)
    country: str = Field(..., description="Country name or ISO code", min_length=2)
    latitude: float = Field(..., description="Latitude (degrees)", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude (degrees)", ge=-180, le=180)
    elevation: float = Field(..., description="Elevation (meters)", ge=-500, le=9000)
    timezone: str = Field(..., description="IANA timezone")
    climate_zone: Optional[str] = Field(None, description="Köppen climate classification")
    population: Optional[int] = Field(None, description="Population", ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Denver",
                "country": "USA",
                "latitude": 39.7392,
                "longitude": -104.9903,
                "elevation": 1609.0,
                "timezone": "America/Denver",
                "climate_zone": "BSk",
                "population": 715522,
            }
        }
    }


class TMYData(BaseModel):
    """
    Typical Meteorological Year (TMY) dataset.

    TMY data represents a synthesized year of hourly weather data
    compiled from multiple years of historical data to represent
    typical climatic conditions at a location.

    Attributes:
        location: Location information
        data_source: Source of the TMY data
        format_type: TMY file format
        temporal_resolution: Time resolution of data
        start_year: First year in source data range
        end_year: Last year in source data range
        hourly_data: List of hourly weather data points
        metadata: Additional metadata
        data_quality: Overall data quality assessment
        completeness_percentage: Percentage of complete data (0-100)
    """

    location: GlobalLocation = Field(..., description="Location information")
    data_source: DataSource = Field(..., description="Data source")
    format_type: TMYFormat = Field(default=TMYFormat.TMY3, description="File format")
    temporal_resolution: TemporalResolution = Field(
        default=TemporalResolution.HOURLY, description="Temporal resolution"
    )
    start_year: int = Field(..., description="Start year of source data", ge=1900, le=2100)
    end_year: int = Field(..., description="End year of source data", ge=1900, le=2100)
    hourly_data: List[WeatherDataPoint] = Field(
        ..., description="Hourly weather data", min_length=1
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    data_quality: DataQuality = Field(
        default=DataQuality.UNKNOWN, description="Overall data quality"
    )
    completeness_percentage: float = Field(
        default=100.0, description="Data completeness (%)", ge=0, le=100
    )

    @field_validator("end_year")
    @classmethod
    def validate_year_range(cls, v: int, info: Any) -> int:
        """Validate that end_year >= start_year."""
        if "start_year" in info.data and v < info.data["start_year"]:
            raise ValueError("end_year must be greater than or equal to start_year")
        return v

    @field_validator("hourly_data")
    @classmethod
    def validate_hourly_data_length(cls, v: List[WeatherDataPoint]) -> List[WeatherDataPoint]:
        """Validate that hourly data has reasonable length."""
        expected_lengths = [8760, 8784]  # Normal year, leap year
        if len(v) not in expected_lengths and len(v) < 8760:
            if len(v) < 100:  # Too short to be valid TMY
                raise ValueError(f"TMY data too short: {len(v)} points (expected ~8760)")
        return v

    def get_annual_irradiation(self) -> float:
        """
        Calculate total annual irradiation in kWh/m².

        Returns:
            Total annual GHI in kWh/m²
        """
        total_wh = sum(point.irradiance_ghi for point in self.hourly_data)
        return total_wh / 1000.0  # Convert Wh to kWh

    def get_average_temperature(self) -> float:
        """
        Calculate average annual temperature.

        Returns:
            Average temperature in °C
        """
        return sum(point.temperature for point in self.hourly_data) / len(self.hourly_data)

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": {
                    "name": "Denver",
                    "country": "USA",
                    "latitude": 39.7392,
                    "longitude": -104.9903,
                    "elevation": 1609.0,
                    "timezone": "America/Denver",
                },
                "data_source": "NSRDB",
                "format_type": "TMY3",
                "temporal_resolution": "hourly",
                "start_year": 2007,
                "end_year": 2021,
                "data_quality": "excellent",
                "completeness_percentage": 99.8,
            }
        }
    }


class ExtremeWeatherEvent(BaseModel):
    """
    Extreme weather event record.

    Attributes:
        event_type: Type of extreme event
        timestamp: When the event occurred
        location: Location of the event
        severity: Severity level (1-5)
        value: Measured value of the extreme
        unit: Unit of measurement
        description: Event description
        impact_score: Estimated impact score (0-10)
    """

    event_type: str = Field(
        ..., description="Event type (e.g., 'heatwave', 'high_wind', 'low_irradiance')"
    )
    timestamp: datetime = Field(..., description="Event timestamp")
    location: GlobalLocation = Field(..., description="Event location")
    severity: int = Field(..., description="Severity level", ge=1, le=5)
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    description: str = Field(default="", description="Event description")
    impact_score: Optional[float] = Field(None, description="Impact score", ge=0, le=10)

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "heatwave",
                "timestamp": "2023-07-15T14:00:00Z",
                "severity": 4,
                "value": 42.5,
                "unit": "°C",
                "description": "Extreme heat event with temperatures exceeding 40°C for 3 consecutive days",
                "impact_score": 7.5,
            }
        }
    }


class HistoricalWeatherStats(BaseModel):
    """
    Multi-year historical weather statistics.

    Attributes:
        location: Location for these statistics
        start_year: First year of analysis
        end_year: Last year of analysis
        num_years: Number of years analyzed
        mean_annual_ghi: Mean annual GHI (kWh/m²/year)
        std_annual_ghi: Standard deviation of annual GHI
        mean_temperature: Mean temperature (°C)
        std_temperature: Standard deviation of temperature
        p90_annual_ghi: P90 annual GHI (90th percentile)
        p50_annual_ghi: P50 annual GHI (50th percentile, median)
        p10_annual_ghi: P10 annual GHI (10th percentile)
        extreme_events: List of extreme weather events
        seasonal_variability: Seasonal statistics
        inter_annual_variability: Year-to-year variability metrics
        climate_change_trend: Climate change trend analysis
    """

    location: GlobalLocation = Field(..., description="Location")
    start_year: int = Field(..., description="Start year", ge=1900, le=2100)
    end_year: int = Field(..., description="End year", ge=1900, le=2100)
    num_years: int = Field(..., description="Number of years", ge=1)

    # Irradiance statistics
    mean_annual_ghi: float = Field(..., description="Mean annual GHI (kWh/m²/year)", ge=0)
    std_annual_ghi: float = Field(..., description="Std dev of annual GHI (kWh/m²/year)", ge=0)
    p90_annual_ghi: float = Field(..., description="P90 annual GHI (kWh/m²/year)", ge=0)
    p50_annual_ghi: float = Field(..., description="P50 annual GHI (kWh/m²/year)", ge=0)
    p10_annual_ghi: float = Field(..., description="P10 annual GHI (kWh/m²/year)", ge=0)

    # Temperature statistics
    mean_temperature: float = Field(..., description="Mean temperature (°C)")
    std_temperature: float = Field(..., description="Std dev of temperature (°C)", ge=0)

    # Additional analytics
    extreme_events: List[ExtremeWeatherEvent] = Field(
        default_factory=list, description="Extreme weather events"
    )
    seasonal_variability: Dict[str, Any] = Field(
        default_factory=dict, description="Seasonal statistics"
    )
    inter_annual_variability: Dict[str, float] = Field(
        default_factory=dict, description="Inter-annual variability metrics"
    )
    climate_change_trend: Optional[Dict[str, Any]] = Field(
        None, description="Climate change trend analysis"
    )

    @field_validator("end_year")
    @classmethod
    def validate_year_range(cls, v: int, info: Any) -> int:
        """Validate that end_year >= start_year."""
        if "start_year" in info.data and v < info.data["start_year"]:
            raise ValueError("end_year must be >= start_year")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": {
                    "name": "Denver",
                    "country": "USA",
                    "latitude": 39.7392,
                    "longitude": -104.9903,
                    "elevation": 1609.0,
                    "timezone": "America/Denver",
                },
                "start_year": 2007,
                "end_year": 2021,
                "num_years": 15,
                "mean_annual_ghi": 1850.0,
                "std_annual_ghi": 85.0,
                "p90_annual_ghi": 1720.0,
                "p50_annual_ghi": 1850.0,
                "p10_annual_ghi": 1980.0,
                "mean_temperature": 10.5,
                "std_temperature": 0.8,
            }
        }
    }
