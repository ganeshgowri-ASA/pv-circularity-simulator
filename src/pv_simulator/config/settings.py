"""
Configuration settings for PV Circularity Simulator.

This module manages all application settings using Pydantic Settings,
with support for environment variables and configuration files.
"""

from typing import Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Weather API Keys
    nsrdb_api_key: str = Field(
        default="DEMO_KEY",
        description="NREL NSRDB API key (use DEMO_KEY for testing)",
    )
    pvgis_api_key: Optional[str] = Field(
        default=None, description="PVGIS API key (optional, not required for basic usage)"
    )
    openweather_api_key: Optional[str] = Field(
        default=None, description="OpenWeather API key for real-time weather data"
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./pv_simulator.db", description="Database connection URL"
    )

    # Default Location Settings
    default_latitude: float = Field(default=39.7392, description="Default latitude (Denver, CO)")
    default_longitude: float = Field(
        default=-104.9903, description="Default longitude (Denver, CO)"
    )
    default_elevation: float = Field(default=1609.0, description="Default elevation in meters")
    default_timezone: str = Field(default="America/Denver", description="Default timezone")

    # API Rate Limiting
    max_api_retries: int = Field(default=3, description="Maximum API retry attempts")
    api_timeout_seconds: int = Field(default=30, description="API request timeout in seconds")

    # Data Directories
    tmy_cache_dir: Path = Field(
        default=Path("./data/tmy_cache"), description="TMY data cache directory"
    )
    weather_data_dir: Path = Field(
        default=Path("./data/weather"), description="Weather data storage directory"
    )
    locations_data_dir: Path = Field(
        default=Path("./data/locations"), description="Global locations database directory"
    )

    # Application Settings
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(
        default="localhost", description="Streamlit server address"
    )

    # API Endpoints
    nsrdb_api_url: str = Field(
        default="https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv",
        description="NREL NSRDB API endpoint",
    )
    nsrdb_tmy_api_url: str = Field(
        default="https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-tmy-download.csv",
        description="NREL NSRDB TMY API endpoint",
    )
    pvgis_api_url: str = Field(
        default="https://re.jrc.ec.europa.eu/api/v5_2",
        description="PVGIS API base URL",
    )

    # TMY Generation Parameters
    tmy_representative_years: int = Field(
        default=12, description="Number of years for TMY generation"
    )
    tmy_start_year: int = Field(default=2007, description="TMY data start year")
    tmy_end_year: int = Field(default=2021, description="TMY data end year")

    # Data Quality
    max_missing_data_percentage: float = Field(
        default=5.0, description="Maximum allowed missing data percentage for TMY"
    )
    interpolation_max_gap_hours: int = Field(
        default=3, description="Maximum gap hours for data interpolation"
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.tmy_cache_dir.mkdir(parents=True, exist_ok=True)
        self.weather_data_dir.mkdir(parents=True, exist_ok=True)
        self.locations_data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.ensure_directories()
