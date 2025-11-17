"""
Configuration management for PV Circularity Simulator.

This module provides centralized configuration management using Pydantic Settings,
supporting environment variables and .env files.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Weather API Keys
    openweathermap_api_key: str = Field(default="", description="OpenWeatherMap API key")
    visualcrossing_api_key: str = Field(default="", description="Visual Crossing API key")
    meteomatics_username: str = Field(default="", description="Meteomatics username")
    meteomatics_password: str = Field(default="", description="Meteomatics password")
    tomorrow_io_api_key: str = Field(default="", description="Tomorrow.io API key")
    nrel_api_key: str = Field(default="", description="NREL API key")

    # Cache Configuration
    cache_type: Literal["sqlite", "redis", "memory"] = Field(
        default="sqlite", description="Cache backend type"
    )
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # API Rate Limiting (requests per minute)
    openweathermap_rate_limit: int = Field(default=60, description="OpenWeatherMap rate limit")
    visualcrossing_rate_limit: int = Field(
        default=1000, description="Visual Crossing rate limit"
    )
    meteomatics_rate_limit: int = Field(default=50, description="Meteomatics rate limit")
    tomorrow_io_rate_limit: int = Field(default=25, description="Tomorrow.io rate limit")
    nrel_rate_limit: int = Field(default=1000, description="NREL rate limit")

    # Database
    database_url: str = Field(
        default="sqlite:///./pv_simulator.db", description="Database connection URL"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Application Settings
    default_timezone: str = Field(default="UTC", description="Default timezone")
    temperature_unit: Literal["celsius", "fahrenheit", "kelvin"] = Field(
        default="celsius", description="Temperature unit"
    )
    irradiance_unit: Literal["w_per_m2", "kw_per_m2"] = Field(
        default="w_per_m2", description="Irradiance unit"
    )
    wind_speed_unit: Literal["m_per_s", "km_per_h", "mph"] = Field(
        default="m_per_s", description="Wind speed unit"
    )

    # Data Quality
    outlier_detection_enabled: bool = Field(
        default=True, description="Enable outlier detection"
    )
    gap_filling_enabled: bool = Field(default=True, description="Enable gap filling")
    max_gap_hours: int = Field(default=3, description="Maximum gap to fill in hours")

    # UI Configuration
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(
        default="localhost", description="Streamlit server address"
    )

    @field_validator("cache_ttl", "max_gap_hours")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that numeric fields are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator(
        "openweathermap_rate_limit",
        "visualcrossing_rate_limit",
        "meteomatics_rate_limit",
        "tomorrow_io_rate_limit",
        "nrel_rate_limit",
    )
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate that rate limits are reasonable."""
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        if v > 10000:
            raise ValueError("Rate limit too high (max 10000)")
        return v

    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a specific weather provider.

        Args:
            provider: Name of the weather provider (e.g., 'openweathermap')

        Returns:
            API key for the provider

        Raises:
            ValueError: If provider is not recognized or API key is not set
        """
        key_map = {
            "openweathermap": self.openweathermap_api_key,
            "visualcrossing": self.visualcrossing_api_key,
            "tomorrow_io": self.tomorrow_io_api_key,
            "nrel": self.nrel_api_key,
        }

        if provider not in key_map:
            raise ValueError(f"Unknown provider: {provider}")

        api_key = key_map[provider]
        if not api_key:
            raise ValueError(f"API key not configured for provider: {provider}")

        return api_key

    def get_rate_limit(self, provider: str) -> int:
        """
        Get rate limit for a specific weather provider.

        Args:
            provider: Name of the weather provider

        Returns:
            Rate limit in requests per minute

        Raises:
            ValueError: If provider is not recognized
        """
        limit_map = {
            "openweathermap": self.openweathermap_rate_limit,
            "visualcrossing": self.visualcrossing_rate_limit,
            "meteomatics": self.meteomatics_rate_limit,
            "tomorrow_io": self.tomorrow_io_rate_limit,
            "nrel": self.nrel_rate_limit,
        }

        if provider not in limit_map:
            raise ValueError(f"Unknown provider: {provider}")

        return limit_map[provider]


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function uses lru_cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.

    Returns:
        Application settings instance
    """
    return Settings()
