"""
Real-time Weather Fetcher with rate limiting and caching.

This module provides high-level functionality for fetching weather data
with automatic caching, rate limiting, and error handling.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from pv_simulator.config import Settings, get_settings
from pv_simulator.models.weather import (
    CurrentWeather,
    ForecastWeather,
    GeoLocation,
    HistoricalWeather,
    WeatherProvider,
)
from pv_simulator.weather.cache import CacheManager, create_cache_manager
from pv_simulator.weather.integrator import WeatherAPIIntegrator

logger = logging.getLogger(__name__)


class RealTimeWeatherFetcher:
    """
    High-level interface for real-time weather data fetching.

    Provides simplified methods for common weather data operations with
    built-in caching, rate limiting, and error handling.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        """
        Initialize Real-time Weather Fetcher.

        Args:
            settings: Application settings (loads from environment if not provided)
            cache_manager: Cache manager instance (creates new if not provided)
        """
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or create_cache_manager(self.settings)
        self.integrator = WeatherAPIIntegrator(self.settings, self.cache_manager)

        logger.info("Initialized Real-time Weather Fetcher")

    def current_conditions(
        self,
        latitude: float,
        longitude: float,
        provider: Optional[WeatherProvider] = None,
    ) -> CurrentWeather:
        """
        Fetch current weather conditions for a location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            provider: Preferred weather provider (automatic selection if not specified)

        Returns:
            Current weather conditions

        Raises:
            ValueError: If coordinates are invalid
            WeatherAPIException: If fetch fails
        """
        location = GeoLocation(latitude=latitude, longitude=longitude)

        logger.info(f"Fetching current conditions for ({latitude}, {longitude})")

        return self.integrator.get_current_weather(location, provider=provider)

    def forecast_data(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
        provider: Optional[WeatherProvider] = None,
    ) -> ForecastWeather:
        """
        Fetch weather forecast for a location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            days: Number of days to forecast (default: 7)
            provider: Preferred weather provider (automatic selection if not specified)

        Returns:
            Weather forecast data

        Raises:
            ValueError: If coordinates are invalid or days < 1
            WeatherAPIException: If fetch fails
        """
        if days < 1:
            raise ValueError("Days must be at least 1")

        location = GeoLocation(latitude=latitude, longitude=longitude)

        logger.info(
            f"Fetching {days}-day forecast for ({latitude}, {longitude})"
        )

        return self.integrator.get_forecast(location, days=days, provider=provider)

    def historical_backfill(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        provider: Optional[WeatherProvider] = None,
    ) -> HistoricalWeather:
        """
        Fetch historical weather data for backfilling gaps.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_date: Start of historical period
            end_date: End of historical period (defaults to now)
            provider: Preferred weather provider (automatic selection if not specified)

        Returns:
            Historical weather data

        Raises:
            ValueError: If coordinates or dates are invalid
            WeatherAPIException: If fetch fails
        """
        if end_date is None:
            end_date = datetime.utcnow()

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        location = GeoLocation(latitude=latitude, longitude=longitude)

        logger.info(
            f"Fetching historical data for ({latitude}, {longitude}) "
            f"from {start_date} to {end_date}"
        )

        return self.integrator.get_historical(
            location, start_date, end_date, provider=provider
        )

    def api_rate_limiting(self, provider: WeatherProvider) -> dict[str, any]:
        """
        Get rate limiting information for a provider.

        Args:
            provider: Weather provider

        Returns:
            Dictionary with rate limit information
        """
        try:
            rate_limit = self.settings.get_rate_limit(provider.value)
            client = self.integrator.clients.get(provider)

            if client:
                rate_limiter = client.rate_limiter
                return {
                    "provider": provider.value,
                    "requests_per_minute": rate_limit,
                    "current_tokens": rate_limiter.tokens,
                    "max_tokens": rate_limiter.max_tokens,
                    "wait_time_seconds": rate_limiter.get_wait_time(),
                }
            else:
                return {
                    "provider": provider.value,
                    "requests_per_minute": rate_limit,
                    "error": "Provider not initialized",
                }

        except ValueError as e:
            return {"error": str(e)}

    def cache_manager(self) -> dict[str, any]:
        """
        Get cache manager status and statistics.

        Returns:
            Dictionary with cache information
        """
        cache_info = {
            "cache_type": self.settings.cache_type,
            "default_ttl": self.settings.cache_ttl,
            "backend": type(self.cache_manager.backend).__name__,
        }

        # Add backend-specific information
        if hasattr(self.cache_manager.backend, "_cache"):
            # Memory cache
            cache_info["entries"] = len(self.cache_manager.backend._cache)
        elif hasattr(self.cache_manager.backend, "db_path"):
            # SQLite cache
            cache_info["db_path"] = str(self.cache_manager.backend.db_path)

        return cache_info

    def clear_cache(self, provider: Optional[WeatherProvider] = None) -> None:
        """
        Clear cached weather data.

        Args:
            provider: Specific provider to clear (clears all if not specified)
        """
        if provider:
            logger.info(f"Clearing cache for {provider.value}")
            # Clear provider-specific cache entries
            # This is a simplified implementation - in production you'd want
            # pattern-based deletion
        else:
            logger.info("Clearing all cached weather data")
            self.cache_manager.clear()

    def get_providers_status(self) -> dict[str, any]:
        """
        Get status of all weather providers.

        Returns:
            Dictionary with provider availability status
        """
        status = {}

        for provider in WeatherProvider:
            is_available = self.integrator.check_provider_availability(provider)
            rate_limit_info = self.api_rate_limiting(provider)

            status[provider.value] = {
                "available": is_available,
                "rate_limit": rate_limit_info,
            }

        return status

    def fetch_multi_provider(
        self, latitude: float, longitude: float, data_type: str = "current"
    ) -> dict[str, any]:
        """
        Fetch data from multiple providers for comparison.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            data_type: Type of data ('current', 'forecast')

        Returns:
            Dictionary with data from each provider
        """
        location = GeoLocation(latitude=latitude, longitude=longitude)
        results = {}

        for provider in self.integrator.get_available_providers():
            # Skip NREL PSM for current/forecast
            if provider == WeatherProvider.NREL_PSM and data_type != "historical":
                continue

            try:
                if data_type == "current":
                    data = self.integrator.get_current_weather(
                        location, provider=provider, fallback=False
                    )
                elif data_type == "forecast":
                    data = self.integrator.get_forecast(
                        location, provider=provider, fallback=False
                    )
                else:
                    continue

                results[provider.value] = {
                    "success": True,
                    "data": data.model_dump(),
                }

            except Exception as e:
                logger.warning(f"Failed to fetch from {provider.value}: {e}")
                results[provider.value] = {
                    "success": False,
                    "error": str(e),
                }

        return results

    def get_solar_irradiance_history(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> HistoricalWeather:
        """
        Fetch historical solar irradiance data (prioritizes NREL PSM).

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_date: Start of period
            end_date: End of period (defaults to now)

        Returns:
            Historical solar irradiance data

        Raises:
            ValueError: If coordinates or dates are invalid
            WeatherAPIException: If fetch fails
        """
        if end_date is None:
            end_date = datetime.utcnow()

        location = GeoLocation(latitude=latitude, longitude=longitude)

        logger.info(
            f"Fetching solar irradiance data for ({latitude}, {longitude}) "
            f"from {start_date} to {end_date}"
        )

        # Prefer NREL PSM for solar data
        return self.integrator.get_historical(
            location,
            start_date,
            end_date,
            provider=WeatherProvider.NREL_PSM,
            fallback=True,
        )
