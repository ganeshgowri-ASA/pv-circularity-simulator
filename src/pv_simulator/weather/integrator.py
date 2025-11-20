"""
Weather API Integrator - Unified interface for multiple weather providers.

This module provides a high-level orchestrator that manages multiple weather API
clients and provides a unified interface for fetching weather data.
"""

import logging
from datetime import datetime
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
from pv_simulator.weather.clients.base import BaseWeatherClient, WeatherAPIException
from pv_simulator.weather.clients.meteomatics import MeteomaticsClient
from pv_simulator.weather.clients.nrel_psm import NRELPSMClient
from pv_simulator.weather.clients.openweathermap import OpenWeatherMapClient
from pv_simulator.weather.clients.tomorrow_io import TomorrowIOClient
from pv_simulator.weather.clients.visualcrossing import VisualCrossingClient

logger = logging.getLogger(__name__)


class WeatherAPIIntegrator:
    """
    Unified interface for multiple weather API providers.

    This class orchestrates multiple weather API clients and provides
    fallback mechanisms, provider selection, and aggregation capabilities.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        """
        Initialize Weather API Integrator.

        Args:
            settings: Application settings (loads from environment if not provided)
            cache_manager: Cache manager instance (creates new if not provided)
        """
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or create_cache_manager(self.settings)

        # Initialize clients
        self.clients: dict[WeatherProvider, BaseWeatherClient] = {}
        self._init_clients()

        logger.info(
            f"Initialized Weather API Integrator with {len(self.clients)} providers"
        )

    def _init_clients(self) -> None:
        """Initialize all available weather API clients."""
        client_map = {
            WeatherProvider.OPENWEATHERMAP: OpenWeatherMapClient,
            WeatherProvider.VISUALCROSSING: VisualCrossingClient,
            WeatherProvider.METEOMATICS: MeteomaticsClient,
            WeatherProvider.TOMORROW_IO: TomorrowIOClient,
            WeatherProvider.NREL_PSM: NRELPSMClient,
        }

        for provider, client_class in client_map.items():
            try:
                client = client_class(self.settings, self.cache_manager)
                self.clients[provider] = client
                logger.info(f"Initialized {provider.value} client")
            except ValueError as e:
                logger.warning(f"Could not initialize {provider.value} client: {e}")

    def openweathermap_api(self, location: GeoLocation, data_type: str = "current") -> dict:
        """
        Fetch data from OpenWeatherMap API.

        Args:
            location: Geographic location
            data_type: Type of data to fetch ('current', 'forecast', 'historical')

        Returns:
            Weather data from OpenWeatherMap

        Raises:
            WeatherAPIException: If request fails
        """
        client = self.clients.get(WeatherProvider.OPENWEATHERMAP)
        if not client:
            raise ValueError("OpenWeatherMap client not initialized")

        if data_type == "current":
            return client.get_current_weather(location).model_dump()
        elif data_type == "forecast":
            return client.get_forecast(location).model_dump()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def visualcrossing_api(self, location: GeoLocation, data_type: str = "current") -> dict:
        """
        Fetch data from Visual Crossing API.

        Args:
            location: Geographic location
            data_type: Type of data to fetch ('current', 'forecast', 'historical')

        Returns:
            Weather data from Visual Crossing

        Raises:
            WeatherAPIException: If request fails
        """
        client = self.clients.get(WeatherProvider.VISUALCROSSING)
        if not client:
            raise ValueError("Visual Crossing client not initialized")

        if data_type == "current":
            return client.get_current_weather(location).model_dump()
        elif data_type == "forecast":
            return client.get_forecast(location).model_dump()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def meteomatics_api(self, location: GeoLocation, data_type: str = "current") -> dict:
        """
        Fetch data from Meteomatics API.

        Args:
            location: Geographic location
            data_type: Type of data to fetch ('current', 'forecast', 'historical')

        Returns:
            Weather data from Meteomatics

        Raises:
            WeatherAPIException: If request fails
        """
        client = self.clients.get(WeatherProvider.METEOMATICS)
        if not client:
            raise ValueError("Meteomatics client not initialized")

        if data_type == "current":
            return client.get_current_weather(location).model_dump()
        elif data_type == "forecast":
            return client.get_forecast(location).model_dump()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def tomorrow_io_api(self, location: GeoLocation, data_type: str = "current") -> dict:
        """
        Fetch data from Tomorrow.io API.

        Args:
            location: Geographic location
            data_type: Type of data to fetch ('current', 'forecast', 'historical')

        Returns:
            Weather data from Tomorrow.io

        Raises:
            WeatherAPIException: If request fails
        """
        client = self.clients.get(WeatherProvider.TOMORROW_IO)
        if not client:
            raise ValueError("Tomorrow.io client not initialized")

        if data_type == "current":
            return client.get_current_weather(location).model_dump()
        elif data_type == "forecast":
            return client.get_forecast(location).model_dump()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def nrel_psm_api(
        self, location: GeoLocation, start_date: datetime, end_date: datetime
    ) -> dict:
        """
        Fetch historical solar irradiance data from NREL PSM API.

        Args:
            location: Geographic location
            start_date: Start of period
            end_date: End of period

        Returns:
            Solar irradiance data from NREL PSM

        Raises:
            WeatherAPIException: If request fails
        """
        client = self.clients.get(WeatherProvider.NREL_PSM)
        if not client:
            raise ValueError("NREL PSM client not initialized")

        return client.get_historical(location, start_date, end_date).model_dump()

    def get_current_weather(
        self,
        location: GeoLocation,
        provider: Optional[WeatherProvider] = None,
        fallback: bool = True,
    ) -> CurrentWeather:
        """
        Get current weather conditions with optional provider selection and fallback.

        Args:
            location: Geographic location
            provider: Preferred provider (tries all if not specified)
            fallback: Whether to try other providers if preferred fails

        Returns:
            Current weather data

        Raises:
            WeatherAPIException: If all providers fail
        """
        providers_to_try = [provider] if provider else list(self.clients.keys())

        # Filter out NREL PSM for current weather (doesn't provide real-time data)
        providers_to_try = [
            p for p in providers_to_try if p != WeatherProvider.NREL_PSM
        ]

        last_error = None

        for p in providers_to_try:
            client = self.clients.get(p)
            if not client:
                continue

            try:
                logger.info(f"Fetching current weather from {p.value}")
                return client.get_current_weather(location)
            except WeatherAPIException as e:
                logger.warning(f"Failed to fetch from {p.value}: {e}")
                last_error = e
                if not fallback:
                    raise

        raise WeatherAPIException(
            f"All providers failed to fetch current weather. Last error: {last_error}"
        )

    def get_forecast(
        self,
        location: GeoLocation,
        days: int = 7,
        provider: Optional[WeatherProvider] = None,
        fallback: bool = True,
    ) -> ForecastWeather:
        """
        Get weather forecast with optional provider selection and fallback.

        Args:
            location: Geographic location
            days: Number of days to forecast
            provider: Preferred provider (tries all if not specified)
            fallback: Whether to try other providers if preferred fails

        Returns:
            Weather forecast data

        Raises:
            WeatherAPIException: If all providers fail
        """
        providers_to_try = [provider] if provider else list(self.clients.keys())

        # Filter out NREL PSM (doesn't provide forecasts)
        providers_to_try = [
            p for p in providers_to_try if p != WeatherProvider.NREL_PSM
        ]

        last_error = None

        for p in providers_to_try:
            client = self.clients.get(p)
            if not client:
                continue

            try:
                logger.info(f"Fetching forecast from {p.value}")
                return client.get_forecast(location, days)
            except WeatherAPIException as e:
                logger.warning(f"Failed to fetch from {p.value}: {e}")
                last_error = e
                if not fallback:
                    raise

        raise WeatherAPIException(
            f"All providers failed to fetch forecast. Last error: {last_error}"
        )

    def get_historical(
        self,
        location: GeoLocation,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[WeatherProvider] = None,
        fallback: bool = True,
    ) -> HistoricalWeather:
        """
        Get historical weather data with optional provider selection and fallback.

        Args:
            location: Geographic location
            start_date: Start of historical period
            end_date: End of historical period
            provider: Preferred provider (tries all if not specified)
            fallback: Whether to try other providers if preferred fails

        Returns:
            Historical weather data

        Raises:
            WeatherAPIException: If all providers fail
        """
        providers_to_try = [provider] if provider else list(self.clients.keys())

        last_error = None

        for p in providers_to_try:
            client = self.clients.get(p)
            if not client:
                continue

            try:
                logger.info(f"Fetching historical data from {p.value}")
                return client.get_historical(location, start_date, end_date)
            except (WeatherAPIException, NotImplementedError) as e:
                logger.warning(f"Failed to fetch from {p.value}: {e}")
                last_error = e
                if not fallback:
                    raise

        raise WeatherAPIException(
            f"All providers failed to fetch historical data. Last error: {last_error}"
        )

    def get_available_providers(self) -> list[WeatherProvider]:
        """
        Get list of available (initialized) providers.

        Returns:
            List of available weather providers
        """
        return list(self.clients.keys())

    def check_provider_availability(self, provider: WeatherProvider) -> bool:
        """
        Check if a specific provider is available and working.

        Args:
            provider: Provider to check

        Returns:
            True if provider is available, False otherwise
        """
        client = self.clients.get(provider)
        if not client:
            return False

        try:
            return client.check_availability()
        except Exception as e:
            logger.error(f"Availability check failed for {provider.value}: {e}")
            return False
