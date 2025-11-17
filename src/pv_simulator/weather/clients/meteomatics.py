"""
Meteomatics API client implementation.

This module provides integration with the Meteomatics API for
high-quality weather and climate data.

API Documentation: https://www.meteomatics.com/en/api/overview/
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from pv_simulator.config import Settings
from pv_simulator.models.weather import (
    CurrentWeather,
    ForecastWeather,
    GeoLocation,
    HistoricalWeather,
    WeatherDataPoint,
    WeatherProvider,
)
from pv_simulator.weather.cache import CacheManager
from pv_simulator.weather.clients.base import BaseWeatherClient

logger = logging.getLogger(__name__)


class MeteomaticsClient(BaseWeatherClient):
    """Meteomatics API client for professional weather data."""

    BASE_URL = "https://api.meteomatics.com"

    def __init__(self, settings: Settings, cache_manager: CacheManager) -> None:
        """
        Initialize Meteomatics client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        super().__init__(settings, cache_manager, WeatherProvider.METEOMATICS)
        self.username = settings.meteomatics_username
        self.password = settings.meteomatics_password

    def _get_auth(self) -> tuple[str, str]:
        """Get authentication credentials."""
        return (self.username, self.password)

    def _make_meteomatics_request(
        self, timestamp: str, parameters: list[str], location: GeoLocation
    ) -> dict[str, Any]:
        """
        Make authenticated request to Meteomatics API.

        Args:
            timestamp: ISO timestamp or time range
            parameters: List of weather parameters to fetch
            location: Geographic location

        Returns:
            Parsed response data
        """
        params_str = ",".join(parameters)
        url = f"{self.BASE_URL}/{timestamp}/{params_str}/{location.latitude},{location.longitude}/json"

        # Meteomatics uses HTTP Basic Auth
        import base64
        credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        return self._make_request(url, headers=headers)

    def _parse_meteomatics_data(
        self, data: dict[str, Any], location: GeoLocation, is_forecast: bool = False
    ) -> list[WeatherDataPoint]:
        """
        Parse Meteomatics response into standard format.

        Args:
            data: Raw API response
            location: Geographic location
            is_forecast: Whether this is forecast data

        Returns:
            List of weather data points
        """
        weather_points = []

        # Meteomatics returns time series for each parameter
        if "data" not in data:
            return weather_points

        # Group data by timestamp
        timestamp_data: dict[datetime, dict[str, float]] = {}

        for param_data in data["data"]:
            param_name = param_data.get("parameter", "")
            coordinates = param_data.get("coordinates", [{}])[0]

            for date_item in coordinates.get("dates", []):
                timestamp = datetime.fromisoformat(date_item["date"].replace("Z", "+00:00"))
                value = date_item["value"]

                if timestamp not in timestamp_data:
                    timestamp_data[timestamp] = {}

                timestamp_data[timestamp][param_name] = value

        # Create WeatherDataPoint for each timestamp
        for timestamp, values in sorted(timestamp_data.items()):
            weather_point = WeatherDataPoint(
                timestamp=timestamp,
                location=location,
                provider=self.provider,
                temperature=values.get("t_2m:C"),
                humidity=values.get("relative_humidity_2m:p"),
                pressure=values.get("msl_pressure:hPa"),
                wind_speed=values.get("wind_speed_10m:ms"),
                wind_direction=values.get("wind_dir_10m:d"),
                precipitation=values.get("precip_1h:mm"),
                ghi=values.get("global_rad:W"),
                dni=values.get("direct_rad:W"),
                dhi=values.get("diffuse_rad:W"),
                cloud_cover=values.get("total_cloud_cover:p"),
                is_forecast=is_forecast,
            )
            weather_points.append(weather_point)

        return weather_points

    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current weather conditions from Meteomatics.

        Args:
            location: Geographic location

        Returns:
            Current weather data

        Raises:
            WeatherAPIException: If request fails
        """
        cache_key = self._get_cache_key("current", location.latitude, location.longitude)

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return CurrentWeather(**cached)

        # Define parameters to fetch
        parameters = [
            "t_2m:C",  # Temperature at 2m
            "relative_humidity_2m:p",  # Relative humidity
            "msl_pressure:hPa",  # Mean sea level pressure
            "wind_speed_10m:ms",  # Wind speed at 10m
            "wind_dir_10m:d",  # Wind direction
            "precip_1h:mm",  # Precipitation
            "global_rad:W",  # Global radiation (GHI)
            "direct_rad:W",  # Direct radiation (DNI)
            "diffuse_rad:W",  # Diffuse radiation (DHI)
            "total_cloud_cover:p",  # Cloud cover
        ]

        timestamp = "now"

        logger.info(
            f"Fetching current weather from Meteomatics for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_meteomatics_request(timestamp, parameters, location)
        weather_points = self._parse_meteomatics_data(response, location)

        if not weather_points:
            raise ValueError("No current weather data received from Meteomatics")

        current = CurrentWeather(data=weather_points[0], cache_key=cache_key)

        # Cache the result
        self._save_to_cache(cache_key, current.model_dump(), ttl=600)

        return current

    def get_forecast(self, location: GeoLocation, days: int = 7) -> ForecastWeather:
        """
        Get weather forecast from Meteomatics.

        Args:
            location: Geographic location
            days: Number of days to forecast (default: 7)

        Returns:
            Weather forecast data

        Raises:
            WeatherAPIException: If request fails
        """
        cache_key = self._get_cache_key("forecast", location.latitude, location.longitude, days)

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return ForecastWeather(**cached)

        # Define parameters
        parameters = [
            "t_2m:C",
            "relative_humidity_2m:p",
            "msl_pressure:hPa",
            "wind_speed_10m:ms",
            "wind_dir_10m:d",
            "precip_1h:mm",
            "global_rad:W",
            "direct_rad:W",
            "diffuse_rad:W",
            "total_cloud_cover:p",
        ]

        # Create time range for forecast
        now = datetime.utcnow()
        end_time = now + timedelta(days=days)
        timestamp = f"{now.isoformat()}--{end_time.isoformat()}:PT1H"  # Hourly data

        logger.info(
            f"Fetching {days}-day forecast from Meteomatics for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_meteomatics_request(timestamp, parameters, location)
        forecast_points = self._parse_meteomatics_data(response, location, is_forecast=True)

        if not forecast_points:
            raise ValueError("No forecast data received from Meteomatics")

        forecast = ForecastWeather(
            location=location,
            provider=self.provider,
            forecast_data=forecast_points,
            forecast_start=forecast_points[0].timestamp,
            forecast_end=forecast_points[-1].timestamp,
        )

        # Cache the result
        self._save_to_cache(cache_key, forecast.model_dump(), ttl=3600)

        return forecast

    def get_historical(
        self,
        location: GeoLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalWeather:
        """
        Get historical weather data from Meteomatics.

        Args:
            location: Geographic location
            start_date: Start of historical period
            end_date: End of historical period

        Returns:
            Historical weather data

        Raises:
            WeatherAPIException: If request fails
        """
        cache_key = self._get_cache_key(
            "historical",
            location.latitude,
            location.longitude,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return HistoricalWeather(**cached)

        # Define parameters
        parameters = [
            "t_2m:C",
            "relative_humidity_2m:p",
            "msl_pressure:hPa",
            "wind_speed_10m:ms",
            "wind_dir_10m:d",
            "precip_1h:mm",
            "global_rad:W",
            "direct_rad:W",
            "diffuse_rad:W",
            "total_cloud_cover:p",
        ]

        # Create time range
        timestamp = f"{start_date.isoformat()}--{end_date.isoformat()}:PT1H"

        logger.info(
            f"Fetching historical data from Meteomatics for "
            f"({location.latitude}, {location.longitude}) "
            f"from {start_date} to {end_date}"
        )

        response = self._make_meteomatics_request(timestamp, parameters, location)
        historical_points = self._parse_meteomatics_data(response, location)

        if not historical_points:
            raise ValueError("No historical data received from Meteomatics")

        historical = HistoricalWeather(
            location=location,
            provider=self.provider,
            historical_data=historical_points,
            period_start=start_date,
            period_end=end_date,
        )

        # Cache the result
        self._save_to_cache(cache_key, historical.model_dump(), ttl=86400)

        return historical
