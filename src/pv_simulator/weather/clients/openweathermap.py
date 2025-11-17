"""
OpenWeatherMap API client implementation.

This module provides integration with the OpenWeatherMap API for current weather,
forecasts, and historical data.

API Documentation: https://openweathermap.org/api
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


class OpenWeatherMapClient(BaseWeatherClient):
    """OpenWeatherMap API client for weather data."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"
    ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

    def __init__(self, settings: Settings, cache_manager: CacheManager) -> None:
        """
        Initialize OpenWeatherMap client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        super().__init__(settings, cache_manager, WeatherProvider.OPENWEATHERMAP)
        self.api_key = settings.get_api_key("openweathermap")

    def _parse_weather_data(
        self,
        data: dict[str, Any],
        location: GeoLocation,
        timestamp: datetime,
        is_forecast: bool = False,
    ) -> WeatherDataPoint:
        """
        Parse OpenWeatherMap weather data into standard format.

        Args:
            data: Raw API response data
            location: Geographic location
            timestamp: Timestamp of the data
            is_forecast: Whether this is forecast data

        Returns:
            Standardized weather data point
        """
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather = data.get("weather", [{}])[0]
        rain = data.get("rain", {})
        snow = data.get("snow", {})

        return WeatherDataPoint(
            timestamp=timestamp,
            location=location,
            provider=self.provider,
            temperature=main.get("temp"),
            feels_like=main.get("feels_like"),
            humidity=main.get("humidity"),
            pressure=main.get("pressure"),
            pressure_sea_level=main.get("sea_level"),
            wind_speed=wind.get("speed"),
            wind_gust=wind.get("gust"),
            wind_direction=wind.get("deg"),
            cloud_cover=clouds.get("all"),
            visibility=data.get("visibility"),
            rain=rain.get("1h", rain.get("3h", 0)) if rain else None,
            snow=snow.get("1h", snow.get("3h", 0)) if snow else None,
            condition=weather.get("description"),
            condition_code=weather.get("id"),
            is_forecast=is_forecast,
        )

    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current weather conditions from OpenWeatherMap.

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

        # Make API request
        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "appid": self.api_key,
            "units": "metric",
        }

        logger.info(
            f"Fetching current weather from OpenWeatherMap for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse response
        timestamp = datetime.utcfromtimestamp(response.get("dt", 0))
        weather_data = self._parse_weather_data(response, location, timestamp)

        current = CurrentWeather(data=weather_data, cache_key=cache_key)

        # Cache the result
        self._save_to_cache(cache_key, current.model_dump(), ttl=600)  # 10 minutes

        return current

    def get_forecast(
        self, location: GeoLocation, days: int = 7
    ) -> ForecastWeather:
        """
        Get weather forecast from OpenWeatherMap.

        Args:
            location: Geographic location
            days: Number of days to forecast (max 5 for free tier)

        Returns:
            Weather forecast data

        Raises:
            WeatherAPIException: If request fails
        """
        # OpenWeatherMap free tier only supports 5 days
        days = min(days, 5)

        cache_key = self._get_cache_key("forecast", location.latitude, location.longitude, days)

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return ForecastWeather(**cached)

        # Make API request
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "appid": self.api_key,
            "units": "metric",
            "cnt": days * 8,  # 8 forecasts per day (3-hour intervals)
        }

        logger.info(
            f"Fetching {days}-day forecast from OpenWeatherMap for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse forecast data
        forecast_list = response.get("list", [])
        forecast_points = []

        for item in forecast_list:
            timestamp = datetime.utcfromtimestamp(item.get("dt", 0))
            weather_data = self._parse_weather_data(
                item, location, timestamp, is_forecast=True
            )
            forecast_points.append(weather_data)

        if not forecast_points:
            raise ValueError("No forecast data received from OpenWeatherMap")

        forecast = ForecastWeather(
            location=location,
            provider=self.provider,
            forecast_data=forecast_points,
            forecast_start=forecast_points[0].timestamp,
            forecast_end=forecast_points[-1].timestamp,
        )

        # Cache the result
        self._save_to_cache(cache_key, forecast.model_dump(), ttl=3600)  # 1 hour

        return forecast

    def get_historical(
        self,
        location: GeoLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalWeather:
        """
        Get historical weather data from OpenWeatherMap.

        Note: Historical data requires a paid subscription.
        This implementation provides a basic structure.

        Args:
            location: Geographic location
            start_date: Start of historical period
            end_date: End of historical period

        Returns:
            Historical weather data

        Raises:
            WeatherAPIException: If request fails
            NotImplementedError: If historical API is not available
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

        logger.warning(
            "OpenWeatherMap historical data requires a paid subscription. "
            "Using timemachine endpoint for limited historical access."
        )

        # Collect data points for each day
        historical_points = []
        current_date = start_date

        while current_date <= end_date:
            timestamp = int(current_date.timestamp())

            url = f"{self.ONECALL_URL}/timemachine"
            params = {
                "lat": location.latitude,
                "lon": location.longitude,
                "dt": timestamp,
                "appid": self.api_key,
                "units": "metric",
            }

            try:
                response = self._make_request(url, params)
                data = response.get("data", [{}])[0]

                if data:
                    weather_data = self._parse_weather_data(
                        data, location, current_date
                    )
                    historical_points.append(weather_data)

            except Exception as e:
                logger.warning(
                    f"Failed to fetch historical data for {current_date}: {e}"
                )

            current_date += timedelta(days=1)

        if not historical_points:
            raise ValueError("No historical data received from OpenWeatherMap")

        historical = HistoricalWeather(
            location=location,
            provider=self.provider,
            historical_data=historical_points,
            period_start=start_date,
            period_end=end_date,
        )

        # Cache the result
        self._save_to_cache(cache_key, historical.model_dump(), ttl=86400)  # 24 hours

        return historical
