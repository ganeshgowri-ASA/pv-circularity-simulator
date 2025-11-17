"""
Visual Crossing Weather API client implementation.

This module provides integration with the Visual Crossing Weather API for
comprehensive weather data including forecasts and historical data.

API Documentation: https://www.visualcrossing.com/resources/documentation/weather-api/
"""

import logging
from datetime import datetime
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


class VisualCrossingClient(BaseWeatherClient):
    """Visual Crossing Weather API client."""

    BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def __init__(self, settings: Settings, cache_manager: CacheManager) -> None:
        """
        Initialize Visual Crossing client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        super().__init__(settings, cache_manager, WeatherProvider.VISUALCROSSING)
        self.api_key = settings.get_api_key("visualcrossing")

    def _parse_weather_data(
        self,
        data: dict[str, Any],
        location: GeoLocation,
        is_forecast: bool = False,
    ) -> WeatherDataPoint:
        """
        Parse Visual Crossing weather data into standard format.

        Args:
            data: Raw API response data
            location: Geographic location
            is_forecast: Whether this is forecast data

        Returns:
            Standardized weather data point
        """
        # Parse timestamp
        datetime_str = data.get("datetime", "")
        datetime_epoch = data.get("datetimeEpoch")

        if datetime_epoch:
            timestamp = datetime.utcfromtimestamp(datetime_epoch)
        else:
            timestamp = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

        return WeatherDataPoint(
            timestamp=timestamp,
            location=location,
            provider=self.provider,
            temperature=data.get("temp"),
            feels_like=data.get("feelslike"),
            dew_point=data.get("dew"),
            humidity=data.get("humidity"),
            pressure=data.get("pressure"),
            wind_speed=data.get("windspeed"),
            wind_gust=data.get("windgust"),
            wind_direction=data.get("winddir"),
            cloud_cover=data.get("cloudcover"),
            visibility=data.get("visibility"),
            precipitation=data.get("precip"),
            precipitation_probability=data.get("precipprob"),
            snow=data.get("snow"),
            ghi=data.get("solarradiation"),  # Visual Crossing provides solar radiation
            uv_index=data.get("uvindex"),
            condition=data.get("conditions"),
            is_forecast=is_forecast,
        )

    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current weather conditions from Visual Crossing.

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
        url = f"{self.BASE_URL}/{location.latitude},{location.longitude}/today"
        params = {
            "key": self.api_key,
            "unitGroup": "metric",
            "include": "current",
        }

        logger.info(
            f"Fetching current weather from Visual Crossing for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse current conditions
        current_conditions = response.get("currentConditions", {})
        weather_data = self._parse_weather_data(current_conditions, location)

        current = CurrentWeather(data=weather_data, cache_key=cache_key)

        # Cache the result
        self._save_to_cache(cache_key, current.model_dump(), ttl=600)

        return current

    def get_forecast(self, location: GeoLocation, days: int = 7) -> ForecastWeather:
        """
        Get weather forecast from Visual Crossing.

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

        # Make API request
        url = f"{self.BASE_URL}/{location.latitude},{location.longitude}/next{days}days"
        params = {
            "key": self.api_key,
            "unitGroup": "metric",
            "include": "hours",
        }

        logger.info(
            f"Fetching {days}-day forecast from Visual Crossing for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse forecast data
        forecast_points = []
        for day in response.get("days", []):
            for hour in day.get("hours", []):
                weather_data = self._parse_weather_data(hour, location, is_forecast=True)
                forecast_points.append(weather_data)

        if not forecast_points:
            raise ValueError("No forecast data received from Visual Crossing")

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
        Get historical weather data from Visual Crossing.

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

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Make API request
        url = f"{self.BASE_URL}/{location.latitude},{location.longitude}/{start_str}/{end_str}"
        params = {
            "key": self.api_key,
            "unitGroup": "metric",
            "include": "hours",
        }

        logger.info(
            f"Fetching historical data from Visual Crossing for "
            f"({location.latitude}, {location.longitude}) "
            f"from {start_str} to {end_str}"
        )

        response = self._make_request(url, params)

        # Parse historical data
        historical_points = []
        for day in response.get("days", []):
            for hour in day.get("hours", []):
                weather_data = self._parse_weather_data(hour, location)
                historical_points.append(weather_data)

        if not historical_points:
            raise ValueError("No historical data received from Visual Crossing")

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
