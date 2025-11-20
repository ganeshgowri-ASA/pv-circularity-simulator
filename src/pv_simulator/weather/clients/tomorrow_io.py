"""
Tomorrow.io API client implementation.

This module provides integration with the Tomorrow.io (formerly ClimaCell) API
for advanced weather data and forecasting.

API Documentation: https://docs.tomorrow.io/reference/welcome
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


class TomorrowIOClient(BaseWeatherClient):
    """Tomorrow.io API client for advanced weather data."""

    BASE_URL = "https://api.tomorrow.io/v4"

    def __init__(self, settings: Settings, cache_manager: CacheManager) -> None:
        """
        Initialize Tomorrow.io client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        super().__init__(settings, cache_manager, WeatherProvider.TOMORROW_IO)
        self.api_key = settings.get_api_key("tomorrow_io")

    def _parse_weather_data(
        self,
        values: dict[str, Any],
        location: GeoLocation,
        timestamp: datetime,
        is_forecast: bool = False,
    ) -> WeatherDataPoint:
        """
        Parse Tomorrow.io weather data into standard format.

        Args:
            values: Weather data values from API
            location: Geographic location
            timestamp: Timestamp of the data
            is_forecast: Whether this is forecast data

        Returns:
            Standardized weather data point
        """
        return WeatherDataPoint(
            timestamp=timestamp,
            location=location,
            provider=self.provider,
            temperature=values.get("temperature"),
            feels_like=values.get("temperatureApparent"),
            dew_point=values.get("dewPoint"),
            humidity=values.get("humidity"),
            pressure=values.get("pressureSurfaceLevel"),
            pressure_sea_level=values.get("pressureSeaLevel"),
            wind_speed=values.get("windSpeed"),
            wind_gust=values.get("windGust"),
            wind_direction=values.get("windDirection"),
            cloud_cover=values.get("cloudCover"),
            visibility=values.get("visibility"),
            precipitation=values.get("precipitationIntensity"),
            precipitation_probability=values.get("precipitationProbability"),
            ghi=values.get("solarGHI"),  # Global Horizontal Irradiance
            dni=values.get("solarDNI"),  # Direct Normal Irradiance
            uv_index=values.get("uvIndex"),
            condition=values.get("weatherCode"),  # Tomorrow.io uses weather codes
            is_forecast=is_forecast,
        )

    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current weather conditions from Tomorrow.io.

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

        # Define fields to fetch
        fields = [
            "temperature",
            "temperatureApparent",
            "dewPoint",
            "humidity",
            "windSpeed",
            "windGust",
            "windDirection",
            "pressureSurfaceLevel",
            "pressureSeaLevel",
            "cloudCover",
            "visibility",
            "precipitationIntensity",
            "solarGHI",
            "solarDNI",
            "uvIndex",
            "weatherCode",
        ]

        url = f"{self.BASE_URL}/weather/realtime"
        params = {
            "location": f"{location.latitude},{location.longitude}",
            "apikey": self.api_key,
            "units": "metric",
        }

        logger.info(
            f"Fetching current weather from Tomorrow.io for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse response
        data = response.get("data", {})
        values = data.get("values", {})
        time_str = data.get("time", "")

        timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        weather_data = self._parse_weather_data(values, location, timestamp)

        current = CurrentWeather(data=weather_data, cache_key=cache_key)

        # Cache the result
        self._save_to_cache(cache_key, current.model_dump(), ttl=600)

        return current

    def get_forecast(self, location: GeoLocation, days: int = 7) -> ForecastWeather:
        """
        Get weather forecast from Tomorrow.io.

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

        # Define fields
        fields = [
            "temperature",
            "temperatureApparent",
            "dewPoint",
            "humidity",
            "windSpeed",
            "windGust",
            "windDirection",
            "pressureSurfaceLevel",
            "pressureSeaLevel",
            "cloudCover",
            "visibility",
            "precipitationIntensity",
            "precipitationProbability",
            "solarGHI",
            "solarDNI",
            "uvIndex",
            "weatherCode",
        ]

        url = f"{self.BASE_URL}/weather/forecast"
        params = {
            "location": f"{location.latitude},{location.longitude}",
            "apikey": self.api_key,
            "units": "metric",
            "timesteps": "1h",  # Hourly forecast
        }

        logger.info(
            f"Fetching {days}-day forecast from Tomorrow.io for "
            f"({location.latitude}, {location.longitude})"
        )

        response = self._make_request(url, params)

        # Parse forecast data
        timelines = response.get("timelines", {})
        hourly = timelines.get("hourly", [])

        forecast_points = []
        end_time = datetime.utcnow() + timedelta(days=days)

        for interval in hourly:
            time_str = interval.get("time", "")
            timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

            if timestamp > end_time:
                break

            values = interval.get("values", {})
            weather_data = self._parse_weather_data(
                values, location, timestamp, is_forecast=True
            )
            forecast_points.append(weather_data)

        if not forecast_points:
            raise ValueError("No forecast data received from Tomorrow.io")

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
        Get historical weather data from Tomorrow.io.

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

        # Define fields
        fields = [
            "temperature",
            "temperatureApparent",
            "dewPoint",
            "humidity",
            "windSpeed",
            "windGust",
            "windDirection",
            "pressureSurfaceLevel",
            "pressureSeaLevel",
            "cloudCover",
            "visibility",
            "precipitationIntensity",
            "solarGHI",
            "solarDNI",
            "uvIndex",
            "weatherCode",
        ]

        url = f"{self.BASE_URL}/timelines"
        params = {
            "location": f"{location.latitude},{location.longitude}",
            "apikey": self.api_key,
            "units": "metric",
            "timesteps": "1h",
            "startTime": start_date.isoformat(),
            "endTime": end_date.isoformat(),
        }

        logger.info(
            f"Fetching historical data from Tomorrow.io for "
            f"({location.latitude}, {location.longitude}) "
            f"from {start_date} to {end_date}"
        )

        response = self._make_request(url, params)

        # Parse historical data
        timelines = response.get("data", {}).get("timelines", [])
        historical_points = []

        for timeline in timelines:
            for interval in timeline.get("intervals", []):
                time_str = interval.get("startTime", "")
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                values = interval.get("values", {})
                weather_data = self._parse_weather_data(values, location, timestamp)
                historical_points.append(weather_data)

        if not historical_points:
            raise ValueError("No historical data received from Tomorrow.io")

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
