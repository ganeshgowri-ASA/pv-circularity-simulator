"""
NREL Physical Solar Model (PSM) API client implementation.

This module provides integration with the NREL PSM API for high-quality
solar irradiance data critical for PV system simulation.

API Documentation: https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

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


class NRELPSMClient(BaseWeatherClient):
    """NREL Physical Solar Model API client for solar irradiance data."""

    BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar"
    PSM_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

    def __init__(self, settings: Settings, cache_manager: CacheManager) -> None:
        """
        Initialize NREL PSM client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
        """
        super().__init__(settings, cache_manager, WeatherProvider.NREL_PSM)
        self.api_key = settings.get_api_key("nrel")

    def _parse_psm_data(
        self, df: pd.DataFrame, location: GeoLocation, is_forecast: bool = False
    ) -> list[WeatherDataPoint]:
        """
        Parse NREL PSM CSV data into standard format.

        Args:
            df: Pandas DataFrame with PSM data
            location: Geographic location
            is_forecast: Whether this is forecast data

        Returns:
            List of weather data points
        """
        weather_points = []

        for _, row in df.iterrows():
            try:
                # Combine date columns into timestamp
                year = int(row.get("Year", 0))
                month = int(row.get("Month", 0))
                day = int(row.get("Day", 0))
                hour = int(row.get("Hour", 0))
                minute = int(row.get("Minute", 0))

                timestamp = datetime(year, month, day, hour, minute)

                weather_point = WeatherDataPoint(
                    timestamp=timestamp,
                    location=location,
                    provider=self.provider,
                    # Solar irradiance data (primary focus of NREL PSM)
                    ghi=row.get("GHI"),  # Global Horizontal Irradiance
                    dni=row.get("DNI"),  # Direct Normal Irradiance
                    dhi=row.get("DHI"),  # Diffuse Horizontal Irradiance
                    # Solar position
                    solar_elevation=row.get("Solar Zenith Angle"),
                    solar_azimuth=row.get("Solar Azimuth Angle"),
                    # Meteorological data
                    temperature=row.get("Temperature"),
                    dew_point=row.get("Dew Point"),
                    pressure=row.get("Pressure"),
                    wind_speed=row.get("Wind Speed"),
                    wind_direction=row.get("Wind Direction"),
                    humidity=row.get("Relative Humidity"),
                    precipitation=row.get("Precipitable Water"),
                    cloud_cover=row.get("Cloud Type"),
                    is_forecast=is_forecast,
                )
                weather_points.append(weather_point)

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse PSM data row: {e}")
                continue

        return weather_points

    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current solar irradiance data from NREL PSM.

        Note: NREL PSM provides historical data, not real-time.
        This method returns the most recent available data.

        Args:
            location: Geographic location

        Returns:
            Current weather data (most recent available)

        Raises:
            WeatherAPIException: If request fails
        """
        cache_key = self._get_cache_key("current", location.latitude, location.longitude)

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return CurrentWeather(**cached)

        logger.info(
            "NREL PSM provides historical data. Fetching most recent available data."
        )

        # Get data for the current year
        current_year = datetime.utcnow().year
        start_date = datetime(current_year, 1, 1)
        end_date = datetime.utcnow()

        historical = self.get_historical(location, start_date, end_date)

        # Get the most recent data point
        if not historical.historical_data:
            raise ValueError("No current data available from NREL PSM")

        latest_data = historical.historical_data[-1]
        current = CurrentWeather(data=latest_data, cache_key=cache_key)

        # Cache for shorter time since it's "current"
        self._save_to_cache(cache_key, current.model_dump(), ttl=1800)  # 30 minutes

        return current

    def get_forecast(self, location: GeoLocation, days: int = 7) -> ForecastWeather:
        """
        Get weather forecast from NREL PSM.

        Note: NREL PSM does not provide forecast data.
        This method raises NotImplementedError.

        Args:
            location: Geographic location
            days: Number of days to forecast

        Raises:
            NotImplementedError: NREL PSM does not provide forecasts
        """
        raise NotImplementedError(
            "NREL PSM provides historical solar irradiance data, not forecasts. "
            "Use other providers (OpenWeatherMap, Tomorrow.io) for forecast data."
        )

    def get_historical(
        self,
        location: GeoLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalWeather:
        """
        Get historical solar irradiance data from NREL PSM.

        Args:
            location: Geographic location
            start_date: Start of historical period
            end_date: End of historical period

        Returns:
            Historical weather data with solar irradiance

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

        # NREL PSM parameters
        params = {
            "api_key": self.api_key,
            "wkt": f"POINT({location.longitude} {location.latitude})",
            "names": start_date.year,  # Year of data
            "attributes": ",".join([
                "ghi",
                "dni",
                "dhi",
                "solar_zenith_angle",
                "solar_azimuth_angle",
                "air_temperature",
                "dew_point",
                "surface_pressure",
                "wind_speed",
                "wind_direction",
                "relative_humidity",
                "precipitable_water",
                "cloud_type",
            ]),
            "leap_day": "true",
            "utc": "true",
            "interval": "60",  # 60-minute intervals
            "email": "user@example.com",  # Required by API
        }

        logger.info(
            f"Fetching historical solar data from NREL PSM for "
            f"({location.latitude}, {location.longitude}) "
            f"from {start_date} to {end_date}"
        )

        try:
            # Make request and parse CSV
            self._check_rate_limit()

            # NREL returns CSV data
            response = self.client.get(self.PSM_URL, params=params)
            response.raise_for_status()

            # Parse CSV data
            # Skip first two rows (metadata)
            import io
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data, skiprows=2)

            # Filter by date range
            df["Timestamp"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]]
            )
            df = df[
                (df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)
            ]

            # Parse data
            historical_points = self._parse_psm_data(df, location)

            if not historical_points:
                raise ValueError("No historical data received from NREL PSM")

            historical = HistoricalWeather(
                location=location,
                provider=self.provider,
                historical_data=historical_points,
                period_start=start_date,
                period_end=end_date,
            )

            # Cache the result (longer TTL for historical data)
            self._save_to_cache(cache_key, historical.model_dump(), ttl=604800)  # 7 days

            return historical

        except Exception as e:
            logger.error(f"Failed to fetch NREL PSM data: {e}")
            raise
