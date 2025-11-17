"""
NREL NSRDB API client for solar radiation and meteorological data.

Provides access to the National Solar Radiation Database (NSRDB)
maintained by the National Renewable Energy Laboratory (NREL).
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pv_simulator.api.base_client import BaseAPIClient
from pv_simulator.config.settings import settings
from pv_simulator.models.weather import (
    DataSource,
    GlobalLocation,
    TMYData,
    TMYFormat,
    TemporalResolution,
    WeatherDataPoint,
)

logger = logging.getLogger(__name__)


class NSRDBClient(BaseAPIClient):
    """
    Client for NREL NSRDB API.

    The NSRDB provides high-quality solar radiation and meteorological data
    for the United States and surrounding areas.

    Attributes:
        api_key: NREL API key (use 'DEMO_KEY' for testing)
    """

    # Available NSRDB attributes
    AVAILABLE_ATTRIBUTES = [
        "ghi",
        "dni",
        "dhi",
        "air_temperature",
        "wind_speed",
        "wind_direction",
        "relative_humidity",
        "surface_pressure",
        "surface_albedo",
        "precipitable_water",
        "cloud_type",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize NSRDB client.

        Args:
            api_key: NREL API key (default: from settings)
        """
        api_key = api_key or settings.nsrdb_api_key
        super().__init__(base_url=settings.nsrdb_api_url, api_key=api_key)

    def get_data(
        self,
        latitude: float,
        longitude: float,
        year: Optional[int] = None,
        attributes: Optional[List[str]] = None,
        interval: int = 60,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetch NSRDB data for a specific location and year.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            year: Year of data (if None, uses TMY data)
            attributes: List of attributes to fetch (if None, fetches all)
            interval: Time interval in minutes (30 or 60)
            **kwargs: Additional API parameters

        Returns:
            Dictionary containing NSRDB data

        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.RequestException: On API request failure
        """
        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude: {latitude}")
        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude: {longitude}")

        # Use all attributes if not specified
        if attributes is None:
            attributes = self.AVAILABLE_ATTRIBUTES

        # Build API parameters
        params = {
            "api_key": self.api_key,
            "lat": latitude,
            "lon": longitude,
            "attributes": ",".join(attributes),
            "interval": interval,
            "utc": "true",
            "names": "tmy",  # Use TMY naming convention
        }

        if year is not None:
            params["year"] = year

        # Add any additional parameters
        params.update(kwargs)

        logger.info(f"Fetching NSRDB data for ({latitude}, {longitude}), year={year}")

        # Make API request
        response = self.get("", params=params)

        # Parse CSV response
        data = self._parse_csv_response(response.text)

        return data

    def get_tmy_data(
        self,
        latitude: float,
        longitude: float,
        attributes: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> TMYData:
        """
        Fetch TMY data from NSRDB.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            attributes: List of attributes to fetch
            cache_dir: Directory to cache downloaded files

        Returns:
            TMY data object

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(f"Fetching TMY data for ({latitude}, {longitude})")

        # Use TMY endpoint
        params = {
            "api_key": self.api_key,
            "lat": latitude,
            "lon": longitude,
            "attributes": ",".join(attributes or self.AVAILABLE_ATTRIBUTES),
            "interval": 60,
            "utc": "true",
            "names": "tmy",
        }

        # Use TMY-specific endpoint
        base_url = self.base_url
        self.base_url = settings.nsrdb_tmy_api_url

        try:
            response = self.get("", params=params)
            csv_data = response.text

            # Parse CSV into TMY data
            tmy_data = self._parse_tmy_csv(csv_data, latitude, longitude)

            # Cache if directory specified
            if cache_dir:
                cache_path = cache_dir / f"nsrdb_tmy_{latitude}_{longitude}.csv"
                cache_path.write_text(csv_data)
                logger.info(f"Cached TMY data to {cache_path}")

            return tmy_data

        finally:
            # Restore base URL
            self.base_url = base_url

    def get_historical_data(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
        attributes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch multi-year historical data from NSRDB.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_year: Start year
            end_year: End year (inclusive)
            attributes: List of attributes to fetch

        Returns:
            List of yearly data dictionaries

        Raises:
            ValueError: If year range is invalid
        """
        if start_year > end_year:
            raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")

        logger.info(f"Fetching historical data for {start_year}-{end_year}")

        historical_data = []
        for year in range(start_year, end_year + 1):
            try:
                data = self.get_data(
                    latitude=latitude,
                    longitude=longitude,
                    year=year,
                    attributes=attributes,
                )
                data["year"] = year
                historical_data.append(data)
                logger.info(f"Fetched data for year {year}")

            except Exception as e:
                logger.error(f"Failed to fetch data for year {year}: {e}")
                continue

        return historical_data

    def _parse_csv_response(self, csv_text: str) -> Dict[str, Any]:
        """
        Parse CSV response from NSRDB API.

        Args:
            csv_text: CSV response text

        Returns:
            Dictionary with parsed data
        """
        # Read CSV with pandas
        df = pd.read_csv(io.StringIO(csv_text), skiprows=2)

        return {
            "data": df,
            "num_records": len(df),
            "columns": list(df.columns),
        }

    def _parse_tmy_csv(self, csv_text: str, latitude: float, longitude: float) -> TMYData:
        """
        Parse TMY CSV data into TMYData object.

        Args:
            csv_text: CSV text from NSRDB
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            TMY data object
        """
        # Parse header
        lines = csv_text.split("\n")
        header_parts = lines[0].split(",")

        # Extract location info
        location = GlobalLocation(
            name=header_parts[1] if len(header_parts) > 1 else "Unknown",
            country="USA",  # NSRDB is US-focused
            latitude=latitude,
            longitude=longitude,
            elevation=float(header_parts[6]) if len(header_parts) > 6 else 0.0,
            timezone=header_parts[7] if len(header_parts) > 7 else "UTC",
        )

        # Read data
        df = pd.read_csv(io.StringIO(csv_text), skiprows=2)

        # Parse weather data points
        hourly_data = []
        for _, row in df.iterrows():
            try:
                # Parse timestamp
                year = int(row.get("Year", 2000))
                month = int(row.get("Month", 1))
                day = int(row.get("Day", 1))
                hour = int(row.get("Hour", 0))
                minute = int(row.get("Minute", 0))

                from datetime import datetime

                timestamp = datetime(year, month, day, hour, minute)

                # Create weather data point
                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(row.get("Temperature", 20)),
                    irradiance_ghi=float(row.get("GHI", 0)),
                    irradiance_dni=float(row.get("DNI", 0)),
                    irradiance_dhi=float(row.get("DHI", 0)),
                    wind_speed=float(row.get("Wind Speed", 0)),
                    wind_direction=float(row.get("Wind Direction", 0)),
                    relative_humidity=float(row.get("Relative Humidity", 50)),
                    pressure=float(row.get("Pressure", 101325)),
                    albedo=float(row.get("Surface Albedo", 0.2)),
                    precipitable_water=float(row.get("Precipitable Water", 0)),
                )
                hourly_data.append(point)

            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        # Create TMY data
        tmy_data = TMYData(
            location=location,
            data_source=DataSource.NSRDB,
            format_type=TMYFormat.TMY3,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=2007,
            end_year=2021,
            hourly_data=hourly_data,
        )

        return tmy_data
