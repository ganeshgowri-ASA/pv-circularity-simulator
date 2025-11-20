"""
PVGIS API client for European solar radiation data.

Provides access to the Photovoltaic Geographical Information System (PVGIS)
maintained by the European Commission's Joint Research Centre.
"""

import json
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


class PVGISClient(BaseAPIClient):
    """
    Client for PVGIS API.

    PVGIS provides solar radiation and meteorological data for Europe,
    Africa, and parts of Asia with high spatial resolution.

    No API key required for basic usage.
    """

    # Available PVGIS databases
    DATABASES = {
        "PVGIS-SARAH2": "Europe, Africa, Asia (satellite-based)",
        "PVGIS-NSRDB": "Americas",
        "PVGIS-ERA5": "Global (reanalysis)",
        "PVGIS-CMSAF": "Europe, Africa (satellite-based)",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize PVGIS client.

        Args:
            api_key: API key (optional, not required for basic usage)
        """
        super().__init__(base_url=settings.pvgis_api_url, api_key=api_key)

    def get_data(
        self,
        latitude: float,
        longitude: float,
        database: str = "PVGIS-SARAH2",
        output_format: str = "json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetch PVGIS data for a specific location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            database: PVGIS database to use
            output_format: Output format (json, csv)
            **kwargs: Additional API parameters

        Returns:
            Dictionary containing PVGIS data

        Raises:
            ValueError: If parameters are invalid
        """
        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude: {latitude}")
        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude: {longitude}")

        # Build API parameters
        params = {
            "lat": latitude,
            "lon": longitude,
            "outputformat": output_format,
            "raddatabase": database,
        }

        # Add any additional parameters
        params.update(kwargs)

        logger.info(f"Fetching PVGIS data for ({latitude}, {longitude}), database={database}")

        # Make API request to hourly radiation endpoint
        response = self.get("seriescalc", params=params)

        # Parse response based on format
        if output_format == "json":
            data = response.json()
        else:
            data = {"text": response.text}

        return data

    def get_tmy_data(
        self,
        latitude: float,
        longitude: float,
        database: str = "PVGIS-SARAH2",
        cache_dir: Optional[Path] = None,
    ) -> TMYData:
        """
        Fetch TMY data from PVGIS.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            database: PVGIS database to use
            cache_dir: Directory to cache downloaded files

        Returns:
            TMY data object
        """
        logger.info(f"Fetching PVGIS TMY data for ({latitude}, {longitude})")

        # Use TMY endpoint
        params = {
            "lat": latitude,
            "lon": longitude,
            "outputformat": "json",
            "raddatabase": database,
        }

        response = self.get("tmy", params=params)
        json_data = response.json()

        # Parse TMY data
        tmy_data = self._parse_pvgis_tmy(json_data, latitude, longitude)

        # Cache if directory specified
        if cache_dir:
            cache_path = cache_dir / f"pvgis_tmy_{latitude}_{longitude}.json"
            cache_path.write_text(json.dumps(json_data, indent=2))
            logger.info(f"Cached TMY data to {cache_path}")

        return tmy_data

    def get_hourly_radiation(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
        database: str = "PVGIS-SARAH2",
    ) -> pd.DataFrame:
        """
        Fetch hourly radiation data for a date range.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_year: Start year
            end_year: End year
            database: PVGIS database to use

        Returns:
            DataFrame with hourly radiation data
        """
        logger.info(f"Fetching hourly radiation for {start_year}-{end_year}")

        params = {
            "lat": latitude,
            "lon": longitude,
            "startyear": start_year,
            "endyear": end_year,
            "outputformat": "json",
            "raddatabase": database,
        }

        response = self.get("seriescalc", params=params)
        data = response.json()

        # Convert to DataFrame
        if "outputs" in data and "hourly" in data["outputs"]:
            df = pd.DataFrame(data["outputs"]["hourly"])
            return df
        else:
            logger.warning("No hourly data in response")
            return pd.DataFrame()

    def get_monthly_radiation(
        self, latitude: float, longitude: float, database: str = "PVGIS-SARAH2"
    ) -> Dict[str, Any]:
        """
        Fetch monthly radiation statistics.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            database: PVGIS database to use

        Returns:
            Dictionary with monthly radiation data
        """
        logger.info(f"Fetching monthly radiation for ({latitude}, {longitude})")

        params = {
            "lat": latitude,
            "lon": longitude,
            "outputformat": "json",
            "raddatabase": database,
        }

        response = self.get("MRcalc", params=params)
        data = response.json()

        return data

    def get_daily_radiation(
        self,
        latitude: float,
        longitude: float,
        month: int,
        day: int,
        database: str = "PVGIS-SARAH2",
    ) -> Dict[str, Any]:
        """
        Fetch daily radiation profile for a specific day.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            month: Month (1-12)
            day: Day of month
            database: PVGIS database to use

        Returns:
            Dictionary with daily radiation profile
        """
        logger.info(f"Fetching daily radiation for {month}/{day}")

        params = {
            "lat": latitude,
            "lon": longitude,
            "month": month,
            "day": day,
            "outputformat": "json",
            "raddatabase": database,
        }

        response = self.get("DRcalc", params=params)
        data = response.json()

        return data

    def _parse_pvgis_tmy(
        self, json_data: Dict[str, Any], latitude: float, longitude: float
    ) -> TMYData:
        """
        Parse PVGIS TMY JSON data into TMYData object.

        Args:
            json_data: JSON response from PVGIS
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            TMY data object
        """
        from datetime import datetime

        # Extract metadata
        metadata = json_data.get("inputs", {})
        location_meta = json_data.get("location", metadata.get("location", {}))

        # Create location
        location = GlobalLocation(
            name=f"{latitude:.2f}N, {longitude:.2f}E",
            country=location_meta.get("country", "Unknown"),
            latitude=latitude,
            longitude=longitude,
            elevation=float(location_meta.get("elevation", 0)),
            timezone="UTC",  # PVGIS uses UTC
        )

        # Parse hourly data
        hourly_data = []
        tmy_records = json_data.get("outputs", {}).get("tmy_hourly", [])

        for record in tmy_records:
            try:
                # Parse timestamp from PVGIS format: YYYYMMDDHHMM
                time_str = str(record.get("time(UTC)", "200001010000"))
                timestamp = datetime.strptime(time_str, "%Y%m%d:%H%M")

                # Create weather data point
                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(record.get("T2m", 20)),  # 2m temperature
                    irradiance_ghi=float(record.get("G(h)", 0)),  # GHI
                    irradiance_dni=float(record.get("Gb(n)", 0)),  # DNI
                    irradiance_dhi=float(record.get("Gd(h)", 0)),  # DHI
                    wind_speed=float(record.get("WS10m", 0)),  # 10m wind speed
                    wind_direction=float(record.get("WD10m", 0)),  # 10m wind direction
                    relative_humidity=float(record.get("RH", 50)),  # Relative humidity
                    pressure=float(record.get("SP", 101325)),  # Surface pressure
                )
                hourly_data.append(point)

            except Exception as e:
                logger.warning(f"Error parsing PVGIS record: {e}")
                continue

        # Create TMY data
        tmy_data = TMYData(
            location=location,
            data_source=DataSource.PVGIS,
            format_type=TMYFormat.JSON,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=2005,
            end_year=2020,
            hourly_data=hourly_data,
            metadata={
                "database": metadata.get("raddatabase", "PVGIS-SARAH2"),
                "pvgis_version": json_data.get("meta", {}).get("version", "5.2"),
            },
        )

        return tmy_data
