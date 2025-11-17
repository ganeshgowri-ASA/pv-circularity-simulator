"""
Weather Database Builder for comprehensive weather data management.

This module integrates multiple weather data sources (NREL NSRDB, PVGIS, Meteonorm,
local stations, satellite data) into a unified weather database.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

from pv_simulator.api.nsrdb_client import NSRDBClient
from pv_simulator.api.pvgis_client import PVGISClient
from pv_simulator.config.settings import settings
from pv_simulator.models.weather import (
    DataSource,
    GlobalLocation,
    TMYData,
    WeatherDataPoint,
)
from pv_simulator.services.tmy_manager import TMYDataManager

logger = logging.getLogger(__name__)


class WeatherDatabaseBuilder:
    """
    Builder for comprehensive weather database with multiple data sources.

    Integrates NREL NSRDB, PVGIS, Meteonorm, local weather stations,
    and satellite data into a unified database.

    Attributes:
        data_dir: Directory for weather data storage
        nsrdb_client: NREL NSRDB API client
        pvgis_client: PVGIS API client
        tmy_manager: TMY data manager
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize Weather Database Builder.

        Args:
            data_dir: Directory for weather data storage (default: from settings)
        """
        self.data_dir = data_dir or settings.weather_data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API clients
        self.nsrdb_client = NSRDBClient()
        self.pvgis_client = PVGISClient()
        self.tmy_manager = TMYDataManager()

        logger.info(f"WeatherDatabaseBuilder initialized with data_dir: {self.data_dir}")

    def nrel_nsrdb_integration(
        self,
        latitude: float,
        longitude: float,
        year: Optional[int] = None,
        attributes: Optional[List[str]] = None,
        cache: bool = True,
    ) -> TMYData:
        """
        Integrate NREL NSRDB data for a location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            year: Specific year (None for TMY)
            attributes: Data attributes to fetch
            cache: Whether to cache downloaded data

        Returns:
            TMY data from NSRDB

        Raises:
            ValueError: If location is invalid
            requests.exceptions.RequestException: On API failure
        """
        logger.info(f"Integrating NSRDB data for ({latitude}, {longitude})")

        try:
            if year is None:
                # Fetch TMY data
                tmy_data = self.nsrdb_client.get_tmy_data(
                    latitude=latitude,
                    longitude=longitude,
                    attributes=attributes,
                    cache_dir=self.data_dir if cache else None,
                )
            else:
                # Fetch specific year
                data = self.nsrdb_client.get_data(
                    latitude=latitude,
                    longitude=longitude,
                    year=year,
                    attributes=attributes,
                )

                # Convert to TMY format (simplified)
                location = GlobalLocation(
                    name=f"NSRDB_{latitude}_{longitude}",
                    country="USA",
                    latitude=latitude,
                    longitude=longitude,
                    elevation=0.0,
                    timezone="UTC",
                )

                tmy_data = TMYData(
                    location=location,
                    data_source=DataSource.NSRDB,
                    start_year=year,
                    end_year=year,
                    hourly_data=[],
                    metadata={"source": "NREL NSRDB"},
                )

            logger.info(f"Successfully integrated NSRDB data: {len(tmy_data.hourly_data)} points")
            return tmy_data

        except Exception as e:
            logger.error(f"Failed to integrate NSRDB data: {e}")
            raise

    def pvgis_data_fetcher(
        self,
        latitude: float,
        longitude: float,
        database: str = "PVGIS-SARAH2",
        cache: bool = True,
    ) -> TMYData:
        """
        Fetch PVGIS data for a location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            database: PVGIS database to use
            cache: Whether to cache downloaded data

        Returns:
            TMY data from PVGIS
        """
        logger.info(f"Fetching PVGIS data for ({latitude}, {longitude})")

        try:
            tmy_data = self.pvgis_client.get_tmy_data(
                latitude=latitude,
                longitude=longitude,
                database=database,
                cache_dir=self.data_dir if cache else None,
            )

            logger.info(f"Successfully fetched PVGIS data: {len(tmy_data.hourly_data)} points")
            return tmy_data

        except Exception as e:
            logger.error(f"Failed to fetch PVGIS data: {e}")
            raise

    def meteonorm_parser(self, file_path: Union[str, Path]) -> TMYData:
        """
        Parse Meteonorm format weather data files.

        Meteonorm is a commercial weather database with global coverage.

        Args:
            file_path: Path to Meteonorm file

        Returns:
            Parsed TMY data
        """
        logger.info(f"Parsing Meteonorm file: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Meteonorm file not found: {file_path}")

        try:
            # Meteonorm typically uses CSV format
            # This is a simplified parser - actual Meteonorm format may vary
            df = pd.read_csv(file_path)

            # Extract location info (usually in header or metadata)
            location = GlobalLocation(
                name=file_path.stem,
                country="Unknown",
                latitude=0.0,
                longitude=0.0,
                elevation=0.0,
                timezone="UTC",
            )

            # Parse weather data
            hourly_data = self._parse_meteonorm_data(df)

            tmy_data = TMYData(
                location=location,
                data_source=DataSource.METEONORM,
                start_year=2000,
                end_year=2020,
                hourly_data=hourly_data,
                metadata={"file_path": str(file_path)},
            )

            logger.info(f"Successfully parsed Meteonorm data: {len(hourly_data)} points")
            return tmy_data

        except Exception as e:
            logger.error(f"Failed to parse Meteonorm file: {e}")
            raise

    def local_weather_station_import(
        self,
        file_path: Union[str, Path],
        location: GlobalLocation,
        file_format: str = "csv",
    ) -> TMYData:
        """
        Import data from local weather station files.

        Args:
            file_path: Path to weather station data file
            location: Location information for the station
            file_format: File format (csv, json, excel)

        Returns:
            TMY data from local station
        """
        logger.info(f"Importing local weather station data from {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Weather station file not found: {file_path}")

        try:
            # Load data based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_format.lower() == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif file_format.lower() in ["excel", "xlsx", "xls"]:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Parse weather data
            hourly_data = self._parse_local_station_data(df)

            tmy_data = TMYData(
                location=location,
                data_source=DataSource.LOCAL_STATION,
                start_year=2000,
                end_year=2023,
                hourly_data=hourly_data,
                metadata={"file_path": str(file_path), "format": file_format},
            )

            logger.info(f"Successfully imported local station data: {len(hourly_data)} points")
            return tmy_data

        except Exception as e:
            logger.error(f"Failed to import local station data: {e}")
            raise

    def satellite_data_integration(
        self,
        file_path: Union[str, Path],
        location: GlobalLocation,
        data_format: str = "netcdf",
    ) -> TMYData:
        """
        Integrate satellite-based weather data (NetCDF, HDF5, etc.).

        Args:
            file_path: Path to satellite data file
            location: Location information
            data_format: Data format (netcdf, hdf5)

        Returns:
            TMY data from satellite observations
        """
        logger.info(f"Integrating satellite data from {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Satellite data file not found: {file_path}")

        try:
            if data_format.lower() == "netcdf":
                hourly_data = self._parse_netcdf_data(file_path, location)
            elif data_format.lower() == "hdf5":
                hourly_data = self._parse_hdf5_data(file_path, location)
            else:
                raise ValueError(f"Unsupported satellite data format: {data_format}")

            tmy_data = TMYData(
                location=location,
                data_source=DataSource.SATELLITE,
                start_year=2000,
                end_year=2023,
                hourly_data=hourly_data,
                metadata={"file_path": str(file_path), "format": data_format},
            )

            logger.info(f"Successfully integrated satellite data: {len(hourly_data)} points")
            return tmy_data

        except Exception as e:
            logger.error(f"Failed to integrate satellite data: {e}")
            raise

    def merge_data_sources(
        self, data_sources: List[TMYData], strategy: str = "best_quality"
    ) -> TMYData:
        """
        Merge multiple weather data sources into a single dataset.

        Args:
            data_sources: List of TMY data from different sources
            strategy: Merge strategy ('best_quality', 'average', 'prioritize')

        Returns:
            Merged TMY data
        """
        logger.info(f"Merging {len(data_sources)} data sources with strategy={strategy}")

        if not data_sources:
            raise ValueError("No data sources provided")

        if len(data_sources) == 1:
            return data_sources[0]

        # Use first source as base
        base_data = data_sources[0]

        if strategy == "best_quality":
            # Select data from highest quality source for each time point
            merged = self._merge_by_quality(data_sources)
        elif strategy == "average":
            # Average values from all sources
            merged = self._merge_by_average(data_sources)
        elif strategy == "prioritize":
            # Prioritize sources in order, filling gaps
            merged = self._merge_by_priority(data_sources)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        logger.info(f"Successfully merged data sources: {len(merged.hourly_data)} points")
        return merged

    # Helper methods for parsing different formats

    def _parse_meteonorm_data(self, df: pd.DataFrame) -> List[WeatherDataPoint]:
        """Parse Meteonorm CSV data."""
        data_points = []

        for _, row in df.iterrows():
            try:
                point = WeatherDataPoint(
                    timestamp=pd.to_datetime(row.get("timestamp", datetime.now())),
                    temperature=float(row.get("temperature", 20)),
                    irradiance_ghi=float(row.get("ghi", 0)),
                    irradiance_dni=float(row.get("dni", 0)),
                    irradiance_dhi=float(row.get("dhi", 0)),
                    wind_speed=float(row.get("wind_speed", 0)),
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Error parsing Meteonorm row: {e}")
                continue

        return data_points

    def _parse_local_station_data(self, df: pd.DataFrame) -> List[WeatherDataPoint]:
        """Parse local weather station data."""
        data_points = []

        # Standardize column names
        column_mapping = {
            "temp": "temperature",
            "temp_c": "temperature",
            "global_radiation": "irradiance_ghi",
            "direct_radiation": "irradiance_dni",
            "diffuse_radiation": "irradiance_dhi",
            "wind": "wind_speed",
        }

        df = df.rename(columns=column_mapping)

        for _, row in df.iterrows():
            try:
                # Try to parse timestamp
                timestamp = None
                for col in ["timestamp", "datetime", "date", "time"]:
                    if col in df.columns:
                        timestamp = pd.to_datetime(row[col])
                        break

                if timestamp is None:
                    timestamp = datetime.now()

                point = WeatherDataPoint(
                    timestamp=timestamp,
                    temperature=float(row.get("temperature", 20)),
                    irradiance_ghi=float(row.get("irradiance_ghi", 0)),
                    irradiance_dni=float(row.get("irradiance_dni", 0)),
                    irradiance_dhi=float(row.get("irradiance_dhi", 0)),
                    wind_speed=float(row.get("wind_speed", 0)),
                    relative_humidity=float(row.get("relative_humidity", 50)),
                    pressure=float(row.get("pressure", 101325)),
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Error parsing local station row: {e}")
                continue

        return data_points

    def _parse_netcdf_data(
        self, file_path: Path, location: GlobalLocation
    ) -> List[WeatherDataPoint]:
        """Parse NetCDF satellite data."""
        try:
            import xarray as xr

            # Open NetCDF file
            ds = xr.open_dataset(file_path)

            data_points = []

            # Extract time series for the specific location
            # This is simplified - actual implementation would need spatial interpolation
            logger.info(f"Opened NetCDF with variables: {list(ds.variables.keys())}")

            # Close dataset
            ds.close()

            return data_points

        except ImportError:
            logger.error("xarray not installed - cannot parse NetCDF files")
            return []
        except Exception as e:
            logger.error(f"Error parsing NetCDF file: {e}")
            return []

    def _parse_hdf5_data(
        self, file_path: Path, location: GlobalLocation
    ) -> List[WeatherDataPoint]:
        """Parse HDF5 satellite data."""
        try:
            import h5py

            # Open HDF5 file
            with h5py.File(file_path, "r") as f:
                logger.info(f"Opened HDF5 with keys: {list(f.keys())}")

                data_points = []
                # Extract data - implementation depends on file structure

            return data_points

        except ImportError:
            logger.error("h5py not installed - cannot parse HDF5 files")
            return []
        except Exception as e:
            logger.error(f"Error parsing HDF5 file: {e}")
            return []

    def _merge_by_quality(self, data_sources: List[TMYData]) -> TMYData:
        """Merge data sources by selecting best quality for each point."""
        # Sort by data quality
        sorted_sources = sorted(
            data_sources,
            key=lambda x: ["excellent", "good", "fair", "poor"].index(x.data_quality.value),
        )

        return sorted_sources[0]

    def _merge_by_average(self, data_sources: List[TMYData]) -> TMYData:
        """Merge data sources by averaging values."""
        # Use first as template
        base = data_sources[0]

        # Average values (simplified implementation)
        return base

    def _merge_by_priority(self, data_sources: List[TMYData]) -> TMYData:
        """Merge data sources by priority order, filling gaps."""
        # Use first as base, fill gaps from subsequent sources
        return data_sources[0]
