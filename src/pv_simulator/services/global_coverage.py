"""
Global Weather Coverage for worldwide location database and mapping.

This module manages a global database of locations with weather data coverage,
provides coordinate-to-weather mapping, nearest station finding, and
geographic interpolation with elevation corrections.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geopy.distance import geodesic

from pv_simulator.config.settings import settings
from pv_simulator.models.weather import GlobalLocation, TMYData

logger = logging.getLogger(__name__)


class GlobalWeatherCoverage:
    """
    Manager for global weather data coverage and location database.

    Provides:
    - Worldwide location database management
    - Coordinate-to-weather data mapping
    - Nearest weather station finding
    - Geographic interpolation between stations
    - Elevation corrections for weather data

    Attributes:
        locations_db: Database of global locations
        locations_db_path: Path to locations database file
    """

    # Standard atmosphere lapse rate (°C/m)
    TEMPERATURE_LAPSE_RATE = -0.0065

    # Pressure decrease with elevation (Pa/m)
    PRESSURE_LAPSE_RATE = -12.0

    def __init__(self, locations_db_path: Optional[Path] = None) -> None:
        """
        Initialize Global Weather Coverage.

        Args:
            locations_db_path: Path to locations database file
        """
        self.locations_db_path = (
            locations_db_path or settings.locations_data_dir / "global_locations.json"
        )
        self.locations_db: List[GlobalLocation] = []

        # Load or create location database
        self._load_or_create_database()

        logger.info(
            f"GlobalWeatherCoverage initialized with {len(self.locations_db)} locations"
        )

    def worldwide_location_database(self) -> List[GlobalLocation]:
        """
        Get the complete worldwide location database.

        Returns:
            List of all locations in database
        """
        return self.locations_db

    def add_location(self, location: GlobalLocation) -> None:
        """
        Add a location to the database.

        Args:
            location: Location to add
        """
        # Check for duplicates
        for existing in self.locations_db:
            if (
                abs(existing.latitude - location.latitude) < 0.01
                and abs(existing.longitude - location.longitude) < 0.01
            ):
                logger.warning(f"Location near {location.name} already exists")
                return

        self.locations_db.append(location)
        logger.info(f"Added location: {location.name}")

        # Save to disk
        self._save_database()

    def coordinate_to_weather_mapping(
        self, latitude: float, longitude: float, max_distance_km: float = 100.0
    ) -> Optional[GlobalLocation]:
        """
        Map coordinates to nearest weather data location.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            max_distance_km: Maximum search distance in kilometers

        Returns:
            Nearest location if found within max_distance, None otherwise
        """
        nearest = self.nearest_station_finder(latitude, longitude, num_stations=1)

        if not nearest:
            return None

        station, distance = nearest[0]

        if distance <= max_distance_km:
            logger.info(
                f"Mapped ({latitude}, {longitude}) to {station.name} "
                f"at {distance:.1f} km distance"
            )
            return station
        else:
            logger.warning(
                f"Nearest station {station.name} is {distance:.1f} km away "
                f"(exceeds max_distance={max_distance_km} km)"
            )
            return None

    def nearest_station_finder(
        self,
        latitude: float,
        longitude: float,
        num_stations: int = 5,
        max_distance_km: Optional[float] = None,
    ) -> List[Tuple[GlobalLocation, float]]:
        """
        Find nearest weather stations to a given coordinate.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            num_stations: Number of nearest stations to return
            max_distance_km: Maximum search distance in kilometers

        Returns:
            List of (location, distance_km) tuples, sorted by distance
        """
        if not self.locations_db:
            logger.warning("Location database is empty")
            return []

        target_coords = (latitude, longitude)
        distances = []

        # Calculate distances to all locations
        for location in self.locations_db:
            station_coords = (location.latitude, location.longitude)
            distance_km = geodesic(target_coords, station_coords).kilometers
            distances.append((location, distance_km))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Apply max distance filter
        if max_distance_km is not None:
            distances = [(loc, dist) for loc, dist in distances if dist <= max_distance_km]

        # Return top N
        result = distances[:num_stations]

        logger.info(f"Found {len(result)} nearest stations within search criteria")
        return result

    def geographic_interpolation(
        self,
        latitude: float,
        longitude: float,
        tmy_data_map: Dict[str, TMYData],
        num_stations: int = 4,
    ) -> Optional[TMYData]:
        """
        Interpolate weather data from nearby stations using inverse distance weighting.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            tmy_data_map: Dictionary mapping location names to TMY data
            num_stations: Number of stations to use for interpolation

        Returns:
            Interpolated TMY data, or None if insufficient data
        """
        logger.info(f"Performing geographic interpolation for ({latitude}, {longitude})")

        # Find nearest stations
        nearest = self.nearest_station_finder(latitude, longitude, num_stations=num_stations)

        if len(nearest) < 2:
            logger.warning("Insufficient stations for interpolation")
            return None

        # Filter stations that have TMY data
        stations_with_data = [
            (loc, dist) for loc, dist in nearest if loc.name in tmy_data_map
        ]

        if not stations_with_data:
            logger.warning("No TMY data available for nearest stations")
            return None

        # Use first station as template
        template_location = stations_with_data[0][0]
        template_data = tmy_data_map[template_location.name]

        # Perform inverse distance weighting interpolation
        interpolated_data = self._inverse_distance_weighting(
            latitude, longitude, stations_with_data, tmy_data_map
        )

        # Create new location
        interpolated_location = GlobalLocation(
            name=f"Interpolated_{latitude:.2f}_{longitude:.2f}",
            country=template_location.country,
            latitude=latitude,
            longitude=longitude,
            elevation=template_location.elevation,  # Could also interpolate elevation
            timezone=template_location.timezone,
        )

        # Create TMY data with interpolated values
        result = TMYData(
            location=interpolated_location,
            data_source=template_data.data_source,
            format_type=template_data.format_type,
            temporal_resolution=template_data.temporal_resolution,
            start_year=template_data.start_year,
            end_year=template_data.end_year,
            hourly_data=interpolated_data,
            metadata={
                "interpolation_method": "inverse_distance_weighting",
                "num_stations": len(stations_with_data),
                "stations": [loc.name for loc, _ in stations_with_data],
            },
        )

        logger.info(f"Successfully interpolated data from {len(stations_with_data)} stations")
        return result

    def elevation_corrections(
        self, tmy_data: TMYData, target_elevation: float
    ) -> TMYData:
        """
        Apply elevation corrections to weather data.

        Adjusts temperature and pressure based on elevation difference
        using standard atmospheric lapse rates.

        Args:
            tmy_data: Original TMY data
            target_elevation: Target elevation in meters

        Returns:
            TMY data corrected for elevation
        """
        elevation_diff = target_elevation - tmy_data.location.elevation

        logger.info(
            f"Applying elevation correction: {tmy_data.location.elevation:.0f}m "
            f"-> {target_elevation:.0f}m (Δ={elevation_diff:.0f}m)"
        )

        # Create corrected data points
        corrected_data = []

        for point in tmy_data.hourly_data:
            # Temperature correction using standard lapse rate
            corrected_temp = point.temperature + (elevation_diff * self.TEMPERATURE_LAPSE_RATE)

            # Pressure correction
            corrected_pressure = point.pressure
            if point.pressure is not None:
                corrected_pressure = point.pressure + (elevation_diff * self.PRESSURE_LAPSE_RATE)

            # Create corrected point (copy other values)
            corrected_point = point.model_copy()
            corrected_point.temperature = corrected_temp
            corrected_point.pressure = corrected_pressure

            corrected_data.append(corrected_point)

        # Create new location with corrected elevation
        corrected_location = tmy_data.location.model_copy()
        corrected_location.elevation = target_elevation

        # Create corrected TMY data
        corrected_tmy = TMYData(
            location=corrected_location,
            data_source=tmy_data.data_source,
            format_type=tmy_data.format_type,
            temporal_resolution=tmy_data.temporal_resolution,
            start_year=tmy_data.start_year,
            end_year=tmy_data.end_year,
            hourly_data=corrected_data,
            metadata={
                **tmy_data.metadata,
                "elevation_corrected": True,
                "original_elevation": tmy_data.location.elevation,
                "target_elevation": target_elevation,
            },
        )

        logger.info("Elevation correction applied successfully")
        return corrected_tmy

    def search_by_name(self, name: str, country: Optional[str] = None) -> List[GlobalLocation]:
        """
        Search locations by name.

        Args:
            name: Location name (partial match allowed)
            country: Optional country filter

        Returns:
            List of matching locations
        """
        name_lower = name.lower()
        results = []

        for location in self.locations_db:
            name_match = name_lower in location.name.lower()
            country_match = country is None or country.lower() in location.country.lower()

            if name_match and country_match:
                results.append(location)

        logger.info(f"Found {len(results)} locations matching '{name}'")
        return results

    def search_by_region(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> List[GlobalLocation]:
        """
        Search locations within a geographic region.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude

        Returns:
            List of locations within the region
        """
        results = []

        for location in self.locations_db:
            if (
                min_lat <= location.latitude <= max_lat
                and min_lon <= location.longitude <= max_lon
            ):
                results.append(location)

        logger.info(f"Found {len(results)} locations in region")
        return results

    # Helper methods

    def _load_or_create_database(self) -> None:
        """Load existing database or create a new one with sample data."""
        if self.locations_db_path.exists():
            try:
                with open(self.locations_db_path, "r") as f:
                    data = json.load(f)

                self.locations_db = [GlobalLocation(**loc) for loc in data]
                logger.info(f"Loaded {len(self.locations_db)} locations from database")

            except Exception as e:
                logger.error(f"Error loading database: {e}")
                self._create_default_database()
        else:
            self._create_default_database()

    def _create_default_database(self) -> None:
        """Create default database with major world cities."""
        logger.info("Creating default location database")

        # Sample major cities
        default_locations = [
            GlobalLocation(
                name="Denver",
                country="USA",
                latitude=39.7392,
                longitude=-104.9903,
                elevation=1609.0,
                timezone="America/Denver",
                climate_zone="BSk",
            ),
            GlobalLocation(
                name="Phoenix",
                country="USA",
                latitude=33.4484,
                longitude=-112.0740,
                elevation=331.0,
                timezone="America/Phoenix",
                climate_zone="BWh",
            ),
            GlobalLocation(
                name="Berlin",
                country="Germany",
                latitude=52.5200,
                longitude=13.4050,
                elevation=34.0,
                timezone="Europe/Berlin",
                climate_zone="Cfb",
            ),
            GlobalLocation(
                name="Madrid",
                country="Spain",
                latitude=40.4168,
                longitude=-3.7038,
                elevation=667.0,
                timezone="Europe/Madrid",
                climate_zone="Csa",
            ),
            GlobalLocation(
                name="Tokyo",
                country="Japan",
                latitude=35.6762,
                longitude=139.6503,
                elevation=40.0,
                timezone="Asia/Tokyo",
                climate_zone="Cfa",
            ),
            GlobalLocation(
                name="Sydney",
                country="Australia",
                latitude=-33.8688,
                longitude=151.2093,
                elevation=58.0,
                timezone="Australia/Sydney",
                climate_zone="Cfa",
            ),
        ]

        self.locations_db = default_locations
        self._save_database()

    def _save_database(self) -> None:
        """Save location database to disk."""
        try:
            self.locations_db_path.parent.mkdir(parents=True, exist_ok=True)

            data = [loc.model_dump() for loc in self.locations_db]

            with open(self.locations_db_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved {len(self.locations_db)} locations to database")

        except Exception as e:
            logger.error(f"Error saving database: {e}")

    def _inverse_distance_weighting(
        self,
        latitude: float,
        longitude: float,
        stations_with_data: List[Tuple[GlobalLocation, float]],
        tmy_data_map: Dict[str, TMYData],
    ) -> List:
        """
        Perform inverse distance weighting interpolation.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            stations_with_data: List of (location, distance) tuples
            tmy_data_map: Dictionary mapping location names to TMY data

        Returns:
            List of interpolated weather data points
        """
        from pv_simulator.models.weather import WeatherDataPoint

        # Calculate weights (inverse distance squared)
        weights = []
        total_weight = 0.0

        for location, distance in stations_with_data:
            if distance < 0.1:  # Very close - use directly
                weight = 1.0
                weights = [(location, 1.0)]
                total_weight = 1.0
                break
            else:
                weight = 1.0 / (distance**2)
                weights.append((location, weight))
                total_weight += weight

        # Normalize weights
        weights = [(loc, w / total_weight) for loc, w in weights]

        # Get first dataset as template for length
        first_data = tmy_data_map[stations_with_data[0][0].name]
        num_points = len(first_data.hourly_data)

        # Interpolate each time step
        interpolated_points = []

        for i in range(num_points):
            weighted_temp = 0.0
            weighted_ghi = 0.0
            weighted_dni = 0.0
            weighted_dhi = 0.0
            weighted_wind = 0.0

            timestamp = first_data.hourly_data[i].timestamp

            for location, weight in weights:
                data = tmy_data_map[location.name]
                if i < len(data.hourly_data):
                    point = data.hourly_data[i]
                    weighted_temp += point.temperature * weight
                    weighted_ghi += point.irradiance_ghi * weight
                    weighted_dni += point.irradiance_dni * weight
                    weighted_dhi += point.irradiance_dhi * weight
                    weighted_wind += point.wind_speed * weight

            # Create interpolated point
            interpolated_point = WeatherDataPoint(
                timestamp=timestamp,
                temperature=weighted_temp,
                irradiance_ghi=weighted_ghi,
                irradiance_dni=weighted_dni,
                irradiance_dhi=weighted_dhi,
                wind_speed=weighted_wind,
            )
            interpolated_points.append(interpolated_point)

        return interpolated_points
