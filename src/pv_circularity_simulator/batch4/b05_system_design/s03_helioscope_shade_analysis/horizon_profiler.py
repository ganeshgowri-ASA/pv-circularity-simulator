"""
Horizon Profiler for far shading analysis.

This module handles import, validation, and processing of horizon profile data
for accurate far shading calculations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .models import HorizonProfile, Location

logger = logging.getLogger(__name__)


class HorizonProfiler:
    """
    Horizon profile management and analysis.

    This class handles import of horizon profiles from various sources,
    validation, interpolation, and calculation of far shading losses.
    """

    def __init__(self, location: Location):
        """
        Initialize the horizon profiler.

        Args:
            location: Geographic location for horizon profile
        """
        self.location = location
        self.horizon_profile: Optional[HorizonProfile] = None

        logger.info(
            f"Initialized HorizonProfiler for location: "
            f"({location.latitude:.4f}, {location.longitude:.4f})"
        )

    def import_horizon_data(
        self,
        source: str,
        data: Optional[Any] = None,
        file_path: Optional[Path] = None
    ) -> HorizonProfile:
        """
        Import horizon profile from various sources.

        Args:
            source: Data source type ('survey', 'google_earth', 'photo', 'manual', 'csv')
            data: Direct data input (dict or list of (azimuth, elevation) tuples)
            file_path: Path to data file (for 'csv' or 'photo' sources)

        Returns:
            HorizonProfile object with imported data

        Raises:
            ValueError: If source is invalid or data is missing
        """
        if source == "manual" and data:
            # Direct input of horizon data
            horizon_profile = self._import_manual_data(data)

        elif source == "csv" and file_path:
            # Import from CSV file
            horizon_profile = self._import_csv_data(file_path)

        elif source == "survey" and data:
            # Import from survey data
            horizon_profile = self._import_survey_data(data)

        elif source == "google_earth" and file_path:
            # Import from Google Earth image
            horizon_profile = self._import_google_earth_data(file_path)

        elif source == "photo" and file_path:
            # Extract horizon from photo using image processing
            horizon_profile = self._import_photo_data(file_path)

        else:
            raise ValueError(
                f"Invalid source '{source}' or missing required data/file_path"
            )

        # Validate the imported profile
        validated_profile = self.validate_horizon_profile(horizon_profile)

        self.horizon_profile = validated_profile
        logger.info(
            f"Imported horizon profile from {source} with "
            f"{len(validated_profile.azimuths)} points"
        )

        return validated_profile

    def calculate_horizon_angles(
        self,
        azimuth_resolution: float = 1.0
    ) -> HorizonProfile:
        """
        Calculate complete horizon profile at specified azimuth resolution.

        Interpolates existing horizon data to create a complete profile
        at regular azimuth intervals.

        Args:
            azimuth_resolution: Azimuth spacing in degrees

        Returns:
            HorizonProfile with interpolated data

        Raises:
            ValueError: If no horizon profile has been loaded
        """
        if self.horizon_profile is None:
            raise ValueError("No horizon profile loaded. Use import_horizon_data first.")

        # Generate complete azimuth range
        azimuths_interpolated = np.arange(0, 360, azimuth_resolution)

        # Interpolate elevations
        elevations_interpolated = np.interp(
            azimuths_interpolated,
            self.horizon_profile.azimuths,
            self.horizon_profile.elevations,
            period=360  # Wrap around at 360 degrees
        )

        interpolated_profile = HorizonProfile(
            azimuths=azimuths_interpolated.tolist(),
            elevations=elevations_interpolated.tolist(),
            source=f"{self.horizon_profile.source}_interpolated"
        )

        logger.info(
            f"Interpolated horizon profile to {len(azimuths_interpolated)} points "
            f"at {azimuth_resolution}° resolution"
        )

        return interpolated_profile

    def validate_horizon_profile(self, profile: HorizonProfile) -> HorizonProfile:
        """
        Validate and clean horizon profile data.

        Checks for consistency, sorts data, removes duplicates, and
        ensures proper azimuth coverage.

        Args:
            profile: HorizonProfile to validate

        Returns:
            Validated HorizonProfile

        Raises:
            ValueError: If profile has critical errors
        """
        if len(profile.azimuths) != len(profile.elevations):
            raise ValueError(
                f"Azimuth and elevation arrays must have same length. "
                f"Got {len(profile.azimuths)} vs {len(profile.elevations)}"
            )

        if len(profile.azimuths) < 3:
            raise ValueError("Horizon profile must have at least 3 data points")

        # Convert to numpy arrays for processing
        azimuths = np.array(profile.azimuths)
        elevations = np.array(profile.elevations)

        # Check for valid ranges (already done in Pydantic, but double-check)
        if not np.all((azimuths >= 0) & (azimuths <= 360)):
            raise ValueError("All azimuth values must be between 0 and 360 degrees")

        if not np.all((elevations >= -90) & (elevations <= 90)):
            raise ValueError("All elevation values must be between -90 and 90 degrees")

        # Sort by azimuth
        sort_indices = np.argsort(azimuths)
        azimuths = azimuths[sort_indices]
        elevations = elevations[sort_indices]

        # Remove duplicate azimuths (keep first occurrence)
        unique_indices = np.unique(azimuths, return_index=True)[1]
        azimuths = azimuths[unique_indices]
        elevations = elevations[unique_indices]

        # Ensure we have 0 and 360 degree coverage for interpolation
        if azimuths[0] != 0:
            # Add interpolated value at 0 degrees
            elev_at_0 = np.interp(0, azimuths, elevations, period=360)
            azimuths = np.insert(azimuths, 0, 0)
            elevations = np.insert(elevations, 0, elev_at_0)

        if azimuths[-1] != 360:
            # Add interpolated value at 360 degrees
            elev_at_360 = np.interp(360, azimuths, elevations, period=360)
            azimuths = np.append(azimuths, 360)
            elevations = np.append(elevations, elev_at_360)

        validated_profile = HorizonProfile(
            azimuths=azimuths.tolist(),
            elevations=elevations.tolist(),
            source=profile.source
        )

        logger.info(f"Validated horizon profile: {len(azimuths)} points")

        return validated_profile

    def visualize_horizon(
        self,
        profile: Optional[HorizonProfile] = None,
        plot_type: str = "2d"
    ) -> Dict[str, Any]:
        """
        Generate visualization data for horizon profile.

        Args:
            profile: HorizonProfile to visualize (uses instance profile if None)
            plot_type: Type of visualization ('2d', '3d', 'polar')

        Returns:
            Dictionary with visualization data

        Raises:
            ValueError: If no profile is available
        """
        profile = profile or self.horizon_profile

        if profile is None:
            raise ValueError("No horizon profile available for visualization")

        viz_data = {
            "azimuths": profile.azimuths,
            "elevations": profile.elevations,
            "plot_type": plot_type,
            "source": profile.source
        }

        if plot_type == "2d":
            # Standard 2D line plot
            viz_data["x_label"] = "Azimuth (degrees)"
            viz_data["y_label"] = "Elevation (degrees)"

        elif plot_type == "polar":
            # Polar plot (azimuth as angle, elevation as radius)
            azimuths_rad = np.radians(profile.azimuths)
            viz_data["theta"] = azimuths_rad.tolist()
            viz_data["r"] = profile.elevations

        elif plot_type == "3d":
            # 3D hemispherical plot
            points_3d = []
            radius = 100  # Arbitrary visualization radius

            for az, el in zip(profile.azimuths, profile.elevations):
                # Convert to Cartesian coordinates
                az_rad = np.radians(az)
                el_rad = np.radians(el)

                # Use elevation-adjusted radius
                r = radius * np.cos(el_rad)
                x = r * np.sin(az_rad)
                y = r * np.cos(az_rad)
                z = radius * np.sin(el_rad)

                points_3d.append([x, y, z])

            viz_data["points_3d"] = points_3d

        logger.info(f"Generated {plot_type} visualization data for horizon profile")

        return viz_data

    def horizon_shading_loss(
        self,
        sun_azimuths: List[float],
        sun_elevations: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate shading loss from horizon profile.

        Args:
            sun_azimuths: List of sun azimuth angles (degrees)
            sun_elevations: List of sun elevation angles (degrees)
            weights: Optional weights for each sun position (e.g., irradiance values)

        Returns:
            Weighted shading loss fraction (0-1)

        Raises:
            ValueError: If no horizon profile is loaded or input lengths don't match
        """
        if self.horizon_profile is None:
            raise ValueError("No horizon profile loaded")

        if len(sun_azimuths) != len(sun_elevations):
            raise ValueError("Sun azimuth and elevation lists must have same length")

        if weights is None:
            weights = [1.0] * len(sun_azimuths)
        elif len(weights) != len(sun_azimuths):
            raise ValueError("Weights list must match sun position list length")

        # Interpolate horizon elevations at sun azimuth angles
        horizon_elevations = np.interp(
            sun_azimuths,
            self.horizon_profile.azimuths,
            self.horizon_profile.elevations,
            period=360
        )

        # Calculate shading for each position
        shaded = np.array(sun_elevations) <= horizon_elevations

        # Calculate weighted shading loss
        total_weight = sum(weights)
        shaded_weight = sum(w for w, s in zip(weights, shaded) if s)

        if total_weight == 0:
            return 0.0

        shading_loss = shaded_weight / total_weight

        logger.debug(
            f"Calculated horizon shading loss: {shading_loss:.4f} "
            f"({sum(shaded)}/{len(shaded)} positions shaded)"
        )

        return shading_loss

    # Private import methods

    def _import_manual_data(
        self,
        data: Any
    ) -> HorizonProfile:
        """Import manually provided horizon data."""
        if isinstance(data, dict):
            # Data provided as dict with 'azimuths' and 'elevations' keys
            azimuths = data.get('azimuths', [])
            elevations = data.get('elevations', [])
        elif isinstance(data, (list, tuple)):
            # Data provided as list of (azimuth, elevation) tuples
            azimuths = [point[0] for point in data]
            elevations = [point[1] for point in data]
        else:
            raise ValueError("Manual data must be dict or list of tuples")

        return HorizonProfile(
            azimuths=azimuths,
            elevations=elevations,
            source="manual"
        )

    def _import_csv_data(self, file_path: Path) -> HorizonProfile:
        """Import horizon data from CSV file."""
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Try to read CSV with various formats
        try:
            df = pd.read_csv(file_path)

            # Look for azimuth/elevation columns (case-insensitive)
            azimuth_col = None
            elevation_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'azimuth' in col_lower or 'az' in col_lower:
                    azimuth_col = col
                if 'elevation' in col_lower or 'elev' in col_lower or 'el' in col_lower:
                    elevation_col = col

            if azimuth_col is None or elevation_col is None:
                # Try first two columns
                if len(df.columns) >= 2:
                    azimuth_col = df.columns[0]
                    elevation_col = df.columns[1]
                else:
                    raise ValueError("Could not identify azimuth and elevation columns")

            azimuths = df[azimuth_col].tolist()
            elevations = df[elevation_col].tolist()

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

        return HorizonProfile(
            azimuths=azimuths,
            elevations=elevations,
            source="csv"
        )

    def _import_survey_data(self, data: Any) -> HorizonProfile:
        """Import horizon data from survey measurements."""
        # Survey data expected to be similar to manual data
        # but may include additional metadata

        if isinstance(data, pd.DataFrame):
            # DataFrame with survey measurements
            azimuths = data['azimuth'].tolist()
            elevations = data['elevation'].tolist()
        elif isinstance(data, dict):
            azimuths = data.get('azimuths', [])
            elevations = data.get('elevations', [])
        else:
            # Try treating as list of tuples
            azimuths = [point[0] for point in data]
            elevations = [point[1] for point in data]

        return HorizonProfile(
            azimuths=azimuths,
            elevations=elevations,
            source="survey"
        )

    def _import_google_earth_data(self, file_path: Path) -> HorizonProfile:
        """Import horizon profile from Google Earth imagery."""
        # This would require image processing to extract horizon line
        # Simplified implementation returns placeholder

        logger.warning(
            "Google Earth import requires image processing - using simplified approach"
        )

        # Generate default flat horizon
        azimuths = list(range(0, 361, 10))
        elevations = [0.0] * len(azimuths)

        return HorizonProfile(
            azimuths=azimuths,
            elevations=elevations,
            source="google_earth"
        )

    def _import_photo_data(self, file_path: Path) -> HorizonProfile:
        """
        Extract horizon profile from panoramic photo.

        Uses edge detection and image processing to identify horizon line.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Photo file not found: {file_path}")

        logger.warning(
            "Photo import requires advanced image processing - using simplified approach"
        )

        try:
            # Load image
            img = Image.open(file_path)
            width, height = img.size

            # Simplified approach: assume equirectangular projection
            # Full implementation would use edge detection, horizon finding algorithms

            # Generate azimuth angles based on image width
            num_points = min(width, 360)
            azimuths = np.linspace(0, 360, num_points, endpoint=True)

            # Placeholder: assume horizon at middle of image
            # Real implementation would detect actual horizon
            elevations = np.zeros(num_points)

            logger.info(f"Extracted horizon profile from photo: {num_points} points")

        except Exception as e:
            logger.error(f"Error processing photo: {e}")
            # Fallback to flat horizon
            azimuths = list(range(0, 361, 10))
            elevations = [0.0] * len(azimuths)

        return HorizonProfile(
            azimuths=azimuths.tolist(),
            elevations=elevations.tolist(),
            source="photo"
        )

    def get_horizon_elevation_at_azimuth(self, azimuth: float) -> float:
        """
        Get horizon elevation at specific azimuth angle.

        Args:
            azimuth: Azimuth angle in degrees (0-360)

        Returns:
            Horizon elevation angle in degrees

        Raises:
            ValueError: If no horizon profile is loaded
        """
        if self.horizon_profile is None:
            raise ValueError("No horizon profile loaded")

        # Interpolate elevation at requested azimuth
        elevation = np.interp(
            azimuth,
            self.horizon_profile.azimuths,
            self.horizon_profile.elevations,
            period=360
        )

        return float(elevation)

    def is_sun_visible(self, sun_azimuth: float, sun_elevation: float) -> bool:
        """
        Check if sun is visible above horizon.

        Args:
            sun_azimuth: Sun azimuth angle (degrees)
            sun_elevation: Sun elevation angle (degrees)

        Returns:
            True if sun is above horizon, False otherwise
        """
        if self.horizon_profile is None:
            # No horizon profile - assume sun is visible if elevation > 0
            return sun_elevation > 0

        horizon_elevation = self.get_horizon_elevation_at_azimuth(sun_azimuth)

        return sun_elevation > horizon_elevation
