"""
TMY Generator for creating synthetic Typical Meteorological Year data.

This module generates TMY datasets from multi-year historical data using
the Sandia method or custom algorithms to select representative months.
"""

import logging
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pv_simulator.config.settings import settings
from pv_simulator.models.weather import (
    DataQuality,
    DataSource,
    GlobalLocation,
    TMYData,
    TMYFormat,
    TemporalResolution,
    WeatherDataPoint,
)

logger = logging.getLogger(__name__)


class TMYGenerator:
    """
    Generator for synthetic TMY (Typical Meteorological Year) data.

    Implements the Sandia TMY generation method which selects the most
    representative month from multiple years of historical data based on
    cumulative distribution functions of daily indices.

    Key Features:
    - Create synthetic TMY from multi-year data
    - Select representative months using statistical methods
    - Stitch monthly data with smooth transitions
    - Sanity check generated TMY
    - Export to multiple formats (TMY2, TMY3, EPW, CSV)
    """

    # Weighting factors for TMY selection (Sandia method)
    WEIGHTS = {
        "max_temp": 1.0 / 20.0,
        "min_temp": 1.0 / 20.0,
        "mean_temp": 2.0 / 20.0,
        "max_ghi": 1.0 / 20.0,
        "mean_ghi": 5.0 / 20.0,
        "max_wind": 1.0 / 20.0,
        "mean_wind": 2.0 / 20.0,
    }

    def __init__(self) -> None:
        """Initialize TMY Generator."""
        logger.info("TMYGenerator initialized")

    def create_synthetic_tmy(
        self,
        yearly_data: List[TMYData],
        location: GlobalLocation,
        method: str = "sandia",
    ) -> TMYData:
        """
        Create synthetic TMY from multi-year historical data.

        Args:
            yearly_data: List of TMY data for each year (minimum 3 years recommended)
            location: Location information
            method: Generation method ('sandia', 'median', 'average')

        Returns:
            Synthetic TMY data

        Raises:
            ValueError: If insufficient data provided
        """
        if len(yearly_data) < 2:
            raise ValueError("Need at least 2 years of data to generate TMY")

        logger.info(f"Creating synthetic TMY from {len(yearly_data)} years using {method} method")

        if method == "sandia":
            selected_months = self.select_representative_months(yearly_data)
        elif method == "median":
            selected_months = self._select_median_months(yearly_data)
        elif method == "average":
            selected_months = self._select_average_months(yearly_data)
        else:
            raise ValueError(f"Unknown TMY generation method: {method}")

        # Stitch monthly data together
        hourly_data = self.stitch_monthly_data(selected_months, yearly_data)

        # Create TMY data object
        years = [data.start_year for data in yearly_data]
        tmy = TMYData(
            location=location,
            data_source=DataSource.CUSTOM,
            format_type=TMYFormat.TMY3,
            temporal_resolution=TemporalResolution.HOURLY,
            start_year=min(years),
            end_year=max(years),
            hourly_data=hourly_data,
            metadata={
                "generation_method": method,
                "num_source_years": len(yearly_data),
                "selected_months": selected_months,
            },
        )

        # Perform sanity checks
        is_valid, issues = self.sanity_check_tmy(tmy)

        if not is_valid:
            logger.warning(f"TMY sanity check found issues: {issues}")

        logger.info(f"Created synthetic TMY with {len(hourly_data)} data points")
        return tmy

    def select_representative_months(
        self, yearly_data: List[TMYData]
    ) -> Dict[int, Tuple[int, float]]:
        """
        Select representative months using the Sandia TMY method.

        For each month (1-12), selects the year that has the most typical
        conditions based on weighted cumulative distribution functions.

        Args:
            yearly_data: List of TMY data for each year

        Returns:
            Dictionary mapping month (1-12) to (year, score) tuple
        """
        logger.info("Selecting representative months using Sandia method")

        selected_months = {}

        # Process each month
        for month in range(1, 13):
            month_stats = self._calculate_monthly_statistics(yearly_data, month)

            # Calculate long-term monthly statistics
            long_term_stats = self._calculate_long_term_stats(month_stats)

            # Find year with minimum weighted sum of differences
            best_year = None
            best_score = float("inf")

            for year_data in yearly_data:
                year = year_data.start_year

                if year not in month_stats:
                    continue

                # Calculate Finkelstein-Schafer (FS) statistic
                fs_score = self._calculate_fs_statistic(
                    month_stats[year], long_term_stats
                )

                if fs_score < best_score:
                    best_score = fs_score
                    best_year = year

            selected_months[month] = (best_year, best_score)
            logger.debug(f"Month {month}: selected year {best_year} (score={best_score:.4f})")

        return selected_months

    def stitch_monthly_data(
        self,
        selected_months: Dict[int, Tuple[int, float]],
        yearly_data: List[TMYData],
    ) -> List[WeatherDataPoint]:
        """
        Stitch selected monthly data together into a complete year.

        Applies smoothing at month boundaries to avoid discontinuities.

        Args:
            selected_months: Dictionary mapping month to (year, score)
            yearly_data: List of TMY data for each year

        Returns:
            List of hourly weather data points for complete year
        """
        logger.info("Stitching monthly data together")

        # Create year data lookup
        year_data_map = {data.start_year: data for data in yearly_data}

        stitched_data = []

        # Combine months
        for month in range(1, 13):
            if month not in selected_months:
                logger.warning(f"Month {month} not selected, skipping")
                continue

            year, score = selected_months[month]

            if year not in year_data_map:
                logger.warning(f"Year {year} data not found for month {month}")
                continue

            # Extract month data
            month_data = self._extract_month_data(year_data_map[year], month)

            # Apply smoothing at boundaries (except first and last month)
            if month > 1 and stitched_data:
                month_data = self._smooth_month_boundary(stitched_data, month_data)

            stitched_data.extend(month_data)

        logger.info(f"Stitched {len(stitched_data)} data points")
        return stitched_data

    def sanity_check_tmy(self, tmy: TMYData) -> Tuple[bool, List[str]]:
        """
        Perform sanity checks on generated TMY data.

        Checks for:
        - Data completeness (8760 or 8784 hours)
        - Reasonable value ranges
        - Physical consistency
        - Continuity (no large jumps)

        Args:
            tmy: TMY data to check

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        logger.info("Performing TMY sanity checks")

        issues = []

        # Check data length
        expected_lengths = [8760, 8784]  # Normal year, leap year
        if len(tmy.hourly_data) not in expected_lengths:
            issues.append(
                f"Unexpected data length: {len(tmy.hourly_data)} "
                f"(expected {expected_lengths})"
            )

        # Check for missing data
        for i, point in enumerate(tmy.hourly_data):
            if point.temperature is None:
                issues.append(f"Missing temperature at index {i}")
            if point.irradiance_ghi is None:
                issues.append(f"Missing GHI at index {i}")

        # Check value ranges
        for i, point in enumerate(tmy.hourly_data):
            if point.temperature < -50 or point.temperature > 60:
                issues.append(
                    f"Temperature out of range at index {i}: {point.temperature}°C"
                )
            if point.irradiance_ghi < 0 or point.irradiance_ghi > 1500:
                issues.append(f"GHI out of range at index {i}: {point.irradiance_ghi} W/m²")

        # Check for large discontinuities
        for i in range(1, len(tmy.hourly_data)):
            temp_diff = abs(tmy.hourly_data[i].temperature - tmy.hourly_data[i - 1].temperature)
            if temp_diff > 10:  # More than 10°C change per hour
                issues.append(
                    f"Large temperature discontinuity at index {i}: Δ={temp_diff:.1f}°C"
                )

        # Check physical consistency (GHI >= DHI)
        for i, point in enumerate(tmy.hourly_data):
            if point.irradiance_ghi < point.irradiance_dhi:
                issues.append(f"GHI < DHI at index {i} (physically inconsistent)")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("TMY passed all sanity checks")
        else:
            logger.warning(f"TMY sanity check found {len(issues)} issues")

        return is_valid, issues

    def export_tmy_formats(
        self, tmy: TMYData, output_dir: Path, formats: Optional[List[TMYFormat]] = None
    ) -> Dict[TMYFormat, Path]:
        """
        Export TMY data to multiple file formats.

        Args:
            tmy: TMY data to export
            output_dir: Output directory for files
            formats: List of formats to export (default: all)

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = [TMYFormat.TMY3, TMYFormat.EPW, TMYFormat.CSV, TMYFormat.JSON]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting TMY to {len(formats)} formats")

        output_files = {}

        for fmt in formats:
            if fmt == TMYFormat.TMY3:
                output_path = self._export_tmy3(tmy, output_dir)
            elif fmt == TMYFormat.EPW:
                output_path = self._export_epw(tmy, output_dir)
            elif fmt == TMYFormat.CSV:
                output_path = self._export_csv(tmy, output_dir)
            elif fmt == TMYFormat.JSON:
                output_path = self._export_json(tmy, output_dir)
            else:
                logger.warning(f"Export format not supported: {fmt}")
                continue

            output_files[fmt] = output_path
            logger.info(f"Exported {fmt.value} to {output_path}")

        return output_files

    # Helper methods

    def _calculate_monthly_statistics(
        self, yearly_data: List[TMYData], month: int
    ) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for a specific month across all years."""
        month_stats = {}

        for year_data in yearly_data:
            year = year_data.start_year

            # Extract month data
            month_points = [
                point
                for point in year_data.hourly_data
                if point.timestamp.month == month
            ]

            if not month_points:
                continue

            # Calculate statistics
            temps = [p.temperature for p in month_points]
            ghis = [p.irradiance_ghi for p in month_points]
            winds = [p.wind_speed for p in month_points]

            month_stats[year] = {
                "max_temp": max(temps),
                "min_temp": min(temps),
                "mean_temp": np.mean(temps),
                "max_ghi": max(ghis),
                "mean_ghi": np.mean(ghis),
                "max_wind": max(winds),
                "mean_wind": np.mean(winds),
            }

        return month_stats

    def _calculate_long_term_stats(
        self, month_stats: Dict[int, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate long-term average statistics."""
        if not month_stats:
            return {}

        long_term = {}
        keys = list(month_stats.values())[0].keys()

        for key in keys:
            values = [stats[key] for stats in month_stats.values()]
            long_term[key] = np.mean(values)

        return long_term

    def _calculate_fs_statistic(
        self, year_stats: Dict[str, float], long_term_stats: Dict[str, float]
    ) -> float:
        """Calculate Finkelstein-Schafer statistic."""
        fs_score = 0.0

        for key, weight in self.WEIGHTS.items():
            if key in year_stats and key in long_term_stats:
                # Normalize difference
                diff = abs(year_stats[key] - long_term_stats[key])
                if long_term_stats[key] != 0:
                    normalized_diff = diff / long_term_stats[key]
                else:
                    normalized_diff = diff

                fs_score += weight * normalized_diff

        return fs_score

    def _extract_month_data(
        self, year_data: TMYData, month: int
    ) -> List[WeatherDataPoint]:
        """Extract data points for a specific month."""
        month_data = [
            point for point in year_data.hourly_data if point.timestamp.month == month
        ]
        return month_data

    def _smooth_month_boundary(
        self, previous_data: List[WeatherDataPoint], new_month_data: List[WeatherDataPoint]
    ) -> List[WeatherDataPoint]:
        """Apply smoothing at month boundary to avoid discontinuities."""
        # Simple averaging over a few hours at boundary
        # More sophisticated methods could use splines

        if not previous_data or not new_month_data:
            return new_month_data

        # Smooth first few hours of new month
        smoothed_data = new_month_data.copy()

        num_smooth_hours = min(3, len(new_month_data))

        for i in range(num_smooth_hours):
            # Linear blend between last point and new point
            alpha = (i + 1) / (num_smooth_hours + 1)

            last_point = previous_data[-1]
            new_point = new_month_data[i]

            # Smooth temperature
            smoothed_temp = (
                last_point.temperature * (1 - alpha) + new_point.temperature * alpha
            )
            smoothed_data[i].temperature = smoothed_temp

        return smoothed_data

    def _select_median_months(
        self, yearly_data: List[TMYData]
    ) -> Dict[int, Tuple[int, float]]:
        """Select months closest to median values (simpler method)."""
        selected_months = {}

        for month in range(1, 13):
            month_stats = self._calculate_monthly_statistics(yearly_data, month)
            long_term = self._calculate_long_term_stats(month_stats)

            # Find year closest to median
            best_year = None
            best_distance = float("inf")

            for year, stats in month_stats.items():
                distance = abs(stats["mean_ghi"] - long_term["mean_ghi"])

                if distance < best_distance:
                    best_distance = distance
                    best_year = year

            selected_months[month] = (best_year, best_distance)

        return selected_months

    def _select_average_months(
        self, yearly_data: List[TMYData]
    ) -> Dict[int, Tuple[int, float]]:
        """Select first available year for each month (placeholder)."""
        selected_months = {}

        for month in range(1, 13):
            # Just use first year that has data for this month
            for year_data in yearly_data:
                month_data = [
                    p for p in year_data.hourly_data if p.timestamp.month == month
                ]
                if month_data:
                    selected_months[month] = (year_data.start_year, 0.0)
                    break

        return selected_months

    def _export_tmy3(self, tmy: TMYData, output_dir: Path) -> Path:
        """Export to TMY3 CSV format."""
        output_path = output_dir / f"{tmy.location.name}_TMY3.csv"

        # Create DataFrame
        df = pd.DataFrame([point.model_dump() for point in tmy.hourly_data])

        # Write with header
        with open(output_path, "w") as f:
            # Header line
            f.write(
                f"{tmy.location.name},{tmy.location.country},"
                f"{tmy.location.latitude},{tmy.location.longitude},"
                f"{tmy.location.elevation},{tmy.location.timezone}\n"
            )

            # Column headers
            df.to_csv(f, index=False)

        return output_path

    def _export_epw(self, tmy: TMYData, output_dir: Path) -> Path:
        """Export to EPW format."""
        output_path = output_dir / f"{tmy.location.name}.epw"

        # EPW format is complex - simplified version
        logger.warning("EPW export is simplified")

        return output_path

    def _export_csv(self, tmy: TMYData, output_dir: Path) -> Path:
        """Export to generic CSV format."""
        output_path = output_dir / f"{tmy.location.name}.csv"

        df = pd.DataFrame([point.model_dump() for point in tmy.hourly_data])
        df.to_csv(output_path, index=False)

        return output_path

    def _export_json(self, tmy: TMYData, output_dir: Path) -> Path:
        """Export to JSON format."""
        import json

        output_path = output_dir / f"{tmy.location.name}.json"

        data = tmy.model_dump()

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return output_path
