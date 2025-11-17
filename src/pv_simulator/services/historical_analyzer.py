"""
Historical Weather Analyzer for multi-year statistics and trends.

This module analyzes historical weather data to identify extreme events,
climate change trends, seasonal variability, and inter-annual variability.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from pv_simulator.models.weather import (
    ExtremeWeatherEvent,
    GlobalLocation,
    HistoricalWeatherStats,
    TMYData,
    WeatherDataPoint,
)

logger = logging.getLogger(__name__)


class HistoricalWeatherAnalyzer:
    """
    Analyzer for historical weather data and climate trends.

    Provides comprehensive analysis of multi-year weather data including:
    - Multi-year statistics (mean, std, percentiles)
    - Extreme weather event detection
    - Climate change trend analysis
    - Seasonal variability
    - Inter-annual variability
    """

    def __init__(self) -> None:
        """Initialize Historical Weather Analyzer."""
        logger.info("HistoricalWeatherAnalyzer initialized")

    def multi_year_statistics(
        self, yearly_data: List[TMYData], location: GlobalLocation
    ) -> HistoricalWeatherStats:
        """
        Calculate multi-year statistics from historical data.

        Args:
            yearly_data: List of TMY data for each year
            location: Location for analysis

        Returns:
            Historical weather statistics

        Raises:
            ValueError: If yearly_data is empty
        """
        if not yearly_data:
            raise ValueError("yearly_data cannot be empty")

        logger.info(f"Calculating multi-year statistics for {len(yearly_data)} years")

        # Extract annual irradiation for each year
        annual_ghi = []
        annual_temp = []

        for year_data in yearly_data:
            # Calculate annual GHI (kWh/m²/year)
            total_ghi = sum(point.irradiance_ghi for point in year_data.hourly_data) / 1000.0
            annual_ghi.append(total_ghi)

            # Calculate average temperature
            avg_temp = np.mean([point.temperature for point in year_data.hourly_data])
            annual_temp.append(avg_temp)

        # Calculate statistics
        annual_ghi_array = np.array(annual_ghi)
        annual_temp_array = np.array(annual_temp)

        mean_ghi = float(np.mean(annual_ghi_array))
        std_ghi = float(np.std(annual_ghi_array))
        p90_ghi = float(np.percentile(annual_ghi_array, 90))
        p50_ghi = float(np.percentile(annual_ghi_array, 50))
        p10_ghi = float(np.percentile(annual_ghi_array, 10))

        mean_temp = float(np.mean(annual_temp_array))
        std_temp = float(np.std(annual_temp_array))

        # Extract year range
        years = [data.start_year for data in yearly_data]
        start_year = min(years)
        end_year = max(years)

        # Analyze seasonal variability
        seasonal_stats = self.seasonal_variability(yearly_data)

        # Analyze inter-annual variability
        inter_annual = self.inter_annual_variability(yearly_data)

        # Detect extreme events
        extreme_events = self.extreme_weather_events(yearly_data, location)

        # Analyze climate trends
        climate_trends = self.climate_change_trends(yearly_data)

        # Create statistics object
        stats_obj = HistoricalWeatherStats(
            location=location,
            start_year=start_year,
            end_year=end_year,
            num_years=len(yearly_data),
            mean_annual_ghi=mean_ghi,
            std_annual_ghi=std_ghi,
            p90_annual_ghi=p90_ghi,
            p50_annual_ghi=p50_ghi,
            p10_annual_ghi=p10_ghi,
            mean_temperature=mean_temp,
            std_temperature=std_temp,
            extreme_events=extreme_events,
            seasonal_variability=seasonal_stats,
            inter_annual_variability=inter_annual,
            climate_change_trend=climate_trends,
        )

        logger.info(f"Multi-year statistics: mean GHI={mean_ghi:.1f} kWh/m²/year")
        return stats_obj

    def extreme_weather_events(
        self,
        yearly_data: List[TMYData],
        location: GlobalLocation,
        temperature_threshold: float = 40.0,
        wind_threshold: float = 20.0,
        low_irradiance_threshold: float = 50.0,
    ) -> List[ExtremeWeatherEvent]:
        """
        Detect extreme weather events in historical data.

        Args:
            yearly_data: List of TMY data for each year
            location: Location for events
            temperature_threshold: Temperature threshold for heat events (°C)
            wind_threshold: Wind speed threshold for high wind events (m/s)
            low_irradiance_threshold: GHI threshold for low irradiance events (W/m²)

        Returns:
            List of extreme weather events
        """
        logger.info("Detecting extreme weather events")

        extreme_events = []

        for year_data in yearly_data:
            # Analyze each data point for extremes
            for point in year_data.hourly_data:
                # Heat waves
                if point.temperature >= temperature_threshold:
                    event = ExtremeWeatherEvent(
                        event_type="heatwave",
                        timestamp=point.timestamp,
                        location=location,
                        severity=self._calculate_severity(
                            point.temperature, temperature_threshold, 50.0
                        ),
                        value=point.temperature,
                        unit="°C",
                        description=f"Extreme high temperature: {point.temperature:.1f}°C",
                        impact_score=self._calculate_impact_score(
                            point.temperature, temperature_threshold
                        ),
                    )
                    extreme_events.append(event)

                # High wind events
                if point.wind_speed >= wind_threshold:
                    event = ExtremeWeatherEvent(
                        event_type="high_wind",
                        timestamp=point.timestamp,
                        location=location,
                        severity=self._calculate_severity(point.wind_speed, wind_threshold, 40.0),
                        value=point.wind_speed,
                        unit="m/s",
                        description=f"Extreme high wind: {point.wind_speed:.1f} m/s",
                        impact_score=self._calculate_impact_score(
                            point.wind_speed, wind_threshold
                        ),
                    )
                    extreme_events.append(event)

                # Detect low irradiance during expected daylight hours
                # (simplified - would need sun position calculation for accuracy)
                hour = point.timestamp.hour
                if 10 <= hour <= 14:  # Midday hours
                    if 0 < point.irradiance_ghi < low_irradiance_threshold:
                        event = ExtremeWeatherEvent(
                            event_type="low_irradiance",
                            timestamp=point.timestamp,
                            location=location,
                            severity=self._calculate_severity(
                                low_irradiance_threshold - point.irradiance_ghi, 0, 100
                            ),
                            value=point.irradiance_ghi,
                            unit="W/m²",
                            description=f"Unusually low irradiance: {point.irradiance_ghi:.1f} W/m²",
                            impact_score=5.0,
                        )
                        extreme_events.append(event)

        logger.info(f"Detected {len(extreme_events)} extreme weather events")
        return extreme_events

    def climate_change_trends(
        self, yearly_data: List[TMYData], confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analyze climate change trends in historical data.

        Uses linear regression to identify trends in temperature and irradiance.

        Args:
            yearly_data: List of TMY data for each year
            confidence_level: Confidence level for trend significance

        Returns:
            Dictionary with trend analysis results
        """
        logger.info("Analyzing climate change trends")

        if len(yearly_data) < 3:
            logger.warning("Insufficient data for trend analysis")
            return {"error": "Insufficient data (need at least 3 years)"}

        # Extract annual metrics
        years = []
        annual_ghi = []
        annual_temp = []
        annual_wind = []

        for year_data in yearly_data:
            years.append(year_data.start_year)

            # Annual GHI
            total_ghi = sum(point.irradiance_ghi for point in year_data.hourly_data) / 1000.0
            annual_ghi.append(total_ghi)

            # Average temperature
            avg_temp = np.mean([point.temperature for point in year_data.hourly_data])
            annual_temp.append(avg_temp)

            # Average wind speed
            avg_wind = np.mean([point.wind_speed for point in year_data.hourly_data])
            annual_wind.append(avg_wind)

        # Perform linear regression for each variable
        years_array = np.array(years)

        # Temperature trend
        temp_slope, temp_intercept, temp_r_value, temp_p_value, temp_std_err = stats.linregress(
            years_array, annual_temp
        )

        # GHI trend
        ghi_slope, ghi_intercept, ghi_r_value, ghi_p_value, ghi_std_err = stats.linregress(
            years_array, annual_ghi
        )

        # Wind trend
        wind_slope, wind_intercept, wind_r_value, wind_p_value, wind_std_err = stats.linregress(
            years_array, annual_wind
        )

        trends = {
            "temperature": {
                "slope": float(temp_slope),
                "slope_per_decade": float(temp_slope * 10),
                "r_squared": float(temp_r_value**2),
                "p_value": float(temp_p_value),
                "significant": temp_p_value < (1 - confidence_level),
                "trend_direction": "warming"
                if temp_slope > 0
                else "cooling" if temp_slope < 0 else "stable",
                "unit": "°C/year",
            },
            "ghi": {
                "slope": float(ghi_slope),
                "slope_per_decade": float(ghi_slope * 10),
                "r_squared": float(ghi_r_value**2),
                "p_value": float(ghi_p_value),
                "significant": ghi_p_value < (1 - confidence_level),
                "trend_direction": "increasing"
                if ghi_slope > 0
                else "decreasing" if ghi_slope < 0 else "stable",
                "unit": "kWh/m²/year/year",
            },
            "wind_speed": {
                "slope": float(wind_slope),
                "slope_per_decade": float(wind_slope * 10),
                "r_squared": float(wind_r_value**2),
                "p_value": float(wind_p_value),
                "significant": wind_p_value < (1 - confidence_level),
                "trend_direction": "increasing"
                if wind_slope > 0
                else "decreasing" if wind_slope < 0 else "stable",
                "unit": "m/s/year",
            },
            "years_analyzed": len(years),
            "year_range": f"{min(years)}-{max(years)}",
        }

        logger.info(
            f"Climate trends: temp {temp_slope:.3f}°C/year, "
            f"GHI {ghi_slope:.2f} kWh/m²/year/year"
        )

        return trends

    def seasonal_variability(self, yearly_data: List[TMYData]) -> Dict[str, Any]:
        """
        Analyze seasonal variability in weather patterns.

        Args:
            yearly_data: List of TMY data for each year

        Returns:
            Dictionary with seasonal statistics
        """
        logger.info("Analyzing seasonal variability")

        # Aggregate data by season
        seasonal_data: Dict[str, Dict[str, List[float]]] = {
            "winter": {"ghi": [], "temp": [], "wind": []},
            "spring": {"ghi": [], "temp": [], "wind": []},
            "summer": {"ghi": [], "temp": [], "wind": []},
            "fall": {"ghi": [], "temp": [], "wind": []},
        }

        for year_data in yearly_data:
            for point in year_data.hourly_data:
                season = self._get_season(point.timestamp.month)
                seasonal_data[season]["ghi"].append(point.irradiance_ghi)
                seasonal_data[season]["temp"].append(point.temperature)
                seasonal_data[season]["wind"].append(point.wind_speed)

        # Calculate statistics for each season
        stats_by_season = {}
        for season, data in seasonal_data.items():
            stats_by_season[season] = {
                "mean_ghi": float(np.mean(data["ghi"])),
                "std_ghi": float(np.std(data["ghi"])),
                "mean_temperature": float(np.mean(data["temp"])),
                "std_temperature": float(np.std(data["temp"])),
                "mean_wind_speed": float(np.mean(data["wind"])),
                "std_wind_speed": float(np.std(data["wind"])),
            }

        logger.info("Seasonal variability analysis complete")
        return stats_by_season

    def inter_annual_variability(self, yearly_data: List[TMYData]) -> Dict[str, float]:
        """
        Analyze inter-annual variability (year-to-year variation).

        Args:
            yearly_data: List of TMY data for each year

        Returns:
            Dictionary with inter-annual variability metrics
        """
        logger.info("Analyzing inter-annual variability")

        annual_ghi = []
        annual_temp = []

        for year_data in yearly_data:
            total_ghi = sum(point.irradiance_ghi for point in year_data.hourly_data) / 1000.0
            annual_ghi.append(total_ghi)

            avg_temp = np.mean([point.temperature for point in year_data.hourly_data])
            annual_temp.append(avg_temp)

        # Calculate coefficient of variation (CV)
        ghi_cv = float(np.std(annual_ghi) / np.mean(annual_ghi) * 100)
        temp_cv = float(np.std(annual_temp) / np.mean(annual_temp) * 100)

        variability = {
            "ghi_coefficient_of_variation": ghi_cv,
            "temperature_coefficient_of_variation": temp_cv,
            "ghi_range": float(max(annual_ghi) - min(annual_ghi)),
            "temperature_range": float(max(annual_temp) - min(annual_temp)),
            "num_years": len(yearly_data),
        }

        logger.info(f"Inter-annual variability: GHI CV={ghi_cv:.2f}%, Temp CV={temp_cv:.2f}%")
        return variability

    # Helper methods

    def _get_season(self, month: int) -> str:
        """Get season name from month (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _calculate_severity(
        self, value: float, threshold: float, max_value: float
    ) -> int:
        """Calculate severity level (1-5) based on how far value exceeds threshold."""
        if value < threshold:
            return 1

        # Normalize to 1-5 scale
        normalized = (value - threshold) / (max_value - threshold)
        severity = int(min(5, max(1, normalized * 4 + 1)))
        return severity

    def _calculate_impact_score(self, value: float, threshold: float) -> float:
        """Calculate impact score (0-10) based on how far value exceeds threshold."""
        if value < threshold:
            return 0.0

        # Simple linear scaling - could be made more sophisticated
        excess = value - threshold
        impact = min(10.0, excess / threshold * 5.0)
        return float(impact)
