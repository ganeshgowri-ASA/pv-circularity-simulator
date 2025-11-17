"""
Performance metrics calculation for PV systems.

This module calculates key performance indicators including performance ratio,
capacity factor, specific yield, availability tracking, and grid export monitoring.
"""

from typing import Optional, List
from datetime import datetime, timedelta

import numpy as np

from pv_circularity.core import get_logger, MonitoringError
from pv_circularity.core.utils import get_utc_now, safe_divide
from pv_circularity.models.monitoring import PerformanceMetrics as PerformanceMetricsModel

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculate PV system performance metrics.

    Provides methods to calculate instantaneous and time-averaged performance
    indicators including PR, capacity factor, specific yield, and availability.

    Example:
        >>> metrics = PerformanceMetrics()
        >>> pr = await metrics.instantaneous_pr(
        ...     actual_power=500,
        ...     rated_power=600,
        ...     irradiance=800,
        ...     reference_irradiance=1000
        ... )
        >>> print(f"Performance Ratio: {pr:.2%}")
    """

    def __init__(self) -> None:
        """Initialize performance metrics calculator."""
        logger.info("PerformanceMetrics initialized")

    async def instantaneous_pr(
        self,
        actual_power: float,
        rated_power: float,
        irradiance: float,
        reference_irradiance: float = 1000.0,
        temperature: Optional[float] = None,
        reference_temperature: float = 25.0,
        temperature_coefficient: float = -0.004,
    ) -> float:
        """
        Calculate instantaneous performance ratio.

        PR = (Actual Power / Rated Power) / (Irradiance / Reference Irradiance)

        Args:
            actual_power: Actual measured power (kW)
            rated_power: Rated system power (kWp)
            irradiance: Actual irradiance (W/m²)
            reference_irradiance: Reference irradiance (W/m²), typically 1000
            temperature: Module temperature (°C), optional
            reference_temperature: Reference temperature (°C), typically 25
            temperature_coefficient: Temperature coefficient (%/°C)

        Returns:
            Performance ratio (0-1+)

        Example:
            >>> pr = await metrics.instantaneous_pr(
            ...     actual_power=500,
            ...     rated_power=600,
            ...     irradiance=800
            ... )
        """
        if irradiance <= 0:
            return 0.0

        # Expected power at current irradiance
        expected_power = rated_power * (irradiance / reference_irradiance)

        # Temperature correction if provided
        if temperature is not None:
            temp_diff = temperature - reference_temperature
            temp_factor = 1 + (temperature_coefficient * temp_diff)
            expected_power *= temp_factor

        # Calculate PR
        pr = safe_divide(actual_power, expected_power, default=0.0)

        logger.debug(
            "Calculated instantaneous PR",
            pr=pr,
            actual_power=actual_power,
            expected_power=expected_power,
        )

        return pr

    async def capacity_factor(
        self,
        actual_energy: float,
        rated_power: float,
        period_hours: float,
    ) -> float:
        """
        Calculate capacity factor.

        CF = Actual Energy / (Rated Power × Period Hours)

        Args:
            actual_energy: Actual energy produced (kWh)
            rated_power: Rated system power (kWp)
            period_hours: Time period in hours

        Returns:
            Capacity factor (0-1)

        Example:
            >>> cf = await metrics.capacity_factor(
            ...     actual_energy=5000,
            ...     rated_power=1000,
            ...     period_hours=24
            ... )
        """
        theoretical_max = rated_power * period_hours
        cf = safe_divide(actual_energy, theoretical_max, default=0.0)

        logger.debug(
            "Calculated capacity factor",
            cf=cf,
            actual_energy=actual_energy,
            theoretical_max=theoretical_max,
        )

        return min(cf, 1.0)  # Cap at 1.0

    async def specific_yield(
        self,
        actual_energy: float,
        rated_power: float,
    ) -> float:
        """
        Calculate specific yield.

        Specific Yield = Actual Energy / Rated Power (kWh/kWp)

        Args:
            actual_energy: Actual energy produced (kWh)
            rated_power: Rated system power (kWp)

        Returns:
            Specific yield (kWh/kWp)

        Example:
            >>> sy = await metrics.specific_yield(
            ...     actual_energy=5000,
            ...     rated_power=1000
            ... )
        """
        sy = safe_divide(actual_energy, rated_power, default=0.0)

        logger.debug(
            "Calculated specific yield",
            specific_yield=sy,
            actual_energy=actual_energy,
            rated_power=rated_power,
        )

        return sy

    async def availability_tracking(
        self,
        total_time: float,
        downtime: float,
    ) -> float:
        """
        Calculate system availability.

        Availability = (Total Time - Downtime) / Total Time

        Args:
            total_time: Total time period (hours)
            downtime: Downtime period (hours)

        Returns:
            Availability (0-1)

        Example:
            >>> avail = await metrics.availability_tracking(
            ...     total_time=24,
            ...     downtime=0.5
            ... )
        """
        if total_time <= 0:
            return 0.0

        uptime = max(0, total_time - downtime)
        availability = safe_divide(uptime, total_time, default=0.0)

        logger.debug(
            "Calculated availability",
            availability=availability,
            uptime=uptime,
            total_time=total_time,
        )

        return min(availability, 1.0)  # Cap at 1.0

    async def grid_export_monitoring(
        self,
        gross_production: float,
        self_consumption: float,
    ) -> dict:
        """
        Monitor grid export statistics.

        Args:
            gross_production: Total energy produced (kWh)
            self_consumption: Energy consumed on-site (kWh)

        Returns:
            Dictionary with grid export metrics

        Example:
            >>> export_data = await metrics.grid_export_monitoring(
            ...     gross_production=1000,
            ...     self_consumption=200
            ... )
        """
        grid_export = max(0, gross_production - self_consumption)
        export_ratio = safe_divide(grid_export, gross_production, default=0.0)
        self_consumption_ratio = safe_divide(self_consumption, gross_production, default=0.0)

        result = {
            "grid_export_energy": grid_export,
            "self_consumption_energy": self_consumption,
            "gross_production": gross_production,
            "export_ratio": export_ratio,
            "self_consumption_ratio": self_consumption_ratio,
        }

        logger.debug("Calculated grid export metrics", **result)

        return result

    async def calculate_period_metrics(
        self,
        site_id: str,
        period_start: datetime,
        period_end: datetime,
        actual_energy: float,
        expected_energy: float,
        rated_power: float,
        avg_irradiance: Optional[float] = None,
        avg_temperature: Optional[float] = None,
        downtime_hours: float = 0.0,
        grid_export_energy: Optional[float] = None,
    ) -> PerformanceMetricsModel:
        """
        Calculate comprehensive performance metrics for a time period.

        Args:
            site_id: Site identifier
            period_start: Period start timestamp
            period_end: Period end timestamp
            actual_energy: Actual energy produced (kWh)
            expected_energy: Expected energy production (kWh)
            rated_power: Rated system power (kWp)
            avg_irradiance: Average irradiance (W/m²)
            avg_temperature: Average temperature (°C)
            downtime_hours: Total downtime (hours)
            grid_export_energy: Energy exported to grid (kWh)

        Returns:
            PerformanceMetrics model with all calculated metrics

        Example:
            >>> metrics_data = await metrics.calculate_period_metrics(
            ...     site_id="SITE001",
            ...     period_start=start_time,
            ...     period_end=end_time,
            ...     actual_energy=5000,
            ...     expected_energy=5500,
            ...     rated_power=1000
            ... )
        """
        # Calculate time period in hours
        period_hours = (period_end - period_start).total_seconds() / 3600

        # Calculate performance ratio
        pr = safe_divide(actual_energy, expected_energy, default=0.0)

        # Calculate capacity factor
        cf = await self.capacity_factor(actual_energy, rated_power, period_hours)

        # Calculate specific yield
        sy = await self.specific_yield(actual_energy, rated_power)

        # Calculate availability
        availability = await self.availability_tracking(period_hours, downtime_hours)

        # Create metrics model
        metrics = PerformanceMetricsModel(
            site_id=site_id,
            timestamp=get_utc_now(),
            period_start=period_start,
            period_end=period_end,
            performance_ratio=pr,
            capacity_factor=cf,
            specific_yield=sy,
            availability=availability,
            actual_energy=actual_energy,
            expected_energy=expected_energy,
            grid_export_energy=grid_export_energy,
            avg_irradiance=avg_irradiance,
            avg_temperature=avg_temperature,
        )

        logger.info(
            "Calculated period metrics",
            site_id=site_id,
            pr=pr,
            cf=cf,
            sy=sy,
            availability=availability,
        )

        return metrics
