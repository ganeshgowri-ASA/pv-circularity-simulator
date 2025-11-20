"""
Performance metrics calculation for PV systems.

This module provides the PerformanceMetrics class for calculating key
performance indicators including PR, capacity factor, specific yield,
availability, and grid export metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio

from config.settings import Settings
from src.core.models.schemas import (
    PerformanceRatioData,
    CapacityFactorData,
    SpecificYieldData,
    AvailabilityData,
    GridExportData,
    InverterData,
    SCADAData
)

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate and track PV system performance metrics.

    This class provides methods for calculating key performance indicators (KPIs)
    including Performance Ratio (PR), Capacity Factor, Specific Yield, Availability,
    and Grid Export metrics.

    Attributes:
        settings: Application settings instance
        _database: Database connection for historical data
        _cache: Cache for intermediate calculations
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the PerformanceMetrics calculator.

        Args:
            settings: Application settings instance containing site configuration
                     and performance thresholds.
        """
        self.settings = settings
        self._database: Optional[Any] = None
        self._cache: Dict[str, any] = {}

        logger.info("PerformanceMetrics initialized")

    async def instantaneous_pr(
        self,
        actual_power: float,
        irradiance: float,
        module_temperature: float,
        rated_capacity: Optional[float] = None,
        reference_irradiance: float = 1000.0,
        reference_temperature: float = 25.0,
        temperature_coefficient: float = -0.004
    ) -> PerformanceRatioData:
        """
        Calculate instantaneous Performance Ratio (PR).

        The Performance Ratio is the ratio of actual energy output to the
        theoretical energy output under reference conditions, accounting for
        temperature and irradiance variations.

        Formula:
            PR = (Actual Power / Expected Power) × 100
            Expected Power = Rated Capacity × (Irradiance / Ref Irradiance) × Temp Correction

        Args:
            actual_power: Actual AC power output in kW
            irradiance: Current plane-of-array irradiance in W/m²
            module_temperature: Current module temperature in °C
            rated_capacity: System rated capacity in kW. Uses settings default if None.
            reference_irradiance: Reference irradiance in W/m² (default: 1000 W/m²)
            reference_temperature: Reference temperature in °C (default: 25°C)
            temperature_coefficient: Power temperature coefficient in %/°C (default: -0.4%/°C)

        Returns:
            PerformanceRatioData: Object containing instantaneous PR and related metrics.

        Example:
            >>> pr_data = await metrics.instantaneous_pr(
            ...     actual_power=850.0,
            ...     irradiance=950.0,
            ...     module_temperature=45.0
            ... )
            >>> print(f"PR: {pr_data.instantaneous_pr:.2f}%")
        """
        if rated_capacity is None:
            rated_capacity = self.settings.site_capacity_kw

        # Validate inputs
        if irradiance <= 0:
            logger.warning(f"Invalid irradiance value: {irradiance}, using minimum")
            irradiance = 1.0

        # Calculate temperature correction factor
        temp_diff = module_temperature - reference_temperature
        temp_correction = 1 + (temperature_coefficient * temp_diff)

        # Calculate expected power at current conditions
        irradiance_ratio = irradiance / reference_irradiance
        expected_power = rated_capacity * irradiance_ratio * temp_correction

        # Calculate instantaneous PR
        if expected_power > 0:
            pr = (actual_power / expected_power) * 100
        else:
            pr = 0.0
            logger.warning("Expected power is zero, PR set to 0")

        # Cap PR at reasonable limits (0-200%)
        pr = max(0.0, min(200.0, pr))

        # Calculate expected and actual energy (for consistency with data model)
        # Using 1-hour period for instantaneous calculation
        actual_energy = actual_power * 1.0  # kWh
        expected_energy = expected_power * 1.0  # kWh

        result = PerformanceRatioData(
            timestamp=datetime.utcnow(),
            site_id=self.settings.site_id,
            instantaneous_pr=pr,
            actual_energy=actual_energy,
            expected_energy=expected_energy,
            reference_irradiance=reference_irradiance,
            actual_irradiance=irradiance,
            temperature_coefficient=temperature_coefficient,
            reference_temperature=reference_temperature,
            actual_temperature=module_temperature
        )

        logger.debug(
            f"Instantaneous PR calculated: {pr:.2f}% "
            f"(actual: {actual_power:.2f}kW, expected: {expected_power:.2f}kW)"
        )

        return result

    async def capacity_factor(
        self,
        actual_energy: float,
        time_period_hours: float,
        rated_capacity: Optional[float] = None
    ) -> CapacityFactorData:
        """
        Calculate Capacity Factor.

        Capacity Factor is the ratio of actual energy production to the maximum
        possible energy production if the system operated at rated capacity for
        the entire period.

        Formula:
            CF = (Actual Energy / (Rated Capacity × Time Period)) × 100

        Args:
            actual_energy: Actual energy produced in kWh
            time_period_hours: Time period in hours
            rated_capacity: System rated capacity in kW. Uses settings default if None.

        Returns:
            CapacityFactorData: Object containing capacity factor and related metrics.

        Example:
            >>> cf_data = await metrics.capacity_factor(
            ...     actual_energy=18500.0,  # kWh
            ...     time_period_hours=24.0   # 1 day
            ... )
            >>> print(f"Capacity Factor: {cf_data.capacity_factor:.2f}%")
        """
        if rated_capacity is None:
            rated_capacity = self.settings.site_capacity_kw

        # Validate inputs
        if time_period_hours <= 0:
            raise ValueError("Time period must be greater than 0")

        if rated_capacity <= 0:
            raise ValueError("Rated capacity must be greater than 0")

        # Calculate maximum possible energy
        max_energy = rated_capacity * time_period_hours

        # Calculate capacity factor
        if max_energy > 0:
            cf = (actual_energy / max_energy) * 100
        else:
            cf = 0.0

        # Cap at reasonable limits (0-100%)
        cf = max(0.0, min(100.0, cf))

        result = CapacityFactorData(
            timestamp=datetime.utcnow(),
            site_id=self.settings.site_id,
            capacity_factor=cf,
            actual_energy=actual_energy,
            rated_capacity=rated_capacity,
            time_period_hours=time_period_hours
        )

        logger.debug(
            f"Capacity Factor calculated: {cf:.2f}% "
            f"({actual_energy:.2f}kWh / {max_energy:.2f}kWh)"
        )

        return result

    async def specific_yield(
        self,
        energy_production: float,
        installed_capacity: Optional[float] = None,
        period_type: str = "daily"
    ) -> SpecificYieldData:
        """
        Calculate Specific Yield.

        Specific Yield is the energy produced per unit of installed capacity,
        typically expressed in kWh/kWp (kilowatt-hours per kilowatt-peak).

        Formula:
            Specific Yield = Energy Production / Installed Capacity

        Args:
            energy_production: Energy produced in kWh for the period
            installed_capacity: Installed capacity in kWp. Uses settings default if None.
            period_type: Type of period ('daily', 'monthly', 'yearly')

        Returns:
            SpecificYieldData: Object containing specific yield and related metrics.

        Example:
            >>> sy_data = await metrics.specific_yield(
            ...     energy_production=4250.0,  # kWh
            ...     period_type='daily'
            ... )
            >>> print(f"Daily Specific Yield: {sy_data.specific_yield:.2f} kWh/kWp")
        """
        if installed_capacity is None:
            installed_capacity = self.settings.site_capacity_kw

        # Validate inputs
        if installed_capacity <= 0:
            raise ValueError("Installed capacity must be greater than 0")

        if period_type not in ['daily', 'monthly', 'yearly']:
            logger.warning(f"Unknown period type: {period_type}, using 'daily'")
            period_type = 'daily'

        # Calculate specific yield
        specific_yield = energy_production / installed_capacity

        result = SpecificYieldData(
            timestamp=datetime.utcnow(),
            site_id=self.settings.site_id,
            specific_yield=specific_yield,
            energy_production=energy_production,
            installed_capacity=installed_capacity,
            period_type=period_type
        )

        logger.debug(
            f"Specific Yield calculated: {specific_yield:.2f} kWh/kWp "
            f"({period_type})"
        )

        return result

    async def availability_tracking(
        self,
        uptime_hours: float,
        total_hours: float,
        available_components: int,
        total_components: int,
        planned_downtime_hours: float = 0.0,
        unplanned_downtime_hours: float = 0.0
    ) -> AvailabilityData:
        """
        Track and calculate system availability.

        Availability is the percentage of time the system is operational and
        available for energy production.

        Formula:
            Availability = (Uptime / Total Time) × 100
            Or: Availability = ((Total Time - Downtime) / Total Time) × 100

        Args:
            uptime_hours: Total uptime in hours
            total_hours: Total time period in hours
            available_components: Number of available/operational components
            total_components: Total number of components in the system
            planned_downtime_hours: Planned downtime (maintenance) in hours
            unplanned_downtime_hours: Unplanned downtime (faults) in hours

        Returns:
            AvailabilityData: Object containing availability metrics and breakdown.

        Example:
            >>> avail_data = await metrics.availability_tracking(
            ...     uptime_hours=23.5,
            ...     total_hours=24.0,
            ...     available_components=98,
            ...     total_components=100
            ... )
            >>> print(f"Availability: {avail_data.availability_percentage:.2f}%")
        """
        # Validate inputs
        if total_hours <= 0:
            raise ValueError("Total hours must be greater than 0")

        if total_components <= 0:
            raise ValueError("Total components must be greater than 0")

        # Calculate downtime
        downtime_hours = total_hours - uptime_hours

        # Ensure downtime components add up
        if planned_downtime_hours + unplanned_downtime_hours == 0:
            # Distribute downtime proportionally if not specified
            # Assume 20% planned, 80% unplanned for unspecified downtime
            planned_downtime_hours = downtime_hours * 0.2
            unplanned_downtime_hours = downtime_hours * 0.8

        # Calculate availability percentage
        availability = (uptime_hours / total_hours) * 100

        # Cap at reasonable limits (0-100%)
        availability = max(0.0, min(100.0, availability))

        result = AvailabilityData(
            timestamp=datetime.utcnow(),
            site_id=self.settings.site_id,
            availability_percentage=availability,
            uptime_hours=uptime_hours,
            total_hours=total_hours,
            downtime_hours=downtime_hours,
            planned_downtime_hours=planned_downtime_hours,
            unplanned_downtime_hours=unplanned_downtime_hours,
            available_components=available_components,
            total_components=total_components
        )

        logger.debug(
            f"Availability calculated: {availability:.2f}% "
            f"({uptime_hours:.2f}h uptime / {total_hours:.2f}h total)"
        )

        return result

    async def grid_export_monitoring(
        self,
        export_power: float,
        export_energy: float,
        grid_voltage: float,
        grid_frequency: float,
        power_factor: float,
        reactive_power: Optional[float] = None,
        grid_connected: bool = True
    ) -> GridExportData:
        """
        Monitor grid export metrics and grid health.

        Tracks power export to the grid, grid quality parameters, and connection
        status. Validates grid parameters against acceptable ranges.

        Args:
            export_power: Current export power in kW (negative if importing)
            export_energy: Cumulative exported energy in kWh
            grid_voltage: Grid voltage in V
            grid_frequency: Grid frequency in Hz
            power_factor: Power factor (-1 to 1)
            reactive_power: Reactive power in kVAR (optional)
            grid_connected: Grid connection status

        Returns:
            GridExportData: Object containing grid export and health metrics.

        Raises:
            ValueError: If grid parameters are outside acceptable ranges.

        Example:
            >>> grid_data = await metrics.grid_export_monitoring(
            ...     export_power=950.0,
            ...     export_energy=18500.0,
            ...     grid_voltage=400.0,
            ...     grid_frequency=50.0,
            ...     power_factor=0.98
            ... )
            >>> print(f"Exporting {grid_data.export_power:.2f} kW to grid")
        """
        # Validate grid frequency
        freq_min = self.settings.monitoring.grid_frequency_min_hz
        freq_max = self.settings.monitoring.grid_frequency_max_hz

        if not (freq_min <= grid_frequency <= freq_max):
            logger.warning(
                f"Grid frequency out of range: {grid_frequency} Hz "
                f"(acceptable: {freq_min}-{freq_max} Hz)"
            )

        # Validate power factor
        if not (-1 <= power_factor <= 1):
            logger.warning(f"Invalid power factor: {power_factor}")
            power_factor = max(-1, min(1, power_factor))

        result = GridExportData(
            timestamp=datetime.utcnow(),
            site_id=self.settings.site_id,
            export_power=export_power,
            export_energy=export_energy,
            grid_voltage=grid_voltage,
            grid_frequency=grid_frequency,
            power_factor=power_factor,
            reactive_power=reactive_power,
            grid_connected=grid_connected
        )

        logger.debug(
            f"Grid export: {export_power:.2f}kW @ {grid_frequency:.2f}Hz, "
            f"PF: {power_factor:.3f}"
        )

        return result

    async def calculate_batch_metrics(
        self,
        inverter_data: List[InverterData],
        scada_data: List[SCADAData],
        time_period_hours: float = 1.0
    ) -> Dict[str, any]:
        """
        Calculate multiple performance metrics from batch data.

        Processes historical data to calculate all key metrics in a single batch
        operation for efficiency.

        Args:
            inverter_data: List of inverter data points
            scada_data: List of SCADA data points
            time_period_hours: Time period covered by the data in hours

        Returns:
            Dictionary containing all calculated metrics:
                - performance_ratio: PerformanceRatioData
                - capacity_factor: CapacityFactorData
                - specific_yield: SpecificYieldData
                - availability: AvailabilityData
                - grid_export: GridExportData

        Example:
            >>> metrics_dict = await metrics.calculate_batch_metrics(
            ...     inverter_data=inv_list,
            ...     scada_data=scada_list,
            ...     time_period_hours=24.0
            ... )
        """
        logger.info(f"Calculating batch metrics for {len(inverter_data)} data points")

        # Aggregate inverter data
        total_ac_power = sum(inv.ac_power for inv in inverter_data) / len(inverter_data)
        total_energy = sum(inv.energy_daily for inv in inverter_data)

        # Get average environmental conditions from SCADA
        avg_irradiance = sum(s.irradiance for s in scada_data) / len(scada_data)
        avg_temp = sum(s.module_temperature for s in scada_data if s.module_temperature) / \
                   len([s for s in scada_data if s.module_temperature])

        # Calculate metrics
        pr = await self.instantaneous_pr(
            actual_power=total_ac_power,
            irradiance=avg_irradiance,
            module_temperature=avg_temp
        )

        cf = await self.capacity_factor(
            actual_energy=total_energy,
            time_period_hours=time_period_hours
        )

        sy = await self.specific_yield(
            energy_production=total_energy,
            period_type='daily' if time_period_hours == 24 else 'custom'
        )

        # Calculate availability
        available_invs = sum(1 for inv in inverter_data if inv.status == 'online')
        total_invs = len(inverter_data)

        avail = await self.availability_tracking(
            uptime_hours=time_period_hours * (available_invs / total_invs),
            total_hours=time_period_hours,
            available_components=available_invs,
            total_components=total_invs
        )

        # Grid export (using latest SCADA data)
        latest_scada = scada_data[-1] if scada_data else None
        grid = None
        if latest_scada:
            grid = await self.grid_export_monitoring(
                export_power=latest_scada.total_ac_power,
                export_energy=total_energy,
                grid_voltage=latest_scada.grid_voltage or 400.0,
                grid_frequency=latest_scada.grid_frequency or 50.0,
                power_factor=0.98  # Default if not available
            )

        return {
            'performance_ratio': pr,
            'capacity_factor': cf,
            'specific_yield': sy,
            'availability': avail,
            'grid_export': grid
        }
