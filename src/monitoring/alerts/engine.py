"""
Alert engine for PV system monitoring.

This module provides the AlertEngine class for detecting and managing alerts
related to underperformance, equipment faults, and grid outages.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict

from config.settings import Settings
from src.core.models.schemas import (
    Alert,
    UnderperformanceAlert,
    EquipmentFaultAlert,
    GridOutageAlert,
    AlertSeverity,
    AlertType,
    InverterData,
    SCADAData,
    PerformanceRatioData
)

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Alert detection and management engine for PV systems.

    This class monitors system performance, equipment health, and grid connectivity
    to detect and generate alerts for various fault conditions and performance issues.

    Attributes:
        settings: Application settings instance
        _active_alerts: Dictionary of currently active alerts
        _alert_history: History of recent alerts for cooldown management
        _alert_callbacks: Registered callback functions for alert notifications
        _suppressed_alerts: Set of suppressed alert IDs
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the AlertEngine.

        Args:
            settings: Application settings instance containing alert thresholds
                     and configuration.
        """
        self.settings = settings
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._alert_callbacks: List[Callable] = []
        self._suppressed_alerts: Set[str] = set()
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._last_alert_time: Dict[str, datetime] = {}

        logger.info("AlertEngine initialized")

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Register a callback function to be called when alerts are generated.

        Args:
            callback: Async function to call with Alert object as parameter.

        Example:
            >>> async def alert_handler(alert: Alert):
            ...     print(f"Alert: {alert.message}")
            >>> engine.register_callback(alert_handler)
        """
        self._alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")

    async def underperformance_detection(
        self,
        performance_data: PerformanceRatioData,
        threshold_pct: Optional[float] = None
    ) -> Optional[UnderperformanceAlert]:
        """
        Detect system underperformance based on Performance Ratio.

        Monitors the Performance Ratio and generates alerts when it falls below
        the configured threshold, indicating underperformance.

        Args:
            performance_data: Performance Ratio data including actual vs expected output
            threshold_pct: Custom threshold percentage (uses config default if None)

        Returns:
            UnderperformanceAlert if underperformance detected, None otherwise.

        Example:
            >>> pr_data = PerformanceRatioData(instantaneous_pr=75.0, ...)
            >>> alert = await engine.underperformance_detection(pr_data)
            >>> if alert:
            ...     print(f"Underperformance detected: {alert.deviation_percentage}%")
        """
        if threshold_pct is None:
            threshold_pct = self.settings.monitoring.underperformance_threshold_pct

        pr = performance_data.instantaneous_pr

        # Check if PR is below threshold
        if pr < threshold_pct:
            # Calculate deviation
            deviation = ((threshold_pct - pr) / threshold_pct) * 100

            # Determine severity based on deviation
            if deviation >= 50:
                severity = AlertSeverity.CRITICAL
            elif deviation >= 30:
                severity = AlertSeverity.HIGH
            elif deviation >= 15:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            # Check cooldown period
            alert_key = f"underperformance_{performance_data.site_id}"
            if not self._check_cooldown(alert_key):
                logger.debug(f"Alert {alert_key} in cooldown period, skipping")
                return None

            # Calculate expected and actual power from energy data
            expected_power = performance_data.expected_energy  # Already in kW from 1hr period
            actual_power = performance_data.actual_energy

            alert = UnderperformanceAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                alert_type=AlertType.UNDERPERFORMANCE,
                severity=severity,
                site_id=performance_data.site_id,
                message=f"System underperformance detected: PR={pr:.1f}% (threshold: {threshold_pct:.1f}%)",
                description=(
                    f"Performance Ratio is {pr:.2f}%, which is {deviation:.1f}% below "
                    f"the threshold of {threshold_pct:.1f}%. "
                    f"Expected power: {expected_power:.2f}kW, Actual: {actual_power:.2f}kW."
                ),
                metric_name="performance_ratio",
                metric_value=pr,
                threshold_value=threshold_pct,
                expected_power=expected_power,
                actual_power=actual_power,
                performance_ratio=pr,
                deviation_percentage=deviation
            )

            await self._process_alert(alert)
            return alert

        return None

    async def equipment_fault_alerts(
        self,
        inverter_data: InverterData
    ) -> Optional[EquipmentFaultAlert]:
        """
        Detect equipment faults from inverter telemetry data.

        Analyzes inverter data for fault conditions including error codes,
        abnormal temperatures, voltage/current anomalies, and status indicators.

        Args:
            inverter_data: Inverter telemetry data to analyze

        Returns:
            EquipmentFaultAlert if fault detected, None otherwise.

        Example:
            >>> inv_data = InverterData(inverter_id='INV001', error_code=42, ...)
            >>> alert = await engine.equipment_fault_alerts(inv_data)
            >>> if alert:
            ...     print(f"Equipment fault: {alert.fault_description}")
        """
        faults = []
        severity = AlertSeverity.MEDIUM

        # Check for error codes
        if inverter_data.error_code is not None and inverter_data.error_code != 0:
            fault_desc = self._get_fault_description(inverter_data.error_code)
            faults.append(f"Error code {inverter_data.error_code}: {fault_desc}")
            severity = AlertSeverity.HIGH

        # Check inverter status
        if inverter_data.status.lower() not in ['online', 'running', 'ok']:
            faults.append(f"Inverter status: {inverter_data.status}")
            severity = AlertSeverity.HIGH if inverter_data.status.lower() in ['fault', 'error'] else AlertSeverity.MEDIUM

        # Check temperature
        temp_threshold = self.settings.monitoring.temperature_alert_threshold_c
        if inverter_data.temperature > temp_threshold:
            faults.append(
                f"High temperature: {inverter_data.temperature:.1f}°C "
                f"(threshold: {temp_threshold:.1f}°C)"
            )
            if inverter_data.temperature > temp_threshold + 10:
                severity = AlertSeverity.CRITICAL

        # Check efficiency
        if inverter_data.efficiency is not None and inverter_data.efficiency < 80:
            faults.append(f"Low efficiency: {inverter_data.efficiency:.1f}%")

        # Check voltage anomalies
        if inverter_data.dc_voltage > 0:
            # Typical DC voltage range: 200-1000V (varies by system)
            if inverter_data.dc_voltage < 200 or inverter_data.dc_voltage > 1000:
                faults.append(f"Abnormal DC voltage: {inverter_data.dc_voltage:.1f}V")
                severity = max(severity, AlertSeverity.HIGH)

        # If faults detected, create alert
        if faults:
            # Check cooldown
            alert_key = f"equipment_fault_{inverter_data.inverter_id}"
            if not self._check_cooldown(alert_key):
                logger.debug(f"Alert {alert_key} in cooldown period, skipping")
                return None

            # Estimate affected capacity (assume each inverter is equal share)
            affected_capacity = inverter_data.ac_power

            alert = EquipmentFaultAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                alert_type=AlertType.EQUIPMENT_FAULT,
                severity=severity,
                site_id=self.settings.site_id,
                component_id=inverter_data.inverter_id,
                component_type="inverter",
                message=f"Equipment fault detected on {inverter_data.inverter_id}",
                description="; ".join(faults),
                fault_code=str(inverter_data.error_code) if inverter_data.error_code else None,
                fault_description="; ".join(faults),
                affected_capacity=affected_capacity
            )

            await self._process_alert(alert)
            return alert

        return None

    async def grid_outage_detection(
        self,
        scada_data: SCADAData,
        previous_scada_data: Optional[SCADAData] = None
    ) -> Optional[GridOutageAlert]:
        """
        Detect grid outage or disconnection events.

        Monitors grid parameters (frequency, voltage) and power export to detect
        grid outages or abnormal grid conditions.

        Args:
            scada_data: Current SCADA data
            previous_scada_data: Previous SCADA data for comparison (optional)

        Returns:
            GridOutageAlert if outage detected, None otherwise.

        Example:
            >>> scada = SCADAData(grid_frequency=0.0, total_ac_power=0.0, ...)
            >>> alert = await engine.grid_outage_detection(scada)
            >>> if alert:
            ...     print(f"Grid outage detected: {alert.message}")
        """
        outage_detected = False
        outage_reason = []

        # Check grid frequency
        if scada_data.grid_frequency is not None:
            freq_min = self.settings.monitoring.grid_frequency_min_hz
            freq_max = self.settings.monitoring.grid_frequency_max_hz

            if scada_data.grid_frequency < freq_min or scada_data.grid_frequency > freq_max:
                outage_reason.append(
                    f"Grid frequency out of range: {scada_data.grid_frequency:.2f}Hz "
                    f"(acceptable: {freq_min}-{freq_max}Hz)"
                )
                outage_detected = True
            elif scada_data.grid_frequency == 0.0:
                outage_reason.append("Grid frequency is zero")
                outage_detected = True

        # Check grid voltage
        if scada_data.grid_voltage is not None and scada_data.grid_voltage == 0.0:
            outage_reason.append("Grid voltage is zero")
            outage_detected = True

        # Check for sudden power drop
        if previous_scada_data:
            # If power dropped by more than 90% suddenly
            if previous_scada_data.total_ac_power > 100:  # Only if we were producing
                power_drop_pct = (
                    (previous_scada_data.total_ac_power - scada_data.total_ac_power) /
                    previous_scada_data.total_ac_power * 100
                )
                if power_drop_pct > 90:
                    outage_reason.append(
                        f"Sudden power drop: {power_drop_pct:.1f}% "
                        f"({previous_scada_data.total_ac_power:.1f}kW → {scada_data.total_ac_power:.1f}kW)"
                    )
                    outage_detected = True

        # Check if all inverters are offline
        if scada_data.available_inverters == 0 and scada_data.total_inverters > 0:
            outage_reason.append("All inverters offline")
            outage_detected = True

        if outage_detected:
            # Check cooldown
            alert_key = f"grid_outage_{scada_data.site_id}"
            if not self._check_cooldown(alert_key):
                logger.debug(f"Alert {alert_key} in cooldown period, skipping")
                return None

            # Estimate energy loss (rough estimate based on capacity and irradiance)
            if scada_data.irradiance > 0:
                # Simple estimation: capacity * (irradiance/1000) * time
                estimated_loss = (
                    self.settings.site_capacity_kw *
                    (scada_data.irradiance / 1000) *
                    0.25  # Assume 15min outage initially
                )
            else:
                estimated_loss = 0.0

            alert = GridOutageAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                alert_type=AlertType.GRID_OUTAGE,
                severity=AlertSeverity.CRITICAL,
                site_id=scada_data.site_id,
                message="Grid outage detected",
                description="; ".join(outage_reason),
                outage_start=datetime.utcnow(),
                affected_power=self.settings.site_capacity_kw,
                estimated_energy_loss=estimated_loss
            )

            await self._process_alert(alert)
            return alert

        return None

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User/system acknowledging the alert

        Returns:
            True if alert was acknowledged, False if not found.
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        logger.warning(f"Alert {alert_id} not found in active alerts")
        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert was resolved, False if not found.
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()

            # Move to history and remove from active
            self._alert_history.append(alert)
            del self._active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved")
            return True

        logger.warning(f"Alert {alert_id} not found in active alerts")
        return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None
    ) -> List[Alert]:
        """
        Get list of active alerts with optional filtering.

        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type

        Returns:
            List of active alerts matching the criteria.
        """
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary containing alert counts by type and severity.
        """
        stats = {
            'total_active': len(self._active_alerts),
            'total_history': len(self._alert_history),
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'unacknowledged': 0
        }

        for alert in self._active_alerts.values():
            stats['by_severity'][alert.severity.value] += 1
            stats['by_type'][alert.alert_type.value] += 1
            if not alert.acknowledged:
                stats['unacknowledged'] += 1

        return dict(stats)

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _process_alert(self, alert: Alert) -> None:
        """
        Process and dispatch a new alert.

        Args:
            alert: Alert to process
        """
        # Check alert rate limiting
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        hour_key = current_hour.isoformat()

        if self._alert_counts[hour_key] >= self.settings.monitoring.max_alerts_per_hour:
            logger.warning(
                f"Alert rate limit exceeded ({self.settings.monitoring.max_alerts_per_hour}/hour), "
                f"suppressing alert: {alert.alert_id}"
            )
            self._suppressed_alerts.add(alert.alert_id)
            return

        self._alert_counts[hour_key] += 1

        # Add to active alerts
        self._active_alerts[alert.alert_id] = alert

        # Update last alert time
        alert_key = f"{alert.alert_type}_{alert.site_id}"
        self._last_alert_time[alert_key] = datetime.utcnow()

        logger.info(
            f"Alert generated: {alert.alert_type.value} - {alert.severity.value} - {alert.message}"
        )

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback {callback.__name__}: {e}", exc_info=True)

    def _check_cooldown(self, alert_key: str) -> bool:
        """
        Check if alert is in cooldown period.

        Args:
            alert_key: Alert key to check

        Returns:
            True if alert can be generated (not in cooldown), False otherwise.
        """
        if alert_key not in self._last_alert_time:
            return True

        last_time = self._last_alert_time[alert_key]
        cooldown = timedelta(seconds=self.settings.monitoring.alert_cooldown_sec)

        return (datetime.utcnow() - last_time) > cooldown

    def _get_fault_description(self, error_code: int) -> str:
        """
        Get fault description from error code.

        Args:
            error_code: Equipment error code

        Returns:
            Human-readable fault description.
        """
        # This is a simplified mapping - in production, this would be
        # a comprehensive lookup table from inverter documentation
        fault_map = {
            1: "Grid overvoltage",
            2: "Grid undervoltage",
            3: "Grid overfrequency",
            4: "Grid underfrequency",
            5: "DC overvoltage",
            6: "DC injection error",
            7: "Ground fault",
            8: "Isolation fault",
            10: "Overtemperature",
            11: "Communication error",
            12: "Hardware fault",
            13: "Fan failure",
            14: "String fault",
            15: "Arc fault detected"
        }

        return fault_map.get(error_code, f"Unknown error (code: {error_code})")
