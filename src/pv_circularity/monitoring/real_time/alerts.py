"""
Alert engine for PV monitoring systems.

This module provides alert detection and notification for underperformance,
equipment faults, and grid outages.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict

from pv_circularity.core import get_logger, AlertError
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.monitoring import (
    AlertData,
    AlertType,
    AlertSeverity,
    MonitoringDataPoint,
    InverterData,
)

logger = get_logger(__name__)


class AlertEngine:
    """
    Alert detection and notification engine.

    Monitors system performance and equipment status to detect and alert on
    anomalies, faults, and outages.

    Args:
        underperformance_threshold: Percentage below expected for underperformance alert
        offline_timeout: Seconds before device considered offline
        check_interval: Interval between alert checks in seconds

    Example:
        >>> engine = AlertEngine(underperformance_threshold=15.0)
        >>> await engine.start()
        >>> engine.subscribe_alerts(alert_callback)
        >>> # Alert callbacks will be triggered automatically
        >>> await engine.stop()
    """

    def __init__(
        self,
        underperformance_threshold: float = 15.0,
        offline_timeout: int = 300,
        check_interval: int = 30,
    ) -> None:
        """
        Initialize alert engine.

        Args:
            underperformance_threshold: Underperformance threshold percentage
            offline_timeout: Device offline timeout in seconds
            check_interval: Alert check interval in seconds
        """
        self.underperformance_threshold = underperformance_threshold
        self.offline_timeout = offline_timeout
        self.check_interval = check_interval

        self._active_alerts: Dict[str, AlertData] = {}
        self._alert_history: List[AlertData] = []
        self._subscribers: List[Callable] = []

        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Track last seen timestamps for devices
        self._device_last_seen: Dict[str, datetime] = {}

        logger.info(
            "AlertEngine initialized",
            underperformance_threshold=underperformance_threshold,
            offline_timeout=offline_timeout,
        )

    async def start(self) -> None:
        """Start the alert engine."""
        if self._is_running:
            logger.warning("Alert engine already running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Alert engine started")

    async def stop(self) -> None:
        """Stop the alert engine."""
        if not self._is_running:
            return

        logger.info("Stopping alert engine")

        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert engine stopped")

    async def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        while self._is_running:
            try:
                # Check for communication loss
                await self._check_communication_loss()

                # Other checks would be triggered by data updates
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error("Error in alert monitoring loop", error=str(e), exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def underperformance_detection(
        self,
        site_id: str,
        device_id: str,
        actual_power: float,
        expected_power: float,
    ) -> Optional[AlertData]:
        """
        Detect underperformance conditions.

        Triggers an alert if actual power is significantly below expected power.

        Args:
            site_id: Site identifier
            device_id: Device identifier
            actual_power: Actual measured power (kW)
            expected_power: Expected power based on conditions (kW)

        Returns:
            AlertData if underperformance detected, None otherwise

        Example:
            >>> alert = await engine.underperformance_detection(
            ...     site_id="SITE001",
            ...     device_id="INV001",
            ...     actual_power=85,
            ...     expected_power=100
            ... )
        """
        if expected_power <= 0:
            return None

        # Calculate performance percentage
        performance_pct = (actual_power / expected_power) * 100
        underperformance = 100 - performance_pct

        if underperformance >= self.underperformance_threshold:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.UNDERPERFORMANCE,
                severity=AlertSeverity.WARNING
                if underperformance < 30
                else AlertSeverity.ERROR,
                message=f"Device underperforming by {underperformance:.1f}%",
                details={
                    "actual_power": actual_power,
                    "expected_power": expected_power,
                    "performance_pct": performance_pct,
                    "underperformance_pct": underperformance,
                },
            )

            logger.warning(
                "Underperformance detected",
                site_id=site_id,
                device_id=device_id,
                underperformance=underperformance,
            )

            return alert

        # Clear alert if performance is back to normal
        await self._clear_alert(device_id, AlertType.UNDERPERFORMANCE)

        return None

    async def equipment_fault_alerts(
        self,
        site_id: str,
        device_id: str,
        inverter_data: InverterData,
    ) -> List[AlertData]:
        """
        Detect equipment faults from inverter data.

        Checks for various fault conditions including alarms, abnormal values,
        and device status.

        Args:
            site_id: Site identifier
            device_id: Device identifier
            inverter_data: Inverter data object

        Returns:
            List of alert data objects for detected faults

        Example:
            >>> alerts = await engine.equipment_fault_alerts(
            ...     site_id="SITE001",
            ...     device_id="INV001",
            ...     inverter_data=inv_data
            ... )
        """
        alerts = []

        # Check for active alarms
        if inverter_data.alarms:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.EQUIPMENT_FAULT,
                severity=AlertSeverity.ERROR,
                message=f"Equipment alarms active: {', '.join(inverter_data.alarms)}",
                details={"alarms": inverter_data.alarms},
            )
            alerts.append(alert)

        # Check for offline/error status
        if inverter_data.status in ["offline", "error"]:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.EQUIPMENT_FAULT,
                severity=AlertSeverity.CRITICAL,
                message=f"Device status: {inverter_data.status}",
                details={"status": inverter_data.status},
            )
            alerts.append(alert)

        # Check for abnormal temperature
        if inverter_data.temperature and inverter_data.temperature > 85:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.EQUIPMENT_FAULT,
                severity=AlertSeverity.WARNING,
                message=f"High temperature: {inverter_data.temperature:.1f}°C",
                details={"temperature": inverter_data.temperature},
            )
            alerts.append(alert)

        # Check for low efficiency
        if inverter_data.efficiency and inverter_data.efficiency < 0.85:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.EQUIPMENT_FAULT,
                severity=AlertSeverity.WARNING,
                message=f"Low efficiency: {inverter_data.efficiency*100:.1f}%",
                details={"efficiency": inverter_data.efficiency},
            )
            alerts.append(alert)

        if not alerts:
            # Clear equipment fault alerts if no issues found
            await self._clear_alert(device_id, AlertType.EQUIPMENT_FAULT)

        return alerts

    async def grid_outage_detection(
        self,
        site_id: str,
        device_id: str,
        ac_voltage: Optional[float],
        frequency: Optional[float],
    ) -> Optional[AlertData]:
        """
        Detect grid outage conditions.

        Monitors AC voltage and frequency for grid connection issues.

        Args:
            site_id: Site identifier
            device_id: Device identifier
            ac_voltage: AC voltage (V)
            frequency: AC frequency (Hz)

        Returns:
            AlertData if grid outage detected, None otherwise

        Example:
            >>> alert = await engine.grid_outage_detection(
            ...     site_id="SITE001",
            ...     device_id="INV001",
            ...     ac_voltage=0,
            ...     frequency=0
            ... )
        """
        grid_issue = False
        issue_details = {}

        # Check voltage
        if ac_voltage is not None and (ac_voltage < 180 or ac_voltage > 260):
            grid_issue = True
            issue_details["ac_voltage"] = ac_voltage
            issue_details["voltage_status"] = "out_of_range"

        # Check frequency (typical range: 49-51 Hz)
        if frequency is not None and (frequency < 49 or frequency > 51):
            grid_issue = True
            issue_details["frequency"] = frequency
            issue_details["frequency_status"] = "out_of_range"

        if grid_issue:
            alert = await self._create_alert(
                site_id=site_id,
                device_id=device_id,
                alert_type=AlertType.GRID_OUTAGE,
                severity=AlertSeverity.CRITICAL,
                message="Grid parameters out of range",
                details=issue_details,
            )

            logger.critical(
                "Grid outage detected",
                site_id=site_id,
                device_id=device_id,
                **issue_details,
            )

            return alert

        # Clear alert if grid is back to normal
        await self._clear_alert(device_id, AlertType.GRID_OUTAGE)

        return None

    async def _check_communication_loss(self) -> None:
        """Check for communication loss with devices."""
        now = get_utc_now()

        for device_id, last_seen in self._device_last_seen.items():
            time_since_last = (now - last_seen).total_seconds()

            if time_since_last > self.offline_timeout:
                alert_key = f"{device_id}_{AlertType.COMMUNICATION_LOSS}"

                if alert_key not in self._active_alerts:
                    alert = await self._create_alert(
                        site_id="unknown",
                        device_id=device_id,
                        alert_type=AlertType.COMMUNICATION_LOSS,
                        severity=AlertSeverity.ERROR,
                        message=f"Communication lost for {int(time_since_last)}s",
                        details={"last_seen": last_seen.isoformat()},
                    )

                    logger.error(
                        "Communication loss detected",
                        device_id=device_id,
                        time_since_last=time_since_last,
                    )

    def update_device_timestamp(self, device_id: str) -> None:
        """Update last seen timestamp for a device."""
        self._device_last_seen[device_id] = get_utc_now()

    async def _create_alert(
        self,
        site_id: str,
        device_id: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any],
    ) -> AlertData:
        """Create and register a new alert."""
        alert_key = f"{device_id}_{alert_type}"

        # Check if alert already exists
        if alert_key in self._active_alerts:
            return self._active_alerts[alert_key]

        # Create new alert
        alert = AlertData(
            site_id=site_id,
            device_id=device_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
            timestamp=get_utc_now(),
        )

        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)

        # Notify subscribers
        await self._notify_subscribers(alert)

        logger.info(
            "Alert created",
            alert_id=str(alert.alert_id),
            device_id=device_id,
            alert_type=alert_type,
            severity=severity,
        )

        return alert

    async def _clear_alert(self, device_id: str, alert_type: AlertType) -> None:
        """Clear an active alert."""
        alert_key = f"{device_id}_{alert_type}"

        if alert_key in self._active_alerts:
            alert = self._active_alerts[alert_key]
            alert.resolved_at = get_utc_now()
            alert.is_active = False
            del self._active_alerts[alert_key]

            logger.info(
                "Alert cleared",
                alert_id=str(alert.alert_id),
                device_id=device_id,
                alert_type=alert_type,
            )

    async def _notify_subscribers(self, alert: AlertData) -> None:
        """Notify all subscribers of a new alert."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Error notifying alert subscriber", error=str(e))

    def subscribe_alerts(self, callback: Callable) -> None:
        """
        Subscribe to alert notifications.

        Args:
            callback: Callback function to receive alerts
                     Signature: callback(alert: AlertData)

        Example:
            >>> def on_alert(alert):
            ...     print(f"Alert: {alert.message}")
            >>> engine.subscribe_alerts(on_alert)
        """
        self._subscribers.append(callback)
        logger.debug("Alert subscriber added", total_subscribers=len(self._subscribers))

    def unsubscribe_alerts(self, callback: Callable) -> None:
        """Unsubscribe from alert notifications."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug("Alert subscriber removed", total_subscribers=len(self._subscribers))

    def get_active_alerts(self, site_id: Optional[str] = None) -> List[AlertData]:
        """
        Get all active alerts, optionally filtered by site.

        Args:
            site_id: Site ID to filter by (None for all sites)

        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())

        if site_id:
            alerts = [a for a in alerts if a.site_id == site_id]

        return alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for alert in self._active_alerts.values():
            severity_counts[alert.severity] += 1
            type_counts[alert.alert_type] += 1

        return {
            "total_active": len(self._active_alerts),
            "total_history": len(self._alert_history),
            "by_severity": dict(severity_counts),
            "by_type": dict(type_counts),
        }
