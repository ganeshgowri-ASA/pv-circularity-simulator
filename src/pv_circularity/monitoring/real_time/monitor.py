"""
Real-time monitoring system for PV installations.

This module provides live data streaming, SCADA integration, inverter data parsing,
and multi-level monitoring (string-level, module-level).
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from collections import deque

from pv_circularity.core import get_logger, MonitoringError
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import SCADADevice
from pv_circularity.models.monitoring import (
    MonitoringDataPoint,
    InverterData,
    StringLevelData,
    ModuleLevelData,
)
from pv_circularity.monitoring.scada import SCADAConnector

logger = get_logger(__name__)


class RealTimeMonitor:
    """
    Real-time monitoring system with live data streaming and SCADA integration.

    Provides continuous monitoring of PV installations with support for
    inverter-level, string-level, and module-level monitoring.

    Args:
        devices: List of SCADA devices to monitor
        update_interval: Data update interval in seconds
        buffer_size: Size of data buffer for streaming

    Example:
        >>> monitor = RealTimeMonitor(devices, update_interval=5)
        >>> await monitor.start_monitoring()
        >>> # In another task:
        >>> async for data in monitor.live_data_stream():
        ...     print(f"Received {len(data)} data points")
    """

    def __init__(
        self,
        devices: List[SCADADevice],
        update_interval: int = 5,
        buffer_size: int = 1000,
    ) -> None:
        """
        Initialize real-time monitor.

        Args:
            devices: List of SCADA devices
            update_interval: Update interval in seconds
            buffer_size: Data buffer size
        """
        self.scada_connector = SCADAConnector(devices)
        self.update_interval = update_interval
        self.buffer_size = buffer_size

        self._data_buffer: deque = deque(maxlen=buffer_size)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._subscribers: List[Callable] = []

        logger.info(
            "RealTimeMonitor initialized",
            devices=len(devices),
            update_interval=update_interval,
        )

    async def start_monitoring(self) -> None:
        """
        Start real-time monitoring.

        Initiates continuous data collection from all configured devices.

        Raises:
            MonitoringError: If monitoring startup fails
        """
        if self._is_monitoring:
            logger.warning("Monitoring already started")
            return

        try:
            # Connect to SCADA devices
            await self.scada_connector.connect()

            # Start monitoring task
            self._is_monitoring = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Real-time monitoring started")

        except Exception as e:
            logger.error("Failed to start monitoring", error=str(e), exc_info=True)
            raise MonitoringError(
                f"Failed to start monitoring: {str(e)}",
                original_exception=e,
            )

    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self._is_monitoring:
            return

        logger.info("Stopping real-time monitoring")

        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        await self.scada_connector.disconnect()

        logger.info("Real-time monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects data periodically."""
        while self._is_monitoring:
            try:
                # Collect data from all devices
                timestamp = get_utc_now()
                device_data = await self.scada_connector.read_all_devices(timestamp)

                # Flatten data and add to buffer
                all_data_points = []
                for device_id, data_points in device_data.items():
                    all_data_points.extend(data_points)

                self._data_buffer.extend(all_data_points)

                # Notify subscribers
                await self._notify_subscribers(all_data_points)

                logger.debug(
                    "Data collection cycle complete",
                    data_points=len(all_data_points),
                    devices=len(device_data),
                )

            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e), exc_info=True)

            # Wait for next update interval
            await asyncio.sleep(self.update_interval)

    async def _notify_subscribers(self, data_points: List[MonitoringDataPoint]) -> None:
        """Notify all subscribers of new data."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_points)
                else:
                    callback(data_points)
            except Exception as e:
                logger.error("Error notifying subscriber", error=str(e))

    async def live_data_stream(self) -> List[MonitoringDataPoint]:
        """
        Stream live monitoring data.

        Yields batches of monitoring data points as they become available.

        Yields:
            List of monitoring data points

        Example:
            >>> async for data_batch in monitor.live_data_stream():
            ...     process_data(data_batch)
        """
        last_processed = 0

        while True:
            if not self._is_monitoring:
                break

            # Get new data from buffer
            current_size = len(self._data_buffer)
            if current_size > last_processed:
                new_data = list(self._data_buffer)[last_processed:]
                last_processed = current_size
                yield new_data

            await asyncio.sleep(0.1)  # Small delay to prevent tight loop

    def subscribe(self, callback: Callable) -> None:
        """
        Subscribe to data updates.

        Args:
            callback: Callback function to receive data updates
                     Signature: callback(data_points: List[MonitoringDataPoint])

        Example:
            >>> def on_data(data_points):
            ...     print(f"Received {len(data_points)} points")
            >>> monitor.subscribe(on_data)
        """
        self._subscribers.append(callback)
        logger.debug("Subscriber added", total_subscribers=len(self._subscribers))

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from data updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug("Subscriber removed", total_subscribers=len(self._subscribers))

    async def scada_integration(self, device_id: str) -> Dict[str, Any]:
        """
        Get SCADA integration status for a device.

        Args:
            device_id: Device identifier

        Returns:
            Dictionary with SCADA connection and data status
        """
        status = self.scada_connector.get_status()
        device_status = status.get(device_id)

        if not device_status:
            return {"connected": False, "error": "Device not found"}

        # Get recent data for device
        recent_data = [dp for dp in self._data_buffer if dp.device_id == device_id]

        return {
            "connected": device_status.get("connected", False),
            "protocol": device_status.get("protocol"),
            "device_type": device_status.get("device_type"),
            "recent_data_points": len(recent_data),
            "last_update": recent_data[-1].timestamp if recent_data else None,
        }

    async def inverter_data_parsing(self, device_id: str) -> Optional[InverterData]:
        """
        Parse inverter-specific data from monitoring data points.

        Args:
            device_id: Inverter device identifier

        Returns:
            InverterData object or None if not available
        """
        # Get recent data for this device
        device_data = [dp for dp in self._data_buffer if dp.device_id == device_id]

        if not device_data:
            return None

        # Group by timestamp (get most recent)
        latest_timestamp = max(dp.timestamp for dp in device_data)
        latest_data = [dp for dp in device_data if dp.timestamp == latest_timestamp]

        # Build InverterData from data points
        data_dict: Dict[str, float] = {dp.parameter: dp.value for dp in latest_data}

        return InverterData(
            device_id=device_id,
            timestamp=latest_timestamp,
            dc_voltage=data_dict.get("dc_voltage"),
            dc_current=data_dict.get("dc_current"),
            dc_power=data_dict.get("dc_power"),
            ac_voltage=data_dict.get("ac_voltage"),
            ac_current=data_dict.get("ac_current"),
            ac_power=data_dict.get("ac_power"),
            frequency=data_dict.get("frequency"),
            power_factor=data_dict.get("power_factor"),
            temperature=data_dict.get("temperature"),
            efficiency=data_dict.get("efficiency"),
            total_energy_today=data_dict.get("energy_today"),
            total_energy_lifetime=data_dict.get("energy_lifetime"),
        )

    async def string_level_monitoring(self, inverter_id: str) -> List[StringLevelData]:
        """
        Get string-level monitoring data for an inverter.

        Args:
            inverter_id: Inverter identifier

        Returns:
            List of StringLevelData objects
        """
        # Get string data from buffer
        string_data_points = [
            dp
            for dp in self._data_buffer
            if dp.metadata.get("inverter_id") == inverter_id
            and dp.parameter.startswith("string_")
        ]

        # Group by string number
        strings: Dict[int, List[MonitoringDataPoint]] = {}
        for dp in string_data_points:
            string_num = dp.metadata.get("string_number")
            if string_num:
                if string_num not in strings:
                    strings[string_num] = []
                strings[string_num].append(dp)

        # Create StringLevelData objects
        string_level_data = []
        for string_num, data_points in strings.items():
            if data_points:
                latest_timestamp = max(dp.timestamp for dp in data_points)
                latest_data = [dp for dp in data_points if dp.timestamp == latest_timestamp]
                data_dict = {dp.parameter: dp.value for dp in latest_data}

                string_data = StringLevelData(
                    device_id=f"{inverter_id}_STR{string_num}",
                    inverter_id=inverter_id,
                    string_number=string_num,
                    timestamp=latest_timestamp,
                    voltage=data_dict.get("string_voltage"),
                    current=data_dict.get("string_current"),
                    power=data_dict.get("string_power"),
                    temperature=data_dict.get("string_temperature"),
                )
                string_level_data.append(string_data)

        return string_level_data

    async def module_level_monitoring(self, string_id: str) -> List[ModuleLevelData]:
        """
        Get module-level monitoring data for a string.

        Args:
            string_id: String identifier

        Returns:
            List of ModuleLevelData objects
        """
        # Get module data from buffer
        module_data_points = [
            dp
            for dp in self._data_buffer
            if dp.metadata.get("string_id") == string_id and dp.parameter.startswith("module_")
        ]

        # Group by module number
        modules: Dict[int, List[MonitoringDataPoint]] = {}
        for dp in module_data_points:
            module_num = dp.metadata.get("module_number")
            if module_num:
                if module_num not in modules:
                    modules[module_num] = []
                modules[module_num].append(dp)

        # Create ModuleLevelData objects
        module_level_data = []
        for module_num, data_points in modules.items():
            if data_points:
                latest_timestamp = max(dp.timestamp for dp in data_points)
                latest_data = [dp for dp in data_points if dp.timestamp == latest_timestamp]
                data_dict = {dp.parameter: dp.value for dp in latest_data}

                module_data = ModuleLevelData(
                    device_id=f"{string_id}_MOD{module_num}",
                    string_id=string_id,
                    module_number=module_num,
                    timestamp=latest_timestamp,
                    voltage=data_dict.get("module_voltage"),
                    current=data_dict.get("module_current"),
                    power=data_dict.get("module_power"),
                    temperature=data_dict.get("module_temperature"),
                )
                module_level_data.append(module_data)

        return module_level_data

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._is_monitoring

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the data buffer."""
        return {
            "buffer_size": len(self._data_buffer),
            "max_size": self.buffer_size,
            "utilization": len(self._data_buffer) / self.buffer_size if self.buffer_size > 0 else 0,
            "unique_devices": len(set(dp.device_id for dp in self._data_buffer)),
        }
