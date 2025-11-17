"""
Real-time monitoring system for PV plant performance.

This module provides the RealTimeMonitor class for collecting, processing,
and streaming live data from PV systems including SCADA, inverters, strings,
and modules.
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any, Callable
from collections import defaultdict
import json

from config.settings import Settings
from src.core.models.schemas import (
    InverterData,
    StringData,
    ModuleData,
    SCADAData,
    DataQualityStatus,
    ProtocolType,
    LiveDataUpdate
)

logger = logging.getLogger(__name__)


class RealTimeMonitor:
    """
    Real-time monitoring system for PV plant performance.

    This class handles live data collection from various sources including
    SCADA systems, inverters, string-level monitors, and module-level sensors
    using MQTT and Modbus protocols.

    Attributes:
        settings: Application settings instance
        _mqtt_client: MQTT client instance
        _modbus_clients: Dictionary of Modbus client instances
        _data_buffer: Buffer for incoming data
        _subscriptions: Active data subscriptions
        _running: Monitor running state
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the RealTimeMonitor.

        Args:
            settings: Application settings instance containing protocol
                     and monitoring configuration.
        """
        self.settings = settings
        self._mqtt_client: Optional[Any] = None
        self._modbus_clients: Dict[str, Any] = {}
        self._data_buffer: Dict[str, List[Any]] = defaultdict(list)
        self._subscriptions: List[Callable] = []
        self._running: bool = False
        self._tasks: List[asyncio.Task] = []

        logger.info("RealTimeMonitor initialized")

    async def start(self) -> None:
        """
        Start the real-time monitoring system.

        Initializes all protocol handlers (MQTT, Modbus) and begins
        data collection from configured sources.

        Raises:
            RuntimeError: If monitor is already running.
        """
        if self._running:
            raise RuntimeError("RealTimeMonitor is already running")

        logger.info("Starting RealTimeMonitor...")
        self._running = True

        # Start protocol handlers
        if ProtocolType.MQTT in self.settings.monitoring.enabled_protocols:
            await self._start_mqtt_handler()

        if ProtocolType.MODBUS_TCP in self.settings.monitoring.enabled_protocols:
            await self._start_modbus_handler()

        logger.info("RealTimeMonitor started successfully")

    async def stop(self) -> None:
        """
        Stop the real-time monitoring system.

        Gracefully shuts down all protocol handlers and cleans up resources.
        """
        logger.info("Stopping RealTimeMonitor...")
        self._running = False

        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cleanup protocol handlers
        if self._mqtt_client:
            await self._stop_mqtt_handler()

        if self._modbus_clients:
            await self._stop_modbus_handler()

        self._tasks.clear()
        logger.info("RealTimeMonitor stopped")

    async def live_data_stream(
        self,
        data_types: Optional[List[str]] = None,
        site_id: Optional[str] = None
    ) -> AsyncIterator[LiveDataUpdate]:
        """
        Stream live data in real-time.

        Provides an async generator that yields live data updates as they
        arrive from various monitoring sources.

        Args:
            data_types: List of data types to stream (e.g., ['inverter', 'scada']).
                       If None, streams all types.
            site_id: Filter data by site ID. If None, streams data from all sites.

        Yields:
            LiveDataUpdate: Real-time data updates containing the latest measurements.

        Example:
            >>> async for update in monitor.live_data_stream(['inverter']):
            ...     print(f"Received: {update.data}")
        """
        logger.info(f"Starting live data stream (types={data_types}, site={site_id})")

        if not self._running:
            logger.warning("Monitor not running, starting it now")
            await self.start()

        queue: asyncio.Queue = asyncio.Queue()

        # Subscribe callback to put data in queue
        async def data_callback(data: Dict[str, Any]) -> None:
            """Callback to handle incoming data."""
            # Filter by data type
            if data_types and data.get("type") not in data_types:
                return

            # Filter by site ID
            if site_id and data.get("site_id") != site_id:
                return

            update = LiveDataUpdate(
                update_type=data.get("type", "unknown"),
                timestamp=data.get("timestamp", datetime.utcnow()),
                site_id=data.get("site_id", self.settings.site_id),
                data=data
            )
            await queue.put(update)

        self._subscriptions.append(data_callback)

        try:
            while self._running:
                try:
                    # Wait for data with timeout
                    update = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield update
                except asyncio.TimeoutError:
                    continue
        finally:
            self._subscriptions.remove(data_callback)
            logger.info("Live data stream ended")

    async def scada_integration(
        self,
        scada_host: Optional[str] = None,
        protocol: ProtocolType = ProtocolType.MQTT
    ) -> AsyncIterator[SCADAData]:
        """
        Integrate with SCADA system for plant-level data.

        Establishes connection to SCADA system and streams aggregated
        plant-level performance data.

        Args:
            scada_host: SCADA system host/endpoint. If None, uses configured host.
            protocol: Communication protocol to use (MQTT or Modbus).

        Yields:
            SCADAData: SCADA system data including aggregated power, environmental
                      conditions, and system availability.

        Raises:
            ConnectionError: If unable to connect to SCADA system.

        Example:
            >>> async for scada_data in monitor.scada_integration():
            ...     print(f"Total power: {scada_data.total_ac_power} kW")
        """
        logger.info(f"Starting SCADA integration (host={scada_host}, protocol={protocol})")

        if protocol == ProtocolType.MQTT:
            async for data in self._scada_mqtt_stream(scada_host):
                yield data
        elif protocol in [ProtocolType.MODBUS_TCP, ProtocolType.MODBUS_RTU]:
            async for data in self._scada_modbus_stream(scada_host):
                yield data
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    async def inverter_data_parsing(
        self,
        inverter_ids: Optional[List[str]] = None
    ) -> AsyncIterator[InverterData]:
        """
        Parse and stream inverter-level data.

        Continuously parses incoming inverter telemetry data and provides
        validated, structured data stream.

        Args:
            inverter_ids: List of inverter IDs to monitor. If None, monitors all inverters.

        Yields:
            InverterData: Parsed and validated inverter telemetry including power,
                         voltage, current, temperature, and status.

        Example:
            >>> async for inv_data in monitor.inverter_data_parsing(['INV001', 'INV002']):
            ...     print(f"{inv_data.inverter_id}: {inv_data.ac_power} kW")
        """
        logger.info(f"Starting inverter data parsing (inverters={inverter_ids})")

        async for update in self.live_data_stream(data_types=['inverter']):
            try:
                # Parse raw data into InverterData model
                inv_data = InverterData(**update.data)

                # Filter by inverter ID if specified
                if inverter_ids and inv_data.inverter_id not in inverter_ids:
                    continue

                # Validate data quality
                inv_data = self._assess_data_quality(inv_data)

                yield inv_data

            except Exception as e:
                logger.error(f"Error parsing inverter data: {e}", exc_info=True)
                continue

    async def string_level_monitoring(
        self,
        string_ids: Optional[List[str]] = None,
        inverter_id: Optional[str] = None
    ) -> AsyncIterator[StringData]:
        """
        Monitor string-level performance data.

        Provides real-time monitoring of individual string performance including
        voltage, current, and power measurements.

        Args:
            string_ids: List of string IDs to monitor. If None, monitors all strings.
            inverter_id: Filter strings by parent inverter ID.

        Yields:
            StringData: String-level measurements including voltage, current, power,
                       and environmental conditions.

        Example:
            >>> async for string_data in monitor.string_level_monitoring(inverter_id='INV001'):
            ...     print(f"String {string_data.string_id}: {string_data.power} kW")
        """
        logger.info(f"Starting string-level monitoring (strings={string_ids}, inverter={inverter_id})")

        async for update in self.live_data_stream(data_types=['string']):
            try:
                # Parse raw data into StringData model
                string_data = StringData(**update.data)

                # Filter by string ID if specified
                if string_ids and string_data.string_id not in string_ids:
                    continue

                # Filter by inverter ID if specified
                if inverter_id and string_data.inverter_id != inverter_id:
                    continue

                # Validate data quality
                string_data = self._assess_data_quality(string_data)

                yield string_data

            except Exception as e:
                logger.error(f"Error parsing string data: {e}", exc_info=True)
                continue

    async def module_level_monitoring(
        self,
        module_ids: Optional[List[str]] = None,
        string_id: Optional[str] = None
    ) -> AsyncIterator[ModuleData]:
        """
        Monitor module-level performance and health data.

        Provides detailed module-level monitoring including hot-spot detection,
        temperature monitoring, and individual module performance.

        Args:
            module_ids: List of module IDs to monitor. If None, monitors all modules.
            string_id: Filter modules by parent string ID.

        Yields:
            ModuleData: Module-level measurements including voltage, current, power,
                       temperature, and hot-spot detection status.

        Example:
            >>> async for module_data in monitor.module_level_monitoring():
            ...     if module_data.hotspot_detected:
            ...         print(f"Alert: Hot spot on module {module_data.module_id}")
        """
        logger.info(f"Starting module-level monitoring (modules={module_ids}, string={string_id})")

        async for update in self.live_data_stream(data_types=['module']):
            try:
                # Parse raw data into ModuleData model
                module_data = ModuleData(**update.data)

                # Filter by module ID if specified
                if module_ids and module_data.module_id not in module_ids:
                    continue

                # Filter by string ID if specified
                if string_id and module_data.string_id != string_id:
                    continue

                # Validate data quality
                module_data = self._assess_data_quality(module_data)

                # Check for hot spots
                if module_data.max_cell_temperature:
                    if module_data.max_cell_temperature - module_data.temperature > 15:
                        module_data.hotspot_detected = True
                        logger.warning(
                            f"Hot spot detected on module {module_data.module_id}: "
                            f"ΔT = {module_data.max_cell_temperature - module_data.temperature}°C"
                        )

                yield module_data

            except Exception as e:
                logger.error(f"Error parsing module data: {e}", exc_info=True)
                continue

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _start_mqtt_handler(self) -> None:
        """Initialize and start MQTT client."""
        try:
            from src.monitoring.protocols.mqtt_handler import MQTTHandler

            self._mqtt_client = MQTTHandler(self.settings)
            await self._mqtt_client.connect()

            # Subscribe to topics and register callback
            await self._mqtt_client.subscribe_all(self._on_mqtt_message)

            logger.info("MQTT handler started")
        except Exception as e:
            logger.error(f"Failed to start MQTT handler: {e}", exc_info=True)
            raise

    async def _stop_mqtt_handler(self) -> None:
        """Stop MQTT client."""
        if self._mqtt_client:
            await self._mqtt_client.disconnect()
            self._mqtt_client = None
            logger.info("MQTT handler stopped")

    async def _start_modbus_handler(self) -> None:
        """Initialize and start Modbus client."""
        try:
            from src.monitoring.protocols.modbus_handler import ModbusHandler

            modbus_handler = ModbusHandler(self.settings)
            await modbus_handler.connect()
            self._modbus_clients['default'] = modbus_handler

            # Start polling task
            task = asyncio.create_task(self._modbus_polling_loop())
            self._tasks.append(task)

            logger.info("Modbus handler started")
        except Exception as e:
            logger.error(f"Failed to start Modbus handler: {e}", exc_info=True)
            raise

    async def _stop_modbus_handler(self) -> None:
        """Stop Modbus clients."""
        for client in self._modbus_clients.values():
            await client.disconnect()
        self._modbus_clients.clear()
        logger.info("Modbus handler stopped")

    async def _on_mqtt_message(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Handle incoming MQTT messages.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            # Notify subscribers
            for callback in self._subscriptions:
                await callback(payload)

        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}", exc_info=True)

    async def _modbus_polling_loop(self) -> None:
        """Continuously poll Modbus devices."""
        while self._running:
            try:
                for client in self._modbus_clients.values():
                    data = await client.poll_data()
                    # Notify subscribers
                    for callback in self._subscriptions:
                        await callback(data)

                await asyncio.sleep(self.settings.modbus.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Modbus polling loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _scada_mqtt_stream(self, host: Optional[str]) -> AsyncIterator[SCADAData]:
        """Stream SCADA data via MQTT."""
        async for update in self.live_data_stream(data_types=['scada']):
            try:
                scada_data = SCADAData(**update.data)
                yield scada_data
            except Exception as e:
                logger.error(f"Error parsing SCADA data: {e}", exc_info=True)

    async def _scada_modbus_stream(self, host: Optional[str]) -> AsyncIterator[SCADAData]:
        """Stream SCADA data via Modbus."""
        async for update in self.live_data_stream(data_types=['scada']):
            try:
                scada_data = SCADAData(**update.data)
                yield scada_data
            except Exception as e:
                logger.error(f"Error parsing SCADA data: {e}", exc_info=True)

    def _assess_data_quality(self, data: Any) -> Any:
        """
        Assess and tag data quality.

        Args:
            data: Data object to assess

        Returns:
            Data object with updated data_quality field
        """
        if not hasattr(data, 'timestamp') or not hasattr(data, 'data_quality'):
            return data

        # Check data staleness
        age_seconds = (datetime.utcnow() - data.timestamp).total_seconds()

        if age_seconds > self.settings.monitoring.data_staleness_threshold_sec:
            data.data_quality = DataQualityStatus.POOR
        elif age_seconds > self.settings.monitoring.data_staleness_threshold_sec / 2:
            data.data_quality = DataQualityStatus.DEGRADED
        else:
            data.data_quality = DataQualityStatus.GOOD

        return data
