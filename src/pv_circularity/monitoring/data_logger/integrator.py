"""
Data Logger Integrator for PV monitoring systems.

This module provides a comprehensive data logger integrator that supports
multiple industrial protocols and data sources.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from pv_circularity.core import get_logger, DataLoggerError
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import SCADADevice, ProtocolType
from pv_circularity.models.monitoring import MonitoringDataPoint
from pv_circularity.monitoring.scada.protocols import (
    ModbusClient,
    SunSpecClient,
    OPCUAClient,
    MQTTClient,
    BACnetClient,
    IEC61850Client,
    SMAClient,
    FroniusClient,
    HuaweiClient,
    SungrowClient,
)

logger = get_logger(__name__)


class DataLoggerIntegrator:
    """
    Integrated data logger supporting multiple industrial protocols.

    This class manages connections to various SCADA devices and protocols,
    collecting data from Modbus TCP, SunSpec, OPC UA, MQTT, and proprietary
    protocols (SMA, Fronius, Huawei, Sungrow).

    Args:
        devices: List of SCADA devices to monitor

    Example:
        >>> devices = [device1, device2, device3]
        >>> integrator = DataLoggerIntegrator(devices)
        >>> await integrator.initialize()
        >>> data = await integrator.collect_all_data()
        >>> await integrator.shutdown()
    """

    def __init__(self, devices: List[SCADADevice]) -> None:
        """
        Initialize data logger integrator.

        Args:
            devices: List of SCADA devices to monitor
        """
        self.devices = {device.device_id: device for device in devices}
        self.clients: Dict[str, Any] = {}
        self._initialized = False

        logger.info(
            "DataLoggerIntegrator initialized",
            device_count=len(devices),
        )

    async def initialize(self) -> None:
        """
        Initialize all protocol clients and establish connections.

        Raises:
            DataLoggerError: If initialization fails for critical devices
        """
        logger.info("Initializing protocol clients", device_count=len(self.devices))

        errors = []

        for device_id, device in self.devices.items():
            if not device.enabled:
                logger.debug("Skipping disabled device", device_id=device_id)
                continue

            try:
                client = await self._create_client(device)
                if client:
                    self.clients[device_id] = client
                    logger.info(
                        "Client initialized",
                        device_id=device_id,
                        protocol=device.protocol_type,
                    )
            except Exception as e:
                error_msg = f"Failed to initialize client for {device_id}: {str(e)}"
                logger.error(error_msg, device_id=device_id, error=str(e))
                errors.append(error_msg)

        self._initialized = True

        if errors:
            logger.warning(
                "Some devices failed to initialize",
                failed_count=len(errors),
                total_count=len(self.devices),
            )

        logger.info(
            "DataLoggerIntegrator initialization complete",
            active_clients=len(self.clients),
        )

    async def _create_client(self, device: SCADADevice) -> Optional[Any]:
        """
        Create and connect appropriate protocol client for device.

        Args:
            device: SCADA device configuration

        Returns:
            Connected protocol client or None if creation fails
        """
        try:
            client = None

            if device.protocol_type == ProtocolType.MODBUS_TCP:
                if device.modbus_config:
                    client = ModbusClient(device.modbus_config)
                    await client.connect()

            elif device.protocol_type == ProtocolType.SUNSPEC:
                if device.modbus_config:
                    client = SunSpecClient(device.modbus_config)
                    await client.connect()
                    await client.discover_sunspec()

            elif device.protocol_type == ProtocolType.OPCUA:
                if device.opcua_config:
                    client = OPCUAClient(device.opcua_config)
                    await client.connect()

            elif device.protocol_type == ProtocolType.MQTT:
                if device.mqtt_config:
                    client = MQTTClient(device.mqtt_config)
                    await client.connect()

            elif device.protocol_type == ProtocolType.BACNET:
                if device.bacnet_config:
                    client = BACnetClient(device.bacnet_config)
                    await client.connect()

            elif device.protocol_type == ProtocolType.IEC61850:
                if device.iec61850_config:
                    client = IEC61850Client(device.iec61850_config)
                    await client.connect()

            elif device.protocol_type == ProtocolType.PROPRIETARY:
                if device.proprietary_config:
                    protocol = device.proprietary_config.protocol
                    if protocol == "sma":
                        client = SMAClient(device.proprietary_config)
                    elif protocol == "fronius":
                        client = FroniusClient(device.proprietary_config)
                    elif protocol == "huawei":
                        client = HuaweiClient(device.proprietary_config)
                    elif protocol == "sungrow":
                        client = SungrowClient(device.proprietary_config)

                    if client:
                        await client.connect()

            return client

        except Exception as e:
            logger.error(
                "Failed to create client",
                device_id=device.device_id,
                protocol=device.protocol_type,
                error=str(e),
            )
            raise

    async def collect_device_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> List[MonitoringDataPoint]:
        """
        Collect data from a single device.

        Args:
            device_id: Device identifier
            timestamp: Timestamp for data points (defaults to current time)

        Returns:
            List of monitoring data points

        Raises:
            DataLoggerError: If device not found or data collection fails
        """
        if not self._initialized:
            raise DataLoggerError("DataLoggerIntegrator not initialized")

        if device_id not in self.clients:
            raise DataLoggerError(
                f"No active client for device {device_id}",
                details={"device_id": device_id},
            )

        timestamp = timestamp or get_utc_now()
        client = self.clients[device_id]

        try:
            # Read parameters based on client type
            if hasattr(client, "read_parameters"):
                data_points = await client.read_parameters(device_id, timestamp)
            elif hasattr(client, "read_inverter_model"):
                # SunSpec client
                data_points = await client.read_inverter_model(device_id, timestamp)
            elif hasattr(client, "read_inverter_data"):
                # Proprietary clients
                inverter_data = await client.read_inverter_data(device_id, timestamp)
                # Convert InverterData to MonitoringDataPoints
                data_points = self._inverter_data_to_points(inverter_data) if inverter_data else []
            else:
                logger.warning(
                    "Client does not support data reading",
                    device_id=device_id,
                )
                return []

            logger.debug(
                "Collected device data",
                device_id=device_id,
                data_points=len(data_points),
            )

            return data_points

        except Exception as e:
            logger.error(
                "Failed to collect device data",
                device_id=device_id,
                error=str(e),
            )
            raise DataLoggerError(
                f"Failed to collect data from device {device_id}: {str(e)}",
                details={"device_id": device_id},
                original_exception=e,
            )

    async def collect_all_data(
        self, timestamp: Optional[datetime] = None
    ) -> Dict[str, List[MonitoringDataPoint]]:
        """
        Collect data from all active devices concurrently.

        Args:
            timestamp: Timestamp for data points (defaults to current time)

        Returns:
            Dictionary mapping device IDs to their data points

        Example:
            >>> data = await integrator.collect_all_data()
            >>> for device_id, data_points in data.items():
            ...     print(f"{device_id}: {len(data_points)} points")
        """
        if not self._initialized:
            raise DataLoggerError("DataLoggerIntegrator not initialized")

        timestamp = timestamp or get_utc_now()
        results = {}

        # Collect data from all devices concurrently
        tasks = []
        device_ids = []

        for device_id in self.clients.keys():
            tasks.append(self.collect_device_data(device_id, timestamp))
            device_ids.append(device_id)

        # Gather results
        collected_data = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for device_id, data in zip(device_ids, collected_data):
            if isinstance(data, Exception):
                logger.error(
                    "Error collecting data from device",
                    device_id=device_id,
                    error=str(data),
                )
                results[device_id] = []
            else:
                results[device_id] = data

        total_points = sum(len(points) for points in results.values())
        logger.info(
            "Collected data from all devices",
            devices=len(results),
            total_data_points=total_points,
        )

        return results

    def _inverter_data_to_points(
        self, inverter_data: Any
    ) -> List[MonitoringDataPoint]:
        """
        Convert InverterData object to list of MonitoringDataPoints.

        Args:
            inverter_data: InverterData object

        Returns:
            List of MonitoringDataPoints
        """
        data_points = []

        # Define mapping of inverter fields to parameters
        field_mapping = {
            "dc_voltage": ("dc_voltage", "V"),
            "dc_current": ("dc_current", "A"),
            "dc_power": ("dc_power", "W"),
            "ac_voltage": ("ac_voltage", "V"),
            "ac_current": ("ac_current", "A"),
            "ac_power": ("ac_power", "kW"),
            "frequency": ("frequency", "Hz"),
            "power_factor": ("power_factor", ""),
            "temperature": ("temperature", "°C"),
            "efficiency": ("efficiency", ""),
            "total_energy_today": ("energy_today", "kWh"),
            "total_energy_lifetime": ("energy_lifetime", "kWh"),
        }

        for field, (parameter, unit) in field_mapping.items():
            value = getattr(inverter_data, field, None)
            if value is not None:
                data_point = MonitoringDataPoint(
                    device_id=inverter_data.device_id,
                    timestamp=inverter_data.timestamp,
                    parameter=parameter,
                    value=float(value),
                    unit=unit,
                    quality=1.0,
                )
                data_points.append(data_point)

        return data_points

    async def shutdown(self) -> None:
        """
        Shutdown all protocol clients and close connections.
        """
        logger.info("Shutting down DataLoggerIntegrator")

        for device_id, client in self.clients.items():
            try:
                if hasattr(client, "disconnect"):
                    await client.disconnect()
                logger.debug("Client disconnected", device_id=device_id)
            except Exception as e:
                logger.error(
                    "Error disconnecting client",
                    device_id=device_id,
                    error=str(e),
                )

        self.clients.clear()
        self._initialized = False
        logger.info("DataLoggerIntegrator shutdown complete")

    @property
    def is_initialized(self) -> bool:
        """Check if integrator is initialized."""
        return self._initialized

    @property
    def active_device_count(self) -> int:
        """Get number of active devices."""
        return len(self.clients)

    def get_device_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all devices.

        Returns:
            Dictionary mapping device IDs to their status information
        """
        status = {}

        for device_id, device in self.devices.items():
            client = self.clients.get(device_id)
            status[device_id] = {
                "enabled": device.enabled,
                "connected": client.is_connected if client else False,
                "protocol": device.protocol_type,
                "device_type": device.device_type,
            }

        return status
