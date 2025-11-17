"""
SCADA Connector for unified device communication.

This module provides a high-level connector that wraps the DataLoggerIntegrator
and provides additional SCADA-specific functionality.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from pv_circularity.core import get_logger
from pv_circularity.models.scada import SCADADevice
from pv_circularity.models.monitoring import MonitoringDataPoint
from pv_circularity.monitoring.data_logger import DataLoggerIntegrator

logger = get_logger(__name__)


class SCADAConnector:
    """
    High-level SCADA connector with OPC UA, BACnet, and IEC 61850 support.

    This connector provides a unified interface for communicating with SCADA
    systems using various industrial protocols.

    Args:
        devices: List of SCADA devices to monitor

    Example:
        >>> connector = SCADAConnector(devices)
        >>> await connector.connect()
        >>> data = await connector.read_all_devices()
        >>> await connector.disconnect()
    """

    def __init__(self, devices: List[SCADADevice]) -> None:
        """Initialize SCADA connector."""
        self.integrator = DataLoggerIntegrator(devices)
        logger.info("SCADAConnector initialized", device_count=len(devices))

    async def connect(self) -> None:
        """Establish connections to all SCADA devices."""
        await self.integrator.initialize()
        logger.info("SCADA connector connected")

    async def disconnect(self) -> None:
        """Disconnect from all SCADA devices."""
        await self.integrator.shutdown()
        logger.info("SCADA connector disconnected")

    async def read_device(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> List[MonitoringDataPoint]:
        """Read data from a specific device."""
        return await self.integrator.collect_device_data(device_id, timestamp)

    async def read_all_devices(
        self, timestamp: Optional[datetime] = None
    ) -> Dict[str, List[MonitoringDataPoint]]:
        """Read data from all devices."""
        return await self.integrator.collect_all_data(timestamp)

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get connection status of all devices."""
        return self.integrator.get_device_status()

    async def opc_ua(self, device_id: str) -> Any:
        """Access OPC UA client for specific device."""
        client = self.integrator.clients.get(device_id)
        if client and hasattr(client, "read_node"):
            return client
        return None

    async def bacnet(self, device_id: str) -> Any:
        """Access BACnet client for specific device."""
        client = self.integrator.clients.get(device_id)
        if client and hasattr(client, "read_property"):
            return client
        return None

    async def iec61850(self, device_id: str) -> Any:
        """Access IEC 61850 client for specific device."""
        client = self.integrator.clients.get(device_id)
        if client and hasattr(client, "read_data_object"):
            return client
        return None
