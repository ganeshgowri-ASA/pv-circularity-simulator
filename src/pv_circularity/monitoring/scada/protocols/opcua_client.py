"""
OPC UA client implementation for industrial automation.

This module provides async OPC UA client for communicating with devices
using the OPC Unified Architecture protocol.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from asyncua import Client
from asyncua.ua.uaerrors import UaError

from pv_circularity.core import get_logger, SCADAConnectionError, retry_on_exception
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import OPCUAConfig
from pv_circularity.models.monitoring import MonitoringDataPoint

logger = get_logger(__name__)


class OPCUAClient:
    """
    Async OPC UA client for data collection from industrial equipment.

    This client provides methods to read data from OPC UA servers with
    support for various security policies and authentication methods.

    Args:
        config: OPC UA configuration

    Example:
        >>> config = OPCUAConfig(endpoint="opc.tcp://192.168.1.100:4840")
        >>> client = OPCUAClient(config)
        >>> await client.connect()
        >>> data = await client.read_node("ns=2;i=1001")
        >>> await client.disconnect()
    """

    def __init__(self, config: OPCUAConfig) -> None:
        """
        Initialize OPC UA client.

        Args:
            config: OPC UA configuration
        """
        self.config = config
        self.client: Optional[Client] = None
        self._connected = False
        logger.info("OPC UA client initialized", endpoint=config.endpoint)

    async def connect(self) -> None:
        """
        Establish connection to OPC UA server.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            self.client = Client(url=self.config.endpoint, timeout=self.config.timeout)

            # Set security policy if specified
            if self.config.security_policy != "None":
                await self.client.set_security_string(
                    f"Basic256Sha256,SignAndEncrypt,{self.config.certificate_path},{self.config.private_key_path}"
                )

            # Set user authentication if provided
            if self.config.username and self.config.password:
                self.client.set_user(self.config.username)
                self.client.set_password(self.config.password)

            await self.client.connect()
            self._connected = True

            logger.info("Connected to OPC UA server", endpoint=self.config.endpoint)

        except Exception as e:
            logger.error("OPC UA connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to OPC UA server: {str(e)}",
                protocol="opcua",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from OPC UA server."""
        if self.client:
            try:
                await self.client.disconnect()
                self._connected = False
                logger.info("Disconnected from OPC UA server")
            except Exception as e:
                logger.error("Error during OPC UA disconnect", error=str(e))

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.client is not None

    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(UaError,))
    async def read_node(self, node_id: str) -> Any:
        """
        Read value from a single OPC UA node.

        Args:
            node_id: Node identifier (e.g., "ns=2;i=1001" or "ns=2;s=Variable1")

        Returns:
            Node value

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to OPC UA server", protocol="opcua")

        try:
            node = self.client.get_node(node_id)
            value = await node.read_value()
            return value

        except Exception as e:
            logger.error("Failed to read OPC UA node", node_id=node_id, error=str(e))
            raise SCADAConnectionError(
                f"Failed to read OPC UA node {node_id}: {str(e)}",
                protocol="opcua",
                original_exception=e,
            )

    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(UaError,))
    async def read_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Read values from multiple OPC UA nodes.

        Args:
            node_ids: List of node identifiers

        Returns:
            Dictionary mapping node IDs to their values

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to OPC UA server", protocol="opcua")

        results = {}
        for node_id in node_ids:
            try:
                value = await self.read_node(node_id)
                results[node_id] = value
            except Exception as e:
                logger.warning(
                    "Failed to read node, skipping",
                    node_id=node_id,
                    error=str(e),
                )
                results[node_id] = None

        return results

    async def read_parameters(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> List[MonitoringDataPoint]:
        """
        Read all configured parameters from the OPC UA server.

        Uses the node_ids from configuration to read and parse data.

        Args:
            device_id: Device identifier for the data points
            timestamp: Timestamp for the data points (defaults to current time)

        Returns:
            List of monitoring data points

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.config.node_ids:
            logger.warning("No node IDs configured for device", device_id=device_id)
            return []

        timestamp = timestamp or get_utc_now()
        data_points: List[MonitoringDataPoint] = []

        # Read all nodes
        values = await self.read_nodes(list(self.config.node_ids.values()))

        # Create data points
        for parameter, node_id in self.config.node_ids.items():
            value = values.get(node_id)
            if value is None:
                continue

            try:
                # Convert value to float (you may need more sophisticated conversion)
                float_value = float(value)

                data_point = MonitoringDataPoint(
                    device_id=device_id,
                    timestamp=timestamp,
                    parameter=parameter,
                    value=float_value,
                    unit="",  # OPC UA nodes may have units in their metadata
                    quality=1.0,
                    metadata={"node_id": node_id},
                )
                data_points.append(data_point)

            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to convert node value to float",
                    parameter=parameter,
                    node_id=node_id,
                    value=value,
                    error=str(e),
                )
                continue

        logger.debug(
            "Read parameters from OPC UA server",
            device_id=device_id,
            parameters=len(data_points),
        )

        return data_points

    async def write_node(self, node_id: str, value: Any) -> None:
        """
        Write value to an OPC UA node.

        Args:
            node_id: Node identifier
            value: Value to write

        Raises:
            SCADAConnectionError: If write operation fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to OPC UA server", protocol="opcua")

        try:
            node = self.client.get_node(node_id)
            await node.write_value(value)
            logger.debug("Wrote value to OPC UA node", node_id=node_id, value=value)

        except Exception as e:
            logger.error(
                "Failed to write OPC UA node",
                node_id=node_id,
                value=value,
                error=str(e),
            )
            raise SCADAConnectionError(
                f"Failed to write OPC UA node {node_id}: {str(e)}",
                protocol="opcua",
                original_exception=e,
            )

    async def __aenter__(self) -> "OPCUAClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
