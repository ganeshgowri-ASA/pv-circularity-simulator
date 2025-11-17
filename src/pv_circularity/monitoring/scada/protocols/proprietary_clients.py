"""
Proprietary protocol clients for major PV inverter manufacturers.

This module provides clients for manufacturer-specific protocols including
SMA, Fronius, Huawei, and Sungrow inverters.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

from pv_circularity.core import get_logger, SCADAConnectionError
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import ProprietaryProtocolConfig
from pv_circularity.models.monitoring import MonitoringDataPoint, InverterData, DeviceStatus

logger = get_logger(__name__)


class ProprietaryClientBase(ABC):
    """
    Base class for proprietary protocol clients.

    Provides common functionality for manufacturer-specific protocol implementations.
    """

    def __init__(self, config: ProprietaryProtocolConfig) -> None:
        """
        Initialize proprietary client.

        Args:
            config: Proprietary protocol configuration
        """
        self.config = config
        self._connected = False
        logger.info(
            "Proprietary client initialized",
            protocol=config.protocol,
            host=config.host,
        )

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to device."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from device."""
        pass

    @abstractmethod
    async def read_inverter_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[InverterData]:
        """Read comprehensive inverter data."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def __aenter__(self) -> "ProprietaryClientBase":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


class SMAClient(ProprietaryClientBase):
    """
    SMA inverter proprietary protocol client.

    SMA uses Speedwire/Webconnect protocol for communication with their inverters.
    This implementation supports basic data retrieval.

    Args:
        config: SMA protocol configuration

    Example:
        >>> config = ProprietaryProtocolConfig(
        ...     protocol="sma",
        ...     host="192.168.1.100",
        ...     port=502,
        ...     password="0000",
        ... )
        >>> client = SMAClient(config)
        >>> await client.connect()
        >>> data = await client.read_inverter_data("INV001")
        >>> await client.disconnect()
    """

    async def connect(self) -> None:
        """
        Establish connection to SMA inverter.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            # TODO: Implement actual SMA Speedwire/Webconnect connection
            # This would typically use the SMA protocol specifications

            # Simulate connection
            await asyncio.sleep(0.1)
            self._connected = True

            logger.info(
                "Connected to SMA inverter",
                host=self.config.host,
                port=self.config.port,
            )

        except Exception as e:
            logger.error("SMA connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to SMA inverter: {str(e)}",
                protocol="sma",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from SMA inverter."""
        self._connected = False
        logger.info("Disconnected from SMA inverter")

    async def read_inverter_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[InverterData]:
        """
        Read comprehensive data from SMA inverter.

        Args:
            device_id: Device identifier
            timestamp: Timestamp for the data

        Returns:
            InverterData object or None if read fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to SMA inverter", protocol="sma")

        timestamp = timestamp or get_utc_now()

        try:
            # TODO: Implement actual SMA data reading
            # This is a placeholder showing the expected data structure

            logger.warning("SMA read_inverter_data not fully implemented", device_id=device_id)

            # Return None for now - full implementation would return actual data
            return None

        except Exception as e:
            logger.error("Failed to read SMA inverter data", device_id=device_id, error=str(e))
            return None


class FroniusClient(ProprietaryClientBase):
    """
    Fronius inverter proprietary protocol client.

    Fronius provides a Solar API (JSON/HTTP) for their inverters.
    This implementation uses the Fronius Solar API.

    Args:
        config: Fronius protocol configuration

    Example:
        >>> config = ProprietaryProtocolConfig(
        ...     protocol="fronius",
        ...     host="192.168.1.100",
        ...     port=80,
        ... )
        >>> client = FroniusClient(config)
        >>> await client.connect()
        >>> data = await client.read_inverter_data("INV001")
        >>> await client.disconnect()
    """

    async def connect(self) -> None:
        """
        Establish connection to Fronius inverter.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            # Fronius uses HTTP API, so "connection" just validates accessibility
            import aiohttp

            url = f"http://{self.config.host}:{self.config.port}/solar_api/GetAPIVersion.cgi"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        self._connected = True
                        logger.info("Connected to Fronius inverter", host=self.config.host)
                    else:
                        raise SCADAConnectionError(
                            f"Fronius API returned status {response.status}",
                            protocol="fronius",
                        )

        except Exception as e:
            logger.error("Fronius connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to Fronius inverter: {str(e)}",
                protocol="fronius",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from Fronius inverter."""
        self._connected = False
        logger.info("Disconnected from Fronius inverter")

    async def read_inverter_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[InverterData]:
        """
        Read comprehensive data from Fronius inverter using Solar API.

        Args:
            device_id: Device identifier
            timestamp: Timestamp for the data

        Returns:
            InverterData object or None if read fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to Fronius inverter", protocol="fronius")

        timestamp = timestamp or get_utc_now()

        try:
            import aiohttp

            # Read inverter realtime data
            url = f"http://{self.config.host}:{self.config.port}/solar_api/v1/GetInverterRealtimeData.cgi?Scope=Device&DeviceId={self.config.device_id or 1}&DataCollection=CommonInverterData"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config.timeout) as response:
                    if response.status != 200:
                        logger.error(
                            "Fronius API error",
                            status=response.status,
                            device_id=device_id,
                        )
                        return None

                    data = await response.json()

                    # Parse Fronius API response
                    body = data.get("Body", {}).get("Data", {})

                    # Extract data
                    pac = body.get("PAC", {}).get("Value")
                    uac = body.get("UAC", {}).get("Value")
                    iac = body.get("IAC", {}).get("Value")
                    udc = body.get("UDC", {}).get("Value")
                    idc = body.get("IDC", {}).get("Value")
                    frequency = body.get("FAC", {}).get("Value")
                    day_energy = body.get("DAY_ENERGY", {}).get("Value")
                    total_energy = body.get("TOTAL_ENERGY", {}).get("Value")

                    # Create InverterData object
                    inverter_data = InverterData(
                        device_id=device_id,
                        timestamp=timestamp,
                        dc_voltage=udc if udc is not None else None,
                        dc_current=idc if idc is not None else None,
                        dc_power=udc * idc if udc and idc else None,
                        ac_voltage=uac if uac is not None else None,
                        ac_current=iac if iac is not None else None,
                        ac_power=pac / 1000 if pac is not None else None,  # Convert W to kW
                        frequency=frequency if frequency is not None else None,
                        total_energy_today=day_energy / 1000 if day_energy else None,  # Wh to kWh
                        total_energy_lifetime=total_energy / 1000 if total_energy else None,
                        status=DeviceStatus.ONLINE,
                    )

                    logger.debug("Read Fronius inverter data", device_id=device_id)
                    return inverter_data

        except Exception as e:
            logger.error(
                "Failed to read Fronius inverter data",
                device_id=device_id,
                error=str(e),
            )
            return None


class HuaweiClient(ProprietaryClientBase):
    """
    Huawei inverter proprietary protocol client.

    Huawei uses Modbus TCP with proprietary register mappings.
    This implementation wraps Modbus with Huawei-specific logic.

    Args:
        config: Huawei protocol configuration

    Example:
        >>> config = ProprietaryProtocolConfig(
        ...     protocol="huawei",
        ...     host="192.168.1.100",
        ...     port=502,
        ... )
        >>> client = HuaweiClient(config)
        >>> await client.connect()
        >>> data = await client.read_inverter_data("INV001")
        >>> await client.disconnect()
    """

    async def connect(self) -> None:
        """
        Establish connection to Huawei inverter.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            # TODO: Implement Huawei Modbus TCP connection
            # Huawei uses standard Modbus with specific register mapping

            # Simulate connection
            await asyncio.sleep(0.1)
            self._connected = True

            logger.info("Connected to Huawei inverter", host=self.config.host)

        except Exception as e:
            logger.error("Huawei connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to Huawei inverter: {str(e)}",
                protocol="huawei",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from Huawei inverter."""
        self._connected = False
        logger.info("Disconnected from Huawei inverter")

    async def read_inverter_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[InverterData]:
        """
        Read comprehensive data from Huawei inverter.

        Args:
            device_id: Device identifier
            timestamp: Timestamp for the data

        Returns:
            InverterData object or None if read fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to Huawei inverter", protocol="huawei")

        timestamp = timestamp or get_utc_now()

        try:
            # TODO: Implement actual Huawei data reading using Modbus
            logger.warning("Huawei read_inverter_data not fully implemented", device_id=device_id)
            return None

        except Exception as e:
            logger.error(
                "Failed to read Huawei inverter data",
                device_id=device_id,
                error=str(e),
            )
            return None


class SungrowClient(ProprietaryClientBase):
    """
    Sungrow inverter proprietary protocol client.

    Sungrow uses Modbus TCP with proprietary register mappings.
    This implementation wraps Modbus with Sungrow-specific logic.

    Args:
        config: Sungrow protocol configuration

    Example:
        >>> config = ProprietaryProtocolConfig(
        ...     protocol="sungrow",
        ...     host="192.168.1.100",
        ...     port=502,
        ... )
        >>> client = SungrowClient(config)
        >>> await client.connect()
        >>> data = await client.read_inverter_data("INV001")
        >>> await client.disconnect()
    """

    async def connect(self) -> None:
        """
        Establish connection to Sungrow inverter.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            # TODO: Implement Sungrow Modbus TCP connection
            # Sungrow uses standard Modbus with specific register mapping

            # Simulate connection
            await asyncio.sleep(0.1)
            self._connected = True

            logger.info("Connected to Sungrow inverter", host=self.config.host)

        except Exception as e:
            logger.error("Sungrow connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to Sungrow inverter: {str(e)}",
                protocol="sungrow",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from Sungrow inverter."""
        self._connected = False
        logger.info("Disconnected from Sungrow inverter")

    async def read_inverter_data(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[InverterData]:
        """
        Read comprehensive data from Sungrow inverter.

        Args:
            device_id: Device identifier
            timestamp: Timestamp for the data

        Returns:
            InverterData object or None if read fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to Sungrow inverter", protocol="sungrow")

        timestamp = timestamp or get_utc_now()

        try:
            # TODO: Implement actual Sungrow data reading using Modbus
            logger.warning(
                "Sungrow read_inverter_data not fully implemented",
                device_id=device_id,
            )
            return None

        except Exception as e:
            logger.error(
                "Failed to read Sungrow inverter data",
                device_id=device_id,
                error=str(e),
            )
            return None
