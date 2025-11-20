"""
Modbus TCP/RTU client implementation with SunSpec protocol support.

This module provides async Modbus clients for communicating with PV equipment
using both standard Modbus and the SunSpec protocol specification.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
from pymodbus.exceptions import ModbusException

from pv_circularity.core import get_logger, SCADAConnectionError, retry_on_exception
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import ModbusConfig
from pv_circularity.models.monitoring import MonitoringDataPoint

logger = get_logger(__name__)


class ModbusClient:
    """
    Async Modbus TCP/RTU client for data collection from industrial devices.

    This client provides methods to read data from Modbus devices including
    holding registers, input registers, coils, and discrete inputs.

    Args:
        config: Modbus configuration

    Example:
        >>> config = ModbusConfig(host="192.168.1.100", port=502, slave_id=1)
        >>> client = ModbusClient(config)
        >>> await client.connect()
        >>> data = await client.read_holding_registers(0, 10)
        >>> await client.disconnect()
    """

    def __init__(self, config: ModbusConfig) -> None:
        """
        Initialize Modbus client.

        Args:
            config: Modbus configuration
        """
        self.config = config
        self.client: Optional[AsyncModbusTcpClient | AsyncModbusSerialClient] = None
        self._connected = False
        logger.info(
            "Modbus client initialized",
            host=config.host,
            port=config.port,
            slave_id=config.slave_id,
        )

    async def connect(self) -> None:
        """
        Establish connection to Modbus device.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            if self.config.serial_port:
                # Modbus RTU over serial
                self.client = AsyncModbusSerialClient(
                    port=self.config.serial_port,
                    baudrate=self.config.baudrate,
                    parity=self.config.parity,
                    stopbits=self.config.stopbits,
                    bytesize=self.config.bytesize,
                    timeout=self.config.timeout,
                )
            else:
                # Modbus TCP
                self.client = AsyncModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                )

            await self.client.connect()
            self._connected = self.client.connected

            if not self._connected:
                raise SCADAConnectionError(
                    "Failed to connect to Modbus device",
                    protocol="modbus",
                    details={"host": self.config.host, "port": self.config.port},
                )

            logger.info(
                "Connected to Modbus device",
                host=self.config.host,
                port=self.config.port,
            )

        except Exception as e:
            logger.error("Modbus connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to Modbus device: {str(e)}",
                protocol="modbus",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from Modbus device")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.client is not None and self.client.connected

    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(ModbusException,))
    async def read_holding_registers(
        self, address: int, count: int, slave: Optional[int] = None
    ) -> List[int]:
        """
        Read holding registers from Modbus device.

        Args:
            address: Starting register address
            count: Number of registers to read
            slave: Slave ID (overrides config if provided)

        Returns:
            List of register values

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.is_connected:
            raise SCADAConnectionError(
                "Not connected to Modbus device", protocol="modbus"
            )

        slave_id = slave if slave is not None else self.config.slave_id

        try:
            response = await self.client.read_holding_registers(
                address=address, count=count, slave=slave_id
            )

            if response.isError():
                raise SCADAConnectionError(
                    f"Modbus read error: {response}",
                    protocol="modbus",
                    details={"address": address, "count": count, "slave": slave_id},
                )

            return response.registers

        except Exception as e:
            logger.error(
                "Failed to read holding registers",
                address=address,
                count=count,
                error=str(e),
            )
            raise SCADAConnectionError(
                f"Failed to read holding registers: {str(e)}",
                protocol="modbus",
                original_exception=e,
            )

    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(ModbusException,))
    async def read_input_registers(
        self, address: int, count: int, slave: Optional[int] = None
    ) -> List[int]:
        """
        Read input registers from Modbus device.

        Args:
            address: Starting register address
            count: Number of registers to read
            slave: Slave ID (overrides config if provided)

        Returns:
            List of register values

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.is_connected:
            raise SCADAConnectionError(
                "Not connected to Modbus device", protocol="modbus"
            )

        slave_id = slave if slave is not None else self.config.slave_id

        try:
            response = await self.client.read_input_registers(
                address=address, count=count, slave=slave_id
            )

            if response.isError():
                raise SCADAConnectionError(
                    f"Modbus read error: {response}",
                    protocol="modbus",
                    details={"address": address, "count": count, "slave": slave_id},
                )

            return response.registers

        except Exception as e:
            logger.error(
                "Failed to read input registers",
                address=address,
                count=count,
                error=str(e),
            )
            raise SCADAConnectionError(
                f"Failed to read input registers: {str(e)}",
                protocol="modbus",
                original_exception=e,
            )

    async def read_parameters(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> List[MonitoringDataPoint]:
        """
        Read all configured parameters from the device.

        Uses the register_map from configuration to read and parse data.

        Args:
            device_id: Device identifier for the data points
            timestamp: Timestamp for the data points (defaults to current time)

        Returns:
            List of monitoring data points

        Raises:
            SCADAConnectionError: If read operation fails
        """
        if not self.config.register_map:
            logger.warning("No register map configured for device", device_id=device_id)
            return []

        timestamp = timestamp or get_utc_now()
        data_points: List[MonitoringDataPoint] = []

        for parameter, register_config in self.config.register_map.items():
            try:
                address = register_config["address"]
                count = register_config.get("count", 1)
                reg_type = register_config.get("type", "holding")
                scale = register_config.get("scale", 1.0)
                offset = register_config.get("offset", 0.0)
                unit = register_config.get("unit", "")

                # Read registers based on type
                if reg_type == "holding":
                    registers = await self.read_holding_registers(address, count)
                elif reg_type == "input":
                    registers = await self.read_input_registers(address, count)
                else:
                    logger.warning(f"Unknown register type: {reg_type}")
                    continue

                # Parse value (simple implementation, can be extended for complex types)
                if count == 1:
                    raw_value = registers[0]
                elif count == 2:
                    # 32-bit value from two 16-bit registers
                    raw_value = (registers[0] << 16) | registers[1]
                else:
                    # For more complex types, just use first register
                    raw_value = registers[0]

                # Apply scaling and offset
                value = (raw_value * scale) + offset

                # Create data point
                data_point = MonitoringDataPoint(
                    device_id=device_id,
                    timestamp=timestamp,
                    parameter=parameter,
                    value=value,
                    unit=unit,
                    quality=1.0,
                    metadata={"address": address, "registers": count},
                )
                data_points.append(data_point)

            except Exception as e:
                logger.error(
                    "Failed to read parameter",
                    parameter=parameter,
                    device_id=device_id,
                    error=str(e),
                )
                # Continue with other parameters
                continue

        logger.debug(
            "Read parameters from Modbus device",
            device_id=device_id,
            parameters=len(data_points),
        )

        return data_points

    async def __aenter__(self) -> "ModbusClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


class SunSpecClient(ModbusClient):
    """
    SunSpec-compliant Modbus client for solar inverters.

    SunSpec is a standardized Modbus register mapping for solar equipment.
    This client automatically discovers and reads SunSpec models.

    Args:
        config: Modbus configuration

    Example:
        >>> config = ModbusConfig(host="192.168.1.100", port=502, slave_id=1)
        >>> client = SunSpecClient(config)
        >>> await client.connect()
        >>> inverter_data = await client.read_inverter_model()
        >>> await client.disconnect()
    """

    # SunSpec constants
    SUNSPEC_BASE_ADDRESS = 40000
    SUNSPEC_ID = 0x53756E53  # "SunS" in ASCII

    # Common SunSpec model IDs
    MODEL_COMMON = 1
    MODEL_INVERTER_SINGLE_PHASE = 101
    MODEL_INVERTER_SPLIT_PHASE = 102
    MODEL_INVERTER_THREE_PHASE = 103
    MODEL_NAMEPLATE = 120
    MODEL_SETTINGS = 121
    MODEL_STATUS = 122
    MODEL_CONTROLS = 123

    def __init__(self, config: ModbusConfig) -> None:
        """
        Initialize SunSpec client.

        Args:
            config: Modbus configuration
        """
        super().__init__(config)
        self.sunspec_base: Optional[int] = None
        self.models: Dict[int, Dict[str, Any]] = {}
        logger.info("SunSpec client initialized")

    async def discover_sunspec(self) -> bool:
        """
        Discover SunSpec base address and available models.

        Returns:
            True if SunSpec device found, False otherwise
        """
        try:
            # Try standard SunSpec base addresses
            base_addresses = [40000, 0, 50000]

            for base in base_addresses:
                try:
                    # Read SunSpec ID
                    registers = await self.read_holding_registers(base, 2)
                    sunspec_id = (registers[0] << 16) | registers[1]

                    if sunspec_id == self.SUNSPEC_ID:
                        self.sunspec_base = base
                        logger.info(
                            "SunSpec device discovered",
                            base_address=base,
                        )
                        await self._scan_models()
                        return True

                except Exception:
                    continue

            logger.warning("SunSpec device not found")
            return False

        except Exception as e:
            logger.error("SunSpec discovery failed", error=str(e))
            return False

    async def _scan_models(self) -> None:
        """Scan for available SunSpec models."""
        if self.sunspec_base is None:
            return

        current_address = self.sunspec_base + 2  # Skip SunSpec ID

        try:
            while True:
                # Read model ID and length
                registers = await self.read_holding_registers(current_address, 2)
                model_id = registers[0]
                model_length = registers[1]

                # End of models
                if model_id == 0xFFFF or model_length == 0:
                    break

                # Store model info
                self.models[model_id] = {
                    "address": current_address,
                    "length": model_length,
                }

                logger.debug(
                    "SunSpec model found",
                    model_id=model_id,
                    address=current_address,
                    length=model_length,
                )

                # Move to next model
                current_address += 2 + model_length

                # Safety check
                if len(self.models) > 50:
                    logger.warning("Too many SunSpec models found, stopping scan")
                    break

        except Exception as e:
            logger.error("SunSpec model scan failed", error=str(e))

    async def read_inverter_model(
        self, device_id: str, timestamp: Optional[datetime] = None
    ) -> List[MonitoringDataPoint]:
        """
        Read inverter model data (models 101, 102, or 103).

        Args:
            device_id: Device identifier
            timestamp: Timestamp for data points

        Returns:
            List of monitoring data points with inverter parameters

        Raises:
            SCADAConnectionError: If no inverter model found
        """
        if self.sunspec_base is None:
            await self.discover_sunspec()

        # Find inverter model
        inverter_model = None
        for model_id in [
            self.MODEL_INVERTER_SINGLE_PHASE,
            self.MODEL_INVERTER_SPLIT_PHASE,
            self.MODEL_INVERTER_THREE_PHASE,
        ]:
            if model_id in self.models:
                inverter_model = self.models[model_id]
                break

        if not inverter_model:
            raise SCADAConnectionError(
                "No SunSpec inverter model found",
                protocol="sunspec",
                device_id=device_id,
            )

        timestamp = timestamp or get_utc_now()
        address = inverter_model["address"] + 2  # Skip header
        length = inverter_model["length"]

        try:
            # Read all inverter registers
            registers = await self.read_holding_registers(address, length)

            # Parse inverter data (simplified - real implementation would be more detailed)
            # This is a basic example - full SunSpec parsing requires model-specific logic
            data_points = [
                MonitoringDataPoint(
                    device_id=device_id,
                    timestamp=timestamp,
                    parameter="ac_current",
                    value=float(registers[0]) * 0.001,  # Scale factor example
                    unit="A",
                    quality=1.0,
                ),
                MonitoringDataPoint(
                    device_id=device_id,
                    timestamp=timestamp,
                    parameter="ac_voltage",
                    value=float(registers[2]) * 0.1,
                    unit="V",
                    quality=1.0,
                ),
                MonitoringDataPoint(
                    device_id=device_id,
                    timestamp=timestamp,
                    parameter="ac_power",
                    value=float(registers[5]) * 1.0,
                    unit="W",
                    quality=1.0,
                ),
            ]

            logger.debug(
                "Read SunSpec inverter data",
                device_id=device_id,
                parameters=len(data_points),
            )

            return data_points

        except Exception as e:
            logger.error("Failed to read SunSpec inverter model", error=str(e))
            raise SCADAConnectionError(
                f"Failed to read SunSpec inverter model: {str(e)}",
                protocol="sunspec",
                original_exception=e,
            )
