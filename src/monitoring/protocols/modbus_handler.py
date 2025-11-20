"""
Modbus protocol handler for PV monitoring data.

This module provides Modbus TCP/RTU client functionality for polling
real-time monitoring data from PV inverters and other equipment.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from pymodbus.client import AsyncModbusTcpClient, ModbusSerialClient
    from pymodbus.exceptions import ModbusException
except ImportError:
    AsyncModbusTcpClient = None
    ModbusSerialClient = None
    ModbusException = Exception

from config.settings import Settings

logger = logging.getLogger(__name__)


class ModbusHandler:
    """
    Modbus protocol handler for equipment polling.

    Manages Modbus TCP/RTU connections and register polling for
    reading telemetry data from inverters and other Modbus devices.

    Attributes:
        settings: Application settings
        _tcp_client: Modbus TCP client instance
        _rtu_client: Modbus RTU client instance
        _connected: Connection status flag
        _register_map: Modbus register mapping configuration
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Modbus handler.

        Args:
            settings: Application settings containing Modbus configuration.
        """
        if AsyncModbusTcpClient is None:
            logger.warning("pymodbus not available, Modbus functionality disabled")

        self.settings = settings
        self._tcp_client: Optional[AsyncModbusTcpClient] = None
        self._rtu_client: Optional[ModbusSerialClient] = None
        self._connected: bool = False

        # Initialize register map for common inverter parameters
        self._register_map = self._initialize_register_map()

        logger.info("ModbusHandler initialized")

    def _initialize_register_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize Modbus register mapping.

        This is a generic register map. In production, this should be loaded
        from configuration files specific to each inverter model.

        Returns:
            Dictionary mapping parameter names to register addresses and types.
        """
        return {
            # Common Modbus registers for solar inverters (SunSpec compatible)
            'ac_power': {'address': 40083, 'count': 2, 'type': 'uint32', 'scale': 0.01, 'unit': 'kW'},
            'dc_power': {'address': 40100, 'count': 2, 'type': 'uint32', 'scale': 0.01, 'unit': 'kW'},
            'ac_voltage_l1': {'address': 40071, 'count': 1, 'type': 'uint16', 'scale': 0.1, 'unit': 'V'},
            'ac_voltage_l2': {'address': 40072, 'count': 1, 'type': 'uint16', 'scale': 0.1, 'unit': 'V'},
            'ac_voltage_l3': {'address': 40073, 'count': 1, 'type': 'uint16', 'scale': 0.1, 'unit': 'V'},
            'ac_current_l1': {'address': 40074, 'count': 1, 'type': 'uint16', 'scale': 0.01, 'unit': 'A'},
            'ac_current_l2': {'address': 40075, 'count': 1, 'type': 'uint16', 'scale': 0.01, 'unit': 'A'},
            'ac_current_l3': {'address': 40076, 'count': 1, 'type': 'uint16', 'scale': 0.01, 'unit': 'A'},
            'dc_voltage': {'address': 40101, 'count': 1, 'type': 'uint16', 'scale': 0.1, 'unit': 'V'},
            'dc_current': {'address': 40102, 'count': 1, 'type': 'uint16', 'scale': 0.01, 'unit': 'A'},
            'temperature': {'address': 40110, 'count': 1, 'type': 'int16', 'scale': 0.1, 'unit': 'C'},
            'frequency': {'address': 40085, 'count': 1, 'type': 'uint16', 'scale': 0.01, 'unit': 'Hz'},
            'energy_daily': {'address': 40090, 'count': 2, 'type': 'uint32', 'scale': 0.001, 'unit': 'kWh'},
            'energy_total': {'address': 40092, 'count': 2, 'type': 'uint32', 'scale': 0.001, 'unit': 'kWh'},
            'status': {'address': 40107, 'count': 1, 'type': 'uint16', 'scale': 1, 'unit': 'code'},
            'error_code': {'address': 40108, 'count': 1, 'type': 'uint16', 'scale': 1, 'unit': 'code'},
        }

    async def connect(self, protocol: str = 'tcp') -> None:
        """
        Connect to Modbus device.

        Args:
            protocol: Protocol type ('tcp' or 'rtu')

        Raises:
            ConnectionError: If unable to connect to Modbus device.
        """
        if AsyncModbusTcpClient is None:
            raise ImportError("pymodbus package is required for Modbus support")

        try:
            if protocol == 'tcp':
                self._tcp_client = AsyncModbusTcpClient(
                    host=self.settings.modbus.tcp_host,
                    port=self.settings.modbus.tcp_port,
                    timeout=self.settings.modbus.timeout
                )
                await self._tcp_client.connect()
                self._connected = self._tcp_client.connected

                logger.info(
                    f"Connected to Modbus TCP at {self.settings.modbus.tcp_host}:"
                    f"{self.settings.modbus.tcp_port}"
                )

            elif protocol == 'rtu':
                # For RTU, we use synchronous client due to serial port limitations
                self._rtu_client = ModbusSerialClient(
                    method='rtu',
                    port=self.settings.modbus.rtu_port,
                    baudrate=self.settings.modbus.rtu_baudrate,
                    parity=self.settings.modbus.rtu_parity,
                    stopbits=self.settings.modbus.rtu_stopbits,
                    bytesize=self.settings.modbus.rtu_bytesize,
                    timeout=self.settings.modbus.timeout
                )
                self._connected = self._rtu_client.connect()

                logger.info(f"Connected to Modbus RTU on {self.settings.modbus.rtu_port}")

            else:
                raise ValueError(f"Unsupported Modbus protocol: {protocol}")

        except Exception as e:
            logger.error(f"Failed to connect to Modbus device: {e}", exc_info=True)
            raise ConnectionError(f"Modbus connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        if self._tcp_client:
            self._tcp_client.close()
            logger.info("Disconnected from Modbus TCP")

        if self._rtu_client:
            self._rtu_client.close()
            logger.info("Disconnected from Modbus RTU")

        self._connected = False

    async def read_holding_registers(
        self,
        address: int,
        count: int,
        unit: int = 1
    ) -> Optional[List[int]]:
        """
        Read holding registers from Modbus device.

        Args:
            address: Starting register address
            count: Number of registers to read
            unit: Modbus unit/slave ID (default: 1)

        Returns:
            List of register values, or None if read failed.
        """
        if not self._connected:
            raise RuntimeError("Modbus client not connected")

        try:
            if self._tcp_client:
                result = await self._tcp_client.read_holding_registers(
                    address=address,
                    count=count,
                    slave=unit
                )
            elif self._rtu_client:
                # RTU read is synchronous, wrap in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._rtu_client.read_holding_registers(
                        address=address,
                        count=count,
                        slave=unit
                    )
                )
            else:
                raise RuntimeError("No Modbus client available")

            if result.isError():
                logger.error(f"Modbus read error at address {address}: {result}")
                return None

            return result.registers

        except Exception as e:
            logger.error(f"Error reading Modbus registers at {address}: {e}", exc_info=True)
            return None

    async def poll_inverter_data(self, unit: int = 1, inverter_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Poll complete inverter data set.

        Reads all configured inverter parameters via Modbus and returns
        structured data dictionary.

        Args:
            unit: Modbus unit/slave ID
            inverter_id: Inverter identifier (generated if None)

        Returns:
            Dictionary containing inverter telemetry data.

        Example:
            >>> data = await handler.poll_inverter_data(unit=1, inverter_id='INV001')
            >>> print(f"AC Power: {data['ac_power']} kW")
        """
        if inverter_id is None:
            inverter_id = f"INV_{unit:03d}"

        data = {
            'inverter_id': inverter_id,
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'inverter',
            'site_id': self.settings.site_id
        }

        # Read each parameter from register map
        for param_name, register_info in self._register_map.items():
            try:
                registers = await self.read_holding_registers(
                    address=register_info['address'],
                    count=register_info['count'],
                    unit=unit
                )

                if registers is None:
                    logger.warning(f"Failed to read {param_name} for unit {unit}")
                    continue

                # Parse value based on type
                value = self._parse_register_value(registers, register_info)
                data[param_name] = value

            except Exception as e:
                logger.error(f"Error reading {param_name}: {e}", exc_info=True)
                continue

        # Add status interpretation
        if 'status' in data:
            data['status'] = self._interpret_status(data['status'])

        logger.debug(f"Polled inverter data for unit {unit}: {data.get('ac_power', 'N/A')} kW")

        return data

    async def poll_data(self, units: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Poll data from multiple Modbus units.

        Args:
            units: List of unit IDs to poll. Defaults to [1] if None.

        Returns:
            Dictionary containing data from all units.
        """
        if units is None:
            units = [1]

        all_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'modbus_poll',
            'inverters': []
        }

        for unit_id in units:
            try:
                inv_data = await self.poll_inverter_data(unit=unit_id)
                all_data['inverters'].append(inv_data)
            except Exception as e:
                logger.error(f"Error polling unit {unit_id}: {e}", exc_info=True)

        return all_data

    def _parse_register_value(
        self,
        registers: List[int],
        register_info: Dict[str, Any]
    ) -> float:
        """
        Parse register values to actual value.

        Args:
            registers: List of register values
            register_info: Register information including type and scale

        Returns:
            Parsed and scaled value.
        """
        reg_type = register_info['type']
        scale = register_info['scale']

        if reg_type == 'uint16':
            value = registers[0]
        elif reg_type == 'int16':
            # Handle signed 16-bit
            value = registers[0]
            if value > 32767:
                value -= 65536
        elif reg_type == 'uint32':
            # Combine two registers (big-endian)
            value = (registers[0] << 16) | registers[1]
        elif reg_type == 'int32':
            # Combine two registers (big-endian) and handle sign
            value = (registers[0] << 16) | registers[1]
            if value > 2147483647:
                value -= 4294967296
        else:
            value = registers[0]

        # Apply scale factor
        return value * scale

    def _interpret_status(self, status_code: int) -> str:
        """
        Interpret inverter status code.

        Args:
            status_code: Numeric status code

        Returns:
            Human-readable status string.
        """
        status_map = {
            0: 'offline',
            1: 'standby',
            2: 'running',
            3: 'fault',
            4: 'maintenance'
        }

        return status_map.get(int(status_code), f'unknown({status_code})')

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Dictionary containing connection information.
        """
        return {
            'connected': self._connected,
            'tcp_configured': self._tcp_client is not None,
            'rtu_configured': self._rtu_client is not None,
            'tcp_host': f"{self.settings.modbus.tcp_host}:{self.settings.modbus.tcp_port}",
            'rtu_port': self.settings.modbus.rtu_port
        }
