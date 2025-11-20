"""
Data models for SCADA systems and protocol configurations.

This module defines Pydantic models for SCADA device configurations,
protocol settings, and device management.
"""

from enum import Enum
from typing import Optional, Dict, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ProtocolType(str, Enum):
    """Supported SCADA protocol types."""

    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPCUA = "opcua"
    MQTT = "mqtt"
    BACNET = "bacnet"
    IEC61850 = "iec61850"
    SUNSPEC = "sunspec"
    PROPRIETARY = "proprietary"


class DeviceType(str, Enum):
    """Types of monitored devices."""

    INVERTER = "inverter"
    STRING_COMBINER = "string_combiner"
    WEATHER_STATION = "weather_station"
    METER = "meter"
    TRACKER = "tracker"
    BATTERY = "battery"
    TRANSFORMER = "transformer"
    SWITCHGEAR = "switchgear"
    SENSOR = "sensor"
    GATEWAY = "gateway"
    OTHER = "other"


class ProprietaryProtocol(str, Enum):
    """Supported proprietary protocols for specific manufacturers."""

    SMA = "sma"
    FRONIUS = "fronius"
    HUAWEI = "huawei"
    SUNGROW = "sungrow"
    SOLAREDGE = "solaredge"
    ABB = "abb"
    OTHER = "other"


class ModbusConfig(BaseModel):
    """
    Modbus protocol configuration.

    Supports both Modbus TCP and RTU with configurable parameters.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "host": "192.168.1.100",
                "port": 502,
                "slave_id": 1,
                "timeout": 10,
                "retries": 3,
            }
        }
    )

    host: str = Field(..., description="Modbus host/IP address")
    port: int = Field(default=502, ge=1, le=65535, description="Modbus TCP port")
    slave_id: int = Field(default=1, ge=0, le=247, description="Modbus slave/unit ID")

    timeout: int = Field(default=10, ge=1, description="Connection timeout in seconds")
    retries: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries (seconds)")

    # For Modbus RTU
    serial_port: Optional[str] = Field(None, description="Serial port (for RTU)")
    baudrate: int = Field(default=9600, description="Baudrate (for RTU)")
    parity: str = Field(default="N", description="Parity: N, E, O (for RTU)")
    stopbits: int = Field(default=1, description="Stop bits (for RTU)")
    bytesize: int = Field(default=8, description="Byte size (for RTU)")

    # Register configuration
    register_map: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Register map: {parameter: {address, count, type, scale, ...}}",
    )

    @field_validator("parity")
    @classmethod
    def validate_parity(cls, v: str) -> str:
        """Validate parity setting."""
        if v.upper() not in ("N", "E", "O"):
            raise ValueError("Parity must be N (none), E (even), or O (odd)")
        return v.upper()


class OPCUAConfig(BaseModel):
    """
    OPC UA protocol configuration.

    Supports various security policies and authentication methods.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "endpoint": "opc.tcp://192.168.1.100:4840",
                "security_policy": "None",
                "timeout": 10,
            }
        }
    )

    endpoint: str = Field(..., description="OPC UA endpoint URL")
    namespace: Optional[int] = Field(default=2, ge=0, description="OPC UA namespace index")

    # Authentication
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")

    # Security
    security_policy: str = Field(
        default="None",
        description="Security policy (None, Basic256Sha256, etc.)",
    )
    certificate_path: Optional[Path] = Field(None, description="Client certificate path")
    private_key_path: Optional[Path] = Field(None, description="Private key path")

    # Connection settings
    timeout: int = Field(default=10, ge=1, description="Connection timeout in seconds")

    # Node configuration
    node_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Node ID map: {parameter: node_id}",
    )


class MQTTConfig(BaseModel):
    """
    MQTT protocol configuration.

    Supports MQTT v3.1.1 and v5.0 with optional TLS.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "broker_host": "mqtt.example.com",
                "broker_port": 1883,
                "client_id": "pv_monitor_001",
                "qos": 1,
            }
        }
    )

    broker_host: str = Field(..., description="MQTT broker hostname/IP")
    broker_port: int = Field(default=1883, ge=1, le=65535, description="MQTT broker port")
    client_id: str = Field(..., description="MQTT client ID", min_length=1)

    # Authentication
    username: Optional[str] = Field(None, description="MQTT username")
    password: Optional[str] = Field(None, description="MQTT password")

    # TLS/SSL
    tls_enabled: bool = Field(default=False, description="Enable TLS/SSL")
    ca_cert_path: Optional[Path] = Field(None, description="CA certificate path")
    client_cert_path: Optional[Path] = Field(None, description="Client certificate path")
    client_key_path: Optional[Path] = Field(None, description="Client key path")

    # Connection settings
    keepalive: int = Field(default=60, ge=1, description="Keepalive interval in seconds")
    qos: int = Field(default=1, ge=0, le=2, description="Quality of Service (0, 1, or 2)")
    clean_session: bool = Field(default=True, description="Clean session flag")

    # Topics
    subscribe_topics: list[str] = Field(
        default_factory=list, description="Topics to subscribe to"
    )
    publish_topic_prefix: Optional[str] = Field(
        None, description="Prefix for published topics"
    )

    @field_validator("qos")
    @classmethod
    def validate_qos(cls, v: int) -> int:
        """Validate QoS level."""
        if v not in (0, 1, 2):
            raise ValueError("QoS must be 0, 1, or 2")
        return v


class BACnetConfig(BaseModel):
    """
    BACnet protocol configuration.

    Supports BACnet/IP for building automation systems.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_address": "192.168.1.100",
                "device_id": 123456,
                "port": 47808,
            }
        }
    )

    device_address: str = Field(..., description="BACnet device IP address")
    device_id: int = Field(..., ge=0, description="BACnet device instance number")
    port: int = Field(default=47808, ge=1, le=65535, description="BACnet port")

    # Network settings
    network_number: Optional[int] = Field(None, ge=0, description="BACnet network number")
    max_apdu_length: int = Field(default=1476, description="Maximum APDU length")

    # Object configuration
    object_map: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Object map: {parameter: {object_type, object_id, property, ...}}",
    )


class IEC61850Config(BaseModel):
    """
    IEC 61850 protocol configuration.

    Used for power utility automation and substations.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "host": "192.168.1.100",
                "port": 102,
                "timeout": 10,
            }
        }
    )

    host: str = Field(..., description="IEC 61850 server IP address")
    port: int = Field(default=102, ge=1, le=65535, description="IEC 61850 port (MMS)")

    timeout: int = Field(default=10, ge=1, description="Connection timeout in seconds")

    # IED (Intelligent Electronic Device) configuration
    ied_name: Optional[str] = Field(None, description="IED name")
    logical_device: Optional[str] = Field(None, description="Logical device name")

    # Data object configuration
    data_object_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Data object map: {parameter: data_object_reference}",
    )


class ProprietaryProtocolConfig(BaseModel):
    """
    Configuration for manufacturer-specific proprietary protocols.

    Supports protocols from major inverter manufacturers.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "protocol": "sma",
                "host": "192.168.1.100",
                "port": 502,
                "device_id": "123456789",
            }
        }
    )

    protocol: ProprietaryProtocol = Field(..., description="Proprietary protocol type")
    host: str = Field(..., description="Device host/IP address")
    port: int = Field(..., ge=1, le=65535, description="Communication port")

    device_id: Optional[str] = Field(None, description="Device serial number or ID")
    password: Optional[str] = Field(None, description="Device password if required")

    timeout: int = Field(default=10, ge=1, description="Connection timeout in seconds")
    retries: int = Field(default=3, ge=0, description="Number of retry attempts")

    # Protocol-specific settings
    protocol_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Protocol-specific configuration"
    )


class SCADADevice(BaseModel):
    """
    SCADA device configuration.

    Represents a monitored device with its communication protocol
    and data collection settings.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "INV001",
                "name": "Inverter 1",
                "device_type": "inverter",
                "protocol_type": "modbus_tcp",
                "enabled": True,
                "poll_interval_seconds": 5,
            }
        }
    )

    device_id: str = Field(..., description="Unique device identifier", min_length=1)
    name: str = Field(..., description="Human-readable device name", min_length=1)
    description: Optional[str] = Field(None, description="Device description")

    device_type: DeviceType = Field(..., description="Type of device")
    protocol_type: ProtocolType = Field(..., description="Communication protocol")

    # Protocol configuration (one of these based on protocol_type)
    modbus_config: Optional[ModbusConfig] = Field(None, description="Modbus configuration")
    opcua_config: Optional[OPCUAConfig] = Field(None, description="OPC UA configuration")
    mqtt_config: Optional[MQTTConfig] = Field(None, description="MQTT configuration")
    bacnet_config: Optional[BACnetConfig] = Field(None, description="BACnet configuration")
    iec61850_config: Optional[IEC61850Config] = Field(
        None, description="IEC 61850 configuration"
    )
    proprietary_config: Optional[ProprietaryProtocolConfig] = Field(
        None, description="Proprietary protocol configuration"
    )

    # Data collection settings
    enabled: bool = Field(default=True, description="Whether device is enabled for monitoring")
    poll_interval_seconds: int = Field(
        default=5, ge=1, description="Data collection interval in seconds"
    )

    # Site information
    site_id: Optional[str] = Field(None, description="Site identifier")
    location: Optional[str] = Field(None, description="Physical location")
    coordinates: Optional[Dict[str, float]] = Field(
        None, description="GPS coordinates {latitude, longitude}"
    )

    # Metadata
    manufacturer: Optional[str] = Field(None, description="Device manufacturer")
    model: Optional[str] = Field(None, description="Device model")
    serial_number: Optional[str] = Field(None, description="Device serial number")
    firmware_version: Optional[str] = Field(None, description="Firmware version")
    installation_date: Optional[str] = Field(None, description="Installation date")

    tags: list[str] = Field(default_factory=list, description="Device tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate GPS coordinates."""
        if v is not None:
            if "latitude" not in v or "longitude" not in v:
                raise ValueError("Coordinates must include latitude and longitude")
            if not (-90 <= v["latitude"] <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if not (-180 <= v["longitude"] <= 180):
                raise ValueError("Longitude must be between -180 and 180")
        return v

    def get_protocol_config(
        self,
    ) -> Union[
        ModbusConfig,
        OPCUAConfig,
        MQTTConfig,
        BACnetConfig,
        IEC61850Config,
        ProprietaryProtocolConfig,
        None,
    ]:
        """
        Get the appropriate protocol configuration for this device.

        Returns:
            Protocol configuration object based on protocol_type

        Raises:
            ValueError: If no configuration is set for the specified protocol
        """
        config_map = {
            ProtocolType.MODBUS_TCP: self.modbus_config,
            ProtocolType.MODBUS_RTU: self.modbus_config,
            ProtocolType.OPCUA: self.opcua_config,
            ProtocolType.MQTT: self.mqtt_config,
            ProtocolType.BACNET: self.bacnet_config,
            ProtocolType.IEC61850: self.iec61850_config,
            ProtocolType.SUNSPEC: self.modbus_config,  # SunSpec uses Modbus
            ProtocolType.PROPRIETARY: self.proprietary_config,
        }

        config = config_map.get(self.protocol_type)
        if config is None:
            raise ValueError(
                f"No configuration set for protocol {self.protocol_type} on device {self.device_id}"
            )
        return config
