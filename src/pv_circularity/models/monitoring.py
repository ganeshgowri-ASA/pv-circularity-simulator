"""
Data models for monitoring system.

This module defines Pydantic models for real-time monitoring data,
performance metrics, alerts, and device status information.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class DeviceStatus(str, Enum):
    """Device operational status."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""

    UNDERPERFORMANCE = "underperformance"
    EQUIPMENT_FAULT = "equipment_fault"
    GRID_OUTAGE = "grid_outage"
    COMMUNICATION_LOSS = "communication_loss"
    THRESHOLD_VIOLATION = "threshold_violation"
    CUSTOM = "custom"


class MonitoringDataPoint(BaseModel):
    """
    Base model for monitoring data points.

    Represents a single measurement from a monitoring device at a specific timestamp.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "INV001",
                "timestamp": "2025-01-17T10:30:00Z",
                "parameter": "ac_power",
                "value": 250.5,
                "unit": "kW",
                "quality": 1.0,
            }
        }
    )

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the data point")
    device_id: str = Field(..., description="Device identifier", min_length=1)
    timestamp: datetime = Field(..., description="Measurement timestamp (UTC)")
    parameter: str = Field(..., description="Parameter name", min_length=1)
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Data quality indicator (0-1)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v


class InverterData(BaseModel):
    """
    Inverter monitoring data.

    Contains comprehensive data from PV inverters including power, voltage,
    current, temperature, and operational status.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "INV001",
                "timestamp": "2025-01-17T10:30:00Z",
                "dc_voltage": 600.5,
                "dc_current": 12.3,
                "dc_power": 7386.15,
                "ac_voltage": 230.0,
                "ac_current": 30.5,
                "ac_power": 7015.0,
                "frequency": 50.0,
                "power_factor": 0.95,
                "temperature": 45.2,
                "efficiency": 0.95,
                "status": "online",
                "total_energy_today": 45.5,
                "total_energy_lifetime": 125000.0,
            }
        }
    )

    device_id: str = Field(..., description="Inverter identifier")
    timestamp: datetime = Field(..., description="Measurement timestamp (UTC)")

    # DC side measurements
    dc_voltage: Optional[float] = Field(None, ge=0, description="DC voltage (V)")
    dc_current: Optional[float] = Field(None, ge=0, description="DC current (A)")
    dc_power: Optional[float] = Field(None, ge=0, description="DC power (W)")

    # AC side measurements
    ac_voltage: Optional[float] = Field(None, ge=0, description="AC voltage (V)")
    ac_current: Optional[float] = Field(None, ge=0, description="AC current (A)")
    ac_power: Optional[float] = Field(None, ge=0, description="AC power (W)")
    frequency: Optional[float] = Field(None, ge=0, description="AC frequency (Hz)")
    power_factor: Optional[float] = Field(None, ge=-1, le=1, description="Power factor")

    # Performance metrics
    temperature: Optional[float] = Field(None, description="Inverter temperature (°C)")
    efficiency: Optional[float] = Field(None, ge=0, le=1, description="Conversion efficiency")

    # Status and energy
    status: DeviceStatus = Field(default=DeviceStatus.UNKNOWN, description="Device status")
    total_energy_today: Optional[float] = Field(None, ge=0, description="Energy today (kWh)")
    total_energy_lifetime: Optional[float] = Field(
        None, ge=0, description="Lifetime energy (kWh)"
    )

    # Additional data
    alarms: list[str] = Field(default_factory=list, description="Active alarms")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v


class StringLevelData(BaseModel):
    """
    String-level monitoring data for PV arrays.

    Monitors individual strings within a PV array to detect anomalies
    and optimize performance.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "STR001",
                "inverter_id": "INV001",
                "string_number": 1,
                "timestamp": "2025-01-17T10:30:00Z",
                "voltage": 600.0,
                "current": 8.5,
                "power": 5100.0,
                "status": "online",
            }
        }
    )

    device_id: str = Field(..., description="String identifier")
    inverter_id: str = Field(..., description="Parent inverter identifier")
    string_number: int = Field(..., ge=1, description="String number within inverter")
    timestamp: datetime = Field(..., description="Measurement timestamp (UTC)")

    voltage: Optional[float] = Field(None, ge=0, description="String voltage (V)")
    current: Optional[float] = Field(None, ge=0, description="String current (A)")
    power: Optional[float] = Field(None, ge=0, description="String power (W)")
    temperature: Optional[float] = Field(None, description="String temperature (°C)")

    status: DeviceStatus = Field(default=DeviceStatus.UNKNOWN, description="String status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v


class ModuleLevelData(BaseModel):
    """
    Module-level monitoring data.

    Provides detailed monitoring at the individual PV module level,
    typically used with module-level power electronics (MLPE).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "MOD001",
                "string_id": "STR001",
                "module_number": 1,
                "timestamp": "2025-01-17T10:30:00Z",
                "voltage": 30.0,
                "current": 8.5,
                "power": 255.0,
                "temperature": 40.0,
                "status": "online",
            }
        }
    )

    device_id: str = Field(..., description="Module identifier")
    string_id: str = Field(..., description="Parent string identifier")
    module_number: int = Field(..., ge=1, description="Module number within string")
    timestamp: datetime = Field(..., description="Measurement timestamp (UTC)")

    voltage: Optional[float] = Field(None, ge=0, description="Module voltage (V)")
    current: Optional[float] = Field(None, ge=0, description="Module current (A)")
    power: Optional[float] = Field(None, ge=0, description="Module power (W)")
    temperature: Optional[float] = Field(None, description="Module temperature (°C)")

    status: DeviceStatus = Field(default=DeviceStatus.UNKNOWN, description="Module status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v


class PerformanceMetrics(BaseModel):
    """
    PV system performance metrics.

    Calculates and stores key performance indicators for PV systems including
    performance ratio, capacity factor, specific yield, and availability.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "site_id": "SITE001",
                "timestamp": "2025-01-17T10:30:00Z",
                "performance_ratio": 0.85,
                "capacity_factor": 0.25,
                "specific_yield": 4.5,
                "availability": 0.98,
                "grid_export_energy": 1250.0,
            }
        }
    )

    site_id: str = Field(..., description="Site identifier")
    timestamp: datetime = Field(..., description="Calculation timestamp (UTC)")
    period_start: datetime = Field(..., description="Metrics period start (UTC)")
    period_end: datetime = Field(..., description="Metrics period end (UTC)")

    # Key performance indicators
    performance_ratio: Optional[float] = Field(
        None, ge=0, le=1.5, description="Performance ratio (PR)"
    )
    capacity_factor: Optional[float] = Field(
        None, ge=0, le=1, description="Capacity factor (CF)"
    )
    specific_yield: Optional[float] = Field(
        None, ge=0, description="Specific yield (kWh/kWp)"
    )
    availability: Optional[float] = Field(None, ge=0, le=1, description="System availability")

    # Energy metrics
    actual_energy: Optional[float] = Field(None, ge=0, description="Actual energy produced (kWh)")
    expected_energy: Optional[float] = Field(
        None, ge=0, description="Expected energy production (kWh)"
    )
    grid_export_energy: Optional[float] = Field(
        None, ge=0, description="Energy exported to grid (kWh)"
    )

    # Environmental conditions
    avg_irradiance: Optional[float] = Field(
        None, ge=0, description="Average irradiance (W/m²)"
    )
    avg_temperature: Optional[float] = Field(
        None, description="Average ambient temperature (°C)"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("timestamp", "period_start", "period_end")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v


class AlertData(BaseModel):
    """
    Alert/notification data model.

    Represents system alerts for underperformance, equipment faults,
    grid outages, and other anomalies.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_id": "123e4567-e89b-12d3-a456-426614174000",
                "site_id": "SITE001",
                "device_id": "INV001",
                "alert_type": "underperformance",
                "severity": "warning",
                "message": "Inverter performance below threshold",
                "timestamp": "2025-01-17T10:30:00Z",
                "is_active": True,
            }
        }
    )

    alert_id: UUID = Field(default_factory=uuid4, description="Unique alert identifier")
    site_id: str = Field(..., description="Site identifier")
    device_id: Optional[str] = Field(None, description="Device identifier (if applicable)")

    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity")

    message: str = Field(..., description="Alert message", min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional alert details")

    timestamp: datetime = Field(..., description="Alert timestamp (UTC)")
    acknowledged_at: Optional[datetime] = Field(
        None, description="Acknowledgment timestamp (UTC)"
    )
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp (UTC)")

    is_active: bool = Field(default=True, description="Whether alert is currently active")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged the alert")

    @field_validator("timestamp", "acknowledged_at", "resolved_at")
    @classmethod
    def validate_timestamp(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are timezone-aware."""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v
