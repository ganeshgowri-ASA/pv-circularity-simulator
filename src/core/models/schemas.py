"""
Pydantic models for data schemas in the PV Circularity Simulator.

This module defines all data models used for validation, serialization,
and type safety across the real-time monitoring system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal


class DataQualityStatus(str, Enum):
    """Enumeration of data quality status values."""
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


class AlertSeverity(str, Enum):
    """Enumeration of alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Enumeration of alert types."""
    UNDERPERFORMANCE = "underperformance"
    EQUIPMENT_FAULT = "equipment_fault"
    GRID_OUTAGE = "grid_outage"
    COMMUNICATION_ERROR = "communication_error"
    DEGRADATION_ANOMALY = "degradation_anomaly"


class ProtocolType(str, Enum):
    """Enumeration of supported communication protocols."""
    MQTT = "mqtt"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    HTTP = "http"
    WEBSOCKET = "websocket"


# ============================================================================
# SCADA and Data Collection Models
# ============================================================================

class InverterData(BaseModel):
    """Model for inverter telemetry data."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    inverter_id: str = Field(..., description="Unique inverter identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")

    # Power metrics
    dc_power: float = Field(..., ge=0, description="DC power input in kW")
    ac_power: float = Field(..., ge=0, description="AC power output in kW")
    reactive_power: Optional[float] = Field(None, description="Reactive power in kVAR")
    power_factor: Optional[float] = Field(None, ge=-1, le=1, description="Power factor")

    # Voltage and current
    dc_voltage: float = Field(..., ge=0, description="DC voltage in V")
    dc_current: float = Field(..., ge=0, description="DC current in A")
    ac_voltage_l1: float = Field(..., ge=0, description="AC voltage phase L1 in V")
    ac_voltage_l2: Optional[float] = Field(None, ge=0, description="AC voltage phase L2 in V")
    ac_voltage_l3: Optional[float] = Field(None, ge=0, description="AC voltage phase L3 in V")
    ac_current_l1: float = Field(..., ge=0, description="AC current phase L1 in A")
    ac_current_l2: Optional[float] = Field(None, ge=0, description="AC current phase L2 in A")
    ac_current_l3: Optional[float] = Field(None, ge=0, description="AC current phase L3 in A")

    # Environmental
    temperature: float = Field(..., description="Inverter temperature in °C")

    # Status
    efficiency: Optional[float] = Field(None, ge=0, le=100, description="Inverter efficiency in %")
    status: str = Field(default="online", description="Inverter status")
    error_code: Optional[int] = Field(None, description="Error code if any")

    # Energy
    energy_daily: float = Field(default=0, ge=0, description="Daily energy production in kWh")
    energy_total: float = Field(default=0, ge=0, description="Total energy production in kWh")

    # Data quality
    data_quality: DataQualityStatus = Field(default=DataQualityStatus.GOOD)

    @field_validator('ac_power')
    @classmethod
    def validate_ac_power(cls, v: float, info) -> float:
        """Ensure AC power doesn't exceed DC power (accounting for losses)."""
        if 'dc_power' in info.data and v > info.data['dc_power'] * 1.05:
            raise ValueError(f"AC power ({v}) cannot exceed DC power by more than 5%")
        return v


class StringData(BaseModel):
    """Model for string-level monitoring data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    string_id: str = Field(..., description="Unique string identifier")
    inverter_id: str = Field(..., description="Parent inverter identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    voltage: float = Field(..., ge=0, description="String voltage in V")
    current: float = Field(..., ge=0, description="String current in A")
    power: float = Field(..., ge=0, description="String power in kW")

    # Optional additional metrics
    temperature: Optional[float] = Field(None, description="String temperature in °C")
    irradiance: Optional[float] = Field(None, ge=0, description="Irradiance in W/m²")

    data_quality: DataQualityStatus = Field(default=DataQualityStatus.GOOD)


class ModuleData(BaseModel):
    """Model for module-level monitoring data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    module_id: str = Field(..., description="Unique module identifier")
    string_id: str = Field(..., description="Parent string identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    voltage: float = Field(..., ge=0, description="Module voltage in V")
    current: float = Field(..., ge=0, description="Module current in A")
    power: float = Field(..., ge=0, description="Module power in W")
    temperature: float = Field(..., description="Module temperature in °C")

    # Performance metrics
    efficiency: Optional[float] = Field(None, ge=0, le=100, description="Module efficiency in %")

    # Hot spot detection
    hotspot_detected: bool = Field(default=False)
    max_cell_temperature: Optional[float] = Field(None, description="Maximum cell temperature in °C")

    data_quality: DataQualityStatus = Field(default=DataQualityStatus.GOOD)


class SCADAData(BaseModel):
    """Model for SCADA system data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    site_id: str = Field(..., description="Site identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Aggregated power
    total_dc_power: float = Field(..., ge=0, description="Total DC power in kW")
    total_ac_power: float = Field(..., ge=0, description="Total AC power in kW")

    # Environmental conditions
    irradiance: float = Field(..., ge=0, description="Plane of array irradiance in W/m²")
    ambient_temperature: float = Field(..., description="Ambient temperature in °C")
    module_temperature: Optional[float] = Field(None, description="Average module temperature in °C")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed in m/s")

    # Grid metrics
    grid_frequency: Optional[float] = Field(None, ge=0, description="Grid frequency in Hz")
    grid_voltage: Optional[float] = Field(None, ge=0, description="Grid voltage in V")

    # Availability
    available_inverters: int = Field(..., ge=0, description="Number of available inverters")
    total_inverters: int = Field(..., ge=1, description="Total number of inverters")

    protocol_type: ProtocolType = Field(default=ProtocolType.MQTT)


# ============================================================================
# Performance Metrics Models
# ============================================================================

class PerformanceRatioData(BaseModel):
    """Model for Performance Ratio (PR) calculation data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    # Instantaneous PR
    instantaneous_pr: float = Field(..., ge=0, le=200, description="Instantaneous PR in %")

    # Energy measurements
    actual_energy: float = Field(..., ge=0, description="Actual energy output in kWh")
    expected_energy: float = Field(..., ge=0, description="Expected energy output in kWh")

    # Reference conditions
    reference_irradiance: float = Field(default=1000.0, description="Reference irradiance in W/m²")
    actual_irradiance: float = Field(..., ge=0, description="Actual irradiance in W/m²")

    # Temperature correction
    temperature_coefficient: Optional[float] = Field(None, description="Temperature coefficient in %/°C")
    reference_temperature: float = Field(default=25.0, description="Reference temperature in °C")
    actual_temperature: float = Field(..., description="Actual module temperature in °C")


class CapacityFactorData(BaseModel):
    """Model for Capacity Factor calculation data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    capacity_factor: float = Field(..., ge=0, le=100, description="Capacity factor in %")

    actual_energy: float = Field(..., ge=0, description="Actual energy production in kWh")
    rated_capacity: float = Field(..., gt=0, description="Rated capacity in kW")
    time_period_hours: float = Field(..., gt=0, description="Time period in hours")


class SpecificYieldData(BaseModel):
    """Model for Specific Yield calculation data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    specific_yield: float = Field(..., ge=0, description="Specific yield in kWh/kWp")

    energy_production: float = Field(..., ge=0, description="Energy production in kWh")
    installed_capacity: float = Field(..., gt=0, description="Installed capacity in kWp")

    # Time period
    period_type: str = Field(..., description="Period type: daily, monthly, yearly")


class AvailabilityData(BaseModel):
    """Model for system availability tracking."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    availability_percentage: float = Field(..., ge=0, le=100, description="Availability in %")

    uptime_hours: float = Field(..., ge=0, description="Uptime in hours")
    total_hours: float = Field(..., gt=0, description="Total period in hours")
    downtime_hours: float = Field(..., ge=0, description="Downtime in hours")

    # Breakdown
    planned_downtime_hours: float = Field(default=0, ge=0)
    unplanned_downtime_hours: float = Field(default=0, ge=0)

    # Equipment counts
    available_components: int = Field(..., ge=0)
    total_components: int = Field(..., ge=1)


class GridExportData(BaseModel):
    """Model for grid export monitoring."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    export_power: float = Field(..., description="Export power in kW (negative if importing)")
    export_energy: float = Field(..., description="Exported energy in kWh")

    grid_voltage: float = Field(..., ge=0, description="Grid voltage in V")
    grid_frequency: float = Field(..., ge=0, description="Grid frequency in Hz")

    power_factor: float = Field(..., ge=-1, le=1, description="Power factor")
    reactive_power: Optional[float] = Field(None, description="Reactive power in kVAR")

    grid_connected: bool = Field(default=True, description="Grid connection status")


# ============================================================================
# Alert Models
# ============================================================================

class Alert(BaseModel):
    """Model for system alerts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    alert_type: AlertType
    severity: AlertSeverity

    site_id: str
    component_id: Optional[str] = Field(None, description="Affected component ID")
    component_type: Optional[str] = Field(None, description="Component type (inverter, string, module)")

    message: str = Field(..., description="Alert message")
    description: Optional[str] = Field(None, description="Detailed description")

    # Metric information
    metric_name: Optional[str] = Field(None, description="Related metric name")
    metric_value: Optional[float] = Field(None, description="Current metric value")
    threshold_value: Optional[float] = Field(None, description="Threshold that was crossed")

    # Status tracking
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)

    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(None)

    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UnderperformanceAlert(Alert):
    """Specialized alert for underperformance detection."""

    expected_power: float = Field(..., ge=0, description="Expected power in kW")
    actual_power: float = Field(..., ge=0, description="Actual power in kW")
    performance_ratio: float = Field(..., ge=0, le=100, description="Performance ratio in %")
    deviation_percentage: float = Field(..., description="Deviation from expected in %")


class EquipmentFaultAlert(Alert):
    """Specialized alert for equipment faults."""

    fault_code: Optional[str] = Field(None, description="Equipment fault code")
    fault_description: Optional[str] = Field(None, description="Fault description")
    affected_capacity: Optional[float] = Field(None, ge=0, description="Affected capacity in kW")


class GridOutageAlert(Alert):
    """Specialized alert for grid outages."""

    outage_start: datetime = Field(..., description="Outage start time")
    outage_end: Optional[datetime] = Field(None, description="Outage end time")
    affected_power: float = Field(..., ge=0, description="Affected power capacity in kW")
    estimated_energy_loss: Optional[float] = Field(None, ge=0, description="Estimated energy loss in kWh")


# ============================================================================
# WebSocket Models
# ============================================================================

class LiveDataUpdate(BaseModel):
    """Model for live data updates via WebSocket."""

    model_config = ConfigDict(str_strip_whitespace=True)

    update_type: str = Field(..., description="Type of update")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    site_id: str

    data: Dict[str, Any] = Field(..., description="Update payload")


class WebSocketMessage(BaseModel):
    """Model for WebSocket messages."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message_type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Configuration Models
# ============================================================================

class MonitoringConfig(BaseModel):
    """Model for monitoring system configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    site_id: str

    # Sampling rates
    inverter_sampling_interval: int = Field(default=5, ge=1, description="Inverter sampling interval in seconds")
    string_sampling_interval: int = Field(default=60, ge=1, description="String sampling interval in seconds")
    module_sampling_interval: int = Field(default=300, ge=1, description="Module sampling interval in seconds")

    # Alert thresholds
    underperformance_threshold: float = Field(default=80.0, ge=0, le=100, description="Underperformance threshold in %")
    temperature_threshold: float = Field(default=85.0, description="Temperature alert threshold in °C")

    # Data retention
    raw_data_retention_days: int = Field(default=90, ge=1, description="Raw data retention in days")
    aggregated_data_retention_days: int = Field(default=3650, ge=1, description="Aggregated data retention in days")

    # Protocol settings
    enabled_protocols: List[ProtocolType] = Field(default_factory=lambda: [ProtocolType.MQTT, ProtocolType.MODBUS_TCP])
