"""
Pydantic data models for PV circularity simulator.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SeverityLevel(str, Enum):
    """Severity levels for defects and anomalies."""

    NORMAL = "normal"
    WARNING = "warning"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    """Status of analysis results."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PARTIAL = "partial"


class ThermalImageMetadata(BaseModel):
    """Metadata for thermal imaging data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(description="Time of thermal image capture")
    camera_model: str = Field(description="Thermal camera model")
    ambient_temp: float = Field(gt=-50, lt=100, description="Ambient temperature in Celsius")
    measurement_distance: float = Field(gt=0, description="Distance from camera to module in meters")
    emissivity: float = Field(ge=0.0, le=1.0, description="Emissivity coefficient")
    wind_speed: Optional[float] = Field(default=None, ge=0, description="Wind speed in m/s")
    irradiance: Optional[float] = Field(default=None, ge=0, le=1500, description="Solar irradiance in W/m²")
    module_id: Optional[str] = Field(default=None, description="Module identifier")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ThermalImageData(BaseModel):
    """Complete thermal image data with temperature matrix and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature_matrix: np.ndarray = Field(description="2D temperature array in Celsius")
    metadata: ThermalImageMetadata = Field(description="Image metadata")
    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")

    @field_validator("temperature_matrix")
    @classmethod
    def validate_temperature_matrix(cls, v: np.ndarray) -> np.ndarray:
        """Validate temperature matrix shape and values."""
        if not isinstance(v, np.ndarray):
            raise ValueError("temperature_matrix must be a numpy array")
        if v.ndim != 2:
            raise ValueError("temperature_matrix must be 2-dimensional")
        if v.size == 0:
            raise ValueError("temperature_matrix cannot be empty")
        return v


class HotspotData(BaseModel):
    """Data for a detected hotspot."""

    location: Tuple[int, int] = Field(description="(row, col) location of hotspot center")
    temperature: float = Field(description="Hotspot temperature in Celsius")
    temperature_delta: float = Field(description="Temperature difference from median in Celsius")
    area_pixels: int = Field(gt=0, description="Area of hotspot in pixels")
    severity: SeverityLevel = Field(description="Severity classification")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        default=None, description="(x_min, y_min, x_max, y_max) bounding box"
    )


class IVCurveData(BaseModel):
    """IV curve measurement data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    voltage: np.ndarray = Field(description="Voltage measurements in Volts")
    current: np.ndarray = Field(description="Current measurements in Amperes")
    temperature: float = Field(description="Cell/module temperature in Celsius")
    irradiance: float = Field(ge=0, le=1500, description="Irradiance in W/m²")
    timestamp: datetime = Field(description="Measurement timestamp")
    module_id: Optional[str] = Field(default=None, description="Module identifier")
    num_cells: Optional[int] = Field(default=60, gt=0, description="Number of cells in series")
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @field_validator("voltage", "current")
    @classmethod
    def validate_arrays(cls, v: np.ndarray) -> np.ndarray:
        """Validate voltage and current arrays."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be a numpy array")
        if v.ndim != 1:
            raise ValueError("Must be 1-dimensional")
        if v.size < 10:
            raise ValueError("Must have at least 10 data points")
        return v


class ElectricalParameters(BaseModel):
    """Extracted electrical parameters from IV curve."""

    voc: float = Field(gt=0, description="Open circuit voltage in Volts")
    isc: float = Field(gt=0, description="Short circuit current in Amperes")
    vmp: float = Field(gt=0, description="Maximum power point voltage in Volts")
    imp: float = Field(gt=0, description="Maximum power point current in Amperes")
    pmp: float = Field(gt=0, description="Maximum power in Watts")
    fill_factor: float = Field(ge=0, le=1, description="Fill factor (dimensionless)")
    efficiency: Optional[float] = Field(default=None, ge=0, le=1, description="Conversion efficiency")
    rs: Optional[float] = Field(default=None, ge=0, description="Series resistance in Ohms")
    rsh: Optional[float] = Field(default=None, ge=0, description="Shunt resistance in Ohms")
    ideality_factor: Optional[float] = Field(
        default=None, ge=1, le=2, description="Diode ideality factor"
    )


class DegradationAnalysis(BaseModel):
    """Results of degradation analysis."""

    power_degradation_percent: float = Field(description="Power degradation percentage")
    voc_degradation_percent: float = Field(description="Voc degradation percentage")
    isc_degradation_percent: float = Field(description="Isc degradation percentage")
    ff_degradation_percent: float = Field(description="Fill factor degradation percentage")
    degradation_rate_per_year: Optional[float] = Field(
        default=None, description="Estimated annual degradation rate"
    )
    estimated_remaining_life_years: Optional[float] = Field(
        default=None, ge=0, description="Estimated remaining useful life"
    )
    severity: SeverityLevel = Field(description="Degradation severity level")


class AnalysisResult(BaseModel):
    """Generic result container for all analyses."""

    status: AnalysisStatus = Field(description="Analysis status")
    data: Dict[str, Any] = Field(description="Analysis results data")
    confidence: float = Field(ge=0, le=1, description="Overall confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    errors: List[str] = Field(default_factory=list, description="Analysis errors")


class ThermalAnalysisResult(BaseModel):
    """Result of thermal image analysis."""

    hotspots: List[HotspotData] = Field(description="Detected hotspots")
    mean_temperature: float = Field(description="Mean module temperature in Celsius")
    median_temperature: float = Field(description="Median module temperature in Celsius")
    max_temperature: float = Field(description="Maximum temperature in Celsius")
    min_temperature: float = Field(description="Minimum temperature in Celsius")
    temperature_std: float = Field(ge=0, description="Temperature standard deviation")
    temperature_uniformity: float = Field(
        ge=0, le=1, description="Temperature uniformity index (1 = perfectly uniform)"
    )
    bypass_diode_failures: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected bypass diode failures"
    )
    overall_severity: SeverityLevel = Field(description="Overall thermal health severity")
    confidence: float = Field(ge=0, le=1, description="Analysis confidence")


class IVAnalysisResult(BaseModel):
    """Result of IV curve analysis."""

    parameters: ElectricalParameters = Field(description="Extracted electrical parameters")
    curve_quality: float = Field(ge=0, le=1, description="Quality of IV curve measurement")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    degradation: Optional[DegradationAnalysis] = Field(
        default=None, description="Degradation analysis if baseline available"
    )
    mismatch_detected: bool = Field(default=False, description="Cell mismatch indicator")
    bypass_diode_active: bool = Field(default=False, description="Bypass diode activation indicator")
    overall_health: SeverityLevel = Field(description="Overall electrical health")
    confidence: float = Field(ge=0, le=1, description="Analysis confidence")
