"""
Pydantic models for IEC 63202 CTM testing and validation.

This module defines comprehensive data models for Cell-to-Module (CTM) testing
according to IEC 63202 standard, including test configurations, measurements,
calibration data, and test results.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field


class CellTechnology(str, Enum):
    """Supported PV cell technologies."""

    MONO_SI = "mono-Si"
    POLY_SI = "poly-Si"
    PERC = "PERC"
    TOPCON = "TOPCon"
    HJT = "HJT"
    IBC = "IBC"
    CIGS = "CIGS"
    CDTE = "CdTe"


class FlashSimulatorType(str, Enum):
    """Flash simulator light source types."""

    XENON = "Xenon"
    LED = "LED"
    HALOGEN = "Halogen"


class TestStatus(str, Enum):
    """Test execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IVCurveData(BaseModel):
    """IV curve measurement data.

    Attributes:
        voltage: Voltage points in Volts (V)
        current: Current points in Amperes (A)
        timestamp: Measurement timestamp
        temperature: Cell/module temperature in Celsius (°C)
        irradiance: Irradiance in W/m²
    """

    voltage: List[float] = Field(..., description="Voltage points (V)")
    current: List[float] = Field(..., description="Current points (A)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")
    temperature: float = Field(..., ge=-40, le=150, description="Temperature (°C)")
    irradiance: float = Field(..., ge=0, le=2000, description="Irradiance (W/m²)")

    @field_validator('voltage', 'current')
    @classmethod
    def validate_arrays(cls, v: List[float]) -> List[float]:
        """Validate that voltage and current arrays have at least 10 points."""
        if len(v) < 10:
            raise ValueError("IV curve must have at least 10 measurement points")
        return v

    @computed_field
    @property
    def voc(self) -> float:
        """Open-circuit voltage (V)."""
        return max(self.voltage)

    @computed_field
    @property
    def isc(self) -> float:
        """Short-circuit current (A)."""
        return max(self.current)

    @computed_field
    @property
    def pmax(self) -> float:
        """Maximum power (W)."""
        power = [v * i for v, i in zip(self.voltage, self.current)]
        return max(power)

    @computed_field
    @property
    def vmp(self) -> float:
        """Voltage at maximum power point (V)."""
        power = [v * i for v, i in zip(self.voltage, self.current)]
        max_idx = power.index(max(power))
        return self.voltage[max_idx]

    @computed_field
    @property
    def imp(self) -> float:
        """Current at maximum power point (A)."""
        power = [v * i for v, i in zip(self.voltage, self.current)]
        max_idx = power.index(max(power))
        return self.current[max_idx]

    @computed_field
    @property
    def fill_factor(self) -> float:
        """Fill factor (dimensionless)."""
        if self.voc * self.isc == 0:
            return 0.0
        return self.pmax / (self.voc * self.isc)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "voltage": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7],
                "current": [8.5, 8.5, 8.4, 8.3, 8.1, 7.5, 6.0, 4.0, 2.0, 0.0],
                "timestamp": "2025-11-17T10:30:00",
                "temperature": 25.0,
                "irradiance": 1000.0
            }
        }


class CellProperties(BaseModel):
    """Reference cell properties and specifications.

    Attributes:
        technology: Cell technology type
        area: Active cell area in cm²
        efficiency: Cell efficiency (%)
        voc: Open-circuit voltage at STC (V)
        isc: Short-circuit current at STC (A)
        vmp: Voltage at max power at STC (V)
        imp: Current at max power at STC (A)
        pmax: Maximum power at STC (W)
        temperature_coefficient_pmax: Pmax temperature coefficient (%/°C)
        temperature_coefficient_voc: Voc temperature coefficient (V/°C)
        temperature_coefficient_isc: Isc temperature coefficient (A/°C)
        thickness: Cell thickness in μm
        manufacturer: Cell manufacturer
        batch_number: Manufacturing batch number
    """

    technology: CellTechnology = Field(..., description="Cell technology")
    area: float = Field(..., gt=0, le=300, description="Active area (cm²)")
    efficiency: float = Field(..., gt=0, le=30, description="Efficiency (%)")
    voc: float = Field(..., gt=0, le=2.0, description="Open-circuit voltage (V)")
    isc: float = Field(..., gt=0, le=15, description="Short-circuit current (A)")
    vmp: float = Field(..., gt=0, description="Voltage at max power (V)")
    imp: float = Field(..., gt=0, description="Current at max power (A)")
    pmax: float = Field(..., gt=0, description="Maximum power (W)")
    temperature_coefficient_pmax: float = Field(
        ..., ge=-1.0, le=0.0, description="Pmax temp coefficient (%/°C)"
    )
    temperature_coefficient_voc: float = Field(
        ..., ge=-0.01, le=0.0, description="Voc temp coefficient (V/°C)"
    )
    temperature_coefficient_isc: float = Field(
        ..., ge=0.0, le=0.01, description="Isc temp coefficient (A/°C)"
    )
    thickness: float = Field(default=180.0, gt=50, le=300, description="Thickness (μm)")
    manufacturer: str = Field(default="", description="Manufacturer name")
    batch_number: str = Field(default="", description="Batch number")

    @computed_field
    @property
    def fill_factor(self) -> float:
        """Calculated fill factor."""
        if self.voc * self.isc == 0:
            return 0.0
        return self.pmax / (self.voc * self.isc)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ModuleConfiguration(BaseModel):
    """Module configuration and layout.

    Attributes:
        num_cells_series: Number of cells in series per string
        num_strings_parallel: Number of parallel strings
        cell_spacing: Cell-to-cell spacing in mm
        encapsulant_type: Encapsulant material (EVA, POE, etc.)
        glass_type: Front glass type
        backsheet_type: Backsheet material
        frame_type: Frame material
        junction_box: Junction box model
        bypass_diodes: Number of bypass diodes
    """

    num_cells_series: int = Field(..., ge=1, le=200, description="Cells in series")
    num_strings_parallel: int = Field(default=1, ge=1, le=10, description="Parallel strings")
    cell_spacing: float = Field(default=2.0, ge=0, le=10, description="Cell spacing (mm)")
    encapsulant_type: str = Field(default="EVA", description="Encapsulant material")
    glass_type: str = Field(default="3.2mm AR-coated", description="Front glass")
    backsheet_type: str = Field(default="White TPT", description="Backsheet material")
    frame_type: str = Field(default="Aluminum", description="Frame material")
    junction_box: str = Field(default="Standard IP67", description="Junction box model")
    bypass_diodes: int = Field(default=3, ge=0, le=10, description="Number of bypass diodes")

    @computed_field
    @property
    def total_cells(self) -> int:
        """Total number of cells in module."""
        return self.num_cells_series * self.num_strings_parallel


class ReferenceDeviceData(BaseModel):
    """Reference device calibration data.

    Attributes:
        device_id: Unique reference device identifier
        calibration_date: Last calibration date
        calibration_lab: Calibrating laboratory
        calibration_certificate: Certificate number
        short_circuit_current: Calibrated Isc at STC (A)
        responsivity: Device responsivity (A/(W/m²))
        temperature_coefficient: Temperature coefficient (A/°C)
        spectral_response: Spectral response curve (wavelength nm, response A/W)
        uncertainty_isc: Isc measurement uncertainty (%)
        uncertainty_temperature: Temperature measurement uncertainty (°C)
        traceability_chain: Traceability to SI units documentation
        next_calibration_due: Next calibration due date
    """

    device_id: str = Field(..., description="Device identifier")
    calibration_date: datetime = Field(..., description="Calibration date")
    calibration_lab: str = Field(..., description="Calibrating laboratory")
    calibration_certificate: str = Field(..., description="Certificate number")
    short_circuit_current: float = Field(..., gt=0, description="Calibrated Isc (A)")
    responsivity: float = Field(..., gt=0, description="Responsivity (A/(W/m²))")
    temperature_coefficient: float = Field(..., description="Temperature coefficient (A/°C)")
    spectral_response: Dict[float, float] = Field(
        default_factory=dict, description="Spectral response (wavelength: response)"
    )
    uncertainty_isc: float = Field(..., gt=0, le=5, description="Isc uncertainty (%)")
    uncertainty_temperature: float = Field(..., gt=0, le=1, description="Temp uncertainty (°C)")
    traceability_chain: str = Field(default="", description="Traceability documentation")
    next_calibration_due: datetime = Field(..., description="Next calibration date")

    @field_validator('next_calibration_due')
    @classmethod
    def validate_calibration_date(cls, v: datetime, info) -> datetime:
        """Ensure next calibration is after current calibration."""
        if 'calibration_date' in info.data and v <= info.data['calibration_date']:
            raise ValueError("Next calibration date must be after current calibration")
        return v


class FlashSimulatorData(BaseModel):
    """Flash simulator characteristics.

    Attributes:
        simulator_type: Type of flash simulator
        spectral_distribution: Spectral distribution data (wavelength: irradiance)
        spatial_uniformity: Spatial uniformity across test area (%)
        temporal_stability: Temporal stability during flash (%)
        flash_duration: Flash duration in milliseconds
        irradiance_set_point: Target irradiance (W/m²)
        temperature_control: Temperature control capability
        class_rating: Simulator class per IEC 60904-9 (A, B, C)
    """

    simulator_type: FlashSimulatorType = Field(..., description="Simulator type")
    spectral_distribution: Dict[float, float] = Field(
        default_factory=dict, description="Spectral data (wavelength: irradiance)"
    )
    spatial_uniformity: float = Field(
        default=98.0, ge=90, le=100, description="Spatial uniformity (%)"
    )
    temporal_stability: float = Field(
        default=99.0, ge=95, le=100, description="Temporal stability (%)"
    )
    flash_duration: float = Field(default=10.0, gt=0, le=100, description="Flash duration (ms)")
    irradiance_set_point: float = Field(
        default=1000.0, ge=100, le=1500, description="Irradiance (W/m²)"
    )
    temperature_control: bool = Field(default=True, description="Temperature control")
    class_rating: str = Field(default="AAA", description="IEC 60904-9 class")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class CTMLossComponents(BaseModel):
    """Detailed CTM loss component breakdown.

    All loss values are expressed as percentages (%).
    Negative values indicate gains.
    """

    optical_reflection: float = Field(default=0.0, description="Reflection losses (%)")
    optical_absorption: float = Field(default=0.0, description="Absorption in encapsulant (%)")
    optical_shading: float = Field(default=0.0, description="Grid/ribbon shading (%)")
    electrical_series_resistance: float = Field(default=0.0, description="Series resistance (%)")
    electrical_mismatch: float = Field(default=0.0, description="Cell mismatch (%)")
    thermal_assembly: float = Field(default=0.0, description="Assembly thermal effects (%)")
    spatial_non_uniformity: float = Field(default=0.0, description="Irradiance non-uniformity (%)")
    spectral_mismatch: float = Field(default=0.0, description="Spectral mismatch (%)")

    @computed_field
    @property
    def total_optical_loss(self) -> float:
        """Total optical losses (%)."""
        return self.optical_reflection + self.optical_absorption + self.optical_shading

    @computed_field
    @property
    def total_electrical_loss(self) -> float:
        """Total electrical losses (%)."""
        return self.electrical_series_resistance + self.electrical_mismatch

    @computed_field
    @property
    def total_loss(self) -> float:
        """Total CTM losses (%)."""
        return (
            self.total_optical_loss +
            self.total_electrical_loss +
            self.thermal_assembly +
            self.spatial_non_uniformity +
            self.spectral_mismatch
        )


class CTMTestConfig(BaseModel):
    """Configuration for CTM testing per IEC 63202.

    Attributes:
        test_id: Unique test identifier
        test_date: Test execution date
        laboratory: Testing laboratory
        operator: Test operator name
        cell_properties: Reference cell properties
        module_config: Module configuration
        reference_device: Reference device calibration data
        flash_simulator: Flash simulator characteristics
        num_cells_tested: Number of reference cells tested
        num_modules_tested: Number of modules tested
        stc_irradiance: STC irradiance (W/m²)
        stc_temperature: STC temperature (°C)
        stc_air_mass: STC air mass coefficient
        acceptance_criteria_min: Minimum acceptable CTM ratio (%)
        acceptance_criteria_max: Maximum acceptable CTM ratio (%)
    """

    test_id: str = Field(..., description="Unique test identifier")
    test_date: datetime = Field(default_factory=datetime.now, description="Test date")
    laboratory: str = Field(..., description="Testing laboratory")
    operator: str = Field(..., description="Test operator")
    cell_properties: CellProperties = Field(..., description="Cell properties")
    module_config: ModuleConfiguration = Field(..., description="Module configuration")
    reference_device: ReferenceDeviceData = Field(..., description="Reference device data")
    flash_simulator: FlashSimulatorData = Field(..., description="Flash simulator data")
    num_cells_tested: int = Field(default=5, ge=1, le=100, description="Number of cells tested")
    num_modules_tested: int = Field(default=3, ge=1, le=50, description="Number of modules tested")
    stc_irradiance: float = Field(default=1000.0, description="STC irradiance (W/m²)")
    stc_temperature: float = Field(default=25.0, description="STC temperature (°C)")
    stc_air_mass: float = Field(default=1.5, description="STC air mass")
    acceptance_criteria_min: float = Field(
        default=95.0, ge=80, le=100, description="Min CTM ratio (%)"
    )
    acceptance_criteria_max: float = Field(
        default=102.0, ge=100, le=110, description="Max CTM ratio (%)"
    )


class CTMTestResult(BaseModel):
    """CTM test result data.

    Attributes:
        config: Test configuration
        cell_measurements: IV curves for tested cells
        module_measurements: IV curves for tested modules
        cell_power_avg: Average cell power (W)
        cell_power_std: Standard deviation of cell power (W)
        module_power_avg: Average module power (W)
        module_power_std: Standard deviation of module power (W)
        ctm_ratio: Measured CTM ratio (%)
        ctm_ratio_uncertainty: CTM ratio uncertainty (%)
        loss_components: Detailed loss breakdown
        spectral_mismatch_factor: Spectral mismatch correction factor
        temperature_correction_factor: Temperature correction factor
        compliance_status: Test compliance status
        test_status: Test execution status
        notes: Additional test notes
    """

    config: CTMTestConfig = Field(..., description="Test configuration")
    cell_measurements: List[IVCurveData] = Field(
        default_factory=list, description="Cell IV curves"
    )
    module_measurements: List[IVCurveData] = Field(
        default_factory=list, description="Module IV curves"
    )
    cell_power_avg: float = Field(default=0.0, ge=0, description="Average cell power (W)")
    cell_power_std: float = Field(default=0.0, ge=0, description="Std dev cell power (W)")
    module_power_avg: float = Field(default=0.0, ge=0, description="Average module power (W)")
    module_power_std: float = Field(default=0.0, ge=0, description="Std dev module power (W)")
    ctm_ratio: float = Field(default=0.0, ge=0, le=110, description="CTM ratio (%)")
    ctm_ratio_uncertainty: float = Field(default=0.0, ge=0, description="CTM uncertainty (%)")
    loss_components: CTMLossComponents = Field(
        default_factory=CTMLossComponents, description="Loss breakdown"
    )
    spectral_mismatch_factor: float = Field(default=1.0, description="Spectral mismatch factor")
    temperature_correction_factor: float = Field(default=1.0, description="Temp correction factor")
    compliance_status: bool = Field(default=False, description="Compliance with IEC 63202")
    test_status: TestStatus = Field(default=TestStatus.PENDING, description="Test status")
    notes: str = Field(default="", description="Additional notes")

    @computed_field
    @property
    def expected_module_power(self) -> float:
        """Expected module power from cell measurements (W)."""
        return self.cell_power_avg * self.config.module_config.total_cells

    @computed_field
    @property
    def power_loss(self) -> float:
        """Absolute power loss (W)."""
        return self.expected_module_power - self.module_power_avg

    @computed_field
    @property
    def power_loss_percentage(self) -> float:
        """Power loss as percentage (%)."""
        if self.expected_module_power == 0:
            return 0.0
        return (self.power_loss / self.expected_module_power) * 100

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class CTMCertificate(BaseModel):
    """IEC 63202 CTM compliance certificate.

    Attributes:
        certificate_number: Unique certificate number
        issue_date: Certificate issue date
        test_result: Associated test result
        certified_by: Certifying authority
        validity_period_months: Certificate validity in months
        certificate_file_path: Path to PDF certificate
    """

    certificate_number: str = Field(..., description="Certificate number")
    issue_date: datetime = Field(default_factory=datetime.now, description="Issue date")
    test_result: CTMTestResult = Field(..., description="Test result")
    certified_by: str = Field(..., description="Certifying authority")
    validity_period_months: int = Field(default=12, ge=1, le=60, description="Validity (months)")
    certificate_file_path: str = Field(default="", description="Certificate file path")

    @computed_field
    @property
    def expiry_date(self) -> datetime:
        """Certificate expiry date."""
        from dateutil.relativedelta import relativedelta
        return self.issue_date + relativedelta(months=self.validity_period_months)

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Check if certificate is still valid."""
        return datetime.now() < self.expiry_date
