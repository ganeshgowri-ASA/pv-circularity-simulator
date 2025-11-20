"""
Pydantic models for PV module configuration and test results.

This module defines the core data models used throughout the PV circularity simulator,
including module configurations, test results, and qualification reports.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class CellTechnology(str, Enum):
    """PV cell technology types."""
    MONO_SI = "mono-Si"
    POLY_SI = "poly-Si"
    PERC = "PERC"
    TOPCON = "TOPCon"
    HJT = "HJT"
    CIGS = "CIGS"
    CDTE = "CdTe"
    PEROVSKITE = "Perovskite"


class ModuleType(str, Enum):
    """Module construction types."""
    STANDARD = "standard"
    BIFACIAL = "bifacial"
    GLASS_GLASS = "glass-glass"
    FLEXIBLE = "flexible"


class TestStatus(str, Enum):
    """Test execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL = "conditional"


class DefectType(str, Enum):
    """Visual defect types."""
    DELAMINATION = "delamination"
    BUBBLE = "bubble"
    CRACK = "crack"
    DISCOLORATION = "discoloration"
    BROKEN_CELL = "broken_cell"
    BROKEN_INTERCONNECT = "broken_interconnect"
    HOTSPOT = "hotspot"


class ModuleConfig(BaseModel):
    """
    Configuration for a PV module design.

    Attributes:
        name: Module identifier/name
        technology: Cell technology type
        module_type: Module construction type
        rated_power: Rated power at STC (W)
        voc: Open-circuit voltage at STC (V)
        isc: Short-circuit current at STC (A)
        vmp: Voltage at maximum power point (V)
        imp: Current at maximum power point (A)
        efficiency: Module efficiency (%)
        area: Module area (m²)
        cells_in_series: Number of cells in series
        cells_in_parallel: Number of parallel strings
        dimensions: Module dimensions [length, width, thickness] in mm
        weight: Module weight (kg)
        glass_thickness_front: Front glass thickness (mm)
        glass_thickness_back: Back glass thickness (mm, 0 if not glass-glass)
        encapsulant_type: Encapsulant material (EVA, POE, etc.)
        backsheet_type: Backsheet material (if applicable)
        frame_material: Frame material (aluminum, frameless, etc.)
        junction_box: Junction box model/type
        bypass_diodes: Number of bypass diodes
        temperature_coeff_pmax: Temperature coefficient of Pmax (%/°C)
        temperature_coeff_voc: Temperature coefficient of Voc (%/°C)
        temperature_coeff_isc: Temperature coefficient of Isc (%/°C)
        noct: Nominal Operating Cell Temperature (°C)
        max_system_voltage: Maximum system voltage (V)
        series_fuse_rating: Series fuse rating (A)
    """

    # Identification
    name: str = Field(..., description="Module identifier")
    technology: CellTechnology = Field(..., description="Cell technology type")
    module_type: ModuleType = Field(default=ModuleType.STANDARD, description="Module construction type")

    # Electrical characteristics at STC (25°C, 1000 W/m², AM1.5)
    rated_power: float = Field(..., gt=0, description="Rated power at STC (W)")
    voc: float = Field(..., gt=0, description="Open-circuit voltage (V)")
    isc: float = Field(..., gt=0, description="Short-circuit current (A)")
    vmp: float = Field(..., gt=0, description="Voltage at MPP (V)")
    imp: float = Field(..., gt=0, description="Current at MPP (A)")
    efficiency: float = Field(..., gt=0, lt=100, description="Module efficiency (%)")

    # Physical characteristics
    area: float = Field(..., gt=0, description="Module area (m²)")
    cells_in_series: int = Field(..., gt=0, description="Number of cells in series")
    cells_in_parallel: int = Field(default=1, gt=0, description="Number of parallel strings")
    dimensions: List[float] = Field(..., description="[length, width, thickness] in mm")
    weight: float = Field(..., gt=0, description="Module weight (kg)")

    # Materials and construction
    glass_thickness_front: float = Field(default=3.2, gt=0, description="Front glass thickness (mm)")
    glass_thickness_back: float = Field(default=0.0, ge=0, description="Back glass thickness (mm)")
    encapsulant_type: str = Field(default="EVA", description="Encapsulant material")
    backsheet_type: Optional[str] = Field(default="Tedlar", description="Backsheet material")
    frame_material: str = Field(default="Aluminum", description="Frame material")
    junction_box: str = Field(default="Standard", description="Junction box type")
    bypass_diodes: int = Field(default=3, ge=0, description="Number of bypass diodes")

    # Temperature coefficients
    temperature_coeff_pmax: float = Field(default=-0.4, description="Temp coeff of Pmax (%/°C)")
    temperature_coeff_voc: float = Field(default=-0.3, description="Temp coeff of Voc (%/°C)")
    temperature_coeff_isc: float = Field(default=0.05, description="Temp coeff of Isc (%/°C)")

    # Operating conditions
    noct: float = Field(default=45.0, gt=0, description="NOCT (°C)")
    max_system_voltage: float = Field(default=1000.0, gt=0, description="Max system voltage (V)")
    series_fuse_rating: float = Field(default=15.0, gt=0, description="Series fuse rating (A)")

    @validator('dimensions')
    def validate_dimensions(cls, v: List[float]) -> List[float]:
        """Validate dimensions list has exactly 3 positive values."""
        if len(v) != 3:
            raise ValueError("Dimensions must be [length, width, thickness]")
        if any(x <= 0 for x in v):
            raise ValueError("All dimensions must be positive")
        return v

    @validator('vmp')
    def validate_vmp(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate Vmp is less than Voc."""
        if 'voc' in values and v >= values['voc']:
            raise ValueError("Vmp must be less than Voc")
        return v

    @validator('imp')
    def validate_imp(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate Imp is less than Isc."""
        if 'isc' in values and v >= values['isc']:
            raise ValueError("Imp must be less than Isc")
        return v


class VisualDefect(BaseModel):
    """Visual defect observation."""
    defect_type: DefectType = Field(..., description="Type of defect")
    severity: str = Field(..., description="Severity: minor, major, critical")
    location: str = Field(..., description="Location description")
    size: Optional[float] = Field(None, description="Defect size (mm or mm²)")
    description: str = Field(..., description="Detailed description")


class IVCurveData(BaseModel):
    """I-V curve measurement data."""
    voltage: List[float] = Field(..., description="Voltage points (V)")
    current: List[float] = Field(..., description="Current points (A)")
    power: List[float] = Field(..., description="Power points (W)")
    voc: float = Field(..., description="Open-circuit voltage (V)")
    isc: float = Field(..., description="Short-circuit current (A)")
    vmp: float = Field(..., description="Voltage at MPP (V)")
    imp: float = Field(..., description="Current at MPP (A)")
    pmax: float = Field(..., description="Maximum power (W)")
    fill_factor: float = Field(..., description="Fill factor")

    @validator('voltage', 'current', 'power')
    def validate_array_length(cls, v: List[float]) -> List[float]:
        """Validate arrays have reasonable length."""
        if len(v) < 10:
            raise ValueError("I-V curve must have at least 10 points")
        return v


class TestResults(BaseModel):
    """
    Results from a single IEC 61215 test.

    Attributes:
        test_id: Test identifier (e.g., "MQT-10")
        test_name: Human-readable test name
        test_date: Date/time of test execution
        status: Test pass/fail status
        initial_power: Power before test (W)
        final_power: Power after test (W)
        power_degradation: Power degradation (%)
        visual_defects: List of visual defects observed
        insulation_resistance: Insulation resistance (MΩ·m²)
        wet_leakage_current: Wet leakage current (mA)
        hotspot_temperature: Maximum hotspot temperature above average (°C)
        iv_curve_before: I-V curve before test
        iv_curve_after: I-V curve after test
        test_parameters: Test-specific parameters
        observations: Additional observations
        compliance_notes: Notes on compliance with standards
    """

    test_id: str = Field(..., description="Test identifier (e.g., MQT-10)")
    test_name: str = Field(..., description="Test name")
    test_date: datetime = Field(default_factory=datetime.now, description="Test execution date")
    status: TestStatus = Field(..., description="Test status")

    # Power measurements
    initial_power: float = Field(..., description="Initial power (W)")
    final_power: float = Field(..., description="Final power (W)")
    power_degradation: float = Field(..., description="Power degradation (%)")

    # Inspections and measurements
    visual_defects: List[VisualDefect] = Field(default_factory=list, description="Visual defects")
    insulation_resistance: Optional[float] = Field(None, description="Insulation resistance (MΩ·m²)")
    wet_leakage_current: Optional[float] = Field(None, description="Wet leakage current (mA)")
    hotspot_temperature: Optional[float] = Field(None, description="Max hotspot temp above avg (°C)")

    # I-V curves
    iv_curve_before: Optional[IVCurveData] = Field(None, description="I-V curve before test")
    iv_curve_after: Optional[IVCurveData] = Field(None, description="I-V curve after test")

    # Additional data
    test_parameters: Dict[str, Any] = Field(default_factory=dict, description="Test parameters")
    observations: str = Field(default="", description="Test observations")
    compliance_notes: str = Field(default="", description="Compliance notes")

    @validator('power_degradation', always=True)
    def calculate_degradation(cls, v: Optional[float], values: Dict[str, Any]) -> float:
        """Calculate power degradation if not provided."""
        if v is not None:
            return v
        if 'initial_power' in values and 'final_power' in values:
            initial = values['initial_power']
            final = values['final_power']
            if initial > 0:
                return ((initial - final) / initial) * 100
        return 0.0


class QualificationReport(BaseModel):
    """
    Complete IEC 61215 qualification report.

    Attributes:
        module_config: Module configuration tested
        test_results: List of all test results
        overall_status: Overall qualification status
        total_power_degradation: Total cumulative power degradation (%)
        critical_failures: List of critical test failures
        report_date: Report generation date
        test_laboratory: Testing laboratory name
        test_operator: Test operator name
        standard_version: IEC 61215 version used
        summary: Executive summary
        recommendations: Recommendations for improvement
    """

    module_config: ModuleConfig = Field(..., description="Module configuration")
    test_results: List[TestResults] = Field(..., description="All test results")
    overall_status: TestStatus = Field(..., description="Overall qualification status")
    total_power_degradation: float = Field(..., description="Total power degradation (%)")
    critical_failures: List[str] = Field(default_factory=list, description="Critical failures")

    report_date: datetime = Field(default_factory=datetime.now, description="Report date")
    test_laboratory: str = Field(default="Simulated Testing", description="Laboratory name")
    test_operator: str = Field(default="Simulator", description="Operator name")
    standard_version: str = Field(default="IEC 61215:2021", description="Standard version")

    summary: str = Field(default="", description="Executive summary")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Compliance checks
    power_retention_check: bool = Field(..., description="Power retention ≥95% check")
    visual_inspection_check: bool = Field(..., description="Visual inspection check")
    insulation_resistance_check: bool = Field(..., description="Insulation resistance check")
    safety_check: bool = Field(..., description="Safety requirements check")

    @validator('overall_status', always=True)
    def determine_overall_status(cls, v: Optional[TestStatus], values: Dict[str, Any]) -> TestStatus:
        """Determine overall status from individual test results."""
        if v is not None:
            return v
        if 'test_results' in values:
            results = values['test_results']
            if any(r.status == TestStatus.FAILED for r in results):
                return TestStatus.FAILED
            elif any(r.status == TestStatus.CONDITIONAL for r in results):
                return TestStatus.CONDITIONAL
            elif all(r.status == TestStatus.PASSED for r in results):
                return TestStatus.PASSED
        return TestStatus.NOT_STARTED
