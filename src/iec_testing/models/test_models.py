"""
Pydantic models for IEC testing results and certification data.

This module defines comprehensive data models for all IEC test standards including
IEC 61215 (Module Qualification), IEC 61730 (Safety), IEC 63202 (CTM Power Loss),
IEC 63209, and IEC 63279.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TestStatus(str, Enum):
    """Test execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL_PASS = "conditional_pass"
    NOT_APPLICABLE = "not_applicable"


class CertificationBodyType(str, Enum):
    """Recognized certification bodies."""

    TUV_RHEINLAND = "tuv_rheinland"
    TUV_SUD = "tuv_sud"
    UL = "ul"
    IEC_CB = "iec_cb"
    CSA = "csa"
    JET = "jet"
    CQC = "cqc"
    VDE = "vde"
    INTERTEK = "intertek"
    SGS = "sgs"


class IECStandard(str, Enum):
    """IEC test standards."""

    IEC_61215 = "iec_61215"
    IEC_61730 = "iec_61730"
    IEC_63202 = "iec_63202"
    IEC_63209 = "iec_63209"
    IEC_63279 = "iec_63279"


class IVCurveData(BaseModel):
    """IV curve measurement data."""

    voltage: List[float] = Field(
        ..., description="Voltage points in volts", min_length=1
    )
    current: List[float] = Field(
        ..., description="Current points in amperes", min_length=1
    )
    temperature: float = Field(..., description="Cell/module temperature in Celsius")
    irradiance: float = Field(..., description="Irradiance in W/m²")
    voc: float = Field(..., description="Open circuit voltage in volts", gt=0)
    isc: float = Field(..., description="Short circuit current in amperes", gt=0)
    vmp: float = Field(..., description="Maximum power point voltage in volts", gt=0)
    imp: float = Field(..., description="Maximum power point current in amperes", gt=0)
    pmax: float = Field(..., description="Maximum power in watts", gt=0)
    fill_factor: float = Field(..., description="Fill factor (0-1)", ge=0, le=1)
    efficiency: Optional[float] = Field(None, description="Module efficiency (%)", ge=0, le=100)
    measurement_timestamp: datetime = Field(
        default_factory=datetime.now, description="Measurement timestamp"
    )

    @field_validator("voltage", "current")
    @classmethod
    def validate_equal_length(cls, v: List[float], info) -> List[float]:
        """Validate that voltage and current arrays have equal length."""
        if info.field_name == "current" and "voltage" in info.data:
            if len(v) != len(info.data["voltage"]):
                raise ValueError("Voltage and current arrays must have equal length")
        return v


class TestPhoto(BaseModel):
    """Photo documentation for test results."""

    photo_path: Path = Field(..., description="Path to photo file")
    caption: str = Field(..., description="Photo caption/description")
    test_name: str = Field(..., description="Associated test name")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Photo timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional photo metadata"
    )


class TestResult(BaseModel):
    """Base test result model."""

    test_id: str = Field(..., description="Unique test identifier")
    test_name: str = Field(..., description="Test name")
    test_standard: IECStandard = Field(..., description="IEC standard")
    module_id: str = Field(..., description="Module identifier")
    test_date: datetime = Field(
        default_factory=datetime.now, description="Test execution date"
    )
    status: TestStatus = Field(..., description="Test result status")
    measured_value: Optional[float] = Field(None, description="Measured value")
    required_value: Optional[float] = Field(None, description="Required/threshold value")
    unit: Optional[str] = Field(None, description="Measurement unit")
    notes: str = Field(default="", description="Additional notes")
    operator: Optional[str] = Field(None, description="Test operator name")
    test_equipment: Dict[str, str] = Field(
        default_factory=dict, description="Test equipment details"
    )
    environmental_conditions: Dict[str, float] = Field(
        default_factory=dict, description="Environmental conditions during test"
    )
    photos: List[TestPhoto] = Field(
        default_factory=list, description="Test photos"
    )


class IEC61215TestSequence(BaseModel):
    """IEC 61215 test sequence results (MQT - Module Qualification Test)."""

    # Visual inspection
    visual_inspection_initial: TestResult = Field(..., description="Initial visual inspection")

    # Electrical performance
    performance_at_stc: TestResult = Field(..., description="Performance at STC")

    # Insulation tests
    wet_leakage_current: TestResult = Field(..., description="Wet leakage current test")

    # Temperature tests
    thermal_cycling: TestResult = Field(..., description="Thermal cycling test (200 cycles)")
    humidity_freeze: TestResult = Field(..., description="Humidity-freeze test")
    damp_heat: TestResult = Field(..., description="Damp heat test (1000h)")

    # UV exposure
    uv_preconditioning: TestResult = Field(..., description="UV preconditioning test")

    # Mechanical tests
    mechanical_load_test: TestResult = Field(..., description="Mechanical load test")
    hail_impact: TestResult = Field(..., description="Hail impact test")

    # Hot spot test
    hot_spot_endurance: TestResult = Field(..., description="Hot spot endurance test")

    # Bypass diode tests
    bypass_diode_thermal: Optional[TestResult] = Field(
        None, description="Bypass diode thermal test"
    )

    # Final tests
    visual_inspection_final: TestResult = Field(..., description="Final visual inspection")
    performance_at_stc_final: TestResult = Field(..., description="Final performance at STC")

    # Power degradation
    power_degradation_percent: float = Field(
        ..., description="Total power degradation (%)", ge=-100, le=100
    )

    # IV curves
    iv_curve_initial: IVCurveData = Field(..., description="Initial IV curve")
    iv_curve_final: IVCurveData = Field(..., description="Final IV curve")


class IEC61215Result(BaseModel):
    """Complete IEC 61215 qualification test results."""

    standard_version: str = Field(
        default="IEC 61215:2021", description="Standard version"
    )
    test_campaign_id: str = Field(..., description="Test campaign identifier")
    module_type: str = Field(..., description="Module type/model")
    manufacturer: str = Field(..., description="Module manufacturer")
    test_lab: str = Field(..., description="Testing laboratory")
    test_start_date: datetime = Field(..., description="Test campaign start date")
    test_end_date: datetime = Field(..., description="Test campaign end date")

    # Test sequences
    test_sequence: IEC61215TestSequence = Field(..., description="Test sequence results")

    # Overall compliance
    overall_status: TestStatus = Field(..., description="Overall qualification status")
    compliance_percentage: float = Field(
        ..., description="Compliance percentage", ge=0, le=100
    )

    # Additional data
    test_report_number: Optional[str] = Field(None, description="Test report number")
    test_photos: List[TestPhoto] = Field(default_factory=list, description="Test photos")
    raw_data_path: Optional[Path] = Field(None, description="Path to raw data files")


class IEC61730SafetyTest(BaseModel):
    """IEC 61730 safety qualification test."""

    # Electrical safety
    insulation_test: TestResult = Field(..., description="Insulation resistance test")
    dielectric_withstand: TestResult = Field(..., description="Dielectric withstand test")
    ground_continuity: TestResult = Field(..., description="Ground continuity test")

    # Fire safety
    fire_test: TestResult = Field(..., description="Fire resistance test")

    # Mechanical safety
    mechanical_stress: TestResult = Field(..., description="Mechanical stress test")
    impact_test: TestResult = Field(..., description="Impact resistance test")

    # Environmental safety
    UV_test: TestResult = Field(..., description="UV exposure test")
    corrosion_test: TestResult = Field(..., description="Corrosion resistance test")


class IEC61730Result(BaseModel):
    """Complete IEC 61730 safety qualification results."""

    standard_version: str = Field(
        default="IEC 61730-1/-2:2016", description="Standard version"
    )
    test_campaign_id: str = Field(..., description="Test campaign identifier")
    module_type: str = Field(..., description="Module type/model")
    manufacturer: str = Field(..., description="Module manufacturer")
    safety_class: str = Field(..., description="Safety class (A, B, C)")
    application_class: str = Field(..., description="Application class")
    test_lab: str = Field(..., description="Testing laboratory")
    test_date: datetime = Field(..., description="Test date")

    # Safety tests
    safety_tests: IEC61730SafetyTest = Field(..., description="Safety test results")

    # Overall compliance
    overall_status: TestStatus = Field(..., description="Overall safety status")
    compliance_percentage: float = Field(
        ..., description="Compliance percentage", ge=0, le=100
    )

    # Additional data
    test_report_number: Optional[str] = Field(None, description="Test report number")
    test_photos: List[TestPhoto] = Field(default_factory=list, description="Test photos")


class CTMLossBreakdown(BaseModel):
    """CTM (Cell-to-Module) power loss breakdown."""

    optical_loss: float = Field(..., description="Optical loss (%)")
    electrical_loss: float = Field(..., description="Electrical loss (%)")
    thermal_loss: float = Field(..., description="Thermal loss (%)")
    mismatch_loss: float = Field(..., description="Cell mismatch loss (%)")
    interconnection_loss: float = Field(..., description="Interconnection loss (%)")
    inactive_area_loss: float = Field(..., description="Inactive area loss (%)")

    total_ctm_loss: float = Field(..., description="Total CTM loss (%)")
    ctm_ratio: float = Field(..., description="CTM ratio (Pmodule/Pcells)")

    @model_validator(mode="after")
    def validate_ctm_ratio(self) -> "CTMLossBreakdown":
        """Validate CTM ratio consistency with total loss."""
        expected_ratio = 1.0 - (self.total_ctm_loss / 100.0)
        if abs(self.ctm_ratio - expected_ratio) > 0.01:
            raise ValueError(
                f"CTM ratio {self.ctm_ratio} inconsistent with "
                f"total loss {self.total_ctm_loss}%"
            )
        return self


class IEC63202Result(BaseModel):
    """Complete IEC 63202 CTM power loss test results."""

    standard_version: str = Field(
        default="IEC TS 63202:2020", description="Standard version"
    )
    test_campaign_id: str = Field(..., description="Test campaign identifier")
    module_type: str = Field(..., description="Module type/model")
    manufacturer: str = Field(..., description="Module manufacturer")
    test_lab: str = Field(..., description="Testing laboratory")
    test_date: datetime = Field(..., description="Test date")

    # Cell performance
    cell_power_avg: float = Field(..., description="Average cell power (W)", gt=0)
    cell_efficiency_avg: float = Field(
        ..., description="Average cell efficiency (%)", gt=0, le=100
    )

    # Module performance
    module_power: float = Field(..., description="Module power (W)", gt=0)
    module_efficiency: float = Field(
        ..., description="Module efficiency (%)", gt=0, le=100
    )

    # Loss analysis
    ctm_loss_breakdown: CTMLossBreakdown = Field(
        ..., description="Detailed CTM loss breakdown"
    )

    # IV curves
    cell_iv_curves: List[IVCurveData] = Field(
        ..., description="Individual cell IV curves"
    )
    module_iv_curve: IVCurveData = Field(..., description="Module IV curve")

    # Overall status
    overall_status: TestStatus = Field(..., description="Overall test status")

    # Additional data
    test_report_number: Optional[str] = Field(None, description="Test report number")
    test_photos: List[TestPhoto] = Field(default_factory=list, description="Test photos")


class ComplianceMatrix(BaseModel):
    """Pass/fail compliance matrix for all tests."""

    iec_61215_tests: Dict[str, TestStatus] = Field(
        ..., description="IEC 61215 test results"
    )
    iec_61730_tests: Dict[str, TestStatus] = Field(
        ..., description="IEC 61730 test results"
    )
    iec_63202_tests: Dict[str, TestStatus] = Field(
        ..., description="IEC 63202 test results"
    )

    overall_compliance: bool = Field(..., description="Overall compliance status")
    total_tests: int = Field(..., description="Total number of tests", ge=0)
    passed_tests: int = Field(..., description="Number of passed tests", ge=0)
    failed_tests: int = Field(..., description="Number of failed tests", ge=0)
    compliance_rate: float = Field(
        ..., description="Compliance rate (%)", ge=0, le=100
    )


class ComplianceReport(BaseModel):
    """Comprehensive compliance report across all IEC standards."""

    report_id: str = Field(..., description="Unique report identifier")
    module_type: str = Field(..., description="Module type/model")
    manufacturer: str = Field(..., description="Module manufacturer")
    report_date: datetime = Field(
        default_factory=datetime.now, description="Report generation date"
    )

    # Test results
    iec_61215_result: Optional[IEC61215Result] = Field(
        None, description="IEC 61215 results"
    )
    iec_61730_result: Optional[IEC61730Result] = Field(
        None, description="IEC 61730 results"
    )
    iec_63202_result: Optional[IEC63202Result] = Field(
        None, description="IEC 63202 results"
    )

    # Compliance matrix
    compliance_matrix: ComplianceMatrix = Field(
        ..., description="Compliance matrix"
    )

    # Overall assessment
    overall_status: TestStatus = Field(..., description="Overall compliance status")
    certification_ready: bool = Field(
        ..., description="Ready for certification submission"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )

    # Report metadata
    report_author: Optional[str] = Field(None, description="Report author")
    reviewed_by: Optional[str] = Field(None, description="Report reviewer")


class CertificationStatus(BaseModel):
    """Certification application status tracking."""

    certification_body: CertificationBodyType = Field(
        ..., description="Certification body"
    )
    application_date: datetime = Field(..., description="Application submission date")
    status: str = Field(..., description="Current status")
    expected_completion_date: Optional[datetime] = Field(
        None, description="Expected completion date"
    )
    actual_completion_date: Optional[datetime] = Field(
        None, description="Actual completion date"
    )
    certificate_number: Optional[str] = Field(None, description="Certificate number")
    certificate_expiry_date: Optional[datetime] = Field(
        None, description="Certificate expiry date"
    )
    certification_cost: Optional[float] = Field(
        None, description="Certification cost", ge=0
    )
    notes: str = Field(default="", description="Status notes")


class CertificationPackage(BaseModel):
    """Complete certification package for submission."""

    package_id: str = Field(..., description="Unique package identifier")
    module_type: str = Field(..., description="Module type/model")
    manufacturer: str = Field(..., description="Module manufacturer")
    created_date: datetime = Field(
        default_factory=datetime.now, description="Package creation date"
    )

    # Target certification
    target_certifications: List[CertificationBodyType] = Field(
        ..., description="Target certification bodies"
    )
    target_standards: List[IECStandard] = Field(
        ..., description="Target IEC standards"
    )

    # Test results
    compliance_report: ComplianceReport = Field(
        ..., description="Compliance report"
    )

    # Documentation
    test_reports_paths: List[Path] = Field(
        ..., description="Paths to test report PDFs"
    )
    technical_drawings_paths: List[Path] = Field(
        default_factory=list, description="Paths to technical drawings"
    )
    bom_path: Optional[Path] = Field(None, description="Bill of materials path")
    datasheet_path: Optional[Path] = Field(None, description="Module datasheet path")

    # Photos and evidence
    test_photos: List[TestPhoto] = Field(
        default_factory=list, description="Test photos"
    )

    # Certification status tracking
    certification_statuses: List[CertificationStatus] = Field(
        default_factory=list, description="Certification status tracking"
    )

    # Metadata
    prepared_by: Optional[str] = Field(None, description="Package preparer")
    reviewed_by: Optional[str] = Field(None, description="Package reviewer")
    approved_by: Optional[str] = Field(None, description="Package approver")

    # Export metadata
    export_format: str = Field(default="pdf", description="Export format")
    custom_branding: Dict[str, Any] = Field(
        default_factory=dict, description="Custom branding settings"
    )


class TestHistory(BaseModel):
    """Historical test data for trend analysis."""

    module_type: str = Field(..., description="Module type/model")
    test_campaigns: List[str] = Field(..., description="Test campaign IDs")
    test_dates: List[datetime] = Field(..., description="Test dates")

    # Performance trends
    power_output_history: List[float] = Field(
        ..., description="Power output over time (W)"
    )
    efficiency_history: List[float] = Field(
        ..., description="Efficiency over time (%)"
    )
    degradation_history: List[float] = Field(
        ..., description="Degradation over time (%)"
    )

    # Test results history
    iec_61215_history: List[Optional[IEC61215Result]] = Field(
        default_factory=list, description="IEC 61215 results history"
    )
    iec_61730_history: List[Optional[IEC61730Result]] = Field(
        default_factory=list, description="IEC 61730 results history"
    )
    iec_63202_history: List[Optional[IEC63202Result]] = Field(
        default_factory=list, description="IEC 63202 results history"
    )

    # Statistical analysis
    mean_power: float = Field(..., description="Mean power output (W)")
    std_power: float = Field(..., description="Power output std deviation (W)", ge=0)
    mean_degradation: float = Field(..., description="Mean degradation rate (%/year)")

    @field_validator("test_campaigns", "test_dates", "power_output_history")
    @classmethod
    def validate_equal_length_lists(cls, v: List, info) -> List:
        """Validate that all history lists have equal length."""
        if info.field_name in ["test_dates", "power_output_history"] and "test_campaigns" in info.data:
            if len(v) != len(info.data["test_campaigns"]):
                raise ValueError("All history lists must have equal length")
        return v
