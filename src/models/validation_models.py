"""Pydantic data models for PV system validation and compliance reporting.

This module defines comprehensive data models for system validation, code compliance,
engineering calculations, performance metrics, and documentation generation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SystemType(str, Enum):
    """Enumeration of supported PV system types."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    UTILITY_SCALE = "utility_scale"
    GROUND_MOUNT = "ground_mount"
    ROOFTOP = "rooftop"
    CARPORT = "carport"
    FLOATING = "floating"
    BIFACIAL = "bifacial"


class IssueSeverity(str, Enum):
    """Severity levels for validation issues."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ComplianceStatus(str, Enum):
    """Status of compliance checks."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class IssueItem(BaseModel):
    """Individual validation issue or finding."""

    severity: IssueSeverity = Field(..., description="Severity level of the issue")
    category: str = Field(..., description="Category of the issue (e.g., 'electrical', 'structural')")
    code: str = Field(..., description="Issue code or identifier")
    message: str = Field(..., description="Detailed description of the issue")
    location: Optional[str] = Field(None, description="Location or component where issue was found")
    recommendation: Optional[str] = Field(None, description="Recommended action to resolve issue")
    reference: Optional[str] = Field(None, description="Code reference or standard citation")
    timestamp: datetime = Field(default_factory=datetime.now, description="When issue was detected")

    class Config:
        json_schema_extra = {
            "example": {
                "severity": "error",
                "category": "electrical",
                "code": "NEC-690.8",
                "message": "Voltage drop exceeds 2% threshold",
                "location": "String 1, Module 15-20",
                "recommendation": "Increase wire gauge from 10AWG to 8AWG",
                "reference": "NEC 2020 Article 690.8(B)",
            }
        }


class SystemConfiguration(BaseModel):
    """PV system configuration parameters."""

    system_type: SystemType = Field(..., description="Type of PV system")
    system_name: str = Field(..., description="Name or identifier for the system")
    location: str = Field(..., description="Installation location")
    capacity_kw: float = Field(..., gt=0, description="System capacity in kW")
    module_count: int = Field(..., gt=0, description="Total number of modules")
    inverter_count: int = Field(..., gt=0, description="Total number of inverters")
    string_count: int = Field(..., gt=0, description="Total number of strings")
    modules_per_string: int = Field(..., gt=0, description="Modules per string")

    # Electrical parameters
    system_voltage_vdc: float = Field(..., gt=0, description="System DC voltage")
    max_voltage_voc: float = Field(..., gt=0, description="Maximum open circuit voltage")
    operating_voltage_vmp: float = Field(..., gt=0, description="Operating voltage at MPP")
    max_current_isc: float = Field(..., gt=0, description="Maximum short circuit current")
    operating_current_imp: float = Field(..., gt=0, description="Operating current at MPP")

    # Environmental
    ambient_temp_min: float = Field(..., description="Minimum ambient temperature (°C)")
    ambient_temp_max: float = Field(..., description="Maximum ambient temperature (°C)")
    wind_speed_max: float = Field(..., ge=0, description="Maximum wind speed (m/s)")
    snow_load: Optional[float] = Field(None, ge=0, description="Snow load (kg/m²)")

    # Code compliance
    jurisdiction: str = Field(..., description="Local jurisdiction")
    applicable_codes: List[str] = Field(
        default_factory=list,
        description="List of applicable electrical codes (NEC, IEC, etc.)"
    )

    # Additional metadata
    design_date: datetime = Field(default_factory=datetime.now, description="Design date")
    designer: Optional[str] = Field(None, description="System designer name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("capacity_kw")
    @classmethod
    def validate_capacity(cls, v: float) -> float:
        """Validate system capacity is within reasonable range."""
        if v > 1000000:  # 1 GW
            raise ValueError("System capacity exceeds reasonable limit (1 GW)")
        return v


class EngineeringCalculation(BaseModel):
    """Engineering calculation result with verification status."""

    calculation_type: str = Field(..., description="Type of calculation")
    description: str = Field(..., description="Detailed description")
    input_parameters: Dict[str, Any] = Field(..., description="Input parameters used")
    calculated_value: float = Field(..., description="Calculated result value")
    unit: str = Field(..., description="Unit of measurement")
    threshold_min: Optional[float] = Field(None, description="Minimum acceptable value")
    threshold_max: Optional[float] = Field(None, description="Maximum acceptable value")
    is_valid: bool = Field(..., description="Whether calculation passes validation")
    formula: Optional[str] = Field(None, description="Formula or equation used")
    reference: Optional[str] = Field(None, description="Reference standard or code")
    notes: Optional[str] = Field(None, description="Additional notes")

    class Config:
        json_schema_extra = {
            "example": {
                "calculation_type": "voltage_drop",
                "description": "DC voltage drop from array to inverter",
                "input_parameters": {
                    "current": 50.0,
                    "distance": 30.0,
                    "wire_gauge": "10AWG",
                    "voltage": 600.0
                },
                "calculated_value": 1.8,
                "unit": "%",
                "threshold_max": 2.0,
                "is_valid": True,
                "formula": "Vdrop = (2 * I * L * R) / V * 100",
                "reference": "NEC 690.8(B)",
            }
        }


class PerformanceMetrics(BaseModel):
    """System performance metrics and validation results."""

    annual_energy_yield_kwh: float = Field(..., ge=0, description="Annual energy yield (kWh)")
    specific_yield_kwh_kwp: float = Field(..., ge=0, description="Specific yield (kWh/kWp)")
    performance_ratio: float = Field(..., ge=0, le=1, description="Performance ratio (0-1)")
    capacity_factor: float = Field(..., ge=0, le=1, description="Capacity factor (0-1)")

    # Loss breakdown
    loss_temperature: float = Field(..., ge=0, description="Temperature losses (%)")
    loss_soiling: float = Field(..., ge=0, description="Soiling losses (%)")
    loss_shading: float = Field(..., ge=0, description="Shading losses (%)")
    loss_mismatch: float = Field(..., ge=0, description="Mismatch losses (%)")
    loss_wiring: float = Field(..., ge=0, description="Wiring losses (%)")
    loss_inverter: float = Field(..., ge=0, description="Inverter losses (%)")
    loss_degradation: float = Field(..., ge=0, description="Module degradation (%)")
    total_losses: float = Field(..., ge=0, description="Total system losses (%)")

    # Validation flags
    is_energy_yield_realistic: bool = Field(..., description="Energy yield sanity check")
    is_pr_in_range: bool = Field(..., description="PR within expected range")
    is_loss_budget_valid: bool = Field(..., description="Loss budget validation")

    benchmark_comparison: Optional[Dict[str, float]] = Field(
        None,
        description="Comparison to industry benchmarks"
    )
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Performance validation notes"
    )

    @field_validator("total_losses")
    @classmethod
    def validate_total_losses(cls, v: float) -> float:
        """Validate total losses are within reasonable range."""
        if v > 50:
            raise ValueError("Total losses exceed 50% - likely calculation error")
        return v


class ComplianceResult(BaseModel):
    """Compliance check result for a specific code or standard."""

    code_name: str = Field(..., description="Name of code or standard (e.g., 'NEC 2020')")
    section: str = Field(..., description="Code section or article")
    requirement: str = Field(..., description="Specific requirement description")
    status: ComplianceStatus = Field(..., description="Compliance status")
    checked_value: Optional[Any] = Field(None, description="Value that was checked")
    required_value: Optional[Any] = Field(None, description="Required value per code")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "code_name": "NEC 2020",
                "section": "690.7(A)",
                "requirement": "Maximum system voltage",
                "status": "passed",
                "checked_value": 1000,
                "required_value": 1500,
                "notes": "System voltage 1000V is below 1500V limit",
            }
        }


class ValidationResult(BaseModel):
    """Overall validation result for a specific validation check."""

    check_name: str = Field(..., description="Name of validation check")
    category: str = Field(..., description="Category (electrical, structural, performance, etc.)")
    status: ComplianceStatus = Field(..., description="Overall status")
    issues: List[IssueItem] = Field(default_factory=list, description="List of issues found")
    calculations: List[EngineeringCalculation] = Field(
        default_factory=list,
        description="Related calculations"
    )
    compliance_checks: List[ComplianceResult] = Field(
        default_factory=list,
        description="Compliance check results"
    )
    summary: str = Field(..., description="Summary of validation results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")


class ValidationReport(BaseModel):
    """Comprehensive system validation report."""

    report_id: str = Field(..., description="Unique report identifier")
    report_date: datetime = Field(default_factory=datetime.now, description="Report generation date")
    system_config: SystemConfiguration = Field(..., description="System configuration")

    # Validation results by category
    electrical_validation: List[ValidationResult] = Field(
        default_factory=list,
        description="Electrical validation results"
    )
    structural_validation: List[ValidationResult] = Field(
        default_factory=list,
        description="Structural validation results"
    )
    performance_validation: List[ValidationResult] = Field(
        default_factory=list,
        description="Performance validation results"
    )
    code_compliance: List[ComplianceResult] = Field(
        default_factory=list,
        description="Code compliance results"
    )

    # Performance metrics
    performance_metrics: Optional[PerformanceMetrics] = Field(
        None,
        description="System performance metrics"
    )

    # Overall summary
    total_issues: int = Field(default=0, description="Total number of issues")
    critical_issues: int = Field(default=0, description="Number of critical issues")
    errors: int = Field(default=0, description="Number of errors")
    warnings: int = Field(default=0, description="Number of warnings")
    overall_status: ComplianceStatus = Field(..., description="Overall validation status")

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations"
    )

    # Metadata
    validator: Optional[str] = Field(None, description="Validator name or system")
    validation_version: str = Field(default="1.0", description="Validation module version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def count_issues(self) -> None:
        """Count and update issue statistics."""
        all_issues: List[IssueItem] = []

        # Collect all issues
        for validation_list in [
            self.electrical_validation,
            self.structural_validation,
            self.performance_validation
        ]:
            for result in validation_list:
                all_issues.extend(result.issues)

        # Count by severity
        self.total_issues = len(all_issues)
        self.critical_issues = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)
        self.errors = sum(1 for i in all_issues if i.severity == IssueSeverity.ERROR)
        self.warnings = sum(1 for i in all_issues if i.severity == IssueSeverity.WARNING)


class DocumentPackage(BaseModel):
    """Complete documentation package for a PV system."""

    package_id: str = Field(..., description="Unique package identifier")
    system_name: str = Field(..., description="System name")
    generation_date: datetime = Field(default_factory=datetime.now, description="Generation date")

    # Document paths
    engineering_package_pdf: Optional[str] = Field(None, description="Engineering package PDF path")
    stamped_drawings_pdf: Optional[str] = Field(None, description="Stamped drawings PDF path")
    specification_sheets_pdf: Optional[str] = Field(None, description="Specification sheets PDF path")
    om_manual_pdf: Optional[str] = Field(None, description="O&M manual PDF path")
    commissioning_checklist_pdf: Optional[str] = Field(
        None,
        description="Commissioning checklist PDF path"
    )
    calculations_spreadsheet: Optional[str] = Field(
        None,
        description="Calculations spreadsheet path"
    )
    cad_drawings: List[str] = Field(default_factory=list, description="CAD drawing file paths")

    # Package metadata
    document_count: int = Field(default=0, description="Total number of documents")
    total_size_mb: float = Field(default=0.0, description="Total package size in MB")
    includes_pe_stamp: bool = Field(default=False, description="Includes PE stamp")
    includes_calculations: bool = Field(default=False, description="Includes detailed calculations")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "package_id": "PKG-2024-001",
                "system_name": "Downtown Solar Array",
                "engineering_package_pdf": "/exports/PKG-2024-001/engineering_package.pdf",
                "stamped_drawings_pdf": "/exports/PKG-2024-001/stamped_drawings.pdf",
                "document_count": 5,
                "total_size_mb": 45.2,
                "includes_pe_stamp": True,
            }
        }
