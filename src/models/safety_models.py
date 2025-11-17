"""Pydantic models for IEC 61730 safety testing and qualification.

This module contains all data models used for safety testing, validation,
and certification of PV modules according to IEC 61730-1 and IEC 61730-2 standards.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, computed_field


class SafetyClass(str, Enum):
    """Safety classification per IEC 61730-1."""
    CLASS_I = "Class I"  # Protective earthing required
    CLASS_II = "Class II"  # Double/reinforced insulation
    CLASS_III = "Class III"  # Extra-low voltage (SELV/PELV)


class FireClass(str, Enum):
    """Fire classification per IEC 61730-2 / UL 790."""
    CLASS_A = "Class A"  # Highest fire resistance
    CLASS_B = "Class B"  # Medium fire resistance
    CLASS_C = "Class C"  # Basic fire resistance
    NOT_RATED = "Not Rated"  # Does not meet minimum requirements


class ApplicationClass(str, Enum):
    """Module application class per IEC 61730."""
    CLASS_A = "Class A"  # Hazardous voltage, not accessible
    CLASS_B = "Class B"  # Hazardous voltage, accessible
    CLASS_C = "Class C"  # Safe voltage, not accessible


class TestStatus(str, Enum):
    """Status of individual test."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL_PASS = "conditional_pass"


class InsulationResistanceTestResult(BaseModel):
    """Results from insulation resistance test per IEC 61730-2 MST 01."""

    test_voltage_v: float = Field(
        ...,
        description="Applied test voltage in volts (typically 500V or 1000V DC)",
        ge=0
    )
    measured_resistance_mohm: float = Field(
        ...,
        description="Measured insulation resistance in megaohms",
        ge=0
    )
    minimum_required_mohm: float = Field(
        default=40.0,
        description="Minimum required resistance per IEC 61730-2",
        ge=0
    )
    test_duration_s: float = Field(
        default=60.0,
        description="Test duration in seconds",
        ge=0
    )
    temperature_c: float = Field(
        ...,
        description="Test temperature in degrees Celsius",
    )
    humidity_percent: float = Field(
        ...,
        description="Relative humidity during test",
        ge=0,
        le=100
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed based on measured resistance."""
        return self.measured_resistance_mohm >= self.minimum_required_mohm


class WetLeakageCurrentTestResult(BaseModel):
    """Results from wet leakage current test per IEC 61730-2 MST 02."""

    leakage_current_ua: float = Field(
        ...,
        description="Measured leakage current in microamperes",
        ge=0
    )
    maximum_allowed_ua: float = Field(
        default=275.0,
        description="Maximum allowed leakage current per IEC 61730-2",
        ge=0
    )
    test_voltage_v: float = Field(
        ...,
        description="Applied test voltage",
        ge=0
    )
    water_spray_duration_min: float = Field(
        default=10.0,
        description="Duration of water spray in minutes",
        ge=0
    )
    water_resistivity_ohm_cm: float = Field(
        ...,
        description="Water resistivity in ohm-cm",
        ge=0
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed based on leakage current."""
        return self.leakage_current_ua <= self.maximum_allowed_ua


class DielectricStrengthTestResult(BaseModel):
    """Results from dielectric strength test per IEC 61730-2 MST 03."""

    test_voltage_v: float = Field(
        ...,
        description="Applied test voltage: 1.5 × Vmax,dc + 1000V",
        ge=0
    )
    vmax_dc_v: float = Field(
        ...,
        description="Maximum DC system voltage",
        ge=0
    )
    test_duration_s: float = Field(
        default=60.0,
        description="Test duration in seconds (typically 60s)",
        ge=0
    )
    breakdown_occurred: bool = Field(
        default=False,
        description="Whether dielectric breakdown occurred"
    )
    breakdown_voltage_v: Optional[float] = Field(
        default=None,
        description="Voltage at which breakdown occurred (if any)",
        ge=0
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (no breakdown)."""
        return not self.breakdown_occurred


class GroundContinuityTestResult(BaseModel):
    """Results from ground continuity test per IEC 61730-2 MST 04."""

    measured_resistance_ohm: float = Field(
        ...,
        description="Measured ground resistance in ohms",
        ge=0
    )
    maximum_allowed_ohm: float = Field(
        default=0.1,
        description="Maximum allowed resistance per IEC 61730-2",
        ge=0
    )
    test_current_a: float = Field(
        default=10.0,
        description="Test current in amperes",
        ge=0
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed based on resistance."""
        return self.measured_resistance_ohm <= self.maximum_allowed_ohm


class BypassDiodeThermalTestResult(BaseModel):
    """Results from bypass diode thermal test per IEC 61730-2 MST 05."""

    peak_temperature_c: float = Field(
        ...,
        description="Peak diode temperature in degrees Celsius",
    )
    maximum_allowed_c: float = Field(
        default=None,
        description="Maximum allowed temperature per diode specification",
    )
    fault_current_a: float = Field(
        ...,
        description="Fault current applied during test",
        ge=0
    )
    test_duration_h: float = Field(
        ...,
        description="Test duration in hours",
        ge=0
    )
    thermal_runaway_detected: bool = Field(
        default=False,
        description="Whether thermal runaway was detected"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (no thermal runaway, within limits)."""
        if self.thermal_runaway_detected:
            return False
        if self.maximum_allowed_c is not None:
            return self.peak_temperature_c <= self.maximum_allowed_c
        return True


class ElectricalSafetyTestResult(BaseModel):
    """Aggregated results from all electrical safety tests."""

    insulation_resistance: Optional[InsulationResistanceTestResult] = None
    wet_leakage_current: Optional[WetLeakageCurrentTestResult] = None
    dielectric_strength: Optional[DielectricStrengthTestResult] = None
    ground_continuity: Optional[GroundContinuityTestResult] = None
    bypass_diode_thermal: Optional[BypassDiodeThermalTestResult] = None

    @computed_field
    @property
    def all_tests_passed(self) -> bool:
        """Check if all applicable electrical tests passed."""
        results = []
        if self.insulation_resistance:
            results.append(self.insulation_resistance.passed)
        if self.wet_leakage_current:
            results.append(self.wet_leakage_current.passed)
        if self.dielectric_strength:
            results.append(self.dielectric_strength.passed)
        if self.ground_continuity:
            results.append(self.ground_continuity.passed)
        if self.bypass_diode_thermal:
            results.append(self.bypass_diode_thermal.passed)

        return all(results) if results else False


class MechanicalLoadTestResult(BaseModel):
    """Results from mechanical load test per IEC 61730-2 MST 06."""

    applied_load_pa: float = Field(
        ...,
        description="Applied mechanical load in Pascals",
        ge=0
    )
    maximum_deflection_mm: float = Field(
        ...,
        description="Maximum measured deflection in millimeters",
        ge=0
    )
    permanent_deformation_mm: float = Field(
        default=0.0,
        description="Permanent deformation after load removal",
        ge=0
    )
    cycles_completed: int = Field(
        default=1,
        description="Number of load cycles completed",
        ge=1
    )
    visual_defects_found: bool = Field(
        default=False,
        description="Whether visual defects were found after test"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (no defects, acceptable deformation)."""
        return not self.visual_defects_found


class ImpactTestResult(BaseModel):
    """Results from impact resistance test per IEC 61730-2 MST 07."""

    ice_ball_diameter_mm: float = Field(
        default=25.0,
        description="Diameter of ice ball in millimeters",
        ge=0
    )
    impact_velocity_ms: float = Field(
        ...,
        description="Impact velocity in meters per second",
        ge=0
    )
    impact_locations: int = Field(
        ...,
        description="Number of impact locations tested",
        ge=1
    )
    cracks_detected: bool = Field(
        default=False,
        description="Whether cracks were detected"
    )
    electrical_safety_maintained: bool = Field(
        default=True,
        description="Whether electrical safety was maintained after impact"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (electrical safety maintained)."""
        return self.electrical_safety_maintained


class RobustnessOfTerminationsTestResult(BaseModel):
    """Results from robustness of terminations test per IEC 61730-2 MST 08."""

    pull_force_n: float = Field(
        ...,
        description="Applied pull force in Newtons",
        ge=0
    )
    torque_nm: float = Field(
        ...,
        description="Applied torque in Newton-meters",
        ge=0
    )
    cable_displaced: bool = Field(
        default=False,
        description="Whether cable was displaced from terminal"
    )
    terminal_damaged: bool = Field(
        default=False,
        description="Whether terminal was damaged"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (no displacement or damage)."""
        return not (self.cable_displaced or self.terminal_damaged)


class MechanicalSafetyTestResult(BaseModel):
    """Aggregated results from all mechanical safety tests."""

    mechanical_load: Optional[MechanicalLoadTestResult] = None
    impact: Optional[ImpactTestResult] = None
    robustness_of_terminations: Optional[RobustnessOfTerminationsTestResult] = None

    @computed_field
    @property
    def all_tests_passed(self) -> bool:
        """Check if all applicable mechanical tests passed."""
        results = []
        if self.mechanical_load:
            results.append(self.mechanical_load.passed)
        if self.impact:
            results.append(self.impact.passed)
        if self.robustness_of_terminations:
            results.append(self.robustness_of_terminations.passed)

        return all(results) if results else False


class SpreadOfFlameTestResult(BaseModel):
    """Results from spread of flame test per UL 790 / IEC 61730-2."""

    flame_spread_distance_cm: float = Field(
        ...,
        description="Distance flame spread across module in centimeters",
        ge=0
    )
    flame_exposure_time_min: float = Field(
        ...,
        description="Duration of flame exposure in minutes",
        ge=0
    )
    sustained_flaming_observed: bool = Field(
        default=False,
        description="Whether sustained flaming was observed"
    )
    roof_deck_penetration: bool = Field(
        default=False,
        description="Whether flame penetrated to roof deck"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)


class FirePenetrationTestResult(BaseModel):
    """Results from fire penetration test per UL 790 / IEC 61730-2."""

    burn_through_occurred: bool = Field(
        default=False,
        description="Whether burn-through occurred"
    )
    test_duration_min: float = Field(
        ...,
        description="Test duration in minutes",
        ge=0
    )
    roof_deck_damage: bool = Field(
        default=False,
        description="Whether roof deck was damaged"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)


class FireBrandTestResult(BaseModel):
    """Results from flying brand test per UL 790 / IEC 61730-2."""

    ignition_occurred: bool = Field(
        default=False,
        description="Whether ignition occurred from flying brand"
    )
    brand_size_class: str = Field(
        ...,
        description="Size classification of brand (A, B, or C)"
    )
    sustained_burning: bool = Field(
        default=False,
        description="Whether sustained burning was observed"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)


class FireSafetyTestResult(BaseModel):
    """Aggregated results from all fire safety tests."""

    spread_of_flame: Optional[SpreadOfFlameTestResult] = None
    fire_penetration: Optional[FirePenetrationTestResult] = None
    fire_brand: Optional[FireBrandTestResult] = None
    fire_classification: FireClass = Field(default=FireClass.NOT_RATED)

    @computed_field
    @property
    def fire_tests_passed(self) -> bool:
        """Check if fire tests meet minimum requirements."""
        return self.fire_classification != FireClass.NOT_RATED


class UVPreconditioningTestResult(BaseModel):
    """Results from UV preconditioning test per IEC 61730-2 MST 09."""

    uv_dose_kwh_m2: float = Field(
        ...,
        description="Total UV dose in kWh/m²",
        ge=0
    )
    required_dose_kwh_m2: float = Field(
        default=15.0,
        description="Required UV dose per IEC 61730-2",
        ge=0
    )
    test_duration_h: float = Field(
        ...,
        description="Test duration in hours",
        ge=0
    )
    visual_degradation: bool = Field(
        default=False,
        description="Whether visual degradation was observed"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (adequate UV exposure without damage)."""
        return self.uv_dose_kwh_m2 >= self.required_dose_kwh_m2


class ThermalCyclingTestResult(BaseModel):
    """Results from thermal cycling test per IEC 61730-2 MST 10."""

    cycles_completed: int = Field(
        ...,
        description="Number of thermal cycles completed",
        ge=0
    )
    required_cycles: int = Field(
        default=200,
        description="Required number of cycles per IEC 61730-2",
        ge=0
    )
    min_temperature_c: float = Field(
        default=-40.0,
        description="Minimum cycle temperature"
    )
    max_temperature_c: float = Field(
        default=85.0,
        description="Maximum cycle temperature"
    )
    electrical_failure: bool = Field(
        default=False,
        description="Whether electrical failure occurred"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (completed cycles without failure)."""
        return (self.cycles_completed >= self.required_cycles and
                not self.electrical_failure)


class HumidityFreezeTestResult(BaseModel):
    """Results from humidity-freeze test per IEC 61730-2 MST 11."""

    cycles_completed: int = Field(
        ...,
        description="Number of humidity-freeze cycles completed",
        ge=0
    )
    required_cycles: int = Field(
        default=10,
        description="Required number of cycles per IEC 61730-2",
        ge=0
    )
    humidity_phase_c: float = Field(
        default=85.0,
        description="Temperature during humidity phase"
    )
    humidity_phase_rh: float = Field(
        default=85.0,
        description="Relative humidity during humidity phase",
        ge=0,
        le=100
    )
    freeze_phase_c: float = Field(
        default=-40.0,
        description="Temperature during freeze phase"
    )
    electrical_failure: bool = Field(
        default=False,
        description="Whether electrical failure occurred"
    )
    status: TestStatus = Field(default=TestStatus.NOT_STARTED)

    @computed_field
    @property
    def passed(self) -> bool:
        """Determine if test passed (completed cycles without failure)."""
        return (self.cycles_completed >= self.required_cycles and
                not self.electrical_failure)


class EnvironmentalSafetyTestResult(BaseModel):
    """Aggregated results from all environmental safety tests."""

    uv_preconditioning: Optional[UVPreconditioningTestResult] = None
    thermal_cycling: Optional[ThermalCyclingTestResult] = None
    humidity_freeze: Optional[HumidityFreezeTestResult] = None

    @computed_field
    @property
    def all_tests_passed(self) -> bool:
        """Check if all applicable environmental tests passed."""
        results = []
        if self.uv_preconditioning:
            results.append(self.uv_preconditioning.passed)
        if self.thermal_cycling:
            results.append(self.thermal_cycling.passed)
        if self.humidity_freeze:
            results.append(self.humidity_freeze.passed)

        return all(results) if results else False


class ConstructionRequirement(BaseModel):
    """Construction requirement check per IEC 61730-1."""

    requirement_id: str = Field(
        ...,
        description="Requirement identifier (e.g., 'IEC61730-1-10.1')"
    )
    requirement_description: str = Field(
        ...,
        description="Description of the requirement"
    )
    compliant: bool = Field(
        default=False,
        description="Whether module is compliant with requirement"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes or observations"
    )


class SafetyClassification(BaseModel):
    """Overall safety classification result."""

    safety_class: SafetyClass = Field(
        ...,
        description="Safety class per IEC 61730-1"
    )
    application_class: ApplicationClass = Field(
        ...,
        description="Application class per IEC 61730"
    )
    fire_class: FireClass = Field(
        default=FireClass.NOT_RATED,
        description="Fire class rating"
    )
    max_system_voltage_v: float = Field(
        ...,
        description="Maximum system voltage in volts",
        ge=0
    )
    classification_rationale: str = Field(
        ...,
        description="Rationale for classification decision"
    )


class SafetyTestConfig(BaseModel):
    """Configuration for safety testing."""

    module_id: str = Field(
        ...,
        description="Unique module identifier"
    )
    manufacturer: str = Field(
        ...,
        description="Module manufacturer name"
    )
    model_number: str = Field(
        ...,
        description="Module model number"
    )
    serial_number: Optional[str] = Field(
        default=None,
        description="Module serial number"
    )
    max_system_voltage_v: float = Field(
        ...,
        description="Maximum system voltage",
        ge=0
    )
    module_area_m2: float = Field(
        ...,
        description="Module area in square meters",
        gt=0
    )
    application_class: ApplicationClass = Field(
        ...,
        description="Intended application class"
    )
    target_safety_class: SafetyClass = Field(
        ...,
        description="Target safety classification"
    )
    target_fire_class: Optional[FireClass] = Field(
        default=None,
        description="Target fire classification"
    )
    test_laboratory: str = Field(
        ...,
        description="Testing laboratory name"
    )
    test_date: datetime = Field(
        default_factory=datetime.now,
        description="Test execution date"
    )
    perform_electrical_tests: bool = Field(
        default=True,
        description="Whether to perform electrical safety tests"
    )
    perform_mechanical_tests: bool = Field(
        default=True,
        description="Whether to perform mechanical safety tests"
    )
    perform_fire_tests: bool = Field(
        default=False,
        description="Whether to perform fire safety tests"
    )
    perform_environmental_tests: bool = Field(
        default=True,
        description="Whether to perform environmental safety tests"
    )


class SafetyTestResult(BaseModel):
    """Complete safety test results for a module."""

    config: SafetyTestConfig
    electrical_tests: Optional[ElectricalSafetyTestResult] = None
    mechanical_tests: Optional[MechanicalSafetyTestResult] = None
    fire_tests: Optional[FireSafetyTestResult] = None
    environmental_tests: Optional[EnvironmentalSafetyTestResult] = None
    construction_requirements: List[ConstructionRequirement] = Field(
        default_factory=list
    )
    classification: Optional[SafetyClassification] = None
    overall_pass: bool = Field(
        default=False,
        description="Overall pass/fail status"
    )
    test_completion_date: Optional[datetime] = None
    notes: Optional[str] = None

    @computed_field
    @property
    def all_required_tests_passed(self) -> bool:
        """Check if all required tests passed."""
        results = []

        if self.config.perform_electrical_tests and self.electrical_tests:
            results.append(self.electrical_tests.all_tests_passed)

        if self.config.perform_mechanical_tests and self.mechanical_tests:
            results.append(self.mechanical_tests.all_tests_passed)

        if self.config.perform_fire_tests and self.fire_tests:
            results.append(self.fire_tests.fire_tests_passed)

        if self.config.perform_environmental_tests and self.environmental_tests:
            results.append(self.environmental_tests.all_tests_passed)

        # Check construction requirements
        if self.construction_requirements:
            results.append(all(req.compliant for req in self.construction_requirements))

        return all(results) if results else False


class SafetyCertificate(BaseModel):
    """IEC 61730 Safety Certificate."""

    certificate_number: str = Field(
        ...,
        description="Unique certificate number"
    )
    issue_date: datetime = Field(
        default_factory=datetime.now,
        description="Certificate issue date"
    )
    expiry_date: Optional[datetime] = Field(
        default=None,
        description="Certificate expiry date"
    )
    certification_body: str = Field(
        ...,
        description="Name of certification body (e.g., TUV, UL, IEC)"
    )
    module_info: SafetyTestConfig
    test_results: SafetyTestResult
    certified_safety_class: SafetyClass
    certified_application_class: ApplicationClass
    certified_fire_class: Optional[FireClass] = None
    special_conditions: Optional[List[str]] = Field(
        default=None,
        description="Special conditions or limitations"
    )

    @field_validator('test_results')
    @classmethod
    def validate_test_results(cls, v: SafetyTestResult) -> SafetyTestResult:
        """Ensure test results indicate pass before certification."""
        if not v.overall_pass:
            raise ValueError("Cannot issue certificate for failed test results")
        return v
