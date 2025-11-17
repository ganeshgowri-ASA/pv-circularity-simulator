"""
Design Validation module for PV systems.

This module provides comprehensive validation of PV system designs including
NEC compliance, string sizing, voltage/current limits, and design issue flagging.
"""

import streamlit as st
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

from src.models.pv_components import (
    SystemDesign, StringConfiguration, Inverter, PVModule,
    MountingStructure, SiteLocation, ValidationResult
)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    code: str
    severity: ValidationSeverity
    message: str
    component: str  # Which component has the issue
    recommendation: Optional[str] = None
    nec_reference: Optional[str] = None  # NEC code reference


@dataclass
class NECCompliance:
    """NEC compliance check results."""
    is_compliant: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_performed: List[str] = field(default_factory=list)
    nec_version: str = "2023"


class DesignValidation:
    """
    Comprehensive design validation engine.

    Validates PV system designs against NEC requirements, electrical limits,
    string sizing rules, and industry best practices.
    """

    def __init__(self, nec_version: str = "2023"):
        """
        Initialize the DesignValidation engine.

        Args:
            nec_version: NEC code version to validate against
        """
        self.nec_version = nec_version
        self.validation_issues: List[ValidationIssue] = []

    def validate_complete_design(self, design: SystemDesign) -> ValidationResult:
        """
        Perform complete validation of a system design.

        Args:
            design: System design to validate

        Returns:
            ValidationResult with all checks
        """
        self.validation_issues = []
        errors = []
        warnings = []
        checks = []

        # Perform all validation checks
        nec_result = self.check_nec_compliance(design)
        checks.extend(nec_result.checks_performed)

        string_result = self.validate_string_sizing(design)
        checks.append("String sizing validation")

        voltage_result = self.check_voltage_limits(design)
        checks.append("Voltage limits check")

        current_result = self.check_current_limits(design)
        checks.append("Current limits check")

        design_issues = self.flag_design_issues(design)
        checks.append("Design issues check")

        # Aggregate results
        for issue in self.validation_issues:
            if issue.severity == ValidationSeverity.ERROR:
                errors.append(f"[{issue.code}] {issue.message}")
            elif issue.severity == ValidationSeverity.WARNING:
                warnings.append(f"[{issue.code}] {issue.message}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks_performed=checks
        )

    def check_nec_compliance(self, design: SystemDesign) -> NECCompliance:
        """
        Check NEC compliance for PV system design.

        Validates against NEC Article 690 and related sections.

        Args:
            design: System design to validate

        Returns:
            NECCompliance results
        """
        compliance = NECCompliance(nec_version=self.nec_version)

        # NEC 690.7 - Maximum Voltage
        compliance.checks_performed.append("NEC 690.7 - Maximum Voltage")
        self._check_maximum_voltage_nec(design, compliance)

        # NEC 690.8 - Circuit Sizing and Current
        compliance.checks_performed.append("NEC 690.8 - Circuit Sizing and Current")
        self._check_circuit_current_nec(design, compliance)

        # NEC 690.9 - Overcurrent Protection
        compliance.checks_performed.append("NEC 690.9 - Overcurrent Protection")
        self._check_overcurrent_protection_nec(design, compliance)

        # NEC 690.12 - Rapid Shutdown
        compliance.checks_performed.append("NEC 690.12 - Rapid Shutdown")
        self._check_rapid_shutdown_nec(design, compliance)

        # NEC 690.13 - Photovoltaic System Disconnecting Means
        compliance.checks_performed.append("NEC 690.13 - Disconnecting Means")
        self._check_disconnecting_means_nec(design, compliance)

        # NEC 690.35 - Unbalanced Interconnections
        compliance.checks_performed.append("NEC 690.35 - Unbalanced Interconnections")
        self._check_balanced_phases_nec(design, compliance)

        # Set overall compliance
        compliance.is_compliant = not any(
            issue.severity == ValidationSeverity.ERROR
            for issue in compliance.issues
        )

        self.validation_issues.extend(compliance.issues)
        return compliance

    def _check_maximum_voltage_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.7 - Maximum Voltage requirements."""
        # NEC 690.7(A) - Calculate maximum voltage at lowest expected temp

        # Assume lowest temperature of -10°C (adjustable based on location)
        if design.site:
            # Could use climate data to determine actual lowest temp
            lowest_temp = -10.0  # °C
        else:
            lowest_temp = -10.0

        # Temperature correction factor
        temp_delta = 25.0 - lowest_temp  # STC is 25°C

        for idx, string_config in enumerate(design.modules):
            module = string_config.module

            # Calculate maximum voltage at lowest temp
            temp_coeff = module.temp_coeff_v_oc / 100.0  # Convert % to decimal
            v_oc_corrected = module.v_oc * (1 + temp_coeff * temp_delta)

            # String voltage
            max_string_voltage = v_oc_corrected * string_config.modules_per_string

            # Check against NEC limit (typically 600V for one/two-family dwellings, 1000V or 1500V for others)
            nec_voltage_limit = design.max_voltage

            if max_string_voltage > nec_voltage_limit:
                compliance.issues.append(ValidationIssue(
                    code="NEC-690.7-001",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"String {idx + 1} maximum voltage ({max_string_voltage:.1f}V at {lowest_temp}°C) "
                        f"exceeds NEC limit ({nec_voltage_limit}V)"
                    ),
                    component=f"String {idx + 1}",
                    recommendation=f"Reduce modules per string to {int(nec_voltage_limit / v_oc_corrected)} or fewer",
                    nec_reference="NEC 690.7(A)"
                ))

            # Check against module rating
            if max_string_voltage > module.max_system_voltage:
                compliance.issues.append(ValidationIssue(
                    code="NEC-690.7-002",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"String {idx + 1} voltage ({max_string_voltage:.1f}V) "
                        f"exceeds module max system voltage ({module.max_system_voltage}V)"
                    ),
                    component=f"String {idx + 1}",
                    recommendation="Reduce modules per string",
                    nec_reference="NEC 690.7(C)"
                ))

    def _check_circuit_current_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.8 - Circuit Sizing and Current."""
        # NEC 690.8(A)(1) - Calculate maximum circuit current

        for idx, string_config in enumerate(design.modules):
            module = string_config.module

            # Maximum circuit current = Isc × 1.25 (NEC 690.8(A)(1))
            max_circuit_current = module.i_sc * 1.25

            # Check against module series fuse rating
            if max_circuit_current > module.series_fuse_rating:
                compliance.issues.append(ValidationIssue(
                    code="NEC-690.8-001",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"String {idx + 1} maximum circuit current ({max_circuit_current:.2f}A) "
                        f"exceeds module series fuse rating ({module.series_fuse_rating}A)"
                    ),
                    component=f"String {idx + 1}",
                    recommendation="Verify module compatibility or use external overcurrent protection",
                    nec_reference="NEC 690.8(A)(1)"
                ))

            # Check parallel strings current
            total_parallel_current = max_circuit_current * string_config.num_strings

            # Verify against inverter limits
            for inv_idx, inverter in enumerate(design.inverters):
                if total_parallel_current > inverter.i_dc_max:
                    compliance.issues.append(ValidationIssue(
                        code="NEC-690.8-002",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Total DC current ({total_parallel_current:.2f}A) "
                            f"exceeds inverter {inv_idx + 1} max current ({inverter.i_dc_max}A)"
                        ),
                        component=f"Inverter {inv_idx + 1}",
                        recommendation="Reduce number of parallel strings or use larger inverter",
                        nec_reference="NEC 690.8(A)"
                    ))

    def _check_overcurrent_protection_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.9 - Overcurrent Protection."""
        # Check that overcurrent protection is sized appropriately

        for idx, string_config in enumerate(design.modules):
            if string_config.num_strings > 2:
                # NEC 690.9(C) requires overcurrent protection when >2 strings in parallel
                compliance.issues.append(ValidationIssue(
                    code="NEC-690.9-001",
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"String configuration {idx + 1} has {string_config.num_strings} "
                        f"parallel strings - overcurrent protection required"
                    ),
                    component=f"String {idx + 1}",
                    recommendation="Install appropriate string fuses or circuit breakers",
                    nec_reference="NEC 690.9(C)"
                ))

    def _check_rapid_shutdown_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.12 - Rapid Shutdown requirements."""
        # Verify inverters have rapid shutdown capability

        has_rapid_shutdown = all(
            inverter.has_rapid_shutdown
            for inverter in design.inverters
        )

        if not has_rapid_shutdown:
            compliance.issues.append(ValidationIssue(
                code="NEC-690.12-001",
                severity=ValidationSeverity.ERROR,
                message="System lacks required rapid shutdown capability",
                component="Inverters",
                recommendation="Select inverters with integrated rapid shutdown or add external RSD devices",
                nec_reference="NEC 690.12"
            ))

    def _check_disconnecting_means_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.13 - Disconnecting Means."""
        # Informational check - physical disconnects must be installed

        compliance.issues.append(ValidationIssue(
            code="NEC-690.13-001",
            severity=ValidationSeverity.INFO,
            message="Ensure DC and AC disconnecting means are installed and labeled",
            component="System",
            recommendation="Install and label all required disconnects per NEC 690.13",
            nec_reference="NEC 690.13"
        ))

    def _check_balanced_phases_nec(
        self,
        design: SystemDesign,
        compliance: NECCompliance
    ) -> None:
        """Check NEC 690.35 - Unbalanced Interconnections."""
        # Check for balanced loading on 3-phase systems

        three_phase_inverters = [
            inv for inv in design.inverters
            if inv.phases == 3
        ]

        if three_phase_inverters:
            total_power_per_phase = sum(inv.p_ac_rated for inv in three_phase_inverters) / 3

            compliance.issues.append(ValidationIssue(
                code="NEC-690.35-001",
                severity=ValidationSeverity.INFO,
                message=(
                    f"Three-phase system - verify balanced loading "
                    f"(~{total_power_per_phase / 1000:.1f}kW per phase)"
                ),
                component="Three-phase inverters",
                recommendation="Ensure phases are balanced within 20% per NEC 690.35",
                nec_reference="NEC 690.35"
            ))

    def validate_string_sizing(self, design: SystemDesign) -> bool:
        """
        Validate string sizing for optimal performance.

        Args:
            design: System design to validate

        Returns:
            True if string sizing is valid
        """
        is_valid = True

        for idx, string_config in enumerate(design.modules):
            # Check MPPT voltage range for each inverter
            for inv_idx, inverter in enumerate(design.inverters):
                # String Vmp should be in MPPT range
                string_vmp = string_config.string_v_mp

                if string_vmp < inverter.v_mpp_min:
                    self.validation_issues.append(ValidationIssue(
                        code="STR-001",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"String {idx + 1} Vmp ({string_vmp:.1f}V) below "
                            f"inverter {inv_idx + 1} MPPT minimum ({inverter.v_mpp_min}V)"
                        ),
                        component=f"String {idx + 1}",
                        recommendation=f"Increase modules per string to at least {int(inverter.v_mpp_min / string_config.module.v_mp) + 1}"
                    ))
                    is_valid = False

                if string_vmp > inverter.v_mpp_max:
                    self.validation_issues.append(ValidationIssue(
                        code="STR-002",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"String {idx + 1} Vmp ({string_vmp:.1f}V) above "
                            f"inverter {inv_idx + 1} MPPT maximum ({inverter.v_mpp_max}V)"
                        ),
                        component=f"String {idx + 1}",
                        recommendation=f"Reduce modules per string to {int(inverter.v_mpp_max / string_config.module.v_mp)}"
                    ))
                    is_valid = False

                # Check if string is within 80-95% of MPPT range (optimal)
                mppt_center = (inverter.v_mpp_min + inverter.v_mpp_max) / 2
                deviation = abs(string_vmp - mppt_center) / mppt_center

                if deviation > 0.3:  # More than 30% from center
                    self.validation_issues.append(ValidationIssue(
                        code="STR-003",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"String {idx + 1} Vmp not optimally positioned in MPPT range "
                            f"(current: {string_vmp:.1f}V, optimal: ~{mppt_center:.1f}V)"
                        ),
                        component=f"String {idx + 1}",
                        recommendation="Adjust modules per string for better MPPT tracking"
                    ))

        return is_valid

    def check_voltage_limits(self, design: SystemDesign) -> bool:
        """
        Check all voltage limits in the system.

        Args:
            design: System design to validate

        Returns:
            True if all voltage limits are satisfied
        """
        is_valid = True

        # Temperature range for calculations
        temp_low = -10.0  # °C
        temp_high = 70.0  # °C (cell temperature)

        for idx, string_config in enumerate(design.modules):
            module = string_config.module

            # Calculate voltage range
            temp_coeff_voc = module.temp_coeff_v_oc / 100.0
            temp_coeff_vmp = module.temp_coeff_p_max / 100.0  # Approximate Vmp coeff

            # Maximum voltage (cold)
            v_oc_max = module.v_oc * (1 + temp_coeff_voc * (25 - temp_low))
            string_v_oc_max = v_oc_max * string_config.modules_per_string

            # Minimum voltage (hot)
            v_mp_min = module.v_mp * (1 + temp_coeff_vmp * (25 - temp_high))
            string_v_mp_min = v_mp_min * string_config.modules_per_string

            # Check against system limits
            if string_v_oc_max > design.max_voltage:
                self.validation_issues.append(ValidationIssue(
                    code="VOLT-001",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"String {idx + 1} max voltage ({string_v_oc_max:.1f}V) "
                        f"exceeds system limit ({design.max_voltage}V)"
                    ),
                    component=f"String {idx + 1}",
                    recommendation="Reduce modules per string"
                ))
                is_valid = False

            if string_v_mp_min < design.min_voltage:
                self.validation_issues.append(ValidationIssue(
                    code="VOLT-002",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"String {idx + 1} min operating voltage ({string_v_mp_min:.1f}V) "
                        f"below system minimum ({design.min_voltage}V) at high temperatures"
                    ),
                    component=f"String {idx + 1}",
                    recommendation="Increase modules per string or verify operating temperature range"
                ))

        return is_valid

    def verify_current_limits(self, design: SystemDesign) -> bool:
        """
        Verify all current limits in the system.

        Args:
            design: System design to validate

        Returns:
            True if all current limits are satisfied
        """
        is_valid = True

        for idx, string_config in enumerate(design.modules):
            module = string_config.module

            # String current (Isc with safety factor)
            string_isc = module.i_sc * 1.25  # NEC safety factor

            # Total parallel current
            total_current = string_isc * string_config.num_strings

            # Check against inverter limits
            for inv_idx, inverter in enumerate(design.inverters):
                if string_isc > inverter.i_dc_max:
                    self.validation_issues.append(ValidationIssue(
                        code="CURR-001",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"String current ({string_isc:.2f}A) exceeds "
                            f"inverter {inv_idx + 1} max current ({inverter.i_dc_max}A)"
                        ),
                        component=f"Inverter {inv_idx + 1}",
                        recommendation="Use inverter with higher current rating"
                    ))
                    is_valid = False

                # Check per-MPPT current
                strings_per_mppt = string_config.num_strings / inverter.num_mppt
                current_per_mppt = string_isc * strings_per_mppt

                # Typical MPPT can handle Idc_max / num_mppt
                mppt_current_limit = inverter.i_dc_max / inverter.num_mppt

                if current_per_mppt > mppt_current_limit:
                    self.validation_issues.append(ValidationIssue(
                        code="CURR-002",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Current per MPPT ({current_per_mppt:.2f}A) may exceed "
                            f"typical limit ({mppt_current_limit:.2f}A)"
                        ),
                        component=f"Inverter {inv_idx + 1}",
                        recommendation="Distribute strings more evenly across MPPTs"
                    ))

        return is_valid

    def flag_design_issues(self, design: SystemDesign) -> List[ValidationIssue]:
        """
        Flag potential design issues and best practice violations.

        Args:
            design: System design to analyze

        Returns:
            List of flagged issues
        """
        issues = []

        # Check DC/AC ratio
        if design.total_ac_power > 0:
            dc_ac_ratio = design.total_dc_power / design.total_ac_power

            if dc_ac_ratio < 1.0:
                issues.append(ValidationIssue(
                    code="DESIGN-001",
                    severity=ValidationSeverity.WARNING,
                    message=f"DC/AC ratio ({dc_ac_ratio:.2f}) below 1.0 - system is undersized",
                    component="System",
                    recommendation="Consider adding more modules to improve capacity factor"
                ))

            if dc_ac_ratio > 1.5:
                issues.append(ValidationIssue(
                    code="DESIGN-002",
                    severity=ValidationSeverity.WARNING,
                    message=f"DC/AC ratio ({dc_ac_ratio:.2f}) above 1.5 - significant clipping expected",
                    component="System",
                    recommendation="Consider adding inverter capacity or reducing module count"
                ))

        # Check module orientation
        if design.mounting:
            if design.mounting.azimuth < 135 or design.mounting.azimuth > 225:
                # Not facing south (in Northern hemisphere)
                if design.site and design.site.latitude > 0:
                    issues.append(ValidationIssue(
                        code="DESIGN-003",
                        severity=ValidationSeverity.WARNING,
                        message=f"Array azimuth ({design.mounting.azimuth}°) not facing south - reduced performance expected",
                        component="Mounting",
                        recommendation="Consider south-facing orientation (180°) for optimal performance"
                    ))

            # Check tilt angle vs latitude
            if design.site:
                optimal_tilt = abs(design.site.latitude)
                tilt_deviation = abs(design.mounting.tilt_angle - optimal_tilt)

                if tilt_deviation > 15:
                    issues.append(ValidationIssue(
                        code="DESIGN-004",
                        severity=ValidationSeverity.INFO,
                        message=(
                            f"Tilt angle ({design.mounting.tilt_angle}°) differs significantly "
                            f"from latitude ({design.site.latitude:.1f}°)"
                        ),
                        component="Mounting",
                        recommendation=f"Consider tilt angle closer to {optimal_tilt:.0f}° for annual optimization"
                    ))

        # Check system losses
        if design.total_system_losses > 20.0:
            issues.append(ValidationIssue(
                code="DESIGN-005",
                severity=ValidationSeverity.WARNING,
                message=f"Total system losses ({design.total_system_losses:.1f}%) are high",
                component="System",
                recommendation="Review loss assumptions and consider design improvements"
            ))

        self.validation_issues.extend(issues)
        return issues

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Returns:
            Dictionary with validation report data
        """
        report = {
            'total_issues': len(self.validation_issues),
            'errors': [i for i in self.validation_issues if i.severity == ValidationSeverity.ERROR],
            'warnings': [i for i in self.validation_issues if i.severity == ValidationSeverity.WARNING],
            'info': [i for i in self.validation_issues if i.severity == ValidationSeverity.INFO],
            'by_component': {},
            'nec_references': []
        }

        # Group by component
        for issue in self.validation_issues:
            if issue.component not in report['by_component']:
                report['by_component'][issue.component] = []
            report['by_component'][issue.component].append(issue)

        # Collect NEC references
        report['nec_references'] = list(set(
            issue.nec_reference
            for issue in self.validation_issues
            if issue.nec_reference
        ))

        return report
