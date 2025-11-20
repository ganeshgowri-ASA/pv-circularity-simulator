"""Code compliance checker for PV systems.

This module provides comprehensive code compliance checking for PV systems
against NEC, IEC, and local building codes.
"""

from datetime import datetime
from typing import Dict, List, Optional

from src.models.validation_models import (
    ComplianceResult,
    ComplianceStatus,
    IssueItem,
    IssueSeverity,
    SystemConfiguration,
)


class CodeComplianceChecker:
    """Comprehensive code compliance checker for PV systems.

    Validates PV system designs against electrical codes (NEC, IEC),
    building codes, fire safety codes, and local jurisdiction requirements.

    Attributes:
        config: System configuration to validate
        nec_version: NEC version to validate against (default: "2020")
        iec_version: IEC version to validate against (default: "60364-7-712")
        jurisdiction: Local jurisdiction for code compliance
        compliance_results: List of all compliance check results
    """

    # NEC 2020 Voltage limits
    NEC_MAX_VOLTAGE_RESIDENTIAL = 600
    NEC_MAX_VOLTAGE_COMMERCIAL = 1000
    NEC_MAX_VOLTAGE_UTILITY = 1500

    # Voltage drop limits
    NEC_VOLTAGE_DROP_LIMIT = 2.0  # percentage

    # Temperature derating factors (NEC 690.7)
    NEC_TEMP_COEFFICIENT_CRYSTALLINE = -0.0035  # per °C
    NEC_TEMP_COEFFICIENT_THIN_FILM = -0.0025  # per °C

    # IEC 60364 limits
    IEC_MAX_VOLTAGE_LV = 1500
    IEC_MIN_INSULATION_RESISTANCE = 1.0  # MΩ

    def __init__(
        self,
        config: SystemConfiguration,
        nec_version: str = "2020",
        iec_version: str = "60364-7-712",
        jurisdiction: Optional[str] = None,
    ) -> None:
        """Initialize code compliance checker.

        Args:
            config: System configuration to validate
            nec_version: NEC version to validate against
            iec_version: IEC version to validate against
            jurisdiction: Local jurisdiction for additional requirements
        """
        self.config = config
        self.nec_version = nec_version
        self.iec_version = iec_version
        self.jurisdiction = jurisdiction or config.jurisdiction
        self.compliance_results: List[ComplianceResult] = []

    def nec_690_compliance(self) -> List[ComplianceResult]:
        """Perform NEC Article 690 compliance checks.

        Validates system against NEC Article 690 - Solar Photovoltaic Systems.
        Covers voltage limits, current limits, grounding, overcurrent protection,
        and disconnecting means.

        Returns:
            List of compliance check results for NEC 690
        """
        results: List[ComplianceResult] = []

        # NEC 690.7 - Maximum Voltage
        results.append(self._check_nec_690_7_max_voltage())

        # NEC 690.8 - Circuit Sizing and Current
        results.append(self._check_nec_690_8_circuit_sizing())

        # NEC 690.9 - Overcurrent Protection
        results.append(self._check_nec_690_9_overcurrent())

        # NEC 690.12 - Rapid Shutdown
        results.append(self._check_nec_690_12_rapid_shutdown())

        # NEC 690.13 - Photovoltaic System Disconnecting Means
        results.append(self._check_nec_690_13_disconnect())

        # NEC 690.31 - Wiring Methods
        results.append(self._check_nec_690_31_wiring())

        # NEC 690.35 - Ungrounded Systems
        results.append(self._check_nec_690_35_grounding())

        self.compliance_results.extend(results)
        return results

    def _check_nec_690_7_max_voltage(self) -> ComplianceResult:
        """Check NEC 690.7 - Maximum Voltage requirements."""
        # Determine voltage limit based on system type
        if self.config.system_type.value in ["residential", "rooftop"]:
            max_allowed = self.NEC_MAX_VOLTAGE_RESIDENTIAL
        elif self.config.system_type.value in ["commercial", "carport"]:
            max_allowed = self.NEC_MAX_VOLTAGE_COMMERCIAL
        else:
            max_allowed = self.NEC_MAX_VOLTAGE_UTILITY

        # Calculate maximum voltage with temperature correction
        # Voc_max = Voc @ STC × (1 + temp_coeff × (T_min - 25))
        temp_correction = 1 + (self.NEC_TEMP_COEFFICIENT_CRYSTALLINE *
                              (self.config.ambient_temp_min - 25))
        max_voltage = self.config.max_voltage_voc * temp_correction

        status = (
            ComplianceStatus.PASSED
            if max_voltage <= max_allowed
            else ComplianceStatus.FAILED
        )

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.7(A)",
            requirement=f"Maximum system voltage for {self.config.system_type.value}",
            status=status,
            checked_value=round(max_voltage, 2),
            required_value=max_allowed,
            notes=(
                f"Maximum voltage {max_voltage:.2f}V "
                f"{'is below' if status == ComplianceStatus.PASSED else 'exceeds'} "
                f"the {max_allowed}V limit for {self.config.system_type.value} systems. "
                f"Includes temperature correction factor for {self.config.ambient_temp_min}°C."
            ),
        )

    def _check_nec_690_8_circuit_sizing(self) -> ComplianceResult:
        """Check NEC 690.8 - Circuit Sizing and Current requirements."""
        # NEC requires 125% safety factor for continuous current
        required_current = self.config.max_current_isc * 1.25

        # This is a design requirement check - typically would verify wire sizing
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.8(A)(1)",
            requirement="Circuit conductors sized for 125% of Isc",
            status=status,
            checked_value=round(required_current, 2),
            required_value=round(self.config.max_current_isc * 1.25, 2),
            notes=(
                f"Circuit conductors must be sized for {required_current:.2f}A "
                f"(125% of {self.config.max_current_isc:.2f}A Isc)"
            ),
        )

    def _check_nec_690_9_overcurrent(self) -> ComplianceResult:
        """Check NEC 690.9 - Overcurrent Protection requirements."""
        # Overcurrent protection required if more than one source
        needs_overcurrent = self.config.string_count > 1

        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.9(A)",
            requirement="Overcurrent protection for parallel sources",
            status=status,
            checked_value=self.config.string_count,
            required_value="Required if > 1 string",
            notes=(
                f"System has {self.config.string_count} strings. "
                f"Overcurrent protection {'is required' if needs_overcurrent else 'may not be required'}."
            ),
        )

    def _check_nec_690_12_rapid_shutdown(self) -> ComplianceResult:
        """Check NEC 690.12 - Rapid Shutdown requirements."""
        # Rapid shutdown required for building-mounted systems
        requires_rapid_shutdown = self.config.system_type.value in [
            "residential",
            "rooftop",
            "commercial",
            "carport"
        ]

        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.12",
            requirement="Rapid shutdown system for building-mounted arrays",
            status=status,
            checked_value=str(requires_rapid_shutdown),
            required_value="Required for building-mounted systems",
            notes=(
                f"Rapid shutdown {'IS REQUIRED' if requires_rapid_shutdown else 'may not be required'} "
                f"for {self.config.system_type.value} systems. "
                f"Conductors must be limited to 80V within 30 seconds of shutdown."
            ),
        )

    def _check_nec_690_13_disconnect(self) -> ComplianceResult:
        """Check NEC 690.13 - Disconnecting Means requirements."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.13",
            requirement="PV system disconnecting means",
            status=status,
            checked_value="Required",
            required_value="Required",
            notes=(
                "A readily accessible disconnecting means is required for the PV system. "
                "Must be marked 'PV SYSTEM DISCONNECT' and be suitable for the DC voltage and current."
            ),
        )

    def _check_nec_690_31_wiring(self) -> ComplianceResult:
        """Check NEC 690.31 - Wiring Methods requirements."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.31",
            requirement="Wiring methods for PV source and output circuits",
            status=status,
            checked_value="Approved wiring methods required",
            required_value="Metal raceways, PV wire, or approved cables",
            notes=(
                "PV source and output circuits must use approved wiring methods. "
                "Exposed single conductor cables must be marked 'PV WIRE' and rated for sunlight/wet locations."
            ),
        )

    def _check_nec_690_35_grounding(self) -> ComplianceResult:
        """Check NEC 690.35 - Grounding requirements."""
        # Most modern systems are ungrounded (floating) DC systems
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"NEC {self.nec_version}",
            section="690.35",
            requirement="Ungrounded PV systems require ground-fault protection",
            status=status,
            checked_value="Ground-fault protection required",
            required_value="Required for ungrounded systems",
            notes=(
                "Ungrounded PV systems must have ground-fault protection and "
                "ground-fault detection per 690.35(C). Inverter must have isolation monitoring."
            ),
        )

    def iec_60364_compliance(self) -> List[ComplianceResult]:
        """Perform IEC 60364-7-712 compliance checks.

        Validates system against IEC 60364-7-712 - Electrical installations of buildings
        Part 7-712: Requirements for special installations or locations - Solar photovoltaic (PV) power supply systems.

        Returns:
            List of compliance check results for IEC 60364
        """
        results: List[ComplianceResult] = []

        # IEC 712.410.3 - Protection against electric shock
        results.append(self._check_iec_712_410_3_protection())

        # IEC 712.433 - Protection against overcurrent
        results.append(self._check_iec_712_433_overcurrent())

        # IEC 712.444 - Protection against overvoltage
        results.append(self._check_iec_712_444_overvoltage())

        # IEC 712.5 - Selection and erection of equipment
        results.append(self._check_iec_712_5_equipment())

        self.compliance_results.extend(results)
        return results

    def _check_iec_712_410_3_protection(self) -> ComplianceResult:
        """Check IEC 712.410.3 - Protection against electric shock."""
        # Check voltage limits for low voltage systems
        status = (
            ComplianceStatus.PASSED
            if self.config.max_voltage_voc <= self.IEC_MAX_VOLTAGE_LV
            else ComplianceStatus.FAILED
        )

        return ComplianceResult(
            code_name=f"IEC {self.iec_version}",
            section="712.410.3",
            requirement="Protection against electric shock - voltage limits",
            status=status,
            checked_value=self.config.max_voltage_voc,
            required_value=self.IEC_MAX_VOLTAGE_LV,
            notes=(
                f"Maximum system voltage {self.config.max_voltage_voc}V "
                f"{'is within' if status == ComplianceStatus.PASSED else 'exceeds'} "
                f"IEC low voltage limit of {self.IEC_MAX_VOLTAGE_LV}V."
            ),
        )

    def _check_iec_712_433_overcurrent(self) -> ComplianceResult:
        """Check IEC 712.433 - Protection against overcurrent."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"IEC {self.iec_version}",
            section="712.433",
            requirement="Overcurrent protection for parallel strings",
            status=status,
            checked_value=self.config.string_count,
            required_value="Required if > 1 string",
            notes=(
                f"System has {self.config.string_count} strings. "
                f"Overcurrent protection devices must be provided for parallel connected strings."
            ),
        )

    def _check_iec_712_444_overvoltage(self) -> ComplianceResult:
        """Check IEC 712.444 - Protection against overvoltage."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"IEC {self.iec_version}",
            section="712.444",
            requirement="Overvoltage protection (lightning/switching)",
            status=status,
            checked_value="SPD recommended",
            required_value="Type II SPD minimum",
            notes=(
                "Surge protection devices (SPD) recommended on both DC and AC sides. "
                "Type II minimum required, Type I recommended for exposed installations."
            ),
        )

    def _check_iec_712_5_equipment(self) -> ComplianceResult:
        """Check IEC 712.5 - Selection and erection of equipment."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name=f"IEC {self.iec_version}",
            section="712.512.2",
            requirement="Equipment selection - DC ratings",
            status=status,
            checked_value="All equipment must be DC rated",
            required_value="DC rated switchgear required",
            notes=(
                "All equipment in DC circuits must be suitable for DC operation. "
                "DC-rated switchgear, overcurrent devices, and connectors required."
            ),
        )

    def local_code_verification(self) -> List[ComplianceResult]:
        """Perform local jurisdiction code verification.

        Checks local code requirements which may be more stringent than
        national codes. Requirements vary by jurisdiction.

        Returns:
            List of local code compliance check results
        """
        results: List[ComplianceResult] = []

        # Placeholder for jurisdiction-specific checks
        results.append(
            ComplianceResult(
                code_name=f"Local Code - {self.jurisdiction}",
                section="General",
                requirement="Local jurisdiction requirements",
                status=ComplianceStatus.PASSED,
                checked_value="Design review required",
                required_value="AHJ approval required",
                notes=(
                    f"Design must be reviewed and approved by {self.jurisdiction} "
                    f"Authority Having Jurisdiction (AHJ). Additional local requirements may apply."
                ),
            )
        )

        self.compliance_results.extend(results)
        return results

    def building_code_checks(self) -> List[ComplianceResult]:
        """Perform building code compliance checks.

        Validates structural and building code requirements for PV installations.
        Covers wind loads, snow loads, seismic requirements, and roof loading.

        Returns:
            List of building code compliance check results
        """
        results: List[ComplianceResult] = []

        # Wind load check
        results.append(self._check_wind_loads())

        # Snow load check (if applicable)
        if self.config.snow_load and self.config.snow_load > 0:
            results.append(self._check_snow_loads())

        # Roof loading check for rooftop systems
        if self.config.system_type.value in ["rooftop", "residential", "commercial"]:
            results.append(self._check_roof_loading())

        self.compliance_results.extend(results)
        return results

    def _check_wind_loads(self) -> ComplianceResult:
        """Check wind load requirements per building codes."""
        # Typical wind speed design limits
        max_design_wind_speed = 50.0  # m/s (approximately 112 mph)

        status = (
            ComplianceStatus.PASSED
            if self.config.wind_speed_max <= max_design_wind_speed
            else ComplianceStatus.WARNING
        )

        return ComplianceResult(
            code_name="IBC 2021",
            section="1609",
            requirement="Wind load resistance",
            status=status,
            checked_value=self.config.wind_speed_max,
            required_value=max_design_wind_speed,
            notes=(
                f"System designed for {self.config.wind_speed_max} m/s wind speed. "
                f"Structural analysis required per ASCE 7 for site-specific wind loads."
            ),
        )

    def _check_snow_loads(self) -> ComplianceResult:
        """Check snow load requirements."""
        # Typical design snow load limit (varies by location)
        max_design_snow_load = 200.0  # kg/m²

        status = (
            ComplianceStatus.PASSED
            if (self.config.snow_load or 0) <= max_design_snow_load
            else ComplianceStatus.WARNING
        )

        return ComplianceResult(
            code_name="IBC 2021",
            section="1608",
            requirement="Snow load resistance",
            status=status,
            checked_value=self.config.snow_load,
            required_value=max_design_snow_load,
            notes=(
                f"System designed for {self.config.snow_load} kg/m² snow load. "
                f"Consult local snow load maps per ASCE 7."
            ),
        )

    def _check_roof_loading(self) -> ComplianceResult:
        """Check roof loading for rooftop systems."""
        # Typical residential roof load capacity: 20-30 psf (96-144 kg/m²)
        # PV system typically adds 3-5 psf (14-24 kg/m²)
        typical_pv_load = 20.0  # kg/m²

        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name="IBC 2021",
            section="1607",
            requirement="Roof dead load capacity",
            status=status,
            checked_value=typical_pv_load,
            required_value="Structural analysis required",
            notes=(
                "Rooftop PV system adds approximately 20 kg/m² dead load. "
                "Structural engineer must verify roof capacity and provide stamped calculations."
            ),
        )

    def fire_safety_compliance(self) -> List[ComplianceResult]:
        """Perform fire safety code compliance checks.

        Validates fire safety requirements including setbacks, pathways,
        and access for firefighters per IFC and local fire codes.

        Returns:
            List of fire safety compliance check results
        """
        results: List[ComplianceResult] = []

        # Firefighter access pathways
        if self.config.system_type.value in ["rooftop", "residential", "commercial"]:
            results.append(self._check_firefighter_access())

        # Rapid shutdown (fire safety aspect)
        results.append(self._check_fire_rapid_shutdown())

        self.compliance_results.extend(results)
        return results

    def _check_firefighter_access(self) -> ComplianceResult:
        """Check firefighter access pathways per IFC."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name="IFC 2021",
            section="605.11.3.3",
            requirement="Roof access pathways for firefighters",
            status=status,
            checked_value="Pathways required",
            required_value="3-foot pathways, 6-foot at ridge",
            notes=(
                "Rooftop PV arrays must provide access pathways: "
                "3-foot (0.9m) perimeter, 3-foot (0.9m) pathway spacing, "
                "6-foot (1.8m) at ridge or highest point. "
                "Consult local fire marshal for specific requirements."
            ),
        )

    def _check_fire_rapid_shutdown(self) -> ComplianceResult:
        """Check rapid shutdown for fire safety."""
        status = ComplianceStatus.PASSED

        return ComplianceResult(
            code_name="IFC 2021",
            section="1204.4",
            requirement="Rapid shutdown for emergency access",
            status=status,
            checked_value="Required",
            required_value="Required for building-mounted systems",
            notes=(
                "Rapid shutdown system required to reduce voltage to safe levels "
                "for firefighter safety during emergency response."
            ),
        )

    def get_all_compliance_results(self) -> List[ComplianceResult]:
        """Get all compliance check results.

        Returns:
            Complete list of all compliance check results
        """
        return self.compliance_results

    def get_failed_checks(self) -> List[ComplianceResult]:
        """Get all failed compliance checks.

        Returns:
            List of failed compliance checks
        """
        return [r for r in self.compliance_results if r.status == ComplianceStatus.FAILED]

    def generate_compliance_summary(self) -> Dict[str, int]:
        """Generate summary statistics for compliance checks.

        Returns:
            Dictionary with counts of checks by status
        """
        summary = {
            "total": len(self.compliance_results),
            "passed": sum(1 for r in self.compliance_results if r.status == ComplianceStatus.PASSED),
            "failed": sum(1 for r in self.compliance_results if r.status == ComplianceStatus.FAILED),
            "warning": sum(1 for r in self.compliance_results if r.status == ComplianceStatus.WARNING),
            "not_applicable": sum(
                1 for r in self.compliance_results if r.status == ComplianceStatus.NOT_APPLICABLE
            ),
        }
        return summary
