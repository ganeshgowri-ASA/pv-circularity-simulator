"""Main IEC 61730 safety testing orchestrator.

This module coordinates all safety testing activities per IEC 61730-1 and IEC 61730-2,
including electrical, mechanical, fire, and environmental tests, along with
construction requirements validation and safety classification.
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
from loguru import logger

from ..models.safety_models import (
    SafetyTestConfig,
    SafetyTestResult,
    ElectricalSafetyTestResult,
    MechanicalSafetyTestResult,
    FireSafetyTestResult,
    EnvironmentalSafetyTestResult,
    ConstructionRequirement,
    SafetyClassification,
    SafetyCertificate,
    SafetyClass,
    FireClass,
    ApplicationClass,
    TestStatus,
    MechanicalLoadTestResult,
    ImpactTestResult,
    RobustnessOfTerminationsTestResult,
    UVPreconditioningTestResult,
    ThermalCyclingTestResult,
    HumidityFreezeTestResult,
)
from .electrical_safety import ElectricalSafetyTest
from .fire_safety import FireSafetyClassification


class IEC61730SafetyTester:
    """Main orchestrator for IEC 61730 safety testing and qualification.

    This class coordinates all aspects of PV module safety testing per IEC 61730-1
    (Construction Requirements) and IEC 61730-2 (Testing Requirements), including:
    - Electrical safety tests (MST 01-05)
    - Mechanical safety tests (MST 06-08)
    - Fire safety tests (Annex C)
    - Environmental safety tests (MST 09-11)
    - Construction requirements validation
    - Safety classification determination
    - Certificate generation

    Attributes:
        config: Safety test configuration.
        electrical_tester: Electrical safety test executor.
        fire_tester: Fire safety test executor.
        results: Aggregated test results.
    """

    def __init__(self, config: SafetyTestConfig) -> None:
        """Initialize IEC 61730 safety tester.

        Args:
            config: Safety test configuration with module details and test parameters.
        """
        self.config = config
        self.results: Optional[SafetyTestResult] = None

        # Initialize test executors
        self.electrical_tester = ElectricalSafetyTest(
            module_id=config.module_id,
            max_system_voltage_v=config.max_system_voltage_v,
            safety_class=config.target_safety_class,
        )

        self.fire_tester = FireSafetyClassification(
            module_id=config.module_id,
            module_area_m2=config.module_area_m2,
        )

        logger.info(
            f"Initialized IEC61730SafetyTester for module {config.module_id}, "
            f"target class {config.target_safety_class}"
        )

    def electrical_safety_tests(self) -> ElectricalSafetyTestResult:
        """Execute all electrical safety tests per IEC 61730-2 MST 01-05.

        Runs the complete suite of electrical safety tests:
        - MST 01: Insulation resistance test
        - MST 02: Wet leakage current test
        - MST 03: Dielectric strength test
        - MST 04: Ground continuity test (Class I only)
        - MST 05: Bypass diode thermal test

        Returns:
            ElectricalSafetyTestResult with all test results.
        """
        logger.info(f"Starting electrical safety tests for {self.config.module_id}")

        # Run all electrical tests
        results = self.electrical_tester.run_all_electrical_tests()

        logger.info(
            f"Electrical safety tests complete: "
            f"{'ALL PASSED' if results.all_tests_passed else 'SOME FAILED'}"
        )

        return results

    def mechanical_safety_tests(self) -> MechanicalSafetyTestResult:
        """Execute all mechanical safety tests per IEC 61730-2 MST 06-08.

        Runs the complete suite of mechanical safety tests:
        - MST 06: Mechanical load test
        - MST 07: Impact resistance test
        - MST 08: Robustness of terminations test

        Returns:
            MechanicalSafetyTestResult with all test results.
        """
        logger.info(f"Starting mechanical safety tests for {self.config.module_id}")

        # MST 06: Mechanical load test
        mechanical_load = self._mechanical_load_test()

        # MST 07: Impact resistance test
        impact = self._impact_resistance_test()

        # MST 08: Robustness of terminations
        terminations = self._robustness_of_terminations_test()

        results = MechanicalSafetyTestResult(
            mechanical_load=mechanical_load,
            impact=impact,
            robustness_of_terminations=terminations,
        )

        logger.info(
            f"Mechanical safety tests complete: "
            f"{'ALL PASSED' if results.all_tests_passed else 'SOME FAILED'}"
        )

        return results

    def fire_safety_tests(self) -> FireSafetyTestResult:
        """Execute all fire safety tests per IEC 61730-2 Annex C / UL 790.

        Runs the complete suite of fire safety tests:
        - Spread of flame test
        - Fire penetration test
        - Flying brand test
        - Fire classification determination

        Returns:
            FireSafetyTestResult with all test results and classification.
        """
        logger.info(f"Starting fire safety tests for {self.config.module_id}")

        target_fire_class = self.config.target_fire_class or FireClass.CLASS_A

        results = self.fire_tester.run_all_fire_tests(
            target_fire_class=target_fire_class
        )

        logger.info(
            f"Fire safety tests complete: Classification = {results.fire_classification}"
        )

        return results

    def environmental_safety_tests(self) -> EnvironmentalSafetyTestResult:
        """Execute all environmental safety tests per IEC 61730-2 MST 09-11.

        Runs the complete suite of environmental conditioning tests:
        - MST 09: UV preconditioning test
        - MST 10: Thermal cycling for safety
        - MST 11: Humidity-freeze test

        Returns:
            EnvironmentalSafetyTestResult with all test results.
        """
        logger.info(f"Starting environmental safety tests for {self.config.module_id}")

        # MST 09: UV preconditioning
        uv_preconditioning = self._uv_preconditioning_test()

        # MST 10: Thermal cycling
        thermal_cycling = self._thermal_cycling_test()

        # MST 11: Humidity-freeze
        humidity_freeze = self._humidity_freeze_test()

        results = EnvironmentalSafetyTestResult(
            uv_preconditioning=uv_preconditioning,
            thermal_cycling=thermal_cycling,
            humidity_freeze=humidity_freeze,
        )

        logger.info(
            f"Environmental safety tests complete: "
            f"{'ALL PASSED' if results.all_tests_passed else 'SOME FAILED'}"
        )

        return results

    def construction_requirements_check(self) -> List[ConstructionRequirement]:
        """Validate IEC 61730-1 construction requirements.

        Checks module construction against IEC 61730-1 requirements including:
        - Sharp edges and corners
        - Materials and components
        - Accessibility of live parts
        - Labeling and marking
        - Bonding and grounding
        - Strain relief

        Returns:
            List of ConstructionRequirement results for each checked requirement.
        """
        logger.info(
            f"Checking construction requirements for {self.config.module_id}"
        )

        requirements = []

        # 10.1: Sharp edges and corners
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.1",
            requirement_description="No sharp edges or corners that could cause injury",
            compliant=np.random.random() > 0.05,  # 95% pass rate
            notes="Visual inspection of module edges and corners"
        ))

        # 10.2: Materials
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.2",
            requirement_description="Materials suitable for intended use and environment",
            compliant=np.random.random() > 0.05,
            notes="Material specifications reviewed for UV, temperature, and moisture resistance"
        ))

        # 10.3: Accessibility of live parts
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.3",
            requirement_description="Live parts not accessible without tools",
            compliant=np.random.random() > 0.02,
            notes="Junction box and connector accessibility assessment"
        ))

        # 10.4: Protection against electric shock
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.4",
            requirement_description=f"Adequate protection for {self.config.target_safety_class}",
            compliant=np.random.random() > 0.05,
            notes=f"Insulation and grounding appropriate for {self.config.target_safety_class}"
        ))

        # 10.5: Labeling and marking
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.5",
            requirement_description="Proper labeling with electrical ratings and warnings",
            compliant=np.random.random() > 0.03,
            notes="Label content, durability, and placement verified"
        ))

        # 10.6: Bonding and grounding (Class I)
        if self.config.target_safety_class == SafetyClass.CLASS_I:
            requirements.append(ConstructionRequirement(
                requirement_id="IEC61730-1-10.6",
                requirement_description="Adequate bonding and grounding provisions",
                compliant=np.random.random() > 0.05,
                notes="Frame bonding points and grounding symbols verified"
            ))

        # 10.7: Strain relief
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.7",
            requirement_description="Adequate strain relief for cables and connections",
            compliant=np.random.random() > 0.05,
            notes="Cable entries and connector strain relief assessed"
        ))

        # 10.8: Bypass diodes
        requirements.append(ConstructionRequirement(
            requirement_id="IEC61730-1-10.8",
            requirement_description="Bypass diodes properly rated and protected",
            compliant=np.random.random() > 0.05,
            notes="Diode ratings and thermal management verified"
        ))

        compliant_count = sum(1 for req in requirements if req.compliant)
        logger.info(
            f"Construction requirements check complete: "
            f"{compliant_count}/{len(requirements)} compliant"
        )

        return requirements

    def generate_safety_classification(
        self,
        electrical_results: ElectricalSafetyTestResult,
        mechanical_results: MechanicalSafetyTestResult,
        fire_results: Optional[FireSafetyTestResult],
        construction_results: List[ConstructionRequirement],
    ) -> SafetyClassification:
        """Determine overall safety classification based on test results.

        Analyzes all test results and determines the appropriate safety class,
        application class, and fire class for the module.

        Args:
            electrical_results: Results from electrical safety tests.
            mechanical_results: Results from mechanical safety tests.
            fire_results: Results from fire safety tests (if performed).
            construction_results: Results from construction requirements check.

        Returns:
            SafetyClassification with determined classes and rationale.
        """
        logger.info(f"Generating safety classification for {self.config.module_id}")

        # Determine safety class based on grounding and insulation
        safety_class = self.config.target_safety_class

        # Verify safety class is appropriate based on test results
        if safety_class == SafetyClass.CLASS_I:
            # Class I requires working ground continuity
            if electrical_results.ground_continuity is None:
                safety_class = SafetyClass.CLASS_II
                logger.warning(
                    "No ground continuity test performed, downgrading to Class II"
                )
            elif not electrical_results.ground_continuity.passed:
                safety_class = SafetyClass.CLASS_II
                logger.warning(
                    "Ground continuity test failed, downgrading to Class II"
                )

        # Application class (based on voltage and accessibility)
        if self.config.max_system_voltage_v <= 35:
            application_class = ApplicationClass.CLASS_C  # Safe voltage
        elif self.config.application_class == ApplicationClass.CLASS_A:
            application_class = ApplicationClass.CLASS_A  # Hazardous, not accessible
        else:
            application_class = ApplicationClass.CLASS_B  # Hazardous, accessible

        # Fire class
        fire_class = FireClass.NOT_RATED
        if fire_results:
            fire_class = fire_results.fire_classification

        # Build rationale
        rationale_parts = []
        rationale_parts.append(
            f"Safety Class {safety_class.value}: "
            f"Based on {'protective earthing' if safety_class == SafetyClass.CLASS_I else 'double/reinforced insulation'}"
        )
        rationale_parts.append(
            f"Application Class {application_class.value}: "
            f"System voltage {self.config.max_system_voltage_v}V"
        )
        if fire_class != FireClass.NOT_RATED:
            rationale_parts.append(
                f"Fire Class {fire_class.value}: "
                f"Based on fire safety test results"
            )

        # Check for test failures
        if not electrical_results.all_tests_passed:
            rationale_parts.append("WARNING: Some electrical tests failed")
        if not mechanical_results.all_tests_passed:
            rationale_parts.append("WARNING: Some mechanical tests failed")
        if not all(req.compliant for req in construction_results):
            rationale_parts.append("WARNING: Some construction requirements not met")

        rationale = ". ".join(rationale_parts)

        classification = SafetyClassification(
            safety_class=safety_class,
            application_class=application_class,
            fire_class=fire_class,
            max_system_voltage_v=self.config.max_system_voltage_v,
            classification_rationale=rationale,
        )

        logger.info(
            f"Safety classification: {safety_class.value}, "
            f"Application: {application_class.value}, "
            f"Fire: {fire_class.value}"
        )

        return classification

    def export_safety_certificate(
        self,
        certification_body: str = "TUV Rheinland",
    ) -> SafetyCertificate:
        """Generate IEC 61730 safety certificate based on test results.

        Creates a formal safety certificate documenting all test results and
        certifying the module's safety classification.

        Args:
            certification_body: Name of certification body (default: "TUV Rheinland").

        Returns:
            SafetyCertificate with all test data and certification details.

        Raises:
            ValueError: If tests have not been completed or did not pass.
        """
        if self.results is None:
            raise ValueError(
                "Cannot generate certificate: tests have not been completed. "
                "Run run_all_tests() first."
            )

        if not self.results.overall_pass:
            raise ValueError(
                "Cannot generate certificate: module did not pass all required tests"
            )

        logger.info(f"Generating safety certificate for {self.config.module_id}")

        # Generate certificate number
        certificate_number = (
            f"IEC61730-{self.config.manufacturer[:3].upper()}-"
            f"{self.config.model_number.replace(' ', '')}-"
            f"{datetime.now().strftime('%Y%m%d')}"
        )

        # Certificate validity (typically 5 years)
        from datetime import timedelta
        expiry_date = datetime.now() + timedelta(days=5*365)

        certificate = SafetyCertificate(
            certificate_number=certificate_number,
            issue_date=datetime.now(),
            expiry_date=expiry_date,
            certification_body=certification_body,
            module_info=self.config,
            test_results=self.results,
            certified_safety_class=self.results.classification.safety_class,
            certified_application_class=self.results.classification.application_class,
            certified_fire_class=self.results.classification.fire_class,
            special_conditions=None,
        )

        logger.info(f"Certificate generated: {certificate_number}")

        return certificate

    def run_all_tests(self) -> SafetyTestResult:
        """Execute complete IEC 61730 safety test suite.

        Runs all required safety tests based on configuration:
        - Electrical safety tests (if enabled)
        - Mechanical safety tests (if enabled)
        - Fire safety tests (if enabled)
        - Environmental safety tests (if enabled)
        - Construction requirements validation
        - Safety classification determination

        Returns:
            SafetyTestResult with complete test results and classification.
        """
        logger.info(
            f"Starting complete IEC 61730 safety test suite for {self.config.module_id}"
        )

        # Execute tests based on configuration
        electrical_results = None
        if self.config.perform_electrical_tests:
            electrical_results = self.electrical_safety_tests()

        mechanical_results = None
        if self.config.perform_mechanical_tests:
            mechanical_results = self.mechanical_safety_tests()

        fire_results = None
        if self.config.perform_fire_tests:
            fire_results = self.fire_safety_tests()

        environmental_results = None
        if self.config.perform_environmental_tests:
            environmental_results = self.environmental_safety_tests()

        # Always check construction requirements
        construction_results = self.construction_requirements_check()

        # Generate classification
        classification = self.generate_safety_classification(
            electrical_results=electrical_results or ElectricalSafetyTestResult(),
            mechanical_results=mechanical_results or MechanicalSafetyTestResult(),
            fire_results=fire_results,
            construction_results=construction_results,
        )

        # Create overall results
        self.results = SafetyTestResult(
            config=self.config,
            electrical_tests=electrical_results,
            mechanical_tests=mechanical_results,
            fire_tests=fire_results,
            environmental_tests=environmental_results,
            construction_requirements=construction_results,
            classification=classification,
            overall_pass=False,  # Will be computed
            test_completion_date=datetime.now(),
        )

        # Determine overall pass/fail
        self.results.overall_pass = self.results.all_required_tests_passed

        logger.info(
            f"Complete test suite finished: "
            f"{'OVERALL PASS' if self.results.overall_pass else 'OVERALL FAIL'}"
        )

        return self.results

    # Private helper methods for mechanical and environmental tests

    def _mechanical_load_test(self) -> MechanicalLoadTestResult:
        """Execute mechanical load test per IEC 61730-2 MST 06."""
        logger.info("Executing mechanical load test")

        # IEC 61730-2 specifies 2400 Pa load (or module design load)
        applied_load_pa = 2400.0

        # Simulate deflection (depends on module rigidity and mounting)
        # Typical deflection: 5-20mm for front load, 5-15mm for rear load
        max_deflection_mm = np.random.uniform(5.0, 20.0)

        # Permanent deformation should be minimal (<2mm)
        permanent_deformation_mm = max(0.0, np.random.normal(0.5, 0.5))

        # Number of cycles (typically 3 cycles: front, rear, front)
        cycles = 3

        # Visual inspection after test
        defect_probability = 0.05  # 5% chance of visual defects
        visual_defects = np.random.random() < defect_probability

        status = TestStatus.PASSED if not visual_defects else TestStatus.FAILED

        return MechanicalLoadTestResult(
            applied_load_pa=applied_load_pa,
            maximum_deflection_mm=max_deflection_mm,
            permanent_deformation_mm=permanent_deformation_mm,
            cycles_completed=cycles,
            visual_defects_found=visual_defects,
            status=status,
        )

    def _impact_resistance_test(self) -> ImpactTestResult:
        """Execute impact resistance test per IEC 61730-2 MST 07."""
        logger.info("Executing impact resistance test")

        # Standard ice ball test: 25mm diameter
        ice_ball_diameter_mm = 25.0

        # Impact velocity depends on drop height (typically 0.5-1m)
        # v = sqrt(2*g*h)
        drop_height_m = 0.75
        impact_velocity_ms = np.sqrt(2 * 9.81 * drop_height_m)

        # Number of impact locations (typically 11 points)
        impact_locations = 11

        # Simulate impact results
        # Cracks may occur but shouldn't compromise electrical safety
        crack_probability = 0.15  # 15% chance of visible cracks
        cracks_detected = np.random.random() < crack_probability

        # Electrical safety is critical
        # Even with cracks, electrical safety must be maintained
        if cracks_detected:
            safety_maintained_probability = 0.9
            electrical_safety_maintained = (
                np.random.random() < safety_maintained_probability
            )
        else:
            electrical_safety_maintained = True

        status = (
            TestStatus.PASSED if electrical_safety_maintained
            else TestStatus.FAILED
        )

        return ImpactTestResult(
            ice_ball_diameter_mm=ice_ball_diameter_mm,
            impact_velocity_ms=impact_velocity_ms,
            impact_locations=impact_locations,
            cracks_detected=cracks_detected,
            electrical_safety_maintained=electrical_safety_maintained,
            status=status,
        )

    def _robustness_of_terminations_test(self) -> RobustnessOfTerminationsTestResult:
        """Execute robustness of terminations test per IEC 61730-2 MST 08."""
        logger.info("Executing robustness of terminations test")

        # Pull force test (typically 50N for cables <1mm², 100N for larger)
        pull_force_n = 100.0

        # Torque test (depends on connector type, typically 0.5-2 Nm)
        torque_nm = 1.0

        # Simulate test results
        # Well-designed terminals should not displace or damage
        displacement_probability = 0.03  # 3% failure rate
        cable_displaced = np.random.random() < displacement_probability

        damage_probability = 0.02  # 2% failure rate
        terminal_damaged = np.random.random() < damage_probability

        status = (
            TestStatus.PASSED if not (cable_displaced or terminal_damaged)
            else TestStatus.FAILED
        )

        return RobustnessOfTerminationsTestResult(
            pull_force_n=pull_force_n,
            torque_nm=torque_nm,
            cable_displaced=cable_displaced,
            terminal_damaged=terminal_damaged,
            status=status,
        )

    def _uv_preconditioning_test(self) -> UVPreconditioningTestResult:
        """Execute UV preconditioning test per IEC 61730-2 MST 09."""
        logger.info("Executing UV preconditioning test")

        # Required UV dose: 15 kWh/m²
        required_dose_kwh_m2 = 15.0

        # Simulate test execution
        # Test duration depends on UV source intensity
        # Typical xenon arc lamp: 0.55 kW/m² → ~27 hours
        test_duration_h = 28.0
        uv_dose_kwh_m2 = 15.5

        # Check for visual degradation
        # Good quality modules should show minimal degradation
        degradation_probability = 0.05
        visual_degradation = np.random.random() < degradation_probability

        status = (
            TestStatus.PASSED if uv_dose_kwh_m2 >= required_dose_kwh_m2
            else TestStatus.FAILED
        )

        return UVPreconditioningTestResult(
            uv_dose_kwh_m2=uv_dose_kwh_m2,
            required_dose_kwh_m2=required_dose_kwh_m2,
            test_duration_h=test_duration_h,
            visual_degradation=visual_degradation,
            status=status,
        )

    def _thermal_cycling_test(self) -> ThermalCyclingTestResult:
        """Execute thermal cycling test per IEC 61730-2 MST 10."""
        logger.info("Executing thermal cycling test")

        # Required: 200 cycles, -40°C to +85°C
        required_cycles = 200
        cycles_completed = 200
        min_temp_c = -40.0
        max_temp_c = 85.0

        # Check for electrical failure
        # Well-designed modules should pass without failure
        failure_probability = 0.05
        electrical_failure = np.random.random() < failure_probability

        status = (
            TestStatus.PASSED if cycles_completed >= required_cycles and not electrical_failure
            else TestStatus.FAILED
        )

        return ThermalCyclingTestResult(
            cycles_completed=cycles_completed,
            required_cycles=required_cycles,
            min_temperature_c=min_temp_c,
            max_temperature_c=max_temp_c,
            electrical_failure=electrical_failure,
            status=status,
        )

    def _humidity_freeze_test(self) -> HumidityFreezeTestResult:
        """Execute humidity-freeze test per IEC 61730-2 MST 11."""
        logger.info("Executing humidity-freeze test")

        # Required: 10 cycles, 85°C/85%RH to -40°C
        required_cycles = 10
        cycles_completed = 10
        humidity_phase_c = 85.0
        humidity_phase_rh = 85.0
        freeze_phase_c = -40.0

        # Check for electrical failure
        failure_probability = 0.05
        electrical_failure = np.random.random() < failure_probability

        status = (
            TestStatus.PASSED if cycles_completed >= required_cycles and not electrical_failure
            else TestStatus.FAILED
        )

        return HumidityFreezeTestResult(
            cycles_completed=cycles_completed,
            required_cycles=required_cycles,
            humidity_phase_c=humidity_phase_c,
            humidity_phase_rh=humidity_phase_rh,
            freeze_phase_c=freeze_phase_c,
            electrical_failure=electrical_failure,
            status=status,
        )
