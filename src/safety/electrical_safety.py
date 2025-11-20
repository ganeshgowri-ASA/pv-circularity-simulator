"""Electrical safety testing per IEC 61730-2 Module Safety Test (MST) procedures.

This module implements all electrical safety tests required by IEC 61730-2,
including insulation resistance, wet leakage current, dielectric strength,
ground continuity, and bypass diode thermal testing.
"""

import logging
from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..models.safety_models import (
    InsulationResistanceTestResult,
    WetLeakageCurrentTestResult,
    DielectricStrengthTestResult,
    GroundContinuityTestResult,
    BypassDiodeThermalTestResult,
    ElectricalSafetyTestResult,
    TestStatus,
    SafetyClass,
)


class ElectricalSafetyTest:
    """Implements electrical safety tests per IEC 61730-2.

    This class provides methods for executing all electrical safety tests
    required for PV module safety qualification according to IEC 61730-2
    standard (MST 01 through MST 05).

    Attributes:
        module_id: Unique identifier for the module under test.
        max_system_voltage_v: Maximum DC system voltage in volts.
        safety_class: Target safety classification.
        test_temperature_c: Ambient test temperature in degrees Celsius.
        test_humidity_percent: Relative humidity during testing.
    """

    def __init__(
        self,
        module_id: str,
        max_system_voltage_v: float,
        safety_class: SafetyClass,
        test_temperature_c: float = 25.0,
        test_humidity_percent: float = 50.0,
    ) -> None:
        """Initialize electrical safety tester.

        Args:
            module_id: Unique module identifier.
            max_system_voltage_v: Maximum DC system voltage.
            safety_class: Target safety class (I, II, or III).
            test_temperature_c: Test temperature (default: 25°C).
            test_humidity_percent: Relative humidity (default: 50%).
        """
        self.module_id = module_id
        self.max_system_voltage_v = max_system_voltage_v
        self.safety_class = safety_class
        self.test_temperature_c = test_temperature_c
        self.test_humidity_percent = test_humidity_percent

        logger.info(
            f"Initialized ElectricalSafetyTest for module {module_id}, "
            f"Vmax={max_system_voltage_v}V, class={safety_class}"
        )

    def insulation_resistance_test(
        self,
        test_voltage_v: Optional[float] = None,
        test_duration_s: float = 60.0,
    ) -> InsulationResistanceTestResult:
        """Perform insulation resistance test per IEC 61730-2 MST 01.

        Measures the insulation resistance between the circuit and the module
        frame/mounting structure. The test voltage is typically 500V DC or 1000V DC
        depending on the maximum system voltage.

        Per IEC 61730-2:
        - Test voltage: 500V DC for Vmax ≤ 50V, 1000V DC for Vmax > 50V
        - Minimum resistance: 40 MΩ
        - Test duration: 60 seconds minimum

        Args:
            test_voltage_v: Applied test voltage. If None, automatically determined
                based on max_system_voltage_v (500V or 1000V).
            test_duration_s: Duration of voltage application in seconds (default: 60).

        Returns:
            InsulationResistanceTestResult containing measured resistance and pass/fail.
        """
        logger.info(f"Starting insulation resistance test for module {self.module_id}")

        # Determine test voltage per IEC 61730-2
        if test_voltage_v is None:
            test_voltage_v = 1000.0 if self.max_system_voltage_v > 50 else 500.0

        # Simulate insulation resistance measurement
        # In real testing, this would interface with a megohmmeter
        # Model: R_insulation depends on temperature, humidity, and material quality
        # Lower humidity and temperature generally increase resistance

        # Temperature coefficient (resistance decreases with temperature)
        temp_factor = np.exp(-0.02 * (self.test_temperature_c - 25.0))

        # Humidity coefficient (resistance decreases with humidity)
        humidity_factor = np.exp(-0.015 * (self.test_humidity_percent - 50.0))

        # Base resistance with some variation (simulating real modules)
        base_resistance_mohm = np.random.normal(400.0, 50.0)

        # Apply environmental factors
        measured_resistance_mohm = (
            base_resistance_mohm * temp_factor * humidity_factor
        )

        # Ensure non-negative
        measured_resistance_mohm = max(0.0, measured_resistance_mohm)

        # Determine pass/fail
        passed = measured_resistance_mohm >= 40.0
        status = TestStatus.PASSED if passed else TestStatus.FAILED

        result = InsulationResistanceTestResult(
            test_voltage_v=test_voltage_v,
            measured_resistance_mohm=measured_resistance_mohm,
            minimum_required_mohm=40.0,
            test_duration_s=test_duration_s,
            temperature_c=self.test_temperature_c,
            humidity_percent=self.test_humidity_percent,
            status=status,
        )

        logger.info(
            f"Insulation resistance test complete: "
            f"{measured_resistance_mohm:.1f} MΩ @ {test_voltage_v}V - "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return result

    def wet_leakage_current_test(
        self,
        water_resistivity_ohm_cm: float = 5000.0,
        spray_duration_min: float = 10.0,
    ) -> WetLeakageCurrentTestResult:
        """Perform wet leakage current test per IEC 61730-2 MST 02.

        Measures leakage current during water spray exposure to simulate
        rain conditions. This test ensures electrical safety during wet conditions.

        Per IEC 61730-2:
        - Water resistivity: 5000 ± 1000 Ω·cm at 20°C
        - Spray duration: 10 minutes minimum
        - Maximum leakage current: 275 μA
        - Test voltage: 1.25 × Vmax,dc

        Args:
            water_resistivity_ohm_cm: Water resistivity in Ω·cm (default: 5000).
            spray_duration_min: Duration of water spray in minutes (default: 10).

        Returns:
            WetLeakageCurrentTestResult containing measured current and pass/fail.
        """
        logger.info(f"Starting wet leakage current test for module {self.module_id}")

        # Test voltage per IEC 61730-2
        test_voltage_v = 1.25 * self.max_system_voltage_v

        # Simulate leakage current measurement
        # Leakage current depends on:
        # 1. Surface contamination
        # 2. Water conductivity (inverse of resistivity)
        # 3. Module design (edge sealing, frame design)
        # 4. Applied voltage

        # Water conductivity factor (higher conductivity = higher leakage)
        conductivity_factor = 5000.0 / water_resistivity_ohm_cm

        # Voltage factor (higher voltage = higher leakage)
        voltage_factor = test_voltage_v / 1000.0

        # Base leakage current (simulating well-designed module)
        base_leakage_ua = np.random.normal(100.0, 30.0)

        # Calculate total leakage current
        leakage_current_ua = (
            base_leakage_ua * conductivity_factor * voltage_factor
        )

        # Ensure non-negative
        leakage_current_ua = max(0.0, leakage_current_ua)

        # Determine pass/fail
        passed = leakage_current_ua <= 275.0
        status = TestStatus.PASSED if passed else TestStatus.FAILED

        result = WetLeakageCurrentTestResult(
            leakage_current_ua=leakage_current_ua,
            maximum_allowed_ua=275.0,
            test_voltage_v=test_voltage_v,
            water_spray_duration_min=spray_duration_min,
            water_resistivity_ohm_cm=water_resistivity_ohm_cm,
            status=status,
        )

        logger.info(
            f"Wet leakage current test complete: "
            f"{leakage_current_ua:.1f} μA @ {test_voltage_v}V - "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return result

    def dielectric_strength_test(
        self,
        test_duration_s: float = 60.0,
    ) -> DielectricStrengthTestResult:
        """Perform dielectric strength (high-pot) test per IEC 61730-2 MST 03.

        Applies high voltage stress to verify insulation integrity and ensure
        no breakdown occurs under overvoltage conditions.

        Per IEC 61730-2:
        - Test voltage: 1.5 × Vmax,dc + 1000V
        - Test duration: 60 seconds
        - Pass criteria: No breakdown or flashover

        Args:
            test_duration_s: Duration of voltage application in seconds (default: 60).

        Returns:
            DielectricStrengthTestResult containing breakdown status and pass/fail.
        """
        logger.info(f"Starting dielectric strength test for module {self.module_id}")

        # Calculate test voltage per IEC 61730-2
        test_voltage_v = 1.5 * self.max_system_voltage_v + 1000.0

        # Simulate dielectric strength test
        # In real testing, this would apply high voltage and monitor for breakdown
        # Breakdown probability depends on:
        # 1. Insulation quality and thickness
        # 2. Manufacturing defects
        # 3. Material aging
        # 4. Applied voltage stress

        # Model breakdown probability using Weibull distribution
        # Higher quality modules have lower breakdown probability
        # Shape parameter beta (higher = more consistent quality)
        beta = 10.0

        # Scale parameter (voltage at which 63% would fail)
        # Set high to represent good quality insulation
        eta = test_voltage_v * 3.0  # Module designed for 3x safety margin

        # Calculate breakdown probability
        breakdown_probability = 1 - np.exp(-((test_voltage_v / eta) ** beta))

        # Simulate breakdown occurrence
        breakdown_occurred = np.random.random() < breakdown_probability

        # If breakdown occurs, estimate breakdown voltage
        breakdown_voltage_v = None
        if breakdown_occurred:
            # Breakdown voltage is somewhere between 0 and test voltage
            # Usually occurs near test voltage for marginal cases
            breakdown_voltage_v = test_voltage_v * np.random.uniform(0.8, 0.99)

        # Determine pass/fail
        passed = not breakdown_occurred
        status = TestStatus.PASSED if passed else TestStatus.FAILED

        result = DielectricStrengthTestResult(
            test_voltage_v=test_voltage_v,
            vmax_dc_v=self.max_system_voltage_v,
            test_duration_s=test_duration_s,
            breakdown_occurred=breakdown_occurred,
            breakdown_voltage_v=breakdown_voltage_v,
            status=status,
        )

        logger.info(
            f"Dielectric strength test complete: "
            f"{test_voltage_v:.0f}V for {test_duration_s}s - "
            f"{'PASS (no breakdown)' if passed else f'FAIL (breakdown at {breakdown_voltage_v:.0f}V)'}"
        )

        return result

    def ground_continuity_test(
        self,
        test_current_a: float = 10.0,
    ) -> GroundContinuityTestResult:
        """Perform ground continuity test per IEC 61730-2 MST 04.

        Verifies that protective grounding connections have sufficiently low
        resistance to safely conduct fault currents. Only applicable to Class I
        modules with protective earthing.

        Per IEC 61730-2:
        - Test current: 10A minimum
        - Maximum resistance: 0.1 Ω
        - Applies to Class I modules only

        Args:
            test_current_a: Test current in amperes (default: 10A).

        Returns:
            GroundContinuityTestResult containing measured resistance and pass/fail.

        Raises:
            ValueError: If test is attempted on non-Class I module.
        """
        if self.safety_class != SafetyClass.CLASS_I:
            logger.warning(
                f"Ground continuity test not applicable to {self.safety_class} module"
            )
            raise ValueError(
                f"Ground continuity test only applies to Class I modules, "
                f"module is {self.safety_class}"
            )

        logger.info(f"Starting ground continuity test for module {self.module_id}")

        # Simulate ground resistance measurement
        # In real testing, this would use a low-resistance ohmmeter
        # Resistance depends on:
        # 1. Connection quality (bolts, crimps, welds)
        # 2. Contact surface area
        # 3. Material conductivity
        # 4. Cable length and gauge

        # Model typical ground connection
        # Good connections: 0.01-0.05 Ω
        # Marginal connections: 0.05-0.10 Ω
        # Poor connections: >0.10 Ω

        # Base resistance with normal distribution
        measured_resistance_ohm = np.random.normal(0.04, 0.02)

        # Ensure non-negative and realistic range
        measured_resistance_ohm = max(0.001, measured_resistance_ohm)

        # Determine pass/fail
        passed = measured_resistance_ohm <= 0.1
        status = TestStatus.PASSED if passed else TestStatus.FAILED

        result = GroundContinuityTestResult(
            measured_resistance_ohm=measured_resistance_ohm,
            maximum_allowed_ohm=0.1,
            test_current_a=test_current_a,
            status=status,
        )

        logger.info(
            f"Ground continuity test complete: "
            f"{measured_resistance_ohm:.4f} Ω @ {test_current_a}A - "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return result

    def bypass_diode_thermal_test(
        self,
        fault_current_a: float,
        test_duration_h: float = 2.0,
        max_diode_temp_c: float = 150.0,
    ) -> BypassDiodeThermalTestResult:
        """Perform bypass diode thermal test per IEC 61730-2 MST 05.

        Verifies that bypass diodes can handle reverse fault currents without
        thermal runaway or exceeding temperature limits. Critical for preventing
        hot-spot failures and fire hazards.

        Per IEC 61730-2:
        - Fault current: As specified in module design
        - Test duration: 2 hours minimum
        - Pass criteria: No thermal runaway, temp within diode ratings

        Args:
            fault_current_a: Fault current to apply in amperes.
            test_duration_h: Test duration in hours (default: 2.0).
            max_diode_temp_c: Maximum allowed diode temperature (default: 150°C).

        Returns:
            BypassDiodeThermalTestResult containing peak temperature and pass/fail.
        """
        logger.info(
            f"Starting bypass diode thermal test for module {self.module_id}: "
            f"{fault_current_a}A for {test_duration_h}h"
        )

        # Simulate bypass diode thermal behavior under fault current
        # Temperature rise depends on:
        # 1. Diode power dissipation (I × Vf)
        # 2. Thermal resistance (junction to ambient)
        # 3. Heat sinking and encapsulation
        # 4. Ambient temperature

        # Typical bypass diode parameters
        forward_voltage_v = 0.7  # Silicon diode forward voltage
        thermal_resistance_c_w = 30.0  # Junction to ambient thermal resistance

        # Power dissipation in diode
        power_dissipation_w = fault_current_a * forward_voltage_v

        # Temperature rise above ambient
        delta_t_c = power_dissipation_w * thermal_resistance_c_w

        # Add some variation for thermal transients and hotspots
        thermal_variation_c = np.random.normal(0.0, 5.0)

        # Peak temperature
        peak_temperature_c = (
            self.test_temperature_c + delta_t_c + thermal_variation_c
        )

        # Check for thermal runaway
        # Thermal runaway occurs if junction temperature exceeds ~175°C for Si diodes
        # and continues to rise uncontrollably
        thermal_runaway_detected = peak_temperature_c > 175.0

        # Determine pass/fail
        passed = (
            not thermal_runaway_detected and
            peak_temperature_c <= max_diode_temp_c
        )
        status = TestStatus.PASSED if passed else TestStatus.FAILED

        result = BypassDiodeThermalTestResult(
            peak_temperature_c=peak_temperature_c,
            maximum_allowed_c=max_diode_temp_c,
            fault_current_a=fault_current_a,
            test_duration_h=test_duration_h,
            thermal_runaway_detected=thermal_runaway_detected,
            status=status,
        )

        logger.info(
            f"Bypass diode thermal test complete: "
            f"Peak temp {peak_temperature_c:.1f}°C - "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return result

    def run_all_electrical_tests(
        self,
        include_ground_continuity: bool = None,
    ) -> ElectricalSafetyTestResult:
        """Execute all applicable electrical safety tests.

        Runs the complete suite of electrical safety tests required by
        IEC 61730-2. Ground continuity test is only performed for Class I modules.

        Args:
            include_ground_continuity: Whether to include ground continuity test.
                If None, automatically determined based on safety class.

        Returns:
            ElectricalSafetyTestResult containing all test results.
        """
        logger.info(
            f"Running complete electrical safety test suite for module {self.module_id}"
        )

        # Determine if ground continuity test is needed
        if include_ground_continuity is None:
            include_ground_continuity = (self.safety_class == SafetyClass.CLASS_I)

        # Run all tests
        insulation = self.insulation_resistance_test()
        wet_leakage = self.wet_leakage_current_test()
        dielectric = self.dielectric_strength_test()

        # Ground continuity only for Class I
        ground = None
        if include_ground_continuity:
            try:
                ground = self.ground_continuity_test()
            except ValueError as e:
                logger.warning(f"Skipping ground continuity test: {e}")

        # Bypass diode thermal test (using typical fault current)
        # Fault current typically 1.25 × Isc
        typical_isc_a = 10.0  # Typical short-circuit current
        fault_current_a = 1.25 * typical_isc_a
        bypass_diode = self.bypass_diode_thermal_test(fault_current_a=fault_current_a)

        # Aggregate results
        result = ElectricalSafetyTestResult(
            insulation_resistance=insulation,
            wet_leakage_current=wet_leakage,
            dielectric_strength=dielectric,
            ground_continuity=ground,
            bypass_diode_thermal=bypass_diode,
        )

        logger.info(
            f"Electrical safety tests complete: "
            f"{'ALL PASSED' if result.all_tests_passed else 'SOME FAILED'}"
        )

        return result
