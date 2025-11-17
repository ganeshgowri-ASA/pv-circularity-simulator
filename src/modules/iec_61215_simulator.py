"""
IEC 61215 PV Module Qualification Test Simulator.

This module provides comprehensive simulation of IEC 61215 module qualification tests (MQT),
including thermal cycling, humidity freeze, damp heat, UV preconditioning, hail impact,
mechanical load testing, and qualification report generation.

IEC 61215 is the international standard for design qualification and type approval of
terrestrial photovoltaic (PV) modules.
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from pathlib import Path

from ..models import (
    ModuleConfig,
    TestResults,
    QualificationReport,
    TestStatus,
    VisualDefect,
    DefectType,
    IVCurveData,
)


class IEC61215TestSimulator:
    """
    Simulator for IEC 61215 module qualification tests.

    This class simulates the full suite of IEC 61215 tests including thermal stress,
    humidity exposure, mechanical loads, and environmental exposure tests. It models
    realistic degradation patterns and generates comprehensive qualification reports.

    Attributes:
        module: Module configuration being tested
        random_seed: Random seed for reproducible simulations
        strictness_factor: Multiplier for degradation (1.0 = typical, >1.0 = stricter)
    """

    # IEC 61215 Test Specifications
    THERMAL_CYCLE_SPEC = {
        "temp_min": -40,  # °C
        "temp_max": 85,   # °C
        "cycle_duration": 6,  # hours
        "max_degradation_per_cycle": 0.05,  # %
    }

    HUMIDITY_FREEZE_SPEC = {
        "humidity_temp": 85,  # °C
        "humidity_rh": 85,    # %
        "freeze_temp": -40,   # °C
        "cycle_duration": 20,  # hours
        "max_degradation_per_cycle": 0.06,  # %
    }

    DAMP_HEAT_SPEC = {
        "temperature": 85,  # °C
        "humidity": 85,     # %
        "standard_duration": 1000,  # hours
        "max_degradation_per_1000h": 5.0,  # %
    }

    UV_PRECONDITIONING_SPEC = {
        "irradiance": 1000,  # W/m²
        "uv_dose_total": 15,  # kWh/m²
        "temperature": 60,    # °C
        "max_degradation": 2.0,  # %
    }

    HAIL_IMPACT_SPEC = {
        "standard_diameter": 25,  # mm
        "standard_velocity": 23,  # m/s
        "kinetic_energy": 0.5,    # J (for standard test)
        "critical_energy": 2.0,   # J (module failure threshold)
    }

    MECHANICAL_LOAD_SPEC = {
        "standard_load": 2400,  # Pa
        "max_deflection": 0.02,  # m (20mm for standard module)
        "cycles": 3,
    }

    INSULATION_RESISTANCE_MIN = 40.0  # MΩ·m² (minimum requirement)
    WET_LEAKAGE_CURRENT_MAX = 1.0     # mA (Class A requirement)
    HOTSPOT_TEMP_MAX = 20.0           # °C above average
    MAX_TOTAL_DEGRADATION = 5.0       # % (maximum allowable degradation)
    MIN_POWER_RETENTION = 95.0        # % (minimum power after all tests)

    def __init__(
        self,
        module: ModuleConfig,
        random_seed: Optional[int] = None,
        strictness_factor: float = 1.0,
    ) -> None:
        """
        Initialize the IEC 61215 test simulator.

        Args:
            module: Module configuration to test
            random_seed: Random seed for reproducible results
            strictness_factor: Degradation multiplier (1.0 = typical, >1.0 = stricter)
        """
        self.module = module
        self.strictness_factor = max(0.1, strictness_factor)
        self._rng = np.random.default_rng(random_seed)
        self._test_sequence_number = 0

    def _generate_iv_curve(
        self,
        voc: float,
        isc: float,
        vmp: float,
        imp: float,
        num_points: int = 100,
    ) -> IVCurveData:
        """
        Generate realistic I-V curve data.

        Args:
            voc: Open-circuit voltage (V)
            isc: Short-circuit current (A)
            vmp: Voltage at maximum power point (V)
            imp: Current at maximum power point (A)
            num_points: Number of points in the curve

        Returns:
            I-V curve data with voltage, current, and power arrays
        """
        # Single-diode model parameters
        voltage = np.linspace(0, voc, num_points)

        # Ideality factor and series resistance (estimated)
        n = 1.2  # Ideality factor
        rs = 0.005  # Series resistance (Ω)
        rsh = 1000  # Shunt resistance (Ω)

        # Thermal voltage
        vt = 0.026  # At 25°C

        # Calculate current using simplified single-diode model
        current = []
        for v in voltage:
            # Iterative solution for current
            i = isc
            for _ in range(10):  # Newton-Raphson iterations
                i_diode = isc * (1 - np.exp((v - i * rs) / (n * vt)) + (v - i * rs) / rsh)
                i = i_diode
            current.append(max(0, i))

        current = np.array(current)
        power = voltage * current

        # Find actual maximum power point
        max_idx = np.argmax(power)
        actual_vmp = voltage[max_idx]
        actual_imp = current[max_idx]
        pmax = power[max_idx]
        fill_factor = pmax / (voc * isc) if (voc * isc) > 0 else 0

        return IVCurveData(
            voltage=voltage.tolist(),
            current=current.tolist(),
            power=power.tolist(),
            voc=voc,
            isc=isc,
            vmp=actual_vmp,
            imp=actual_imp,
            pmax=pmax,
            fill_factor=fill_factor,
        )

    def _apply_degradation(
        self,
        initial_power: float,
        degradation_percent: float,
    ) -> Tuple[float, float, float, float]:
        """
        Apply degradation to module parameters.

        Args:
            initial_power: Initial module power (W)
            degradation_percent: Degradation percentage

        Returns:
            Tuple of (final_power, voc_new, isc_new, vmp_new, imp_new)
        """
        degradation_factor = 1 - (degradation_percent / 100)
        final_power = initial_power * degradation_factor

        # Distribute degradation between voltage and current
        # Typically: 60% voltage loss, 40% current loss
        voc_degradation = 0.6 * degradation_percent
        isc_degradation = 0.4 * degradation_percent

        voc_new = self.module.voc * (1 - voc_degradation / 100)
        isc_new = self.module.isc * (1 - isc_degradation / 100)
        vmp_new = self.module.vmp * (1 - voc_degradation / 100)
        imp_new = self.module.imp * (1 - isc_degradation / 100)

        return final_power, voc_new, isc_new, vmp_new, imp_new

    def _generate_visual_defects(
        self,
        test_severity: float,
        defect_probability: float = 0.3,
    ) -> List[VisualDefect]:
        """
        Generate visual defects based on test severity.

        Args:
            test_severity: Severity factor (0-1, higher = more severe)
            defect_probability: Probability of defects occurring

        Returns:
            List of visual defects
        """
        defects = []

        if self._rng.random() < defect_probability * test_severity:
            # Possible defect types based on test
            defect_types = [
                (DefectType.DELAMINATION, "minor", "Edge delamination observed"),
                (DefectType.BUBBLE, "minor", "Small bubbles in encapsulant"),
                (DefectType.DISCOLORATION, "minor", "Slight discoloration of encapsulant"),
            ]

            if test_severity > 0.7:
                defect_types.extend([
                    (DefectType.CRACK, "major", "Micro-cracks in cells"),
                    (DefectType.BROKEN_INTERCONNECT, "major", "Broken cell interconnect"),
                ])

            num_defects = min(3, int(test_severity * 5))
            for _ in range(num_defects):
                if self._rng.random() < 0.5:
                    defect_type, severity, desc = self._rng.choice(defect_types)
                    defects.append(
                        VisualDefect(
                            defect_type=defect_type,
                            severity=severity,
                            location=f"Cell {self._rng.integers(1, self.module.cells_in_series)}",
                            size=self._rng.uniform(1, 10),
                            description=desc,
                        )
                    )

        return defects

    def simulate_thermal_cycling(
        self,
        module: ModuleConfig,
        cycles: int = 200,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-10: Thermal Cycling Test.

        Modules are cycled between -40°C and +85°C to test thermal stress resistance.
        Standard requirement: 200 cycles.

        Args:
            module: Module configuration to test
            cycles: Number of thermal cycles (default: 200)

        Returns:
            Test results including power degradation and defects
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Degradation model: increases with cycles, affected by materials
        base_degradation_per_cycle = self.THERMAL_CYCLE_SPEC["max_degradation_per_cycle"]

        # Material factors
        material_factor = 1.0
        if module.encapsulant_type.upper() == "POE":
            material_factor *= 0.7  # POE better thermal stability
        if module.module_type.name == "GLASS_GLASS":
            material_factor *= 0.8  # Better thermal expansion matching

        total_degradation = (
            base_degradation_per_cycle * cycles * material_factor * self.strictness_factor
        )

        # Add random variation (±20%)
        total_degradation *= self._rng.uniform(0.8, 1.2)
        total_degradation = min(total_degradation, 5.0)  # Cap at 5%

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Visual defects (thermal stress can cause delamination)
        test_severity = min(cycles / 200, 1.0) * self.strictness_factor
        visual_defects = self._generate_visual_defects(test_severity, defect_probability=0.4)

        # Insulation resistance (should remain high)
        insulation_resistance = self._rng.uniform(50, 100)  # MΩ·m²

        # Determine pass/fail
        status = TestStatus.PASSED
        if total_degradation > 5.0:
            status = TestStatus.FAILED
        elif any(d.severity == "major" for d in visual_defects):
            status = TestStatus.FAILED
        elif insulation_resistance < self.INSULATION_RESISTANCE_MIN:
            status = TestStatus.FAILED
        elif total_degradation > 3.0 or len(visual_defects) > 2:
            status = TestStatus.CONDITIONAL

        return TestResults(
            test_id="MQT-10",
            test_name="Thermal Cycling Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "cycles": cycles,
                "temp_min": self.THERMAL_CYCLE_SPEC["temp_min"],
                "temp_max": self.THERMAL_CYCLE_SPEC["temp_max"],
                "cycle_duration": self.THERMAL_CYCLE_SPEC["cycle_duration"],
            },
            observations=f"Completed {cycles} thermal cycles between -40°C and +85°C",
            compliance_notes=f"IEC 61215 MQT-10: {status.value}",
        )

    def simulate_humidity_freeze(
        self,
        module: ModuleConfig,
        cycles: int = 10,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-11: Humidity Freeze Test.

        Modules undergo cycles of humidity exposure (85°C/85%RH) followed by freezing (-40°C).
        Standard requirement: 10 cycles.

        Args:
            module: Module configuration to test
            cycles: Number of humidity-freeze cycles (default: 10)

        Returns:
            Test results including power degradation and defects
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Degradation model
        base_degradation_per_cycle = self.HUMIDITY_FREEZE_SPEC["max_degradation_per_cycle"]

        # Material factors
        material_factor = 1.0
        if module.backsheet_type and "Tedlar" in module.backsheet_type:
            material_factor *= 0.9  # Better moisture barrier
        if module.module_type.name == "GLASS_GLASS":
            material_factor *= 0.6  # Excellent moisture barrier

        total_degradation = (
            base_degradation_per_cycle * cycles * material_factor * self.strictness_factor
        )
        total_degradation *= self._rng.uniform(0.8, 1.2)
        total_degradation = min(total_degradation, 5.0)

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Visual defects (moisture can cause delamination, corrosion)
        test_severity = min(cycles / 10, 1.0) * self.strictness_factor
        visual_defects = self._generate_visual_defects(test_severity, defect_probability=0.5)

        # Insulation resistance (critical for humidity test)
        insulation_resistance = self._rng.uniform(45, 80)  # Lower due to moisture

        # Wet leakage current
        wet_leakage_current = self._rng.uniform(0.1, 0.8)  # mA

        # Determine pass/fail
        status = TestStatus.PASSED
        if total_degradation > 5.0:
            status = TestStatus.FAILED
        elif insulation_resistance < self.INSULATION_RESISTANCE_MIN:
            status = TestStatus.FAILED
        elif wet_leakage_current > self.WET_LEAKAGE_CURRENT_MAX:
            status = TestStatus.FAILED
        elif any(d.defect_type == DefectType.DELAMINATION and d.severity == "major" for d in visual_defects):
            status = TestStatus.FAILED
        elif total_degradation > 3.0:
            status = TestStatus.CONDITIONAL

        return TestResults(
            test_id="MQT-11",
            test_name="Humidity Freeze Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            wet_leakage_current=wet_leakage_current,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "cycles": cycles,
                "humidity_temp": self.HUMIDITY_FREEZE_SPEC["humidity_temp"],
                "humidity_rh": self.HUMIDITY_FREEZE_SPEC["humidity_rh"],
                "freeze_temp": self.HUMIDITY_FREEZE_SPEC["freeze_temp"],
            },
            observations=f"Completed {cycles} humidity-freeze cycles",
            compliance_notes=f"IEC 61215 MQT-11: {status.value}",
        )

    def simulate_damp_heat(
        self,
        module: ModuleConfig,
        hours: int = 1000,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-12: Damp Heat Test.

        Modules are exposed to 85°C and 85% relative humidity for extended periods.
        Standard requirement: 1000 hours.

        Args:
            module: Module configuration to test
            hours: Duration of exposure in hours (default: 1000)

        Returns:
            Test results including power degradation and defects
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Degradation model (linear with time)
        base_degradation_per_1000h = self.DAMP_HEAT_SPEC["max_degradation_per_1000h"]

        # Material factors
        material_factor = 1.0
        if module.encapsulant_type.upper() == "POE":
            material_factor *= 0.7  # POE better moisture resistance
        if module.module_type.name == "GLASS_GLASS":
            material_factor *= 0.5  # Excellent protection
        if module.backsheet_type and "Tedlar" in module.backsheet_type:
            material_factor *= 0.9

        total_degradation = (
            base_degradation_per_1000h * (hours / 1000) * material_factor * self.strictness_factor
        )
        total_degradation *= self._rng.uniform(0.8, 1.2)
        total_degradation = min(total_degradation, 5.0)

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Visual defects (moisture penetration)
        test_severity = min(hours / 1000, 1.0) * self.strictness_factor
        visual_defects = self._generate_visual_defects(test_severity, defect_probability=0.6)

        # Add discoloration (common in damp heat)
        if test_severity > 0.5 and self._rng.random() < 0.7:
            visual_defects.append(
                VisualDefect(
                    defect_type=DefectType.DISCOLORATION,
                    severity="minor",
                    location="Encapsulant",
                    description="Yellowing/browning of encapsulant material",
                )
            )

        # Insulation resistance
        insulation_resistance = self._rng.uniform(40, 70)

        # Wet leakage current
        wet_leakage_current = self._rng.uniform(0.2, 0.9)

        # Determine pass/fail
        status = TestStatus.PASSED
        if total_degradation > 5.0:
            status = TestStatus.FAILED
        elif insulation_resistance < self.INSULATION_RESISTANCE_MIN:
            status = TestStatus.FAILED
        elif wet_leakage_current > self.WET_LEAKAGE_CURRENT_MAX:
            status = TestStatus.FAILED
        elif total_degradation > 3.5:
            status = TestStatus.CONDITIONAL

        return TestResults(
            test_id="MQT-12",
            test_name="Damp Heat Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            wet_leakage_current=wet_leakage_current,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "duration_hours": hours,
                "temperature": self.DAMP_HEAT_SPEC["temperature"],
                "humidity": self.DAMP_HEAT_SPEC["humidity"],
            },
            observations=f"Exposed to 85°C/85%RH for {hours} hours",
            compliance_notes=f"IEC 61215 MQT-12: {status.value}",
        )

    def simulate_uv_preconditioning(
        self,
        module: ModuleConfig,
        hours: float = 48.0,
        dose: float = 15.0,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-13: UV Preconditioning Test.

        Modules are exposed to UV radiation to test encapsulant and backsheet UV stability.
        Standard: 15 kWh/m² UV dose.

        Args:
            module: Module configuration to test
            hours: Exposure duration (hours)
            dose: UV dose (kWh/m²)

        Returns:
            Test results including power degradation and defects
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Degradation model (based on UV dose)
        base_degradation = self.UV_PRECONDITIONING_SPEC["max_degradation"]

        # Material factors
        material_factor = 1.0
        if module.encapsulant_type.upper() == "POE":
            material_factor *= 0.6  # POE excellent UV resistance
        if module.backsheet_type and "Tedlar" in module.backsheet_type:
            material_factor *= 0.8  # Tedlar good UV resistance

        total_degradation = (
            base_degradation * (dose / 15.0) * material_factor * self.strictness_factor
        )
        total_degradation *= self._rng.uniform(0.8, 1.2)
        total_degradation = min(total_degradation, 3.0)

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Visual defects (UV mainly causes discoloration)
        visual_defects = []
        if self._rng.random() < 0.8:
            visual_defects.append(
                VisualDefect(
                    defect_type=DefectType.DISCOLORATION,
                    severity="minor",
                    location="Backsheet/encapsulant",
                    description="Slight yellowing from UV exposure",
                )
            )

        # Insulation resistance (should not be affected)
        insulation_resistance = self._rng.uniform(60, 100)

        # Determine pass/fail
        status = TestStatus.PASSED
        if total_degradation > 5.0:
            status = TestStatus.FAILED
        elif total_degradation > 2.5:
            status = TestStatus.CONDITIONAL

        return TestResults(
            test_id="MQT-13",
            test_name="UV Preconditioning Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "duration_hours": hours,
                "uv_dose_kwh_m2": dose,
                "irradiance": self.UV_PRECONDITIONING_SPEC["irradiance"],
                "temperature": self.UV_PRECONDITIONING_SPEC["temperature"],
            },
            observations=f"UV exposure: {dose} kWh/m² over {hours} hours",
            compliance_notes=f"IEC 61215 MQT-13: {status.value}",
        )

    def simulate_hail_impact(
        self,
        module: ModuleConfig,
        diameter: float = 25.0,
        velocity: float = 23.0,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-17: Hail Impact Test.

        Ice balls are shot at the module to test mechanical robustness.
        Standard: 25mm diameter at 23 m/s.

        Args:
            module: Module configuration to test
            diameter: Hail ball diameter (mm)
            velocity: Impact velocity (m/s)

        Returns:
            Test results including structural integrity
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Calculate impact energy
        # E = 0.5 * m * v^2, where m ∝ d³
        standard_energy = self.HAIL_IMPACT_SPEC["kinetic_energy"]
        impact_energy = standard_energy * ((diameter / 25) ** 3) * ((velocity / 23) ** 2)

        # Damage threshold depends on glass thickness and module construction
        glass_strength_factor = module.glass_thickness_front / 3.2  # Normalized to 3.2mm
        if module.module_type.name == "GLASS_GLASS":
            glass_strength_factor *= 1.3  # Additional back support

        failure_threshold = self.HAIL_IMPACT_SPEC["critical_energy"] * glass_strength_factor

        # Determine damage
        if impact_energy < failure_threshold * 0.5:
            # No damage
            total_degradation = 0.0
            visual_defects = []
            status = TestStatus.PASSED
            observations = "No visible damage from hail impact"
        elif impact_energy < failure_threshold * 0.8:
            # Minor damage
            total_degradation = self._rng.uniform(0.1, 0.5)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="minor",
                    location="Front glass surface",
                    description="Minor surface cracks, cells intact",
                )
            ]
            status = TestStatus.PASSED
            observations = "Minor surface cracks, no cell damage"
        elif impact_energy < failure_threshold:
            # Moderate damage
            total_degradation = self._rng.uniform(1.0, 3.0)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="major",
                    location="Impact zone",
                    description="Glass cracks, possible cell micro-cracks",
                ),
                VisualDefect(
                    defect_type=DefectType.BROKEN_CELL,
                    severity="minor",
                    location="Impact zone",
                    description="Cell micro-cracks detected",
                ),
            ]
            status = TestStatus.CONDITIONAL
            observations = "Significant cracks but module still functional"
        else:
            # Severe damage / failure
            total_degradation = self._rng.uniform(5.0, 15.0)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="critical",
                    location="Impact zone",
                    description="Glass penetration, cell fracture",
                ),
                VisualDefect(
                    defect_type=DefectType.BROKEN_CELL,
                    severity="critical",
                    location="Multiple cells",
                    description="Multiple cells broken, interconnects damaged",
                ),
            ]
            status = TestStatus.FAILED
            observations = "Module failure: glass penetration and cell destruction"

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Insulation resistance (may be compromised if glass broken)
        if status == TestStatus.FAILED:
            insulation_resistance = self._rng.uniform(10, 35)  # Below minimum
        else:
            insulation_resistance = self._rng.uniform(50, 100)

        return TestResults(
            test_id="MQT-17",
            test_name="Hail Impact Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "hail_diameter_mm": diameter,
                "impact_velocity_ms": velocity,
                "impact_energy_J": impact_energy,
                "failure_threshold_J": failure_threshold,
            },
            observations=observations,
            compliance_notes=f"IEC 61215 MQT-17: {status.value}",
        )

    def simulate_mechanical_load(
        self,
        module: ModuleConfig,
        front_load: float = 2400.0,
        back_load: float = 2400.0,
    ) -> TestResults:
        """
        Simulate IEC 61215 MQT-18: Mechanical Load Test.

        Static and cyclic loads are applied to test structural integrity.
        Standard: ±2400 Pa for 3 cycles.

        Args:
            module: Module configuration to test
            front_load: Front surface load (Pa)
            back_load: Back surface load (Pa)

        Returns:
            Test results including structural deformation
        """
        self._test_sequence_number += 1
        initial_power = module.rated_power

        # Calculate deflection based on module dimensions and load
        # Simplified beam deflection: δ ∝ F * L³ / (E * I)
        length = module.dimensions[0] / 1000  # Convert mm to m
        width = module.dimensions[1] / 1000
        thickness = module.dimensions[2] / 1000

        # Moment of inertia (simplified)
        moment_of_inertia = (width * thickness ** 3) / 12

        # Effective modulus (glass-dominated)
        effective_modulus = 70e9  # Pa (glass modulus)

        # Maximum load
        max_load = max(front_load, back_load)

        # Deflection (simplified)
        deflection = (max_load * length ** 4) / (384 * effective_modulus * moment_of_inertia)

        # Frame support reduces deflection
        if module.frame_material and "Aluminum" in module.frame_material:
            deflection *= 0.3  # Frame provides significant support

        # Determine damage based on deflection
        max_allowable_deflection = self.MECHANICAL_LOAD_SPEC["max_deflection"]

        if deflection < max_allowable_deflection * 0.5:
            total_degradation = 0.0
            visual_defects = []
            status = TestStatus.PASSED
            observations = "No visible deformation or damage"
        elif deflection < max_allowable_deflection:
            total_degradation = self._rng.uniform(0.1, 0.5)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="minor",
                    location="Cell interconnects",
                    description="Minor stress on interconnects",
                )
            ]
            status = TestStatus.PASSED
            observations = f"Deflection {deflection*1000:.1f}mm, within limits"
        elif deflection < max_allowable_deflection * 1.5:
            total_degradation = self._rng.uniform(1.0, 3.0)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="major",
                    location="Multiple cells",
                    description="Cell cracks from excessive bending",
                ),
                VisualDefect(
                    defect_type=DefectType.BROKEN_INTERCONNECT,
                    severity="major",
                    location="Center region",
                    description="Broken interconnect ribbons",
                ),
            ]
            status = TestStatus.CONDITIONAL
            observations = f"Excessive deflection {deflection*1000:.1f}mm"
        else:
            total_degradation = self._rng.uniform(5.0, 10.0)
            visual_defects = [
                VisualDefect(
                    defect_type=DefectType.CRACK,
                    severity="critical",
                    location="Throughout module",
                    description="Widespread cell fracture",
                ),
                VisualDefect(
                    defect_type=DefectType.BROKEN_CELL,
                    severity="critical",
                    location="Center region",
                    description="Multiple broken cells",
                ),
            ]
            status = TestStatus.FAILED
            observations = f"Structural failure: deflection {deflection*1000:.1f}mm"

        final_power, voc_new, isc_new, vmp_new, imp_new = self._apply_degradation(
            initial_power, total_degradation
        )

        # Generate I-V curves
        iv_before = self._generate_iv_curve(module.voc, module.isc, module.vmp, module.imp)
        iv_after = self._generate_iv_curve(voc_new, isc_new, vmp_new, imp_new)

        # Insulation resistance
        insulation_resistance = self._rng.uniform(50, 100)

        return TestResults(
            test_id="MQT-18",
            test_name="Mechanical Load Test",
            status=status,
            initial_power=initial_power,
            final_power=final_power,
            power_degradation=total_degradation,
            visual_defects=visual_defects,
            insulation_resistance=insulation_resistance,
            iv_curve_before=iv_before,
            iv_curve_after=iv_after,
            test_parameters={
                "front_load_Pa": front_load,
                "back_load_Pa": back_load,
                "cycles": self.MECHANICAL_LOAD_SPEC["cycles"],
                "deflection_m": deflection,
                "max_allowable_deflection_m": max_allowable_deflection,
            },
            observations=observations,
            compliance_notes=f"IEC 61215 MQT-18: {status.value}",
        )

    def generate_qualification_report(
        self,
        all_tests: List[TestResults],
    ) -> QualificationReport:
        """
        Generate comprehensive IEC 61215 qualification report.

        Args:
            all_tests: List of all test results

        Returns:
            Complete qualification report with pass/fail determination
        """
        # Calculate total degradation
        total_degradation = sum(test.power_degradation for test in all_tests)

        # Determine overall status
        failed_tests = [t for t in all_tests if t.status == TestStatus.FAILED]
        conditional_tests = [t for t in all_tests if t.status == TestStatus.CONDITIONAL]

        if failed_tests or total_degradation > self.MAX_TOTAL_DEGRADATION:
            overall_status = TestStatus.FAILED
        elif conditional_tests or total_degradation > 4.0:
            overall_status = TestStatus.CONDITIONAL
        else:
            overall_status = TestStatus.PASSED

        # Critical failures
        critical_failures = []
        for test in failed_tests:
            critical_failures.append(
                f"{test.test_id} - {test.test_name}: {test.status.value}"
            )

        # Compliance checks
        final_power = self.module.rated_power * (1 - total_degradation / 100)
        power_retention = (final_power / self.module.rated_power) * 100
        power_retention_check = power_retention >= self.MIN_POWER_RETENTION

        # Visual inspection check
        major_defects = [
            d for test in all_tests for d in test.visual_defects
            if d.severity in ["major", "critical"]
        ]
        visual_inspection_check = len(major_defects) == 0

        # Insulation resistance check
        insulation_tests = [t for t in all_tests if t.insulation_resistance is not None]
        min_insulation = min(
            (t.insulation_resistance for t in insulation_tests),
            default=100.0
        )
        insulation_resistance_check = min_insulation >= self.INSULATION_RESISTANCE_MIN

        # Safety check
        leakage_tests = [t for t in all_tests if t.wet_leakage_current is not None]
        max_leakage = max(
            (t.wet_leakage_current for t in leakage_tests),
            default=0.0
        )
        safety_check = max_leakage <= self.WET_LEAKAGE_CURRENT_MAX

        # Generate summary
        summary = f"""
IEC 61215 Qualification Test Report - {self.module.name}

Module Type: {self.module.technology.value} - {self.module.module_type.value}
Rated Power: {self.module.rated_power}W

Total Tests Conducted: {len(all_tests)}
Tests Passed: {sum(1 for t in all_tests if t.status == TestStatus.PASSED)}
Tests Failed: {len(failed_tests)}
Tests Conditional: {len(conditional_tests)}

Total Power Degradation: {total_degradation:.2f}% (Limit: {self.MAX_TOTAL_DEGRADATION}%)
Final Power Retention: {power_retention:.2f}% (Required: ≥{self.MIN_POWER_RETENTION}%)

Overall Status: {overall_status.value.upper()}
        """.strip()

        # Generate recommendations
        recommendations = []
        if total_degradation > 3.0:
            recommendations.append(
                "Consider improved encapsulant materials to reduce degradation"
            )
        if not visual_inspection_check:
            recommendations.append(
                "Review cell interconnect design and lamination process"
            )
        if not insulation_resistance_check:
            recommendations.append(
                "Improve moisture barrier in backsheet and edge sealing"
            )
        if len(conditional_tests) > 0:
            recommendations.append(
                f"Further investigation recommended for: {', '.join(t.test_name for t in conditional_tests)}"
            )
        if overall_status == TestStatus.PASSED:
            recommendations.append(
                "Module design meets IEC 61215 qualification requirements"
            )

        return QualificationReport(
            module_config=self.module,
            test_results=all_tests,
            overall_status=overall_status,
            total_power_degradation=total_degradation,
            critical_failures=critical_failures,
            power_retention_check=power_retention_check,
            visual_inspection_check=visual_inspection_check,
            insulation_resistance_check=insulation_resistance_check,
            safety_check=safety_check,
            summary=summary,
            recommendations=recommendations,
        )

    def plot_power_degradation_timeline(
        self,
        test_results: List[TestResults],
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create power degradation timeline chart.

        Args:
            test_results: List of test results in sequence
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate cumulative degradation
        cumulative_degradation = [0]
        power_values = [self.module.rated_power]
        test_names = ["Initial"]

        for test in test_results:
            cumulative_degradation.append(
                cumulative_degradation[-1] + test.power_degradation
            )
            power_values.append(
                self.module.rated_power * (1 - cumulative_degradation[-1] / 100)
            )
            test_names.append(test.test_id)

        # Plot
        x = range(len(power_values))
        ax.plot(x, power_values, 'o-', linewidth=2, markersize=8, label='Module Power')

        # Add threshold line
        threshold = self.module.rated_power * (self.MIN_POWER_RETENTION / 100)
        ax.axhline(y=threshold, color='r', linestyle='--', label='95% Threshold')

        # Formatting
        ax.set_xlabel('Test Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'IEC 61215 Power Degradation Timeline - {self.module.name}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add degradation percentages
        for i, (xi, yi, deg) in enumerate(zip(x[1:], power_values[1:], cumulative_degradation[1:]), 1):
            ax.annotate(
                f'-{deg:.1f}%',
                xy=(xi, yi),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='red'
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_iv_curve_comparison(
        self,
        test_result: TestResults,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create before/after I-V curve comparison.

        Args:
            test_result: Test result with before/after I-V curves
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        if not test_result.iv_curve_before or not test_result.iv_curve_after:
            raise ValueError("Test result must have before and after I-V curves")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        before = test_result.iv_curve_before
        after = test_result.iv_curve_after

        # I-V curves
        ax1.plot(before.voltage, before.current, 'b-', linewidth=2, label='Before')
        ax1.plot(after.voltage, after.current, 'r--', linewidth=2, label='After')
        ax1.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Current (A)', fontsize=11, fontweight='bold')
        ax1.set_title('I-V Curves', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Mark MPP points
        ax1.plot(before.vmp, before.imp, 'bo', markersize=10, label='MPP Before')
        ax1.plot(after.vmp, after.imp, 'ro', markersize=10, label='MPP After')

        # P-V curves
        ax2.plot(before.voltage, before.power, 'b-', linewidth=2, label='Before')
        ax2.plot(after.voltage, after.power, 'r--', linewidth=2, label='After')
        ax2.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Power (W)', fontsize=11, fontweight='bold')
        ax2.set_title('P-V Curves', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add text box with key metrics
        textstr = f'''Test: {test_result.test_name}
Before: Pmax={before.pmax:.1f}W, FF={before.fill_factor:.3f}
After: Pmax={after.pmax:.1f}W, FF={after.fill_factor:.3f}
Degradation: {test_result.power_degradation:.2f}%'''

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(
            0.05, 0.95, textstr,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )

        plt.suptitle(
            f'{test_result.test_id} - I-V Curve Comparison',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_report_to_excel(
        self,
        report: QualificationReport,
        file_path: Path,
    ) -> None:
        """
        Export qualification report to Excel file.

        Args:
            report: Qualification report to export
            file_path: Path for Excel file
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Parameter': [
                    'Module Name',
                    'Technology',
                    'Rated Power (W)',
                    'Overall Status',
                    'Total Degradation (%)',
                    'Power Retention (%)',
                    'Tests Passed',
                    'Tests Failed',
                    'Report Date',
                ],
                'Value': [
                    report.module_config.name,
                    report.module_config.technology.value,
                    report.module_config.rated_power,
                    report.overall_status.value,
                    f"{report.total_power_degradation:.2f}",
                    f"{(1 - report.total_power_degradation/100) * 100:.2f}",
                    sum(1 for t in report.test_results if t.status == TestStatus.PASSED),
                    len(report.critical_failures),
                    report.report_date.strftime('%Y-%m-%d %H:%M'),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Test results sheet
            test_data = []
            for test in report.test_results:
                test_data.append({
                    'Test ID': test.test_id,
                    'Test Name': test.test_name,
                    'Status': test.status.value,
                    'Initial Power (W)': f"{test.initial_power:.2f}",
                    'Final Power (W)': f"{test.final_power:.2f}",
                    'Degradation (%)': f"{test.power_degradation:.2f}",
                    'Visual Defects': len(test.visual_defects),
                    'Insulation Resistance (MΩ·m²)': f"{test.insulation_resistance:.1f}" if test.insulation_resistance else "N/A",
                    'Observations': test.observations,
                })
            pd.DataFrame(test_data).to_excel(writer, sheet_name='Test Results', index=False)

            # Defects sheet
            defect_data = []
            for test in report.test_results:
                for defect in test.visual_defects:
                    defect_data.append({
                        'Test ID': test.test_id,
                        'Defect Type': defect.defect_type.value,
                        'Severity': defect.severity,
                        'Location': defect.location,
                        'Description': defect.description,
                    })
            if defect_data:
                pd.DataFrame(defect_data).to_excel(writer, sheet_name='Visual Defects', index=False)

            # Compliance sheet
            compliance_data = {
                'Check': [
                    'Power Retention ≥95%',
                    'Visual Inspection',
                    'Insulation Resistance ≥40 MΩ·m²',
                    'Safety (Leakage Current <1mA)',
                ],
                'Status': [
                    'PASS' if report.power_retention_check else 'FAIL',
                    'PASS' if report.visual_inspection_check else 'FAIL',
                    'PASS' if report.insulation_resistance_check else 'FAIL',
                    'PASS' if report.safety_check else 'FAIL',
                ],
            }
            pd.DataFrame(compliance_data).to_excel(writer, sheet_name='Compliance', index=False)

    def export_report_to_pdf(
        self,
        report: QualificationReport,
        file_path: Path,
    ) -> None:
        """
        Export qualification report to PDF file.

        Args:
            report: Qualification report to export
            file_path: Path for PDF file
        """
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(str(file_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        story.append(Paragraph('IEC 61215 Module Qualification Report', title_style))
        story.append(Spacer(1, 0.2*inch))

        # Module information
        story.append(Paragraph('<b>Module Information</b>', styles['Heading2']))
        module_data = [
            ['Parameter', 'Value'],
            ['Module Name', report.module_config.name],
            ['Technology', report.module_config.technology.value],
            ['Module Type', report.module_config.module_type.value],
            ['Rated Power', f"{report.module_config.rated_power} W"],
            ['Efficiency', f"{report.module_config.efficiency} %"],
            ['Dimensions', f"{report.module_config.dimensions[0]}×{report.module_config.dimensions[1]}×{report.module_config.dimensions[2]} mm"],
        ]
        module_table = Table(module_data, colWidths=[3*inch, 3*inch])
        module_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(module_table)
        story.append(Spacer(1, 0.3*inch))

        # Test summary
        story.append(Paragraph('<b>Test Summary</b>', styles['Heading2']))
        summary_data = [
            ['Metric', 'Value', 'Status'],
            ['Overall Status', report.overall_status.value.upper(),
             'PASS' if report.overall_status == TestStatus.PASSED else 'FAIL'],
            ['Total Power Degradation', f"{report.total_power_degradation:.2f}%",
             'PASS' if report.total_power_degradation <= 5.0 else 'FAIL'],
            ['Power Retention', f"{(1-report.total_power_degradation/100)*100:.2f}%",
             'PASS' if report.power_retention_check else 'FAIL'],
            ['Tests Conducted', str(len(report.test_results)), '-'],
            ['Tests Passed', str(sum(1 for t in report.test_results if t.status == TestStatus.PASSED)), '-'],
            ['Tests Failed', str(len(report.critical_failures)), '-'],
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(PageBreak())

        # Detailed test results
        story.append(Paragraph('<b>Detailed Test Results</b>', styles['Heading2']))
        test_data = [['Test ID', 'Test Name', 'Status', 'Degradation (%)']]
        for test in report.test_results:
            test_data.append([
                test.test_id,
                test.test_name,
                test.status.value,
                f"{test.power_degradation:.2f}",
            ])
        test_table = Table(test_data, colWidths=[1*inch, 3*inch, 1.2*inch, 1.3*inch])
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(test_table)
        story.append(Spacer(1, 0.3*inch))

        # Recommendations
        if report.recommendations:
            story.append(Paragraph('<b>Recommendations</b>', styles['Heading2']))
            for i, rec in enumerate(report.recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

        # Build PDF
        doc.build(story)
