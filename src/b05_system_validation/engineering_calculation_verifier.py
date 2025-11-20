"""Engineering calculation verifier for PV systems.

This module provides comprehensive verification of electrical engineering
calculations for PV systems including string calculations, voltage drop,
short circuit analysis, grounding, and overcurrent protection.
"""

import math
from typing import Any, Dict, List, Optional

from src.models.validation_models import (
    EngineeringCalculation,
    IssueItem,
    IssueSeverity,
    SystemConfiguration,
)


class EngineeringCalculationVerifier:
    """Comprehensive engineering calculation verifier for PV systems.

    Verifies electrical calculations including string sizing, voltage drop,
    short circuit current, grounding conductor sizing, and overcurrent
    protection device selection.

    Attributes:
        config: System configuration to verify
        calculations: List of all calculation results
        wire_resistance_table: Standard wire resistance values (Ω/km)
    """

    # Standard wire resistance at 75°C (Ω/km for copper)
    WIRE_RESISTANCE_TABLE: Dict[str, float] = {
        "14AWG": 8.28,
        "12AWG": 5.21,
        "10AWG": 3.28,
        "8AWG": 2.06,
        "6AWG": 1.30,
        "4AWG": 0.815,
        "2AWG": 0.513,
        "1AWG": 0.407,
        "1/0AWG": 0.323,
        "2/0AWG": 0.256,
        "3/0AWG": 0.203,
        "4/0AWG": 0.161,
        "250MCM": 0.137,
        "300MCM": 0.114,
        "350MCM": 0.0977,
        "400MCM": 0.0854,
        "500MCM": 0.0683,
    }

    # Wire ampacity at 75°C (A for copper, 3 current-carrying conductors)
    WIRE_AMPACITY_TABLE: Dict[str, float] = {
        "14AWG": 20,
        "12AWG": 25,
        "10AWG": 35,
        "8AWG": 50,
        "6AWG": 65,
        "4AWG": 85,
        "2AWG": 115,
        "1AWG": 130,
        "1/0AWG": 150,
        "2/0AWG": 175,
        "3/0AWG": 200,
        "4/0AWG": 230,
        "250MCM": 255,
        "300MCM": 285,
        "350MCM": 310,
        "400MCM": 335,
        "500MCM": 380,
    }

    # Standard OCPD sizes (A)
    STANDARD_OCPD_SIZES = [
        15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
        110, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
        500, 600, 700, 800, 1000, 1200
    ]

    def __init__(self, config: SystemConfiguration) -> None:
        """Initialize engineering calculation verifier.

        Args:
            config: System configuration to verify
        """
        self.config = config
        self.calculations: List[EngineeringCalculation] = []

    def verify_string_calculations(
        self,
        modules_per_string: Optional[int] = None,
        module_voc: Optional[float] = None,
        module_vmp: Optional[float] = None,
        module_isc: Optional[float] = None,
        module_imp: Optional[float] = None,
    ) -> List[EngineeringCalculation]:
        """Verify string voltage and current calculations.

        Validates that string configurations meet voltage and current requirements
        considering temperature effects and safety factors.

        Args:
            modules_per_string: Number of modules in series (uses config if not provided)
            module_voc: Module open circuit voltage (V)
            module_vmp: Module max power voltage (V)
            module_isc: Module short circuit current (A)
            module_imp: Module max power current (A)

        Returns:
            List of calculation results for string design
        """
        results: List[EngineeringCalculation] = []

        # Use provided values or derive from system config
        mps = modules_per_string or self.config.modules_per_string
        voc = module_voc or (self.config.max_voltage_voc / mps)
        vmp = module_vmp or (self.config.operating_voltage_vmp / mps)
        isc = module_isc or self.config.max_current_isc
        imp = module_imp or self.config.operating_current_imp

        # Calculate string voltage at STC
        string_voc_stc = voc * mps
        string_vmp_stc = vmp * mps

        # Temperature coefficient (typical for crystalline silicon)
        temp_coeff = -0.0035  # per °C

        # Calculate string voltage at minimum temperature
        temp_delta = self.config.ambient_temp_min - 25  # STC is 25°C
        string_voc_min_temp = string_voc_stc * (1 + temp_coeff * temp_delta)

        # Verify maximum voltage
        results.append(
            EngineeringCalculation(
                calculation_type="string_voltage_max",
                description="Maximum string voltage at minimum temperature",
                input_parameters={
                    "modules_per_string": mps,
                    "module_voc": voc,
                    "ambient_temp_min": self.config.ambient_temp_min,
                    "temp_coefficient": temp_coeff,
                },
                calculated_value=round(string_voc_min_temp, 2),
                unit="V",
                threshold_max=1500.0,
                is_valid=string_voc_min_temp <= 1500.0,
                formula="Voc_max = Voc_STC × modules × (1 + temp_coeff × (T_min - 25))",
                reference="NEC 690.7(A)",
                notes=(
                    f"String Voc at {self.config.ambient_temp_min}°C: {string_voc_min_temp:.2f}V. "
                    f"Must not exceed inverter maximum input voltage."
                ),
            )
        )

        # Calculate string voltage at maximum temperature
        temp_delta_max = self.config.ambient_temp_max - 25
        string_vmp_max_temp = string_vmp_stc * (1 + temp_coeff * temp_delta_max)

        results.append(
            EngineeringCalculation(
                calculation_type="string_voltage_operating",
                description="Operating string voltage at maximum temperature",
                input_parameters={
                    "modules_per_string": mps,
                    "module_vmp": vmp,
                    "ambient_temp_max": self.config.ambient_temp_max,
                    "temp_coefficient": temp_coeff,
                },
                calculated_value=round(string_vmp_max_temp, 2),
                unit="V",
                threshold_min=200.0,  # Minimum for efficient inverter operation
                is_valid=string_vmp_max_temp >= 200.0,
                formula="Vmp_operating = Vmp_STC × modules × (1 + temp_coeff × (T_max - 25))",
                reference="Inverter MPPT range",
                notes=(
                    f"String Vmp at {self.config.ambient_temp_max}°C: {string_vmp_max_temp:.2f}V. "
                    f"Must be within inverter MPPT voltage range."
                ),
            )
        )

        # String current calculations
        results.append(
            EngineeringCalculation(
                calculation_type="string_current_max",
                description="Maximum string short-circuit current",
                input_parameters={
                    "module_isc": isc,
                    "irradiance_factor": 1.25,  # 125% per NEC
                },
                calculated_value=round(isc * 1.25, 2),
                unit="A",
                threshold_max=None,
                is_valid=True,
                formula="Isc_max = Isc_STC × 1.25",
                reference="NEC 690.8(A)(1)",
                notes=(
                    f"Maximum string current: {isc * 1.25:.2f}A. "
                    f"Conductors and OCPD must be rated for this current."
                ),
            )
        )

        self.calculations.extend(results)
        return results

    def check_voltage_drop(
        self,
        current: float,
        distance: float,
        wire_gauge: str,
        voltage: Optional[float] = None,
        max_drop_percent: float = 2.0,
    ) -> EngineeringCalculation:
        """Calculate and verify voltage drop in DC conductors.

        Args:
            current: Current in conductor (A)
            distance: One-way distance (m)
            wire_gauge: Wire gauge (e.g., "10AWG", "4/0AWG")
            voltage: System voltage for percentage calculation (uses config if not provided)
            max_drop_percent: Maximum allowable voltage drop (%)

        Returns:
            Voltage drop calculation result
        """
        if wire_gauge not in self.WIRE_RESISTANCE_TABLE:
            raise ValueError(f"Unknown wire gauge: {wire_gauge}")

        # Get wire resistance (Ω/km)
        resistance_per_km = self.WIRE_RESISTANCE_TABLE[wire_gauge]

        # Calculate total resistance for round trip (Ω)
        # Distance is one-way, so multiply by 2 for round trip
        distance_km = distance / 1000.0
        total_resistance = resistance_per_km * distance_km * 2

        # Calculate voltage drop (V)
        voltage_drop_v = current * total_resistance

        # Calculate percentage drop
        sys_voltage = voltage or self.config.system_voltage_vdc
        voltage_drop_percent = (voltage_drop_v / sys_voltage) * 100

        is_valid = voltage_drop_percent <= max_drop_percent

        calc = EngineeringCalculation(
            calculation_type="voltage_drop",
            description=f"DC voltage drop for {wire_gauge} conductor",
            input_parameters={
                "current_A": current,
                "distance_m": distance,
                "wire_gauge": wire_gauge,
                "resistance_ohm_per_km": resistance_per_km,
                "system_voltage_V": sys_voltage,
            },
            calculated_value=round(voltage_drop_percent, 2),
            unit="%",
            threshold_max=max_drop_percent,
            is_valid=is_valid,
            formula="Vdrop% = (2 × I × L × R / V) × 100",
            reference="NEC 690.8(B)",
            notes=(
                f"Voltage drop: {voltage_drop_v:.2f}V ({voltage_drop_percent:.2f}%) "
                f"for {distance:.1f}m run at {current:.1f}A. "
                f"{'PASS' if is_valid else 'EXCEEDS'} {max_drop_percent}% limit."
            ),
        )

        self.calculations.append(calc)
        return calc

    def validate_short_circuit(
        self,
        parallel_strings: int,
        string_isc: float,
        max_system_isc: Optional[float] = None,
    ) -> EngineeringCalculation:
        """Validate short circuit current calculations.

        Args:
            parallel_strings: Number of parallel strings
            string_isc: Short circuit current per string (A)
            max_system_isc: Maximum allowable system short circuit current (A)

        Returns:
            Short circuit current calculation result
        """
        # Calculate total system short circuit current
        # Include 125% safety factor per NEC
        system_isc = parallel_strings * string_isc * 1.25

        # Validate against maximum (if provided)
        is_valid = True
        if max_system_isc:
            is_valid = system_isc <= max_system_isc

        calc = EngineeringCalculation(
            calculation_type="short_circuit_current",
            description="System short circuit current with safety factor",
            input_parameters={
                "parallel_strings": parallel_strings,
                "string_isc": string_isc,
                "safety_factor": 1.25,
            },
            calculated_value=round(system_isc, 2),
            unit="A",
            threshold_max=max_system_isc,
            is_valid=is_valid,
            formula="Isc_system = N_strings × Isc_string × 1.25",
            reference="NEC 690.8(A)",
            notes=(
                f"Total system short circuit current: {system_isc:.2f}A. "
                f"Inverter and conductors must be rated for this current."
            ),
        )

        self.calculations.append(calc)
        return calc

    def verify_grounding(
        self,
        system_voltage: Optional[float] = None,
        equipment_current: Optional[float] = None,
    ) -> List[EngineeringCalculation]:
        """Verify grounding conductor sizing.

        Args:
            system_voltage: System voltage (V)
            equipment_current: Maximum fault current (A)

        Returns:
            List of grounding calculation results
        """
        results: List[EngineeringCalculation] = []

        voltage = system_voltage or self.config.system_voltage_vdc
        fault_current = equipment_current or self.config.max_current_isc

        # Equipment grounding conductor (EGC) sizing per NEC 250.122
        # Simplified - based on overcurrent device size
        ocpd_size = math.ceil(fault_current * 1.25)

        # EGC size based on OCPD
        if ocpd_size <= 15:
            egc_size = "14AWG"
        elif ocpd_size <= 20:
            egc_size = "12AWG"
        elif ocpd_size <= 60:
            egc_size = "10AWG"
        elif ocpd_size <= 100:
            egc_size = "8AWG"
        elif ocpd_size <= 200:
            egc_size = "6AWG"
        elif ocpd_size <= 300:
            egc_size = "4AWG"
        elif ocpd_size <= 400:
            egc_size = "3AWG"
        elif ocpd_size <= 600:
            egc_size = "2AWG"
        else:
            egc_size = "1/0AWG"

        results.append(
            EngineeringCalculation(
                calculation_type="equipment_grounding_conductor",
                description="Equipment grounding conductor (EGC) sizing",
                input_parameters={
                    "ocpd_rating": ocpd_size,
                    "system_voltage": voltage,
                },
                calculated_value=0,  # Categorical result
                unit="AWG",
                threshold_min=None,
                is_valid=True,
                formula="Per NEC Table 250.122",
                reference="NEC 250.122",
                notes=(
                    f"Minimum EGC size: {egc_size} for {ocpd_size}A OCPD. "
                    f"EGC runs with circuit conductors to equipment."
                ),
            )
        )

        # Grounding electrode conductor (GEC) for system grounding
        results.append(
            EngineeringCalculation(
                calculation_type="grounding_electrode_conductor",
                description="Grounding electrode conductor (GEC) sizing",
                input_parameters={
                    "system_type": "ungrounded",
                },
                calculated_value=0,
                unit="AWG",
                threshold_min=None,
                is_valid=True,
                formula="Per NEC 250.166",
                reference="NEC 250.166",
                notes=(
                    "For ungrounded PV systems, GEC per NEC 250.166. "
                    "Typically 6AWG minimum for equipment grounding."
                ),
            )
        )

        self.calculations.extend(results)
        return results

    def confirm_overcurrent_protection(
        self,
        continuous_current: float,
        max_fault_current: Optional[float] = None,
    ) -> EngineeringCalculation:
        """Confirm overcurrent protection device (OCPD) sizing.

        Args:
            continuous_current: Continuous operating current (A)
            max_fault_current: Maximum fault current for interrupting rating (A)

        Returns:
            OCPD sizing calculation result
        """
        # Per NEC 690.8, OCPD must be rated at least 125% of continuous current
        min_ocpd_rating = continuous_current * 1.25

        # Select next standard OCPD size
        selected_ocpd = None
        for size in self.STANDARD_OCPD_SIZES:
            if size >= min_ocpd_rating:
                selected_ocpd = size
                break

        if selected_ocpd is None:
            selected_ocpd = self.STANDARD_OCPD_SIZES[-1]
            is_valid = False
        else:
            is_valid = True

        calc = EngineeringCalculation(
            calculation_type="overcurrent_protection",
            description="Overcurrent protection device (OCPD) selection",
            input_parameters={
                "continuous_current": continuous_current,
                "minimum_rating": min_ocpd_rating,
            },
            calculated_value=selected_ocpd,
            unit="A",
            threshold_min=min_ocpd_rating,
            is_valid=is_valid,
            formula="OCPD_rating ≥ Icontinuous × 1.25",
            reference="NEC 690.8(B)",
            notes=(
                f"Required OCPD rating: {min_ocpd_rating:.1f}A. "
                f"Selected standard size: {selected_ocpd}A. "
                f"Must be DC-rated with appropriate voltage rating."
            ),
        )

        self.calculations.append(calc)
        return calc

    def calculate_wire_sizing(
        self,
        current: float,
        length: float,
        max_voltage_drop_percent: float = 2.0,
        temperature_correction: float = 1.0,
    ) -> EngineeringCalculation:
        """Calculate minimum wire size based on ampacity and voltage drop.

        Args:
            current: Conductor current (A)
            length: One-way conductor length (m)
            max_voltage_drop_percent: Maximum allowable voltage drop (%)
            temperature_correction: Temperature correction factor

        Returns:
            Wire sizing calculation result
        """
        # Apply NEC 125% continuous current factor
        design_current = current * 1.25

        # Apply temperature correction
        required_ampacity = design_current / temperature_correction

        # Find minimum wire size based on ampacity
        ampacity_wire = None
        for gauge, ampacity in self.WIRE_AMPACITY_TABLE.items():
            if ampacity >= required_ampacity:
                ampacity_wire = gauge
                break

        # Find minimum wire size based on voltage drop
        voltage_drop_wire = None
        for gauge in self.WIRE_RESISTANCE_TABLE.keys():
            drop_calc = self.check_voltage_drop(
                current=current,
                distance=length,
                wire_gauge=gauge,
                max_drop_percent=max_voltage_drop_percent,
            )
            if drop_calc.is_valid:
                voltage_drop_wire = gauge
                break

        # Select the larger of the two wire sizes
        wire_sizes = list(self.WIRE_AMPACITY_TABLE.keys())
        ampacity_idx = wire_sizes.index(ampacity_wire) if ampacity_wire else 0
        vdrop_idx = wire_sizes.index(voltage_drop_wire) if voltage_drop_wire else 0
        selected_idx = max(ampacity_idx, vdrop_idx)
        selected_wire = wire_sizes[selected_idx]

        calc = EngineeringCalculation(
            calculation_type="wire_sizing",
            description="Conductor sizing based on ampacity and voltage drop",
            input_parameters={
                "current": current,
                "design_current": design_current,
                "length": length,
                "temp_correction": temperature_correction,
            },
            calculated_value=0,  # Categorical result
            unit="AWG",
            threshold_min=None,
            is_valid=True,
            formula="Size based on max(ampacity requirement, voltage drop requirement)",
            reference="NEC 690.8, 310.15",
            notes=(
                f"Selected wire size: {selected_wire}. "
                f"Based on {design_current:.1f}A design current "
                f"and {max_voltage_drop_percent}% voltage drop limit."
            ),
        )

        self.calculations.append(calc)
        return calc

    def get_all_calculations(self) -> List[EngineeringCalculation]:
        """Get all calculation results.

        Returns:
            Complete list of all calculation results
        """
        return self.calculations

    def get_failed_calculations(self) -> List[EngineeringCalculation]:
        """Get all failed calculations.

        Returns:
            List of calculations that failed validation
        """
        return [c for c in self.calculations if not c.is_valid]

    def generate_calculation_summary(self) -> Dict[str, int]:
        """Generate summary statistics for calculations.

        Returns:
            Dictionary with counts of calculations by type and status
        """
        summary = {
            "total": len(self.calculations),
            "valid": sum(1 for c in self.calculations if c.is_valid),
            "invalid": sum(1 for c in self.calculations if not c.is_valid),
        }

        # Count by type
        calc_types = set(c.calculation_type for c in self.calculations)
        for calc_type in calc_types:
            summary[f"type_{calc_type}"] = sum(
                1 for c in self.calculations if c.calculation_type == calc_type
            )

        return summary
