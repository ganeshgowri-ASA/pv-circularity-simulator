"""
Simplified device physics engine for PV cell simulation.

This module provides fast analytical/semi-analytical models for
quick simulation and real-time UI feedback.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_models.cell_architecture import CellArchitecture
from core.data_models.materials import MaterialDatabase
from core.constants import Q, K_B, STC_TEMPERATURE, STC_IRRADIANCE


class DevicePhysicsEngine:
    """
    Simplified physics engine for quick PV cell simulation.

    Uses analytical and semi-analytical models for fast computation.
    """

    def __init__(self):
        """Initialize device physics engine."""
        pass

    def simulate_jv_curve(
        self,
        architecture: CellArchitecture,
        material_db: MaterialDatabase,
        temperature: float = STC_TEMPERATURE,
        illumination: float = STC_IRRADIANCE,
        voltage_points: int = 201
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate JV curve using two-diode model.

        Args:
            architecture: Cell architecture
            material_db: Materials database
            temperature: Temperature (K)
            illumination: Illumination (W/m²)
            voltage_points: Number of voltage points

        Returns:
            Tuple of (voltage, current_density) arrays
        """
        # Estimate parameters from architecture
        params = self._estimate_parameters(
            architecture,
            material_db,
            temperature
        )

        # Scale photocurrent with illumination
        jph = params['jph'] * (illumination / STC_IRRADIANCE)

        # Voltage range
        voltage = np.linspace(-0.1, params['voc'] * 1.2, voltage_points)

        # Two-diode equation
        vt = K_B * temperature / Q  # Thermal voltage

        # Series resistance effect
        v_internal = voltage - params['rs'] * jph / 1000  # Convert mA to A

        # Diode currents
        j_d1 = params['j01'] * (np.exp(v_internal / (params['n1'] * vt)) - 1)
        j_d2 = params['j02'] * (np.exp(v_internal / (params['n2'] * vt)) - 1)

        # Shunt current
        j_sh = v_internal / params['rsh']

        # Total current
        current_density = jph - j_d1 - j_d2 - j_sh

        return voltage, current_density

    def estimate_efficiency(
        self,
        architecture: CellArchitecture,
        material_db: MaterialDatabase,
        temperature: float = STC_TEMPERATURE,
        illumination: float = STC_IRRADIANCE
    ) -> Dict[str, float]:
        """
        Estimate cell efficiency and performance metrics.

        Args:
            architecture: Cell architecture
            material_db: Materials database
            temperature: Temperature (K)
            illumination: Illumination (W/m²)

        Returns:
            Dictionary with performance metrics
        """
        voltage, current = self.simulate_jv_curve(
            architecture,
            material_db,
            temperature,
            illumination
        )

        # Calculate metrics
        # Voc (J = 0)
        voc = float(np.interp(0, current[::-1], voltage[::-1]))

        # Jsc (V = 0)
        jsc = abs(float(np.interp(0, voltage, current)))

        # Maximum power point
        power = voltage * np.abs(current)
        mpp_idx = np.argmax(power)
        vmpp = float(voltage[mpp_idx])
        jmpp = abs(float(current[mpp_idx]))
        pmpp = float(power[mpp_idx])

        # Fill factor
        ff = (pmpp / (voc * jsc)) * 100 if (voc > 0 and jsc > 0) else 0

        # Efficiency (Pin = 100 mW/cm² at STC)
        pin = illumination / 10.0  # W/m² to mW/cm²
        efficiency = (pmpp / pin) * 100

        return {
            'voc': voc,
            'jsc': jsc,
            'ff': ff,
            'efficiency': efficiency,
            'vmpp': vmpp,
            'jmpp': jmpp,
            'pmpp': pmpp
        }

    def simulate_quantum_efficiency(
        self,
        architecture: CellArchitecture,
        material_db: MaterialDatabase,
        wavelength_range: Tuple[float, float] = (300, 1200),
        wavelength_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate external and internal quantum efficiency.

        Args:
            architecture: Cell architecture
            material_db: Materials database
            wavelength_range: (min_wl, max_wl) in nm
            wavelength_points: Number of wavelength points

        Returns:
            Tuple of (wavelength, EQE, IQE) arrays
        """
        wavelength = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            wavelength_points
        )

        # Simplified QE model
        # This should be replaced with proper optical modeling
        eqe = self._calculate_eqe(
            wavelength,
            architecture,
            material_db
        )

        # IQE = EQE / (1 - R) where R is reflection
        reflection = self._calculate_reflection(
            wavelength,
            architecture,
            material_db
        )

        iqe = np.divide(
            eqe,
            1 - reflection,
            out=np.zeros_like(eqe),
            where=(1 - reflection) > 0
        )

        return wavelength, eqe, iqe

    def _estimate_parameters(
        self,
        architecture: CellArchitecture,
        material_db: MaterialDatabase,
        temperature: float
    ) -> Dict[str, float]:
        """
        Estimate diode model parameters from architecture.

        This is a simplified estimation - real values would come from
        detailed simulation or measurement.
        """
        # Default parameters for c-Si cell
        params = {
            'jph': 40.0,      # Photocurrent (mA/cm²)
            'j01': 1e-12,     # Saturation current 1 (mA/cm²)
            'j02': 1e-8,      # Saturation current 2 (mA/cm²)
            'n1': 1.0,        # Ideality factor 1
            'n2': 2.0,        # Ideality factor 2
            'rs': 1.0,        # Series resistance (Ohm·cm²)
            'rsh': 1000.0,    # Shunt resistance (Ohm·cm²)
            'voc': 0.65,      # Estimated Voc (V)
        }

        # Adjust based on architecture type
        arch_type = architecture.architecture_type.upper()

        if 'PERC' in arch_type:
            params['jph'] = 41.0
            params['voc'] = 0.68
            params['j01'] = 5e-13
        elif 'TOPCON' in arch_type or 'TUNNEL' in arch_type:
            params['jph'] = 41.5
            params['voc'] = 0.71
            params['j01'] = 2e-13
            params['rs'] = 0.8
        elif 'HJT' in arch_type or 'HETEROJUNCTION' in arch_type:
            params['jph'] = 41.0
            params['voc'] = 0.75
            params['j01'] = 1e-13
            params['rs'] = 0.7
        elif 'IBC' in arch_type:
            params['jph'] = 42.0
            params['voc'] = 0.72
            params['j01'] = 2e-13
            params['rs'] = 0.6

        # Adjust for temperature
        # Simplified temperature dependence
        dt = temperature - STC_TEMPERATURE
        params['voc'] -= 0.002 * dt  # -2 mV/K
        params['jph'] += 0.0005 * params['jph'] * dt  # +0.05%/K

        return params

    def _calculate_eqe(
        self,
        wavelength: np.ndarray,
        architecture: CellArchitecture,
        material_db: MaterialDatabase
    ) -> np.ndarray:
        """
        Calculate external quantum efficiency.

        Simplified model - real implementation needs optical modeling.
        """
        eqe = np.zeros_like(wavelength)

        for i, wl in enumerate(wavelength):
            # Simple model: peak in visible, drop in UV and IR
            if wl < 400:
                # UV: reduced due to recombination and absorption in front layers
                eqe[i] = 0.5 * (wl - 300) / 100
            elif wl < 900:
                # Visible: high efficiency
                eqe[i] = 0.90
            elif wl < 1100:
                # Near-IR: decreasing due to bandgap
                eqe[i] = 0.90 * (1100 - wl) / 200
            else:
                # Far-IR: below bandgap
                eqe[i] = 0.05 * np.exp(-(wl - 1100) / 50)

        # Apply architecture-specific adjustments
        arch_type = architecture.architecture_type.upper()

        if 'PERC' in arch_type:
            # Better red response
            eqe[wavelength > 900] *= 1.2
        elif 'HJT' in arch_type:
            # Better blue response, slight loss in red
            eqe[wavelength < 600] *= 1.1
            eqe[wavelength > 900] *= 0.95

        return np.clip(eqe, 0, 1)

    def _calculate_reflection(
        self,
        wavelength: np.ndarray,
        architecture: CellArchitecture,
        material_db: MaterialDatabase
    ) -> np.ndarray:
        """
        Calculate reflection losses.

        Simplified model - real implementation needs transfer matrix method.
        """
        # Check for ARC layers
        has_arc = any(
            layer.layer_type == 'arc'
            for layer in architecture.layers
        )

        if has_arc:
            # With ARC: ~5% reflection with wavelength dependence
            reflection = 0.03 + 0.02 * np.cos(2 * np.pi * wavelength / 400)
        else:
            # Without ARC: ~30% reflection
            reflection = 0.25 + 0.05 * np.cos(2 * np.pi * wavelength / 400)

        return np.clip(reflection, 0, 1)


def quick_efficiency_estimate(
    architecture_type: str,
    has_passivation: bool = False,
    has_arc: bool = False
) -> Dict[str, float]:
    """
    Very quick efficiency estimate based on architecture type.

    Args:
        architecture_type: Type of cell architecture
        has_passivation: Whether cell has good passivation
        has_arc: Whether cell has ARC

    Returns:
        Estimated performance metrics
    """
    # Base efficiencies by architecture
    base_eff = {
        'AL-BSF': 18.5,
        'PERC': 21.5,
        'TOPCON': 23.5,
        'HJT': 24.0,
        'IBC': 24.5
    }

    efficiency = base_eff.get(architecture_type.upper(), 20.0)

    # Apply modifiers
    if has_passivation:
        efficiency += 0.5
    if has_arc:
        efficiency += 0.3

    # Estimate other parameters (simplified)
    voc = 0.60 + (efficiency - 18) * 0.01  # Rough correlation
    jsc = 38 + (efficiency - 18) * 0.5
    ff = 75 + (efficiency - 18) * 0.3

    return {
        'efficiency': efficiency,
        'voc': voc,
        'jsc': jsc,
        'ff': ff,
        'vmpp': voc * 0.85,
        'jmpp': jsc * 0.95,
        'pmpp': efficiency * 100 / 100  # mW/cm²
    }
