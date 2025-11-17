"""
Electrical Shading Model with bypass diode simulation and mismatch analysis.

This module provides detailed electrical modeling of shading effects including
bypass diode activation, current mismatch, and hotspot risk assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    ElectricalShadeResult,
    ModuleElectricalParams,
    SunPosition,
)

logger = logging.getLogger(__name__)


class ElectricalShadingModel:
    """
    Electrical modeling of shading effects on PV modules.

    Simulates bypass diode behavior, substring shading, current mismatch,
    and hotspot risks under partial shading conditions.
    """

    def __init__(self, module_params: ModuleElectricalParams):
        """
        Initialize the electrical shading model.

        Args:
            module_params: Module electrical parameters
        """
        self.module_params = module_params

        # Calculate cells per substring
        self.cells_per_substring = module_params.cells_per_diode

        logger.info(
            f"Initialized ElectricalShadingModel: "
            f"{module_params.cells_in_series} cells, "
            f"{module_params.bypass_diodes} bypass diodes"
        )

    def bypass_diode_simulation(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: Optional[List[float]] = None,
        temperature: float = 25.0
    ) -> ElectricalShadeResult:
        """
        Simulate bypass diode activation under partial shading.

        Args:
            shaded_cells: List of cell indices that are shaded
            irradiance_on_cells: Irradiance on each cell (W/m²). If None, assumes
                                shaded cells get 0 and unshaded get 1000 W/m²
            temperature: Cell temperature in °C

        Returns:
            ElectricalShadeResult with bypass diode simulation results
        """
        from datetime import datetime

        # Generate default irradiance if not provided
        if irradiance_on_cells is None:
            irradiance_on_cells = [
                0.0 if i in shaded_cells else 1000.0
                for i in range(self.module_params.cells_in_series)
            ]

        # Determine which substrings are affected
        active_bypass_diodes = self._determine_active_diodes(
            shaded_cells, irradiance_on_cells
        )

        # Calculate voltage loss from bypassed substrings
        voltage_loss = len(active_bypass_diodes) * (
            self.module_params.v_mp / self.module_params.bypass_diodes
        )

        # Calculate power loss
        power_loss_fraction = self._calculate_power_loss(
            shaded_cells,
            irradiance_on_cells,
            active_bypass_diodes
        )

        # Calculate current mismatch
        current_mismatch = self._calculate_current_mismatch(irradiance_on_cells)

        # Check for hotspot risk
        hotspot_risk, hotspot_cells = self._assess_hotspot_risk(
            shaded_cells,
            irradiance_on_cells,
            active_bypass_diodes
        )

        result = ElectricalShadeResult(
            timestamp=datetime.now(),
            module_id=0,
            shaded_cells=shaded_cells,
            active_bypass_diodes=active_bypass_diodes,
            voltage_loss=voltage_loss,
            power_loss=power_loss_fraction,
            current_mismatch_loss=current_mismatch,
            hotspot_risk=hotspot_risk,
            hotspot_cells=hotspot_cells
        )

        logger.debug(
            f"Bypass diode simulation: {len(active_bypass_diodes)} diodes active, "
            f"power loss: {power_loss_fraction:.2%}"
        )

        return result

    def substring_shading(
        self,
        substring_index: int,
        shading_fraction: float
    ) -> Dict[str, float]:
        """
        Calculate impact of shading on a specific substring.

        Args:
            substring_index: Index of substring (0 to num_bypass_diodes-1)
            shading_fraction: Fraction of substring that is shaded (0-1)

        Returns:
            Dictionary with substring performance metrics
        """
        if not 0 <= substring_index < self.module_params.bypass_diodes:
            raise ValueError(
                f"Substring index must be 0 to {self.module_params.bypass_diodes - 1}"
            )

        # Calculate current reduction in shaded substring
        # Assumes current is proportional to lowest cell irradiance
        current_reduction = shading_fraction

        # If shading is severe enough, bypass diode will activate
        bypass_threshold = 0.3  # Activate if more than 30% shaded

        if shading_fraction >= bypass_threshold:
            # Bypass diode active - substring produces no power
            power_output = 0.0
            voltage_output = 0.0
            current_output = 0.0
            bypass_active = True
        else:
            # Bypass diode inactive - reduced performance
            current_output = 1.0 - current_reduction
            voltage_output = 1.0  # Voltage roughly maintained
            power_output = current_output * voltage_output
            bypass_active = False

        return {
            "power_fraction": power_output,
            "voltage_fraction": voltage_output,
            "current_fraction": current_output,
            "bypass_active": bypass_active,
            "shading_fraction": shading_fraction
        }

    def mismatch_losses(
        self,
        modules_irradiances: List[float]
    ) -> float:
        """
        Calculate current mismatch losses in parallel strings.

        Args:
            modules_irradiances: List of irradiance values for each module in string (W/m²)

        Returns:
            Mismatch loss fraction (0-1)
        """
        if not modules_irradiances:
            return 0.0

        # Current is limited by module with lowest irradiance
        min_irradiance = min(modules_irradiances)
        avg_irradiance = np.mean(modules_irradiances)

        if avg_irradiance == 0:
            return 0.0

        # Mismatch loss is the difference between average and minimum
        mismatch_loss = (avg_irradiance - min_irradiance) / avg_irradiance

        logger.debug(
            f"Current mismatch loss: {mismatch_loss:.2%} "
            f"(min: {min_irradiance:.0f}, avg: {avg_irradiance:.0f} W/m²)"
        )

        return mismatch_loss

    def module_iv_under_shade(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float],
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate module I-V curve under partial shading.

        Args:
            shaded_cells: List of shaded cell indices
            irradiance_on_cells: Irradiance on each cell (W/m²)
            num_points: Number of points in I-V curve

        Returns:
            Tuple of (voltage_array, current_array) for I-V curve
        """
        # Determine which bypass diodes are active
        active_diodes = self._determine_active_diodes(shaded_cells, irradiance_on_cells)

        # Number of active substrings (those not bypassed)
        active_substrings = self.module_params.bypass_diodes - len(active_diodes)

        if active_substrings == 0:
            # All substrings bypassed - no output
            return np.zeros(num_points), np.zeros(num_points)

        # Calculate effective voltage and current
        # Voltage is reduced proportionally to bypassed substrings
        v_oc_effective = self.module_params.v_oc * active_substrings / self.module_params.bypass_diodes

        # Current limited by most shaded active substring
        active_irradiances = []
        for diode_idx in range(self.module_params.bypass_diodes):
            if diode_idx not in active_diodes:
                # This substring is active
                substring_cells = self._get_substring_cells(diode_idx)
                substring_irr = [irradiance_on_cells[i] for i in substring_cells]
                # Current limited by weakest cell
                active_irradiances.append(min(substring_irr))

        if active_irradiances:
            min_active_irr = min(active_irradiances)
            i_sc_effective = self.module_params.i_sc * min_active_irr / 1000.0
        else:
            i_sc_effective = 0.0

        # Generate I-V curve using single-diode model (simplified)
        voltage = np.linspace(0, v_oc_effective, num_points)
        current = i_sc_effective * (1 - np.exp((voltage / v_oc_effective - 1) / 0.08))

        return voltage, current

    def mppt_behavior_under_shade(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float]
    ) -> Dict[str, float]:
        """
        Simulate MPPT tracking behavior under non-uniform irradiance.

        Args:
            shaded_cells: List of shaded cell indices
            irradiance_on_cells: Irradiance on each cell (W/m²)

        Returns:
            Dictionary with MPPT tracking results
        """
        # Generate I-V curve
        voltage, current = self.module_iv_under_shade(
            shaded_cells, irradiance_on_cells, num_points=200
        )

        # Calculate power curve
        power = voltage * current

        # Find maximum power point
        max_power_idx = np.argmax(power)
        v_mpp = voltage[max_power_idx]
        i_mpp = current[max_power_idx]
        p_mpp = power[max_power_idx]

        # Calculate ideal power (no shading)
        avg_irradiance = np.mean(irradiance_on_cells)
        p_ideal = self.module_params.p_max * avg_irradiance / 1000.0

        # MPPT efficiency
        mppt_efficiency = p_mpp / p_ideal if p_ideal > 0 else 0.0

        # Check for multiple local maxima (can confuse MPPT)
        local_maxima = self._find_local_maxima(power)
        has_multiple_peaks = len(local_maxima) > 1

        return {
            "v_mpp": v_mpp,
            "i_mpp": i_mpp,
            "p_mpp": p_mpp,
            "p_ideal": p_ideal,
            "mppt_efficiency": mppt_efficiency,
            "multiple_peaks": has_multiple_peaks,
            "num_peaks": len(local_maxima)
        }

    def hotspot_risk_analysis(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float]
    ) -> Dict[str, any]:
        """
        Analyze hotspot risk for shaded cells.

        Args:
            shaded_cells: List of shaded cell indices
            irradiance_on_cells: Irradiance on each cell (W/m²)

        Returns:
            Dictionary with hotspot risk analysis
        """
        # Determine active bypass diodes
        active_diodes = self._determine_active_diodes(shaded_cells, irradiance_on_cells)

        # Check for hotspot conditions
        hotspot_risk, hotspot_cells = self._assess_hotspot_risk(
            shaded_cells, irradiance_on_cells, active_diodes
        )

        # Calculate power dissipation in shaded cells
        cell_power_dissipation = []

        for cell_idx in shaded_cells:
            # Find which substring this cell belongs to
            substring_idx = cell_idx // self.cells_per_substring

            if substring_idx in active_diodes:
                # Bypass diode active - minimal dissipation
                dissipation = 0.0
            else:
                # Cell forced to conduct string current despite low irradiance
                # This causes reverse bias and power dissipation
                cell_irr = irradiance_on_cells[cell_idx]
                substring_cells = self._get_substring_cells(substring_idx)
                avg_substring_irr = np.mean([irradiance_on_cells[i] for i in substring_cells])

                # Estimate dissipation (simplified)
                if avg_substring_irr > 0:
                    irr_ratio = cell_irr / avg_substring_irr
                    dissipation = (1 - irr_ratio) * (self.module_params.p_max / self.module_params.cells_in_series)
                else:
                    dissipation = 0.0

            cell_power_dissipation.append({
                "cell_index": cell_idx,
                "dissipation_watts": dissipation
            })

        return {
            "hotspot_risk": hotspot_risk,
            "hotspot_cells": hotspot_cells,
            "cell_power_dissipation": cell_power_dissipation,
            "max_dissipation": max([c["dissipation_watts"] for c in cell_power_dissipation]) if cell_power_dissipation else 0.0
        }

    # Private helper methods

    def _determine_active_diodes(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float]
    ) -> List[int]:
        """Determine which bypass diodes are active."""
        active_diodes = []

        for diode_idx in range(self.module_params.bypass_diodes):
            # Get cells in this substring
            substring_cells = self._get_substring_cells(diode_idx)

            # Check if any cells in this substring are significantly shaded
            substring_irradiances = [irradiance_on_cells[i] for i in substring_cells]
            min_irr = min(substring_irradiances)
            avg_irr = np.mean(substring_irradiances)

            # Bypass diode activates if minimum irradiance is much less than average
            # or if irradiance drops below threshold
            bypass_threshold_irr = 300.0  # W/m²
            bypass_threshold_ratio = 0.5

            if min_irr < bypass_threshold_irr or (avg_irr > 0 and min_irr / avg_irr < bypass_threshold_ratio):
                active_diodes.append(diode_idx)

        return active_diodes

    def _get_substring_cells(self, substring_index: int) -> List[int]:
        """Get list of cell indices in a substring."""
        start_cell = substring_index * self.cells_per_substring
        end_cell = min(start_cell + self.cells_per_substring, self.module_params.cells_in_series)

        return list(range(start_cell, end_cell))

    def _calculate_power_loss(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float],
        active_bypass_diodes: List[int]
    ) -> float:
        """Calculate power loss from shading and bypass diode activation."""
        # Power loss from bypassed substrings
        bypassed_fraction = len(active_bypass_diodes) / self.module_params.bypass_diodes

        # Additional loss from current limiting in active substrings
        active_substrings_irr = []

        for diode_idx in range(self.module_params.bypass_diodes):
            if diode_idx not in active_bypass_diodes:
                substring_cells = self._get_substring_cells(diode_idx)
                substring_irr = [irradiance_on_cells[i] for i in substring_cells]
                active_substrings_irr.extend(substring_irr)

        if active_substrings_irr:
            avg_active_irr = np.mean(active_substrings_irr)
            min_active_irr = min(active_substrings_irr)
            current_limit_loss = (avg_active_irr - min_active_irr) / avg_active_irr if avg_active_irr > 0 else 0.0
        else:
            current_limit_loss = 0.0

        # Total power loss
        total_power_loss = bypassed_fraction + (1 - bypassed_fraction) * current_limit_loss

        return min(1.0, total_power_loss)

    def _calculate_current_mismatch(self, irradiance_on_cells: List[float]) -> float:
        """Calculate current mismatch across substrings."""
        substring_currents = []

        for diode_idx in range(self.module_params.bypass_diodes):
            substring_cells = self._get_substring_cells(diode_idx)
            substring_irr = [irradiance_on_cells[i] for i in substring_cells]

            # Current limited by weakest cell in substring
            min_irr = min(substring_irr)
            substring_currents.append(min_irr)

        if not substring_currents:
            return 0.0

        avg_current = np.mean(substring_currents)
        min_current = min(substring_currents)

        if avg_current == 0:
            return 0.0

        mismatch = (avg_current - min_current) / avg_current

        return mismatch

    def _assess_hotspot_risk(
        self,
        shaded_cells: List[int],
        irradiance_on_cells: List[float],
        active_bypass_diodes: List[int]
    ) -> Tuple[bool, List[int]]:
        """Assess hotspot risk for shaded cells."""
        hotspot_cells = []

        for cell_idx in shaded_cells:
            substring_idx = cell_idx // self.cells_per_substring

            # Hotspot risk exists if:
            # 1. Cell is shaded
            # 2. Bypass diode is NOT active (cell is forced to conduct)
            # 3. Irradiance on cell is very low

            if substring_idx not in active_bypass_diodes:
                cell_irr = irradiance_on_cells[cell_idx]
                if cell_irr < 200.0:  # Very low irradiance
                    hotspot_cells.append(cell_idx)

        hotspot_risk = len(hotspot_cells) > 0

        return hotspot_risk, hotspot_cells

    def _find_local_maxima(self, power_curve: np.ndarray) -> List[int]:
        """Find local maxima in power curve."""
        local_maxima = []

        for i in range(1, len(power_curve) - 1):
            if power_curve[i] > power_curve[i - 1] and power_curve[i] > power_curve[i + 1]:
                # Check if this is a significant peak (not just noise)
                if power_curve[i] > 0.01 * np.max(power_curve):
                    local_maxima.append(i)

        return local_maxima
