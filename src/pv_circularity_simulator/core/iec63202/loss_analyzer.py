"""
CTM Power Loss Analysis for IEC 63202 testing.

This module provides detailed analysis of Cell-to-Module power losses including
optical losses (reflection, absorption, shading), electrical losses (series
resistance, mismatch), thermal effects, spatial non-uniformity, and spectral
mismatch factors.
"""

from typing import Dict, Tuple, Optional
import logging

import numpy as np
from scipy import integrate

from pv_circularity_simulator.core.iec63202.models import (
    CellProperties,
    ModuleConfiguration,
    CTMLossComponents,
    FlashSimulatorData,
)
from pv_circularity_simulator.core.utils.constants import (
    AM15_SPECTRUM,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    ELEMENTARY_CHARGE,
)

logger = logging.getLogger(__name__)


class CTMPowerLossAnalyzer:
    """
    Comprehensive CTM power loss analyzer.

    This class analyzes and quantifies individual CTM loss mechanisms:
    - Optical losses: reflection, absorption, grid shading
    - Electrical losses: series resistance, cell mismatch
    - Thermal losses: temperature effects during assembly
    - Spatial non-uniformity: irradiance distribution
    - Spectral mismatch: simulator vs. solar spectrum

    The analyzer integrates with the B03 CTM model (k1-k24 factors) to provide
    comprehensive loss budgeting and validation.
    """

    def __init__(self) -> None:
        """Initialize CTM power loss analyzer."""
        self.am15_spectrum = AM15_SPECTRUM
        logger.info("CTM Power Loss Analyzer initialized")

    def optical_losses(
        self,
        glass_transmission: float = 0.96,
        encapsulant_absorption: float = 0.015,
        grid_coverage_ratio: float = 0.025,
        num_busbars: int = 3
    ) -> Dict[str, float]:
        """
        Calculate optical losses including reflection, absorption, and grid shading.

        Optical losses occur due to:
        1. Front glass reflection (reduced by AR coating)
        2. Encapsulant absorption (EVA, POE degradation)
        3. Metallization grid shading (fingers and busbars)

        Args:
            glass_transmission: Front glass transmission coefficient (0-1)
            encapsulant_absorption: Encapsulant absorption coefficient (0-1)
            grid_coverage_ratio: Fraction of cell area covered by fingers
            num_busbars: Number of busbars (affects shading)

        Returns:
            Dictionary with individual optical loss components (%)

        Example:
            >>> analyzer = CTMPowerLossAnalyzer()
            >>> losses = analyzer.optical_losses(
            ...     glass_transmission=0.96,
            ...     encapsulant_absorption=0.015,
            ...     grid_coverage_ratio=0.025,
            ...     num_busbars=5
            ... )
            >>> print(f"Reflection: {losses['reflection']:.2f}%")
            >>> print(f"Total optical: {losses['total']:.2f}%")
        """
        # Front glass reflection loss
        reflection_loss = (1.0 - glass_transmission) * 100

        # Encapsulant absorption loss (EVA/POE)
        absorption_loss = encapsulant_absorption * 100

        # Grid shading loss (fingers + busbars)
        # Busbar width typically 1.5-2mm, more busbars = more shading
        busbar_shading = num_busbars * 0.002  # ~0.2% per busbar
        finger_shading = grid_coverage_ratio * 100
        shading_loss = (busbar_shading + grid_coverage_ratio) * 100

        total_optical_loss = reflection_loss + absorption_loss + shading_loss

        losses = {
            "reflection": reflection_loss,
            "absorption": absorption_loss,
            "grid_shading": shading_loss,
            "busbar_shading": busbar_shading * 100,
            "finger_shading": finger_shading,
            "total": total_optical_loss,
        }

        logger.info(
            f"Optical losses: Reflection={reflection_loss:.2f}%, "
            f"Absorption={absorption_loss:.2f}%, "
            f"Shading={shading_loss:.2f}%, "
            f"Total={total_optical_loss:.2f}%"
        )

        return losses

    def electrical_losses(
        self,
        cell_properties: CellProperties,
        module_config: ModuleConfiguration,
        ribbon_resistivity: float = 1.8e-8,  # Ohm·m (copper)
        ribbon_length: float = 0.16,  # m
        ribbon_cross_section: float = 2.0e-7,  # m² (0.2mm × 1mm)
        solder_resistance: float = 0.001,  # Ohm per joint
    ) -> Dict[str, float]:
        """
        Calculate electrical losses from series resistance and cell mismatch.

        Electrical losses include:
        1. Ribbon resistance (fingers, bus ribbons)
        2. Busbar resistance
        3. Solder joint resistance
        4. Cell-to-cell mismatch losses

        Args:
            cell_properties: Cell electrical properties
            module_config: Module configuration
            ribbon_resistivity: Ribbon material resistivity (Ohm·m)
            ribbon_length: Average ribbon length per cell (m)
            ribbon_cross_section: Ribbon cross-sectional area (m²)
            solder_resistance: Resistance per solder joint (Ohm)

        Returns:
            Dictionary with individual electrical loss components (%)

        Example:
            >>> losses = analyzer.electrical_losses(
            ...     cell_properties=cell_props,
            ...     module_config=module_config,
            ...     ribbon_resistivity=1.8e-8
            ... )
            >>> print(f"Series resistance: {losses['series_resistance']:.2f}%")
        """
        # Calculate ribbon resistance
        r_ribbon = (ribbon_resistivity * ribbon_length) / ribbon_cross_section

        # Solder joint resistance (2 joints per cell)
        r_solder = 2 * solder_resistance

        # Total series resistance per cell
        r_series_total = r_ribbon + r_solder

        # Calculate power loss due to series resistance
        # P_loss = I²R, relative to P_max
        i_mp = cell_properties.imp
        series_resistance_loss = (
            (i_mp**2 * r_series_total * module_config.num_cells_series) /
            (cell_properties.pmax * module_config.total_cells)
        ) * 100

        # Cell mismatch loss (typically 0.5-2% depending on binning)
        # Estimated from fill factor variation
        mismatch_loss = 1.0  # Default 1% for standard binning

        total_electrical_loss = series_resistance_loss + mismatch_loss

        losses = {
            "series_resistance": series_resistance_loss,
            "ribbon_resistance": (i_mp**2 * r_ribbon / cell_properties.pmax) * 100,
            "solder_resistance": (i_mp**2 * r_solder / cell_properties.pmax) * 100,
            "mismatch": mismatch_loss,
            "total": total_electrical_loss,
        }

        logger.info(
            f"Electrical losses: Series R={series_resistance_loss:.2f}%, "
            f"Mismatch={mismatch_loss:.2f}%, "
            f"Total={total_electrical_loss:.2f}%"
        )

        return losses

    def thermal_losses(
        self,
        cell_properties: CellProperties,
        lamination_temp: float = 145.0,  # °C (typical EVA lamination)
        cooldown_rate: float = 2.0,  # °C/min
        assembly_time: float = 15.0,  # minutes
    ) -> Dict[str, float]:
        """
        Calculate thermal losses due to temperature coefficient effects during assembly.

        Thermal stresses during lamination and assembly can cause:
        1. Temporary power degradation during high-temperature exposure
        2. Thermal stress-induced microcracks
        3. Solder joint degradation

        Args:
            cell_properties: Cell properties including temp coefficients
            lamination_temp: Peak lamination temperature (°C)
            cooldown_rate: Cooling rate after lamination (°C/min)
            assembly_time: Total assembly time at elevated temperature (min)

        Returns:
            Dictionary with thermal loss components (%)

        Example:
            >>> losses = analyzer.thermal_losses(
            ...     cell_properties=cell_props,
            ...     lamination_temp=150.0
            ... )
            >>> print(f"Thermal assembly loss: {losses['total']:.2f}%")
        """
        from pv_circularity_simulator.core.utils.constants import STC_TEMPERATURE

        # Temperature difference from STC
        temp_diff = lamination_temp - STC_TEMPERATURE

        # Temporary power loss during lamination
        temp_coeff_pmax = cell_properties.temperature_coefficient_pmax
        instantaneous_loss = abs(temp_coeff_pmax * temp_diff)

        # Time-weighted thermal exposure (assume linear cooldown)
        avg_temp_elevation = temp_diff / 2  # Average during cooldown
        exposure_factor = assembly_time / 60  # Convert to hours

        # Permanent thermal stress loss (empirical)
        thermal_stress_loss = 0.2  # ~0.2% typical for standard process

        total_thermal_loss = thermal_stress_loss

        losses = {
            "thermal_stress": thermal_stress_loss,
            "instantaneous_effect": instantaneous_loss,
            "time_weighted_exposure": avg_temp_elevation * exposure_factor * 0.01,
            "total": total_thermal_loss,
        }

        logger.info(
            f"Thermal losses: Stress={thermal_stress_loss:.2f}%, "
            f"Total={total_thermal_loss:.2f}%"
        )

        return losses

    def spatial_non_uniformity(
        self,
        flash_simulator: FlashSimulatorData,
        module_area: float = 1.6,  # m² (typical module)
    ) -> float:
        """
        Calculate losses due to irradiance non-uniformity across module.

        Flash simulators have spatial non-uniformity in irradiance distribution
        across the test plane. IEC 60904-9 Class A requires ≤2% non-uniformity.
        Higher non-uniformity causes measurement errors.

        Args:
            flash_simulator: Flash simulator specifications
            module_area: Module active area (m²)

        Returns:
            Spatial non-uniformity loss (%)

        Example:
            >>> loss = analyzer.spatial_non_uniformity(
            ...     flash_simulator=flash_sim_data,
            ...     module_area=1.6
            ... )
            >>> print(f"Spatial uniformity loss: {loss:.2f}%")
        """
        # Get uniformity percentage (e.g., 98% = 2% non-uniformity)
        uniformity_percent = flash_simulator.spatial_uniformity

        # Non-uniformity as deviation from mean
        non_uniformity = 100 - uniformity_percent

        # Loss is approximately equal to non-uniformity
        # For large modules, edge effects can increase losses
        area_factor = 1.0 + (module_area - 1.6) * 0.1  # Scale with area

        spatial_loss = non_uniformity * area_factor

        logger.info(
            f"Spatial non-uniformity: {uniformity_percent:.1f}% uniformity, "
            f"{spatial_loss:.2f}% loss"
        )

        return spatial_loss

    def spectral_mismatch_factor(
        self,
        simulator_spectrum: Dict[float, float],
        cell_spectral_response: Optional[Dict[float, float]] = None
    ) -> float:
        """
        Calculate spectral mismatch correction factor.

        Spectral mismatch occurs when flash simulator spectrum differs from
        AM1.5 reference spectrum. The mismatch factor M is calculated per
        IEC 60904-7 as:

        M = (∫E_ref(λ)·SR(λ)dλ / ∫E_sim(λ)·SR(λ)dλ) ×
            (∫E_sim(λ)dλ / ∫E_ref(λ)dλ)

        Args:
            simulator_spectrum: Flash simulator spectral distribution
                               (wavelength nm: irradiance W/m²/nm)
            cell_spectral_response: Cell spectral response curve
                                   (wavelength nm: response A/W)
                                   If None, uses typical crystalline Si response

        Returns:
            Spectral mismatch correction factor (typically 0.95-1.05)

        Example:
            >>> factor = analyzer.spectral_mismatch_factor(
            ...     simulator_spectrum={300: 0.1, 500: 1.8, 900: 0.7, ...},
            ...     cell_spectral_response={300: 0.2, 500: 0.8, 900: 0.6, ...}
            ... )
            >>> print(f"Spectral mismatch: {(1-factor)*100:.2f}% loss")
        """
        if cell_spectral_response is None:
            # Use typical c-Si spectral response
            cell_spectral_response = self._default_spectral_response()

        # Get common wavelength range
        wavelengths = sorted(set(simulator_spectrum.keys()) & set(self.am15_spectrum.keys()))

        if len(wavelengths) < 5:
            logger.warning(
                "Insufficient spectral data points, using default mismatch factor"
            )
            return 0.98  # Typical value for LED/Xenon simulators

        # Interpolate to common wavelength grid
        wl_grid = np.linspace(min(wavelengths), max(wavelengths), 100)

        # Interpolate spectra and response
        e_ref = np.interp(wl_grid, list(self.am15_spectrum.keys()), list(self.am15_spectrum.values()))
        e_sim = np.interp(wl_grid, list(simulator_spectrum.keys()), list(simulator_spectrum.values()))
        sr = np.interp(wl_grid, list(cell_spectral_response.keys()), list(cell_spectral_response.values()))

        # Calculate integrals
        integral_ref_sr = np.trapz(e_ref * sr, wl_grid)
        integral_sim_sr = np.trapz(e_sim * sr, wl_grid)
        integral_sim = np.trapz(e_sim, wl_grid)
        integral_ref = np.trapz(e_ref, wl_grid)

        # Calculate mismatch factor
        if integral_sim_sr == 0 or integral_ref == 0:
            logger.error("Invalid spectral data, returning default factor")
            return 1.0

        mismatch_factor = (integral_ref_sr / integral_sim_sr) * (integral_sim / integral_ref)

        logger.info(
            f"Spectral mismatch factor: {mismatch_factor:.4f} "
            f"({(1-mismatch_factor)*100:.2f}% {'loss' if mismatch_factor < 1 else 'gain'})"
        )

        return mismatch_factor

    def total_ctm_loss_budget(
        self,
        cell_properties: CellProperties,
        module_config: ModuleConfiguration,
        flash_simulator: Optional[FlashSimulatorData] = None,
        glass_transmission: float = 0.96,
        encapsulant_absorption: float = 0.015,
    ) -> CTMLossComponents:
        """
        Calculate comprehensive CTM loss budget matching k1-k24 factor categories.

        This method integrates all loss mechanisms to provide a complete
        CTM loss breakdown consistent with the B03 model.

        Args:
            cell_properties: Cell electrical and optical properties
            module_config: Module design and configuration
            flash_simulator: Flash simulator characteristics (optional)
            glass_transmission: Front glass transmission coefficient
            encapsulant_absorption: Encapsulant absorption coefficient

        Returns:
            Complete CTM loss component breakdown

        Example:
            >>> loss_budget = analyzer.total_ctm_loss_budget(
            ...     cell_properties=cell_props,
            ...     module_config=module_config
            ... )
            >>> print(f"Total CTM loss: {loss_budget.total_loss:.2f}%")
            >>> print(f"Optical: {loss_budget.total_optical_loss:.2f}%")
            >>> print(f"Electrical: {loss_budget.total_electrical_loss:.2f}%")
        """
        # Calculate optical losses
        optical = self.optical_losses(
            glass_transmission=glass_transmission,
            encapsulant_absorption=encapsulant_absorption,
            grid_coverage_ratio=0.025,
            num_busbars=module_config.bypass_diodes,
        )

        # Calculate electrical losses
        electrical = self.electrical_losses(
            cell_properties=cell_properties,
            module_config=module_config,
        )

        # Calculate thermal losses
        thermal = self.thermal_losses(cell_properties=cell_properties)

        # Calculate spatial non-uniformity if flash simulator provided
        spatial_loss = 0.0
        if flash_simulator:
            spatial_loss = self.spatial_non_uniformity(flash_simulator)

        # Calculate spectral mismatch if flash simulator provided
        spectral_loss = 0.0
        if flash_simulator and flash_simulator.spectral_distribution:
            spectral_factor = self.spectral_mismatch_factor(
                flash_simulator.spectral_distribution
            )
            spectral_loss = (1.0 - spectral_factor) * 100

        # Create comprehensive loss breakdown
        loss_components = CTMLossComponents(
            optical_reflection=optical["reflection"],
            optical_absorption=optical["absorption"],
            optical_shading=optical["grid_shading"],
            electrical_series_resistance=electrical["series_resistance"],
            electrical_mismatch=electrical["mismatch"],
            thermal_assembly=thermal["total"],
            spatial_non_uniformity=spatial_loss,
            spectral_mismatch=spectral_loss,
        )

        logger.info(
            f"Total CTM loss budget: {loss_components.total_loss:.2f}% "
            f"(Optical: {loss_components.total_optical_loss:.2f}%, "
            f"Electrical: {loss_components.total_electrical_loss:.2f}%, "
            f"Thermal: {thermal['total']:.2f}%)"
        )

        return loss_components

    def _default_spectral_response(self) -> Dict[float, float]:
        """
        Generate default crystalline silicon spectral response.

        Returns:
            Spectral response curve (wavelength nm: response A/W)
        """
        # Typical c-Si spectral response (normalized)
        wavelengths = np.array([
            300, 350, 400, 450, 500, 550, 600, 650,
            700, 750, 800, 850, 900, 950, 1000, 1100, 1200
        ])

        # Normalized response (peak around 800-900 nm for c-Si)
        response = np.array([
            0.10, 0.30, 0.50, 0.65, 0.75, 0.82, 0.88, 0.92,
            0.95, 0.97, 0.98, 0.97, 0.94, 0.88, 0.80, 0.60, 0.35
        ])

        return dict(zip(wavelengths, response))

    def visualize_loss_waterfall(
        self,
        loss_components: CTMLossComponents,
        initial_power: float = 100.0
    ) -> Dict[str, float]:
        """
        Create waterfall chart data showing cumulative CTM losses.

        Args:
            loss_components: CTM loss breakdown
            initial_power: Initial power reference (%, default 100%)

        Returns:
            Dictionary with cumulative power at each loss stage

        Example:
            >>> waterfall = analyzer.visualize_loss_waterfall(loss_components)
            >>> for stage, power in waterfall.items():
            ...     print(f"{stage}: {power:.2f}%")
        """
        waterfall = {
            "Initial (Cell Power × N)": initial_power,
            "After Reflection": initial_power - loss_components.optical_reflection,
            "After Absorption": (
                initial_power -
                loss_components.optical_reflection -
                loss_components.optical_absorption
            ),
            "After Shading": (
                initial_power -
                loss_components.total_optical_loss
            ),
            "After Series R": (
                initial_power -
                loss_components.total_optical_loss -
                loss_components.electrical_series_resistance
            ),
            "After Mismatch": (
                initial_power -
                loss_components.total_optical_loss -
                loss_components.total_electrical_loss
            ),
            "After Thermal": (
                initial_power -
                loss_components.total_optical_loss -
                loss_components.total_electrical_loss -
                loss_components.thermal_assembly
            ),
            "Final (Module Power)": initial_power - loss_components.total_loss,
        }

        return waterfall
