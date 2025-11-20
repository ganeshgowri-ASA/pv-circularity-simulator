"""
CTM Loss Modeling Engine implementing Fraunhofer ISE SmartCalc methodology.

This module provides comprehensive Cell-to-Module (CTM) loss analysis with k1-k24 factors
covering optical, electrical, coupling, and environmental effects. Supports advanced module
architectures including half-cut, quarter-cut, shingled, IBC, and bifacial modules.

Reference: Fraunhofer ISE SmartCalc, Cell-to-Module.com
"""

from typing import Dict, Optional, List, Tuple, Literal
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go


class ModuleType(str, Enum):
    """Supported module architectures."""
    STANDARD = "standard"
    HALF_CUT = "half_cut"
    QUARTER_CUT = "quarter_cut"
    SHINGLED = "shingled"
    IBC = "ibc"
    BIFACIAL = "bifacial"


class EncapsulantType(str, Enum):
    """Encapsulant material types."""
    STANDARD_EVA = "standard_eva"
    YELLOW_EVA = "yellow_eva"
    FAST_CURE_EVA = "fast_cure_eva"
    POE = "poe"
    SILICONE = "silicone"


class CellParameters(BaseModel):
    """Cell-level electrical and physical parameters."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    # Electrical characteristics at STC
    power_stc: float = Field(..., gt=0, description="Cell power at STC (W)")
    voltage_mpp: float = Field(..., gt=0, description="Voltage at MPP (V)")
    current_mpp: float = Field(..., gt=0, description="Current at MPP (A)")
    voltage_oc: float = Field(..., gt=0, description="Open circuit voltage (V)")
    current_sc: float = Field(..., gt=0, description="Short circuit current (A)")
    efficiency: float = Field(..., gt=0, le=100, description="Cell efficiency (%)")

    # Physical dimensions
    width: float = Field(..., gt=0, description="Cell width (mm)")
    height: float = Field(..., gt=0, description="Cell height (mm)")
    thickness: float = Field(default=180, gt=0, description="Cell thickness (μm)")

    # Optical properties
    front_grid_coverage: float = Field(default=2.5, ge=0, le=100,
                                       description="Front metallization shading (%)")
    inactive_area_fraction: float = Field(default=0.5, ge=0, le=100,
                                          description="Inactive cell area (%)")

    # Temperature characteristics
    temp_coeff_power: float = Field(default=-0.40, description="Power temp coefficient (%/°C)")
    temp_coeff_voltage: float = Field(default=-0.30, description="Voltage temp coefficient (%/°C)")
    temp_coeff_current: float = Field(default=0.05, description="Current temp coefficient (%/°C)")

    # Degradation
    lid_factor: float = Field(default=1.5, ge=0, le=10,
                              description="Light-induced degradation (%)")
    letid_factor: float = Field(default=0.0, ge=0, le=10,
                                description="Light and elevated temp induced degradation (%)")

    # Low irradiance behavior
    low_irradiance_loss: float = Field(default=0.5, ge=0, le=5,
                                       description="Loss at low irradiance (%)")

    # Spectral response
    spectral_mismatch: float = Field(default=0.0, ge=-5, le=5,
                                     description="Spectral mismatch factor (%)")

    @field_validator('power_stc')
    @classmethod
    def validate_power(cls, v: float) -> float:
        """Validate cell power is in reasonable range."""
        if not (0.5 <= v <= 10):
            raise ValueError("Cell power should be between 0.5W and 10W")
        return v


class ModuleParameters(BaseModel):
    """Module-level configuration and design parameters."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    # Module architecture
    module_type: ModuleType = Field(default=ModuleType.STANDARD,
                                    description="Module architecture type")
    cells_in_series: int = Field(..., gt=0, description="Number of cells in series")
    cells_in_parallel: int = Field(default=1, gt=0, description="Number of cell strings in parallel")

    # Layout and dimensions
    cell_gap: float = Field(default=2.0, ge=0, description="Gap between cells (mm)")
    border_width: float = Field(default=10.0, ge=0, description="Module border width (mm)")

    # Front glass
    glass_thickness: float = Field(default=3.2, gt=0, description="Front glass thickness (mm)")
    glass_transmittance: float = Field(default=91.5, gt=0, le=100,
                                       description="Glass transmittance (%)")
    glass_ar_coating: bool = Field(default=True, description="Anti-reflective coating")
    glass_reflection_gain: float = Field(default=1.0, ge=0, le=5,
                                         description="Reflection gain from glass (%)")

    # Encapsulant
    encapsulant_type: EncapsulantType = Field(default=EncapsulantType.STANDARD_EVA,
                                              description="Encapsulant material")
    encapsulant_thickness: float = Field(default=0.45, gt=0,
                                         description="Encapsulant thickness (mm)")
    encapsulant_yellowing: float = Field(default=0.0, ge=0, le=10,
                                         description="UV-induced yellowing loss (%)")
    encapsulant_current_gain: float = Field(default=0.5, ge=0, le=5,
                                            description="Current gain from encapsulant (%)")

    # Backsheet/rear glass
    is_bifacial: bool = Field(default=False, description="Bifacial module")
    rear_glass: bool = Field(default=False, description="Glass-glass construction")
    bifaciality_factor: float = Field(default=0.70, ge=0, le=1.0,
                                      description="Rear/front efficiency ratio")
    rear_optical_gain: float = Field(default=0.0, ge=0, le=5,
                                     description="Rear side optical gain (%)")

    # Electrical connections
    ribbon_width: float = Field(default=1.5, gt=0, description="Interconnect ribbon width (mm)")
    ribbon_thickness: float = Field(default=0.2, gt=0, description="Ribbon thickness (mm)")
    ribbon_resistivity: float = Field(default=1.7e-8, gt=0,
                                      description="Ribbon resistivity (Ω·m)")
    busbar_count: int = Field(default=5, gt=0, description="Number of busbars per cell")
    junction_box_loss: float = Field(default=0.5, ge=0, le=5,
                                     description="Junction box resistive loss (%)")

    # Manufacturing quality
    cell_mismatch: float = Field(default=1.0, ge=0, le=5,
                                 description="Internal cell mismatch loss (%)")
    module_mismatch: float = Field(default=0.5, ge=0, le=3,
                                   description="Module-to-module mismatch (%)")
    manufacturing_damage: float = Field(default=0.5, ge=0, le=5,
                                        description="Manufacturing-induced damage (%)")

    # Environmental conditions (for k21-k24)
    operating_temperature: float = Field(default=25.0, description="Operating temperature (°C)")
    irradiance: float = Field(default=1000.0, gt=0, description="Irradiance (W/m²)")
    aoi_angle: float = Field(default=0.0, ge=0, le=90,
                             description="Angle of incidence (degrees)")

    @property
    def total_cells(self) -> int:
        """Total number of cells in module."""
        return self.cells_in_series * self.cells_in_parallel


class CTMLossModel:
    """
    Cell-to-Module Loss Modeling Engine based on Fraunhofer ISE SmartCalc methodology.

    Implements comprehensive k1-k24 factor analysis covering:
    - Optical losses/gains (k1-k7)
    - Coupling effects (k8-k11)
    - Electrical losses (k12-k15)
    - Environmental factors (k21-k24)

    Supports advanced module architectures: half-cut, quarter-cut, shingled, IBC, bifacial.

    Example:
        >>> cell = CellParameters(power_stc=5.2, voltage_mpp=0.65, current_mpp=8.0, ...)
        >>> module = ModuleParameters(cells_in_series=60, module_type=ModuleType.HALF_CUT)
        >>> model = CTMLossModel(cell, module)
        >>> k_factors = model.calculate_all_k_factors()
        >>> module_power = model.calculate_module_power()
    """

    def __init__(
        self,
        cell_params: CellParameters,
        module_params: ModuleParameters,
        validate: bool = True
    ):
        """
        Initialize CTM Loss Model.

        Args:
            cell_params: Cell-level parameters
            module_params: Module-level parameters
            validate: Run validation checks

        Raises:
            ValueError: If parameters are inconsistent
        """
        self.cell = cell_params
        self.module = module_params
        self.k_factors: Dict[str, float] = {}

        if validate:
            self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate parameter consistency."""
        # Check bifacial configuration
        if self.module.is_bifacial and self.module.bifaciality_factor == 0:
            raise ValueError("Bifacial module must have bifaciality_factor > 0")

        # Check cell dimensions
        cell_area = self.cell.width * self.cell.height / 1e6  # mm² to m²
        expected_power = cell_area * 1000 * self.cell.efficiency / 100
        if abs(expected_power - self.cell.power_stc) / self.cell.power_stc > 0.1:
            print(f"Warning: Cell power ({self.cell.power_stc}W) deviates from "
                  f"expected {expected_power:.2f}W based on dimensions and efficiency")

    # ========================================================================
    # OPTICAL LOSSES/GAINS (k1-k7)
    # ========================================================================

    def calculate_k1_glass_reflection_gain(self) -> float:
        """
        Calculate k1: Reflection gain from front glass.

        Glass with AR coating can reflect light back to the cell that would
        otherwise escape, providing a small current gain.

        Returns:
            k1 factor (typically 1.005 to 1.02, i.e., 0.5% to 2% gain)
        """
        base_gain = self.module.glass_reflection_gain

        # AR coating provides additional gain
        if self.module.glass_ar_coating:
            ar_bonus = 0.5  # Additional 0.5% from AR coating
        else:
            ar_bonus = 0.0

        k1 = 1.0 + (base_gain + ar_bonus) / 100
        return k1

    def calculate_k2_encapsulant_gain(self) -> float:
        """
        Calculate k2: Current gain from encapsulant optical properties.

        Yellow EVA and certain encapsulants can shift blue light to red,
        improving spectral match and current generation.

        Returns:
            k2 factor (typically 1.003 to 1.015)
        """
        base_gain = self.module.encapsulant_current_gain

        # Yellow EVA provides higher gain
        if self.module.encapsulant_type == EncapsulantType.YELLOW_EVA:
            type_bonus = 0.5  # Additional 0.5% gain
        else:
            type_bonus = 0.0

        # Thicker encapsulant provides more effect
        thickness_factor = min(self.module.encapsulant_thickness / 0.45, 1.2)

        k2 = 1.0 + (base_gain + type_bonus) * thickness_factor / 100
        return k2

    def calculate_k3_front_grid_correction(self) -> float:
        """
        Calculate k3: Front grid shading loss correction.

        Accounts for difference between cell test (standard grid) and actual
        module grid configuration. Includes busbar and finger coverage.

        Returns:
            k3 factor (typically 0.97 to 0.99)
        """
        # Base shading from cell measurement
        cell_shading = self.cell.front_grid_coverage / 100

        # Module-level adjustments based on architecture
        if self.module.module_type == ModuleType.IBC:
            # IBC has no front metallization
            module_shading = 0.0
        elif self.module.module_type == ModuleType.SHINGLED:
            # Shingled cells have reduced visible shading
            module_shading = cell_shading * 0.5
        else:
            # Standard, half-cut, quarter-cut
            # More busbars = more shading
            busbar_factor = 1.0 + (self.module.busbar_count - 3) * 0.1
            module_shading = cell_shading * busbar_factor

        # k3 accounts for the difference
        k3 = (1.0 - module_shading) / (1.0 - cell_shading)
        return k3

    def calculate_k4_inactive_area_loss(self) -> float:
        """
        Calculate k4: Inactive cell area loss.

        Cell edges near the busbar/junction may be less active. This loss
        is partially recovered in module configuration.

        Returns:
            k4 factor (typically 0.995 to 1.0)
        """
        inactive_fraction = self.cell.inactive_area_fraction / 100

        # Module configuration can partially recover this
        if self.module.module_type == ModuleType.SHINGLED:
            # Shingled cells overlap, recovering some inactive area
            recovery = 0.5
        elif self.module.module_type in [ModuleType.HALF_CUT, ModuleType.QUARTER_CUT]:
            # Cut cells have smaller inactive edges
            recovery = 0.3
        else:
            recovery = 0.0

        effective_loss = inactive_fraction * (1.0 - recovery)
        k4 = 1.0 - effective_loss
        return k4

    def calculate_k5_glass_absorption(self) -> float:
        """
        Calculate k5: Front glass absorption loss.

        Glass absorbs some light, especially UV. Thicker glass increases loss.
        AR coating reduces this effect.

        Returns:
            k5 factor (typically 0.96 to 0.985)
        """
        # Base transmission
        base_transmission = self.module.glass_transmittance / 100

        # Thickness correction (thicker = more absorption)
        thickness_factor = 1.0 - (self.module.glass_thickness - 3.2) * 0.005

        # AR coating improves transmission
        if self.module.glass_ar_coating:
            ar_improvement = 1.02
        else:
            ar_improvement = 1.0

        k5 = base_transmission * thickness_factor * ar_improvement
        return k5

    def calculate_k6_encapsulant_absorption(self) -> float:
        """
        Calculate k6: Encapsulant absorption and yellowing loss.

        EVA can yellow under UV exposure, reducing transmission. POE and
        silicone are more stable.

        Returns:
            k6 factor (typically 0.985 to 1.0)
        """
        # Base yellowing loss
        yellowing_loss = self.module.encapsulant_yellowing / 100

        # Material-dependent UV stability
        if self.module.encapsulant_type == EncapsulantType.POE:
            uv_stability = 0.99
        elif self.module.encapsulant_type == EncapsulantType.SILICONE:
            uv_stability = 0.995
        elif self.module.encapsulant_type == EncapsulantType.YELLOW_EVA:
            uv_stability = 0.97  # More susceptible to degradation
        else:  # Standard EVA
            uv_stability = 0.985

        k6 = uv_stability * (1.0 - yellowing_loss)
        return k6

    def calculate_k7_rear_optical_properties(self) -> float:
        """
        Calculate k7: Rear side optical properties (bifacial gain).

        For bifacial modules, rear side can generate additional power.
        For monofacial, accounts for any rear reflectance effects.

        Returns:
            k7 factor (1.0 for monofacial, up to 1.3 for bifacial with high albedo)
        """
        if not self.module.is_bifacial:
            # Monofacial: small gain from rear reflectance
            k7 = 1.0 + self.module.rear_optical_gain / 100
        else:
            # Bifacial: significant gain from rear irradiance
            # Assuming 20% rear irradiance under standard conditions
            rear_contribution = 0.20 * self.module.bifaciality_factor

            # Glass-glass construction improves rear optics
            if self.module.rear_glass:
                rear_contribution *= 1.05

            k7 = 1.0 + rear_contribution + self.module.rear_optical_gain / 100

        return k7

    # ========================================================================
    # COUPLING EFFECTS (k8-k11)
    # ========================================================================

    def calculate_k8_cell_gap_losses(self) -> float:
        """
        Calculate k8: Cell-to-cell gap losses.

        Gaps between cells reduce active area. Smaller gaps and shingled
        designs minimize this loss.

        Returns:
            k8 factor (typically 0.95 to 0.99)
        """
        if self.module.module_type == ModuleType.SHINGLED:
            # Shingled cells overlap, no gap loss
            k8 = 1.0
        else:
            # Calculate module active area
            cell_area = self.cell.width * self.cell.height  # mm²
            total_cell_area = cell_area * self.module.total_cells

            # Gap area (simplified)
            num_gaps_x = self.module.cells_in_series - 1
            num_gaps_y = self.module.cells_in_parallel - 1

            gap_area_x = num_gaps_x * self.module.cell_gap * self.cell.height
            gap_area_y = num_gaps_y * self.module.cell_gap * self.cell.width
            total_gap_area = gap_area_x + gap_area_y

            # Also account for border
            module_width = self.cell.width * self.module.cells_in_parallel + \
                          (self.module.cells_in_parallel - 1) * self.module.cell_gap + \
                          2 * self.module.border_width
            module_height = self.cell.height * self.module.cells_in_series + \
                           (self.module.cells_in_series - 1) * self.module.cell_gap + \
                           2 * self.module.border_width
            module_area = module_width * module_height

            k8 = total_cell_area / module_area

        return k8

    def calculate_k9_internal_mismatch(self) -> float:
        """
        Calculate k9: Internal cell mismatch loss.

        Cells in series are limited by weakest cell. Binning quality and
        module design affect this. Half-cut reduces impact.

        Returns:
            k9 factor (typically 0.98 to 0.995)
        """
        base_mismatch = self.module.cell_mismatch / 100

        # Half-cut and quarter-cut reduce mismatch impact
        if self.module.module_type == ModuleType.HALF_CUT:
            mismatch_reduction = 0.5  # 50% reduction in mismatch loss
        elif self.module.module_type == ModuleType.QUARTER_CUT:
            mismatch_reduction = 0.7  # 70% reduction
        else:
            mismatch_reduction = 0.0

        effective_mismatch = base_mismatch * (1.0 - mismatch_reduction)
        k9 = 1.0 - effective_mismatch
        return k9

    def calculate_k10_module_mismatch(self) -> float:
        """
        Calculate k10: Module-to-module mismatch.

        Accounts for variation between modules in production. This is a
        system-level factor but included for completeness.

        Returns:
            k10 factor (typically 0.995 to 0.998)
        """
        k10 = 1.0 - self.module.module_mismatch / 100
        return k10

    def calculate_k11_lid_letid(self) -> float:
        """
        Calculate k11: Light-Induced Degradation (LID) and LETID effects.

        LID occurs in first hours/days of operation. LETID affects certain
        cell types under elevated temperature and light.

        Returns:
            k11 factor (typically 0.97 to 0.99)
        """
        lid_loss = self.cell.lid_factor / 100
        letid_loss = self.cell.letid_factor / 100

        # Combined effect (assuming independent)
        k11 = (1.0 - lid_loss) * (1.0 - letid_loss)
        return k11

    # ========================================================================
    # ELECTRICAL LOSSES (k12-k15)
    # ========================================================================

    def calculate_k12_resistive_losses(self) -> float:
        """
        Calculate k12: Series resistance losses (ribbon, busbar, junction box).

        Accounts for I²R losses in interconnects and junction box. Half-cut
        cells reduce current, thus reducing resistive losses.

        Returns:
            k12 factor (typically 0.97 to 0.99)
        """
        # Calculate total module current at MPP
        if self.module.module_type in [ModuleType.HALF_CUT, ModuleType.QUARTER_CUT]:
            # Current is reduced due to parallel connection
            module_current = self.cell.current_mpp * self.module.cells_in_parallel
        else:
            module_current = self.cell.current_mpp

        # Ribbon resistance (simplified)
        ribbon_length = self.cell.width * self.module.cells_in_series / 1000  # m
        ribbon_area = self.module.ribbon_width * self.module.ribbon_thickness / 1e6  # m²
        ribbon_resistance = self.module.ribbon_resistivity * ribbon_length / ribbon_area

        # Junction box and busbar (empirical)
        jbox_loss = self.module.junction_box_loss / 100

        # Total resistive loss
        i2r_loss = (module_current ** 2 * ribbon_resistance) / \
                   (self.cell.power_stc * self.module.total_cells)
        total_loss = i2r_loss + jbox_loss

        k12 = 1.0 - total_loss
        return k12

    def calculate_k13_interconnection_resistance(self) -> float:
        """
        Calculate k13: Cell interconnection resistance loss.

        Contact resistance between ribbon and cell metallization.
        Shingled cells have different interconnection characteristics.

        Returns:
            k13 factor (typically 0.995 to 0.999)
        """
        if self.module.module_type == ModuleType.SHINGLED:
            # Shingled: conductive adhesive, slightly higher resistance
            contact_loss = 0.8  # 0.8% loss
        elif self.module.module_type == ModuleType.IBC:
            # IBC: all contacts on rear, optimized
            contact_loss = 0.3  # 0.3% loss
        else:
            # Standard soldered connection
            contact_loss = 0.5  # 0.5% loss

        k13 = 1.0 - contact_loss / 100
        return k13

    def calculate_k14_manufacturing_damage(self) -> float:
        """
        Calculate k14: Quality/damage losses from manufacturing.

        Handling, soldering thermal stress, and lamination can cause
        micro-cracks and other damage.

        Returns:
            k14 factor (typically 0.995 to 0.998)
        """
        base_damage = self.module.manufacturing_damage / 100

        # Shingled and IBC may have different damage profiles
        if self.module.module_type == ModuleType.SHINGLED:
            # Shingled: less thermal stress, but more handling
            damage_factor = 1.0
        elif self.module.module_type == ModuleType.IBC:
            # IBC: specialized process, potentially less damage
            damage_factor = 0.7
        else:
            damage_factor = 1.0

        k14 = 1.0 - base_damage * damage_factor
        return k14

    def calculate_k15_inactive_electrical_loss(self) -> float:
        """
        Calculate k15: Inactive cell area electrical loss.

        Similar to k4 but accounts for electrical (not optical) effects
        of inactive regions near cell edges.

        Returns:
            k15 factor (typically 0.997 to 1.0)
        """
        # Electrical inactive area is smaller than optical
        electrical_inactive = self.cell.inactive_area_fraction * 0.5 / 100

        k15 = 1.0 - electrical_inactive
        return k15

    # ========================================================================
    # ENVIRONMENTAL FACTORS (k21-k24)
    # ========================================================================

    def calculate_k21_temperature_coefficient(self) -> float:
        """
        Calculate k21: Temperature coefficient effect.

        Accounts for power loss at operating temperature vs. STC (25°C).

        Returns:
            k21 factor (typically 0.85 to 1.0 depending on operating temp)
        """
        temp_difference = self.module.operating_temperature - 25.0
        temp_loss = self.cell.temp_coeff_power * temp_difference / 100

        k21 = 1.0 + temp_loss
        return k21

    def calculate_k22_low_irradiance_losses(self) -> float:
        """
        Calculate k22: Low irradiance performance losses.

        Modules perform differently at low light levels due to shunt
        resistance and recombination effects.

        Returns:
            k22 factor (1.0 at high irradiance, lower at low irradiance)
        """
        # Normalized irradiance
        norm_irradiance = self.module.irradiance / 1000.0

        if norm_irradiance >= 0.8:
            # High irradiance: minimal effect
            k22 = 1.0
        else:
            # Low irradiance: apply loss
            # Loss increases as irradiance decreases
            base_loss = self.cell.low_irradiance_loss / 100
            irradiance_factor = (0.8 - norm_irradiance) / 0.8
            k22 = 1.0 - base_loss * irradiance_factor

        return k22

    def calculate_k23_spectral_response(self) -> float:
        """
        Calculate k23: Spectral response differences.

        Solar spectrum varies with air mass, cloud cover, etc. Cell spectral
        response may differ from standard test spectrum.

        Returns:
            k23 factor (typically 0.98 to 1.02)
        """
        # Use cell spectral mismatch parameter
        k23 = 1.0 + self.cell.spectral_mismatch / 100
        return k23

    def calculate_k24_angle_of_incidence(self) -> float:
        """
        Calculate k24: Angle of incidence (AOI) effects.

        Light hitting module at an angle experiences different reflection
        and absorption. AR coating helps reduce this effect.

        Returns:
            k24 factor (1.0 at normal incidence, lower at high angles)
        """
        aoi_rad = np.deg2rad(self.module.aoi_angle)

        # Basic cosine loss
        cosine_factor = np.cos(aoi_rad)

        # Additional reflection losses at high angles (Fresnel equations, simplified)
        if self.module.aoi_angle > 50:
            reflection_loss = 0.01 * (self.module.aoi_angle - 50) / 40  # Up to 1% at 90°
        else:
            reflection_loss = 0.0

        # AR coating reduces AOI losses
        if self.module.glass_ar_coating:
            reflection_loss *= 0.5

        k24 = cosine_factor * (1.0 - reflection_loss)
        return k24

    # ========================================================================
    # MAIN CALCULATION METHODS
    # ========================================================================

    def calculate_all_k_factors(self) -> Dict[str, float]:
        """
        Calculate all k1-k24 factors.

        Returns:
            Dictionary mapping factor name to value

        Example:
            >>> k_factors = model.calculate_all_k_factors()
            >>> print(f"Glass reflection gain: {k_factors['k1']:.4f}")
        """
        self.k_factors = {
            # Optical (k1-k7)
            'k1_glass_reflection': self.calculate_k1_glass_reflection_gain(),
            'k2_encapsulant_gain': self.calculate_k2_encapsulant_gain(),
            'k3_grid_correction': self.calculate_k3_front_grid_correction(),
            'k4_inactive_area': self.calculate_k4_inactive_area_loss(),
            'k5_glass_absorption': self.calculate_k5_glass_absorption(),
            'k6_encapsulant_absorption': self.calculate_k6_encapsulant_absorption(),
            'k7_rear_optical': self.calculate_k7_rear_optical_properties(),

            # Coupling (k8-k11)
            'k8_cell_gaps': self.calculate_k8_cell_gap_losses(),
            'k9_internal_mismatch': self.calculate_k9_internal_mismatch(),
            'k10_module_mismatch': self.calculate_k10_module_mismatch(),
            'k11_lid_letid': self.calculate_k11_lid_letid(),

            # Electrical (k12-k15)
            'k12_resistive': self.calculate_k12_resistive_losses(),
            'k13_interconnection': self.calculate_k13_interconnection_resistance(),
            'k14_manufacturing': self.calculate_k14_manufacturing_damage(),
            'k15_inactive_electrical': self.calculate_k15_inactive_electrical_loss(),

            # Environmental (k21-k24)
            'k21_temperature': self.calculate_k21_temperature_coefficient(),
            'k22_low_irradiance': self.calculate_k22_low_irradiance_losses(),
            'k23_spectral': self.calculate_k23_spectral_response(),
            'k24_aoi': self.calculate_k24_angle_of_incidence(),
        }

        return self.k_factors

    def calculate_module_power(
        self,
        cell_power: Optional[float] = None,
        k_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate module power from cell power and k-factors.

        Args:
            cell_power: Cell power in watts (uses cell_params.power_stc if None)
            k_factors: Pre-calculated k-factors (calculates if None)

        Returns:
            Module power in watts

        Example:
            >>> module_power = model.calculate_module_power()
            >>> print(f"Module power: {module_power:.2f} W")
        """
        if cell_power is None:
            cell_power = self.cell.power_stc

        if k_factors is None:
            k_factors = self.calculate_all_k_factors()

        # Start with sum of cell powers
        total_cell_power = cell_power * self.module.total_cells

        # Apply all k-factors multiplicatively
        module_power = total_cell_power
        for k_value in k_factors.values():
            module_power *= k_value

        return module_power

    def get_ctm_ratio(self) -> float:
        """
        Calculate Cell-to-Module ratio (CTM power gain/loss).

        Returns:
            CTM ratio (module_power / sum_of_cell_powers)

        Example:
            >>> ctm = model.get_ctm_ratio()
            >>> print(f"CTM ratio: {ctm:.1%}")  # e.g., "CTM ratio: 96.5%"
        """
        module_power = self.calculate_module_power()
        total_cell_power = self.cell.power_stc * self.module.total_cells
        return module_power / total_cell_power

    def get_loss_breakdown(self) -> Dict[str, float]:
        """
        Get percentage contribution of each k-factor category.

        Returns:
            Dictionary with category names and their loss/gain percentages
        """
        if not self.k_factors:
            self.calculate_all_k_factors()

        # Calculate category products
        optical = np.prod([self.k_factors[f'k{i}_' + name]
                          for i, name in enumerate([
                              'glass_reflection', 'encapsulant_gain', 'grid_correction',
                              'inactive_area', 'glass_absorption', 'encapsulant_absorption',
                              'rear_optical'
                          ], 1)])

        coupling = np.prod([self.k_factors[f'k{i}_' + name]
                           for i, name in enumerate([
                               'cell_gaps', 'internal_mismatch', 'module_mismatch', 'lid_letid'
                           ], 8)])

        electrical = np.prod([self.k_factors[f'k{i}_' + name]
                             for i, name in enumerate([
                                 'resistive', 'interconnection', 'manufacturing',
                                 'inactive_electrical'
                             ], 12)])

        environmental = np.prod([self.k_factors[f'k{i}_' + name]
                                for i, name in enumerate([
                                    'temperature', 'low_irradiance', 'spectral', 'aoi'
                                ], 21)])

        return {
            'Optical (k1-k7)': (optical - 1.0) * 100,
            'Coupling (k8-k11)': (coupling - 1.0) * 100,
            'Electrical (k12-k15)': (electrical - 1.0) * 100,
            'Environmental (k21-k24)': (environmental - 1.0) * 100,
            'Total CTM': (self.get_ctm_ratio() - 1.0) * 100,
        }

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def generate_loss_waterfall(
        self,
        use_plotly: bool = False,
        title: Optional[str] = None
    ) -> Figure:
        """
        Generate waterfall chart showing cumulative k-factor effects.

        Args:
            use_plotly: Use Plotly instead of Matplotlib
            title: Custom chart title

        Returns:
            Matplotlib Figure or Plotly Figure object

        Example:
            >>> fig = model.generate_loss_waterfall()
            >>> fig.savefig('ctm_waterfall.png')
        """
        if not self.k_factors:
            self.calculate_all_k_factors()

        # Calculate cumulative power
        cell_power_total = self.cell.power_stc * self.module.total_cells
        powers = [cell_power_total]
        labels = ['Cell Power\nTotal']

        # Add each k-factor
        current_power = cell_power_total
        for name, k_value in self.k_factors.items():
            current_power *= k_value
            powers.append(current_power)
            # Format label
            factor_num = name.split('_')[0]
            factor_desc = ' '.join(name.split('_')[1:]).title()
            change = (k_value - 1.0) * 100
            labels.append(f'{factor_num}\n{factor_desc}\n({change:+.2f}%)')

        labels.append('Module\nPower')

        if use_plotly:
            # Plotly waterfall
            fig = go.Figure(go.Waterfall(
                x=labels,
                measure=['absolute'] + ['relative'] * len(self.k_factors) + ['total'],
                y=[cell_power_total] + [powers[i] - powers[i-1]
                                        for i in range(1, len(powers))] + [powers[-1]],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "rgba(255, 0, 0, 0.6)"}},
                increasing={"marker": {"color": "rgba(0, 255, 0, 0.6)"}},
                totals={"marker": {"color": "rgba(0, 0, 255, 0.6)"}},
            ))

            fig.update_layout(
                title=title or "CTM Loss/Gain Waterfall Analysis",
                yaxis_title="Power (W)",
                showlegend=False,
                height=600,
            )

            return fig
        else:
            # Matplotlib waterfall
            fig, ax = plt.subplots(figsize=(16, 8))

            # Calculate bar positions and colors
            x_pos = np.arange(len(powers))
            colors = []
            bottom_values = []
            bar_heights = []

            # First bar (cell power)
            colors.append('blue')
            bottom_values.append(0)
            bar_heights.append(powers[0])

            # K-factor bars
            for i in range(1, len(powers) - 1):
                delta = powers[i] - powers[i-1]
                if delta >= 0:
                    colors.append('green')
                    bottom_values.append(powers[i-1])
                else:
                    colors.append('red')
                    bottom_values.append(powers[i])
                bar_heights.append(abs(delta))

            # Last bar (module power)
            colors.append('blue')
            bottom_values.append(0)
            bar_heights.append(powers[-1])

            # Plot bars
            bars = ax.bar(x_pos, bar_heights, bottom=bottom_values, color=colors, alpha=0.7, width=0.8)

            # Add connecting lines
            for i in range(len(powers) - 1):
                ax.plot([i + 0.4, i + 1.4], [powers[i], powers[i]], 'k--', alpha=0.3, linewidth=1)

            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Power (W)', fontsize=12)
            ax.set_title(title or 'CTM Loss/Gain Waterfall Analysis', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (bar, power) in enumerate(zip(bars, powers)):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2,
                           f'{power:.1f}W', ha='center', va='center', fontsize=7,
                           fontweight='bold', color='white',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            plt.tight_layout()
            return fig

    # ========================================================================
    # SENSITIVITY ANALYSIS
    # ========================================================================

    def sensitivity_analysis(
        self,
        parameter: str,
        variation_range: Tuple[float, float] = (0.8, 1.2),
        num_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on a specific parameter.

        Args:
            parameter: Parameter name (e.g., 'cell.efficiency', 'module.glass_thickness')
            variation_range: Tuple of (min_factor, max_factor) to vary parameter
            num_points: Number of points to evaluate

        Returns:
            Dictionary with 'parameter_values', 'module_power', 'ctm_ratio' arrays

        Example:
            >>> results = model.sensitivity_analysis('cell.efficiency', (0.9, 1.1))
            >>> plt.plot(results['parameter_values'], results['module_power'])
        """
        # Parse parameter path
        parts = parameter.split('.')
        if len(parts) != 2:
            raise ValueError(f"Parameter must be in format 'object.attribute', got {parameter}")

        obj_name, attr_name = parts
        if obj_name == 'cell':
            obj = self.cell
        elif obj_name == 'module':
            obj = self.module
        else:
            raise ValueError(f"Unknown object: {obj_name}")

        if not hasattr(obj, attr_name):
            raise ValueError(f"{obj_name} has no attribute {attr_name}")

        # Get original value
        original_value = getattr(obj, attr_name)

        # Generate variation range
        min_val = original_value * variation_range[0]
        max_val = original_value * variation_range[1]
        param_values = np.linspace(min_val, max_val, num_points)

        # Calculate results
        module_powers = []
        ctm_ratios = []

        for val in param_values:
            # Temporarily set parameter
            setattr(obj, attr_name, val)

            # Recalculate
            power = self.calculate_module_power()
            ctm = self.get_ctm_ratio()

            module_powers.append(power)
            ctm_ratios.append(ctm)

        # Restore original value
        setattr(obj, attr_name, original_value)

        return {
            'parameter_values': param_values,
            'module_power': np.array(module_powers),
            'ctm_ratio': np.array(ctm_ratios),
        }

    def multi_parameter_sensitivity(
        self,
        parameters: List[str],
        variation_range: Tuple[float, float] = (0.9, 1.1),
        num_points: int = 10
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Perform sensitivity analysis on multiple parameters.

        Args:
            parameters: List of parameter names
            variation_range: Range to vary each parameter
            num_points: Points per parameter

        Returns:
            Dictionary mapping parameter names to sensitivity results
        """
        results = {}
        for param in parameters:
            results[param] = self.sensitivity_analysis(param, variation_range, num_points)
        return results

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_report(self) -> str:
        """
        Generate comprehensive text report of CTM analysis.

        Returns:
            Formatted report string
        """
        if not self.k_factors:
            self.calculate_all_k_factors()

        module_power = self.calculate_module_power()
        ctm_ratio = self.get_ctm_ratio()
        breakdown = self.get_loss_breakdown()

        report = []
        report.append("=" * 80)
        report.append("CTM LOSS MODELING REPORT - Fraunhofer ISE SmartCalc Methodology")
        report.append("=" * 80)
        report.append("")

        # Cell info
        report.append("CELL PARAMETERS:")
        report.append(f"  Power (STC):        {self.cell.power_stc:.3f} W")
        report.append(f"  Efficiency:         {self.cell.efficiency:.2f}%")
        report.append(f"  Dimensions:         {self.cell.width} x {self.cell.height} mm")
        report.append(f"  Vmp / Imp:          {self.cell.voltage_mpp:.3f} V / {self.cell.current_mpp:.3f} A")
        report.append("")

        # Module info
        report.append("MODULE PARAMETERS:")
        report.append(f"  Type:               {self.module.module_type.value}")
        report.append(f"  Configuration:      {self.module.cells_in_series}S{self.module.cells_in_parallel}P "
                     f"({self.module.total_cells} cells)")
        report.append(f"  Encapsulant:        {self.module.encapsulant_type.value}")
        report.append(f"  Bifacial:           {'Yes' if self.module.is_bifacial else 'No'}")
        report.append("")

        # Results
        report.append("RESULTS:")
        report.append(f"  Total Cell Power:   {self.cell.power_stc * self.module.total_cells:.2f} W")
        report.append(f"  Module Power:       {module_power:.2f} W")
        report.append(f"  CTM Ratio:          {ctm_ratio:.4f} ({(ctm_ratio - 1) * 100:+.2f}%)")
        report.append("")

        # Breakdown by category
        report.append("LOSS/GAIN BREAKDOWN BY CATEGORY:")
        for category, value in breakdown.items():
            report.append(f"  {category:30s} {value:+6.2f}%")
        report.append("")

        # Individual k-factors
        report.append("DETAILED K-FACTORS:")
        for name, value in self.k_factors.items():
            change = (value - 1.0) * 100
            factor_desc = ' '.join(name.split('_')[1:]).replace('_', ' ').title()
            report.append(f"  {name:30s} {value:8.5f}  ({change:+6.2f}%)  {factor_desc}")

        report.append("")
        report.append("=" * 80)

        return '\n'.join(report)

    def __repr__(self) -> str:
        """String representation."""
        return (f"CTMLossModel(cell_power={self.cell.power_stc}W, "
                f"module_type={self.module.module_type.value}, "
                f"cells={self.module.total_cells})")
