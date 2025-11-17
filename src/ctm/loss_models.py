"""Loss models for CTM calculations."""

from typing import Dict
from pydantic import BaseModel, Field
import numpy as np


class OpticalLosses(BaseModel):
    """Model for optical losses in module."""

    glass_thickness_mm: float = Field(default=3.2, description="Front glass thickness in mm")
    encapsulant_thickness_mm: float = Field(default=0.45, description="Encapsulant thickness in mm")
    cell_coverage_ratio: float = Field(default=0.95, description="Cell coverage ratio (active area / total area)")

    @property
    def transmission_loss(self) -> float:
        """Calculate transmission loss through glass and encapsulant."""
        # Beer-Lambert law approximation
        glass_transmission = 0.90 + 0.06 * (1 - self.glass_thickness_mm / 4.0)
        encapsulant_transmission = 0.98 + 0.005 * (1 - self.encapsulant_thickness_mm / 0.5)
        return glass_transmission * encapsulant_transmission

    @property
    def reflection_loss(self) -> float:
        """Calculate reflection loss at interfaces."""
        # Fresnel reflection at air-glass interface (typically 4% per surface without AR coating)
        # With AR coating: ~1-2% total
        return 0.98

    @property
    def inactive_area_loss(self) -> float:
        """Calculate loss due to inactive area (gaps between cells)."""
        return self.cell_coverage_ratio


class ElectricalLosses(BaseModel):
    """Model for electrical losses in module."""

    num_cells_series: int = Field(..., description="Number of cells in series")
    num_strings_parallel: int = Field(default=1, description="Number of parallel strings")
    interconnect_resistance_ohm: float = Field(default=0.005, description="Interconnect resistance in Ohms")
    operating_current_a: float = Field(..., description="Operating current in Amps")

    @property
    def series_resistance_loss(self) -> float:
        """Calculate series resistance power loss."""
        # Power loss = I^2 * R
        total_resistance = self.interconnect_resistance_ohm * self.num_cells_series
        power_loss_fraction = (self.operating_current_a ** 2 * total_resistance) / 100  # Normalized
        return 1 - min(0.05, power_loss_fraction)  # Cap at 5% loss

    @property
    def mismatch_loss(self) -> float:
        """Calculate cell-to-cell mismatch loss."""
        # Typically 1-3% for quality matched cells
        std_dev = 0.02  # 2% standard deviation in cell efficiency
        mismatch_factor = 1 - (std_dev ** 2) * (self.num_cells_series - 1) / 2
        return max(0.95, mismatch_factor)

    @property
    def bypass_diode_loss(self) -> float:
        """Calculate bypass diode forward voltage loss."""
        # Typically 0.3-0.5V per diode when conducting
        # Under normal operation, diodes are reverse-biased (no loss)
        # Small leakage current: ~0.1-0.3% loss
        return 0.997


class ThermalLosses(BaseModel):
    """Model for thermal losses and effects."""

    cell_temp_coefficient_pct_c: float = Field(default=-0.4, description="Temperature coefficient %/°C")
    noct_c: float = Field(default=45.0, description="NOCT temperature in °C")
    ambient_temp_c: float = Field(default=25.0, description="Ambient temperature in °C")
    stc_temp_c: float = Field(default=25.0, description="STC temperature in °C")

    @property
    def noct_power_loss(self) -> float:
        """Calculate power loss at NOCT."""
        temp_delta = self.noct_c - self.stc_temp_c
        power_loss_pct = abs(self.cell_temp_coefficient_pct_c) * temp_delta
        return 1 - (power_loss_pct / 100)

    @property
    def operating_temp_loss(self) -> float:
        """Calculate power loss at operating temperature."""
        temp_delta = self.noct_c - self.ambient_temp_c
        # Simplified model: cell temp rises above ambient
        power_loss_pct = abs(self.cell_temp_coefficient_pct_c) * temp_delta
        return 1 - (power_loss_pct / 100)

    @property
    def thermal_gradient_loss(self) -> float:
        """Calculate loss due to temperature gradients across module."""
        # Typically 1-2% loss due to non-uniform temperature
        return 0.99
