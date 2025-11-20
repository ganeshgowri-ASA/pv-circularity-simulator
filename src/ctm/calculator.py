"""CTM (Cell-to-Module) calculator with k-factors."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np


class CTMFactors(BaseModel):
    """CTM k-factors based on industry standards.

    These factors represent various losses in the cell-to-module conversion process.
    Reference: IEC 61853 and industry best practices.
    """

    # Optical losses (k1-k8)
    k1_reflection_loss: float = Field(default=0.98, ge=0.9, le=1.0, description="Anti-reflection coating effectiveness")
    k2_glass_transmission: float = Field(default=0.96, ge=0.90, le=1.0, description="Glass transmission factor")
    k3_encapsulant_transmission: float = Field(default=0.985, ge=0.95, le=1.0, description="Encapsulant transmission")
    k4_soiling_factor: float = Field(default=0.98, ge=0.90, le=1.0, description="Soiling/cleanliness factor")
    k5_spectral_mismatch: float = Field(default=0.99, ge=0.95, le=1.0, description="Spectral mismatch factor")
    k6_angular_losses: float = Field(default=0.97, ge=0.90, le=1.0, description="Angular incidence losses")
    k7_inactive_area: float = Field(default=0.95, ge=0.85, le=1.0, description="Packing/inactive area factor")
    k8_optical_coupling: float = Field(default=0.995, ge=0.98, le=1.0, description="Optical coupling efficiency")

    # Electrical losses (k9-k16)
    k9_interconnection_loss: float = Field(default=0.98, ge=0.95, le=1.0, description="Cell interconnection losses")
    k10_series_resistance: float = Field(default=0.99, ge=0.95, le=1.0, description="Series resistance losses")
    k11_shunt_resistance: float = Field(default=0.998, ge=0.99, le=1.0, description="Shunt resistance losses")
    k12_cell_mismatch: float = Field(default=0.98, ge=0.95, le=1.0, description="Cell-to-cell mismatch losses")
    k13_diode_losses: float = Field(default=0.997, ge=0.99, le=1.0, description="Bypass diode forward voltage losses")
    k14_junction_box: float = Field(default=0.995, ge=0.99, le=1.0, description="Junction box connection losses")
    k15_cable_resistance: float = Field(default=0.998, ge=0.99, le=1.0, description="Cable resistance losses")
    k16_contact_resistance: float = Field(default=0.997, ge=0.99, le=1.0, description="Contact resistance losses")

    # Thermal losses (k17-k20)
    k17_thermal_mismatch: float = Field(default=0.99, ge=0.95, le=1.0, description="Thermal mismatch between cells")
    k18_heat_dissipation: float = Field(default=0.985, ge=0.95, le=1.0, description="Heat dissipation factor")
    k19_noct_effect: float = Field(default=0.96, ge=0.90, le=1.0, description="NOCT temperature effect")
    k20_hot_spot_risk: float = Field(default=0.995, ge=0.98, le=1.0, description="Hot spot risk mitigation")

    # Manufacturing and quality (k21-k24)
    k21_manufacturing_tolerance: float = Field(default=0.97, ge=0.95, le=1.0, description="Manufacturing tolerance")
    k22_lamination_quality: float = Field(default=0.995, ge=0.98, le=1.0, description="Lamination quality factor")
    k23_edge_deletion: float = Field(default=0.998, ge=0.99, le=1.0, description="Edge deletion losses")
    k24_measurement_uncertainty: float = Field(default=0.99, ge=0.97, le=1.0, description="Measurement uncertainty")

    @property
    def optical_loss_total(self) -> float:
        """Calculate total optical losses."""
        return (
            self.k1_reflection_loss
            * self.k2_glass_transmission
            * self.k3_encapsulant_transmission
            * self.k4_soiling_factor
            * self.k5_spectral_mismatch
            * self.k6_angular_losses
            * self.k7_inactive_area
            * self.k8_optical_coupling
        )

    @property
    def electrical_loss_total(self) -> float:
        """Calculate total electrical losses."""
        return (
            self.k9_interconnection_loss
            * self.k10_series_resistance
            * self.k11_shunt_resistance
            * self.k12_cell_mismatch
            * self.k13_diode_losses
            * self.k14_junction_box
            * self.k15_cable_resistance
            * self.k16_contact_resistance
        )

    @property
    def thermal_loss_total(self) -> float:
        """Calculate total thermal losses."""
        return (
            self.k17_thermal_mismatch
            * self.k18_heat_dissipation
            * self.k19_noct_effect
            * self.k20_hot_spot_risk
        )

    @property
    def manufacturing_loss_total(self) -> float:
        """Calculate total manufacturing and quality losses."""
        return (
            self.k21_manufacturing_tolerance
            * self.k22_lamination_quality
            * self.k23_edge_deletion
            * self.k24_measurement_uncertainty
        )

    @property
    def ctm_ratio(self) -> float:
        """Calculate overall CTM ratio."""
        return (
            self.optical_loss_total
            * self.electrical_loss_total
            * self.thermal_loss_total
            * self.manufacturing_loss_total
        )

    def get_all_factors(self) -> Dict[str, float]:
        """Get all k-factors as a dictionary."""
        return {
            "k1_reflection_loss": self.k1_reflection_loss,
            "k2_glass_transmission": self.k2_glass_transmission,
            "k3_encapsulant_transmission": self.k3_encapsulant_transmission,
            "k4_soiling_factor": self.k4_soiling_factor,
            "k5_spectral_mismatch": self.k5_spectral_mismatch,
            "k6_angular_losses": self.k6_angular_losses,
            "k7_inactive_area": self.k7_inactive_area,
            "k8_optical_coupling": self.k8_optical_coupling,
            "k9_interconnection_loss": self.k9_interconnection_loss,
            "k10_series_resistance": self.k10_series_resistance,
            "k11_shunt_resistance": self.k11_shunt_resistance,
            "k12_cell_mismatch": self.k12_cell_mismatch,
            "k13_diode_losses": self.k13_diode_losses,
            "k14_junction_box": self.k14_junction_box,
            "k15_cable_resistance": self.k15_cable_resistance,
            "k16_contact_resistance": self.k16_contact_resistance,
            "k17_thermal_mismatch": self.k17_thermal_mismatch,
            "k18_heat_dissipation": self.k18_heat_dissipation,
            "k19_noct_effect": self.k19_noct_effect,
            "k20_hot_spot_risk": self.k20_hot_spot_risk,
            "k21_manufacturing_tolerance": self.k21_manufacturing_tolerance,
            "k22_lamination_quality": self.k22_lamination_quality,
            "k23_edge_deletion": self.k23_edge_deletion,
            "k24_measurement_uncertainty": self.k24_measurement_uncertainty,
        }

    def get_loss_waterfall(self) -> List[Dict[str, float]]:
        """Get loss waterfall data for visualization."""
        waterfall = [
            {"category": "Cell Pmax", "value": 1.0, "loss_pct": 0.0},
        ]

        # Optical losses
        current_value = 1.0
        for k, v in [
            ("Reflection Loss", self.k1_reflection_loss),
            ("Glass Transmission", self.k2_glass_transmission),
            ("Encapsulant Transmission", self.k3_encapsulant_transmission),
            ("Soiling Factor", self.k4_soiling_factor),
            ("Spectral Mismatch", self.k5_spectral_mismatch),
            ("Angular Losses", self.k6_angular_losses),
            ("Inactive Area", self.k7_inactive_area),
            ("Optical Coupling", self.k8_optical_coupling),
        ]:
            new_value = current_value * v
            loss_pct = (1 - v) * 100
            waterfall.append({"category": k, "value": new_value, "loss_pct": loss_pct})
            current_value = new_value

        # Electrical losses
        for k, v in [
            ("Interconnection Loss", self.k9_interconnection_loss),
            ("Series Resistance", self.k10_series_resistance),
            ("Shunt Resistance", self.k11_shunt_resistance),
            ("Cell Mismatch", self.k12_cell_mismatch),
            ("Diode Losses", self.k13_diode_losses),
            ("Junction Box", self.k14_junction_box),
            ("Cable Resistance", self.k15_cable_resistance),
            ("Contact Resistance", self.k16_contact_resistance),
        ]:
            new_value = current_value * v
            loss_pct = (1 - v) * 100
            waterfall.append({"category": k, "value": new_value, "loss_pct": loss_pct})
            current_value = new_value

        # Thermal losses
        for k, v in [
            ("Thermal Mismatch", self.k17_thermal_mismatch),
            ("Heat Dissipation", self.k18_heat_dissipation),
            ("NOCT Effect", self.k19_noct_effect),
            ("Hot Spot Risk", self.k20_hot_spot_risk),
        ]:
            new_value = current_value * v
            loss_pct = (1 - v) * 100
            waterfall.append({"category": k, "value": new_value, "loss_pct": loss_pct})
            current_value = new_value

        # Manufacturing losses
        for k, v in [
            ("Manufacturing Tolerance", self.k21_manufacturing_tolerance),
            ("Lamination Quality", self.k22_lamination_quality),
            ("Edge Deletion", self.k23_edge_deletion),
            ("Measurement Uncertainty", self.k24_measurement_uncertainty),
        ]:
            new_value = current_value * v
            loss_pct = (1 - v) * 100
            waterfall.append({"category": k, "value": new_value, "loss_pct": loss_pct})
            current_value = new_value

        waterfall.append({"category": "Module Pmax", "value": current_value, "loss_pct": 0.0})

        return waterfall


class CTMResult(BaseModel):
    """Result of CTM calculation."""

    cell_pmax_total_w: float = Field(..., description="Total cell Pmax in watts")
    module_pmax_w: float = Field(..., description="Module Pmax in watts")
    ctm_ratio: float = Field(..., description="CTM ratio (module/cell)")
    total_loss_pct: float = Field(..., description="Total loss percentage")
    optical_loss_pct: float = Field(..., description="Optical loss percentage")
    electrical_loss_pct: float = Field(..., description="Electrical loss percentage")
    thermal_loss_pct: float = Field(..., description="Thermal loss percentage")
    manufacturing_loss_pct: float = Field(..., description="Manufacturing loss percentage")
    waterfall_data: List[Dict[str, float]] = Field(..., description="Loss waterfall data for visualization")


class CTMCalculator:
    """Calculator for Cell-to-Module conversion with loss analysis."""

    def __init__(self, factors: Optional[CTMFactors] = None):
        """Initialize CTM calculator with k-factors.

        Args:
            factors: CTM k-factors. If None, use defaults.
        """
        self.factors = factors or CTMFactors()

    def calculate(self, cell_pmax_w: float, num_cells: int, cell_configuration: str = "full-cell") -> CTMResult:
        """Calculate module Pmax from cell specifications.

        Args:
            cell_pmax_w: Single cell Pmax in watts
            num_cells: Number of cells in module
            cell_configuration: Cell cutting configuration

        Returns:
            CTMResult with detailed loss breakdown
        """
        # Apply configuration-specific adjustments
        config_factor = 1.0
        if cell_configuration == "half-cut":
            config_factor = 1.0  # Same total power, better performance under shade
        elif cell_configuration == "quarter-cut":
            config_factor = 1.0  # Same total power
        elif cell_configuration == "shingled":
            config_factor = 0.98  # 2% loss for shingling overlap

        # Calculate total cell power
        cell_pmax_total = cell_pmax_w * num_cells * config_factor

        # Calculate module power with CTM losses
        module_pmax = cell_pmax_total * self.factors.ctm_ratio

        # Calculate loss percentages
        optical_loss = (1 - self.factors.optical_loss_total) * 100
        electrical_loss = (1 - self.factors.electrical_loss_total) * 100
        thermal_loss = (1 - self.factors.thermal_loss_total) * 100
        manufacturing_loss = (1 - self.factors.manufacturing_loss_total) * 100
        total_loss = (1 - self.factors.ctm_ratio) * 100

        return CTMResult(
            cell_pmax_total_w=cell_pmax_total,
            module_pmax_w=module_pmax,
            ctm_ratio=self.factors.ctm_ratio,
            total_loss_pct=total_loss,
            optical_loss_pct=optical_loss,
            electrical_loss_pct=electrical_loss,
            thermal_loss_pct=thermal_loss,
            manufacturing_loss_pct=manufacturing_loss,
            waterfall_data=self.factors.get_loss_waterfall(),
        )

    def sensitivity_analysis(
        self, cell_pmax_w: float, num_cells: int, factor_name: str, range_pct: float = 10, steps: int = 20
    ) -> Dict[str, np.ndarray]:
        """Perform sensitivity analysis on a specific k-factor.

        Args:
            cell_pmax_w: Single cell Pmax in watts
            num_cells: Number of cells in module
            factor_name: Name of k-factor to vary
            range_pct: Percentage range to vary (e.g., 10 means ±10%)
            steps: Number of steps in the analysis

        Returns:
            Dictionary with factor values and corresponding module Pmax
        """
        base_value = getattr(self.factors, factor_name)
        min_val = base_value * (1 - range_pct / 100)
        max_val = base_value * (1 + range_pct / 100)

        # Clamp to valid range (0.85 to 1.0 for most factors)
        min_val = max(0.85, min_val)
        max_val = min(1.0, max_val)

        factor_values = np.linspace(min_val, max_val, steps)
        module_pmax_values = []

        for val in factor_values:
            # Create temporary factors with modified value
            temp_factors = self.factors.model_copy()
            setattr(temp_factors, factor_name, val)
            temp_calc = CTMCalculator(temp_factors)
            result = temp_calc.calculate(cell_pmax_w, num_cells)
            module_pmax_values.append(result.module_pmax_w)

        return {"factor_values": factor_values, "module_pmax_values": np.array(module_pmax_values)}
