"""
B03 CTM (Cell-to-Module) Loss Model with k1-k24 factors.

This module implements a comprehensive cell-to-module power loss model
based on IEC 63202 standard and industry best practices. The model
calculates CTM power ratio using 24 individual loss factors (k1-k24).
"""

from typing import Dict, Tuple, Optional
import logging

import numpy as np
from pydantic import BaseModel, Field

from pv_circularity_simulator.core.utils.constants import (
    CTM_LOSS_FACTORS,
    DEFAULT_CTM_SCENARIOS,
    calculate_total_ctm_factor,
)

logger = logging.getLogger(__name__)


class B03CTMConfiguration(BaseModel):
    """Configuration for B03 CTM loss model.

    Attributes:
        k1_cell_binning: Cell binning quality level
        k2_cell_degradation_storage: Cell storage degradation level
        k3_cell_breakage: Cell breakage severity
        k4_measurement_uncertainty_cell: Cell measurement uncertainty class
        k5_cell_temperature_variation: Cell temperature control quality
        k6_ribbon_resistance: Ribbon electrical resistance level
        k7_solder_joint_quality: Solder joint quality level
        k8_busbar_resistance: Busbar resistance level
        k9_cell_mismatch: Cell-to-cell mismatch level
        k10_interconnect_shading: Interconnection shading configuration
        k11_glass_transmission: Glass transmission quality
        k12_encapsulant_transmission: Encapsulant transmission quality
        k13_encapsulant_absorption: Encapsulant absorption level
        k14_backsheet_reflectance: Backsheet reflectance type
        k15_lamination_bubbles_delamination: Lamination defect level
        k16_junction_box_diode_loss: Junction box and diode losses
        k17_frame_shading: Frame shading configuration
        k18_module_edge_effects: Edge effect severity
        k19_thermal_stress_assembly: Thermal stress during assembly
        k20_quality_control_process: Quality control stringency
        k21_flash_simulator_spectrum: Flash simulator quality class
        k22_spatial_uniformity: Spatial uniformity level
        k23_measurement_uncertainty_module: Module measurement uncertainty
        k24_module_temperature_variation: Module temperature control
    """

    # Cell-level factors (k1-k5)
    k1_cell_binning: str = Field(default="medium", description="Cell binning quality")
    k2_cell_degradation_storage: str = Field(default="good", description="Storage degradation")
    k3_cell_breakage: str = Field(default="minor_microcracks", description="Cell breakage")
    k4_measurement_uncertainty_cell: str = Field(default="class_b", description="Cell measurement")
    k5_cell_temperature_variation: str = Field(default="good", description="Cell temperature")

    # Interconnection factors (k6-k10)
    k6_ribbon_resistance: str = Field(default="medium_resistance", description="Ribbon resistance")
    k7_solder_joint_quality: str = Field(default="good", description="Solder quality")
    k8_busbar_resistance: str = Field(default="medium", description="Busbar resistance")
    k9_cell_mismatch: str = Field(default="medium_binning", description="Cell mismatch")
    k10_interconnect_shading: str = Field(default="standard_3bb", description="Interconnect shading")

    # Encapsulation factors (k11-k15)
    k11_glass_transmission: str = Field(default="ar_coated_standard", description="Glass transmission")
    k12_encapsulant_transmission: str = Field(default="eva_high_quality", description="Encapsulant transmission")
    k13_encapsulant_absorption: str = Field(default="medium_absorption", description="Encapsulant absorption")
    k14_backsheet_reflectance: str = Field(default="white_standard", description="Backsheet reflectance")
    k15_lamination_bubbles_delamination: str = Field(default="minor_defects", description="Lamination defects")

    # Module assembly factors (k16-k20)
    k16_junction_box_diode_loss: str = Field(default="medium_voltage_drop", description="Junction box losses")
    k17_frame_shading: str = Field(default="standard_frame", description="Frame shading")
    k18_module_edge_effects: str = Field(default="minor", description="Edge effects")
    k19_thermal_stress_assembly: str = Field(default="medium_stress", description="Thermal stress")
    k20_quality_control_process: str = Field(default="standard", description="Quality control")

    # Measurement factors (k21-k24)
    k21_flash_simulator_spectrum: str = Field(default="class_aba", description="Flash simulator")
    k22_spatial_uniformity: str = Field(default="good", description="Spatial uniformity")
    k23_measurement_uncertainty_module: str = Field(default="class_b", description="Module measurement")
    k24_module_temperature_variation: str = Field(default="good", description="Module temperature")

    @classmethod
    def from_scenario(cls, scenario: str = "standard_quality") -> "B03CTMConfiguration":
        """
        Create configuration from predefined quality scenario.

        Args:
            scenario: Quality scenario name

        Returns:
            Configuration instance with scenario settings

        Raises:
            ValueError: If scenario is unknown
        """
        if scenario not in DEFAULT_CTM_SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Choose from {list(DEFAULT_CTM_SCENARIOS.keys())}"
            )

        return cls(**DEFAULT_CTM_SCENARIOS[scenario])


class B03CTMLossResult(BaseModel):
    """Result of B03 CTM loss calculation.

    Attributes:
        configuration: Input configuration
        individual_factors: Dictionary of individual k factors
        cell_level_factor: Product of k1-k5
        interconnection_factor: Product of k6-k10
        encapsulation_factor: Product of k11-k15
        assembly_factor: Product of k16-k20
        measurement_factor: Product of k21-k24
        total_ctm_factor: Product of all factors (CTM power ratio)
        total_ctm_ratio_percent: CTM ratio as percentage
        total_loss_percent: Total power loss as percentage
    """

    configuration: B03CTMConfiguration = Field(..., description="Input configuration")
    individual_factors: Dict[str, float] = Field(..., description="Individual k factors")
    cell_level_factor: float = Field(..., description="Cell-level factor (k1-k5)")
    interconnection_factor: float = Field(..., description="Interconnection factor (k6-k10)")
    encapsulation_factor: float = Field(..., description="Encapsulation factor (k11-k15)")
    assembly_factor: float = Field(..., description="Assembly factor (k16-k20)")
    measurement_factor: float = Field(..., description="Measurement factor (k21-k24)")
    total_ctm_factor: float = Field(..., description="Total CTM factor")
    total_ctm_ratio_percent: float = Field(..., description="CTM ratio (%)")
    total_loss_percent: float = Field(..., description="Total loss (%)")

    def get_loss_breakdown(self) -> Dict[str, float]:
        """
        Get loss breakdown by category.

        Returns:
            Dictionary with loss percentages by category
        """
        return {
            "cell_level_loss": (1.0 - self.cell_level_factor) * 100,
            "interconnection_loss": (1.0 - self.interconnection_factor) * 100,
            "encapsulation_loss": (1.0 - self.encapsulation_factor) * 100,
            "assembly_loss": (1.0 - self.assembly_factor) * 100,
            "measurement_loss": (1.0 - self.measurement_factor) * 100,
        }


class B03CTMLossModel:
    """
    B03 CTM Loss Model implementation.

    This class implements the comprehensive B03 Cell-to-Module (CTM) loss model
    using 24 individual loss factors (k1-k24) organized into five categories:
    - Cell-level losses (k1-k5)
    - Interconnection losses (k6-k10)
    - Encapsulation losses (k11-k15)
    - Assembly losses (k16-k20)
    - Measurement losses (k21-k24)

    The model calculates the overall CTM power ratio as the product of all factors.
    """

    def __init__(self) -> None:
        """Initialize the B03 CTM loss model."""
        self.loss_factors = CTM_LOSS_FACTORS
        logger.info("B03 CTM Loss Model initialized")

    def calculate_ctm_losses(
        self,
        configuration: B03CTMConfiguration
    ) -> B03CTMLossResult:
        """
        Calculate CTM losses using B03 model with k1-k24 factors.

        Args:
            configuration: CTM configuration specifying all k factors

        Returns:
            Comprehensive CTM loss calculation result

        Example:
            >>> model = B03CTMLossModel()
            >>> config = B03CTMConfiguration.from_scenario("premium_quality")
            >>> result = model.calculate_ctm_losses(config)
            >>> print(f"CTM Ratio: {result.total_ctm_ratio_percent:.2f}%")
            CTM Ratio: 98.25%
        """
        individual_factors = self._get_individual_factors(configuration)

        # Calculate category-level factors
        cell_level = self._calculate_category_factor(individual_factors, "k1", "k5")
        interconnection = self._calculate_category_factor(individual_factors, "k6", "k10")
        encapsulation = self._calculate_category_factor(individual_factors, "k11", "k15")
        assembly = self._calculate_category_factor(individual_factors, "k16", "k20")
        measurement = self._calculate_category_factor(individual_factors, "k21", "k24")

        # Calculate total CTM factor
        total_ctm_factor = (
            cell_level *
            interconnection *
            encapsulation *
            assembly *
            measurement
        )

        # Convert to percentage
        total_ctm_ratio_percent = total_ctm_factor * 100
        total_loss_percent = (1.0 - total_ctm_factor) * 100

        result = B03CTMLossResult(
            configuration=configuration,
            individual_factors=individual_factors,
            cell_level_factor=cell_level,
            interconnection_factor=interconnection,
            encapsulation_factor=encapsulation,
            assembly_factor=assembly,
            measurement_factor=measurement,
            total_ctm_factor=total_ctm_factor,
            total_ctm_ratio_percent=total_ctm_ratio_percent,
            total_loss_percent=total_loss_percent,
        )

        logger.info(
            f"CTM calculation complete: Ratio={total_ctm_ratio_percent:.2f}%, "
            f"Loss={total_loss_percent:.2f}%"
        )

        return result

    def _get_individual_factors(
        self,
        configuration: B03CTMConfiguration
    ) -> Dict[str, float]:
        """
        Extract individual k factor values from configuration.

        Args:
            configuration: CTM configuration

        Returns:
            Dictionary mapping factor names to values

        Raises:
            ValueError: If any factor configuration is invalid
        """
        individual_factors = {}

        for i in range(1, 25):
            factor_name = f"k{i}_{self._get_factor_suffix(i)}"
            quality_level = getattr(configuration, factor_name)

            if factor_name not in self.loss_factors:
                raise ValueError(f"Unknown loss factor: {factor_name}")

            if quality_level not in self.loss_factors[factor_name]:
                raise ValueError(
                    f"Unknown quality level '{quality_level}' for {factor_name}. "
                    f"Valid options: {list(self.loss_factors[factor_name].keys())}"
                )

            factor_value = self.loss_factors[factor_name][quality_level]
            individual_factors[factor_name] = factor_value

        return individual_factors

    def _get_factor_suffix(self, factor_number: int) -> str:
        """
        Get the suffix for a k factor name.

        Args:
            factor_number: Factor number (1-24)

        Returns:
            Factor name suffix
        """
        suffixes = {
            1: "cell_binning",
            2: "cell_degradation_storage",
            3: "cell_breakage",
            4: "measurement_uncertainty_cell",
            5: "cell_temperature_variation",
            6: "ribbon_resistance",
            7: "solder_joint_quality",
            8: "busbar_resistance",
            9: "cell_mismatch",
            10: "interconnect_shading",
            11: "glass_transmission",
            12: "encapsulant_transmission",
            13: "encapsulant_absorption",
            14: "backsheet_reflectance",
            15: "lamination_bubbles_delamination",
            16: "junction_box_diode_loss",
            17: "frame_shading",
            18: "module_edge_effects",
            19: "thermal_stress_assembly",
            20: "quality_control_process",
            21: "flash_simulator_spectrum",
            22: "spatial_uniformity",
            23: "measurement_uncertainty_module",
            24: "module_temperature_variation",
        }
        return suffixes[factor_number]

    def _calculate_category_factor(
        self,
        individual_factors: Dict[str, float],
        start_prefix: str,
        end_prefix: str
    ) -> float:
        """
        Calculate product of factors in a category range.

        Args:
            individual_factors: All individual factors
            start_prefix: Starting factor prefix (e.g., "k1")
            end_prefix: Ending factor prefix (e.g., "k5")

        Returns:
            Product of factors in range
        """
        start_num = int(start_prefix[1:])
        end_num = int(end_prefix[1:])

        category_factor = 1.0
        for i in range(start_num, end_num + 1):
            factor_name = f"k{i}_{self._get_factor_suffix(i)}"
            category_factor *= individual_factors[factor_name]

        return category_factor

    def compare_scenarios(
        self,
        scenarios: Optional[list[str]] = None
    ) -> Dict[str, B03CTMLossResult]:
        """
        Compare CTM losses across multiple quality scenarios.

        Args:
            scenarios: List of scenario names to compare.
                      If None, compares all default scenarios.

        Returns:
            Dictionary mapping scenario names to results

        Example:
            >>> model = B03CTMLossModel()
            >>> comparison = model.compare_scenarios()
            >>> for name, result in comparison.items():
            ...     print(f"{name}: {result.total_ctm_ratio_percent:.2f}%")
            premium_quality: 98.25%
            standard_quality: 96.50%
            economy_quality: 94.20%
        """
        if scenarios is None:
            scenarios = list(DEFAULT_CTM_SCENARIOS.keys())

        results = {}
        for scenario in scenarios:
            config = B03CTMConfiguration.from_scenario(scenario)
            result = self.calculate_ctm_losses(config)
            results[scenario] = result

        return results

    def sensitivity_analysis(
        self,
        base_configuration: B03CTMConfiguration,
        factor_to_vary: str,
        quality_levels: Optional[list[str]] = None
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis by varying a single factor.

        Args:
            base_configuration: Base configuration
            factor_to_vary: Name of factor to vary (e.g., "k1_cell_binning")
            quality_levels: List of quality levels to test.
                           If None, tests all available levels for the factor.

        Returns:
            Dictionary mapping quality levels to CTM ratios

        Raises:
            ValueError: If factor name is invalid

        Example:
            >>> model = B03CTMLossModel()
            >>> config = B03CTMConfiguration.from_scenario("standard_quality")
            >>> sensitivity = model.sensitivity_analysis(
            ...     config,
            ...     "k10_interconnect_shading"
            ... )
            >>> for level, ratio in sensitivity.items():
            ...     print(f"{level}: {ratio:.2f}%")
            mbb_5bb: 97.0%
            standard_3bb: 96.5%
            conventional_2bb: 96.0%
        """
        if factor_to_vary not in self.loss_factors:
            raise ValueError(
                f"Unknown factor: {factor_to_vary}. "
                f"Valid factors: {list(self.loss_factors.keys())}"
            )

        if quality_levels is None:
            quality_levels = list(self.loss_factors[factor_to_vary].keys())

        results = {}
        for quality_level in quality_levels:
            # Create modified configuration
            config_dict = base_configuration.model_dump()
            config_dict[factor_to_vary] = quality_level
            modified_config = B03CTMConfiguration(**config_dict)

            # Calculate CTM with modified factor
            result = self.calculate_ctm_losses(modified_config)
            results[quality_level] = result.total_ctm_ratio_percent

        return results
