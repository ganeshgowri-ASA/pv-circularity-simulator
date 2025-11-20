"""
Griddler Pro Integration & Metallization Optimization Module

This module provides comprehensive metallization pattern design and optimization
for photovoltaic solar cells, including advanced grid patterns, multi-busbar
configurations, and cost analysis.

Author: PV Circularity Simulator Team
Date: 2025-11-17
"""

from typing import Dict, List, Optional, Tuple, Literal, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator, ConfigDict
import json
import math


# ============================================================================
# Enumerations
# ============================================================================

class MetallizationType(str, Enum):
    """Types of metallization processes."""
    SCREEN_PRINTING = "screen_printing"
    COPPER_PLATING = "copper_plating"
    EVAPORATION = "evaporation"
    ELECTROLESS_PLATING = "electroless_plating"


class GridPatternType(str, Enum):
    """Types of grid patterns."""
    STANDARD_H_PATTERN = "standard_h_pattern"
    MULTI_BUSBAR = "multi_busbar"
    SHINGLED = "shingled"
    SMARTWIRE = "smartwire"
    IBC = "ibc"  # Interdigitated Back Contact
    BIFACIAL = "bifacial"
    HALF_CUT = "half_cut"


class BusbarConfiguration(str, Enum):
    """Busbar configurations."""
    BB2 = "2BB"
    BB3 = "3BB"
    BB4 = "4BB"
    BB5 = "5BB"
    BB6 = "6BB"
    BB9 = "9BB"
    BB12 = "12BB"
    BB16 = "16BB"
    MBB = "MBB"  # Multi-busbar (>16)


class OptimizationObjective(str, Enum):
    """Optimization objectives for metallization."""
    MINIMIZE_RESISTANCE = "minimize_resistance"
    MINIMIZE_SHADING = "minimize_shading"
    MINIMIZE_SILVER = "minimize_silver"
    MAXIMIZE_FILL_FACTOR = "maximize_fill_factor"
    BALANCED = "balanced"


# ============================================================================
# Pydantic Models
# ============================================================================

class MetallizationParameters(BaseModel):
    """Parameters defining the metallization design."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Finger parameters
    finger_width: float = Field(
        default=50.0,
        ge=20.0,
        le=200.0,
        description="Finger width in micrometers"
    )
    finger_spacing: float = Field(
        default=2000.0,
        ge=500.0,
        le=5000.0,
        description="Spacing between fingers in micrometers"
    )
    finger_count: int = Field(
        default=100,
        ge=20,
        le=300,
        description="Number of fingers"
    )
    finger_height: float = Field(
        default=15.0,
        ge=5.0,
        le=50.0,
        description="Finger height in micrometers"
    )
    finger_aspect_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Finger aspect ratio (height/width)"
    )

    # Busbar parameters
    busbar_width: float = Field(
        default=1500.0,
        ge=500.0,
        le=3000.0,
        description="Busbar width in micrometers"
    )
    busbar_count: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Number of busbars"
    )
    busbar_height: float = Field(
        default=20.0,
        ge=10.0,
        le=60.0,
        description="Busbar height in micrometers"
    )

    # Material properties
    contact_resistance: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Contact resistance in mOhm·cm²"
    )
    sheet_resistance: float = Field(
        default=50.0,
        ge=20.0,
        le=200.0,
        description="Sheet resistance of emitter in Ohm/sq"
    )
    silver_paste_resistivity: float = Field(
        default=3.0e-6,
        ge=1.0e-6,
        le=10.0e-6,
        description="Silver paste resistivity in Ohm·cm"
    )
    silver_paste_density: float = Field(
        default=10.49,
        ge=8.0,
        le=12.0,
        description="Silver paste density in g/cm³"
    )

    # Cell dimensions
    cell_width: float = Field(
        default=156.75,
        ge=100.0,
        le=210.0,
        description="Cell width in mm"
    )
    cell_length: float = Field(
        default=156.75,
        ge=100.0,
        le=210.0,
        description="Cell length in mm"
    )

    @validator('finger_aspect_ratio')
    def validate_aspect_ratio(cls, v, values):
        """Ensure aspect ratio is consistent with height/width."""
        if 'finger_height' in values and 'finger_width' in values:
            calculated_ratio = values['finger_height'] / values['finger_width']
            if abs(calculated_ratio - v) > 0.1:
                return calculated_ratio
        return v


class GridPattern(BaseModel):
    """Represents a complete grid pattern design."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern_type: GridPatternType = Field(
        description="Type of grid pattern"
    )
    busbar_config: BusbarConfiguration = Field(
        description="Busbar configuration"
    )

    # Geometric parameters
    finger_positions: List[float] = Field(
        default_factory=list,
        description="Y-positions of fingers in mm"
    )
    busbar_positions: List[float] = Field(
        default_factory=list,
        description="X-positions of busbars in mm"
    )

    # Calculated properties
    total_finger_length: float = Field(
        default=0.0,
        description="Total finger length in mm"
    )
    total_busbar_length: float = Field(
        default=0.0,
        description="Total busbar length in mm"
    )
    shading_area: float = Field(
        default=0.0,
        description="Total shading area in mm²"
    )
    shading_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of cell area shaded"
    )

    # Electrical properties
    series_resistance: float = Field(
        default=0.0,
        description="Total series resistance in Ohm·cm²"
    )
    finger_resistance: float = Field(
        default=0.0,
        description="Finger resistance contribution in Ohm·cm²"
    )
    busbar_resistance: float = Field(
        default=0.0,
        description="Busbar resistance contribution in Ohm·cm²"
    )
    contact_resistance_total: float = Field(
        default=0.0,
        description="Contact resistance contribution in Ohm·cm²"
    )
    emitter_resistance: float = Field(
        default=0.0,
        description="Emitter resistance contribution in Ohm·cm²"
    )

    # Material usage
    silver_mass: float = Field(
        default=0.0,
        description="Total silver mass in mg"
    )

    # Performance metrics
    fill_factor_loss: float = Field(
        default=0.0,
        description="Fill factor loss due to metallization"
    )
    efficiency_loss: float = Field(
        default=0.0,
        description="Efficiency loss percentage"
    )


class OptimizedPattern(BaseModel):
    """Optimized metallization pattern with performance metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern: GridPattern = Field(
        description="Optimized grid pattern"
    )
    parameters: MetallizationParameters = Field(
        description="Metallization parameters used"
    )

    # Optimization results
    objective_value: float = Field(
        description="Optimization objective function value"
    )
    optimization_objective: OptimizationObjective = Field(
        description="Optimization objective used"
    )

    # Performance breakdown
    optical_efficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="Optical efficiency (1 - shading loss)"
    )
    electrical_efficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="Electrical efficiency (accounting for resistance)"
    )
    combined_efficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined optical and electrical efficiency"
    )

    # Cost metrics
    silver_cost_per_cell: float = Field(
        default=0.0,
        description="Silver cost per cell in USD"
    )
    processing_cost_per_cell: float = Field(
        default=0.0,
        description="Processing cost per cell in USD"
    )

    # Trade-off metrics
    performance_to_cost_ratio: float = Field(
        default=0.0,
        description="Performance to cost ratio"
    )


class CostAnalysis(BaseModel):
    """Cost analysis for metallization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metallization_type: MetallizationType = Field(
        description="Type of metallization process"
    )

    # Material costs
    silver_mass_mg: float = Field(
        ge=0.0,
        description="Silver mass in mg per cell"
    )
    silver_price_per_gram: float = Field(
        default=0.75,
        ge=0.0,
        description="Silver price in USD per gram"
    )
    silver_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total silver cost in USD per cell"
    )

    # Processing costs
    screen_printing_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Screen printing cost in USD per cell"
    )
    firing_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Firing cost in USD per cell"
    )
    alternative_process_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Alternative metallization cost in USD per cell"
    )

    # Total costs
    total_material_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total material cost in USD per cell"
    )
    total_processing_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing cost in USD per cell"
    )
    total_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total metallization cost in USD per cell"
    )

    # Screen printing parameters
    paste_consumption_mg_per_cell: float = Field(
        default=0.0,
        ge=0.0,
        description="Total paste consumption in mg per cell"
    )
    screens_per_hour: float = Field(
        default=3000.0,
        ge=0.0,
        description="Screen printing throughput in cells per hour"
    )
    firing_temperature: float = Field(
        default=800.0,
        ge=600.0,
        le=1000.0,
        description="Firing temperature in Celsius"
    )
    firing_time_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Firing time in seconds"
    )


class CADExport(BaseModel):
    """CAD export format specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    format: Literal["DXF", "GDSII", "SVG", "JSON"] = Field(
        description="Export format"
    )
    pattern: GridPattern = Field(
        description="Grid pattern to export"
    )
    layer_mapping: Dict[str, int] = Field(
        default_factory=lambda: {"fingers": 1, "busbars": 2},
        description="Layer mapping for CAD export"
    )
    units: Literal["mm", "um", "nm"] = Field(
        default="mm",
        description="Units for export"
    )
    include_annotations: bool = Field(
        default=True,
        description="Include dimensional annotations"
    )


# ============================================================================
# GriddlerInterface Class
# ============================================================================

class GriddlerInterface:
    """
    Main interface for Griddler Pro metallization design and optimization.

    This class provides comprehensive functionality for designing, optimizing,
    and analyzing metallization patterns for photovoltaic solar cells.
    """

    def __init__(
        self,
        default_params: Optional[MetallizationParameters] = None,
        silver_price_per_gram: float = 0.75
    ):
        """
        Initialize the GriddlerInterface.

        Args:
            default_params: Default metallization parameters
            silver_price_per_gram: Current silver price in USD per gram
        """
        self.default_params = default_params or MetallizationParameters()
        self.silver_price = silver_price_per_gram

    def design_finger_pattern(
        self,
        cell_params: Dict[str, Any]
    ) -> GridPattern:
        """
        Design the finger pattern for a solar cell.

        This method calculates the optimal finger positions and configuration
        based on cell parameters and current operating point.

        Args:
            cell_params: Dictionary containing:
                - cell_width: Cell width in mm
                - cell_length: Cell length in mm
                - finger_count: Number of fingers
                - finger_width: Finger width in um
                - busbar_count: Number of busbars
                - busbar_width: Busbar width in um
                - pattern_type: Type of grid pattern (optional)
                - busbar_config: Busbar configuration (optional)

        Returns:
            GridPattern object with complete finger design
        """
        # Extract parameters
        cell_width = cell_params.get('cell_width', self.default_params.cell_width)
        cell_length = cell_params.get('cell_length', self.default_params.cell_length)
        finger_count = cell_params.get('finger_count', self.default_params.finger_count)
        finger_width = cell_params.get('finger_width', self.default_params.finger_width)
        busbar_count = cell_params.get('busbar_count', self.default_params.busbar_count)
        busbar_width = cell_params.get('busbar_width', self.default_params.busbar_width)

        pattern_type = cell_params.get(
            'pattern_type',
            GridPatternType.STANDARD_H_PATTERN
        )
        busbar_config = cell_params.get(
            'busbar_config',
            self._get_busbar_config(busbar_count)
        )

        # Calculate finger positions (evenly distributed along cell width)
        finger_spacing = cell_width / (finger_count + 1)
        finger_positions = [
            finger_spacing * (i + 1) for i in range(finger_count)
        ]

        # Calculate busbar positions (evenly distributed along cell length)
        if busbar_count == 2:
            busbar_positions = [busbar_width / 2000, cell_length - busbar_width / 2000]
        else:
            busbar_spacing = cell_length / (busbar_count - 1)
            busbar_positions = [
                busbar_spacing * i for i in range(busbar_count)
            ]

        # Calculate total lengths
        total_finger_length = finger_count * cell_length
        total_busbar_length = busbar_count * cell_width

        # Calculate shading area
        finger_shading = finger_count * (finger_width / 1000) * cell_length
        busbar_shading = busbar_count * (busbar_width / 1000) * cell_width
        total_shading_area = finger_shading + busbar_shading
        cell_area = cell_width * cell_length
        shading_fraction = total_shading_area / cell_area

        pattern = GridPattern(
            pattern_type=pattern_type,
            busbar_config=busbar_config,
            finger_positions=finger_positions,
            busbar_positions=busbar_positions,
            total_finger_length=total_finger_length,
            total_busbar_length=total_busbar_length,
            shading_area=total_shading_area,
            shading_fraction=shading_fraction
        )

        return pattern

    def optimize_busbar_width(
        self,
        params: Dict[str, Any]
    ) -> float:
        """
        Optimize busbar width to minimize total losses.

        This method finds the optimal busbar width that balances shading losses
        and resistive losses.

        Args:
            params: Dictionary containing:
                - cell_width: Cell width in mm
                - cell_length: Cell length in mm
                - busbar_count: Number of busbars
                - current_density: Operating current density in A/cm²
                - voltage: Operating voltage in V
                - resistivity: Busbar material resistivity in Ohm·cm
                - height: Busbar height in um

        Returns:
            Optimal busbar width in micrometers
        """
        cell_width = params.get('cell_width', self.default_params.cell_width)
        cell_length = params.get('cell_length', self.default_params.cell_length)
        busbar_count = params.get('busbar_count', self.default_params.busbar_count)
        current_density = params.get('current_density', 0.04)  # A/cm²
        voltage = params.get('voltage', 0.65)  # V
        resistivity = params.get('resistivity', self.default_params.silver_paste_resistivity)
        height = params.get('height', self.default_params.busbar_height)

        # Cell area in cm²
        cell_area_cm2 = (cell_width * cell_length) / 100

        # Total current per busbar
        current_per_busbar = (current_density * cell_area_cm2) / busbar_count

        # Optimize busbar width using analytical approach
        # Power loss = shading loss + resistive loss
        # dP/dW = 0 for optimal width

        def total_power_loss(width_um):
            """Calculate total power loss for given busbar width."""
            width_cm = width_um / 10000

            # Shading loss
            shading_area = busbar_count * (width_cm / 10) * cell_width
            shading_loss = (shading_area / (cell_width * cell_length)) * \
                          (current_density * cell_area_cm2 * voltage)

            # Resistive loss
            busbar_length_cm = cell_width / 10
            cross_section = width_cm * (height / 10000)
            resistance = resistivity * busbar_length_cm / cross_section
            resistive_loss = current_per_busbar**2 * resistance * busbar_count

            return shading_loss + resistive_loss

        # Search for optimal width between 500 and 3000 um
        widths = np.linspace(500, 3000, 100)
        losses = [total_power_loss(w) for w in widths]
        optimal_idx = np.argmin(losses)
        optimal_width = widths[optimal_idx]

        return float(optimal_width)

    def calculate_shading_losses(
        self,
        pattern: GridPattern
    ) -> float:
        """
        Calculate shading losses for a given grid pattern.

        Args:
            pattern: Grid pattern to analyze

        Returns:
            Shading loss as a fraction of total power (0-1)
        """
        # Shading loss is approximately equal to the shading fraction
        # for a uniform illumination
        return pattern.shading_fraction

    def calculate_series_resistance(
        self,
        pattern: GridPattern,
        params: Optional[MetallizationParameters] = None
    ) -> float:
        """
        Calculate total series resistance for a grid pattern.

        This method computes the series resistance including contributions from:
        - Fingers
        - Busbars
        - Contact resistance
        - Emitter sheet resistance

        Args:
            pattern: Grid pattern to analyze
            params: Metallization parameters (uses default if None)

        Returns:
            Total series resistance in Ohm·cm²
        """
        if params is None:
            params = self.default_params

        cell_area_cm2 = (params.cell_width * params.cell_length) / 100

        # Finger resistance
        # R_finger = ρ * L / A, where A = width * height
        finger_width_cm = params.finger_width / 10000
        finger_height_cm = params.finger_height / 10000
        finger_cross_section = finger_width_cm * finger_height_cm

        finger_length_cm = params.cell_length / 10
        single_finger_resistance = (
            params.silver_paste_resistivity * finger_length_cm / finger_cross_section
        )

        # Effective finger resistance considering current collection
        # (current increases along finger)
        finger_resistance = single_finger_resistance / 3 * params.finger_count

        # Busbar resistance
        busbar_width_cm = params.busbar_width / 10000
        busbar_height_cm = params.busbar_height / 10000
        busbar_cross_section = busbar_width_cm * busbar_height_cm

        busbar_length_cm = params.cell_width / 10
        single_busbar_resistance = (
            params.silver_paste_resistivity * busbar_length_cm / busbar_cross_section
        )

        # Effective busbar resistance (parallel busbars)
        busbar_resistance = single_busbar_resistance / params.busbar_count / 3

        # Contact resistance
        # Distributed over the finger contact area
        contact_area = (
            params.finger_count * (params.finger_width / 10000) *
            (params.cell_length / 10)
        )
        contact_resistance = params.contact_resistance * cell_area_cm2 / contact_area

        # Emitter resistance
        # Resistance between fingers
        finger_spacing_cm = params.finger_spacing / 10000
        emitter_resistance = (
            params.sheet_resistance * finger_spacing_cm**2 / (12 * cell_area_cm2)
        )

        # Total series resistance
        total_resistance = (
            finger_resistance + busbar_resistance +
            contact_resistance + emitter_resistance
        )

        # Update pattern with detailed breakdown
        pattern.finger_resistance = finger_resistance
        pattern.busbar_resistance = busbar_resistance
        pattern.contact_resistance_total = contact_resistance
        pattern.emitter_resistance = emitter_resistance
        pattern.series_resistance = total_resistance

        return total_resistance

    def optimize_metallization(
        self,
        cell_design: Dict[str, Any],
        objective: OptimizationObjective = OptimizationObjective.BALANCED
    ) -> OptimizedPattern:
        """
        Optimize metallization pattern for given cell design and objective.

        This method performs multi-objective optimization considering:
        - Series resistance minimization
        - Shading loss minimization
        - Silver consumption minimization
        - Fill factor maximization

        Args:
            cell_design: Dictionary containing:
                - cell_width: Cell width in mm
                - cell_length: Cell length in mm
                - jsc: Short circuit current density in A/cm²
                - voc: Open circuit voltage in V
                - target_efficiency: Target efficiency (optional)
                - max_silver_mg: Maximum silver consumption in mg (optional)
                - busbar_config: Desired busbar configuration (optional)
            objective: Optimization objective

        Returns:
            OptimizedPattern with optimized design and performance metrics
        """
        # Extract cell parameters
        cell_width = cell_design.get('cell_width', 156.75)
        cell_length = cell_design.get('cell_length', 156.75)
        jsc = cell_design.get('jsc', 0.042)  # A/cm²
        voc = cell_design.get('voc', 0.68)  # V

        cell_area_cm2 = (cell_width * cell_length) / 100

        # Optimization bounds
        finger_width_range = np.linspace(30, 100, 15)
        finger_count_range = np.linspace(60, 150, 15).astype(int)
        busbar_width_range = np.linspace(800, 2000, 10)

        busbar_config = cell_design.get('busbar_config', BusbarConfiguration.BB5)
        busbar_count = self._get_busbar_count(busbar_config)

        best_score = float('inf')
        best_params = None
        best_pattern = None

        # Grid search optimization
        for finger_width in finger_width_range:
            for finger_count in finger_count_range:
                for busbar_width in busbar_width_range:
                    # Create test parameters
                    test_params = MetallizationParameters(
                        finger_width=finger_width,
                        finger_count=int(finger_count),
                        finger_spacing=cell_width * 1000 / (finger_count + 1),
                        busbar_width=busbar_width,
                        busbar_count=busbar_count,
                        cell_width=cell_width,
                        cell_length=cell_length
                    )

                    # Design pattern
                    pattern = self.design_finger_pattern({
                        'cell_width': cell_width,
                        'cell_length': cell_length,
                        'finger_count': int(finger_count),
                        'finger_width': finger_width,
                        'busbar_count': busbar_count,
                        'busbar_width': busbar_width,
                        'busbar_config': busbar_config
                    })

                    # Calculate resistance
                    rs = self.calculate_series_resistance(pattern, test_params)

                    # Calculate shading loss
                    shading_loss = self.calculate_shading_losses(pattern)

                    # Calculate silver consumption
                    silver_mg = self._calculate_silver_mass(pattern, test_params)
                    pattern.silver_mass = silver_mg

                    # Calculate fill factor loss
                    ff_loss = self._calculate_ff_loss(rs, jsc, voc)
                    pattern.fill_factor_loss = ff_loss

                    # Calculate objective function
                    score = self._calculate_objective(
                        objective, rs, shading_loss, silver_mg, ff_loss
                    )

                    if score < best_score:
                        best_score = score
                        best_params = test_params
                        best_pattern = pattern

        # Calculate performance metrics
        optical_efficiency = 1 - best_pattern.shading_fraction
        electrical_efficiency = 1 - best_pattern.fill_factor_loss
        combined_efficiency = optical_efficiency * electrical_efficiency

        # Calculate costs
        cost_analysis = self.calculate_cost_analysis(
            best_pattern,
            best_params,
            MetallizationType.SCREEN_PRINTING
        )

        optimized = OptimizedPattern(
            pattern=best_pattern,
            parameters=best_params,
            objective_value=best_score,
            optimization_objective=objective,
            optical_efficiency=optical_efficiency,
            electrical_efficiency=electrical_efficiency,
            combined_efficiency=combined_efficiency,
            silver_cost_per_cell=cost_analysis.silver_cost,
            processing_cost_per_cell=cost_analysis.total_processing_cost,
            performance_to_cost_ratio=combined_efficiency / cost_analysis.total_cost
            if cost_analysis.total_cost > 0 else 0.0
        )

        return optimized

    def generate_advanced_pattern(
        self,
        pattern_type: GridPatternType,
        cell_params: Dict[str, Any]
    ) -> GridPattern:
        """
        Generate advanced metallization patterns.

        Supports:
        - Multi-busbar (MBB 9-16BB)
        - Shingled cell interconnection
        - Smartwire connection technology
        - IBC finger patterns
        - Bifacial optimization

        Args:
            pattern_type: Type of advanced pattern
            cell_params: Cell parameters and specifications

        Returns:
            GridPattern with advanced configuration
        """
        if pattern_type == GridPatternType.MULTI_BUSBAR:
            return self._generate_mbb_pattern(cell_params)
        elif pattern_type == GridPatternType.SHINGLED:
            return self._generate_shingled_pattern(cell_params)
        elif pattern_type == GridPatternType.SMARTWIRE:
            return self._generate_smartwire_pattern(cell_params)
        elif pattern_type == GridPatternType.IBC:
            return self._generate_ibc_pattern(cell_params)
        elif pattern_type == GridPatternType.BIFACIAL:
            return self._generate_bifacial_pattern(cell_params)
        elif pattern_type == GridPatternType.HALF_CUT:
            return self._generate_half_cut_pattern(cell_params)
        else:
            return self.design_finger_pattern(cell_params)

    def calculate_cost_analysis(
        self,
        pattern: GridPattern,
        params: MetallizationParameters,
        metallization_type: MetallizationType
    ) -> CostAnalysis:
        """
        Perform comprehensive cost analysis for metallization.

        Args:
            pattern: Grid pattern to analyze
            params: Metallization parameters
            metallization_type: Type of metallization process

        Returns:
            CostAnalysis with detailed cost breakdown
        """
        # Calculate silver mass
        silver_mass_mg = self._calculate_silver_mass(pattern, params)

        # Calculate silver cost
        silver_cost = (silver_mass_mg / 1000) * self.silver_price

        # Calculate processing costs
        if metallization_type == MetallizationType.SCREEN_PRINTING:
            screen_printing_cost = 0.015  # USD per cell
            firing_cost = 0.008  # USD per cell
            alternative_cost = 0.0
        elif metallization_type == MetallizationType.COPPER_PLATING:
            screen_printing_cost = 0.010
            firing_cost = 0.005
            alternative_cost = 0.012  # Plating cost
        else:
            screen_printing_cost = 0.015
            firing_cost = 0.008
            alternative_cost = 0.0

        total_processing = screen_printing_cost + firing_cost + alternative_cost
        total_cost = silver_cost + total_processing

        analysis = CostAnalysis(
            metallization_type=metallization_type,
            silver_mass_mg=silver_mass_mg,
            silver_price_per_gram=self.silver_price,
            silver_cost=silver_cost,
            screen_printing_cost=screen_printing_cost,
            firing_cost=firing_cost,
            alternative_process_cost=alternative_cost,
            total_material_cost=silver_cost,
            total_processing_cost=total_processing,
            total_cost=total_cost,
            paste_consumption_mg_per_cell=silver_mass_mg * 1.15  # Account for waste
        )

        return analysis

    def export_to_cad(
        self,
        pattern: GridPattern,
        params: MetallizationParameters,
        export_format: Literal["DXF", "GDSII", "SVG", "JSON"]
    ) -> str:
        """
        Export grid pattern to CAD format.

        Args:
            pattern: Grid pattern to export
            params: Metallization parameters
            export_format: Target CAD format

        Returns:
            CAD data as string (format-specific)
        """
        export = CADExport(
            format=export_format,
            pattern=pattern,
            layer_mapping={"fingers": 1, "busbars": 2},
            units="mm"
        )

        if export_format == "JSON":
            return self._export_json(pattern, params)
        elif export_format == "SVG":
            return self._export_svg(pattern, params)
        elif export_format == "DXF":
            return self._export_dxf(pattern, params)
        elif export_format == "GDSII":
            return self._export_gdsii(pattern, params)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _get_busbar_config(self, count: int) -> BusbarConfiguration:
        """Map busbar count to configuration enum."""
        mapping = {
            2: BusbarConfiguration.BB2,
            3: BusbarConfiguration.BB3,
            4: BusbarConfiguration.BB4,
            5: BusbarConfiguration.BB5,
            6: BusbarConfiguration.BB6,
            9: BusbarConfiguration.BB9,
            12: BusbarConfiguration.BB12,
            16: BusbarConfiguration.BB16,
        }
        return mapping.get(count, BusbarConfiguration.MBB)

    def _get_busbar_count(self, config: BusbarConfiguration) -> int:
        """Map busbar configuration to count."""
        mapping = {
            BusbarConfiguration.BB2: 2,
            BusbarConfiguration.BB3: 3,
            BusbarConfiguration.BB4: 4,
            BusbarConfiguration.BB5: 5,
            BusbarConfiguration.BB6: 6,
            BusbarConfiguration.BB9: 9,
            BusbarConfiguration.BB12: 12,
            BusbarConfiguration.BB16: 16,
            BusbarConfiguration.MBB: 20,
        }
        return mapping.get(config, 5)

    def _calculate_silver_mass(
        self,
        pattern: GridPattern,
        params: MetallizationParameters
    ) -> float:
        """
        Calculate total silver mass in mg.

        Args:
            pattern: Grid pattern
            params: Metallization parameters

        Returns:
            Silver mass in mg
        """
        # Finger volume
        finger_width_cm = params.finger_width / 10000
        finger_height_cm = params.finger_height / 10000
        finger_length_cm = pattern.total_finger_length / 10
        finger_volume = finger_width_cm * finger_height_cm * finger_length_cm

        # Busbar volume
        busbar_width_cm = params.busbar_width / 10000
        busbar_height_cm = params.busbar_height / 10000
        busbar_length_cm = pattern.total_busbar_length / 10
        busbar_volume = busbar_width_cm * busbar_height_cm * busbar_length_cm

        # Total volume in cm³
        total_volume = finger_volume + busbar_volume

        # Mass = volume × density (convert to mg)
        mass_mg = total_volume * params.silver_paste_density * 1000

        return mass_mg

    def _calculate_ff_loss(
        self,
        rs: float,
        jsc: float,
        voc: float
    ) -> float:
        """
        Calculate fill factor loss due to series resistance.

        Args:
            rs: Series resistance in Ohm·cm²
            jsc: Short circuit current density in A/cm²
            voc: Open circuit voltage in V

        Returns:
            Fill factor loss (fractional)
        """
        # Normalized series resistance
        rs_normalized = rs * jsc / voc

        # Fill factor loss using Green's approximation
        ff_loss = rs_normalized * (1 - 1.1 * rs_normalized)

        return max(0, min(1, ff_loss))

    def _calculate_objective(
        self,
        objective: OptimizationObjective,
        rs: float,
        shading: float,
        silver_mg: float,
        ff_loss: float
    ) -> float:
        """Calculate objective function value."""
        if objective == OptimizationObjective.MINIMIZE_RESISTANCE:
            return rs
        elif objective == OptimizationObjective.MINIMIZE_SHADING:
            return shading
        elif objective == OptimizationObjective.MINIMIZE_SILVER:
            return silver_mg
        elif objective == OptimizationObjective.MAXIMIZE_FILL_FACTOR:
            return ff_loss
        else:  # BALANCED
            # Weighted sum of normalized objectives
            rs_norm = rs / 1.0  # Normalize to typical value
            shading_norm = shading / 0.05
            silver_norm = silver_mg / 100
            ff_norm = ff_loss / 0.05

            return 0.3 * rs_norm + 0.3 * shading_norm + 0.2 * silver_norm + 0.2 * ff_norm

    def _generate_mbb_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate multi-busbar pattern (9-16 busbars)."""
        busbar_count = cell_params.get('busbar_count', 12)
        finger_count = cell_params.get('finger_count', 120)

        # MBB typically uses thinner busbars and more of them
        cell_params['busbar_width'] = cell_params.get('busbar_width', 400)
        cell_params['busbar_config'] = self._get_busbar_config(busbar_count)

        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.MULTI_BUSBAR

        return pattern

    def _generate_shingled_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate shingled cell pattern."""
        # Shingled cells have no busbars, only fingers
        cell_params['busbar_count'] = 0
        cell_params['busbar_width'] = 0
        cell_params['finger_count'] = cell_params.get('finger_count', 20)

        # Fingers run along the short edge
        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.SHINGLED
        pattern.busbar_config = BusbarConfiguration.BB2  # Placeholder

        return pattern

    def _generate_smartwire_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate SmartWire connection pattern."""
        # SmartWire uses many thin wires instead of busbars
        cell_params['busbar_count'] = cell_params.get('wire_count', 12)
        cell_params['busbar_width'] = cell_params.get('wire_diameter', 300)

        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.SMARTWIRE

        return pattern

    def _generate_ibc_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate interdigitated back contact pattern."""
        # IBC has alternating n+ and p+ regions on the back
        finger_count = cell_params.get('finger_count', 60)

        # Double the fingers for interdigitated pattern
        cell_params['finger_count'] = finger_count * 2
        cell_params['busbar_count'] = 2  # One for each polarity

        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.IBC

        # IBC has no front-side shading
        pattern.shading_fraction = 0.0
        pattern.shading_area = 0.0

        return pattern

    def _generate_bifacial_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate bifacial cell pattern."""
        # Bifacial cells optimize for both front and rear light capture
        cell_params['busbar_width'] = cell_params.get('busbar_width', 800)
        cell_params['finger_width'] = cell_params.get('finger_width', 40)

        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.BIFACIAL

        return pattern

    def _generate_half_cut_pattern(self, cell_params: Dict[str, Any]) -> GridPattern:
        """Generate half-cut cell pattern."""
        # Half-cut cells are essentially two cells in series
        cell_length = cell_params.get('cell_length', 156.75)
        cell_params['cell_length'] = cell_length / 2

        pattern = self.design_finger_pattern(cell_params)
        pattern.pattern_type = GridPatternType.HALF_CUT

        return pattern

    def _export_json(
        self,
        pattern: GridPattern,
        params: MetallizationParameters
    ) -> str:
        """Export pattern to JSON format."""
        data = {
            "pattern_type": pattern.pattern_type.value,
            "busbar_config": pattern.busbar_config.value,
            "cell_dimensions": {
                "width_mm": params.cell_width,
                "length_mm": params.cell_length
            },
            "fingers": {
                "count": params.finger_count,
                "width_um": params.finger_width,
                "height_um": params.finger_height,
                "positions_mm": pattern.finger_positions
            },
            "busbars": {
                "count": params.busbar_count,
                "width_um": params.busbar_width,
                "height_um": params.busbar_height,
                "positions_mm": pattern.busbar_positions
            },
            "performance": {
                "shading_fraction": pattern.shading_fraction,
                "series_resistance_ohm_cm2": pattern.series_resistance,
                "silver_mass_mg": pattern.silver_mass
            }
        }

        return json.dumps(data, indent=2)

    def _export_svg(
        self,
        pattern: GridPattern,
        params: MetallizationParameters
    ) -> str:
        """Export pattern to SVG format."""
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg width="{params.cell_width}mm" height="{params.cell_length}mm" '
            'xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.finger { fill: silver; stroke: none; }',
            '.busbar { fill: gray; stroke: none; }',
            '</style>',
            '</defs>'
        ]

        # Draw fingers
        finger_width_mm = params.finger_width / 1000
        for y_pos in pattern.finger_positions:
            svg_parts.append(
                f'<rect class="finger" x="0" y="{y_pos - finger_width_mm/2}" '
                f'width="{params.cell_width}" height="{finger_width_mm}"/>'
            )

        # Draw busbars
        busbar_width_mm = params.busbar_width / 1000
        for x_pos in pattern.busbar_positions:
            svg_parts.append(
                f'<rect class="busbar" x="{x_pos - busbar_width_mm/2}" y="0" '
                f'width="{busbar_width_mm}" height="{params.cell_length}"/>'
            )

        svg_parts.append('</svg>')

        return '\n'.join(svg_parts)

    def _export_dxf(
        self,
        pattern: GridPattern,
        params: MetallizationParameters
    ) -> str:
        """Export pattern to DXF format (simplified)."""
        dxf_parts = [
            "0",
            "SECTION",
            "2",
            "ENTITIES"
        ]

        # Draw fingers as lines
        finger_width_mm = params.finger_width / 1000
        for y_pos in pattern.finger_positions:
            dxf_parts.extend([
                "0",
                "LINE",
                "8",
                "FINGERS",
                "10",
                "0.0",
                "20",
                f"{y_pos}",
                "11",
                f"{params.cell_width}",
                "21",
                f"{y_pos}"
            ])

        # Draw busbars as lines
        for x_pos in pattern.busbar_positions:
            dxf_parts.extend([
                "0",
                "LINE",
                "8",
                "BUSBARS",
                "10",
                f"{x_pos}",
                "20",
                "0.0",
                "11",
                f"{x_pos}",
                "21",
                f"{params.cell_length}"
            ])

        dxf_parts.extend([
            "0",
            "ENDSEC",
            "0",
            "EOF"
        ])

        return '\n'.join(dxf_parts)

    def _export_gdsii(
        self,
        pattern: GridPattern,
        params: MetallizationParameters
    ) -> str:
        """Export pattern to GDSII format (text representation)."""
        # GDSII is a binary format, returning text description
        gds_text = [
            "GDSII Layout Export",
            f"Cell: PV_CELL_{pattern.pattern_type.value}",
            "",
            "Layer 1: Fingers",
        ]

        finger_width_nm = int(params.finger_width * 1000)
        for i, y_pos in enumerate(pattern.finger_positions):
            y_nm = int(y_pos * 1e6)
            gds_text.append(
                f"  RECT {i}: (0, {y_nm}) to ({int(params.cell_width*1e6)}, {y_nm + finger_width_nm})"
            )

        gds_text.append("\nLayer 2: Busbars")
        busbar_width_nm = int(params.busbar_width * 1000)
        for i, x_pos in enumerate(pattern.busbar_positions):
            x_nm = int(x_pos * 1e6)
            gds_text.append(
                f"  RECT {i}: ({x_nm}, 0) to ({x_nm + busbar_width_nm}, {int(params.cell_length*1e6)})"
            )

        return '\n'.join(gds_text)


# ============================================================================
# Utility Functions
# ============================================================================

def compare_patterns(
    patterns: List[OptimizedPattern]
) -> Dict[str, Any]:
    """
    Compare multiple optimized patterns.

    Args:
        patterns: List of optimized patterns to compare

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "count": len(patterns),
        "patterns": []
    }

    for i, pattern in enumerate(patterns):
        comparison["patterns"].append({
            "index": i,
            "pattern_type": pattern.pattern.pattern_type.value,
            "busbar_config": pattern.pattern.busbar_config.value,
            "shading_fraction": pattern.pattern.shading_fraction,
            "series_resistance": pattern.pattern.series_resistance,
            "silver_mass_mg": pattern.pattern.silver_mass,
            "combined_efficiency": pattern.combined_efficiency,
            "total_cost": pattern.silver_cost_per_cell + pattern.processing_cost_per_cell,
            "performance_to_cost": pattern.performance_to_cost_ratio
        })

    # Find best in each category
    if patterns:
        comparison["best_efficiency"] = max(
            range(len(patterns)),
            key=lambda i: patterns[i].combined_efficiency
        )
        comparison["lowest_cost"] = min(
            range(len(patterns)),
            key=lambda i: patterns[i].silver_cost_per_cell + patterns[i].processing_cost_per_cell
        )
        comparison["best_performance_to_cost"] = max(
            range(len(patterns)),
            key=lambda i: patterns[i].performance_to_cost_ratio
        )

    return comparison


def calculate_module_level_impact(
    cell_pattern: OptimizedPattern,
    module_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate module-level impact of cell metallization.

    Args:
        cell_pattern: Optimized cell pattern
        module_config: Module configuration (cells in series/parallel, etc.)

    Returns:
        Dictionary with module-level metrics
    """
    cells_in_series = module_config.get('cells_in_series', 60)
    cells_in_parallel = module_config.get('cells_in_parallel', 1)
    total_cells = cells_in_series * cells_in_parallel

    # Module-level calculations
    module_resistance = (
        cell_pattern.pattern.series_resistance * cells_in_series / cells_in_parallel
    )

    total_silver_mass_g = (cell_pattern.pattern.silver_mass * total_cells) / 1000
    total_cost = (
        cell_pattern.silver_cost_per_cell + cell_pattern.processing_cost_per_cell
    ) * total_cells

    module_efficiency_loss = (
        1 - cell_pattern.combined_efficiency
    ) * 100  # Convert to percentage

    return {
        "module_series_resistance_ohm": module_resistance,
        "total_silver_mass_g": total_silver_mass_g,
        "total_metallization_cost_usd": total_cost,
        "module_efficiency_loss_percent": module_efficiency_loss,
        "cells_count": total_cells
    }
