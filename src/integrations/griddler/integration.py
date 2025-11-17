"""
Griddler integration for metallization grid design and optimization.

This module provides an interface to Griddler for optimizing front contact
grid patterns to minimize resistive losses while maximizing light capture.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class GridPattern:
    """Front contact grid pattern definition."""

    # Busbar configuration
    num_busbars: int = 3
    busbar_width: float = 1.5  # mm

    # Finger configuration
    num_fingers: int = 80
    finger_width: float = 0.05  # mm
    finger_spacing: float = 2.0  # mm

    # Cell dimensions
    cell_width: float = 156.0  # mm
    cell_height: float = 156.0  # mm

    # Metal properties
    metal_resistivity: float = 2.65e-6  # Ohm·cm (Ag)
    metal_thickness: float = 20.0  # µm

    # Contact resistance
    contact_resistivity: float = 1e-3  # Ohm·cm²


@dataclass
class GridOptimizationResults:
    """Results from grid optimization."""

    # Optimized pattern
    pattern: GridPattern

    # Loss analysis
    shading_loss: float = 0.0  # %
    series_resistance: float = 0.0  # Ohm·cm²
    resistance_loss: float = 0.0  # %
    total_loss: float = 0.0  # %

    # Power metrics
    power_loss: float = 0.0  # W
    efficiency_loss: float = 0.0  # % absolute

    # Cost metrics
    metal_usage: float = 0.0  # mg
    metal_cost: float = 0.0  # $/cell


class GriddlerIntegration:
    """
    Integration with Griddler for grid optimization.

    Provides methods to design and optimize front contact grids
    for PV cells.
    """

    def __init__(self):
        """Initialize Griddler integration."""
        self.demo_mode = True  # Always demo mode for now

    def calculate_grid_losses(
        self,
        pattern: GridPattern,
        current_density: float = 40.0,  # mA/cm²
        voltage: float = 0.6  # V
    ) -> GridOptimizationResults:
        """
        Calculate losses for a given grid pattern.

        Args:
            pattern: Grid pattern definition
            current_density: Operating current density (mA/cm²)
            voltage: Operating voltage (V)

        Returns:
            GridOptimizationResults with loss analysis
        """
        results = GridOptimizationResults(pattern=pattern)

        # Calculate shading loss
        cell_area = pattern.cell_width * pattern.cell_height  # mm²

        # Busbar area
        busbar_area = (
            pattern.num_busbars *
            pattern.busbar_width *
            pattern.cell_height
        )

        # Finger area
        finger_area = (
            pattern.num_fingers *
            pattern.finger_width *
            pattern.cell_width
        )

        # Overlap area (fingers crossing busbars)
        overlap_area = (
            pattern.num_busbars *
            pattern.num_fingers *
            pattern.busbar_width *
            pattern.finger_width
        )

        total_metal_area = busbar_area + finger_area - overlap_area
        results.shading_loss = (total_metal_area / cell_area) * 100

        # Calculate series resistance
        # Finger resistance
        finger_length = pattern.cell_width / 2  # Current flows to nearest busbar
        finger_resistance = (
            pattern.metal_resistivity * finger_length * 10 /  # mm to cm
            (pattern.finger_width * 1e-1 * pattern.metal_thickness * 1e-4)  # cm²
        )

        # Busbar resistance (simplified)
        busbar_resistance = (
            pattern.metal_resistivity * pattern.cell_height * 10 /
            (pattern.busbar_width * 1e-1 * pattern.metal_thickness * 1e-4)
        ) / pattern.num_busbars

        # Contact resistance
        contact_area = total_metal_area * 1e-2  # mm² to cm²
        contact_resistance = pattern.contact_resistivity / contact_area

        # Total series resistance (simplified model)
        results.series_resistance = (
            finger_resistance / pattern.num_fingers +
            busbar_resistance +
            contact_resistance
        )

        # Calculate resistance loss
        # Power loss = I²R
        cell_area_cm2 = cell_area * 1e-2  # mm² to cm²
        total_current = current_density * cell_area_cm2 / 1000  # A
        power_loss = total_current**2 * results.series_resistance  # W

        # Power at MPP
        power_mpp = voltage * total_current  # W
        results.power_loss = power_loss
        results.resistance_loss = (power_loss / power_mpp) * 100 if power_mpp > 0 else 0

        # Total loss
        results.total_loss = results.shading_loss + results.resistance_loss

        # Calculate metal usage
        results.metal_usage = (
            total_metal_area *  # mm²
            pattern.metal_thickness * 1e-3 *  # mm
            10.49  # g/cm³ for Ag
        ) * 1000  # Convert to mg

        # Estimate cost ($50/g for Ag)
        results.metal_cost = results.metal_usage * 1e-3 * 50

        return results

    def optimize_grid(
        self,
        cell_width: float = 156.0,
        cell_height: float = 156.0,
        current_density: float = 40.0,
        voltage: float = 0.6,
        num_busbars: Optional[int] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> GridOptimizationResults:
        """
        Optimize grid pattern for minimum losses.

        Args:
            cell_width: Cell width (mm)
            cell_height: Cell height (mm)
            current_density: Operating current density (mA/cm²)
            voltage: Operating voltage (V)
            num_busbars: Fixed number of busbars (None for optimization)
            constraints: Additional constraints

        Returns:
            Optimized GridOptimizationResults
        """
        best_results = None
        best_loss = float('inf')

        # Try different configurations
        busbar_options = [num_busbars] if num_busbars else [3, 4, 5, 6]
        finger_options = range(60, 121, 10)
        finger_width_options = [0.03, 0.04, 0.05, 0.06]  # mm

        for nb in busbar_options:
            for nf in finger_options:
                for fw in finger_width_options:
                    pattern = GridPattern(
                        num_busbars=nb,
                        num_fingers=nf,
                        finger_width=fw,
                        finger_spacing=cell_width / nf,
                        cell_width=cell_width,
                        cell_height=cell_height
                    )

                    results = self.calculate_grid_losses(
                        pattern,
                        current_density,
                        voltage
                    )

                    if results.total_loss < best_loss:
                        best_loss = results.total_loss
                        best_results = results

        return best_results if best_results else GridOptimizationResults(
            pattern=GridPattern(cell_width=cell_width, cell_height=cell_height)
        )

    def visualize_grid(
        self,
        pattern: GridPattern
    ) -> np.ndarray:
        """
        Generate visual representation of grid pattern.

        Args:
            pattern: Grid pattern to visualize

        Returns:
            2D array representing grid (1=metal, 0=clear)
        """
        # Create grid image (10 pixels per mm)
        scale = 10
        width = int(pattern.cell_width * scale)
        height = int(pattern.cell_height * scale)

        grid = np.zeros((height, width))

        # Draw busbars
        busbar_positions = np.linspace(
            0,
            pattern.cell_width,
            pattern.num_busbars + 2
        )[1:-1]  # Exclude edges

        for pos in busbar_positions:
            x_start = int((pos - pattern.busbar_width / 2) * scale)
            x_end = int((pos + pattern.busbar_width / 2) * scale)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            grid[:, x_start:x_end] = 1

        # Draw fingers
        finger_positions = np.linspace(
            0,
            pattern.cell_height,
            pattern.num_fingers + 2
        )[1:-1]

        for pos in finger_positions:
            y_start = int((pos - pattern.finger_width / 2) * scale)
            y_end = int((pos + pattern.finger_width / 2) * scale)
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            grid[y_start:y_end, :] = 1

        return grid
