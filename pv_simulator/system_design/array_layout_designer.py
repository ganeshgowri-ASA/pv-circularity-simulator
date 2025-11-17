"""Array layout designer for various PV mounting configurations."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from pv_simulator.system_design.models import (
    ModuleParameters,
    MountingType,
    ArrayLayout,
)

logger = logging.getLogger(__name__)


class ArrayLayoutDesigner:
    """
    Design PV array layouts for multiple mounting configurations.

    Supports ground-mounted (fixed-tilt, trackers), rooftop, carport, floating,
    agrivoltaic, and BIPV systems with optimized spacing and layout.
    """

    def __init__(
        self,
        module: ModuleParameters,
        mounting_type: MountingType = MountingType.GROUND_FIXED,
        site_latitude: float = 35.0,
    ):
        """
        Initialize array layout designer.

        Args:
            module: Module parameters for sizing calculations
            mounting_type: Type of mounting system
            site_latitude: Site latitude for tilt optimization (degrees)
        """
        self.module = module
        self.mounting_type = mounting_type
        self.site_latitude = site_latitude

        logger.info(
            f"Initialized ArrayLayoutDesigner for {mounting_type.value} "
            f"at latitude {site_latitude}°"
        )

    def ground_mount_layout(
        self,
        total_modules: int,
        target_gcr: float = 0.4,
        tilt_angle: Optional[float] = None,
        azimuth: float = 180.0,
        modules_per_row: int = 20,
        tracker_type: str = "fixed",
    ) -> ArrayLayout:
        """
        Design ground-mounted array layout.

        Args:
            total_modules: Total number of modules
            target_gcr: Target ground coverage ratio (0.3-0.5 typical)
            tilt_angle: Module tilt angle (None = latitude for fixed-tilt)
            azimuth: Array azimuth (degrees, 180=south)
            modules_per_row: Number of modules per row
            tracker_type: "fixed", "single_axis", or "dual_axis"

        Returns:
            ArrayLayout configuration
        """
        # Determine tilt angle
        if tilt_angle is None:
            if tracker_type == "fixed":
                tilt_angle = abs(self.site_latitude)  # Latitude rule
            elif tracker_type == "single_axis":
                tilt_angle = 0.0  # Horizontal axis
            else:  # dual_axis
                tilt_angle = 0.0  # Variable

        # Calculate row spacing for target GCR
        module_height = self.module.length if self.module.length > self.module.width else self.module.width
        module_width = self.module.width if self.module.length > self.module.width else self.module.length

        # Row width (collector width)
        if tracker_type == "single_axis":
            # Tracker: use diagonal dimension
            row_width = np.sqrt(module_height**2 + module_width**2)
        else:
            # Fixed tilt: use projected width
            row_width = module_height * np.cos(np.radians(tilt_angle))

        # Calculate row spacing from GCR
        # GCR = row_width / row_spacing
        row_spacing = row_width / target_gcr if target_gcr > 0 else row_width * 2.5

        # Calculate number of rows
        num_rows = int(np.ceil(total_modules / modules_per_row))

        # Backtracking for single-axis trackers
        backtracking_enabled = tracker_type == "single_axis"
        max_rotation_angle = 60.0 if tracker_type == "single_axis" else None

        layout = ArrayLayout(
            mounting_type=(
                MountingType.GROUND_SINGLE_AXIS if tracker_type == "single_axis"
                else MountingType.GROUND_DUAL_AXIS if tracker_type == "dual_axis"
                else MountingType.GROUND_FIXED
            ),
            tilt_angle=tilt_angle,
            azimuth=azimuth,
            rows=num_rows,
            modules_per_row=modules_per_row,
            row_spacing=row_spacing,
            module_spacing=0.02,  # 2cm gap between modules
            gcr=target_gcr,
            backtracking_enabled=backtracking_enabled,
            max_rotation_angle=max_rotation_angle,
        )

        logger.info(
            f"Ground-mount layout: {num_rows} rows × {modules_per_row} modules, "
            f"tilt={tilt_angle}°, GCR={target_gcr:.2f}, spacing={row_spacing:.2f}m"
        )

        return layout

    def rooftop_layout(
        self,
        total_modules: int,
        roof_width_m: float,
        roof_length_m: float,
        roof_tilt: float = 5.0,
        roof_azimuth: float = 180.0,
        portrait_orientation: bool = True,
        fire_setback_m: float = 1.0,
    ) -> ArrayLayout:
        """
        Design rooftop system layout with setbacks and fire access.

        Args:
            total_modules: Total number of modules
            roof_width_m: Available roof width (m)
            roof_length_m: Available roof length (m)
            roof_tilt: Roof tilt angle (degrees)
            roof_azimuth: Roof azimuth (degrees, 180=south)
            portrait_orientation: True for portrait, False for landscape
            fire_setback_m: Fire access setback from roof edges (m)

        Returns:
            ArrayLayout configuration
        """
        # Calculate usable roof area after setbacks
        usable_width = roof_width_m - (2 * fire_setback_m)
        usable_length = roof_length_m - (2 * fire_setback_m)

        if usable_width <= 0 or usable_length <= 0:
            raise ValueError("Roof dimensions too small after applying setbacks")

        # Module dimensions
        if portrait_orientation:
            module_length = self.module.length
            module_width = self.module.width
        else:
            module_length = self.module.width
            module_width = self.module.length

        # Calculate modules per row and number of rows
        modules_per_row = int(np.floor(usable_width / module_width))
        max_rows = int(np.floor(usable_length / module_length))

        # For sloped roofs, rows are along the roof
        # For flat roofs, may need to consider row spacing
        if roof_tilt < 10:  # Flat roof
            # Add tilted racking
            effective_tilt = max(10.0, abs(self.site_latitude))
            projected_length = module_length * np.cos(np.radians(effective_tilt))
            row_spacing = projected_length * 2.0  # Avoid self-shading
            max_rows = int(np.floor(usable_length / row_spacing))
        else:
            effective_tilt = roof_tilt
            row_spacing = module_length + 0.05  # 5cm gap

        num_rows = min(max_rows, int(np.ceil(total_modules / modules_per_row)))

        # Calculate actual number of modules that fit
        actual_modules = min(total_modules, num_rows * modules_per_row)

        if actual_modules < total_modules:
            logger.warning(
                f"Only {actual_modules} of {total_modules} modules fit on roof"
            )

        layout = ArrayLayout(
            mounting_type=(
                MountingType.ROOFTOP_FLAT if roof_tilt < 10
                else MountingType.ROOFTOP_SLOPED
            ),
            tilt_angle=effective_tilt,
            azimuth=roof_azimuth,
            rows=num_rows,
            modules_per_row=modules_per_row,
            row_spacing=row_spacing,
            module_spacing=0.02,
            gcr=None,  # Not applicable for rooftop
            setback_front=fire_setback_m,
            setback_back=fire_setback_m,
            setback_side=fire_setback_m,
        )

        logger.info(
            f"Rooftop layout: {num_rows} rows × {modules_per_row} modules "
            f"({actual_modules} total), tilt={effective_tilt}°, setback={fire_setback_m}m"
        )

        return layout

    def carport_canopy_layout(
        self,
        total_modules: int,
        num_parking_spaces: int,
        space_width_m: float = 2.5,
        space_length_m: float = 5.5,
        clearance_height_m: float = 2.5,
        tilt_angle: float = 5.0,
    ) -> ArrayLayout:
        """
        Design carport or canopy structure layout.

        Args:
            total_modules: Total number of modules
            num_parking_spaces: Number of parking spaces
            space_width_m: Width of each parking space (m)
            space_length_m: Length of each parking space (m)
            clearance_height_m: Minimum clearance height (m)
            tilt_angle: Module tilt angle (degrees, typically 5-15°)

        Returns:
            ArrayLayout configuration
        """
        # Carport typically covers 2 rows of parking
        spaces_per_row = int(np.ceil(num_parking_spaces / 2))

        # Canopy width covers 2 parking spaces
        canopy_width = space_width_m * 2

        # Modules along the length of parking spaces
        module_length = max(self.module.length, self.module.width)
        modules_per_row = int(np.floor(space_length_m * spaces_per_row / self.module.width))

        # Number of rows (typically 2-3 modules wide per canopy)
        modules_across = int(np.floor(canopy_width / module_length))
        num_rows = modules_across

        # Calculate actual modules
        actual_modules = min(total_modules, num_rows * modules_per_row)

        if actual_modules < total_modules:
            logger.warning(
                f"Only {actual_modules} of {total_modules} modules fit on carport"
            )

        layout = ArrayLayout(
            mounting_type=MountingType.CARPORT,
            tilt_angle=tilt_angle,
            azimuth=180.0,  # Typically south-facing
            rows=num_rows,
            modules_per_row=modules_per_row,
            row_spacing=module_length + 0.05,
            module_spacing=0.02,
            clearance_height=clearance_height_m,
        )

        logger.info(
            f"Carport layout: {num_parking_spaces} spaces, "
            f"{num_rows} rows × {modules_per_row} modules ({actual_modules} total)"
        )

        return layout

    def floating_pv_layout(
        self,
        total_modules: int,
        water_area_m2: float,
        max_coverage_ratio: float = 0.3,
        tilt_angle: float = 10.0,
        pontoon_spacing_m: float = 0.5,
    ) -> ArrayLayout:
        """
        Design floating solar system layout.

        Args:
            total_modules: Total number of modules
            water_area_m2: Available water surface area (m²)
            max_coverage_ratio: Maximum water coverage ratio (typically 0.2-0.4)
            tilt_angle: Module tilt angle (degrees, typically 5-15° for floating)
            pontoon_spacing_m: Spacing between pontoon platforms (m)

        Returns:
            ArrayLayout configuration
        """
        # Calculate module area
        module_area = self.module.area
        total_module_area = total_modules * module_area

        # Account for tilt (increases footprint)
        tilt_factor = np.cos(np.radians(tilt_angle))
        projected_area = total_module_area / tilt_factor

        # Add pontoon spacing
        pontoon_factor = 1.2  # Pontoons add ~20% to footprint
        required_area = projected_area * pontoon_factor

        # Check coverage ratio
        coverage_ratio = required_area / water_area_m2

        if coverage_ratio > max_coverage_ratio:
            # Reduce number of modules to meet coverage limit
            adjusted_modules = int(total_modules * max_coverage_ratio / coverage_ratio)
            logger.warning(
                f"Reducing modules from {total_modules} to {adjusted_modules} "
                f"to meet {max_coverage_ratio*100:.0f}% coverage limit"
            )
            total_modules = adjusted_modules
            coverage_ratio = max_coverage_ratio

        # Layout in rows (typically long rows for floating)
        modules_per_row = 50  # Typical floating array row length
        num_rows = int(np.ceil(total_modules / modules_per_row))

        row_spacing = self.module.length + pontoon_spacing_m

        layout = ArrayLayout(
            mounting_type=MountingType.FLOATING,
            tilt_angle=tilt_angle,
            azimuth=180.0,  # South-facing
            rows=num_rows,
            modules_per_row=modules_per_row,
            row_spacing=row_spacing,
            module_spacing=pontoon_spacing_m,
            pontoon_spacing=pontoon_spacing_m,
            water_coverage_ratio=coverage_ratio,
        )

        logger.info(
            f"Floating PV layout: {num_rows} rows × {modules_per_row} modules, "
            f"coverage={coverage_ratio*100:.1f}%, tilt={tilt_angle}°"
        )

        return layout

    def agrivoltaic_layout(
        self,
        total_modules: int,
        field_area_m2: float,
        crop_type: str = "generic",
        clearance_height_m: float = 3.0,
        crop_row_spacing_m: float = 3.0,
        tilt_angle: float = 25.0,
    ) -> ArrayLayout:
        """
        Design agrivoltaic system layout with crop spacing.

        Args:
            total_modules: Total number of modules
            field_area_m2: Available field area (m²)
            crop_type: Type of crop (affects spacing requirements)
            clearance_height_m: Minimum ground clearance for equipment (m)
            crop_row_spacing_m: Spacing for crop rows between PV arrays (m)
            tilt_angle: Module tilt angle (degrees)

        Returns:
            ArrayLayout configuration
        """
        # Agrivoltaic systems typically have wider row spacing for light penetration
        # and equipment access

        # Module row width
        row_width = self.module.length * np.cos(np.radians(tilt_angle))

        # Spacing between PV rows (includes crop rows)
        # Typically 8-12m for tractor access and light penetration
        row_spacing = max(crop_row_spacing_m * 2, 8.0)

        # Calculate GCR
        gcr = row_width / row_spacing

        # Modules per row (typically shorter rows for agrivoltaics)
        modules_per_row = 30
        num_rows = int(np.ceil(total_modules / modules_per_row))

        # Check if fits in field
        required_length = num_rows * row_spacing
        available_width = np.sqrt(field_area_m2 / required_length) if required_length > 0 else 0

        if available_width < modules_per_row * self.module.width:
            # Adjust layout
            modules_per_row = int(available_width / self.module.width)
            num_rows = int(np.ceil(total_modules / modules_per_row))
            logger.warning(f"Adjusted to {num_rows} rows × {modules_per_row} modules")

        layout = ArrayLayout(
            mounting_type=MountingType.AGRIVOLTAIC,
            tilt_angle=tilt_angle,
            azimuth=180.0,
            rows=num_rows,
            modules_per_row=modules_per_row,
            row_spacing=row_spacing,
            module_spacing=0.02,
            gcr=gcr,
            clearance_height=clearance_height_m,
            crop_row_spacing=crop_row_spacing_m,
        )

        logger.info(
            f"Agrivoltaic layout: {num_rows} rows × {modules_per_row} modules, "
            f"row spacing={row_spacing}m, clearance={clearance_height_m}m, GCR={gcr:.2f}"
        )

        return layout

    def calculate_ground_coverage_ratio(
        self,
        tilt_angle: float,
        row_spacing: float,
    ) -> float:
        """
        Calculate ground coverage ratio for tilted arrays.

        Args:
            tilt_angle: Module tilt angle (degrees)
            row_spacing: Distance between row centers (m)

        Returns:
            Ground coverage ratio (0-1)
        """
        # Collector width (projected width of tilted module)
        module_length = max(self.module.length, self.module.width)
        collector_width = module_length * np.cos(np.radians(tilt_angle))

        # GCR = collector width / row spacing
        gcr = collector_width / row_spacing if row_spacing > 0 else 0.0

        logger.debug(
            f"GCR calculation: tilt={tilt_angle}°, "
            f"collector={collector_width:.2f}m, spacing={row_spacing:.2f}m, GCR={gcr:.3f}"
        )

        return gcr

    def design_string_routing(
        self,
        layout: ArrayLayout,
        modules_per_string: int,
    ) -> Dict[str, any]:
        """
        Design combiner box placement and string routing.

        Args:
            layout: ArrayLayout configuration
            modules_per_string: Number of modules per string

        Returns:
            Dictionary with string routing design
        """
        total_modules = layout.rows * layout.modules_per_row
        num_strings = int(np.ceil(total_modules / modules_per_string))

        # Strings per row
        strings_per_row = int(np.ceil(layout.modules_per_row / modules_per_string))

        # Combiner box placement (one per 10-20 strings typically)
        strings_per_combiner = 12  # Typical
        num_combiners = int(np.ceil(num_strings / strings_per_combiner))

        # Place combiners at row ends or centrally
        if layout.rows <= 4:
            combiner_placement = "row_ends"
        else:
            combiner_placement = "central_pads"

        # Calculate average cable length
        if combiner_placement == "row_ends":
            avg_string_cable_m = layout.modules_per_row * self.module.width / 2
        else:
            avg_string_cable_m = (
                np.sqrt(
                    (layout.modules_per_row * self.module.width / 2) ** 2
                    + (layout.rows * layout.row_spacing / 2) ** 2
                )
            )

        # Combiner to inverter cable length
        avg_combiner_to_inv_m = max(50.0, layout.rows * layout.row_spacing / 2)

        routing = {
            "total_modules": total_modules,
            "modules_per_string": modules_per_string,
            "num_strings": num_strings,
            "strings_per_row": strings_per_row,
            "num_combiners": num_combiners,
            "strings_per_combiner": strings_per_combiner,
            "combiner_placement": combiner_placement,
            "avg_string_cable_length_m": avg_string_cable_m,
            "avg_combiner_to_inverter_m": avg_combiner_to_inv_m,
            "total_dc_cable_estimate_m": num_strings * avg_string_cable_m
            + num_combiners * avg_combiner_to_inv_m,
        }

        logger.info(
            f"String routing: {num_strings} strings, {num_combiners} combiners, "
            f"{combiner_placement} placement, "
            f"avg cable: {avg_string_cable_m:.1f}m (string), "
            f"{avg_combiner_to_inv_m:.1f}m (combiner-inverter)"
        )

        return routing

    def optimize_row_spacing(
        self,
        tilt_angle: float,
        target_gcr: Optional[float] = None,
        max_shading_loss: float = 5.0,
        solstice_shading: bool = True,
    ) -> float:
        """
        Optimize row spacing to minimize shading while maximizing land use.

        Args:
            tilt_angle: Module tilt angle (degrees)
            target_gcr: Target GCR (if None, optimize for max_shading_loss)
            max_shading_loss: Maximum acceptable shading loss (%)
            solstice_shading: Consider worst-case winter solstice shading

        Returns:
            Optimal row spacing (m)
        """
        module_length = max(self.module.length, self.module.width)

        if target_gcr is not None:
            # Direct calculation from target GCR
            collector_width = module_length * np.cos(np.radians(tilt_angle))
            optimal_spacing = collector_width / target_gcr
            logger.info(f"Row spacing for GCR={target_gcr:.2f}: {optimal_spacing:.2f}m")
            return optimal_spacing

        # Optimize based on shading
        if solstice_shading:
            # Winter solstice: sun elevation at solar noon
            # Simplified: elevation ≈ 90° - latitude - 23.45° (tilt)
            sun_elevation = 90 - abs(self.site_latitude) - 23.45
            sun_elevation = max(15, sun_elevation)  # Minimum 15° elevation
        else:
            # Equinox condition
            sun_elevation = 90 - abs(self.site_latitude)

        # Shadow length = module_height / tan(sun_elevation)
        module_height = module_length * np.sin(np.radians(tilt_angle))
        shadow_length = module_height / np.tan(np.radians(sun_elevation))

        # Add buffer for acceptable shading
        # ~5% shading ≈ 10% of shadow penetrating next row
        buffer_factor = 1 - (max_shading_loss / 50.0)
        optimal_spacing = shadow_length / buffer_factor

        calculated_gcr = self.calculate_ground_coverage_ratio(tilt_angle, optimal_spacing)

        logger.info(
            f"Optimized row spacing: {optimal_spacing:.2f}m "
            f"(sun elevation: {sun_elevation:.1f}°, GCR: {calculated_gcr:.2f})"
        )

        return optimal_spacing
