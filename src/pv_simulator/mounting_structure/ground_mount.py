"""Ground-mount PV system design: fixed-tilt, single-axis, and dual-axis trackers."""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

from .models import (
    GroundMountConfig,
    ModuleOrientation,
    RackingConfiguration,
    FoundationType,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
)
from .structural_calculator import StructuralCalculator
from .foundation_engineer import FoundationEngineer


logger = logging.getLogger(__name__)


class GroundMountDesign:
    """
    Ground-mount PV system design and engineering.

    Supports:
    - Fixed-tilt racking (1P, 2P, 3P, 4P configurations)
    - Single-axis trackers (horizontal, tilted axis, backtracking)
    - Dual-axis trackers with sun tracking
    - Row spacing optimization
    - Foundation design for ground mount
    """

    def __init__(
        self,
        config: GroundMountConfig,
        structural_calc: Optional[StructuralCalculator] = None,
        foundation_eng: Optional[FoundationEngineer] = None,
    ):
        """
        Initialize ground mount designer.

        Args:
            config: Ground mount configuration
            structural_calc: Structural calculator instance
            foundation_eng: Foundation engineer instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)
        self.foundation_eng = foundation_eng or FoundationEngineer(config.site_parameters)

    def fixed_tilt_structure(self) -> StructuralAnalysisResult:
        """
        Design fixed-tilt racking structure.

        Returns:
            Complete structural analysis with BOM
        """
        logger.info(f"Designing fixed-tilt structure: {self.config.racking_config.value}, tilt={self.config.tilt_angle}°")

        # Calculate module layout
        layout = self._calculate_module_layout()

        # Calculate loads
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=layout["max_height"],
            additional_dead_load=0.15,  # Racking weight ~0.15 kN/m²
            is_rooftop=False,
        )

        # Design foundation
        foundation = self._design_ground_foundation(
            uplift_force=load_analysis.wind_load_uplift * layout["area_per_post"],
            compression_force=(load_analysis.dead_load + load_analysis.snow_load) * layout["area_per_post"],
            lateral_force=0.5,  # kN
        )

        # Design structural members
        members = self._design_fixed_tilt_members(layout, load_analysis)

        # Calculate BOM
        bom = self._generate_bom(layout, members, foundation)

        # Total steel weight
        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.025,  # 25mm typical
            deflection_limit=layout["purlin_length"] / 180,
            connection_details=self._get_connection_details(),
            compliance_notes=[
                "Design per ASCE 7-22 wind and snow loads",
                "Foundation design per IBC 2021",
                "Deflection limits: L/180 for purlins",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def single_axis_tracker(self) -> StructuralAnalysisResult:
        """
        Design single-axis tracker system.

        Returns:
            Complete structural analysis with tracker specifications
        """
        logger.info(f"Designing single-axis tracker: backtracking={self.config.backtracking_enabled}")

        # Tracker geometry
        tracker_layout = self._calculate_tracker_layout()

        # Load analysis at max tracking angle
        max_angle = self.config.max_tracking_angle or 60.0
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=max_angle,  # Worst case
            height=tracker_layout["max_height"],
            additional_dead_load=0.25,  # Tracker adds more weight
            is_rooftop=False,
        )

        # Tracker torque tube design
        torque_tube = self._design_torque_tube(tracker_layout, load_analysis)

        # Foundation for tracker posts
        foundation = self._design_ground_foundation(
            uplift_force=load_analysis.wind_load_uplift * tracker_layout["area_per_post"],
            compression_force=(load_analysis.dead_load + load_analysis.snow_load) * tracker_layout["area_per_post"],
            lateral_force=1.5,  # Higher lateral from tracker movement
        )

        # Structural members
        members = [torque_tube] + self._design_tracker_components(tracker_layout, load_analysis)

        # BOM
        bom = self._generate_tracker_bom(tracker_layout, members, foundation)

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.040,  # 40mm typical for trackers
            deflection_limit=tracker_layout["torque_tube_length"] / 150,
            connection_details={
                "tracker_motor": "Slew drive, 0.5 kW per tracker row",
                "backtracking": "Enabled" if self.config.backtracking_enabled else "Disabled",
                "max_tracking_angle": f"{max_angle}°",
                "stow_position": "Horizontal (0°) for high wind",
            },
            compliance_notes=[
                "Design per ASCE 7-22 with tracker-specific wind coefficients",
                "Stow position at wind speeds >15 m/s",
                "Backtracking algorithm to minimize shading",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def dual_axis_tracker(self) -> StructuralAnalysisResult:
        """
        Design dual-axis tracker system with sun tracking algorithms.

        Returns:
            Complete structural analysis with dual-axis specifications
        """
        logger.info("Designing dual-axis tracker")

        # Dual-axis typically has one tracker per module or small group
        modules_per_tracker = 4  # Typical

        # Pedestal height
        pedestal_height = 3.0  # m

        # Load analysis (worst case - horizontal position)
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=modules_per_tracker,
            tilt_angle=45.0,  # Mid-range
            height=pedestal_height,
            additional_dead_load=0.30,  # Dual-axis adds significant weight
            is_rooftop=False,
        )

        # Foundation for pedestal
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width * modules_per_tracker

        foundation = self._design_ground_foundation(
            uplift_force=load_analysis.wind_load_uplift * module_area,
            compression_force=(load_analysis.dead_load + load_analysis.snow_load) * module_area,
            lateral_force=2.0,  # Dual-axis has significant lateral loads
        )

        # Structural members
        members = [
            StructuralMember(
                member_type="pedestal",
                material=MaterialType.STEEL_GALVANIZED,
                profile="HSS6x6x1/4",
                length=pedestal_height,
                spacing=5.0,
                quantity=int(self.config.num_modules / modules_per_tracker),
                capacity=50.0,
                utilization=0.6,
            ),
            StructuralMember(
                member_type="tracker_frame",
                material=MaterialType.ALUMINUM,
                profile="Custom tracker frame",
                length=2.5,
                spacing=0.0,
                quantity=int(self.config.num_modules / modules_per_tracker),
                capacity=15.0,
                utilization=0.5,
            ),
        ]

        # BOM
        bom = self._generate_dual_axis_bom(modules_per_tracker, members, foundation)

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.030,
            deflection_limit=2.0 / 100,  # 2m frame, 1/100 limit
            connection_details={
                "azimuth_motor": "Slew drive for azimuth rotation",
                "tilt_motor": "Linear actuator for tilt adjustment",
                "tracking_accuracy": "±0.5°",
                "sun_tracking_algorithm": "Astronomical with sensor backup",
            },
            compliance_notes=[
                "Design per ASCE 7-22",
                "Stow position: horizontal at wind speeds >12 m/s",
                "Sun tracking: +/- 60° tilt, 360° azimuth",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def calculate_row_spacing(
        self,
        min_solar_access: float = 0.85,
    ) -> Dict[str, float]:
        """
        Calculate optimal row spacing to minimize shading.

        Args:
            min_solar_access: Minimum solar access fraction (0-1)

        Returns:
            Dictionary with row spacing parameters
        """
        logger.info(f"Calculating row spacing for tilt={self.config.tilt_angle}°, latitude={self.config.site_parameters.latitude}°")

        # Module dimensions
        module_length = self.config.module_dimensions.length

        # Height of module top edge
        if self.config.orientation == ModuleOrientation.PORTRAIT:
            slant_height = module_length
        else:
            slant_height = self.config.module_dimensions.width

        # Vertical height of tilted module
        h = slant_height * math.sin(math.radians(self.config.tilt_angle))

        # Horizontal projection
        L = slant_height * math.cos(math.radians(self.config.tilt_angle))

        # Solar altitude angle at winter solstice, solar noon
        # Simplified: altitude = 90° - latitude + 23.5° (winter) or - 23.5° (summer)
        lat_rad = math.radians(abs(self.config.site_parameters.latitude))

        # Winter solstice (worst case for shading)
        declination_winter = -23.5  # degrees
        solar_altitude_winter = 90 - abs(self.config.site_parameters.latitude) + declination_winter

        if solar_altitude_winter <= 0:
            logger.warning("Solar altitude is negative in winter - extreme latitude")
            solar_altitude_winter = 10  # Minimum

        # Shadow length
        shadow_length = h / math.tan(math.radians(solar_altitude_winter))

        # Row spacing to achieve desired solar access
        # GCR = ground coverage ratio = slant_height / row_spacing
        row_spacing = (L + shadow_length) / min_solar_access

        # Ground coverage ratio
        gcr = slant_height / row_spacing

        return {
            "row_spacing": row_spacing,
            "ground_coverage_ratio": gcr,
            "shadow_length_winter": shadow_length,
            "solar_altitude_winter": solar_altitude_winter,
            "vertical_height": h,
            "horizontal_projection": L,
            "shading_loss": (1 - min_solar_access) * 100,
        }

    def foundation_design(self) -> Dict[str, Any]:
        """
        Complete foundation design for ground mount system.

        Returns:
            Dictionary with foundation design details
        """
        # Calculate loads (typical)
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=1,  # Per module for unit load
            tilt_angle=self.config.tilt_angle,
            height=2.0,
            additional_dead_load=0.15,
            is_rooftop=False,
        )

        # Area per post (typical)
        area_per_post = self.config.post_spacing * 2.0  # 2m module width typical

        # Foundation options
        foundations = {}

        # Driven pile
        if self.config.foundation_type in [FoundationType.DRIVEN_PILE, FoundationType.GROUND_SCREW]:
            foundations["driven_pile"] = self.foundation_eng.pile_foundation_design(
                uplift_force=load_analysis.wind_load_uplift * area_per_post,
                compression_force=(load_analysis.dead_load + load_analysis.snow_load) * area_per_post,
                lateral_force=0.5,
            )

        # Helical pile
        if self.config.foundation_type == FoundationType.HELICAL_PILE:
            foundations["helical_pile"] = self.foundation_eng.helical_pile_design(
                uplift_force=load_analysis.wind_load_uplift * area_per_post,
                compression_force=(load_analysis.dead_load + load_analysis.snow_load) * area_per_post,
            )

        # Ballasted (if applicable)
        if self.config.foundation_type == FoundationType.BALLASTED:
            foundations["ballasted"] = self.foundation_eng.ballast_design(
                uplift_force=load_analysis.wind_load_uplift * area_per_post,
                wind_moment=load_analysis.wind_load_uplift * area_per_post * 1.5,
            )

        # Geotechnical requirements
        geo_requirements = self.foundation_eng.geotechnical_requirements(
            foundation_type=self.config.foundation_type,
        )

        # Frost protection
        frost_protection = self.foundation_eng.frost_depth_consideration(
            foundation_type=self.config.foundation_type,
        )

        return {
            "foundations": foundations,
            "geotechnical_requirements": geo_requirements,
            "frost_protection": frost_protection,
            "recommended_type": self.config.foundation_type.value,
        }

    def calculate_post_spacing(
        self,
        max_deflection_ratio: float = 180,
    ) -> Dict[str, float]:
        """
        Calculate optimal post spacing based on wind/snow loads.

        Args:
            max_deflection_ratio: Maximum deflection ratio (L/ratio)

        Returns:
            Dictionary with post spacing parameters
        """
        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=1,
            tilt_angle=self.config.tilt_angle,
            height=2.0,
            additional_dead_load=0.15,
            is_rooftop=False,
        )

        # Critical load (worst case)
        critical_load = max(
            load_analysis.dead_load + load_analysis.snow_load,
            abs(load_analysis.wind_load_uplift),
            abs(load_analysis.wind_load_downward),
        )

        # Purlin properties (typical C-channel)
        # C6x8.2: I = 13.1 in⁴ = 5.45e-6 m⁴
        I = 5.45e-6  # m⁴
        E = 200e6  # kN/m² (steel)

        # Maximum spacing for deflection limit
        # δ = 5wL⁴/(384EI) <= L/180
        # Solve for L: L³ <= (384EI)/(5w*180)
        w = critical_load * self.config.module_dimensions.width  # kN/m
        L_max_deflection = ((384 * E * I) / (5 * w * max_deflection_ratio)) ** (1/3)

        # Maximum spacing for stress (simplified - allowable stress design)
        # M = wL²/8, σ = M*c/I <= allowable
        # For C6x8.2: S = 3.09 in³ = 5.06e-5 m³
        S = 5.06e-5  # m³
        allowable_stress = 150e3  # kN/m² (Fy = 250 MPa, factor of safety)
        L_max_stress = math.sqrt(8 * allowable_stress * S / w)

        # Governing spacing
        L_max = min(L_max_deflection, L_max_stress)

        # Practical spacing (round down to nearest 0.5m)
        post_spacing = math.floor(L_max * 2) / 2

        # Ensure within practical range (1.5m - 4.0m)
        post_spacing = max(1.5, min(4.0, post_spacing))

        return {
            "post_spacing": post_spacing,
            "max_spacing_deflection": L_max_deflection,
            "max_spacing_stress": L_max_stress,
            "governing_criteria": "deflection" if L_max_deflection < L_max_stress else "stress",
            "applied_load": critical_load,
        }

    def racking_bom(self) -> List[BillOfMaterials]:
        """
        Generate complete bill of materials for racking system.

        Returns:
            List of BOM items
        """
        # Calculate layout
        layout = self._calculate_module_layout()

        # Foundation
        foundation = self._design_ground_foundation(10, 5, 0.5)  # Typical loads

        # Structural members
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=layout["max_height"],
            additional_dead_load=0.15,
            is_rooftop=False,
        )

        members = self._design_fixed_tilt_members(layout, load_analysis)

        return self._generate_bom(layout, members, foundation)

    # Private helper methods

    def _calculate_module_layout(self) -> Dict[str, Any]:
        """Calculate module layout geometry."""
        module_length = self.config.module_dimensions.length
        module_width = self.config.module_dimensions.width

        # Modules per row based on racking config
        if self.config.orientation == ModuleOrientation.PORTRAIT:
            if self.config.racking_config == RackingConfiguration.ONE_PORTRAIT:
                modules_high = 1
                row_width = module_width
            elif self.config.racking_config == RackingConfiguration.TWO_PORTRAIT:
                modules_high = 2
                row_width = module_width * 2
            elif self.config.racking_config == RackingConfiguration.THREE_PORTRAIT:
                modules_high = 3
                row_width = module_width * 3
            else:  # 4P
                modules_high = 4
                row_width = module_width * 4
            purlin_length = module_length
        else:  # Landscape
            modules_high = int(self.config.racking_config.value[0])
            row_width = module_length * modules_high
            purlin_length = module_width

        # Number of rows
        num_rows = math.ceil(self.config.num_modules / (modules_high * 10))  # Assume 10 modules per row length

        # Heights
        max_height = row_width * math.sin(math.radians(self.config.tilt_angle)) + 0.5  # 0.5m clearance

        # Posts per row
        posts_per_row = math.ceil(purlin_length / self.config.post_spacing) + 1

        return {
            "modules_high": modules_high,
            "row_width": row_width,
            "purlin_length": purlin_length,
            "num_rows": num_rows,
            "max_height": max_height,
            "posts_per_row": posts_per_row,
            "area_per_post": self.config.post_spacing * (row_width / modules_high),
        }

    def _calculate_tracker_layout(self) -> Dict[str, Any]:
        """Calculate single-axis tracker layout."""
        # Tracker row length (typical 50-100m)
        row_length = 80.0  # m

        # Modules per tracker row
        module_length = self.config.module_dimensions.length
        modules_per_row = int(row_length / module_length)

        # Tracker height (at horizontal position)
        tracker_height = 2.5  # m

        # Post spacing (trackers typically 10-20m apart)
        post_spacing = 15.0  # m

        # Number of posts per row
        posts_per_row = int(row_length / post_spacing) + 1

        return {
            "row_length": row_length,
            "modules_per_row": modules_per_row,
            "tracker_height": tracker_height,
            "post_spacing": post_spacing,
            "posts_per_row": posts_per_row,
            "max_height": tracker_height + self.config.module_dimensions.width * 0.5,
            "torque_tube_length": row_length,
            "area_per_post": post_spacing * self.config.module_dimensions.width * 2,
        }

    def _design_ground_foundation(
        self,
        uplift_force: float,
        compression_force: float,
        lateral_force: float,
    ):
        """Design foundation based on config."""
        if self.config.foundation_type == FoundationType.DRIVEN_PILE:
            return self.foundation_eng.pile_foundation_design(uplift_force, compression_force, lateral_force)
        elif self.config.foundation_type == FoundationType.HELICAL_PILE:
            return self.foundation_eng.helical_pile_design(uplift_force, compression_force)
        elif self.config.foundation_type == FoundationType.BALLASTED:
            return self.foundation_eng.ballast_design(uplift_force, uplift_force * 1.5)
        else:
            return self.foundation_eng.pile_foundation_design(uplift_force, compression_force, lateral_force)

    def _design_fixed_tilt_members(self, layout, load_analysis) -> List[StructuralMember]:
        """Design structural members for fixed-tilt."""
        members = [
            StructuralMember(
                member_type="purlin",
                material=MaterialType.STEEL_GALVANIZED,
                profile="C6x8.2",
                length=layout["purlin_length"],
                spacing=layout["row_width"] / layout["modules_high"],
                quantity=layout["num_rows"] * layout["modules_high"],
                capacity=15.0,
                utilization=0.65,
            ),
            StructuralMember(
                member_type="post",
                material=MaterialType.STEEL_GALVANIZED,
                profile="HSS4x4x1/4",
                length=layout["max_height"],
                spacing=self.config.post_spacing,
                quantity=layout["num_rows"] * layout["posts_per_row"],
                capacity=50.0,
                utilization=0.55,
            ),
            StructuralMember(
                member_type="cross_brace",
                material=MaterialType.STEEL_GALVANIZED,
                profile="L2x2x1/4",
                length=2.0,
                spacing=self.config.post_spacing,
                quantity=layout["num_rows"] * layout["posts_per_row"],
                capacity=25.0,
                utilization=0.45,
            ),
        ]
        return members

    def _design_torque_tube(self, layout, load_analysis) -> StructuralMember:
        """Design torque tube for single-axis tracker."""
        return StructuralMember(
            member_type="torque_tube",
            material=MaterialType.STEEL_GALVANIZED,
            profile="HSS8x8x3/8",
            length=layout["torque_tube_length"],
            spacing=0,
            quantity=int(self.config.num_modules / layout["modules_per_row"]),
            capacity=200.0,
            utilization=0.70,
        )

    def _design_tracker_components(self, layout, load_analysis) -> List[StructuralMember]:
        """Design tracker components."""
        num_trackers = int(self.config.num_modules / layout["modules_per_row"])
        return [
            StructuralMember(
                member_type="tracker_post",
                material=MaterialType.STEEL_GALVANIZED,
                profile="HSS6x6x3/8",
                length=layout["tracker_height"],
                spacing=layout["post_spacing"],
                quantity=layout["posts_per_row"] * num_trackers,
                capacity=100.0,
                utilization=0.60,
            ),
            StructuralMember(
                member_type="bearing_assembly",
                material=MaterialType.STEEL_GALVANIZED,
                profile="Tracker bearing",
                length=0.5,
                spacing=layout["post_spacing"],
                quantity=layout["posts_per_row"] * num_trackers,
                capacity=50.0,
                utilization=0.50,
            ),
        ]

    def _generate_bom(self, layout, members, foundation) -> List[BillOfMaterials]:
        """Generate bill of materials."""
        bom = []

        # Foundations
        total_foundations = foundation.quantity * layout["num_rows"] * layout["posts_per_row"]
        bom.append(BillOfMaterials(
            item_number="FND-001",
            description=f"{foundation.foundation_type.value} foundation",
            material=foundation.material,
            specification=f"{foundation.depth}m deep, {foundation.diameter}m dia",
            quantity=total_foundations,
            unit="ea",
            unit_weight=50.0,
            total_weight=50.0 * total_foundations,
            unit_cost=75.0,
            total_cost=75.0 * total_foundations,
        ))

        # Structural members
        for i, member in enumerate(members):
            weight_per_unit = 10.0 if "purlin" in member.member_type else 25.0
            bom.append(BillOfMaterials(
                item_number=f"STR-{i+1:03d}",
                description=f"{member.member_type} - {member.profile}",
                material=member.material,
                specification=f"{member.length}m long",
                quantity=member.quantity,
                unit="ea",
                unit_weight=weight_per_unit,
                total_weight=weight_per_unit * member.quantity,
                unit_cost=50.0,
                total_cost=50.0 * member.quantity,
            ))

        # Module clamps
        bom.append(BillOfMaterials(
            item_number="HW-001",
            description="Module mid clamp",
            material=MaterialType.ALUMINUM,
            specification="Universal mid clamp",
            quantity=self.config.num_modules * 4,
            unit="ea",
            unit_weight=0.2,
            total_weight=0.2 * self.config.num_modules * 4,
            unit_cost=2.5,
            total_cost=2.5 * self.config.num_modules * 4,
        ))

        return bom

    def _generate_tracker_bom(self, layout, members, foundation) -> List[BillOfMaterials]:
        """Generate BOM for tracker."""
        bom = self._generate_bom({"num_rows": int(self.config.num_modules / layout["modules_per_row"]), "posts_per_row": layout["posts_per_row"]}, members, foundation)

        # Add tracker-specific items
        num_trackers = int(self.config.num_modules / layout["modules_per_row"])
        bom.append(BillOfMaterials(
            item_number="TRK-001",
            description="Tracker motor assembly",
            material=MaterialType.STEEL_GALVANIZED,
            specification="Slew drive, 0.5kW",
            quantity=num_trackers,
            unit="ea",
            unit_weight=15.0,
            total_weight=15.0 * num_trackers,
            unit_cost=850.0,
            total_cost=850.0 * num_trackers,
        ))

        return bom

    def _generate_dual_axis_bom(self, modules_per_tracker, members, foundation) -> List[BillOfMaterials]:
        """Generate BOM for dual-axis tracker."""
        num_trackers = int(self.config.num_modules / modules_per_tracker)

        bom = [
            BillOfMaterials(
                item_number="FND-001",
                description="Pedestal foundation",
                material=MaterialType.CONCRETE,
                specification="1.5m x 1.5m x 1.0m deep",
                quantity=num_trackers,
                unit="ea",
                unit_weight=1500.0,
                total_weight=1500.0 * num_trackers,
                unit_cost=500.0,
                total_cost=500.0 * num_trackers,
            ),
            BillOfMaterials(
                item_number="TRK-001",
                description="Dual-axis tracker unit",
                material=MaterialType.STEEL_GALVANIZED,
                specification=f"{modules_per_tracker} module capacity",
                quantity=num_trackers,
                unit="ea",
                unit_weight=250.0,
                total_weight=250.0 * num_trackers,
                unit_cost=3500.0,
                total_cost=3500.0 * num_trackers,
            ),
        ]

        return bom

    def _get_connection_details(self) -> Dict[str, Any]:
        """Get connection design details."""
        return {
            "purlin_to_post": "3/8\" bolts, grade 5, (4) per connection",
            "post_to_foundation": "Embedded anchor bolts, 5/8\" dia",
            "module_clamps": "Aluminum mid/end clamps per manufacturer specs",
            "bracing": "Bolted L-bracket connections",
        }

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
