"""Rooftop PV mounting system design for flat and pitched roofs."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    RooftopMountConfig,
    ModuleDimensions,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
    FoundationType,
)
from .structural_calculator import StructuralCalculator
from .foundation_engineer import FoundationEngineer


logger = logging.getLogger(__name__)


class RooftopMountDesign:
    """
    Rooftop PV mounting system design.

    Supports:
    - Flat roof systems (ballasted, attached)
    - Pitched roof systems (rail-mounted, shared rail)
    - Roof load calculations
    - Attachment point design
    - Fire setback requirements
    - Wind zone analysis
    """

    def __init__(
        self,
        config: RooftopMountConfig,
        structural_calc: Optional[StructuralCalculator] = None,
        foundation_eng: Optional[FoundationEngineer] = None,
    ):
        """
        Initialize rooftop mount designer.

        Args:
            config: Rooftop mount configuration
            structural_calc: Structural calculator instance
            foundation_eng: Foundation engineer instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)
        self.foundation_eng = foundation_eng or FoundationEngineer(config.site_parameters)

    def flat_roof_design(
        self,
        ballasted: bool = True,
    ) -> StructuralAnalysisResult:
        """
        Design flat roof mounting system.

        Args:
            ballasted: Use ballasted system (True) or attached (False)

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing flat roof system: ballasted={ballasted}")

        # Module layout
        layout = self._calculate_rooftop_layout()

        # Roof height for wind analysis
        roof_height = 10.0  # Typical commercial roof height

        # Load analysis with rooftop wind effects
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.tilt_angle * 0.05,  # Module height above roof
            additional_dead_load=0.20 if ballasted else 0.10,
            is_rooftop=True,
        )

        # Check roof capacity
        roof_check = self.structural_capacity_check(load_analysis)

        if not roof_check["passes"]:
            logger.warning(f"Roof capacity insufficient: {roof_check['utilization']:.2f}")

        # Foundation (ballast or attachment)
        if ballasted:
            foundation = self.foundation_eng.ballast_design(
                uplift_force=load_analysis.wind_load_uplift * layout["area_per_support"],
                wind_moment=load_analysis.wind_load_uplift * layout["area_per_support"] * 0.5,
            )
        else:
            foundation = self._design_roof_attachment(
                uplift_force=load_analysis.wind_load_uplift * layout["area_per_support"],
                compression_force=(load_analysis.dead_load + load_analysis.snow_load) * layout["area_per_support"],
            )

        # Structural members
        members = self._design_flat_roof_members(layout, load_analysis, ballasted)

        # BOM
        bom = self._generate_rooftop_bom(layout, members, foundation, ballasted)

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        # Fire setback calculation
        setback_info = self.setback_requirements()

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.015,
            deflection_limit=2.0 / 240,  # L/240 for rooftop
            connection_details={
                "type": "ballasted" if ballasted else "attached",
                "roof_penetrations": 0 if ballasted else layout["num_attachments"],
                "waterproofing": "Required for attached systems" if not ballasted else "Not required",
            },
            compliance_notes=[
                f"Fire setback: {setback_info['perimeter_setback']}m from roof edge",
                f"Access pathway: {setback_info['pathway_width']}m wide",
                "Design per IFC 2021 Section 605.11",
                f"Roof load check: {'PASS' if roof_check['passes'] else 'FAIL'}",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def pitched_roof_design(
        self,
        attachment_method: str = "rail",
    ) -> StructuralAnalysisResult:
        """
        Design pitched roof mounting system.

        Args:
            attachment_method: Attachment method (rail, shared_rail, direct)

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing pitched roof system: {attachment_method}, pitch={self.config.roof_pitch}°")

        # Module layout on pitched roof
        layout = self._calculate_pitched_roof_layout()

        # Effective tilt angle (roof pitch + module tilt)
        effective_tilt = self.config.roof_pitch + self.config.tilt_angle

        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=effective_tilt,
            height=0.2,  # Low profile on pitched roof
            additional_dead_load=0.08,  # Lightweight rail system
            is_rooftop=True,
        )

        # Roof capacity check
        roof_check = self.structural_capacity_check(load_analysis)

        # Attachment points
        attachment_design = self.attachment_point_design(
            uplift_force=load_analysis.wind_load_uplift * layout["area_per_attachment"],
            dead_load=load_analysis.dead_load * layout["area_per_attachment"],
        )

        # Structural members (rails)
        members = self._design_pitched_roof_members(layout, load_analysis, attachment_method)

        # BOM
        bom = self._generate_pitched_roof_bom(layout, members, attachment_design, attachment_method)

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        # Fire setback
        setback_info = self.setback_requirements()

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=attachment_design,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.010,
            deflection_limit=layout["rail_length"] / 240,
            connection_details={
                "attachment_method": attachment_method,
                "roof_penetrations": layout["num_attachments"],
                "flashing": "L-foot with flashing boot",
                "waterproofing": "Sealant + flashing per manufacturer specs",
            },
            compliance_notes=[
                f"Fire setback: {setback_info['perimeter_setback']}m",
                f"Ridge setback: {setback_info['ridge_setback']}m",
                f"Roof load check: {'PASS' if roof_check['passes'] else 'FAIL'}",
                "Rapid shutdown per NEC 690.12",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def calculate_roof_loading(
        self,
        include_existing: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate total roof loading including existing and PV system.

        Args:
            include_existing: Include existing roof loads

        Returns:
            Dictionary with load components
        """
        # PV system loads
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=1,  # Per unit area
            tilt_angle=self.config.tilt_angle,
            height=0.2,
            additional_dead_load=0.15,
            is_rooftop=True,
        )

        # Existing roof loads (if applicable)
        existing_dead = 0.5 if include_existing else 0  # kN/m² (roofing, insulation, deck)
        existing_live = 1.0 if include_existing else 0  # kN/m² (maintenance)

        # Total loads
        total_dead = existing_dead + load_analysis.dead_load
        total_live = existing_live + load_analysis.live_load
        total_snow = load_analysis.snow_load
        total_wind_uplift = load_analysis.wind_load_uplift

        # Load combinations (governing)
        combo_1 = 1.2 * total_dead + 1.6 * total_live + 0.5 * total_snow
        combo_2 = 1.2 * total_dead + 1.6 * total_snow + total_live
        combo_3 = 1.2 * total_dead + 1.0 * abs(total_wind_uplift) + total_live + 0.5 * total_snow
        combo_4 = 0.9 * total_dead + 1.0 * abs(total_wind_uplift)

        governing = max(combo_1, combo_2, combo_3, abs(combo_4))

        return {
            "pv_dead_load": load_analysis.dead_load,
            "pv_live_load": load_analysis.live_load,
            "existing_dead_load": existing_dead,
            "existing_live_load": existing_live,
            "total_dead_load": total_dead,
            "total_live_load": total_live,
            "snow_load": total_snow,
            "wind_uplift": total_wind_uplift,
            "governing_combination": governing,
        }

    def attachment_point_design(
        self,
        uplift_force: float,
        dead_load: float,
        attachment_type: str = "L-foot",
    ) -> Any:
        """
        Design roof attachment points.

        Args:
            uplift_force: Uplift force per attachment (kN)
            dead_load: Dead load per attachment (kN)
            attachment_type: Type of attachment (L-foot, S-5, standoff)

        Returns:
            Foundation design for roof attachment
        """
        logger.info(f"Designing roof attachment: type={attachment_type}, uplift={uplift_force}kN")

        # Lag screw capacity (typical)
        lag_diameter = 0.0095  # m (3/8")
        lag_length = 0.15  # m (6")

        # Withdrawal capacity in wood (simplified)
        # P = 1800 * G^2 * D * L (lbs) where G=specific gravity, D=diameter(in), L=length(in)
        # For Douglas Fir: G = 0.5
        G = 0.5
        D_inch = lag_diameter * 39.37  # Convert to inches
        L_inch = lag_length * 39.37
        P_lbs = 1800 * G**2 * D_inch * L_inch
        P_kn = P_lbs * 0.00444822  # Convert lbs to kN

        # Safety factor
        sf = 4.0
        allowable_per_lag = P_kn / sf

        # Number of lags required
        num_lags = max(2, math.ceil(uplift_force / allowable_per_lag))

        # Attachment spacing (typical 1.2m for rails)
        spacing = 1.2  # m

        from .models import FoundationDesign

        return FoundationDesign(
            foundation_type=FoundationType.ROOF_ATTACHMENT,
            depth=lag_length,
            diameter=lag_diameter,
            length=0.0,
            width=0.0,
            capacity=allowable_per_lag * num_lags,
            spacing=spacing,
            quantity=1,
            material=MaterialType.STEEL_GALVANIZED,
            embedment_depth=lag_length,
            concrete_volume=0.0,
            reinforcement=f"{attachment_type} with {num_lags} x 3/8\" lag screws",
        )

    def setback_requirements(self) -> Dict[str, float]:
        """
        Calculate fire setback and access pathway requirements per IFC.

        Returns:
            Dictionary with setback dimensions
        """
        # IFC 2021 Section 605.11 requirements
        # Based on roof area and building type

        # Typical commercial building
        perimeter_setback = 1.5  # m (5 ft) from roof edge
        ridge_setback = 0.5  # m for pitched roofs
        valley_setback = 0.5  # m
        hip_setback = 0.5  # m

        # Access pathways
        pathway_width = 1.2  # m (4 ft) minimum
        pathway_spacing = 15.0  # m (50 ft) maximum between pathways

        # Smoke ventilation (for large arrays)
        smoke_vent_opening = 4.0  # m x 4.0 m every 600 m² of array

        return {
            "perimeter_setback": perimeter_setback,
            "ridge_setback": ridge_setback,
            "valley_setback": valley_setback,
            "hip_setback": hip_setback,
            "pathway_width": pathway_width,
            "pathway_spacing": pathway_spacing,
            "smoke_vent_size": smoke_vent_opening,
            "code_reference": "IFC 2021 Section 605.11",
        }

    def wind_zone_analysis(self) -> Dict[str, Any]:
        """
        Analyze ASCE 7 wind pressure coefficients for rooftop.

        Returns:
            Dictionary with wind zone details
        """
        # ASCE 7-22 Figure 29.4-7: Rooftop solar panel systems

        # Wind zones on roof
        zones = {
            "zone_1": {
                "description": "Corner zone",
                "width": min(0.1 * 30, 3.0),  # 10% of building width or 3m
                "pressure_coefficient": -2.2,  # Most critical
            },
            "zone_2": {
                "description": "Edge zone",
                "width": min(0.4 * 30, 12.0),  # 40% of building width or 12m
                "pressure_coefficient": -1.5,
            },
            "zone_3": {
                "description": "Interior zone",
                "width": "remainder",
                "pressure_coefficient": -0.9,  # Least critical
            },
        }

        # Apply wind zone factor to base pressure
        base_pressure = self.structural_calc.wind_load_analysis(
            tilt_angle=self.config.tilt_angle,
            height=10.0,
            module_dimensions=self.config.module_dimensions,
            is_rooftop=True,
            roof_height=10.0,
        )

        for zone, data in zones.items():
            data["design_pressure"] = abs(base_pressure["uplift_pressure"] * data["pressure_coefficient"])

        return {
            "zones": zones,
            "wind_zone_classification": self.config.wind_zone,
            "governing_zone": "zone_1",
            "notes": "Ballast or attachment design varies by zone",
        }

    def structural_capacity_check(
        self,
        load_analysis: Any,
    ) -> Dict[str, Any]:
        """
        Verify roof can support additional PV loading.

        Args:
            load_analysis: Load analysis results

        Returns:
            Dictionary with capacity check results
        """
        # Existing roof capacities
        existing_dead_capacity = self.config.roof_dead_load_capacity
        existing_live_capacity = self.config.roof_live_load_capacity

        # Additional loads from PV
        pv_dead = load_analysis.dead_load
        pv_live = load_analysis.live_load

        # Utilization ratios
        dead_utilization = pv_dead / existing_dead_capacity
        live_utilization = pv_live / existing_live_capacity

        # Check if within capacity
        passes_dead = dead_utilization < 0.8  # 80% utilization limit
        passes_live = live_utilization < 0.8

        passes_overall = passes_dead and passes_live

        return {
            "existing_dead_capacity": existing_dead_capacity,
            "existing_live_capacity": existing_live_capacity,
            "pv_dead_load": pv_dead,
            "pv_live_load": pv_live,
            "dead_utilization": dead_utilization,
            "live_utilization": live_utilization,
            "utilization": max(dead_utilization, live_utilization),
            "passes": passes_overall,
            "recommendation": "Roof structural upgrade required" if not passes_overall else "Roof capacity adequate",
        }

    # Private helper methods

    def _calculate_rooftop_layout(self) -> Dict[str, Any]:
        """Calculate flat rooftop layout."""
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width

        # Typical flat roof layout (rows with spacing)
        row_spacing = 2.0  # m between rows
        support_spacing = 2.5  # m between supports

        # Area per support point
        area_per_support = support_spacing * 2.0  # 2m module width

        # Number of support points
        num_supports = math.ceil(self.config.num_modules / 4)  # 4 modules per support typical

        return {
            "module_area": module_area,
            "row_spacing": row_spacing,
            "support_spacing": support_spacing,
            "area_per_support": area_per_support,
            "num_supports": num_supports,
            "total_roof_area": self.config.num_modules * module_area * 1.5,  # With spacing
        }

    def _calculate_pitched_roof_layout(self) -> Dict[str, Any]:
        """Calculate pitched roof layout."""
        module_length = self.config.module_dimensions.length
        module_width = self.config.module_dimensions.width

        # Rail spacing (typically 2 rails per module row)
        rail_spacing = 1.0  # m

        # Attachment spacing along rail
        attachment_spacing = 1.2  # m

        # Rail length (typical)
        rail_length = 5.0  # m

        # Number of attachments
        attachments_per_rail = int(rail_length / attachment_spacing) + 1
        num_rails = math.ceil(self.config.num_modules / 3)  # 3 modules per rail
        num_attachments = num_rails * attachments_per_rail

        # Area per attachment
        area_per_attachment = (rail_length * module_width) / attachments_per_rail

        return {
            "rail_spacing": rail_spacing,
            "attachment_spacing": attachment_spacing,
            "rail_length": rail_length,
            "num_rails": num_rails,
            "num_attachments": num_attachments,
            "area_per_attachment": area_per_attachment,
        }

    def _design_flat_roof_members(self, layout, load_analysis, ballasted) -> List[StructuralMember]:
        """Design structural members for flat roof."""
        members = [
            StructuralMember(
                member_type="tilt_frame",
                material=MaterialType.ALUMINUM,
                profile="Custom tilt frame",
                length=2.0,
                spacing=layout["support_spacing"],
                quantity=layout["num_supports"],
                capacity=10.0,
                utilization=0.55,
            ),
        ]

        if ballasted:
            members.append(
                StructuralMember(
                    member_type="ballast_tray",
                    material=MaterialType.STEEL_GALVANIZED,
                    profile="Ballast tray assembly",
                    length=0.6,
                    spacing=layout["support_spacing"],
                    quantity=layout["num_supports"],
                    capacity=20.0,
                    utilization=0.60,
                )
            )

        return members

    def _design_pitched_roof_members(self, layout, load_analysis, attachment_method) -> List[StructuralMember]:
        """Design structural members for pitched roof."""
        if attachment_method == "shared_rail":
            rail_profile = "Shared rail 40mm"
            rail_quantity = layout["num_rails"]
        else:
            rail_profile = "Standard rail 40mm"
            rail_quantity = layout["num_rails"] * 2  # Two rails per module row

        members = [
            StructuralMember(
                member_type="rail",
                material=MaterialType.ALUMINUM,
                profile=rail_profile,
                length=layout["rail_length"],
                spacing=layout["rail_spacing"],
                quantity=rail_quantity,
                capacity=8.0,
                utilization=0.50,
            ),
            StructuralMember(
                member_type="L-foot",
                material=MaterialType.ALUMINUM,
                profile="L-foot with flashing",
                length=0.2,
                spacing=layout["attachment_spacing"],
                quantity=layout["num_attachments"],
                capacity=2.5,
                utilization=0.45,
            ),
        ]

        return members

    def _design_roof_attachment(self, uplift_force, compression_force):
        """Design roof attachment (for attached flat roof systems)."""
        return self.attachment_point_design(uplift_force, compression_force, "roof_anchor")

    def _generate_rooftop_bom(self, layout, members, foundation, ballasted) -> List[BillOfMaterials]:
        """Generate BOM for flat roof."""
        bom = []

        # Foundations/ballast
        if ballasted:
            bom.append(BillOfMaterials(
                item_number="BAL-001",
                description="Concrete ballast block",
                material=MaterialType.CONCRETE,
                specification=f"{foundation.length}m x {foundation.width}m",
                quantity=layout["num_supports"] * 4,  # 4 ballasts per support
                unit="ea",
                unit_weight=foundation.concrete_volume * 23.5 if foundation.concrete_volume else 100,
                total_weight=(foundation.concrete_volume * 23.5 if foundation.concrete_volume else 100) * layout["num_supports"] * 4,
                unit_cost=25.0,
                total_cost=25.0 * layout["num_supports"] * 4,
            ))

        # Structural members
        for i, member in enumerate(members):
            unit_weight = 15.0 if "frame" in member.member_type else 5.0
            unit_cost = 120.0 if "frame" in member.member_type else 30.0

            bom.append(BillOfMaterials(
                item_number=f"RF-{i+1:03d}",
                description=f"{member.member_type} - {member.profile}",
                material=member.material,
                specification=f"{member.length}m",
                quantity=member.quantity,
                unit="ea",
                unit_weight=unit_weight,
                total_weight=unit_weight * member.quantity,
                unit_cost=unit_cost,
                total_cost=unit_cost * member.quantity,
            ))

        # Module clamps
        bom.append(BillOfMaterials(
            item_number="HW-001",
            description="Module clamp assembly",
            material=MaterialType.ALUMINUM,
            specification="Universal clamp",
            quantity=self.config.num_modules * 4,
            unit="ea",
            unit_weight=0.3,
            total_weight=0.3 * self.config.num_modules * 4,
            unit_cost=3.5,
            total_cost=3.5 * self.config.num_modules * 4,
        ))

        return bom

    def _generate_pitched_roof_bom(self, layout, members, attachment, attachment_method) -> List[BillOfMaterials]:
        """Generate BOM for pitched roof."""
        bom = []

        # Attachments
        bom.append(BillOfMaterials(
            item_number="ATT-001",
            description=f"Roof attachment - {attachment.reinforcement}",
            material=MaterialType.ALUMINUM,
            specification="L-foot with flashing",
            quantity=layout["num_attachments"],
            unit="ea",
            unit_weight=1.5,
            total_weight=1.5 * layout["num_attachments"],
            unit_cost=18.0,
            total_cost=18.0 * layout["num_attachments"],
        ))

        # Rails and members
        for i, member in enumerate(members):
            if member.member_type == "rail":
                unit_weight = 2.5
                unit_cost = 45.0
            else:
                unit_weight = 1.5
                unit_cost = 18.0

            bom.append(BillOfMaterials(
                item_number=f"PR-{i+1:03d}",
                description=f"{member.member_type} - {member.profile}",
                material=member.material,
                specification=f"{member.length}m",
                quantity=member.quantity,
                unit="ea",
                unit_weight=unit_weight,
                total_weight=unit_weight * member.quantity,
                unit_cost=unit_cost,
                total_cost=unit_cost * member.quantity,
            ))

        # Module clamps
        bom.append(BillOfMaterials(
            item_number="HW-001",
            description="Module mid/end clamp",
            material=MaterialType.ALUMINUM,
            specification="Universal clamp",
            quantity=self.config.num_modules * 4,
            unit="ea",
            unit_weight=0.25,
            total_weight=0.25 * self.config.num_modules * 4,
            unit_cost=3.0,
            total_cost=3.0 * self.config.num_modules * 4,
        ))

        # Flashing boots
        bom.append(BillOfMaterials(
            item_number="WP-001",
            description="Flashing boot",
            material=MaterialType.COMPOSITE,
            specification="EPDM rubber boot",
            quantity=layout["num_attachments"],
            unit="ea",
            unit_weight=0.3,
            total_weight=0.3 * layout["num_attachments"],
            unit_cost=8.0,
            total_cost=8.0 * layout["num_attachments"],
        ))

        return bom

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
