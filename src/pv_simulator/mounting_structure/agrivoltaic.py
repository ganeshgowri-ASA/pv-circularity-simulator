"""Agrivoltaic (agriculture + PV) system design."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    AgrivoltaicConfig,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
)
from .structural_calculator import StructuralCalculator
from .foundation_engineer import FoundationEngineer


logger = logging.getLogger(__name__)


class AgrivoltaicDesign:
    """
    Agrivoltaic PV system design.

    Supports:
    - High-clearance structures for farm equipment
    - Row spacing optimization for crops
    - Bifacial module configurations
    - Crop-specific designs
    - Irrigation system integration
    - Seasonal tilt adjustment
    """

    # Crop-specific parameters
    CROP_PARAMETERS = {
        "wheat": {"min_light": 0.70, "equipment_height": 3.5, "row_spacing": 12},
        "corn": {"min_light": 0.75, "equipment_height": 4.0, "row_spacing": 15},
        "soybeans": {"min_light": 0.65, "equipment_height": 3.0, "row_spacing": 10},
        "vegetables": {"min_light": 0.60, "equipment_height": 2.5, "row_spacing": 8},
        "vineyard": {"min_light": 0.55, "equipment_height": 2.5, "row_spacing": 6},
        "grazing": {"min_light": 0.50, "equipment_height": 4.5, "row_spacing": 20},
        "orchards": {"min_light": 0.60, "equipment_height": 5.0, "row_spacing": 18},
    }

    def __init__(
        self,
        config: AgrivoltaicConfig,
        structural_calc: Optional[StructuralCalculator] = None,
        foundation_eng: Optional[FoundationEngineer] = None,
    ):
        """
        Initialize agrivoltaic designer.

        Args:
            config: Agrivoltaic configuration
            structural_calc: Structural calculator instance
            foundation_eng: Foundation engineer instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)
        self.foundation_eng = foundation_eng or FoundationEngineer(config.site_parameters)

    def high_clearance_structure(self) -> StructuralAnalysisResult:
        """
        Design elevated structure for farm equipment access.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing agrivoltaic system: clearance={self.config.clearance_height}m, crop={self.config.crop_type}")

        # Layout with high clearance
        layout = self._calculate_agrivoltaic_layout()

        # Load analysis at elevated height
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.clearance_height + 1.0,
            additional_dead_load=0.20,  # Elevated structure
            is_rooftop=False,
        )

        # Design elevated structure
        members = self._design_elevated_structure(layout, load_analysis)

        # Foundation design
        foundation = self.foundation_eng.pile_foundation_design(
            uplift_force=load_analysis.wind_load_uplift * layout["area_per_post"],
            compression_force=(load_analysis.dead_load + load_analysis.snow_load) * layout["area_per_post"],
            lateral_force=1.0,
        )

        # BOM
        bom = self._generate_agrivoltaic_bom(layout, members, foundation)

        # Crop compatibility analysis
        crop_analysis = self.crop_specific_design()

        # Row spacing analysis
        row_spacing_analysis = self.row_spacing_for_crops()

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.ALUMINUM])

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.030,
            deflection_limit=layout["span_length"] / 180,
            connection_details={
                "post_to_foundation": "Deep foundation for elevated loads",
                "beam_to_post": "Bolted connections for maintenance access",
                "equipment_clearance": f"{self.config.clearance_height}m minimum",
            },
            compliance_notes=[
                f"Equipment access: {layout['equipment_access_width']}m pathways",
                f"Crop light requirement: {crop_analysis['min_light_requirement']*100:.0f}% PAR",
                f"Row spacing: {row_spacing_analysis['recommended_spacing']}m",
                f"Bifacial modules: {'Yes' if self.config.bifacial_modules else 'No'}",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def row_spacing_for_crops(self) -> Dict[str, Any]:
        """
        Optimize row spacing for crop sunlight requirements.

        Returns:
            Dictionary with row spacing analysis
        """
        # Get crop parameters
        crop_params = self.CROP_PARAMETERS.get(
            self.config.crop_type.lower(),
            {"min_light": 0.65, "equipment_height": 3.5, "row_spacing": 12}
        )

        min_light_fraction = crop_params["min_light"]

        # Current row spacing
        current_spacing = self.config.row_spacing_for_crops

        # Module dimensions
        module_length = self.config.module_dimensions.length
        if self.config.tilt_angle > 0:
            # Slant height for tilted modules
            slant_height = module_length
        else:
            slant_height = module_length

        # Module width (perpendicular to row)
        module_width = slant_height * math.cos(math.radians(self.config.tilt_angle))

        # Shadow length calculation
        latitude = abs(self.config.site_parameters.latitude)
        # Solar altitude at solar noon (winter solstice)
        solar_altitude = 90 - latitude - 23.5

        if solar_altitude <= 0:
            solar_altitude = 10  # Minimum

        # Height of elevated modules
        h = self.config.clearance_height + slant_height * math.sin(math.radians(self.config.tilt_angle))

        # Shadow length
        shadow_length = h / math.tan(math.radians(solar_altitude))

        # Required row spacing for light requirement
        # Light fraction = 1 - (module_width / row_spacing)
        # Solve for row_spacing: row_spacing = module_width / (1 - light_fraction)
        required_spacing = (module_width + shadow_length) / min_light_fraction

        # Recommended spacing (from crop parameters or calculated)
        recommended_spacing = max(required_spacing, crop_params["row_spacing"])

        # Ground coverage ratio
        gcr = module_width / recommended_spacing

        # Light reaching crops
        light_fraction = 1.0 - gcr + (0.1 if self.config.bifacial_modules else 0)  # Bifacial adds diffuse light

        return {
            "crop_type": self.config.crop_type,
            "min_light_requirement": min_light_fraction,
            "current_spacing": current_spacing,
            "recommended_spacing": recommended_spacing,
            "shadow_length": shadow_length,
            "ground_coverage_ratio": gcr,
            "light_reaching_crops": min(light_fraction, 1.0),
            "spacing_adequate": current_spacing >= recommended_spacing,
        }

    def bifacial_agrivoltaic(self) -> Dict[str, Any]:
        """
        Design bifacial module agrivoltaic system.

        Returns:
            Dictionary with bifacial design parameters
        """
        if not self.config.bifacial_modules:
            return {"bifacial": False}

        # Bifacial gain calculation
        # Ground albedo (agricultural land)
        if "grazing" in self.config.crop_type.lower():
            albedo = 0.25  # Grass
        elif "vineyard" in self.config.crop_type.lower():
            albedo = 0.20  # Soil with vegetation
        else:
            albedo = 0.30  # Light-colored crop/soil

        # Bifacial gain depends on:
        # - Ground clearance (higher = more rear irradiance)
        # - Ground albedo
        # - Row spacing
        # - Module bifaciality (typically 70-80%)

        bifaciality = 0.75  # Typical bifacial factor

        # Simplified bifacial gain model
        # Rear irradiance fraction ≈ albedo * view_factor
        # View factor increases with clearance and row spacing

        clearance_factor = min(1.0, self.config.clearance_height / 4.0)  # Saturates at 4m
        spacing_factor = min(1.0, self.config.row_spacing_for_crops / 10.0)

        view_factor = 0.3 * clearance_factor * spacing_factor

        rear_irradiance_fraction = albedo * view_factor
        bifacial_gain = rear_irradiance_fraction * bifaciality * 100  # Percent

        return {
            "bifacial": True,
            "bifaciality": bifaciality,
            "ground_albedo": albedo,
            "clearance_height": self.config.clearance_height,
            "view_factor": view_factor,
            "rear_irradiance_fraction": rear_irradiance_fraction,
            "bifacial_gain_percent": bifacial_gain,
            "annual_energy_boost": bifacial_gain,  # Approximate
        }

    def crop_specific_design(self) -> Dict[str, Any]:
        """
        Crop-specific agrivoltaic design parameters.

        Returns:
            Dictionary with crop-specific recommendations
        """
        crop_params = self.CROP_PARAMETERS.get(
            self.config.crop_type.lower(),
            {"min_light": 0.65, "equipment_height": 3.5, "row_spacing": 12}
        )

        # Check if clearance meets requirement
        clearance_adequate = self.config.clearance_height >= crop_params["equipment_height"]

        # Recommendations based on crop
        crop_recommendations = {
            "wheat": [
                "Use higher tilt in winter for snow shedding",
                "Wide row spacing (12-15m) for combine harvester",
                "Bifacial modules recommended for ground reflection",
            ],
            "corn": [
                "Minimum 4m clearance for tall crop growth",
                "Consider adjustable tilt for growing season",
                "Wider spacing (15-20m) for equipment access",
            ],
            "vegetables": [
                "Lower tilt (10-20°) to reduce shading",
                "Closer spacing acceptable (6-10m)",
                "Consider translucent modules for uniform light",
            ],
            "vineyard": [
                "Align rows with vineyard rows",
                "Partial shading beneficial in hot climates",
                "Integrate with trellis system",
            ],
            "grazing": [
                "Very high clearance (4-5m) for livestock",
                "Wide spacing (20-25m) for pasture management",
                "Bifacial modules ideal for grass albedo",
            ],
        }

        recommendations = crop_recommendations.get(
            self.config.crop_type.lower(),
            ["Consult agricultural specialist for crop-specific design"]
        )

        return {
            "crop_type": self.config.crop_type,
            "min_light_requirement": crop_params["min_light"],
            "equipment_height_required": crop_params["equipment_height"],
            "recommended_row_spacing": crop_params["row_spacing"],
            "clearance_adequate": clearance_adequate,
            "recommendations": recommendations,
        }

    def irrigation_integration(self) -> Dict[str, Any]:
        """
        Integration with irrigation systems.

        Returns:
            Dictionary with irrigation integration details
        """
        if not self.config.irrigation_integration:
            return {"integrated": False}

        # Irrigation types compatible with agrivoltaics
        irrigation_types = {
            "drip": {
                "description": "Drip irrigation under panels",
                "compatibility": "Excellent",
                "installation": "Can route lines along post bases",
            },
            "sprinkler": {
                "description": "Sprinkler irrigation",
                "compatibility": "Good",
                "installation": "Design around panel rows, avoid wetting modules",
            },
            "pivot": {
                "description": "Center pivot irrigation",
                "compatibility": "Challenging",
                "installation": "Requires very high clearance and wide spacing",
            },
        }

        # Water collection from panels
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        total_panel_area = module_area * self.config.num_modules

        # Annual rainfall (assumed 600mm)
        annual_rainfall = 0.6  # m

        # Collectable water (80% efficiency)
        water_collected = total_panel_area * annual_rainfall * 0.8  # m³/year

        return {
            "integrated": True,
            "irrigation_types": irrigation_types,
            "rainwater_harvesting": {
                "panel_area": total_panel_area,
                "annual_rainfall": annual_rainfall,
                "water_collected_m3_year": water_collected,
                "irrigation_potential": "Supplement to irrigation system",
            },
            "design_considerations": [
                "Route irrigation lines along post bases",
                "Avoid wetting PV modules (reduces performance)",
                "Use collected water for crop irrigation",
                "Design drainage to prevent erosion under panels",
            ],
        }

    def seasonal_tilt_adjustment(self) -> Dict[str, float]:
        """
        Calculate optimal seasonal tilt adjustments.

        Returns:
            Dictionary with seasonal tilt recommendations
        """
        if not self.config.adjustable_tilt:
            return {
                "adjustable": False,
                "current_tilt": self.config.tilt_angle,
            }

        latitude = abs(self.config.site_parameters.latitude)

        # Optimal tilt by season (simplified)
        winter_tilt = min(latitude + 15, 60)  # Steeper in winter
        summer_tilt = max(latitude - 15, 10)  # Flatter in summer
        spring_fall_tilt = latitude  # At latitude

        # Crop considerations
        crop_params = self.CROP_PARAMETERS.get(
            self.config.crop_type.lower(),
            {"min_light": 0.65, "equipment_height": 3.5, "row_spacing": 12}
        )

        # Adjust for crop growing season
        # During growing season, flatten tilt to allow more light to crops
        growing_season_tilt = summer_tilt

        return {
            "adjustable": True,
            "winter_tilt": winter_tilt,
            "summer_tilt": summer_tilt,
            "spring_fall_tilt": spring_fall_tilt,
            "growing_season_tilt": growing_season_tilt,
            "adjustment_frequency": "Seasonal (4 times per year)",
            "energy_benefit": "5-10% annual energy increase vs fixed tilt",
            "crop_benefit": "Optimized light distribution by season",
        }

    # Private helper methods

    def _calculate_agrivoltaic_layout(self) -> Dict[str, Any]:
        """Calculate agrivoltaic layout."""
        # Elevated structure with wide row spacing
        row_spacing = self.config.row_spacing_for_crops

        # Equipment access width
        equipment_width = self.config.equipment_access_width

        # Module dimensions
        module_length = self.config.module_dimensions.length
        module_width = self.config.module_dimensions.width

        # Number of rows
        num_rows = max(1, int(self.config.num_modules / 20))  # 20 modules per row

        # Post spacing (wider for agrivoltaic)
        post_spacing = 6.0  # m

        # Span length between posts
        span_length = post_spacing

        # Posts per row
        posts_per_row = int(20 * module_length / post_spacing) + 1

        # Area per post
        area_per_post = post_spacing * row_spacing / 2

        return {
            "row_spacing": row_spacing,
            "equipment_access_width": equipment_width,
            "clearance_height": self.config.clearance_height,
            "post_spacing": post_spacing,
            "span_length": span_length,
            "num_rows": num_rows,
            "posts_per_row": posts_per_row,
            "area_per_post": area_per_post,
        }

    def _design_elevated_structure(self, layout: Dict, load_analysis: Any) -> List[StructuralMember]:
        """Design elevated agrivoltaic structure."""
        members = [
            StructuralMember(
                member_type="elevated_post",
                material=MaterialType.STEEL_GALVANIZED,
                profile="HSS6x6x3/8",
                length=layout["clearance_height"] + 1.5,
                spacing=layout["post_spacing"],
                quantity=layout["num_rows"] * layout["posts_per_row"],
                capacity=100.0,
                utilization=0.60,
            ),
            StructuralMember(
                member_type="primary_beam",
                material=MaterialType.STEEL_GALVANIZED,
                profile="W10x33",
                length=layout["span_length"],
                spacing=layout["row_spacing"],
                quantity=layout["num_rows"] * 2,
                capacity=80.0,
                utilization=0.65,
            ),
            StructuralMember(
                member_type="module_rail",
                material=MaterialType.ALUMINUM,
                profile="C-channel 6\"",
                length=3.0,
                spacing=1.0,
                quantity=self.config.num_modules * 2,
                capacity=10.0,
                utilization=0.50,
            ),
        ]

        # Add adjustable tilt mechanism if required
        if self.config.adjustable_tilt:
            members.append(
                StructuralMember(
                    member_type="tilt_actuator",
                    material=MaterialType.STEEL_GALVANIZED,
                    profile="Linear actuator assembly",
                    length=1.0,
                    spacing=6.0,
                    quantity=layout["num_rows"] * 5,
                    capacity=5.0,
                    utilization=0.40,
                )
            )

        return members

    def _generate_agrivoltaic_bom(
        self,
        layout: Dict,
        members: List[StructuralMember],
        foundation: Any,
    ) -> List[BillOfMaterials]:
        """Generate BOM for agrivoltaic system."""
        bom = []

        # Foundations
        total_foundations = layout["num_rows"] * layout["posts_per_row"]
        bom.append(BillOfMaterials(
            item_number="FND-001",
            description=f"Deep foundation - {foundation.foundation_type.value}",
            material=foundation.material,
            specification=f"{foundation.depth}m deep",
            quantity=total_foundations,
            unit="ea",
            unit_weight=100.0,
            total_weight=100.0 * total_foundations,
            unit_cost=150.0,
            total_cost=150.0 * total_foundations,
        ))

        # Structural members
        for i, member in enumerate(members):
            if "post" in member.member_type:
                unit_weight = 80.0
                unit_cost = 280.0
            elif "beam" in member.member_type:
                unit_weight = 50.0
                unit_cost = 220.0
            elif "actuator" in member.member_type:
                unit_weight = 10.0
                unit_cost = 850.0
            else:
                unit_weight = 8.0
                unit_cost = 45.0

            bom.append(BillOfMaterials(
                item_number=f"AGR-{i+1:03d}",
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
            specification="Heavy-duty clamp",
            quantity=self.config.num_modules * 4,
            unit="ea",
            unit_weight=0.4,
            total_weight=0.4 * self.config.num_modules * 4,
            unit_cost=4.5,
            total_cost=4.5 * self.config.num_modules * 4,
        ))

        # Irrigation integration (if applicable)
        if self.config.irrigation_integration:
            bom.append(BillOfMaterials(
                item_number="IRR-001",
                description="Rainwater collection system",
                material=MaterialType.HDPE,
                specification="Gutters and downspouts",
                quantity=layout["num_rows"] * 10,
                unit="m",
                unit_weight=2.0,
                total_weight=2.0 * layout["num_rows"] * 10,
                unit_cost=15.0,
                total_cost=15.0 * layout["num_rows"] * 10,
            ))

        return bom

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
