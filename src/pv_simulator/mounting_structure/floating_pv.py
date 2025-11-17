"""Floating PV system design for water bodies."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    FloatingPVConfig,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
    FoundationType,
    FoundationDesign,
)
from .structural_calculator import StructuralCalculator


logger = logging.getLogger(__name__)


class FloatingPVDesign:
    """
    Floating PV system design.

    Supports:
    - Pontoon layout and spacing
    - Anchoring and mooring systems
    - Wave impact analysis
    - Evaporative cooling modeling
    - Environmental considerations
    - Tilt angle optimization for floating systems
    """

    def __init__(
        self,
        config: FloatingPVConfig,
        structural_calc: Optional[StructuralCalculator] = None,
    ):
        """
        Initialize floating PV designer.

        Args:
            config: Floating PV configuration
            structural_calc: Structural calculator instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)

    def pontoon_layout(self) -> StructuralAnalysisResult:
        """
        Design pontoon layout and module mounting.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing floating PV: coverage={self.config.coverage_ratio}, tilt={self.config.tilt_angle}°")

        # Layout calculations
        layout = self._calculate_floating_layout()

        # Load analysis (reduced wind loads for low-tilt floating systems)
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=0.5,  # Low height above water
            additional_dead_load=0.10,  # Lightweight floating structure
            is_rooftop=False,
        )

        # Wave impact analysis
        wave_analysis = self.wave_impact_analysis()

        # Anchoring system
        anchoring = self.anchoring_system()

        # Structural members (pontoons and frames)
        members = self._design_floating_structure(layout, load_analysis)

        # BOM
        bom = self._generate_floating_bom(layout, members, anchoring)

        # Cooling benefit
        cooling_benefit = self.cooling_benefit_modeling() if self.config.cooling_benefit else {}

        total_weight = sum(item.total_weight or 0 for item in bom)

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=anchoring["foundation_design"],
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=0.05,  # Flexible system
            deflection_limit=0.10,  # More tolerant for floating
            connection_details={
                "pontoon_connections": "HDPE hinged connections for flexibility",
                "anchoring": anchoring["type"],
                "mooring_lines": f"{anchoring['num_anchors']} anchor points",
            },
            compliance_notes=[
                f"Water coverage ratio: {self.config.coverage_ratio*100:.1f}%",
                f"Wave design height: {wave_analysis['design_wave_height']}m",
                f"Cooling benefit: {cooling_benefit.get('temperature_reduction', 0):.1f}°C" if cooling_benefit else "No cooling model",
                "Environmental impact assessment required",
            ],
            total_steel_weight=0,  # Primarily HDPE
            total_cost_estimate=self._estimate_cost(bom),
        )

    def anchoring_system(self) -> Dict[str, Any]:
        """
        Design anchoring and mooring system.

        Returns:
            Dictionary with anchoring specifications
        """
        logger.info(f"Designing anchoring system for {self.config.water_body_type}")

        # Total array area
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        total_array_area = module_area * self.config.num_modules

        # Array dimensions (assumed square-ish)
        array_side = math.sqrt(total_array_area)

        # Number of anchor points (perimeter + internal)
        # Typical: one anchor per 500-1000 m² of array
        num_anchors = max(4, int(total_array_area / 750))

        # Anchor type based on water depth and bottom type
        if self.config.water_depth < 10:
            anchor_type = "driven_pile"
            anchor_capacity = 50  # kN per anchor
        elif self.config.water_depth < 30:
            anchor_type = "screw_anchor"
            anchor_capacity = 40  # kN
        else:
            anchor_type = "deadweight_anchor"
            anchor_capacity = 30  # kN

        # Mooring line tension (wind + current)
        # Simplified: based on wind load on array
        wind_force_total = abs(self.structural_calc.wind_load_analysis(
            tilt_angle=self.config.tilt_angle,
            height=0.5,
            module_dimensions=self.config.module_dimensions,
        )["uplift_pressure"]) * total_array_area

        # Mooring line tension per anchor
        tension_per_anchor = wind_force_total / num_anchors * 1.5  # Safety factor

        # Mooring line specification
        if tension_per_anchor < 20:
            line_spec = "25mm polyester rope"
            line_strength = 30  # kN
        elif tension_per_anchor < 40:
            line_spec = "32mm polyester rope"
            line_strength = 60  # kN
        else:
            line_spec = "40mm polyester rope"
            line_strength = 100  # kN

        foundation = FoundationDesign(
            foundation_type=FoundationType.PONTOON,
            depth=self.config.water_depth,
            diameter=None,
            length=array_side,
            width=array_side,
            capacity=anchor_capacity,
            spacing=array_side / math.sqrt(num_anchors),
            quantity=num_anchors,
            material=MaterialType.STEEL_GALVANIZED if "pile" in anchor_type else MaterialType.CONCRETE,
            embedment_depth=5.0 if "pile" in anchor_type else 0,
            reinforcement=f"{anchor_type} with {line_spec}",
        )

        return {
            "type": anchor_type,
            "num_anchors": num_anchors,
            "anchor_capacity": anchor_capacity,
            "mooring_line_spec": line_spec,
            "mooring_line_strength": line_strength,
            "tension_per_anchor": tension_per_anchor,
            "water_depth": self.config.water_depth,
            "foundation_design": foundation,
        }

    def wave_impact_analysis(self) -> Dict[str, Any]:
        """
        Analyze wave action and water level variation.

        Returns:
            Dictionary with wave impact assessment
        """
        # Design wave height (based on fetch and wind)
        wave_height = self.config.max_wave_height

        # Wave period (simplified Airy wave theory)
        # T ≈ 2π * sqrt(H/g) for deep water
        g = 9.81  # m/s²
        wave_period = 2 * math.pi * math.sqrt(wave_height / g) if wave_height > 0 else 0

        # Wave forces (simplified)
        # F = 0.5 * ρ * g * H² * W (per unit width)
        rho_water = 1000  # kg/m³
        module_width = self.config.module_dimensions.width
        wave_force_per_m = 0.5 * rho_water * g * wave_height**2 / 1000  # kN/m

        # Total wave force on array
        num_rows = int(math.sqrt(self.config.num_modules))
        total_wave_force = wave_force_per_m * module_width * num_rows

        # Freeboard requirement (height above water)
        # Minimum freeboard = 1.5 * wave height + water level variation
        min_freeboard = 1.5 * wave_height + self.config.water_level_variation

        return {
            "max_wave_height": wave_height,
            "wave_period": wave_period,
            "wave_force_per_meter": wave_force_per_m,
            "total_wave_force": total_wave_force,
            "design_wave_height": wave_height * 1.2,  # Design factor
            "min_freeboard": min_freeboard,
            "water_level_variation": self.config.water_level_variation,
        }

    def cooling_benefit_modeling(self) -> Dict[str, float]:
        """
        Model evaporative cooling effects on module temperature.

        Returns:
            Dictionary with cooling benefit analysis
        """
        # Floating PV typically runs 5-15°C cooler than ground-mount
        # Due to evaporative cooling and air circulation underneath

        # Ambient temperature (assumed)
        ambient_temp = 25  # °C

        # Ground-mount operating temperature (simplified)
        # T_module = T_ambient + (NOCT - 20) * (Irradiance / 800)
        # Assume NOCT = 45°C, Irradiance = 1000 W/m²
        NOCT = 45
        irradiance = 1000  # W/m²
        temp_ground_mount = ambient_temp + (NOCT - 20) * (irradiance / 800)

        # Floating PV cooling benefit (empirical)
        # Cooling benefit increases with water temperature differential
        # Assume water temp = ambient temp (conservative)
        cooling_benefit = 10  # °C typical reduction

        temp_floating = temp_ground_mount - cooling_benefit

        # Power output increase due to lower temperature
        # PV temperature coefficient: typically -0.4%/°C
        temp_coeff = -0.004  # per °C

        power_gain = cooling_benefit * abs(temp_coeff) * 100  # Percentage

        return {
            "ambient_temperature": ambient_temp,
            "ground_mount_temperature": temp_ground_mount,
            "floating_temperature": temp_floating,
            "temperature_reduction": cooling_benefit,
            "power_gain_percent": power_gain,
        }

    def environmental_considerations(self) -> Dict[str, Any]:
        """
        Environmental impact considerations for floating PV.

        Returns:
            Dictionary with environmental guidelines
        """
        # Water coverage limits (ecological)
        max_coverage = 0.40  # 40% maximum recommended
        current_coverage = self.config.coverage_ratio

        # Shading impact on aquatic life
        if current_coverage < 0.10:
            shading_impact = "Minimal"
        elif current_coverage < 0.25:
            shading_impact = "Low to moderate"
        elif current_coverage < 0.40:
            shading_impact = "Moderate"
        else:
            shading_impact = "High - ecological study required"

        # Water quality monitoring points
        monitoring_points = [
            "Dissolved oxygen levels",
            "Water temperature",
            "pH levels",
            "Algae growth",
            "Fish population",
        ]

        # Setbacks from shore (typical)
        shore_setback = 15  # m

        return {
            "coverage_ratio": current_coverage,
            "max_recommended_coverage": max_coverage,
            "coverage_acceptable": current_coverage <= max_coverage,
            "shading_impact": shading_impact,
            "monitoring_requirements": monitoring_points,
            "shore_setback": shore_setback,
            "ecological_study_required": current_coverage > 0.30,
            "recommendations": [
                "Minimize coverage to protect aquatic ecosystem",
                "Avoid spawning areas and migration routes",
                "Allow light penetration for photosynthesis",
                "Monitor water quality regularly",
            ],
        }

    def tilt_angle_optimization(self) -> Dict[str, float]:
        """
        Optimize tilt angle for floating PV (typically low tilt).

        Returns:
            Dictionary with tilt angle recommendations
        """
        # Floating PV typically uses low tilt (5-15°) for:
        # 1. Reduced wind loads
        # 2. Lower wave impact
        # 3. Easier installation
        # 4. More stable platform

        latitude = abs(self.config.site_parameters.latitude)

        # Optimal tilt for ground-mount would be ≈ latitude
        optimal_ground_tilt = latitude

        # Floating PV compromise (lower for stability)
        if latitude < 15:
            recommended_tilt = 10
        elif latitude < 30:
            recommended_tilt = 12
        elif latitude < 45:
            recommended_tilt = 15
        else:
            recommended_tilt = 15  # Maximum for floating

        current_tilt = self.config.tilt_angle

        # Energy yield impact (simplified)
        # Lower tilt reduces annual energy but improves summer production
        if current_tilt < 10:
            energy_penalty = 2  # % relative to optimal
        elif current_tilt < 15:
            energy_penalty = 1
        else:
            energy_penalty = 0

        return {
            "current_tilt": current_tilt,
            "recommended_tilt": recommended_tilt,
            "optimal_ground_tilt": optimal_ground_tilt,
            "energy_penalty_percent": energy_penalty,
            "benefits_of_low_tilt": [
                "Reduced wind loads",
                "Lower wave impact",
                "Improved platform stability",
                "Better summer production",
            ],
        }

    # Private helper methods

    def _calculate_floating_layout(self) -> Dict[str, Any]:
        """Calculate floating array layout."""
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        total_module_area = module_area * self.config.num_modules

        # Water surface area required
        water_area_required = total_module_area / self.config.coverage_ratio

        # Array dimensions (assumed rectangular)
        aspect_ratio = 1.5  # Typical L:W ratio
        array_width = math.sqrt(water_area_required / aspect_ratio)
        array_length = array_width * aspect_ratio

        # Number of pontoons
        # Typical: one pontoon per module or per 2 modules
        pontoons_per_module = 1.0
        num_pontoons = int(self.config.num_modules * pontoons_per_module)

        # Pontoon dimensions (typical HDPE floats)
        pontoon_length = 4.0  # m
        pontoon_width = 0.4  # m
        pontoon_height = 0.3  # m

        # Buoyancy calculation
        # Buoyancy = ρ * g * V_displaced
        # Must exceed total weight (modules + structure)
        module_weight = self.config.module_dimensions.weight * self.config.num_modules  # kg
        structure_weight = num_pontoons * 20  # kg (20 kg per pontoon assembly)
        total_weight = module_weight + structure_weight

        # Required displaced volume
        rho_water = 1000  # kg/m³
        safety_factor = 1.5  # Freeboard
        volume_required = (total_weight * safety_factor) / rho_water  # m³

        # Actual volume from pontoons
        volume_per_pontoon = pontoon_length * pontoon_width * pontoon_height * 0.7  # 70% submerged
        total_volume = volume_per_pontoon * num_pontoons

        buoyancy_adequate = total_volume >= volume_required

        return {
            "array_length": array_length,
            "array_width": array_width,
            "water_area": water_area_required,
            "num_pontoons": num_pontoons,
            "pontoon_length": pontoon_length,
            "pontoon_width": pontoon_width,
            "pontoon_height": pontoon_height,
            "pontoon_spacing": self.config.pontoon_spacing,
            "total_weight": total_weight,
            "buoyancy_volume": total_volume,
            "buoyancy_adequate": buoyancy_adequate,
        }

    def _design_floating_structure(self, layout: Dict, load_analysis: Any) -> List[StructuralMember]:
        """Design floating structure members."""
        members = [
            StructuralMember(
                member_type="pontoon",
                material=MaterialType.HDPE,
                profile=f"{layout['pontoon_length']}m x {layout['pontoon_width']}m HDPE float",
                length=layout["pontoon_length"],
                spacing=layout["pontoon_spacing"],
                quantity=layout["num_pontoons"],
                capacity=500,  # kg buoyancy per pontoon
                utilization=layout["total_weight"] / (layout["num_pontoons"] * 500) if layout["num_pontoons"] > 0 else 0,
            ),
            StructuralMember(
                member_type="mounting_frame",
                material=MaterialType.ALUMINUM,
                profile="Aluminum frame for floating modules",
                length=2.0,
                spacing=2.0,
                quantity=int(self.config.num_modules / 4),  # One frame per 4 modules
                capacity=20,  # kN
                utilization=0.40,
            ),
        ]
        return members

    def _generate_floating_bom(
        self,
        layout: Dict,
        members: List[StructuralMember],
        anchoring: Dict,
    ) -> List[BillOfMaterials]:
        """Generate BOM for floating PV."""
        bom = []

        # Pontoons
        bom.append(BillOfMaterials(
            item_number="FLT-001",
            description="HDPE pontoon float",
            material=MaterialType.HDPE,
            specification=f"{layout['pontoon_length']}m x {layout['pontoon_width']}m x {layout['pontoon_height']}m",
            quantity=layout["num_pontoons"],
            unit="ea",
            unit_weight=20.0,
            total_weight=20.0 * layout["num_pontoons"],
            unit_cost=180.0,
            total_cost=180.0 * layout["num_pontoons"],
        ))

        # Mounting frames
        bom.append(BillOfMaterials(
            item_number="FLT-002",
            description="Aluminum mounting frame",
            material=MaterialType.ALUMINUM,
            specification="Lightweight floating frame",
            quantity=int(self.config.num_modules / 4),
            unit="ea",
            unit_weight=15.0,
            total_weight=15.0 * int(self.config.num_modules / 4),
            unit_cost=250.0,
            total_cost=250.0 * int(self.config.num_modules / 4),
        ))

        # Anchors and mooring
        bom.append(BillOfMaterials(
            item_number="ANC-001",
            description=f"Anchor system - {anchoring['type']}",
            material=MaterialType.STEEL_GALVANIZED,
            specification=f"{anchoring['anchor_capacity']}kN capacity",
            quantity=anchoring["num_anchors"],
            unit="ea",
            unit_weight=50.0,
            total_weight=50.0 * anchoring["num_anchors"],
            unit_cost=450.0,
            total_cost=450.0 * anchoring["num_anchors"],
        ))

        # Mooring lines
        bom.append(BillOfMaterials(
            item_number="MOR-001",
            description=f"Mooring line - {anchoring['mooring_line_spec']}",
            material=MaterialType.COMPOSITE,
            specification=f"{anchoring['mooring_line_strength']}kN strength",
            quantity=anchoring["num_anchors"] * 50,  # 50m per anchor
            unit="m",
            unit_weight=0.5,
            total_weight=0.5 * anchoring["num_anchors"] * 50,
            unit_cost=8.0,
            total_cost=8.0 * anchoring["num_anchors"] * 50,
        ))

        # Module clamps
        bom.append(BillOfMaterials(
            item_number="HW-001",
            description="Module clamp for floating",
            material=MaterialType.STAINLESS_STEEL,
            specification="Corrosion-resistant clamp",
            quantity=self.config.num_modules * 4,
            unit="ea",
            unit_weight=0.3,
            total_weight=0.3 * self.config.num_modules * 4,
            unit_cost=5.0,
            total_cost=5.0 * self.config.num_modules * 4,
        ))

        return bom

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
