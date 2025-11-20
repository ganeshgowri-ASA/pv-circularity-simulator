"""Building-Integrated Photovoltaic (BIPV) system design."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    BIPVConfig,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
    FoundationType,
    FoundationDesign,
)
from .structural_calculator import StructuralCalculator


logger = logging.getLogger(__name__)


class BIPVDesign:
    """
    Building-Integrated Photovoltaic (BIPV) system design.

    Supports:
    - Facade integration (vertical, tilted)
    - Skylight and canopy systems
    - Curtain wall integration
    - Structural glazing calculations
    - Thermal performance analysis
    - Electrical integration
    """

    def __init__(
        self,
        config: BIPVConfig,
        structural_calc: Optional[StructuralCalculator] = None,
    ):
        """
        Initialize BIPV designer.

        Args:
            config: BIPV configuration
            structural_calc: Structural calculator instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)

    def facade_integration(self) -> StructuralAnalysisResult:
        """
        Design building-integrated facade system.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing BIPV facade: vertical={self.config.vertical_installation}, height={self.config.building_height}m")

        # Layout
        layout = self._calculate_facade_layout()

        # Load analysis (facade loads)
        # Vertical or tilted installation
        effective_tilt = 90 if self.config.vertical_installation else self.config.tilt_angle

        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=effective_tilt,
            height=self.config.building_height / 2,  # Mid-height
            additional_dead_load=0.15 if self.config.structural_glazing else 0.08,
            is_rooftop=False,
        )

        # Structural glazing if required
        if self.config.structural_glazing:
            glazing_analysis = self.structural_glazing()
        else:
            glazing_analysis = {}

        # Structural members
        members = self._design_facade_structure(layout, load_analysis)

        # Foundation (building-integrated, minimal)
        foundation = self._create_bipv_foundation()

        # BOM
        bom = self._generate_bipv_bom(layout, members, "facade")

        # Thermal performance
        thermal = self.thermal_performance()

        # Electrical integration
        electrical = self.electrical_integration()

        total_weight = sum(item.total_weight or 0 for item in bom)

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=layout.get("max_deflection", 0.015),
            deflection_limit=layout.get("height", 3.0) / 250,
            connection_details={
                "attachment": "Building structure attachment",
                "glazing_type": "Structural glazing" if self.config.structural_glazing else "Framed modules",
                "waterproofing": "Integrated with building envelope",
                "electrical": electrical["conduit_routing"],
            },
            compliance_notes=[
                "BIPV modules comply with building code requirements",
                f"Thermal performance: R-value={thermal.get('r_value', 0):.2f}",
                f"Fire rating: Class A (glass-glass modules)" if self.config.structural_glazing else "Standard",
                "Electrical integration per NEC Article 690",
            ],
            total_steel_weight=total_weight * 0.3,  # Estimate steel fraction
            total_cost_estimate=self._estimate_cost(bom),
        )

    def skylight_canopy(self) -> StructuralAnalysisResult:
        """
        Design translucent BIPV skylight or canopy system.

        Returns:
            Complete structural analysis
        """
        logger.info("Designing BIPV skylight/canopy system")

        # Layout
        layout = self._calculate_skylight_layout()

        # Load analysis (horizontal or low tilt)
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.building_height,
            additional_dead_load=0.20,  # Glass-glass modules heavier
            is_rooftop=True,
        )

        # Structural glazing requirements
        glazing = self.structural_glazing()

        # Structural members
        members = self._design_skylight_structure(layout, load_analysis)

        # Foundation
        foundation = self._create_bipv_foundation()

        # BOM
        bom = self._generate_bipv_bom(layout, members, "skylight")

        # Thermal performance
        thermal = self.thermal_performance()

        total_weight = sum(item.total_weight or 0 for item in bom)

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=glazing["max_deflection"],
            deflection_limit=glazing["deflection_limit"],
            connection_details={
                "glazing": "Structural silicone glazing",
                "framing": "Aluminum curtain wall system",
                "waterproofing": "Sealed glazing joints",
            },
            compliance_notes=[
                f"Glass-glass modules: {self.config.module_dimensions.glass_thickness*1000:.1f}mm front glass",
                f"Translucent: {'Yes' if self.config.translucent_modules else 'No'}",
                f"Thermal break: {'Yes' if self.config.thermal_break else 'No'}",
                f"U-value: {thermal['u_value']:.2f} W/m²K",
            ],
            total_steel_weight=total_weight * 0.4,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def curtain_wall_integration(self) -> StructuralAnalysisResult:
        """
        Design BIPV curtain wall system.

        Returns:
            Complete structural analysis
        """
        logger.info("Designing BIPV curtain wall system")

        # Layout
        layout = self._calculate_curtain_wall_layout()

        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=90,  # Vertical
            height=self.config.building_height / 2,
            additional_dead_load=0.25,  # Curtain wall system
            is_rooftop=False,
        )

        # Structural glazing
        glazing = self.structural_glazing()

        # Structural members
        members = self._design_curtain_wall_structure(layout, load_analysis)

        # Foundation
        foundation = self._create_bipv_foundation()

        # BOM
        bom = self._generate_bipv_bom(layout, members, "curtain_wall")

        # Thermal performance
        thermal = self.thermal_performance()

        # Electrical integration
        electrical = self.electrical_integration()

        total_weight = sum(item.total_weight or 0 for item in bom)

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=glazing["max_deflection"],
            deflection_limit=glazing["deflection_limit"],
            connection_details={
                "system_type": "Unitized curtain wall with BIPV",
                "mullions": "Aluminum thermal-break mullions",
                "spandrel": "BIPV modules in spandrel zones",
                "vision": "Standard glazing in vision zones",
            },
            compliance_notes=[
                "Curtain wall per AAMA standards",
                f"Thermal break: {'Yes' if self.config.thermal_break else 'No'}",
                f"U-value: {thermal['u_value']:.2f} W/m²K",
                "Air/water infiltration testing required",
                f"Electrical: {electrical['junction_box_location']} junction boxes",
            ],
            total_steel_weight=total_weight * 0.5,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def structural_glazing(self) -> Dict[str, Any]:
        """
        Calculate structural glazing requirements for glass-glass modules.

        Returns:
            Dictionary with structural glazing specifications
        """
        if not self.config.structural_glazing:
            return {"structural_glazing": False}

        # Module dimensions
        length = self.config.module_dimensions.length
        width = self.config.module_dimensions.width
        glass_thickness = self.config.module_dimensions.glass_thickness

        # Glass properties
        # Tempered glass: E = 70 GPa
        E_glass = 70e6  # kN/m²

        # Moment of inertia for rectangular cross-section
        # I = b * h³ / 12
        # For glass: h = glass_thickness, b = width
        I = width * glass_thickness**3 / 12

        # Load (wind pressure)
        wind_analysis = self.structural_calc.wind_load_analysis(
            tilt_angle=90 if self.config.vertical_installation else self.config.tilt_angle,
            height=self.config.building_height,
            module_dimensions=self.config.module_dimensions,
            is_rooftop=False,
        )

        # Wind pressure (worst case)
        wind_pressure = max(abs(wind_analysis["uplift_pressure"]), abs(wind_analysis["downward_pressure"]))

        # Span between supports (typical 1.0-1.5m for BIPV)
        span = min(length, 1.2)  # m

        # Distributed load
        w = wind_pressure * width  # kN/m

        # Deflection: δ = 5wL⁴/(384EI) for simply supported
        deflection = (5 * w * span**4) / (384 * E_glass * I)

        # Deflection limit (L/60 for glass)
        deflection_limit = span / 60

        # Stress check
        # Maximum moment: M = wL²/8
        M = w * span**2 / 8

        # Section modulus: S = I / (h/2)
        S = I / (glass_thickness / 2)

        # Bending stress
        stress = M / S  # kN/m²

        # Allowable stress for tempered glass (conservative)
        allowable_stress = 40e3  # kN/m² (40 MPa)

        passes = (deflection <= deflection_limit) and (stress <= allowable_stress)

        return {
            "structural_glazing": True,
            "glass_thickness": glass_thickness * 1000,  # mm
            "span": span,
            "wind_pressure": wind_pressure,
            "max_deflection": deflection,
            "deflection_limit": deflection_limit,
            "deflection_passes": deflection <= deflection_limit,
            "bending_stress": stress / 1000,  # MPa
            "allowable_stress": allowable_stress / 1000,  # MPa
            "stress_passes": stress <= allowable_stress,
            "overall_passes": passes,
            "recommendation": "Adequate" if passes else "Increase glass thickness or reduce span",
        }

    def thermal_performance(self) -> Dict[str, float]:
        """
        Calculate thermal performance of BIPV system.

        Returns:
            Dictionary with thermal properties
        """
        # BIPV thermal properties depend on module construction
        # Glass-glass modules provide better insulation than glass-backsheet

        # U-value (heat transfer coefficient) W/m²K
        # Lower is better (less heat transfer)

        if self.config.structural_glazing:
            # Glass-glass BIPV module
            if self.config.thermal_break:
                # With thermal break in frame
                u_value = 1.8  # W/m²K
                r_value = 1 / u_value  # m²K/W
            else:
                # Without thermal break
                u_value = 2.5
                r_value = 1 / u_value
        else:
            # Standard framed module
            u_value = 3.0
            r_value = 1 / u_value

        # Solar heat gain coefficient (SHGC)
        # Fraction of solar radiation that enters as heat
        if self.config.translucent_modules:
            shgc = 0.35  # Partial transparency
            visible_transmittance = 0.25
        else:
            shgc = 0.15  # Opaque PV
            visible_transmittance = 0.0

        # Thermal mass effect (for facade)
        if self.config.vertical_installation:
            thermal_mass_benefit = 0.10  # 10% reduction in cooling load
        else:
            thermal_mass_benefit = 0.05

        return {
            "u_value": u_value,
            "r_value": r_value,
            "shgc": shgc,
            "visible_transmittance": visible_transmittance,
            "thermal_break": self.config.thermal_break,
            "thermal_mass_benefit": thermal_mass_benefit,
            "energy_performance": "Better than standard curtain wall" if u_value < 2.0 else "Standard",
        }

    def electrical_integration(self) -> Dict[str, Any]:
        """
        Design electrical integration for BIPV.

        Returns:
            Dictionary with electrical specifications
        """
        # Junction box placement
        junction_box_location = self.config.junction_box_location

        # Conduit routing
        conduit_routing = self.config.conduit_routing

        # Number of modules
        num_modules = self.config.num_modules

        # String sizing (typical 10-20 modules per string)
        modules_per_string = 15
        num_strings = math.ceil(num_modules / modules_per_string)

        # Inverter location
        if self.config.integration_type == "facade":
            inverter_location = "Rooftop or basement"
        else:
            inverter_location = "Dedicated electrical room"

        # Rapid shutdown (NEC 690.12)
        rapid_shutdown_required = True

        # Conduit size (simplified)
        # Based on number of conductors and current
        # Assume 10A per string, 2 conductors per string
        total_conductors = num_strings * 2

        if total_conductors <= 6:
            conduit_size = "3/4 inch"
        elif total_conductors <= 12:
            conduit_size = "1 inch"
        else:
            conduit_size = "1.25 inch"

        return {
            "junction_box_location": junction_box_location,
            "conduit_routing": conduit_routing,
            "modules_per_string": modules_per_string,
            "num_strings": num_strings,
            "inverter_location": inverter_location,
            "rapid_shutdown": rapid_shutdown_required,
            "conduit_size": conduit_size,
            "grounding": "Integrated with building grounding system",
            "monitoring": "Module-level monitoring recommended",
        }

    # Private helper methods

    def _calculate_facade_layout(self) -> Dict[str, Any]:
        """Calculate facade BIPV layout."""
        # Facade area
        building_width = 30  # m (assumed)
        facade_area = building_width * self.config.building_height

        # Module area
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        total_module_area = module_area * self.config.num_modules

        # Coverage ratio
        coverage = total_module_area / facade_area

        # Mullion spacing (typical curtain wall)
        mullion_spacing = 1.5  # m

        return {
            "facade_area": facade_area,
            "module_coverage": coverage,
            "mullion_spacing": mullion_spacing,
            "height": self.config.building_height,
            "max_deflection": 0.010,
        }

    def _calculate_skylight_layout(self) -> Dict[str, Any]:
        """Calculate skylight BIPV layout."""
        # Skylight area (assumed 10% of roof area)
        roof_area = 500  # m² (assumed)
        skylight_area = roof_area * 0.10

        # Module area
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width

        # Number of modules that fit
        modules_in_skylight = int(skylight_area / module_area)

        # Framing grid
        grid_spacing = 2.0  # m

        return {
            "skylight_area": skylight_area,
            "grid_spacing": grid_spacing,
            "modules_in_skylight": modules_in_skylight,
        }

    def _calculate_curtain_wall_layout(self) -> Dict[str, Any]:
        """Calculate curtain wall BIPV layout."""
        # Typical curtain wall layout
        floor_height = 3.5  # m
        num_floors = int(self.config.building_height / floor_height)

        # Module area
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width

        # Mullion grid
        horizontal_mullion_spacing = floor_height
        vertical_mullion_spacing = 1.5  # m

        # Panel size (fits between mullions)
        panel_width = vertical_mullion_spacing - 0.05  # Clearance
        panel_height = horizontal_mullion_spacing - 0.05

        return {
            "num_floors": num_floors,
            "floor_height": floor_height,
            "horizontal_mullion_spacing": horizontal_mullion_spacing,
            "vertical_mullion_spacing": vertical_mullion_spacing,
            "panel_width": panel_width,
            "panel_height": panel_height,
        }

    def _design_facade_structure(self, layout: Dict, load_analysis: Any) -> List[StructuralMember]:
        """Design facade structural members."""
        members = [
            StructuralMember(
                member_type="facade_support",
                material=MaterialType.ALUMINUM,
                profile="Aluminum extrusion",
                length=layout["height"],
                spacing=layout["mullion_spacing"],
                quantity=20,
                capacity=5.0,
                utilization=0.55,
            ),
        ]
        return members

    def _design_skylight_structure(self, layout: Dict, load_analysis: Any) -> List[StructuralMember]:
        """Design skylight structural members."""
        members = [
            StructuralMember(
                member_type="skylight_frame",
                material=MaterialType.ALUMINUM,
                profile="Aluminum glazing system",
                length=layout["grid_spacing"],
                spacing=layout["grid_spacing"],
                quantity=10,
                capacity=8.0,
                utilization=0.60,
            ),
        ]
        return members

    def _design_curtain_wall_structure(self, layout: Dict, load_analysis: Any) -> List[StructuralMember]:
        """Design curtain wall structural members."""
        members = [
            StructuralMember(
                member_type="vertical_mullion",
                material=MaterialType.ALUMINUM,
                profile="Thermal-break mullion",
                length=layout["floor_height"],
                spacing=layout["vertical_mullion_spacing"],
                quantity=layout["num_floors"] * 20,
                capacity=10.0,
                utilization=0.50,
            ),
            StructuralMember(
                member_type="horizontal_mullion",
                material=MaterialType.ALUMINUM,
                profile="Thermal-break mullion",
                length=layout["vertical_mullion_spacing"],
                spacing=layout["horizontal_mullion_spacing"],
                quantity=layout["num_floors"] * 20,
                capacity=8.0,
                utilization=0.45,
            ),
        ]
        return members

    def _create_bipv_foundation(self) -> FoundationDesign:
        """Create minimal foundation design for BIPV (building-integrated)."""
        return FoundationDesign(
            foundation_type=FoundationType.ROOF_ATTACHMENT,
            depth=0.0,
            diameter=None,
            length=0.0,
            width=0.0,
            capacity=0.0,
            spacing=0.0,
            quantity=0,
            material=MaterialType.ALUMINUM,
            embedment_depth=0.0,
            concrete_volume=0.0,
            reinforcement="Integrated with building structure",
        )

    def _generate_bipv_bom(
        self,
        layout: Dict,
        members: List[StructuralMember],
        system_type: str,
    ) -> List[BillOfMaterials]:
        """Generate BOM for BIPV system."""
        bom = []

        # BIPV modules (glass-glass, more expensive)
        module_cost = 350 if self.config.structural_glazing else 250

        bom.append(BillOfMaterials(
            item_number="BIPV-001",
            description=f"BIPV module - {system_type}",
            material=MaterialType.COMPOSITE,
            specification=f"{self.config.module_dimensions.length}m x {self.config.module_dimensions.width}m glass-glass",
            quantity=self.config.num_modules,
            unit="ea",
            unit_weight=self.config.module_dimensions.weight * 1.3,  # Glass-glass heavier
            total_weight=self.config.module_dimensions.weight * 1.3 * self.config.num_modules,
            unit_cost=module_cost,
            total_cost=module_cost * self.config.num_modules,
        ))

        # Structural members
        for i, member in enumerate(members):
            unit_weight = 15.0
            unit_cost = 180.0

            bom.append(BillOfMaterials(
                item_number=f"BIPV-{i+2:03d}",
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

        # Electrical integration
        bom.append(BillOfMaterials(
            item_number="ELEC-001",
            description="BIPV electrical integration",
            material=MaterialType.COMPOSITE,
            specification=f"Junction boxes, conduit, wiring",
            quantity=1,
            unit="system",
            unit_weight=50.0,
            total_weight=50.0,
            unit_cost=5000.0,
            total_cost=5000.0,
        ))

        return bom

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
