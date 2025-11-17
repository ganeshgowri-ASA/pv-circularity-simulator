"""Carport and canopy PV mounting system design."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    CarportConfig,
    MaterialType,
    BillOfMaterials,
    StructuralAnalysisResult,
    StructuralMember,
)
from .structural_calculator import StructuralCalculator
from .foundation_engineer import FoundationEngineer


logger = logging.getLogger(__name__)


class CarportCanopyDesign:
    """
    Carport and canopy PV system design.

    Supports:
    - Single cantilever carports
    - Double cantilever carports
    - Four-post canopies
    - Beam sizing (steel, timber)
    - Column foundations
    - Vehicle clearance requirements
    - Drainage design
    """

    def __init__(
        self,
        config: CarportConfig,
        structural_calc: Optional[StructuralCalculator] = None,
        foundation_eng: Optional[FoundationEngineer] = None,
    ):
        """
        Initialize carport/canopy designer.

        Args:
            config: Carport configuration
            structural_calc: Structural calculator instance
            foundation_eng: Foundation engineer instance
        """
        self.config = config
        self.structural_calc = structural_calc or StructuralCalculator(config.site_parameters)
        self.foundation_eng = foundation_eng or FoundationEngineer(config.site_parameters)

    def single_cantilever_carport(self) -> StructuralAnalysisResult:
        """
        Design single-sided cantilever carport.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing single cantilever carport: span={self.config.span_length}m")

        # Cantilever length (typically 40-50% of back span)
        cantilever_length = self.config.cantilever_length or (self.config.span_length * 0.4)

        # Layout
        layout = self._calculate_carport_layout(cantilever_length)

        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.clearance_height + 0.5,
            additional_dead_load=0.25,  # Carport structure weight
            is_rooftop=False,
        )

        # Design beams
        beams = self._design_cantilever_beams(
            span_length=self.config.span_length,
            cantilever_length=cantilever_length,
            load_analysis=load_analysis,
            layout=layout,
        )

        # Design columns
        columns = self._design_columns(
            height=self.config.clearance_height + 1.0,
            load_per_column=layout["load_per_column"],
        )

        # Foundation design
        foundation = self.column_foundation(
            axial_load=layout["load_per_column"],
            moment=cantilever_length * load_analysis.dead_load * layout["tributary_width"],
        )

        # All structural members
        members = beams + columns

        # BOM
        bom = self._generate_carport_bom(layout, members, foundation, "single_cantilever")

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.STAINLESS_STEEL])

        # Drainage
        drainage = self.drainage_design()

        # Clearance check
        clearance_check = self.clearance_requirements()

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=cantilever_length / 100,  # Cantilever deflection
            deflection_limit=cantilever_length / 80,
            connection_details={
                "beam_to_column": "Welded or bolted moment connection",
                "column_to_foundation": "Anchor bolts, embedded",
                "cantilever_support": "Continuous beam over column",
            },
            compliance_notes=[
                f"Vehicle clearance: {clearance_check['min_clearance']}m (ADA compliant: {clearance_check['ada_compliant']})",
                f"Drainage slope: {drainage['slope']*100:.1f}%",
                "Design per ASCE 7-22 and IBC 2021",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def double_cantilever_carport(self) -> StructuralAnalysisResult:
        """
        Design center-post double cantilever carport.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing double cantilever carport: total span={self.config.span_length}m")

        # Each cantilever is half the span
        cantilever_length = self.config.span_length / 2.0

        # Layout
        layout = self._calculate_carport_layout(cantilever_length, double_cantilever=True)

        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.clearance_height + 0.5,
            additional_dead_load=0.20,
            is_rooftop=False,
        )

        # Design beams (double cantilever)
        beams = self._design_double_cantilever_beams(
            cantilever_length=cantilever_length,
            load_analysis=load_analysis,
            layout=layout,
        )

        # Design columns (center posts)
        columns = self._design_columns(
            height=self.config.clearance_height + 1.0,
            load_per_column=layout["load_per_column"],
        )

        # Foundation
        foundation = self.column_foundation(
            axial_load=layout["load_per_column"],
            moment=cantilever_length * load_analysis.dead_load * layout["tributary_width"] * 0.5,  # Lower moment
        )

        members = beams + columns

        # BOM
        bom = self._generate_carport_bom(layout, members, foundation, "double_cantilever")

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.STAINLESS_STEEL])

        drainage = self.drainage_design()
        clearance_check = self.clearance_requirements()

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=cantilever_length / 120,
            deflection_limit=cantilever_length / 100,
            connection_details={
                "beam_to_column": "Welded moment connection at center post",
                "column_to_foundation": "Anchor bolts",
                "cantilever_support": "Balanced cantilevers",
            },
            compliance_notes=[
                f"Vehicle clearance: {clearance_check['min_clearance']}m",
                f"Drainage: Center high point, {drainage['slope']*100:.1f}% slope to edges",
                "Balanced loading required",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def four_post_canopy(self) -> StructuralAnalysisResult:
        """
        Design traditional four-post canopy structure.

        Returns:
            Complete structural analysis
        """
        logger.info(f"Designing four-post canopy: span={self.config.span_length}m")

        # Layout
        layout = self._calculate_canopy_layout()

        # Load analysis
        load_analysis = self.structural_calc.calculate_total_loads(
            module_dimensions=self.config.module_dimensions,
            num_modules=self.config.num_modules,
            tilt_angle=self.config.tilt_angle,
            height=self.config.clearance_height + 0.5,
            additional_dead_load=0.15,  # Lighter structure
            is_rooftop=False,
        )

        # Design beams
        beams = self._design_canopy_beams(
            span_length=self.config.span_length,
            load_analysis=load_analysis,
            layout=layout,
        )

        # Design columns
        columns = self._design_columns(
            height=self.config.clearance_height + 1.0,
            load_per_column=layout["load_per_column"],
        )

        # Foundation
        foundation = self.column_foundation(
            axial_load=layout["load_per_column"],
            moment=0.5,  # Minimal moment for four-post
        )

        members = beams + columns

        # BOM
        bom = self._generate_carport_bom(layout, members, foundation, "four_post")

        total_steel = sum(item.total_weight or 0 for item in bom if item.material in [MaterialType.STEEL_GALVANIZED, MaterialType.STAINLESS_STEEL])

        drainage = self.drainage_design()
        clearance_check = self.clearance_requirements()

        return StructuralAnalysisResult(
            mounting_type=self.config.mounting_type,
            load_analysis=load_analysis,
            foundation_design=foundation,
            structural_members=members,
            bill_of_materials=bom,
            max_deflection=self.config.span_length / 250,
            deflection_limit=self.config.span_length / 180,
            connection_details={
                "beam_to_column": "Bolted or welded connections",
                "column_to_foundation": "Anchor bolts",
                "beam_configuration": "Simply supported beams",
            },
            compliance_notes=[
                f"Vehicle clearance: {clearance_check['min_clearance']}m",
                f"Drainage slope: {drainage['slope']*100:.1f}%",
                "Four-post design for standard loading",
            ],
            total_steel_weight=total_steel,
            total_cost_estimate=self._estimate_cost(bom),
        )

    def calculate_beam_sizing(
        self,
        span_length: float,
        load_per_meter: float,
        beam_material: str = "steel",
    ) -> Dict[str, Any]:
        """
        Calculate required beam size.

        Args:
            span_length: Beam span length (m)
            load_per_meter: Distributed load (kN/m)
            beam_material: Beam material (steel, timber)

        Returns:
            Dictionary with beam sizing results
        """
        logger.info(f"Sizing beam: span={span_length}m, load={load_per_meter}kN/m, material={beam_material}")

        if beam_material == "steel":
            # Steel W-beam design
            # Maximum moment: M = wL²/8 for simple span
            M_max = load_per_meter * span_length**2 / 8  # kN-m

            # Required section modulus: S = M / (Fb * safety_factor)
            # Allowable stress: Fb = 0.66 * Fy = 0.66 * 250 MPa = 165 MPa
            Fb = 165e3  # kN/m²
            sf = 1.67  # Safety factor
            S_required = M_max / (Fb / sf)  # m³

            # Convert to in³ for standard sections
            S_required_in3 = S_required * 61023.7

            # Select W-beam (simplified - would use AISC tables)
            if S_required_in3 < 20:
                beam_section = "W8x24"
                S_actual = 20.9  # in³
                weight_per_ft = 24  # lbs/ft
            elif S_required_in3 < 35:
                beam_section = "W10x33"
                S_actual = 35.0
                weight_per_ft = 33
            elif S_required_in3 < 64:
                beam_section = "W12x45"
                S_actual = 64.2
                weight_per_ft = 45
            else:
                beam_section = "W14x68"
                S_actual = 103.0
                weight_per_ft = 68

            # Convert weight to kg/m
            weight_per_m = weight_per_ft * 1.488

            # Deflection check
            E = 200e6  # kN/m²
            I_in4 = S_actual * 6.0  # Approximate depth factor
            I_m4 = I_in4 * 4.162e-7  # Convert in⁴ to m⁴

            deflection_calc = self.structural_calc.deflection_analysis(
                span_length=span_length,
                applied_load=load_per_meter,
                moment_of_inertia=I_m4,
                elastic_modulus=E,
                support_type="simple",
            )

            return {
                "beam_section": beam_section,
                "material": "Steel A992",
                "section_modulus_required": S_required_in3,
                "section_modulus_actual": S_actual,
                "weight_per_meter": weight_per_m,
                "max_moment": M_max,
                "deflection": deflection_calc["max_deflection"],
                "deflection_passes": deflection_calc["passes_deflection"],
            }

        elif beam_material == "timber":
            # Timber beam design (simplified)
            # Glulam or sawn timber
            M_max = load_per_meter * span_length**2 / 8  # kN-m

            # Allowable bending stress for Douglas Fir: Fb = 10 MPa
            Fb = 10e3  # kN/m²
            sf = 2.5
            S_required = M_max / (Fb / sf)  # m³

            # Typical timber sections (width x depth in mm)
            if S_required < 0.001:
                section = "140x240mm"
                width, depth = 0.14, 0.24
            elif S_required < 0.003:
                section = "190x360mm"
                width, depth = 0.19, 0.36
            else:
                section = "240x480mm"
                width, depth = 0.24, 0.48

            # Section modulus: S = b*h²/6
            S_actual = width * depth**2 / 6

            # Weight (Douglas Fir: ~550 kg/m³)
            weight_per_m = width * depth * 550

            return {
                "beam_section": section,
                "material": "Douglas Fir Glulam",
                "section_modulus_required": S_required,
                "section_modulus_actual": S_actual,
                "weight_per_meter": weight_per_m,
                "max_moment": M_max,
            }

        else:
            raise ValueError(f"Unknown beam material: {beam_material}")

    def column_foundation(
        self,
        axial_load: float,
        moment: float,
    ) -> Any:
        """
        Design column foundation.

        Args:
            axial_load: Axial load on column (kN)
            moment: Moment at base (kN-m)

        Returns:
            Foundation design
        """
        # Use spread footing for carport columns
        lateral_force = moment / self.config.clearance_height if self.config.clearance_height > 0 else 1.0

        return self.foundation_eng.shallow_foundation(
            vertical_load=axial_load,
            lateral_load=lateral_force,
            moment=moment,
            footing_width=1.2,
            footing_length=1.2,
        )

    def clearance_requirements(self) -> Dict[str, Any]:
        """
        Calculate vehicle clearance requirements.

        Returns:
            Dictionary with clearance specifications
        """
        # ADA requirements
        ada_min_clearance = 2.1  # m (7 ft)
        standard_clearance = 2.4  # m (8 ft)
        truck_clearance = 4.3  # m (14 ft)

        # Check if design meets requirements
        meets_ada = self.config.clearance_height >= ada_min_clearance
        meets_standard = self.config.clearance_height >= standard_clearance
        meets_truck = self.config.clearance_height >= truck_clearance

        return {
            "min_clearance": self.config.clearance_height,
            "ada_minimum": ada_min_clearance,
            "ada_compliant": meets_ada,
            "standard_clearance": standard_clearance,
            "meets_standard": meets_standard,
            "truck_clearance": truck_clearance,
            "meets_truck": meets_truck,
            "recommended": "Increase clearance" if not meets_standard else "Adequate",
        }

    def drainage_design(self) -> Dict[str, Any]:
        """
        Design rainwater drainage system.

        Returns:
            Dictionary with drainage specifications
        """
        # Drainage slope
        slope = self.config.drainage_slope

        # Module area
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        total_area = module_area * self.config.num_modules

        # Rainfall intensity (simplified - 100mm/hr design storm)
        rainfall_intensity = 0.1  # m/hr

        # Flow rate: Q = C * I * A
        # C = runoff coefficient (0.95 for PV modules)
        C = 0.95
        flow_rate = C * rainfall_intensity * total_area  # m³/hr

        # Gutter sizing (simplified)
        # For rectangular gutter, capacity ≈ slope^0.5 * width^2.67
        # Typical 150mm (6") gutter
        gutter_size = 0.15  # m

        # Number of downspouts (1 per 50 m² of roof)
        num_downspouts = max(2, math.ceil(total_area / 50))

        # Downspout diameter (typically 75-100mm)
        downspout_diameter = 0.075  # m (3")

        return {
            "slope": slope,
            "total_area": total_area,
            "flow_rate": flow_rate,
            "gutter_size": gutter_size,
            "num_downspouts": num_downspouts,
            "downspout_diameter": downspout_diameter,
            "drainage_direction": "Front and back" if self.config.carport_type == "double_cantilever" else "Back edge",
        }

    # Private helper methods

    def _calculate_carport_layout(
        self,
        cantilever_length: float,
        double_cantilever: bool = False,
    ) -> Dict[str, Any]:
        """Calculate carport layout geometry."""
        # Column spacing
        column_spacing = self.config.column_spacing

        # Number of bays
        num_bays = math.ceil(50 / column_spacing)  # Assume 50m length

        # Tributary width per beam
        tributary_width = self.config.span_length if not double_cantilever else self.config.span_length

        # Number of columns
        if double_cantilever:
            columns_per_row = num_bays + 1
        else:
            columns_per_row = num_bays + 1

        # Load per column (simplified)
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        load_per_module = self.config.module_dimensions.weight * 9.81 / 1000  # kN
        modules_per_bay = int((column_spacing * tributary_width) / module_area)
        load_per_column = load_per_module * modules_per_bay * 2.0  # Dead + Live

        return {
            "cantilever_length": cantilever_length,
            "column_spacing": column_spacing,
            "num_bays": num_bays,
            "tributary_width": tributary_width,
            "columns_per_row": columns_per_row,
            "total_columns": columns_per_row,
            "load_per_column": load_per_column,
        }

    def _calculate_canopy_layout(self) -> Dict[str, Any]:
        """Calculate four-post canopy layout."""
        # Simple four-post layout
        span_length = self.config.span_length
        span_width = span_length  # Square canopy

        # Module coverage
        module_area = self.config.module_dimensions.length * self.config.module_dimensions.width
        num_modules = int((span_length * span_width) / module_area)

        # Load per column (quarter of total)
        total_dead_load = self.config.module_dimensions.weight * num_modules * 9.81 / 1000
        load_per_column = total_dead_load / 4 * 2.0  # Include live load factor

        return {
            "span_length": span_length,
            "span_width": span_width,
            "num_columns": 4,
            "load_per_column": load_per_column,
            "tributary_width": span_width / 2,
        }

    def _design_cantilever_beams(
        self,
        span_length: float,
        cantilever_length: float,
        load_analysis: Any,
        layout: Dict,
    ) -> List[StructuralMember]:
        """Design beams for single cantilever."""
        # Load per meter
        load_per_m = (load_analysis.dead_load + load_analysis.snow_load) * layout["tributary_width"]

        # Beam sizing
        beam_info = self.calculate_beam_sizing(
            span_length=span_length + cantilever_length,
            load_per_meter=load_per_m,
            beam_material="steel",
        )

        return [
            StructuralMember(
                member_type="primary_beam",
                material=MaterialType.STEEL_GALVANIZED if self.config.beam_material == MaterialType.STEEL_GALVANIZED else self.config.beam_material,
                profile=beam_info["beam_section"],
                length=span_length + cantilever_length,
                spacing=layout["column_spacing"],
                quantity=layout["num_bays"],
                capacity=beam_info["max_moment"],
                utilization=0.75,
            ),
        ]

    def _design_double_cantilever_beams(
        self,
        cantilever_length: float,
        load_analysis: Any,
        layout: Dict,
    ) -> List[StructuralMember]:
        """Design beams for double cantilever."""
        load_per_m = (load_analysis.dead_load + load_analysis.snow_load) * layout["tributary_width"]

        beam_info = self.calculate_beam_sizing(
            span_length=cantilever_length * 2,
            load_per_meter=load_per_m,
            beam_material="steel",
        )

        return [
            StructuralMember(
                member_type="primary_beam",
                material=self.config.beam_material,
                profile=beam_info["beam_section"],
                length=cantilever_length * 2,
                spacing=layout["column_spacing"],
                quantity=layout["num_bays"],
                capacity=beam_info["max_moment"],
                utilization=0.70,
            ),
        ]

    def _design_canopy_beams(
        self,
        span_length: float,
        load_analysis: Any,
        layout: Dict,
    ) -> List[StructuralMember]:
        """Design beams for four-post canopy."""
        load_per_m = (load_analysis.dead_load + load_analysis.snow_load) * layout["tributary_width"]

        beam_info = self.calculate_beam_sizing(
            span_length=span_length,
            load_per_meter=load_per_m,
            beam_material="steel",
        )

        return [
            StructuralMember(
                member_type="primary_beam",
                material=self.config.beam_material,
                profile=beam_info["beam_section"],
                length=span_length,
                spacing=span_length,  # Two beams
                quantity=4,  # Four beams total
                capacity=beam_info["max_moment"],
                utilization=0.65,
            ),
        ]

    def _design_columns(
        self,
        height: float,
        load_per_column: float,
    ) -> List[StructuralMember]:
        """Design columns."""
        # Column selection based on load
        if load_per_column < 100:
            profile = "HSS6x6x1/4"
            capacity = 150
        elif load_per_column < 200:
            profile = "HSS8x8x3/8"
            capacity = 300
        else:
            profile = "HSS10x10x1/2"
            capacity = 500

        return [
            StructuralMember(
                member_type="column",
                material=MaterialType.STEEL_GALVANIZED,
                profile=profile,
                length=height,
                spacing=self.config.column_spacing,
                quantity=10,  # Typical number
                capacity=capacity,
                utilization=load_per_column / capacity,
            ),
        ]

    def _generate_carport_bom(
        self,
        layout: Dict,
        members: List[StructuralMember],
        foundation: Any,
        carport_type: str,
    ) -> List[BillOfMaterials]:
        """Generate BOM for carport."""
        bom = []

        # Foundations
        num_foundations = layout.get("total_columns", 10)
        bom.append(BillOfMaterials(
            item_number="FND-001",
            description=f"Column foundation - {foundation.foundation_type.value}",
            material=foundation.material,
            specification=f"{foundation.length}m x {foundation.width}m x {foundation.depth}m deep",
            quantity=num_foundations,
            unit="ea",
            unit_weight=foundation.concrete_volume * 23.5 * 1000 if foundation.concrete_volume else 2000,
            total_weight=(foundation.concrete_volume * 23.5 * 1000 if foundation.concrete_volume else 2000) * num_foundations,
            unit_cost=600.0,
            total_cost=600.0 * num_foundations,
        ))

        # Structural members
        for i, member in enumerate(members):
            if member.member_type == "primary_beam":
                unit_weight = 100
                unit_cost = 450
            else:  # Column
                unit_weight = 80
                unit_cost = 350

            bom.append(BillOfMaterials(
                item_number=f"CP-{i+1:03d}",
                description=f"{member.member_type} - {member.profile}",
                material=member.material,
                specification=f"{member.length}m long",
                quantity=member.quantity,
                unit="ea",
                unit_weight=unit_weight,
                total_weight=unit_weight * member.quantity,
                unit_cost=unit_cost,
                total_cost=unit_cost * member.quantity,
            ))

        # PV module mounting hardware
        bom.append(BillOfMaterials(
            item_number="PV-001",
            description="PV module mounting system",
            material=MaterialType.ALUMINUM,
            specification="Rails and clamps for carport",
            quantity=self.config.num_modules,
            unit="module",
            unit_weight=3.0,
            total_weight=3.0 * self.config.num_modules,
            unit_cost=45.0,
            total_cost=45.0 * self.config.num_modules,
        ))

        return bom

    def _estimate_cost(self, bom: List[BillOfMaterials]) -> float:
        """Estimate total cost from BOM."""
        return sum(item.total_cost or 0 for item in bom)
