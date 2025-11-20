"""Foundation engineering and design for PV mounting structures."""

import math
import logging
from typing import Dict, Any, List, Optional

from .models import (
    SiteParameters,
    FoundationType,
    FoundationDesign,
    MaterialType,
    SoilType,
)


logger = logging.getLogger(__name__)


class FoundationEngineer:
    """
    Foundation engineering and design for PV mounting structures.

    Supports:
    - Pile foundations (driven, helical, ground screws)
    - Ballasted foundations
    - Shallow foundations (spread footings)
    - Geotechnical analysis
    """

    # Soil bearing capacity estimates (kN/m²) - conservative values
    SOIL_BEARING_CAPACITY = {
        SoilType.CLAY: 75,
        SoilType.SAND: 100,
        SoilType.SILT: 50,
        SoilType.GRAVEL: 200,
        SoilType.ROCK: 1000,
        SoilType.MIXED: 80,
    }

    # Pile friction coefficients (kN/m² per meter depth)
    PILE_FRICTION = {
        SoilType.CLAY: 15,
        SoilType.SAND: 10,
        SoilType.SILT: 8,
        SoilType.GRAVEL: 20,
        SoilType.ROCK: 50,
        SoilType.MIXED: 12,
    }

    def __init__(self, site_parameters: SiteParameters):
        """
        Initialize foundation engineer with site parameters.

        Args:
            site_parameters: Site-specific design parameters
        """
        self.site = site_parameters
        self.g = 9.81  # Gravity (m/s²)

    def pile_foundation_design(
        self,
        uplift_force: float,
        compression_force: float,
        lateral_force: float,
        pile_diameter: float = 0.15,  # m
        min_depth: float = 1.5,  # m
    ) -> FoundationDesign:
        """
        Design driven pile foundation.

        Args:
            uplift_force: Design uplift force (kN)
            compression_force: Design compression force (kN)
            lateral_force: Design lateral force (kN)
            pile_diameter: Pile diameter (m)
            min_depth: Minimum pile depth (m)

        Returns:
            FoundationDesign with pile specifications
        """
        logger.info(f"Designing driven pile: uplift={uplift_force}kN, compression={compression_force}kN")

        # Safety factor
        sf = 2.5

        # Required capacities
        uplift_required = uplift_force * sf
        compression_required = compression_force * sf

        # Pile perimeter
        perimeter = math.pi * pile_diameter

        # Soil friction per unit length
        unit_friction = self.PILE_FRICTION.get(self.site.soil_type, 10.0)

        # Calculate required depth for uplift (friction only)
        depth_uplift = uplift_required / (perimeter * unit_friction)

        # Calculate required depth for compression (friction + end bearing)
        # End bearing capacity
        end_area = math.pi * (pile_diameter / 2)**2
        bearing_capacity = self.SOIL_BEARING_CAPACITY.get(self.site.soil_type, 75.0)
        end_bearing = end_area * bearing_capacity

        # Compression = friction + end bearing
        # friction = perimeter * unit_friction * depth
        # compression_required = perimeter * unit_friction * depth + end_bearing
        depth_compression = (compression_required - end_bearing) / (perimeter * unit_friction)

        # Required depth is max of uplift, compression, frost depth, and minimum
        depth_required = max(depth_uplift, depth_compression, self.site.frost_depth + 0.3, min_depth)

        # Round up to nearest 0.5m
        depth = math.ceil(depth_required * 2) / 2

        # Calculate actual capacities
        friction_capacity = perimeter * unit_friction * depth
        compression_capacity = friction_capacity + end_bearing
        uplift_capacity = friction_capacity

        # Lateral capacity (simplified - Broms method)
        # For short rigid piles in cohesive soil
        lateral_capacity = 2.0 * pile_diameter * depth * 50  # Simplified

        # Check if capacity is adequate
        if uplift_capacity < uplift_required:
            logger.warning(f"Pile uplift capacity ({uplift_capacity:.1f}kN) < required ({uplift_required:.1f}kN)")

        return FoundationDesign(
            foundation_type=FoundationType.DRIVEN_PILE,
            depth=depth,
            diameter=pile_diameter,
            length=depth,
            width=pile_diameter,
            capacity=min(uplift_capacity, compression_capacity) / sf,
            spacing=3.0,  # Typical spacing, will be refined
            quantity=1,  # Per pile
            material=MaterialType.STEEL_GALVANIZED,
            embedment_depth=depth,
            concrete_volume=0.0,
            reinforcement="Steel pipe pile",
        )

    def helical_pile_design(
        self,
        uplift_force: float,
        compression_force: float,
        helix_diameter: float = 0.30,  # m
        shaft_diameter: float = 0.089,  # m (3.5")
        num_helixes: int = 3,
    ) -> FoundationDesign:
        """
        Design helical pile/anchor foundation.

        Args:
            uplift_force: Design uplift force (kN)
            compression_force: Design compression force (kN)
            helix_diameter: Helix plate diameter (m)
            shaft_diameter: Central shaft diameter (m)
            num_helixes: Number of helix plates

        Returns:
            FoundationDesign with helical pile specifications
        """
        logger.info(f"Designing helical pile: uplift={uplift_force}kN, compression={compression_force}kN")

        # Safety factor
        sf = 2.0

        # Required capacities
        uplift_required = uplift_force * sf
        compression_required = compression_force * sf

        # Helix area
        helix_area = math.pi * (helix_diameter / 2)**2

        # Individual helix capacity (uplift) - based on soil bearing
        bearing_capacity = self.SOIL_BEARING_CAPACITY.get(self.site.soil_type, 75.0)

        # Uplift capacity per helix (empirical factor 0.7)
        capacity_per_helix = helix_area * bearing_capacity * 0.7

        # Total uplift capacity
        uplift_capacity = capacity_per_helix * num_helixes

        # Compression capacity (higher than uplift)
        compression_capacity = capacity_per_helix * num_helixes * 1.5

        # Helix spacing (typically 3x diameter)
        helix_spacing = 3 * helix_diameter

        # Required depth (helixes must be below frost line)
        depth_required = self.site.frost_depth + helix_spacing * (num_helixes - 1) + 0.5

        # Round up
        depth = math.ceil(depth_required * 2) / 2

        # Installation torque (simplified correlation)
        # Kt = empirical torque correlation factor (typically 3-10 m⁻¹)
        Kt = 6.0  # m⁻¹
        installation_torque = uplift_capacity / Kt  # kN-m

        logger.info(f"Helical pile depth: {depth}m, torque: {installation_torque:.1f} kN-m")

        return FoundationDesign(
            foundation_type=FoundationType.HELICAL_PILE,
            depth=depth,
            diameter=helix_diameter,
            length=depth,
            width=helix_diameter,
            capacity=min(uplift_capacity, compression_capacity) / sf,
            spacing=3.0,
            quantity=1,
            material=MaterialType.STEEL_GALVANIZED,
            embedment_depth=depth,
            concrete_volume=0.0,
            reinforcement=f"{num_helixes} helix plates, torque={installation_torque:.0f} kN-m",
        )

    def ballast_design(
        self,
        uplift_force: float,
        wind_moment: float,
        ballast_density: float = 23.5,  # kN/m³ (concrete)
        friction_coefficient: float = 0.4,
    ) -> FoundationDesign:
        """
        Design ballasted foundation (typically for rooftop or flat ground).

        Args:
            uplift_force: Design uplift force per ballast (kN)
            wind_moment: Overturning moment (kN-m)
            ballast_density: Ballast material density (kN/m³)
            friction_coefficient: Friction coefficient with roof/ground

        Returns:
            FoundationDesign with ballast specifications
        """
        logger.info(f"Designing ballast: uplift={uplift_force}kN, moment={wind_moment}kN-m")

        # Safety factor for overturning
        sf_overturning = 1.5
        sf_uplift = 1.2

        # Required ballast weight to resist uplift
        weight_uplift = uplift_force * sf_uplift

        # Required ballast weight to resist overturning
        # Assume ballast spacing and calculate required weight
        lever_arm = 2.0  # m (typical racking width)
        weight_overturning = (wind_moment * sf_overturning) / lever_arm

        # Governing weight
        required_weight = max(weight_uplift, weight_overturning)

        # Ballast volume
        volume = required_weight / ballast_density

        # Typical ballast block dimensions (0.6m x 0.6m x h)
        block_footprint = 0.6 * 0.6  # m²
        block_height = volume / block_footprint

        # Round up to practical height
        block_height = max(0.15, math.ceil(block_height * 20) / 20)  # 0.05m increments

        # Actual volume and weight
        volume_actual = block_footprint * block_height
        weight_actual = volume_actual * ballast_density

        # Sliding resistance
        sliding_resistance = weight_actual * friction_coefficient

        return FoundationDesign(
            foundation_type=FoundationType.BALLASTED,
            depth=0.0,
            diameter=None,
            length=0.6,
            width=0.6,
            capacity=weight_actual / sf_uplift,
            spacing=2.0,
            quantity=1,
            material=MaterialType.CONCRETE,
            embedment_depth=0.0,
            concrete_volume=volume_actual,
            reinforcement=f"Ballast block: {block_height:.2f}m high, {weight_actual:.0f}kN, sliding resistance: {sliding_resistance:.0f}kN",
        )

    def shallow_foundation(
        self,
        vertical_load: float,
        lateral_load: float,
        moment: float,
        footing_width: float = 1.0,  # m
        footing_length: float = 1.0,  # m
    ) -> FoundationDesign:
        """
        Design spread footing (shallow foundation).

        Args:
            vertical_load: Vertical load (kN)
            lateral_load: Lateral load (kN)
            moment: Overturning moment (kN-m)
            footing_width: Footing width (m)
            footing_length: Footing length (m)

        Returns:
            FoundationDesign with spread footing specifications
        """
        logger.info(f"Designing spread footing: load={vertical_load}kN, moment={moment}kN-m")

        # Safety factors
        sf_bearing = 3.0
        sf_overturning = 1.5

        # Bearing capacity
        bearing_capacity = self.site.bearing_capacity or self.SOIL_BEARING_CAPACITY.get(
            self.site.soil_type, 75.0
        )

        # Footing area
        area = footing_width * footing_length

        # Eccentricity due to moment
        e = moment / vertical_load if vertical_load > 0 else 0

        # Effective width (accounting for eccentricity)
        B_eff = footing_width - 2 * e

        if B_eff <= 0:
            logger.error("Footing eccentricity too large - unstable")
            B_eff = footing_width * 0.5

        # Effective area
        area_eff = B_eff * footing_length

        # Bearing pressure
        bearing_pressure = vertical_load / area_eff

        # Check against allowable
        allowable_bearing = bearing_capacity / sf_bearing

        if bearing_pressure > allowable_bearing:
            # Resize footing
            area_required = vertical_load * sf_bearing / bearing_capacity
            footing_width = math.sqrt(area_required)
            footing_length = footing_width
            area_eff = area_required
            bearing_pressure = vertical_load / area_eff

        # Footing thickness (simplified - typically 0.3-0.6m)
        footing_thickness = max(0.3, footing_width * 0.3)

        # Concrete volume
        concrete_volume = footing_width * footing_length * footing_thickness

        # Depth below grade (below frost line)
        depth = max(self.site.frost_depth + 0.3, 0.6)

        # Reinforcement (simplified)
        reinforcement = "#5 bars @ 200mm each way"

        return FoundationDesign(
            foundation_type=FoundationType.SPREAD_FOOTING,
            depth=depth,
            diameter=None,
            length=footing_length,
            width=footing_width,
            capacity=allowable_bearing * area_eff,
            spacing=3.0,
            quantity=1,
            material=MaterialType.CONCRETE,
            embedment_depth=depth,
            concrete_volume=concrete_volume,
            reinforcement=reinforcement,
        )

    def geotechnical_requirements(
        self,
        foundation_type: FoundationType,
    ) -> Dict[str, Any]:
        """
        Specify geotechnical investigation requirements.

        Args:
            foundation_type: Type of foundation being designed

        Returns:
            Dictionary with geotechnical requirements
        """
        requirements = {
            "foundation_type": foundation_type.value,
            "site_investigation_required": True,
            "recommended_tests": [],
            "minimum_borings": 0,
            "boring_depth": 0.0,
        }

        if foundation_type in [FoundationType.DRIVEN_PILE, FoundationType.HELICAL_PILE, FoundationType.GROUND_SCREW]:
            requirements["recommended_tests"] = [
                "Standard Penetration Test (SPT)",
                "Cone Penetration Test (CPT)",
                "Soil classification",
                "Moisture content",
                "Shear strength",
            ]
            requirements["minimum_borings"] = 3
            requirements["boring_depth"] = 1.5 * 6.0  # 1.5x expected pile depth

        elif foundation_type == FoundationType.SPREAD_FOOTING:
            requirements["recommended_tests"] = [
                "Bearing capacity test",
                "Soil classification",
                "Moisture content",
                "Compaction test",
            ]
            requirements["minimum_borings"] = 2
            requirements["boring_depth"] = 3.0

        elif foundation_type == FoundationType.BALLASTED:
            requirements["site_investigation_required"] = False
            requirements["recommended_tests"] = [
                "Roof structural capacity verification",
                "Surface friction test",
            ]

        return requirements

    def frost_depth_consideration(
        self,
        foundation_type: FoundationType,
    ) -> Dict[str, float]:
        """
        Calculate frost heave protection requirements.

        Args:
            foundation_type: Type of foundation

        Returns:
            Dictionary with frost protection requirements
        """
        frost_depth = self.site.frost_depth

        if foundation_type == FoundationType.BALLASTED:
            # No frost concerns for ballasted
            return {
                "frost_depth": frost_depth,
                "minimum_embedment": 0.0,
                "frost_protection_required": False,
            }

        # Minimum embedment below frost line
        min_embedment = frost_depth + 0.3  # 300mm below frost line

        # For cold climates, may need additional measures
        frost_susceptible = self.site.soil_type in [SoilType.SILT, SoilType.CLAY]

        return {
            "frost_depth": frost_depth,
            "minimum_embedment": min_embedment,
            "frost_protection_required": frost_depth > 0.5,
            "frost_susceptible_soil": frost_susceptible,
            "recommendations": [
                "Embed foundation below frost line",
                "Use free-draining backfill if soil is frost-susceptible",
                "Consider insulated foundation in extreme climates",
            ] if frost_depth > 0.5 else [],
        }

    def calculate_foundation_quantity(
        self,
        total_array_width: float,
        total_array_length: float,
        foundation_spacing: float,
        perimeter_factor: float = 1.2,
    ) -> int:
        """
        Calculate required number of foundations for array.

        Args:
            total_array_width: Total array width (m)
            total_array_length: Total array length (m)
            foundation_spacing: Foundation spacing (m)
            perimeter_factor: Additional foundations for perimeter (factor)

        Returns:
            Number of foundations required
        """
        # Number of foundations along width
        num_width = math.ceil(total_array_width / foundation_spacing) + 1

        # Number of foundations along length
        num_length = math.ceil(total_array_length / foundation_spacing) + 1

        # Total foundations
        num_foundations = num_width * num_length

        # Add perimeter reinforcement
        num_foundations = int(num_foundations * perimeter_factor)

        logger.info(f"Foundation quantity: {num_foundations} for {total_array_width}m x {total_array_length}m array")

        return num_foundations
