"""Structural calculator with ASCE 7 wind, snow, and seismic load analysis."""

import math
import logging
from typing import Dict, Any, Optional, Tuple

from .models import (
    SiteParameters,
    ModuleDimensions,
    ExposureCategory,
    SeismicDesignCategory,
    LoadAnalysis,
)


logger = logging.getLogger(__name__)


class StructuralCalculator:
    """
    Structural engineering calculator for PV mounting systems.

    Implements ASCE 7 standards for:
    - Wind load analysis (Chapter 27, 28, 29)
    - Snow load analysis (Chapter 7)
    - Seismic analysis (Chapter 11, 12)
    - Load combinations (Chapter 2)
    - Deflection and stress analysis
    """

    def __init__(self, site_parameters: SiteParameters):
        """
        Initialize structural calculator with site parameters.

        Args:
            site_parameters: Site-specific design parameters
        """
        self.site = site_parameters
        self.g = 9.81  # Gravity (m/s²)

    def wind_load_analysis(
        self,
        tilt_angle: float,
        height: float,
        module_dimensions: ModuleDimensions,
        is_rooftop: bool = False,
        roof_height: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate wind loads per ASCE 7-22.

        Args:
            tilt_angle: Module tilt angle in degrees
            height: Height above ground (m)
            module_dimensions: Module physical dimensions
            is_rooftop: Whether installation is on rooftop
            roof_height: Building height if rooftop installation

        Returns:
            Dictionary with wind pressure components (kN/m²)
        """
        logger.info(f"Calculating wind loads for tilt={tilt_angle}°, height={height}m")

        # Basic wind speed (already in m/s)
        V = self.site.wind_speed

        # Velocity pressure exposure coefficient (Kz) - ASCE 7 Table 26.10-1
        Kz = self._calculate_kz(height)

        # Topographic factor (Kzt) - simplified, assume 1.0
        Kzt = 1.0

        # Directionality factor (Kd) - ASCE 7 Table 26.6-1
        Kd = 0.85  # For building structures

        # Wind importance factor (based on Risk Category II)
        Iw = 1.0

        # Velocity pressure: qz = 0.613 * Kz * Kzt * Kd * V² (N/m²)
        qz = 0.613 * Kz * Kzt * Kd * V**2  # N/m²
        qz_kn = qz / 1000.0  # kN/m²

        # Gust effect factor (G) - simplified
        G = 0.85

        # Pressure coefficients for tilted solar panels - ASCE 7 Fig 29.4-5
        # These are simplified values based on tilt angle
        theta_rad = math.radians(tilt_angle)

        if tilt_angle <= 7:
            # Nearly flat, use flat plate coefficients
            Cp_upper = -0.9
            Cp_lower = -0.3
        elif tilt_angle <= 45:
            # Moderate tilt
            Cp_upper = -1.3 + 0.3 * math.sin(theta_rad)
            Cp_lower = -0.5
        else:
            # Steep tilt
            Cp_upper = -1.1
            Cp_lower = -0.7

        # Net pressure coefficients
        Cn_uplift = abs(Cp_upper - Cp_lower)  # Uplift
        Cn_downward = 0.8  # Simplified downward coefficient

        # Calculate wind pressures
        p_uplift = qz_kn * G * Cn_uplift * Iw
        p_downward = qz_kn * G * Cn_downward * Iw

        # Roof pressure if applicable (ASCE 7-22 Chapter 27)
        if is_rooftop and roof_height:
            qh = 0.613 * self._calculate_kz(roof_height) * Kzt * Kd * V**2 / 1000.0
            # Additional roof uplift
            Cp_roof = -1.2  # Edge zone
            p_roof_uplift = qh * G * abs(Cp_roof) * Iw
            p_uplift += p_roof_uplift * 0.5  # Combined effect

        return {
            "velocity_pressure_qz": qz_kn,
            "uplift_pressure": p_uplift,
            "downward_pressure": p_downward,
            "gust_factor": G,
            "exposure_coefficient_kz": Kz,
            "wind_speed": V,
        }

    def _calculate_kz(self, height: float) -> float:
        """
        Calculate velocity pressure exposure coefficient (Kz) per ASCE 7 Table 26.10-1.

        Args:
            height: Height above ground (m)

        Returns:
            Exposure coefficient Kz
        """
        # Convert to feet for ASCE 7 formulas
        z_ft = height * 3.28084

        # Minimum height
        z_ft = max(z_ft, 15.0)

        # Alpha and zg based on exposure category
        if self.site.exposure_category == ExposureCategory.B:
            alpha = 7.0
            zg = 1200.0  # ft
        elif self.site.exposure_category == ExposureCategory.C:
            alpha = 9.5
            zg = 900.0  # ft
        else:  # Exposure D
            alpha = 11.5
            zg = 700.0  # ft

        # Case 1: z >= 15 ft
        Kz = 2.01 * (z_ft / zg) ** (2.0 / alpha)

        return Kz

    def snow_load_analysis(
        self,
        tilt_angle: float,
        module_length: float,
        is_sheltered: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate snow loads per ASCE 7 Chapter 7.

        Args:
            tilt_angle: Module tilt angle in degrees
            module_length: Module length along slope (m)
            is_sheltered: Whether installation is sheltered/warm roof

        Returns:
            Dictionary with snow load components (kN/m²)
        """
        logger.info(f"Calculating snow loads for tilt={tilt_angle}°")

        # Ground snow load (pg)
        pg = self.site.ground_snow_load

        if pg == 0:
            return {
                "ground_snow_load": 0.0,
                "flat_roof_snow_load": 0.0,
                "sloped_roof_snow_load": 0.0,
                "drift_load": 0.0,
            }

        # Exposure factor (Ce) - ASCE 7 Table 7.3-1
        if self.site.exposure_category == ExposureCategory.B:
            Ce = 1.2 if is_sheltered else 1.0
        elif self.site.exposure_category == ExposureCategory.C:
            Ce = 1.0 if is_sheltered else 0.9
        else:  # Exposure D
            Ce = 0.8

        # Thermal factor (Ct) - ASCE 7 Table 7.3-2
        Ct = 1.0 if is_sheltered else 1.2  # Unheated structure

        # Importance factor (Is) - Risk Category II
        Is = 1.0

        # Flat roof snow load (pf)
        pf = 0.7 * Ce * Ct * Is * pg

        # Slope factor (Cs) - ASCE 7 Fig 7.4-1
        theta = tilt_angle
        if theta < 30:
            Cs = 1.0
        elif theta <= 70:
            Cs = (70 - theta) / 40.0
        else:
            Cs = 0.0

        # Sloped roof snow load
        ps = Cs * pf

        # Drift load for arrays (simplified)
        # ASCE 7 Section 7.7-7.8
        if module_length > 5.0 and pg > 0.5:  # Significant drift potential
            hd = 0.43 * math.sqrt(module_length) * math.sqrt(pg) ** 0.33
            drift_load = 0.8 * pg * hd / module_length  # Distributed
        else:
            drift_load = 0.0

        return {
            "ground_snow_load": pg,
            "flat_roof_snow_load": pf,
            "sloped_roof_snow_load": ps,
            "drift_load": drift_load,
            "exposure_factor_ce": Ce,
            "thermal_factor_ct": Ct,
            "slope_factor_cs": Cs,
        }

    def seismic_analysis(
        self,
        total_weight: float,
        height: float,
        importance_factor: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate seismic loads per ASCE 7 Chapter 11-12.

        Args:
            total_weight: Total weight of PV system (kN)
            height: Center of mass height above base (m)
            importance_factor: Seismic importance factor

        Returns:
            Dictionary with seismic force components
        """
        logger.info(f"Calculating seismic loads for SDC={self.site.seismic_category}")

        # For SDC A and B, seismic design is often not required
        if self.site.seismic_category in [SeismicDesignCategory.A, SeismicDesignCategory.B]:
            return {
                "seismic_force": 0.0,
                "seismic_category": self.site.seismic_category.value,
                "requires_seismic_design": False,
            }

        # Simplified seismic design (ASCE 7-22 Section 12.14 - Nonbuilding Structures)
        # Seismic response coefficient (Cs)

        # Mapped spectral accelerations (simplified - would need site-specific values)
        # These are placeholder values - in practice, use USGS seismic maps
        if self.site.seismic_category == SeismicDesignCategory.C:
            SDS = 0.33  # 5% damped design spectral acceleration
            SD1 = 0.20
        elif self.site.seismic_category == SeismicDesignCategory.D:
            SDS = 0.50
            SD1 = 0.30
        elif self.site.seismic_category == SeismicDesignCategory.E:
            SDS = 0.75
            SD1 = 0.45
        else:  # F
            SDS = 1.00
            SD1 = 0.60

        # Response modification factor (R) for PV arrays
        R = 3.5  # ASCE 7 Table 15.4-2

        # Seismic response coefficient
        Cs = SDS / (R / importance_factor)

        # Minimum Cs
        Cs_min = 0.044 * SDS * importance_factor
        Cs = max(Cs, Cs_min)

        # Maximum Cs (if period calculated)
        # For simplicity, we'll use the calculated value

        # Seismic base shear
        V = Cs * total_weight

        return {
            "seismic_force": V,
            "seismic_coefficient_cs": Cs,
            "spectral_acceleration_sds": SDS,
            "spectral_acceleration_sd1": SD1,
            "response_modification_r": R,
            "seismic_category": self.site.seismic_category.value,
            "requires_seismic_design": True,
        }

    def deflection_analysis(
        self,
        span_length: float,
        applied_load: float,
        moment_of_inertia: float,
        elastic_modulus: float = 200e6,  # kN/m² for steel
        support_type: str = "simple",
    ) -> Dict[str, float]:
        """
        Calculate beam/member deflection under load.

        Args:
            span_length: Span length (m)
            applied_load: Applied distributed load (kN/m)
            moment_of_inertia: Moment of inertia (m⁴)
            elastic_modulus: Elastic modulus (kN/m²)
            support_type: Support type (simple, fixed, cantilever)

        Returns:
            Dictionary with deflection values
        """
        L = span_length
        w = applied_load
        E = elastic_modulus
        I = moment_of_inertia

        # Calculate maximum deflection based on support type
        if support_type == "simple":
            # Simply supported: δ = 5wL⁴/(384EI)
            delta_max = (5 * w * L**4) / (384 * E * I)
            deflection_limit_ratio = 180  # L/180
        elif support_type == "fixed":
            # Fixed-fixed: δ = wL⁴/(384EI)
            delta_max = (w * L**4) / (384 * E * I)
            deflection_limit_ratio = 240  # L/240
        elif support_type == "cantilever":
            # Cantilever: δ = wL⁴/(8EI)
            delta_max = (w * L**4) / (8 * E * I)
            deflection_limit_ratio = 120  # L/120
        else:
            raise ValueError(f"Unknown support type: {support_type}")

        # Deflection limits
        delta_limit = L / deflection_limit_ratio

        # Check ratio
        passes = delta_max <= delta_limit

        return {
            "max_deflection": delta_max,
            "deflection_limit": delta_limit,
            "deflection_ratio": L / delta_max if delta_max > 0 else float('inf'),
            "required_ratio": deflection_limit_ratio,
            "passes_deflection": passes,
        }

    def connection_design(
        self,
        applied_force: float,
        connection_type: str = "bolted",
        num_fasteners: int = 4,
        fastener_diameter: float = 0.016,  # m (5/8")
        material_strength: float = 400,  # MPa
    ) -> Dict[str, float]:
        """
        Design structural connections (bolts, welds).

        Args:
            applied_force: Applied force on connection (kN)
            connection_type: Connection type (bolted, welded)
            num_fasteners: Number of fasteners
            fastener_diameter: Fastener diameter (m)
            material_strength: Material ultimate strength (MPa)

        Returns:
            Dictionary with connection design results
        """
        if connection_type == "bolted":
            # Bolt shear capacity (simplified)
            # A307 bolts: Fv = 0.4 * Fu
            # Area per bolt
            Ab = math.pi * (fastener_diameter / 2)**2  # m²
            Ab_mm2 = Ab * 1e6  # mm²

            # Shear strength
            Fv = 0.4 * material_strength  # MPa

            # Capacity per bolt
            capacity_per_bolt = Fv * Ab_mm2 / 1000.0  # kN

            # Total capacity
            total_capacity = capacity_per_bolt * num_fasteners

            # Safety factor (AISC)
            safety_factor = 2.0
            allowable_capacity = total_capacity / safety_factor

            # Utilization
            utilization = applied_force / allowable_capacity

            return {
                "connection_type": connection_type,
                "num_fasteners": num_fasteners,
                "capacity_per_fastener": capacity_per_bolt,
                "total_capacity": total_capacity,
                "allowable_capacity": allowable_capacity,
                "applied_force": applied_force,
                "utilization": utilization,
                "passes": utilization <= 1.0,
            }

        elif connection_type == "welded":
            # Simplified weld design
            # Assume fillet weld, throat dimension
            throat = fastener_diameter  # Use as weld size

            # Weld length (simplified)
            weld_length = num_fasteners * 0.1  # m

            # Weld capacity (simplified)
            Fw = 0.6 * material_strength  # MPa

            # Throat area
            Aw = throat * weld_length * 1e6  # mm²

            # Capacity
            capacity = Fw * Aw / 1000.0  # kN

            safety_factor = 2.5
            allowable_capacity = capacity / safety_factor

            utilization = applied_force / allowable_capacity

            return {
                "connection_type": connection_type,
                "weld_size": throat,
                "weld_length": weld_length,
                "total_capacity": capacity,
                "allowable_capacity": allowable_capacity,
                "applied_force": applied_force,
                "utilization": utilization,
                "passes": utilization <= 1.0,
            }

        else:
            raise ValueError(f"Unknown connection type: {connection_type}")

    def safety_factors(
        self,
        dead_load: float,
        live_load: float,
        wind_load: float,
        snow_load: float,
        seismic_load: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate load combinations per ASCE 7 Chapter 2 and IBC.

        Args:
            dead_load: Dead load (kN or kN/m²)
            live_load: Live load (kN or kN/m²)
            wind_load: Wind load (kN or kN/m²)
            snow_load: Snow load (kN or kN/m²)
            seismic_load: Seismic load (kN)

        Returns:
            Dictionary with load combinations
        """
        # ASCE 7-22 Section 2.3 Load Combinations
        combinations = {}

        # Strength design (LRFD) combinations
        combinations["1.4D"] = 1.4 * dead_load
        combinations["1.2D+1.6L+0.5S"] = 1.2 * dead_load + 1.6 * live_load + 0.5 * snow_load
        combinations["1.2D+1.6S+L"] = 1.2 * dead_load + 1.6 * snow_load + live_load
        combinations["1.2D+1.0W+L+0.5S"] = 1.2 * dead_load + 1.0 * wind_load + live_load + 0.5 * snow_load
        combinations["0.9D+1.0W"] = 0.9 * dead_load + 1.0 * wind_load

        if seismic_load > 0:
            combinations["1.2D+1.0E+L+0.2S"] = 1.2 * dead_load + 1.0 * seismic_load + live_load + 0.2 * snow_load
            combinations["0.9D+1.0E"] = 0.9 * dead_load + 1.0 * seismic_load

        # Find governing combination
        governing_combo = max(combinations.items(), key=lambda x: abs(x[1]))

        return {
            "combinations": combinations,
            "governing_combination": governing_combo[0],
            "governing_load": governing_combo[1],
        }

    def calculate_total_loads(
        self,
        module_dimensions: ModuleDimensions,
        num_modules: int,
        tilt_angle: float,
        height: float,
        additional_dead_load: float = 0.0,
        is_rooftop: bool = False,
    ) -> LoadAnalysis:
        """
        Calculate complete load analysis for mounting structure.

        Args:
            module_dimensions: Module physical dimensions
            num_modules: Total number of modules
            tilt_angle: Module tilt angle (degrees)
            height: Height above ground/roof (m)
            additional_dead_load: Additional dead load from racking (kN/m²)
            is_rooftop: Whether installation is on rooftop

        Returns:
            LoadAnalysis model with all load components
        """
        # Module area
        module_area = module_dimensions.length * module_dimensions.width
        total_area = module_area * num_modules

        # Dead load (module weight + racking)
        module_weight_kn = module_dimensions.weight * self.g / 1000.0  # kN
        dead_load_total = module_weight_kn * num_modules
        dead_load_per_area = dead_load_total / total_area + additional_dead_load  # kN/m²

        # Live load (maintenance, typically 0.25-0.5 kN/m²)
        live_load = 0.5  # kN/m²

        # Wind loads
        wind_results = self.wind_load_analysis(
            tilt_angle=tilt_angle,
            height=height,
            module_dimensions=module_dimensions,
            is_rooftop=is_rooftop,
        )
        wind_uplift = wind_results["uplift_pressure"]
        wind_downward = wind_results["downward_pressure"]

        # Snow loads
        snow_results = self.snow_load_analysis(
            tilt_angle=tilt_angle,
            module_length=module_dimensions.length,
        )
        snow_load = snow_results["sloped_roof_snow_load"] + snow_results["drift_load"]

        # Seismic loads (if applicable)
        seismic_results = self.seismic_analysis(
            total_weight=dead_load_total,
            height=height,
        )
        seismic_load = seismic_results.get("seismic_force", 0.0)

        # Load combinations
        load_combos = self.safety_factors(
            dead_load=dead_load_per_area,
            live_load=live_load,
            wind_load=max(abs(wind_uplift), abs(wind_downward)),
            snow_load=snow_load,
            seismic_load=seismic_load / total_area if total_area > 0 else 0,
        )

        return LoadAnalysis(
            dead_load=dead_load_per_area,
            live_load=live_load,
            wind_load_uplift=wind_uplift,
            wind_load_downward=wind_downward,
            snow_load=snow_load,
            seismic_load=seismic_load,
            total_load_combination=load_combos["governing_load"],
            safety_factor=2.0,
        )
