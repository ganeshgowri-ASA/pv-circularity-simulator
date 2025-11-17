"""
Environmental Impact & LCA Analysis (B11-S05)

This module provides tools for life cycle assessment of PV modules,
including carbon footprint calculation, energy payback analysis, and
environmental impact indicators.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum


class LifeCycleStage(str, Enum):
    """Life cycle stages for LCA."""
    RAW_MATERIAL = "raw_material"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    INSTALLATION = "installation"
    OPERATION = "operation"
    MAINTENANCE = "maintenance"
    END_OF_LIFE = "end_of_life"


class ImpactCategory(str, Enum):
    """Environmental impact categories."""
    CLIMATE_CHANGE = "climate_change"
    ACIDIFICATION = "acidification"
    EUTROPHICATION = "eutrophication"
    OZONE_DEPLETION = "ozone_depletion"
    PHOTOCHEMICAL_OXIDATION = "photochemical_oxidation"
    HUMAN_TOXICITY = "human_toxicity"
    ECOTOXICITY = "ecotoxicity"
    RESOURCE_DEPLETION = "resource_depletion"


class CarbonFootprint(BaseModel):
    """Carbon footprint analysis results."""

    raw_materials_kg_co2eq: float = Field(ge=0, description="Raw materials CO2eq emissions")
    manufacturing_kg_co2eq: float = Field(ge=0, description="Manufacturing CO2eq emissions")
    transportation_kg_co2eq: float = Field(ge=0, description="Transportation CO2eq emissions")
    installation_kg_co2eq: float = Field(ge=0, description="Installation CO2eq emissions")
    operation_kg_co2eq: float = Field(ge=0, description="Operation CO2eq emissions")
    maintenance_kg_co2eq: float = Field(ge=0, description="Maintenance CO2eq emissions")
    end_of_life_kg_co2eq: float = Field(description="End-of-life CO2eq emissions (negative if recycling benefit)")

    @property
    def total_kg_co2eq(self) -> float:
        """Total carbon footprint in kg CO2eq."""
        return (
            self.raw_materials_kg_co2eq + self.manufacturing_kg_co2eq +
            self.transportation_kg_co2eq + self.installation_kg_co2eq +
            self.operation_kg_co2eq + self.maintenance_kg_co2eq +
            self.end_of_life_kg_co2eq
        )

    @property
    def total_tons_co2eq(self) -> float:
        """Total carbon footprint in tons CO2eq."""
        return self.total_kg_co2eq / 1000


class EnergyPayback(BaseModel):
    """Energy payback analysis results."""

    embodied_energy_kwh: float = Field(ge=0, description="Total embodied energy in kWh")
    annual_energy_generation_kwh: float = Field(ge=0, description="Annual energy generation")
    energy_payback_time_years: float = Field(ge=0, description="Energy payback time in years")
    energy_return_on_investment: float = Field(ge=0, description="Energy return on investment (EROI)")
    lifetime_net_energy_kwh: float = Field(description="Lifetime net energy production")
    carbon_payback_time_years: float = Field(ge=0, description="Carbon payback time in years")


class EnvironmentalIndicators(BaseModel):
    """Comprehensive environmental impact indicators."""

    climate_change_kg_co2eq: float = Field(description="Climate change impact")
    acidification_kg_so2eq: float = Field(ge=0, description="Acidification potential")
    eutrophication_kg_po4eq: float = Field(ge=0, description="Eutrophication potential")
    ozone_depletion_kg_cfc11eq: float = Field(ge=0, description="Ozone depletion potential")
    photochemical_oxidation_kg_c2h4eq: float = Field(ge=0, description="Photochemical oxidation")
    water_consumption_m3: float = Field(ge=0, description="Water consumption")
    land_use_m2_years: float = Field(ge=0, description="Land use")
    resource_depletion_score: float = Field(ge=0, description="Resource depletion score")


class CircularityMetrics(BaseModel):
    """Circularity and circular economy metrics."""

    material_circularity_index: float = Field(ge=0, le=1, description="Material circularity index (0-1)")
    recycled_content_fraction: float = Field(ge=0, le=1, description="Fraction of recycled content")
    end_of_life_recovery_rate: float = Field(ge=0, le=1, description="End-of-life recovery rate")
    lifetime_extension_factor: float = Field(ge=1, description="Lifetime extension through maintenance/repair")
    virgin_material_use_kg: float = Field(ge=0, description="Virgin material use")
    recycled_material_use_kg: float = Field(ge=0, description="Recycled material use")


class LCAAnalyzer:
    """
    Life cycle assessment analyzer for PV modules.

    Provides methods for carbon footprint calculation, energy payback analysis,
    and comprehensive environmental impact assessment.
    """

    def __init__(
        self,
        grid_carbon_intensity_kg_per_kwh: float = 0.5,
        analysis_period_years: int = 30,
        discount_rate: float = 0.03
    ):
        """
        Initialize LCA analyzer.

        Args:
            grid_carbon_intensity_kg_per_kwh: Grid carbon intensity (kg CO2eq/kWh)
            analysis_period_years: Analysis period for LCA
            discount_rate: Discount rate for future impacts
        """
        self.grid_carbon_intensity = grid_carbon_intensity_kg_per_kwh
        self.analysis_period = analysis_period_years
        self.discount_rate = discount_rate

    def carbon_footprint(
        self,
        module_power_w: float,
        module_weight_kg: float,
        manufacturing_location: str = "China",
        transportation_km: float = 10000,
        lifetime_years: int = 25,
        recycling_at_eol: bool = True,
        degradation_rate: float = 0.005
    ) -> CarbonFootprint:
        """
        Calculate comprehensive carbon footprint of PV module lifecycle.

        Args:
            module_power_w: Module power rating in watts
            module_weight_kg: Module weight in kg
            manufacturing_location: Manufacturing location affecting energy mix
            transportation_km: Total transportation distance
            lifetime_years: Expected lifetime in years
            recycling_at_eol: Whether module is recycled at end-of-life
            degradation_rate: Annual power degradation rate

        Returns:
            CarbonFootprint with detailed emissions breakdown
        """
        # Raw materials CO2eq (based on material composition)
        # Typical PV module: ~70% glass, 15% aluminum, 10% silicon, 5% other
        raw_materials_co2 = self._calculate_raw_materials_carbon(module_weight_kg)

        # Manufacturing CO2eq (depends on location and energy mix)
        manufacturing_factors = {
            "China": 1.2,  # High coal usage
            "EU": 0.7,
            "US": 0.9,
            "India": 1.1,
            "Other": 1.0
        }
        manufacturing_factor = manufacturing_factors.get(manufacturing_location, 1.0)
        # Typical: 50-150 kg CO2eq per kWp, using 100 kg CO2eq/kWp as baseline
        manufacturing_co2 = (module_power_w / 1000) * 100 * manufacturing_factor

        # Transportation CO2eq
        # Typical: 0.05 kg CO2eq per ton-km for sea freight
        transportation_co2 = (module_weight_kg / 1000) * transportation_km * 0.05

        # Installation CO2eq
        # Typical: 10-20 kg CO2eq per kWp
        installation_co2 = (module_power_w / 1000) * 15

        # Operation CO2eq (minimal for PV, mainly monitoring)
        operation_co2 = 0.5 * lifetime_years

        # Maintenance CO2eq
        # Includes cleaning, repairs, inverter replacement
        maintenance_co2 = (module_power_w / 1000) * 2 * lifetime_years

        # End-of-life CO2eq
        if recycling_at_eol:
            # Recycling has costs but avoids virgin material production
            # Net benefit: -50 to -100 kg CO2eq per ton recycled
            eol_co2 = -(module_weight_kg / 1000) * 75  # Negative = benefit
        else:
            # Landfilling has minimal direct emissions
            eol_co2 = (module_weight_kg / 1000) * 10

        return CarbonFootprint(
            raw_materials_kg_co2eq=raw_materials_co2,
            manufacturing_kg_co2eq=manufacturing_co2,
            transportation_kg_co2eq=transportation_co2,
            installation_kg_co2eq=installation_co2,
            operation_kg_co2eq=operation_co2,
            maintenance_kg_co2eq=maintenance_co2,
            end_of_life_kg_co2eq=eol_co2
        )

    def energy_payback(
        self,
        module_power_w: float,
        module_weight_kg: float,
        annual_irradiation_kwh_per_m2: float,
        module_area_m2: float,
        performance_ratio: float = 0.80,
        lifetime_years: int = 25,
        degradation_rate: float = 0.005,
        manufacturing_location: str = "China"
    ) -> EnergyPayback:
        """
        Calculate energy payback time and energy return on investment.

        Args:
            module_power_w: Module power rating in watts
            module_weight_kg: Module weight in kg
            annual_irradiation_kwh_per_m2: Annual solar irradiation
            module_area_m2: Module area in m²
            performance_ratio: System performance ratio
            lifetime_years: Expected lifetime in years
            degradation_rate: Annual power degradation rate
            manufacturing_location: Manufacturing location

        Returns:
            EnergyPayback with detailed energy analysis
        """
        # Calculate embodied energy (energy required for production)
        embodied_energy = self._calculate_embodied_energy(
            module_power_w,
            module_weight_kg,
            manufacturing_location
        )

        # Calculate annual energy generation (first year)
        annual_generation = (
            module_power_w / 1000 *  # Convert to kW
            annual_irradiation_kwh_per_m2 *
            performance_ratio
        )

        # Calculate energy payback time
        # Account for degradation by using average output over EPBT period
        epbt = embodied_energy / annual_generation if annual_generation > 0 else 0

        # Calculate lifetime energy generation (accounting for degradation)
        lifetime_generation = 0
        for year in range(lifetime_years):
            year_output = annual_generation * (1 - degradation_rate) ** year
            lifetime_generation += year_output

        # Energy return on investment (EROI)
        eroi = lifetime_generation / embodied_energy if embodied_energy > 0 else 0

        # Lifetime net energy
        lifetime_net_energy = lifetime_generation - embodied_energy

        # Carbon payback time
        carbon_footprint = self.carbon_footprint(
            module_power_w,
            module_weight_kg,
            manufacturing_location
        )

        # Calculate avoided emissions per year
        avoided_emissions_per_year = annual_generation * self.grid_carbon_intensity

        # Carbon payback time
        cpbt = (
            carbon_footprint.total_kg_co2eq / avoided_emissions_per_year
            if avoided_emissions_per_year > 0 else 0
        )

        return EnergyPayback(
            embodied_energy_kwh=embodied_energy,
            annual_energy_generation_kwh=annual_generation,
            energy_payback_time_years=epbt,
            energy_return_on_investment=eroi,
            lifetime_net_energy_kwh=lifetime_net_energy,
            carbon_payback_time_years=cpbt
        )

    def environmental_indicators(
        self,
        module_power_w: float,
        module_weight_kg: float,
        manufacturing_location: str = "China",
        lifetime_years: int = 25,
        annual_irradiation_kwh_per_m2: float = 1800
    ) -> EnvironmentalIndicators:
        """
        Calculate comprehensive environmental impact indicators.

        Args:
            module_power_w: Module power rating in watts
            module_weight_kg: Module weight in kg
            manufacturing_location: Manufacturing location
            lifetime_years: Expected lifetime in years
            annual_irradiation_kwh_per_m2: Annual solar irradiation

        Returns:
            EnvironmentalIndicators with comprehensive impact assessment
        """
        # Get carbon footprint
        carbon = self.carbon_footprint(
            module_power_w,
            module_weight_kg,
            manufacturing_location,
            lifetime_years=lifetime_years
        )

        # Acidification potential (kg SO2eq)
        # Mainly from manufacturing and material production
        acidification = (module_power_w / 1000) * 0.05 + module_weight_kg * 0.002

        # Eutrophication potential (kg PO4eq)
        eutrophication = (module_power_w / 1000) * 0.01 + module_weight_kg * 0.0005

        # Ozone depletion potential (kg CFC-11eq)
        # Very low for modern PV manufacturing
        ozone_depletion = module_weight_kg * 0.000001

        # Photochemical oxidation (kg C2H4eq)
        photochemical = (module_power_w / 1000) * 0.008

        # Water consumption (m³)
        # Mainly in silicon and glass production
        water_consumption = module_weight_kg * 0.05

        # Land use (m² × years)
        # Assuming 15% efficiency and 1 m² = 150W
        module_area = module_power_w / 150
        land_use = module_area * lifetime_years

        # Resource depletion score (dimensionless, 0-100)
        # Based on use of scarce materials (silver, silicon)
        resource_score = self._calculate_resource_depletion_score(module_weight_kg)

        return EnvironmentalIndicators(
            climate_change_kg_co2eq=carbon.total_kg_co2eq,
            acidification_kg_so2eq=acidification,
            eutrophication_kg_po4eq=eutrophication,
            ozone_depletion_kg_cfc11eq=ozone_depletion,
            photochemical_oxidation_kg_c2h4eq=photochemical,
            water_consumption_m3=water_consumption,
            land_use_m2_years=land_use,
            resource_depletion_score=resource_score
        )

    def circularity_assessment(
        self,
        module_weight_kg: float,
        recycled_content_kg: float,
        expected_recovery_kg: float,
        lifetime_extension_years: float = 0,
        baseline_lifetime_years: int = 25
    ) -> CircularityMetrics:
        """
        Assess circular economy metrics for PV module.

        Args:
            module_weight_kg: Total module weight
            recycled_content_kg: Weight of recycled materials in module
            expected_recovery_kg: Expected recovered materials at end-of-life
            lifetime_extension_years: Additional lifetime from repair/maintenance
            baseline_lifetime_years: Baseline expected lifetime

        Returns:
            CircularityMetrics with circular economy indicators
        """
        # Recycled content fraction
        recycled_content_fraction = recycled_content_kg / module_weight_kg if module_weight_kg > 0 else 0

        # End-of-life recovery rate
        eol_recovery_rate = expected_recovery_kg / module_weight_kg if module_weight_kg > 0 else 0

        # Lifetime extension factor
        total_lifetime = baseline_lifetime_years + lifetime_extension_years
        lifetime_extension_factor = total_lifetime / baseline_lifetime_years

        # Material Circularity Index (MCI) - simplified Ellen MacArthur Foundation approach
        # MCI = (recycled input + recovered output) / 2, adjusted for utility
        virgin_material = module_weight_kg - recycled_content_kg
        utility_factor = total_lifetime / baseline_lifetime_years

        # Linear flow index (0 = fully circular, 1 = fully linear)
        lfi = (virgin_material + (module_weight_kg - expected_recovery_kg)) / (2 * module_weight_kg)

        # Material circularity index
        mci = (1 - lfi) * utility_factor
        mci = max(0, min(1, mci))  # Bound between 0 and 1

        return CircularityMetrics(
            material_circularity_index=mci,
            recycled_content_fraction=recycled_content_fraction,
            end_of_life_recovery_rate=eol_recovery_rate,
            lifetime_extension_factor=lifetime_extension_factor,
            virgin_material_use_kg=virgin_material,
            recycled_material_use_kg=recycled_content_kg
        )

    def comparative_analysis(
        self,
        scenarios: List[Dict]
    ) -> List[Dict]:
        """
        Compare environmental impacts of different scenarios.

        Args:
            scenarios: List of scenario configurations

        Returns:
            List of comparative analysis results
        """
        results = []

        for scenario in scenarios:
            name = scenario.get("name", "Unnamed")
            module_power = scenario.get("module_power_w", 400)
            module_weight = scenario.get("module_weight_kg", 20)
            location = scenario.get("manufacturing_location", "China")
            irradiation = scenario.get("annual_irradiation_kwh_per_m2", 1800)
            module_area = scenario.get("module_area_m2", 2.0)

            # Calculate all metrics
            carbon = self.carbon_footprint(module_power, module_weight, location)
            energy = self.energy_payback(
                module_power,
                module_weight,
                irradiation,
                module_area,
                manufacturing_location=location
            )
            indicators = self.environmental_indicators(
                module_power,
                module_weight,
                location,
                annual_irradiation_kwh_per_m2=irradiation
            )

            results.append({
                "scenario_name": name,
                "carbon_footprint_kg_co2eq": carbon.total_kg_co2eq,
                "energy_payback_years": energy.energy_payback_time_years,
                "carbon_payback_years": energy.carbon_payback_time_years,
                "eroi": energy.energy_return_on_investment,
                "environmental_indicators": indicators.model_dump(),
                "carbon_intensity_g_per_kwh": (
                    carbon.total_kg_co2eq * 1000 / energy.annual_energy_generation_kwh / 25
                    if energy.annual_energy_generation_kwh > 0 else 0
                )
            })

        return results

    @staticmethod
    def _calculate_raw_materials_carbon(module_weight_kg: float) -> float:
        """
        Calculate CO2eq emissions from raw materials.

        Typical composition:
        - Glass: 70% (0.5 kg CO2eq/kg)
        - Aluminum: 15% (8.5 kg CO2eq/kg)
        - Silicon: 10% (60 kg CO2eq/kg)
        - Other: 5% (2 kg CO2eq/kg)
        """
        glass_co2 = module_weight_kg * 0.70 * 0.5
        aluminum_co2 = module_weight_kg * 0.15 * 8.5
        silicon_co2 = module_weight_kg * 0.10 * 60
        other_co2 = module_weight_kg * 0.05 * 2

        return glass_co2 + aluminum_co2 + silicon_co2 + other_co2

    @staticmethod
    def _calculate_embodied_energy(
        module_power_w: float,
        module_weight_kg: float,
        manufacturing_location: str
    ) -> float:
        """
        Calculate embodied energy in PV module.

        Returns energy in kWh.
        """
        # Base embodied energy: ~600-800 kWh/kWp for crystalline silicon
        base_energy_per_kwp = 700  # kWh/kWp

        # Adjust for manufacturing location (energy mix efficiency)
        location_factors = {
            "China": 1.1,
            "EU": 0.9,
            "US": 1.0,
            "India": 1.15,
            "Other": 1.0
        }
        location_factor = location_factors.get(manufacturing_location, 1.0)

        embodied_energy = (module_power_w / 1000) * base_energy_per_kwp * location_factor

        return embodied_energy

    @staticmethod
    def _calculate_resource_depletion_score(module_weight_kg: float) -> float:
        """
        Calculate resource depletion score.

        Higher score = more resource depletion
        Based on use of scarce materials relative to reserves.
        """
        # Simplified scoring based on typical module composition
        # Silver: high scarcity (score: 80)
        # Silicon: medium scarcity (score: 30)
        # Aluminum, glass: low scarcity (score: 10)

        silver_kg = module_weight_kg * 0.0005  # ~0.05%
        silicon_kg = module_weight_kg * 0.10  # ~10%
        common_kg = module_weight_kg * 0.8995  # ~90%

        score = (
            silver_kg * 80 +
            silicon_kg * 30 +
            common_kg * 10
        )

        # Normalize to 0-100 scale
        normalized_score = min(100, score / module_weight_kg)

        return normalized_score
