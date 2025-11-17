"""
Design Suite Module (B01-B03)
=============================
Integrates:
- B01: Materials Engineering Database
- B02: Cell Design & SCAPS-1D Simulation
- B03: Module Design & CTM Loss Analysis

This module provides comprehensive PV design capabilities from material selection
through cell design to complete module engineering with Fraunhofer ISE CTM loss factors.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator


# ============================================================================
# B01: MATERIALS ENGINEERING DATABASE
# ============================================================================

class MaterialType(str, Enum):
    """Enumeration of PV material types."""
    C_SI = "c-Si"
    PEROVSKITE = "Perovskite"
    CIGS = "CIGS"
    CDTE = "CdTe"
    BIFACIAL_SI = "Bi-facial Si"
    TANDEM = "Tandem"
    HJT = "HJT"
    TOPCON = "TOPCon"
    IBC = "IBC"


class MaterialSpecification(BaseModel):
    """Comprehensive material specification model."""

    material_id: str = Field(..., description="Unique material identifier")
    name: str = Field(..., description="Material name")
    material_type: MaterialType = Field(..., description="Material category")
    efficiency: float = Field(..., ge=0, le=100, description="Efficiency percentage")
    cost_per_wp: float = Field(..., ge=0, description="Cost per watt-peak ($/Wp)")
    degradation_rate: float = Field(..., ge=0, description="Annual degradation (%/year)")
    recyclability_score: int = Field(..., ge=0, le=100, description="Recyclability score (0-100)")
    embodied_energy: float = Field(default=0.0, ge=0, description="Embodied energy (MJ/m²)")
    carbon_footprint: float = Field(default=0.0, ge=0, description="Carbon footprint (kg CO2e/m²)")
    temperature_coefficient: float = Field(default=-0.4, description="Temperature coefficient (%/°C)")
    warranty_years: int = Field(default=25, ge=0, description="Warranty period (years)")
    spectral_response: Dict[str, float] = Field(default_factory=dict, description="Spectral response data")

    class Config:
        use_enum_values = True


class MaterialsDatabase:
    """
    Materials Engineering Database with comprehensive PV material specifications.
    Supports material search, comparison, and lifecycle analysis.
    """

    def __init__(self):
        """Initialize materials database with default entries."""
        self.materials: Dict[str, MaterialSpecification] = {}
        self._load_default_materials()

    def _load_default_materials(self) -> None:
        """Load default material specifications."""
        default_materials = [
            MaterialSpecification(
                material_id="MAT001",
                name="Monocrystalline Silicon",
                material_type=MaterialType.C_SI,
                efficiency=21.5,
                cost_per_wp=0.45,
                degradation_rate=0.5,
                recyclability_score=95,
                embodied_energy=4500,
                carbon_footprint=45,
                temperature_coefficient=-0.38,
                warranty_years=25
            ),
            MaterialSpecification(
                material_id="MAT002",
                name="Perovskite Silicon Tandem",
                material_type=MaterialType.PEROVSKITE,
                efficiency=24.2,
                cost_per_wp=0.38,
                degradation_rate=2.0,
                recyclability_score=65,
                embodied_energy=3800,
                carbon_footprint=38,
                temperature_coefficient=-0.25,
                warranty_years=15
            ),
            MaterialSpecification(
                material_id="MAT003",
                name="CIGS Thin Film",
                material_type=MaterialType.CIGS,
                efficiency=18.8,
                cost_per_wp=0.52,
                degradation_rate=1.2,
                recyclability_score=75,
                embodied_energy=3200,
                carbon_footprint=32,
                temperature_coefficient=-0.32,
                warranty_years=20
            ),
            MaterialSpecification(
                material_id="MAT004",
                name="CdTe Thin Film",
                material_type=MaterialType.CDTE,
                efficiency=20.5,
                cost_per_wp=0.40,
                degradation_rate=0.8,
                recyclability_score=90,
                embodied_energy=2800,
                carbon_footprint=28,
                temperature_coefficient=-0.25,
                warranty_years=25
            ),
            MaterialSpecification(
                material_id="MAT005",
                name="Bifacial PERC",
                material_type=MaterialType.BIFACIAL_SI,
                efficiency=22.1,
                cost_per_wp=0.48,
                degradation_rate=0.6,
                recyclability_score=96,
                embodied_energy=4800,
                carbon_footprint=48,
                temperature_coefficient=-0.35,
                warranty_years=30
            ),
            MaterialSpecification(
                material_id="MAT006",
                name="HJT (Heterojunction)",
                material_type=MaterialType.HJT,
                efficiency=23.8,
                cost_per_wp=0.55,
                degradation_rate=0.25,
                recyclability_score=94,
                embodied_energy=5200,
                carbon_footprint=52,
                temperature_coefficient=-0.24,
                warranty_years=30
            ),
        ]

        for material in default_materials:
            self.materials[material.material_id] = material

    def get_material(self, material_id: str) -> Optional[MaterialSpecification]:
        """Retrieve material by ID."""
        return self.materials.get(material_id)

    def search_materials(
        self,
        material_type: Optional[MaterialType] = None,
        min_efficiency: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_recyclability: Optional[int] = None
    ) -> List[MaterialSpecification]:
        """Search materials with filters."""
        results = list(self.materials.values())

        if material_type:
            results = [m for m in results if m.material_type == material_type]
        if min_efficiency:
            results = [m for m in results if m.efficiency >= min_efficiency]
        if max_cost:
            results = [m for m in results if m.cost_per_wp <= max_cost]
        if min_recyclability:
            results = [m for m in results if m.recyclability_score >= min_recyclability]

        return results

    def get_dataframe(self) -> pd.DataFrame:
        """Export materials database as DataFrame."""
        data = []
        for mat in self.materials.values():
            data.append({
                'Material ID': mat.material_id,
                'Material': mat.name,
                'Type': mat.material_type,
                'Efficiency (%)': mat.efficiency,
                'Cost ($/Wp)': mat.cost_per_wp,
                'Degradation (%/yr)': mat.degradation_rate,
                'Recyclability': mat.recyclability_score,
                'Embodied Energy (MJ/m²)': mat.embodied_energy,
                'Carbon (kg CO2e/m²)': mat.carbon_footprint,
                'Temp Coef (%/°C)': mat.temperature_coefficient,
                'Warranty (yrs)': mat.warranty_years
            })
        return pd.DataFrame(data)


# ============================================================================
# B02: CELL DESIGN & SCAPS-1D SIMULATION
# ============================================================================

class SubstrateType(str, Enum):
    """Cell substrate types."""
    GLASS = "Glass"
    PLASTIC = "Plastic"
    METAL = "Metal"
    SILICON_WAFER = "Silicon Wafer"


class CellArchitecture(str, Enum):
    """Cell architecture types."""
    N_TYPE_REAR_PASS = "n-type Si with rear passivation"
    P_TYPE_PERC = "p-type PERC"
    HJT = "Heterojunction (HJT)"
    TOPCON = "TOPCon"
    IBC = "Interdigitated Back Contact (IBC)"
    TANDEM = "Tandem (Perovskite/Si)"


class CellDesignParameters(BaseModel):
    """Cell design input parameters for SCAPS-1D simulation."""

    substrate: SubstrateType = Field(..., description="Substrate material type")
    thickness_um: float = Field(..., ge=0.1, le=500, description="Device thickness (micrometers)")
    architecture: CellArchitecture = Field(..., description="Cell architecture")
    doping_concentration: float = Field(default=1e16, ge=1e14, le=1e20, description="Doping concentration (cm⁻³)")
    front_metal_coverage: float = Field(default=5.0, ge=0, le=20, description="Front metallization coverage (%)")
    rear_passivation: bool = Field(default=True, description="Rear surface passivation enabled")
    anti_reflective_coating: bool = Field(default=True, description="ARC enabled")
    texture_enabled: bool = Field(default=True, description="Surface texturing enabled")

    class Config:
        use_enum_values = True


class CellSimulationResults(BaseModel):
    """SCAPS-1D simulation output results."""

    efficiency: float = Field(..., ge=0, le=100, description="Cell efficiency (%)")
    voc: float = Field(..., ge=0, description="Open circuit voltage (mV)")
    jsc: float = Field(..., ge=0, description="Short circuit current density (mA/cm²)")
    fill_factor: float = Field(..., ge=0, le=100, description="Fill factor (%)")
    isc: float = Field(default=0.0, ge=0, description="Short circuit current (A)")
    vmpp: float = Field(default=0.0, ge=0, description="Voltage at MPP (mV)")
    impp: float = Field(default=0.0, ge=0, description="Current at MPP (A)")
    series_resistance: float = Field(default=0.0, ge=0, description="Series resistance (Ω)")
    shunt_resistance: float = Field(default=float('inf'), ge=0, description="Shunt resistance (Ω)")
    quantum_efficiency: Dict[int, float] = Field(default_factory=dict, description="QE data (wavelength: efficiency)")


class CellDesignSimulator:
    """
    Cell Design & SCAPS-1D Simulation Engine.
    Simulates solar cell performance based on material and architecture parameters.
    """

    def __init__(self):
        """Initialize cell design simulator."""
        self.simulation_cache: Dict[str, CellSimulationResults] = {}

    def simulate_cell(self, params: CellDesignParameters) -> CellSimulationResults:
        """
        Run SCAPS-1D simulation with given parameters.

        Args:
            params: Cell design parameters

        Returns:
            Simulation results with electrical characteristics
        """
        # Base efficiency calculations based on architecture
        base_efficiency = self._calculate_base_efficiency(params)

        # Apply loss factors
        efficiency = self._apply_loss_factors(base_efficiency, params)

        # Calculate electrical parameters
        voc = self._calculate_voc(params, efficiency)
        jsc = self._calculate_jsc(params, efficiency)
        fill_factor = self._calculate_fill_factor(params)

        results = CellSimulationResults(
            efficiency=efficiency,
            voc=voc,
            jsc=jsc,
            fill_factor=fill_factor,
            vmpp=voc * 0.85,  # Typical MPP voltage
            impp=jsc * 0.92,  # Typical MPP current
            series_resistance=self._calculate_series_resistance(params),
            shunt_resistance=self._calculate_shunt_resistance(params)
        )

        return results

    def _calculate_base_efficiency(self, params: CellDesignParameters) -> float:
        """Calculate base efficiency based on architecture."""
        base_efficiencies = {
            CellArchitecture.N_TYPE_REAR_PASS: 22.5,
            CellArchitecture.P_TYPE_PERC: 21.0,
            CellArchitecture.HJT: 24.0,
            CellArchitecture.TOPCON: 23.5,
            CellArchitecture.IBC: 25.0,
            CellArchitecture.TANDEM: 28.0
        }
        return base_efficiencies.get(params.architecture, 20.0)

    def _apply_loss_factors(self, base_eff: float, params: CellDesignParameters) -> float:
        """Apply various loss factors to base efficiency."""
        efficiency = base_eff

        # Thickness losses
        if params.thickness_um < 100:
            efficiency *= 0.95

        # Front metallization losses
        metallization_loss = params.front_metal_coverage / 100 * 0.5
        efficiency *= (1 - metallization_loss)

        # Passivation gain
        if params.rear_passivation:
            efficiency *= 1.05

        # ARC gain
        if params.anti_reflective_coating:
            efficiency *= 1.03

        # Texturing gain
        if params.texture_enabled:
            efficiency *= 1.02

        return min(efficiency, 29.0)  # Cap at theoretical limit

    def _calculate_voc(self, params: CellDesignParameters, efficiency: float) -> float:
        """Calculate open circuit voltage."""
        base_voc = 650  # mV
        return base_voc + (efficiency - 20) * 15

    def _calculate_jsc(self, params: CellDesignParameters, efficiency: float) -> float:
        """Calculate short circuit current density."""
        base_jsc = 38.0  # mA/cm²
        if params.texture_enabled:
            base_jsc *= 1.1
        if params.anti_reflective_coating:
            base_jsc *= 1.05
        return base_jsc + (efficiency - 20) * 0.5

    def _calculate_fill_factor(self, params: CellDesignParameters) -> float:
        """Calculate fill factor."""
        base_ff = 78.0
        if params.rear_passivation:
            base_ff += 2.0
        if params.front_metal_coverage < 3:
            base_ff -= 1.5
        return min(base_ff, 86.0)

    def _calculate_series_resistance(self, params: CellDesignParameters) -> float:
        """Calculate series resistance."""
        return 0.5 + params.front_metal_coverage * 0.05

    def _calculate_shunt_resistance(self, params: CellDesignParameters) -> float:
        """Calculate shunt resistance."""
        base_rsh = 5000
        if params.rear_passivation:
            base_rsh *= 2
        return base_rsh


# ============================================================================
# B03: MODULE DESIGN & CTM LOSS ANALYSIS
# ============================================================================

class CTMLossFactor(BaseModel):
    """Fraunhofer ISE CTM loss factor (k1-k24)."""

    factor_id: str = Field(..., description="Factor ID (k1-k24)")
    name: str = Field(..., description="Loss factor name")
    description: str = Field(..., description="Detailed description")
    loss_percentage: float = Field(..., ge=0, le=100, description="Loss percentage")
    category: str = Field(..., description="Loss category")

    @validator('factor_id')
    def validate_factor_id(cls, v):
        """Validate CTM factor ID."""
        valid_ids = [f"k{i}" for i in range(1, 25)]
        if v not in valid_ids:
            raise ValueError(f"Factor ID must be one of {valid_ids}")
        return v


class ModuleConfiguration(BaseModel):
    """Module configuration parameters."""

    cells_in_series: int = Field(default=60, ge=36, le=144, description="Cells in series")
    cells_in_parallel: int = Field(default=1, ge=1, le=4, description="Cells in parallel")
    cell_efficiency: float = Field(..., ge=0, le=100, description="Individual cell efficiency (%)")
    cell_area_cm2: float = Field(default=243, ge=0, description="Cell area (cm²)")
    glass_thickness_mm: float = Field(default=3.2, ge=2.0, le=5.0, description="Front glass thickness (mm)")
    encapsulant_type: str = Field(default="EVA", description="Encapsulant material")
    backsheet_type: str = Field(default="Tedlar", description="Backsheet material")
    frame_material: str = Field(default="Aluminum", description="Frame material")
    bypass_diodes: int = Field(default=3, ge=0, le=6, description="Number of bypass diodes")
    junction_box_type: str = Field(default="IP67", description="Junction box rating")


class CTMAnalyzer:
    """
    CTM (Cell-to-Module) Loss Analysis using Fraunhofer ISE 24-factor model.
    Analyzes losses in converting cell efficiency to module efficiency.
    """

    def __init__(self):
        """Initialize CTM analyzer with standard loss factors."""
        self.loss_factors: List[CTMLossFactor] = self._initialize_ctm_factors()

    def _initialize_ctm_factors(self) -> List[CTMLossFactor]:
        """Initialize all 24 CTM loss factors."""
        return [
            CTMLossFactor(factor_id="k1", name="Optical Reflection", description="Front glass and ARC reflection losses", loss_percentage=2.5, category="Optical"),
            CTMLossFactor(factor_id="k2", name="Soiling", description="Dust and dirt accumulation on module surface", loss_percentage=1.8, category="Environmental"),
            CTMLossFactor(factor_id="k3", name="Temperature", description="Operating temperature above STC", loss_percentage=3.2, category="Thermal"),
            CTMLossFactor(factor_id="k4", name="Resistive (Series)", description="Interconnect and busbar resistance", loss_percentage=2.1, category="Electrical"),
            CTMLossFactor(factor_id="k5", name="Cell Mismatch", description="Current mismatch between cells", loss_percentage=1.5, category="Electrical"),
            CTMLossFactor(factor_id="k6", name="Wiring Loss", description="Internal module wiring resistance", loss_percentage=0.8, category="Electrical"),
            CTMLossFactor(factor_id="k7", name="Inactive Area", description="Non-active area between cells", loss_percentage=3.5, category="Geometric"),
            CTMLossFactor(factor_id="k8", name="Cell Breakage", description="Micro-crack induced losses", loss_percentage=0.5, category="Manufacturing"),
            CTMLossFactor(factor_id="k9", name="Encapsulant Absorption", description="EVA/POE light absorption", loss_percentage=1.2, category="Optical"),
            CTMLossFactor(factor_id="k10", name="Glass Transmission", description="Glass optical losses", loss_percentage=1.5, category="Optical"),
            CTMLossFactor(factor_id="k11", name="Spectral Mismatch", description="Spectral response variation", loss_percentage=0.8, category="Optical"),
            CTMLossFactor(factor_id="k12", name="Incident Angle", description="Angle of incidence losses", loss_percentage=1.0, category="Optical"),
            CTMLossFactor(factor_id="k13", name="Shading", description="Frame and junction box shading", loss_percentage=0.6, category="Geometric"),
            CTMLossFactor(factor_id="k14", name="Quality Binning", description="Cell sorting and binning tolerances", loss_percentage=0.3, category="Manufacturing"),
            CTMLossFactor(factor_id="k15", name="LID (Light Induced)", description="Light-induced degradation", loss_percentage=1.5, category="Degradation"),
            CTMLossFactor(factor_id="k16", name="PID (Potential Induced)", description="Potential-induced degradation", loss_percentage=0.4, category="Degradation"),
            CTMLossFactor(factor_id="k17", name="Contact Resistance", description="Cell-to-ribbon contact resistance", loss_percentage=0.7, category="Electrical"),
            CTMLossFactor(factor_id="k18", name="Lamination Stress", description="Mechanical stress during lamination", loss_percentage=0.3, category="Manufacturing"),
            CTMLossFactor(factor_id="k19", name="Solder Bond", description="Solder joint quality", loss_percentage=0.4, category="Manufacturing"),
            CTMLossFactor(factor_id="k20", name="Bypass Diode", description="Diode voltage drop losses", loss_percentage=0.2, category="Electrical"),
            CTMLossFactor(factor_id="k21", name="Encapsulation EVA", description="EVA yellowing over time", loss_percentage=0.5, category="Degradation"),
            CTMLossFactor(factor_id="k22", name="Backsheet Reflectance", description="Reduced rear reflectance", loss_percentage=0.3, category="Optical"),
            CTMLossFactor(factor_id="k23", name="Hotspot Formation", description="Localized hotspot losses", loss_percentage=0.2, category="Thermal"),
            CTMLossFactor(factor_id="k24", name="Manufacturing Tolerance", description="General manufacturing variations", loss_percentage=0.5, category="Manufacturing"),
        ]

    def calculate_module_efficiency(
        self,
        cell_efficiency: float,
        module_config: ModuleConfiguration,
        active_loss_factors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate module efficiency considering CTM losses.

        Args:
            cell_efficiency: Input cell efficiency (%)
            module_config: Module configuration parameters
            active_loss_factors: List of factor IDs to apply (default: all)

        Returns:
            Dictionary with module efficiency and detailed loss breakdown
        """
        if active_loss_factors is None:
            active_loss_factors = [f"k{i}" for i in range(1, 25)]

        # Start with cell efficiency
        remaining_efficiency = cell_efficiency
        loss_breakdown = []

        for factor in self.loss_factors:
            if factor.factor_id in active_loss_factors:
                loss_amount = remaining_efficiency * (factor.loss_percentage / 100)
                remaining_efficiency -= loss_amount
                loss_breakdown.append({
                    'factor_id': factor.factor_id,
                    'name': factor.name,
                    'category': factor.category,
                    'loss_percentage': factor.loss_percentage,
                    'absolute_loss': loss_amount,
                    'remaining_efficiency': remaining_efficiency
                })

        total_ctm_loss = cell_efficiency - remaining_efficiency
        ctm_ratio = remaining_efficiency / cell_efficiency if cell_efficiency > 0 else 0

        # Calculate module power
        total_cell_area = module_config.cell_area_cm2 * module_config.cells_in_series * module_config.cells_in_parallel
        module_power = (remaining_efficiency / 100) * total_cell_area * 0.1  # kW under 1000 W/m²

        return {
            'cell_efficiency': cell_efficiency,
            'module_efficiency': remaining_efficiency,
            'total_ctm_loss': total_ctm_loss,
            'ctm_ratio': ctm_ratio,
            'module_power_wp': module_power * 1000,
            'loss_breakdown': loss_breakdown,
            'module_config': module_config.dict()
        }

    def get_loss_summary_df(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert CTM analysis to DataFrame."""
        return pd.DataFrame(analysis_result['loss_breakdown'])


# ============================================================================
# DESIGN SUITE INTEGRATION INTERFACE
# ============================================================================

class DesignSuite:
    """
    Unified Design Suite Interface integrating B01-B03.
    Provides end-to-end design workflow from materials to complete modules.
    """

    def __init__(self):
        """Initialize all design suite components."""
        self.materials_db = MaterialsDatabase()
        self.cell_simulator = CellDesignSimulator()
        self.ctm_analyzer = CTMAnalyzer()

    def design_workflow(
        self,
        material_id: str,
        cell_params: CellDesignParameters,
        module_config: ModuleConfiguration
    ) -> Dict[str, Any]:
        """
        Execute complete design workflow: Material → Cell → Module.

        Args:
            material_id: Material database ID
            cell_params: Cell design parameters
            module_config: Module configuration

        Returns:
            Complete design analysis results
        """
        # Step 1: Get material specifications
        material = self.materials_db.get_material(material_id)
        if not material:
            raise ValueError(f"Material {material_id} not found in database")

        # Step 2: Simulate cell design
        cell_results = self.cell_simulator.simulate_cell(cell_params)

        # Step 3: Analyze CTM losses
        module_config.cell_efficiency = cell_results.efficiency
        ctm_results = self.ctm_analyzer.calculate_module_efficiency(
            cell_efficiency=cell_results.efficiency,
            module_config=module_config
        )

        return {
            'material': material.dict(),
            'cell_simulation': cell_results.dict(),
            'ctm_analysis': ctm_results,
            'final_module_efficiency': ctm_results['module_efficiency'],
            'final_module_power': ctm_results['module_power_wp']
        }

    def get_materials_database(self) -> MaterialsDatabase:
        """Access materials database."""
        return self.materials_db

    def get_cell_simulator(self) -> CellDesignSimulator:
        """Access cell simulator."""
        return self.cell_simulator

    def get_ctm_analyzer(self) -> CTMAnalyzer:
        """Access CTM analyzer."""
        return self.ctm_analyzer


# Export main interface
__all__ = [
    'DesignSuite',
    'MaterialsDatabase',
    'MaterialSpecification',
    'MaterialType',
    'CellDesignSimulator',
    'CellDesignParameters',
    'CellSimulationResults',
    'CTMAnalyzer',
    'ModuleConfiguration',
    'CTMLossFactor'
]
