"""
Data Models
===========

Pydantic models for data validation and structure.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CellType(str, Enum):
    """PV cell technology types."""
    MONO_PERC = "mono_perc"
    POLY = "poly"
    TOPCON = "topcon"
    HJT = "hjt"
    IBC = "ibc"
    PEROVSKITE = "perovskite"


class MaterialType(str, Enum):
    """Material types for PV modules."""
    GLASS = "glass"
    EVA = "eva"
    POE = "poe"
    BACKSHEET = "backsheet"
    FRAME = "frame"
    JUNCTION_BOX = "junction_box"
    CELL = "cell"
    RIBBON = "ribbon"


class Material(BaseModel):
    """Material specification."""
    name: str
    type: MaterialType
    manufacturer: Optional[str] = None
    cost_per_unit: float = Field(gt=0)
    recyclability_score: float = Field(ge=0, le=100)
    carbon_footprint: float = Field(ge=0)
    properties: Dict[str, Any] = {}


class ModuleDesign(BaseModel):
    """PV module design specification."""
    name: str
    cell_type: CellType
    num_cells: int = Field(gt=0)
    cell_efficiency: float = Field(gt=0, le=100)
    module_power: float = Field(gt=0)
    voltage_voc: float = Field(gt=0)
    current_isc: float = Field(gt=0)
    voltage_vmpp: float = Field(gt=0)
    current_impp: float = Field(gt=0)
    materials: List[Material] = []
    dimensions: Dict[str, float] = {}
    weight: Optional[float] = None

    @validator('cell_efficiency')
    def validate_efficiency(cls, v):
        """Validate cell efficiency is reasonable."""
        if v > 30:
            raise ValueError('Cell efficiency exceeds current technological limits')
        return v


class CTMLosses(BaseModel):
    """Cell-to-Module loss factors (k1-k15, k21-k24)."""
    # Optical losses
    k1_reflection: float = Field(default=0.98, ge=0, le=1)
    k2_shading: float = Field(default=0.97, ge=0, le=1)
    k3_absorption: float = Field(default=0.99, ge=0, le=1)

    # Electrical losses
    k4_resistive: float = Field(default=0.98, ge=0, le=1)
    k5_mismatch: float = Field(default=0.98, ge=0, le=1)
    k6_junction_box: float = Field(default=0.995, ge=0, le=1)

    # Thermal losses
    k7_temperature: float = Field(default=0.96, ge=0, le=1)
    k8_hotspot: float = Field(default=0.99, ge=0, le=1)

    # Assembly losses
    k9_encapsulation: float = Field(default=0.99, ge=0, le=1)
    k10_lamination: float = Field(default=0.995, ge=0, le=1)

    # Long-term degradation
    k11_lid: float = Field(default=0.98, ge=0, le=1)  # Light-induced degradation
    k12_pid: float = Field(default=0.99, ge=0, le=1)  # Potential-induced degradation
    k13_mechanical: float = Field(default=0.995, ge=0, le=1)
    k14_cell_degradation: float = Field(default=0.995, ge=0, le=1)
    k15_interconnect: float = Field(default=0.995, ge=0, le=1)

    # Environmental factors
    k21_humidity: float = Field(default=0.99, ge=0, le=1)
    k22_uv_exposure: float = Field(default=0.99, ge=0, le=1)
    k23_thermal_cycling: float = Field(default=0.995, ge=0, le=1)
    k24_corrosion: float = Field(default=0.995, ge=0, le=1)

    def calculate_total_ctm_ratio(self) -> float:
        """Calculate total CTM ratio from all k-factors."""
        factors = [
            self.k1_reflection, self.k2_shading, self.k3_absorption,
            self.k4_resistive, self.k5_mismatch, self.k6_junction_box,
            self.k7_temperature, self.k8_hotspot,
            self.k9_encapsulation, self.k10_lamination,
            self.k11_lid, self.k12_pid, self.k13_mechanical,
            self.k14_cell_degradation, self.k15_interconnect,
            self.k21_humidity, self.k22_uv_exposure,
            self.k23_thermal_cycling, self.k24_corrosion
        ]
        total = 1.0
        for factor in factors:
            total *= factor
        return total


class SystemDesign(BaseModel):
    """PV system design specification."""
    name: str
    location: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    module_design: Optional[ModuleDesign] = None
    num_modules: int = Field(gt=0)
    system_capacity: float = Field(gt=0)
    inverter_capacity: float = Field(gt=0)
    tilt_angle: float = Field(ge=0, le=90)
    azimuth: float = Field(ge=0, le=360)
    mounting_type: str = "fixed"
    dc_ac_ratio: Optional[float] = None

    @validator('dc_ac_ratio', always=True)
    def calculate_dc_ac_ratio(cls, v, values):
        """Calculate DC/AC ratio if not provided."""
        if v is None and 'system_capacity' in values and 'inverter_capacity' in values:
            return values['system_capacity'] / values['inverter_capacity']
        return v


class ProjectData(BaseModel):
    """Complete project data structure."""
    project_name: str
    created_at: datetime
    last_modified: datetime
    material_selections: List[Material] = []
    module_design: Optional[ModuleDesign] = None
    ctm_losses: Optional[CTMLosses] = None
    system_design: Optional[SystemDesign] = None
    eya_results: Dict[str, Any] = {}
    performance_data: Dict[str, Any] = {}
    fault_data: Dict[str, Any] = {}
    hya_results: Dict[str, Any] = {}
    forecast_data: Dict[str, Any] = {}
    revamp_data: Dict[str, Any] = {}
    circularity_data: Dict[str, Any] = {}
