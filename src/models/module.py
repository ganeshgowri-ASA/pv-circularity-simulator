"""Module design data models."""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from .cell import CellDesign


class ModuleLayout(BaseModel):
    """Module layout configuration."""

    num_cells: Literal[60, 72, 120, 132, 144] = Field(..., description="Number of cells in module")
    rows: int = Field(..., gt=0, description="Number of rows")
    columns: int = Field(..., gt=0, description="Number of columns")
    num_strings: int = Field(..., gt=0, description="Number of parallel strings")
    cells_per_string: int = Field(..., gt=0, description="Cells per string (series)")
    bypass_diodes: int = Field(..., gt=0, description="Number of bypass diodes")
    cells_per_diode: int = Field(..., gt=0, description="Cells per bypass diode")

    @property
    def total_cells(self) -> int:
        """Calculate total number of cells."""
        return self.num_strings * self.cells_per_string


class ModuleConfiguration(BaseModel):
    """Complete module configuration."""

    name: str = Field(..., description="Module configuration name")
    cell_design: CellDesign
    layout: ModuleLayout

    # Dimensions
    length_mm: float = Field(..., gt=0, description="Module length in mm")
    width_mm: float = Field(..., gt=0, description="Module width in mm")
    thickness_mm: float = Field(..., gt=0, description="Module thickness in mm")

    # Frame and encapsulation
    frame_type: Literal["aluminum", "frameless", "composite"] = Field(default="aluminum")
    glass_front_mm: float = Field(default=3.2, description="Front glass thickness in mm")
    glass_back_mm: Optional[float] = Field(None, description="Back glass thickness in mm (for bifacial)")
    backsheet_type: Optional[str] = Field(None, description="Backsheet type (for monofacial)")
    encapsulant_type: Literal["EVA", "POE", "TPO"] = Field(default="EVA")

    # Electrical
    junction_box_type: str = Field(default="standard", description="Junction box type")
    cable_length_mm: float = Field(default=1200, description="Cable length in mm")
    connector_type: str = Field(default="MC4", description="Connector type")

    # Weight and cost (optional, can be calculated)
    weight_kg: Optional[float] = Field(None, description="Module weight in kg")
    cost_usd: Optional[float] = Field(None, description="Module cost in USD")

    @property
    def area_m2(self) -> float:
        """Calculate module area in m²."""
        return (self.length_mm * self.width_mm) / 1_000_000

    @property
    def is_bifacial(self) -> bool:
        """Check if module is bifacial."""
        return self.glass_back_mm is not None and self.cell_design.template.bifacial


class ModuleDesign(BaseModel):
    """Complete module design with performance specifications."""

    configuration: ModuleConfiguration

    # STC Performance (Standard Test Conditions: 1000 W/m², 25°C, AM1.5)
    pmax_stc_w: float = Field(..., gt=0, description="Maximum power at STC in watts")
    voc_stc_v: float = Field(..., gt=0, description="Open circuit voltage at STC")
    isc_stc_a: float = Field(..., gt=0, description="Short circuit current at STC")
    vmp_stc_v: float = Field(..., gt=0, description="Voltage at max power at STC")
    imp_stc_a: float = Field(..., gt=0, description="Current at max power at STC")
    efficiency_stc_pct: float = Field(..., gt=0, description="Module efficiency at STC")

    # NOCT Performance (Nominal Operating Cell Temperature: 800 W/m², 20°C ambient, 1 m/s wind)
    pmax_noct_w: float = Field(..., gt=0, description="Maximum power at NOCT")
    noct_temp_c: float = Field(default=45.0, description="NOCT temperature in °C")

    # Temperature coefficients
    temp_coeff_pmax_pct: float = Field(..., description="Temperature coefficient of Pmax in %/°C")
    temp_coeff_voc_pct: float = Field(..., description="Temperature coefficient of Voc in %/°C")
    temp_coeff_isc_pct: float = Field(..., description="Temperature coefficient of Isc in %/°C")

    # Bifacial parameters (if applicable)
    bifacial_gain_pct: Optional[float] = Field(None, ge=0, le=100, description="Bifacial gain in %")

    # Degradation
    initial_degradation_pct: float = Field(default=2.0, description="Initial degradation in first year (%)")
    annual_degradation_pct: float = Field(default=0.5, description="Annual degradation after first year (%)")

    # CTM (Cell-to-Module) ratio
    ctm_ratio: float = Field(..., gt=0, le=1, description="Cell-to-Module power ratio")

    # Certifications and ratings
    max_system_voltage_v: float = Field(default=1500, description="Maximum system voltage in V")
    max_series_fuse_a: float = Field(default=20, description="Maximum series fuse rating in A")
    operating_temp_range: tuple[float, float] = Field(default=(-40, 85), description="Operating temperature range in °C")
    max_load_pa: float = Field(default=5400, description="Maximum load in Pa")

    @property
    def power_density_w_m2(self) -> float:
        """Calculate power density in W/m²."""
        return self.pmax_stc_w / self.configuration.area_m2

    @property
    def power_at_25_years_w(self) -> float:
        """Calculate power after 25 years of degradation."""
        year_1_power = self.pmax_stc_w * (1 - self.initial_degradation_pct / 100)
        year_25_power = year_1_power * ((1 - self.annual_degradation_pct / 100) ** 24)
        return year_25_power

    @property
    def degradation_25_years_pct(self) -> float:
        """Calculate total degradation over 25 years."""
        return (1 - self.power_at_25_years_w / self.pmax_stc_w) * 100
