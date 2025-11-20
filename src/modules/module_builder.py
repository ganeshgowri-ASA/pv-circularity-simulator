"""
Module Configuration Builder & PAN File Generator

This module provides comprehensive functionality for designing PV modules,
calculating specifications, and generating PVsyst PAN files.

Author: PV Circularity Simulator
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import csv
import math
from io import StringIO

from pydantic import BaseModel, Field, field_validator, computed_field


# ============================================================================
# Enumerations
# ============================================================================

class CellType(str, Enum):
    """Cell technology types"""
    MONO_PERC = "mono_perc"
    MONO_TOPCON = "mono_topcon"
    MONO_HJT = "mono_hjt"
    MONO_IBC = "mono_ibc"
    MULTI_SI = "multi_si"
    PEROVSKITE = "perovskite"
    TANDEM = "tandem"


class LayoutType(str, Enum):
    """Module layout configurations"""
    STANDARD = "standard"
    HALF_CUT = "half_cut"
    QUARTER_CUT = "quarter_cut"
    SHINGLED = "shingled"
    IBC = "ibc"
    BIFACIAL = "bifacial"


class ConnectionType(str, Enum):
    """Cell connection types"""
    BUSBAR = "busbar"
    MULTI_BUSBAR = "multi_busbar"
    SHINGLED = "shingled"
    WIRE = "wire"


class ValidationLevel(str, Enum):
    """Validation severity levels"""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Pydantic Models - Cell Design
# ============================================================================

class CellDesign(BaseModel):
    """
    Photovoltaic cell design specifications

    Attributes:
        cell_type: Type of cell technology
        efficiency: Cell efficiency (0-1)
        area: Cell area in m²
        voltage_oc: Open circuit voltage in V
        current_sc: Short circuit current in A
        voltage_mpp: Voltage at maximum power point in V
        current_mpp: Current at maximum power point in A
        temp_coeff_voc: Temperature coefficient of Voc in %/°C
        temp_coeff_isc: Temperature coefficient of Isc in %/°C
        temp_coeff_pmax: Temperature coefficient of Pmax in %/°C
        series_resistance: Series resistance in Ω
        shunt_resistance: Shunt resistance in Ω
        ideality_factor: Diode ideality factor
        is_bifacial: Whether cell is bifacial
        bifacial_factor: Bifacial factor (rear/front efficiency ratio)
        busbar_count: Number of busbars (0 for IBC)
        thickness: Cell thickness in μm
    """
    cell_type: CellType
    efficiency: float = Field(gt=0, le=1, description="Cell efficiency fraction")
    area: float = Field(gt=0, description="Cell area in m²")
    voltage_oc: float = Field(gt=0, description="Open circuit voltage in V")
    current_sc: float = Field(gt=0, description="Short circuit current in A")
    voltage_mpp: float = Field(gt=0, description="Voltage at MPP in V")
    current_mpp: float = Field(gt=0, description="Current at MPP in A")
    temp_coeff_voc: float = Field(description="Temp coeff Voc in %/°C")
    temp_coeff_isc: float = Field(description="Temp coeff Isc in %/°C")
    temp_coeff_pmax: float = Field(description="Temp coeff Pmax in %/°C")
    series_resistance: float = Field(ge=0, description="Series resistance in Ω")
    shunt_resistance: float = Field(gt=0, description="Shunt resistance in Ω")
    ideality_factor: float = Field(gt=0, le=2, description="Diode ideality factor")
    is_bifacial: bool = False
    bifacial_factor: float = Field(default=0.0, ge=0, le=1)
    busbar_count: int = Field(ge=0, le=16, default=5)
    thickness: float = Field(gt=0, default=180, description="Cell thickness in μm")

    @computed_field
    @property
    def power_max(self) -> float:
        """Maximum power in W"""
        return self.voltage_mpp * self.current_mpp

    @field_validator('bifacial_factor')
    @classmethod
    def validate_bifacial(cls, v: float, info) -> float:
        """Ensure bifacial factor is only set for bifacial cells"""
        if 'is_bifacial' in info.data and not info.data['is_bifacial'] and v > 0:
            raise ValueError("Bifacial factor must be 0 for non-bifacial cells")
        return v


# ============================================================================
# Pydantic Models - Module Configuration
# ============================================================================

class ModuleLayout(BaseModel):
    """
    Module physical layout configuration

    Attributes:
        layout_type: Type of layout configuration
        cells_series: Number of cells in series
        cells_parallel: Number of cells in parallel
        submodules: Number of submodules (for half/quarter-cut)
        bypass_diodes: Number of bypass diodes
        cell_gap: Gap between cells in mm
        overlap: Overlap for shingled cells in mm
        connection_type: Type of cell interconnection
        busbar_count: Number of busbars (for MBB)
    """
    layout_type: LayoutType
    cells_series: int = Field(gt=0, description="Cells in series")
    cells_parallel: int = Field(gt=0, default=1, description="Cells in parallel")
    submodules: int = Field(ge=1, default=1, description="Number of submodules")
    bypass_diodes: int = Field(ge=0, description="Number of bypass diodes")
    cell_gap: float = Field(ge=0, default=2.0, description="Gap between cells in mm")
    overlap: float = Field(ge=0, default=0.0, description="Shingled overlap in mm")
    connection_type: ConnectionType = ConnectionType.MULTI_BUSBAR
    busbar_count: int = Field(ge=0, le=16, default=9)

    @computed_field
    @property
    def total_cells(self) -> int:
        """Total number of cells in module"""
        return self.cells_series * self.cells_parallel

    @field_validator('bypass_diodes')
    @classmethod
    def validate_diodes(cls, v: int, info) -> int:
        """Ensure adequate bypass diodes"""
        if v == 0:
            return 0
        if 'submodules' in info.data and v < info.data['submodules']:
            raise ValueError("Should have at least one bypass diode per submodule")
        return v


class ModuleConfig(BaseModel):
    """
    Complete module configuration

    Attributes:
        name: Module model name
        manufacturer: Manufacturer name
        cell_design: Cell design specifications
        layout: Module layout configuration
        length: Module length in mm
        width: Module width in mm
        thickness: Module thickness in mm
        weight: Module weight in kg
        frame_type: Frame material type
        glass_thickness_front: Front glass thickness in mm
        glass_thickness_rear: Rear glass thickness in mm (0 for monofacial)
        junction_box_type: Junction box type
        cable_length: Cable length in mm
        connector_type: Connector type
        noct: Nominal Operating Cell Temperature in °C
        operating_temp_min: Minimum operating temperature in °C
        operating_temp_max: Maximum operating temperature in °C
        max_system_voltage: Maximum system voltage in V
        max_series_fuse: Maximum series fuse rating in A
        fire_class: Fire safety class
        hail_resistance: Hail resistance diameter in mm
    """
    name: str = Field(min_length=1, description="Module model name")
    manufacturer: str = Field(min_length=1, description="Manufacturer name")
    cell_design: CellDesign
    layout: ModuleLayout

    # Mechanical specifications
    length: float = Field(gt=0, description="Module length in mm")
    width: float = Field(gt=0, description="Module width in mm")
    thickness: float = Field(gt=0, default=35.0, description="Module thickness in mm")
    weight: float = Field(gt=0, description="Module weight in kg")

    # Construction details
    frame_type: str = Field(default="Anodized Aluminum")
    glass_thickness_front: float = Field(gt=0, default=3.2, description="Front glass in mm")
    glass_thickness_rear: float = Field(ge=0, default=0.0, description="Rear glass in mm")
    junction_box_type: str = Field(default="IP68")
    cable_length: float = Field(gt=0, default=1200, description="Cable length in mm")
    connector_type: str = Field(default="MC4")

    # Operating conditions
    noct: float = Field(gt=0, le=100, default=45.0, description="NOCT in °C")
    operating_temp_min: float = Field(default=-40.0, description="Min temp in °C")
    operating_temp_max: float = Field(default=85.0, description="Max temp in °C")
    max_system_voltage: int = Field(gt=0, default=1500, description="Max system voltage in V")
    max_series_fuse: float = Field(gt=0, default=25.0, description="Max fuse in A")

    # Certifications
    fire_class: str = Field(default="Class C")
    hail_resistance: float = Field(ge=0, default=25.0, description="Hail resistance in mm")

    @computed_field
    @property
    def area(self) -> float:
        """Module area in m²"""
        return (self.length * self.width) / 1_000_000

    @computed_field
    @property
    def is_bifacial(self) -> bool:
        """Whether module is bifacial"""
        return self.cell_design.is_bifacial and self.glass_thickness_rear > 0


class ModuleSpecs(BaseModel):
    """
    Calculated module electrical and thermal specifications

    Attributes:
        pmax: Maximum power in W (STC)
        voc: Open circuit voltage in V
        isc: Short circuit current in A
        vmpp: Voltage at maximum power point in V
        impp: Current at maximum power point in A
        efficiency: Module efficiency (0-1)
        temp_coeff_voc: Temperature coefficient of Voc in %/°C
        temp_coeff_isc: Temperature coefficient of Isc in %/°C
        temp_coeff_pmax: Temperature coefficient of Pmax in %/°C
        noct: Nominal Operating Cell Temperature in °C
        temp_coeff_vmpp: Temperature coefficient of Vmpp in %/°C
        temp_coeff_impp: Temperature coefficient of Impp in %/°C
        series_resistance: Module series resistance in Ω
        power_tolerance: Power tolerance in %
        fill_factor: Fill factor
        bifacial_factor: Bifacial factor (0 for monofacial)
        low_irradiance_loss: Performance loss at 200W/m² (%)
        iam_loss_50deg: IAM loss at 50° incidence angle (%)
    """
    pmax: float = Field(gt=0, description="Maximum power in W")
    voc: float = Field(gt=0, description="Open circuit voltage in V")
    isc: float = Field(gt=0, description="Short circuit current in A")
    vmpp: float = Field(gt=0, description="Voltage at MPP in V")
    impp: float = Field(gt=0, description="Current at MPP in A")
    efficiency: float = Field(gt=0, le=1, description="Module efficiency")

    # Temperature coefficients
    temp_coeff_voc: float
    temp_coeff_isc: float
    temp_coeff_pmax: float
    temp_coeff_vmpp: float
    temp_coeff_impp: float

    # Thermal
    noct: float = Field(gt=0, le=100, description="NOCT in °C")

    # Electrical characteristics
    series_resistance: float = Field(ge=0, description="Series resistance in Ω")
    power_tolerance: float = Field(default=3.0, description="Power tolerance in %")
    fill_factor: float = Field(gt=0, le=1, description="Fill factor")

    # Advanced features
    bifacial_factor: float = Field(ge=0, le=1, default=0.0)
    low_irradiance_loss: float = Field(ge=0, le=100, default=1.5, description="Loss at 200W/m²")
    iam_loss_50deg: float = Field(ge=0, le=100, default=3.0, description="IAM loss at 50°")

    # CTM (Cell-to-Module) losses
    ctm_loss_resistance: float = Field(ge=0, le=100, default=2.0, description="Resistance loss %")
    ctm_loss_reflection: float = Field(ge=0, le=100, default=1.5, description="Reflection loss %")
    ctm_loss_mismatch: float = Field(ge=0, le=100, default=0.5, description="Mismatch loss %")
    ctm_loss_inactive: float = Field(ge=0, le=100, default=1.0, description="Inactive area loss %")

    @computed_field
    @property
    def ctm_total_loss(self) -> float:
        """Total CTM loss in %"""
        return (self.ctm_loss_resistance + self.ctm_loss_reflection +
                self.ctm_loss_mismatch + self.ctm_loss_inactive)


# ============================================================================
# Pydantic Models - Validation & Optimization
# ============================================================================

class ValidationIssue(BaseModel):
    """Individual validation issue"""
    level: ValidationLevel
    category: str
    message: str
    field: Optional[str] = None
    value: Optional[Union[float, str, int]] = None
    recommendation: Optional[str] = None


class ValidationReport(BaseModel):
    """
    Module design validation report

    Attributes:
        is_valid: Whether design passes validation
        issues: List of validation issues
        timestamp: Validation timestamp
        design_name: Name of validated design
    """
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    design_name: str

    @computed_field
    @property
    def error_count(self) -> int:
        """Count of errors"""
        return sum(1 for i in self.issues if i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL])

    @computed_field
    @property
    def warning_count(self) -> int:
        """Count of warnings"""
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)


class OptimalLayout(BaseModel):
    """
    Optimized module layout result

    Attributes:
        layout: Optimized layout configuration
        efficiency_gain: Efficiency improvement over baseline (%)
        cost_delta: Cost delta over baseline (%)
        performance_score: Overall performance score (0-100)
        optimization_notes: Notes about optimization choices
    """
    layout: ModuleLayout
    efficiency_gain: float = Field(description="Efficiency improvement %")
    cost_delta: float = Field(description="Cost delta %")
    performance_score: float = Field(ge=0, le=100, description="Performance score")
    optimization_notes: List[str] = Field(default_factory=list)


# ============================================================================
# Module Configuration Builder
# ============================================================================

class ModuleConfigBuilder:
    """
    Comprehensive module configuration builder and analyzer

    This class provides methods to:
    - Create module configurations from cell designs
    - Calculate module specifications
    - Generate PVsyst PAN files
    - Validate module designs
    - Optimize cell layouts
    """

    # Constants for calculations
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
    STANDARD_IRRADIANCE = 1000  # W/m²
    STANDARD_TEMPERATURE = 25   # °C
    AMBIENT_TEMPERATURE = 20    # °C
    WIND_SPEED = 1              # m/s

    def __init__(self):
        """Initialize the module configuration builder"""
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> Dict:
        """Initialize validation rules for module designs"""
        return {
            'voltage_limits': {'min': 0, 'max': 200},
            'current_limits': {'min': 0, 'max': 25},
            'efficiency_limits': {'min': 0.10, 'max': 0.30},
            'noct_limits': {'min': 38, 'max': 55},
            'fill_factor_limits': {'min': 0.70, 'max': 0.85},
        }

    # ========================================================================
    # Core Methods
    # ========================================================================

    def create_module_config(
        self,
        cell_design: CellDesign,
        layout: Dict,
        name: str = "Custom Module",
        manufacturer: str = "Generic Manufacturer",
        **kwargs
    ) -> ModuleConfig:
        """
        Create a complete module configuration from cell design and layout

        Args:
            cell_design: Cell design specifications
            layout: Layout dictionary with configuration
            name: Module model name
            manufacturer: Manufacturer name
            **kwargs: Additional module parameters

        Returns:
            Complete module configuration

        Example:
            >>> cell = CellDesign(
            ...     cell_type=CellType.MONO_PERC,
            ...     efficiency=0.225,
            ...     area=0.0244,  # 156mm x 156mm
            ...     voltage_oc=0.65,
            ...     current_sc=10.2,
            ...     voltage_mpp=0.55,
            ...     current_mpp=9.8,
            ...     temp_coeff_voc=-0.28,
            ...     temp_coeff_isc=0.05,
            ...     temp_coeff_pmax=-0.35,
            ...     series_resistance=0.005,
            ...     shunt_resistance=500,
            ...     ideality_factor=1.2
            ... )
            >>> layout_dict = {
            ...     'layout_type': 'half_cut',
            ...     'cells_series': 120,
            ...     'cells_parallel': 1,
            ...     'submodules': 2,
            ...     'bypass_diodes': 3
            ... }
            >>> config = builder.create_module_config(cell, layout_dict)
        """
        # Parse layout configuration
        module_layout = self._create_layout(layout, cell_design)

        # Calculate module dimensions
        dimensions = self._calculate_dimensions(cell_design, module_layout)

        # Set default values
        defaults = {
            'length': dimensions['length'],
            'width': dimensions['width'],
            'thickness': 35.0,
            'weight': dimensions['weight'],
            'noct': 45.0,
            'glass_thickness_rear': 2.0 if cell_design.is_bifacial else 0.0,
        }
        defaults.update(kwargs)

        # Create module configuration
        module_config = ModuleConfig(
            name=name,
            manufacturer=manufacturer,
            cell_design=cell_design,
            layout=module_layout,
            **defaults
        )

        return module_config

    def calculate_module_specs(self, config: ModuleConfig) -> ModuleSpecs:
        """
        Calculate complete module specifications from configuration

        This method computes all electrical and thermal parameters including:
        - Power, voltage, and current at STC
        - Temperature coefficients
        - Fill factor and efficiency
        - CTM losses
        - Low irradiance and IAM performance

        Args:
            config: Module configuration

        Returns:
            Complete module specifications

        Example:
            >>> specs = builder.calculate_module_specs(config)
            >>> print(f"Module power: {specs.pmax:.1f} W")
            >>> print(f"Efficiency: {specs.efficiency*100:.2f}%")
        """
        cell = config.cell_design
        layout = config.layout

        # Calculate basic electrical parameters
        if layout.layout_type == LayoutType.HALF_CUT:
            # Half-cut: cells in series doubled, voltage doubled
            voc = cell.voltage_oc * layout.cells_series
            isc = cell.current_sc * layout.cells_parallel * 0.5
            vmpp = cell.voltage_mpp * layout.cells_series
            impp = cell.current_mpp * layout.cells_parallel * 0.5
        elif layout.layout_type == LayoutType.QUARTER_CUT:
            # Quarter-cut: similar to half-cut but 4 submodules
            voc = cell.voltage_oc * layout.cells_series
            isc = cell.current_sc * layout.cells_parallel * 0.25
            vmpp = cell.voltage_mpp * layout.cells_series
            impp = cell.current_mpp * layout.cells_parallel * 0.25
        elif layout.layout_type == LayoutType.SHINGLED:
            # Shingled: reduced resistance, slightly higher current
            voc = cell.voltage_oc * layout.cells_series * 0.98  # Slight reduction
            isc = cell.current_sc * layout.cells_parallel * 1.02  # Slight gain
            vmpp = cell.voltage_mpp * layout.cells_series * 0.98
            impp = cell.current_mpp * layout.cells_parallel * 1.02
        else:
            # Standard configuration
            voc = cell.voltage_oc * layout.cells_series
            isc = cell.current_sc * layout.cells_parallel
            vmpp = cell.voltage_mpp * layout.cells_series
            impp = cell.current_mpp * layout.cells_parallel

        # Calculate power
        pmax = vmpp * impp

        # Calculate CTM losses
        ctm_losses = self._calculate_ctm_losses(config)

        # Apply CTM losses to power
        ctm_factor = 1 - (ctm_losses['total'] / 100)
        pmax_with_ctm = pmax * ctm_factor

        # Adjust Vmpp and Impp proportionally
        vmpp_final = vmpp * math.sqrt(ctm_factor)
        impp_final = impp * math.sqrt(ctm_factor)

        # Calculate fill factor
        fill_factor = pmax_with_ctm / (voc * isc) if (voc * isc) > 0 else 0

        # Calculate efficiency
        efficiency = pmax_with_ctm / (config.area * 1000)  # At 1000 W/m²

        # Temperature coefficients (module level)
        temp_coeff_voc = cell.temp_coeff_voc
        temp_coeff_isc = cell.temp_coeff_isc
        temp_coeff_pmax = cell.temp_coeff_pmax

        # Vmpp and Impp temperature coefficients
        # Vmpp has similar behavior to Voc, Impp to Isc
        temp_coeff_vmpp = temp_coeff_voc * 0.95  # Slightly less negative
        temp_coeff_impp = temp_coeff_isc * 0.95

        # Calculate module series resistance
        series_resistance = self._calculate_module_resistance(config)

        # NOCT calculation
        noct = self._calculate_noct(config)

        # Low irradiance and IAM losses
        low_irr_loss = self._calculate_low_irradiance_loss(config)
        iam_loss = self._calculate_iam_loss(config)

        # Bifacial factor
        bifacial_factor = cell.bifacial_factor if config.is_bifacial else 0.0

        return ModuleSpecs(
            pmax=pmax_with_ctm,
            voc=voc,
            isc=isc,
            vmpp=vmpp_final,
            impp=impp_final,
            efficiency=efficiency,
            temp_coeff_voc=temp_coeff_voc,
            temp_coeff_isc=temp_coeff_isc,
            temp_coeff_pmax=temp_coeff_pmax,
            temp_coeff_vmpp=temp_coeff_vmpp,
            temp_coeff_impp=temp_coeff_impp,
            noct=noct,
            series_resistance=series_resistance,
            fill_factor=fill_factor,
            bifacial_factor=bifacial_factor,
            low_irradiance_loss=low_irr_loss,
            iam_loss_50deg=iam_loss,
            ctm_loss_resistance=ctm_losses['resistance'],
            ctm_loss_reflection=ctm_losses['reflection'],
            ctm_loss_mismatch=ctm_losses['mismatch'],
            ctm_loss_inactive=ctm_losses['inactive'],
        )

    def generate_pvsyst_pan_file(self, module: ModuleConfig) -> str:
        """
        Generate PVsyst PAN file format for module

        Generates a complete PAN file with all electrical, mechanical,
        and thermal specifications compatible with PVsyst 7.x

        Args:
            module: Module configuration

        Returns:
            PAN file content as string

        Example:
            >>> pan_content = builder.generate_pvsyst_pan_file(config)
            >>> with open('module.PAN', 'w') as f:
            ...     f.write(pan_content)
        """
        specs = self.calculate_module_specs(module)

        # Generate PAN file header
        pan_lines = [
            "PVObject_=pvModule",
            f"  Version=7.3.1",
            f"  Flags=$0043",
            "",
            f"  PVObject_Commercial=pvCommercial",
            f"    Comment={module.name}",
            f"    Flags=$0041",
            f"    Manufacturer={module.manufacturer}",
            f"    Model={module.name}",
            f"    DataSource=Calculated",
            f"    YearBeg={datetime.now().year}",
            f"    Width={module.width / 1000:.3f}",
            f"    Height={module.length / 1000:.3f}",
            f"    Depth={module.thickness / 1000:.3f}",
            f"    Weight={module.weight:.1f}",
            f"    NPieces={module.layout.total_cells}",
            f"    PriceDate={datetime.now().strftime('%d/%m/%y')}",
            f"  End of PVObject pvCommercial",
            "",
        ]

        # Technology parameters
        tech_type = self._get_pvsyst_technology(module.cell_design.cell_type)
        pan_lines.extend([
            f"  Technol={tech_type}",
            f"  NCelS={module.layout.cells_series}",
            f"  NCelP={module.layout.cells_parallel}",
            f"  NDiode={module.layout.bypass_diodes}",
            "",
        ])

        # Bifacial parameters
        if module.is_bifacial:
            pan_lines.extend([
                f"  BifacialityFactor={specs.bifacial_factor:.3f}",
                f"  Bifacial=Yes",
                "",
            ])

        # STC parameters
        pan_lines.extend([
            f"  GRef=1000",
            f"  TRef=25.0",
            f"  PNom={specs.pmax:.1f}",
            f"  PNomTolLow={specs.power_tolerance:.1f}",
            f"  PNomTolUp={specs.power_tolerance:.1f}",
            f"  Isc={specs.isc:.3f}",
            f"  Voc={specs.voc:.2f}",
            f"  Imp={specs.impp:.3f}",
            f"  Vmp={specs.vmpp:.2f}",
            "",
        ])

        # Module efficiency
        pan_lines.extend([
            f"  muISC={specs.temp_coeff_isc / 100:.6f}",
            f"  muVocSpec={specs.temp_coeff_voc:.3f}",
            f"  muPmpReq={specs.temp_coeff_pmax:.3f}",
            "",
        ])

        # Operating conditions
        pan_lines.extend([
            f"  TModule_NOCT={specs.noct:.1f}",
            f"  T_NOCT=45.0",
            f"  G_NOCT=800",
            f"  muPmpReq_NOCT={specs.temp_coeff_pmax:.3f}",
            "",
        ])

        # Series resistance and model parameters
        gamma = self._calculate_gamma(specs)
        pan_lines.extend([
            f"  RShunt={module.cell_design.shunt_resistance * module.layout.cells_series / module.layout.cells_parallel:.1f}",
            f"  Rp_0={module.cell_design.shunt_resistance * module.layout.cells_series / module.layout.cells_parallel * 1.5:.1f}",
            f"  Rp_Exp=5.50",
            f"  RSerie={specs.series_resistance:.4f}",
            f"  Gamma={gamma:.3f}",
            f"  muGamma={-0.0003:.6f}",
            "",
        ])

        # IAM (Incidence Angle Modifier) profile
        iam_profile = self._generate_iam_profile(module)
        pan_lines.extend([
            f"  PVObject_IAM=pvIAM",
            f"    Flags=$00",
            f"    IAMMode=UserProfile",
            f"    IAMProfile=TCubicProfile",
        ])
        for angle, modifier in iam_profile.items():
            pan_lines.append(f"      {angle:2d}  {modifier:.4f}")
        pan_lines.extend([
            f"  End of PVObject pvIAM",
            "",
        ])

        # Low irradiance behavior
        pan_lines.extend([
            f"  OperPoints=0",
            "",
        ])

        # End of module object
        pan_lines.extend([
            f"End of PVObject pvModule",
        ])

        return "\n".join(pan_lines)

    def validate_module_design(self, config: ModuleConfig) -> ValidationReport:
        """
        Validate module design against industry standards and best practices

        Performs comprehensive validation including:
        - Electrical parameter ranges
        - Thermal characteristics
        - Mechanical specifications
        - Layout configuration
        - Safety requirements

        Args:
            config: Module configuration to validate

        Returns:
            Validation report with issues and recommendations

        Example:
            >>> report = builder.validate_module_design(config)
            >>> if not report.is_valid:
            ...     for issue in report.issues:
            ...         print(f"{issue.level}: {issue.message}")
        """
        issues: List[ValidationIssue] = []
        specs = self.calculate_module_specs(config)

        # Validate electrical parameters
        if specs.voc > self.validation_rules['voltage_limits']['max']:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="Electrical",
                message=f"Open circuit voltage {specs.voc:.1f}V exceeds typical limit",
                field="voc",
                value=specs.voc,
                recommendation="Reduce cells in series or use lower voltage cells"
            ))

        if specs.isc > self.validation_rules['current_limits']['max']:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Electrical",
                message=f"Short circuit current {specs.isc:.1f}A is high",
                field="isc",
                value=specs.isc,
                recommendation="Verify cable and connector ratings"
            ))

        # Validate efficiency
        if specs.efficiency < self.validation_rules['efficiency_limits']['min']:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Performance",
                message=f"Module efficiency {specs.efficiency*100:.1f}% is low",
                field="efficiency",
                value=specs.efficiency,
                recommendation="Consider higher efficiency cells or optimize layout"
            ))

        if specs.efficiency > self.validation_rules['efficiency_limits']['max']:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Performance",
                message=f"Module efficiency {specs.efficiency*100:.1f}% seems unusually high",
                field="efficiency",
                value=specs.efficiency,
                recommendation="Verify cell specifications and CTM losses"
            ))

        # Validate fill factor
        if specs.fill_factor < self.validation_rules['fill_factor_limits']['min']:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="Performance",
                message=f"Fill factor {specs.fill_factor:.3f} is too low",
                field="fill_factor",
                value=specs.fill_factor,
                recommendation="Check series resistance and cell quality"
            ))

        # Validate NOCT
        if specs.noct < self.validation_rules['noct_limits']['min']:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Thermal",
                message=f"NOCT {specs.noct:.1f}°C is unusually low",
                field="noct",
                value=specs.noct,
                recommendation="Verify thermal model assumptions"
            ))

        if specs.noct > self.validation_rules['noct_limits']['max']:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Thermal",
                message=f"NOCT {specs.noct:.1f}°C is high, expect temperature losses",
                field="noct",
                value=specs.noct,
                recommendation="Consider improved thermal design"
            ))

        # Validate bypass diodes
        cells_per_diode = config.layout.total_cells / max(config.layout.bypass_diodes, 1)
        if cells_per_diode > 24:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Design",
                message=f"Too many cells ({cells_per_diode:.0f}) per bypass diode",
                field="bypass_diodes",
                value=config.layout.bypass_diodes,
                recommendation="Add more bypass diodes for better shading tolerance"
            ))

        # Validate mechanical specs
        if config.weight / config.area > 15:  # kg/m²
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Mechanical",
                message=f"Module is heavy ({config.weight/config.area:.1f} kg/m²)",
                field="weight",
                value=config.weight,
                recommendation="Consider structural loading requirements"
            ))

        # Validate bifacial configuration
        if config.cell_design.is_bifacial and config.glass_thickness_rear == 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="Design",
                message="Bifacial cells require rear glass",
                field="glass_thickness_rear",
                value=0.0,
                recommendation="Add rear glass (typically 2mm)"
            ))

        # Validate system voltage
        safety_margin = 1.15  # 15% safety margin for cold weather
        max_cold_voc = specs.voc * safety_margin
        if max_cold_voc * 30 > config.max_system_voltage:  # Assume up to 30 modules in series
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="System",
                message=f"Limited string size due to system voltage limit",
                field="max_system_voltage",
                value=config.max_system_voltage,
                recommendation=f"Max ~{int(config.max_system_voltage/(max_cold_voc))} modules per string"
            ))

        # Determine overall validation status
        has_critical_errors = any(
            i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for i in issues
        )

        if not issues:
            issues.append(ValidationIssue(
                level=ValidationLevel.PASS,
                category="Overall",
                message="Module design passes all validation checks"
            ))

        return ValidationReport(
            is_valid=not has_critical_errors,
            issues=issues,
            design_name=config.name
        )

    def optimize_cell_layout(
        self,
        cell_design: CellDesign,
        constraints: Dict
    ) -> OptimalLayout:
        """
        Optimize cell layout for given constraints

        Finds optimal layout configuration considering:
        - Target power output
        - Voltage and current requirements
        - Cost constraints
        - Physical size limitations
        - Performance objectives

        Args:
            cell_design: Cell design to use
            constraints: Dictionary of constraints and objectives
                - target_power: Target module power in W
                - max_voltage: Maximum voltage limit in V
                - max_current: Maximum current limit in A
                - max_area: Maximum module area in m²
                - optimize_for: 'efficiency' | 'cost' | 'performance'
                - allow_half_cut: Whether to allow half-cut configuration
                - allow_shingled: Whether to allow shingled configuration

        Returns:
            Optimal layout configuration with performance metrics

        Example:
            >>> constraints = {
            ...     'target_power': 450,
            ...     'max_voltage': 50,
            ...     'optimize_for': 'efficiency',
            ...     'allow_half_cut': True
            ... }
            >>> optimal = builder.optimize_cell_layout(cell_design, constraints)
            >>> print(f"Optimal: {optimal.layout.layout_type}, "
            ...       f"{optimal.layout.total_cells} cells, "
            ...       f"{optimal.efficiency_gain:.1f}% gain")
        """
        target_power = constraints.get('target_power', 400)
        max_voltage = constraints.get('max_voltage', 50)
        max_current = constraints.get('max_current', 15)
        max_area = constraints.get('max_area', 2.5)
        optimize_for = constraints.get('optimize_for', 'performance')
        allow_half_cut = constraints.get('allow_half_cut', True)
        allow_shingled = constraints.get('allow_shingled', False)

        # Calculate required cells for target power
        cell_power = cell_design.power_max
        cells_needed = int(math.ceil(target_power / cell_power * 1.1))  # Add 10% margin for CTM

        # Standard cell counts
        standard_counts = [60, 72, 96, 108, 120, 132, 144]
        selected_count = min([c for c in standard_counts if c >= cells_needed], default=144)

        # Evaluate different layout types
        candidates = []

        # Standard layout
        standard_layout = self._evaluate_layout_option(
            cell_design=cell_design,
            layout_type=LayoutType.STANDARD,
            cells_series=selected_count,
            cells_parallel=1,
            submodules=1,
            bypass_diodes=3,
            max_voltage=max_voltage,
            max_current=max_current,
            max_area=max_area
        )
        if standard_layout:
            candidates.append(('standard', standard_layout))

        # Half-cut layout
        if allow_half_cut and selected_count >= 60:
            half_cut_layout = self._evaluate_layout_option(
                cell_design=cell_design,
                layout_type=LayoutType.HALF_CUT,
                cells_series=selected_count,
                cells_parallel=2,
                submodules=2,
                bypass_diodes=3,
                max_voltage=max_voltage,
                max_current=max_current,
                max_area=max_area
            )
            if half_cut_layout:
                candidates.append(('half_cut', half_cut_layout))

        # Shingled layout
        if allow_shingled:
            shingled_layout = self._evaluate_layout_option(
                cell_design=cell_design,
                layout_type=LayoutType.SHINGLED,
                cells_series=selected_count,
                cells_parallel=1,
                submodules=1,
                bypass_diodes=3,
                max_voltage=max_voltage,
                max_current=max_current,
                max_area=max_area
            )
            if shingled_layout:
                candidates.append(('shingled', shingled_layout))

        # Bifacial if applicable
        if cell_design.is_bifacial:
            bifacial_layout = self._evaluate_layout_option(
                cell_design=cell_design,
                layout_type=LayoutType.BIFACIAL,
                cells_series=selected_count,
                cells_parallel=1,
                submodules=1,
                bypass_diodes=3,
                max_voltage=max_voltage,
                max_current=max_current,
                max_area=max_area
            )
            if bifacial_layout:
                candidates.append(('bifacial', bifacial_layout))

        # Score candidates based on optimization objective
        best_layout = None
        best_score = -float('inf')
        best_name = None

        for name, layout_data in candidates:
            score = self._score_layout(layout_data, optimize_for, target_power)
            if score > best_score:
                best_score = score
                best_layout = layout_data
                best_name = name

        if not best_layout:
            # Fallback to standard layout
            best_layout = {
                'layout': ModuleLayout(
                    layout_type=LayoutType.STANDARD,
                    cells_series=selected_count,
                    cells_parallel=1,
                    submodules=1,
                    bypass_diodes=3
                ),
                'power': cell_power * selected_count * 0.95,
                'efficiency': cell_design.efficiency * 0.95,
                'cost_factor': 1.0
            }
            best_name = 'standard'
            # Calculate a baseline score
            best_score = self._score_layout(best_layout, optimize_for, target_power)

        # Calculate metrics relative to standard
        baseline_efficiency = cell_design.efficiency * 0.95  # Standard CTM
        efficiency_gain = (best_layout['efficiency'] / baseline_efficiency - 1) * 100
        cost_delta = (best_layout['cost_factor'] - 1.0) * 100

        # Generate optimization notes
        notes = [
            f"Selected {best_name} layout with {selected_count} cells",
            f"Achieves {best_layout['power']:.0f}W (target: {target_power}W)",
            f"Module efficiency: {best_layout['efficiency']*100:.2f}%"
        ]

        if best_name == 'half_cut':
            notes.append("Half-cut reduces resistive losses by ~1-2%")
        elif best_name == 'shingled':
            notes.append("Shingled eliminates cell gaps, improving efficiency")
        elif best_name == 'bifacial':
            notes.append(f"Bifacial with {cell_design.bifacial_factor*100:.0f}% rear gain")

        return OptimalLayout(
            layout=best_layout['layout'],
            efficiency_gain=efficiency_gain,
            cost_delta=cost_delta,
            performance_score=min(best_score, 100),
            optimization_notes=notes
        )

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_to_json(
        self,
        config: ModuleConfig,
        include_specs: bool = True,
        filepath: Optional[Path] = None
    ) -> str:
        """
        Export module configuration to JSON format

        Args:
            config: Module configuration
            include_specs: Whether to include calculated specifications
            filepath: Optional file path to save JSON

        Returns:
            JSON string
        """
        data = {
            'module_config': config.model_dump(mode='json'),
        }

        if include_specs:
            specs = self.calculate_module_specs(config)
            data['module_specs'] = specs.model_dump(mode='json')

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            filepath.write_text(json_str)

        return json_str

    def export_to_csv(
        self,
        configs: List[ModuleConfig],
        filepath: Optional[Path] = None
    ) -> str:
        """
        Export multiple module configurations to CSV format

        Args:
            configs: List of module configurations
            filepath: Optional file path to save CSV

        Returns:
            CSV string
        """
        output = StringIO()

        # Define CSV columns
        fieldnames = [
            'Name', 'Manufacturer', 'Cell Type', 'Layout Type',
            'Total Cells', 'Pmax (W)', 'Voc (V)', 'Isc (A)',
            'Vmpp (V)', 'Impp (A)', 'Efficiency (%)', 'Length (mm)',
            'Width (mm)', 'Weight (kg)', 'Area (m²)', 'NOCT (°C)',
            'Temp Coeff Pmax (%/°C)', 'Bifacial', 'Bifacial Factor'
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for config in configs:
            specs = self.calculate_module_specs(config)

            row = {
                'Name': config.name,
                'Manufacturer': config.manufacturer,
                'Cell Type': config.cell_design.cell_type.value,
                'Layout Type': config.layout.layout_type.value,
                'Total Cells': config.layout.total_cells,
                'Pmax (W)': f"{specs.pmax:.1f}",
                'Voc (V)': f"{specs.voc:.2f}",
                'Isc (A)': f"{specs.isc:.2f}",
                'Vmpp (V)': f"{specs.vmpp:.2f}",
                'Impp (A)': f"{specs.impp:.2f}",
                'Efficiency (%)': f"{specs.efficiency*100:.2f}",
                'Length (mm)': f"{config.length:.0f}",
                'Width (mm)': f"{config.width:.0f}",
                'Weight (kg)': f"{config.weight:.1f}",
                'Area (m²)': f"{config.area:.3f}",
                'NOCT (°C)': f"{specs.noct:.1f}",
                'Temp Coeff Pmax (%/°C)': f"{specs.temp_coeff_pmax:.3f}",
                'Bifacial': 'Yes' if config.is_bifacial else 'No',
                'Bifacial Factor': f"{specs.bifacial_factor:.2f}" if config.is_bifacial else 'N/A'
            }

            writer.writerow(row)

        csv_str = output.getvalue()

        if filepath:
            filepath.write_text(csv_str)

        return csv_str

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_layout(self, layout_dict: Dict, cell_design: CellDesign) -> ModuleLayout:
        """Create ModuleLayout from dictionary"""
        layout_type = LayoutType(layout_dict.get('layout_type', 'standard'))

        # Set defaults based on layout type
        if layout_type == LayoutType.HALF_CUT:
            defaults = {
                'cells_parallel': 2,
                'submodules': 2,
                'bypass_diodes': 3,
                'connection_type': ConnectionType.MULTI_BUSBAR,
                'busbar_count': cell_design.busbar_count
            }
        elif layout_type == LayoutType.QUARTER_CUT:
            defaults = {
                'cells_parallel': 4,
                'submodules': 4,
                'bypass_diodes': 4,
                'connection_type': ConnectionType.MULTI_BUSBAR,
                'busbar_count': cell_design.busbar_count
            }
        elif layout_type == LayoutType.SHINGLED:
            defaults = {
                'cells_parallel': 1,
                'submodules': 1,
                'bypass_diodes': 3,
                'cell_gap': 0.0,
                'overlap': 2.0,
                'connection_type': ConnectionType.SHINGLED,
                'busbar_count': 0
            }
        elif layout_type == LayoutType.IBC:
            defaults = {
                'cells_parallel': 1,
                'submodules': 1,
                'bypass_diodes': 3,
                'connection_type': ConnectionType.WIRE,
                'busbar_count': 0
            }
        else:
            defaults = {
                'cells_parallel': 1,
                'submodules': 1,
                'bypass_diodes': 3,
                'connection_type': ConnectionType.MULTI_BUSBAR,
                'busbar_count': cell_design.busbar_count
            }

        # Merge with provided values
        layout_params = {**defaults, **layout_dict}
        layout_params['layout_type'] = layout_type

        return ModuleLayout(**layout_params)

    def _calculate_dimensions(
        self,
        cell_design: CellDesign,
        layout: ModuleLayout
    ) -> Dict[str, float]:
        """Calculate module physical dimensions"""
        # Estimate cell dimension (assuming square cells)
        cell_size = math.sqrt(cell_design.area) * 1000  # mm

        # Common configurations
        if layout.total_cells == 60:
            rows, cols = 10, 6
        elif layout.total_cells == 72:
            rows, cols = 12, 6
        elif layout.total_cells == 96:
            rows, cols = 12, 8
        elif layout.total_cells == 120:
            rows, cols = 10, 12
        elif layout.total_cells == 132:
            rows, cols = 11, 12
        elif layout.total_cells == 144:
            rows, cols = 12, 12
        else:
            # Estimate reasonable configuration
            cols = min(12, layout.total_cells // 6)
            rows = layout.total_cells // cols

        # Account for gaps and overlaps
        if layout.layout_type == LayoutType.SHINGLED:
            effective_gap = -layout.overlap  # Negative for overlap
        else:
            effective_gap = layout.cell_gap

        # Calculate dimensions
        length = rows * cell_size + (rows - 1) * effective_gap + 30  # 30mm frame margin
        width = cols * cell_size + (cols - 1) * effective_gap + 30

        # Estimate weight (glass + cells + frame + encapsulant)
        area_m2 = (length * width) / 1_000_000
        weight = area_m2 * 11.5  # Typical ~11.5 kg/m²

        return {
            'length': length,
            'width': width,
            'weight': weight
        }

    def _calculate_ctm_losses(self, config: ModuleConfig) -> Dict[str, float]:
        """Calculate Cell-to-Module losses"""
        layout = config.layout
        cell = config.cell_design

        # Resistance losses
        if layout.layout_type == LayoutType.HALF_CUT:
            # Half-cut reduces resistive losses
            resistance_loss = 1.5
        elif layout.layout_type == LayoutType.SHINGLED:
            # Shingled has minimal resistive losses
            resistance_loss = 0.8
        elif layout.connection_type == ConnectionType.MULTI_BUSBAR:
            # MBB reduces resistance
            resistance_loss = 1.8
        else:
            # Standard busbar
            resistance_loss = 2.5

        # Reflection losses (glass, encapsulant)
        if config.glass_thickness_front > 0:
            reflection_loss = 1.5  # AR coating helps
        else:
            reflection_loss = 3.0

        # Mismatch losses
        if layout.layout_type == LayoutType.SHINGLED:
            mismatch_loss = 0.3  # Very low for shingled
        elif layout.cells_parallel > 1:
            mismatch_loss = 0.8  # Higher for parallel strings
        else:
            mismatch_loss = 0.5

        # Inactive area losses
        if layout.layout_type == LayoutType.SHINGLED:
            inactive_loss = 0.5  # Minimal gaps
        elif layout.cell_gap < 1.0:
            inactive_loss = 0.8
        else:
            inactive_loss = 1.2

        total_loss = resistance_loss + reflection_loss + mismatch_loss + inactive_loss

        return {
            'resistance': resistance_loss,
            'reflection': reflection_loss,
            'mismatch': mismatch_loss,
            'inactive': inactive_loss,
            'total': total_loss
        }

    def _calculate_module_resistance(self, config: ModuleConfig) -> float:
        """Calculate module series resistance"""
        cell_resistance = config.cell_design.series_resistance

        # Scale by series/parallel configuration
        module_resistance = (
            cell_resistance * config.layout.cells_series / config.layout.cells_parallel
        )

        # Add interconnect resistance
        if config.layout.connection_type == ConnectionType.MULTI_BUSBAR:
            interconnect_r = 0.002 * config.layout.cells_series
        elif config.layout.connection_type == ConnectionType.SHINGLED:
            interconnect_r = 0.001 * config.layout.cells_series  # Very low
        else:
            interconnect_r = 0.003 * config.layout.cells_series

        return module_resistance + interconnect_r

    def _calculate_noct(self, config: ModuleConfig) -> float:
        """Calculate Nominal Operating Cell Temperature"""
        # Base NOCT depends on construction
        if config.is_bifacial:
            base_noct = 43.0  # Better cooling for bifacial
        elif config.glass_thickness_rear > 0:
            base_noct = 44.0  # Glass-glass
        else:
            base_noct = 45.0  # Glass-backsheet

        # Adjust for frame and thickness
        if config.thickness > 40:
            base_noct += 1.0  # Thicker modules run hotter

        return base_noct

    def _calculate_low_irradiance_loss(self, config: ModuleConfig) -> float:
        """Calculate performance loss at low irradiance (200 W/m²)"""
        # Better cells perform better at low light
        if config.cell_design.cell_type in [CellType.MONO_HJT, CellType.MONO_IBC]:
            return 0.8  # Excellent low-light performance
        elif config.cell_design.cell_type in [CellType.MONO_PERC, CellType.MONO_TOPCON]:
            return 1.2
        else:
            return 2.0

    def _calculate_iam_loss(self, config: ModuleConfig) -> float:
        """Calculate IAM loss at 50° incidence angle"""
        # Glass type affects IAM
        if config.glass_thickness_front > 0:
            # AR coating assumed
            return 3.0
        else:
            return 5.0

    def _get_pvsyst_technology(self, cell_type: CellType) -> str:
        """Map cell type to PVsyst technology string"""
        mapping = {
            CellType.MONO_PERC: "mtSiMono",
            CellType.MONO_TOPCON: "mtSiMono",
            CellType.MONO_HJT: "mtSiMono",
            CellType.MONO_IBC: "mtSiMono",
            CellType.MULTI_SI: "mtSiPoly",
            CellType.PEROVSKITE: "mtThinFilm",
            CellType.TANDEM: "mtThinFilm"
        }
        return mapping.get(cell_type, "mtSiMono")

    def _calculate_gamma(self, specs: ModuleSpecs) -> float:
        """Calculate gamma parameter for PVsyst model"""
        # Gamma relates to the temperature dependence of Imp
        # Typically ranges from 0.9 to 1.1
        return 1.0 + (specs.temp_coeff_impp / 100) * 10

    def _generate_iam_profile(self, module: ModuleConfig) -> Dict[int, float]:
        """Generate IAM (Incidence Angle Modifier) profile"""
        # Standard IAM profile for glass modules
        # Based on Sandia model and typical measurements

        if module.glass_thickness_front > 0:
            # Glass module with AR coating
            profile = {
                0: 1.0000,
                10: 1.0000,
                20: 0.9995,
                30: 0.9985,
                40: 0.9960,
                50: 0.9900,
                60: 0.9750,
                70: 0.9350,
                75: 0.8950,
                80: 0.8200,
                85: 0.6800,
                90: 0.0000
            }
        else:
            # Without AR coating
            profile = {
                0: 1.0000,
                10: 0.9995,
                20: 0.9980,
                30: 0.9950,
                40: 0.9880,
                50: 0.9750,
                60: 0.9500,
                70: 0.9000,
                75: 0.8500,
                80: 0.7500,
                85: 0.6000,
                90: 0.0000
            }

        return profile

    def _evaluate_layout_option(
        self,
        cell_design: CellDesign,
        layout_type: LayoutType,
        cells_series: int,
        cells_parallel: int,
        submodules: int,
        bypass_diodes: int,
        max_voltage: float,
        max_current: float,
        max_area: float
    ) -> Optional[Dict]:
        """Evaluate a layout option against constraints"""
        try:
            # Create layout
            layout = ModuleLayout(
                layout_type=layout_type,
                cells_series=cells_series,
                cells_parallel=cells_parallel,
                submodules=submodules,
                bypass_diodes=bypass_diodes
            )

            # Create temporary config
            dimensions = self._calculate_dimensions(cell_design, layout)

            temp_config = ModuleConfig(
                name="temp",
                manufacturer="temp",
                cell_design=cell_design,
                layout=layout,
                length=dimensions['length'],
                width=dimensions['width'],
                weight=dimensions['weight']
            )

            # Calculate specs
            specs = self.calculate_module_specs(temp_config)

            # Check constraints
            if specs.voc > max_voltage:
                return None
            if specs.isc > max_current:
                return None
            if temp_config.area > max_area:
                return None

            # Calculate cost factor
            cost_factor = 1.0
            if layout_type == LayoutType.HALF_CUT:
                cost_factor = 1.02  # Slightly more expensive
            elif layout_type == LayoutType.SHINGLED:
                cost_factor = 1.10  # More expensive manufacturing
            elif layout_type == LayoutType.BIFACIAL:
                cost_factor = 1.08  # Bifacial glass premium

            return {
                'layout': layout,
                'power': specs.pmax,
                'efficiency': specs.efficiency,
                'voltage': specs.voc,
                'current': specs.isc,
                'area': temp_config.area,
                'cost_factor': cost_factor
            }

        except Exception:
            return None

    def _score_layout(
        self,
        layout_data: Dict,
        optimize_for: str,
        target_power: float
    ) -> float:
        """Score a layout option based on optimization objective"""
        if optimize_for == 'efficiency':
            # Maximize efficiency
            score = layout_data['efficiency'] * 100
        elif optimize_for == 'cost':
            # Minimize cost per watt
            cost_per_watt = layout_data['cost_factor'] / layout_data['power']
            score = 100 / (cost_per_watt * 1000)  # Invert so lower cost = higher score
        else:  # 'performance'
            # Balance power, efficiency, and cost
            power_score = min(layout_data['power'] / target_power * 50, 50)
            efficiency_score = layout_data['efficiency'] * 100 * 0.3
            cost_score = (2.0 - layout_data['cost_factor']) * 10
            score = power_score + efficiency_score + cost_score

        return score


# ============================================================================
# Convenience Functions
# ============================================================================

def create_standard_module(
    power_class: int = 450,
    cell_type: CellType = CellType.MONO_PERC,
    layout_type: LayoutType = LayoutType.HALF_CUT,
    manufacturer: str = "Generic Solar"
) -> ModuleConfig:
    """
    Create a standard module configuration for common power classes

    Args:
        power_class: Target power in W (300, 400, 450, 500, 550, 600)
        cell_type: Type of cell technology
        layout_type: Type of layout configuration
        manufacturer: Manufacturer name

    Returns:
        Module configuration

    Example:
        >>> module = create_standard_module(power_class=450, layout_type=LayoutType.HALF_CUT)
        >>> builder = ModuleConfigBuilder()
        >>> specs = builder.calculate_module_specs(module)
        >>> print(f"Created {specs.pmax:.0f}W module")
    """
    # Standard cell specs by type
    cell_specs = {
        CellType.MONO_PERC: {
            'efficiency': 0.225,
            'voltage_oc': 0.68,
            'current_sc': 10.3,
            'voltage_mpp': 0.58,
            'current_mpp': 9.8,
            'temp_coeff_pmax': -0.35,
        },
        CellType.MONO_TOPCON: {
            'efficiency': 0.245,
            'voltage_oc': 0.70,
            'current_sc': 10.5,
            'voltage_mpp': 0.60,
            'current_mpp': 10.0,
            'temp_coeff_pmax': -0.30,
        },
        CellType.MONO_HJT: {
            'efficiency': 0.255,
            'voltage_oc': 0.74,
            'current_sc': 10.2,
            'voltage_mpp': 0.64,
            'current_mpp': 9.8,
            'temp_coeff_pmax': -0.26,
        },
    }

    specs = cell_specs.get(cell_type, cell_specs[CellType.MONO_PERC])

    # Create cell design
    cell = CellDesign(
        cell_type=cell_type,
        efficiency=specs['efficiency'],
        area=0.0244,  # 156mm x 156mm M6 cell
        voltage_oc=specs['voltage_oc'],
        current_sc=specs['current_sc'],
        voltage_mpp=specs['voltage_mpp'],
        current_mpp=specs['current_mpp'],
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        temp_coeff_pmax=specs['temp_coeff_pmax'],
        series_resistance=0.005,
        shunt_resistance=500,
        ideality_factor=1.2,
        busbar_count=9
    )

    # Determine cell count for power class
    cell_power = specs['voltage_mpp'] * specs['current_mpp']
    cells_needed = int(power_class / cell_power / 0.95)  # Account for CTM

    # Select standard cell count
    standard_counts = [72, 120, 132, 144]
    cell_count = min([c for c in standard_counts if c >= cells_needed], default=144)

    # Create layout
    if layout_type == LayoutType.HALF_CUT:
        layout_dict = {
            'layout_type': layout_type,
            'cells_series': cell_count,
            'cells_parallel': 2,
            'submodules': 2,
            'bypass_diodes': 3
        }
    else:
        layout_dict = {
            'layout_type': layout_type,
            'cells_series': cell_count,
            'cells_parallel': 1,
            'submodules': 1,
            'bypass_diodes': 3
        }

    # Build module
    builder = ModuleConfigBuilder()
    module = builder.create_module_config(
        cell_design=cell,
        layout=layout_dict,
        name=f"{manufacturer} {power_class}W",
        manufacturer=manufacturer
    )

    return module


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Enums
    'CellType',
    'LayoutType',
    'ConnectionType',
    'ValidationLevel',

    # Models
    'CellDesign',
    'ModuleLayout',
    'ModuleConfig',
    'ModuleSpecs',
    'ValidationIssue',
    'ValidationReport',
    'OptimalLayout',

    # Main class
    'ModuleConfigBuilder',

    # Convenience functions
    'create_standard_module',
]
