"""Data models for PV system design using Pydantic."""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class MountingType(str, Enum):
    """Types of PV system mounting configurations."""

    GROUND_FIXED = "ground_fixed"
    GROUND_SINGLE_AXIS = "ground_single_axis"
    GROUND_DUAL_AXIS = "ground_dual_axis"
    ROOFTOP_FLAT = "rooftop_flat"
    ROOFTOP_SLOPED = "rooftop_sloped"
    CARPORT = "carport"
    CANOPY = "canopy"
    FLOATING = "floating"
    AGRIVOLTAIC = "agrivoltaic"
    BIPV_FACADE = "bipv_facade"
    BIPV_ROOF = "bipv_roof"


class InverterType(str, Enum):
    """Types of inverter configurations."""

    CENTRAL = "central"
    STRING = "string"
    MICRO = "micro"
    POWER_OPTIMIZER = "power_optimizer"


class SystemType(str, Enum):
    """Types of PV systems by scale."""

    UTILITY = "utility"
    COMMERCIAL = "commercial"
    RESIDENTIAL = "residential"


class ModuleParameters(BaseModel):
    """PV module electrical and physical parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    manufacturer: str = Field(..., description="Module manufacturer name")
    model: str = Field(..., description="Module model number")
    technology: str = Field(..., description="Cell technology (mono-Si, poly-Si, CdTe, etc.)")

    # Electrical parameters at STC
    pmax: float = Field(..., gt=0, description="Maximum power at STC (W)")
    voc: float = Field(..., gt=0, description="Open circuit voltage at STC (V)")
    isc: float = Field(..., gt=0, description="Short circuit current at STC (A)")
    vmp: float = Field(..., gt=0, description="Voltage at maximum power point at STC (V)")
    imp: float = Field(..., gt=0, description="Current at maximum power point at STC (A)")

    # Temperature coefficients
    temp_coeff_pmax: float = Field(..., description="Temperature coefficient of Pmax (%/°C)")
    temp_coeff_voc: float = Field(..., description="Temperature coefficient of Voc (%/°C)")
    temp_coeff_isc: float = Field(..., description="Temperature coefficient of Isc (%/°C)")

    # Physical parameters
    length: float = Field(..., gt=0, description="Module length (m)")
    width: float = Field(..., gt=0, description="Module width (m)")
    thickness: float = Field(..., gt=0, description="Module thickness (m)")
    weight: float = Field(..., gt=0, description="Module weight (kg)")

    # Additional parameters
    cells_in_series: int = Field(..., gt=0, description="Number of cells in series")
    efficiency: float = Field(..., gt=0, le=100, description="Module efficiency (%)")
    noct: float = Field(default=45.0, description="Nominal operating cell temperature (°C)")

    @property
    def area(self) -> float:
        """Calculate module area in m²."""
        return self.length * self.width


class InverterParameters(BaseModel):
    """Inverter electrical parameters and specifications."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    manufacturer: str = Field(..., description="Inverter manufacturer name")
    model: str = Field(..., description="Inverter model number")
    inverter_type: InverterType = Field(..., description="Type of inverter")

    # AC parameters
    pac_max: float = Field(..., gt=0, description="Maximum AC power output (W)")
    vac_nom: float = Field(..., gt=0, description="Nominal AC voltage (V)")
    iac_max: float = Field(..., gt=0, description="Maximum AC current (A)")
    frequency: float = Field(default=60.0, description="AC frequency (Hz)")

    # DC parameters
    pdc_max: float = Field(..., gt=0, description="Maximum DC power input (W)")
    vdc_max: float = Field(..., gt=0, description="Maximum DC voltage (V)")
    vdc_nom: float = Field(..., gt=0, description="Nominal DC voltage (V)")
    vdc_min: float = Field(..., gt=0, description="Minimum DC operating voltage (V)")
    idc_max: float = Field(..., gt=0, description="Maximum DC current (A)")

    # MPPT parameters
    num_mppt: int = Field(..., gt=0, description="Number of MPPT inputs")
    mppt_vmin: float = Field(..., gt=0, description="MPPT minimum voltage (V)")
    mppt_vmax: float = Field(..., gt=0, description="MPPT maximum voltage (V)")
    strings_per_mppt: int = Field(..., gt=0, description="Maximum strings per MPPT")

    # Efficiency
    max_efficiency: float = Field(..., gt=0, le=100, description="Maximum efficiency (%)")
    euro_efficiency: Optional[float] = Field(None, description="European efficiency (%)")
    cec_efficiency: Optional[float] = Field(None, description="CEC weighted efficiency (%)")

    # Physical
    weight: float = Field(..., gt=0, description="Inverter weight (kg)")
    dimensions: Optional[Tuple[float, float, float]] = Field(None, description="Dimensions (L, W, H) in m")

    # Environmental
    operating_temp_min: float = Field(default=-25.0, description="Minimum operating temperature (°C)")
    operating_temp_max: float = Field(default=60.0, description="Maximum operating temperature (°C)")
    ip_rating: str = Field(default="IP65", description="Ingress protection rating")

    @field_validator('vdc_min')
    @classmethod
    def validate_vdc_min(cls, v: float, info) -> float:
        """Validate that minimum DC voltage is less than nominal."""
        if 'vdc_nom' in info.data and v >= info.data['vdc_nom']:
            raise ValueError("vdc_min must be less than vdc_nom")
        return v


class StringConfiguration(BaseModel):
    """Configuration for a PV string."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    modules_per_string: int = Field(..., gt=0, description="Number of modules per string")
    strings_per_mppt: int = Field(..., gt=0, description="Number of strings per MPPT")
    orientation_azimuth: float = Field(..., ge=-180, le=180, description="String azimuth angle (degrees)")
    tilt_angle: float = Field(..., ge=0, le=90, description="String tilt angle (degrees)")

    # Voltage calculations (computed)
    voc_stc: Optional[float] = Field(None, description="String Voc at STC (V)")
    vmp_stc: Optional[float] = Field(None, description="String Vmp at STC (V)")
    isc_stc: Optional[float] = Field(None, description="String Isc at STC (A)")
    imp_stc: Optional[float] = Field(None, description="String Imp at STC (A)")

    # Temperature corrected values
    voc_min_temp: Optional[float] = Field(None, description="String Voc at max temp (V)")
    voc_max_temp: Optional[float] = Field(None, description="String Voc at min temp (V)")
    vmp_min_temp: Optional[float] = Field(None, description="String Vmp at max temp (V)")
    vmp_max_temp: Optional[float] = Field(None, description="String Vmp at min temp (V)")


class SystemLosses(BaseModel):
    """System loss parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Environmental losses
    soiling: float = Field(default=2.0, ge=0, le=100, description="Soiling losses (%)")
    shading_near: float = Field(default=0.0, ge=0, le=100, description="Near shading losses (%)")
    shading_far: float = Field(default=0.0, ge=0, le=100, description="Far shading/horizon losses (%)")

    # Electrical losses
    dc_wiring: float = Field(default=1.5, ge=0, le=100, description="DC wiring losses (%)")
    ac_wiring: float = Field(default=1.0, ge=0, le=100, description="AC wiring losses (%)")
    transformer: float = Field(default=1.0, ge=0, le=100, description="Transformer losses (%)")

    # Inverter losses (computed from efficiency curve)
    inverter: float = Field(default=2.5, ge=0, le=100, description="Inverter losses (%)")
    clipping: float = Field(default=0.0, ge=0, le=100, description="Inverter clipping losses (%)")

    # Availability losses
    availability: float = Field(default=98.0, ge=0, le=100, description="System availability (%)")
    grid_curtailment: float = Field(default=0.0, ge=0, le=100, description="Grid curtailment (%)")

    # Degradation
    lid: float = Field(default=1.5, ge=0, le=100, description="Light induced degradation (%)")
    mismatch: float = Field(default=1.0, ge=0, le=100, description="Module mismatch losses (%)")

    def total_losses(self) -> float:
        """Calculate total system losses using multiplication method."""
        factors = [
            (100 - self.soiling) / 100,
            (100 - self.shading_near) / 100,
            (100 - self.shading_far) / 100,
            (100 - self.dc_wiring) / 100,
            (100 - self.ac_wiring) / 100,
            (100 - self.transformer) / 100,
            (100 - self.inverter) / 100,
            (100 - self.clipping) / 100,
            self.availability / 100,
            (100 - self.grid_curtailment) / 100,
            (100 - self.lid) / 100,
            (100 - self.mismatch) / 100,
        ]
        total_factor = 1.0
        for factor in factors:
            total_factor *= factor
        return (1.0 - total_factor) * 100


class ArrayLayout(BaseModel):
    """Array layout configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mounting_type: MountingType = Field(..., description="Type of mounting system")
    tilt_angle: float = Field(..., ge=0, le=90, description="Array tilt angle (degrees)")
    azimuth: float = Field(..., ge=-180, le=180, description="Array azimuth (degrees, 0=North)")

    # Layout parameters
    rows: int = Field(..., gt=0, description="Number of rows")
    modules_per_row: int = Field(..., gt=0, description="Modules per row")
    row_spacing: float = Field(..., gt=0, description="Row-to-row spacing (m)")
    module_spacing: float = Field(default=0.02, ge=0, description="Module-to-module spacing (m)")

    # Ground coverage ratio
    gcr: Optional[float] = Field(None, ge=0, le=1, description="Ground coverage ratio")

    # Tracker-specific
    backtracking_enabled: bool = Field(default=False, description="Enable backtracking for trackers")
    max_rotation_angle: Optional[float] = Field(None, description="Maximum tracker rotation (degrees)")

    # Rooftop-specific
    setback_front: Optional[float] = Field(None, description="Front setback (m)")
    setback_back: Optional[float] = Field(None, description="Back setback (m)")
    setback_side: Optional[float] = Field(None, description="Side setback (m)")

    # Floating-specific
    pontoon_spacing: Optional[float] = Field(None, description="Pontoon spacing (m)")
    water_coverage_ratio: Optional[float] = Field(None, description="Water surface coverage ratio")

    # Agrivoltaic-specific
    clearance_height: Optional[float] = Field(None, description="Ground clearance (m)")
    crop_row_spacing: Optional[float] = Field(None, description="Crop row spacing (m)")


class SystemConfiguration(BaseModel):
    """Complete PV system configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # System identification
    project_name: str = Field(..., description="Project name")
    system_type: SystemType = Field(..., description="System type (utility/commercial/residential)")
    location: str = Field(..., description="System location")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (degrees)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (degrees)")
    elevation: float = Field(..., description="Elevation above sea level (m)")

    # Design date and environmental
    design_date: datetime = Field(default_factory=datetime.now, description="Design date")
    site_temp_min: float = Field(..., description="Site minimum temperature (°C)")
    site_temp_max: float = Field(..., description="Site maximum temperature (°C)")
    avg_ambient_temp: float = Field(..., description="Average ambient temperature (°C)")

    # Module and inverter
    module: ModuleParameters = Field(..., description="Module parameters")
    inverter: InverterParameters = Field(..., description="Inverter parameters")

    # Array configuration
    array_layout: ArrayLayout = Field(..., description="Array layout configuration")
    string_config: StringConfiguration = Field(..., description="String configuration")

    # System sizing
    num_modules: int = Field(..., gt=0, description="Total number of modules")
    num_inverters: int = Field(..., gt=0, description="Total number of inverters")
    dc_capacity: Optional[float] = Field(None, description="Total DC capacity (kW)")
    ac_capacity: Optional[float] = Field(None, description="Total AC capacity (kW)")
    dc_ac_ratio: Optional[float] = Field(None, description="DC/AC ratio")

    # Losses
    losses: SystemLosses = Field(default_factory=SystemLosses, description="System losses")

    # Performance metrics (computed)
    performance_ratio: Optional[float] = Field(None, description="Performance ratio (%)")
    specific_yield: Optional[float] = Field(None, description="Specific yield (kWh/kWp/year)")
    annual_energy: Optional[float] = Field(None, description="Annual energy production (kWh)")

    def calculate_dc_capacity(self) -> float:
        """Calculate total DC capacity in kW."""
        return (self.num_modules * self.module.pmax) / 1000

    def calculate_ac_capacity(self) -> float:
        """Calculate total AC capacity in kW."""
        return (self.num_inverters * self.inverter.pac_max) / 1000

    def calculate_dc_ac_ratio(self) -> float:
        """Calculate DC to AC ratio."""
        dc_kw = self.calculate_dc_capacity()
        ac_kw = self.calculate_ac_capacity()
        return dc_kw / ac_kw if ac_kw > 0 else 0.0


class WeatherData(BaseModel):
    """Weather and irradiance data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    location: str = Field(..., description="Weather data location")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (degrees)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (degrees)")

    # Irradiance data (W/m²)
    ghi: List[float] = Field(..., description="Global horizontal irradiance time series")
    dni: List[float] = Field(..., description="Direct normal irradiance time series")
    dhi: List[float] = Field(..., description="Diffuse horizontal irradiance time series")

    # Temperature data (°C)
    temp_air: List[float] = Field(..., description="Air temperature time series")
    temp_dew: Optional[List[float]] = Field(None, description="Dew point temperature time series")

    # Wind data (m/s)
    wind_speed: Optional[List[float]] = Field(None, description="Wind speed time series")
    wind_direction: Optional[List[float]] = Field(None, description="Wind direction time series")

    # Other meteorological data
    relative_humidity: Optional[List[float]] = Field(None, description="Relative humidity time series (%)")
    pressure: Optional[List[float]] = Field(None, description="Atmospheric pressure time series (Pa)")
    albedo: Optional[List[float]] = Field(None, description="Ground albedo time series")

    # Timestamps
    timestamps: List[datetime] = Field(..., description="Timestamp for each data point")

    @field_validator('ghi', 'dni', 'dhi', 'temp_air')
    @classmethod
    def validate_equal_length(cls, v: List[float], info) -> List[float]:
        """Validate that all time series have equal length."""
        if 'timestamps' in info.data and len(v) != len(info.data['timestamps']):
            raise ValueError(f"Time series length {len(v)} does not match timestamps length {len(info.data['timestamps'])}")
        return v


class PVsystPanFile(BaseModel):
    """Structure for PVsyst .PAN file data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Header information
    pan_file_version: str = Field(..., description="PAN file version")
    technology: str = Field(..., description="Cell technology")

    # Module parameters (maps to ModuleParameters)
    module_params: ModuleParameters = Field(..., description="Module electrical parameters")

    # Additional PVsyst-specific parameters
    gamma_ref: Optional[float] = Field(None, description="Diode ideality factor at reference")
    mu_gamma: Optional[float] = Field(None, description="Temperature coefficient of gamma")

    # IAM (Incidence Angle Modifier) parameters
    iam_model: Optional[str] = Field(None, description="IAM model type")
    iam_parameters: Optional[Dict[str, float]] = Field(None, description="IAM model parameters")


class PVsystOndFile(BaseModel):
    """Structure for PVsyst .OND file data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Header information
    ond_file_version: str = Field(..., description="OND file version")

    # Inverter parameters (maps to InverterParameters)
    inverter_params: InverterParameters = Field(..., description="Inverter electrical parameters")

    # Efficiency curve
    efficiency_curve: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Efficiency curve as (power_fraction, efficiency) pairs"
    )

    # Night consumption
    night_consumption: Optional[float] = Field(None, description="Night consumption power (W)")

    # Additional parameters
    auxiliary_consumption: Optional[float] = Field(None, description="Auxiliary consumption (W)")


class SimulationResults(BaseModel):
    """PV system simulation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # System configuration reference
    system_config: SystemConfiguration = Field(..., description="System configuration used")

    # Time series results
    timestamps: List[datetime] = Field(..., description="Simulation timestamps")
    dc_power: List[float] = Field(..., description="DC power output (W)")
    ac_power: List[float] = Field(..., description="AC power output (W)")
    dc_voltage: List[float] = Field(..., description="DC voltage (V)")
    dc_current: List[float] = Field(..., description="DC current (A)")

    # Temperature
    module_temp: List[float] = Field(..., description="Module temperature (°C)")

    # Losses breakdown
    soiling_loss: List[float] = Field(..., description="Soiling losses (W)")
    shading_loss: List[float] = Field(..., description="Shading losses (W)")
    wiring_loss: List[float] = Field(..., description="Wiring losses (W)")
    inverter_loss: List[float] = Field(..., description="Inverter losses (W)")
    clipping_loss: List[float] = Field(..., description="Clipping losses (W)")

    # Summary metrics
    total_dc_energy: float = Field(..., description="Total DC energy (kWh)")
    total_ac_energy: float = Field(..., description="Total AC energy (kWh)")
    performance_ratio: float = Field(..., description="Performance ratio (%)")
    capacity_factor: float = Field(..., description="Capacity factor (%)")
    specific_yield: float = Field(..., description="Specific yield (kWh/kWp/year)")
