"""
System models for PV circularity simulator.

This module defines comprehensive Pydantic models for complete photovoltaic
systems, including:
- System configuration (modules, inverters, mounting)
- Location and orientation
- Electrical protection and wiring
- Monitoring systems
- Grid connection parameters

All models include full validation for physical constraints and
production-ready error handling.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.core import NamedModel, UUIDModel
from pv_circularity_simulator.models.modules import ModuleModel


class MountingType(str, Enum):
    """Enumeration of PV system mounting types."""

    ROOF_MOUNTED = "roof_mounted"
    GROUND_MOUNTED = "ground_mounted"
    POLE_MOUNTED = "pole_mounted"
    TRACKING_SINGLE_AXIS = "tracking_single_axis"
    TRACKING_DUAL_AXIS = "tracking_dual_axis"
    CARPORT = "carport"
    FACADE = "facade"
    FLOATING = "floating"


class InverterType(str, Enum):
    """Enumeration of inverter types."""

    STRING = "string"
    CENTRAL = "central"
    MICROINVERTER = "microinverter"
    POWER_OPTIMIZER = "power_optimizer"
    HYBRID = "hybrid"


class GridConnectionType(str, Enum):
    """Enumeration of grid connection types."""

    ON_GRID = "on_grid"
    OFF_GRID = "off_grid"
    HYBRID = "hybrid"


class LocationCoordinates(NamedModel):
    """
    Geographic location of the PV system.

    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        altitude_m: Altitude above sea level in meters
        timezone: Timezone string (e.g., 'America/New_York', 'Europe/Berlin')
        location_name: Human-readable location name
    """

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude in decimal degrees (-90 to 90)",
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude in decimal degrees (-180 to 180)",
    )
    altitude_m: float = Field(
        default=0.0,
        ge=-500,
        le=9000,
        description="Altitude above sea level in meters",
    )
    timezone: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Timezone string (e.g., 'UTC', 'America/New_York', 'Europe/Berlin')",
    )
    location_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable location name (city, state, country)",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone string format."""
        # Basic validation - could be enhanced with pytz if needed
        if "/" not in v and v != "UTC":
            raise ValueError(
                "Timezone must be in format 'Region/City' (e.g., 'America/New_York') or 'UTC'"
            )
        return v


class Orientation(NamedModel):
    """
    Orientation of the PV array.

    Attributes:
        tilt_angle_deg: Tilt angle from horizontal in degrees (0-90)
        azimuth_angle_deg: Azimuth angle in degrees (0-360, 180=South in N hemisphere)
        tracking_type: Type of tracking system (if any)
        row_spacing_m: Spacing between rows in meters (for ground-mounted)
        gcr: Ground coverage ratio (0-1, module area / ground area)
    """

    tilt_angle_deg: float = Field(
        ...,
        ge=0,
        le=90,
        description="Tilt angle from horizontal in degrees (0=flat, 90=vertical)",
    )
    azimuth_angle_deg: float = Field(
        ...,
        ge=0,
        lt=360,
        description="Azimuth angle in degrees (0=N, 90=E, 180=S, 270=W)",
    )
    tracking_type: Optional[MountingType] = Field(
        None,
        description="Tracking system type (if tracking is used)",
    )
    row_spacing_m: Optional[float] = Field(
        None,
        gt=0,
        le=100,
        description="Spacing between rows in meters (for ground-mounted systems)",
    )
    gcr: Optional[float] = Field(
        None,
        gt=0,
        le=1.0,
        description="Ground coverage ratio (module area / ground area)",
    )

    @model_validator(mode="after")
    def validate_tracking_consistency(self) -> "Orientation":
        """Validate tracking configuration consistency."""
        if self.tracking_type in [
            MountingType.TRACKING_SINGLE_AXIS,
            MountingType.TRACKING_DUAL_AXIS,
        ]:
            # For tracking systems, tilt might be variable
            import warnings
            warnings.warn(
                f"Tracking system specified ({self.tracking_type}), "
                f"tilt angle represents tracker orientation at rest"
            )
        return self


class InverterConfiguration(NamedModel):
    """
    Inverter configuration for the PV system.

    Attributes:
        inverter_type: Type of inverter
        rated_power_w: Rated AC power output in watts
        max_dc_voltage_v: Maximum DC input voltage in volts
        mppt_voltage_range_min_v: Minimum MPPT voltage in volts
        mppt_voltage_range_max_v: Maximum MPPT voltage in volts
        max_dc_current_a: Maximum DC input current in amperes
        number_of_mppt: Number of MPPT trackers
        efficiency_max_percentage: Maximum efficiency percentage
        efficiency_weighted_percentage: Weighted efficiency percentage
        manufacturer: Inverter manufacturer
        model: Inverter model number
    """

    inverter_type: InverterType = Field(
        ...,
        description="Type of inverter",
    )
    rated_power_w: float = Field(
        ...,
        gt=0,
        le=10_000_000,
        description="Rated AC power output in watts",
    )
    max_dc_voltage_v: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Maximum DC input voltage in volts",
    )
    mppt_voltage_range_min_v: float = Field(
        ...,
        gt=0,
        description="Minimum MPPT voltage range in volts",
    )
    mppt_voltage_range_max_v: float = Field(
        ...,
        gt=0,
        description="Maximum MPPT voltage range in volts",
    )
    max_dc_current_a: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Maximum DC input current in amperes",
    )
    number_of_mppt: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of MPPT (Maximum Power Point Tracking) inputs",
    )
    efficiency_max_percentage: float = Field(
        ...,
        gt=80,
        le=100,
        description="Maximum efficiency percentage (typical: 95-99%)",
    )
    efficiency_weighted_percentage: float = Field(
        ...,
        gt=80,
        le=100,
        description="Weighted efficiency percentage (CEC or Euro efficiency)",
    )
    manufacturer: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Inverter manufacturer name",
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Inverter model number",
    )

    @model_validator(mode="after")
    def validate_inverter_consistency(self) -> "InverterConfiguration":
        """Validate inverter parameter consistency."""
        # MPPT range should be valid
        if self.mppt_voltage_range_min_v >= self.mppt_voltage_range_max_v:
            raise ValueError(
                f"MPPT min voltage ({self.mppt_voltage_range_min_v}V) must be less than "
                f"max voltage ({self.mppt_voltage_range_max_v}V)"
            )

        # MPPT max should not exceed max DC voltage
        if self.mppt_voltage_range_max_v > self.max_dc_voltage_v:
            raise ValueError(
                f"MPPT max voltage ({self.mppt_voltage_range_max_v}V) cannot exceed "
                f"max DC voltage ({self.max_dc_voltage_v}V)"
            )

        # Weighted efficiency should be <= max efficiency
        if self.efficiency_weighted_percentage > self.efficiency_max_percentage:
            raise ValueError(
                f"Weighted efficiency ({self.efficiency_weighted_percentage}%) cannot exceed "
                f"max efficiency ({self.efficiency_max_percentage}%)"
            )

        return self


class MountingStructure(NamedModel):
    """
    Mounting structure configuration.

    Attributes:
        mounting_type: Type of mounting system
        structure_material: Material of mounting structure
        foundation_type: Type of foundation (for ground-mount)
        wind_load_capacity_pa: Wind load capacity in Pascals
        snow_load_capacity_pa: Snow load capacity in Pascals
        corrosion_resistance: Corrosion resistance rating
        expected_lifetime_years: Expected structural lifetime in years
    """

    mounting_type: MountingType = Field(
        ...,
        description="Type of mounting system",
    )
    structure_material: str = Field(
        default="aluminum",
        max_length=50,
        description="Material of mounting structure (aluminum, steel, etc.)",
    )
    foundation_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Type of foundation (concrete, pile, ballasted, etc.)",
    )
    wind_load_capacity_pa: float = Field(
        default=2400,
        gt=0,
        le=10000,
        description="Wind load capacity in Pascals (typical: 2400 Pa)",
    )
    snow_load_capacity_pa: float = Field(
        default=5400,
        gt=0,
        le=20000,
        description="Snow load capacity in Pascals (typical: 5400 Pa)",
    )
    corrosion_resistance: str = Field(
        default="C3",
        max_length=20,
        description="Corrosion resistance rating (ISO 12944: C1-C5)",
    )
    expected_lifetime_years: int = Field(
        default=25,
        ge=10,
        le=50,
        description="Expected structural lifetime in years",
    )


class ElectricalProtection(NamedModel):
    """
    Electrical protection and safety equipment.

    Attributes:
        dc_disconnect: Whether DC disconnect switch is present
        ac_disconnect: Whether AC disconnect switch is present
        surge_protection_dc: Whether DC surge protection is installed
        surge_protection_ac: Whether AC surge protection is installed
        ground_fault_protection: Whether ground fault protection is present
        arc_fault_protection: Whether arc fault protection is present
        rapid_shutdown: Whether rapid shutdown system is installed
        isolation_monitoring: Whether isolation monitoring is present
    """

    dc_disconnect: bool = Field(
        default=True,
        description="DC disconnect switch present",
    )
    ac_disconnect: bool = Field(
        default=True,
        description="AC disconnect switch present",
    )
    surge_protection_dc: bool = Field(
        default=True,
        description="DC surge protection device (SPD) installed",
    )
    surge_protection_ac: bool = Field(
        default=True,
        description="AC surge protection device (SPD) installed",
    )
    ground_fault_protection: bool = Field(
        default=True,
        description="Ground fault detection and interruption present",
    )
    arc_fault_protection: bool = Field(
        default=True,
        description="Arc fault circuit interrupter (AFCI) present",
    )
    rapid_shutdown: bool = Field(
        default=False,
        description="Rapid shutdown system installed (required in some jurisdictions)",
    )
    isolation_monitoring: bool = Field(
        default=False,
        description="Insulation/isolation monitoring device present",
    )


class SystemConfiguration(NamedModel):
    """
    Complete PV system electrical configuration.

    Attributes:
        modules_per_string: Number of modules connected in series per string
        strings_in_parallel: Number of strings connected in parallel
        total_modules: Total number of modules in the system
        dc_capacity_w: Total DC capacity in watts
        ac_capacity_w: Total AC capacity (inverter rating) in watts
        dc_ac_ratio: DC to AC ratio (oversizing factor)
        string_voltage_voc: String open-circuit voltage in volts
        string_voltage_vmpp: String operating voltage at MPP in volts
        string_current_isc: String short-circuit current in amperes
    """

    modules_per_string: int = Field(
        ...,
        ge=1,
        le=50,
        description="Number of modules in series per string (typical: 10-30)",
    )
    strings_in_parallel: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of strings in parallel",
    )
    total_modules: int = Field(
        ...,
        ge=1,
        description="Total number of modules in the system",
    )
    dc_capacity_w: float = Field(
        ...,
        gt=0,
        description="Total DC capacity (STC) in watts",
    )
    ac_capacity_w: float = Field(
        ...,
        gt=0,
        description="Total AC capacity (inverter rating) in watts",
    )
    dc_ac_ratio: float = Field(
        ...,
        gt=0,
        le=3.0,
        description="DC to AC ratio (typical: 1.0-1.3)",
    )
    string_voltage_voc: float = Field(
        ...,
        gt=0,
        le=2000,
        description="String open-circuit voltage in volts",
    )
    string_voltage_vmpp: float = Field(
        ...,
        gt=0,
        description="String operating voltage at maximum power point in volts",
    )
    string_current_isc: float = Field(
        ...,
        gt=0,
        description="String short-circuit current in amperes",
    )

    @model_validator(mode="after")
    def validate_system_configuration(self) -> "SystemConfiguration":
        """Validate system configuration consistency."""
        # Total modules should equal modules_per_string × strings_in_parallel
        expected_total = self.modules_per_string * self.strings_in_parallel
        if self.total_modules != expected_total:
            raise ValueError(
                f"Total modules ({self.total_modules}) should equal "
                f"modules_per_string × strings_in_parallel ({expected_total})"
            )

        # DC/AC ratio should match capacities
        calculated_ratio = self.dc_capacity_w / self.ac_capacity_w
        if abs(calculated_ratio - self.dc_ac_ratio) > 0.01:
            raise ValueError(
                f"DC/AC ratio ({self.dc_ac_ratio:.2f}) should match "
                f"dc_capacity / ac_capacity ({calculated_ratio:.2f})"
            )

        # String voltage at MPP should be less than Voc
        if self.string_voltage_vmpp >= self.string_voltage_voc:
            raise ValueError(
                f"String Vmpp ({self.string_voltage_vmpp}V) must be less than "
                f"Voc ({self.string_voltage_voc}V)"
            )

        return self


class SystemModel(UUIDModel):
    """
    Comprehensive PV system model.

    This model represents a complete photovoltaic system installation,
    including modules, inverters, mounting, location, and all configuration
    parameters.

    Attributes:
        name: Human-readable name/identifier for the system
        system_id: External system identifier (optional)
        module: Reference module model used in this system
        location: Geographic location and coordinates
        orientation: Array orientation (tilt, azimuth, tracking)
        configuration: Electrical system configuration
        inverter: Inverter configuration
        mounting: Mounting structure details
        protection: Electrical protection equipment
        grid_connection: Grid connection type
        monitoring_system: Monitoring system information
        installer: Installation company name
        installation_date: Date of installation
        commissioning_date: Date of commissioning
        expected_lifetime_years: Expected system lifetime
        owner: System owner information
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name/identifier for the system",
    )
    system_id: Optional[str] = Field(
        None,
        max_length=100,
        description="External system identifier (utility or monitoring system ID)",
    )
    module: ModuleModel = Field(
        ...,
        description="Reference module model used in this system",
    )
    location: LocationCoordinates = Field(
        ...,
        description="Geographic location and coordinates of the system",
    )
    orientation: Orientation = Field(
        ...,
        description="Array orientation (tilt angle, azimuth, tracking)",
    )
    configuration: SystemConfiguration = Field(
        ...,
        description="Electrical system configuration (strings, modules, capacity)",
    )
    inverter: InverterConfiguration = Field(
        ...,
        description="Inverter specifications and configuration",
    )
    mounting: MountingStructure = Field(
        ...,
        description="Mounting structure and foundation details",
    )
    protection: ElectricalProtection = Field(
        default_factory=ElectricalProtection,
        description="Electrical protection and safety equipment",
    )
    grid_connection: GridConnectionType = Field(
        ...,
        description="Type of grid connection (on-grid, off-grid, hybrid)",
    )
    monitoring_system: Optional[str] = Field(
        None,
        max_length=255,
        description="Monitoring system brand/model",
    )
    installer: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Installation company name",
    )
    installation_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Installation date in YYYY-MM-DD format",
    )
    commissioning_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Commissioning date in YYYY-MM-DD format",
    )
    expected_lifetime_years: int = Field(
        default=25,
        ge=10,
        le=50,
        description="Expected system lifetime in years",
    )
    owner: Optional[str] = Field(
        None,
        max_length=255,
        description="System owner name or organization",
    )

    @model_validator(mode="after")
    def validate_system_consistency(self) -> "SystemModel":
        """Validate consistency across system components."""
        # Check string voltage is within inverter MPPT range
        if not (
            self.inverter.mppt_voltage_range_min_v
            <= self.configuration.string_voltage_vmpp
            <= self.inverter.mppt_voltage_range_max_v
        ):
            raise ValueError(
                f"String Vmpp ({self.configuration.string_voltage_vmpp}V) is outside "
                f"inverter MPPT range ({self.inverter.mppt_voltage_range_min_v}V - "
                f"{self.inverter.mppt_voltage_range_max_v}V)"
            )

        # Check string Voc doesn't exceed inverter max voltage
        if self.configuration.string_voltage_voc > self.inverter.max_dc_voltage_v:
            raise ValueError(
                f"String Voc ({self.configuration.string_voltage_voc}V) exceeds "
                f"inverter max DC voltage ({self.inverter.max_dc_voltage_v}V)"
            )

        # Verify DC capacity matches module count
        expected_dc_capacity = self.module.electrical.pmax_w * self.configuration.total_modules
        if abs(self.configuration.dc_capacity_w - expected_dc_capacity) > 100:
            raise ValueError(
                f"DC capacity ({self.configuration.dc_capacity_w}W) should match "
                f"module power × total modules ({expected_dc_capacity:.0f}W)"
            )

        return self

    def calculate_total_area_m2(self) -> float:
        """
        Calculate total system area in square meters.

        Returns:
            float: Total area in m²
        """
        module_area_m2 = self.module.mechanical.calculate_area_m2()
        return module_area_m2 * self.configuration.total_modules

    def calculate_annual_energy_kwh(
        self,
        annual_irradiation_kwh_m2: float,
        performance_ratio: float = 0.80,
    ) -> float:
        """
        Estimate annual energy production for the entire system.

        Args:
            annual_irradiation_kwh_m2: Annual irradiation in kWh/m²
            performance_ratio: System performance ratio (0-1)

        Returns:
            float: Estimated annual energy in kWh

        Raises:
            ValueError: If inputs are invalid
        """
        module_annual_energy = self.module.estimate_annual_energy_kwh(
            annual_irradiation_kwh_m2, performance_ratio
        )
        return module_annual_energy * self.configuration.total_modules

    def calculate_specific_yield_kwh_kwp(
        self,
        annual_energy_kwh: float,
    ) -> float:
        """
        Calculate specific yield (kWh/kWp).

        Args:
            annual_energy_kwh: Annual energy production in kWh

        Returns:
            float: Specific yield in kWh/kWp
        """
        dc_capacity_kwp = self.configuration.dc_capacity_w / 1000.0
        return annual_energy_kwh / dc_capacity_kwp
