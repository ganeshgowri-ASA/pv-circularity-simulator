"""
Module Temperature and NOCT Calculation Module

This module provides comprehensive photovoltaic module temperature modeling
including NOCT (Nominal Operating Cell Temperature) calculations, various
temperature models, and thermal effects from different mounting configurations.

Key Features:
- Multiple temperature models: Ross/Faiman, Sandia, King (SAPM), Simple Linear
- NOCT calculation with mounting configuration factors
- Wind speed and convective cooling effects
- Temperature coefficient loss calculations
- Thermal time constant modeling
- Seasonal and time-of-day variations

Author: PV Circularity Simulator Team
Date: 2025-11-17
"""

from typing import Dict, Optional, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math


# ============================================================================
# Enumerations and Constants
# ============================================================================

class MountingType(str, Enum):
    """
    Mounting configuration types for PV modules.

    Different mounting configurations significantly affect module operating
    temperatures due to varying levels of ventilation and heat dissipation.
    """
    OPEN_RACK = "open_rack"
    CLOSE_ROOF = "close_roof"
    BUILDING_INTEGRATED = "building_integrated"
    GROUND_MOUNT = "ground_mount"
    TRACKER_SINGLE_AXIS = "tracker_single_axis"
    TRACKER_DUAL_AXIS = "tracker_dual_axis"


class TemperatureModelType(str, Enum):
    """Temperature calculation model types."""
    SIMPLE_LINEAR = "simple_linear"
    ROSS_FAIMAN = "ross_faiman"
    SANDIA = "sandia"
    KING_SAPM = "king_sapm"


class ModuleTechnology(str, Enum):
    """PV module technology types with different temperature coefficients."""
    MONO_SI = "mono_si"
    POLY_SI = "poly_si"
    HJT = "hjt"  # Heterojunction
    PERC = "perc"
    TOPCON = "topcon"
    THIN_FILM_CDTE = "thin_film_cdte"
    THIN_FILM_CIGS = "thin_film_cigs"
    BIFACIAL = "bifacial"


# Standard NOCT test conditions
NOCT_STANDARD_IRRADIANCE = 800.0  # W/m²
NOCT_STANDARD_AMBIENT = 20.0  # °C
NOCT_STANDARD_WIND = 1.0  # m/s
STC_TEMPERATURE = 25.0  # °C (Standard Test Conditions)


# Mounting configuration thermal adjustment factors (°C increase over open rack)
MOUNTING_THERMAL_ADJUSTMENT = {
    MountingType.OPEN_RACK: 0.0,  # Baseline - full ventilation
    MountingType.CLOSE_ROOF: 12.5,  # <6 inches clearance, limited airflow (+10-15°C)
    MountingType.BUILDING_INTEGRATED: 25.0,  # Worst case, minimal ventilation (+20-30°C)
    MountingType.GROUND_MOUNT: -2.0,  # Optimized clearance, slight improvement
    MountingType.TRACKER_SINGLE_AXIS: -3.0,  # Better ventilation, tracking advantages
    MountingType.TRACKER_DUAL_AXIS: -4.0,  # Best ventilation and orientation
}


# Typical temperature coefficients by technology (%/°C)
DEFAULT_TEMP_COEFFICIENTS = {
    ModuleTechnology.MONO_SI: -0.40,
    ModuleTechnology.POLY_SI: -0.43,
    ModuleTechnology.HJT: -0.25,  # Better temperature performance
    ModuleTechnology.PERC: -0.37,
    ModuleTechnology.TOPCON: -0.35,
    ModuleTechnology.THIN_FILM_CDTE: -0.25,
    ModuleTechnology.THIN_FILM_CIGS: -0.32,
    ModuleTechnology.BIFACIAL: -0.38,
}


# ============================================================================
# Pydantic Models
# ============================================================================

class NOCTCalculationInput(BaseModel):
    """
    Input parameters for NOCT calculation.

    NOCT represents the module temperature under specific standardized
    conditions: 800 W/m² irradiance, 20°C ambient, 1 m/s wind speed.
    """
    model_config = ConfigDict(use_enum_values=True)

    ambient_temp: float = Field(
        ...,
        description="Ambient air temperature (°C)",
        ge=-50.0,
        le=60.0
    )
    irradiance: float = Field(
        ...,
        description="Plane of array irradiance (W/m²)",
        ge=0.0,
        le=1500.0
    )
    wind_speed: float = Field(
        ...,
        description="Wind speed at module height (m/s)",
        ge=0.0,
        le=30.0
    )
    mounting: MountingType = Field(
        default=MountingType.OPEN_RACK,
        description="Mounting configuration type"
    )
    base_noct: Optional[float] = Field(
        default=45.0,
        description="Base NOCT value from datasheet (°C)",
        ge=25.0,
        le=65.0
    )


class ModuleTemperatureInput(BaseModel):
    """Input parameters for module temperature calculation."""
    model_config = ConfigDict(use_enum_values=True)

    ambient_temp: float = Field(
        ...,
        description="Ambient air temperature (°C)",
        ge=-50.0,
        le=60.0
    )
    irradiance: float = Field(
        ...,
        description="Plane of array irradiance (W/m²)",
        ge=0.0,
        le=1500.0
    )
    wind_speed: float = Field(
        ...,
        description="Wind speed at module height (m/s)",
        ge=0.0,
        le=30.0
    )
    noct: float = Field(
        default=45.0,
        description="Nominal Operating Cell Temperature (°C)",
        ge=25.0,
        le=65.0
    )
    mounting: MountingType = Field(
        default=MountingType.OPEN_RACK,
        description="Mounting configuration"
    )
    model_type: TemperatureModelType = Field(
        default=TemperatureModelType.SIMPLE_LINEAR,
        description="Temperature model to use"
    )

    # Model-specific parameters (optional)
    ross_a: Optional[float] = Field(
        default=None,
        description="Ross model coefficient a (W/m²°C)",
        ge=0.0
    )
    ross_b: Optional[float] = Field(
        default=None,
        description="Ross model coefficient b (W s/m³°C)",
        ge=0.0
    )
    sandia_a: Optional[float] = Field(
        default=None,
        description="Sandia model parameter a",
        ge=-10.0,
        le=0.0
    )
    sandia_b: Optional[float] = Field(
        default=None,
        description="Sandia model parameter b",
        ge=-1.0,
        le=0.0
    )
    sandia_deltac: Optional[float] = Field(
        default=None,
        description="Sandia model DeltaT parameter",
        ge=0.0
    )


class TemperatureCoefficientInput(BaseModel):
    """Input parameters for temperature coefficient loss calculation."""
    model_config = ConfigDict(use_enum_values=True)

    module_temp: float = Field(
        ...,
        description="Module operating temperature (°C)",
        ge=-50.0,
        le=120.0
    )
    stc_temp: float = Field(
        default=STC_TEMPERATURE,
        description="Standard Test Condition temperature (°C)",
        ge=20.0,
        le=30.0
    )
    temp_coeff: float = Field(
        ...,
        description="Temperature coefficient of power (%/°C)",
        ge=-1.0,
        le=0.0
    )
    technology: Optional[ModuleTechnology] = Field(
        default=None,
        description="Module technology type (overrides temp_coeff if provided)"
    )


class ModuleSpecification(BaseModel):
    """Physical and thermal specifications of a PV module."""
    model_config = ConfigDict(use_enum_values=True)

    area: float = Field(
        ...,
        description="Module area (m²)",
        gt=0.0,
        le=5.0
    )
    thickness: float = Field(
        default=0.04,
        description="Module thickness including frame (m)",
        gt=0.0,
        le=0.1
    )
    mass: float = Field(
        default=20.0,
        description="Module mass (kg)",
        gt=0.0,
        le=50.0
    )
    specific_heat: float = Field(
        default=900.0,
        description="Specific heat capacity (J/kg°C)",
        gt=0.0,
        le=2000.0
    )
    absorptance: float = Field(
        default=0.9,
        description="Solar absorptance (0-1)",
        ge=0.0,
        le=1.0
    )
    emissivity: float = Field(
        default=0.84,
        description="Thermal emissivity (0-1)",
        ge=0.0,
        le=1.0
    )
    technology: ModuleTechnology = Field(
        default=ModuleTechnology.MONO_SI,
        description="Module technology type"
    )
    noct: float = Field(
        default=45.0,
        description="Nominal Operating Cell Temperature (°C)",
        ge=25.0,
        le=65.0
    )
    temp_coeff_power: Optional[float] = Field(
        default=None,
        description="Temperature coefficient of power (%/°C), uses tech default if None",
        ge=-1.0,
        le=0.0
    )


class TemperatureCalculationResult(BaseModel):
    """Result of module temperature calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_temperature: float = Field(
        ...,
        description="Calculated module temperature (°C)"
    )
    temp_above_ambient: float = Field(
        ...,
        description="Temperature rise above ambient (°C)"
    )
    model_used: TemperatureModelType = Field(
        ...,
        description="Temperature model used for calculation"
    )
    mounting_adjustment: float = Field(
        default=0.0,
        description="Temperature adjustment due to mounting (°C)"
    )
    power_loss_factor: Optional[float] = Field(
        default=None,
        description="Power loss factor due to temperature (0-1)"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional calculation metadata"
    )


# ============================================================================
# Main Temperature Model Class
# ============================================================================

class ModuleTemperatureModel:
    """
    Comprehensive module temperature modeling for PV systems.

    This class provides multiple industry-standard methods for calculating
    PV module operating temperatures under various environmental conditions
    and mounting configurations.

    Supported Models:
    - Simple Linear: Basic NOCT-based linear model
    - Ross/Faiman: Empirical model with convective cooling
    - Sandia: Detailed model from Sandia National Laboratories
    - King (SAPM): Sandia Array Performance Model temperature equations

    Example:
        >>> model = ModuleTemperatureModel()
        >>> noct = model.calculate_noct(
        ...     ambient_temp=25.0,
        ...     irradiance=1000.0,
        ...     wind_speed=2.0,
        ...     mounting=MountingType.OPEN_RACK
        ... )
        >>> print(f"NOCT: {noct:.2f}°C")
    """

    def __init__(self):
        """Initialize the module temperature model."""
        self.stefan_boltzmann = 5.67e-8  # W/m²K⁴

    def calculate_noct(
        self,
        ambient_temp: float,
        irradiance: float,
        wind_speed: float,
        mounting: Union[MountingType, str] = MountingType.OPEN_RACK,
        base_noct: float = 45.0
    ) -> float:
        """
        Calculate Nominal Operating Cell Temperature (NOCT).

        NOCT is defined as the module temperature under standard conditions:
        - Irradiance: 800 W/m²
        - Ambient temperature: 20°C
        - Wind speed: 1 m/s
        - Mounting: Open rack

        This method adjusts the base NOCT for different conditions and
        mounting configurations.

        Args:
            ambient_temp: Ambient air temperature (°C)
            irradiance: Plane of array irradiance (W/m²)
            wind_speed: Wind speed at module height (m/s)
            mounting: Mounting configuration type
            base_noct: Base NOCT from manufacturer datasheet (°C)

        Returns:
            Adjusted NOCT value (°C)

        Notes:
            - Higher wind speeds improve cooling (lower NOCT)
            - Different mounting types add thermal resistance
            - Building-integrated systems can be 20-30°C hotter than open rack

        References:
            - IEC 61215: Crystalline silicon terrestrial photovoltaic modules
            - ASTM E1036: Standard Test Methods for Electrical Performance of NOCT
        """
        # Convert string to enum if needed
        if isinstance(mounting, str):
            mounting = MountingType(mounting)

        # Validate inputs
        input_data = NOCTCalculationInput(
            ambient_temp=ambient_temp,
            irradiance=irradiance,
            wind_speed=wind_speed,
            mounting=mounting,
            base_noct=base_noct
        )

        # Calculate temperature rise at standard NOCT conditions
        standard_rise = base_noct - NOCT_STANDARD_AMBIENT

        # Adjust for actual irradiance (linear scaling)
        irradiance_factor = irradiance / NOCT_STANDARD_IRRADIANCE if irradiance > 0 else 0.0

        # Adjust for wind speed (convective cooling)
        # Higher wind speed -> better cooling -> lower temperature
        # Using empirical relationship: wind effect ~ 1/sqrt(wind_speed)
        wind_factor = math.sqrt(NOCT_STANDARD_WIND / max(wind_speed, 0.1))

        # Calculate adjusted temperature rise
        adjusted_rise = standard_rise * irradiance_factor * wind_factor

        # Apply mounting configuration adjustment
        mounting_adj = MOUNTING_THERMAL_ADJUSTMENT.get(mounting, 0.0)

        # Calculate final NOCT
        noct = ambient_temp + adjusted_rise + mounting_adj

        return round(noct, 2)

    def calculate_module_temp(
        self,
        ambient: float,
        irr: float,
        wind: float,
        noct: float,
        mounting: Union[MountingType, str] = MountingType.OPEN_RACK,
        model_type: Union[TemperatureModelType, str] = TemperatureModelType.SIMPLE_LINEAR,
        **kwargs
    ) -> float:
        """
        Calculate module operating temperature using specified model.

        This is the main method for temperature calculation, supporting multiple
        industry-standard models with varying complexity and accuracy.

        Args:
            ambient: Ambient air temperature (°C)
            irr: Plane of array irradiance (W/m²)
            wind: Wind speed at module height (m/s)
            noct: Nominal Operating Cell Temperature (°C)
            mounting: Mounting configuration type
            model_type: Temperature model to use
            **kwargs: Additional model-specific parameters

        Returns:
            Module operating temperature (°C)

        Model Selection Guide:
            - SIMPLE_LINEAR: Fast, reasonable accuracy for most applications
            - ROSS_FAIMAN: Better wind speed modeling, requires coefficients
            - SANDIA: Most accurate, requires detailed module parameters
            - KING_SAPM: Industry standard, good balance of accuracy/complexity
        """
        # Convert string to enum if needed
        if isinstance(mounting, str):
            mounting = MountingType(mounting)
        if isinstance(model_type, str):
            model_type = TemperatureModelType(model_type)

        # Route to appropriate model
        if model_type == TemperatureModelType.SIMPLE_LINEAR:
            temp = self._calculate_simple_linear(ambient, irr, noct)
        elif model_type == TemperatureModelType.ROSS_FAIMAN:
            temp = self._calculate_ross_faiman(ambient, irr, wind, **kwargs)
        elif model_type == TemperatureModelType.SANDIA:
            temp = self._calculate_sandia(ambient, irr, wind, **kwargs)
        elif model_type == TemperatureModelType.KING_SAPM:
            temp = self._calculate_king_sapm(ambient, irr, wind, noct, **kwargs)
        else:
            raise ValueError(f"Unknown temperature model type: {model_type}")

        # Apply mounting configuration adjustment
        mounting_adj = MOUNTING_THERMAL_ADJUSTMENT.get(mounting, 0.0)
        temp += mounting_adj

        return round(temp, 2)

    def _calculate_simple_linear(
        self,
        ambient: float,
        irradiance: float,
        noct: float
    ) -> float:
        """
        Simple linear temperature model.

        Formula: Tmod = Tamb + (NOCT - 20) * (G / 800)

        This is the most basic model, assuming linear relationship between
        irradiance and temperature rise.

        Args:
            ambient: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            noct: NOCT value (°C)

        Returns:
            Module temperature (°C)
        """
        if irradiance <= 0:
            return ambient

        temp_rise_at_noct = noct - NOCT_STANDARD_AMBIENT
        irradiance_ratio = irradiance / NOCT_STANDARD_IRRADIANCE
        temp = ambient + (temp_rise_at_noct * irradiance_ratio)

        return temp

    def _calculate_ross_faiman(
        self,
        ambient: float,
        irradiance: float,
        wind_speed: float,
        ross_a: Optional[float] = None,
        ross_b: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Ross/Faiman temperature model with wind speed effects.

        Formula: Tmod = Tamb + (G / (a + b * v))

        Where:
        - a: Heat loss coefficient at zero wind speed (W/m²°C)
        - b: Heat loss coefficient for wind dependency (W s/m³°C)
        - v: Wind speed (m/s)

        Args:
            ambient: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            ross_a: Coefficient a (default: 25.0 W/m²°C)
            ross_b: Coefficient b (default: 6.84 W s/m³°C)

        Returns:
            Module temperature (°C)

        References:
            - Faiman, D. (2008). "Assessing the outdoor operating temperature of PV modules"
        """
        if irradiance <= 0:
            return ambient

        # Default Ross/Faiman coefficients for typical silicon modules
        a = ross_a if ross_a is not None else 25.0
        b = ross_b if ross_b is not None else 6.84

        # Calculate heat loss coefficient with wind dependency
        heat_loss_coeff = a + (b * wind_speed)

        # Avoid division by zero
        if heat_loss_coeff <= 0:
            heat_loss_coeff = 1.0

        # Calculate module temperature
        temp = ambient + (irradiance / heat_loss_coeff)

        return temp

    def _calculate_sandia(
        self,
        ambient: float,
        irradiance: float,
        wind_speed: float,
        sandia_a: Optional[float] = None,
        sandia_b: Optional[float] = None,
        sandia_deltac: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Sandia National Laboratories temperature model.

        Formula: Tmod = G * exp(a + b*v) + Tamb + G/1000 * DeltaC

        This empirical model was developed from extensive field measurements
        and provides excellent accuracy across various conditions.

        Args:
            ambient: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            sandia_a: Model parameter a (default: -3.47)
            sandia_b: Model parameter b (default: -0.0594)
            sandia_deltac: DeltaT parameter (default: 3.0)

        Returns:
            Module temperature (°C)

        References:
            - King, D.L., et al. (2004). "Sandia Photovoltaic Array Performance Model"
        """
        if irradiance <= 0:
            return ambient

        # Default Sandia parameters for glass/cell/polymer sheet - open rack
        a = sandia_a if sandia_a is not None else -3.47
        b = sandia_b if sandia_b is not None else -0.0594
        deltac = sandia_deltac if sandia_deltac is not None else 3.0

        # Calculate temperature rise due to irradiance and wind
        exp_term = math.exp(a + (b * wind_speed))
        irradiance_effect = irradiance * exp_term

        # Additional correction term
        correction = (irradiance / 1000.0) * deltac

        # Calculate module temperature
        temp = irradiance_effect + ambient + correction

        return temp

    def _calculate_king_sapm(
        self,
        ambient: float,
        irradiance: float,
        wind_speed: float,
        noct: float,
        **kwargs
    ) -> float:
        """
        King model from Sandia Array Performance Model (SAPM).

        This model derives coefficients from NOCT and uses them in a
        temperature calculation that accounts for wind speed effects.

        Args:
            ambient: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            noct: Nominal Operating Cell Temperature (°C)

        Returns:
            Module temperature (°C)

        References:
            - King, D.L., et al. (2004). "Sandia Array Performance Model"
        """
        if irradiance <= 0:
            return ambient

        # Derive a and b from NOCT
        # At NOCT conditions: NOCT = 800 * exp(a + b*1) + 20
        # Typical relationship for open rack mounting
        temp_rise_noct = noct - NOCT_STANDARD_AMBIENT

        # Empirical coefficients derived from NOCT
        # These are approximations for typical modules
        a = math.log(temp_rise_noct / NOCT_STANDARD_IRRADIANCE) + (0.05 * NOCT_STANDARD_WIND)
        b = -0.05

        # Calculate module temperature
        exp_term = math.exp(a + (b * wind_speed))
        temp = (irradiance * exp_term) + ambient

        return temp

    def calculate_temp_coefficient_losses(
        self,
        module_temp: float,
        stc_temp: float = STC_TEMPERATURE,
        temp_coeff: Optional[float] = None,
        technology: Optional[Union[ModuleTechnology, str]] = None
    ) -> float:
        """
        Calculate power loss factor due to temperature effects.

        PV module power output decreases with increasing temperature above STC
        (Standard Test Conditions, 25°C). This method calculates the fractional
        power loss based on temperature coefficient.

        Args:
            module_temp: Module operating temperature (°C)
            stc_temp: Standard Test Condition temperature (°C, default: 25)
            temp_coeff: Temperature coefficient of power (%/°C)
            technology: Module technology (overrides temp_coeff if provided)

        Returns:
            Power loss factor (0-1), where 1.0 = no loss, 0.5 = 50% loss

        Examples:
            >>> model = ModuleTemperatureModel()
            >>> # Mono-Si module at 50°C (25°C above STC)
            >>> loss_factor = model.calculate_temp_coefficient_losses(
            ...     module_temp=50.0,
            ...     technology=ModuleTechnology.MONO_SI
            ... )
            >>> print(f"Power: {loss_factor * 100:.1f}% of STC")
            Power: 90.0% of STC

        Notes:
            - Typical crystalline silicon: -0.35 to -0.45%/°C
            - HJT technology: Better performance at -0.25%/°C
            - Thin film (CdTe): Better performance at -0.25%/°C
        """
        # Convert string to enum if needed
        if isinstance(technology, str):
            technology = ModuleTechnology(technology)

        # Determine temperature coefficient
        if technology is not None:
            temp_coeff_pct = DEFAULT_TEMP_COEFFICIENTS.get(
                technology,
                DEFAULT_TEMP_COEFFICIENTS[ModuleTechnology.MONO_SI]
            )
        elif temp_coeff is not None:
            temp_coeff_pct = temp_coeff
        else:
            # Default to mono-Si if nothing specified
            temp_coeff_pct = DEFAULT_TEMP_COEFFICIENTS[ModuleTechnology.MONO_SI]

        # Validate inputs
        input_data = TemperatureCoefficientInput(
            module_temp=module_temp,
            stc_temp=stc_temp,
            temp_coeff=temp_coeff_pct
        )

        # Calculate temperature difference from STC
        temp_diff = module_temp - stc_temp

        # Calculate power loss (percentage)
        # temp_coeff is negative (e.g., -0.40 %/°C)
        # For temp above STC: power decreases
        # For temp below STC: power increases
        power_loss_pct = temp_coeff_pct * temp_diff

        # Convert to power factor (1.0 = 100%, 0.9 = 90%)
        power_factor = 1.0 + (power_loss_pct / 100.0)

        # Ensure physically reasonable bounds
        power_factor = max(0.0, min(1.5, power_factor))

        return round(power_factor, 4)

    def model_thermal_time_constant(
        self,
        module: Union[ModuleSpecification, Dict]
    ) -> float:
        """
        Calculate thermal time constant for module temperature dynamics.

        The thermal time constant (τ) represents how quickly a module responds
        to changes in environmental conditions. It's crucial for:
        - Transient temperature modeling
        - Thermal stress analysis
        - Short-term performance prediction

        Formula: τ = (m * Cp) / (h * A)

        Where:
        - m: Module mass (kg)
        - Cp: Specific heat capacity (J/kg°C)
        - h: Heat transfer coefficient (W/m²°C)
        - A: Module area (m²)

        Args:
            module: Module specification (ModuleSpecification or dict)

        Returns:
            Thermal time constant (seconds)

        Notes:
            - Typical values: 5-15 minutes for standard modules
            - Lighter modules: Faster response (lower τ)
            - Better ventilation: Faster response (lower τ)
            - Thermal mass affects: Temperature stability

        References:
            - Kurnik, J., et al. (2011). "Outdoor testing of PV module temperature"
        """
        # Convert dict to ModuleSpecification if needed
        if isinstance(module, dict):
            module = ModuleSpecification(**module)

        # Calculate heat transfer coefficient
        # Includes convection (natural and forced) and radiation
        h_conv_natural = 5.0  # W/m²°C (natural convection)
        h_conv_forced = 10.0  # W/m²°C (forced convection, typical wind)

        # Radiative heat transfer coefficient (linearized)
        # h_rad ≈ 4 * ε * σ * T³
        t_kelvin = 273.15 + 45.0  # Typical operating temp
        h_rad = 4.0 * module.emissivity * self.stefan_boltzmann * (t_kelvin ** 3)

        # Total heat transfer coefficient
        h_total = h_conv_natural + h_conv_forced + h_rad

        # Calculate thermal capacitance
        thermal_capacitance = module.mass * module.specific_heat

        # Calculate thermal conductance
        thermal_conductance = h_total * module.area

        # Avoid division by zero
        if thermal_conductance <= 0:
            thermal_conductance = 1.0

        # Calculate time constant
        tau = thermal_capacitance / thermal_conductance

        return round(tau, 2)

    def calculate_seasonal_adjustment(
        self,
        day_of_year: int,
        latitude: float,
        base_ambient: float
    ) -> float:
        """
        Calculate seasonal ambient temperature adjustment.

        Estimates daily ambient temperature variation based on latitude and
        time of year using sinusoidal approximation.

        Args:
            day_of_year: Day of year (1-365)
            latitude: Site latitude (degrees, -90 to 90)
            base_ambient: Annual average ambient temperature (°C)

        Returns:
            Adjusted ambient temperature (°C)

        Notes:
            - Northern hemisphere: Peak in summer (day ~200)
            - Southern hemisphere: Peak in winter (day ~20)
            - Higher latitudes: Greater seasonal variation
        """
        # Validate inputs
        day_of_year = max(1, min(365, day_of_year))
        latitude = max(-90.0, min(90.0, latitude))

        # Seasonal amplitude (°C) varies with latitude
        # Higher latitudes have greater seasonal swings
        amplitude = 10.0 * (abs(latitude) / 45.0)

        # Phase shift for Northern vs Southern hemisphere
        if latitude >= 0:
            # Northern: warmest around day 200 (mid-July)
            phase_shift = 200
        else:
            # Southern: warmest around day 20 (mid-January)
            phase_shift = 20

        # Calculate seasonal adjustment using sinusoid
        angle = 2.0 * math.pi * (day_of_year - phase_shift) / 365.0
        seasonal_adjustment = amplitude * math.sin(angle)

        adjusted_temp = base_ambient + seasonal_adjustment

        return round(adjusted_temp, 2)

    def calculate_time_of_day_adjustment(
        self,
        hour: float,
        daily_amplitude: float = 8.0
    ) -> float:
        """
        Calculate time-of-day temperature adjustment.

        Models diurnal temperature variation using sinusoidal approximation.
        Peak temperature typically occurs around 14:00-15:00 local solar time.

        Args:
            hour: Hour of day (0-24, decimal allowed)
            daily_amplitude: Daily temperature swing amplitude (°C, default: 8)

        Returns:
            Temperature adjustment relative to daily average (°C)

        Notes:
            - Peak temperature: ~14:00-15:00 (2-3 PM)
            - Minimum temperature: ~05:00-06:00 (5-6 AM)
            - Typical amplitude: 5-12°C depending on climate
        """
        # Validate hour
        hour = hour % 24.0

        # Peak temperature occurs around 15:00 (3 PM)
        peak_hour = 15.0

        # Calculate phase (radians)
        phase = 2.0 * math.pi * (hour - peak_hour) / 24.0

        # Calculate adjustment (sinusoidal)
        adjustment = daily_amplitude * math.sin(phase)

        return round(adjustment, 2)

    def calculate_comprehensive_temperature(
        self,
        ambient_base: float,
        irradiance: float,
        wind_speed: float,
        module_spec: Union[ModuleSpecification, Dict],
        day_of_year: int = 180,
        hour: float = 12.0,
        latitude: float = 35.0,
        mounting: Union[MountingType, str] = MountingType.OPEN_RACK,
        model_type: Union[TemperatureModelType, str] = TemperatureModelType.SIMPLE_LINEAR
    ) -> TemperatureCalculationResult:
        """
        Comprehensive temperature calculation with all adjustments.

        This method combines multiple factors:
        - Seasonal temperature variation
        - Time-of-day effects
        - NOCT-based or advanced temperature modeling
        - Mounting configuration adjustments
        - Temperature coefficient losses

        Args:
            ambient_base: Base ambient temperature (°C)
            irradiance: Plane of array irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            module_spec: Module specifications
            day_of_year: Day of year (1-365)
            hour: Hour of day (0-24)
            latitude: Site latitude (degrees)
            mounting: Mounting configuration
            model_type: Temperature calculation model

        Returns:
            TemperatureCalculationResult with complete information
        """
        # Convert dict to ModuleSpecification if needed
        if isinstance(module_spec, dict):
            module_spec = ModuleSpecification(**module_spec)

        # Convert strings to enums if needed
        if isinstance(mounting, str):
            mounting = MountingType(mounting)
        if isinstance(model_type, str):
            model_type = TemperatureModelType(model_type)

        # Calculate seasonal adjustment
        seasonal_adj = self.calculate_seasonal_adjustment(
            day_of_year, latitude, ambient_base
        )

        # Calculate time-of-day adjustment
        tod_adj = self.calculate_time_of_day_adjustment(hour)

        # Combined ambient temperature
        ambient_adjusted = seasonal_adj + tod_adj

        # Calculate module temperature
        module_temp = self.calculate_module_temp(
            ambient=ambient_adjusted,
            irr=irradiance,
            wind=wind_speed,
            noct=module_spec.noct,
            mounting=mounting,
            model_type=model_type
        )

        # Calculate temperature coefficient losses
        temp_coeff = (
            module_spec.temp_coeff_power
            if module_spec.temp_coeff_power is not None
            else DEFAULT_TEMP_COEFFICIENTS[module_spec.technology]
        )

        power_loss_factor = self.calculate_temp_coefficient_losses(
            module_temp=module_temp,
            temp_coeff=temp_coeff
        )

        # Calculate mounting adjustment
        mounting_adj = MOUNTING_THERMAL_ADJUSTMENT.get(mounting, 0.0)

        # Create result
        result = TemperatureCalculationResult(
            module_temperature=module_temp,
            temp_above_ambient=module_temp - ambient_adjusted,
            model_used=model_type,
            mounting_adjustment=mounting_adj,
            power_loss_factor=power_loss_factor,
            metadata={
                "ambient_base": ambient_base,
                "ambient_adjusted": ambient_adjusted,
                "seasonal_adjustment": seasonal_adj - ambient_base,
                "tod_adjustment": tod_adj,
                "irradiance": irradiance,
                "wind_speed": wind_speed,
                "noct": module_spec.noct,
                "temp_coefficient": temp_coeff,
                "day_of_year": day_of_year,
                "hour": hour,
                "latitude": latitude,
                "technology": module_spec.technology,
            }
        )

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_default_temp_coefficient(
    technology: Union[ModuleTechnology, str]
) -> float:
    """
    Get default temperature coefficient for a module technology.

    Args:
        technology: Module technology type

    Returns:
        Temperature coefficient (%/°C)
    """
    if isinstance(technology, str):
        technology = ModuleTechnology(technology)

    return DEFAULT_TEMP_COEFFICIENTS.get(
        technology,
        DEFAULT_TEMP_COEFFICIENTS[ModuleTechnology.MONO_SI]
    )


def estimate_noct_from_mounting(
    base_noct: float,
    mounting: Union[MountingType, str]
) -> float:
    """
    Estimate NOCT adjustment based on mounting configuration.

    Args:
        base_noct: Base NOCT for open rack (°C)
        mounting: Mounting configuration

    Returns:
        Adjusted NOCT (°C)
    """
    if isinstance(mounting, str):
        mounting = MountingType(mounting)

    adjustment = MOUNTING_THERMAL_ADJUSTMENT.get(mounting, 0.0)
    return base_noct + adjustment


def calculate_power_at_temperature(
    stc_power: float,
    module_temp: float,
    temp_coeff: float,
    stc_temp: float = STC_TEMPERATURE
) -> float:
    """
    Calculate actual power output accounting for temperature effects.

    Args:
        stc_power: Rated power at STC (W)
        module_temp: Module operating temperature (°C)
        temp_coeff: Temperature coefficient (%/°C)
        stc_temp: STC temperature (°C)

    Returns:
        Temperature-adjusted power (W)
    """
    model = ModuleTemperatureModel()
    loss_factor = model.calculate_temp_coefficient_losses(
        module_temp=module_temp,
        stc_temp=stc_temp,
        temp_coeff=temp_coeff
    )

    return stc_power * loss_factor


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    "ModuleTemperatureModel",

    # Enums
    "MountingType",
    "TemperatureModelType",
    "ModuleTechnology",

    # Pydantic models
    "NOCTCalculationInput",
    "ModuleTemperatureInput",
    "TemperatureCoefficientInput",
    "ModuleSpecification",
    "TemperatureCalculationResult",

    # Utility functions
    "get_default_temp_coefficient",
    "estimate_noct_from_mounting",
    "calculate_power_at_temperature",

    # Constants
    "NOCT_STANDARD_IRRADIANCE",
    "NOCT_STANDARD_AMBIENT",
    "NOCT_STANDARD_WIND",
    "STC_TEMPERATURE",
    "MOUNTING_THERMAL_ADJUSTMENT",
    "DEFAULT_TEMP_COEFFICIENTS",
]
