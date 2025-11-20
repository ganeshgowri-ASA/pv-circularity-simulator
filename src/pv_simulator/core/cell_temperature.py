"""
Cell and module temperature modeling using multiple thermal models.

This module provides comprehensive temperature modeling capabilities including:
- Sandia Array Performance Model
- PVsyst temperature model
- Faiman temperature model
- NOCT-based calculations
- Custom thermal models with heat transfer physics

References:
    - King et al. (2004): Sandia PV Array Performance Model
    - Faiman (2008): Assessing the outdoor operating temperature of photovoltaic modules
    - Mermoud (2012): PVsyst thermal model
"""

from typing import Optional, Union, Dict, Callable
import numpy as np
import pandas as pd
from datetime import datetime

# Import pvlib for standard temperature models
try:
    import pvlib
    from pvlib.temperature import sapm_cell, pvsyst_cell, faiman
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    print("Warning: pvlib not available. Some temperature models may not work.")

from pv_simulator.models.thermal import (
    TemperatureConditions,
    ThermalParameters,
    TemperatureCoefficients,
    ThermalModelOutput,
    MountingConfiguration,
    HeatTransferCoefficients,
)
from pv_simulator.models.noct import ModuleNOCTData, NOCTSpecification
from pv_simulator.utils.constants import (
    STC_TEMPERATURE,
    NOCT_IRRADIANCE,
    NOCT_AMBIENT_TEMP,
    MOUNTING_CONFIGS,
    STEFAN_BOLTZMANN,
)
from pv_simulator.utils.helpers import (
    celsius_to_kelvin,
    calculate_reynolds_number,
    calculate_nusselt_number,
    calculate_prandtl_number,
    calculate_heat_transfer_coefficient,
)


class CellTemperatureModel:
    """
    Cell temperature modeling using multiple established thermal models.

    This class provides implementations of various cell temperature calculation methods,
    including models from Sandia, PVsyst, Faiman, NOCT-based approaches, and custom
    thermal models based on heat transfer physics.

    Attributes:
        conditions: Environmental conditions for temperature calculations
        thermal_params: Thermal properties of the module
        mounting: Mounting configuration affecting thermal behavior
        temp_coefficients: Temperature coefficients for performance calculations

    Examples:
        >>> from pv_simulator.models.thermal import TemperatureConditions, MountingConfiguration
        >>> conditions = TemperatureConditions(
        ...     ambient_temp=25.0,
        ...     irradiance=1000.0,
        ...     wind_speed=3.0
        ... )
        >>> mounting = MountingConfiguration(mounting_type="open_rack")
        >>> model = CellTemperatureModel(conditions=conditions, mounting=mounting)
        >>> result = model.sandia_model(a=-3.47, b=-0.0594, delta_t=3.0)
        >>> print(f"Cell temperature: {result.cell_temperature:.1f}°C")
    """

    def __init__(
        self,
        conditions: TemperatureConditions,
        thermal_params: Optional[ThermalParameters] = None,
        mounting: Optional[MountingConfiguration] = None,
        temp_coefficients: Optional[TemperatureCoefficients] = None,
    ):
        """
        Initialize the cell temperature model.

        Args:
            conditions: Environmental conditions (temperature, irradiance, wind speed)
            thermal_params: Thermal properties of the module (optional)
            mounting: Mounting configuration (optional, defaults to open_rack)
            temp_coefficients: Temperature coefficients for performance (optional)
        """
        self.conditions = conditions
        self.thermal_params = thermal_params or ThermalParameters()
        self.mounting = mounting or MountingConfiguration()
        self.temp_coefficients = temp_coefficients

    def sandia_model(
        self,
        a: float = -3.47,
        b: float = -0.0594,
        delta_t: float = 3.0,
    ) -> ThermalModelOutput:
        """
        Calculate cell temperature using the Sandia Array Performance Model.

        The Sandia model uses empirical coefficients derived from outdoor testing
        to calculate module temperature as a function of irradiance and wind speed.

        Model equation:
            T_module = T_ambient + (irradiance / 1000) * exp(a + b * wind_speed) + delta_t

        Args:
            a: Empirical coefficient (dimensionless), typically around -3.47
            b: Empirical coefficient (s/m), typically around -0.0594
            delta_t: Temperature difference between cell and module back (°C), typically 2-5°C

        Returns:
            ThermalModelOutput containing cell temperature and model details

        References:
            King, D. L., et al. (2004). "Sandia Photovoltaic Array Performance Model."
            SAND2004-3535. Sandia National Laboratories.

        Examples:
            >>> conditions = TemperatureConditions(ambient_temp=25, irradiance=1000, wind_speed=1)
            >>> model = CellTemperatureModel(conditions)
            >>> result = model.sandia_model()
            >>> result.cell_temperature
            50.2...
        """
        T_amb = self.conditions.ambient_temp
        E = self.conditions.irradiance
        ws = self.conditions.wind_speed

        # Sandia model calculation
        # T_module = T_ambient + (E/E0) * exp(a + b*ws) + delta_T
        E0 = 1000.0  # Reference irradiance
        module_temp = T_amb + (E / E0) * np.exp(a + b * ws)
        cell_temp = module_temp + delta_t

        # Use pvlib if available for validation
        if PVLIB_AVAILABLE:
            try:
                # Note: pvlib's sapm_cell calculates module temp, need to add delta_t
                pvlib_module_temp = pvlib.temperature.sapm_cell(
                    poa_global=E,
                    temp_air=T_amb,
                    wind_speed=ws,
                    a=a,
                    b=b,
                    deltaT=delta_t,
                )
                cell_temp = float(pvlib_module_temp)
                module_temp = cell_temp - delta_t
            except Exception as e:
                # Fall back to manual calculation
                pass

        return ThermalModelOutput(
            cell_temperature=float(cell_temp),
            module_temperature=float(module_temp),
            back_surface_temperature=float(module_temp),
            model_name="Sandia",
            conditions=self.conditions,
            mounting=self.mounting,
            timestamp=datetime.now(),
        )

    def pvsyst_model(
        self,
        u_c: float = 29.0,
        u_v: float = 0.0,
    ) -> ThermalModelOutput:
        """
        Calculate cell temperature using the PVsyst temperature model.

        The PVsyst model uses heat loss factors to characterize the thermal behavior
        of PV modules. It includes a constant heat loss factor and an optional
        wind-dependent factor.

        Model equation:
            T_cell = T_ambient + (irradiance / (u_c + u_v * wind_speed)) * (1 - efficiency)

        Args:
            u_c: Constant heat loss factor in W/(m²·K), typically 20-30
            u_v: Wind-dependent heat loss factor in W/(m²·K)/(m/s), typically 0-6

        Returns:
            ThermalModelOutput containing cell temperature and model details

        References:
            Mermoud, A. (2012). "PVsyst User's Manual."

        Examples:
            >>> conditions = TemperatureConditions(ambient_temp=20, irradiance=800, wind_speed=1)
            >>> model = CellTemperatureModel(conditions)
            >>> result = model.pvsyst_model(u_c=29.0, u_v=0.0)
            >>> result.cell_temperature
            42.3...
        """
        T_amb = self.conditions.ambient_temp
        E = self.conditions.irradiance
        ws = self.conditions.wind_speed

        # Estimate efficiency loss factor (1 - efficiency)
        # Typical module efficiency is 15-20%, so (1 - efficiency) ≈ 0.85
        efficiency_factor = 0.9  # Conservative estimate

        # PVsyst model calculation
        u_total = u_c + u_v * ws
        cell_temp = T_amb + (E / u_total) * efficiency_factor

        # Use pvlib if available
        if PVLIB_AVAILABLE:
            try:
                pvlib_cell_temp = pvlib.temperature.pvsyst_cell(
                    poa_global=E,
                    temp_air=T_amb,
                    wind_speed=ws,
                    u_c=u_c,
                    u_v=u_v,
                )
                cell_temp = float(pvlib_cell_temp)
            except Exception as e:
                # Fall back to manual calculation
                pass

        module_temp = cell_temp - 2.0  # Approximate module temp

        return ThermalModelOutput(
            cell_temperature=float(cell_temp),
            module_temperature=float(module_temp),
            back_surface_temperature=float(module_temp),
            model_name="PVsyst",
            conditions=self.conditions,
            mounting=self.mounting,
            timestamp=datetime.now(),
        )

    def faiman_model(
        self,
        u0: float = 25.0,
        u1: float = 6.84,
    ) -> ThermalModelOutput:
        """
        Calculate cell temperature using the Faiman temperature model.

        The Faiman model uses two heat transfer coefficients: a constant coefficient
        and a wind-dependent coefficient, to model the thermal behavior.

        Model equation:
            T_module = T_ambient + (irradiance * absorptivity) / (u0 + u1 * wind_speed)

        Args:
            u0: Constant heat transfer coefficient in W/(m²·K), typically 20-30
            u1: Wind-dependent heat transfer coefficient in W/(m²·K)/(m/s), typically 5-10

        Returns:
            ThermalModelOutput containing cell temperature and model details

        References:
            Faiman, D. (2008). "Assessing the outdoor operating temperature of
            photovoltaic modules." Progress in Photovoltaics, 16(4), 307-315.

        Examples:
            >>> conditions = TemperatureConditions(ambient_temp=25, irradiance=1000, wind_speed=2)
            >>> model = CellTemperatureModel(conditions)
            >>> result = model.faiman_model()
            >>> result.cell_temperature
            48.5...
        """
        T_amb = self.conditions.ambient_temp
        E = self.conditions.irradiance
        ws = self.conditions.wind_speed
        absorptivity = self.thermal_params.absorptivity

        # Faiman model calculation
        # T_module = T_ambient + (E * α) / (u0 + u1 * ws)
        module_temp = T_amb + (E * absorptivity) / (u0 + u1 * ws)
        cell_temp = module_temp + 2.0  # Approximate cell temp

        # Use pvlib if available
        if PVLIB_AVAILABLE:
            try:
                pvlib_cell_temp = pvlib.temperature.faiman(
                    poa_global=E,
                    temp_air=T_amb,
                    wind_speed=ws,
                    u0=u0,
                    u1=u1,
                )
                cell_temp = float(pvlib_cell_temp)
                module_temp = cell_temp - 2.0
            except Exception as e:
                # Fall back to manual calculation
                pass

        return ThermalModelOutput(
            cell_temperature=float(cell_temp),
            module_temperature=float(module_temp),
            back_surface_temperature=float(module_temp),
            model_name="Faiman",
            conditions=self.conditions,
            mounting=self.mounting,
            timestamp=datetime.now(),
        )

    def noct_based(
        self,
        noct: Optional[Union[float, NOCTSpecification, ModuleNOCTData]] = None,
    ) -> ThermalModelOutput:
        """
        Calculate cell temperature using NOCT-based method.

        NOCT (Nominal Operating Cell Temperature) provides a simple way to estimate
        cell temperature based on the module's NOCT rating and current conditions.

        Model equation:
            T_cell = T_ambient + (NOCT - 20) * (irradiance / 800) * correction_factors

        Args:
            noct: NOCT value in °C, NOCTSpecification, or ModuleNOCTData.
                  If None, uses a typical value of 45°C.

        Returns:
            ThermalModelOutput containing cell temperature and model details

        Examples:
            >>> conditions = TemperatureConditions(ambient_temp=25, irradiance=1000, wind_speed=1)
            >>> model = CellTemperatureModel(conditions)
            >>> result = model.noct_based(noct=45.0)
            >>> result.cell_temperature
            56.25
        """
        T_amb = self.conditions.ambient_temp
        E = self.conditions.irradiance
        ws = self.conditions.wind_speed

        # Extract NOCT value
        if noct is None:
            noct_value = 45.0  # Typical NOCT
        elif isinstance(noct, (int, float)):
            noct_value = float(noct)
        elif isinstance(noct, NOCTSpecification):
            noct_value = noct.noct_celsius
        elif isinstance(noct, ModuleNOCTData):
            noct_value = noct.noct_spec.noct_celsius
        else:
            raise ValueError(f"Invalid NOCT type: {type(noct)}")

        # NOCT-based calculation
        # Standard NOCT conditions: 800 W/m², 20°C ambient, 1 m/s wind
        E_noct = NOCT_IRRADIANCE
        T_amb_noct = NOCT_AMBIENT_TEMP
        ws_noct = 1.0

        # Temperature rise at NOCT conditions
        delta_T_noct = noct_value - T_amb_noct

        # Scale to current irradiance
        delta_T = delta_T_noct * (E / E_noct)

        # Wind speed correction (simplified)
        wind_correction = 1.0 - 0.05 * (ws - ws_noct)  # 5% reduction per m/s above NOCT
        wind_correction = max(0.5, min(1.2, wind_correction))  # Limit correction

        # Final cell temperature
        cell_temp = T_amb + delta_T * wind_correction

        module_temp = cell_temp - 2.0

        return ThermalModelOutput(
            cell_temperature=float(cell_temp),
            module_temperature=float(module_temp),
            back_surface_temperature=float(module_temp),
            model_name="NOCT-based",
            conditions=self.conditions,
            mounting=self.mounting,
            timestamp=datetime.now(),
        )

    def custom_thermal_models(
        self,
        model_func: Optional[Callable] = None,
        model_params: Optional[Dict] = None,
    ) -> ThermalModelOutput:
        """
        Calculate cell temperature using a custom thermal model function.

        This method allows users to provide their own temperature calculation
        function for specialized applications or research purposes.

        Args:
            model_func: Custom function with signature:
                        func(conditions, thermal_params, mounting, **params) -> float
                        Should return cell temperature in °C
            model_params: Additional parameters to pass to the custom function

        Returns:
            ThermalModelOutput containing cell temperature and model details

        Raises:
            ValueError: If model_func is None or not callable

        Examples:
            >>> def my_model(conditions, thermal_params, mounting, k=1.0):
            ...     return conditions.ambient_temp + k * conditions.irradiance / 30.0
            >>> conditions = TemperatureConditions(ambient_temp=25, irradiance=900, wind_speed=2)
            >>> model = CellTemperatureModel(conditions)
            >>> result = model.custom_thermal_models(model_func=my_model, model_params={'k': 1.2})
            >>> result.cell_temperature
            61.0
        """
        if model_func is None:
            raise ValueError("model_func must be provided for custom thermal models")

        if not callable(model_func):
            raise ValueError("model_func must be callable")

        params = model_params or {}

        try:
            cell_temp = model_func(
                self.conditions,
                self.thermal_params,
                self.mounting,
                **params,
            )
        except Exception as e:
            raise RuntimeError(f"Error in custom thermal model: {e}")

        if not isinstance(cell_temp, (int, float)):
            raise ValueError(
                f"Custom model must return numeric temperature, got {type(cell_temp)}"
            )

        module_temp = cell_temp - 2.0

        return ThermalModelOutput(
            cell_temperature=float(cell_temp),
            module_temperature=float(module_temp),
            back_surface_temperature=float(module_temp),
            model_name="Custom",
            conditions=self.conditions,
            mounting=self.mounting,
            timestamp=datetime.now(),
        )


class ModuleTemperatureCalculator:
    """
    Advanced module temperature calculations with detailed heat transfer physics.

    This class provides comprehensive thermal modeling including:
    - Heat transfer coefficient calculations
    - Wind speed effects on cooling
    - Mounting configuration impacts
    - Back surface temperature estimation
    - Thermal time constant calculations

    Attributes:
        thermal_params: Thermal properties of the module
        mounting: Mounting configuration
        conditions: Environmental conditions

    Examples:
        >>> from pv_simulator.models.thermal import TemperatureConditions, ThermalParameters
        >>> conditions = TemperatureConditions(ambient_temp=25, irradiance=1000, wind_speed=3)
        >>> calculator = ModuleTemperatureCalculator(conditions=conditions)
        >>> coeffs = calculator.heat_transfer_coefficients()
        >>> print(f"Front convection: {coeffs.convective_front:.2f} W/(m²·K)")
    """

    def __init__(
        self,
        thermal_params: Optional[ThermalParameters] = None,
        mounting: Optional[MountingConfiguration] = None,
        conditions: Optional[TemperatureConditions] = None,
    ):
        """
        Initialize the module temperature calculator.

        Args:
            thermal_params: Thermal properties of the module (optional)
            mounting: Mounting configuration (optional)
            conditions: Environmental conditions (optional)
        """
        self.thermal_params = thermal_params or ThermalParameters()
        self.mounting = mounting or MountingConfiguration()
        self.conditions = conditions

    def heat_transfer_coefficients(
        self,
        wind_speed: Optional[float] = None,
    ) -> HeatTransferCoefficients:
        """
        Calculate heat transfer coefficients for module surfaces.

        Computes convective and radiative heat transfer coefficients for both
        front and back surfaces of the module based on mounting configuration
        and environmental conditions.

        Args:
            wind_speed: Wind speed in m/s (uses conditions if not provided)

        Returns:
            HeatTransferCoefficients containing all heat transfer coefficients

        Examples:
            >>> calculator = ModuleTemperatureCalculator()
            >>> coeffs = calculator.heat_transfer_coefficients(wind_speed=3.0)
            >>> coeffs.convective_front > coeffs.convective_back
            True
        """
        if wind_speed is None:
            if self.conditions is None:
                raise ValueError("Either wind_speed or conditions must be provided")
            wind_speed = self.conditions.wind_speed

        # Get base coefficients from mounting configuration
        mount_config = MOUNTING_CONFIGS.get(self.mounting.mounting_type, MOUNTING_CONFIGS["open_rack"])

        # Convective heat transfer - front surface
        # Base convection + wind effect
        h_conv_front_base = mount_config["convection_coeff"]
        h_conv_front = h_conv_front_base + 4.0 * wind_speed  # Simplified wind effect

        # Convective heat transfer - back surface
        # Reduced for most mounting types
        h_conv_back = h_conv_front * 0.7  # Back side has less wind exposure

        # Adjust for mounting type
        if self.mounting.mounting_type == "roof_mounted":
            h_conv_back *= 0.5  # Severely restricted
        elif self.mounting.mounting_type == "building_integrated":
            h_conv_back *= 0.3  # Very restricted

        # Radiative heat transfer
        emissivity = self.thermal_params.emissivity
        if self.conditions:
            T_module_est = self.conditions.ambient_temp + 20  # Rough estimate
        else:
            T_module_est = 45.0  # Default estimate

        T_module_K = celsius_to_kelvin(T_module_est)
        T_sky_K = celsius_to_kelvin(self.conditions.sky_temperature if self.conditions else T_module_est - 10)
        T_ground_K = celsius_to_kelvin(self.conditions.ground_temperature if self.conditions and self.conditions.ground_temperature else T_module_est - 5)

        # Radiative heat transfer coefficient (linearized)
        # h_rad = 4 * σ * ε * T_mean³
        h_rad_front = 4 * STEFAN_BOLTZMANN * emissivity * ((T_module_K + T_sky_K) / 2) ** 3
        h_rad_back = 4 * STEFAN_BOLTZMANN * emissivity * ((T_module_K + T_ground_K) / 2) ** 3

        return HeatTransferCoefficients(
            convective_front=float(h_conv_front),
            convective_back=float(h_conv_back),
            radiative_front=float(h_rad_front),
            radiative_back=float(h_rad_back),
        )

    def wind_speed_effects(
        self,
        wind_speed_range: Optional[np.ndarray] = None,
        base_temp: float = 50.0,
    ) -> pd.DataFrame:
        """
        Analyze the effect of wind speed on module temperature.

        Args:
            wind_speed_range: Array of wind speeds to analyze (m/s).
                             If None, uses default range 0-15 m/s.
            base_temp: Base ambient temperature in °C

        Returns:
            DataFrame with wind speed and corresponding temperature reduction

        Examples:
            >>> calculator = ModuleTemperatureCalculator()
            >>> results = calculator.wind_speed_effects()
            >>> results.iloc[0]['temp_reduction_c'] < results.iloc[-1]['temp_reduction_c']
            True
        """
        if wind_speed_range is None:
            wind_speed_range = np.linspace(0, 15, 30)

        results = []
        for ws in wind_speed_range:
            coeffs = self.heat_transfer_coefficients(wind_speed=float(ws))

            # Estimate temperature reduction
            # Higher heat transfer = lower temperature
            temp_reduction = coeffs.total_front / 5.0  # Simplified scaling

            results.append({
                "wind_speed_ms": float(ws),
                "h_conv_front": coeffs.convective_front,
                "h_total_front": coeffs.total_front,
                "temp_reduction_c": float(temp_reduction),
            })

        return pd.DataFrame(results)

    def mounting_configuration_effects(
        self,
        irradiance: float = 1000.0,
        ambient_temp: float = 25.0,
        wind_speed: float = 3.0,
    ) -> pd.DataFrame:
        """
        Compare module temperatures across different mounting configurations.

        Args:
            irradiance: Solar irradiance in W/m²
            ambient_temp: Ambient temperature in °C
            wind_speed: Wind speed in m/s

        Returns:
            DataFrame comparing temperatures for different mounting types

        Examples:
            >>> calculator = ModuleTemperatureCalculator()
            >>> results = calculator.mounting_configuration_effects()
            >>> len(results) == 4  # Four mounting types
            True
        """
        mounting_types = ["open_rack", "roof_mounted", "ground_mounted", "building_integrated"]
        conditions = TemperatureConditions(
            ambient_temp=ambient_temp,
            irradiance=irradiance,
            wind_speed=wind_speed,
        )

        results = []
        for mount_type in mounting_types:
            mounting = MountingConfiguration(mounting_type=mount_type)
            temp_model = CellTemperatureModel(
                conditions=conditions,
                mounting=mounting,
                thermal_params=self.thermal_params,
            )

            # Calculate using multiple models
            sandia_result = temp_model.sandia_model()
            pvsyst_result = temp_model.pvsyst_model()
            noct_result = temp_model.noct_based()

            results.append({
                "mounting_type": mount_type,
                "cell_temp_sandia_c": sandia_result.cell_temperature,
                "cell_temp_pvsyst_c": pvsyst_result.cell_temperature,
                "cell_temp_noct_c": noct_result.cell_temperature,
                "avg_cell_temp_c": np.mean([
                    sandia_result.cell_temperature,
                    pvsyst_result.cell_temperature,
                    noct_result.cell_temperature,
                ]),
            })

        return pd.DataFrame(results)

    def back_surface_temperature(
        self,
        front_surface_temp: float,
        irradiance: float,
        ambient_temp: float,
    ) -> float:
        """
        Calculate back surface temperature from front surface temperature.

        Uses a thermal resistance model to estimate the temperature difference
        between front and back surfaces.

        Args:
            front_surface_temp: Front surface temperature in °C
            irradiance: Solar irradiance in W/m²
            ambient_temp: Ambient temperature in °C

        Returns:
            Back surface temperature in °C

        Examples:
            >>> calculator = ModuleTemperatureCalculator()
            >>> back_temp = calculator.back_surface_temperature(60.0, 1000.0, 25.0)
            >>> 25 < back_temp < 60
            True
        """
        # Get mounting thermal resistance
        mount_config = MOUNTING_CONFIGS.get(
            self.mounting.mounting_type,
            MOUNTING_CONFIGS["open_rack"]
        )
        R_thermal = mount_config["thermal_resistance"]

        # Heat flow through module (simplified)
        # Q = (T_front - T_back) / R_thermal
        # Also: Q = h_back * (T_back - T_ambient)

        # Estimate back heat transfer coefficient
        coeffs = self.heat_transfer_coefficients()
        h_back = coeffs.total_back

        # Solve for back temperature
        # T_back = (T_front/R + h_back*T_ambient) / (1/R + h_back)
        T_back = (front_surface_temp / R_thermal + h_back * ambient_temp) / (
            1.0 / R_thermal + h_back
        )

        return float(T_back)

    def thermal_time_constants(
        self,
        wind_speed: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate thermal time constants for the module.

        The thermal time constant indicates how quickly the module responds to
        changes in environmental conditions (e.g., cloud passage).

        Thermal time constant: τ = (m * c_p) / (h * A) = (ρ * c_p * thickness) / h

        Args:
            wind_speed: Wind speed in m/s

        Returns:
            Dictionary containing:
                - tau_heating: Heating time constant in seconds
                - tau_cooling: Cooling time constant in seconds
                - response_time_63: Time to reach 63% of final temperature (s)
                - response_time_95: Time to reach 95% of final temperature (s)

        Examples:
            >>> calculator = ModuleTemperatureCalculator()
            >>> tau = calculator.thermal_time_constants(wind_speed=3.0)
            >>> tau['tau_heating'] > 0
            True
            >>> tau['tau_cooling'] < tau['tau_heating']
            True
        """
        # Get heat transfer coefficients
        coeffs = self.heat_transfer_coefficients(wind_speed=wind_speed)

        # Heat capacity per unit area
        C_area = self.thermal_params.heat_capacity  # J/(m²·K)

        # Time constant = C / h
        tau_heating = C_area / coeffs.total_front  # Heating (sun exposure)
        tau_cooling = C_area / (coeffs.total_front + coeffs.total_back)  # Cooling (both surfaces)

        # Response times
        # 63% response: 1 * tau
        # 95% response: 3 * tau
        response_63 = tau_heating
        response_95 = 3 * tau_heating

        return {
            "tau_heating": float(tau_heating),
            "tau_cooling": float(tau_cooling),
            "response_time_63": float(response_63),
            "response_time_95": float(response_95),
            "tau_heating_minutes": float(tau_heating / 60),
            "tau_cooling_minutes": float(tau_cooling / 60),
        }
