"""
Module Degradation Modeling & Lifetime Prediction

This module provides comprehensive modeling of photovoltaic module degradation
mechanisms and lifetime prediction capabilities. It includes:

- Multiple degradation mechanisms (LID, LETID, PID, UV, thermal, mechanical)
- Environmental stress factor modeling
- Technology-specific degradation rates
- Linear and non-linear lifetime prediction models
- Monte Carlo statistical analysis
- Warranty compliance checking
- Integration with financial models

Author: PV Circularity Simulator
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
import math
from pydantic import BaseModel, Field, validator, field_validator
import numpy as np
from datetime import datetime


class ModuleType(str, Enum):
    """Supported photovoltaic module technologies."""
    MONO_PERC = "mono_perc"
    POLY = "poly"
    TOPCON = "topcon"
    HJT = "hjt"
    THIN_FILM_CDTE = "thin_film_cdte"
    THIN_FILM_CIGS = "thin_film_cigs"
    BIFACIAL_PERC = "bifacial_perc"
    BIFACIAL_TOPCON = "bifacial_topcon"


class DegradationMode(str, Enum):
    """Types of degradation mechanisms."""
    LID = "lid"  # Light Induced Degradation
    LETID = "letid"  # Light and Elevated Temperature Induced Degradation
    PID = "pid"  # Potential Induced Degradation
    UV = "uv"  # UV-induced encapsulant degradation
    THERMAL_CYCLING = "thermal_cycling"  # Solder fatigue, interconnect
    MECHANICAL = "mechanical"  # Micro-cracks from wind/snow
    CORROSION = "corrosion"  # Humidity-induced corrosion


class LifetimeModelType(str, Enum):
    """Lifetime prediction model types."""
    LINEAR = "linear"
    NON_LINEAR = "non_linear"
    MONTE_CARLO = "monte_carlo"
    IEC_61215 = "iec_61215"


class EnvironmentalStressFactors(BaseModel):
    """Environmental stress factors affecting module degradation.

    Attributes:
        temperature_cycling_amplitude: Daily temperature variation (°C)
        temperature_cycling_frequency: Number of thermal cycles per year
        avg_ambient_temperature: Average ambient temperature (°C)
        max_module_temperature: Maximum module operating temperature (°C)
        relative_humidity_avg: Average relative humidity (%)
        damp_heat_hours: Cumulative hours at >85°C and >85% RH
        uv_exposure_dose: Annual UV radiation dose (kWh/m²/year)
        system_voltage: DC system voltage (V)
        soiling_accumulation_rate: Monthly soiling loss rate (%)
        wind_speed_avg: Average wind speed (m/s)
        snow_load_events: Number of significant snow events per year
        hail_exposure_risk: Hail risk factor (0-1, 0=none, 1=extreme)
    """
    temperature_cycling_amplitude: float = Field(
        ge=0, le=100, description="Daily temperature variation in °C"
    )
    temperature_cycling_frequency: int = Field(
        ge=0, le=365, description="Number of thermal cycles per year"
    )
    avg_ambient_temperature: float = Field(
        ge=-50, le=70, description="Average ambient temperature in °C"
    )
    max_module_temperature: float = Field(
        ge=0, le=120, description="Maximum module operating temperature in °C"
    )
    relative_humidity_avg: float = Field(
        ge=0, le=100, description="Average relative humidity in %"
    )
    damp_heat_hours: float = Field(
        ge=0, description="Cumulative hours at >85°C and >85% RH"
    )
    uv_exposure_dose: float = Field(
        ge=0, description="Annual UV radiation dose in kWh/m²/year"
    )
    system_voltage: float = Field(
        ge=0, le=2000, description="DC system voltage in V"
    )
    soiling_accumulation_rate: float = Field(
        ge=0, le=10, description="Monthly soiling loss rate in %"
    )
    wind_speed_avg: float = Field(
        ge=0, le=50, description="Average wind speed in m/s"
    )
    snow_load_events: int = Field(
        ge=0, le=100, description="Number of significant snow events per year"
    )
    hail_exposure_risk: float = Field(
        ge=0, le=1, description="Hail risk factor (0-1)"
    )


class TechnologyDegradationRates(BaseModel):
    """Technology-specific baseline degradation rates (%/year).

    Attributes:
        base_degradation_rate: Baseline annual degradation rate
        lid_susceptibility: Light-induced degradation susceptibility (0-1)
        letid_susceptibility: LETID susceptibility (0-1)
        pid_susceptibility: PID susceptibility (0-1)
        uv_resistance: UV resistance factor (0-1, higher is better)
        thermal_coefficient: Temperature coefficient of degradation
    """
    base_degradation_rate: float = Field(
        ge=0, le=5, description="Baseline annual degradation rate in %/year"
    )
    lid_susceptibility: float = Field(
        ge=0, le=1, description="LID susceptibility factor"
    )
    letid_susceptibility: float = Field(
        ge=0, le=1, description="LETID susceptibility factor"
    )
    pid_susceptibility: float = Field(
        ge=0, le=1, description="PID susceptibility factor"
    )
    uv_resistance: float = Field(
        ge=0, le=1, description="UV resistance factor (higher is better)"
    )
    thermal_coefficient: float = Field(
        description="Temperature coefficient of degradation (%/year/°C)"
    )


class WarrantySpecification(BaseModel):
    """Module warranty specification.

    Attributes:
        duration_years: Warranty duration in years
        initial_guarantee: Guaranteed power at year 1 (% of nameplate)
        end_guarantee: Guaranteed power at end of warranty (% of nameplate)
        is_linear: Whether warranty degrades linearly
        tiered_guarantees: Optional tiered guarantees {year: min_power%}
    """
    duration_years: int = Field(ge=1, le=50, description="Warranty duration in years")
    initial_guarantee: float = Field(
        ge=80, le=100, description="Year 1 power guarantee (%)"
    )
    end_guarantee: float = Field(
        ge=70, le=100, description="End of warranty power guarantee (%)"
    )
    is_linear: bool = Field(
        default=True, description="Linear warranty degradation"
    )
    tiered_guarantees: Optional[Dict[int, float]] = Field(
        default=None, description="Tiered guarantees by year"
    )


class DegradationResult(BaseModel):
    """Results from degradation analysis.

    Attributes:
        years: List of years
        power_retention: Power retention at each year (% of initial)
        degradation_rate: Effective degradation rate (%/year)
        mechanism_contributions: Breakdown by degradation mechanism
        warranty_compliant: Whether warranty is maintained
        warranty_margin: Safety margin above warranty (%)
        expected_lifetime: Expected lifetime to 80% retention (years)
    """
    years: List[int]
    power_retention: List[float]
    degradation_rate: float
    mechanism_contributions: Dict[str, float]
    warranty_compliant: bool
    warranty_margin: Optional[List[float]] = None
    expected_lifetime: Optional[float] = None


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo lifetime simulation.

    Attributes:
        n_simulations: Number of Monte Carlo simulations
        degradation_rate_std: Standard deviation of degradation rate (%)
        stress_factor_uncertainty: Uncertainty in stress factors (fraction)
        random_seed: Random seed for reproducibility
    """
    n_simulations: int = Field(
        ge=100, le=100000, default=1000, description="Number of simulations"
    )
    degradation_rate_std: float = Field(
        ge=0, le=2, default=0.15, description="Degradation rate std dev (%)"
    )
    stress_factor_uncertainty: float = Field(
        ge=0, le=0.5, default=0.1, description="Stress factor uncertainty"
    )
    random_seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )


class ModuleDegradationModel:
    """
    Comprehensive module degradation modeling and lifetime prediction.

    This class provides methods to model various degradation mechanisms,
    predict power evolution over time, and validate warranty compliance.

    Example:
        >>> stress_factors = EnvironmentalStressFactors(
        ...     temperature_cycling_amplitude=30,
        ...     temperature_cycling_frequency=300,
        ...     avg_ambient_temperature=25,
        ...     max_module_temperature=75,
        ...     relative_humidity_avg=60,
        ...     damp_heat_hours=500,
        ...     uv_exposure_dose=1800,
        ...     system_voltage=1000,
        ...     soiling_accumulation_rate=0.5,
        ...     wind_speed_avg=4.5,
        ...     snow_load_events=10,
        ...     hail_exposure_risk=0.2
        ... )
        >>> model = ModuleDegradationModel()
        >>> deg_rate = model.calculate_degradation_rate(
        ...     stress_factors.dict(),
        ...     ModuleType.MONO_PERC
        ... )
        >>> power_profile = model.predict_power_over_lifetime(
        ...     initial=400.0,
        ...     years=25,
        ...     deg_rate=deg_rate
        ... )
    """

    # Technology-specific degradation parameters
    TECHNOLOGY_PARAMS: Dict[ModuleType, TechnologyDegradationRates] = {
        ModuleType.MONO_PERC: TechnologyDegradationRates(
            base_degradation_rate=0.6,
            lid_susceptibility=0.8,
            letid_susceptibility=0.9,
            pid_susceptibility=0.7,
            uv_resistance=0.8,
            thermal_coefficient=0.015
        ),
        ModuleType.POLY: TechnologyDegradationRates(
            base_degradation_rate=0.8,
            lid_susceptibility=0.6,
            letid_susceptibility=0.5,
            pid_susceptibility=0.8,
            uv_resistance=0.75,
            thermal_coefficient=0.018
        ),
        ModuleType.TOPCON: TechnologyDegradationRates(
            base_degradation_rate=0.5,
            lid_susceptibility=0.3,
            letid_susceptibility=0.4,
            pid_susceptibility=0.5,
            uv_resistance=0.85,
            thermal_coefficient=0.012
        ),
        ModuleType.HJT: TechnologyDegradationRates(
            base_degradation_rate=0.4,
            lid_susceptibility=0.2,
            letid_susceptibility=0.2,
            pid_susceptibility=0.3,
            uv_resistance=0.9,
            thermal_coefficient=0.01
        ),
        ModuleType.THIN_FILM_CDTE: TechnologyDegradationRates(
            base_degradation_rate=1.0,
            lid_susceptibility=0.4,
            letid_susceptibility=0.3,
            pid_susceptibility=0.6,
            uv_resistance=0.7,
            thermal_coefficient=0.02
        ),
        ModuleType.THIN_FILM_CIGS: TechnologyDegradationRates(
            base_degradation_rate=1.2,
            lid_susceptibility=0.5,
            letid_susceptibility=0.4,
            pid_susceptibility=0.7,
            uv_resistance=0.65,
            thermal_coefficient=0.022
        ),
        ModuleType.BIFACIAL_PERC: TechnologyDegradationRates(
            base_degradation_rate=0.55,
            lid_susceptibility=0.75,
            letid_susceptibility=0.85,
            pid_susceptibility=0.65,
            uv_resistance=0.82,
            thermal_coefficient=0.014
        ),
        ModuleType.BIFACIAL_TOPCON: TechnologyDegradationRates(
            base_degradation_rate=0.45,
            lid_susceptibility=0.25,
            letid_susceptibility=0.35,
            pid_susceptibility=0.45,
            uv_resistance=0.87,
            thermal_coefficient=0.011
        ),
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the module degradation model.

        Args:
            config: Optional configuration dictionary for model parameters
        """
        self.config = config or {}
        self.degradation_history: List[DegradationResult] = []

    def calculate_degradation_rate(
        self,
        stress_factors: Dict,
        module_type: str
    ) -> float:
        """
        Calculate overall degradation rate based on stress factors and module type.

        This method combines baseline technology degradation with environmental
        stress factors to compute an effective annual degradation rate.

        Args:
            stress_factors: Dictionary of environmental stress factors
            module_type: Module technology type (e.g., 'mono_perc', 'topcon')

        Returns:
            Effective annual degradation rate in %/year

        Example:
            >>> stress = {
            ...     'temperature_cycling_amplitude': 30,
            ...     'avg_ambient_temperature': 25,
            ...     'relative_humidity_avg': 60,
            ...     'uv_exposure_dose': 1800,
            ...     'system_voltage': 1000,
            ... }
            >>> model = ModuleDegradationModel()
            >>> rate = model.calculate_degradation_rate(stress, 'mono_perc')
            >>> print(f"Degradation rate: {rate:.2f}%/year")
        """
        # Parse inputs
        env_stress = EnvironmentalStressFactors(**stress_factors)
        mod_type = ModuleType(module_type)
        tech_params = self.TECHNOLOGY_PARAMS[mod_type]

        # Start with baseline degradation rate
        base_rate = tech_params.base_degradation_rate

        # Calculate stress multipliers
        thermal_stress = self._calculate_thermal_stress(env_stress, tech_params)
        humidity_stress = self._calculate_humidity_stress(env_stress)
        uv_stress = self._calculate_uv_stress(env_stress, tech_params)
        voltage_stress = self._calculate_voltage_stress(env_stress, tech_params)
        mechanical_stress = self._calculate_mechanical_stress(env_stress)

        # Combine stress factors (multiplicative model)
        total_stress_multiplier = (
            thermal_stress *
            humidity_stress *
            uv_stress *
            voltage_stress *
            mechanical_stress
        )

        # Calculate effective degradation rate
        effective_rate = base_rate * total_stress_multiplier

        # Cap at reasonable maximum (5%/year)
        return min(effective_rate, 5.0)

    def _calculate_thermal_stress(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate thermal stress multiplier."""
        # Temperature cycling contribution
        cycling_factor = 1.0 + (
            env.temperature_cycling_amplitude / 50.0 *
            env.temperature_cycling_frequency / 365.0 *
            0.3  # 30% max increase from cycling
        )

        # High temperature contribution (Arrhenius-like)
        temp_excess = max(0, env.max_module_temperature - 50)
        temp_factor = 1.0 + temp_excess * tech.thermal_coefficient

        return cycling_factor * temp_factor

    def _calculate_humidity_stress(self, env: EnvironmentalStressFactors) -> float:
        """Calculate humidity stress multiplier."""
        # Base humidity effect
        humidity_factor = 1.0 + (env.relative_humidity_avg - 40) / 100.0 * 0.2

        # Damp heat acceleration (hours > 85°C and 85% RH)
        damp_heat_factor = 1.0 + env.damp_heat_hours / 8760.0 * 0.5

        return humidity_factor * damp_heat_factor

    def _calculate_uv_stress(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate UV exposure stress multiplier."""
        # Nominal UV dose is ~1800 kWh/m²/year
        uv_excess = max(0, env.uv_exposure_dose - 1800) / 1800.0

        # UV resistance reduces impact
        uv_susceptibility = 1.0 - tech.uv_resistance

        return 1.0 + uv_excess * uv_susceptibility * 0.25

    def _calculate_voltage_stress(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate voltage-induced (PID) stress multiplier."""
        # PID risk increases with voltage, humidity, and temperature
        if env.system_voltage < 600:
            voltage_risk = 0.0
        else:
            # Voltage above 600V increases PID risk
            voltage_risk = (env.system_voltage - 600) / 1000.0

            # Humidity and temperature accelerate PID
            pid_acceleration = (
                (env.relative_humidity_avg / 100.0) *
                (env.avg_ambient_temperature / 50.0)
            )

            voltage_risk *= pid_acceleration * tech.pid_susceptibility

        return 1.0 + voltage_risk * 0.4  # Max 40% increase from PID

    def _calculate_mechanical_stress(self, env: EnvironmentalStressFactors) -> float:
        """Calculate mechanical stress multiplier from wind and snow."""
        # Wind loading contribution
        wind_factor = 1.0 + (env.wind_speed_avg / 20.0) * 0.15

        # Snow loading contribution
        snow_factor = 1.0 + (env.snow_load_events / 50.0) * 0.1

        # Hail damage risk
        hail_factor = 1.0 + env.hail_exposure_risk * 0.2

        return wind_factor * snow_factor * hail_factor

    def model_lid(self, initial_power: float, lid_effect: float) -> float:
        """
        Model Light Induced Degradation (LID).

        LID typically occurs in the first few hours/days of exposure for
        p-type silicon modules, causing 1-3% initial power loss.

        Args:
            initial_power: Initial module power rating (W)
            lid_effect: LID effect as fraction (e.g., 0.02 for 2% loss)

        Returns:
            Power after LID stabilization (W)

        Example:
            >>> model = ModuleDegradationModel()
            >>> stabilized = model.model_lid(400.0, 0.02)
            >>> print(f"Power after LID: {stabilized:.2f}W")
            Power after LID: 392.00W
        """
        if not 0 <= lid_effect <= 0.05:
            raise ValueError("LID effect should be between 0 and 0.05 (0-5%)")

        return initial_power * (1.0 - lid_effect)

    def model_letid(
        self,
        exposure_time: int,
        temp: float,
        current: float,
        module_type: str = "mono_perc"
    ) -> float:
        """
        Model Light and Elevated Temperature Induced Degradation (LETID).

        LETID is particularly significant in PERC modules, causing 2-10%
        degradation over several months, followed by partial recovery.

        Args:
            exposure_time: Exposure time in hours
            temp: Module operating temperature in °C
            current: Operating current in A
            module_type: Module technology type

        Returns:
            LETID degradation factor (fraction, e.g., 0.95 for 5% loss)

        Example:
            >>> model = ModuleDegradationModel()
            >>> # After 1000 hours at 60°C
            >>> letid_factor = model.model_letid(1000, 60, 9.5, 'mono_perc')
            >>> print(f"LETID factor: {letid_factor:.4f}")
        """
        mod_type = ModuleType(module_type)
        tech_params = self.TECHNOLOGY_PARAMS[mod_type]
        letid_susceptibility = tech_params.letid_susceptibility

        # LETID kinetics model (simplified)
        # Based on exponential decay with temperature acceleration

        # Activation energy approximation (eV)
        Ea = 0.7
        k_boltzmann = 8.617e-5  # eV/K
        T_kelvin = temp + 273.15
        T_ref = 333.15  # 60°C reference

        # Temperature acceleration factor
        temp_accel = math.exp(
            (Ea / k_boltzmann) * (1/T_ref - 1/T_kelvin)
        )

        # Current density effect (normalized to ~10 A)
        current_factor = (current / 10.0) ** 0.5

        # Maximum LETID degradation for this technology
        max_letid = 0.02 + 0.08 * letid_susceptibility  # 2-10% range

        # Time constant (hours) - typical 500-2000 hours
        tau = 1000 / (temp_accel * current_factor)

        # Exponential approach to maximum degradation
        degradation = max_letid * (1 - math.exp(-exposure_time / tau))

        # Recovery phase (partial, after ~5*tau)
        if exposure_time > 5 * tau:
            recovery_factor = 0.3  # 30% recovery typical
            degradation *= (1 - recovery_factor)

        return 1.0 - degradation

    def predict_power_over_lifetime(
        self,
        initial: float,
        years: int,
        deg_rate: float,
        model_type: str = "non_linear",
        include_lid: bool = True,
        include_letid: bool = True,
        module_type: str = "mono_perc"
    ) -> List[float]:
        """
        Predict power output over module lifetime.

        Args:
            initial: Initial power rating (W)
            years: Number of years to predict
            deg_rate: Annual degradation rate (%/year)
            model_type: Prediction model ('linear', 'non_linear', 'monte_carlo')
            include_lid: Include LID effect in year 0
            include_letid: Include LETID effect in early years
            module_type: Module technology type

        Returns:
            List of power values for each year (W)

        Example:
            >>> model = ModuleDegradationModel()
            >>> power = model.predict_power_over_lifetime(
            ...     initial=400.0,
            ...     years=25,
            ...     deg_rate=0.6,
            ...     model_type='non_linear'
            ... )
            >>> print(f"Year 25: {power[25]:.2f}W")
        """
        mod_type = ModuleType(module_type)
        tech_params = self.TECHNOLOGY_PARAMS[mod_type]

        power_profile = [initial]
        current_power = initial

        # Apply LID in first year if applicable
        if include_lid:
            lid_loss = 0.01 + 0.02 * tech_params.lid_susceptibility  # 1-3%
            current_power = self.model_lid(current_power, lid_loss)

        # Year-by-year prediction
        for year in range(1, years + 1):
            if model_type == "linear":
                # Simple linear degradation
                annual_loss = deg_rate / 100.0
                current_power *= (1 - annual_loss)

            elif model_type == "non_linear":
                # Non-linear model with initial drop + linear

                # LETID effect in early years (months 0-24)
                if include_letid and year <= 2:
                    letid_factor = self.model_letid(
                        exposure_time=year * 4380,  # ~half year in hours
                        temp=60,  # Typical operating temp
                        current=9.5,  # Typical current
                        module_type=module_type
                    )
                    # Apply LETID additional to degradation
                    letid_loss = 1 - letid_factor
                else:
                    letid_loss = 0

                # Base degradation with slight acceleration over time
                time_acceleration = 1.0 + (year / years) * 0.1  # 10% acceleration
                annual_loss = (deg_rate / 100.0) * time_acceleration

                # Combine effects
                total_loss = annual_loss + letid_loss * (1 if year <= 2 else 0)
                current_power *= (1 - total_loss)

            else:  # Default to linear if unknown
                annual_loss = deg_rate / 100.0
                current_power *= (1 - annual_loss)

            power_profile.append(current_power)

        return power_profile

    def calculate_warranty_compliance(
        self,
        actual_degradation: List[float],
        warranty: Dict
    ) -> bool:
        """
        Check if actual degradation meets warranty specifications.

        Args:
            actual_degradation: List of actual power values over time (W)
            warranty: Warranty specification dictionary

        Returns:
            True if warranty is maintained, False otherwise

        Example:
            >>> model = ModuleDegradationModel()
            >>> power_profile = model.predict_power_over_lifetime(400, 25, 0.6)
            >>> warranty_spec = {
            ...     'duration_years': 25,
            ...     'initial_guarantee': 98,
            ...     'end_guarantee': 85,
            ...     'is_linear': True
            ... }
            >>> compliant = model.calculate_warranty_compliance(
            ...     power_profile,
            ...     warranty_spec
            ... )
            >>> print(f"Warranty compliant: {compliant}")
        """
        warranty_spec = WarrantySpecification(**warranty)

        initial_power = actual_degradation[0]

        for year in range(len(actual_degradation)):
            if year > warranty_spec.duration_years:
                break

            # Calculate warranty requirement for this year
            if warranty_spec.tiered_guarantees and year in warranty_spec.tiered_guarantees:
                min_power_pct = warranty_spec.tiered_guarantees[year]
            elif warranty_spec.is_linear:
                # Linear interpolation
                if year == 0:
                    min_power_pct = 100.0
                elif year == 1:
                    min_power_pct = warranty_spec.initial_guarantee
                else:
                    # Linear between year 1 and end
                    years_elapsed = year - 1
                    total_years = warranty_spec.duration_years - 1
                    degradation_range = (
                        warranty_spec.initial_guarantee -
                        warranty_spec.end_guarantee
                    )
                    min_power_pct = (
                        warranty_spec.initial_guarantee -
                        (degradation_range * years_elapsed / total_years)
                    )
            else:
                # Non-linear or stepped warranty
                min_power_pct = warranty_spec.end_guarantee

            # Check compliance
            actual_power_pct = (actual_degradation[year] / initial_power) * 100

            if actual_power_pct < min_power_pct:
                return False

        return True

    def run_monte_carlo_simulation(
        self,
        initial_power: float,
        years: int,
        base_deg_rate: float,
        stress_factors: Dict,
        module_type: str,
        config: Optional[MonteCarloConfig] = None
    ) -> Dict[str, Union[List[float], float]]:
        """
        Run Monte Carlo simulation for lifetime prediction uncertainty.

        Args:
            initial_power: Initial power rating (W)
            years: Number of years to simulate
            base_deg_rate: Base degradation rate (%/year)
            stress_factors: Environmental stress factors
            module_type: Module technology type
            config: Monte Carlo configuration

        Returns:
            Dictionary with percentile predictions (P10, P50, P90) and statistics

        Example:
            >>> model = ModuleDegradationModel()
            >>> mc_config = MonteCarloConfig(n_simulations=1000)
            >>> results = model.run_monte_carlo_simulation(
            ...     initial_power=400,
            ...     years=25,
            ...     base_deg_rate=0.6,
            ...     stress_factors=stress_dict,
            ...     module_type='mono_perc',
            ...     config=mc_config
            ... )
            >>> print(f"P50 at year 25: {results['p50'][25]:.2f}W")
        """
        if config is None:
            config = MonteCarloConfig()

        np.random.seed(config.random_seed)

        n_sims = config.n_simulations
        all_simulations = np.zeros((n_sims, years + 1))

        for sim in range(n_sims):
            # Add uncertainty to degradation rate
            deg_rate_sample = np.random.normal(
                base_deg_rate,
                config.degradation_rate_std
            )
            deg_rate_sample = max(0.1, min(5.0, deg_rate_sample))  # Bounds

            # Add uncertainty to stress factors
            stress_sample = stress_factors.copy()
            for key, value in stress_sample.items():
                if isinstance(value, (int, float)):
                    uncertainty = value * config.stress_factor_uncertainty
                    stress_sample[key] = value + np.random.normal(0, uncertainty)

            # Recalculate degradation rate with sampled parameters
            try:
                adjusted_rate = self.calculate_degradation_rate(
                    stress_sample,
                    module_type
                )
            except:
                adjusted_rate = deg_rate_sample

            # Generate power profile for this simulation
            power_profile = self.predict_power_over_lifetime(
                initial=initial_power,
                years=years,
                deg_rate=adjusted_rate,
                model_type="non_linear",
                module_type=module_type
            )

            all_simulations[sim, :] = power_profile

        # Calculate percentiles
        p10 = np.percentile(all_simulations, 10, axis=0).tolist()
        p50 = np.percentile(all_simulations, 50, axis=0).tolist()
        p90 = np.percentile(all_simulations, 90, axis=0).tolist()
        mean = np.mean(all_simulations, axis=0).tolist()
        std = np.std(all_simulations, axis=0).tolist()

        return {
            'p10': p10,  # Conservative (worse case)
            'p50': p50,  # Median
            'p90': p90,  # Optimistic (best case)
            'mean': mean,
            'std': std,
            'n_simulations': n_sims
        }

    def analyze_degradation_mechanisms(
        self,
        stress_factors: Dict,
        module_type: str
    ) -> Dict[str, float]:
        """
        Analyze contribution of individual degradation mechanisms.

        Args:
            stress_factors: Environmental stress factors
            module_type: Module technology type

        Returns:
            Dictionary mapping mechanism names to degradation contributions (%/year)

        Example:
            >>> model = ModuleDegradationModel()
            >>> contributions = model.analyze_degradation_mechanisms(
            ...     stress_factors=stress_dict,
            ...     module_type='mono_perc'
            ... )
            >>> for mechanism, rate in contributions.items():
            ...     print(f"{mechanism}: {rate:.3f}%/year")
        """
        env_stress = EnvironmentalStressFactors(**stress_factors)
        mod_type = ModuleType(module_type)
        tech_params = self.TECHNOLOGY_PARAMS[mod_type]

        base_rate = tech_params.base_degradation_rate

        mechanisms = {
            'baseline': base_rate,
            'lid': 0.0,  # One-time, not annual
            'letid': tech_params.letid_susceptibility * 0.3,  # ~0.3%/yr during active phase
            'pid': self._calculate_pid_contribution(env_stress, tech_params),
            'uv_degradation': self._calculate_uv_contribution(env_stress, tech_params),
            'thermal_cycling': self._calculate_thermal_contribution(env_stress, tech_params),
            'mechanical_stress': self._calculate_mechanical_contribution(env_stress),
            'corrosion': self._calculate_corrosion_contribution(env_stress)
        }

        return mechanisms

    def _calculate_pid_contribution(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate PID degradation contribution."""
        if env.system_voltage < 600:
            return 0.0

        voltage_factor = (env.system_voltage - 600) / 1000.0
        humidity_factor = env.relative_humidity_avg / 100.0
        temp_factor = env.avg_ambient_temperature / 40.0

        pid_rate = (
            voltage_factor *
            humidity_factor *
            temp_factor *
            tech.pid_susceptibility *
            0.5  # Max ~0.5%/year from PID
        )

        return pid_rate

    def _calculate_uv_contribution(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate UV degradation contribution."""
        uv_excess = max(0, env.uv_exposure_dose - 1800) / 1800.0
        uv_susceptibility = 1.0 - tech.uv_resistance

        return uv_excess * uv_susceptibility * 0.3  # Max ~0.3%/year

    def _calculate_thermal_contribution(
        self,
        env: EnvironmentalStressFactors,
        tech: TechnologyDegradationRates
    ) -> float:
        """Calculate thermal cycling degradation contribution."""
        cycling_severity = (
            env.temperature_cycling_amplitude / 50.0 *
            env.temperature_cycling_frequency / 365.0
        )

        return cycling_severity * tech.thermal_coefficient * 20  # Scaled contribution

    def _calculate_mechanical_contribution(
        self,
        env: EnvironmentalStressFactors
    ) -> float:
        """Calculate mechanical stress degradation contribution."""
        wind_contrib = (env.wind_speed_avg / 20.0) * 0.1
        snow_contrib = (env.snow_load_events / 50.0) * 0.05
        hail_contrib = env.hail_exposure_risk * 0.15

        return wind_contrib + snow_contrib + hail_contrib

    def _calculate_corrosion_contribution(
        self,
        env: EnvironmentalStressFactors
    ) -> float:
        """Calculate humidity-induced corrosion contribution."""
        humidity_factor = max(0, env.relative_humidity_avg - 60) / 40.0
        damp_heat_factor = env.damp_heat_hours / 8760.0

        return (humidity_factor + damp_heat_factor) * 0.2  # Max ~0.4%/year

    def generate_warranty_model(
        self,
        duration_years: int = 25,
        initial_guarantee: float = 98.0,
        end_guarantee: float = 84.6,
        warranty_type: str = "linear"
    ) -> WarrantySpecification:
        """
        Generate standard warranty model.

        Args:
            duration_years: Warranty duration (typically 25-30 years)
            initial_guarantee: Year 1 guarantee (% of nameplate)
            end_guarantee: End of warranty guarantee (% of nameplate)
            warranty_type: 'linear' or 'tiered'

        Returns:
            WarrantySpecification object

        Example:
            >>> model = ModuleDegradationModel()
            >>> warranty = model.generate_warranty_model(
            ...     duration_years=25,
            ...     initial_guarantee=98,
            ...     end_guarantee=84.8
            ... )
        """
        if warranty_type == "linear":
            return WarrantySpecification(
                duration_years=duration_years,
                initial_guarantee=initial_guarantee,
                end_guarantee=end_guarantee,
                is_linear=True
            )

        elif warranty_type == "tiered":
            # Standard tiered warranty (common industry practice)
            tiered = {
                1: initial_guarantee,
                5: initial_guarantee - 1.0,
                10: 90.0,
                15: 87.5,
                20: 85.0,
                25: end_guarantee
            }

            return WarrantySpecification(
                duration_years=duration_years,
                initial_guarantee=initial_guarantee,
                end_guarantee=end_guarantee,
                is_linear=False,
                tiered_guarantees=tiered
            )

        else:
            raise ValueError(f"Unknown warranty type: {warranty_type}")

    def calculate_lifetime_energy_yield(
        self,
        power_profile: List[float],
        annual_irradiation: float,
        performance_ratio: float = 0.85,
        system_losses: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate lifetime energy yield for financial modeling.

        Args:
            power_profile: Power values over lifetime (W)
            annual_irradiation: Annual solar irradiation (kWh/m²/year)
            performance_ratio: System performance ratio (0-1)
            system_losses: Additional system losses (fraction)

        Returns:
            Dictionary with energy production metrics

        Example:
            >>> model = ModuleDegradationModel()
            >>> power = model.predict_power_over_lifetime(400, 25, 0.6)
            >>> energy = model.calculate_lifetime_energy_yield(
            ...     power_profile=power,
            ...     annual_irradiation=1800,
            ...     performance_ratio=0.85
            ... )
            >>> print(f"Total lifetime energy: {energy['total_kwh']:.0f} kWh")
        """
        # Standard Test Conditions: 1000 W/m²
        stc_irradiance = 1.0  # kW/m²

        yearly_energy = []
        total_energy = 0.0

        for year, power_w in enumerate(power_profile[1:], start=1):  # Skip year 0
            # Convert power to kW
            power_kw = power_w / 1000.0

            # Annual energy = Power × (Irradiation / STC) × PR × (1 - losses)
            annual_energy = (
                power_kw *
                (annual_irradiation / stc_irradiance) *
                performance_ratio *
                (1 - system_losses)
            )

            yearly_energy.append(annual_energy)
            total_energy += annual_energy

        return {
            'total_kwh': total_energy,
            'yearly_kwh': yearly_energy,
            'average_annual_kwh': total_energy / len(yearly_energy) if yearly_energy else 0,
            'first_year_kwh': yearly_energy[0] if yearly_energy else 0,
            'final_year_kwh': yearly_energy[-1] if yearly_energy else 0
        }

    def compare_technologies(
        self,
        stress_factors: Dict,
        initial_power: float = 400.0,
        years: int = 25
    ) -> Dict[str, Dict]:
        """
        Compare degradation across different module technologies.

        Args:
            stress_factors: Environmental stress factors
            initial_power: Initial power rating (W)
            years: Comparison period (years)

        Returns:
            Dictionary mapping technology names to performance metrics

        Example:
            >>> model = ModuleDegradationModel()
            >>> comparison = model.compare_technologies(
            ...     stress_factors=stress_dict,
            ...     initial_power=400,
            ...     years=25
            ... )
            >>> for tech, metrics in comparison.items():
            ...     print(f"{tech}: {metrics['final_power']:.1f}W at year 25")
        """
        results = {}

        for tech in ModuleType:
            # Calculate degradation rate
            deg_rate = self.calculate_degradation_rate(
                stress_factors,
                tech.value
            )

            # Predict power over lifetime
            power_profile = self.predict_power_over_lifetime(
                initial=initial_power,
                years=years,
                deg_rate=deg_rate,
                module_type=tech.value
            )

            # Calculate metrics
            final_power = power_profile[-1]
            total_degradation = ((initial_power - final_power) / initial_power) * 100
            avg_annual_degradation = total_degradation / years

            results[tech.value] = {
                'degradation_rate': deg_rate,
                'final_power': final_power,
                'total_degradation_pct': total_degradation,
                'avg_annual_degradation': avg_annual_degradation,
                'power_profile': power_profile
            }

        return results


# Utility functions for common use cases

def create_typical_desert_environment() -> EnvironmentalStressFactors:
    """Create typical desert climate stress factors."""
    return EnvironmentalStressFactors(
        temperature_cycling_amplitude=35,
        temperature_cycling_frequency=330,
        avg_ambient_temperature=32,
        max_module_temperature=85,
        relative_humidity_avg=25,
        damp_heat_hours=100,
        uv_exposure_dose=2400,
        system_voltage=1000,
        soiling_accumulation_rate=2.5,
        wind_speed_avg=5.5,
        snow_load_events=0,
        hail_exposure_risk=0.1
    )


def create_typical_coastal_environment() -> EnvironmentalStressFactors:
    """Create typical coastal climate stress factors."""
    return EnvironmentalStressFactors(
        temperature_cycling_amplitude=15,
        temperature_cycling_frequency=300,
        avg_ambient_temperature=22,
        max_module_temperature=65,
        relative_humidity_avg=75,
        damp_heat_hours=1500,
        uv_exposure_dose=1800,
        system_voltage=1000,
        soiling_accumulation_rate=0.8,
        wind_speed_avg=7.0,
        snow_load_events=0,
        hail_exposure_risk=0.05
    )


def create_typical_continental_environment() -> EnvironmentalStressFactors:
    """Create typical continental climate stress factors."""
    return EnvironmentalStressFactors(
        temperature_cycling_amplitude=25,
        temperature_cycling_frequency=320,
        avg_ambient_temperature=18,
        max_module_temperature=70,
        relative_humidity_avg=55,
        damp_heat_hours=500,
        uv_exposure_dose=1600,
        system_voltage=1000,
        soiling_accumulation_rate=1.2,
        wind_speed_avg=4.0,
        snow_load_events=15,
        hail_exposure_risk=0.3
    )


if __name__ == "__main__":
    # Example usage
    print("PV Module Degradation Model - Example Usage\n")

    # Create model
    model = ModuleDegradationModel()

    # Define environmental conditions (continental climate)
    stress_factors = create_typical_continental_environment()

    # Calculate degradation rate for Mono PERC
    deg_rate = model.calculate_degradation_rate(
        stress_factors.dict(),
        ModuleType.MONO_PERC.value
    )
    print(f"Mono PERC degradation rate: {deg_rate:.3f}%/year\n")

    # Predict power over 25 years
    power_profile = model.predict_power_over_lifetime(
        initial=400.0,
        years=25,
        deg_rate=deg_rate,
        module_type=ModuleType.MONO_PERC.value
    )

    print(f"Power at year 0: {power_profile[0]:.2f}W")
    print(f"Power at year 1: {power_profile[1]:.2f}W (after LID)")
    print(f"Power at year 10: {power_profile[10]:.2f}W")
    print(f"Power at year 25: {power_profile[25]:.2f}W\n")

    # Check warranty compliance
    warranty = model.generate_warranty_model(
        duration_years=25,
        initial_guarantee=98.0,
        end_guarantee=84.8
    )

    compliant = model.calculate_warranty_compliance(
        power_profile,
        warranty.dict()
    )
    print(f"Warranty compliant: {compliant}\n")

    # Analyze degradation mechanisms
    mechanisms = model.analyze_degradation_mechanisms(
        stress_factors.dict(),
        ModuleType.MONO_PERC.value
    )

    print("Degradation mechanism contributions:")
    for mechanism, contribution in mechanisms.items():
        print(f"  {mechanism}: {contribution:.4f}%/year")

    print("\n" + "="*60)
    print("Module Degradation Model initialized successfully!")
    print("="*60)
