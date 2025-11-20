"""
Performance models for PV circularity simulator.

This module defines comprehensive Pydantic models for PV performance tracking,
including:
- Real-time and historical performance metrics
- Temperature coefficients and effects
- Degradation models (initial, annual, cumulative)
- Loss analysis (CTM, optical, thermal, shading, soiling, etc.)
- Performance ratio calculations

All models include full validation for physical constraints and
production-ready error handling.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.core import NamedModel, TimestampedModel, UUIDModel


class PerformanceMetrics(TimestampedModel):
    """
    Real-time or historical performance metrics for a PV system.

    These metrics can represent instantaneous measurements or
    aggregated values over a time period.

    Attributes:
        power_output_w: Actual power output in watts
        voltage_v: Operating voltage in volts
        current_a: Operating current in amperes
        irradiance_w_m2: Irradiance on module plane in W/m²
        cell_temperature_c: Cell temperature in Celsius
        ambient_temperature_c: Ambient air temperature in Celsius
        wind_speed_m_s: Wind speed in meters per second
        efficiency_percentage: Instantaneous efficiency percentage
        performance_ratio: Performance ratio (actual/expected output)
        energy_cumulative_kwh: Cumulative energy production in kWh
        measurement_timestamp: Timestamp of measurement
    """

    power_output_w: float = Field(
        ...,
        ge=0,
        description="Actual power output in watts",
    )
    voltage_v: float = Field(
        ...,
        ge=0,
        description="Operating voltage in volts",
    )
    current_a: float = Field(
        ...,
        ge=0,
        description="Operating current in amperes",
    )
    irradiance_w_m2: float = Field(
        ...,
        ge=0,
        le=1500,
        description="Irradiance on module plane in W/m² (max ~1400 W/m²)",
    )
    cell_temperature_c: float = Field(
        ...,
        ge=-40,
        le=125,
        description="Cell/module temperature in Celsius",
    )
    ambient_temperature_c: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Ambient air temperature in Celsius",
    )
    wind_speed_m_s: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Wind speed in meters per second",
    )
    efficiency_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Instantaneous conversion efficiency percentage",
    )
    performance_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=1.5,
        description="Performance ratio (actual output / expected output at STC)",
    )
    energy_cumulative_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Cumulative energy production in kWh",
    )
    measurement_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of this measurement (UTC)",
    )

    @field_validator("irradiance_w_m2")
    @classmethod
    def validate_irradiance(cls, v: float) -> float:
        """Validate irradiance is within realistic range."""
        if v > 1400:
            import warnings
            warnings.warn(
                f"Irradiance {v:.0f} W/m² exceeds typical maximum (~1400 W/m²). "
                f"This may indicate measurement error or unusual conditions."
            )
        return v

    @model_validator(mode="after")
    def validate_power_consistency(self) -> "PerformanceMetrics":
        """Validate that power = voltage × current."""
        calculated_power = self.voltage_v * self.current_a
        if abs(calculated_power - self.power_output_w) > max(1.0, self.power_output_w * 0.01):
            raise ValueError(
                f"Power ({self.power_output_w:.2f}W) should equal voltage × current "
                f"({calculated_power:.2f}W)"
            )
        return self


class TemperatureCoefficients(NamedModel):
    """
    Temperature coefficients for PV performance calculations.

    These coefficients describe how module parameters change with temperature.
    Typically provided by manufacturers.

    Attributes:
        alpha_isc_percent_c: Temperature coefficient of Isc in %/°C (positive)
        beta_voc_percent_c: Temperature coefficient of Voc in %/°C (negative)
        gamma_pmax_percent_c: Temperature coefficient of Pmax in %/°C (negative)
        alpha_isc_a_c: Temperature coefficient of Isc in A/°C (absolute)
        beta_voc_v_c: Temperature coefficient of Voc in V/°C (absolute)
    """

    alpha_isc_percent_c: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Temperature coefficient of Isc in %/°C (typical: 0.03-0.06)",
    )
    beta_voc_percent_c: float = Field(
        default=-0.30,
        ge=-1.0,
        le=0.0,
        description="Temperature coefficient of Voc in %/°C (typical: -0.28 to -0.35)",
    )
    gamma_pmax_percent_c: float = Field(
        default=-0.40,
        ge=-1.0,
        le=0.0,
        description="Temperature coefficient of Pmax in %/°C (typical: -0.35 to -0.50)",
    )
    alpha_isc_a_c: Optional[float] = Field(
        None,
        description="Temperature coefficient of Isc in A/°C (absolute value)",
    )
    beta_voc_v_c: Optional[float] = Field(
        None,
        description="Temperature coefficient of Voc in V/°C (absolute value)",
    )

    def calculate_power_derating(
        self,
        cell_temp_c: float,
        reference_temp_c: float = 25.0,
    ) -> float:
        """
        Calculate power derating factor due to temperature.

        Args:
            cell_temp_c: Actual cell temperature in Celsius
            reference_temp_c: Reference temperature in Celsius (default: 25°C STC)

        Returns:
            float: Power derating factor (1.0 = no derating, <1.0 = reduced power)
        """
        delta_t = cell_temp_c - reference_temp_c
        derating = 1.0 + (self.gamma_pmax_percent_c / 100.0) * delta_t
        return max(0.0, derating)  # Ensure non-negative


class DegradationModel(NamedModel):
    """
    Degradation model for PV modules and systems.

    Models the reduction in performance over time due to various
    degradation mechanisms.

    Attributes:
        initial_degradation_percentage: Initial degradation in first year (LID, PID)
        annual_degradation_rate_percentage: Annual degradation rate after first year
        lifetime_years: Expected lifetime of the system
        degradation_type: Type of degradation model (linear, exponential, custom)
        light_induced_degradation_percentage: Light-induced degradation (LID)
        potential_induced_degradation_percentage: Potential-induced degradation (PID)
        mechanical_stress_degradation_percentage: Mechanical stress degradation
        cumulative_degradation_percentage: Total cumulative degradation to date
        age_years: Current age of system in years
    """

    initial_degradation_percentage: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="Initial degradation in first year (LID, PID) in %",
    )
    annual_degradation_rate_percentage: float = Field(
        default=0.5,
        ge=0,
        le=5.0,
        description="Annual degradation rate after first year in %/year (typical: 0.3-0.8)",
    )
    lifetime_years: int = Field(
        default=25,
        ge=10,
        le=50,
        description="Expected system lifetime in years",
    )
    degradation_type: str = Field(
        default="linear",
        pattern="^(linear|exponential|custom)$",
        description="Type of degradation model (linear, exponential, or custom)",
    )
    light_induced_degradation_percentage: float = Field(
        default=1.5,
        ge=0,
        le=5,
        description="Light-induced degradation (LID) in % (typical: 1-3%)",
    )
    potential_induced_degradation_percentage: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Potential-induced degradation (PID) in % (0 if mitigated)",
    )
    mechanical_stress_degradation_percentage: float = Field(
        default=0.0,
        ge=0,
        le=5,
        description="Mechanical stress degradation in %",
    )
    cumulative_degradation_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Total cumulative degradation to date in %",
    )
    age_years: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Current age of system in years",
    )

    def calculate_power_retention(self, years: Optional[float] = None) -> float:
        """
        Calculate power retention factor at a given age.

        Args:
            years: Age in years (uses age_years if not specified)

        Returns:
            float: Power retention factor (1.0 = 100%, 0.8 = 80% of original)

        Raises:
            ValueError: If years is negative
        """
        age = years if years is not None else self.age_years
        if age < 0:
            raise ValueError("Age cannot be negative")

        if self.degradation_type == "linear":
            # Linear degradation model
            if age <= 1.0:
                degradation = self.initial_degradation_percentage * age
            else:
                degradation = (
                    self.initial_degradation_percentage
                    + self.annual_degradation_rate_percentage * (age - 1.0)
                )
        elif self.degradation_type == "exponential":
            # Exponential degradation model
            import math
            rate = self.annual_degradation_rate_percentage / 100.0
            degradation = 100.0 * (1.0 - math.exp(-rate * age))
        else:
            # Custom: use cumulative degradation
            degradation = self.cumulative_degradation_percentage

        retention = 1.0 - (degradation / 100.0)
        return max(0.0, min(1.0, retention))  # Clamp to [0, 1]

    def calculate_degradation_at_year(self, year: int) -> float:
        """
        Calculate total degradation percentage at specific year.

        Args:
            year: Year number (0 = new, 1 = after 1 year, etc.)

        Returns:
            float: Total degradation percentage

        Raises:
            ValueError: If year is negative
        """
        if year < 0:
            raise ValueError("Year cannot be negative")

        retention = self.calculate_power_retention(float(year))
        return (1.0 - retention) * 100.0

    @field_validator("cumulative_degradation_percentage")
    @classmethod
    def validate_cumulative_degradation(cls, v: float) -> float:
        """Validate cumulative degradation is reasonable."""
        if v > 50:
            import warnings
            warnings.warn(
                f"Cumulative degradation {v:.1f}% is very high. "
                f"System may be approaching end of life."
            )
        return v


class LossAnalysis(NamedModel):
    """
    Comprehensive loss analysis for PV systems.

    Breaks down all loss mechanisms that reduce system performance
    from ideal STC conditions to actual operating conditions.

    Attributes:
        soiling_loss_percentage: Soiling/dirt accumulation loss
        shading_loss_percentage: Shading loss (trees, buildings, etc.)
        spectral_loss_percentage: Spectral mismatch loss
        temperature_loss_percentage: Temperature-related loss
        irradiance_loss_percentage: Low irradiance loss
        module_mismatch_loss_percentage: Module mismatch loss
        wiring_loss_percentage: DC wiring resistance loss
        connection_loss_percentage: Connection and termination loss
        inverter_loss_percentage: Inverter conversion loss
        transformer_loss_percentage: Transformer loss (if present)
        availability_loss_percentage: Downtime/availability loss
        degradation_loss_percentage: Age-related degradation loss
        ctm_loss_percentage: Contact Transport Mechanism loss
        optical_loss_percentage: Optical losses (reflection, absorption)
        total_loss_percentage: Total system loss (calculated)
    """

    soiling_loss_percentage: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Soiling/dirt accumulation loss in % (typical: 1-5%)",
    )
    shading_loss_percentage: float = Field(
        default=1.0,
        ge=0,
        le=50,
        description="Shading loss in % (highly site-specific)",
    )
    spectral_loss_percentage: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Spectral mismatch loss in %",
    )
    temperature_loss_percentage: float = Field(
        default=5.0,
        ge=0,
        le=30,
        description="Temperature-related loss in % (climate-dependent)",
    )
    irradiance_loss_percentage: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Low irradiance loss in %",
    )
    module_mismatch_loss_percentage: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Module-to-module mismatch loss in %",
    )
    wiring_loss_percentage: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="DC wiring resistance loss in %",
    )
    connection_loss_percentage: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Connection and termination loss in %",
    )
    inverter_loss_percentage: float = Field(
        default=3.0,
        ge=0,
        le=20,
        description="Inverter conversion loss in % (100 - efficiency)",
    )
    transformer_loss_percentage: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Transformer loss in % (0 if no transformer)",
    )
    availability_loss_percentage: float = Field(
        default=1.0,
        ge=0,
        le=20,
        description="System downtime/availability loss in %",
    )
    degradation_loss_percentage: float = Field(
        default=0.0,
        ge=0,
        le=50,
        description="Age-related degradation loss in %",
    )
    ctm_loss_percentage: float = Field(
        default=1.5,
        ge=0,
        le=10,
        description="Contact Transport Mechanism loss in %",
    )
    optical_loss_percentage: float = Field(
        default=3.0,
        ge=0,
        le=15,
        description="Optical losses (reflection, absorption) in %",
    )
    total_loss_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Total system loss in % (auto-calculated if not provided)",
    )

    @model_validator(mode="after")
    def calculate_total_loss(self) -> "LossAnalysis":
        """
        Calculate total loss from individual loss components.

        Uses compound loss calculation: total_retention = ∏(1 - loss_i)
        Then: total_loss = 1 - total_retention
        """
        # Calculate compound losses (multiplicative, not additive)
        retention_factor = 1.0
        losses = [
            self.soiling_loss_percentage,
            self.shading_loss_percentage,
            self.spectral_loss_percentage,
            self.temperature_loss_percentage,
            self.irradiance_loss_percentage,
            self.module_mismatch_loss_percentage,
            self.wiring_loss_percentage,
            self.connection_loss_percentage,
            self.inverter_loss_percentage,
            self.transformer_loss_percentage,
            self.availability_loss_percentage,
            self.degradation_loss_percentage,
            self.ctm_loss_percentage,
            self.optical_loss_percentage,
        ]

        for loss in losses:
            retention_factor *= 1.0 - (loss / 100.0)

        calculated_total_loss = (1.0 - retention_factor) * 100.0

        # If total_loss is not provided, use calculated value
        if self.total_loss_percentage is None:
            self.total_loss_percentage = calculated_total_loss
        else:
            # Validate provided value is close to calculated
            if abs(self.total_loss_percentage - calculated_total_loss) > 5.0:
                import warnings
                warnings.warn(
                    f"Provided total loss ({self.total_loss_percentage:.1f}%) differs "
                    f"significantly from calculated ({calculated_total_loss:.1f}%)"
                )

        return self

    def calculate_performance_ratio(self) -> float:
        """
        Calculate performance ratio from losses.

        Returns:
            float: Performance ratio (0-1, typically 0.75-0.85)
        """
        return 1.0 - (self.total_loss_percentage / 100.0)


class PerformanceModel(UUIDModel):
    """
    Comprehensive performance model for PV systems.

    Combines real-time metrics, temperature effects, degradation,
    and loss analysis into a complete performance model.

    Attributes:
        name: Human-readable name for this performance record
        system_id: Reference to the PV system
        metrics: Current or averaged performance metrics
        temperature_coefficients: Temperature coefficient specifications
        degradation: Degradation model and current state
        losses: Comprehensive loss analysis
        performance_ratio: Overall system performance ratio
        capacity_factor_percentage: Capacity factor (actual/rated output)
        notes: Optional notes about performance conditions or issues
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for this performance record",
    )
    system_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Reference ID to the PV system being monitored",
    )
    metrics: PerformanceMetrics = Field(
        ...,
        description="Current or time-averaged performance metrics",
    )
    temperature_coefficients: TemperatureCoefficients = Field(
        ...,
        description="Temperature coefficients for performance calculations",
    )
    degradation: DegradationModel = Field(
        ...,
        description="Degradation model and current degradation state",
    )
    losses: LossAnalysis = Field(
        ...,
        description="Comprehensive breakdown of all loss mechanisms",
    )
    performance_ratio: float = Field(
        ...,
        ge=0,
        le=1.5,
        description="Overall system performance ratio (actual/expected, 0-1)",
    )
    capacity_factor_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Capacity factor: (actual energy / rated capacity) × 100%",
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional notes about performance conditions or issues",
    )

    @model_validator(mode="after")
    def validate_performance_consistency(self) -> "PerformanceModel":
        """Validate consistency between performance components."""
        # Performance ratio should be consistent with loss analysis
        calculated_pr = self.losses.calculate_performance_ratio()
        if abs(self.performance_ratio - calculated_pr) > 0.1:
            import warnings
            warnings.warn(
                f"Performance ratio ({self.performance_ratio:.3f}) differs from "
                f"calculated value based on losses ({calculated_pr:.3f})"
            )

        # Update degradation loss if degradation model is available
        retention = self.degradation.calculate_power_retention()
        expected_degradation_loss = (1.0 - retention) * 100.0
        if abs(self.losses.degradation_loss_percentage - expected_degradation_loss) > 2.0:
            import warnings
            warnings.warn(
                f"Degradation loss in loss analysis ({self.losses.degradation_loss_percentage:.1f}%) "
                f"differs from degradation model ({expected_degradation_loss:.1f}%)"
            )

        return self

    def calculate_expected_power(
        self,
        rated_power_w: float,
        irradiance_w_m2: float,
        reference_irradiance_w_m2: float = 1000.0,
    ) -> float:
        """
        Calculate expected power output under given conditions.

        Args:
            rated_power_w: Rated power at STC in watts
            irradiance_w_m2: Actual irradiance in W/m²
            reference_irradiance_w_m2: Reference irradiance (STC) in W/m²

        Returns:
            float: Expected power output in watts

        Raises:
            ValueError: If inputs are invalid
        """
        if rated_power_w <= 0:
            raise ValueError("Rated power must be positive")
        if irradiance_w_m2 < 0:
            raise ValueError("Irradiance cannot be negative")

        # Irradiance scaling
        power = rated_power_w * (irradiance_w_m2 / reference_irradiance_w_m2)

        # Temperature derating
        temp_derating = self.temperature_coefficients.calculate_power_derating(
            self.metrics.cell_temperature_c
        )
        power *= temp_derating

        # Degradation
        degradation_retention = self.degradation.calculate_power_retention()
        power *= degradation_retention

        # Performance ratio (all other losses)
        power *= self.performance_ratio

        return power
