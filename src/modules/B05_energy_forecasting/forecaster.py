"""Energy Forecasting Engine.

This module implements comprehensive energy forecasting for PV systems using
pvlib and advanced modeling techniques.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

from ...models.eya_models import (
    ProjectInfo,
    SystemConfiguration,
    WeatherData,
    EnergyOutput,
    PerformanceMetrics,
)


class EnergyForecaster:
    """Energy forecasting engine for PV systems.

    This class provides comprehensive energy forecasting capabilities including:
    - Hourly energy production forecasting
    - Performance ratio calculations
    - Loss analysis
    - Uncertainty quantification

    Attributes:
        project_info: Project information and location data
        system_config: System configuration parameters
    """

    def __init__(self, project_info: ProjectInfo, system_config: SystemConfiguration):
        """Initialize the energy forecaster.

        Args:
            project_info: Project information and metadata
            system_config: System configuration parameters
        """
        self.project_info = project_info
        self.system_config = system_config
        self._weather_data: List[WeatherData] = []

    def load_weather_data(self, weather_data: List[WeatherData]) -> None:
        """Load weather data for forecasting.

        Args:
            weather_data: List of weather data points
        """
        self._weather_data = sorted(weather_data, key=lambda x: x.timestamp)

    def generate_synthetic_weather_data(
        self, start_date: datetime, end_date: datetime, hourly: bool = True
    ) -> List[WeatherData]:
        """Generate synthetic weather data for demonstration.

        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            hourly: Generate hourly data if True, else daily

        Returns:
            List of synthetic weather data points
        """
        weather_data = []
        freq = "H" if hourly else "D"
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        for timestamp in dates:
            # Simple sinusoidal model for GHI (W/m²)
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday

            # Daily pattern (cosine wave)
            if 6 <= hour <= 18:
                hour_angle = (hour - 12) * 15  # degrees
                ghi = 800 * np.cos(np.radians(hour_angle)) * max(0, 1)
            else:
                ghi = 0

            # Seasonal variation
            seasonal_factor = 0.8 + 0.2 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            ghi *= seasonal_factor

            # Add some randomness
            ghi *= np.random.uniform(0.7, 1.0)
            ghi = max(0, ghi)

            # DNI and DHI estimation
            dni = ghi * 0.7 if ghi > 0 else 0
            dhi = ghi * 0.3 if ghi > 0 else 0

            # Temperature model
            temp_base = 15 + 10 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
            temp_daily = 7 * np.cos(2 * np.pi * (hour - 14) / 24)
            temperature = temp_base + temp_daily + np.random.normal(0, 2)

            weather_data.append(
                WeatherData(
                    timestamp=timestamp,
                    ghi=ghi,
                    dni=dni,
                    dhi=dhi,
                    temperature=temperature,
                    wind_speed=np.random.uniform(0, 8),
                    humidity=np.random.uniform(0.3, 0.9),
                    pressure=101325,
                )
            )

        return weather_data

    def calculate_poa_irradiance(self, weather: WeatherData) -> float:
        """Calculate Plane of Array (POA) irradiance.

        Args:
            weather: Weather data point

        Returns:
            POA irradiance in W/m²
        """
        # Simplified POA calculation (in production, use pvlib)
        tilt = np.radians(self.system_config.tilt_angle)
        ghi = weather.ghi
        dhi = weather.dhi

        # Simplified transposition
        poa = ghi * np.cos(tilt) + dhi * (1 + np.cos(tilt)) / 2
        return max(0, poa)

    def calculate_cell_temperature(self, weather: WeatherData, poa: float) -> float:
        """Calculate PV cell temperature.

        Args:
            weather: Weather data point
            poa: Plane of array irradiance in W/m²

        Returns:
            Cell temperature in °C
        """
        # Simplified cell temperature model
        # T_cell = T_ambient + (NOCT - 20) / 800 * POA
        noct = 45  # Nominal Operating Cell Temperature
        t_cell = weather.temperature + (noct - 20) / 800 * poa
        return t_cell

    def calculate_dc_power(self, weather: WeatherData) -> float:
        """Calculate DC power output.

        Args:
            weather: Weather data point

        Returns:
            DC power in kW
        """
        poa = self.calculate_poa_irradiance(weather)
        if poa < 1:  # Negligible irradiance
            return 0.0

        # Cell temperature
        t_cell = self.calculate_cell_temperature(weather, poa)

        # Temperature coefficient (typical: -0.4%/°C)
        temp_coeff = -0.004
        t_ref = 25  # Reference temperature

        # Power calculation
        efficiency_factor = 1 + temp_coeff * (t_cell - t_ref)
        dc_power = (
            self.system_config.capacity_dc
            * (poa / 1000)  # Normalize to STC (1000 W/m²)
            * efficiency_factor
        )

        return max(0, dc_power)

    def calculate_ac_power(self, dc_power: float) -> float:
        """Calculate AC power output.

        Args:
            dc_power: DC power in kW

        Returns:
            AC power in kW
        """
        # Inverter efficiency curve (simplified)
        inv_eff = self.system_config.inverter_efficiency

        # Clipping at inverter capacity
        ac_power = min(dc_power * inv_eff, self.system_config.capacity_ac)

        return max(0, ac_power)

    def forecast_energy(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[EnergyOutput]:
        """Forecast energy production.

        Args:
            start_date: Start date for forecast (uses project commissioning if None)
            end_date: End date for forecast (1 year from start if None)

        Returns:
            List of energy output forecasts
        """
        if start_date is None:
            start_date = self.project_info.commissioning_date

        if end_date is None:
            end_date = start_date + timedelta(days=365)

        # Generate or use existing weather data
        if not self._weather_data:
            self._weather_data = self.generate_synthetic_weather_data(start_date, end_date)

        energy_outputs = []

        for weather in self._weather_data:
            if not (start_date <= weather.timestamp <= end_date):
                continue

            # Calculate power
            dc_power = self.calculate_dc_power(weather)
            ac_power = self.calculate_ac_power(dc_power)

            # Energy (assuming 1-hour intervals)
            dc_energy = dc_power  # kWh (power * 1 hour)
            ac_energy = ac_power  # kWh

            # Exported energy (simplified - assume all AC energy is exported)
            exported_energy = ac_energy

            # Specific yield
            specific_yield = ac_energy / self.system_config.capacity_dc

            # Capacity factor
            max_possible_energy = self.system_config.capacity_ac  # for 1 hour
            capacity_factor = ac_energy / max_possible_energy if max_possible_energy > 0 else 0

            energy_outputs.append(
                EnergyOutput(
                    timestamp=weather.timestamp,
                    dc_energy=dc_energy,
                    ac_energy=ac_energy,
                    exported_energy=exported_energy,
                    specific_yield=specific_yield,
                    capacity_factor=capacity_factor,
                )
            )

        return energy_outputs

    def calculate_annual_energy(self, energy_outputs: List[EnergyOutput]) -> float:
        """Calculate total annual energy production.

        Args:
            energy_outputs: List of energy output forecasts

        Returns:
            Total annual AC energy in kWh/year
        """
        return sum(output.ac_energy for output in energy_outputs)

    def calculate_performance_metrics(
        self, energy_outputs: List[EnergyOutput], weather_data: List[WeatherData]
    ) -> PerformanceMetrics:
        """Calculate performance metrics.

        Args:
            energy_outputs: List of energy outputs
            weather_data: List of weather data points

        Returns:
            Performance metrics
        """
        # Calculate yields
        total_ac_energy = sum(output.ac_energy for output in energy_outputs)
        final_yield = total_ac_energy / self.system_config.capacity_dc  # kWh/kWp

        # Reference yield (based on POA irradiation)
        total_poa = sum(self.calculate_poa_irradiance(w) for w in weather_data)
        reference_yield = total_poa / 1000  # kWh/m²/kWp (assuming STC 1000 W/m²)

        # Array yield (DC)
        total_dc_energy = sum(output.dc_energy for output in energy_outputs)
        array_yield = total_dc_energy / self.system_config.capacity_dc

        # Losses
        capture_losses = reference_yield - array_yield
        system_losses = array_yield - final_yield

        # Performance ratio
        performance_ratio = final_yield / reference_yield if reference_yield > 0 else 0

        return PerformanceMetrics(
            performance_ratio=min(1.0, performance_ratio),
            reference_yield=reference_yield,
            final_yield=final_yield,
            array_yield=array_yield,
            system_losses=system_losses,
            capture_losses=capture_losses,
        )

    def calculate_uncertainty(
        self, annual_energy: float, confidence_level: float = 0.90
    ) -> Dict[str, float]:
        """Calculate uncertainty in energy forecast.

        Args:
            annual_energy: Annual energy forecast in kWh/year
            confidence_level: Confidence level for uncertainty (default: 90%)

        Returns:
            Dictionary with uncertainty metrics
        """
        # Typical uncertainty sources (%)
        weather_uncertainty = 5.0  # Weather data uncertainty
        model_uncertainty = 3.0  # Model uncertainty
        degradation_uncertainty = 2.0  # Degradation uncertainty
        availability_uncertainty = 1.5  # Availability uncertainty

        # Combined uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(
            weather_uncertainty**2
            + model_uncertainty**2
            + degradation_uncertainty**2
            + availability_uncertainty**2
        )

        # Confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        uncertainty_range = annual_energy * (total_uncertainty / 100) * z_score

        return {
            "total_uncertainty_pct": total_uncertainty,
            "confidence_level": confidence_level,
            "annual_energy_low": annual_energy - uncertainty_range,
            "annual_energy_high": annual_energy + uncertainty_range,
            "uncertainty_range": uncertainty_range,
        }
