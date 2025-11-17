"""Performance prediction for PV modules."""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from ..models.module import ModuleDesign
import numpy as np


class PerformanceResult(BaseModel):
    """Result of performance prediction."""

    # STC conditions
    pmax_stc_w: float = Field(..., description="Maximum power at STC")
    voc_stc_v: float = Field(..., description="Open circuit voltage at STC")
    isc_stc_a: float = Field(..., description="Short circuit current at STC")
    efficiency_stc_pct: float = Field(..., description="Efficiency at STC")

    # NOCT conditions
    pmax_noct_w: float = Field(..., description="Maximum power at NOCT")
    voc_noct_v: float = Field(..., description="Open circuit voltage at NOCT")
    isc_noct_a: float = Field(..., description="Short circuit current at NOCT")
    efficiency_noct_pct: float = Field(..., description="Efficiency at NOCT")

    # Temperature effects
    power_at_25c_w: float = Field(..., description="Power at 25°C cell temperature")
    power_at_50c_w: float = Field(..., description="Power at 50°C cell temperature")
    power_at_75c_w: float = Field(..., description="Power at 75°C cell temperature")

    # Bifacial gain (if applicable)
    bifacial_gain_w: Optional[float] = Field(None, description="Additional power from bifacial gain")
    total_power_with_bifacial_w: Optional[float] = Field(None, description="Total power including bifacial")

    # Annual energy prediction (kWh)
    annual_energy_kwh: float = Field(..., description="Predicted annual energy production")


class PerformancePredictor:
    """Predictor for module performance under various conditions."""

    def __init__(self, module: ModuleDesign):
        """Initialize performance predictor.

        Args:
            module: Module design specification
        """
        self.module = module

    def predict_stc(self) -> Dict[str, float]:
        """Predict performance at STC (1000 W/m², 25°C, AM1.5).

        Returns:
            Dictionary with STC performance metrics
        """
        return {
            "pmax_w": self.module.pmax_stc_w,
            "voc_v": self.module.voc_stc_v,
            "isc_a": self.module.isc_stc_a,
            "vmp_v": self.module.vmp_stc_v,
            "imp_a": self.module.imp_stc_a,
            "efficiency_pct": self.module.efficiency_stc_pct,
        }

    def predict_noct(self) -> Dict[str, float]:
        """Predict performance at NOCT (800 W/m², 20°C ambient, 1 m/s wind).

        Returns:
            Dictionary with NOCT performance metrics
        """
        # Temperature delta from STC
        temp_delta = self.module.noct_temp_c - 25.0

        # Adjust voltage (decreases with temperature)
        voc_noct = self.module.voc_stc_v * (1 + self.module.temp_coeff_voc_pct / 100 * temp_delta)
        vmp_noct = self.module.vmp_stc_v * (1 + self.module.temp_coeff_voc_pct / 100 * temp_delta)

        # Adjust current (increases slightly with temperature)
        isc_noct = self.module.isc_stc_a * (1 + self.module.temp_coeff_isc_pct / 100 * temp_delta)
        imp_noct = self.module.imp_stc_a * (1 + self.module.temp_coeff_isc_pct / 100 * temp_delta)

        # Irradiance adjustment (800 vs 1000 W/m²)
        irradiance_factor = 0.8

        isc_noct *= irradiance_factor
        imp_noct *= irradiance_factor
        pmax_noct = vmp_noct * imp_noct

        efficiency_noct = (pmax_noct / (self.module.configuration.area_m2 * 800)) * 100

        return {
            "pmax_w": pmax_noct,
            "voc_v": voc_noct,
            "isc_a": isc_noct,
            "vmp_v": vmp_noct,
            "imp_a": imp_noct,
            "efficiency_pct": efficiency_noct,
        }

    def predict_at_temperature(self, cell_temp_c: float, irradiance_w_m2: float = 1000) -> Dict[str, float]:
        """Predict performance at specific cell temperature and irradiance.

        Args:
            cell_temp_c: Cell temperature in °C
            irradiance_w_m2: Irradiance in W/m²

        Returns:
            Dictionary with performance metrics
        """
        # Temperature delta from STC
        temp_delta = cell_temp_c - 25.0

        # Adjust power
        pmax = self.module.pmax_stc_w * (1 + self.module.temp_coeff_pmax_pct / 100 * temp_delta)

        # Adjust voltage
        voc = self.module.voc_stc_v * (1 + self.module.temp_coeff_voc_pct / 100 * temp_delta)
        vmp = self.module.vmp_stc_v * (1 + self.module.temp_coeff_voc_pct / 100 * temp_delta)

        # Adjust current
        isc = self.module.isc_stc_a * (1 + self.module.temp_coeff_isc_pct / 100 * temp_delta)
        imp = self.module.imp_stc_a * (1 + self.module.temp_coeff_isc_pct / 100 * temp_delta)

        # Irradiance adjustment
        irradiance_factor = irradiance_w_m2 / 1000
        isc *= irradiance_factor
        imp *= irradiance_factor
        pmax *= irradiance_factor

        return {
            "pmax_w": pmax,
            "voc_v": voc,
            "isc_a": isc,
            "vmp_v": vmp,
            "imp_a": imp,
        }

    def predict_bifacial_gain(
        self, albedo: float = 0.2, rear_irradiance_factor: float = 0.3
    ) -> Dict[str, float]:
        """Predict bifacial gain.

        Args:
            albedo: Ground albedo (reflectivity)
            rear_irradiance_factor: Rear irradiance as fraction of front

        Returns:
            Dictionary with bifacial gain metrics
        """
        if not self.module.configuration.is_bifacial:
            return {"bifacial_gain_w": 0, "total_power_w": self.module.pmax_stc_w, "gain_pct": 0}

        # Calculate rear-side power
        bifaciality = self.module.configuration.cell_design.template.bifaciality_factor or 0.7
        rear_power_factor = rear_irradiance_factor * bifaciality * albedo

        bifacial_gain_w = self.module.pmax_stc_w * rear_power_factor
        total_power_w = self.module.pmax_stc_w + bifacial_gain_w
        gain_pct = (bifacial_gain_w / self.module.pmax_stc_w) * 100

        return {
            "bifacial_gain_w": bifacial_gain_w,
            "total_power_w": total_power_w,
            "gain_pct": gain_pct,
        }

    def predict_annual_energy(
        self,
        location_irradiance_kwh_m2: float = 1800,
        performance_ratio: float = 0.80,
        soiling_loss_pct: float = 2.0,
    ) -> float:
        """Predict annual energy production.

        Args:
            location_irradiance_kwh_m2: Annual irradiance at location in kWh/m²
            performance_ratio: System performance ratio (0-1)
            soiling_loss_pct: Soiling losses in %

        Returns:
            Annual energy in kWh
        """
        # Module area
        area_m2 = self.module.configuration.area_m2

        # Total incident energy
        incident_energy_kwh = location_irradiance_kwh_m2 * area_m2

        # Module efficiency at STC
        efficiency = self.module.efficiency_stc_pct / 100

        # Ideal energy production
        ideal_energy_kwh = incident_energy_kwh * efficiency

        # Apply performance ratio and soiling
        annual_energy_kwh = ideal_energy_kwh * performance_ratio * (1 - soiling_loss_pct / 100)

        return annual_energy_kwh

    def predict(
        self,
        location_irradiance_kwh_m2: float = 1800,
        albedo: float = 0.2,
        performance_ratio: float = 0.80,
    ) -> PerformanceResult:
        """Comprehensive performance prediction.

        Args:
            location_irradiance_kwh_m2: Annual irradiance at location
            albedo: Ground albedo for bifacial gain
            performance_ratio: System performance ratio

        Returns:
            PerformanceResult with all predictions
        """
        stc = self.predict_stc()
        noct = self.predict_noct()

        temp_25c = self.predict_at_temperature(25)
        temp_50c = self.predict_at_temperature(50)
        temp_75c = self.predict_at_temperature(75)

        bifacial = self.predict_bifacial_gain(albedo)
        annual_energy = self.predict_annual_energy(location_irradiance_kwh_m2, performance_ratio)

        return PerformanceResult(
            pmax_stc_w=stc["pmax_w"],
            voc_stc_v=stc["voc_v"],
            isc_stc_a=stc["isc_a"],
            efficiency_stc_pct=stc["efficiency_pct"],
            pmax_noct_w=noct["pmax_w"],
            voc_noct_v=noct["voc_v"],
            isc_noct_a=noct["isc_a"],
            efficiency_noct_pct=noct["efficiency_pct"],
            power_at_25c_w=temp_25c["pmax_w"],
            power_at_50c_w=temp_50c["pmax_w"],
            power_at_75c_w=temp_75c["pmax_w"],
            bifacial_gain_w=bifacial["bifacial_gain_w"] if self.module.configuration.is_bifacial else None,
            total_power_with_bifacial_w=bifacial["total_power_w"] if self.module.configuration.is_bifacial else None,
            annual_energy_kwh=annual_energy,
        )
