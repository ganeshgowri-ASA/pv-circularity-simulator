"""System loss modeling for PV systems."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    SystemLosses,
    MountingType,
)

logger = logging.getLogger(__name__)


class SystemLossModel:
    """
    Comprehensive loss modeling for PV systems.

    Calculates soiling, shading, wiring, inverter, transformer, clipping,
    and availability losses following industry best practices and PVsyst methodology.
    """

    # Geographic soiling loss database (annual average %)
    SOILING_DATABASE = {
        "desert": 4.5,
        "arid": 3.5,
        "semi_arid": 2.5,
        "temperate": 2.0,
        "tropical": 3.0,
        "coastal": 3.5,
        "urban_industrial": 4.0,
        "rural": 1.5,
        "arctic": 1.0,
    }

    # Cable resistivity (Ω·mm²/m at 20°C)
    COPPER_RESISTIVITY = 0.01724
    ALUMINUM_RESISTIVITY = 0.02826

    def __init__(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        mounting_type: MountingType = MountingType.GROUND_FIXED,
    ):
        """
        Initialize system loss model.

        Args:
            module: Module parameters
            inverter: Inverter parameters
            mounting_type: Type of mounting system
        """
        self.module = module
        self.inverter = inverter
        self.mounting_type = mounting_type

        logger.info(f"Initialized SystemLossModel for {mounting_type.value} system")

    def soiling_losses(
        self,
        geographic_type: str = "temperate",
        cleaning_frequency_days: int = 30,
        rainfall_mm_per_year: Optional[float] = None,
    ) -> float:
        """
        Calculate soiling losses based on geography and cleaning schedule.

        Args:
            geographic_type: Geographic climate type (desert, arid, semi_arid, temperate, etc.)
            cleaning_frequency_days: Days between manual cleaning (0 = no cleaning)
            rainfall_mm_per_year: Annual rainfall in mm (if provided, overrides geographic type)

        Returns:
            Soiling loss percentage (%)
        """
        # Base soiling from geographic database
        base_soiling = self.SOILING_DATABASE.get(geographic_type, 2.0)

        # Adjust for rainfall if provided
        if rainfall_mm_per_year is not None:
            # Rainfall reduces soiling: > 1000mm = low, 500-1000 = medium, < 500 = high
            if rainfall_mm_per_year > 1000:
                rainfall_factor = 0.7
            elif rainfall_mm_per_year > 500:
                rainfall_factor = 1.0
            else:
                rainfall_factor = 1.3
            base_soiling *= rainfall_factor

        # Adjust for cleaning frequency
        if cleaning_frequency_days > 0:
            # More frequent cleaning reduces soiling
            # Assume 30-day cleaning gives base value, scale accordingly
            cleaning_factor = np.sqrt(cleaning_frequency_days / 30.0)
            soiling_loss = base_soiling * min(cleaning_factor, 2.0)
        else:
            # No cleaning: increase by 50%
            soiling_loss = base_soiling * 1.5

        # Mounting type adjustment
        if self.mounting_type in [MountingType.GROUND_FIXED, MountingType.GROUND_SINGLE_AXIS]:
            # Ground mounted: more dust accumulation
            soiling_loss *= 1.1
        elif self.mounting_type == MountingType.FLOATING:
            # Floating: less soiling due to water proximity
            soiling_loss *= 0.8

        logger.info(
            f"Soiling losses: {soiling_loss:.2f}% "
            f"(type: {geographic_type}, cleaning: {cleaning_frequency_days} days)"
        )

        return soiling_loss

    def shading_losses(
        self,
        horizon_profile: Optional[List[Tuple[float, float]]] = None,
        near_shading_objects: Optional[List[Dict]] = None,
        backtracking_enabled: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate shading losses from horizon and near objects.

        Args:
            horizon_profile: List of (azimuth, elevation) tuples defining horizon
            near_shading_objects: List of nearby shading objects with dimensions
            backtracking_enabled: Whether tracker backtracking is enabled

        Returns:
            Dictionary with shading loss components:
                - far_shading: Far shading/horizon losses (%)
                - near_shading: Near object shading losses (%)
                - electrical_shading: Additional electrical losses from shading (%)
                - total_shading: Total shading losses (%)
        """
        far_shading = 0.0
        near_shading = 0.0
        electrical_shading = 0.0

        # Far shading (horizon profile)
        if horizon_profile:
            # Simplified model: average horizon elevation
            avg_elevation = np.mean([h[1] for h in horizon_profile])
            # ~0.5% loss per degree of horizon elevation
            far_shading = min(avg_elevation * 0.5, 10.0)

        # Near shading (objects)
        if near_shading_objects:
            # Simplified shading calculation
            # In production, this would use detailed 3D modeling
            total_shading_factor = 0.0
            for obj in near_shading_objects:
                # Each object contributes based on size and distance
                shading_impact = obj.get("shading_factor", 0.0)
                total_shading_factor += shading_impact

            near_shading = min(total_shading_factor * 100, 20.0)

        # Electrical shading losses (module bypass diode behavior)
        if near_shading > 0:
            # Partial shading causes additional electrical losses
            # ~1.5x the geometric shading due to bypass diode losses
            electrical_shading = near_shading * 0.5

        # Backtracking reduces shading for trackers
        if backtracking_enabled and self.mounting_type == MountingType.GROUND_SINGLE_AXIS:
            near_shading *= 0.3  # Backtracking can reduce row-to-row shading by ~70%
            logger.info("Backtracking enabled: reduced near shading by 70%")

        total_shading = far_shading + near_shading + electrical_shading

        logger.info(
            f"Shading losses - far: {far_shading:.2f}%, "
            f"near: {near_shading:.2f}%, electrical: {electrical_shading:.2f}%, "
            f"total: {total_shading:.2f}%"
        )

        return {
            "far_shading": far_shading,
            "near_shading": near_shading,
            "electrical_shading": electrical_shading,
            "total_shading": total_shading,
        }

    def dc_wiring_losses(
        self,
        cable_length_m: float,
        cable_cross_section_mm2: float,
        current_a: float,
        cable_material: str = "copper",
        ambient_temp_c: float = 25.0,
    ) -> float:
        """
        Calculate DC wiring resistance losses.

        Args:
            cable_length_m: Total cable length (both conductors) in meters
            cable_cross_section_mm2: Cable cross-sectional area in mm²
            current_a: Operating current in amperes
            cable_material: Cable material ("copper" or "aluminum")
            ambient_temp_c: Ambient temperature in °C

        Returns:
            DC wiring loss percentage (%)
        """
        # Get resistivity based on material
        if cable_material.lower() == "copper":
            resistivity_20c = self.COPPER_RESISTIVITY
        elif cable_material.lower() == "aluminum":
            resistivity_20c = self.ALUMINUM_RESISTIVITY
        else:
            raise ValueError(f"Unknown cable material: {cable_material}")

        # Temperature correction for resistance (0.004/°C for copper, 0.004/°C for aluminum)
        temp_coeff = 0.004
        resistivity = resistivity_20c * (1 + temp_coeff * (ambient_temp_c - 20.0))

        # Calculate cable resistance
        resistance = (resistivity * cable_length_m) / cable_cross_section_mm2

        # Calculate power loss
        power_loss_w = current_a ** 2 * resistance

        # Calculate percentage loss (relative to module power)
        voltage = self.module.vmp
        power_w = voltage * current_a
        loss_percent = (power_loss_w / power_w * 100) if power_w > 0 else 0.0

        logger.debug(
            f"DC wiring: {cable_length_m}m of {cable_cross_section_mm2}mm² {cable_material}, "
            f"R={resistance:.4f}Ω, loss={loss_percent:.2f}%"
        )

        return loss_percent

    def ac_wiring_losses(
        self,
        cable_length_m: float,
        cable_cross_section_mm2: float,
        power_kw: float,
        voltage_v: float = 480.0,
        num_phases: int = 3,
        cable_material: str = "copper",
    ) -> float:
        """
        Calculate AC wiring losses.

        Args:
            cable_length_m: Cable length in meters
            cable_cross_section_mm2: Cable cross-sectional area in mm²
            power_kw: AC power in kW
            voltage_v: AC voltage in volts
            num_phases: Number of phases (1 or 3)
            cable_material: Cable material ("copper" or "aluminum")

        Returns:
            AC wiring loss percentage (%)
        """
        # Calculate current
        if num_phases == 3:
            current_a = (power_kw * 1000) / (np.sqrt(3) * voltage_v * 0.95)  # 0.95 = power factor
        else:
            current_a = (power_kw * 1000) / (voltage_v * 0.95)

        # Use DC wiring loss calculation (similar principle)
        loss_percent = self.dc_wiring_losses(
            cable_length_m=cable_length_m,
            cable_cross_section_mm2=cable_cross_section_mm2,
            current_a=current_a,
            cable_material=cable_material,
        )

        logger.debug(f"AC wiring: {num_phases}-phase, {current_a:.1f}A, loss={loss_percent:.2f}%")

        return loss_percent

    def inverter_efficiency_curve(
        self, power_fraction: float, voltage_fraction: float = 1.0
    ) -> float:
        """
        Calculate inverter efficiency at given loading.

        Uses a simplified efficiency curve based on European/CEC weighting.

        Args:
            power_fraction: Fraction of rated power (0-1.2)
            voltage_fraction: Fraction of nominal DC voltage (0.5-1.2)

        Returns:
            Inverter efficiency (0-1, not percentage)
        """
        # Simplified efficiency curve (typical string inverter)
        # Peak efficiency around 30-50% loading
        max_eff = self.inverter.max_efficiency / 100.0

        if power_fraction < 0.05:
            # Very low power: poor efficiency
            efficiency = max_eff * 0.80
        elif power_fraction < 0.10:
            efficiency = max_eff * 0.90
        elif power_fraction < 0.20:
            efficiency = max_eff * 0.95
        elif power_fraction < 0.30:
            efficiency = max_eff * 0.98
        elif power_fraction <= 1.0:
            # Peak efficiency region
            efficiency = max_eff
        elif power_fraction <= 1.1:
            # Slight overloading: still near peak
            efficiency = max_eff * 0.99
        else:
            # Clipping region: efficiency drops
            efficiency = max_eff * 0.97

        # Voltage dependency (reduced efficiency at low voltage)
        if voltage_fraction < 0.9:
            voltage_penalty = (0.9 - voltage_fraction) * 0.1
            efficiency *= (1.0 - voltage_penalty)

        return min(efficiency, 1.0)

    def transformer_losses(
        self,
        transformer_rating_kva: float,
        load_fraction: float,
        no_load_loss_percent: float = 0.2,
        load_loss_percent: float = 0.8,
    ) -> float:
        """
        Calculate transformer losses (no-load + load losses).

        Args:
            transformer_rating_kva: Transformer rating in kVA
            load_fraction: Fraction of rated load (0-1)
            no_load_loss_percent: No-load (iron) losses as % of rating
            load_loss_percent: Full-load copper losses as % of rating

        Returns:
            Transformer loss percentage (%)
        """
        # No-load losses (constant)
        no_load_loss = no_load_loss_percent

        # Load losses (proportional to square of current)
        load_loss = load_loss_percent * (load_fraction ** 2)

        total_loss = no_load_loss + load_loss

        logger.debug(
            f"Transformer losses at {load_fraction*100:.0f}% load: "
            f"no-load={no_load_loss:.2f}%, load={load_loss:.2f}%, total={total_loss:.2f}%"
        )

        return total_loss

    def clipping_losses(
        self,
        dc_power_w: float,
        inverter_max_power_w: float,
    ) -> float:
        """
        Calculate power clipping losses.

        Args:
            dc_power_w: DC power input to inverter (W)
            inverter_max_power_w: Inverter maximum DC power rating (W)

        Returns:
            Clipped power in watts
        """
        if dc_power_w > inverter_max_power_w:
            clipped_power = dc_power_w - inverter_max_power_w
            logger.debug(f"Clipping: {clipped_power:.0f}W ({clipped_power/dc_power_w*100:.1f}%)")
            return clipped_power
        return 0.0

    def availability_losses(
        self,
        grid_availability: float = 99.5,
        scheduled_maintenance_hours: float = 24.0,
        unscheduled_downtime_hours: float = 12.0,
    ) -> float:
        """
        Calculate availability losses from grid and maintenance.

        Args:
            grid_availability: Grid availability percentage (%)
            scheduled_maintenance_hours: Annual scheduled maintenance hours
            unscheduled_downtime_hours: Annual unscheduled downtime hours

        Returns:
            Availability loss percentage (%)
        """
        # Grid unavailability
        grid_loss = 100.0 - grid_availability

        # Maintenance downtime (assume during daylight hours)
        hours_per_year = 8760
        daylight_fraction = 0.5  # Assume 12 hours daylight
        daylight_hours = hours_per_year * daylight_fraction

        maintenance_loss = (
            (scheduled_maintenance_hours + unscheduled_downtime_hours)
            / daylight_hours
            * 100.0
        )

        total_availability_loss = grid_loss + maintenance_loss

        logger.info(
            f"Availability losses - grid: {grid_loss:.2f}%, "
            f"maintenance: {maintenance_loss:.2f}%, total: {total_availability_loss:.2f}%"
        )

        return total_availability_loss

    def calculate_system_losses(
        self,
        dc_power_w: float,
        location_type: str = "temperate",
        cable_length_dc_m: float = 100.0,
        cable_length_ac_m: float = 200.0,
        has_transformer: bool = False,
        **kwargs,
    ) -> SystemLosses:
        """
        Calculate comprehensive system losses for given operating point.

        Args:
            dc_power_w: DC power at module output (W)
            location_type: Geographic location type for soiling
            cable_length_dc_m: DC cable length (m)
            cable_length_ac_m: AC cable length (m)
            has_transformer: Whether system includes transformer
            **kwargs: Additional parameters for specific loss calculations

        Returns:
            SystemLosses object with all loss components
        """
        # Soiling
        soiling = self.soiling_losses(geographic_type=location_type)

        # Shading
        shading_results = self.shading_losses(
            horizon_profile=kwargs.get("horizon_profile"),
            near_shading_objects=kwargs.get("near_shading_objects"),
            backtracking_enabled=kwargs.get("backtracking_enabled", False),
        )

        # DC wiring (assume 6mm² copper cable, typical string current)
        dc_wiring = self.dc_wiring_losses(
            cable_length_m=cable_length_dc_m,
            cable_cross_section_mm2=6.0,
            current_a=self.module.imp,
        )

        # Inverter
        power_fraction = dc_power_w / self.inverter.pdc_max
        inverter_eff = self.inverter_efficiency_curve(power_fraction)
        inverter_loss = (1.0 - inverter_eff) * 100.0

        # Clipping
        clipped_power_w = self.clipping_losses(dc_power_w, self.inverter.pdc_max)
        clipping = (clipped_power_w / dc_power_w * 100.0) if dc_power_w > 0 else 0.0

        # AC power after inverter
        ac_power_kw = (dc_power_w * inverter_eff) / 1000.0

        # AC wiring
        ac_wiring = self.ac_wiring_losses(
            cable_length_m=cable_length_ac_m,
            cable_cross_section_mm2=25.0,  # Typical for commercial systems
            power_kw=ac_power_kw,
            voltage_v=480.0,
            num_phases=3,
        )

        # Transformer
        transformer = 0.0
        if has_transformer:
            transformer = self.transformer_losses(
                transformer_rating_kva=self.inverter.pac_max / 1000.0 * 1.1,
                load_fraction=power_fraction,
            )

        # Availability
        availability_loss = self.availability_losses()

        losses = SystemLosses(
            soiling=soiling,
            shading_near=shading_results["near_shading"],
            shading_far=shading_results["far_shading"],
            dc_wiring=dc_wiring,
            ac_wiring=ac_wiring,
            transformer=transformer,
            inverter=inverter_loss,
            clipping=clipping,
            availability=100.0 - availability_loss,
            grid_curtailment=kwargs.get("grid_curtailment", 0.0),
            lid=kwargs.get("lid", 1.5),
            mismatch=kwargs.get("mismatch", 1.0),
        )

        logger.info(f"Total system losses: {losses.total_losses():.2f}%")

        return losses

    def loss_waterfall_data(self, losses: SystemLosses) -> Dict[str, float]:
        """
        Generate waterfall chart data for visualization.

        Args:
            losses: SystemLosses object

        Returns:
            Dictionary of loss categories and values for waterfall chart
        """
        waterfall = {
            "Nameplate Power": 100.0,
            "After Soiling": 100.0 - losses.soiling,
            "After Near Shading": None,
            "After Far Shading": None,
            "After LID": None,
            "After Mismatch": None,
            "After DC Wiring": None,
            "After Inverter": None,
            "After Clipping": None,
            "After AC Wiring": None,
            "After Transformer": None,
            "After Availability": None,
            "Final Output": None,
        }

        # Calculate cascading losses
        current_power = 100.0
        for loss_type, loss_value in [
            ("After Soiling", losses.soiling),
            ("After Near Shading", losses.shading_near),
            ("After Far Shading", losses.shading_far),
            ("After LID", losses.lid),
            ("After Mismatch", losses.mismatch),
            ("After DC Wiring", losses.dc_wiring),
            ("After Inverter", losses.inverter),
            ("After Clipping", losses.clipping),
            ("After AC Wiring", losses.ac_wiring),
            ("After Transformer", losses.transformer),
        ]:
            current_power *= (100.0 - loss_value) / 100.0
            waterfall[loss_type] = current_power

        # Apply availability
        current_power *= losses.availability / 100.0
        waterfall["After Availability"] = current_power
        waterfall["Final Output"] = current_power

        return waterfall
