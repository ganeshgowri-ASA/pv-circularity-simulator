"""Inverter selection and sizing with database integration."""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
from pathlib import Path

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    InverterType,
    SystemType,
)

logger = logging.getLogger(__name__)


class InverterSelector:
    """
    Inverter selection and sizing with optimization for DC/AC ratio and clipping.

    Includes database search functionality and comparative analysis between
    central, string, and microinverter configurations.
    """

    def __init__(
        self,
        module: ModuleParameters,
        system_type: SystemType = SystemType.UTILITY,
        database_path: Optional[str] = None,
    ):
        """
        Initialize inverter selector.

        Args:
            module: Module parameters for string sizing
            system_type: Type of PV system (utility/commercial/residential)
            database_path: Path to inverter database JSON file
        """
        self.module = module
        self.system_type = system_type
        self.database_path = database_path
        self.inverter_database: List[InverterParameters] = []

        if database_path and Path(database_path).exists():
            self._load_database(database_path)

        logger.info(
            f"Initialized InverterSelector for {system_type.value} system "
            f"with {len(self.inverter_database)} inverters in database"
        )

    def _load_database(self, database_path: str) -> None:
        """
        Load inverter database from JSON file.

        Args:
            database_path: Path to JSON database file
        """
        try:
            with open(database_path, 'r') as f:
                data = json.load(f)

            for inv_data in data.get('inverters', []):
                # Parse inverter data to InverterParameters
                inverter = InverterParameters(**inv_data)
                self.inverter_database.append(inverter)

            logger.info(f"Loaded {len(self.inverter_database)} inverters from database")

        except Exception as e:
            logger.error(f"Failed to load inverter database: {e}")
            self.inverter_database = []

    def search_inverter_database(
        self,
        dc_power_kw: float,
        inverter_type: Optional[InverterType] = None,
        manufacturer: Optional[str] = None,
        voltage_range: Optional[Tuple[float, float]] = None,
    ) -> List[InverterParameters]:
        """
        Search inverter database for suitable inverters.

        Args:
            dc_power_kw: Target DC power in kW
            inverter_type: Filter by inverter type
            manufacturer: Filter by manufacturer name
            voltage_range: Required (min, max) voltage range in volts

        Returns:
            List of matching InverterParameters sorted by suitability
        """
        if not self.inverter_database:
            logger.warning("Inverter database is empty")
            return []

        matching_inverters = []

        for inverter in self.inverter_database:
            # Filter by type
            if inverter_type and inverter.inverter_type != inverter_type:
                continue

            # Filter by manufacturer
            if manufacturer and manufacturer.lower() not in inverter.manufacturer.lower():
                continue

            # Filter by voltage range
            if voltage_range:
                min_v, max_v = voltage_range
                if inverter.mppt_vmin > min_v or inverter.mppt_vmax < max_v:
                    continue

            # Check power rating (should be within reasonable range)
            ac_power_kw = inverter.pac_max / 1000.0
            power_ratio = dc_power_kw / ac_power_kw

            # Accept if DC/AC ratio is between 0.8 and 1.5
            if 0.8 <= power_ratio <= 1.5:
                matching_inverters.append(inverter)

        # Sort by how close the power rating is to target
        matching_inverters.sort(
            key=lambda inv: abs(dc_power_kw - (inv.pac_max / 1000.0))
        )

        logger.info(
            f"Found {len(matching_inverters)} matching inverters for {dc_power_kw}kW DC"
        )

        return matching_inverters

    def calculate_dc_ac_ratio(
        self,
        dc_power_kw: float,
        ac_power_kw: float,
    ) -> float:
        """
        Calculate DC to AC power ratio.

        Args:
            dc_power_kw: DC power in kW
            ac_power_kw: AC power in kW

        Returns:
            DC/AC ratio (typically 1.15-1.35 for optimal design)
        """
        if ac_power_kw <= 0:
            raise ValueError("AC power must be greater than 0")

        dc_ac_ratio = dc_power_kw / ac_power_kw

        logger.debug(f"DC/AC ratio: {dc_ac_ratio:.3f} ({dc_power_kw}kW DC / {ac_power_kw}kW AC)")

        return dc_ac_ratio

    def analyze_clipping_losses(
        self,
        dc_power_profile: np.ndarray,
        inverter_ac_max_kw: float,
        inverter_efficiency: float = 0.98,
    ) -> Dict[str, float]:
        """
        Analyze power clipping losses for a given DC power profile.

        Args:
            dc_power_profile: Array of DC power values in kW (hourly or sub-hourly)
            inverter_ac_max_kw: Inverter maximum AC power output in kW
            inverter_efficiency: Average inverter efficiency (0-1)

        Returns:
            Dictionary with clipping analysis:
                - total_dc_energy: Total DC energy (kWh)
                - clipped_energy: Energy lost to clipping (kWh)
                - clipping_loss_percent: Clipping as percentage of DC energy
                - hours_clipped: Number of hours with clipping
                - max_clipped_power: Maximum clipped power (kW)
        """
        # Calculate AC power output
        ac_power_profile = dc_power_profile * inverter_efficiency

        # Identify clipping
        clipped_mask = ac_power_profile > inverter_ac_max_kw
        ac_power_clipped = np.minimum(ac_power_profile, inverter_ac_max_kw)

        # Calculate clipped energy (assuming hourly data)
        total_dc_energy = np.sum(dc_power_profile)
        clipped_energy = np.sum(ac_power_profile - ac_power_clipped)
        clipping_loss_percent = (clipped_energy / total_dc_energy * 100) if total_dc_energy > 0 else 0.0

        hours_clipped = np.sum(clipped_mask)
        max_clipped_power = np.max(ac_power_profile - ac_power_clipped) if hours_clipped > 0 else 0.0

        logger.info(
            f"Clipping analysis: {clipping_loss_percent:.2f}% loss, "
            f"{hours_clipped} hours clipped, max clip: {max_clipped_power:.1f}kW"
        )

        return {
            "total_dc_energy": float(total_dc_energy),
            "clipped_energy": float(clipped_energy),
            "clipping_loss_percent": float(clipping_loss_percent),
            "hours_clipped": int(hours_clipped),
            "max_clipped_power": float(max_clipped_power),
        }

    def optimize_dc_ac_ratio(
        self,
        dc_power_profile: np.ndarray,
        target_clipping_percent: float = 2.0,
        inverter_efficiency: float = 0.98,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize DC/AC ratio for target clipping losses.

        Args:
            dc_power_profile: Array of DC power values in kW
            target_clipping_percent: Target clipping loss percentage (typically 1-3%)
            inverter_efficiency: Average inverter efficiency (0-1)

        Returns:
            Tuple of (optimal_dc_ac_ratio, clipping_analysis_dict)
        """
        # Find peak DC power
        peak_dc_kw = np.max(dc_power_profile)

        if peak_dc_kw <= 0:
            raise ValueError("DC power profile must have positive values")

        # Iterate to find optimal ratio
        best_ratio = 1.0
        best_diff = float('inf')

        for ratio in np.arange(1.0, 1.5, 0.05):
            inverter_ac_max_kw = peak_dc_kw / ratio
            clipping_analysis = self.analyze_clipping_losses(
                dc_power_profile, inverter_ac_max_kw, inverter_efficiency
            )

            diff = abs(clipping_analysis["clipping_loss_percent"] - target_clipping_percent)

            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio

        # Final analysis with optimal ratio
        optimal_inverter_ac_kw = peak_dc_kw / best_ratio
        final_analysis = self.analyze_clipping_losses(
            dc_power_profile, optimal_inverter_ac_kw, inverter_efficiency
        )

        logger.info(
            f"Optimal DC/AC ratio: {best_ratio:.3f} "
            f"(clipping: {final_analysis['clipping_loss_percent']:.2f}%)"
        )

        return best_ratio, final_analysis

    def central_vs_string_analysis(
        self,
        total_dc_power_kw: float,
        num_strings: int,
    ) -> Dict[str, any]:
        """
        Compare central inverter vs. string inverter configurations.

        Args:
            total_dc_power_kw: Total DC power of the system in kW
            num_strings: Number of strings in the array

        Returns:
            Dictionary comparing central and string inverter options
        """
        analysis = {
            "total_dc_power_kw": total_dc_power_kw,
            "num_strings": num_strings,
            "central": {},
            "string": {},
        }

        # Central inverter option (1-3 large inverters)
        num_central_inverters = max(1, int(np.ceil(total_dc_power_kw / 1000)))  # 1 per MW
        central_power_per_inv = total_dc_power_kw / num_central_inverters

        analysis["central"] = {
            "num_inverters": num_central_inverters,
            "power_per_inverter_kw": central_power_per_inv,
            "advantages": [
                "Lower cost per kW",
                "Fewer monitoring points",
                "Centralized maintenance",
            ],
            "disadvantages": [
                "Single point of failure",
                "Higher DC wiring losses",
                "Requires inverter building/pad",
                "Lower MPPT granularity",
            ],
            "typical_efficiency": 98.5,
            "relative_cost": 1.0,  # Base cost
        }

        # String inverter option (distributed)
        strings_per_inverter = 10  # Typical for utility-scale string inverters
        num_string_inverters = max(1, int(np.ceil(num_strings / strings_per_inverter)))
        string_power_per_inv = total_dc_power_kw / num_string_inverters

        analysis["string"] = {
            "num_inverters": num_string_inverters,
            "power_per_inverter_kw": string_power_per_inv,
            "advantages": [
                "Better MPPT granularity",
                "Reduced DC wiring",
                "No single point of failure",
                "Easier installation",
            ],
            "disadvantages": [
                "Higher cost per kW",
                "More monitoring points",
                "Distributed maintenance",
            ],
            "typical_efficiency": 98.0,
            "relative_cost": 1.15,  # ~15% higher cost
        }

        # Recommendation based on system type
        if self.system_type == SystemType.UTILITY:
            if total_dc_power_kw > 2000:
                analysis["recommendation"] = "central"
                analysis["reason"] = "Large utility-scale: central inverters more cost-effective"
            else:
                analysis["recommendation"] = "string"
                analysis["reason"] = "Mid-size utility: string inverters balance cost and performance"

        elif self.system_type == SystemType.COMMERCIAL:
            analysis["recommendation"] = "string"
            analysis["reason"] = "Commercial scale: string inverters optimal for reliability"

        else:  # RESIDENTIAL
            analysis["recommendation"] = "string"
            analysis["reason"] = "Residential: string inverters or microinverters appropriate"

        logger.info(
            f"Central vs. String analysis: {num_central_inverters} central "
            f"or {num_string_inverters} string inverters, recommend: {analysis['recommendation']}"
        )

        return analysis

    def microinverter_design(
        self,
        num_modules: int,
        module_power_w: float,
    ) -> Dict[str, any]:
        """
        Design microinverter system for residential applications.

        Args:
            num_modules: Total number of modules
            module_power_w: Module power rating in watts

        Returns:
            Dictionary with microinverter system design
        """
        # Typical microinverter handles 1-4 modules
        modules_per_microinverter = 2  # Common configuration

        num_microinverters = int(np.ceil(num_modules / modules_per_microinverter))
        total_dc_power_kw = (num_modules * module_power_w) / 1000

        design = {
            "num_modules": num_modules,
            "module_power_w": module_power_w,
            "modules_per_microinverter": modules_per_microinverter,
            "num_microinverters": num_microinverters,
            "total_dc_power_kw": total_dc_power_kw,
            "advantages": [
                "Module-level MPPT",
                "No high-voltage DC",
                "Easy expansion",
                "Module-level monitoring",
                "Better shading performance",
            ],
            "disadvantages": [
                "Highest cost per watt",
                "Roof-mounted electronics (heat)",
                "Many potential failure points",
                "Complex monitoring",
            ],
            "typical_efficiency": 96.5,
            "relative_cost": 1.3,  # ~30% higher than string
            "suitable_for": [
                "Complex roof layouts",
                "Shading issues",
                "Future expansion plans",
            ],
        }

        logger.info(
            f"Microinverter design: {num_microinverters} units for {num_modules} modules"
        )

        return design

    def inverter_mppt_optimization(
        self,
        inverter: InverterParameters,
        module: ModuleParameters,
        strings_per_mppt: int,
        site_temp_range: Tuple[float, float] = (-10, 70),
    ) -> Dict[str, any]:
        """
        Optimize MPPT configuration for maximum energy harvest.

        Args:
            inverter: Inverter parameters
            module: Module parameters
            strings_per_mppt: Number of strings per MPPT input
            site_temp_range: (min, max) site temperature in °C

        Returns:
            Dictionary with MPPT optimization analysis
        """
        temp_min, temp_max = site_temp_range

        # Calculate voltage range at STC
        vmp_stc = module.vmp
        voc_stc = module.voc

        # Temperature corrections
        temp_coeff_v = module.temp_coeff_voc / 100.0

        vmp_cold = vmp_stc * (1 + temp_coeff_v * (temp_min - 25))
        vmp_hot = vmp_stc * (1 + temp_coeff_v * (temp_max - 25))
        voc_cold = voc_stc * (1 + temp_coeff_v * (temp_min - 25))

        # MPPT tracking analysis
        mppt_utilization = {
            "mppt_range_v": (inverter.mppt_vmin, inverter.mppt_vmax),
            "operating_range_v": (vmp_hot, vmp_cold),
            "vmp_at_25c": vmp_stc,
            "vmp_at_cold": vmp_cold,
            "vmp_at_hot": vmp_hot,
            "mppt_headroom_low": vmp_hot - inverter.mppt_vmin,
            "mppt_headroom_high": inverter.mppt_vmax - vmp_cold,
        }

        # Check if voltage is well-centered in MPPT range
        mppt_center = (inverter.mppt_vmin + inverter.mppt_vmax) / 2
        vmp_deviation_percent = abs(vmp_stc - mppt_center) / mppt_center * 100

        if vmp_deviation_percent < 10:
            mppt_rating = "Excellent"
        elif vmp_deviation_percent < 20:
            mppt_rating = "Good"
        elif vmp_deviation_percent < 30:
            mppt_rating = "Acceptable"
        else:
            mppt_rating = "Poor"

        mppt_utilization["rating"] = mppt_rating
        mppt_utilization["vmp_deviation_percent"] = vmp_deviation_percent

        # String current
        string_imp = module.imp * strings_per_mppt
        current_utilization_percent = string_imp / inverter.idc_max * 100

        mppt_utilization["string_current_a"] = string_imp
        mppt_utilization["inverter_max_current_a"] = inverter.idc_max
        mppt_utilization["current_utilization_percent"] = current_utilization_percent

        logger.info(
            f"MPPT optimization: {mppt_rating} rating, "
            f"Vmp deviation: {vmp_deviation_percent:.1f}%, "
            f"current utilization: {current_utilization_percent:.1f}%"
        )

        return mppt_utilization

    def select_optimal_inverter(
        self,
        dc_power_kw: float,
        target_dc_ac_ratio: float = 1.25,
        preferred_manufacturer: Optional[str] = None,
    ) -> Optional[InverterParameters]:
        """
        Select optimal inverter from database.

        Args:
            dc_power_kw: Target DC power in kW
            target_dc_ac_ratio: Preferred DC/AC ratio
            preferred_manufacturer: Preferred manufacturer name

        Returns:
            Best matching InverterParameters or None if no match found
        """
        # Calculate target AC power
        target_ac_kw = dc_power_kw / target_dc_ac_ratio

        # Search database
        candidates = self.search_inverter_database(
            dc_power_kw=dc_power_kw,
            manufacturer=preferred_manufacturer,
        )

        if not candidates:
            logger.warning(f"No inverters found for {dc_power_kw}kW DC power")
            return None

        # Rank candidates by DC/AC ratio proximity to target
        best_inverter = None
        best_score = float('inf')

        for inverter in candidates:
            ac_kw = inverter.pac_max / 1000.0
            dc_ac_ratio = dc_power_kw / ac_kw
            ratio_diff = abs(dc_ac_ratio - target_dc_ac_ratio)

            # Also consider efficiency
            efficiency_score = (100 - inverter.max_efficiency) / 10.0

            # Combined score (lower is better)
            score = ratio_diff + efficiency_score

            if score < best_score:
                best_score = score
                best_inverter = inverter

        if best_inverter:
            logger.info(
                f"Selected inverter: {best_inverter.manufacturer} {best_inverter.model} "
                f"({best_inverter.pac_max/1000:.1f}kW AC, {best_inverter.max_efficiency}% eff)"
            )

        return best_inverter
