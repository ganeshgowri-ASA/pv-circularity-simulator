"""
EconomicOptimizer: Economic optimization for PV systems.

This module implements optimization methods focused on economic objectives
including LCOE minimization, NPV maximization, and component selection.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize, differential_evolution

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    DesignPoint,
)


class EconomicOptimizer:
    """
    Economic optimization for PV systems.

    This class provides methods to optimize economic performance through
    LCOE minimization, NPV maximization, DC/AC ratio optimization,
    module selection, and balance of system optimization.

    Attributes:
        parameters: PV system parameters
        constraints: Optimization constraints
    """

    def __init__(
        self,
        parameters: PVSystemParameters,
        constraints: OptimizationConstraints,
    ):
        """
        Initialize economic optimizer.

        Args:
            parameters: PV system parameters
            constraints: Optimization constraints
        """
        self.parameters = parameters
        self.constraints = constraints

    def minimize_lcoe(
        self,
        vary_dc_ac_ratio: bool = True,
        vary_gcr: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Minimize levelized cost of energy (LCOE).

        LCOE is calculated as:
        LCOE = (CAPEX + PV(O&M)) / Total Energy Production

        Args:
            vary_dc_ac_ratio: Whether to optimize DC/AC ratio
            vary_gcr: Whether to optimize GCR

        Returns:
            Tuple of (minimum_lcoe, optimal_parameters)
        """
        def calculate_lcoe(x: np.ndarray) -> float:
            """Calculate LCOE for given parameters."""
            if vary_dc_ac_ratio and vary_gcr:
                dc_ac_ratio, gcr = x
            elif vary_dc_ac_ratio:
                dc_ac_ratio = x[0]
                gcr = self.parameters.gcr
            elif vary_gcr:
                gcr = x[0]
                dc_ac_ratio = self.parameters.dc_ac_ratio
            else:
                dc_ac_ratio = self.parameters.dc_ac_ratio
                gcr = self.parameters.gcr

            # System sizing
            capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6
            inverter_capacity_kw = capacity_mw * 1000 / dc_ac_ratio

            # CAPEX calculation
            module_capex = self.parameters.num_modules * self.parameters.module_cost
            inverter_capex = inverter_capacity_kw * self.parameters.inverter_cost_per_kw

            # Land use
            module_area_total = self.parameters.num_modules * self.parameters.module_area
            land_area_acres = (module_area_total / gcr) / 4046.86
            land_capex = land_area_acres * self.parameters.land_cost_per_acre

            # BOS costs (racking, wiring, etc.)
            bos_capex = capacity_mw * 250000  # $250k/MW

            total_capex = module_capex + inverter_capex + land_capex + bos_capex

            # Annual energy production
            annual_energy_kwh = self._estimate_annual_energy(
                capacity_mw, dc_ac_ratio, gcr
            )

            # Lifetime energy with degradation
            total_energy_kwh = sum(
                annual_energy_kwh * (1 - self.parameters.degradation_rate) ** year
                for year in range(self.parameters.project_lifetime)
            )

            # O&M costs (present value)
            annual_om = self.parameters.om_cost_per_kw_year * capacity_mw * 1000
            om_pv = sum(
                annual_om / (1 + self.parameters.discount_rate) ** year
                for year in range(1, self.parameters.project_lifetime + 1)
            )

            # LCOE
            lcoe = (total_capex + om_pv) / total_energy_kwh if total_energy_kwh > 0 else 999

            return lcoe

        # Set up optimization
        if vary_dc_ac_ratio and vary_gcr:
            x0 = [self.parameters.dc_ac_ratio, self.parameters.gcr]
            bounds = [
                (self.constraints.min_dc_ac_ratio, self.constraints.max_dc_ac_ratio),
                (self.constraints.min_gcr, self.constraints.max_gcr),
            ]
        elif vary_dc_ac_ratio:
            x0 = [self.parameters.dc_ac_ratio]
            bounds = [(self.constraints.min_dc_ac_ratio, self.constraints.max_dc_ac_ratio)]
        elif vary_gcr:
            x0 = [self.parameters.gcr]
            bounds = [(self.constraints.min_gcr, self.constraints.max_gcr)]
        else:
            # Nothing to optimize
            lcoe = calculate_lcoe(np.array([]))
            return lcoe, {
                "lcoe": lcoe,
                "dc_ac_ratio": self.parameters.dc_ac_ratio,
                "gcr": self.parameters.gcr,
            }

        # Optimize
        result = minimize(
            calculate_lcoe,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
        )

        optimal_lcoe = result.fun

        if vary_dc_ac_ratio and vary_gcr:
            optimal_dc_ac, optimal_gcr = result.x
        elif vary_dc_ac_ratio:
            optimal_dc_ac = result.x[0]
            optimal_gcr = self.parameters.gcr
        else:
            optimal_dc_ac = self.parameters.dc_ac_ratio
            optimal_gcr = result.x[0]

        return optimal_lcoe, {
            "lcoe": optimal_lcoe,
            "dc_ac_ratio": optimal_dc_ac,
            "gcr": optimal_gcr,
            "success": result.success,
        }

    def maximize_npv(
        self,
        electricity_price: float = 0.06,
        vary_capacity: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Maximize net present value (NPV).

        NPV = PV(Revenues) - CAPEX - PV(O&M)

        Args:
            electricity_price: Electricity price in $/kWh
            vary_capacity: Whether to optimize system capacity

        Returns:
            Tuple of (maximum_npv, optimal_parameters)
        """
        def calculate_npv(num_modules: int) -> float:
            """Calculate NPV for given system size."""
            # System sizing
            capacity_mw = num_modules * self.parameters.module_power / 1e6
            inverter_capacity_kw = capacity_mw * 1000 / self.parameters.dc_ac_ratio

            # CAPEX
            module_capex = num_modules * self.parameters.module_cost
            inverter_capex = inverter_capacity_kw * self.parameters.inverter_cost_per_kw

            module_area_total = num_modules * self.parameters.module_area
            land_area_acres = (module_area_total / self.parameters.gcr) / 4046.86
            land_capex = land_area_acres * self.parameters.land_cost_per_acre

            bos_capex = capacity_mw * 250000

            total_capex = module_capex + inverter_capex + land_capex + bos_capex

            # Annual energy
            annual_energy_kwh = self._estimate_annual_energy(
                capacity_mw,
                self.parameters.dc_ac_ratio,
                self.parameters.gcr,
            )

            # Revenue (present value)
            revenue_pv = sum(
                annual_energy_kwh
                * (1 - self.parameters.degradation_rate) ** year
                * electricity_price
                / (1 + self.parameters.discount_rate) ** year
                for year in range(1, self.parameters.project_lifetime + 1)
            )

            # O&M costs (present value)
            annual_om = self.parameters.om_cost_per_kw_year * capacity_mw * 1000
            om_pv = sum(
                annual_om / (1 + self.parameters.discount_rate) ** year
                for year in range(1, self.parameters.project_lifetime + 1)
            )

            # NPV
            npv = revenue_pv - total_capex - om_pv

            return -npv  # Negative for minimization

        if vary_capacity:
            # Optimize capacity (number of modules)
            max_modules = int(
                self.parameters.available_land_acres
                * 4046.86
                * self.parameters.gcr
                / self.parameters.module_area
            )

            result = minimize(
                lambda x: calculate_npv(int(x[0])),
                [self.parameters.num_modules],
                bounds=[(1000, max_modules)],
                method='L-BFGS-B',
            )

            optimal_modules = int(result.x[0])
            optimal_npv = -result.fun
        else:
            optimal_modules = self.parameters.num_modules
            optimal_npv = -calculate_npv(optimal_modules)

        optimal_capacity_mw = optimal_modules * self.parameters.module_power / 1e6

        return optimal_npv, {
            "npv": optimal_npv,
            "num_modules": optimal_modules,
            "capacity_mw": optimal_capacity_mw,
            "electricity_price": electricity_price,
        }

    def optimize_dc_ac_ratio(
        self,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize DC/AC ratio for best economic performance.

        The DC/AC ratio affects:
        - Inverter costs (lower ratio = more inverters)
        - Clipping losses (higher ratio = more clipping)
        - Energy yield
        - LCOE

        Returns:
            Tuple of (optimal_dc_ac_ratio, performance_metrics)
        """
        def objective(dc_ac_ratio: float) -> float:
            """Calculate economic metric (LCOE) for given DC/AC ratio."""
            capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6
            inverter_capacity_kw = capacity_mw * 1000 / dc_ac_ratio

            # Costs
            inverter_capex = inverter_capacity_kw * self.parameters.inverter_cost_per_kw

            # Energy with clipping
            annual_energy_kwh = self._estimate_annual_energy(
                capacity_mw, dc_ac_ratio, self.parameters.gcr
            )

            # Clipping losses (simplified model)
            clipping_loss = max(0, (dc_ac_ratio - 1.2) * 0.05)
            annual_energy_kwh *= 1 - clipping_loss

            # LCOE impact (simplified)
            # Just optimize based on clipping vs inverter cost tradeoff
            metric = inverter_capex / (annual_energy_kwh + 1)

            return metric

        result = minimize(
            lambda x: objective(x[0]),
            [self.parameters.dc_ac_ratio],
            bounds=[(self.constraints.min_dc_ac_ratio, self.constraints.max_dc_ac_ratio)],
            method='L-BFGS-B',
        )

        optimal_dc_ac = result.x[0]

        # Calculate metrics at optimal point
        capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6
        clipping_loss = max(0, (optimal_dc_ac - 1.2) * 0.05)
        inverter_capacity_kw = capacity_mw * 1000 / optimal_dc_ac
        inverter_cost = inverter_capacity_kw * self.parameters.inverter_cost_per_kw

        return optimal_dc_ac, {
            "dc_ac_ratio": optimal_dc_ac,
            "clipping_loss": clipping_loss,
            "inverter_capacity_kw": inverter_capacity_kw,
            "inverter_cost": inverter_cost,
            "success": result.success,
        }

    def optimize_module_selection(
        self,
        module_options: List[Dict[str, float]],
    ) -> Tuple[int, Dict[str, float]]:
        """
        Optimize module selection from available options.

        Args:
            module_options: List of module specifications, each containing:
                - power: Module power in watts
                - efficiency: Module efficiency
                - cost: Module cost in USD
                - area: Module area in m²

        Returns:
            Tuple of (best_module_index, performance_metrics)
        """
        best_lcoe = float('inf')
        best_idx = 0
        results = []

        for idx, module in enumerate(module_options):
            # Calculate system performance with this module
            params = self.parameters.model_copy(deep=True)
            params.module_power = module['power']
            params.module_efficiency = module['efficiency']
            params.module_cost = module['cost']
            params.module_area = module['area']

            # Calculate LCOE
            capacity_mw = params.num_modules * params.module_power / 1e6
            annual_energy = self._estimate_annual_energy(
                capacity_mw, params.dc_ac_ratio, params.gcr
            )

            module_capex = params.num_modules * params.module_cost
            inverter_capacity_kw = capacity_mw * 1000 / params.dc_ac_ratio
            inverter_capex = inverter_capacity_kw * params.inverter_cost_per_kw
            bos_capex = capacity_mw * 250000

            total_capex = module_capex + inverter_capex + bos_capex

            total_energy = sum(
                annual_energy * (1 - params.degradation_rate) ** year
                for year in range(params.project_lifetime)
            )

            annual_om = params.om_cost_per_kw_year * capacity_mw * 1000
            om_pv = sum(
                annual_om / (1 + params.discount_rate) ** year
                for year in range(1, params.project_lifetime + 1)
            )

            lcoe = (total_capex + om_pv) / total_energy if total_energy > 0 else 999

            results.append({
                "module_idx": idx,
                "module_power": module['power'],
                "lcoe": lcoe,
                "annual_energy_kwh": annual_energy,
                "total_capex": total_capex,
            })

            if lcoe < best_lcoe:
                best_lcoe = lcoe
                best_idx = idx

        return best_idx, results[best_idx]

    def balance_of_system_optimization(
        self,
    ) -> Dict[str, float]:
        """
        Optimize balance of system (BOS) components.

        This includes:
        - Racking and mounting structures
        - Electrical infrastructure (wiring, transformers)
        - Installation costs
        - Project development costs

        Returns:
            Dictionary with optimized BOS cost breakdown
        """
        capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6

        # BOS cost components ($/W-DC)
        racking_cost_per_w = 0.10  # Can be optimized based on GCR, tilt, etc.
        electrical_cost_per_w = 0.08
        installation_cost_per_w = 0.15
        development_cost_per_w = 0.05

        # Optimization: reduce racking cost with higher GCR (economies of scale)
        gcr_factor = 1.0 - 0.2 * (self.parameters.gcr - 0.3)
        racking_cost_per_w *= gcr_factor

        # Optimization: reduce electrical cost with larger system
        size_factor = 1.0 - 0.1 * min(1.0, capacity_mw / 100)
        electrical_cost_per_w *= size_factor

        # Total BOS
        total_bos_per_w = (
            racking_cost_per_w
            + electrical_cost_per_w
            + installation_cost_per_w
            + development_cost_per_w
        )

        total_bos_cost = total_bos_per_w * capacity_mw * 1e6

        return {
            "racking_cost_per_w": racking_cost_per_w,
            "electrical_cost_per_w": electrical_cost_per_w,
            "installation_cost_per_w": installation_cost_per_w,
            "development_cost_per_w": development_cost_per_w,
            "total_bos_per_w": total_bos_per_w,
            "total_bos_cost": total_bos_cost,
            "capacity_mw": capacity_mw,
        }

    def _estimate_annual_energy(
        self,
        capacity_mw: float,
        dc_ac_ratio: float,
        gcr: float,
    ) -> float:
        """
        Estimate annual energy production.

        Args:
            capacity_mw: System capacity in MW
            dc_ac_ratio: DC/AC ratio
            gcr: Ground coverage ratio

        Returns:
            Annual energy in kWh
        """
        # Base peak sun hours
        psh = 5.0 - 0.02 * abs(self.parameters.latitude)

        # Clipping losses
        clipping_loss = max(0, (dc_ac_ratio - 1.2) * 0.05)

        # Shading losses (function of GCR)
        shading_loss = max(0, (gcr - 0.35) * 0.15)

        # System efficiency
        system_efficiency = (
            self.parameters.inverter_efficiency
            * (1 - clipping_loss)
            * (1 - shading_loss)
            * 0.95  # Other losses
        )

        annual_energy_kwh = capacity_mw * 1000 * psh * 365 * system_efficiency

        return annual_energy_kwh
