"""
Pydantic models for PV system optimization.

This module defines all data models used in the optimization engine,
including system parameters, constraints, objectives, and results.
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np
from datetime import datetime


class PVSystemParameters(BaseModel):
    """
    Complete PV system parameters for optimization.

    Attributes:
        module_power: Nominal module power in watts
        module_efficiency: Module efficiency (0-1)
        module_area: Module area in square meters
        module_cost: Cost per module in USD
        bifacial: Whether modules are bifacial
        bifaciality: Bifaciality factor (0-1) for bifacial modules
        tracker_type: Type of tracking system
        tilt_angle: Fixed tilt angle in degrees (for fixed tilt)
        azimuth: Azimuth angle in degrees
        gcr: Ground coverage ratio (0-1)
        dc_ac_ratio: DC to AC ratio
        inverter_efficiency: Inverter efficiency (0-1)
        inverter_cost_per_kw: Inverter cost in USD/kW
        land_cost_per_acre: Land cost in USD/acre
        available_land_acres: Available land in acres
        latitude: Site latitude in degrees
        longitude: Site longitude in degrees
        elevation: Site elevation in meters
        albedo: Ground albedo (0-1)
        num_modules: Number of modules (design variable)
        row_spacing: Row spacing in meters (design variable)
        string_length: Number of modules per string
        discount_rate: Discount rate for NPV calculation (0-1)
        project_lifetime: Project lifetime in years
        degradation_rate: Annual degradation rate (0-1)
        om_cost_per_kw_year: O&M cost in USD/kW/year
    """

    # Module parameters
    module_power: float = Field(gt=0, description="Module power in watts")
    module_efficiency: float = Field(ge=0, le=1, description="Module efficiency")
    module_area: float = Field(gt=0, description="Module area in m²")
    module_cost: float = Field(ge=0, description="Module cost in USD")
    bifacial: bool = Field(default=False, description="Bifacial module flag")
    bifaciality: float = Field(default=0.7, ge=0, le=1, description="Bifaciality factor")

    # Tracking system
    tracker_type: Literal["fixed", "single_axis", "dual_axis"] = Field(
        default="fixed", description="Tracker type"
    )
    tilt_angle: float = Field(default=25.0, ge=0, le=90, description="Tilt angle in degrees")
    azimuth: float = Field(default=180.0, ge=0, le=360, description="Azimuth in degrees")

    # Layout parameters (design variables)
    gcr: float = Field(default=0.4, ge=0.1, le=0.9, description="Ground coverage ratio")
    dc_ac_ratio: float = Field(default=1.25, ge=1.0, le=2.0, description="DC/AC ratio")

    # Inverter parameters
    inverter_efficiency: float = Field(default=0.98, ge=0.9, le=1.0, description="Inverter efficiency")
    inverter_cost_per_kw: float = Field(default=100.0, ge=0, description="Inverter cost USD/kW")

    # Site parameters
    land_cost_per_acre: float = Field(default=5000.0, ge=0, description="Land cost USD/acre")
    available_land_acres: float = Field(default=100.0, gt=0, description="Available land in acres")
    latitude: float = Field(ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(ge=-180, le=180, description="Longitude in degrees")
    elevation: float = Field(default=0.0, description="Elevation in meters")
    albedo: float = Field(default=0.2, ge=0, le=1, description="Ground albedo")

    # System sizing (design variables)
    num_modules: int = Field(default=10000, gt=0, description="Number of modules")
    row_spacing: float = Field(default=5.0, gt=0, description="Row spacing in meters")
    string_length: int = Field(default=20, gt=0, description="Modules per string")

    # Economic parameters
    discount_rate: float = Field(default=0.08, ge=0, le=1, description="Discount rate")
    project_lifetime: int = Field(default=25, gt=0, description="Project lifetime in years")
    degradation_rate: float = Field(default=0.005, ge=0, le=0.1, description="Annual degradation")
    om_cost_per_kw_year: float = Field(default=15.0, ge=0, description="O&M cost USD/kW/year")

    model_config = {"frozen": False, "validate_assignment": True}


class OptimizationConstraints(BaseModel):
    """
    Constraints for optimization problem.

    Attributes:
        min_gcr: Minimum ground coverage ratio
        max_gcr: Maximum ground coverage ratio
        min_dc_ac_ratio: Minimum DC/AC ratio
        max_dc_ac_ratio: Maximum DC/AC ratio
        min_tilt: Minimum tilt angle in degrees
        max_tilt: Maximum tilt angle in degrees
        max_land_use_acres: Maximum land use in acres
        min_capacity_mw: Minimum system capacity in MW
        max_capacity_mw: Maximum system capacity in MW
        max_capex: Maximum capital expenditure in USD
        min_energy_yield_kwh_per_kwp: Minimum specific yield in kWh/kWp/year
        max_shading_loss: Maximum acceptable shading loss fraction
        min_bifacial_gain: Minimum bifacial gain fraction (if bifacial)
    """

    min_gcr: float = Field(default=0.2, ge=0, le=1)
    max_gcr: float = Field(default=0.6, ge=0, le=1)
    min_dc_ac_ratio: float = Field(default=1.1, ge=1.0)
    max_dc_ac_ratio: float = Field(default=1.5, ge=1.0)
    min_tilt: float = Field(default=10.0, ge=0, le=90)
    max_tilt: float = Field(default=40.0, ge=0, le=90)
    max_land_use_acres: Optional[float] = Field(default=None, ge=0)
    min_capacity_mw: Optional[float] = Field(default=None, ge=0)
    max_capacity_mw: Optional[float] = Field(default=None, ge=0)
    max_capex: Optional[float] = Field(default=None, ge=0)
    min_energy_yield_kwh_per_kwp: Optional[float] = Field(default=None, ge=0)
    max_shading_loss: float = Field(default=0.1, ge=0, le=1)
    min_bifacial_gain: float = Field(default=0.05, ge=0, le=1)

    @field_validator("max_gcr")
    @classmethod
    def validate_gcr_range(cls, v: float, info: Any) -> float:
        """Validate GCR range."""
        if "min_gcr" in info.data and v < info.data["min_gcr"]:
            raise ValueError("max_gcr must be >= min_gcr")
        return v

    @field_validator("max_dc_ac_ratio")
    @classmethod
    def validate_dc_ac_range(cls, v: float, info: Any) -> float:
        """Validate DC/AC ratio range."""
        if "min_dc_ac_ratio" in info.data and v < info.data["min_dc_ac_ratio"]:
            raise ValueError("max_dc_ac_ratio must be >= min_dc_ac_ratio")
        return v

    model_config = {"frozen": False}


class OptimizationObjectives(BaseModel):
    """
    Multi-objective optimization objectives and weights.

    Attributes:
        maximize_energy: Weight for energy yield maximization
        minimize_lcoe: Weight for LCOE minimization
        minimize_land_use: Weight for land use minimization
        maximize_npv: Weight for NPV maximization
        minimize_shading: Weight for shading loss minimization
        maximize_bifacial_gain: Weight for bifacial gain maximization
    """

    maximize_energy: float = Field(default=1.0, ge=0, le=1)
    minimize_lcoe: float = Field(default=1.0, ge=0, le=1)
    minimize_land_use: float = Field(default=0.5, ge=0, le=1)
    maximize_npv: float = Field(default=0.8, ge=0, le=1)
    minimize_shading: float = Field(default=0.6, ge=0, le=1)
    maximize_bifacial_gain: float = Field(default=0.3, ge=0, le=1)

    def normalize(self) -> "OptimizationObjectives":
        """Normalize weights to sum to 1."""
        total = (
            self.maximize_energy
            + self.minimize_lcoe
            + self.minimize_land_use
            + self.maximize_npv
            + self.minimize_shading
            + self.maximize_bifacial_gain
        )
        if total == 0:
            return self

        return OptimizationObjectives(
            maximize_energy=self.maximize_energy / total,
            minimize_lcoe=self.minimize_lcoe / total,
            minimize_land_use=self.minimize_land_use / total,
            maximize_npv=self.maximize_npv / total,
            minimize_shading=self.minimize_shading / total,
            maximize_bifacial_gain=self.maximize_bifacial_gain / total,
        )

    model_config = {"frozen": False}


class DesignPoint(BaseModel):
    """
    Single design point in the optimization space.

    Attributes:
        gcr: Ground coverage ratio
        dc_ac_ratio: DC/AC ratio
        tilt_angle: Tilt angle in degrees
        num_modules: Number of modules
        row_spacing: Row spacing in meters
        annual_energy_kwh: Annual energy production in kWh
        capacity_mw: System capacity in MW
        lcoe: Levelized cost of energy in USD/kWh
        npv: Net present value in USD
        land_use_acres: Land use in acres
        shading_loss: Shading loss fraction
        bifacial_gain: Bifacial gain fraction (if applicable)
        capex: Capital expenditure in USD
    """

    # Design variables
    gcr: float = Field(ge=0, le=1)
    dc_ac_ratio: float = Field(ge=1.0)
    tilt_angle: float = Field(ge=0, le=90)
    num_modules: int = Field(gt=0)
    row_spacing: float = Field(gt=0)

    # Performance metrics
    annual_energy_kwh: float = Field(ge=0)
    capacity_mw: float = Field(ge=0)
    lcoe: float = Field(ge=0)
    npv: float
    land_use_acres: float = Field(ge=0)
    shading_loss: float = Field(ge=0, le=1)
    bifacial_gain: float = Field(default=0.0, ge=0)
    capex: float = Field(ge=0)

    model_config = {"frozen": False}


class ParetoSolution(BaseModel):
    """
    Solution on the Pareto frontier.

    Attributes:
        design: Design point
        objectives: Dictionary of objective values
        rank: Pareto rank (0 = non-dominated)
        crowding_distance: Crowding distance for diversity
    """

    design: DesignPoint
    objectives: Dict[str, float]
    rank: int = Field(ge=0)
    crowding_distance: float = Field(ge=0)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


class OptimizationResult(BaseModel):
    """
    Results from an optimization run.

    Attributes:
        algorithm: Optimization algorithm used
        best_solution: Best solution found
        pareto_front: List of Pareto-optimal solutions
        convergence_history: Convergence history over iterations
        execution_time_seconds: Total execution time
        num_iterations: Number of iterations performed
        num_evaluations: Number of function evaluations
        success: Whether optimization succeeded
        message: Status message
        metadata: Additional metadata
    """

    algorithm: Literal["genetic", "pso", "linear", "multi_objective"]
    best_solution: DesignPoint
    pareto_front: List[ParetoSolution] = Field(default_factory=list)
    convergence_history: List[float] = Field(default_factory=list)
    execution_time_seconds: float = Field(ge=0)
    num_iterations: int = Field(ge=0)
    num_evaluations: int = Field(ge=0)
    success: bool
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


class SensitivityResult(BaseModel):
    """
    Results from sensitivity analysis.

    Attributes:
        parameter_name: Name of parameter being varied
        parameter_values: Array of parameter values tested
        output_values: Corresponding output values
        sensitivity_index: Sensitivity index (normalized)
        correlation: Correlation coefficient with output
    """

    parameter_name: str
    parameter_values: List[float]
    output_values: List[float]
    sensitivity_index: float
    correlation: float = Field(ge=-1, le=1)

    model_config = {"frozen": False}
