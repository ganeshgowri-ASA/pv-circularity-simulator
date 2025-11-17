"""Degradation modeling for PV modules."""

from typing import List, Dict
from pydantic import BaseModel, Field
import numpy as np


class DegradationResult(BaseModel):
    """Result of degradation calculation."""

    years: List[int] = Field(..., description="Year numbers")
    power_retention_pct: List[float] = Field(..., description="Power retention percentage")
    absolute_power_w: List[float] = Field(..., description="Absolute power in watts")
    cumulative_degradation_pct: List[float] = Field(..., description="Cumulative degradation percentage")


class DegradationModel:
    """Model for PV module degradation over time."""

    def __init__(
        self,
        initial_power_w: float,
        initial_degradation_pct: float = 2.0,
        annual_degradation_pct: float = 0.5,
        degradation_mode: str = "linear",
    ):
        """Initialize degradation model.

        Args:
            initial_power_w: Initial module power in watts
            initial_degradation_pct: First year degradation percentage
            annual_degradation_pct: Annual degradation after first year
            degradation_mode: Degradation mode ('linear', 'exponential', 'piecewise')
        """
        self.initial_power_w = initial_power_w
        self.initial_degradation_pct = initial_degradation_pct
        self.annual_degradation_pct = annual_degradation_pct
        self.degradation_mode = degradation_mode

    def calculate_linear(self, years: int = 25) -> DegradationResult:
        """Calculate linear degradation model.

        Args:
            years: Number of years to project

        Returns:
            DegradationResult with yearly projections
        """
        year_list = list(range(years + 1))
        power_retention = []
        absolute_power = []
        cumulative_degradation = []

        for year in year_list:
            if year == 0:
                retention = 100.0
            elif year == 1:
                retention = 100.0 - self.initial_degradation_pct
            else:
                retention = (100.0 - self.initial_degradation_pct) - (
                    self.annual_degradation_pct * (year - 1)
                )

            power = self.initial_power_w * (retention / 100)
            deg = 100.0 - retention

            power_retention.append(retention)
            absolute_power.append(power)
            cumulative_degradation.append(deg)

        return DegradationResult(
            years=year_list,
            power_retention_pct=power_retention,
            absolute_power_w=absolute_power,
            cumulative_degradation_pct=cumulative_degradation,
        )

    def calculate_exponential(self, years: int = 25) -> DegradationResult:
        """Calculate exponential degradation model.

        Args:
            years: Number of years to project

        Returns:
            DegradationResult with yearly projections
        """
        year_list = list(range(years + 1))
        power_retention = []
        absolute_power = []
        cumulative_degradation = []

        # First year degradation
        year_1_factor = 1 - (self.initial_degradation_pct / 100)

        for year in year_list:
            if year == 0:
                retention = 100.0
            elif year == 1:
                retention = 100.0 * year_1_factor
            else:
                # Exponential decay after year 1
                annual_factor = 1 - (self.annual_degradation_pct / 100)
                retention = 100.0 * year_1_factor * (annual_factor ** (year - 1))

            power = self.initial_power_w * (retention / 100)
            deg = 100.0 - retention

            power_retention.append(retention)
            absolute_power.append(power)
            cumulative_degradation.append(deg)

        return DegradationResult(
            years=year_list,
            power_retention_pct=power_retention,
            absolute_power_w=absolute_power,
            cumulative_degradation_pct=cumulative_degradation,
        )

    def calculate_piecewise(self, years: int = 25, breakpoints: Dict[int, float] = None) -> DegradationResult:
        """Calculate piecewise degradation model with different rates at different periods.

        Args:
            years: Number of years to project
            breakpoints: Dictionary mapping year to degradation rate change

        Returns:
            DegradationResult with yearly projections
        """
        if breakpoints is None:
            # Default: faster degradation in first 5 years, then slower
            breakpoints = {
                0: self.initial_degradation_pct,
                1: self.annual_degradation_pct * 1.2,
                5: self.annual_degradation_pct,
                15: self.annual_degradation_pct * 0.8,
            }

        year_list = list(range(years + 1))
        power_retention = []
        absolute_power = []
        cumulative_degradation = []

        current_retention = 100.0
        current_rate = self.initial_degradation_pct

        for year in year_list:
            if year in breakpoints:
                current_rate = breakpoints[year]

            if year == 0:
                retention = 100.0
            else:
                current_retention -= current_rate
                retention = current_retention

            power = self.initial_power_w * (retention / 100)
            deg = 100.0 - retention

            power_retention.append(retention)
            absolute_power.append(power)
            cumulative_degradation.append(deg)

        return DegradationResult(
            years=year_list,
            power_retention_pct=power_retention,
            absolute_power_w=absolute_power,
            cumulative_degradation_pct=cumulative_degradation,
        )

    def calculate(self, years: int = 25) -> DegradationResult:
        """Calculate degradation based on selected mode.

        Args:
            years: Number of years to project

        Returns:
            DegradationResult with yearly projections
        """
        if self.degradation_mode == "linear":
            return self.calculate_linear(years)
        elif self.degradation_mode == "exponential":
            return self.calculate_exponential(years)
        elif self.degradation_mode == "piecewise":
            return self.calculate_piecewise(years)
        else:
            return self.calculate_linear(years)

    def calculate_warranty_compliance(
        self, years: int = 25, warranty_requirements: Dict[int, float] = None
    ) -> Dict[str, any]:
        """Check if degradation meets warranty requirements.

        Args:
            years: Number of years to check
            warranty_requirements: Dictionary mapping year to minimum required power retention %

        Returns:
            Dictionary with compliance results
        """
        if warranty_requirements is None:
            # Typical tier-1 warranty: 98% @ 1yr, 90% @ 10yr, 84.8% @ 25yr
            warranty_requirements = {
                1: 98.0,
                5: 95.0,
                10: 90.0,
                15: 87.4,
                20: 84.8,
                25: 82.2,
            }

        result = self.calculate(years)

        compliance = {}
        all_compliant = True

        for year, required_retention in warranty_requirements.items():
            if year <= years:
                actual_retention = result.power_retention_pct[year]
                compliant = actual_retention >= required_retention
                compliance[year] = {
                    "required_pct": required_retention,
                    "actual_pct": actual_retention,
                    "compliant": compliant,
                    "margin_pct": actual_retention - required_retention,
                }
                if not compliant:
                    all_compliant = False

        return {"all_compliant": all_compliant, "yearly_compliance": compliance}
