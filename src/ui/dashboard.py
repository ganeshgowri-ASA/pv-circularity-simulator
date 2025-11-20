"""EYA Dashboard - Main dashboard controller.

This module provides the main dashboard controller with core functions for
project overview, energy output analysis, performance ratio, losses waterfall,
and financial metrics.
"""

from typing import Dict, Optional
import pandas as pd
import streamlit as st

from ..models.eya_models import (
    ProjectInfo,
    SystemConfiguration,
    PerformanceMetrics,
    LossBreakdown,
    FinancialMetrics,
)
from ..modules.B05_energy_forecasting.forecaster import EnergyForecaster
from ..modules.B06_energy_yield_analysis.analyzer import EnergyYieldAnalyzer


class EYADashboard:
    """Energy Yield Analysis Dashboard controller.

    This class manages the main dashboard functionality including:
    - Project overview
    - Annual energy output analysis
    - Performance ratio calculations
    - Losses waterfall analysis
    - Financial metrics display

    Attributes:
        forecaster: Energy forecasting engine
        analyzer: Energy yield analysis engine
    """

    def __init__(
        self,
        project_info: ProjectInfo,
        system_config: SystemConfiguration,
        loss_breakdown: Optional[LossBreakdown] = None,
    ):
        """Initialize the EYA Dashboard.

        Args:
            project_info: Project information
            system_config: System configuration
            loss_breakdown: Custom loss breakdown (optional)
        """
        self.project_info = project_info
        self.system_config = system_config
        self.forecaster = EnergyForecaster(project_info, system_config)
        self.analyzer = EnergyYieldAnalyzer(project_info, system_config, loss_breakdown)

        # Cache for computed data
        self._energy_outputs = None
        self._weather_data = None
        self._performance_metrics = None

    def project_overview(self) -> Dict:
        """Generate project overview information.

        Returns:
            Dictionary containing project overview data
        """
        overview = {
            "Project Information": {
                "Project Name": self.project_info.project_name,
                "Location": self.project_info.location,
                "Coordinates": f"{self.project_info.latitude:.4f}°, {self.project_info.longitude:.4f}°",
                "Altitude": f"{self.project_info.altitude:.0f} m",
                "Commissioning Date": self.project_info.commissioning_date.strftime("%Y-%m-%d"),
                "Project Lifetime": f"{self.project_info.project_lifetime} years",
            },
            "System Configuration": {
                "DC Capacity": f"{self.system_config.capacity_dc:.2f} kWp",
                "AC Capacity": f"{self.system_config.capacity_ac:.2f} kWac",
                "Module Type": self.system_config.module_type.value,
                "Module Efficiency": f"{self.system_config.module_efficiency * 100:.2f}%",
                "Module Count": f"{self.system_config.module_count:,}",
                "Inverter Efficiency": f"{self.system_config.inverter_efficiency * 100:.2f}%",
                "Mounting Type": self.system_config.mounting_type.value.replace("_", " ").title(),
                "Tilt Angle": f"{self.system_config.tilt_angle:.1f}°",
                "Azimuth Angle": f"{self.system_config.azimuth_angle:.1f}°",
                "DC/AC Ratio": f"{self.system_config.dc_ac_ratio:.2f}",
                "Ground Coverage Ratio": f"{self.system_config.gcr:.2f}",
            },
        }

        return overview

    def annual_energy_output(self) -> Dict:
        """Calculate and return annual energy output metrics.

        Returns:
            Dictionary containing annual energy output data
        """
        # Generate energy forecast if not cached
        if self._energy_outputs is None:
            self._weather_data = self.forecaster.generate_synthetic_weather_data(
                self.project_info.commissioning_date,
                self.project_info.commissioning_date.replace(year=self.project_info.commissioning_date.year + 1),
            )
            self.forecaster.load_weather_data(self._weather_data)
            self._energy_outputs = self.forecaster.forecast_energy()

        # Calculate annual totals
        annual_dc_energy = sum(output.dc_energy for output in self._energy_outputs)
        annual_ac_energy = sum(output.ac_energy for output in self._energy_outputs)
        annual_exported = sum(output.exported_energy for output in self._energy_outputs)

        # Calculate specific yield
        specific_yield = annual_ac_energy / self.system_config.capacity_dc

        # Calculate capacity factor
        max_possible_energy = self.system_config.capacity_ac * 8760  # hours in year
        capacity_factor = annual_ac_energy / max_possible_energy

        # Monthly breakdown
        monthly_df = self.analyzer.calculate_monthly_performance(self._energy_outputs)

        output_data = {
            "Annual Totals": {
                "DC Energy": f"{annual_dc_energy:,.0f} kWh",
                "AC Energy": f"{annual_ac_energy:,.0f} kWh",
                "Exported Energy": f"{annual_exported:,.0f} kWh",
                "System Losses": f"{annual_dc_energy - annual_ac_energy:,.0f} kWh",
            },
            "Performance Indicators": {
                "Specific Yield": f"{specific_yield:.2f} kWh/kWp",
                "Capacity Factor": f"{capacity_factor * 100:.2f}%",
                "Average Daily Energy": f"{annual_ac_energy / 365:.2f} kWh/day",
                "Peak Month Production": f"{monthly_df['ac_energy'].max():,.0f} kWh",
                "Lowest Month Production": f"{monthly_df['ac_energy'].min():,.0f} kWh",
            },
            "monthly_data": monthly_df,
        }

        return output_data

    def performance_ratio(self) -> Dict:
        """Calculate and return performance ratio analysis.

        Returns:
            Dictionary containing performance ratio data
        """
        # Ensure energy outputs are calculated
        if self._energy_outputs is None:
            self.annual_energy_output()

        # Calculate performance metrics
        self._performance_metrics = self.forecaster.calculate_performance_metrics(
            self._energy_outputs, self._weather_data
        )

        pr_data = {
            "Performance Ratio": {
                "PR": f"{self._performance_metrics.performance_ratio * 100:.2f}%",
                "Reference Yield": f"{self._performance_metrics.reference_yield:.2f} kWh/kWp",
                "Array Yield": f"{self._performance_metrics.array_yield:.2f} kWh/kWp",
                "Final Yield": f"{self._performance_metrics.final_yield:.2f} kWh/kWp",
            },
            "Yield Analysis": {
                "Capture Losses": f"{self._performance_metrics.capture_losses:.2f} kWh/kWp",
                "System Losses": f"{self._performance_metrics.system_losses:.2f} kWh/kWp",
                "Total Losses": f"{self._performance_metrics.capture_losses + self._performance_metrics.system_losses:.2f} kWh/kWp",
            },
            "Efficiency Metrics": {
                "Array Efficiency": f"{(self._performance_metrics.array_yield / self._performance_metrics.reference_yield) * 100:.2f}%",
                "System Efficiency": f"{(self._performance_metrics.final_yield / self._performance_metrics.array_yield) * 100:.2f}%",
                "Overall Efficiency": f"{self._performance_metrics.performance_ratio * 100:.2f}%",
            },
            "metrics": self._performance_metrics,
        }

        return pr_data

    def losses_waterfall(self) -> Dict:
        """Generate losses waterfall analysis.

        Returns:
            Dictionary containing detailed loss breakdown
        """
        # Get detailed loss analysis
        losses = self.analyzer.analyze_losses()

        # Calculate energy impact
        if self._energy_outputs is None:
            self.annual_energy_output()

        annual_ac_energy = sum(output.ac_energy for output in self._energy_outputs)

        # Calculate waterfall data (cumulative losses)
        waterfall_steps = [
            ("Rated Capacity", self.system_config.capacity_dc * 1000, 0),  # Start at DC capacity
            ("Soiling", None, losses["soiling_loss"]),
            ("Shading", None, losses["shading_loss"]),
            ("Snow", None, losses["snow_loss"]),
            ("Mismatch", None, losses["mismatch_loss"]),
            ("Wiring", None, losses["wiring_loss"]),
            ("Connections", None, losses["connection_loss"]),
            ("LID", None, losses["lid_loss"]),
            ("Nameplate", None, losses["nameplate_loss"]),
            ("Age", None, losses["age_loss"]),
            ("Temperature", None, losses["temperature_loss"]),
            ("Inverter", None, losses["inverter_loss"]),
            ("Transformer", None, losses["transformer_loss"]),
            ("Availability", None, losses["availability_loss"]),
        ]

        # Calculate cumulative values
        current_value = waterfall_steps[0][1]
        waterfall_data = []

        for name, value, loss_pct in waterfall_steps:
            if value is not None:
                waterfall_data.append({"stage": name, "value": value, "loss_pct": 0})
                current_value = value
            else:
                loss_value = current_value * (loss_pct / 100)
                current_value -= loss_value
                waterfall_data.append({
                    "stage": name,
                    "value": current_value,
                    "loss_pct": loss_pct,
                    "loss_value": loss_value,
                })

        losses_data = {
            "Total System Loss": f"{losses['total_loss']:.2f}%",
            "Loss Categories": {
                "Optical Losses": f"{losses['optical_losses']:.2f}%",
                "Electrical Losses": f"{losses['electrical_losses']:.2f}%",
                "Degradation Losses": f"{losses['degradation_losses']:.2f}%",
                "Environmental Losses": f"{losses['environmental_losses']:.2f}%",
                "System Availability Losses": f"{losses['system_losses']:.2f}%",
            },
            "Detailed Breakdown": losses,
            "waterfall_data": pd.DataFrame(waterfall_data),
        }

        return losses_data

    def financial_metrics(self, financial_params: FinancialMetrics) -> Dict:
        """Calculate and return financial metrics.

        Args:
            financial_params: Financial parameters (CAPEX, OPEX, energy price, etc.)

        Returns:
            Dictionary containing financial analysis
        """
        # Ensure energy outputs are calculated
        if self._energy_outputs is None:
            self.annual_energy_output()

        annual_energy = sum(output.ac_energy for output in self._energy_outputs)

        # Calculate financial metrics
        financial = self.analyzer.calculate_financial_metrics(annual_energy, financial_params)

        # Calculate lifetime metrics
        degradation_df = self.analyzer.calculate_degradation_impact(
            annual_energy, self.project_info.project_lifetime
        )

        total_lifetime_energy = degradation_df["cumulative_energy_kwh"].iloc[-1]
        total_lifetime_revenue = total_lifetime_energy * financial.energy_price

        financial_data = {
            "Investment Analysis": {
                "CAPEX": f"${financial.capex:,.0f}",
                "Annual OPEX": f"${financial.opex_annual:,.0f}",
                "Total Lifetime OPEX": f"${financial.opex_annual * self.project_info.project_lifetime:,.0f}",
                "Energy Price": f"${financial.energy_price:.4f}/kWh",
            },
            "Economic Metrics": {
                "LCOE": f"${financial.lcoe:.4f}/kWh" if financial.lcoe else "N/A",
                "NPV": f"${financial.npv:,.0f}" if financial.npv else "N/A",
                "IRR": f"{financial.irr * 100:.2f}%" if financial.irr else "N/A",
                "Payback Period": f"{financial.payback_period:.1f} years" if financial.payback_period else "N/A",
            },
            "Revenue Projections": {
                "First Year Revenue": f"${annual_energy * financial.energy_price:,.0f}",
                "Lifetime Energy Production": f"{total_lifetime_energy:,.0f} kWh",
                "Lifetime Revenue": f"${total_lifetime_revenue:,.0f}",
            },
            "degradation_data": degradation_df,
            "financial_metrics": financial,
        }

        return financial_data

    def get_comprehensive_summary(self, financial_params: Optional[FinancialMetrics] = None) -> Dict:
        """Get comprehensive dashboard summary.

        Args:
            financial_params: Optional financial parameters

        Returns:
            Complete dashboard summary
        """
        summary = {
            "project_overview": self.project_overview(),
            "annual_energy": self.annual_energy_output(),
            "performance_ratio": self.performance_ratio(),
            "losses": self.losses_waterfall(),
        }

        if financial_params:
            summary["financial"] = self.financial_metrics(financial_params)

        return summary
