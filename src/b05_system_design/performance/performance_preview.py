"""
Performance Preview module for PV systems.

This module provides comprehensive performance estimation including annual energy
yield, performance ratio, loss analysis, financial preview, and report generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import io

from src.models.pv_components import SystemDesign, SiteLocation


@dataclass
class EnergyEstimate:
    """Energy production estimate."""
    annual_energy: float  # kWh/year
    monthly_energy: List[float]  # kWh/month
    specific_yield: float  # kWh/kWp/year
    capacity_factor: float  # %
    first_year_output: float  # kWh
    degradation_rate: float = 0.5  # %/year


@dataclass
class PerformanceRatio:
    """Performance ratio calculation."""
    pr_value: float  # Performance ratio (0-1)
    reference_yield: float  # kWh/kWp
    final_yield: float  # kWh/kWp
    losses_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class FinancialMetrics:
    """Financial analysis metrics."""
    total_cost: float  # $
    lcoe: float  # $/kWh
    payback_period: float  # years
    npv: float  # $ Net present value
    irr: float  # % Internal rate of return
    annual_savings: float  # $/year
    lifetime_savings: float  # $


@dataclass
class LossAnalysis:
    """Detailed loss analysis."""
    poa_irradiance: float  # kWh/m²/year
    dc_energy: float  # kWh
    ac_energy: float  # kWh
    losses: Dict[str, float] = field(default_factory=dict)
    loss_percentages: Dict[str, float] = field(default_factory=dict)


class PerformancePreview:
    """
    Performance preview and estimation engine.

    Calculates energy production, performance ratio, losses, and financial
    metrics for PV system designs.
    """

    def __init__(self):
        """Initialize the PerformancePreview engine."""
        self.monthly_irradiance_default = [
            3.2, 4.1, 5.3, 6.2, 6.8, 7.1, 7.0, 6.5, 5.6, 4.5, 3.5, 2.9  # kWh/m²/day
        ]

    def annual_energy_estimate(
        self,
        design: SystemDesign,
        site: Optional[SiteLocation] = None
    ) -> EnergyEstimate:
        """
        Calculate annual energy production estimate.

        Args:
            design: System design
            site: Site location (optional, uses defaults if not provided)

        Returns:
            EnergyEstimate with annual and monthly production
        """
        # System capacity
        dc_capacity = design.total_dc_power  # kW

        # Get irradiance data (use defaults or site-specific)
        if site and site.annual_ghi:
            annual_irradiance = site.annual_ghi  # kWh/m²/year
            monthly_irradiance = self._estimate_monthly_irradiance(annual_irradiance)
        else:
            # Use default values (typical for mid-latitude location)
            monthly_irradiance = self.monthly_irradiance_default
            annual_irradiance = sum(m * 30 for m in monthly_irradiance)  # Approximate

        # Calculate POA (Plane of Array) irradiance
        if design.mounting:
            tilt = design.mounting.tilt_angle
            azimuth = design.mounting.azimuth
        else:
            tilt = 25.0
            azimuth = 180.0

        # Tilt factor (simplified - assumes optimal tilt increases irradiance)
        if site:
            optimal_tilt = abs(site.latitude)
        else:
            optimal_tilt = 35.0

        tilt_factor = 1.0 + 0.01 * (optimal_tilt - abs(tilt - optimal_tilt))
        tilt_factor = max(0.85, min(1.15, tilt_factor))  # Limit to ±15%

        # Azimuth factor (south-facing = 1.0, east/west = ~0.9)
        azimuth_deviation = abs(180 - azimuth)
        azimuth_factor = 1.0 - (azimuth_deviation / 180) * 0.15

        # Monthly energy production
        monthly_energy = []
        for month_irrad in monthly_irradiance:
            # POA irradiance for month
            poa_irrad = month_irrad * 30 * tilt_factor * azimuth_factor  # kWh/m²/month

            # DC energy production
            # Standard test conditions: 1000 W/m²
            dc_energy = dc_capacity * (poa_irrad / 1.0)  # kWh

            # Apply system losses
            loss_factor = 1.0 - (design.total_system_losses / 100.0)
            ac_energy = dc_energy * loss_factor

            # Apply inverter efficiency
            if design.inverters:
                avg_efficiency = np.mean([inv.euro_efficiency for inv in design.inverters]) / 100.0
            else:
                avg_efficiency = 0.97

            final_energy = ac_energy * avg_efficiency

            monthly_energy.append(final_energy)

        # Annual totals
        annual_energy = sum(monthly_energy)

        # Calculate metrics
        specific_yield = annual_energy / dc_capacity if dc_capacity > 0 else 0
        hours_per_year = 8760
        capacity_factor = (annual_energy / (dc_capacity * hours_per_year)) * 100 if dc_capacity > 0 else 0

        # Get degradation rate from modules
        if design.modules and len(design.modules) > 0:
            degradation_rate = design.modules[0].module.degradation_rate
        else:
            degradation_rate = 0.5

        first_year_output = annual_energy

        return EnergyEstimate(
            annual_energy=annual_energy,
            monthly_energy=monthly_energy,
            specific_yield=specific_yield,
            capacity_factor=capacity_factor,
            first_year_output=first_year_output,
            degradation_rate=degradation_rate
        )

    def pr_calculation(
        self,
        design: SystemDesign,
        site: Optional[SiteLocation] = None
    ) -> PerformanceRatio:
        """
        Calculate Performance Ratio (PR).

        PR = Final Yield / Reference Yield

        Args:
            design: System design
            site: Site location

        Returns:
            PerformanceRatio with detailed breakdown
        """
        # Get energy estimate
        energy_estimate = self.annual_energy_estimate(design, site)

        # Reference yield (irradiance-based)
        # Assumes ~1800 kWh/m²/year for reference location
        reference_irradiance = 1800.0  # kWh/m²/year
        reference_yield = reference_irradiance / 1.0  # kWh/kWp

        # Final yield (actual production)
        final_yield = energy_estimate.specific_yield

        # Performance ratio
        pr_value = final_yield / reference_yield if reference_yield > 0 else 0

        # Detailed losses breakdown
        losses_breakdown = design.system_losses.copy()

        # Add inverter losses
        if design.inverters:
            avg_efficiency = np.mean([inv.euro_efficiency for inv in design.inverters])
            inverter_loss = 100 - avg_efficiency
            losses_breakdown['inverter'] = inverter_loss

        # Add DC/AC clipping losses
        if design.total_ac_power > 0:
            dc_ac_ratio = design.total_dc_power / design.total_ac_power
            if dc_ac_ratio > 1.2:
                clipping_loss = (dc_ac_ratio - 1.2) * 2.0  # Rough estimate
                losses_breakdown['clipping'] = min(clipping_loss, 5.0)

        return PerformanceRatio(
            pr_value=pr_value,
            reference_yield=reference_yield,
            final_yield=final_yield,
            losses_breakdown=losses_breakdown
        )

    def shading_loss_summary(
        self,
        design: SystemDesign
    ) -> Dict[str, float]:
        """
        Summarize shading losses.

        Args:
            design: System design

        Returns:
            Dictionary with shading loss breakdown
        """
        # Extract shading loss from system losses
        shading_loss = design.system_losses.get('shading', 3.0)

        # Breakdown by source
        summary = {
            'near_shading': shading_loss * 0.6,  # Trees, structures
            'far_shading': shading_loss * 0.2,   # Horizon
            'row_to_row': shading_loss * 0.2,    # Inter-row shading
            'total': shading_loss
        }

        # Adjust for mounting type
        if design.mounting:
            if design.mounting.is_tracking:
                # Tracking systems have less shading
                summary = {k: v * 0.7 for k, v in summary.items()}
            elif design.mounting.mounting_type.value == "rooftop":
                # Rooftop may have more shading
                summary = {k: v * 1.2 for k, v in summary.items()}

        return summary

    def financial_preview(
        self,
        design: SystemDesign,
        site: Optional[SiteLocation] = None,
        electricity_rate: float = 0.12,  # $/kWh
        system_cost_per_watt: float = 2.50,  # $/W
        discount_rate: float = 0.05,  # 5%
        lifetime_years: int = 25
    ) -> FinancialMetrics:
        """
        Calculate financial metrics for the system.

        Args:
            design: System design
            site: Site location
            electricity_rate: Cost of electricity ($/kWh)
            system_cost_per_watt: System cost ($/W)
            discount_rate: Discount rate for NPV calculation
            lifetime_years: System lifetime (years)

        Returns:
            FinancialMetrics with cost and savings analysis
        """
        # System cost
        dc_capacity_w = design.total_dc_power * 1000
        total_cost = dc_capacity_w * system_cost_per_watt

        # Energy production
        energy_estimate = self.annual_energy_estimate(design, site)
        first_year_energy = energy_estimate.annual_energy
        degradation_rate = energy_estimate.degradation_rate / 100.0

        # Annual revenue/savings
        first_year_savings = first_year_energy * electricity_rate

        # Calculate lifetime metrics
        lifetime_energy = 0
        lifetime_savings = 0
        npv = -total_cost  # Initial investment

        for year in range(1, lifetime_years + 1):
            # Energy production with degradation
            year_energy = first_year_energy * ((1 - degradation_rate) ** (year - 1))
            lifetime_energy += year_energy

            # Revenue/savings
            year_savings = year_energy * electricity_rate
            lifetime_savings += year_savings

            # NPV calculation
            discounted_savings = year_savings / ((1 + discount_rate) ** year)
            npv += discounted_savings

        # LCOE calculation
        # LCOE = Total lifetime cost / Total lifetime energy
        # Simplified: assumes no O&M costs
        discounted_energy = sum(
            first_year_energy * ((1 - degradation_rate) ** (year - 1)) / ((1 + discount_rate) ** year)
            for year in range(1, lifetime_years + 1)
        )
        lcoe = total_cost / discounted_energy if discounted_energy > 0 else 0

        # Simple payback period
        payback_period = total_cost / first_year_savings if first_year_savings > 0 else 0

        # IRR calculation (simplified)
        # Using NPV formula, solve for rate where NPV = 0
        # Approximate using goal-seek approach
        irr = self._calculate_irr(
            total_cost, first_year_savings, degradation_rate, lifetime_years
        )

        return FinancialMetrics(
            total_cost=total_cost,
            lcoe=lcoe,
            payback_period=payback_period,
            npv=npv,
            irr=irr,
            annual_savings=first_year_savings,
            lifetime_savings=lifetime_savings
        )

    def _calculate_irr(
        self,
        initial_cost: float,
        first_year_savings: float,
        degradation_rate: float,
        years: int
    ) -> float:
        """Calculate Internal Rate of Return using iterative method."""
        # Simple approximation
        # Try different rates to find where NPV = 0
        for rate in np.linspace(0.01, 0.30, 100):
            npv = -initial_cost
            for year in range(1, years + 1):
                year_savings = first_year_savings * ((1 - degradation_rate) ** (year - 1))
                npv += year_savings / ((1 + rate) ** year)

            if npv < 0:
                return (rate - 0.003) * 100  # Return rate just before NPV goes negative

        return 15.0  # Default if calculation doesn't converge

    def export_design_report(
        self,
        design: SystemDesign,
        site: Optional[SiteLocation] = None,
        format: str = "pdf"
    ) -> io.BytesIO:
        """
        Export comprehensive design report.

        Args:
            design: System design
            site: Site location
            format: Export format ("pdf", "excel", "json")

        Returns:
            BytesIO buffer with report data
        """
        # Calculate all metrics
        energy_estimate = self.annual_energy_estimate(design, site)
        pr = self.pr_calculation(design, site)
        shading = self.shading_loss_summary(design)
        financial = self.financial_preview(design, site)

        if format == "excel":
            return self._export_excel_report(
                design, energy_estimate, pr, shading, financial
            )
        elif format == "json":
            return self._export_json_report(
                design, energy_estimate, pr, shading, financial
            )
        else:  # Default to Excel for this implementation
            return self._export_excel_report(
                design, energy_estimate, pr, shading, financial
            )

    def _export_excel_report(
        self,
        design: SystemDesign,
        energy: EnergyEstimate,
        pr: PerformanceRatio,
        shading: Dict[str, float],
        financial: FinancialMetrics
    ) -> io.BytesIO:
        """Export report as Excel file."""
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Design Name',
                    'DC Capacity (kW)',
                    'AC Capacity (kW)',
                    'DC/AC Ratio',
                    'Module Count',
                    'Annual Energy (kWh)',
                    'Specific Yield (kWh/kWp)',
                    'Capacity Factor (%)',
                    'Performance Ratio',
                    'Total Cost ($)',
                    'LCOE ($/kWh)',
                    'Payback Period (years)',
                    'NPV ($)',
                    'IRR (%)'
                ],
                'Value': [
                    design.design_name,
                    f"{design.total_dc_power:.2f}",
                    f"{design.total_ac_power:.2f}",
                    f"{design.dc_ac_ratio:.2f}",
                    design.total_modules_count,
                    f"{energy.annual_energy:.0f}",
                    f"{energy.specific_yield:.0f}",
                    f"{energy.capacity_factor:.1f}",
                    f"{pr.pr_value:.3f}",
                    f"{financial.total_cost:,.0f}",
                    f"{financial.lcoe:.3f}",
                    f"{financial.payback_period:.1f}",
                    f"{financial.npv:,.0f}",
                    f"{financial.irr:.1f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Monthly production
            monthly_data = {
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Energy (kWh)': [f"{e:.0f}" for e in energy.monthly_energy]
            }
            pd.DataFrame(monthly_data).to_excel(writer, sheet_name='Monthly Production', index=False)

            # Losses breakdown
            losses_data = {
                'Loss Category': list(pr.losses_breakdown.keys()),
                'Loss (%)': [f"{v:.2f}" for v in pr.losses_breakdown.values()]
            }
            pd.DataFrame(losses_data).to_excel(writer, sheet_name='Losses', index=False)

            # Component details
            if design.modules:
                module_data = []
                for idx, config in enumerate(design.modules):
                    module_data.append({
                        'String #': idx + 1,
                        'Module': f"{config.module.manufacturer} {config.module.model}",
                        'Modules per String': config.modules_per_string,
                        'Number of Strings': config.num_strings,
                        'Total Modules': config.total_modules,
                        'Total Power (kW)': config.total_power / 1000
                    })
                pd.DataFrame(module_data).to_excel(writer, sheet_name='Modules', index=False)

            if design.inverters:
                inverter_counts = {}
                for inv in design.inverters:
                    key = f"{inv.manufacturer} {inv.model}"
                    if key not in inverter_counts:
                        inverter_counts[key] = {'count': 0, 'inverter': inv}
                    inverter_counts[key]['count'] += 1

                inverter_data = []
                for name, data in inverter_counts.items():
                    inv = data['inverter']
                    inverter_data.append({
                        'Model': name,
                        'Quantity': data['count'],
                        'AC Power (kW)': inv.p_ac_rated / 1000,
                        'Total AC (kW)': (inv.p_ac_rated * data['count']) / 1000,
                        'Efficiency (%)': inv.euro_efficiency
                    })
                pd.DataFrame(inverter_data).to_excel(writer, sheet_name='Inverters', index=False)

        buffer.seek(0)
        return buffer

    def _export_json_report(
        self,
        design: SystemDesign,
        energy: EnergyEstimate,
        pr: PerformanceRatio,
        shading: Dict[str, float],
        financial: FinancialMetrics
    ) -> io.BytesIO:
        """Export report as JSON file."""
        import json

        report = {
            'design': {
                'name': design.design_name,
                'dc_capacity_kw': design.total_dc_power,
                'ac_capacity_kw': design.total_ac_power,
                'dc_ac_ratio': design.dc_ac_ratio,
                'module_count': design.total_modules_count
            },
            'performance': {
                'annual_energy_kwh': energy.annual_energy,
                'specific_yield': energy.specific_yield,
                'capacity_factor': energy.capacity_factor,
                'performance_ratio': pr.pr_value,
                'monthly_energy': energy.monthly_energy
            },
            'losses': pr.losses_breakdown,
            'shading': shading,
            'financial': {
                'total_cost': financial.total_cost,
                'lcoe': financial.lcoe,
                'payback_period': financial.payback_period,
                'npv': financial.npv,
                'irr': financial.irr,
                'annual_savings': financial.annual_savings,
                'lifetime_savings': financial.lifetime_savings
            }
        }

        buffer = io.BytesIO()
        buffer.write(json.dumps(report, indent=2).encode('utf-8'))
        buffer.seek(0)
        return buffer

    def _estimate_monthly_irradiance(self, annual_irradiance: float) -> List[float]:
        """Estimate monthly irradiance from annual value."""
        # Use typical distribution pattern
        monthly_pattern = np.array(self.monthly_irradiance_default)
        scale_factor = annual_irradiance / (sum(monthly_pattern) * 30)
        return (monthly_pattern * scale_factor).tolist()

    def plot_monthly_production(self, energy: EnergyEstimate) -> go.Figure:
        """Create monthly production chart."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=months,
            y=energy.monthly_energy,
            marker_color='steelblue',
            text=[f"{e:.0f}" for e in energy.monthly_energy],
            textposition='outside'
        ))

        fig.update_layout(
            title="Monthly Energy Production",
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            height=400
        )

        return fig

    def plot_loss_waterfall(self, pr: PerformanceRatio) -> go.Figure:
        """Create loss waterfall chart."""
        # Sort losses by magnitude
        sorted_losses = sorted(pr.losses_breakdown.items(), key=lambda x: x[1], reverse=True)

        categories = ['POA Irradiance'] + [name for name, _ in sorted_losses] + ['Final Output']
        values = [100] + [-loss for _, loss in sorted_losses] + [0]

        # Calculate cumulative for positioning
        cumulative = [100]
        for _, loss in sorted_losses:
            cumulative.append(cumulative[-1] - loss)

        fig = go.Figure(go.Waterfall(
            x=categories,
            y=values,
            measure=['relative'] + ['relative'] * len(sorted_losses) + ['total'],
            text=[f"{v:.1f}%" if v != 0 else f"{cumulative[-1]:.1f}%" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="Energy Loss Waterfall",
            yaxis_title="Energy (%)",
            showlegend=False,
            height=500
        )

        return fig
