"""
CTM Test Report Generation for IEC 63202.

This module generates comprehensive CTM test reports including:
- CTM ratio calculation and uncertainty analysis
- Comparison with IEC 63202 requirements
- Loss breakdown visualization
- Cell and module IV curve comparison
- Compliance status and certification
- Export to PDF and Excel formats
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pv_circularity_simulator.core.iec63202.models import (
    CTMTestResult,
    CTMCertificate,
    IVCurveData,
)

logger = logging.getLogger(__name__)


class CTMTestReport:
    """
    CTM test report generator with visualization and export capabilities.

    This class generates comprehensive IEC 63202 CTM test reports including:
    - Executive summary with pass/fail status
    - Detailed CTM ratio analysis with uncertainty
    - Loss breakdown by category
    - IV curve comparisons (cell vs. module)
    - Statistical analysis of measurements
    - Compliance verification against IEC 63202
    - Export to multiple formats (PDF, Excel, HTML)

    Attributes:
        test_result: CTM test result data
        certificate: Optional IEC 63202 certificate
    """

    def __init__(
        self,
        test_result: CTMTestResult,
        certificate: Optional[CTMCertificate] = None
    ) -> None:
        """
        Initialize CTM test report generator.

        Args:
            test_result: Complete CTM test result
            certificate: Optional IEC 63202 compliance certificate
        """
        self.test_result = test_result
        self.certificate = certificate

        logger.info(
            f"CTM Test Report initialized for test {test_result.config.test_id}"
        )

    def generate_summary(self) -> Dict[str, any]:
        """
        Generate executive summary of CTM test results.

        Returns:
            Dictionary containing summary statistics and compliance status

        Example:
            >>> report = CTMTestReport(test_result)
            >>> summary = report.generate_summary()
            >>> print(f"CTM Ratio: {summary['ctm_ratio']:.2f}%")
            >>> print(f"Status: {summary['compliance_status']}")
        """
        summary = {
            "test_id": self.test_result.config.test_id,
            "test_date": self.test_result.config.test_date,
            "laboratory": self.test_result.config.laboratory,
            "operator": self.test_result.config.operator,
            "cell_technology": self.test_result.config.cell_properties.technology,
            "num_cells": self.test_result.config.module_config.total_cells,
            "num_cells_tested": len(self.test_result.cell_measurements),
            "num_modules_tested": len(self.test_result.module_measurements),
            "cell_power_avg": self.test_result.cell_power_avg,
            "cell_power_std": self.test_result.cell_power_std,
            "module_power_avg": self.test_result.module_power_avg,
            "module_power_std": self.test_result.module_power_std,
            "expected_module_power": self.test_result.expected_module_power,
            "ctm_ratio": self.test_result.ctm_ratio,
            "ctm_uncertainty": self.test_result.ctm_ratio_uncertainty,
            "total_loss": self.test_result.loss_components.total_loss,
            "compliance_status": "PASS" if self.test_result.compliance_status else "FAIL",
            "certified": self.certificate is not None,
        }

        logger.info(f"Summary generated: CTM={summary['ctm_ratio']:.2f}%")

        return summary

    def create_iv_curve_comparison(self) -> go.Figure:
        """
        Create IV curve comparison plot (cell vs. module).

        Generates a multi-panel plot showing:
        - Cell IV curves (individual and average)
        - Module IV curves (individual and average)
        - Power curves for both

        Returns:
            Plotly figure with IV curve comparisons

        Example:
            >>> fig = report.create_iv_curve_comparison()
            >>> fig.show()
            >>> fig.write_html("ctm_iv_curves.html")
        """
        # Create subplots: 2 rows x 2 cols
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Cell IV Curves",
                "Cell Power Curves",
                "Module IV Curves",
                "Module Power Curves"
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        # Plot cell IV curves
        for i, cell_iv in enumerate(self.test_result.cell_measurements):
            fig.add_trace(
                go.Scatter(
                    x=cell_iv.voltage,
                    y=cell_iv.current,
                    name=f"Cell {i+1}",
                    mode="lines",
                    line=dict(width=1),
                    showlegend=True,
                ),
                row=1, col=1
            )

            # Calculate and plot power
            power = [v * i for v, i in zip(cell_iv.voltage, cell_iv.current)]
            fig.add_trace(
                go.Scatter(
                    x=cell_iv.voltage,
                    y=power,
                    name=f"Cell {i+1} Power",
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                ),
                row=1, col=2
            )

        # Plot module IV curves
        for i, module_iv in enumerate(self.test_result.module_measurements):
            fig.add_trace(
                go.Scatter(
                    x=module_iv.voltage,
                    y=module_iv.current,
                    name=f"Module {i+1}",
                    mode="lines",
                    line=dict(width=2),
                    showlegend=True,
                ),
                row=2, col=1
            )

            # Calculate and plot power
            power = [v * i for v, i in zip(module_iv.voltage, module_iv.current)]
            fig.add_trace(
                go.Scatter(
                    x=module_iv.voltage,
                    y=power,
                    name=f"Module {i+1} Power",
                    mode="lines",
                    line=dict(width=2),
                    showlegend=False,
                ),
                row=2, col=2
            )

        # Update axes labels
        fig.update_xaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text="Voltage (V)", row=1, col=2)
        fig.update_xaxes(title_text="Voltage (V)", row=2, col=1)
        fig.update_xaxes(title_text="Voltage (V)", row=2, col=2)

        fig.update_yaxes(title_text="Current (A)", row=1, col=1)
        fig.update_yaxes(title_text="Power (W)", row=1, col=2)
        fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        fig.update_yaxes(title_text="Power (W)", row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text=f"CTM Test IV Curves - Test {self.test_result.config.test_id}",
            height=800,
            showlegend=True,
        )

        logger.info("IV curve comparison plot created")

        return fig

    def create_loss_waterfall_chart(self) -> go.Figure:
        """
        Create waterfall chart showing cumulative CTM losses.

        The waterfall chart visualizes how power is reduced from initial
        cell power through various loss mechanisms to final module power.

        Returns:
            Plotly waterfall chart figure

        Example:
            >>> fig = report.create_loss_waterfall_chart()
            >>> fig.show()
        """
        losses = self.test_result.loss_components

        # Build waterfall data
        categories = [
            "Cell Power × N",
            "Reflection Loss",
            "Absorption Loss",
            "Shading Loss",
            "Series R Loss",
            "Mismatch Loss",
            "Thermal Loss",
            "Spatial Loss",
            "Spectral Loss",
            "Module Power"
        ]

        values = [
            100.0,  # Start at 100%
            -losses.optical_reflection,
            -losses.optical_absorption,
            -losses.optical_shading,
            -losses.electrical_series_resistance,
            -losses.electrical_mismatch,
            -losses.thermal_assembly,
            -losses.spatial_non_uniformity,
            -losses.spectral_mismatch,
            None,  # Final value calculated automatically
        ]

        # Determine measure type for each bar
        measures = ["absolute"] + ["relative"] * 8 + ["total"]

        fig = go.Figure(go.Waterfall(
            name="CTM Losses",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            text=[f"{v:.2f}%" if v is not None else "" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}},
        ))

        fig.update_layout(
            title=f"CTM Power Loss Waterfall - Test {self.test_result.config.test_id}",
            xaxis_title="Loss Category",
            yaxis_title="Relative Power (%)",
            showlegend=False,
            height=600,
        )

        logger.info("Loss waterfall chart created")

        return fig

    def create_loss_breakdown_pie(self) -> go.Figure:
        """
        Create pie chart showing CTM loss breakdown by category.

        Returns:
            Plotly pie chart figure

        Example:
            >>> fig = report.create_loss_breakdown_pie()
            >>> fig.show()
        """
        losses = self.test_result.loss_components

        labels = [
            "Optical Reflection",
            "Optical Absorption",
            "Optical Shading",
            "Electrical Series R",
            "Electrical Mismatch",
            "Thermal",
            "Spatial Non-uniformity",
            "Spectral Mismatch"
        ]

        values = [
            losses.optical_reflection,
            losses.optical_absorption,
            losses.optical_shading,
            losses.electrical_series_resistance,
            losses.electrical_mismatch,
            losses.thermal_assembly,
            losses.spatial_non_uniformity,
            losses.spectral_mismatch,
        ]

        # Filter out zero values
        labels_filtered = [l for l, v in zip(labels, values) if v > 0.01]
        values_filtered = [v for v in values if v > 0.01]

        fig = go.Figure(data=[go.Pie(
            labels=labels_filtered,
            values=values_filtered,
            hole=0.3,
            textinfo="label+percent",
            textposition="auto",
        )])

        fig.update_layout(
            title=f"CTM Loss Breakdown - Total: {losses.total_loss:.2f}%",
            height=500,
        )

        logger.info("Loss breakdown pie chart created")

        return fig

    def create_compliance_dashboard(self) -> go.Figure:
        """
        Create compliance dashboard showing pass/fail status.

        Returns:
            Plotly figure with compliance indicators

        Example:
            >>> fig = report.create_compliance_dashboard()
            >>> fig.show()
        """
        from pv_circularity_simulator.core.utils.constants import IEC_63202_COMPLIANCE

        # Compliance criteria
        criteria = {
            "CTM Ratio Range": {
                "value": self.test_result.ctm_ratio,
                "min": self.test_result.config.acceptance_criteria_min,
                "max": self.test_result.config.acceptance_criteria_max,
                "pass": (
                    self.test_result.config.acceptance_criteria_min <=
                    self.test_result.ctm_ratio <=
                    self.test_result.config.acceptance_criteria_max
                )
            },
            "Measurement Uncertainty": {
                "value": self.test_result.ctm_ratio_uncertainty,
                "max": IEC_63202_COMPLIANCE["max_uncertainty"],
                "pass": self.test_result.ctm_ratio_uncertainty <= IEC_63202_COMPLIANCE["max_uncertainty"]
            },
            "Overall Compliance": {
                "value": 1.0 if self.test_result.compliance_status else 0.0,
                "pass": self.test_result.compliance_status
            }
        }

        # Create indicator plots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=list(criteria.keys()),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )

        # CTM Ratio gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=criteria["CTM Ratio Range"]["value"],
                title={"text": "CTM Ratio (%)"},
                delta={"reference": 100.0},
                gauge={
                    "axis": {"range": [90, 105]},
                    "bar": {"color": "green" if criteria["CTM Ratio Range"]["pass"] else "red"},
                    "steps": [
                        {"range": [90, criteria["CTM Ratio Range"]["min"]], "color": "lightgray"},
                        {"range": [criteria["CTM Ratio Range"]["max"], 105], "color": "lightgray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": criteria["CTM Ratio Range"]["max"]
                    }
                }
            ),
            row=1, col=1
        )

        # Uncertainty gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=criteria["Measurement Uncertainty"]["value"],
                title={"text": "Uncertainty (%)"},
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": "green" if criteria["Measurement Uncertainty"]["pass"] else "red"},
                    "steps": [
                        {"range": [0, criteria["Measurement Uncertainty"]["max"]], "color": "lightgreen"},
                        {"range": [criteria["Measurement Uncertainty"]["max"], 5], "color": "lightcoral"},
                    ],
                }
            ),
            row=1, col=2
        )

        # Overall compliance
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=criteria["Overall Compliance"]["value"],
                title={"text": "Overall Status"},
                number={"suffix": "", "font": {"size": 60}},
                delta={
                    "reference": 0.5,
                    "position": "bottom",
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ),
            row=1, col=3
        )

        fig.update_layout(
            title_text=f"IEC 63202 Compliance Dashboard - Test {self.test_result.config.test_id}",
            height=400,
        )

        logger.info("Compliance dashboard created")

        return fig

    def export_to_excel(self, file_path: str) -> None:
        """
        Export CTM test report to Excel format.

        Creates a multi-sheet Excel workbook containing:
        - Summary sheet with key results
        - Cell measurements data
        - Module measurements data
        - Loss breakdown
        - Compliance status

        Args:
            file_path: Output file path (.xlsx)

        Example:
            >>> report.export_to_excel("ctm_test_report.xlsx")
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary = self.generate_summary()
            summary_df = pd.DataFrame([summary]).T
            summary_df.columns = ["Value"]
            summary_df.to_excel(writer, sheet_name="Summary")

            # Cell measurements
            cell_data = []
            for i, iv in enumerate(self.test_result.cell_measurements, 1):
                cell_data.append({
                    "Cell": i,
                    "Voc (V)": iv.voc,
                    "Isc (A)": iv.isc,
                    "Vmp (V)": iv.vmp,
                    "Imp (A)": iv.imp,
                    "Pmax (W)": iv.pmax,
                    "FF": iv.fill_factor,
                    "Temp (°C)": iv.temperature,
                    "Irradiance (W/m²)": iv.irradiance,
                })
            cell_df = pd.DataFrame(cell_data)
            cell_df.to_excel(writer, sheet_name="Cell Measurements", index=False)

            # Module measurements
            module_data = []
            for i, iv in enumerate(self.test_result.module_measurements, 1):
                module_data.append({
                    "Module": i,
                    "Voc (V)": iv.voc,
                    "Isc (A)": iv.isc,
                    "Vmp (V)": iv.vmp,
                    "Imp (A)": iv.imp,
                    "Pmax (W)": iv.pmax,
                    "FF": iv.fill_factor,
                    "Temp (°C)": iv.temperature,
                    "Irradiance (W/m²)": iv.irradiance,
                })
            module_df = pd.DataFrame(module_data)
            module_df.to_excel(writer, sheet_name="Module Measurements", index=False)

            # Loss breakdown
            losses = self.test_result.loss_components
            loss_data = {
                "Loss Category": [
                    "Optical Reflection",
                    "Optical Absorption",
                    "Optical Shading",
                    "Total Optical",
                    "Electrical Series R",
                    "Electrical Mismatch",
                    "Total Electrical",
                    "Thermal Assembly",
                    "Spatial Non-uniformity",
                    "Spectral Mismatch",
                    "TOTAL LOSS",
                ],
                "Loss (%)": [
                    losses.optical_reflection,
                    losses.optical_absorption,
                    losses.optical_shading,
                    losses.total_optical_loss,
                    losses.electrical_series_resistance,
                    losses.electrical_mismatch,
                    losses.total_electrical_loss,
                    losses.thermal_assembly,
                    losses.spatial_non_uniformity,
                    losses.spectral_mismatch,
                    losses.total_loss,
                ]
            }
            loss_df = pd.DataFrame(loss_data)
            loss_df.to_excel(writer, sheet_name="Loss Breakdown", index=False)

        logger.info(f"Report exported to Excel: {file_path}")

    def export_to_pdf(self, file_path: str) -> None:
        """
        Export CTM test report to PDF format.

        Args:
            file_path: Output file path (.pdf)

        Example:
            >>> report.export_to_pdf("ctm_test_report.pdf")
        """
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
        )
        story.append(Paragraph(
            f"IEC 63202 CTM Test Report",
            title_style
        ))

        # Test Information
        summary = self.generate_summary()
        story.append(Paragraph("Test Information", styles['Heading2']))
        test_info = [
            ["Test ID:", summary['test_id']],
            ["Date:", summary['test_date'].strftime("%Y-%m-%d %H:%M")],
            ["Laboratory:", summary['laboratory']],
            ["Operator:", summary['operator']],
            ["Cell Technology:", summary['cell_technology']],
        ]
        t = Table(test_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))

        # CTM Results
        story.append(Paragraph("CTM Test Results", styles['Heading2']))
        ctm_results = [
            ["Metric", "Value"],
            ["Cell Power (avg)", f"{summary['cell_power_avg']:.3f} W"],
            ["Module Power (avg)", f"{summary['module_power_avg']:.3f} W"],
            ["Expected Module Power", f"{summary['expected_module_power']:.3f} W"],
            ["CTM Ratio", f"{summary['ctm_ratio']:.2f}%"],
            ["Uncertainty", f"±{summary['ctm_uncertainty']:.2f}%"],
            ["Total Loss", f"{summary['total_loss']:.2f}%"],
            ["Compliance Status", summary['compliance_status']],
        ]
        t = Table(ctm_results, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, -1), (-1, -1),
             colors.green if summary['compliance_status'] == 'PASS' else colors.red),
        ]))
        story.append(t)

        # Build PDF
        doc.build(story)

        logger.info(f"Report exported to PDF: {file_path}")

    def generate_html_report(self) -> str:
        """
        Generate HTML report with interactive plots.

        Returns:
            HTML string containing complete report

        Example:
            >>> html = report.generate_html_report()
            >>> with open("ctm_report.html", "w") as f:
            ...     f.write(html)
        """
        summary = self.generate_summary()

        html_parts = []
        html_parts.append("<html><head>")
        html_parts.append("<title>IEC 63202 CTM Test Report</title>")
        html_parts.append("<style>")
        html_parts.append("""
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #1f77b4; }
            h2 { color: #2ca02c; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #1f77b4; color: white; }
            .pass { background-color: #90EE90; }
            .fail { background-color: #FFB6C1; }
        """)
        html_parts.append("</style></head><body>")

        # Title and summary
        html_parts.append(f"<h1>IEC 63202 CTM Test Report</h1>")
        html_parts.append(f"<p><strong>Test ID:</strong> {summary['test_id']}</p>")
        html_parts.append(f"<p><strong>Date:</strong> {summary['test_date']}</p>")
        html_parts.append(f"<p><strong>Laboratory:</strong> {summary['laboratory']}</p>")

        # Results table
        html_parts.append("<h2>Test Results</h2>")
        status_class = "pass" if summary['compliance_status'] == 'PASS' else "fail"
        html_parts.append(f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Cell Power (avg)</td><td>{summary['cell_power_avg']:.3f} W</td></tr>
                <tr><td>Module Power (avg)</td><td>{summary['module_power_avg']:.3f} W</td></tr>
                <tr><td>CTM Ratio</td><td>{summary['ctm_ratio']:.2f}%</td></tr>
                <tr><td>Uncertainty</td><td>±{summary['ctm_uncertainty']:.2f}%</td></tr>
                <tr><td>Total Loss</td><td>{summary['total_loss']:.2f}%</td></tr>
                <tr class="{status_class}"><td>Compliance</td><td>{summary['compliance_status']}</td></tr>
            </table>
        """)

        # Add plots
        html_parts.append("<h2>IV Curve Comparison</h2>")
        iv_fig = self.create_iv_curve_comparison()
        html_parts.append(iv_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        html_parts.append("<h2>Loss Breakdown</h2>")
        loss_fig = self.create_loss_waterfall_chart()
        html_parts.append(loss_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        html_parts.append("</body></html>")

        html_report = "\n".join(html_parts)

        logger.info("HTML report generated")

        return html_report
