"""Documentation generator for PV systems.

This module provides comprehensive documentation generation capabilities
including PDF reports, CAD drawings, calculation spreadsheets, O&M manuals,
and commissioning checklists.
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models.validation_models import (
    ComplianceResult,
    DocumentPackage,
    EngineeringCalculation,
    PerformanceMetrics,
    SystemConfiguration,
    ValidationReport,
)


class DocumentationGenerator:
    """Comprehensive documentation generator for PV systems.

    Generates professional engineering documentation packages including
    PDF reports, stamped drawings, specification sheets, O&M manuals,
    commissioning checklists, and CAD drawings.

    Attributes:
        config: System configuration
        validation_report: Validation report to document
        output_dir: Output directory for generated documents
        include_pe_stamp: Whether to include PE stamp placeholder
    """

    def __init__(
        self,
        config: SystemConfiguration,
        validation_report: Optional[ValidationReport] = None,
        output_dir: str = "./exports",
        include_pe_stamp: bool = False,
    ) -> None:
        """Initialize documentation generator.

        Args:
            config: System configuration
            validation_report: Validation report to document
            output_dir: Output directory for generated documents
            include_pe_stamp: Whether to include PE stamp placeholder
        """
        self.config = config
        self.validation_report = validation_report
        self.output_dir = Path(output_dir)
        self.include_pe_stamp = include_pe_stamp

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self) -> None:
        """Create custom paragraph styles for documents."""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=30,
                alignment=1,  # Center
            )
        )

        # Heading1 style
        self.styles.add(
            ParagraphStyle(
                name="CustomHeading1",
                parent=self.styles["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#1f4788"),
                spaceBefore=12,
                spaceAfter=6,
            )
        )

        # Heading2 style
        self.styles.add(
            ParagraphStyle(
                name="CustomHeading2",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#2e5c8a"),
                spaceBefore=10,
                spaceAfter=4,
            )
        )

    def generate_engineering_package(
        self,
        filename: Optional[str] = None
    ) -> str:
        """Generate complete engineering package PDF.

        Creates comprehensive engineering documentation including system
        overview, calculations, compliance checks, and recommendations.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"engineering_package_{self.config.system_name}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build document content
        story = []

        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())

        # System overview
        story.extend(self._create_system_overview())
        story.append(PageBreak())

        # Compliance summary
        if self.validation_report:
            story.extend(self._create_compliance_section())
            story.append(PageBreak())

        # Engineering calculations
        if self.validation_report:
            story.extend(self._create_calculations_section())
            story.append(PageBreak())

        # Performance analysis
        if self.validation_report and self.validation_report.performance_metrics:
            story.extend(self._create_performance_section())
            story.append(PageBreak())

        # Recommendations
        if self.validation_report:
            story.extend(self._create_recommendations_section())

        # Build PDF
        doc.build(story)

        return str(filepath)

    def _create_title_page(self) -> List[Any]:
        """Create title page for engineering package."""
        content = []

        # Title
        content.append(Spacer(1, 2 * inch))
        content.append(Paragraph("PV System Engineering Package", self.styles["CustomTitle"]))
        content.append(Spacer(1, 0.5 * inch))

        # System name
        content.append(
            Paragraph(
                f"<b>{self.config.system_name}</b>",
                self.styles["CustomHeading1"]
            )
        )
        content.append(Spacer(1, 0.3 * inch))

        # System details table
        details = [
            ["System Type:", self.config.system_type.value.title()],
            ["Location:", self.config.location],
            ["Capacity:", f"{self.config.capacity_kw:.2f} kW"],
            ["Module Count:", str(self.config.module_count)],
            ["Design Date:", self.config.design_date.strftime("%Y-%m-%d")],
        ]

        if self.config.designer:
            details.append(["Designer:", self.config.designer])

        table = Table(details, colWidths=[2 * inch, 4 * inch])
        table.setStyle(
            TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 12),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 12),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1f4788")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e6f2ff")),
            ])
        )

        content.append(table)
        content.append(Spacer(1, 1 * inch))

        # PE stamp placeholder
        if self.include_pe_stamp:
            content.append(
                Paragraph(
                    "<i>[Professional Engineer Stamp]</i>",
                    self.styles["Normal"]
                )
            )

        # Disclaimer
        content.append(Spacer(1, 0.5 * inch))
        disclaimer = (
            "<i>This document contains engineering calculations and analysis "
            "for the photovoltaic system described herein. Design must be reviewed "
            "and approved by the Authority Having Jurisdiction (AHJ) prior to installation.</i>"
        )
        content.append(Paragraph(disclaimer, self.styles["Normal"]))

        return content

    def _create_system_overview(self) -> List[Any]:
        """Create system overview section."""
        content = []

        content.append(Paragraph("1. System Overview", self.styles["CustomHeading1"]))
        content.append(Spacer(1, 0.2 * inch))

        # System configuration
        content.append(Paragraph("1.1 System Configuration", self.styles["CustomHeading2"]))

        config_data = [
            ["Parameter", "Value", "Unit"],
            ["System Type", self.config.system_type.value.title(), ""],
            ["Total Capacity", f"{self.config.capacity_kw:.2f}", "kW"],
            ["Module Count", str(self.config.module_count), "units"],
            ["String Count", str(self.config.string_count), "strings"],
            ["Modules per String", str(self.config.modules_per_string), "modules"],
            ["Inverter Count", str(self.config.inverter_count), "units"],
        ]

        config_table = Table(config_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        config_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ])
        )

        content.append(config_table)
        content.append(Spacer(1, 0.3 * inch))

        # Electrical parameters
        content.append(Paragraph("1.2 Electrical Parameters", self.styles["CustomHeading2"]))

        electrical_data = [
            ["Parameter", "Value", "Unit"],
            ["System DC Voltage", f"{self.config.system_voltage_vdc:.1f}", "V"],
            ["Max Voc (Cold)", f"{self.config.max_voltage_voc:.1f}", "V"],
            ["Operating Vmp", f"{self.config.operating_voltage_vmp:.1f}", "V"],
            ["Max Isc", f"{self.config.max_current_isc:.1f}", "A"],
            ["Operating Imp", f"{self.config.operating_current_imp:.1f}", "A"],
        ]

        electrical_table = Table(electrical_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        electrical_table.setStyle(config_table._cellstyles)

        content.append(electrical_table)
        content.append(Spacer(1, 0.3 * inch))

        # Environmental conditions
        content.append(Paragraph("1.3 Environmental Design Conditions", self.styles["CustomHeading2"]))

        env_data = [
            ["Parameter", "Value", "Unit"],
            ["Minimum Temperature", f"{self.config.ambient_temp_min:.1f}", "°C"],
            ["Maximum Temperature", f"{self.config.ambient_temp_max:.1f}", "°C"],
            ["Maximum Wind Speed", f"{self.config.wind_speed_max:.1f}", "m/s"],
        ]

        if self.config.snow_load:
            env_data.append(["Snow Load", f"{self.config.snow_load:.1f}", "kg/m²"])

        env_table = Table(env_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        env_table.setStyle(config_table._cellstyles)

        content.append(env_table)

        return content

    def _create_compliance_section(self) -> List[Any]:
        """Create code compliance section."""
        content = []

        content.append(Paragraph("2. Code Compliance Summary", self.styles["CustomHeading1"]))
        content.append(Spacer(1, 0.2 * inch))

        if not self.validation_report:
            content.append(Paragraph("No validation report available.", self.styles["Normal"]))
            return content

        # Compliance statistics
        total_checks = len(self.validation_report.code_compliance)
        passed = sum(1 for c in self.validation_report.code_compliance if c.status.value == "passed")
        failed = sum(1 for c in self.validation_report.code_compliance if c.status.value == "failed")
        warnings = sum(1 for c in self.validation_report.code_compliance if c.status.value == "warning")

        stats_text = (
            f"<b>Total Checks:</b> {total_checks} | "
            f"<b>Passed:</b> {passed} | "
            f"<b>Failed:</b> {failed} | "
            f"<b>Warnings:</b> {warnings}"
        )
        content.append(Paragraph(stats_text, self.styles["Normal"]))
        content.append(Spacer(1, 0.2 * inch))

        # Compliance results table
        compliance_data = [["Code", "Section", "Requirement", "Status"]]

        for result in self.validation_report.code_compliance[:20]:  # Limit to 20 for brevity
            status_color = {
                "passed": "✓",
                "failed": "✗",
                "warning": "⚠",
            }.get(result.status.value, "?")

            compliance_data.append([
                result.code_name,
                result.section,
                result.requirement[:60] + "..." if len(result.requirement) > 60 else result.requirement,
                status_color,
            ])

        compliance_table = Table(
            compliance_data,
            colWidths=[1.5 * inch, 1 * inch, 3 * inch, 0.5 * inch]
        )
        compliance_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ])
        )

        content.append(compliance_table)

        return content

    def _create_calculations_section(self) -> List[Any]:
        """Create engineering calculations section."""
        content = []

        content.append(Paragraph("3. Engineering Calculations", self.styles["CustomHeading1"]))
        content.append(Spacer(1, 0.2 * inch))

        if not self.validation_report:
            content.append(Paragraph("No validation report available.", self.styles["Normal"]))
            return content

        # Collect all calculations from validation results
        all_calculations: List[EngineeringCalculation] = []
        for validation_list in [
            self.validation_report.electrical_validation,
            self.validation_report.structural_validation,
        ]:
            for result in validation_list:
                all_calculations.extend(result.calculations)

        if not all_calculations:
            content.append(Paragraph("No calculations available.", self.styles["Normal"]))
            return content

        # Create calculations table
        calc_data = [["Calculation", "Result", "Unit", "Status"]]

        for calc in all_calculations[:15]:  # Limit to 15 for brevity
            status_mark = "✓" if calc.is_valid else "✗"
            calc_data.append([
                calc.description[:50] + "..." if len(calc.description) > 50 else calc.description,
                f"{calc.calculated_value:.2f}" if isinstance(calc.calculated_value, (int, float)) else str(calc.calculated_value),
                calc.unit,
                status_mark,
            ])

        calc_table = Table(calc_data, colWidths=[3 * inch, 1.5 * inch, 1 * inch, 0.5 * inch])
        calc_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ])
        )

        content.append(calc_table)

        return content

    def _create_performance_section(self) -> List[Any]:
        """Create performance analysis section."""
        content = []

        content.append(Paragraph("4. Performance Analysis", self.styles["CustomHeading1"]))
        content.append(Spacer(1, 0.2 * inch))

        if not self.validation_report or not self.validation_report.performance_metrics:
            content.append(Paragraph("No performance data available.", self.styles["Normal"]))
            return content

        metrics = self.validation_report.performance_metrics

        # Key performance indicators
        kpi_data = [
            ["Metric", "Value", "Unit"],
            ["Annual Energy Yield", f"{metrics.annual_energy_yield_kwh:,.0f}", "kWh"],
            ["Specific Yield", f"{metrics.specific_yield_kwh_kwp:.1f}", "kWh/kWp"],
            ["Performance Ratio", f"{metrics.performance_ratio:.2%}", "%"],
            ["Capacity Factor", f"{metrics.capacity_factor:.2%}", "%"],
            ["Total System Losses", f"{metrics.total_losses:.1f}", "%"],
        ]

        kpi_table = Table(kpi_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        kpi_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ])
        )

        content.append(kpi_table)
        content.append(Spacer(1, 0.3 * inch))

        # Loss breakdown
        content.append(Paragraph("4.1 Loss Budget Breakdown", self.styles["CustomHeading2"]))

        loss_data = [
            ["Loss Category", "Value (%)", ""],
            ["Temperature Losses", f"{metrics.loss_temperature:.2f}", "%"],
            ["Soiling Losses", f"{metrics.loss_soiling:.2f}", "%"],
            ["Shading Losses", f"{metrics.loss_shading:.2f}", "%"],
            ["Mismatch Losses", f"{metrics.loss_mismatch:.2f}", "%"],
            ["Wiring Losses", f"{metrics.loss_wiring:.2f}", "%"],
            ["Inverter Losses", f"{metrics.loss_inverter:.2f}", "%"],
            ["Degradation", f"{metrics.loss_degradation:.2f}", "%"],
        ]

        loss_table = Table(loss_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        loss_table.setStyle(kpi_table._cellstyles)

        content.append(loss_table)

        return content

    def _create_recommendations_section(self) -> List[Any]:
        """Create recommendations section."""
        content = []

        content.append(Paragraph("5. Recommendations", self.styles["CustomHeading1"]))
        content.append(Spacer(1, 0.2 * inch))

        if not self.validation_report or not self.validation_report.recommendations:
            content.append(Paragraph("No specific recommendations at this time.", self.styles["Normal"]))
            return content

        for i, recommendation in enumerate(self.validation_report.recommendations, 1):
            content.append(Paragraph(f"{i}. {recommendation}", self.styles["Normal"]))
            content.append(Spacer(1, 0.1 * inch))

        return content

    def create_stamped_drawings(self, filename: Optional[str] = None) -> str:
        """Create stamped engineering drawings (placeholder implementation).

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stamped_drawings_{self.config.system_name}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        # Create a simple placeholder drawing
        with PdfPages(str(filepath)) as pdf:
            fig, ax = plt.subplots(figsize=(11, 8.5))

            # Title
            ax.text(
                0.5, 0.9,
                f"PV System Drawings - {self.config.system_name}",
                ha="center",
                fontsize=20,
                fontweight="bold"
            )

            # System diagram placeholder
            ax.text(
                0.5, 0.5,
                "[Single Line Diagram]\n[Site Plan]\n[Array Layout]\n[Electrical Details]",
                ha="center",
                va="center",
                fontsize=14
            )

            # PE stamp placeholder
            if self.include_pe_stamp:
                ax.text(
                    0.85, 0.1,
                    "[PE Stamp]",
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                )

            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        return str(filepath)

    def produce_specification_sheets(self, filename: Optional[str] = None) -> str:
        """Produce equipment specification sheets.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"specification_sheets_{self.config.system_name}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        story.append(Paragraph("Equipment Specification Sheets", self.styles["CustomTitle"]))
        story.append(Spacer(1, 0.3 * inch))

        # Module specifications
        story.append(Paragraph("PV Module Specifications", self.styles["CustomHeading1"]))
        story.append(Spacer(1, 0.1 * inch))

        module_specs = [
            ["Parameter", "Value"],
            ["Total Module Count", str(self.config.module_count)],
            ["Rated Power (STC)", "TBD kW"],
            ["Module Efficiency", "TBD %"],
            ["Temperature Coefficient", "-0.35 %/°C (typical)"],
            ["Dimensions", "TBD mm"],
            ["Weight", "TBD kg"],
        ]

        spec_table = Table(module_specs, colWidths=[3 * inch, 3 * inch])
        spec_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ])
        )

        story.append(spec_table)
        story.append(Spacer(1, 0.3 * inch))

        # Inverter specifications
        story.append(Paragraph("Inverter Specifications", self.styles["CustomHeading1"]))
        story.append(Spacer(1, 0.1 * inch))

        inverter_specs = [
            ["Parameter", "Value"],
            ["Total Inverter Count", str(self.config.inverter_count)],
            ["Rated AC Power", "TBD kW"],
            ["Max Efficiency", "TBD %"],
            ["MPPT Voltage Range", "TBD V"],
            ["Max DC Voltage", "TBD V"],
            ["Max DC Current", "TBD A"],
        ]

        inv_table = Table(inverter_specs, colWidths=[3 * inch, 3 * inch])
        inv_table.setStyle(spec_table._cellstyles)

        story.append(inv_table)

        doc.build(story)
        return str(filepath)

    def generate_O_and_M_manual(self, filename: Optional[str] = None) -> str:
        """Generate Operations & Maintenance manual.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"om_manual_{self.config.system_name}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        story.append(Paragraph("Operations & Maintenance Manual", self.styles["CustomTitle"]))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph(f"System: {self.config.system_name}", self.styles["CustomHeading1"]))
        story.append(Spacer(1, 0.3 * inch))

        # Safety section
        story.append(Paragraph("1. Safety Information", self.styles["CustomHeading1"]))
        safety_items = [
            "High voltage present - only qualified personnel should service equipment",
            "Follow all lockout/tagout procedures before maintenance",
            "Wear appropriate PPE including arc flash protection",
            "Verify system is de-energized before servicing",
            "Follow manufacturer safety guidelines for all equipment",
        ]

        for item in safety_items:
            story.append(Paragraph(f"• {item}", self.styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

        story.append(Spacer(1, 0.2 * inch))

        # Maintenance schedule
        story.append(Paragraph("2. Maintenance Schedule", self.styles["CustomHeading1"]))
        story.append(Spacer(1, 0.1 * inch))

        maintenance_schedule = [
            ["Task", "Frequency", "Description"],
            ["Visual Inspection", "Monthly", "Check for physical damage, debris, shading"],
            ["Module Cleaning", "Quarterly", "Clean modules if soiling detected"],
            ["Electrical Testing", "Annual", "IR testing, IV curve trace, string testing"],
            ["Inverter Inspection", "Annual", "Check filters, cooling, connections"],
            ["Torque Check", "Biennial", "Verify all electrical connections"],
        ]

        maint_table = Table(maintenance_schedule, colWidths=[1.5 * inch, 1.5 * inch, 3.5 * inch])
        maint_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ])
        )

        story.append(maint_table)
        story.append(Spacer(1, 0.3 * inch))

        # Troubleshooting
        story.append(Paragraph("3. Troubleshooting Guide", self.styles["CustomHeading1"]))
        story.append(Spacer(1, 0.1 * inch))

        troubleshooting = [
            ["Issue", "Possible Cause", "Action"],
            ["Low energy production", "Soiling, shading, equipment failure", "Inspect modules, check inverter"],
            ["Inverter fault", "Grid disturbance, DC overvoltage", "Check error codes, reset if safe"],
            ["Communication loss", "Network issue, gateway fault", "Check network connections"],
        ]

        trouble_table = Table(troubleshooting, colWidths=[2 * inch, 2.5 * inch, 2 * inch])
        trouble_table.setStyle(maint_table._cellstyles)

        story.append(trouble_table)

        doc.build(story)
        return str(filepath)

    def create_commissioning_checklist(self, filename: Optional[str] = None) -> str:
        """Create commissioning checklist.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"commissioning_checklist_{self.config.system_name}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        story.append(Paragraph("Commissioning Checklist", self.styles["CustomTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph(f"System: {self.config.system_name}", self.styles["CustomHeading1"]))
        story.append(Paragraph(f"Date: _____________", self.styles["Normal"]))
        story.append(Paragraph(f"Technician: _____________", self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Pre-commissioning checks
        story.append(Paragraph("1. Pre-Commissioning Checks", self.styles["CustomHeading1"]))

        pre_comm_items = [
            ["☐", "Verify all modules are installed per design"],
            ["☐", "Check all DC wiring connections are secure"],
            ["☐", "Verify proper polarity on all strings"],
            ["☐", "Check all AC connections"],
            ["☐", "Verify grounding system is complete"],
            ["☐", "Inspect racking for proper installation"],
            ["☐", "Confirm inverter installation per manufacturer specs"],
        ]

        for item in pre_comm_items:
            story.append(Paragraph(f"{item[0]} {item[1]}", self.styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

        story.append(Spacer(1, 0.2 * inch))

        # Electrical testing
        story.append(Paragraph("2. Electrical Testing", self.styles["CustomHeading1"]))

        test_table_data = [
            ["Test", "Expected", "Actual", "Pass/Fail"],
            ["String Voc", f"{self.config.max_voltage_voc:.0f}V ±5%", "_______", "☐"],
            ["String Isc", f"{self.config.max_current_isc:.0f}A ±5%", "_______", "☐"],
            ["Insulation Resistance", "> 1 MΩ", "_______", "☐"],
            ["Ground Continuity", "< 1 Ω", "_______", "☐"],
            ["Inverter Startup", "Normal", "_______", "☐"],
        ]

        test_table = Table(test_table_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1 * inch])
        test_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ])
        )

        story.append(test_table)
        story.append(Spacer(1, 0.3 * inch))

        # Final verification
        story.append(Paragraph("3. Final Verification", self.styles["CustomHeading1"]))

        final_items = [
            ["☐", "System producing power"],
            ["☐", "Monitoring system operational"],
            ["☐", "All labels installed"],
            ["☐", "As-built drawings provided"],
            ["☐", "O&M manual provided to owner"],
            ["☐", "Owner training completed"],
        ]

        for item in final_items:
            story.append(Paragraph(f"{item[0]} {item[1]}", self.styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

        story.append(Spacer(1, 0.5 * inch))

        # Signature
        story.append(Paragraph("Commissioned by: _______________________  Date: __________", self.styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Approved by: _______________________  Date: __________", self.styles["Normal"]))

        doc.build(story)
        return str(filepath)

    def export_cad_drawing(self, filename: Optional[str] = None) -> str:
        """Export simple CAD drawing in DXF format.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated DXF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cad_drawing_{self.config.system_name}_{timestamp}.dxf"

        filepath = self.output_dir / filename

        # Create new DXF document
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Add title block
        msp.add_text(
            f"PV System: {self.config.system_name}",
            dxfattribs={"height": 0.5}
        ).set_pos((0, 10))

        msp.add_text(
            f"Capacity: {self.config.capacity_kw} kW",
            dxfattribs={"height": 0.25}
        ).set_pos((0, 9))

        # Simple array representation (placeholder)
        # Draw rectangles representing module arrays
        rows = int(np.sqrt(self.config.module_count))
        cols = int(np.ceil(self.config.module_count / rows))

        module_width = 1.0
        module_height = 2.0
        spacing = 0.1

        for row in range(rows):
            for col in range(cols):
                if row * cols + col >= self.config.module_count:
                    break

                x = col * (module_width + spacing)
                y = row * (module_height + spacing)

                # Draw rectangle for module
                points = [
                    (x, y),
                    (x + module_width, y),
                    (x + module_width, y + module_height),
                    (x, y + module_height),
                    (x, y),
                ]
                msp.add_lwpolyline(points)

        # Save DXF
        doc.saveas(str(filepath))
        return str(filepath)

    def create_calculations_spreadsheet(self, filename: Optional[str] = None) -> str:
        """Create calculations spreadsheet with all engineering calculations.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to generated Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calculations_{self.config.system_name}_{timestamp}.xlsx"

        filepath = self.output_dir / filename

        with pd.ExcelWriter(str(filepath), engine="xlsxwriter") as writer:
            # System configuration sheet
            config_data = {
                "Parameter": [
                    "System Name",
                    "System Type",
                    "Location",
                    "Capacity (kW)",
                    "Module Count",
                    "String Count",
                    "Modules per String",
                    "System Voltage (V)",
                    "Max Voc (V)",
                    "Operating Vmp (V)",
                    "Max Isc (A)",
                    "Operating Imp (A)",
                ],
                "Value": [
                    self.config.system_name,
                    self.config.system_type.value,
                    self.config.location,
                    self.config.capacity_kw,
                    self.config.module_count,
                    self.config.string_count,
                    self.config.modules_per_string,
                    self.config.system_voltage_vdc,
                    self.config.max_voltage_voc,
                    self.config.operating_voltage_vmp,
                    self.config.max_current_isc,
                    self.config.operating_current_imp,
                ],
            }

            df_config = pd.DataFrame(config_data)
            df_config.to_excel(writer, sheet_name="System Configuration", index=False)

            # Calculations sheet (if validation report available)
            if self.validation_report:
                all_calcs = []
                for validation_list in [
                    self.validation_report.electrical_validation,
                    self.validation_report.structural_validation,
                ]:
                    for result in validation_list:
                        for calc in result.calculations:
                            all_calcs.append({
                                "Type": calc.calculation_type,
                                "Description": calc.description,
                                "Value": calc.calculated_value,
                                "Unit": calc.unit,
                                "Valid": calc.is_valid,
                                "Formula": calc.formula or "",
                                "Reference": calc.reference or "",
                            })

                if all_calcs:
                    df_calcs = pd.DataFrame(all_calcs)
                    df_calcs.to_excel(writer, sheet_name="Calculations", index=False)

            # Compliance sheet (if validation report available)
            if self.validation_report:
                compliance_data = []
                for result in self.validation_report.code_compliance:
                    compliance_data.append({
                        "Code": result.code_name,
                        "Section": result.section,
                        "Requirement": result.requirement,
                        "Status": result.status.value,
                        "Checked Value": str(result.checked_value) if result.checked_value else "",
                        "Required Value": str(result.required_value) if result.required_value else "",
                        "Notes": result.notes or "",
                    })

                if compliance_data:
                    df_compliance = pd.DataFrame(compliance_data)
                    df_compliance.to_excel(writer, sheet_name="Code Compliance", index=False)

        return str(filepath)

    def generate_complete_package(self, package_id: Optional[str] = None) -> DocumentPackage:
        """Generate complete documentation package.

        Creates all documentation types and returns a DocumentPackage
        with paths to all generated files.

        Args:
            package_id: Unique package identifier (auto-generated if not provided)

        Returns:
            DocumentPackage with paths to all generated documents
        """
        if package_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_id = f"PKG-{timestamp}"

        # Create package subdirectory
        package_dir = self.output_dir / package_id
        package_dir.mkdir(parents=True, exist_ok=True)

        # Store original output dir and temporarily change it
        original_output_dir = self.output_dir
        self.output_dir = package_dir

        # Generate all documents
        engineering_pdf = self.generate_engineering_package("engineering_package.pdf")
        stamped_pdf = self.create_stamped_drawings("stamped_drawings.pdf")
        spec_pdf = self.produce_specification_sheets("specification_sheets.pdf")
        om_pdf = self.generate_O_and_M_manual("om_manual.pdf")
        comm_pdf = self.create_commissioning_checklist("commissioning_checklist.pdf")
        calc_xlsx = self.create_calculations_spreadsheet("calculations.xlsx")
        cad_dxf = self.export_cad_drawing("array_layout.dxf")

        # Restore original output dir
        self.output_dir = original_output_dir

        # Calculate total package size
        total_size_bytes = sum(
            os.path.getsize(f) for f in [
                engineering_pdf, stamped_pdf, spec_pdf, om_pdf,
                comm_pdf, calc_xlsx, cad_dxf
            ] if os.path.exists(f)
        )
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Create DocumentPackage
        package = DocumentPackage(
            package_id=package_id,
            system_name=self.config.system_name,
            engineering_package_pdf=engineering_pdf,
            stamped_drawings_pdf=stamped_pdf,
            specification_sheets_pdf=spec_pdf,
            om_manual_pdf=om_pdf,
            commissioning_checklist_pdf=comm_pdf,
            calculations_spreadsheet=calc_xlsx,
            cad_drawings=[cad_dxf],
            document_count=7,
            total_size_mb=round(total_size_mb, 2),
            includes_pe_stamp=self.include_pe_stamp,
            includes_calculations=True,
        )

        return package
