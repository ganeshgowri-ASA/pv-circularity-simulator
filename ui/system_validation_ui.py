"""Streamlit UI for PV System Validation & Engineering Reports.

This module provides a comprehensive Streamlit dashboard for system validation,
compliance checking, performance analysis, and documentation generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.b05_system_validation.documentation_generator import DocumentationGenerator
from src.b05_system_validation.system_validator import SystemValidator
from src.models.validation_models import (
    ComplianceStatus,
    IssueSeverity,
    PerformanceMetrics,
    SystemConfiguration,
    SystemType,
)


class SystemValidationUI:
    """Streamlit UI for System Validation & Engineering Reports.

    Provides comprehensive dashboard interface for:
    - System configuration input
    - Automated validation checks
    - Compliance matrix display
    - Issue tracking
    - Report generation
    - Documentation export
    """

    def __init__(self) -> None:
        """Initialize SystemValidationUI."""
        st.set_page_config(
            page_title="PV System Validation & Engineering Reports",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize session state
        if "validator" not in st.session_state:
            st.session_state.validator = None
        if "validation_report" not in st.session_state:
            st.session_state.validation_report = None
        if "doc_generator" not in st.session_state:
            st.session_state.doc_generator = None

    def run(self) -> None:
        """Run the Streamlit application."""
        # Header
        st.title("⚡ PV System Validation & Engineering Reports")
        st.markdown("**BATCH4-B05-S06: System Validation & Engineering Reports Module**")
        st.markdown("---")

        # Sidebar navigation
        page = st.sidebar.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "⚙️ System Configuration",
                "✅ Validation Results",
                "📊 Performance Analysis",
                "📋 Compliance Matrix",
                "🔧 Issue Tracking",
                "📄 Report Generation",
            ]
        )

        # Route to appropriate page
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "⚙️ System Configuration":
            self.show_configuration_page()
        elif page == "✅ Validation Results":
            self.show_validation_results()
        elif page == "📊 Performance Analysis":
            self.show_performance_analysis()
        elif page == "📋 Compliance Matrix":
            self.show_compliance_matrix()
        elif page == "🔧 Issue Tracking":
            self.show_issue_tracking()
        elif page == "📄 Report Generation":
            self.show_report_generation()

    def show_dashboard(self) -> None:
        """Show main dashboard with overview."""
        st.header("System Validation Dashboard")

        if st.session_state.validation_report is None:
            st.info("👈 Configure your PV system in the sidebar to begin validation.")

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Systems Validated", "0")
            with col2:
                st.metric("Active Issues", "0")
            with col3:
                st.metric("Compliance Rate", "N/A")
            with col4:
                st.metric("Reports Generated", "0")

        else:
            report = st.session_state.validation_report

            # Status banner
            status_color = {
                "passed": "🟢",
                "warning": "🟡",
                "failed": "🔴",
            }.get(report.overall_status.value, "⚪")

            st.markdown(f"## {status_color} Overall Status: {report.overall_status.value.upper()}")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Issues", report.total_issues)
            with col2:
                st.metric("Critical Issues", report.critical_issues, delta=None if report.critical_issues == 0 else -report.critical_issues)
            with col3:
                compliance_rate = (
                    sum(1 for c in report.code_compliance if c.status == ComplianceStatus.PASSED) /
                    len(report.code_compliance) * 100
                    if report.code_compliance else 0
                )
                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
            with col4:
                st.metric("Validation Date", report.report_date.strftime("%Y-%m-%d"))

            # Issues by severity chart
            st.subheader("Issues by Severity")
            issues_data = {
                "Severity": ["Critical", "Errors", "Warnings", "Info"],
                "Count": [
                    report.critical_issues,
                    report.errors,
                    report.warnings,
                    report.total_issues - report.critical_issues - report.errors - report.warnings
                ],
                "Color": ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"]
            }

            fig = px.bar(
                issues_data,
                x="Severity",
                y="Count",
                color="Color",
                color_discrete_map="identity",
                title="Issue Distribution"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Compliance overview
            st.subheader("Compliance Overview")
            col1, col2 = st.columns(2)

            with col1:
                # Compliance by status
                compliance_counts = {
                    "Passed": sum(1 for c in report.code_compliance if c.status == ComplianceStatus.PASSED),
                    "Failed": sum(1 for c in report.code_compliance if c.status == ComplianceStatus.FAILED),
                    "Warning": sum(1 for c in report.code_compliance if c.status == ComplianceStatus.WARNING),
                }

                fig_compliance = go.Figure(data=[go.Pie(
                    labels=list(compliance_counts.keys()),
                    values=list(compliance_counts.values()),
                    marker_colors=["#4caf50", "#f44336", "#ff9800"]
                )])
                fig_compliance.update_layout(title="Compliance Status Distribution")
                st.plotly_chart(fig_compliance, use_container_width=True)

            with col2:
                # Top recommendations
                st.markdown("**Top Recommendations:**")
                for i, rec in enumerate(report.recommendations[:5], 1):
                    st.markdown(f"{i}. {rec}")

    def show_configuration_page(self) -> None:
        """Show system configuration page."""
        st.header("System Configuration")

        with st.form("system_config"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("System Information")
                system_name = st.text_input("System Name", value="Example PV System")
                system_type = st.selectbox(
                    "System Type",
                    options=[t.value for t in SystemType],
                    format_func=lambda x: x.replace("_", " ").title()
                )
                location = st.text_input("Location", value="San Francisco, CA")
                jurisdiction = st.text_input("Jurisdiction", value="San Francisco")

                st.subheader("System Sizing")
                capacity_kw = st.number_input("System Capacity (kW)", min_value=1.0, value=100.0)
                module_count = st.number_input("Module Count", min_value=1, value=250)
                inverter_count = st.number_input("Inverter Count", min_value=1, value=2)
                string_count = st.number_input("String Count", min_value=1, value=20)
                modules_per_string = st.number_input("Modules per String", min_value=1, value=12)

            with col2:
                st.subheader("Electrical Parameters")
                system_voltage_vdc = st.number_input("System DC Voltage (V)", min_value=100.0, value=600.0)
                max_voltage_voc = st.number_input("Max Voc (V)", min_value=100.0, value=800.0)
                operating_voltage_vmp = st.number_input("Operating Vmp (V)", min_value=100.0, value=650.0)
                max_current_isc = st.number_input("Max Isc (A)", min_value=1.0, value=10.0)
                operating_current_imp = st.number_input("Operating Imp (A)", min_value=1.0, value=9.5)

                st.subheader("Environmental Conditions")
                ambient_temp_min = st.number_input("Min Temperature (°C)", value=-10.0)
                ambient_temp_max = st.number_input("Max Temperature (°C)", value=45.0)
                wind_speed_max = st.number_input("Max Wind Speed (m/s)", min_value=0.0, value=40.0)
                snow_load = st.number_input("Snow Load (kg/m²)", min_value=0.0, value=50.0)

            # Performance metrics (optional)
            st.subheader("Performance Metrics (Optional)")
            include_performance = st.checkbox("Include Performance Analysis")

            perf_metrics = None
            if include_performance:
                col3, col4 = st.columns(2)
                with col3:
                    annual_energy = st.number_input("Annual Energy Yield (kWh)", min_value=0.0, value=150000.0)
                    specific_yield = st.number_input("Specific Yield (kWh/kWp)", min_value=0.0, value=1500.0)
                    performance_ratio = st.slider("Performance Ratio", 0.0, 1.0, 0.82)
                    capacity_factor = st.slider("Capacity Factor", 0.0, 1.0, 0.20)

                with col4:
                    loss_temp = st.number_input("Temperature Losses (%)", min_value=0.0, value=5.0)
                    loss_soiling = st.number_input("Soiling Losses (%)", min_value=0.0, value=2.0)
                    loss_shading = st.number_input("Shading Losses (%)", min_value=0.0, value=1.0)
                    loss_mismatch = st.number_input("Mismatch Losses (%)", min_value=0.0, value=2.0)
                    loss_wiring = st.number_input("Wiring Losses (%)", min_value=0.0, value=2.0)
                    loss_inverter = st.number_input("Inverter Losses (%)", min_value=0.0, value=3.0)
                    loss_degradation = st.number_input("Degradation (%)", min_value=0.0, value=0.5)

            submit_button = st.form_submit_button("🚀 Run Validation")

            if submit_button:
                # Create system configuration
                config = SystemConfiguration(
                    system_type=SystemType(system_type),
                    system_name=system_name,
                    location=location,
                    capacity_kw=capacity_kw,
                    module_count=module_count,
                    inverter_count=inverter_count,
                    string_count=string_count,
                    modules_per_string=modules_per_string,
                    system_voltage_vdc=system_voltage_vdc,
                    max_voltage_voc=max_voltage_voc,
                    operating_voltage_vmp=operating_voltage_vmp,
                    max_current_isc=max_current_isc,
                    operating_current_imp=operating_current_imp,
                    ambient_temp_min=ambient_temp_min,
                    ambient_temp_max=ambient_temp_max,
                    wind_speed_max=wind_speed_max,
                    snow_load=snow_load,
                    jurisdiction=jurisdiction,
                    applicable_codes=["NEC 2020", "IEC 60364", "IBC 2021", "IFC 2021"],
                )

                # Create performance metrics if included
                if include_performance:
                    total_losses = (
                        loss_temp + loss_soiling + loss_shading +
                        loss_mismatch + loss_wiring + loss_inverter + loss_degradation
                    )

                    perf_metrics = PerformanceMetrics(
                        annual_energy_yield_kwh=annual_energy,
                        specific_yield_kwh_kwp=specific_yield,
                        performance_ratio=performance_ratio,
                        capacity_factor=capacity_factor,
                        loss_temperature=loss_temp,
                        loss_soiling=loss_soiling,
                        loss_shading=loss_shading,
                        loss_mismatch=loss_mismatch,
                        loss_wiring=loss_wiring,
                        loss_inverter=loss_inverter,
                        loss_degradation=loss_degradation,
                        total_losses=total_losses,
                        is_energy_yield_realistic=True,
                        is_pr_in_range=True,
                        is_loss_budget_valid=True,
                    )

                # Run validation
                with st.spinner("Running comprehensive validation..."):
                    validator = SystemValidator(config, perf_metrics)
                    report = validator.validate_complete_design()

                    st.session_state.validator = validator
                    st.session_state.validation_report = report
                    st.session_state.doc_generator = DocumentationGenerator(
                        config,
                        report,
                        output_dir="./exports"
                    )

                st.success("✅ Validation completed successfully!")
                st.info("Navigate to 'Validation Results' to see detailed results.")

    def show_validation_results(self) -> None:
        """Show validation results page."""
        st.header("Validation Results")

        if st.session_state.validation_report is None:
            st.warning("No validation results available. Please configure and validate a system first.")
            return

        report = st.session_state.validation_report

        # Overall status
        status_emoji = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌",
        }.get(report.overall_status.value, "❓")

        st.markdown(f"## {status_emoji} Overall Status: {report.overall_status.value.upper()}")

        # Tabs for different validation categories
        tab1, tab2, tab3 = st.tabs(["⚡ Electrical", "🏗️ Structural", "📊 Performance"])

        with tab1:
            self._show_validation_category(report.electrical_validation, "Electrical Validation")

        with tab2:
            self._show_validation_category(report.structural_validation, "Structural Validation")

        with tab3:
            self._show_validation_category(report.performance_validation, "Performance Validation")

    def _show_validation_category(self, validation_results: List, category: str) -> None:
        """Show validation results for a category."""
        st.subheader(category)

        if not validation_results:
            st.info(f"No {category.lower()} checks performed.")
            return

        for result in validation_results:
            # Status indicator
            status_color = {
                "passed": "🟢",
                "warning": "🟡",
                "failed": "🔴",
                "not_applicable": "⚪",
            }.get(result.status.value, "❓")

            with st.expander(f"{status_color} {result.check_name} - {result.status.value.upper()}"):
                st.write(result.summary)

                if result.issues:
                    st.markdown("**Issues:**")
                    for issue in result.issues:
                        severity_emoji = {
                            "critical": "🔴",
                            "error": "🟠",
                            "warning": "🟡",
                            "info": "🔵",
                        }.get(issue.severity.value, "⚪")

                        st.markdown(f"{severity_emoji} **{issue.code}**: {issue.message}")
                        if issue.recommendation:
                            st.markdown(f"  *Recommendation:* {issue.recommendation}")

                if result.calculations:
                    st.markdown("**Calculations:**")
                    calc_data = []
                    for calc in result.calculations:
                        calc_data.append({
                            "Type": calc.calculation_type,
                            "Description": calc.description,
                            "Value": f"{calc.calculated_value:.2f}" if isinstance(calc.calculated_value, (int, float)) else str(calc.calculated_value),
                            "Unit": calc.unit,
                            "Valid": "✅" if calc.is_valid else "❌",
                        })
                    st.dataframe(pd.DataFrame(calc_data), use_container_width=True)

    def show_performance_analysis(self) -> None:
        """Show performance analysis page."""
        st.header("Performance Analysis")

        if st.session_state.validation_report is None:
            st.warning("No validation results available. Please configure and validate a system first.")
            return

        report = st.session_state.validation_report

        if report.performance_metrics is None:
            st.info("No performance metrics provided for this system.")
            return

        metrics = report.performance_metrics

        # Key performance indicators
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Annual Energy Yield", f"{metrics.annual_energy_yield_kwh:,.0f} kWh")
        with col2:
            st.metric("Specific Yield", f"{metrics.specific_yield_kwh_kwp:.0f} kWh/kWp")
        with col3:
            st.metric("Performance Ratio", f"{metrics.performance_ratio:.2%}")
        with col4:
            st.metric("Capacity Factor", f"{metrics.capacity_factor:.2%}")

        # Loss breakdown
        st.subheader("Loss Budget Breakdown")

        loss_data = {
            "Category": [
                "Temperature",
                "Soiling",
                "Shading",
                "Mismatch",
                "Wiring",
                "Inverter",
                "Degradation"
            ],
            "Loss (%)": [
                metrics.loss_temperature,
                metrics.loss_soiling,
                metrics.loss_shading,
                metrics.loss_mismatch,
                metrics.loss_wiring,
                metrics.loss_inverter,
                metrics.loss_degradation
            ]
        }

        fig = px.bar(
            loss_data,
            x="Category",
            y="Loss (%)",
            title="System Loss Budget",
            color="Loss (%)",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Total losses
        st.metric("Total System Losses", f"{metrics.total_losses:.1f}%")

        # Performance validation flags
        st.subheader("Performance Validation Checks")
        col1, col2, col3 = st.columns(3)

        with col1:
            status = "✅" if metrics.is_energy_yield_realistic else "❌"
            st.markdown(f"{status} Energy Yield Realistic")

        with col2:
            status = "✅" if metrics.is_pr_in_range else "❌"
            st.markdown(f"{status} PR Within Range")

        with col3:
            status = "✅" if metrics.is_loss_budget_valid else "❌"
            st.markdown(f"{status} Loss Budget Valid")

    def show_compliance_matrix(self) -> None:
        """Show compliance matrix page."""
        st.header("Compliance Matrix")

        if st.session_state.validation_report is None:
            st.warning("No validation results available. Please configure and validate a system first.")
            return

        report = st.session_state.validation_report

        # Compliance summary
        total = len(report.code_compliance)
        passed = sum(1 for c in report.code_compliance if c.status == ComplianceStatus.PASSED)
        failed = sum(1 for c in report.code_compliance if c.status == ComplianceStatus.FAILED)
        warnings = sum(1 for c in report.code_compliance if c.status == ComplianceStatus.WARNING)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Checks", total)
        with col2:
            st.metric("Passed", passed)
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Warnings", warnings)

        # Compliance table
        st.subheader("Detailed Compliance Checks")

        # Create DataFrame
        compliance_data = []
        for check in report.code_compliance:
            status_symbol = {
                "passed": "✅",
                "failed": "❌",
                "warning": "⚠️",
                "not_applicable": "➖",
            }.get(check.status.value, "❓")

            compliance_data.append({
                "Status": status_symbol,
                "Code": check.code_name,
                "Section": check.section,
                "Requirement": check.requirement,
                "Checked Value": str(check.checked_value) if check.checked_value else "N/A",
                "Required Value": str(check.required_value) if check.required_value else "N/A",
            })

        df = pd.DataFrame(compliance_data)

        # Filter options
        filter_status = st.multiselect(
            "Filter by Status",
            options=["✅ Passed", "❌ Failed", "⚠️ Warning"],
            default=[]
        )

        if filter_status:
            status_map = {"✅ Passed": "✅", "❌ Failed": "❌", "⚠️ Warning": "⚠️"}
            filter_symbols = [status_map[s] for s in filter_status]
            df = df[df["Status"].isin(filter_symbols)]

        st.dataframe(df, use_container_width=True, height=500)

    def show_issue_tracking(self) -> None:
        """Show issue tracking page."""
        st.header("Issue Tracking")

        if st.session_state.validation_report is None:
            st.warning("No validation results available. Please configure and validate a system first.")
            return

        report = st.session_state.validation_report

        # Collect all issues
        all_issues = []
        for validation_list in [
            report.electrical_validation,
            report.structural_validation,
            report.performance_validation
        ]:
            for result in validation_list:
                all_issues.extend(result.issues)

        if not all_issues:
            st.success("🎉 No issues found! System passes all validation checks.")
            return

        # Issue summary
        st.subheader("Issue Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            critical = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)
            st.metric("Critical", critical)
        with col2:
            errors = sum(1 for i in all_issues if i.severity == IssueSeverity.ERROR)
            st.metric("Errors", errors)
        with col3:
            warnings = sum(1 for i in all_issues if i.severity == IssueSeverity.WARNING)
            st.metric("Warnings", warnings)
        with col4:
            info = sum(1 for i in all_issues if i.severity == IssueSeverity.INFO)
            st.metric("Info", info)

        # Filter by severity
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["Critical", "Error", "Warning", "Info"],
            default=["Critical", "Error"]
        )

        # Display issues
        for issue in all_issues:
            if issue.severity.value.title() not in severity_filter:
                continue

            severity_emoji = {
                "critical": "🔴",
                "error": "🟠",
                "warning": "🟡",
                "info": "🔵",
            }.get(issue.severity.value, "⚪")

            with st.expander(f"{severity_emoji} [{issue.severity.value.upper()}] {issue.code}: {issue.message}"):
                st.markdown(f"**Category:** {issue.category}")
                if issue.location:
                    st.markdown(f"**Location:** {issue.location}")
                if issue.recommendation:
                    st.markdown(f"**Recommendation:** {issue.recommendation}")
                if issue.reference:
                    st.markdown(f"**Reference:** {issue.reference}")
                st.markdown(f"**Detected:** {issue.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def show_report_generation(self) -> None:
        """Show report generation page."""
        st.header("Report Generation & Export")

        if st.session_state.validation_report is None:
            st.warning("No validation results available. Please configure and validate a system first.")
            return

        doc_gen = st.session_state.doc_generator

        st.markdown("Generate professional engineering documentation packages:")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Individual Documents")

            if st.button("📄 Generate Engineering Package (PDF)"):
                with st.spinner("Generating engineering package..."):
                    filepath = doc_gen.generate_engineering_package()
                st.success(f"✅ Generated: {filepath}")

            if st.button("📐 Generate Stamped Drawings (PDF)"):
                with st.spinner("Generating stamped drawings..."):
                    filepath = doc_gen.create_stamped_drawings()
                st.success(f"✅ Generated: {filepath}")

            if st.button("📋 Generate Specification Sheets (PDF)"):
                with st.spinner("Generating specification sheets..."):
                    filepath = doc_gen.produce_specification_sheets()
                st.success(f"✅ Generated: {filepath}")

            if st.button("📖 Generate O&M Manual (PDF)"):
                with st.spinner("Generating O&M manual..."):
                    filepath = doc_gen.generate_O_and_M_manual()
                st.success(f"✅ Generated: {filepath}")

            if st.button("✓ Generate Commissioning Checklist (PDF)"):
                with st.spinner("Generating commissioning checklist..."):
                    filepath = doc_gen.create_commissioning_checklist()
                st.success(f"✅ Generated: {filepath}")

        with col2:
            st.subheader("Technical Documents")

            if st.button("📊 Generate Calculations Spreadsheet (XLSX)"):
                with st.spinner("Generating calculations spreadsheet..."):
                    filepath = doc_gen.create_calculations_spreadsheet()
                st.success(f"✅ Generated: {filepath}")

            if st.button("🔧 Export CAD Drawing (DXF)"):
                with st.spinner("Exporting CAD drawing..."):
                    filepath = doc_gen.export_cad_drawing()
                st.success(f"✅ Generated: {filepath}")

            if st.button("💾 Export Validation Report (JSON)"):
                with st.spinner("Exporting validation report..."):
                    filepath = "./exports/validation_report.json"
                    st.session_state.validator.export_report_json(filepath)
                st.success(f"✅ Generated: {filepath}")

        st.markdown("---")

        # Complete package
        st.subheader("📦 Complete Documentation Package")
        st.markdown("Generate all documents in a single package:")

        if st.button("🚀 Generate Complete Package", type="primary"):
            with st.spinner("Generating complete documentation package..."):
                package = doc_gen.generate_complete_package()

            st.success("✅ Complete package generated successfully!")

            st.json({
                "package_id": package.package_id,
                "system_name": package.system_name,
                "document_count": package.document_count,
                "total_size_mb": package.total_size_mb,
                "includes_pe_stamp": package.includes_pe_stamp,
            })

            st.markdown(f"**Package Location:** `{Path(package.engineering_package_pdf).parent}`")


def main() -> None:
    """Main entry point for the Streamlit app."""
    app = SystemValidationUI()
    app.run()


if __name__ == "__main__":
    main()
