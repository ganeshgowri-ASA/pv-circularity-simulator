"""
IEC Testing Dashboard - Streamlit Application

Comprehensive interactive dashboard for IEC testing results, compliance reporting,
and certification management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.iec_testing.B04_S01_iec61215 import IEC61215Tester
from src.iec_testing.B04_S02_iec61730 import IEC61730Tester
from src.iec_testing.B04_S03_iec63202 import IEC63202Tester
from src.iec_testing.B04_S04_reporting_dashboard import (
    CertificationWorkflow,
    ComplianceVisualization,
    IECTestResultsManager,
    TestReportGenerator,
)
from src.iec_testing.models.test_models import (
    CertificationBodyType,
    ComplianceReport,
    IEC61215Result,
    IEC61730Result,
    IEC63202Result,
    IECStandard,
    TestStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IECTestingUI:
    """
    Main Streamlit UI component for IEC testing dashboard.

    Provides multi-tab interface for test results exploration, compliance reporting,
    and certification management.
    """

    def __init__(self) -> None:
        """Initialize IEC Testing UI."""
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="IEC Testing Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "test_results_manager" not in st.session_state:
            st.session_state.test_results_manager = IECTestResultsManager()

        if "visualization" not in st.session_state:
            st.session_state.visualization = ComplianceVisualization()

        if "report_generator" not in st.session_state:
            st.session_state.report_generator = TestReportGenerator()

        if "certification_workflow" not in st.session_state:
            st.session_state.certification_workflow = CertificationWorkflow()

        if "iec_61215_results" not in st.session_state:
            st.session_state.iec_61215_results = []

        if "iec_61730_results" not in st.session_state:
            st.session_state.iec_61730_results = []

        if "iec_63202_results" not in st.session_state:
            st.session_state.iec_63202_results = []

    def render_sidebar(self) -> Dict[str, Any]:
        """
        Render sidebar with controls and filters.

        Returns:
            Dict[str, Any]: User selections from sidebar
        """
        st.sidebar.title("⚡ IEC Testing Dashboard")
        st.sidebar.markdown("---")

        st.sidebar.header("Test Configuration")

        module_type = st.sidebar.text_input(
            "Module Type", value="PV-400W-PERC", help="Enter module type/model"
        )

        manufacturer = st.sidebar.text_input(
            "Manufacturer", value="Solar Innovations Inc.", help="Enter manufacturer name"
        )

        test_campaign_id = st.sidebar.text_input(
            "Test Campaign ID",
            value=f"TC-{datetime.now().strftime('%Y%m%d')}",
            help="Unique test campaign identifier",
        )

        st.sidebar.markdown("---")
        st.sidebar.header("Actions")

        run_tests = st.sidebar.button("🔬 Run IEC Tests", use_container_width=True)

        generate_reports = st.sidebar.button(
            "📄 Generate Reports", use_container_width=True
        )

        export_data = st.sidebar.button("💾 Export Data", use_container_width=True)

        return {
            "module_type": module_type,
            "manufacturer": manufacturer,
            "test_campaign_id": test_campaign_id,
            "run_tests": run_tests,
            "generate_reports": generate_reports,
            "export_data": export_data,
        }

    def run_all_tests(
        self, module_type: str, manufacturer: str, test_campaign_id: str
    ) -> Dict[str, Any]:
        """
        Run all IEC tests and store results.

        Args:
            module_type: Module type/model
            manufacturer: Manufacturer name
            test_campaign_id: Test campaign ID

        Returns:
            Dict[str, Any]: Test results
        """
        with st.spinner("Running IEC test suite... This may take a moment."):
            # Run IEC 61215
            tester_61215 = IEC61215Tester()
            result_61215 = tester_61215.run_full_qualification(
                module_id=f"{module_type}_001",
                module_type=module_type,
                manufacturer=manufacturer,
                test_campaign_id=test_campaign_id,
            )

            # Run IEC 61730
            tester_61730 = IEC61730Tester()
            result_61730 = tester_61730.run_full_safety_qualification(
                module_id=f"{module_type}_001",
                module_type=module_type,
                manufacturer=manufacturer,
                test_campaign_id=test_campaign_id,
            )

            # Run IEC 63202
            tester_63202 = IEC63202Tester()
            result_63202 = tester_63202.run_full_ctm_analysis(
                module_id=f"{module_type}_001",
                module_type=module_type,
                manufacturer=manufacturer,
                test_campaign_id=test_campaign_id,
            )

            # Store in session state
            st.session_state.iec_61215_results.append(result_61215)
            st.session_state.iec_61730_results.append(result_61730)
            st.session_state.iec_63202_results.append(result_63202)

            # Load into results manager
            st.session_state.test_results_manager.load_test_results(
                result_61215=result_61215,
                result_61730=result_61730,
                result_63202=result_63202,
            )

        return {
            "iec_61215": result_61215,
            "iec_61730": result_61730,
            "iec_63202": result_63202,
        }

    def render_iec61215_tab(self) -> None:
        """Render IEC 61215 results tab."""
        st.header("IEC 61215: Module Qualification Testing (MQT)")

        if not st.session_state.iec_61215_results:
            st.info("No IEC 61215 test results available. Run tests from the sidebar.")
            return

        # Select result to display
        result_index = st.selectbox(
            "Select Test Campaign",
            range(len(st.session_state.iec_61215_results)),
            format_func=lambda i: st.session_state.iec_61215_results[
                i
            ].test_campaign_id,
        )

        result: IEC61215Result = st.session_state.iec_61215_results[result_index]

        # Display overall status
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Status",
                result.overall_status.value.upper(),
                delta="PASS" if result.overall_status == TestStatus.PASSED else "FAIL",
            )

        with col2:
            st.metric("Compliance Rate", f"{result.compliance_percentage:.1f}%")

        with col3:
            st.metric(
                "Power Degradation",
                f"{result.test_sequence.power_degradation_percent:.2f}%",
                delta=f"{5.0 - result.test_sequence.power_degradation_percent:.2f}% margin",
            )

        with col4:
            st.metric(
                "Test Duration",
                f"{(result.test_end_date - result.test_start_date).days} days",
            )

        st.markdown("---")

        # Test results table
        st.subheader("Test Results Summary")

        seq = result.test_sequence
        test_data = {
            "Test Name": [
                "Visual Inspection (Initial)",
                "Performance at STC",
                "Wet Leakage Current",
                "Thermal Cycling",
                "Humidity-Freeze",
                "Damp Heat",
                "UV Preconditioning",
                "Mechanical Load",
                "Hail Impact",
                "Hot Spot Endurance",
                "Visual Inspection (Final)",
                "Performance at STC (Final)",
            ],
            "Status": [
                seq.visual_inspection_initial.status.value,
                seq.performance_at_stc.status.value,
                seq.wet_leakage_current.status.value,
                seq.thermal_cycling.status.value,
                seq.humidity_freeze.status.value,
                seq.damp_heat.status.value,
                seq.uv_preconditioning.status.value,
                seq.mechanical_load_test.status.value,
                seq.hail_impact.status.value,
                seq.hot_spot_endurance.status.value,
                seq.visual_inspection_final.status.value,
                seq.performance_at_stc_final.status.value,
            ],
            "Result": [
                "Pass",
                f"{seq.performance_at_stc.measured_value:.1f} W",
                f"{seq.wet_leakage_current.measured_value:.2f} mA",
                f"{seq.thermal_cycling.measured_value:.1f}% degradation",
                f"{seq.humidity_freeze.measured_value:.1f}% degradation",
                f"{seq.damp_heat.measured_value:.1f}% degradation",
                "Pass",
                "Pass",
                "Pass",
                "Pass",
                "Pass",
                f"{seq.performance_at_stc_final.measured_value:.1f} W",
            ],
        }

        df = pd.DataFrame(test_data)

        # Color-code status column
        def color_status(val: str) -> str:
            if val == "passed":
                return "background-color: #90EE90"
            elif val == "failed":
                return "background-color: #FFB6C1"
            else:
                return ""

        styled_df = df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True)

        # IV Curve Comparison
        st.subheader("IV Curve Comparison")
        viz = st.session_state.visualization
        iv_fig = viz.iv_curve_comparison(
            seq.iv_curve_initial, seq.iv_curve_final, "Initial vs Final IV Curves"
        )
        st.plotly_chart(iv_fig, use_container_width=True)

    def render_iec61730_tab(self) -> None:
        """Render IEC 61730 results tab."""
        st.header("IEC 61730: Safety Qualification")

        if not st.session_state.iec_61730_results:
            st.info("No IEC 61730 test results available. Run tests from the sidebar.")
            return

        # Select result to display
        result_index = st.selectbox(
            "Select Test Campaign",
            range(len(st.session_state.iec_61730_results)),
            format_func=lambda i: st.session_state.iec_61730_results[
                i
            ].test_campaign_id,
            key="iec61730_select",
        )

        result: IEC61730Result = st.session_state.iec_61730_results[result_index]

        # Display overall status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Status",
                result.overall_status.value.upper(),
                delta="PASS" if result.overall_status == TestStatus.PASSED else "FAIL",
            )

        with col2:
            st.metric("Safety Class", result.safety_class)

        with col3:
            st.metric("Compliance Rate", f"{result.compliance_percentage:.1f}%")

        st.markdown("---")

        # Safety test results
        st.subheader("Safety Test Results")

        safety = result.safety_tests
        test_data = {
            "Test Name": [
                "Insulation Resistance",
                "Dielectric Withstand",
                "Ground Continuity",
                "Fire Resistance",
                "Mechanical Stress",
                "Impact Resistance",
                "UV Exposure",
                "Corrosion Resistance",
            ],
            "Status": [
                safety.insulation_test.status.value,
                safety.dielectric_withstand.status.value,
                safety.ground_continuity.status.value,
                safety.fire_test.status.value,
                safety.mechanical_stress.status.value,
                safety.impact_test.status.value,
                safety.UV_test.status.value,
                safety.corrosion_test.status.value,
            ],
            "Result": [
                f"{safety.insulation_test.measured_value:.0f} MΩ (Req: >{safety.insulation_test.required_value:.0f} MΩ)",
                f"{safety.dielectric_withstand.measured_value:.2f} mA (Req: <{safety.dielectric_withstand.required_value:.0f} mA)",
                f"{safety.ground_continuity.measured_value:.3f} Ω (Req: <{safety.ground_continuity.required_value:.1f} Ω)",
                "Class C",
                "Pass",
                "Pass",
                "Pass",
                "Pass",
            ],
        }

        df = pd.DataFrame(test_data)

        def color_status(val: str) -> str:
            if val == "passed":
                return "background-color: #90EE90"
            elif val == "failed":
                return "background-color: #FFB6C1"
            else:
                return ""

        styled_df = df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True)

    def render_iec63202_tab(self) -> None:
        """Render IEC 63202 results tab."""
        st.header("IEC 63202: CTM Power Loss Analysis")

        if not st.session_state.iec_63202_results:
            st.info("No IEC 63202 test results available. Run tests from the sidebar.")
            return

        # Select result to display
        result_index = st.selectbox(
            "Select Test Campaign",
            range(len(st.session_state.iec_63202_results)),
            format_func=lambda i: st.session_state.iec_63202_results[
                i
            ].test_campaign_id,
            key="iec63202_select",
        )

        result: IEC63202Result = st.session_state.iec_63202_results[result_index]

        # Display overall metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CTM Ratio", f"{result.ctm_loss_breakdown.ctm_ratio:.3f}")

        with col2:
            st.metric(
                "Total CTM Loss", f"{result.ctm_loss_breakdown.total_ctm_loss:.2f}%"
            )

        with col3:
            st.metric("Module Power", f"{result.module_power:.2f} W")

        with col4:
            st.metric("Average Cell Power", f"{result.cell_power_avg:.2f} W")

        st.markdown("---")

        # CTM Loss Breakdown
        st.subheader("CTM Loss Breakdown")

        col1, col2 = st.columns([1, 1])

        with col1:
            ctm = result.ctm_loss_breakdown
            loss_data = {
                "Loss Component": [
                    "Optical Loss",
                    "Electrical Loss",
                    "Thermal Loss",
                    "Mismatch Loss",
                    "Interconnection Loss",
                    "Inactive Area Loss",
                ],
                "Loss (%)": [
                    f"{ctm.optical_loss:.2f}",
                    f"{ctm.electrical_loss:.2f}",
                    f"{ctm.thermal_loss:.2f}",
                    f"{ctm.mismatch_loss:.2f}",
                    f"{ctm.interconnection_loss:.2f}",
                    f"{ctm.inactive_area_loss:.2f}",
                ],
            }

            df = pd.DataFrame(loss_data)
            st.dataframe(df, use_container_width=True)

        with col2:
            # Waterfall chart
            viz = st.session_state.visualization
            waterfall_fig = viz.ctm_loss_waterfall(result.ctm_loss_breakdown)
            st.plotly_chart(waterfall_fig, use_container_width=True)

        # Module IV Curve
        st.subheader("Module IV Curve")
        module_iv = result.module_iv_curve
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=module_iv.voltage,
                y=module_iv.current,
                mode="lines",
                name="Module IV Curve",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_layout(
            xaxis_title="Voltage (V)",
            yaxis_title="Current (A)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_combined_view_tab(self) -> None:
        """Render combined view with all test results."""
        st.header("Combined Test Results & Compliance Dashboard")

        if not any(
            [
                st.session_state.iec_61215_results,
                st.session_state.iec_61730_results,
                st.session_state.iec_63202_results,
            ]
        ):
            st.info(
                "No test results available. Run tests from the sidebar to see combined view."
            )
            return

        # Aggregate compliance status
        compliance_status = st.session_state.test_results_manager.aggregate_compliance_status()

        # Display overall metrics
        st.subheader("Overall Compliance Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Status", compliance_status["overall_status"])

        with col2:
            st.metric("Total Tests", compliance_status["total_tests"])

        with col3:
            st.metric("Passed Tests", compliance_status["passed_tests"])

        with col4:
            st.metric("Compliance Rate", f"{compliance_status['compliance_rate']:.1f}%")

        st.markdown("---")

        # Compliance matrix
        st.subheader("Compliance Matrix")

        compliance_matrix = st.session_state.test_results_manager.generate_compliance_matrix()

        # Pass/Fail Summary Chart
        viz = st.session_state.visualization
        summary_fig = viz.pass_fail_summary(compliance_matrix)
        st.plotly_chart(summary_fig, use_container_width=True)

        # Detailed compliance table
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**IEC 61215 Tests**")
            df_61215 = pd.DataFrame(
                [
                    {"Test": k, "Status": v.value}
                    for k, v in compliance_matrix.iec_61215_tests.items()
                ]
            )
            st.dataframe(df_61215, use_container_width=True, height=300)

        with col2:
            st.markdown("**IEC 61730 Tests**")
            df_61730 = pd.DataFrame(
                [
                    {"Test": k, "Status": v.value}
                    for k, v in compliance_matrix.iec_61730_tests.items()
                ]
            )
            st.dataframe(df_61730, use_container_width=True, height=300)

        with col3:
            st.markdown("**IEC 63202 Tests**")
            df_63202 = pd.DataFrame(
                [
                    {"Test": k, "Status": v.value}
                    for k, v in compliance_matrix.iec_63202_tests.items()
                ]
            )
            st.dataframe(df_63202, use_container_width=True, height=300)

    def render_certification_tab(self) -> None:
        """Render certification management tab."""
        st.header("Certification Management")

        st.subheader("Target Certifications")

        # Select certification bodies
        selected_bodies = st.multiselect(
            "Select Certification Bodies",
            [body.value for body in CertificationBodyType],
            default=["tuv_rheinland", "ul"],
        )

        if st.button("Prepare Certification Package"):
            if not any(
                [
                    st.session_state.iec_61215_results,
                    st.session_state.iec_61730_results,
                    st.session_state.iec_63202_results,
                ]
            ):
                st.error("No test results available. Run tests first.")
                return

            # Create compliance report
            compliance_matrix = (
                st.session_state.test_results_manager.generate_compliance_matrix()
            )

            compliance_report = ComplianceReport(
                report_id=f"COMP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                module_type=st.session_state.iec_61215_results[0].module_type
                if st.session_state.iec_61215_results
                else "Unknown",
                manufacturer=st.session_state.iec_61215_results[0].manufacturer
                if st.session_state.iec_61215_results
                else "Unknown",
                iec_61215_result=st.session_state.iec_61215_results[0]
                if st.session_state.iec_61215_results
                else None,
                iec_61730_result=st.session_state.iec_61730_results[0]
                if st.session_state.iec_61730_results
                else None,
                iec_63202_result=st.session_state.iec_63202_results[0]
                if st.session_state.iec_63202_results
                else None,
                compliance_matrix=compliance_matrix,
                overall_status=TestStatus.PASSED
                if compliance_matrix.overall_compliance
                else TestStatus.FAILED,
                certification_ready=compliance_matrix.overall_compliance,
            )

            # Prepare package
            target_certs = [
                CertificationBodyType(body) for body in selected_bodies
            ]
            package = st.session_state.certification_workflow.prepare_certification_package(
                compliance_report=compliance_report,
                target_certifications=target_certs,
                module_type=compliance_report.module_type,
                manufacturer=compliance_report.manufacturer,
            )

            st.success(f"Certification package prepared: {package.package_id}")

            # Display package summary
            st.subheader("Package Summary")
            st.write(f"**Package ID:** {package.package_id}")
            st.write(f"**Module Type:** {package.module_type}")
            st.write(f"**Manufacturer:** {package.manufacturer}")
            st.write(
                f"**Target Certifications:** {', '.join([c.value for c in package.target_certifications])}"
            )
            st.write(f"**Certification Ready:** {'Yes' if package.compliance_report.certification_ready else 'No'}")

        # Certification cost tracking
        st.markdown("---")
        st.subheader("Certification Cost Tracking")

        # Track status for mock package
        statuses = st.session_state.certification_workflow.track_certification_status(
            "CERT-DEMO"
        )

        costs = st.session_state.certification_workflow.manage_certification_costs(
            statuses
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Certification Cost", f"${costs['total_cost']:,.2f}")

        with col2:
            avg_timeline = 45  # days
            st.metric("Average Timeline", f"{avg_timeline} days")

        # Status table
        status_data = {
            "Certification Body": [s.certification_body.value for s in statuses],
            "Status": [s.status for s in statuses],
            "Cost": [f"${s.certification_cost:,.2f}" if s.certification_cost else "N/A" for s in statuses],
            "Expected Completion": [
                s.expected_completion_date.strftime("%Y-%m-%d")
                if s.expected_completion_date
                else "N/A"
                for s in statuses
            ],
        }

        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True)

    def render_export_tab(self) -> None:
        """Render data export tab."""
        st.header("Export Test Data")

        if not any(
            [
                st.session_state.iec_61215_results,
                st.session_state.iec_61730_results,
                st.session_state.iec_63202_results,
            ]
        ):
            st.info("No test results available to export. Run tests first.")
            return

        # Export format selection
        export_format = st.selectbox(
            "Select Export Format", ["JSON", "Excel", "PDF Report"]
        )

        output_dir = Path("./exports")
        output_dir.mkdir(exist_ok=True)

        if st.button("Export Data"):
            with st.spinner(f"Exporting data in {export_format} format..."):
                if export_format == "JSON":
                    exported = st.session_state.test_results_manager.export_test_package(
                        output_dir, format="json"
                    )
                    st.success(f"Exported {len(exported)} JSON files to {output_dir}")
                    for name, path in exported.items():
                        st.write(f"- {name}: `{path}`")

                elif export_format == "Excel":
                    exported = st.session_state.test_results_manager.export_test_package(
                        output_dir, format="excel"
                    )
                    st.success(f"Exported Excel file to {output_dir}")
                    for name, path in exported.items():
                        st.write(f"- {name}: `{path}`")

                elif export_format == "PDF Report":
                    # Generate PDF reports
                    report_gen = st.session_state.report_generator

                    if st.session_state.iec_61215_results:
                        path = output_dir / "iec_61215_report.pdf"
                        report_gen.generate_iec61215_report(
                            st.session_state.iec_61215_results[0], path
                        )
                        st.success(f"Generated IEC 61215 report: `{path}`")

                    if st.session_state.iec_61730_results:
                        path = output_dir / "iec_61730_report.pdf"
                        report_gen.generate_iec61730_report(
                            st.session_state.iec_61730_results[0], path
                        )
                        st.success(f"Generated IEC 61730 report: `{path}`")

                    if st.session_state.iec_63202_results:
                        path = output_dir / "iec_63202_report.pdf"
                        report_gen.generate_iec63202_report(
                            st.session_state.iec_63202_results[0], path
                        )
                        st.success(f"Generated IEC 63202 report: `{path}`")

    def run(self) -> None:
        """Run the main Streamlit application."""
        # Render sidebar and get user selections
        selections = self.render_sidebar()

        # Handle test execution
        if selections["run_tests"]:
            results = self.run_all_tests(
                selections["module_type"],
                selections["manufacturer"],
                selections["test_campaign_id"],
            )
            st.success("✅ All IEC tests completed successfully!")

        # Main content area with tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "IEC 61215",
                "IEC 61730",
                "IEC 63202",
                "Combined View",
                "Certification",
                "Export",
            ]
        )

        with tab1:
            self.render_iec61215_tab()

        with tab2:
            self.render_iec61730_tab()

        with tab3:
            self.render_iec63202_tab()

        with tab4:
            self.render_combined_view_tab()

        with tab5:
            self.render_certification_tab()

        with tab6:
            self.render_export_tab()


def main() -> None:
    """Main entry point for Streamlit application."""
    app = IECTestingUI()
    app.run()


if __name__ == "__main__":
    main()
