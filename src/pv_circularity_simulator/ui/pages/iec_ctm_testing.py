"""
Streamlit UI for IEC 63202 CTM Testing and Validation.

This module provides an interactive web interface for:
- CTM test configuration
- Cell and module data input
- Real-time CTM ratio calculation
- Loss waterfall visualization
- IEC 63202 compliance verification
- Report generation and export
"""

from datetime import datetime
from typing import Optional, List
import logging

import streamlit as st
import pandas as pd
import numpy as np

from pv_circularity_simulator.core.iec63202.models import (
    CTMTestConfig,
    CellProperties,
    ModuleConfiguration,
    ReferenceDeviceData,
    FlashSimulatorData,
    IVCurveData,
    CellTechnology,
    FlashSimulatorType,
)
from pv_circularity_simulator.core.iec63202.tester import IEC63202CTMTester
from pv_circularity_simulator.core.iec63202.loss_analyzer import CTMPowerLossAnalyzer
from pv_circularity_simulator.core.iec63202.calibration import ReferenceDeviceCalibration
from pv_circularity_simulator.core.iec63202.report import CTMTestReport
from pv_circularity_simulator.core.ctm.b03_ctm_loss_model import (
    B03CTMLossModel,
    B03CTMConfiguration,
)
from pv_circularity_simulator.core.utils.constants import (
    STC_IRRADIANCE,
    STC_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class CTMTestUI:
    """
    Streamlit UI for IEC 63202 CTM testing.

    This class provides a comprehensive web interface for CTM testing including:
    - Interactive test configuration
    - Cell and module data input (manual or CSV upload)
    - Real-time CTM calculation and visualization
    - Loss breakdown analysis
    - Compliance checking
    - Report generation and download
    """

    def __init__(self) -> None:
        """Initialize CTM Test UI."""
        self.setup_page_config()
        logger.info("CTM Test UI initialized")

    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="IEC 63202 CTM Testing",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def run(self) -> None:
        """
        Run the main CTM testing interface.

        This is the entry point for the Streamlit application.
        """
        st.title("⚡ IEC 63202 Cell-to-Module (CTM) Testing")
        st.markdown("""
        This tool performs comprehensive IEC 63202 CTM testing and power loss validation.
        Configure your test, input cell and module measurements, and generate compliance reports.
        """)

        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select Page",
                [
                    "Test Configuration",
                    "Cell Measurements",
                    "Module Measurements",
                    "CTM Analysis",
                    "B03 Loss Model",
                    "Reports & Export"
                ]
            )

        # Route to appropriate page
        if page == "Test Configuration":
            self.test_configuration_page()
        elif page == "Cell Measurements":
            self.cell_measurements_page()
        elif page == "Module Measurements":
            self.module_measurements_page()
        elif page == "CTM Analysis":
            self.ctm_analysis_page()
        elif page == "B03 Loss Model":
            self.b03_loss_model_page()
        elif page == "Reports & Export":
            self.reports_page()

    def test_configuration_page(self) -> None:
        """Test configuration input page."""
        st.header("Test Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Test Information")
            test_id = st.text_input("Test ID", value=f"CTM-{datetime.now().strftime('%Y%m%d')}")
            laboratory = st.text_input("Laboratory", value="PV Testing Lab")
            operator = st.text_input("Operator Name", value="Lab Technician")

            st.subheader("Cell Properties")
            cell_tech = st.selectbox(
                "Cell Technology",
                options=[t.value for t in CellTechnology],
                index=2  # PERC
            )
            cell_area = st.number_input("Cell Area (cm²)", value=244.0, min_value=100.0, max_value=300.0)
            cell_efficiency = st.number_input("Cell Efficiency (%)", value=22.5, min_value=15.0, max_value=28.0)
            cell_pmax = st.number_input("Cell Pmax (W)", value=5.2, min_value=3.0, max_value=8.0)

        with col2:
            st.subheader("Module Configuration")
            num_cells_series = st.number_input("Cells in Series", value=60, min_value=30, max_value=150)
            num_strings = st.number_input("Parallel Strings", value=1, min_value=1, max_value=5)
            bypass_diodes = st.number_input("Bypass Diodes", value=3, min_value=0, max_value=10)

            st.subheader("Acceptance Criteria")
            ctm_min = st.number_input("Min CTM Ratio (%)", value=95.0, min_value=90.0, max_value=100.0)
            ctm_max = st.number_input("Max CTM Ratio (%)", value=102.0, min_value=100.0, max_value=110.0)

        # Calculate derived cell parameters
        voc = 0.68  # Typical for PERC
        isc = cell_pmax / (voc * 0.80)  # Assuming FF ~0.80
        vmp = voc * 0.85
        imp = cell_pmax / vmp

        # Store configuration in session state
        if st.button("Save Configuration", type="primary"):
            try:
                cell_props = CellProperties(
                    technology=CellTechnology(cell_tech),
                    area=cell_area,
                    efficiency=cell_efficiency,
                    voc=voc,
                    isc=isc,
                    vmp=vmp,
                    imp=imp,
                    pmax=cell_pmax,
                    temperature_coefficient_pmax=-0.39,
                    temperature_coefficient_voc=-0.0029,
                    temperature_coefficient_isc=0.0005,
                )

                module_config = ModuleConfiguration(
                    num_cells_series=num_cells_series,
                    num_strings_parallel=num_strings,
                    bypass_diodes=bypass_diodes,
                )

                # Create reference device (using defaults)
                ref_device = ReferenceDeviceData(
                    device_id="REF-001",
                    calibration_date=datetime.now(),
                    calibration_lab=laboratory,
                    calibration_certificate="CAL-2025-001",
                    short_circuit_current=isc,
                    responsivity=isc / STC_IRRADIANCE,
                    temperature_coefficient=0.0005,
                    uncertainty_isc=1.5,
                    uncertainty_temperature=0.2,
                    next_calibration_due=datetime(2026, 1, 1),
                )

                # Create flash simulator
                flash_sim = FlashSimulatorData(
                    simulator_type=FlashSimulatorType.LED,
                    spatial_uniformity=98.5,
                    temporal_stability=99.2,
                )

                # Create test config
                test_config = CTMTestConfig(
                    test_id=test_id,
                    laboratory=laboratory,
                    operator=operator,
                    cell_properties=cell_props,
                    module_config=module_config,
                    reference_device=ref_device,
                    flash_simulator=flash_sim,
                    acceptance_criteria_min=ctm_min,
                    acceptance_criteria_max=ctm_max,
                )

                st.session_state['ctm_config'] = test_config
                st.success("✓ Configuration saved successfully!")

                # Display summary
                st.info(f"""
                **Configuration Summary:**
                - Cell: {cell_tech}, {cell_pmax:.2f}W
                - Module: {module_config.total_cells} cells ({num_cells_series}S × {num_strings}P)
                - Expected Module Power: {cell_pmax * module_config.total_cells:.1f}W
                - Acceptance: {ctm_min:.1f}% - {ctm_max:.1f}%
                """)

            except Exception as e:
                st.error(f"Configuration error: {str(e)}")
                logger.error(f"Configuration error: {e}")

    def cell_measurements_page(self) -> None:
        """Cell measurements input page."""
        st.header("Reference Cell Measurements")

        if 'ctm_config' not in st.session_state:
            st.warning("⚠️ Please configure test settings first!")
            return

        st.markdown("""
        Enter IV curve measurements for reference cells tested under STC conditions.
        You can input data manually or upload a CSV file.
        """)

        # Input method selection
        input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload"])

        if input_method == "Manual Entry":
            self._manual_cell_input()
        else:
            self._csv_cell_upload()

        # Display current measurements
        if 'cell_measurements' in st.session_state:
            st.subheader("Current Cell Measurements")
            cell_data = []
            for i, iv in enumerate(st.session_state['cell_measurements'], 1):
                cell_data.append({
                    "Cell": i,
                    "Voc (V)": f"{iv.voc:.3f}",
                    "Isc (A)": f"{iv.isc:.3f}",
                    "Pmax (W)": f"{iv.pmax:.3f}",
                    "FF": f"{iv.fill_factor:.3f}",
                })
            st.dataframe(pd.DataFrame(cell_data), use_container_width=True)

    def module_measurements_page(self) -> None:
        """Module measurements input page."""
        st.header("Module Flash Test Measurements")

        if 'ctm_config' not in st.session_state:
            st.warning("⚠️ Please configure test settings first!")
            return

        st.markdown("""
        Enter IV curve measurements for modules tested using flash simulator under STC conditions.
        """)

        # Input method selection
        input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload"])

        if input_method == "Manual Entry":
            self._manual_module_input()
        else:
            self._csv_module_upload()

        # Display current measurements
        if 'module_measurements' in st.session_state:
            st.subheader("Current Module Measurements")
            module_data = []
            for i, iv in enumerate(st.session_state['module_measurements'], 1):
                module_data.append({
                    "Module": i,
                    "Voc (V)": f"{iv.voc:.2f}",
                    "Isc (A)": f"{iv.isc:.3f}",
                    "Pmax (W)": f"{iv.pmax:.2f}",
                    "FF": f"{iv.fill_factor:.3f}",
                })
            st.dataframe(pd.DataFrame(module_data), use_container_width=True)

    def ctm_analysis_page(self) -> None:
        """CTM analysis and results page."""
        st.header("CTM Analysis & Results")

        if 'ctm_config' not in st.session_state:
            st.warning("⚠️ Please configure test settings first!")
            return

        if 'cell_measurements' not in st.session_state or 'module_measurements' not in st.session_state:
            st.warning("⚠️ Please input cell and module measurements first!")
            return

        config = st.session_state['ctm_config']
        cell_measurements = st.session_state['cell_measurements']
        module_measurements = st.session_state['module_measurements']

        # Run CTM test
        if st.button("Run CTM Analysis", type="primary"):
            with st.spinner("Calculating CTM ratio..."):
                try:
                    # Create analyzer
                    loss_analyzer = CTMPowerLossAnalyzer()

                    # Create tester
                    tester = IEC63202CTMTester(
                        config=config,
                        power_loss_analyzer=loss_analyzer
                    )

                    # Run CTM test
                    result = tester.ctm_power_ratio_test(
                        cell_measurements=cell_measurements,
                        module_measurements=module_measurements
                    )

                    st.session_state['ctm_result'] = result
                    st.success("✓ CTM analysis complete!")

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    logger.error(f"CTM analysis error: {e}")
                    return

        # Display results
        if 'ctm_result' in st.session_state:
            result = st.session_state['ctm_result']

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "CTM Ratio",
                    f"{result.ctm_ratio:.2f}%",
                    delta=f"{result.ctm_ratio - 100:.2f}%"
                )
            with col2:
                st.metric(
                    "Uncertainty",
                    f"±{result.ctm_ratio_uncertainty:.2f}%"
                )
            with col3:
                st.metric(
                    "Total Loss",
                    f"{result.loss_components.total_loss:.2f}%"
                )
            with col4:
                compliance = "PASS ✓" if result.compliance_status else "FAIL ✗"
                st.metric("Compliance", compliance)

            # Loss breakdown
            st.subheader("Loss Breakdown")
            col1, col2 = st.columns(2)

            with col1:
                loss_data = {
                    "Category": [
                        "Optical Reflection",
                        "Optical Absorption",
                        "Optical Shading",
                        "Electrical Series R",
                        "Electrical Mismatch",
                        "Thermal",
                        "Spatial",
                        "Spectral"
                    ],
                    "Loss (%)": [
                        result.loss_components.optical_reflection,
                        result.loss_components.optical_absorption,
                        result.loss_components.optical_shading,
                        result.loss_components.electrical_series_resistance,
                        result.loss_components.electrical_mismatch,
                        result.loss_components.thermal_assembly,
                        result.loss_components.spatial_non_uniformity,
                        result.loss_components.spectral_mismatch,
                    ]
                }
                st.dataframe(pd.DataFrame(loss_data), use_container_width=True)

            with col2:
                # Create report and show waterfall
                report = CTMTestReport(result)
                fig = report.create_loss_waterfall_chart()
                st.plotly_chart(fig, use_container_width=True)

            # IV curves
            st.subheader("IV Curve Comparison")
            fig = report.create_iv_curve_comparison()
            st.plotly_chart(fig, use_container_width=True)

    def b03_loss_model_page(self) -> None:
        """B03 CTM loss model analysis page."""
        st.header("B03 CTM Loss Model (k1-k24 Factors)")

        st.markdown("""
        Analyze CTM losses using the comprehensive B03 model with 24 individual loss factors.
        Select a quality scenario or configure individual factors.
        """)

        # Scenario selection
        scenario = st.selectbox(
            "Quality Scenario",
            ["premium_quality", "standard_quality", "economy_quality"],
            index=1
        )

        if st.button("Calculate B03 CTM Losses", type="primary"):
            with st.spinner("Calculating B03 CTM model..."):
                try:
                    model = B03CTMLossModel()
                    config = B03CTMConfiguration.from_scenario(scenario)
                    result = model.calculate_ctm_losses(config)

                    st.session_state['b03_result'] = result

                except Exception as e:
                    st.error(f"B03 calculation error: {str(e)}")
                    return

        # Display results
        if 'b03_result' in st.session_state:
            result = st.session_state['b03_result']

            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CTM Ratio", f"{result.total_ctm_ratio_percent:.2f}%")
            with col2:
                st.metric("Total Loss", f"{result.total_loss_percent:.2f}%")
            with col3:
                st.metric("Scenario", scenario)

            # Category breakdown
            st.subheader("Loss by Category")
            breakdown = result.get_loss_breakdown()
            breakdown_df = pd.DataFrame([
                {"Category": k.replace("_", " ").title(), "Loss (%)": f"{v:.3f}"}
                for k, v in breakdown.items()
            ])
            st.dataframe(breakdown_df, use_container_width=True)

            # Individual factors
            with st.expander("Individual k Factors"):
                factors_df = pd.DataFrame([
                    {"Factor": k, "Value": f"{v:.6f}"}
                    for k, v in result.individual_factors.items()
                ])
                st.dataframe(factors_df, use_container_width=True)

    def reports_page(self) -> None:
        """Reports and export page."""
        st.header("Reports & Export")

        if 'ctm_result' not in st.session_state:
            st.warning("⚠️ Please run CTM analysis first!")
            return

        result = st.session_state['ctm_result']
        report = CTMTestReport(result)

        st.subheader("Download Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Excel export
            if st.button("Generate Excel Report"):
                try:
                    file_path = f"ctm_report_{result.config.test_id}.xlsx"
                    report.export_to_excel(file_path)
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "Download Excel",
                            data=f,
                            file_name=file_path,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Excel export error: {str(e)}")

        with col2:
            # HTML export
            if st.button("Generate HTML Report"):
                try:
                    html = report.generate_html_report()
                    st.download_button(
                        "Download HTML",
                        data=html,
                        file_name=f"ctm_report_{result.config.test_id}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"HTML export error: {str(e)}")

        with col3:
            # PDF export
            if st.button("Generate PDF Report"):
                try:
                    file_path = f"ctm_report_{result.config.test_id}.pdf"
                    report.export_to_pdf(file_path)
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            data=f,
                            file_name=file_path,
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"PDF export error: {str(e)}")

        # Preview compliance dashboard
        st.subheader("Compliance Dashboard")
        fig = report.create_compliance_dashboard()
        st.plotly_chart(fig, use_container_width=True)

    def _manual_cell_input(self) -> None:
        """Manual cell measurement input."""
        st.subheader("Manual Cell Data Entry")

        # Simple IV curve input
        col1, col2 = st.columns(2)
        with col1:
            voc = st.number_input("Voc (V)", value=0.68, min_value=0.5, max_value=1.0, step=0.01)
            isc = st.number_input("Isc (A)", value=8.5, min_value=5.0, max_value=15.0, step=0.1)
        with col2:
            pmax = st.number_input("Pmax (W)", value=5.2, min_value=3.0, max_value=8.0, step=0.1)
            temperature = st.number_input("Temperature (°C)", value=25.0, min_value=20.0, max_value=30.0)

        if st.button("Add Cell Measurement"):
            # Generate simple IV curve
            v_points = np.linspace(0, voc, 20)
            # Simple diode model approximation
            i_points = isc * (1 - (v_points / voc) ** 3)

            iv_curve = IVCurveData(
                voltage=v_points.tolist(),
                current=i_points.tolist(),
                temperature=temperature,
                irradiance=STC_IRRADIANCE,
            )

            if 'cell_measurements' not in st.session_state:
                st.session_state['cell_measurements'] = []

            st.session_state['cell_measurements'].append(iv_curve)
            st.success(f"✓ Cell measurement added (Pmax: {iv_curve.pmax:.3f}W)")

    def _manual_module_input(self) -> None:
        """Manual module measurement input."""
        st.subheader("Manual Module Data Entry")

        config = st.session_state['ctm_config']
        n_cells = config.module_config.total_cells

        col1, col2 = st.columns(2)
        with col1:
            voc = st.number_input("Voc (V)", value=40.8, min_value=30.0, max_value=60.0, step=0.1)
            isc = st.number_input("Isc (A)", value=8.5, min_value=5.0, max_value=15.0, step=0.1)
        with col2:
            pmax = st.number_input("Pmax (W)", value=300.0, min_value=200.0, max_value=500.0, step=1.0)
            temperature = st.number_input("Temperature (°C)", value=25.0, min_value=20.0, max_value=30.0)

        if st.button("Add Module Measurement"):
            # Generate simple IV curve
            v_points = np.linspace(0, voc, 30)
            i_points = isc * (1 - (v_points / voc) ** 3)

            iv_curve = IVCurveData(
                voltage=v_points.tolist(),
                current=i_points.tolist(),
                temperature=temperature,
                irradiance=STC_IRRADIANCE,
            )

            if 'module_measurements' not in st.session_state:
                st.session_state['module_measurements'] = []

            st.session_state['module_measurements'].append(iv_curve)
            st.success(f"✓ Module measurement added (Pmax: {iv_curve.pmax:.2f}W)")

    def _csv_cell_upload(self) -> None:
        """CSV cell data upload."""
        st.subheader("Upload Cell Data CSV")
        st.markdown("CSV format: voltage,current,temperature,irradiance")

        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Process CSV and create IV curves
                st.info("CSV upload functionality - implement parsing logic")
            except Exception as e:
                st.error(f"CSV upload error: {str(e)}")

    def _csv_module_upload(self) -> None:
        """CSV module data upload."""
        st.subheader("Upload Module Data CSV")
        st.markdown("CSV format: voltage,current,temperature,irradiance")

        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Process CSV and create IV curves
                st.info("CSV upload functionality - implement parsing logic")
            except Exception as e:
                st.error(f"CSV upload error: {str(e)}")


def main() -> None:
    """Main entry point for Streamlit app."""
    ui = CTMTestUI()
    ui.run()


if __name__ == "__main__":
    main()
