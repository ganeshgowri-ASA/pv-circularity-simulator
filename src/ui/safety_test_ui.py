"""Streamlit UI component for IEC 61730 safety testing.

This module provides an interactive web interface for configuring, executing,
and visualizing PV module safety tests per IEC 61730 standards.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from loguru import logger

from ..models.safety_models import (
    SafetyTestConfig,
    SafetyClass,
    ApplicationClass,
    FireClass,
)
from ..safety.iec61730_tester import IEC61730SafetyTester
from ..safety.safety_report import SafetyQualificationReport


class SafetyTestUI:
    """Interactive Streamlit UI for IEC 61730 safety testing.

    Provides a comprehensive web interface for:
    - Safety test configuration
    - Test execution and monitoring
    - Real-time visualization of test results
    - Safety certificate generation
    - Comparison with standard requirements
    - Report export (PDF, JSON)

    The UI is organized into multiple tabs for different aspects of testing.
    """

    def __init__(self) -> None:
        """Initialize the safety test UI."""
        self.tester: Optional[IEC61730SafetyTester] = None
        logger.info("Initialized SafetyTestUI")

    def render(self) -> None:
        """Render the complete safety test UI.

        This is the main entry point for the Streamlit app.
        """
        st.set_page_config(
            page_title="IEC 61730 Safety Testing",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("⚡ IEC 61730 PV Module Safety Testing")
        st.markdown(
            "Comprehensive safety testing and qualification per IEC 61730-1 "
            "(Construction Requirements) and IEC 61730-2 (Testing Requirements)"
        )

        # Sidebar for configuration
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Configuration",
            "🔬 Test Execution",
            "📊 Results & Visualization",
            "📜 Certification",
            "📄 Reports"
        ])

        with tab1:
            self._render_configuration_tab()

        with tab2:
            self._render_test_execution_tab()

        with tab3:
            self._render_results_tab()

        with tab4:
            self._render_certification_tab()

        with tab5:
            self._render_reports_tab()

    def _render_sidebar(self) -> None:
        """Render the sidebar with quick info and controls."""
        st.sidebar.title("Quick Info")

        st.sidebar.markdown("### Standards")
        st.sidebar.markdown("""
        - **IEC 61730-1:2016** - Construction Requirements
        - **IEC 61730-2:2016** - Testing Requirements
        - **UL 790** - Fire Classification
        """)

        st.sidebar.markdown("### Test Categories")
        st.sidebar.markdown("""
        - ⚡ Electrical Safety (MST 01-05)
        - 🔧 Mechanical Safety (MST 06-08)
        - 🔥 Fire Safety (Annex C)
        - 🌡️ Environmental Safety (MST 09-11)
        - 📐 Construction Requirements
        """)

        if st.sidebar.button("Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    def _render_configuration_tab(self) -> None:
        """Render the configuration tab for test setup."""
        st.header("Test Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Module Information")

            module_id = st.text_input(
                "Module ID *",
                value="MOD-001",
                help="Unique identifier for the module under test"
            )

            manufacturer = st.text_input(
                "Manufacturer *",
                value="SolarTech Inc.",
                help="Module manufacturer name"
            )

            model_number = st.text_input(
                "Model Number *",
                value="ST-400-72MH",
                help="Module model number"
            )

            serial_number = st.text_input(
                "Serial Number",
                value="SN-2024-001",
                help="Module serial number (optional)"
            )

            module_area_m2 = st.number_input(
                "Module Area (m²) *",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Module area in square meters"
            )

            max_system_voltage_v = st.number_input(
                "Maximum System Voltage (V) *",
                min_value=0.0,
                max_value=2000.0,
                value=1000.0,
                step=10.0,
                help="Maximum DC system voltage"
            )

        with col2:
            st.subheader("Test Parameters")

            target_safety_class = st.selectbox(
                "Target Safety Class *",
                options=[cls.value for cls in SafetyClass],
                index=0,
                help="Target safety classification"
            )

            application_class = st.selectbox(
                "Application Class *",
                options=[cls.value for cls in ApplicationClass],
                index=1,
                help="Module application class"
            )

            target_fire_class = st.selectbox(
                "Target Fire Class",
                options=["None"] + [cls.value for cls in FireClass],
                index=1,
                help="Target fire classification (if fire tests are performed)"
            )

            test_laboratory = st.text_input(
                "Test Laboratory *",
                value="TUV Rheinland",
                help="Name of testing laboratory"
            )

            test_date = st.date_input(
                "Test Date *",
                value=datetime.now(),
                help="Date of test execution"
            )

        st.subheader("Test Selection")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            perform_electrical = st.checkbox(
                "Electrical Safety Tests",
                value=True,
                help="MST 01-05: Insulation, leakage, dielectric, ground, bypass diode"
            )

        with col2:
            perform_mechanical = st.checkbox(
                "Mechanical Safety Tests",
                value=True,
                help="MST 06-08: Load, impact, terminations"
            )

        with col3:
            perform_fire = st.checkbox(
                "Fire Safety Tests",
                value=False,
                help="Annex C: Spread of flame, penetration, brand"
            )

        with col4:
            perform_environmental = st.checkbox(
                "Environmental Safety Tests",
                value=True,
                help="MST 09-11: UV, thermal cycling, humidity-freeze"
            )

        st.markdown("---")

        if st.button("💾 Save Configuration", type="primary"):
            # Create config
            config = SafetyTestConfig(
                module_id=module_id,
                manufacturer=manufacturer,
                model_number=model_number,
                serial_number=serial_number if serial_number else None,
                max_system_voltage_v=max_system_voltage_v,
                module_area_m2=module_area_m2,
                application_class=ApplicationClass(application_class),
                target_safety_class=SafetyClass(target_safety_class),
                target_fire_class=(
                    FireClass(target_fire_class) if target_fire_class != "None" else None
                ),
                test_laboratory=test_laboratory,
                test_date=datetime.combine(test_date, datetime.min.time()),
                perform_electrical_tests=perform_electrical,
                perform_mechanical_tests=perform_mechanical,
                perform_fire_tests=perform_fire,
                perform_environmental_tests=perform_environmental,
            )

            # Save to session state
            st.session_state['config'] = config
            st.session_state['tester'] = IEC61730SafetyTester(config)

            st.success("✅ Configuration saved successfully!")
            logger.info(f"Configuration saved for module {module_id}")

    def _render_test_execution_tab(self) -> None:
        """Render the test execution tab."""
        st.header("Test Execution")

        if 'config' not in st.session_state:
            st.warning("⚠️ Please configure the test in the Configuration tab first.")
            return

        config: SafetyTestConfig = st.session_state['config']
        tester: IEC61730SafetyTester = st.session_state['tester']

        st.info(f"**Module:** {config.model_number} | **Target Class:** {config.target_safety_class.value}")

        st.subheader("Test Suite Execution")

        # Test execution controls
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            if st.button("▶️ Run All Tests", type="primary"):
                with st.spinner("Running complete test suite..."):
                    results = tester.run_all_tests()
                    st.session_state['results'] = results
                    logger.info(f"Test suite completed for {config.module_id}")

                if results.overall_pass:
                    st.success("✅ ALL TESTS PASSED!")
                else:
                    st.error("❌ SOME TESTS FAILED")

        with col2:
            if st.button("🔄 Reset Tests"):
                if 'results' in st.session_state:
                    del st.session_state['results']
                st.session_state['tester'] = IEC61730SafetyTester(config)
                st.success("Tests reset")

        # Individual test execution
        st.markdown("---")
        st.subheader("Individual Test Execution")

        col1, col2 = st.columns(2)

        with col1:
            if config.perform_electrical_tests:
                if st.button("⚡ Run Electrical Tests"):
                    with st.spinner("Running electrical safety tests..."):
                        results = tester.electrical_safety_tests()
                        if 'partial_results' not in st.session_state:
                            st.session_state['partial_results'] = {}
                        st.session_state['partial_results']['electrical'] = results

                    if results.all_tests_passed:
                        st.success("✅ Electrical tests passed")
                    else:
                        st.error("❌ Electrical tests failed")

            if config.perform_fire_tests:
                if st.button("🔥 Run Fire Tests"):
                    with st.spinner("Running fire safety tests..."):
                        results = tester.fire_safety_tests()
                        if 'partial_results' not in st.session_state:
                            st.session_state['partial_results'] = {}
                        st.session_state['partial_results']['fire'] = results

                    st.info(f"Fire Classification: {results.fire_classification.value}")

        with col2:
            if config.perform_mechanical_tests:
                if st.button("🔧 Run Mechanical Tests"):
                    with st.spinner("Running mechanical safety tests..."):
                        results = tester.mechanical_safety_tests()
                        if 'partial_results' not in st.session_state:
                            st.session_state['partial_results'] = {}
                        st.session_state['partial_results']['mechanical'] = results

                    if results.all_tests_passed:
                        st.success("✅ Mechanical tests passed")
                    else:
                        st.error("❌ Mechanical tests failed")

            if config.perform_environmental_tests:
                if st.button("🌡️ Run Environmental Tests"):
                    with st.spinner("Running environmental safety tests..."):
                        results = tester.environmental_safety_tests()
                        if 'partial_results' not in st.session_state:
                            st.session_state['partial_results'] = {}
                        st.session_state['partial_results']['environmental'] = results

                    if results.all_tests_passed:
                        st.success("✅ Environmental tests passed")
                    else:
                        st.error("❌ Environmental tests failed")

        # Display partial results if available
        if 'partial_results' in st.session_state:
            st.markdown("---")
            st.subheader("Partial Test Results")
            self._display_partial_results(st.session_state['partial_results'])

    def _render_results_tab(self) -> None:
        """Render the results and visualization tab."""
        st.header("Test Results & Visualization")

        if 'results' not in st.session_state:
            st.warning("⚠️ No test results available. Run tests in the Test Execution tab.")
            return

        results = st.session_state['results']

        # Overall status
        col1, col2, col3 = st.columns(3)

        with col1:
            status_color = "green" if results.overall_pass else "red"
            status_text = "PASS" if results.overall_pass else "FAIL"
            st.markdown(f"### Overall Status: :{status_color}[{status_text}]")

        with col2:
            if results.classification:
                st.markdown(f"### Safety Class: {results.classification.safety_class.value}")

        with col3:
            if results.classification and results.classification.fire_class != FireClass.NOT_RATED:
                st.markdown(f"### Fire Class: {results.classification.fire_class.value}")

        st.markdown("---")

        # Test results breakdown
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚡ Electrical",
            "🔧 Mechanical",
            "🔥 Fire",
            "🌡️ Environmental",
            "📐 Construction"
        ])

        with tab1:
            if results.electrical_tests:
                self._display_electrical_results(results.electrical_tests)

        with tab2:
            if results.mechanical_tests:
                self._display_mechanical_results(results.mechanical_tests)

        with tab3:
            if results.fire_tests:
                self._display_fire_results(results.fire_tests)

        with tab4:
            if results.environmental_tests:
                self._display_environmental_results(results.environmental_tests)

        with tab5:
            if results.construction_requirements:
                self._display_construction_results(results.construction_requirements)

    def _render_certification_tab(self) -> None:
        """Render the certification tab."""
        st.header("Safety Certification")

        if 'results' not in st.session_state:
            st.warning("⚠️ No test results available. Complete tests first.")
            return

        results = st.session_state['results']

        if not results.overall_pass:
            st.error("❌ Module did not pass all required tests. Certification cannot be issued.")
            st.markdown("### Failed Tests:")

            failures = []
            if results.electrical_tests and not results.electrical_tests.all_tests_passed:
                failures.append("- Electrical safety tests")
            if results.mechanical_tests and not results.mechanical_tests.all_tests_passed:
                failures.append("- Mechanical safety tests")
            if results.fire_tests and not results.fire_tests.fire_tests_passed:
                failures.append("- Fire safety tests")
            if results.environmental_tests and not results.environmental_tests.all_tests_passed:
                failures.append("- Environmental safety tests")

            for failure in failures:
                st.markdown(failure)

            return

        st.success("✅ Module passed all required tests and is eligible for certification")

        st.subheader("Generate Safety Certificate")

        certification_body = st.selectbox(
            "Certification Body",
            options=[
                "TUV Rheinland",
                "TUV SUD",
                "UL (Underwriters Laboratories)",
                "IEC System for Conformity Assessment",
                "CSA Group",
                "Intertek",
                "VDE Testing and Certification Institute",
            ],
            help="Select the certification body"
        )

        if st.button("📜 Generate Certificate", type="primary"):
            with st.spinner("Generating safety certificate..."):
                tester: IEC61730SafetyTester = st.session_state['tester']
                certificate = tester.export_safety_certificate(
                    certification_body=certification_body
                )
                st.session_state['certificate'] = certificate

            st.success("✅ Certificate generated successfully!")

        # Display certificate if generated
        if 'certificate' in st.session_state:
            certificate = st.session_state['certificate']

            st.markdown("---")
            st.subheader("Certificate Details")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Certificate Number:** {certificate.certificate_number}")
                st.markdown(f"**Issue Date:** {certificate.issue_date.strftime('%Y-%m-%d')}")
                if certificate.expiry_date:
                    st.markdown(f"**Expiry Date:** {certificate.expiry_date.strftime('%Y-%m-%d')}")
                st.markdown(f"**Certification Body:** {certificate.certification_body}")

            with col2:
                st.markdown(f"**Safety Class:** {certificate.certified_safety_class.value}")
                st.markdown(f"**Application Class:** {certificate.certified_application_class.value}")
                if certificate.certified_fire_class:
                    st.markdown(f"**Fire Class:** {certificate.certified_fire_class.value}")

            st.markdown("---")
            st.markdown("### Module Information")
            st.markdown(f"**Manufacturer:** {certificate.module_info.manufacturer}")
            st.markdown(f"**Model:** {certificate.module_info.model_number}")
            st.markdown(f"**Serial Number:** {certificate.module_info.serial_number}")

    def _render_reports_tab(self) -> None:
        """Render the reports tab."""
        st.header("Test Reports")

        if 'results' not in st.session_state:
            st.warning("⚠️ No test results available. Complete tests first.")
            return

        results = st.session_state['results']
        certificate = st.session_state.get('certificate', None)

        report_generator = SafetyQualificationReport(
            test_results=results,
            certificate=certificate,
        )

        st.subheader("Report Summary")
        summary = report_generator.generate_summary()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Module:** {summary['model_number']}")
            st.markdown(f"**Manufacturer:** {summary['manufacturer']}")
            st.markdown(f"**Overall Status:** {summary['overall_status']}")

        with col2:
            if summary['safety_classification']:
                st.markdown(f"**Safety Class:** {summary['safety_classification']['safety_class']}")
                st.markdown(f"**Fire Class:** {summary['safety_classification']['fire_class']}")

        # Test results summary
        st.markdown("---")
        st.subheader("Test Results Summary")

        summary_df = pd.DataFrame([
            {"Test Category": key.capitalize(), "Result": value}
            for key, value in summary['test_results_summary'].items()
        ])
        st.dataframe(summary_df, use_container_width=True)

        # Export options
        st.markdown("---")
        st.subheader("Export Reports")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export to PDF"):
                output_dir = Path("./reports")
                output_dir.mkdir(exist_ok=True)

                pdf_path = output_dir / f"{results.config.module_id}_safety_report.pdf"

                with st.spinner("Generating PDF report..."):
                    report_generator.export_to_pdf(pdf_path)

                st.success(f"✅ PDF report saved to {pdf_path}")

        with col2:
            if st.button("📊 Export to JSON"):
                output_dir = Path("./reports")
                output_dir.mkdir(exist_ok=True)

                json_path = output_dir / f"{results.config.module_id}_safety_report.json"

                with st.spinner("Generating JSON report..."):
                    report_generator.export_to_json(json_path)

                st.success(f"✅ JSON report saved to {json_path}")

        # Display detailed report
        if st.checkbox("Show Detailed Report"):
            st.markdown("---")
            st.subheader("Detailed Test Report")

            detailed_report = report_generator.generate_detailed_report()
            st.json(detailed_report)

    # Helper methods for displaying results

    def _display_partial_results(self, partial_results: dict) -> None:
        """Display partial test results."""
        for test_type, results in partial_results.items():
            st.markdown(f"**{test_type.capitalize()}:** {type(results).__name__}")

    def _display_electrical_results(self, results) -> None:
        """Display electrical test results with visualizations."""
        st.subheader("Electrical Safety Test Results")

        # Create metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            if results.insulation_resistance:
                value = results.insulation_resistance.measured_resistance_mohm
                delta = value - results.insulation_resistance.minimum_required_mohm
                st.metric(
                    "Insulation Resistance",
                    f"{value:.1f} MΩ",
                    f"{delta:+.1f} MΩ vs min",
                    delta_color="normal" if delta > 0 else "inverse"
                )

        with col2:
            if results.wet_leakage_current:
                value = results.wet_leakage_current.leakage_current_ua
                delta = results.wet_leakage_current.maximum_allowed_ua - value
                st.metric(
                    "Wet Leakage Current",
                    f"{value:.1f} μA",
                    f"{delta:+.1f} μA margin",
                    delta_color="normal" if delta > 0 else "inverse"
                )

        with col3:
            if results.bypass_diode_thermal:
                value = results.bypass_diode_thermal.peak_temperature_c
                st.metric(
                    "Bypass Diode Peak Temp",
                    f"{value:.1f} °C",
                )

        # Detailed results table
        st.markdown("---")
        test_data = []

        if results.insulation_resistance:
            test_data.append({
                "Test": "Insulation Resistance",
                "Status": results.insulation_resistance.status.value,
                "Result": f"{results.insulation_resistance.measured_resistance_mohm:.1f} MΩ",
                "Requirement": f"≥ {results.insulation_resistance.minimum_required_mohm} MΩ",
                "Pass": "✅" if results.insulation_resistance.passed else "❌"
            })

        if results.wet_leakage_current:
            test_data.append({
                "Test": "Wet Leakage Current",
                "Status": results.wet_leakage_current.status.value,
                "Result": f"{results.wet_leakage_current.leakage_current_ua:.1f} μA",
                "Requirement": f"≤ {results.wet_leakage_current.maximum_allowed_ua} μA",
                "Pass": "✅" if results.wet_leakage_current.passed else "❌"
            })

        if results.dielectric_strength:
            test_data.append({
                "Test": "Dielectric Strength",
                "Status": results.dielectric_strength.status.value,
                "Result": "No breakdown" if not results.dielectric_strength.breakdown_occurred else "Breakdown",
                "Requirement": f"{results.dielectric_strength.test_voltage_v:.0f}V for 60s",
                "Pass": "✅" if results.dielectric_strength.passed else "❌"
            })

        if results.ground_continuity:
            test_data.append({
                "Test": "Ground Continuity",
                "Status": results.ground_continuity.status.value,
                "Result": f"{results.ground_continuity.measured_resistance_ohm:.4f} Ω",
                "Requirement": f"≤ {results.ground_continuity.maximum_allowed_ohm} Ω",
                "Pass": "✅" if results.ground_continuity.passed else "❌"
            })

        if results.bypass_diode_thermal:
            test_data.append({
                "Test": "Bypass Diode Thermal",
                "Status": results.bypass_diode_thermal.status.value,
                "Result": f"{results.bypass_diode_thermal.peak_temperature_c:.1f} °C",
                "Requirement": "No thermal runaway",
                "Pass": "✅" if results.bypass_diode_thermal.passed else "❌"
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

    def _display_mechanical_results(self, results) -> None:
        """Display mechanical test results."""
        st.subheader("Mechanical Safety Test Results")

        test_data = []

        if results.mechanical_load:
            test_data.append({
                "Test": "Mechanical Load",
                "Status": results.mechanical_load.status.value,
                "Load": f"{results.mechanical_load.applied_load_pa} Pa",
                "Max Deflection": f"{results.mechanical_load.maximum_deflection_mm:.1f} mm",
                "Visual Defects": "Yes" if results.mechanical_load.visual_defects_found else "No",
                "Pass": "✅" if results.mechanical_load.passed else "❌"
            })

        if results.impact:
            test_data.append({
                "Test": "Impact Resistance",
                "Status": results.impact.status.value,
                "Impact Velocity": f"{results.impact.impact_velocity_ms:.2f} m/s",
                "Cracks": "Yes" if results.impact.cracks_detected else "No",
                "Safety Maintained": "Yes" if results.impact.electrical_safety_maintained else "No",
                "Pass": "✅" if results.impact.passed else "❌"
            })

        if results.robustness_of_terminations:
            test_data.append({
                "Test": "Robustness of Terminations",
                "Status": results.robustness_of_terminations.status.value,
                "Pull Force": f"{results.robustness_of_terminations.pull_force_n} N",
                "Displaced": "Yes" if results.robustness_of_terminations.cable_displaced else "No",
                "Damaged": "Yes" if results.robustness_of_terminations.terminal_damaged else "No",
                "Pass": "✅" if results.robustness_of_terminations.passed else "❌"
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

    def _display_fire_results(self, results) -> None:
        """Display fire test results."""
        st.subheader("Fire Safety Test Results")

        st.markdown(f"### Fire Classification: **{results.fire_classification.value}**")

        test_data = []

        if results.spread_of_flame:
            test_data.append({
                "Test": "Spread of Flame",
                "Status": results.spread_of_flame.status.value,
                "Flame Spread": f"{results.spread_of_flame.flame_spread_distance_cm:.1f} cm",
                "Sustained Flaming": "Yes" if results.spread_of_flame.sustained_flaming_observed else "No",
                "Roof Penetration": "Yes" if results.spread_of_flame.roof_deck_penetration else "No",
            })

        if results.fire_penetration:
            test_data.append({
                "Test": "Fire Penetration",
                "Status": results.fire_penetration.status.value,
                "Burn Through": "Yes" if results.fire_penetration.burn_through_occurred else "No",
                "Deck Damage": "Yes" if results.fire_penetration.roof_deck_damage else "No",
                "Duration": f"{results.fire_penetration.test_duration_min} min",
            })

        if results.fire_brand:
            test_data.append({
                "Test": "Fire Brand",
                "Status": results.fire_brand.status.value,
                "Brand Class": results.fire_brand.brand_size_class,
                "Ignition": "Yes" if results.fire_brand.ignition_occurred else "No",
                "Sustained Burning": "Yes" if results.fire_brand.sustained_burning else "No",
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

    def _display_environmental_results(self, results) -> None:
        """Display environmental test results."""
        st.subheader("Environmental Safety Test Results")

        test_data = []

        if results.uv_preconditioning:
            test_data.append({
                "Test": "UV Preconditioning",
                "Status": results.uv_preconditioning.status.value,
                "UV Dose": f"{results.uv_preconditioning.uv_dose_kwh_m2:.1f} kWh/m²",
                "Required": f"{results.uv_preconditioning.required_dose_kwh_m2} kWh/m²",
                "Degradation": "Yes" if results.uv_preconditioning.visual_degradation else "No",
                "Pass": "✅" if results.uv_preconditioning.passed else "❌"
            })

        if results.thermal_cycling:
            test_data.append({
                "Test": "Thermal Cycling",
                "Status": results.thermal_cycling.status.value,
                "Cycles": f"{results.thermal_cycling.cycles_completed}/{results.thermal_cycling.required_cycles}",
                "Range": f"{results.thermal_cycling.min_temperature_c}°C to {results.thermal_cycling.max_temperature_c}°C",
                "Electrical Failure": "Yes" if results.thermal_cycling.electrical_failure else "No",
                "Pass": "✅" if results.thermal_cycling.passed else "❌"
            })

        if results.humidity_freeze:
            test_data.append({
                "Test": "Humidity-Freeze",
                "Status": results.humidity_freeze.status.value,
                "Cycles": f"{results.humidity_freeze.cycles_completed}/{results.humidity_freeze.required_cycles}",
                "Range": f"{results.humidity_freeze.freeze_phase_c}°C to {results.humidity_freeze.humidity_phase_c}°C/{results.humidity_freeze.humidity_phase_rh}%RH",
                "Electrical Failure": "Yes" if results.humidity_freeze.electrical_failure else "No",
                "Pass": "✅" if results.humidity_freeze.passed else "❌"
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

    def _display_construction_results(self, requirements) -> None:
        """Display construction requirements results."""
        st.subheader("Construction Requirements (IEC 61730-1)")

        test_data = []

        for req in requirements:
            test_data.append({
                "Requirement ID": req.requirement_id,
                "Description": req.requirement_description,
                "Compliant": "✅" if req.compliant else "❌",
                "Notes": req.notes or "-"
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

        # Compliance summary
        compliant_count = sum(1 for req in requirements if req.compliant)
        total_count = len(requirements)
        compliance_rate = (compliant_count / total_count) * 100 if total_count > 0 else 0

        st.metric(
            "Compliance Rate",
            f"{compliance_rate:.1f}%",
            f"{compliant_count}/{total_count} requirements"
        )


def main():
    """Main entry point for the Streamlit app."""
    ui = SafetyTestUI()
    ui.render()


if __name__ == "__main__":
    main()
