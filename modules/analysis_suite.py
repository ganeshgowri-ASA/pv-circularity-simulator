"""
Analysis Suite Module - Branches B04-B06

This module provides functionality for:
- B04: IEC Testing Standards (IEC 61215, 61730, 62804, 61853)
- B05: System Design & Optimization
- B06: Weather Data & Energy Yield Assessment

Author: PV Circularity Simulator Team
Version: 1.0 (71 Sessions Integrated)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    IEC_STANDARDS,
    INVERTER_TYPES,
    MOUNTING_TYPES,
    WEATHER_PRESETS,
    STC_CONDITIONS,
    COLOR_PALETTE,
)
from utils.validators import (
    IECTest,
    SystemDesign,
    WeatherData,
    GeoLocation,
)


# ============================================================================
# BRANCH 04: IEC TESTING STANDARDS
# ============================================================================

def render_iec_testing() -> None:
    """
    Render the IEC Testing Standards interface.

    Features:
    - IEC 61215 (Crystalline Silicon Testing)
    - IEC 61730 (Safety Qualification)
    - IEC 62804 (PID Testing)
    - IEC 61853 (Performance Testing)
    - Test tracking and results
    - Compliance reporting
    """
    st.header("🔬 IEC Testing Standards & Compliance")
    st.markdown("*Comprehensive PV module testing per international standards*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Test Overview",
        "🧪 Active Tests",
        "📊 Results & Reports",
        "✅ Compliance Matrix"
    ])

    # Tab 1: Test Overview
    with tab1:
        st.subheader("IEC Testing Standards Overview")

        # Display all standards
        for standard_id, standard_info in IEC_STANDARDS.items():
            with st.expander(f"**{standard_id}: {standard_info['name']}**"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Test Sequence:**")
                    for i, test in enumerate(standard_info['tests'], 1):
                        st.markdown(f"{i}. {test}")

                with col2:
                    st.metric("Total Tests", len(standard_info['tests']))
                    st.metric("Duration", f"{standard_info['duration_hours']} hours")
                    st.metric("Est. Days", f"{standard_info['duration_hours'] / 24:.1f}")

        # Summary metrics
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_tests = sum(len(s['tests']) for s in IEC_STANDARDS.values())
            st.metric("Total Test Types", total_tests)
        with col2:
            total_hours = sum(s['duration_hours'] for s in IEC_STANDARDS.values())
            st.metric("Total Test Hours", total_hours)
        with col3:
            st.metric("Standards Covered", len(IEC_STANDARDS))
        with col4:
            st.metric("Completion Time", f"{total_hours / 24:.0f} days")

    # Tab 2: Active Tests
    with tab2:
        st.subheader("Configure and Run Tests")

        # Select standard
        selected_standard = st.selectbox(
            "Select IEC Standard:",
            list(IEC_STANDARDS.keys()),
            format_func=lambda x: f"{x}: {IEC_STANDARDS[x]['name']}"
        )

        standard_info = IEC_STANDARDS[selected_standard]

        # Select specific tests
        st.markdown("**Select Tests to Perform:**")
        selected_tests = []

        # Group tests in columns
        tests = standard_info['tests']
        num_cols = 3
        cols = st.columns(num_cols)

        for i, test_name in enumerate(tests):
            with cols[i % num_cols]:
                if st.checkbox(test_name, key=f"test_{i}"):
                    selected_tests.append(test_name)

        if selected_tests:
            st.success(f"✓ {len(selected_tests)} tests selected")

            # Test parameters
            with st.form("test_configuration"):
                st.markdown("**Test Configuration**")

                col1, col2 = st.columns(2)
                with col1:
                    module_id = st.text_input("Module ID", "MOD-2025-001")
                    manufacturer = st.text_input("Manufacturer", "Test Manufacturer")
                    test_date = st.date_input("Test Date", datetime.now())

                with col2:
                    test_lab = st.text_input("Testing Laboratory", "Certified Lab")
                    technician = st.text_input("Technician", "John Doe")
                    ambient_conditions = st.text_area("Ambient Conditions", "25°C, 45% RH")

                submitted = st.form_submit_button("🚀 Start Test Sequence", type="primary")

                if submitted:
                    st.success(f"✓ Test sequence initiated for {module_id}")

                    # Create test records
                    test_records = []
                    for test_name in selected_tests:
                        test_record = IECTest(
                            standard=selected_standard,
                            test_name=test_name,
                            status="In Progress",
                            pass_criteria="Per IEC specification",
                            test_date=datetime.combine(test_date, datetime.min.time()),
                            duration_hours=standard_info['duration_hours'] / len(tests),
                            result="In Progress"
                        )
                        test_records.append(test_record)

                    # Display test queue
                    st.markdown("**Test Queue:**")
                    test_df = pd.DataFrame([
                        {
                            'Test Name': t.test_name,
                            'Standard': t.standard,
                            'Status': t.status,
                            'Duration (hrs)': f"{t.duration_hours:.1f}"
                        }
                        for t in test_records
                    ])
                    st.dataframe(test_df, use_container_width=True)

        # Simulate ongoing tests
        if st.checkbox("Show Simulated Test Progress"):
            st.markdown("**Current Test Progress**")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate some progress (in real app, this would be actual test data)
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f"Running: Thermal Cycling Test - Cycle {i}/200")
                if i >= 100:
                    break

            st.success("✓ Test sequence completed!")

    # Tab 3: Results & Reports
    with tab3:
        st.subheader("Test Results & Analysis")

        # Generate sample test results
        test_results = generate_sample_test_results()

        # Results table
        st.markdown("**Recent Test Results:**")
        results_df = pd.DataFrame(test_results)

        # Color code results
        def color_result(val):
            if val == "Pass":
                return f'background-color: {COLOR_PALETTE["success"]}40'
            elif val == "Fail":
                return f'background-color: {COLOR_PALETTE["danger"]}40'
            else:
                return f'background-color: {COLOR_PALETTE["warning"]}40'

        styled_df = results_df.style.applymap(color_result, subset=['Result'])
        st.dataframe(styled_df, use_container_width=True)

        # Pass/Fail statistics
        col1, col2, col3, col4 = st.columns(4)
        passed = len([r for r in test_results if r['Result'] == 'Pass'])
        failed = len([r for r in test_results if r['Result'] == 'Fail'])
        in_progress = len([r for r in test_results if r['Result'] == 'In Progress'])

        with col1:
            st.metric("Tests Passed", passed, delta=f"{passed/(passed+failed)*100:.0f}%")
        with col2:
            st.metric("Tests Failed", failed, delta=f"-{failed/(passed+failed)*100:.0f}%", delta_color="inverse")
        with col3:
            st.metric("In Progress", in_progress)
        with col4:
            st.metric("Total Tests", len(test_results))

        # Results visualization
        fig = go.Figure(data=[
            go.Bar(name='Pass', x=['IEC 61215', 'IEC 61730', 'IEC 62804'], y=[14, 5, 2], marker_color=COLOR_PALETTE['success']),
            go.Bar(name='Fail', x=['IEC 61215', 'IEC 61730', 'IEC 62804'], y=[1, 0, 0], marker_color=COLOR_PALETTE['danger']),
            go.Bar(name='In Progress', x=['IEC 61215', 'IEC 61730', 'IEC 62804'], y=[2, 0, 0], marker_color=COLOR_PALETTE['warning']),
        ])
        fig.update_layout(
            barmode='stack',
            title="Test Results by Standard",
            xaxis_title="IEC Standard",
            yaxis_title="Number of Tests",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export options
        if st.button("📄 Generate PDF Test Report"):
            st.info("PDF report generation would be implemented here")
            st.download_button(
                label="Download Test Report",
                data="Sample test report data",
                file_name="iec_test_report.pdf",
                mime="application/pdf"
            )

    # Tab 4: Compliance Matrix
    with tab4:
        st.subheader("Compliance Matrix")

        # Create compliance matrix
        compliance_data = {
            'Standard': [],
            'Required Tests': [],
            'Completed': [],
            'Passed': [],
            'Compliance %': [],
            'Status': []
        }

        for standard_id, standard_info in IEC_STANDARDS.items():
            total_tests = len(standard_info['tests'])
            completed = int(total_tests * 0.85)  # Simulate 85% completion
            passed = int(completed * 0.95)  # Simulate 95% pass rate

            compliance_data['Standard'].append(standard_id)
            compliance_data['Required Tests'].append(total_tests)
            compliance_data['Completed'].append(completed)
            compliance_data['Passed'].append(passed)
            compliance_data['Compliance %'].append(f"{passed/total_tests*100:.1f}%")
            compliance_data['Status'].append("✓ Compliant" if passed/total_tests >= 0.8 else "⚠ Pending")

        compliance_df = pd.DataFrame(compliance_data)
        st.dataframe(compliance_df, use_container_width=True)

        # Compliance visualization
        fig_compliance = go.Figure()

        fig_compliance.add_trace(go.Scatterpolar(
            r=[85, 100, 90, 80],
            theta=['IEC 61215', 'IEC 61730', 'IEC 62804', 'IEC 61853'],
            fill='toself',
            name='Compliance %'
        ))

        fig_compliance.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Overall Compliance Status"
        )
        st.plotly_chart(fig_compliance, use_container_width=True)


def generate_sample_test_results() -> List[Dict[str, Any]]:
    """
    Generate sample test results for demonstration.

    Returns:
        List of test result dictionaries
    """
    tests = [
        {"Test": "Visual Inspection", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-15"},
        {"Test": "Maximum Power", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-15"},
        {"Test": "Insulation Test", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-15"},
        {"Test": "Thermal Cycling", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-16"},
        {"Test": "Humidity-Freeze", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-16"},
        {"Test": "Damp Heat", "Standard": "IEC 61215", "Result": "In Progress", "Date": "2025-11-17"},
        {"Test": "Hot-Spot Endurance", "Standard": "IEC 61215", "Result": "Pass", "Date": "2025-11-14"},
        {"Test": "PID Stress Test", "Standard": "IEC 62804", "Result": "Pass", "Date": "2025-11-13"},
        {"Test": "Safety Qualification", "Standard": "IEC 61730", "Result": "Pass", "Date": "2025-11-12"},
    ]
    return tests


# ============================================================================
# BRANCH 05: SYSTEM DESIGN & OPTIMIZATION
# ============================================================================

def render_system_design() -> None:
    """
    Render the System Design & Optimization interface.

    Features:
    - String configuration
    - Inverter selection and sizing
    - DC/AC ratio optimization
    - Mounting structure design
    - System layout
    - Electrical balance of system
    """
    st.header("⚡ System Design & Optimization")
    st.markdown("*Complete PV system design from modules to grid*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ System Configuration",
        "🔌 Inverter Selection",
        "📐 Layout Design",
        "💰 System Economics"
    ])

    # Tab 1: System Configuration
    with tab1:
        st.subheader("System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Module Specifications**")
            module_power_w = st.number_input("Module Power (W)", 200, 700, 450)
            module_voc = st.number_input("Module Voc (V)", 30.0, 60.0, 49.5, 0.1)
            module_isc = st.number_input("Module Isc (A)", 5.0, 15.0, 11.5, 0.1)
            module_vmpp = st.number_input("Module Vmpp (V)", 25.0, 50.0, 41.2, 0.1)
            module_impp = st.number_input("Module Impp (A)", 5.0, 15.0, 10.9, 0.1)

        with col2:
            st.markdown("**System Sizing**")
            system_capacity_kw = st.number_input("Target System Capacity (kW)", 1.0, 10000.0, 100.0, 1.0)
            num_modules = int(system_capacity_kw * 1000 / module_power_w)

            st.metric("Calculated Modules", num_modules)

            num_strings = st.number_input("Number of Strings", 1, 100, 10, 1)
            modules_per_string = int(num_modules / num_strings)

            st.metric("Modules per String", modules_per_string)

        # String voltage calculations
        string_voc = modules_per_string * module_voc
        string_vmpp = modules_per_string * module_vmpp

        st.divider()
        st.subheader("String Electrical Characteristics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("String Voc", f"{string_voc:.1f} V")
        with col2:
            st.metric("String Vmpp", f"{string_vmpp:.1f} V")
        with col3:
            st.metric("String Isc", f"{module_isc:.1f} A")
        with col4:
            st.metric("String Power", f"{modules_per_string * module_power_w / 1000:.1f} kW")

        # Mounting type selection
        st.markdown("**Mounting Configuration**")
        col1, col2, col3 = st.columns(3)

        with col1:
            mounting_type = st.selectbox("Mounting Type", list(MOUNTING_TYPES.keys()))
            tilt_angle = st.slider("Tilt Angle (°)", 0, 90, 25)

        with col2:
            azimuth = st.slider("Azimuth (°)", 0, 360, 180)
            row_spacing = st.number_input("Row Spacing (m)", 0.5, 10.0, 3.0, 0.1)

        with col3:
            mounting_info = MOUNTING_TYPES[mounting_type]
            st.metric("Mounting Cost", f"${mounting_info['cost_per_kw'] * system_capacity_kw:,.0f}")
            st.metric("Energy Gain", f"{mounting_info['energy_gain']:+.0f}%")
            st.metric("Maintenance", mounting_info['maintenance'])

        # Validate system design
        try:
            system = SystemDesign(
                capacity_kw=system_capacity_kw,
                num_modules=num_modules,
                module_power_w=module_power_w,
                num_strings=num_strings,
                modules_per_string=modules_per_string,
                inverter_type="String",  # Default
                inverter_capacity_kw=system_capacity_kw * 1.1,
                mounting_type=mounting_type,
                tilt_angle=tilt_angle,
                azimuth=azimuth,
                location="Test Location"
            )
            st.success("✓ System design validated")

        except Exception as e:
            st.error(f"Validation error: {str(e)}")

    # Tab 2: Inverter Selection
    with tab2:
        st.subheader("Inverter Selection & Sizing")

        # Display inverter options
        selected_inverter = st.selectbox(
            "Select Inverter Type:",
            list(INVERTER_TYPES.keys())
        )

        inverter_info = INVERTER_TYPES[selected_inverter]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Inverter Specifications**")
            st.write(f"**Efficiency:** {inverter_info['efficiency']}%")
            st.write(f"**Cost:** ${inverter_info['cost_per_kw']}/kW")
            st.write(f"**Lifespan:** {inverter_info['lifespan_years']} years")
            st.write(f"**Power Range:** {inverter_info['power_range'][0]}-{inverter_info['power_range'][1]} kW")
            st.write(f"**Max DC Voltage:** {inverter_info['max_dc_voltage']} V")
            st.write(f"**MPPT Efficiency:** {inverter_info['mppt_efficiency']}%")

        with col2:
            st.markdown("**Sizing Calculator**")
            dc_ac_ratio = st.slider("DC/AC Ratio", 1.0, 1.5, 1.2, 0.05)

            inverter_capacity_kw = system_capacity_kw / dc_ac_ratio
            num_inverters = int(np.ceil(inverter_capacity_kw / inverter_info['power_range'][1]))

            st.metric("Inverter Capacity", f"{inverter_capacity_kw:.1f} kW")
            st.metric("Number of Inverters", num_inverters)
            st.metric("DC/AC Ratio", f"{dc_ac_ratio:.2f}")

            total_inverter_cost = inverter_capacity_kw * inverter_info['cost_per_kw']
            st.metric("Total Inverter Cost", f"${total_inverter_cost:,.0f}")

        # Voltage check
        st.divider()
        st.markdown("**Voltage Compatibility Check**")

        max_string_voltage = string_voc * 1.15  # Safety factor for cold conditions

        if max_string_voltage <= inverter_info['max_dc_voltage']:
            st.success(f"✓ String voltage ({max_string_voltage:.0f}V) is within inverter limit ({inverter_info['max_dc_voltage']}V)")
        else:
            st.error(f"⚠ String voltage ({max_string_voltage:.0f}V) exceeds inverter limit ({inverter_info['max_dc_voltage']}V)")
            st.warning("Recommendation: Reduce modules per string or use inverter with higher voltage rating")

        # Efficiency curve
        st.subheader("Inverter Efficiency Curve")
        load_pct = np.linspace(0, 110, 50)
        efficiency = inverter_info['efficiency'] * (0.88 + 0.12 * np.exp(-(load_pct - 50)**2 / 500))

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(
            x=load_pct,
            y=efficiency,
            mode='lines',
            line=dict(color=COLOR_PALETTE['primary'], width=3),
            fill='tozeroy'
        ))
        fig_eff.update_layout(
            title=f"{selected_inverter} Inverter Efficiency vs Load",
            xaxis_title="Load (%)",
            yaxis_title="Efficiency (%)",
            height=400
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    # Tab 3: Layout Design
    with tab3:
        st.subheader("System Layout Design")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Site Parameters**")
            site_length = st.number_input("Site Length (m)", 10, 1000, 100)
            site_width = st.number_input("Site Width (m)", 10, 1000, 80)
            site_area_m2 = site_length * site_width

            st.metric("Site Area", f"{site_area_m2:,.0f} m²")

            # Calculate module area
            module_length = 1.65  # m (typical)
            module_width = 0.99  # m (typical)
            module_area = module_length * module_width

            total_module_area = num_modules * module_area
            gcr = total_module_area / site_area_m2  # Ground Coverage Ratio

            st.metric("Module Area", f"{total_module_area:,.0f} m²")
            st.metric("GCR", f"{gcr:.2%}")

        with col2:
            st.markdown("**Layout Metrics**")
            strings_per_row = st.number_input("Strings per Row", 1, 20, 5)
            num_rows = int(np.ceil(num_strings / strings_per_row))

            st.metric("Number of Rows", num_rows)

            total_length_needed = num_rows * (module_length * np.cos(np.radians(tilt_angle)) + row_spacing)
            total_width_needed = strings_per_row * module_width * modules_per_string

            if total_length_needed <= site_length and total_width_needed <= site_width:
                st.success("✓ Layout fits within site boundaries")
            else:
                st.error("⚠ Layout exceeds site boundaries")

            st.metric("Required Length", f"{total_length_needed:.1f} m")
            st.metric("Required Width", f"{total_width_needed:.1f} m")

        # Visualize layout (simplified)
        st.subheader("Layout Visualization")

        fig_layout = go.Figure()

        # Draw site boundary
        fig_layout.add_trace(go.Scatter(
            x=[0, site_length, site_length, 0, 0],
            y=[0, 0, site_width, site_width, 0],
            mode='lines',
            name='Site Boundary',
            line=dict(color='black', width=2)
        ))

        # Draw module rows (simplified)
        current_y = 5
        for row in range(min(num_rows, 10)):  # Limit visualization to 10 rows
            row_length = modules_per_string * module_length
            fig_layout.add_trace(go.Scatter(
                x=[5, 5 + row_length],
                y=[current_y, current_y],
                mode='lines',
                name=f'Row {row + 1}',
                line=dict(width=10)
            ))
            current_y += row_spacing

        fig_layout.update_layout(
            title="System Layout (Top View)",
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            height=500,
            showlegend=False,
            xaxis=dict(scaleanchor="y", scaleratio=1)
        )
        st.plotly_chart(fig_layout, use_container_width=True)

    # Tab 4: System Economics
    with tab4:
        st.subheader("System Economics")

        # Cost breakdown
        module_cost = num_modules * module_power_w * 0.35  # $/W
        inverter_cost = inverter_capacity_kw * inverter_info['cost_per_kw']
        mounting_cost = system_capacity_kw * mounting_info['cost_per_kw']
        bos_cost = system_capacity_kw * 200  # Balance of system
        installation_cost = (module_cost + inverter_cost + mounting_cost + bos_cost) * 0.15

        total_capex = module_cost + inverter_cost + mounting_cost + bos_cost + installation_cost

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Modules", f"${module_cost:,.0f}")
        with col2:
            st.metric("Inverters", f"${inverter_cost:,.0f}")
        with col3:
            st.metric("Mounting", f"${mounting_cost:,.0f}")
        with col4:
            st.metric("BOS", f"${bos_cost:,.0f}")
        with col5:
            st.metric("Installation", f"${installation_cost:,.0f}")

        st.metric("**Total CAPEX**", f"${total_capex:,.0f}", delta=f"${total_capex/system_capacity_kw:,.0f}/kW")

        # Cost breakdown pie chart
        fig_cost = go.Figure(data=[go.Pie(
            labels=['Modules', 'Inverters', 'Mounting', 'BOS', 'Installation'],
            values=[module_cost, inverter_cost, mounting_cost, bos_cost, installation_cost],
            hole=.3
        )])
        fig_cost.update_layout(title="CAPEX Breakdown", height=400)
        st.plotly_chart(fig_cost, use_container_width=True)


# ============================================================================
# BRANCH 06: WEATHER DATA & ENERGY YIELD ASSESSMENT
# ============================================================================

def render_weather_analysis() -> None:
    """
    Render the Weather Data & Energy Yield Assessment interface.

    Features:
    - Climate zone selection
    - TMY data integration
    - GHI/DNI/DHI analysis
    - Temperature impact
    - Soiling modeling
    - Energy yield prediction
    """
    st.header("🌤️ Weather Data & Energy Yield Assessment")
    st.markdown("*Climate analysis and energy production forecasting*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Location & Climate",
        "☀️ Irradiance Analysis",
        "📊 Energy Yield",
        "🎯 P50/P90 Analysis"
    ])

    # Tab 1: Location & Climate
    with tab1:
        st.subheader("Location Selection & Climate Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Geographic Location**")
            latitude = st.number_input("Latitude", -90.0, 90.0, 28.6, 0.1)
            longitude = st.number_input("Longitude", -180.0, 180.0, 77.2, 0.1)
            altitude = st.number_input("Altitude (m)", 0, 5000, 216)

            location = GeoLocation(
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude
            )

            st.success(f"✓ Location: {latitude:.2f}°N, {longitude:.2f}°E")

        with col2:
            st.markdown("**Climate Preset**")
            climate_preset = st.selectbox("Select Climate Type", list(WEATHER_PRESETS.keys()))

            if st.button("Apply Climate Preset"):
                st.session_state.weather_preset = WEATHER_PRESETS[climate_preset]
                st.success(f"✓ Applied {climate_preset} climate preset")

        # Display/edit weather parameters
        st.divider()
        st.subheader("Annual Climate Parameters")

        if 'weather_preset' not in st.session_state:
            st.session_state.weather_preset = WEATHER_PRESETS[climate_preset]

        weather_data = st.session_state.weather_preset

        col1, col2, col3 = st.columns(3)

        with col1:
            annual_ghi = st.number_input(
                "Annual GHI (kWh/m²)",
                800, 3000,
                int(weather_data['annual_ghi'])
            )
            avg_temp = st.number_input(
                "Avg Temperature (°C)",
                -20, 50,
                int(weather_data['avg_temp'])
            )

        with col2:
            humidity = st.slider(
                "Relative Humidity (%)",
                0, 100,
                int(weather_data['humidity'])
            )
            wind_speed = st.number_input(
                "Avg Wind Speed (m/s)",
                0.0, 20.0,
                weather_data['wind_speed']
            )

        with col3:
            soiling_rate = st.number_input(
                "Soiling Rate (%/day)",
                0.0, 1.0,
                weather_data['soiling_rate'],
                0.01
            )
            rainfall = st.number_input(
                "Annual Rainfall (mm)",
                0, 5000,
                int(weather_data['rainfall_mm'])
            )

        # Validate weather data
        try:
            weather = WeatherData(
                location=f"{latitude:.2f}°N, {longitude:.2f}°E",
                annual_ghi_kwh_m2=annual_ghi,
                avg_temp_c=avg_temp,
                humidity_pct=humidity,
                wind_speed_ms=wind_speed,
                soiling_rate_pct=soiling_rate,
                rainfall_mm=rainfall
            )
            st.success("✓ Weather data validated")

        except Exception as e:
            st.error(f"Validation error: {str(e)}")

    # Tab 2: Irradiance Analysis
    with tab2:
        st.subheader("Solar Irradiance Analysis")

        # Generate synthetic annual irradiance profile
        days = np.arange(1, 366)
        base_ghi = annual_ghi / 365
        seasonal_variation = 0.3 * np.sin(2 * np.pi * (days - 80) / 365)
        daily_ghi = base_ghi * (1 + seasonal_variation) + np.random.normal(0, base_ghi * 0.1, 365)

        # Monthly aggregation
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        monthly_ghi = []
        day_counter = 0

        for days_in_month in days_per_month:
            month_data = daily_ghi[day_counter:day_counter + days_in_month]
            monthly_ghi.append(month_data.sum())
            day_counter += days_in_month

        # Plot monthly irradiance
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=months,
            y=monthly_ghi,
            marker_color=COLOR_PALETTE['warning'],
            name='GHI'
        ))
        fig_monthly.update_layout(
            title="Monthly Global Horizontal Irradiance",
            xaxis_title="Month",
            yaxis_title="GHI (kWh/m²)",
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Daily profile
        st.subheader("Typical Daily Irradiance Profile")

        hours = np.arange(0, 24, 0.5)
        hourly_irradiance = 1000 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 1.5
        hourly_irradiance[hours < 6] = 0
        hourly_irradiance[hours > 18] = 0

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=hours,
            y=hourly_irradiance,
            mode='lines',
            fill='tozeroy',
            line=dict(color=COLOR_PALETTE['warning'], width=3)
        ))
        fig_daily.update_layout(
            title="Typical Clear Sky Daily Irradiance",
            xaxis_title="Hour of Day",
            yaxis_title="Irradiance (W/m²)",
            height=400
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Peak Month", months[np.argmax(monthly_ghi)])
        with col2:
            st.metric("Max Monthly GHI", f"{max(monthly_ghi):.0f} kWh/m²")
        with col3:
            st.metric("Min Monthly GHI", f"{min(monthly_ghi):.0f} kWh/m²")
        with col4:
            st.metric("Avg Daily GHI", f"{annual_ghi/365:.1f} kWh/m²")

    # Tab 3: Energy Yield
    with tab3:
        st.subheader("Annual Energy Yield Estimation")

        # System parameters for yield calculation
        col1, col2 = st.columns(2)

        with col1:
            system_capacity = st.number_input("System Capacity (kWp)", 1, 10000, 100)
            system_pr = st.slider("Performance Ratio", 0.60, 0.95, 0.80, 0.01)
            degradation_rate = st.slider("Annual Degradation (%)", 0.0, 2.0, 0.5, 0.1)

        with col2:
            availability = st.slider("System Availability", 0.90, 1.00, 0.98, 0.01)
            grid_availability = st.slider("Grid Availability", 0.90, 1.00, 0.99, 0.01)
            soiling_loss = st.slider("Avg Soiling Loss (%)", 0.0, 10.0, 2.0, 0.5)

        # Calculate annual yield
        poa_irradiance = annual_ghi * 1.05  # Tilt factor
        annual_yield_year1 = (system_capacity * poa_irradiance * system_pr *
                              availability * grid_availability * (1 - soiling_loss / 100))

        # 25-year projection
        years = np.arange(1, 26)
        annual_yields = annual_yield_year1 * (1 - degradation_rate / 100) ** (years - 1)
        cumulative_yield = np.cumsum(annual_yields)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Year 1 Yield", f"{annual_yield_year1:,.0f} kWh")
        with col2:
            st.metric("Year 25 Yield", f"{annual_yields[-1]:,.0f} kWh")
        with col3:
            st.metric("25-Year Total", f"{cumulative_yield[-1]:,.0f} kWh")
        with col4:
            specific_yield = annual_yield_year1 / system_capacity
            st.metric("Specific Yield", f"{specific_yield:.0f} kWh/kWp")

        # Annual yield projection
        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(
            x=years,
            y=annual_yields / 1000,
            mode='lines+markers',
            name='Annual Yield',
            line=dict(color=COLOR_PALETTE['primary'], width=3)
        ))
        fig_yield.update_layout(
            title="25-Year Energy Yield Projection",
            xaxis_title="Year",
            yaxis_title="Annual Yield (MWh)",
            height=400
        )
        st.plotly_chart(fig_yield, use_container_width=True)

        # Monthly yield distribution
        monthly_yields = np.array(monthly_ghi) / annual_ghi * annual_yield_year1

        fig_monthly_yield = go.Figure()
        fig_monthly_yield.add_trace(go.Bar(
            x=months,
            y=monthly_yields,
            marker_color=COLOR_PALETTE['success']
        ))
        fig_monthly_yield.update_layout(
            title="Monthly Energy Production (Year 1)",
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            height=400
        )
        st.plotly_chart(fig_monthly_yield, use_container_width=True)

    # Tab 4: P50/P90 Analysis
    with tab4:
        st.subheader("Probabilistic Energy Assessment (P50/P90)")

        st.markdown("""
        **P-values represent exceedance probability:**
        - **P50**: 50% probability that actual yield will exceed this value (median)
        - **P75**: 75% probability of exceedance
        - **P90**: 90% probability of exceedance (conservative for financing)
        """)

        # Uncertainty factors
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Uncertainty Factors (%)**")
            resource_uncertainty = st.slider("Resource Data", 0.0, 15.0, 5.0, 0.5)
            model_uncertainty = st.slider("Simulation Model", 0.0, 10.0, 3.0, 0.5)
            performance_uncertainty = st.slider("System Performance", 0.0, 10.0, 4.0, 0.5)

        with col2:
            st.markdown("**Inter-annual Variability (%)**")
            interannual_variability = st.slider("Year-to-Year Variation", 0.0, 15.0, 6.0, 0.5)

        # Calculate combined uncertainty
        total_uncertainty = np.sqrt(
            resource_uncertainty**2 +
            model_uncertainty**2 +
            performance_uncertainty**2 +
            interannual_variability**2
        )

        # Calculate P-values (simplified approach)
        p50_yield = annual_yield_year1
        p75_yield = p50_yield * (1 - 0.674 * total_uncertainty / 100)  # 0.674 is z-score for P75
        p90_yield = p50_yield * (1 - 1.282 * total_uncertainty / 100)  # 1.282 is z-score for P90
        p99_yield = p50_yield * (1 - 2.326 * total_uncertainty / 100)  # 2.326 is z-score for P99

        st.metric("**Total Uncertainty**", f"{total_uncertainty:.1f}%")

        # Display P-values
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("P50 Yield", f"{p50_yield:,.0f} kWh")
        with col2:
            st.metric("P75 Yield", f"{p75_yield:,.0f} kWh", delta=f"{(p75_yield/p50_yield-1)*100:.1f}%")
        with col3:
            st.metric("P90 Yield", f"{p90_yield:,.0f} kWh", delta=f"{(p90_yield/p50_yield-1)*100:.1f}%")
        with col4:
            st.metric("P99 Yield", f"{p99_yield:,.0f} kWh", delta=f"{(p99_yield/p50_yield-1)*100:.1f}%")

        # Probability distribution curve
        yields = np.linspace(p99_yield * 0.9, p50_yield * 1.2, 1000)
        probability = 100 * (1 - 0.5 * (1 + np.erf((yields - p50_yield) / (total_uncertainty * p50_yield / 100 * np.sqrt(2)))))

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=yields / 1000,
            y=probability,
            mode='lines',
            fill='tozeroy',
            line=dict(color=COLOR_PALETTE['primary'], width=3)
        ))

        # Add P-value markers
        for p_val, yield_val, label in [(50, p50_yield, 'P50'), (75, p75_yield, 'P75'), (90, p90_yield, 'P90')]:
            fig_prob.add_vline(x=yield_val / 1000, line_dash="dash", annotation_text=label)

        fig_prob.update_layout(
            title="Exceedance Probability Curve",
            xaxis_title="Annual Energy Yield (MWh)",
            yaxis_title="Exceedance Probability (%)",
            height=500
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Uncertainty breakdown
        fig_uncertainty = go.Figure(data=[go.Pie(
            labels=['Resource Data', 'Model', 'Performance', 'Inter-annual'],
            values=[resource_uncertainty**2, model_uncertainty**2,
                   performance_uncertainty**2, interannual_variability**2],
            hole=.3
        )])
        fig_uncertainty.update_layout(
            title="Uncertainty Contribution Breakdown (Variance)",
            height=400
        )
        st.plotly_chart(fig_uncertainty, use_container_width=True)
