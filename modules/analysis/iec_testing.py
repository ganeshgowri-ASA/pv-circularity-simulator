"""
IEC Standards Testing Module (Branch B04).

Features:
- IEC 61215: Design qualification and type approval
- IEC 61730: Safety qualification
- IEC 63202: Light-induced degradation measurement
- IEC 63209: Extended-stress testing
- IEC TS 63279: Thermal characteristics modeling
- Test result tracking and certification
- Pass/fail criteria evaluation
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import IEC_STANDARDS
from utils.validators import IECTestResult
from utils.helpers import create_heatmap, create_comparison_bar_chart


class IECTestingSimulator:
    """IEC standards testing and certification."""

    def __init__(self):
        """Initialize IEC testing simulator."""
        self.standards = IEC_STANDARDS
        self.test_database = []

    def run_iec_61215_test(
        self,
        module_id: str,
        module_power: float,
        test_conditions: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Run IEC 61215 design qualification tests.

        Args:
            module_id: Module identifier
            module_power: Module rated power (W)
            test_conditions: Test condition parameters

        Returns:
            Test results dictionary
        """
        results = {
            "test_id": f"IEC61215_{module_id}_{datetime.now().strftime('%Y%m%d')}",
            "standard": "IEC_61215",
            "module_id": module_id,
            "test_date": datetime.now(),
            "tests": {}
        }

        # Visual Inspection
        results["tests"]["visual_inspection"] = {
            "name": "Visual Inspection",
            "passed": True,
            "notes": "No visible defects, cracks, or delamination"
        }

        # Maximum Power Determination
        measured_power = module_power * np.random.normal(1.0, 0.02)
        power_tolerance = 0.03  # ±3%
        results["tests"]["max_power"] = {
            "name": "Maximum Power Determination",
            "measured": measured_power,
            "rated": module_power,
            "deviation": ((measured_power - module_power) / module_power) * 100,
            "tolerance": power_tolerance * 100,
            "passed": abs(measured_power - module_power) <= module_power * power_tolerance
        }

        # Insulation Test (1000V + 2×Voc)
        insulation_resistance = np.random.uniform(100, 500)  # MΩ
        results["tests"]["insulation"] = {
            "name": "Insulation Test",
            "measured": insulation_resistance,
            "minimum": 40,  # MΩ
            "passed": insulation_resistance >= 40
        }

        # Temperature Coefficients
        temp_coeff_pmax = np.random.normal(-0.40, 0.05)
        temp_coeff_voc = np.random.normal(-0.30, 0.03)
        temp_coeff_isc = np.random.normal(0.05, 0.01)

        results["tests"]["temperature_coefficients"] = {
            "name": "Temperature Coefficients",
            "pmax": temp_coeff_pmax,
            "voc": temp_coeff_voc,
            "isc": temp_coeff_isc,
            "passed": True
        }

        # NOCT Measurement
        noct = np.random.normal(45, 2)
        results["tests"]["noct"] = {
            "name": "NOCT Measurement",
            "measured": noct,
            "typical_range": (42, 48),
            "passed": 40 <= noct <= 50
        }

        # Low Irradiance Performance
        low_irr_efficiency = np.random.normal(95, 3)  # % of STC efficiency
        results["tests"]["low_irradiance"] = {
            "name": "Low Irradiance Performance (200 W/m²)",
            "relative_efficiency": low_irr_efficiency,
            "minimum": 90,
            "passed": low_irr_efficiency >= 90
        }

        # Thermal Cycling (200 cycles)
        thermal_cycles = self.standards["IEC_61215"]["thermal_cycling_count"]
        power_degradation_thermal = np.random.uniform(0.5, 2.5)
        results["tests"]["thermal_cycling"] = {
            "name": f"Thermal Cycling ({thermal_cycles} cycles)",
            "cycles": thermal_cycles,
            "power_degradation": power_degradation_thermal,
            "max_degradation": 5.0,
            "passed": power_degradation_thermal <= 5.0
        }

        # Humidity-Freeze (10 cycles)
        hf_cycles = self.standards["IEC_61215"]["humidity_freeze_cycles"]
        power_degradation_hf = np.random.uniform(0.3, 1.8)
        results["tests"]["humidity_freeze"] = {
            "name": f"Humidity-Freeze ({hf_cycles} cycles)",
            "cycles": hf_cycles,
            "power_degradation": power_degradation_hf,
            "max_degradation": 5.0,
            "passed": power_degradation_hf <= 5.0
        }

        # Damp Heat (1000 hours)
        dh_duration = self.standards["IEC_61215"]["damp_heat_duration"]
        power_degradation_dh = np.random.uniform(1.0, 4.0)
        results["tests"]["damp_heat"] = {
            "name": f"Damp Heat ({dh_duration} hours)",
            "duration": dh_duration,
            "power_degradation": power_degradation_dh,
            "max_degradation": 5.0,
            "passed": power_degradation_dh <= 5.0
        }

        # Mechanical Load (2400 Pa)
        mechanical_load = 2400  # Pa
        power_degradation_mech = np.random.uniform(0.2, 1.5)
        results["tests"]["mechanical_load"] = {
            "name": "Mechanical Load Test",
            "load": mechanical_load,
            "power_degradation": power_degradation_mech,
            "max_degradation": 5.0,
            "passed": power_degradation_mech <= 5.0
        }

        # Hail Impact (25mm ice ball at 23 m/s)
        results["tests"]["hail_impact"] = {
            "name": "Hail Impact Test",
            "ice_ball_diameter": 25,  # mm
            "velocity": 23,  # m/s
            "impact_locations": 11,
            "cracks_detected": 0,
            "passed": True
        }

        # Bypass Diode Thermal Test
        diode_temp = np.random.uniform(75, 95)
        results["tests"]["bypass_diode"] = {
            "name": "Bypass Diode Thermal Test",
            "max_temperature": diode_temp,
            "limit": 100,  # °C
            "passed": diode_temp <= 100
        }

        # Overall pass/fail
        all_passed = all(test.get("passed", False) for test in results["tests"].values())
        results["overall_passed"] = all_passed
        results["certification_eligible"] = all_passed

        return results

    def run_iec_61730_safety_test(
        self,
        module_id: str,
        construction_class: str = "Class_A"
    ) -> Dict[str, any]:
        """
        Run IEC 61730 safety qualification tests.

        Args:
            module_id: Module identifier
            construction_class: Construction class (Class_A, Class_B, Class_C)

        Returns:
            Safety test results
        """
        results = {
            "test_id": f"IEC61730_{module_id}_{datetime.now().strftime('%Y%m%d')}",
            "standard": "IEC_61730",
            "module_id": module_id,
            "construction_class": construction_class,
            "test_date": datetime.now(),
            "tests": {}
        }

        # Fire Test
        fire_rating = np.random.choice(["Class_A", "Class_B", "Class_C"], p=[0.7, 0.2, 0.1])
        results["tests"]["fire"] = {
            "name": "Fire Test",
            "rating": fire_rating,
            "required": construction_class,
            "passed": fire_rating == construction_class
        }

        # Electrical Shock Protection
        touch_current = np.random.uniform(0.1, 0.5)  # mA
        results["tests"]["electrical_shock"] = {
            "name": "Electrical Shock Protection",
            "touch_current": touch_current,
            "limit": 1.0,  # mA
            "passed": touch_current <= 1.0
        }

        # Mechanical Stress Test
        edge_load = np.random.uniform(200, 300)  # N
        results["tests"]["mechanical_stress"] = {
            "name": "Mechanical Stress Test",
            "edge_load": edge_load,
            "no_sharp_edges": True,
            "passed": True
        }

        # Environmental Stress
        results["tests"]["environmental_stress"] = {
            "name": "Environmental Stress",
            "uv_exposure": "Pass",
            "thermal_stress": "Pass",
            "humidity_exposure": "Pass",
            "passed": True
        }

        # Wet Leakage Current
        wet_leakage = np.random.uniform(0.5, 2.0)  # mA
        results["tests"]["wet_leakage"] = {
            "name": "Wet Leakage Current",
            "measured": wet_leakage,
            "limit": 3.5,  # mA
            "passed": wet_leakage <= 3.5
        }

        # Dielectric Withstand
        dielectric_voltage = np.random.uniform(2000, 3000)  # V
        results["tests"]["dielectric"] = {
            "name": "Dielectric Withstand Test",
            "test_voltage": dielectric_voltage,
            "minimum": 1000,  # V
            "breakdown": False,
            "passed": True
        }

        all_passed = all(test.get("passed", False) for test in results["tests"].values())
        results["overall_passed"] = all_passed
        results["safety_certified"] = all_passed

        return results

    def run_iec_63202_lid_test(
        self,
        module_id: str,
        initial_power: float,
        irradiance: float = 1000,
        duration_hours: int = 168
    ) -> Dict[str, any]:
        """
        Run IEC 63202 Light-Induced Degradation test.

        Args:
            module_id: Module identifier
            initial_power: Initial power (W)
            irradiance: Test irradiance (W/m²)
            duration_hours: Test duration (hours)

        Returns:
            LID test results
        """
        results = {
            "test_id": f"IEC63202_{module_id}_{datetime.now().strftime('%Y%m%d')}",
            "standard": "IEC_63202",
            "module_id": module_id,
            "test_date": datetime.now(),
            "test_conditions": {
                "irradiance": irradiance,
                "duration_hours": duration_hours,
                "temperature": 25  # °C
            }
        }

        # Simulate LID degradation over time
        time_points = np.linspace(0, duration_hours, 50)

        # LID model: Power = P0 * (1 - LID_max * (1 - exp(-t/tau)))
        lid_max = np.random.uniform(1.5, 3.0)  # % maximum degradation
        tau = np.random.uniform(20, 40)  # time constant (hours)

        power_over_time = initial_power * (1 - (lid_max / 100) * (1 - np.exp(-time_points / tau)))

        final_power = power_over_time[-1]
        total_degradation = ((initial_power - final_power) / initial_power) * 100

        results["measurements"] = {
            "time_hours": time_points.tolist(),
            "power_watts": power_over_time.tolist()
        }

        results["degradation_analysis"] = {
            "initial_power": initial_power,
            "final_power": final_power,
            "total_degradation_percent": total_degradation,
            "lid_type": "LID" if total_degradation < 2.5 else "LETID",
            "stabilized": total_degradation < 3.0,
            "max_acceptable": 3.0,
            "passed": total_degradation <= 3.0
        }

        return results

    def run_iec_63209_extended_stress(
        self,
        module_id: str,
        module_power: float
    ) -> Dict[str, any]:
        """
        Run IEC 63209 Extended-Stress Testing.

        Args:
            module_id: Module identifier
            module_power: Initial module power (W)

        Returns:
            Extended-stress test results
        """
        results = {
            "test_id": f"IEC63209_{module_id}_{datetime.now().strftime('%Y%m%d')}",
            "standard": "IEC_63209",
            "module_id": module_id,
            "test_date": datetime.now(),
            "tests": {}
        }

        # Dynamic Mechanical Load (more severe than IEC 61215)
        dml_cycles = 1000
        dml_degradation = np.random.uniform(1.0, 3.5)
        results["tests"]["dynamic_mechanical_load"] = {
            "name": "Dynamic Mechanical Load",
            "cycles": dml_cycles,
            "load_profile": "Sinusoidal 1000-3000 Pa",
            "power_degradation": dml_degradation,
            "max_degradation": 5.0,
            "passed": dml_degradation <= 5.0
        }

        # Extended Thermal Cycling (400 cycles vs 200 in IEC 61215)
        etc_cycles = 400
        etc_degradation = np.random.uniform(2.0, 4.5)
        results["tests"]["extended_thermal_cycling"] = {
            "name": "Extended Thermal Cycling",
            "cycles": etc_cycles,
            "temperature_range": "-40°C to +85°C",
            "power_degradation": etc_degradation,
            "max_degradation": 5.0,
            "passed": etc_degradation <= 5.0
        }

        # Damp Heat + High Voltage Bias
        dh_hv_duration = 1000  # hours
        dh_hv_degradation = np.random.uniform(2.5, 5.0)
        leakage_current = np.random.uniform(10, 50)  # μA

        results["tests"]["damp_heat_hv"] = {
            "name": "Damp Heat + High Voltage Bias",
            "duration_hours": dh_hv_duration,
            "voltage_bias": "-1000V",
            "power_degradation": dh_hv_degradation,
            "leakage_current_ua": leakage_current,
            "max_degradation": 5.0,
            "max_leakage": 100,  # μA
            "passed": dh_hv_degradation <= 5.0 and leakage_current <= 100
        }

        all_passed = all(test.get("passed", False) for test in results["tests"].values())
        results["overall_passed"] = all_passed
        results["premium_certified"] = all_passed

        return results

    def run_iec_ts_63279_thermal_model(
        self,
        module_id: str,
        ambient_temps: np.ndarray,
        irradiances: np.ndarray,
        wind_speeds: np.ndarray
    ) -> Dict[str, any]:
        """
        Run IEC TS 63279 Thermal Characteristics Modeling.

        Args:
            module_id: Module identifier
            ambient_temps: Array of ambient temperatures (°C)
            irradiances: Array of irradiances (W/m²)
            wind_speeds: Array of wind speeds (m/s)

        Returns:
            Thermal modeling results
        """
        results = {
            "test_id": f"IECTS63279_{module_id}_{datetime.now().strftime('%Y%m%d')}",
            "standard": "IEC_TS_63279",
            "module_id": module_id,
            "test_date": datetime.now()
        }

        # Thermal model parameters
        u_value = np.random.uniform(25, 35)  # W/(m²·K)
        thermal_capacitance = np.random.uniform(10000, 15000)  # J/(m²·K)

        # Calculate module temperatures using simplified thermal model
        # T_module = T_ambient + (irradiance * (1 - efficiency)) / (U + wind_factor * wind_speed)
        efficiency = 0.20
        wind_factor = 5.0

        module_temps = ambient_temps + (irradiances * (1 - efficiency)) / (u_value + wind_factor * wind_speeds)

        results["thermal_parameters"] = {
            "u_value": u_value,
            "thermal_capacitance": thermal_capacitance,
            "wind_coefficient": wind_factor
        }

        results["temperature_predictions"] = {
            "ambient_temp": ambient_temps.tolist(),
            "irradiance": irradiances.tolist(),
            "wind_speed": wind_speeds.tolist(),
            "predicted_module_temp": module_temps.tolist()
        }

        # Validation metrics
        results["model_validation"] = {
            "max_module_temp": float(np.max(module_temps)),
            "min_module_temp": float(np.min(module_temps)),
            "avg_temp_rise": float(np.mean(module_temps - ambient_temps)),
            "model_accuracy": "±3°C",
            "validated": True
        }

        return results

    def generate_test_certificate(
        self,
        test_results: Dict[str, any],
        certificate_number: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate test certificate.

        Args:
            test_results: Test results dictionary
            certificate_number: Optional certificate number

        Returns:
            Certificate data
        """
        if certificate_number is None:
            certificate_number = f"CERT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        certificate = {
            "certificate_number": certificate_number,
            "issue_date": datetime.now(),
            "expiry_date": datetime.now() + timedelta(days=365 * 5),  # 5 years
            "standard": test_results.get("standard", "N/A"),
            "module_id": test_results.get("module_id", "N/A"),
            "test_id": test_results.get("test_id", "N/A"),
            "overall_result": "PASS" if test_results.get("overall_passed", False) else "FAIL",
            "test_laboratory": "PV Circularity Simulator Testing Lab",
            "accreditation": "ISO/IEC 17025:2017",
            "test_summary": test_results
        }

        return certificate


def render_iec_testing():
    """Render IEC testing interface in Streamlit."""
    st.header("🔬 IEC Standards Testing & Certification")
    st.markdown("Comprehensive IEC testing for module qualification and safety certification.")

    simulator = IECTestingSimulator()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 IEC 61215",
        "🛡️ IEC 61730",
        "☀️ IEC 63202 (LID)",
        "⚡ IEC 63209",
        "🌡️ IEC TS 63279",
        "📜 Certificates"
    ])

    with tab1:
        st.subheader("IEC 61215: Design Qualification & Type Approval")
        st.markdown("Crystalline silicon terrestrial PV modules - Design qualification and type approval")

        col1, col2 = st.columns(2)

        with col1:
            module_id_61215 = st.text_input("Module ID:", value="MOD-2024-001", key="mod_61215")
            module_power = st.number_input("Rated Power (W):", min_value=100, max_value=700, value=400, step=10)

        with col2:
            test_temp = st.number_input("Test Temperature (°C):", value=25, min_value=-40, max_value=85)
            test_irradiance = st.number_input("Test Irradiance (W/m²):", value=1000, min_value=100, max_value=1200)

        if st.button("🧪 Run IEC 61215 Test Suite", key="run_61215"):
            test_conditions = {"temperature": test_temp, "irradiance": test_irradiance}

            with st.spinner("Running comprehensive test suite..."):
                results = simulator.run_iec_61215_test(module_id_61215, module_power, test_conditions)

            st.success(f"Test Complete: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")

            # Display test results
            st.subheader("Test Results Summary")

            test_data = []
            for test_name, test_info in results["tests"].items():
                test_data.append({
                    "Test": test_info.get("name", test_name),
                    "Status": "✅ Pass" if test_info.get("passed", False) else "❌ Fail",
                    "Details": str(test_info.get("power_degradation", test_info.get("measured", "N/A")))
                })

            df_tests = pd.DataFrame(test_data)
            st.dataframe(df_tests, use_container_width=True)

            # Degradation comparison chart
            degradation_tests = {k: v for k, v in results["tests"].items()
                               if "power_degradation" in v}

            if degradation_tests:
                fig = go.Figure()

                test_names = [v["name"] for v in degradation_tests.values()]
                degradations = [v["power_degradation"] for v in degradation_tests.values()]
                max_allowed = [v.get("max_degradation", 5.0) for v in degradation_tests.values()]

                fig.add_trace(go.Bar(
                    x=test_names,
                    y=degradations,
                    name="Measured Degradation",
                    marker_color='#3498DB'
                ))

                fig.add_trace(go.Scatter(
                    x=test_names,
                    y=max_allowed,
                    name="Max Allowed (5%)",
                    line=dict(color='#E74C3C', width=2, dash='dash'),
                    mode='lines'
                ))

                fig.update_layout(
                    title="Power Degradation by Test Type",
                    xaxis_title="Test",
                    yaxis_title="Power Degradation (%)",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("IEC 61730: Safety Qualification")
        st.markdown("PV module safety qualification - Requirements for construction and testing")

        col1, col2 = st.columns(2)

        with col1:
            module_id_61730 = st.text_input("Module ID:", value="MOD-2024-002", key="mod_61730")

        with col2:
            construction_class = st.selectbox("Construction Class:", ["Class_A", "Class_B", "Class_C"])

        if st.button("🛡️ Run IEC 61730 Safety Tests", key="run_61730"):
            with st.spinner("Running safety qualification tests..."):
                results = simulator.run_iec_61730_safety_test(module_id_61730, construction_class)

            st.success(f"Safety Test: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Fire Rating", results["tests"]["fire"]["rating"])

            with col2:
                st.metric("Touch Current", f"{results['tests']['electrical_shock']['touch_current']:.2f} mA")

            with col3:
                st.metric("Safety Certified", "✅ Yes" if results["safety_certified"] else "❌ No")

            # Detailed results
            st.subheader("Detailed Safety Test Results")
            safety_data = []
            for test_name, test_info in results["tests"].items():
                safety_data.append({
                    "Test": test_info.get("name", test_name),
                    "Status": "✅ Pass" if test_info.get("passed", False) else "❌ Fail",
                    "Result": str(test_info.get("rating", test_info.get("measured", "Pass")))
                })

            df_safety = pd.DataFrame(safety_data)
            st.dataframe(df_safety, use_container_width=True)

    with tab3:
        st.subheader("IEC 63202: Light-Induced Degradation (LID)")
        st.markdown("Measurement and characterization of light-induced degradation")

        col1, col2 = st.columns(2)

        with col1:
            module_id_63202 = st.text_input("Module ID:", value="MOD-2024-003", key="mod_63202")
            initial_power_lid = st.number_input("Initial Power (W):", min_value=100, max_value=700, value=400, step=10, key="power_lid")

        with col2:
            irradiance_lid = st.number_input("Test Irradiance (W/m²):", value=1000, min_value=500, max_value=1500, step=100, key="irr_lid")
            duration_lid = st.number_input("Test Duration (hours):", value=168, min_value=24, max_value=500, step=24)

        if st.button("☀️ Run LID Test", key="run_lid"):
            with st.spinner(f"Running {duration_lid}-hour LID test simulation..."):
                results = simulator.run_iec_63202_lid_test(
                    module_id_63202, initial_power_lid, irradiance_lid, duration_lid
                )

            degradation = results["degradation_analysis"]["total_degradation_percent"]
            st.success(f"LID Test Complete: {degradation:.2f}% degradation - {'✅ PASSED' if results['degradation_analysis']['passed'] else '❌ FAILED'}")

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Initial Power", f"{results['degradation_analysis']['initial_power']:.1f} W")

            with col2:
                st.metric("Final Power", f"{results['degradation_analysis']['final_power']:.1f} W")

            with col3:
                st.metric("Total Degradation", f"{degradation:.2f}%",
                         delta=f"-{degradation:.2f}%")

            with col4:
                st.metric("LID Type", results['degradation_analysis']['lid_type'])

            # Plot degradation over time
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results["measurements"]["time_hours"],
                y=results["measurements"]["power_watts"],
                mode='lines',
                name='Module Power',
                line=dict(color='#E74C3C', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))

            fig.add_hline(
                y=initial_power_lid * 0.97,
                line_dash="dash",
                line_color="green",
                annotation_text="97% Power (3% degradation limit)"
            )

            fig.update_layout(
                title="Light-Induced Degradation Over Time",
                xaxis_title="Time (hours)",
                yaxis_title="Power (W)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("IEC 63209: Extended-Stress Testing")
        st.markdown("Enhanced testing for premium module qualification")

        col1, col2 = st.columns(2)

        with col1:
            module_id_63209 = st.text_input("Module ID:", value="MOD-2024-004", key="mod_63209")

        with col2:
            module_power_63209 = st.number_input("Module Power (W):", min_value=100, max_value=700, value=450, step=10, key="power_63209")

        if st.button("⚡ Run Extended-Stress Tests", key="run_63209"):
            with st.spinner("Running extended-stress test sequence..."):
                results = simulator.run_iec_63209_extended_stress(module_id_63209, module_power_63209)

            st.success(f"Extended-Stress Test: {'✅ PASSED (Premium Certified)' if results['overall_passed'] else '❌ FAILED'}")

            # Results comparison
            test_names = [v["name"] for v in results["tests"].values()]
            degradations = [v["power_degradation"] for v in results["tests"].values()]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=test_names,
                y=degradations,
                marker_color=['#2ECC71' if d <= 5.0 else '#E74C3C' for d in degradations],
                text=[f"{d:.1f}%" for d in degradations],
                textposition='auto'
            ))

            fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                         annotation_text="Max Allowed (5%)")

            fig.update_layout(
                title="Extended-Stress Test Results",
                xaxis_title="Test Type",
                yaxis_title="Power Degradation (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics
            st.subheader("Test Details")
            for test_name, test_info in results["tests"].items():
                with st.expander(test_info["name"]):
                    cols = st.columns(3)
                    for i, (key, value) in enumerate(test_info.items()):
                        if key not in ["name", "passed"]:
                            cols[i % 3].metric(key.replace("_", " ").title(), str(value))

    with tab5:
        st.subheader("IEC TS 63279: Thermal Characteristics Modeling")
        st.markdown("Module thermal behavior prediction and validation")

        module_id_63279 = st.text_input("Module ID:", value="MOD-2024-005", key="mod_63279")

        col1, col2, col3 = st.columns(3)

        with col1:
            num_points = st.slider("Number of Test Points:", 10, 100, 50)

        with col2:
            max_irradiance = st.slider("Max Irradiance (W/m²):", 400, 1200, 1000, step=100)

        with col3:
            max_wind = st.slider("Max Wind Speed (m/s):", 0, 20, 10)

        if st.button("🌡️ Run Thermal Modeling", key="run_thermal"):
            # Generate test conditions
            ambient_temps = np.random.uniform(15, 35, num_points)
            irradiances = np.random.uniform(200, max_irradiance, num_points)
            wind_speeds = np.random.uniform(0, max_wind, num_points)

            with st.spinner("Running thermal characterization..."):
                results = simulator.run_iec_ts_63279_thermal_model(
                    module_id_63279, ambient_temps, irradiances, wind_speeds
                )

            st.success("✅ Thermal Model Validated")

            # Display thermal parameters
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("U-value", f"{results['thermal_parameters']['u_value']:.1f} W/(m²·K)")

            with col2:
                st.metric("Thermal Capacitance", f"{results['thermal_parameters']['thermal_capacitance']:.0f} J/(m²·K)")

            with col3:
                st.metric("Max Module Temp", f"{results['model_validation']['max_module_temp']:.1f}°C")

            # 3D scatter plot
            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=results["temperature_predictions"]["irradiance"],
                y=results["temperature_predictions"]["wind_speed"],
                z=results["temperature_predictions"]["predicted_module_temp"],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results["temperature_predictions"]["predicted_module_temp"],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Module<br>Temp (°C)")
                ),
                text=[f"Temp: {t:.1f}°C<br>Irr: {i:.0f} W/m²<br>Wind: {w:.1f} m/s"
                      for t, i, w in zip(
                          results["temperature_predictions"]["predicted_module_temp"],
                          results["temperature_predictions"]["irradiance"],
                          results["temperature_predictions"]["wind_speed"]
                      )],
                hoverinfo='text'
            ))

            fig.update_layout(
                title="Module Temperature vs Environmental Conditions",
                scene=dict(
                    xaxis_title="Irradiance (W/m²)",
                    yaxis_title="Wind Speed (m/s)",
                    zaxis_title="Module Temperature (°C)"
                ),
                height=500,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("📜 Test Certificates & Documentation")

        st.markdown("""
        ### Available Certifications

        Upon successful completion of testing, modules receive certification according to:

        - **IEC 61215**: Design qualification and type approval
        - **IEC 61730**: Safety qualification (Class A/B/C)
        - **IEC 63202**: LID resistance certification
        - **IEC 63209**: Premium module certification
        - **IEC TS 63279**: Thermal model validation

        Certificates are valid for **5 years** from issue date and include:
        - Complete test results and measurements
        - Pass/fail status for all test categories
        - Laboratory accreditation information
        - Unique certificate number for traceability
        """)

        st.info("💡 Run any test suite above to generate a certificate")

        # Certificate template preview
        with st.expander("📋 View Sample Certificate"):
            st.markdown("""
            ```
            ═══════════════════════════════════════════════════════════
                      PV MODULE TEST CERTIFICATE
            ═══════════════════════════════════════════════════════════

            Certificate Number: CERT-20241117123456
            Issue Date: 2024-11-17
            Expiry Date: 2029-11-17

            Standard: IEC 61215
            Module ID: MOD-2024-001
            Test Laboratory: PV Circularity Simulator Testing Lab
            Accreditation: ISO/IEC 17025:2017

            OVERALL RESULT: ✅ PASS

            This certificate confirms that the module has successfully
            passed all required tests according to the specified standard.
            ═══════════════════════════════════════════════════════════
            ```
            """)

    st.divider()
    st.info("💡 **IEC Standards Testing** - Branch B04 | Comprehensive Testing & Certification Suite")
