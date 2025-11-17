"""
Fault Detection & Diagnostics Module (Branch B08).

Features:
- Hot spot detection (thermal imaging)
- Cell crack detection (EL imaging)
- Bypass diode failure detection
- Soiling detection and quantification
- Delamination detection
- PID detection and mitigation
- Fault severity classification
- Recommended corrective actions
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import FAULT_TYPES
from utils.validators import FaultDetection


class FaultDiagnostics:
    """Fault detection and diagnostics system."""

    def __init__(self):
        """Initialize fault diagnostics system."""
        self.fault_types = FAULT_TYPES

    def detect_hotspots(
        self,
        thermal_image: np.ndarray,
        ambient_temp: float = 25.0,
        threshold_delta: float = 15.0
    ) -> Dict[str, any]:
        """
        Detect hot spots using thermal imaging analysis.

        Args:
            thermal_image: Thermal image array (simulated)
            ambient_temp: Ambient temperature (°C)
            threshold_delta: Temperature threshold above ambient (°C)

        Returns:
            Hot spot detection results
        """
        # Simulate thermal imaging analysis
        mean_temp = thermal_image.mean()
        max_temp = thermal_image.max()
        min_temp = thermal_image.min()

        temp_delta = max_temp - mean_temp

        # Detect hotspots
        hotspot_mask = thermal_image > (mean_temp + threshold_delta)
        num_hotspots = np.sum(hotspot_mask)

        # Calculate severity
        if temp_delta > 30:
            severity = 'critical'
            power_loss_estimate = 25
        elif temp_delta > 20:
            severity = 'high'
            power_loss_estimate = 15
        elif temp_delta > threshold_delta:
            severity = 'medium'
            power_loss_estimate = 8
        else:
            severity = 'low'
            power_loss_estimate = 2

        # Recommended actions
        if severity in ['critical', 'high']:
            recommended_action = "Immediate inspection required. Check for bypass diode failure, poor connections, or cell damage."
        elif severity == 'medium':
            recommended_action = "Schedule inspection within 1 week. Monitor temperature trends."
        else:
            recommended_action = "Continue monitoring. No immediate action required."

        return {
            'detected': num_hotspots > 0,
            'num_hotspots': int(num_hotspots),
            'max_temperature': float(max_temp),
            'mean_temperature': float(mean_temp),
            'temperature_delta': float(temp_delta),
            'severity': severity,
            'power_loss_estimate': power_loss_estimate,
            'recommended_action': recommended_action,
            'hotspot_locations': hotspot_mask
        }

    def detect_cell_cracks(
        self,
        el_image: np.ndarray,
        crack_threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Detect cell cracks using electroluminescence imaging.

        Args:
            el_image: EL image array (simulated)
            crack_threshold: Threshold for crack detection

        Returns:
            Cell crack detection results
        """
        # Simulate EL imaging analysis
        # Look for discontinuities and dark areas

        # Calculate gradient to find discontinuities
        gradient_x = np.gradient(el_image, axis=0)
        gradient_y = np.gradient(el_image, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Detect cracks based on high gradients
        crack_mask = gradient_magnitude > crack_threshold
        num_cracks = np.sum(crack_mask)

        # Calculate affected area
        affected_area_percent = (num_cracks / el_image.size) * 100

        # Estimate power loss
        power_loss_estimate = min(affected_area_percent * 2, 25)

        # Severity classification
        if affected_area_percent > 15:
            severity = 'high'
        elif affected_area_percent > 8:
            severity = 'medium'
        elif affected_area_percent > 3:
            severity = 'low'
        else:
            severity = 'none'

        # Recommended actions
        if severity == 'high':
            recommended_action = "Replace module. Significant power loss detected."
        elif severity == 'medium':
            recommended_action = "Monitor closely. Consider replacement in next maintenance cycle."
        elif severity == 'low':
            recommended_action = "Continue monitoring. Document for trend analysis."
        else:
            recommended_action = "No action required."

        return {
            'detected': num_cracks > 0,
            'num_cracks': int(num_cracks),
            'affected_area_percent': float(affected_area_percent),
            'severity': severity,
            'power_loss_estimate': float(power_loss_estimate),
            'recommended_action': recommended_action,
            'crack_locations': crack_mask
        }

    def detect_bypass_diode_failure(
        self,
        iv_curve_data: Dict[str, np.ndarray],
        expected_curve: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Detect bypass diode failures using I-V curve analysis.

        Args:
            iv_curve_data: I-V curve data (voltage, current)
            expected_curve: Expected I-V curve for comparison

        Returns:
            Bypass diode failure detection results
        """
        voltage = iv_curve_data['voltage']
        current = iv_curve_data['current']

        # Calculate power
        power = voltage * current
        max_power_idx = np.argmax(power)
        max_power = power[max_power_idx]

        # Look for characteristic signs of bypass diode failure
        # 1. Steps in I-V curve
        # 2. Reduced Voc or Isc
        # 3. Multiple power peaks

        # Detect steps (high second derivative)
        current_derivative = np.gradient(current)
        steps_detected = np.sum(np.abs(current_derivative) > np.std(current_derivative) * 3)

        # Check for multiple local maxima in power curve
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(power, height=max_power * 0.3)
        num_peaks = len(peaks)

        # Determine if bypass diode failure is likely
        failure_detected = (steps_detected > 2) or (num_peaks > 1)

        # Estimate power loss
        if failure_detected:
            if num_peaks > 1:
                power_loss_estimate = 20
                severity = 'high'
            else:
                power_loss_estimate = 12
                severity = 'medium'
        else:
            power_loss_estimate = 0
            severity = 'none'

        # Recommended actions
        if severity == 'high':
            recommended_action = "Replace bypass diode immediately. Module operating in degraded mode."
        elif severity == 'medium':
            recommended_action = "Schedule bypass diode replacement. Monitor performance."
        else:
            recommended_action = "No bypass diode issues detected."

        return {
            'detected': failure_detected,
            'num_steps': int(steps_detected),
            'num_power_peaks': int(num_peaks),
            'severity': severity,
            'power_loss_estimate': float(power_loss_estimate),
            'recommended_action': recommended_action
        }

    def detect_soiling(
        self,
        current_power: float,
        expected_power: float,
        irradiance: float,
        recent_rain: bool = False
    ) -> Dict[str, any]:
        """
        Detect and quantify soiling losses.

        Args:
            current_power: Current measured power (kW)
            expected_power: Expected power at current conditions (kW)
            irradiance: Current irradiance (W/m²)
            recent_rain: Whether there was recent rainfall

        Returns:
            Soiling detection results
        """
        # Calculate soiling ratio
        soiling_ratio = current_power / expected_power if expected_power > 0 else 1.0

        # Estimate soiling loss
        soiling_loss_percent = (1 - soiling_ratio) * 100

        # Adjust for recent rain (natural cleaning)
        if recent_rain:
            soiling_loss_percent *= 0.3

        # Severity classification
        if soiling_loss_percent > 8:
            severity = 'high'
        elif soiling_loss_percent > 4:
            severity = 'medium'
        elif soiling_loss_percent > 1:
            severity = 'low'
        else:
            severity = 'none'

        # Recommended actions
        if severity == 'high':
            recommended_action = "Schedule cleaning immediately. Significant energy loss detected."
        elif severity == 'medium':
            recommended_action = "Schedule cleaning within 2 weeks."
        elif severity == 'low':
            recommended_action = "Monitor. Consider cleaning in next maintenance cycle."
        else:
            recommended_action = "No cleaning required at this time."

        # Calculate cleaning economics
        daily_loss_kwh = (expected_power - current_power) * 5  # Assume 5 peak hours
        annual_loss_kwh = daily_loss_kwh * 365
        annual_revenue_loss = annual_loss_kwh * 0.12  # $0.12/kWh

        return {
            'detected': soiling_loss_percent > 1,
            'soiling_ratio': float(soiling_ratio),
            'soiling_loss_percent': float(soiling_loss_percent),
            'severity': severity,
            'power_loss_estimate': float(soiling_loss_percent),
            'recommended_action': recommended_action,
            'daily_energy_loss_kwh': float(daily_loss_kwh),
            'annual_revenue_loss_usd': float(annual_revenue_loss)
        }

    def detect_delamination(
        self,
        visual_image: np.ndarray,
        age_years: float
    ) -> Dict[str, any]:
        """
        Detect delamination using visual inspection analysis.

        Args:
            visual_image: Visual image array (simulated)
            age_years: Module age in years

        Returns:
            Delamination detection results
        """
        # Simulate visual inspection
        # Look for color changes, bubbles, or separation

        # Calculate image statistics
        mean_intensity = visual_image.mean()
        std_intensity = visual_image.std()

        # Detect anomalies (areas with significantly different intensity)
        anomaly_threshold = mean_intensity + 2 * std_intensity
        delamination_mask = visual_image > anomaly_threshold
        affected_area_percent = (np.sum(delamination_mask) / visual_image.size) * 100

        # Age factor (delamination more likely in older modules)
        age_factor = min(age_years / 15, 2.0)
        adjusted_severity_score = affected_area_percent * age_factor

        # Severity classification
        if adjusted_severity_score > 10:
            severity = 'high'
            power_loss_estimate = 18
        elif adjusted_severity_score > 5:
            severity = 'medium'
            power_loss_estimate = 10
        elif adjusted_severity_score > 2:
            severity = 'low'
            power_loss_estimate = 4
        else:
            severity = 'none'
            power_loss_estimate = 0

        # Recommended actions
        if severity == 'high':
            recommended_action = "Replace module. Delamination causing significant performance degradation and safety risk."
        elif severity == 'medium':
            recommended_action = "Monitor closely. Plan for replacement. Check warranty coverage."
        elif severity == 'low':
            recommended_action = "Document and monitor. Inspect in next scheduled maintenance."
        else:
            recommended_action = "No delamination detected."

        return {
            'detected': affected_area_percent > 1,
            'affected_area_percent': float(affected_area_percent),
            'severity': severity,
            'power_loss_estimate': float(power_loss_estimate),
            'recommended_action': recommended_action,
            'warranty_claim_eligible': age_years < 10 and severity in ['high', 'medium']
        }

    def detect_pid(
        self,
        power_degradation: float,
        system_voltage: float,
        humidity: float,
        temperature: float
    ) -> Dict[str, any]:
        """
        Detect Potential-Induced Degradation (PID).

        Args:
            power_degradation: Observed power degradation (%)
            system_voltage: System operating voltage (V)
            humidity: Average humidity (%)
            temperature: Average temperature (°C)

        Returns:
            PID detection results
        """
        # PID risk factors
        high_voltage = system_voltage > 600
        high_humidity = humidity > 70
        high_temp = temperature > 30

        # Calculate PID risk score
        risk_score = 0
        if high_voltage:
            risk_score += 40
        if high_humidity:
            risk_score += 30
        if high_temp:
            risk_score += 30

        # Normalize degradation (PID typically shows 10-50% degradation)
        pid_likelihood = min(power_degradation * 2, 100)

        # Combine with risk factors
        combined_score = (pid_likelihood + risk_score) / 2

        # Detection and severity
        if combined_score > 70 and power_degradation > 15:
            detected = True
            severity = 'high'
            power_loss_estimate = min(power_degradation, 50)
        elif combined_score > 50 and power_degradation > 8:
            detected = True
            severity = 'medium'
            power_loss_estimate = min(power_degradation, 30)
        elif combined_score > 30:
            detected = True
            severity = 'low'
            power_loss_estimate = min(power_degradation, 15)
        else:
            detected = False
            severity = 'none'
            power_loss_estimate = 0

        # Recommended actions and mitigation
        if severity == 'high':
            recommended_action = "Immediate PID mitigation required. Options: 1) Install PID recovery system (reverse bias at night), 2) Grounding reconfiguration, 3) Replace affected modules."
            mitigation_options = ["Night-time reverse bias", "Grounding system modification", "Module replacement"]
        elif severity == 'medium':
            recommended_action = "Implement PID prevention measures. Consider PID recovery system installation."
            mitigation_options = ["Anti-PID coating", "System grounding optimization", "Monitor and trend"]
        elif severity == 'low':
            recommended_action = "Monitor for PID progression. Optimize system grounding."
            mitigation_options = ["Enhanced monitoring", "Grounding check"]
        else:
            recommended_action = "No PID detected. Continue monitoring."
            mitigation_options = []

        return {
            'detected': detected,
            'pid_risk_score': float(risk_score),
            'combined_score': float(combined_score),
            'severity': severity,
            'power_loss_estimate': float(power_loss_estimate),
            'recommended_action': recommended_action,
            'mitigation_options': mitigation_options,
            'reversible': severity in ['low', 'medium']
        }

    def classify_fault_severity(
        self,
        fault_type: str,
        power_loss: float,
        safety_risk: bool = False
    ) -> Dict[str, str]:
        """
        Classify fault severity and prioritize response.

        Args:
            fault_type: Type of fault
            power_loss: Estimated power loss (%)
            safety_risk: Whether fault poses safety risk

        Returns:
            Severity classification and priority
        """
        # Safety-critical faults
        if safety_risk:
            severity = 'critical'
            priority = 'immediate'
            response_time = 'Within 24 hours'
        # High power loss
        elif power_loss > 20:
            severity = 'high'
            priority = 'urgent'
            response_time = 'Within 1 week'
        # Moderate power loss
        elif power_loss > 10:
            severity = 'medium'
            priority = 'scheduled'
            response_time = 'Within 1 month'
        # Low power loss
        elif power_loss > 5:
            severity = 'low'
            priority = 'routine'
            response_time = 'Next maintenance cycle'
        # Minimal impact
        else:
            severity = 'minimal'
            priority = 'monitoring'
            response_time = 'Continue monitoring'

        return {
            'severity': severity,
            'priority': priority,
            'response_time': response_time,
            'fault_type': fault_type
        }

    def generate_fault_report(
        self,
        all_faults: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Generate comprehensive fault diagnostics report.

        Args:
            all_faults: List of detected faults

        Returns:
            Comprehensive fault report
        """
        total_faults = len(all_faults)

        # Count by severity
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'none': 0
        }

        total_power_loss = 0
        high_priority_faults = []

        for fault in all_faults:
            severity = fault.get('severity', 'none')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            power_loss = fault.get('power_loss_estimate', 0)
            total_power_loss += power_loss

            if severity in ['critical', 'high']:
                high_priority_faults.append(fault)

        # Overall system health score (0-100)
        health_score = max(0, 100 - total_power_loss)

        return {
            'total_faults': total_faults,
            'severity_counts': severity_counts,
            'total_power_loss_estimate': total_power_loss,
            'high_priority_faults': high_priority_faults,
            'health_score': health_score,
            'health_status': 'Good' if health_score > 80 else ('Fair' if health_score > 60 else 'Poor')
        }


def render_fault_diagnostics():
    """Render fault diagnostics interface in Streamlit."""
    st.header("🔍 Fault Detection & Diagnostics")
    st.markdown("Advanced fault detection using thermal imaging, EL imaging, and performance analysis.")

    diagnostics = FaultDiagnostics()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔥 Hot Spot Detection",
        "💔 Cell Crack Detection",
        "⚡ Bypass Diode",
        "🧹 Soiling",
        "🔬 Delamination & PID",
        "📋 Fault Report"
    ])

    with tab1:
        st.subheader("Hot Spot Detection (Thermal Imaging)")

        col1, col2 = st.columns(2)

        with col1:
            image_size = st.slider("Module Grid Size:", 10, 50, 20)
            ambient_temp = st.slider("Ambient Temperature (°C):", 0, 50, 25)

        with col2:
            temp_threshold = st.slider("Hot Spot Threshold (ΔT °C):", 5, 30, 15)
            hotspot_severity = st.slider("Simulated Hotspot Severity:", 0, 40, 20)

        if st.button("🔥 Run Hot Spot Detection", key="hotspot"):
            with st.spinner("Analyzing thermal image..."):
                # Generate simulated thermal image
                thermal_image = np.random.normal(ambient_temp + 20, 5, (image_size, image_size))

                # Add simulated hotspots
                num_hotspots = np.random.randint(0, 3)
                for _ in range(num_hotspots):
                    x, y = np.random.randint(0, image_size, 2)
                    size = 2
                    thermal_image[max(0, x-size):min(image_size, x+size),
                                max(0, y-size):min(image_size, y+size)] += hotspot_severity

                results = diagnostics.detect_hotspots(thermal_image, ambient_temp, temp_threshold)

            st.session_state['hotspot_results'] = results
            st.session_state['thermal_image'] = thermal_image

            # Display results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Hot Spots Detected", results['num_hotspots'])

            with col2:
                st.metric("Max Temperature", f"{results['max_temperature']:.1f}°C")

            with col3:
                st.metric("Temperature Delta", f"{results['temperature_delta']:.1f}°C")

            with col4:
                severity_colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                st.metric("Severity", f"{severity_colors.get(results['severity'], '⚪')} {results['severity'].title()}")

            st.metric("Power Loss Estimate", f"{results['power_loss_estimate']}%")

            st.info(f"**Recommendation:** {results['recommended_action']}")

            # Visualize thermal image
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Thermal Image**")
                fig = go.Figure(data=go.Heatmap(
                    z=thermal_image,
                    colorscale='Hot',
                    colorbar=dict(title='Temperature (°C)')
                ))
                fig.update_layout(
                    title="Thermal Image",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Hot Spot Locations**")
                fig = go.Figure(data=go.Heatmap(
                    z=results['hotspot_locations'].astype(int),
                    colorscale='Reds',
                    showscale=False
                ))
                fig.update_layout(
                    title="Detected Hot Spots",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cell Crack Detection (EL Imaging)")

        col1, col2 = st.columns(2)

        with col1:
            el_image_size = st.slider("Cell Grid Size:", 10, 50, 30, key="el_size")

        with col2:
            crack_severity = st.slider("Simulated Crack Severity:", 0.0, 1.0, 0.5, key="crack_sev")

        if st.button("💔 Run Cell Crack Detection", key="crack"):
            with st.spinner("Analyzing EL image..."):
                # Generate simulated EL image
                el_image = np.random.random((el_image_size, el_image_size))

                # Add simulated cracks (sharp discontinuities)
                num_cracks = np.random.randint(0, 5)
                for _ in range(num_cracks):
                    if np.random.random() < 0.5:
                        # Vertical crack
                        x = np.random.randint(0, el_image_size)
                        el_image[x, :] *= crack_severity
                    else:
                        # Horizontal crack
                        y = np.random.randint(0, el_image_size)
                        el_image[:, y] *= crack_severity

                results = diagnostics.detect_cell_cracks(el_image, crack_threshold=0.3)

            st.session_state['crack_results'] = results
            st.session_state['el_image'] = el_image

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Cracks Detected", results['num_cracks'])

            with col2:
                st.metric("Affected Area", f"{results['affected_area_percent']:.2f}%")

            with col3:
                severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'none': '⚪'}
                st.metric("Severity", f"{severity_colors.get(results['severity'], '⚪')} {results['severity'].title()}")

            st.metric("Power Loss Estimate", f"{results['power_loss_estimate']:.1f}%")

            st.info(f"**Recommendation:** {results['recommended_action']}")

            # Visualize EL image
            col1, col2 = st.columns(2)

            with col1:
                st.write("**EL Image**")
                fig = go.Figure(data=go.Heatmap(
                    z=el_image,
                    colorscale='Greys',
                    colorbar=dict(title='Intensity')
                ))
                fig.update_layout(
                    title="Electroluminescence Image",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Crack Locations**")
                fig = go.Figure(data=go.Heatmap(
                    z=results['crack_locations'].astype(int),
                    colorscale='Reds',
                    showscale=False
                ))
                fig.update_layout(
                    title="Detected Cracks",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Bypass Diode Failure Detection")

        col1, col2 = st.columns(2)

        with col1:
            num_points = st.slider("I-V Curve Points:", 50, 200, 100)

        with col2:
            simulate_failure = st.checkbox("Simulate Bypass Diode Failure")

        if st.button("⚡ Analyze I-V Curve", key="diode"):
            with st.spinner("Analyzing I-V curve..."):
                # Generate simulated I-V curve
                voltage = np.linspace(0, 40, num_points)

                # Normal I-V curve
                isc = 9.0
                voc = 40.0
                current = isc * (1 - (voltage / voc) ** 2)

                # Add bypass diode failure effect
                if simulate_failure:
                    # Add a step in the curve
                    step_idx = num_points // 2
                    current[step_idx:] *= 0.7

                iv_curve_data = {
                    'voltage': voltage,
                    'current': current
                }

                results = diagnostics.detect_bypass_diode_failure(iv_curve_data)

            st.session_state['diode_results'] = results
            st.session_state['iv_curve'] = iv_curve_data

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                detection_status = "⚠️ Detected" if results['detected'] else "✅ Normal"
                st.metric("Bypass Diode Status", detection_status)

            with col2:
                st.metric("I-V Curve Steps", results['num_steps'])

            with col3:
                st.metric("Power Peaks", results['num_power_peaks'])

            if results['detected']:
                st.error(f"**Power Loss:** {results['power_loss_estimate']:.1f}%")
                st.warning(f"**Recommendation:** {results['recommended_action']}")
            else:
                st.success(f"**Status:** {results['recommended_action']}")

            # Plot I-V and P-V curves
            voltage = iv_curve_data['voltage']
            current = iv_curve_data['current']
            power = voltage * current

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('I-V Curve', 'P-V Curve')
            )

            # I-V curve
            fig.add_trace(
                go.Scatter(x=voltage, y=current, mode='lines',
                          name='I-V Curve', line=dict(color='#3498DB', width=3)),
                row=1, col=1
            )

            # P-V curve
            fig.add_trace(
                go.Scatter(x=voltage, y=power, mode='lines',
                          name='P-V Curve', line=dict(color='#2ECC71', width=3)),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Voltage (V)", row=1, col=1)
            fig.update_xaxes(title_text="Voltage (V)", row=1, col=2)
            fig.update_yaxes(title_text="Current (A)", row=1, col=1)
            fig.update_yaxes(title_text="Power (W)", row=1, col=2)

            fig.update_layout(height=400, showlegend=False, template='plotly_white')

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Soiling Detection & Quantification")

        col1, col2 = st.columns(2)

        with col1:
            current_power = st.number_input("Current Power Output (kW):", 0.0, 1000.0, 85.0, 1.0)
            expected_power = st.number_input("Expected Power Output (kW):", 0.0, 1000.0, 100.0, 1.0)

        with col2:
            irradiance = st.slider("Current Irradiance (W/m²):", 0, 1200, 1000, 50)
            recent_rain = st.checkbox("Recent Rainfall?")

        if st.button("🧹 Detect Soiling", key="soiling"):
            results = diagnostics.detect_soiling(current_power, expected_power, irradiance, recent_rain)

            st.session_state['soiling_results'] = results

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Soiling Ratio", f"{results['soiling_ratio']:.3f}")

            with col2:
                st.metric("Soiling Loss", f"{results['soiling_loss_percent']:.2f}%")

            with col3:
                severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'none': '⚪'}
                st.metric("Severity", f"{severity_colors.get(results['severity'], '⚪')} {results['severity'].title()}")

            st.divider()

            st.write("### Economic Impact")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Daily Energy Loss", f"{results['daily_energy_loss_kwh']:.1f} kWh")

            with col2:
                st.metric("Annual Revenue Loss", f"${results['annual_revenue_loss_usd']:,.2f}")

            st.info(f"**Recommendation:** {results['recommended_action']}")

            # Soiling impact visualization
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Expected Power', 'Actual Power', 'Soiling Loss'],
                y=[expected_power, current_power, expected_power - current_power],
                marker_color=['#2ECC71', '#3498DB', '#E74C3C'],
                text=[f"{expected_power:.1f} kW", f"{current_power:.1f} kW", f"{expected_power - current_power:.1f} kW"],
                textposition='auto'
            ))

            fig.update_layout(
                title="Soiling Impact on Power Output",
                xaxis_title="Category",
                yaxis_title="Power (kW)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Delamination & PID Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Delamination Detection**")

            module_age = st.slider("Module Age (years):", 0, 30, 8)
            delam_severity = st.slider("Simulated Delamination:", 0.0, 20.0, 5.0)

            if st.button("🔬 Detect Delamination", key="delam"):
                # Generate simulated visual inspection image
                visual_image = np.random.normal(128, 20, (30, 30))

                # Add delamination effects
                num_spots = int(delam_severity)
                for _ in range(num_spots):
                    x, y = np.random.randint(0, 30, 2)
                    visual_image[x, y] += 100

                results = diagnostics.detect_delamination(visual_image, module_age)

                st.session_state['delam_results'] = results

                st.metric("Affected Area", f"{results['affected_area_percent']:.2f}%")
                st.metric("Severity", results['severity'].title())
                st.metric("Power Loss", f"{results['power_loss_estimate']:.1f}%")

                warranty_status = "✅ Yes" if results['warranty_claim_eligible'] else "❌ No"
                st.metric("Warranty Eligible", warranty_status)

                st.info(f"**Recommendation:** {results['recommended_action']}")

        with col2:
            st.write("**PID Detection**")

            power_deg = st.slider("Observed Power Degradation (%):", 0.0, 50.0, 15.0)
            system_voltage = st.number_input("System Voltage (V):", 0, 1500, 800, 50)
            humidity = st.slider("Average Humidity (%):", 0, 100, 70)
            temperature = st.slider("Average Temperature (°C):", 0, 50, 30)

            if st.button("⚡ Detect PID", key="pid"):
                results = diagnostics.detect_pid(power_deg, system_voltage, humidity, temperature)

                st.session_state['pid_results'] = results

                st.metric("PID Risk Score", f"{results['pid_risk_score']:.0f}/100")
                st.metric("Combined Score", f"{results['combined_score']:.0f}/100")
                st.metric("Severity", results['severity'].title())
                st.metric("Power Loss", f"{results['power_loss_estimate']:.1f}%")

                reversible_status = "✅ Yes" if results['reversible'] else "❌ No"
                st.metric("Reversible", reversible_status)

                st.info(f"**Recommendation:** {results['recommended_action']}")

                if results['mitigation_options']:
                    st.write("**Mitigation Options:**")
                    for option in results['mitigation_options']:
                        st.write(f"• {option}")

    with tab6:
        st.subheader("Comprehensive Fault Report")

        st.write("### Generate System Health Report")

        if st.button("📋 Generate Fault Report", key="report"):
            # Collect all fault results from session state
            all_faults = []

            if 'hotspot_results' in st.session_state:
                fault = st.session_state['hotspot_results'].copy()
                fault['fault_type'] = 'hotspot'
                all_faults.append(fault)

            if 'crack_results' in st.session_state:
                fault = st.session_state['crack_results'].copy()
                fault['fault_type'] = 'cell_crack'
                all_faults.append(fault)

            if 'diode_results' in st.session_state:
                fault = st.session_state['diode_results'].copy()
                fault['fault_type'] = 'bypass_diode'
                all_faults.append(fault)

            if 'soiling_results' in st.session_state:
                fault = st.session_state['soiling_results'].copy()
                fault['fault_type'] = 'soiling'
                all_faults.append(fault)

            if 'delam_results' in st.session_state:
                fault = st.session_state['delam_results'].copy()
                fault['fault_type'] = 'delamination'
                all_faults.append(fault)

            if 'pid_results' in st.session_state:
                fault = st.session_state['pid_results'].copy()
                fault['fault_type'] = 'pid'
                all_faults.append(fault)

            if len(all_faults) == 0:
                st.warning("⚠️ No fault detection results available. Please run fault detection tests first.")
            else:
                report = diagnostics.generate_fault_report(all_faults)

                # Display overall health
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("System Health Score", f"{report['health_score']:.1f}/100")

                with col2:
                    st.metric("Health Status", report['health_status'])

                with col3:
                    st.metric("Total Power Loss", f"{report['total_power_loss_estimate']:.1f}%")

                # Severity distribution
                st.write("### Fault Severity Distribution")

                severity_data = report['severity_counts']
                severity_labels = list(severity_data.keys())
                severity_values = list(severity_data.values())

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=severity_labels,
                    y=severity_values,
                    marker_color=['#E74C3C', '#E67E22', '#F39C12', '#2ECC71', '#95A5A6'],
                    text=severity_values,
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Fault Severity Breakdown",
                    xaxis_title="Severity Level",
                    yaxis_title="Number of Faults",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # High priority faults
                if report['high_priority_faults']:
                    st.write("### High Priority Faults (Immediate Action Required)")

                    for fault in report['high_priority_faults']:
                        fault_type = fault.get('fault_type', 'Unknown')
                        severity = fault.get('severity', 'unknown')
                        power_loss = fault.get('power_loss_estimate', 0)
                        action = fault.get('recommended_action', 'No action specified')

                        st.error(f"""
                        **Fault Type:** {fault_type.replace('_', ' ').title()}
                        **Severity:** {severity.title()}
                        **Power Loss:** {power_loss:.1f}%
                        **Action:** {action}
                        """)

                # Detailed fault summary
                st.write("### All Detected Faults")

                fault_summary = []
                for fault in all_faults:
                    fault_summary.append({
                        'Fault Type': fault.get('fault_type', 'Unknown').replace('_', ' ').title(),
                        'Severity': fault.get('severity', 'none').title(),
                        'Power Loss (%)': f"{fault.get('power_loss_estimate', 0):.1f}",
                        'Detected': '✅ Yes' if fault.get('detected', False) else '❌ No'
                    })

                st.dataframe(pd.DataFrame(fault_summary), use_container_width=True)

    st.divider()
    st.info("💡 **Fault Detection & Diagnostics** - Branch B08 | Advanced AI-Powered Fault Analysis")
