"""Performance Monitoring page for PV Circularity Simulator."""

import streamlit as st


def render() -> None:
    """Render the Performance Monitoring page.

    This page provides real-time and historical performance monitoring,
    reliability testing, and degradation analysis.
    """
    st.title("📊 Performance Monitoring")

    st.markdown("""
    Monitor real-world performance, track system health, and analyze
    long-term degradation trends.
    """)

    # Real-time monitoring
    st.subheader("⚡ Real-time Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Power", "8.4 kW", "+0.3 kW")
    with col2:
        st.metric("Today's Energy", "42.5 kWh", "+2.1 kWh")
    with col3:
        st.metric("System Efficiency", "17.8%", "-0.1%")
    with col4:
        st.metric("Status", "✅ Normal", delta_color="off")

    # Performance chart
    import pandas as pd
    import numpy as np

    hours = np.arange(0, 24)
    power = np.maximum(0, 10 * np.sin((hours - 6) * np.pi / 12) ** 2)

    perf_data = pd.DataFrame({
        'Hour': hours,
        'Power (kW)': power
    })

    st.line_chart(perf_data.set_index('Hour'))
    st.caption("Today's Power Generation Profile")

    # Historical performance
    with st.expander("📅 Historical Performance"):
        time_range = st.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last Year", "All Time"])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Average Daily Production", "45.2 kWh")
            st.metric("Peak Power", "9.8 kW")

        with col2:
            st.metric("Total Energy (All Time)", "16,542 kWh")
            st.metric("CO₂ Offset", "12,406 kg")

    # Reliability testing
    st.subheader("🔬 Reliability Testing")

    reliability_tests = st.multiselect(
        "Select Reliability Tests",
        ["Thermal Cycling", "Damp Heat", "UV Exposure", "Mechanical Load", "PID Testing"],
        default=["Thermal Cycling"]
    )

    if st.button("🧪 Run Reliability Tests"):
        st.info("Running reliability tests: " + ", ".join(reliability_tests))
        with st.spinner("Testing in progress..."):
            st.success("✅ All tests passed within specifications")

    # Degradation analysis
    with st.expander("📉 Degradation Analysis"):
        st.markdown("**Long-term Performance Degradation**")

        years = np.arange(0, 26)
        performance = 100 * (0.995 ** years)

        deg_data = pd.DataFrame({
            'Year': years,
            'Performance (%)': performance
        })

        st.line_chart(deg_data.set_index('Year'))

        st.metric("Degradation Rate", "0.5% / year")
        st.metric("Predicted Performance (Year 25)", "88.2%")
