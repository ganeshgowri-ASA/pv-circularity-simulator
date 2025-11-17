"""Advanced Visualizations Page - Interactive Charts and Correlation Analysis."""

import streamlit as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType
from src.ui.dashboard import EYADashboard
from src.ui.visualizations import InteractiveVisualizations

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

st.title("📈 Advanced Visualizations")
st.markdown("Interactive charts and correlation analysis")
st.markdown("---")

# Configuration
project_info = ProjectInfo(
    project_name="Solar PV Project",
    location="San Francisco, CA",
    latitude=37.7749,
    longitude=-122.4194,
    commissioning_date=datetime(2024, 1, 1),
    project_lifetime=25,
)

system_config = SystemConfiguration(
    capacity_dc=1000.0,
    capacity_ac=850.0,
    module_type=ModuleType.MONO_SI,
    module_efficiency=0.20,
    module_count=5000,
    tilt_angle=30.0,
    azimuth_angle=180.0,
)

# Initialize dashboard
dashboard = EYADashboard(project_info, system_config)
viz = InteractiveVisualizations()

# Visualization selector
st.sidebar.markdown("### Visualization Options")
viz_type = st.sidebar.selectbox(
    "Select Visualization",
    [
        "Monthly Production Charts",
        "Performance Heatmap",
        "Weather Correlation",
        "Loss Waterfall",
        "Annual Degradation",
        "All Visualizations",
    ],
)

# Calculate data
with st.spinner("Calculating data..."):
    energy_data = dashboard.annual_energy_output()
    pr_data = dashboard.performance_ratio()
    losses_data = dashboard.losses_waterfall()

# Monthly Production Charts
if viz_type in ["Monthly Production Charts", "All Visualizations"]:
    st.markdown("### 📊 Monthly Production Analysis")

    monthly_chart = viz.monthly_production_charts(
        energy_data["monthly_data"],
        title="Monthly Energy Production Breakdown",
    )
    st.plotly_chart(monthly_chart, use_container_width=True)

    if viz_type != "All Visualizations":
        with st.expander("ℹ️ About Monthly Production Charts"):
            st.markdown("""
            This multi-panel visualization shows:

            1. **Monthly AC Energy Production**: Bar chart of total AC energy per month
            2. **Specific Yield**: Line chart showing kWh/kWp production trend
            3. **Capacity Factor**: Monthly capacity factor as a percentage
            4. **DC vs AC Energy**: Comparison of DC and AC energy production

            **Insights:**
            - Identify seasonal patterns in energy production
            - Detect underperforming months
            - Understand the relationship between DC and AC energy
            - Monitor capacity factor trends
            """)

    if viz_type == "All Visualizations":
        st.markdown("---")

# Performance Heatmap
if viz_type in ["Performance Heatmap", "All Visualizations"]:
    st.markdown("### 🔥 Performance Heatmap")

    heatmap_chart = viz.performance_heatmap(
        energy_data["monthly_data"],
        title="Monthly Performance Metrics Heatmap",
    )
    st.plotly_chart(heatmap_chart, use_container_width=True)

    if viz_type != "All Visualizations":
        with st.expander("ℹ️ About Performance Heatmap"):
            st.markdown("""
            The heatmap shows normalized performance across months for key metrics:

            - **AC Energy**: Total AC energy production
            - **Specific Yield**: Energy per installed kWp
            - **Capacity Factor**: Utilization of system capacity

            **Color Scale:**
            - 🟢 Green: High performance (good)
            - 🟡 Yellow: Medium performance
            - 🔴 Red: Low performance (needs attention)

            **Use Cases:**
            - Quick identification of performance trends
            - Seasonal pattern recognition
            - Comparison across different metrics
            """)

    if viz_type == "All Visualizations":
        st.markdown("---")

# Weather Correlation
if viz_type in ["Weather Correlation", "All Visualizations"]:
    st.markdown("### 🌤️ Weather Correlation Analysis")

    # Generate weather correlation plots
    if dashboard._weather_data and dashboard._energy_outputs:
        weather_chart = viz.weather_correlation_plots(
            dashboard._weather_data,
            dashboard._energy_outputs,
        )
        st.plotly_chart(weather_chart, use_container_width=True)

        if viz_type != "All Visualizations":
            with st.expander("ℹ️ About Weather Correlation"):
                st.markdown("""
                This analysis shows the relationship between weather conditions and energy production:

                1. **Energy vs GHI (Global Horizontal Irradiance)**
                   - Strong positive correlation expected
                   - Points colored by temperature

                2. **Energy vs Temperature**
                   - Typically shows negative correlation
                   - Higher temperatures reduce module efficiency

                3. **Energy vs Wind Speed**
                   - Complex relationship
                   - Wind can cool modules (positive) or cause vibration (negative)

                4. **Daily Energy Profile**
                   - Average production by hour of day
                   - Shows typical generation pattern

                **Applications:**
                - Validate simulation models
                - Identify anomalies
                - Understand environmental impacts
                - Optimize system design
                """)
    else:
        st.info("Weather data not available. Generating correlation plots requires weather data.")

    if viz_type == "All Visualizations":
        st.markdown("---")

# Loss Waterfall
if viz_type in ["Loss Waterfall", "All Visualizations"]:
    st.markdown("### 🔻 Energy Loss Waterfall")

    waterfall_chart = viz.loss_breakdown_sankey(
        losses_data["waterfall_data"],
        title="System Energy Loss Waterfall",
    )
    st.plotly_chart(waterfall_chart, use_container_width=True)

    # Loss summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total System Loss", losses_data["Total System Loss"])

    with col2:
        categories = losses_data["Loss Categories"]
        st.metric("Largest Category", f"{max(categories.values())}")

    with col3:
        detailed = losses_data["Detailed Breakdown"]
        # Find largest individual loss
        loss_values = {k: v for k, v in detailed.items() if k != "total_loss"}
        max_loss = max(loss_values.items(), key=lambda x: x[1])
        st.metric("Largest Individual Loss", f"{max_loss[1]:.2f}% ({max_loss[0].replace('_', ' ').title()})")

    if viz_type != "All Visualizations":
        with st.expander("ℹ️ About Loss Waterfall"):
            st.markdown("""
            The Sankey diagram shows the flow of energy through the system with losses at each stage:

            **Loss Categories:**
            - Optical: Soiling, shading, snow
            - Electrical: Mismatch, wiring, inverter
            - Degradation: LID, nameplate, age
            - Environmental: Temperature effects
            - System: Availability and downtime

            **Reading the Diagram:**
            - Width of flows represents energy amount
            - Each stage shows cumulative effect of losses
            - Final value is net AC energy delivered

            **Optimization:**
            - Focus on largest loss contributors first
            - Consider cost-benefit of mitigation strategies
            - Monitor losses over time for degradation trends
            """)

    if viz_type == "All Visualizations":
        st.markdown("---")

# Annual Degradation
if viz_type in ["Annual Degradation", "All Visualizations"]:
    st.markdown("### 📉 Annual Degradation Projection")

    # Need financial data for degradation
    from src.models.eya_models import FinancialMetrics

    financial_params = FinancialMetrics(
        capex=1000000.0,
        opex_annual=15000.0,
        energy_price=0.12,
        degradation_rate=0.005,
        discount_rate=0.05,
    )

    financial_data = dashboard.financial_metrics(financial_params)
    degradation_chart = viz.annual_degradation_chart(
        financial_data["degradation_data"],
        title="25-Year Energy Production Projection",
    )
    st.plotly_chart(degradation_chart, use_container_width=True)

    # Degradation stats
    deg_df = financial_data["degradation_data"]

    col1, col2, col3 = st.columns(3)

    with col1:
        year1_energy = deg_df.iloc[0]["annual_energy_kwh"]
        st.metric("Year 1 Production", f"{year1_energy:,.0f} kWh")

    with col2:
        year25_energy = deg_df.iloc[-1]["annual_energy_kwh"]
        total_degradation = ((year1_energy - year25_energy) / year1_energy) * 100
        st.metric("Year 25 Production", f"{year25_energy:,.0f} kWh", delta=f"-{total_degradation:.1f}%")

    with col3:
        total_lifetime = deg_df.iloc[-1]["cumulative_energy_kwh"]
        st.metric("Lifetime Production", f"{total_lifetime:,.0f} kWh")

    if viz_type != "All Visualizations":
        with st.expander("ℹ️ About Degradation Projection"):
            st.markdown("""
            This visualization shows expected energy production over the project lifetime accounting for:

            **Degradation Factors:**
            - Annual degradation rate (typically 0.3-0.8%/year)
            - Cumulative effect over project lifetime
            - Impact on total energy production

            **Key Insights:**
            - **Left Chart**: Annual energy production trend
            - **Right Chart**: Cumulative lifetime production

            **Planning Implications:**
            - Revenue projections
            - Maintenance planning
            - Replacement timing
            - Performance guarantees

            **Typical Degradation Rates:**
            - Crystalline Silicon: 0.5%/year
            - Thin Film: 0.4-0.6%/year
            - Premium Modules: <0.3%/year
            """)

# Additional Analysis Tools
st.markdown("---")
st.markdown("### 🔧 Custom Analysis Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Altair Chart Builder")

    chart_type = st.selectbox("Chart Type", ["bar", "line", "scatter"])
    x_col = st.selectbox("X-Axis", energy_data["monthly_data"].columns.tolist(), index=0)
    y_col = st.selectbox("Y-Axis", energy_data["monthly_data"].columns.tolist(), index=2)

    if st.button("Generate Custom Chart"):
        custom_chart = viz.create_altair_chart(
            energy_data["monthly_data"],
            x=x_col,
            y=y_col,
            chart_type=chart_type,
        )
        st.altair_chart(custom_chart, use_container_width=True)

with col2:
    st.markdown("#### Export Visualization Data")

    st.info("""
    Export data for external analysis:
    - Monthly production data
    - Performance metrics
    - Loss breakdown
    - Weather correlations
    """)

    if st.button("📊 Export to CSV"):
        import pandas as pd

        csv = energy_data["monthly_data"].to_csv(index=False)
        st.download_button(
            label="⬇️ Download Monthly Data CSV",
            data=csv,
            file_name=f"monthly_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# Tips and Best Practices
with st.expander("💡 Visualization Best Practices"):
    st.markdown("""
    ### Making the Most of Visualizations

    #### 1. Compare Across Time Periods
    - Look for seasonal patterns
    - Identify trends and anomalies
    - Benchmark against expectations

    #### 2. Cross-Reference Multiple Charts
    - Combine weather correlation with monthly production
    - Link loss analysis to performance metrics
    - Use degradation projections for financial planning

    #### 3. Interactive Features
    - Hover over data points for details
    - Zoom in on specific time periods
    - Click legend items to show/hide data series

    #### 4. Export and Share
    - Download charts as images (camera icon)
    - Export data to CSV for further analysis
    - Include in reports and presentations

    #### 5. Regular Monitoring
    - Track key metrics over time
    - Set performance benchmarks
    - Investigate significant deviations
    - Use for predictive maintenance

    ### Common Use Cases

    **Project Development:**
    - Validate energy estimates
    - Support financing applications
    - Communicate with stakeholders

    **Operations:**
    - Monitor actual vs expected performance
    - Identify maintenance needs
    - Optimize cleaning schedules

    **Asset Management:**
    - Track long-term degradation
    - Plan for component replacement
    - Maximize lifetime value
    """)
