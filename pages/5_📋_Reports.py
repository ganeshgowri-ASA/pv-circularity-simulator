"""Reports Page - Generate PDF and Excel Reports."""

import streamlit as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType, FinancialMetrics
from src.ui.dashboard import EYADashboard
from src.ui.reports import ComprehensiveReports

st.set_page_config(page_title="Reports", page_icon="📋", layout="wide")

st.title("📋 Comprehensive Reports")
st.markdown("Generate professional PDF and Excel reports for your PV project")
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

financial_params = FinancialMetrics(
    capex=1000000.0,
    opex_annual=15000.0,
    energy_price=0.12,
    degradation_rate=0.005,
    discount_rate=0.05,
)

# Initialize dashboard
dashboard = EYADashboard(project_info, system_config)
reports = ComprehensiveReports(project_info, system_config, dashboard.analyzer)

# Report Options
st.markdown("### 📝 Report Configuration")

col1, col2 = st.columns(2)

with col1:
    include_financial = st.checkbox("Include Financial Analysis", value=True)
    include_monthly = st.checkbox("Include Monthly Data", value=True)
    include_losses = st.checkbox("Include Loss Analysis", value=True)

with col2:
    include_sensitivity = st.checkbox("Include Sensitivity Analysis", value=True)
    include_probabilistic = st.checkbox("Include P50/P90/P99 Analysis", value=True)

# Generate reports
st.markdown("### 📄 Generate Reports")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### PDF Report")
    st.info("""
    **Professional PDF Report includes:**
    - Project overview and system configuration
    - Annual energy production summary
    - Performance metrics and PR analysis
    - Loss breakdown
    - Financial analysis (if selected)
    - Professional formatting with tables and charts
    """)

    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                # Calculate all data
                energy_data = dashboard.annual_energy_output()
                pr_data = dashboard.performance_ratio()
                annual_energy = sum([output.ac_energy for output in dashboard._energy_outputs])

                financial = None
                if include_financial:
                    financial_data = dashboard.financial_metrics(financial_params)
                    financial = financial_data["financial_metrics"]

                # Generate PDF
                pdf_buffer = reports.eya_pdf_generator(
                    annual_energy=annual_energy,
                    performance_metrics=pr_data["metrics"],
                    financial_metrics=financial if include_financial else None,
                )

                st.success("✅ PDF report generated successfully!")

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"EYA_Report_{project_info.project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
                st.exception(e)

with col2:
    st.markdown("#### Excel Report")
    st.info("""
    **Comprehensive Excel Workbook includes:**
    - Project Overview sheet
    - Energy Production data
    - Monthly production breakdown
    - Performance metrics
    - Loss analysis
    - Financial analysis (if selected)
    - Multiple formatted worksheets
    """)

    if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
        with st.spinner("Generating Excel report..."):
            try:
                # Calculate all data
                energy_data = dashboard.annual_energy_output()
                pr_data = dashboard.performance_ratio()
                annual_energy = sum([output.ac_energy for output in dashboard._energy_outputs])

                financial = None
                if include_financial:
                    financial_data = dashboard.financial_metrics(financial_params)
                    financial = financial_data["financial_metrics"]

                # Generate Excel
                excel_buffer = reports.excel_export(
                    annual_energy=annual_energy,
                    performance_metrics=pr_data["metrics"],
                    monthly_data=energy_data["monthly_data"],
                    financial_metrics=financial if include_financial else None,
                )

                st.success("✅ Excel report generated successfully!")

                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_buffer,
                    file_name=f"EYA_Report_{project_info.project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Error generating Excel: {str(e)}")
                st.exception(e)

# Sensitivity Analysis
if include_sensitivity:
    st.markdown("---")
    st.markdown("### 🎯 Sensitivity Analysis")

    col1, col2 = st.columns(2)

    with col1:
        param_name = st.selectbox(
            "Parameter to Analyze",
            ["Capacity Factor", "Energy Price", "CAPEX", "Degradation Rate"]
        )

        variation = st.slider("Variation Range (±%)", 5, 50, 20)

    with col2:
        st.markdown("#### Parameter Details")

        base_values = {
            "Capacity Factor": 0.25,
            "Energy Price": 0.12,
            "CAPEX": 1000000.0,
            "Degradation Rate": 0.005,
        }

        base_value = base_values[param_name]
        st.metric("Base Value", f"{base_value}")
        st.metric("Variation Range", f"±{variation}%")

    if st.button("📊 Generate Sensitivity Analysis"):
        with st.spinner("Calculating sensitivity..."):
            try:
                # Calculate base annual energy
                energy_data = dashboard.annual_energy_output()
                annual_energy = sum([output.ac_energy for output in dashboard._energy_outputs])

                # Generate sensitivity table
                parameters = [
                    (param_name, base_value, variation),
                ]

                sensitivity_table = reports.sensitivity_analysis_tables(annual_energy, parameters)

                st.success("✅ Sensitivity analysis complete!")

                st.dataframe(
                    sensitivity_table.style.format({
                        "Base Value": "{:.4f}",
                        "Test Value": "{:.4f}",
                        "Change (%)": "{:.2f}%",
                        "Annual Energy (kWh)": "{:,.0f}",
                        "Energy Change (%)": "{:.2f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Visualization
                from src.ui.visualizations import InteractiveVisualizations
                viz = InteractiveVisualizations()
                sens_chart = viz.sensitivity_chart(sensitivity_table)
                st.plotly_chart(sens_chart, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Probabilistic Analysis
if include_probabilistic:
    st.markdown("---")
    st.markdown("### 📈 Probabilistic Analysis (P50/P90/P99)")

    col1, col2 = st.columns(2)

    with col1:
        uncertainty = st.slider("Total Uncertainty (%)", 5, 20, 10)
        num_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=2)

    with col2:
        st.info("""
        **Probabilistic Analysis** uses Monte Carlo simulation to estimate
        the probability of different energy production outcomes.

        - **P99**: 99% probability of exceeding this value
        - **P90**: 90% probability of exceeding this value
        - **P50**: 50% probability (median)
        """)

    if st.button("🎲 Generate Probabilistic Analysis"):
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Calculate base annual energy
                energy_data = dashboard.annual_energy_output()
                annual_energy = sum([output.ac_energy for output in dashboard._energy_outputs])

                # Generate P50/P90/P99 analysis
                prob_analysis = reports.p50_p90_p99_analysis(annual_energy, uncertainty)

                st.success("✅ Probabilistic analysis complete!")

                # Display results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    p99_val = prob_analysis["Exceedance Probability"]["P99 (99%)"]
                    st.metric("P99", f"{p99_val:,.0f} kWh/year")

                with col2:
                    p90_val = prob_analysis["Exceedance Probability"]["P90 (90%)"]
                    st.metric("P90", f"{p90_val:,.0f} kWh/year")

                with col3:
                    p50_val = prob_analysis["Exceedance Probability"]["P50 (50% - Median)"]
                    st.metric("P50", f"{p50_val:,.0f} kWh/year")

                with col4:
                    mean_val = prob_analysis["Exceedance Probability"]["Mean"]
                    st.metric("Mean", f"{mean_val:,.0f} kWh/year")

                # Detailed table
                st.markdown("#### Exceedance Probability Table")
                st.dataframe(
                    prob_analysis["table"].style.format({
                        "Annual Energy (kWh)": "{:,.0f}",
                        "% of P50": "{:.2f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Visualization
                from src.ui.visualizations import InteractiveVisualizations
                viz = InteractiveVisualizations()
                prob_chart = viz.p50_p90_p99_chart(prob_analysis["table"])
                st.plotly_chart(prob_chart, use_container_width=True)

                # Analysis insights
                st.markdown("#### Analysis Insights")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Risk Assessment**")
                    downside = prob_analysis["Analysis"]["Downside (P50 - P99)"]
                    upside = prob_analysis["Analysis"]["Upside (P90 - P50)"]
                    st.markdown(f"- **Downside Risk (P50-P99)**: {downside:,.0f} kWh/year")
                    st.markdown(f"- **Upside Potential (P90-P50)**: {upside:,.0f} kWh/year")

                with col2:
                    st.markdown("**Statistical Metrics**")
                    std_dev = prob_analysis["Statistics"]["Standard Deviation (kWh)"]
                    cov = prob_analysis["Statistics"]["Coefficient of Variation (%)"]
                    st.markdown(f"- **Standard Deviation**: {std_dev:,.0f} kWh")
                    st.markdown(f"- **Coefficient of Variation**: {cov:.2f}%")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Help Section
with st.expander("ℹ️ About These Reports"):
    st.markdown("""
    ### Report Types

    #### PDF Report
    Professional PDF report suitable for:
    - Client presentations
    - Project documentation
    - Financing applications
    - Technical documentation

    #### Excel Report
    Comprehensive workbook suitable for:
    - Detailed analysis
    - Custom calculations
    - Data sharing with stakeholders
    - Financial modeling

    #### Sensitivity Analysis
    Analyzes how changes in key parameters affect energy production:
    - Identifies critical parameters
    - Quantifies risk exposure
    - Supports decision-making
    - Validates assumptions

    #### Probabilistic Analysis
    Monte Carlo simulation to estimate:
    - Probability distributions of outcomes
    - Risk-adjusted production estimates
    - Confidence intervals
    - P50 (median), P90, and P99 values

    ### Usage Tips

    1. **Generate Multiple Scenarios**: Create reports with different assumptions
    2. **Save for Comparison**: Download reports to compare design alternatives
    3. **Share with Stakeholders**: Use PDF for presentations, Excel for detailed review
    4. **Update Regularly**: Regenerate reports as project parameters change
    """)
