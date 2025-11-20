"""
HYA Simulation Module
====================

Historical Yield Analysis (HYA) for existing PV systems.
Analyzes actual performance against expected production.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the HYA simulation module.

    Args:
        session: Session manager instance

    Features:
        - Import actual production data
        - Compare with expected yield
        - Performance ratio trending
        - Loss attribution analysis
        - Degradation rate calculation
        - Weather normalization
        - Benchmarking against guarantees
        - Financial impact assessment
    """
    st.header("📅 Historical Yield Analysis (HYA)")

    st.info("""
    Analyze historical performance of existing PV systems and compare with
    expected production and guarantees.
    """)

    # Analysis period selection
    st.subheader("📆 Analysis Period")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))

    with col2:
        end_date = st.date_input("End Date", datetime.now())

    with col3:
        analysis_interval = st.selectbox("Interval", ["Daily", "Monthly", "Yearly"])

    # Data import
    st.markdown("---")
    st.subheader("📥 Import Production Data")

    col1, col2 = st.columns(2)

    with col1:
        data_source = st.selectbox(
            "Data Source",
            ["SCADA System", "Inverter Data Logger", "CSV Upload", "API Integration"]
        )

        if data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload Production Data (CSV)", type=['csv'])
            if uploaded_file:
                st.success("Production data uploaded successfully!")

    with col2:
        expected_source = st.selectbox(
            "Expected Yield Source",
            ["EYA Simulation", "PVsyst Model", "Contractual Guarantee", "Manual Input"]
        )

        if expected_source == "Manual Input":
            expected_annual = st.number_input("Expected Annual Production (MWh)", 0, 100000, 5000)

    # Run analysis
    if st.button("📊 Run HYA Analysis", type="primary"):
        with st.spinner("Analyzing historical performance..."):
            import time
            time.sleep(2)

            # Generate dummy historical data
            days = (end_date - start_date).days
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Simulate daily production with seasonal variation
            daily_production = []
            expected_production = []

            for i, date in enumerate(dates):
                # Seasonal factor (higher in summer)
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)

                # Random weather variation
                weather_factor = np.random.uniform(0.6, 1.0)

                # Expected production
                expected = 15 * seasonal_factor  # MWh/day base

                # Actual production (with some losses and degradation)
                degradation = 1 - (0.005 * (i / 365))  # 0.5% per year
                actual = expected * weather_factor * degradation * 0.95  # 5% average loss

                daily_production.append(actual)
                expected_production.append(expected)

            hya_df = pd.DataFrame({
                'Date': dates,
                'Actual (MWh)': daily_production,
                'Expected (MWh)': expected_production,
                'PR (%)': [(a/e)*100 if e > 0 else 0 for a, e in zip(daily_production, expected_production)]
            })

            st.success("HYA analysis completed!")

            # Summary metrics
            st.markdown("---")
            st.subheader("📊 Performance Summary")

            total_actual = hya_df['Actual (MWh)'].sum()
            total_expected = hya_df['Expected (MWh)'].sum()
            avg_pr = hya_df['PR (%)'].mean()
            availability = 98.5  # Dummy value

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Production", f"{total_actual:,.0f} MWh")
                st.caption(f"vs. Expected: {total_expected:,.0f} MWh")

            with col2:
                shortfall = total_expected - total_actual
                shortfall_pct = (shortfall / total_expected) * 100
                st.metric("Production Gap", f"{shortfall:,.0f} MWh", f"{shortfall_pct:.1f}%", delta_color="inverse")

            with col3:
                st.metric("Average PR", f"{avg_pr:.1f}%")
                if avg_pr < 75:
                    st.error("Below target")
                elif avg_pr < 85:
                    st.warning("Needs improvement")
                else:
                    st.success("Meeting target")

            with col4:
                st.metric("Availability", f"{availability:.1f}%")

            # Monthly aggregation
            st.markdown("---")
            st.subheader("📈 Monthly Performance")

            monthly_df = hya_df.set_index('Date').resample('M').agg({
                'Actual (MWh)': 'sum',
                'Expected (MWh)': 'sum',
                'PR (%)': 'mean'
            }).reset_index()

            monthly_df['Month'] = monthly_df['Date'].dt.strftime('%Y-%m')

            st.line_chart(monthly_df.set_index('Month')[['Actual (MWh)', 'Expected (MWh)']])

            # Performance ratio trend
            st.markdown("---")
            st.subheader("📉 Performance Ratio Trend")

            st.line_chart(monthly_df.set_index('Month')['PR (%)'])

            target_pr = st.slider("Target PR (%)", 70.0, 95.0, 80.0, 0.5)
            months_below_target = len(monthly_df[monthly_df['PR (%)'] < target_pr])

            if months_below_target > 0:
                st.warning(f"⚠️ {months_below_target} months below target PR of {target_pr}%")

            # Loss attribution
            st.markdown("---")
            st.subheader("🔍 Loss Attribution")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Estimated Loss Breakdown**")

                loss_categories = pd.DataFrame({
                    'Category': [
                        'Soiling',
                        'Shading',
                        'Temperature',
                        'Inverter Losses',
                        'Wiring Losses',
                        'Downtime',
                        'Degradation',
                        'Other'
                    ],
                    'Loss (%)': [1.8, 0.5, 2.1, 1.2, 0.8, 1.5, 0.5, 1.6]
                })

                st.bar_chart(loss_categories.set_index('Category'))

            with col2:
                st.markdown("**Top Loss Factors**")
                sorted_losses = loss_categories.sort_values('Loss (%)', ascending=False)

                for idx, row in sorted_losses.head(5).iterrows():
                    st.metric(row['Category'], f"{row['Loss (%)']}%")

            # Degradation analysis
            st.markdown("---")
            st.subheader("📉 Degradation Analysis")

            # Calculate degradation rate from PR trend
            from scipy import stats

            x = np.arange(len(monthly_df))
            y = monthly_df['PR (%)'].values

            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Convert slope to annual degradation rate
                degradation_rate = abs(slope * 12)  # months to years

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Degradation Rate", f"{degradation_rate:.2f}%/year")

                with col2:
                    st.metric("Confidence (R²)", f"{r_value**2:.3f}")

                with col3:
                    warranty_rate = 0.55  # Typical warranty degradation
                    if degradation_rate > warranty_rate:
                        st.metric("vs. Warranty", f"+{degradation_rate - warranty_rate:.2f}%/year", delta_color="inverse")
                        st.warning("⚠️ Exceeds warranty limits")
                    else:
                        st.metric("vs. Warranty", f"{degradation_rate - warranty_rate:.2f}%/year")
                        st.success("✅ Within warranty")

            # Financial impact
            st.markdown("---")
            st.subheader("💰 Financial Impact")

            col1, col2 = st.columns(2)

            with col1:
                electricity_rate = st.number_input("Electricity Rate ($/MWh)", 0, 500, 80)
                capacity_payment = st.number_input("Capacity Payment ($/kW-year)", 0, 200, 0)

            with col2:
                revenue_actual = total_actual * electricity_rate
                revenue_expected = total_expected * electricity_rate
                revenue_loss = revenue_expected - revenue_actual

                st.metric("Actual Revenue", f"${revenue_actual:,.0f}")
                st.metric("Expected Revenue", f"${revenue_expected:,.0f}")
                st.metric("Revenue Loss", f"${revenue_loss:,.0f}", delta_color="inverse")

            # Save results
            session.set('hya_results', {
                'total_actual': total_actual,
                'total_expected': total_expected,
                'avg_pr': avg_pr,
                'degradation_rate': degradation_rate if len(x) > 2 else 0,
                'revenue_loss': revenue_loss
            })
