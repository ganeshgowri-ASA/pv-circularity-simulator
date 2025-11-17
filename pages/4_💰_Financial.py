"""Financial Analysis Page - LCOE, NPV, IRR, and Economic Metrics."""

import streamlit as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType, FinancialMetrics
from src.ui.dashboard import EYADashboard
from src.ui.visualizations import InteractiveVisualizations

st.set_page_config(page_title="Financial Analysis", page_icon="💰", layout="wide")

st.title("💰 Financial Analysis")
st.markdown("Economic metrics and investment analysis for your PV project")
st.markdown("---")

# Financial Parameters Input
with st.sidebar:
    st.subheader("Financial Parameters")

    capex = st.number_input("CAPEX ($)", value=1000000.0, min_value=0.0, step=10000.0, help="Total capital expenditure")
    opex_annual = st.number_input("Annual OPEX ($)", value=15000.0, min_value=0.0, step=1000.0, help="Annual operational expenditure")
    energy_price = st.number_input("Energy Price ($/kWh)", value=0.12, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
    degradation_rate = st.number_input("Degradation Rate (%/year)", value=0.5, min_value=0.0, max_value=2.0, step=0.1) / 100
    discount_rate = st.number_input("Discount Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.5) / 100

    capacity_dc = st.number_input("DC Capacity (kWp)", value=1000.0, min_value=1.0)

# Create configuration
project_info = ProjectInfo(
    project_name="Solar PV Project",
    location="San Francisco, CA",
    latitude=37.7749,
    longitude=-122.4194,
    commissioning_date=datetime(2024, 1, 1),
    project_lifetime=25,
)

system_config = SystemConfiguration(
    capacity_dc=capacity_dc,
    capacity_ac=850.0,
    module_type=ModuleType.MONO_SI,
    module_efficiency=0.20,
    module_count=5000,
    tilt_angle=30.0,
    azimuth_angle=180.0,
)

financial_params = FinancialMetrics(
    capex=capex,
    opex_annual=opex_annual,
    energy_price=energy_price,
    degradation_rate=degradation_rate,
    discount_rate=discount_rate,
)

# Initialize dashboard
dashboard = EYADashboard(project_info, system_config)

# Calculate financial metrics
with st.spinner("Calculating financial metrics..."):
    financial_data = dashboard.financial_metrics(financial_params)

st.success("✅ Financial analysis complete!")

# Key Financial Metrics
st.markdown("### 💵 Key Financial Metrics")

col1, col2, col3, col4 = st.columns(4)

econ_metrics = financial_data["Economic Metrics"]

with col1:
    lcoe = econ_metrics["LCOE"]
    st.metric(
        "LCOE",
        lcoe,
        delta=f"vs ${energy_price:.4f}/kWh price",
        help="Levelized Cost of Energy"
    )

with col2:
    npv = econ_metrics["NPV"]
    st.metric(
        "NPV",
        npv,
        delta="Positive" if "(" not in npv else "Negative",
        help="Net Present Value"
    )

with col3:
    irr = econ_metrics["IRR"]
    st.metric(
        "IRR",
        irr,
        delta=f"vs {discount_rate*100:.1f}% discount rate",
        help="Internal Rate of Return"
    )

with col4:
    payback = econ_metrics["Payback Period"]
    st.metric(
        "Payback Period",
        payback,
        help="Simple payback period"
    )

# Investment Overview
st.markdown("### 📊 Investment Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Capital Investment")

    inv_analysis = financial_data["Investment Analysis"]

    st.markdown(f"""
    - **Initial Investment (CAPEX)**: {inv_analysis["CAPEX"]}
    - **Annual Operating Cost (OPEX)**: {inv_analysis["Annual OPEX"]}
    - **Lifetime Operating Cost**: {inv_analysis["Total Lifetime OPEX"]}
    - **Energy Tariff**: {inv_analysis["Energy Price"]}
    """)

    # Calculate specific costs
    capex_per_kw = capex / capacity_dc
    st.metric("CAPEX per kWp", f"${capex_per_kw:,.2f}")

with col2:
    st.markdown("#### Revenue Projections")

    revenue_proj = financial_data["Revenue Projections"]

    st.markdown(f"""
    - **First Year Revenue**: {revenue_proj["First Year Revenue"]}
    - **Lifetime Energy Production**: {revenue_proj["Lifetime Energy Production"]}
    - **Lifetime Revenue**: {revenue_proj["Lifetime Revenue"]}
    """)

    # Calculate ROI
    total_investment = capex + (opex_annual * project_info.project_lifetime)
    lifetime_revenue_val = float(revenue_proj["Lifetime Revenue"].replace("$", "").replace(",", ""))
    roi = ((lifetime_revenue_val - total_investment) / total_investment) * 100
    st.metric("ROI", f"{roi:.1f}%")

# Annual Cash Flow
st.markdown("### 💸 Annual Cash Flow Projection")

degradation_df = financial_data["degradation_data"]

import pandas as pd

# Calculate cash flows
cash_flow_data = []
for idx, row in degradation_df.iterrows():
    year = row["year"]
    annual_energy = row["annual_energy_kwh"]
    revenue = annual_energy * energy_price
    cost = opex_annual
    net_cash_flow = revenue - cost

    if year == 1:
        net_cash_flow -= capex  # Initial investment

    cumulative_cash_flow = sum([
        (degradation_df.loc[i, "annual_energy_kwh"] * energy_price - opex_annual)
        for i in range(year)
    ]) - capex

    cash_flow_data.append({
        "Year": year,
        "Energy (kWh)": annual_energy,
        "Revenue ($)": revenue,
        "OPEX ($)": cost,
        "Net Cash Flow ($)": net_cash_flow,
        "Cumulative ($)": cumulative_cash_flow,
    })

cash_flow_df = pd.DataFrame(cash_flow_data)

# Visualize degradation
viz = InteractiveVisualizations()
degradation_chart = viz.annual_degradation_chart(degradation_df)
st.plotly_chart(degradation_chart, use_container_width=True)

# Cash flow table
with st.expander("📋 View Detailed Cash Flow Table"):
    st.dataframe(
        cash_flow_df.style.format({
            "Energy (kWh)": "{:,.0f}",
            "Revenue ($)": "${:,.0f}",
            "OPEX ($)": "${:,.0f}",
            "Net Cash Flow ($)": "${:,.0f}",
            "Cumulative ($)": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# Financial Analysis Summary
st.markdown("### 📈 Financial Analysis Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Project Economics")

    # Parse values
    lcoe_val = float(lcoe.replace("$", "").replace("/kWh", ""))
    npv_val = float(npv.replace("$", "").replace(",", "").replace("N/A", "0"))
    irr_val = float(irr.replace("%", "").replace("N/A", "0"))

    if lcoe_val < energy_price:
        st.success(f"✅ **Profitable Project**: LCOE (${lcoe_val:.4f}/kWh) is below energy price (${energy_price:.4f}/kWh)")
    else:
        st.warning(f"⚠️ **Review Needed**: LCOE (${lcoe_val:.4f}/kWh) exceeds energy price (${energy_price:.4f}/kWh)")

    if npv_val > 0:
        st.success(f"✅ **Positive NPV**: ${npv_val:,.0f} - Project adds value")
    else:
        st.warning("⚠️ **Negative NPV**: Project may not be economically viable")

    if irr_val > discount_rate * 100:
        st.success(f"✅ **Attractive IRR**: {irr_val:.2f}% exceeds discount rate ({discount_rate*100:.1f}%)")
    else:
        st.warning(f"⚠️ **Low IRR**: {irr_val:.2f}% is below discount rate ({discount_rate*100:.1f}%)")

with col2:
    st.markdown("#### Key Takeaways")

    payback_val = float(payback.replace(" years", "").replace("N/A", "0"))

    st.markdown(f"""
    - **Investment Efficiency**: ${capex_per_kw:,.2f} per kWp
    - **Return Period**: {payback_val:.1f} years
    - **Project Duration**: {project_info.project_lifetime} years
    - **Break-even Year**: Year {int(payback_val)} (approximately)
    - **Post-payback Earnings**: {project_info.project_lifetime - int(payback_val)} years
    """)

# Sensitivity to Energy Price
st.markdown("### 🎯 Sensitivity to Energy Price")

price_range = [energy_price * 0.7, energy_price * 0.85, energy_price, energy_price * 1.15, energy_price * 1.3]
sensitivity_data = []

for price in price_range:
    temp_financial = FinancialMetrics(
        capex=capex,
        opex_annual=opex_annual,
        energy_price=price,
        degradation_rate=degradation_rate,
        discount_rate=discount_rate,
    )

    annual_energy = float(revenue_proj["First Year Revenue"].replace("$", "").replace(",", "")) / energy_price

    temp_result = dashboard.analyzer.calculate_financial_metrics(annual_energy, temp_financial)

    sensitivity_data.append({
        "Energy Price ($/kWh)": price,
        "LCOE ($/kWh)": temp_result.lcoe,
        "NPV ($)": temp_result.npv,
        "IRR (%)": temp_result.irr * 100 if temp_result.irr else 0,
        "Payback (years)": temp_result.payback_period,
    })

sensitivity_df = pd.DataFrame(sensitivity_data)

st.dataframe(
    sensitivity_df.style.format({
        "Energy Price ($/kWh)": "${:.4f}",
        "LCOE ($/kWh)": "${:.4f}",
        "NPV ($)": "${:,.0f}",
        "IRR (%)": "{:.2f}%",
        "Payback (years)": "{:.1f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Recommendations
with st.expander("💡 Financial Optimization Recommendations"):
    st.markdown("""
    ### Ways to Improve Project Economics

    #### Reduce CAPEX
    - Competitive bidding for equipment and installation
    - Consider lower-cost mounting solutions
    - Optimize system design to reduce balance of system costs
    - Take advantage of economies of scale

    #### Reduce OPEX
    - Implement predictive maintenance
    - Use remote monitoring to reduce site visits
    - Negotiate long-term service contracts
    - Train in-house staff for routine maintenance

    #### Increase Revenue
    - Optimize system design for maximum energy production
    - Implement module-level optimization
    - Regular cleaning and maintenance
    - Explore energy storage for time-of-use optimization
    - Consider renewable energy certificates (RECs)

    #### Financial Structure
    - Explore tax incentives and rebates
    - Consider power purchase agreements (PPAs)
    - Investigate green bonds or favorable financing
    - Look into investment tax credits (ITC) or production tax credits (PTC)
    """)
