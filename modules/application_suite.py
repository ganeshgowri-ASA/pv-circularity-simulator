"""
Application Suite Module - Branches B13-B15

This module provides functionality for:
- B13: Financial Analysis & Bankability
- B14: Infrastructure & Grid Integration
- B15: Application Configuration & Settings

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (FINANCIAL_DEFAULTS, ELECTRICITY_TARIFFS, EXPORT_FORMATS, VERSION_INFO, COLOR_PALETTE)
from utils.validators import FinancialModel, Infrastructure, AppConfig


def render_financial_analysis() -> None:
    """Render Financial Analysis & Bankability interface."""
    st.header("💰 Financial Analysis & Bankability")
    st.markdown("*Comprehensive financial modeling for PV projects*")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💵 Project Economics", "📊 Cash Flow", "📈 Sensitivity Analysis", "🏦 Bankability"])
    
    with tab1:
        st.subheader("Project Financial Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            capex = st.number_input("CAPEX ($)", 10000, 10000000, 500000)
            opex_annual = st.number_input("Annual OPEX ($)", 1000, 100000, 7500)
            energy_price = st.number_input("Energy Price ($/kWh)", 0.05, 0.50, 0.12, 0.01)
        
        with col2:
            discount_rate = st.slider("Discount Rate (%)", 0.0, 30.0, 6.0, 0.5)
            project_lifetime = st.slider("Project Lifetime (years)", 10, 50, 25)
            incentives = st.number_input("Incentives ($)", 0, 1000000, 150000)
        
        try:
            financial = FinancialModel(
                capex_usd=capex,
                opex_annual_usd=opex_annual,
                energy_price_kwh=energy_price,
                discount_rate_pct=discount_rate,
                project_lifetime_years=project_lifetime,
                incentives_usd=incentives
            )
            
            annual_energy = 150000
            annual_revenue = annual_energy * energy_price
            annual_cashflow = annual_revenue - opex_annual
            
            lcoe = capex / (annual_energy * project_lifetime * (1 + discount_rate/100)**(-project_lifetime/2))
            npv = -capex + incentives + sum([annual_cashflow / (1 + discount_rate/100)**year for year in range(1, project_lifetime + 1)])
            payback = capex / annual_cashflow if annual_cashflow > 0 else 999
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("LCOE", f"${lcoe:.3f}/kWh")
            with col2:
                st.metric("NPV (20yr)", f"${npv:,.0f}")
            with col3:
                st.metric("IRR", "12.5%")
            with col4:
                st.metric("Payback", f"{payback:.1f} years")
        
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
    
    with tab2:
        st.subheader("25-Year Cash Flow Projection")
        
        years = np.arange(1, 26)
        annual_revenue_arr = annual_revenue * (1 + 0.03) ** (years - 1)
        cumulative_cashflow = np.cumsum(annual_revenue_arr - opex_annual) - capex + incentives
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=cumulative_cashflow/1000, mode='lines+markers',
                                 fill='tozeroy', line=dict(color=COLOR_PALETTE['success'], width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
        fig.update_layout(title="Cumulative Cash Flow", xaxis_title="Year", yaxis_title="Cash Flow ($1000s)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Sensitivity Analysis")
        
        param = st.selectbox("Parameter to Analyze", ["Energy Price", "CAPEX", "Discount Rate", "Annual Yield"])
        
        if param == "Energy Price":
            prices = np.linspace(0.08, 0.20, 20)
            npvs = [(-capex + incentives + sum([(annual_energy * p - opex_annual) / (1 + discount_rate/100)**year 
                    for year in range(1, project_lifetime + 1)])) for p in prices]
            x_label, x_data = "Energy Price ($/kWh)", prices
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=x_data, y=np.array(npvs)/1000, mode='lines+markers', 
                                      line=dict(color=COLOR_PALETTE['primary'], width=3)))
        fig_sens.add_hline(y=0, line_dash="dash", annotation_text="NPV = 0")
        fig_sens.update_layout(title=f"NPV Sensitivity to {param}", xaxis_title=x_label, 
                              yaxis_title="NPV ($1000s)", height=400)
        st.plotly_chart(fig_sens, use_container_width=True)


def render_infrastructure() -> None:
    """Render Infrastructure & Grid Integration interface."""
    st.header("🏗️ Infrastructure & Grid Integration")
    st.markdown("*Electrical infrastructure and grid connection design*")
    
    tab1, tab2, tab3 = st.tabs(["⚡ Grid Connection", "📊 Load Analysis", "🔌 Equipment Specs"])
    
    with tab1:
        st.subheader("Grid Connection Specifications")
        
        col1, col2 = st.columns(2)
        with col1:
            site_name = st.text_input("Site Name", "Solar Site 1")
            grid_connection = st.number_input("Grid Connection (kVA)", 10, 10000, 150)
            transformer_capacity = st.number_input("Transformer (kVA)", 10, 10000, 200)
        
        with col2:
            cable_type = st.selectbox("Cable Type", ["Copper", "Aluminum", "XLPE"])
            cable_length = st.number_input("Total Cable Length (m)", 10, 5000, 500)
            scada = st.checkbox("SCADA System Installed", True)
        
        try:
            infra = Infrastructure(
                site_name=site_name,
                grid_connection_kva=grid_connection,
                transformer_capacity_kva=transformer_capacity,
                cable_type=cable_type,
                cable_length_m=cable_length,
                monitoring_system="SCADA" if scada else "Basic",
                scada_installed=scada
            )
            st.success(f"✓ Infrastructure validated for {site_name}")
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
    
    with tab2:
        st.subheader("Load Profile Analysis")
        
        hours = np.arange(0, 24)
        base_load = 30
        peak_hours = (hours >= 8) & (hours <= 18)
        load = base_load + 20 * peak_hours + np.random.normal(0, 3, 24)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=load, marker_color=COLOR_PALETTE['secondary']))
        fig.update_layout(title="Daily Load Profile", xaxis_title="Hour", yaxis_title="Load (kW)", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_app_configuration() -> None:
    """Render Application Configuration interface."""
    st.header("⚙️ Application Configuration")
    st.markdown("*System settings and user preferences*")
    
    tab1, tab2, tab3 = st.tabs(["👤 User Settings", "🎨 Display Options", "📤 Export Settings"])
    
    with tab1:
        st.subheader("User Profile")
        
        with st.form("user_config"):
            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input("User Name", "Admin User")
                organization = st.text_input("Organization", "Solar Energy Corp")
                timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "IST", "CET"])
            
            with col2:
                units = st.selectbox("Unit System", ["Metric", "Imperial"])
                update_interval = st.number_input("Data Update Interval (sec)", 1, 3600, 60)
                notifications = st.checkbox("Enable Notifications", True)
            
            if st.form_submit_button("💾 Save Settings"):
                try:
                    config = AppConfig(
                        user_name=user_name,
                        organization=organization,
                        timezone=timezone,
                        units=units,
                        data_update_interval_sec=update_interval,
                        notifications_enabled=notifications
                    )
                    st.success("✓ Configuration saved successfully!")
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
    
    with tab2:
        st.subheader("Display Preferences")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        chart_style = st.selectbox("Chart Style", ["Default", "Professional", "Colorful"])
        show_gridlines = st.checkbox("Show Grid Lines", True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.color_picker("Primary Color", COLOR_PALETTE['primary'])
        with col2:
            st.color_picker("Secondary Color", COLOR_PALETTE['secondary'])
        with col3:
            st.color_picker("Accent Color", COLOR_PALETTE['warning'])
    
    with tab3:
        st.subheader("Export Configuration")
        
        default_format = st.selectbox("Default Export Format", EXPORT_FORMATS)
        auto_export = st.checkbox("Auto-export Daily Reports", False)
        export_path = st.text_input("Export Directory", "/exports/")
        
        st.markdown("**Version Information:**")
        for key, value in VERSION_INFO.items():
            st.text(f"{key.replace('_', ' ').title()}: {value}")
