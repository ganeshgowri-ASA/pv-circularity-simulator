"""
Circularity Suite Module - Branches B10-B12

This module provides functionality for:
- B10: System Revamp & Retrofit Planning
- B11: Circular Economy Assessment (3R: Reduce, Reuse, Recycle)
- B12: Hybrid Energy Systems (PV + Storage)

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

from utils.constants import (RECYCLING_PROCESSES, MATERIAL_RECOVERY_RATES, REVAMP_OPTIONS, BATTERY_TYPES, COLOR_PALETTE)
from utils.validators import RevampOption, CircularityAssessment, HybridSystem


def render_revamp_planning() -> None:
    """Render System Revamp & Retrofit Planning interface."""
    st.header("🔄 System Revamp & Retrofit Planning")
    st.markdown("*Evaluate and plan system upgrades for aging PV installations*")
    
    tab1, tab2, tab3 = st.tabs(["⚙️ Revamp Options", "📊 Economic Analysis", "📅 Planning Timeline"])
    
    with tab1:
        st.subheader("Available Revamp Options")
        for option_name, specs in REVAMP_OPTIONS.items():
            with st.expander(f"**{option_name}**"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost", f"${specs['cost_per_kw']}/kW")
                with col2:
                    st.metric("Efficiency Gain", f"+{specs['efficiency_gain']}%")
                with col3:
                    st.metric("Lifespan Extension", f"{specs['lifespan_extension']} years")
    
    with tab2:
        st.subheader("Economic Analysis")
        system_size = st.number_input("System Size (kW)", 10, 1000, 100)
        option = st.selectbox("Select Revamp Option", list(REVAMP_OPTIONS.keys()))
        
        specs = REVAMP_OPTIONS[option]
        total_cost = specs['cost_per_kw'] * system_size
        annual_gain = system_size * 1500 * specs['efficiency_gain'] / 100 * 0.12
        payback = total_cost / annual_gain if annual_gain > 0 else 999
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Investment", f"${total_cost:,.0f}")
        with col2:
            st.metric("Annual Benefit", f"${annual_gain:,.0f}/yr")
        with col3:
            st.metric("Payback Period", f"{payback:.1f} years")


def render_circularity_assessment() -> None:
    """Render Circular Economy Assessment interface."""
    st.header("♻️ Circular Economy Assessment (3R)")
    st.markdown("*Reduce, Reuse, Recycle framework for PV modules*")
    
    tab1, tab2, tab3 = st.tabs(["📊 Module Assessment", "🔄 Material Recovery", "💰 Value Analysis"])
    
    with tab1:
        st.subheader("Module Lifecycle Assessment")
        module_age = st.slider("Module Age (years)", 0, 35, 15)
        condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
        
        reuse_potential = max(0, 100 - module_age * 2.5 - ({"Excellent": 0, "Good": 10, "Fair": 25, "Poor": 50}[condition]))
        repair_value = max(0, 50 - module_age * 1.5) * 10
        recycling_revenue = 15 + module_age * 0.5
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reuse Potential", f"{reuse_potential:.0f}%")
        with col2:
            st.metric("Repair Value", f"${repair_value:.0f}")
        with col3:
            st.metric("Recycling Revenue", f"${recycling_revenue:.0f}")
        
        circularity_score = (reuse_potential + repair_value/5 + recycling_revenue*2) / 3
        st.metric("**Circularity Score**", f"{circularity_score:.0f}/100")
    
    with tab2:
        st.subheader("Material Recovery Rates")
        process = st.selectbox("Recycling Process", list(RECYCLING_PROCESSES.keys()))
        
        recovery_df = pd.DataFrame.from_dict(MATERIAL_RECOVERY_RATES, orient='index', columns=['Recovery Rate (%)'])
        st.dataframe(recovery_df.style.background_gradient(cmap='Greens'), use_container_width=True)
        
        process_info = RECYCLING_PROCESSES[process]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Process Efficiency", f"{process_info['efficiency']}%")
        with col2:
            st.metric("Cost per Module", f"${process_info['cost_per_module']}")
        with col3:
            st.metric("Energy Use", f"{process_info['energy_consumption']} kWh")


def render_hybrid_systems() -> None:
    """Render Hybrid Energy Systems interface."""
    st.header("🔋 Hybrid Energy Systems (PV + Storage)")
    st.markdown("*Design and optimize PV + Battery Energy Storage Systems*")
    
    tab1, tab2, tab3 = st.tabs(["⚙️ System Design", "📊 Energy Flow", "💰 Economics"])
    
    with tab1:
        st.subheader("Hybrid System Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            pv_capacity = st.number_input("PV Capacity (kW)", 1.0, 1000.0, 50.0)
            battery_type = st.selectbox("Battery Technology", list(BATTERY_TYPES.keys()))
        
        with col2:
            battery_capacity = st.number_input("Battery Capacity (kWh)", 1.0, 500.0, 100.0)
            battery_power = st.number_input("Battery Power (kW)", 1.0, 200.0, 50.0)
        
        battery_info = BATTERY_TYPES[battery_type]
        
        try:
            hybrid = HybridSystem(
                pv_capacity_kw=pv_capacity,
                battery_capacity_kwh=battery_capacity,
                battery_type=battery_type,
                battery_power_kw=battery_power,
                roundtrip_efficiency_pct=battery_info['efficiency'],
                cycle_life=battery_info['cycle_life'],
                depth_of_discharge_pct=battery_info['depth_of_discharge']
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Usable Capacity", f"{hybrid.usable_capacity_kwh:.1f} kWh")
            with col2:
                st.metric("Efficiency", f"{hybrid.roundtrip_efficiency_pct}%")
            with col3:
                st.metric("Cycle Life", f"{hybrid.cycle_life:,}")
            with col4:
                st.metric("DoD", f"{hybrid.depth_of_discharge_pct}%")
        
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
    
    with tab2:
        st.subheader("Daily Energy Flow Simulation")
        
        hours = np.arange(0, 24, 0.5)
        pv_generation = pv_capacity * np.maximum(0, np.sin(np.pi * (hours - 6) / 12) ** 1.5)
        pv_generation[hours < 6] = 0
        pv_generation[hours > 18] = 0
        
        load_demand = 30 + 10 * (hours > 8) * (hours < 18) + 5 * np.random.random(len(hours))
        
        battery_charge = np.minimum(pv_generation - load_demand, battery_power)
        battery_discharge = -np.minimum(load_demand - pv_generation, battery_power)
        net_flow = battery_charge + battery_discharge
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=pv_generation, name='PV Generation', fill='tozeroy', 
                                 line=dict(color=COLOR_PALETTE['warning'])))
        fig.add_trace(go.Scatter(x=hours, y=load_demand, name='Load Demand', 
                                 line=dict(color=COLOR_PALETTE['danger'])))
        fig.add_trace(go.Scatter(x=hours, y=net_flow, name='Battery Flow', 
                                 line=dict(color=COLOR_PALETTE['primary'])))
        
        fig.update_layout(title="24-Hour Energy Flow", xaxis_title="Hour", yaxis_title="Power (kW)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Economic Analysis")
        
        battery_cost = battery_capacity * battery_info['cost_per_kwh']
        annual_savings = pv_capacity * 1500 * 0.4 * 0.12
        payback = battery_cost / annual_savings if annual_savings > 0 else 999
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Battery Investment", f"${battery_cost:,.0f}")
        with col2:
            st.metric("Annual Savings", f"${annual_savings:,.0f}")
        with col3:
            st.metric("Payback Period", f"{payback:.1f} years")
