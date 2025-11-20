"""
B12-S05: Hybrid Systems UI
Production-ready Streamlit UI for hybrid renewable energy systems.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Optional

from ..core.data_models import (
    HybridSystemConfiguration,
    BatterySpecification,
    EnergyStorageTechnology
)
from .battery_integration import BatteryIntegrator, create_battery_system
from .wind_hybrid import WindHybridDesigner


class HybridUI:
    """
    Interactive UI for hybrid renewable energy system design and analysis.
    """

    def system_topology(self, config: HybridSystemConfiguration) -> go.Figure:
        """
        Display system topology diagram.

        Args:
            config: Hybrid system configuration

        Returns:
            Plotly figure with system diagram
        """
        # Create sankey diagram for system topology
        labels = ["Solar PV", "Wind", "Battery", "Load", "Grid Export", "Grid Import"]

        # Simplified flow values (example)
        solar_total = config.solar_capacity_kw * 0.2 * 8760  # 20% CF
        wind_total = config.wind_capacity_kw * 0.3 * 8760  # 30% CF
        load_total = (solar_total + wind_total) * 0.8

        source = [0, 1, 2, 0, 1]  # From
        target = [3, 3, 3, 4, 2]  # To
        value = [solar_total * 0.7, wind_total * 0.7,
                config.battery_capacity_kwh * 300, solar_total * 0.3, wind_total * 0.3]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=labels,
                color=["#FDB462", "#80B1D3", "#FB8072", "#BEBADA", "#8DD3C7", "#FFFFB3"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])

        fig.update_layout(
            title="Hybrid System Energy Flow",
            font_size=12,
            height=500
        )

        return fig

    def optimization_dashboard(self,
                              solar_gen: np.ndarray,
                              wind_gen: np.ndarray,
                              load: np.ndarray) -> None:
        """
        Display optimization results dashboard.

        Args:
            solar_gen: Solar generation profile
            wind_gen: Wind generation profile
            load: Load profile
        """
        st.subheader("⚡ Hybrid System Optimization")

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Generation Mix", "Storage Dispatch", "Grid Interaction"])

        with tab1:
            # Generation mix over time
            df = pd.DataFrame({
                'Hour': range(len(solar_gen)),
                'Solar': solar_gen,
                'Wind': wind_gen,
                'Load': load
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Hour'], y=df['Solar'],
                                    mode='lines', name='Solar',
                                    line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['Hour'], y=df['Wind'],
                                    mode='lines', name='Wind',
                                    line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Hour'], y=df['Load'],
                                    mode='lines', name='Load',
                                    line=dict(color='red', dash='dash')))

            fig.update_layout(
                title="Generation and Load Profiles",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Battery dispatch strategy
            st.write("Battery charge/discharge optimization based on:")
            col1, col2 = st.columns(2)

            with col1:
                st.info("✅ Charge during low prices / excess generation")
                st.info("✅ Peak shaving capability")

            with col2:
                st.info("✅ Discharge during high prices / peak demand")
                st.info("✅ Grid services provision")

        with tab3:
            # Grid interaction metrics
            total_solar = np.sum(solar_gen)
            total_wind = np.sum(wind_gen)
            total_load = np.sum(load)
            total_gen = total_solar + total_wind

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Solar Contribution", f"{total_solar / total_gen * 100:.1f}%")

            with col2:
                st.metric("Wind Contribution", f"{total_wind / total_gen * 100:.1f}%")

            with col3:
                self_sufficiency = min(total_gen / total_load, 1.0) if total_load > 0 else 0
                st.metric("Self-Sufficiency", f"{self_sufficiency * 100:.1f}%")

    def dispatch_strategies(self, battery: BatteryIntegrator,
                           price_profile: np.ndarray) -> go.Figure:
        """
        Display battery dispatch strategies.

        Args:
            battery: Battery integrator instance
            price_profile: Electricity price profile

        Returns:
            Plotly figure
        """
        # Run arbitrage optimization
        time_hours = np.arange(len(price_profile))
        result = battery.arbitrage_optimization(price_profile, time_hours)

        # Extract data
        power = [s.power_kw for s in result['schedule']]
        soc = [s.state_of_charge for s in result['schedule']]
        profit = [s.arbitrage_profit for s in result['schedule']]

        # Create subplot figure
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Battery Power", "State of Charge", "Electricity Price"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Power trace
        fig.add_trace(
            go.Scatter(x=time_hours, y=power, name="Power", line=dict(color='blue')),
            row=1, col=1
        )

        # SoC trace
        fig.add_trace(
            go.Scatter(x=time_hours, y=soc, name="SoC", line=dict(color='green')),
            row=2, col=1
        )

        # Price trace
        fig.add_trace(
            go.Scatter(x=time_hours, y=price_profile, name="Price",
                      line=dict(color='red')),
            row=3, col=1
        )

        fig.update_layout(height=800, showlegend=True,
                         title_text="Battery Dispatch Strategy")
        fig.update_xaxes(title_text="Hour", row=3, col=1)
        fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
        fig.update_yaxes(title_text="SoC", row=2, col=1)
        fig.update_yaxes(title_text="Price ($/kWh)", row=3, col=1)

        return fig

    def render_dashboard(self) -> None:
        """Render complete hybrid systems dashboard."""
        st.title("🔋 Hybrid Renewable Energy Systems")

        # Sidebar configuration
        st.sidebar.header("System Configuration")

        solar_capacity = st.sidebar.number_input("Solar Capacity (kW)", value=5000, step=100)
        wind_capacity = st.sidebar.number_input("Wind Capacity (kW)", value=3000, step=100)
        battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", value=2000, step=100)

        config = HybridSystemConfiguration(
            system_name="Hybrid System",
            solar_capacity_kw=solar_capacity,
            wind_capacity_kw=wind_capacity,
            battery_capacity_kwh=battery_capacity,
            location="Sample Location",
            optimization_objective="cost",
            grid_connected=True
        )

        # Display topology
        st.subheader("System Topology")
        topology_fig = self.system_topology(config)
        st.plotly_chart(topology_fig, use_container_width=True)

        # Generate sample profiles
        hours = 168  # One week
        solar_gen = solar_capacity * np.maximum(0, np.sin(np.linspace(0, 7 * np.pi, hours)))
        wind_gen = wind_capacity * (0.3 + 0.2 * np.random.random(hours))
        load = (solar_capacity + wind_capacity) * 0.3 * (0.5 + 0.5 * np.random.random(hours))

        # Optimization dashboard
        self.optimization_dashboard(solar_gen, wind_gen, load)


__all__ = ["HybridUI"]
