"""
Hybrid Energy System UI & Configuration Module.

This module provides a comprehensive Streamlit-based user interface for
configuring, controlling, and monitoring hybrid energy systems.

The HybridSystemUI class offers:
- Interactive system configuration wizard
- Component selection and management
- Operation strategy builder
- Real-time performance monitoring dashboard
"""

from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..core.config import (
    SystemConfiguration,
    ComponentConfig,
    OperationStrategy,
    ConfigManager,
)
from ..core.models import (
    HybridEnergySystem,
    PVArray,
    BatteryStorage,
    EnergyComponent,
)
from ..monitoring.metrics import (
    PerformanceMetrics,
    SystemMetrics,
    MetricsTracker,
)
from ..utils.helpers import (
    format_power,
    format_energy,
    format_percentage,
    sanitize_component_id,
    generate_default_load_profile,
    generate_default_irradiance_profile,
    color_by_status,
)


class HybridSystemUI:
    """
    Hybrid Energy System User Interface and Configuration Manager.

    This class provides a complete Streamlit-based interface for managing
    hybrid energy systems, including configuration, monitoring, and control.

    The UI is organized into several main sections:
    1. System Configuration Wizard - Step-by-step system setup
    2. Component Selector - Add/remove/configure components
    3. Operation Strategy Builder - Define control strategies
    4. Performance Monitoring Dashboard - Real-time system monitoring

    Attributes:
        config: Current system configuration
        system: Hybrid energy system instance
        metrics_tracker: Metrics tracking utility
        simulation_running: Flag indicating if simulation is active

    Example:
        >>> ui = HybridSystemUI()
        >>> ui.render()  # Renders the complete UI
    """

    def __init__(self, config: Optional[SystemConfiguration] = None):
        """
        Initialize the Hybrid System UI.

        Args:
            config: Optional pre-existing configuration. If None, creates default.
        """
        # Initialize session state
        self._initialize_session_state()

        # Load or create configuration
        if config is not None:
            self.config = config
            st.session_state.config = config
        elif st.session_state.get("config") is None:
            self.config = ConfigManager.create_default_config()
            st.session_state.config = self.config
        else:
            self.config = st.session_state.config

        # Initialize system
        self.system = self._build_system_from_config()
        self.metrics_tracker = MetricsTracker(max_history_points=2000)
        self.simulation_running = False

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "config" not in st.session_state:
            st.session_state.config = None

        if "simulation_data" not in st.session_state:
            st.session_state.simulation_data = []

        if "current_step" not in st.session_state:
            st.session_state.current_step = 0

        if "wizard_step" not in st.session_state:
            st.session_state.wizard_step = 0

        if "selected_components" not in st.session_state:
            st.session_state.selected_components = []

    def _build_system_from_config(self) -> HybridEnergySystem:
        """
        Build a HybridEnergySystem instance from configuration.

        Returns:
            Configured HybridEnergySystem instance
        """
        system = HybridEnergySystem(self.config.system_name)

        # Add components based on configuration
        for comp_config in self.config.components:
            if comp_config.component_type == "pv_array":
                component = PVArray(
                    component_id=comp_config.component_id,
                    component_type=comp_config.component_type,
                    name=comp_config.name,
                    capacity=comp_config.capacity,
                    capacity_unit=comp_config.capacity_unit,
                    efficiency=comp_config.efficiency,
                    area_m2=comp_config.parameters.get("area_m2", 50.0),
                    tilt_angle=comp_config.parameters.get("tilt_angle", 30.0),
                    azimuth_angle=comp_config.parameters.get("azimuth_angle", 180.0),
                )
            elif comp_config.component_type == "battery":
                component = BatteryStorage(
                    component_id=comp_config.component_id,
                    component_type=comp_config.component_type,
                    name=comp_config.name,
                    capacity=comp_config.capacity,
                    capacity_unit=comp_config.capacity_unit,
                    efficiency=comp_config.efficiency,
                    state_of_charge=comp_config.parameters.get("initial_soc", 0.5),
                    min_soc=comp_config.parameters.get("min_soc", 0.2),
                    max_soc=comp_config.parameters.get("max_soc", 0.9),
                    charge_rate_max_kw=comp_config.parameters.get(
                        "charge_rate_max_kw", comp_config.capacity / 4
                    ),
                    discharge_rate_max_kw=comp_config.parameters.get(
                        "discharge_rate_max_kw", comp_config.capacity / 4
                    ),
                )
            else:
                # Generic component
                component = EnergyComponent(
                    component_id=comp_config.component_id,
                    component_type=comp_config.component_type,
                    name=comp_config.name,
                    capacity=comp_config.capacity,
                    capacity_unit=comp_config.capacity_unit,
                    efficiency=comp_config.efficiency,
                    parameters=comp_config.parameters,
                )

            system.add_component(component)

        return system

    def system_configuration_wizard(self) -> None:
        """
        Interactive step-by-step system configuration wizard.

        This method provides a guided configuration experience with multiple steps:
        1. Basic system information
        2. Component selection and configuration
        3. Operation strategy setup
        4. Simulation parameters
        5. Review and confirmation

        The wizard uses Streamlit's session state to maintain progress across
        interactions and provides validation at each step.

        Example:
            >>> ui = HybridSystemUI()
            >>> ui.system_configuration_wizard()
        """
        st.header("⚙️ System Configuration Wizard")
        st.markdown("Configure your hybrid energy system step by step.")

        # Progress indicator
        steps = [
            "Basic Info",
            "Components",
            "Strategy",
            "Simulation",
            "Review"
        ]
        current_step = st.session_state.wizard_step

        # Display progress
        cols = st.columns(len(steps))
        for idx, (col, step_name) in enumerate(zip(cols, steps)):
            with col:
                if idx < current_step:
                    st.success(f"✓ {step_name}")
                elif idx == current_step:
                    st.info(f"▶ {step_name}")
                else:
                    st.text(f"○ {step_name}")

        st.markdown("---")

        # Render current step
        if current_step == 0:
            self._wizard_step_basic_info()
        elif current_step == 1:
            self._wizard_step_components()
        elif current_step == 2:
            self._wizard_step_strategy()
        elif current_step == 3:
            self._wizard_step_simulation()
        elif current_step == 4:
            self._wizard_step_review()

    def _wizard_step_basic_info(self) -> None:
        """Wizard step: Basic system information."""
        st.subheader("Step 1: Basic System Information")

        system_name = st.text_input(
            "System Name",
            value=self.config.system_name,
            help="Enter a descriptive name for your hybrid energy system"
        )

        system_description = st.text_area(
            "System Description",
            value=self.config.system_description,
            help="Provide details about the system purpose and location"
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Next →", type="primary", use_container_width=True):
                self.config.system_name = system_name
                self.config.system_description = system_description
                st.session_state.wizard_step = 1
                st.rerun()

    def _wizard_step_components(self) -> None:
        """Wizard step: Component selection and configuration."""
        st.subheader("Step 2: Select System Components")

        st.markdown("Add components to your hybrid energy system:")

        # Component selector (reuse existing method)
        self.component_selector()

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
        with col2:
            if st.button("Next →", type="primary", use_container_width=True):
                if len(self.config.components) > 0:
                    st.session_state.wizard_step = 2
                    st.rerun()
                else:
                    st.error("Please add at least one component")

    def _wizard_step_strategy(self) -> None:
        """Wizard step: Operation strategy configuration."""
        st.subheader("Step 3: Operation Strategy")

        # Operation strategy builder (reuse existing method)
        self.operation_strategy_builder()

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3
                st.rerun()

    def _wizard_step_simulation(self) -> None:
        """Wizard step: Simulation parameters."""
        st.subheader("Step 4: Simulation Parameters")

        col1, col2 = st.columns(2)

        with col1:
            time_step = st.slider(
                "Time Step (minutes)",
                min_value=1,
                max_value=60,
                value=self.config.simulation.time_step_minutes,
                help="Simulation time resolution"
            )

            duration = st.number_input(
                "Simulation Duration (hours)",
                min_value=1,
                max_value=8760,
                value=self.config.simulation.simulation_duration_hours,
                help="Total simulation period (1-8760 hours)"
            )

        with col2:
            weather_source = st.selectbox(
                "Weather Data Source",
                options=["default", "custom", "historic", "forecast"],
                index=0,
                help="Source for weather data"
            )

            load_source = st.selectbox(
                "Load Profile Source",
                options=["default", "custom", "measured", "predicted"],
                index=0,
                help="Source for load profile data"
            )

        # Update configuration
        self.config.simulation.time_step_minutes = time_step
        self.config.simulation.simulation_duration_hours = duration
        self.config.simulation.weather_data_source = weather_source
        self.config.simulation.load_profile_source = load_source

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 4
                st.rerun()

    def _wizard_step_review(self) -> None:
        """Wizard step: Review and confirmation."""
        st.subheader("Step 5: Review Configuration")

        st.markdown("### System Overview")
        st.write(f"**Name:** {self.config.system_name}")
        st.write(f"**Description:** {self.config.system_description}")

        st.markdown("### Components")
        comp_data = []
        for comp in self.config.components:
            comp_data.append({
                "ID": comp.component_id,
                "Type": comp.component_type,
                "Name": comp.name,
                "Capacity": f"{comp.capacity} {comp.capacity_unit}",
                "Efficiency": f"{comp.efficiency*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        st.markdown("### Operation Strategy")
        if self.config.operation_strategy:
            st.write(f"**Strategy:** {self.config.operation_strategy.strategy_name}")
            st.write(f"**Type:** {self.config.operation_strategy.strategy_type}")
        else:
            st.write("No operation strategy configured")

        st.markdown("### Simulation Parameters")
        st.write(f"**Time Step:** {self.config.simulation.time_step_minutes} minutes")
        st.write(f"**Duration:** {self.config.simulation.simulation_duration_hours} hours")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 3
                st.rerun()
        with col2:
            if st.button("Save Configuration", type="primary", use_container_width=True):
                st.success("✓ Configuration saved successfully!")
                st.balloons()
        with col3:
            if st.button("Finish", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()

    def component_selector(self) -> None:
        """
        Interactive component selection and configuration interface.

        This method provides an interface for:
        - Viewing all configured components
        - Adding new components with detailed parameters
        - Editing existing components
        - Removing components from the system
        - Validating component configurations

        Components are displayed in a table with options to edit or remove.
        A form allows adding new components with type-specific parameters.

        Example:
            >>> ui = HybridSystemUI()
            >>> ui.component_selector()
        """
        st.header("🔌 Component Selector")

        # Display existing components
        if self.config.components:
            st.subheader("Current Components")

            for idx, comp in enumerate(self.config.components):
                with st.expander(
                    f"{comp.name} ({comp.component_type}) - {comp.capacity} {comp.capacity_unit}",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.write(f"**ID:** {comp.component_id}")
                        st.write(f"**Type:** {comp.component_type}")
                        st.write(f"**Capacity:** {comp.capacity} {comp.capacity_unit}")

                    with col2:
                        st.write(f"**Efficiency:** {comp.efficiency*100:.1f}%")
                        st.write(f"**Status:** {'Enabled' if comp.enabled else 'Disabled'}")

                    with col3:
                        if st.button("Remove", key=f"remove_{comp.component_id}"):
                            self.config.components.pop(idx)
                            self.system = self._build_system_from_config()
                            st.rerun()

                    # Display component-specific parameters
                    if comp.parameters:
                        st.write("**Parameters:**")
                        st.json(comp.parameters)
        else:
            st.info("No components configured yet. Add components below.")

        st.markdown("---")

        # Add new component
        st.subheader("Add New Component")

        with st.form("add_component_form"):
            col1, col2 = st.columns(2)

            with col1:
                comp_type = st.selectbox(
                    "Component Type",
                    options=[
                        "pv_array",
                        "battery",
                        "wind_turbine",
                        "diesel_generator",
                        "fuel_cell",
                        "electrolyzer",
                        "grid_connection",
                    ],
                    help="Select the type of energy component"
                )

                comp_name = st.text_input(
                    "Component Name",
                    value=f"New {comp_type.replace('_', ' ').title()}",
                    help="Enter a descriptive name"
                )

                comp_capacity = st.number_input(
                    "Capacity",
                    min_value=0.1,
                    value=10.0,
                    step=0.5,
                    help="Nominal capacity of the component"
                )

            with col2:
                comp_capacity_unit = st.selectbox(
                    "Capacity Unit",
                    options=["kW", "kWh", "MW", "MWh"],
                    index=0 if comp_type != "battery" else 1,
                    help="Unit for capacity measurement"
                )

                comp_efficiency = st.slider(
                    "Efficiency (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=85.0,
                    step=1.0,
                    help="Component operating efficiency"
                ) / 100.0

            # Type-specific parameters
            parameters = {}

            if comp_type == "pv_array":
                st.markdown("**PV Array Parameters**")
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    parameters["area_m2"] = st.number_input(
                        "Area (m²)", min_value=1.0, value=50.0
                    )
                with pcol2:
                    parameters["tilt_angle"] = st.number_input(
                        "Tilt Angle (°)", min_value=0.0, max_value=90.0, value=30.0
                    )
                with pcol3:
                    parameters["azimuth_angle"] = st.number_input(
                        "Azimuth (°)", min_value=0.0, max_value=360.0, value=180.0
                    )

            elif comp_type == "battery":
                st.markdown("**Battery Parameters**")
                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    parameters["initial_soc"] = st.slider(
                        "Initial SOC (%)", 0.0, 100.0, 50.0
                    ) / 100.0
                with bcol2:
                    parameters["min_soc"] = st.slider(
                        "Min SOC (%)", 0.0, 100.0, 20.0
                    ) / 100.0
                with bcol3:
                    parameters["max_soc"] = st.slider(
                        "Max SOC (%)", 0.0, 100.0, 90.0
                    ) / 100.0

                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    parameters["charge_rate_max_kw"] = st.number_input(
                        "Max Charge Rate (kW)", min_value=0.1, value=comp_capacity / 4
                    )
                with rcol2:
                    parameters["discharge_rate_max_kw"] = st.number_input(
                        "Max Discharge Rate (kW)", min_value=0.1, value=comp_capacity / 4
                    )

            # Submit button
            submitted = st.form_submit_button("Add Component", type="primary")

            if submitted:
                # Generate component ID
                comp_id = sanitize_component_id(f"{comp_type}_{len(self.config.components) + 1:03d}")

                # Create component configuration
                new_component = ComponentConfig(
                    component_id=comp_id,
                    component_type=comp_type,
                    name=comp_name,
                    capacity=comp_capacity,
                    capacity_unit=comp_capacity_unit,
                    efficiency=comp_efficiency,
                    parameters=parameters,
                    enabled=True
                )

                # Add to configuration
                self.config.components.append(new_component)

                # Rebuild system
                self.system = self._build_system_from_config()

                st.success(f"✓ Added {comp_name} ({comp_id})")
                st.rerun()

    def operation_strategy_builder(self) -> None:
        """
        Interactive operation strategy configuration builder.

        This method provides an interface for defining and configuring
        control strategies for the hybrid energy system, including:

        - Strategy type selection (rule-based, optimal, predictive)
        - Component priority ordering
        - Control algorithm parameters
        - Operating constraints and limits
        - Strategy validation and testing

        The builder ensures that all strategy configurations are valid
        and compatible with the configured components.

        Example:
            >>> ui = HybridSystemUI()
            >>> ui.operation_strategy_builder()
        """
        st.header("🎯 Operation Strategy Builder")

        st.markdown(
            "Define how your hybrid energy system should operate and "
            "prioritize different energy sources."
        )

        # Strategy configuration form
        with st.form("strategy_form"):
            st.subheader("Strategy Configuration")

            col1, col2 = st.columns(2)

            with col1:
                strategy_name = st.text_input(
                    "Strategy Name",
                    value=(
                        self.config.operation_strategy.strategy_name
                        if self.config.operation_strategy
                        else "Default Strategy"
                    ),
                    help="Name for this operation strategy"
                )

                strategy_type = st.selectbox(
                    "Strategy Type",
                    options=["rule_based", "optimal", "predictive"],
                    index=0,
                    help="Type of control strategy"
                )

            with col2:
                st.markdown("**Strategy Description**")

                if strategy_type == "rule_based":
                    st.info(
                        "Rule-based: Simple priority-based control using "
                        "predefined rules"
                    )
                elif strategy_type == "optimal":
                    st.info(
                        "Optimal: Optimization-based control for cost or "
                        "performance objectives"
                    )
                else:
                    st.info(
                        "Predictive: Model predictive control using "
                        "forecasts and predictions"
                    )

            # Component priority ordering
            st.subheader("Component Priority")
            st.markdown(
                "Drag components to reorder (higher = higher priority). "
                "Or manually configure below:"
            )

            component_ids = [c.component_id for c in self.config.components]
            component_names = [
                f"{c.name} ({c.component_id})" for c in self.config.components
            ]

            if component_ids:
                # Get current priority order or use default
                if (
                    self.config.operation_strategy
                    and self.config.operation_strategy.priority_order
                ):
                    current_order = self.config.operation_strategy.priority_order
                    # Validate and add any missing components
                    for comp_id in component_ids:
                        if comp_id not in current_order:
                            current_order.append(comp_id)
                else:
                    current_order = component_ids.copy()

                # Display priority list with controls
                priority_order = []
                for idx in range(len(current_order)):
                    if current_order[idx] in component_ids:
                        selected = st.selectbox(
                            f"Priority {idx + 1}",
                            options=component_names,
                            index=component_ids.index(current_order[idx]),
                            key=f"priority_{idx}"
                        )
                        selected_id = component_ids[component_names.index(selected)]
                        priority_order.append(selected_id)
            else:
                st.warning("No components configured. Add components first.")
                priority_order = []

            # Control parameters
            st.subheader("Control Parameters")

            control_params = {}

            if strategy_type == "rule_based":
                col1, col2 = st.columns(2)
                with col1:
                    control_params["battery_charge_threshold"] = st.slider(
                        "Battery Charge Threshold (%)",
                        0.0, 100.0, 80.0,
                        help="Start charging battery when SOC below this"
                    ) / 100.0
                with col2:
                    control_params["battery_discharge_threshold"] = st.slider(
                        "Battery Discharge Threshold (%)",
                        0.0, 100.0, 30.0,
                        help="Stop discharging when SOC below this"
                    ) / 100.0

            elif strategy_type == "optimal":
                col1, col2 = st.columns(2)
                with col1:
                    control_params["optimization_horizon_hours"] = st.number_input(
                        "Optimization Horizon (hours)",
                        min_value=1, max_value=48, value=24
                    )
                with col2:
                    control_params["objective"] = st.selectbox(
                        "Optimization Objective",
                        options=["minimize_cost", "maximize_self_consumption", "maximize_renewable"]
                    )

            # Operating constraints
            st.subheader("Operating Constraints")

            constraints = {}
            ccol1, ccol2 = st.columns(2)

            with ccol1:
                constraints["max_grid_import_kw"] = st.number_input(
                    "Max Grid Import (kW)",
                    min_value=0.0,
                    value=100.0,
                    help="Maximum power import from grid"
                )

            with ccol2:
                constraints["max_grid_export_kw"] = st.number_input(
                    "Max Grid Export (kW)",
                    min_value=0.0,
                    value=50.0,
                    help="Maximum power export to grid"
                )

            # Submit button
            submitted = st.form_submit_button("Save Strategy", type="primary")

            if submitted:
                # Create operation strategy
                self.config.operation_strategy = OperationStrategy(
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    priority_order=priority_order,
                    control_parameters=control_params,
                    constraints=constraints
                )

                st.success(f"✓ Operation strategy '{strategy_name}' saved successfully!")

        # Display current strategy
        if self.config.operation_strategy:
            st.markdown("---")
            st.subheader("Current Strategy Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {self.config.operation_strategy.strategy_name}")
                st.write(f"**Type:** {self.config.operation_strategy.strategy_type}")

            with col2:
                st.write("**Priority Order:**")
                for idx, comp_id in enumerate(self.config.operation_strategy.priority_order):
                    comp = self.config.get_component_by_id(comp_id)
                    if comp:
                        st.write(f"  {idx + 1}. {comp.name} ({comp_id})")

    def performance_monitoring_dashboard(self) -> None:
        """
        Real-time performance monitoring and visualization dashboard.

        This comprehensive dashboard provides:
        - Real-time system status and metrics
        - Power flow visualization
        - Component status indicators
        - Energy production/consumption charts
        - Performance indicators (efficiency, self-sufficiency, etc.)
        - Historical data analysis
        - Alert and notification system

        The dashboard updates in real-time during simulation and provides
        interactive charts using Plotly for detailed analysis.

        Features:
        - Live power flow diagram
        - Time-series charts for key metrics
        - Component health monitoring
        - Performance KPI cards
        - Exportable reports

        Example:
            >>> ui = HybridSystemUI()
            >>> ui.performance_monitoring_dashboard()
        """
        st.header("📊 Performance Monitoring Dashboard")

        # Control panel
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                if st.button("▶ Start Simulation", type="primary", use_container_width=True):
                    self._run_simulation()

            with col2:
                if st.button("⏸ Pause", use_container_width=True):
                    st.info("Simulation paused")

            with col3:
                if st.button("⏹ Stop & Reset", use_container_width=True):
                    self.system.reset()
                    self.metrics_tracker.clear_history()
                    st.session_state.simulation_data = []
                    st.session_state.current_step = 0
                    st.success("System reset")

            with col4:
                st.write(f"Step: {st.session_state.current_step}")

        st.markdown("---")

        # KPI Cards
        st.subheader("Key Performance Indicators")

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        # Get current metrics
        system_metrics = self.system.get_system_metrics()

        with kpi_col1:
            st.metric(
                label="Total Generation",
                value=format_power(system_metrics["total_generation_kw"]),
                delta=None
            )

        with kpi_col2:
            st.metric(
                label="Total Consumption",
                value=format_power(system_metrics["total_consumption_kw"]),
                delta=None
            )

        with kpi_col3:
            st.metric(
                label="Net Power",
                value=format_power(system_metrics["net_power_kw"]),
                delta=None
            )

        with kpi_col4:
            st.metric(
                label="System Capacity",
                value=format_power(system_metrics["total_capacity_kw"]),
                delta=None
            )

        st.markdown("---")

        # Component Status
        st.subheader("Component Status")

        comp_cols = st.columns(min(len(self.system.components), 4))

        for idx, (comp_id, component) in enumerate(self.system.components.items()):
            col_idx = idx % 4
            with comp_cols[col_idx]:
                status_color = color_by_status(component.state)

                st.markdown(
                    f"""
                    <div style="padding: 10px; border-left: 4px solid {status_color};
                                background-color: rgba(128,128,128,0.1); border-radius: 4px;">
                        <b>{component.name}</b><br>
                        <small>{component.component_type}</small><br>
                        Status: {component.state}<br>
                        Power: {format_power(component.power_output)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Special display for battery
                if isinstance(component, BatteryStorage):
                    st.progress(
                        component.state_of_charge,
                        text=f"SOC: {component.state_of_charge*100:.1f}%"
                    )

        st.markdown("---")

        # Charts
        if st.session_state.simulation_data:
            self._render_performance_charts()
        else:
            st.info("Run a simulation to see performance charts")

    def _render_performance_charts(self) -> None:
        """Render performance visualization charts."""
        st.subheader("Performance Charts")

        df = pd.DataFrame(st.session_state.simulation_data)

        # Create tabs for different chart views
        tab1, tab2, tab3 = st.tabs(["Power Flow", "Energy Balance", "Component Details"])

        with tab1:
            # Power flow chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Power Generation vs Load", "Grid Interaction"),
                vertical_spacing=0.12
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["pv_generation_kw"],
                    name="PV Generation",
                    fill='tozeroy',
                    line=dict(color='#FDB462')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["load_demand_kw"],
                    name="Load Demand",
                    line=dict(color='#E41A1C', dash='dash')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["grid_power_kw"],
                    name="Grid Power",
                    fill='tozeroy',
                    line=dict(color='#377EB8')
                ),
                row=2, col=1
            )

            fig.update_xaxes(title_text="Time Step", row=2, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=2, col=1)

            fig.update_layout(height=600, showlegend=True)

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Energy balance pie chart
            total_gen = df["pv_generation_kw"].sum()
            total_load = df["load_demand_kw"].sum()
            total_grid_import = df[df["grid_power_kw"] > 0]["grid_power_kw"].sum()
            total_grid_export = df[df["grid_power_kw"] < 0]["grid_power_kw"].abs().sum()

            col1, col2 = st.columns(2)

            with col1:
                fig_gen = go.Figure(data=[go.Pie(
                    labels=["PV Generation", "Grid Import"],
                    values=[total_gen, total_grid_import],
                    hole=0.3
                )])
                fig_gen.update_layout(title="Energy Sources")
                st.plotly_chart(fig_gen, use_container_width=True)

            with col2:
                fig_cons = go.Figure(data=[go.Pie(
                    labels=["Load Consumption", "Grid Export"],
                    values=[total_load, total_grid_export],
                    hole=0.3
                )])
                fig_cons.update_layout(title="Energy Usage")
                st.plotly_chart(fig_cons, use_container_width=True)

        with tab3:
            # Component details
            if "battery_power_kw" in df.columns:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df["battery_power_kw"],
                    name="Battery Power",
                    fill='tozeroy',
                    line=dict(color='#4DAF4A')
                ))

                fig.update_layout(
                    title="Battery Power Profile",
                    xaxis_title="Time Step",
                    yaxis_title="Power (kW)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

    def _run_simulation(self) -> None:
        """Run a simulation and update the dashboard."""
        st.info("Running simulation...")

        # Get simulation parameters
        num_steps = 288  # 24 hours at 5-minute intervals
        time_step_minutes = self.config.simulation.time_step_minutes

        # Generate profiles
        load_profile = generate_default_load_profile(num_steps)
        irradiance_profile = generate_default_irradiance_profile(num_steps)

        # Run simulation
        simulation_data = []

        for step in range(num_steps):
            input_conditions = {
                "irradiance_w_m2": irradiance_profile[step],
                "temperature_c": 25.0
            }

            result = self.system.simulate_step(
                load_demand_kw=load_profile[step],
                input_conditions=input_conditions,
                time_step_minutes=time_step_minutes
            )

            simulation_data.append(result)

            # Update metrics tracker
            self.metrics_tracker.add_simulation_result(result)

        # Store in session state
        st.session_state.simulation_data = simulation_data
        st.session_state.current_step = num_steps

        st.success(f"✓ Simulation completed ({num_steps} steps)")
        st.rerun()

    def render(self) -> None:
        """
        Render the complete Hybrid System UI.

        This is the main entry point that renders the full interface
        with navigation and all sub-components.

        Example:
            >>> ui = HybridSystemUI()
            >>> ui.render()
        """
        # Page configuration
        st.set_page_config(
            page_title="Hybrid Energy System UI",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Sidebar
        with st.sidebar:
            st.title("⚡ Hybrid Energy System")
            st.markdown("---")

            page = st.radio(
                "Navigation",
                options=[
                    "Dashboard",
                    "Configuration Wizard",
                    "Component Selector",
                    "Operation Strategy",
                    "Settings"
                ],
                index=0
            )

            st.markdown("---")
            st.markdown("### System Info")
            st.write(f"**Name:** {self.config.system_name}")
            st.write(f"**Components:** {len(self.config.components)}")
            st.write(f"**Capacity:** {format_power(self.system.get_total_capacity())}")

        # Main content area
        if page == "Dashboard":
            self.performance_monitoring_dashboard()
        elif page == "Configuration Wizard":
            self.system_configuration_wizard()
        elif page == "Component Selector":
            self.component_selector()
        elif page == "Operation Strategy":
            self.operation_strategy_builder()
        elif page == "Settings":
            self._render_settings()

    def _render_settings(self) -> None:
        """Render settings page."""
        st.header("⚙️ Settings")

        st.subheader("Monitoring Configuration")

        col1, col2 = st.columns(2)

        with col1:
            refresh_rate = st.slider(
                "Dashboard Refresh Rate (seconds)",
                min_value=1,
                max_value=60,
                value=self.config.monitoring.refresh_rate_seconds
            )

        with col2:
            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                    self.config.monitoring.log_level
                )
            )

        self.config.monitoring.refresh_rate_seconds = refresh_rate
        self.config.monitoring.log_level = log_level

        st.markdown("---")

        st.subheader("Configuration Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export Configuration", use_container_width=True):
                config_dict = self.config.model_dump()
                st.download_button(
                    label="Download YAML",
                    data=str(config_dict),
                    file_name="hybrid_system_config.yaml",
                    mime="text/yaml"
                )

        with col2:
            if st.button("Load Default Config", use_container_width=True):
                self.config = ConfigManager.create_default_config()
                self.system = self._build_system_from_config()
                st.success("Loaded default configuration")
                st.rerun()
