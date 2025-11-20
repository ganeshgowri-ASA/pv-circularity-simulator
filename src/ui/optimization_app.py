"""
OptimizationUI: Interactive Streamlit interface for PV system optimization.

This module provides a comprehensive web-based UI for running optimizations,
visualizing results, comparing designs, and exploring the design space.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    OptimizationObjectives,
    DesignPoint,
)
from ..optimization.system_optimizer import SystemOptimizer
from ..optimization.energy_yield_optimizer import EnergyYieldOptimizer
from ..optimization.economic_optimizer import EconomicOptimizer
from ..optimization.layout_optimizer import LayoutOptimizer
from ..optimization.design_space_explorer import DesignSpaceExplorer, ParameterRange


class OptimizationUI:
    """
    Interactive Streamlit UI for PV system optimization.

    This class provides a comprehensive web interface with:
    - Multi-objective optimization sliders
    - Pareto curve visualization
    - Design comparison tools
    - Parameter sweep visualization
    - Sensitivity analysis charts
    - Results export functionality
    """

    def __init__(self) -> None:
        """Initialize the optimization UI."""
        st.set_page_config(
            page_title="PV System Optimization Engine",
            page_icon="☀️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize session state
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = []
        if 'current_parameters' not in st.session_state:
            st.session_state.current_parameters = None
        if 'pareto_front' not in st.session_state:
            st.session_state.pareto_front = None

    def run(self) -> None:
        """Run the Streamlit application."""
        st.title("☀️ PV System Optimization Engine")
        st.markdown("Advanced multi-objective optimization for photovoltaic system design")

        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "System Configuration",
                "Single-Objective Optimization",
                "Multi-Objective Optimization",
                "Design Space Exploration",
                "Sensitivity Analysis",
                "Results Comparison",
            ],
        )

        if page == "System Configuration":
            self.page_system_configuration()
        elif page == "Single-Objective Optimization":
            self.page_single_objective()
        elif page == "Multi-Objective Optimization":
            self.page_multi_objective()
        elif page == "Design Space Exploration":
            self.page_design_space()
        elif page == "Sensitivity Analysis":
            self.page_sensitivity_analysis()
        elif page == "Results Comparison":
            self.page_results_comparison()

    def page_system_configuration(self) -> None:
        """System configuration page."""
        st.header("System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Module Parameters")
            module_power = st.number_input("Module Power (W)", value=450.0, min_value=100.0, max_value=1000.0)
            module_efficiency = st.slider("Module Efficiency", 0.15, 0.25, 0.20, 0.01)
            module_area = st.number_input("Module Area (m²)", value=2.5, min_value=1.0, max_value=5.0)
            module_cost = st.number_input("Module Cost ($)", value=150.0, min_value=50.0, max_value=500.0)
            bifacial = st.checkbox("Bifacial Modules", value=False)
            bifaciality = st.slider("Bifaciality Factor", 0.6, 0.9, 0.7, 0.05) if bifacial else 0.7

            st.subheader("Tracker Configuration")
            tracker_type = st.selectbox("Tracker Type", ["fixed", "single_axis", "dual_axis"])

        with col2:
            st.subheader("Site Parameters")
            latitude = st.number_input("Latitude (°)", value=35.0, min_value=-90.0, max_value=90.0)
            longitude = st.number_input("Longitude (°)", value=-120.0, min_value=-180.0, max_value=180.0)
            available_land = st.number_input("Available Land (acres)", value=100.0, min_value=1.0, max_value=10000.0)
            land_cost = st.number_input("Land Cost ($/acre)", value=5000.0, min_value=0.0)
            albedo = st.slider("Ground Albedo", 0.1, 0.9, 0.2, 0.05)

            st.subheader("Economic Parameters")
            discount_rate = st.slider("Discount Rate", 0.03, 0.15, 0.08, 0.01)
            project_lifetime = st.number_input("Project Lifetime (years)", value=25, min_value=10, max_value=50)
            degradation_rate = st.slider("Annual Degradation", 0.001, 0.01, 0.005, 0.001)

        # System sizing
        st.subheader("System Sizing")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_modules = st.number_input("Number of Modules", value=10000, min_value=100, max_value=1000000)
        with col2:
            gcr = st.slider("Ground Coverage Ratio", 0.2, 0.7, 0.4, 0.05)
        with col3:
            dc_ac_ratio = st.slider("DC/AC Ratio", 1.0, 2.0, 1.25, 0.05)

        # Create parameters object
        if st.button("Save Configuration", type="primary"):
            parameters = PVSystemParameters(
                module_power=module_power,
                module_efficiency=module_efficiency,
                module_area=module_area,
                module_cost=module_cost,
                bifacial=bifacial,
                bifaciality=bifaciality,
                tracker_type=tracker_type,
                latitude=latitude,
                longitude=longitude,
                available_land_acres=available_land,
                land_cost_per_acre=land_cost,
                albedo=albedo,
                discount_rate=discount_rate,
                project_lifetime=project_lifetime,
                degradation_rate=degradation_rate,
                num_modules=num_modules,
                gcr=gcr,
                dc_ac_ratio=dc_ac_ratio,
            )
            st.session_state.current_parameters = parameters
            st.success("Configuration saved!")

        # Display current configuration
        if st.session_state.current_parameters:
            capacity_mw = st.session_state.current_parameters.num_modules * st.session_state.current_parameters.module_power / 1e6
            st.info(f"System Capacity: {capacity_mw:.2f} MW DC")

    def page_single_objective(self) -> None:
        """Single-objective optimization page."""
        st.header("Single-Objective Optimization")

        if st.session_state.current_parameters is None:
            st.warning("Please configure system parameters first!")
            return

        # Algorithm selection
        algorithm = st.selectbox(
            "Optimization Algorithm",
            ["Genetic Algorithm", "Particle Swarm Optimization", "Linear Programming"],
        )

        # Objective selection
        objective = st.selectbox(
            "Optimization Objective",
            ["Minimize LCOE", "Maximize Energy", "Maximize NPV", "Minimize Land Use"],
        )

        # Constraints
        st.subheader("Optimization Constraints")
        col1, col2 = st.columns(2)

        with col1:
            min_gcr = st.slider("Min GCR", 0.1, 0.5, 0.2, 0.05)
            max_gcr = st.slider("Max GCR", 0.4, 0.9, 0.6, 0.05)
            min_tilt = st.slider("Min Tilt (°)", 0, 30, 10, 5)
        with col2:
            max_tilt = st.slider("Max Tilt (°)", 20, 60, 40, 5)
            min_dc_ac = st.slider("Min DC/AC", 1.0, 1.3, 1.1, 0.1)
            max_dc_ac = st.slider("Max DC/AC", 1.2, 2.0, 1.5, 0.1)

        constraints = OptimizationConstraints(
            min_gcr=min_gcr,
            max_gcr=max_gcr,
            min_tilt=min_tilt,
            max_tilt=max_tilt,
            min_dc_ac_ratio=min_dc_ac,
            max_dc_ac_ratio=max_dc_ac,
        )

        # Algorithm parameters
        if algorithm == "Genetic Algorithm":
            col1, col2 = st.columns(2)
            with col1:
                population = st.number_input("Population Size", value=100, min_value=10, max_value=500)
            with col2:
                generations = st.number_input("Generations", value=50, min_value=10, max_value=200)

        elif algorithm == "Particle Swarm Optimization":
            col1, col2 = st.columns(2)
            with col1:
                swarm_size = st.number_input("Swarm Size", value=50, min_value=10, max_value=200)
            with col2:
                iterations = st.number_input("Iterations", value=100, min_value=10, max_value=500)

        # Run optimization
        if st.button("Run Optimization", type="primary"):
            with st.spinner("Running optimization..."):
                # Create objectives based on selection
                objectives = OptimizationObjectives()
                if objective == "Minimize LCOE":
                    objectives.minimize_lcoe = 1.0
                    objectives.maximize_energy = 0.0
                    objectives.maximize_npv = 0.0
                    objectives.minimize_land_use = 0.0
                elif objective == "Maximize Energy":
                    objectives.maximize_energy = 1.0
                    objectives.minimize_lcoe = 0.0
                elif objective == "Maximize NPV":
                    objectives.maximize_npv = 1.0
                    objectives.minimize_lcoe = 0.0
                elif objective == "Minimize Land Use":
                    objectives.minimize_land_use = 1.0

                optimizer = SystemOptimizer(
                    st.session_state.current_parameters,
                    constraints,
                    objectives,
                )

                # Run selected algorithm
                if algorithm == "Genetic Algorithm":
                    result = optimizer.genetic_algorithm_optimizer(
                        population_size=population,
                        num_generations=generations,
                    )
                elif algorithm == "Particle Swarm Optimization":
                    result = optimizer.particle_swarm_optimizer(
                        swarm_size=swarm_size,
                        max_iterations=iterations,
                    )
                else:  # Linear Programming
                    result = optimizer.linear_programming_optimizer()

                # Store result
                st.session_state.optimization_results.append({
                    'timestamp': datetime.now().isoformat(),
                    'algorithm': algorithm,
                    'objective': objective,
                    'result': result,
                })

                # Display results
                st.success("Optimization completed!")
                self._display_optimization_result(result)

    def page_multi_objective(self) -> None:
        """Multi-objective optimization page."""
        st.header("Multi-Objective Optimization")

        if st.session_state.current_parameters is None:
            st.warning("Please configure system parameters first!")
            return

        st.markdown("""
        Multi-objective optimization finds the **Pareto frontier** of trade-offs
        between competing objectives. Adjust the weights to emphasize different goals.
        """)

        # Objective weights
        st.subheader("Objective Weights")
        col1, col2, col3 = st.columns(3)

        with col1:
            w_energy = st.slider("Energy Yield", 0.0, 1.0, 0.8, 0.1)
            w_lcoe = st.slider("LCOE", 0.0, 1.0, 0.8, 0.1)
        with col2:
            w_land = st.slider("Land Use", 0.0, 1.0, 0.3, 0.1)
            w_npv = st.slider("NPV", 0.0, 1.0, 0.6, 0.1)
        with col3:
            w_shading = st.slider("Shading Loss", 0.0, 1.0, 0.4, 0.1)
            w_bifacial = st.slider("Bifacial Gain", 0.0, 1.0, 0.2, 0.1)

        objectives = OptimizationObjectives(
            maximize_energy=w_energy,
            minimize_lcoe=w_lcoe,
            minimize_land_use=w_land,
            maximize_npv=w_npv,
            minimize_shading=w_shading,
            maximize_bifacial_gain=w_bifacial,
        )

        # Constraints (simplified)
        constraints = OptimizationConstraints()

        # Optimization parameters
        col1, col2 = st.columns(2)
        with col1:
            population = st.number_input("Population Size", value=100, min_value=20, max_value=500)
        with col2:
            generations = st.number_input("Generations", value=50, min_value=10, max_value=200)

        # Run optimization
        if st.button("Run Multi-Objective Optimization", type="primary"):
            with st.spinner("Running NSGA-II optimization..."):
                optimizer = SystemOptimizer(
                    st.session_state.current_parameters,
                    constraints,
                    objectives,
                )

                result = optimizer.multi_objective_optimization(
                    population_size=population,
                    num_generations=generations,
                )

                st.session_state.pareto_front = result.pareto_front

                st.success(f"Found {len(result.pareto_front)} Pareto-optimal solutions!")

                # Visualize Pareto front
                self._plot_pareto_front(result.pareto_front)

    def page_design_space(self) -> None:
        """Design space exploration page."""
        st.header("Design Space Exploration")

        if st.session_state.current_parameters is None:
            st.warning("Please configure system parameters first!")
            return

        exploration_type = st.selectbox(
            "Exploration Type",
            ["Parameter Sweep", "Monte Carlo Simulation"],
        )

        if exploration_type == "Parameter Sweep":
            st.subheader("Parameter Sweep")

            # Select parameters to sweep
            param1 = st.selectbox("Parameter 1", ["gcr", "dc_ac_ratio", "tilt_angle"])
            param1_min = st.number_input(f"{param1} Min", value=0.2)
            param1_max = st.number_input(f"{param1} Max", value=0.6)

            sweep_2d = st.checkbox("2D Sweep (add second parameter)")

            if sweep_2d:
                param2 = st.selectbox("Parameter 2", ["gcr", "dc_ac_ratio", "tilt_angle"])
                param2_min = st.number_input(f"{param2} Min", value=1.1)
                param2_max = st.number_input(f"{param2} Max", value=1.5)

            output_metric = st.selectbox("Output Metric", ["lcoe", "annual_energy_kwh", "npv"])

            if st.button("Run Parameter Sweep"):
                with st.spinner("Running sweep..."):
                    explorer = DesignSpaceExplorer(
                        st.session_state.current_parameters,
                        OptimizationConstraints(),
                    )

                    ranges = [ParameterRange(param1, param1_min, param1_max, 20)]
                    if sweep_2d:
                        ranges.append(ParameterRange(param2, param2_min, param2_max, 20))

                    results = explorer.parameter_sweep(
                        ranges,
                        output_metric=output_metric,
                        parallel=True,
                    )

                    # Visualize
                    if results['dimension'] == 1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['values'],
                            y=results['results'],
                            mode='lines+markers',
                        ))
                        fig.update_layout(
                            title=f"{output_metric} vs {param1}",
                            xaxis_title=param1,
                            yaxis_title=output_metric,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif results['dimension'] == 2:
                        fig = go.Figure(data=[go.Surface(
                            x=results['param1_values'],
                            y=results['param2_values'],
                            z=results['results'],
                        )])
                        fig.update_layout(
                            title=f"{output_metric} Surface",
                            scene=dict(
                                xaxis_title=param1,
                                yaxis_title=param2,
                                zaxis_title=output_metric,
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)

        else:  # Monte Carlo
            st.subheader("Monte Carlo Simulation")

            num_samples = st.number_input("Number of Samples", value=1000, min_value=100, max_value=10000)

            if st.button("Run Monte Carlo"):
                with st.spinner(f"Running {num_samples} simulations..."):
                    explorer = DesignSpaceExplorer(
                        st.session_state.current_parameters,
                        OptimizationConstraints(),
                    )

                    results = explorer.monte_carlo_simulation(
                        num_samples=num_samples,
                        parallel=True,
                    )

                    # Display statistics
                    st.subheader("Results Statistics")
                    df_stats = pd.DataFrame(results['statistics']).T
                    st.dataframe(df_stats)

                    # Plot distributions
                    for metric in results['output_metrics']:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=results['output_data'][metric],
                            nbinsx=50,
                            name=metric,
                        ))
                        fig.update_layout(
                            title=f"{metric} Distribution",
                            xaxis_title=metric,
                            yaxis_title="Frequency",
                        )
                        st.plotly_chart(fig, use_container_width=True)

    def page_sensitivity_analysis(self) -> None:
        """Sensitivity analysis page."""
        st.header("Sensitivity Analysis")

        if st.session_state.current_parameters is None:
            st.warning("Please configure system parameters first!")
            return

        # Select parameters to analyze
        parameters_to_vary = st.multiselect(
            "Parameters to Analyze",
            ["module_efficiency", "module_cost", "discount_rate", "degradation_rate", "gcr"],
            default=["module_efficiency", "module_cost"],
        )

        variation_percent = st.slider("Variation (%)", 5, 30, 10, 5)

        if st.button("Run Sensitivity Analysis"):
            with st.spinner("Analyzing..."):
                explorer = DesignSpaceExplorer(
                    st.session_state.current_parameters,
                    OptimizationConstraints(),
                )

                results = explorer.sensitivity_analysis(
                    parameters_to_vary=parameters_to_vary,
                    variation_percent=variation_percent,
                )

                # Create tornado chart
                sensitivities = {}
                for result in results:
                    if "lcoe" in result.parameter_name:
                        param = result.parameter_name.replace("_lcoe", "")
                        sensitivities[param] = result.sensitivity_index

                if sensitivities:
                    sorted_params = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=[p[0] for p in sorted_params],
                        x=[p[1] for p in sorted_params],
                        orientation='h',
                    ))
                    fig.update_layout(
                        title="Sensitivity Analysis (Tornado Chart)",
                        xaxis_title="Sensitivity Index",
                        yaxis_title="Parameter",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show detailed results
                st.subheader("Detailed Results")
                for result in results:
                    with st.expander(result.parameter_name):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=result.parameter_values,
                            y=result.output_values,
                            mode='lines+markers',
                        ))
                        fig.update_layout(
                            xaxis_title="Parameter Value",
                            yaxis_title="Output",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("Correlation", f"{result.correlation:.3f}")

    def page_results_comparison(self) -> None:
        """Results comparison page."""
        st.header("Results Comparison")

        if not st.session_state.optimization_results:
            st.info("No optimization results to compare yet. Run some optimizations first!")
            return

        # Create comparison table
        comparison_data = []
        for i, result in enumerate(st.session_state.optimization_results):
            best = result['result'].best_solution
            comparison_data.append({
                'Run': i + 1,
                'Algorithm': result['algorithm'],
                'Objective': result['objective'],
                'LCOE ($/kWh)': f"{best.lcoe:.4f}",
                'Energy (MWh)': f"{best.annual_energy_kwh/1000:.0f}",
                'NPV ($M)': f"{best.npv/1e6:.2f}",
                'GCR': f"{best.gcr:.2f}",
                'DC/AC': f"{best.dc_ac_ratio:.2f}",
                'Timestamp': result['timestamp'],
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(df)

        # Export results
        if st.button("Export Results"):
            json_str = json.dumps(
                [
                    {
                        **r,
                        'result': {
                            'best_solution': r['result'].best_solution.model_dump(),
                            'execution_time': r['result'].execution_time_seconds,
                            'success': r['result'].success,
                        }
                    }
                    for r in st.session_state.optimization_results
                ],
                indent=2,
            )
            st.download_button(
                label="Download Results (JSON)",
                data=json_str,
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    def _display_optimization_result(self, result: Any) -> None:
        """Display optimization result."""
        best = result.best_solution

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LCOE", f"${best.lcoe:.4f}/kWh")
        col2.metric("Annual Energy", f"{best.annual_energy_kwh/1000:.0f} MWh")
        col3.metric("NPV", f"${best.npv/1e6:.2f}M")
        col4.metric("Capacity", f"{best.capacity_mw:.2f} MW")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GCR", f"{best.gcr:.3f}")
        col2.metric("DC/AC Ratio", f"{best.dc_ac_ratio:.2f}")
        col3.metric("Tilt Angle", f"{best.tilt_angle:.1f}°")
        col4.metric("Land Use", f"{best.land_use_acres:.1f} acres")

        # Convergence plot
        if result.convergence_history:
            st.subheader("Convergence History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.convergence_history,
                mode='lines',
            ))
            fig.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Objective Value",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Performance summary
        st.info(f"Optimization completed in {result.execution_time_seconds:.2f} seconds "
                f"with {result.num_evaluations} function evaluations")

    def _plot_pareto_front(self, pareto_front: List[Any]) -> None:
        """Plot Pareto frontier."""
        if not pareto_front:
            return

        st.subheader("Pareto Frontier")

        # Extract objectives
        lcoe_values = [s.design.lcoe for s in pareto_front]
        energy_values = [s.design.annual_energy_kwh / 1000 for s in pareto_front]
        land_values = [s.design.land_use_acres for s in pareto_front]

        # 2D Pareto plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lcoe_values,
            y=energy_values,
            mode='markers',
            marker=dict(
                size=10,
                color=land_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Land Use (acres)"),
            ),
            text=[f"GCR: {s.design.gcr:.2f}<br>DC/AC: {s.design.dc_ac_ratio:.2f}"
                  for s in pareto_front],
            hovertemplate="LCOE: $%{x:.4f}/kWh<br>Energy: %{y:.0f} MWh<br>%{text}",
        ))
        fig.update_layout(
            title="Pareto Frontier: LCOE vs Energy",
            xaxis_title="LCOE ($/kWh)",
            yaxis_title="Annual Energy (MWh)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3D Pareto plot
        fig3d = go.Figure(data=[go.Scatter3d(
            x=lcoe_values,
            y=energy_values,
            z=land_values,
            mode='markers',
            marker=dict(size=6, color=energy_values, colorscale='Viridis'),
        )])
        fig3d.update_layout(
            title="3D Pareto Frontier",
            scene=dict(
                xaxis_title="LCOE ($/kWh)",
                yaxis_title="Energy (MWh)",
                zaxis_title="Land Use (acres)",
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)


def main() -> None:
    """Main entry point for the Streamlit app."""
    app = OptimizationUI()
    app.run()


if __name__ == "__main__":
    main()
