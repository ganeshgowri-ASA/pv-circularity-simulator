"""
Interactive forecast dashboards using Streamlit and Plotly.

This module provides visualization tools for forecasting results including
interactive charts, confidence intervals, and scenario analysis.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pv_simulator.core.schemas import (
    ForecastResult,
    ModelMetrics,
    SeasonalDecomposition,
    TimeSeriesData,
)


class ForecastDashboard:
    """
    Interactive forecast visualization dashboard.

    Provides methods to create interactive charts with Plotly and
    build Streamlit dashboards for forecast analysis.
    """

    def __init__(self, title: str = "Forecast Dashboard") -> None:
        """
        Initialize forecast dashboard.

        Args:
            title: Dashboard title
        """
        self.title = title

    def interactive_charts(
        self,
        actual: Optional[TimeSeriesData] = None,
        forecast: Optional[ForecastResult] = None,
        title: str = "Time Series Forecast",
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create interactive forecast chart with Plotly.

        Args:
            actual: Actual time series data
            forecast: Forecast results
            title: Chart title
            **kwargs: Additional plotting parameters

        Returns:
            Plotly Figure object

        Example:
            >>> dashboard = ForecastDashboard()
            >>> fig = dashboard.interactive_charts(actual_data, forecast_result)
            >>> fig.show()
        """
        fig = go.Figure()

        # Plot actual data
        if actual is not None:
            fig.add_trace(
                go.Scatter(
                    x=actual.timestamps,
                    y=actual.values,
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="blue", width=2),
                    marker=dict(size=4),
                )
            )

        # Plot forecast
        if forecast is not None:
            # Main forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast.timestamps,
                    y=forecast.predictions,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="red", width=2, dash="dash"),
                    marker=dict(size=4),
                )
            )

            # Confidence intervals
            if forecast.lower_bound and forecast.upper_bound:
                # Upper bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast.timestamps,
                        y=forecast.upper_bound,
                        mode="lines",
                        name=f"Upper {forecast.confidence_level:.0%} CI",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )

                # Lower bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast.timestamps,
                        y=forecast.lower_bound,
                        mode="lines",
                        name=f"Confidence Interval ({forecast.confidence_level:.0%})",
                        fill="tonexty",
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        line=dict(width=0),
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def confidence_intervals(
        self,
        forecast: ForecastResult,
        additional_levels: Optional[List[float]] = None,
    ) -> go.Figure:
        """
        Create chart showing multiple confidence intervals.

        Args:
            forecast: Forecast with confidence intervals
            additional_levels: Additional confidence levels to show

        Returns:
            Plotly Figure with confidence bands

        Example:
            >>> fig = dashboard.confidence_intervals(
            ...     forecast, additional_levels=[0.50, 0.80, 0.95]
            ... )
        """
        fig = go.Figure()

        # Main forecast
        fig.add_trace(
            go.Scatter(
                x=forecast.timestamps,
                y=forecast.predictions,
                mode="lines",
                name="Forecast",
                line=dict(color="black", width=2),
            )
        )

        # Confidence intervals (if available)
        if forecast.lower_bound and forecast.upper_bound:
            colors = ["rgba(255, 0, 0, 0.1)", "rgba(255, 100, 0, 0.15)", "rgba(255, 150, 0, 0.2)"]

            if additional_levels:
                # Simulate different confidence levels
                # (In practice, these would come from the model)
                for i, level in enumerate(sorted(additional_levels, reverse=True)):
                    scale = (1.0 - level) / (1.0 - forecast.confidence_level)
                    upper = [
                        p + (ub - p) * scale
                        for p, ub in zip(forecast.predictions, forecast.upper_bound)
                    ]
                    lower = [
                        p - (p - lb) * scale
                        for p, lb in zip(forecast.predictions, forecast.lower_bound)
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=forecast.timestamps,
                            y=upper,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=forecast.timestamps,
                            y=lower,
                            mode="lines",
                            fill="tonexty",
                            fillcolor=colors[i % len(colors)],
                            line=dict(width=0),
                            name=f"{level:.0%} CI",
                        )
                    )

        fig.update_layout(
            title="Forecast with Confidence Intervals",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def scenario_analysis(
        self, scenarios: Dict[str, ForecastResult], actual: Optional[TimeSeriesData] = None
    ) -> go.Figure:
        """
        Create chart comparing multiple forecast scenarios.

        Args:
            scenarios: Dictionary mapping scenario names to forecasts
            actual: Optional actual data to include

        Returns:
            Plotly Figure with scenario comparison

        Example:
            >>> scenarios = {
            ...     'base': base_forecast,
            ...     'optimistic': optimistic_forecast,
            ...     'pessimistic': pessimistic_forecast
            ... }
            >>> fig = dashboard.scenario_analysis(scenarios)
        """
        fig = go.Figure()

        # Plot actual data
        if actual is not None:
            fig.add_trace(
                go.Scatter(
                    x=actual.timestamps,
                    y=actual.values,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue", width=2),
                )
            )

        # Define colors for scenarios
        colors = {
            "base": "green",
            "optimistic": "lightgreen",
            "pessimistic": "orange",
            "best": "darkgreen",
            "worst": "red",
        }

        # Plot each scenario
        for scenario_name, forecast in scenarios.items():
            color = colors.get(scenario_name.lower(), "gray")

            fig.add_trace(
                go.Scatter(
                    x=forecast.timestamps,
                    y=forecast.predictions,
                    mode="lines",
                    name=scenario_name.capitalize(),
                    line=dict(color=color, width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title="Scenario Analysis",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def seasonal_decomposition_chart(
        self, decomposition: SeasonalDecomposition, timestamps: List
    ) -> go.Figure:
        """
        Create chart showing seasonal decomposition components.

        Args:
            decomposition: Seasonal decomposition results
            timestamps: Timestamps for the components

        Returns:
            Plotly Figure with decomposition components

        Example:
            >>> fig = dashboard.seasonal_decomposition_chart(decomp, timestamps)
        """
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Trend", "Seasonal", "Residual"),
            vertical_spacing=0.1,
        )

        # Trend component
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=decomposition.trend,
                mode="lines",
                name="Trend",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Seasonal component
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=decomposition.seasonal,
                mode="lines",
                name="Seasonal",
                line=dict(color="green", width=2),
            ),
            row=2,
            col=1,
        )

        # Residual component
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=decomposition.residual,
                mode="lines",
                name="Residual",
                line=dict(color="red", width=1),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=900,
            title_text="Seasonal Decomposition",
            showlegend=False,
            template="plotly_white",
        )

        return fig

    def metrics_comparison_chart(
        self, metrics_dict: Dict[str, ModelMetrics]
    ) -> go.Figure:
        """
        Create bar chart comparing metrics across models.

        Args:
            metrics_dict: Dictionary mapping model names to metrics

        Returns:
            Plotly Figure with metrics comparison

        Example:
            >>> metrics = {
            ...     'ARIMA': arima_metrics,
            ...     'Prophet': prophet_metrics,
            ...     'XGBoost': xgb_metrics
            ... }
            >>> fig = dashboard.metrics_comparison_chart(metrics)
        """
        models = list(metrics_dict.keys())
        mae_values = [m.mae for m in metrics_dict.values()]
        rmse_values = [m.rmse for m in metrics_dict.values()]
        mape_values = [m.mape for m in metrics_dict.values()]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("MAE", "RMSE", "MAPE"),
        )

        # MAE
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name="MAE", marker_color="blue"),
            row=1,
            col=1,
        )

        # RMSE
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name="RMSE", marker_color="green"),
            row=1,
            col=2,
        )

        # MAPE
        fig.add_trace(
            go.Bar(x=models, y=mape_values, name="MAPE (%)", marker_color="red"),
            row=1,
            col=3,
        )

        fig.update_layout(
            height=400,
            title_text="Model Performance Comparison",
            showlegend=False,
            template="plotly_white",
        )

        return fig

    def residual_analysis_chart(
        self, actual: TimeSeriesData, forecast: ForecastResult
    ) -> go.Figure:
        """
        Create residual analysis charts.

        Args:
            actual: Actual time series data
            forecast: Forecast results

        Returns:
            Plotly Figure with residual analysis

        Example:
            >>> fig = dashboard.residual_analysis_chart(actual, forecast)
        """
        # Calculate residuals
        residuals = np.array(actual.values) - np.array(forecast.predictions)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Residuals Over Time",
                "Residual Distribution",
                "Q-Q Plot",
                "ACF of Residuals",
            ),
        )

        # Residuals over time
        fig.add_trace(
            go.Scatter(
                x=actual.timestamps,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="blue", size=4),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # Histogram of residuals
        fig.add_trace(
            go.Histogram(x=residuals, name="Distribution", marker_color="green"),
            row=1,
            col=2,
        )

        # Q-Q plot (simplified)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(
            sorted_residuals.min(), sorted_residuals.max(), len(sorted_residuals)
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode="markers",
                name="Q-Q",
                marker=dict(color="purple", size=4),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles,
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # ACF of residuals (simplified)
        from statsmodels.tsa.stattools import acf

        acf_values = acf(residuals, nlags=20)
        fig.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color="orange"),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title_text="Residual Analysis",
            showlegend=False,
            template="plotly_white",
        )

        return fig

    def build_streamlit_dashboard(
        self,
        actual: Optional[TimeSeriesData] = None,
        forecast: Optional[ForecastResult] = None,
        metrics: Optional[ModelMetrics] = None,
        decomposition: Optional[SeasonalDecomposition] = None,
    ) -> None:
        """
        Build complete Streamlit dashboard.

        This method creates a full interactive dashboard using Streamlit.
        Should be called from a Streamlit app script.

        Args:
            actual: Actual time series data
            forecast: Forecast results
            metrics: Model metrics
            decomposition: Seasonal decomposition

        Example:
            >>> # In a Streamlit script:
            >>> dashboard = ForecastDashboard("PV Energy Forecast")
            >>> dashboard.build_streamlit_dashboard(
            ...     actual=data, forecast=forecast, metrics=metrics
            ... )
        """
        st.set_page_config(page_title=self.title, layout="wide")
        st.title(self.title)

        # Sidebar
        st.sidebar.header("Dashboard Controls")

        # Main forecast chart
        st.header("Forecast Overview")
        if actual is not None and forecast is not None:
            fig = self.interactive_charts(actual, forecast)
            st.plotly_chart(fig, use_container_width=True)

        # Metrics
        if metrics is not None:
            st.header("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("MAE", f"{metrics.mae:.2f}")
            with col2:
                st.metric("RMSE", f"{metrics.rmse:.2f}")
            with col3:
                st.metric("MAPE", f"{metrics.mape:.2f}%")
            with col4:
                st.metric("R²", f"{metrics.r2:.3f}")

        # Confidence intervals
        if forecast is not None and forecast.lower_bound is not None:
            st.header("Confidence Intervals")
            fig = self.confidence_intervals(forecast)
            st.plotly_chart(fig, use_container_width=True)

        # Seasonal decomposition
        if decomposition is not None and actual is not None:
            st.header("Seasonal Decomposition")
            fig = self.seasonal_decomposition_chart(decomposition, actual.timestamps)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Seasonality Strength", f"{decomposition.seasonality_strength:.3f}")
            with col2:
                st.metric("Trend Strength", f"{decomposition.trend_strength:.3f}")

        # Residual analysis
        if actual is not None and forecast is not None:
            st.header("Residual Analysis")
            fig = self.residual_analysis_chart(actual, forecast)
            st.plotly_chart(fig, use_container_width=True)

        # Footer
        st.markdown("---")
        st.caption(f"Dashboard generated by PV Circularity Simulator v0.1.0")
