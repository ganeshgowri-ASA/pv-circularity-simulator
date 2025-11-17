"""
Forecast Dashboard with accuracy metrics and visualization.

This module provides comprehensive tools for:
- Calculating forecast accuracy metrics (MAE, RMSE, MAPE, etc.)
- Generating confidence intervals for predictions
- Creating interactive Plotly dashboards for forecast visualization
- Statistical validation and error analysis

The dashboard is production-ready with full type hints, validation,
and comprehensive documentation.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pv_simulator.forecasting.models import (
    ForecastPoint,
    ForecastSeries,
    AccuracyMetrics,
    ForecastData,
)


def mae_rmse_calculation(
    actual: Union[np.ndarray, List[float]],
    predicted: Union[np.ndarray, List[float]],
) -> Tuple[float, float, float]:
    """
    Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    This function computes fundamental forecast accuracy metrics:
    - MAE: Average absolute difference between predicted and actual values
    - RMSE: Square root of average squared differences
    - MSE: Mean Squared Error (intermediate calculation)

    Args:
        actual: Array of actual observed values
        predicted: Array of predicted values

    Returns:
        Tuple[float, float, float]: (MAE, RMSE, MSE)

    Raises:
        ValueError: If arrays are empty, have different lengths, or contain NaN/Inf

    Examples:
        >>> actual = np.array([100, 110, 105, 115])
        >>> predicted = np.array([98, 112, 107, 113])
        >>> mae, rmse, mse = mae_rmse_calculation(actual, predicted)
        >>> mae >= 0
        True
        >>> rmse >= mae  # RMSE is always >= MAE
        True

    Notes:
        - RMSE penalizes larger errors more heavily than MAE
        - Both metrics are in the same units as the original data
        - Lower values indicate better forecast accuracy
        - RMSE is more sensitive to outliers than MAE

    References:
        - Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: principles and practice
    """
    # Convert to numpy arrays
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    # Validation
    if len(actual_arr) == 0 or len(predicted_arr) == 0:
        raise ValueError("Input arrays cannot be empty")

    if len(actual_arr) != len(predicted_arr):
        raise ValueError(
            f"Array length mismatch: actual={len(actual_arr)}, predicted={len(predicted_arr)}"
        )

    if np.any(np.isnan(actual_arr)) or np.any(np.isnan(predicted_arr)):
        raise ValueError("Input arrays cannot contain NaN values")

    if np.any(np.isinf(actual_arr)) or np.any(np.isinf(predicted_arr)):
        raise ValueError("Input arrays cannot contain infinite values")

    # Calculate metrics
    mae = mean_absolute_error(actual_arr, predicted_arr)
    mse = mean_squared_error(actual_arr, predicted_arr)
    rmse = np.sqrt(mse)

    return mae, rmse, mse


def accuracy_metrics(
    actual: Union[np.ndarray, List[float]],
    predicted: Union[np.ndarray, List[float]],
    include_mape: bool = True,
    include_r2: bool = True,
) -> AccuracyMetrics:
    """
    Calculate comprehensive forecast accuracy metrics.

    Computes a full suite of accuracy metrics including:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MSE (Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)
    - Bias (Mean Error)

    Args:
        actual: Array of actual observed values
        predicted: Array of predicted values
        include_mape: Whether to calculate MAPE (requires non-zero actuals)
        include_r2: Whether to calculate R² score

    Returns:
        AccuracyMetrics: Pydantic model containing all calculated metrics

    Raises:
        ValueError: If arrays are invalid or incompatible

    Examples:
        >>> actual = np.array([100, 110, 105, 115, 120])
        >>> predicted = np.array([98, 112, 107, 113, 122])
        >>> metrics = accuracy_metrics(actual, predicted)
        >>> metrics.mae > 0
        True
        >>> metrics.rmse > 0
        True
        >>> metrics.n_samples
        5

    Notes:
        - MAPE is undefined when actual values are zero
        - R² can be negative for very poor models
        - Bias indicates systematic over/under-prediction
        - All metrics are computed on the same sample set

    References:
        - Willmott, C.J., & Matsuura, K. (2005). Advantages of the mean absolute error
    """
    # Convert to numpy arrays
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    # Calculate MAE, RMSE, MSE
    mae, rmse, mse = mae_rmse_calculation(actual_arr, predicted_arr)

    # Calculate bias (mean error)
    bias = float(np.mean(predicted_arr - actual_arr))

    # Calculate MAPE if requested and possible
    mape = None
    if include_mape:
        # Avoid division by zero
        non_zero_mask = actual_arr != 0
        if np.any(non_zero_mask):
            mape_values = np.abs((actual_arr[non_zero_mask] - predicted_arr[non_zero_mask]) /
                                actual_arr[non_zero_mask])
            mape = float(np.mean(mape_values) * 100)  # Convert to percentage

    # Calculate R² if requested
    r2 = None
    if include_r2:
        try:
            r2 = float(r2_score(actual_arr, predicted_arr))
        except Exception:
            # R² calculation can fail for certain edge cases
            r2 = None

    # Create and return AccuracyMetrics model
    return AccuracyMetrics(
        mae=mae,
        rmse=rmse,
        mse=mse,
        mape=mape,
        r2_score=r2,
        bias=bias,
        n_samples=len(actual_arr),
    )


def confidence_intervals(
    predicted: Union[np.ndarray, List[float]],
    residuals: Optional[Union[np.ndarray, List[float]]] = None,
    confidence_level: float = 0.95,
    method: str = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for forecast predictions.

    Generates upper and lower bounds for predictions based on residual
    distribution and specified confidence level. Supports multiple methods
    for interval calculation.

    Args:
        predicted: Array of predicted values
        residuals: Array of residuals (actual - predicted) from training/validation.
                  If None, uses 10% of predicted values as estimated std
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Method for interval calculation:
               - 'normal': Assumes normal distribution of residuals
               - 'percentile': Uses empirical percentiles of residuals
               - 'bootstrap': Bootstrap-based intervals (if residuals provided)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (lower_bounds, upper_bounds)

    Raises:
        ValueError: If inputs are invalid or method is unsupported

    Examples:
        >>> predicted = np.array([100, 110, 105, 115])
        >>> lower, upper = confidence_intervals(predicted, confidence_level=0.95)
        >>> len(lower) == len(predicted)
        True
        >>> np.all(lower <= predicted)
        True
        >>> np.all(upper >= predicted)
        True

    Notes:
        - 'normal' method assumes normally distributed residuals
        - 'percentile' is more robust to non-normal residuals
        - Wider intervals indicate higher uncertainty
        - Confidence level of 0.95 means 95% of actuals should fall within bounds

    References:
        - Chatfield, C. (1993). Calculating interval forecasts. Journal of Business & Economic Statistics
    """
    # Convert to numpy array
    predicted_arr = np.asarray(predicted, dtype=float)

    # Validation
    if len(predicted_arr) == 0:
        raise ValueError("Predicted array cannot be empty")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")

    if method not in ["normal", "percentile", "bootstrap"]:
        raise ValueError(f"Unsupported method: {method}. Use 'normal', 'percentile', or 'bootstrap'")

    # Calculate residual standard deviation
    if residuals is not None:
        residuals_arr = np.asarray(residuals, dtype=float)
        if len(residuals_arr) < 2:
            raise ValueError("Need at least 2 residuals for interval calculation")
        residual_std = np.std(residuals_arr, ddof=1)
    else:
        # Estimate uncertainty as percentage of predicted values
        # Use 10% as a reasonable default for energy forecasting
        residual_std = np.mean(np.abs(predicted_arr)) * 0.10

    if method == "normal":
        # Normal distribution method
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std
        lower_bounds = predicted_arr - margin
        upper_bounds = predicted_arr + margin

    elif method == "percentile":
        if residuals is None:
            # Fall back to normal method if no residuals
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * residual_std
            lower_bounds = predicted_arr - margin
            upper_bounds = predicted_arr + margin
        else:
            # Use empirical percentiles
            residuals_arr = np.asarray(residuals, dtype=float)
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100

            lower_margin = np.abs(np.percentile(residuals_arr, lower_percentile))
            upper_margin = np.abs(np.percentile(residuals_arr, upper_percentile))

            lower_bounds = predicted_arr - upper_margin
            upper_bounds = predicted_arr + lower_margin

    elif method == "bootstrap":
        if residuals is None:
            raise ValueError("Bootstrap method requires residuals")

        # Bootstrap resampling
        residuals_arr = np.asarray(residuals, dtype=float)
        n_bootstrap = 1000
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100

        bootstrap_bounds = []
        for _ in range(n_bootstrap):
            # Resample residuals
            resampled = np.random.choice(residuals_arr, size=len(predicted_arr), replace=True)
            bootstrap_bounds.append(predicted_arr + resampled)

        bootstrap_bounds = np.array(bootstrap_bounds)
        lower_bounds = np.percentile(bootstrap_bounds, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_bounds, upper_percentile, axis=0)

    else:
        raise ValueError(f"Unsupported method: {method}")

    return lower_bounds, upper_bounds


class ForecastDashboard:
    """
    Interactive dashboard for forecast visualization and analysis.

    This class provides comprehensive visualization and analysis tools for
    energy forecasting, including:
    - Time series plots with confidence intervals
    - Accuracy metrics visualization
    - Error distribution analysis
    - Residual plots and diagnostics
    - Interactive Plotly dashboards

    Attributes:
        forecast_data: The forecast data to visualize
        title: Dashboard title
        width: Dashboard width in pixels
        height: Dashboard height in pixels

    Examples:
        >>> from datetime import datetime, timedelta
        >>> # Create sample forecast
        >>> points = [
        ...     ForecastPoint(
        ...         timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
        ...         predicted=100 + i * 2,
        ...         actual=100 + i * 2 + np.random.randn()
        ...     )
        ...     for i in range(24)
        ... ]
        >>> series = ForecastSeries(points=points, model_name="ARIMA")
        >>> forecast = ForecastData(series=series)
        >>> dashboard = ForecastDashboard(forecast)
        >>> fig = dashboard.create_dashboard()
        >>> fig is not None
        True

    Notes:
        - Requires Plotly for visualization
        - Automatically calculates metrics if actuals are available
        - Supports multiple forecast series comparison
        - Production-ready with error handling and validation
    """

    def __init__(
        self,
        forecast_data: ForecastData,
        title: str = "Forecast Dashboard",
        width: int = 1400,
        height: int = 1000,
    ):
        """
        Initialize the Forecast Dashboard.

        Args:
            forecast_data: ForecastData object containing predictions and optionally actuals
            title: Dashboard title
            width: Dashboard width in pixels
            height: Dashboard height in pixels

        Raises:
            ValueError: If forecast_data is invalid
        """
        if not isinstance(forecast_data, ForecastData):
            raise ValueError("forecast_data must be a ForecastData instance")

        if forecast_data.series.length == 0:
            raise ValueError("Forecast series cannot be empty")

        self.forecast_data = forecast_data
        self.title = title
        self.width = width
        self.height = height

        # Calculate metrics if actuals are available
        if self.forecast_data.can_calculate_metrics and self.forecast_data.metrics is None:
            self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """
        Calculate accuracy metrics for the forecast.

        This internal method computes metrics when actuals are available
        and stores them in the forecast_data object.
        """
        actual_values = self.forecast_data.series.get_actual_values()
        predicted_values = self.forecast_data.series.get_predicted_values()

        if actual_values is not None and len(actual_values) == len(predicted_values):
            self.forecast_data.metrics = accuracy_metrics(
                actual=actual_values,
                predicted=predicted_values,
            )

    def forecast_visualization(
        self,
        show_confidence: bool = True,
        show_actuals: bool = True,
        confidence_level: float = 0.95,
    ) -> go.Figure:
        """
        Create an interactive forecast visualization with Plotly.

        Generates a comprehensive time series plot showing:
        - Predicted values
        - Actual values (if available)
        - Confidence intervals (if requested)
        - Forecast metadata

        Args:
            show_confidence: Whether to display confidence intervals
            show_actuals: Whether to display actual values
            confidence_level: Confidence level for intervals (0.0 to 1.0)

        Returns:
            go.Figure: Plotly figure object

        Examples:
            >>> # Assuming dashboard is initialized
            >>> fig = dashboard.forecast_visualization(
            ...     show_confidence=True,
            ...     confidence_level=0.95
            ... )
            >>> fig.layout.title.text  # Check title exists
            'Forecast Dashboard'

        Notes:
            - Returns interactive Plotly figure
            - Can be displayed in Jupyter or exported to HTML
            - Automatically handles missing confidence intervals
        """
        fig = go.Figure()

        # Get data
        timestamps = self.forecast_data.series.get_timestamps()
        predicted = self.forecast_data.series.get_predicted_values()
        actual = self.forecast_data.series.get_actual_values()

        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predicted,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
            )
        )

        # Add actual values if available
        if show_actuals and actual is not None:
            # Filter timestamps for points with actual values
            actual_timestamps = [
                point.timestamp for point in self.forecast_data.series.points
                if point.actual is not None
            ]
            actual_values = [
                point.actual for point in self.forecast_data.series.points
                if point.actual is not None
            ]

            fig.add_trace(
                go.Scatter(
                    x=actual_timestamps,
                    y=actual_values,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='green', width=2),
                    marker=dict(size=6),
                )
            )

        # Add confidence intervals
        if show_confidence:
            # Check if points already have confidence bounds
            has_bounds = any(
                point.lower_bound is not None and point.upper_bound is not None
                for point in self.forecast_data.series.points
            )

            if has_bounds:
                # Use existing bounds
                lower = np.array([
                    point.lower_bound if point.lower_bound is not None else point.predicted
                    for point in self.forecast_data.series.points
                ])
                upper = np.array([
                    point.upper_bound if point.upper_bound is not None else point.predicted
                    for point in self.forecast_data.series.points
                ])
            else:
                # Calculate confidence intervals
                if actual is not None and len(actual) == len(predicted):
                    residuals = actual - predicted[:len(actual)]
                    lower, upper = confidence_intervals(
                        predicted=predicted,
                        residuals=residuals,
                        confidence_level=confidence_level,
                    )
                else:
                    lower, upper = confidence_intervals(
                        predicted=predicted,
                        confidence_level=confidence_level,
                    )

            # Add confidence interval as filled area
            fig.add_trace(
                go.Scatter(
                    x=timestamps + timestamps[::-1],
                    y=np.concatenate([upper, lower[::-1]]).tolist(),
                    fill='toself',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level*100:.0f}% Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip',
                )
            )

        # Update layout
        fig.update_layout(
            title=self.title,
            xaxis_title="Timestamp",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            width=self.width,
            height=self.height // 2,  # Half height for single plot
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        return fig

    def create_error_analysis(self) -> Optional[go.Figure]:
        """
        Create error analysis visualization.

        Generates plots showing:
        - Error distribution histogram
        - Residual plot over time
        - Q-Q plot for normality assessment

        Returns:
            Optional[go.Figure]: Plotly figure or None if no actuals available

        Notes:
            - Only available when actual values exist
            - Helps diagnose forecast bias and patterns
        """
        if not self.forecast_data.can_calculate_metrics:
            return None

        actual = self.forecast_data.series.get_actual_values()
        predicted = self.forecast_data.series.get_predicted_values()

        if actual is None or len(actual) != len(predicted):
            return None

        # Calculate errors
        errors = predicted - actual
        timestamps = [
            point.timestamp for point in self.forecast_data.series.points
            if point.actual is not None
        ]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Error Distribution', 'Residuals Over Time'),
        )

        # Error histogram
        fig.add_trace(
            go.Histogram(
                x=errors,
                name='Errors',
                nbinsx=30,
                marker_color='lightblue',
                showlegend=False,
            ),
            row=1, col=1
        )

        # Residual plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=errors,
                mode='markers',
                name='Residuals',
                marker=dict(color='red', size=6),
                showlegend=False,
            ),
            row=1, col=2
        )

        # Add zero line to residual plot
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

        # Update layout
        fig.update_xaxes(title_text="Error", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Timestamp", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)

        fig.update_layout(
            title_text="Error Analysis",
            template='plotly_white',
            width=self.width,
            height=self.height // 2,
        )

        return fig

    def create_metrics_table(self) -> Optional[go.Figure]:
        """
        Create a formatted table displaying accuracy metrics.

        Returns:
            Optional[go.Figure]: Plotly table figure or None if no metrics

        Notes:
            - Displays all calculated accuracy metrics
            - Formatted for easy reading
        """
        if self.forecast_data.metrics is None:
            return None

        metrics_dict = self.forecast_data.metrics.to_summary_dict()

        # Create table data
        metrics_names = list(metrics_dict.keys())
        metrics_values = [
            str(v) if v is not None else "N/A"
            for v in metrics_dict.values()
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=['Metric', 'Value'],
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(size=14, color='black'),
                    ),
                    cells=dict(
                        values=[metrics_names, metrics_values],
                        fill_color='lavender',
                        align='left',
                        font=dict(size=12),
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Accuracy Metrics",
            width=self.width // 2,
            height=400,
        )

        return fig

    def create_dashboard(
        self,
        show_confidence: bool = True,
        show_actuals: bool = True,
        show_error_analysis: bool = True,
        confidence_level: float = 0.95,
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with all visualizations.

        Generates a multi-panel dashboard including:
        - Forecast visualization with confidence intervals
        - Error analysis (if actuals available)
        - Accuracy metrics summary

        Args:
            show_confidence: Whether to show confidence intervals
            show_actuals: Whether to show actual values
            show_error_analysis: Whether to include error analysis panel
            confidence_level: Confidence level for intervals

        Returns:
            go.Figure: Complete dashboard figure

        Examples:
            >>> # Create comprehensive dashboard
            >>> dashboard = ForecastDashboard(forecast_data)
            >>> fig = dashboard.create_dashboard(
            ...     show_confidence=True,
            ...     show_error_analysis=True,
            ...     confidence_level=0.95
            ... )
            >>> # Display in Jupyter
            >>> # fig.show()
            >>> # Or save to HTML
            >>> # fig.write_html("forecast_dashboard.html")

        Notes:
            - Automatically adapts based on data availability
            - Returns interactive Plotly figure
            - Can be exported to various formats
        """
        has_actuals = self.forecast_data.can_calculate_metrics

        if has_actuals and show_error_analysis:
            # Create dashboard with error analysis
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.6, 0.4],
                subplot_titles=(
                    'Forecast Time Series',
                    'Error Analysis'
                ),
                vertical_spacing=0.12,
            )

            # Create individual figures
            forecast_fig = self.forecast_visualization(
                show_confidence=show_confidence,
                show_actuals=show_actuals,
                confidence_level=confidence_level,
            )

            error_fig = self.create_error_analysis()

            # Add forecast traces
            for trace in forecast_fig.data:
                fig.add_trace(trace, row=1, col=1)

            # Add error traces if available
            if error_fig is not None:
                actual = self.forecast_data.series.get_actual_values()
                predicted = self.forecast_data.series.get_predicted_values()
                if actual is not None:
                    errors = predicted[:len(actual)] - actual
                    timestamps = [
                        point.timestamp for point in self.forecast_data.series.points
                        if point.actual is not None
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=errors,
                            mode='markers',
                            name='Residuals',
                            marker=dict(color='red', size=6),
                            showlegend=False,
                        ),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

            # Update axes
            fig.update_xaxes(title_text="Timestamp", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_xaxes(title_text="Timestamp", row=2, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)

        else:
            # Simple dashboard with just forecast
            fig = self.forecast_visualization(
                show_confidence=show_confidence,
                show_actuals=show_actuals,
                confidence_level=confidence_level,
            )

        # Update overall layout
        fig.update_layout(
            title_text=self.title,
            template='plotly_white',
            width=self.width,
            height=self.height,
            showlegend=True,
        )

        # Add metrics annotation if available
        if self.forecast_data.metrics is not None:
            metrics_text = (
                f"<b>Accuracy Metrics</b><br>"
                f"MAE: {self.forecast_data.metrics.mae:.4f}<br>"
                f"RMSE: {self.forecast_data.metrics.rmse:.4f}<br>"
                f"Bias: {self.forecast_data.metrics.bias:.4f}<br>"
                f"Samples: {self.forecast_data.metrics.n_samples}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                xanchor='right',
                yanchor='top',
                text=metrics_text,
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                borderpad=10,
                bgcolor="white",
                opacity=0.9,
            )

        return fig

    def export_html(
        self,
        filename: str,
        show_confidence: bool = True,
        show_actuals: bool = True,
        show_error_analysis: bool = True,
    ) -> None:
        """
        Export dashboard to HTML file.

        Args:
            filename: Output HTML filename
            show_confidence: Whether to show confidence intervals
            show_actuals: Whether to show actual values
            show_error_analysis: Whether to include error analysis

        Examples:
            >>> dashboard = ForecastDashboard(forecast_data)
            >>> dashboard.export_html("forecast_dashboard.html")
        """
        fig = self.create_dashboard(
            show_confidence=show_confidence,
            show_actuals=show_actuals,
            show_error_analysis=show_error_analysis,
        )
        fig.write_html(filename)
