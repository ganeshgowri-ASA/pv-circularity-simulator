"""
Financial Visualization Module using Plotly.

Provides comprehensive charting capabilities for financial analysis including
cash flow waterfalls, LCOE breakdowns, sensitivity plots, and tornado diagrams.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..models.financial_models import CashFlowModel, CostStructure
from ..calculators.lcoe_calculator import LCOEResult, LCOECalculator
from ..calculators.sensitivity_analysis import (
    SensitivityResult,
    TornadoData,
    SensitivityAnalyzer,
)


class FinancialChartBuilder:
    """
    Builder class for creating financial visualizations using Plotly.

    Provides methods for various chart types commonly used in financial analysis.
    """

    def __init__(self, template: str = "plotly_white"):
        """
        Initialize chart builder.

        Args:
            template: Plotly template to use for styling
        """
        self.template = template
        self.color_scheme = {
            'revenue': '#2E7D32',  # Green
            'costs': '#C62828',    # Red
            'net_positive': '#1976D2',  # Blue
            'net_negative': '#D32F2F',  # Dark red
            'circularity': '#00897B',  # Teal
            'baseline': '#757575',  # Gray
        }

    def create_cash_flow_waterfall(
        self,
        cash_flow_model: CashFlowModel,
        title: str = "Project Cash Flow Analysis",
    ) -> go.Figure:
        """
        Create waterfall chart showing cash flows over project lifetime.

        Args:
            cash_flow_model: Cash flow model to visualize
            title: Chart title

        Returns:
            Plotly Figure object
        """
        df = cash_flow_model.generate_cash_flow_series()

        # Prepare data for waterfall
        years = df['year'].values
        net_cf = df['net_cash_flow'].values

        fig = go.Figure()

        # Add bars for each year
        colors = [self.color_scheme['net_positive'] if cf >= 0
                 else self.color_scheme['net_negative']
                 for cf in net_cf]

        fig.add_trace(go.Bar(
            x=years,
            y=net_cf,
            marker_color=colors,
            name='Net Cash Flow',
            text=[f"${cf:,.0f}" for cf in net_cf],
            textposition='outside',
            hovertemplate='Year %{x}<br>Net CF: $%{y:,.2f}<extra></extra>',
        ))

        # Add cumulative line
        fig.add_trace(go.Scatter(
            x=years,
            y=df['cumulative_cash_flow'].values,
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(color=self.color_scheme['baseline'], width=2),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate='Year %{x}<br>Cumulative: $%{y:,.2f}<extra></extra>',
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Annual Cash Flow ($)",
            yaxis2=dict(
                title="Cumulative Cash Flow ($)",
                overlaying='y',
                side='right',
            ),
            template=self.template,
            hovermode='x unified',
            showlegend=True,
            height=500,
        )

        return fig

    def create_cash_flow_breakdown(
        self,
        cash_flow_model: CashFlowModel,
        title: str = "Revenue vs Costs Over Time",
    ) -> go.Figure:
        """
        Create stacked area chart showing revenue and costs over time.

        Args:
            cash_flow_model: Cash flow model to visualize
            title: Chart title

        Returns:
            Plotly Figure object
        """
        df = cash_flow_model.generate_cash_flow_series()

        fig = go.Figure()

        # Revenue area
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['revenue'],
            fill='tonexty',
            name='Revenue',
            line=dict(color=self.color_scheme['revenue']),
            stackgroup='one',
            hovertemplate='Year %{x}<br>Revenue: $%{y:,.2f}<extra></extra>',
        ))

        # Costs area (negative values for stacking)
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=-df['costs'],
            fill='tonexty',
            name='Costs',
            line=dict(color=self.color_scheme['costs']),
            stackgroup='two',
            hovertemplate='Year %{x}<br>Costs: $%{y:,.2f}<extra></extra>',
        ))

        # Net cash flow line
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['net_cash_flow'],
            mode='lines',
            name='Net Cash Flow',
            line=dict(color=self.color_scheme['net_positive'], width=3),
            hovertemplate='Year %{x}<br>Net: $%{y:,.2f}<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Cash Flow ($)",
            template=self.template,
            hovermode='x unified',
            height=500,
        )

        return fig

    def create_lcoe_breakdown_pie(
        self,
        lcoe_result: LCOEResult,
        title: str = "LCOE Cost Breakdown",
    ) -> go.Figure:
        """
        Create pie chart showing LCOE cost breakdown.

        Args:
            lcoe_result: LCOE calculation result
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Filter out zero or negative values
        breakdown = {k: v for k, v in lcoe_result.cost_breakdown.items()
                    if v > 0}

        labels = list(breakdown.keys())
        values = list(breakdown.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='%{label}<br>$%{value:,.2f}<br>%{percent}<extra></extra>',
        )])

        fig.update_layout(
            title=title,
            template=self.template,
            height=500,
        )

        return fig

    def create_lcoe_breakdown_waterfall(
        self,
        lcoe_result: LCOEResult,
        title: str = "LCOE Cost Components",
    ) -> go.Figure:
        """
        Create waterfall chart for LCOE cost breakdown.

        Args:
            lcoe_result: LCOE calculation result
            title: Chart title

        Returns:
            Plotly Figure object
        """
        breakdown = lcoe_result.cost_breakdown

        # Separate positive and negative components
        labels = []
        values = []
        measures = []

        for label, value in breakdown.items():
            labels.append(label)
            values.append(value)
            if value < 0:
                measures.append('relative')
            else:
                measures.append('relative')

        # Add total
        labels.append('Total')
        values.append(sum(breakdown.values()))
        measures.append('total')

        fig = go.Figure(go.Waterfall(
            name="Cost Components",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.color_scheme['revenue']}},
            increasing={"marker": {"color": self.color_scheme['costs']}},
            totals={"marker": {"color": self.color_scheme['baseline']}},
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Cost Component",
            yaxis_title="Cost ($)",
            template=self.template,
            height=500,
        )

        return fig

    def create_sensitivity_chart(
        self,
        sensitivity_result: SensitivityResult,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create line chart for one-way sensitivity analysis.

        Args:
            sensitivity_result: Sensitivity analysis result
            title: Chart title (auto-generated if None)

        Returns:
            Plotly Figure object
        """
        if title is None:
            title = f"Sensitivity: {sensitivity_result.metric_name} vs {sensitivity_result.parameter_name}"

        fig = go.Figure()

        # Main sensitivity line
        fig.add_trace(go.Scatter(
            x=sensitivity_result.parameter_values,
            y=sensitivity_result.metric_values,
            mode='lines+markers',
            name=sensitivity_result.metric_name.upper(),
            line=dict(color=self.color_scheme['net_positive'], width=3),
            marker=dict(size=6),
            hovertemplate='%{x:.2f}<br>%{y:.4f}<extra></extra>',
        ))

        # Add base case marker
        fig.add_trace(go.Scatter(
            x=[sensitivity_result.base_parameter_value],
            y=[sensitivity_result.base_metric_value],
            mode='markers',
            name='Base Case',
            marker=dict(
                size=12,
                color=self.color_scheme['baseline'],
                symbol='star',
            ),
            hovertemplate='Base Case<br>%{x:.2f}<br>%{y:.4f}<extra></extra>',
        ))

        # Add reference line at base metric value
        fig.add_hline(
            y=sensitivity_result.base_metric_value,
            line_dash="dash",
            line_color=self.color_scheme['baseline'],
            opacity=0.5,
        )

        fig.update_layout(
            title=title,
            xaxis_title=sensitivity_result.parameter_name,
            yaxis_title=sensitivity_result.metric_name.upper(),
            template=self.template,
            hovermode='closest',
            height=500,
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=f'Elasticity: {sensitivity_result.elasticity:.2f}',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                )
            ]
        )

        return fig

    def create_tornado_diagram(
        self,
        tornado_data: TornadoData,
        title: str = "Tornado Diagram - Parameter Impact",
    ) -> go.Figure:
        """
        Create tornado diagram showing parameter impact on metric.

        Args:
            tornado_data: Tornado diagram data
            title: Chart title

        Returns:
            Plotly Figure object
        """
        df = tornado_data.to_dataframe()

        # Calculate deviations from base
        low_delta = df['low'] - tornado_data.base_value
        high_delta = df['high'] - tornado_data.base_value

        fig = go.Figure()

        # Low values (left side)
        fig.add_trace(go.Bar(
            name='Low Value',
            y=df['parameter'],
            x=low_delta,
            orientation='h',
            marker=dict(color=self.color_scheme['costs']),
            hovertemplate='%{y}<br>Low: %{x:+.4f}<extra></extra>',
        ))

        # High values (right side)
        fig.add_trace(go.Bar(
            name='High Value',
            y=df['parameter'],
            x=high_delta,
            orientation='h',
            marker=dict(color=self.color_scheme['revenue']),
            hovertemplate='%{y}<br>High: %{x:+.4f}<extra></extra>',
        ))

        # Add vertical line at base case
        fig.add_vline(x=0, line_color=self.color_scheme['baseline'], line_width=2)

        fig.update_layout(
            title=title,
            xaxis_title="Change in Metric from Base Case",
            yaxis_title="Parameter",
            barmode='overlay',
            template=self.template,
            height=max(400, len(df) * 40),
            showlegend=True,
        )

        return fig

    def create_2d_sensitivity_heatmap(
        self,
        df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create heatmap for two-way sensitivity analysis.

        Args:
            df: DataFrame with two-way sensitivity results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric name
            title: Chart title (auto-generated if None)

        Returns:
            Plotly Figure object
        """
        if title is None:
            title = f"2D Sensitivity: {metric} vs {param1} and {param2}"

        # Pivot data for heatmap
        pivot_df = df.pivot(index=param2, columns=param1, values=metric)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn_r',  # Red (high) to Green (low) for costs
            colorbar=dict(title=metric.upper()),
            hovertemplate=f'{param1}: %{{x}}<br>{param2}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title=param1,
            yaxis_title=param2,
            template=self.template,
            height=500,
        )

        return fig

    def create_monte_carlo_histogram(
        self,
        df: pd.DataFrame,
        metric: str,
        title: str = "Monte Carlo Simulation Results",
        n_bins: int = 50,
    ) -> go.Figure:
        """
        Create histogram of Monte Carlo simulation results.

        Args:
            df: DataFrame with simulation results
            metric: Metric column name
            title: Chart title
            n_bins: Number of histogram bins

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=df[metric],
            nbinsx=n_bins,
            name='Frequency',
            marker_color=self.color_scheme['net_positive'],
            opacity=0.7,
        ))

        # Add percentile lines
        percentiles = [10, 50, 90]
        colors = ['red', 'orange', 'red']

        for p, color in zip(percentiles, colors):
            value = df[metric].quantile(p / 100)
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"P{p}: {value:.4f}",
                annotation_position="top",
            )

        # Mean line
        mean_value = df[metric].mean()
        fig.add_vline(
            x=mean_value,
            line_dash="solid",
            line_color=self.color_scheme['baseline'],
            line_width=2,
            annotation_text=f"Mean: {mean_value:.4f}",
            annotation_position="top",
        )

        fig.update_layout(
            title=title,
            xaxis_title=metric.upper(),
            yaxis_title="Frequency",
            template=self.template,
            height=500,
            showlegend=False,
        )

        return fig

    def create_financial_metrics_dashboard(
        self,
        cash_flow_model: CashFlowModel,
        lcoe_result: LCOEResult,
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple financial metrics.

        Args:
            cash_flow_model: Cash flow model
            lcoe_result: LCOE calculation result

        Returns:
            Plotly Figure with subplots
        """
        # Calculate metrics
        npv = cash_flow_model.calculate_npv()
        irr = cash_flow_model.calculate_irr()
        payback = cash_flow_model.calculate_payback_period()
        roi = cash_flow_model.calculate_roi()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Cash Flow Over Time',
                'Key Financial Metrics',
                'LCOE Breakdown',
                'Cumulative Cash Flow'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'indicator'}],
                [{'type': 'pie'}, {'type': 'scatter'}]
            ]
        )

        # 1. Annual cash flow
        df = cash_flow_model.generate_cash_flow_series()
        colors = [self.color_scheme['net_positive'] if cf >= 0
                 else self.color_scheme['net_negative']
                 for cf in df['net_cash_flow']]

        fig.add_trace(
            go.Bar(
                x=df['year'],
                y=df['net_cash_flow'],
                marker_color=colors,
                name='Net Cash Flow',
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 2. Key metrics (using annotations since indicator doesn't work in subplots well)
        metrics_text = f"""
        NPV: ${npv:,.0f}<br>
        IRR: {irr*100:.2f}%<br>
        Payback: {payback:.1f} years<br>
        ROI: {roi:.1f}%<br>
        LCOE: ${lcoe_result.lcoe:.4f}/kWh
        """

        fig.add_annotation(
            text=metrics_text,
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
            row=1,
            col=2,
        )

        # 3. LCOE breakdown pie
        breakdown = {k: v for k, v in lcoe_result.cost_breakdown.items() if v > 0}
        fig.add_trace(
            go.Pie(
                labels=list(breakdown.keys()),
                values=list(breakdown.values()),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        # 4. Cumulative cash flow
        fig.add_trace(
            go.Scatter(
                x=df['year'],
                y=df['cumulative_cash_flow'],
                mode='lines+markers',
                line=dict(color=self.color_scheme['net_positive'], width=2),
                name='Cumulative CF',
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Financial Analysis Dashboard",
            template=self.template,
            height=800,
            showlegend=True,
        )

        return fig

    def create_circularity_impact_chart(
        self,
        lcoe_with_circularity: LCOEResult,
        lcoe_without_circularity: LCOEResult,
        title: str = "Circular Economy Impact on LCOE",
    ) -> go.Figure:
        """
        Create comparison chart showing circularity benefits.

        Args:
            lcoe_with_circularity: LCOE result with circular economy
            lcoe_without_circularity: LCOE result without circular economy
            title: Chart title

        Returns:
            Plotly Figure object
        """
        categories = ['LCOE', 'Total Lifetime Cost']
        with_circ = [
            lcoe_with_circularity.lcoe,
            lcoe_with_circularity.total_lifetime_cost
        ]
        without_circ = [
            lcoe_without_circularity.lcoe,
            lcoe_without_circularity.total_lifetime_cost
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Without Circularity',
            x=categories,
            y=without_circ,
            marker_color=self.color_scheme['costs'],
            text=[f"${v:.4f}/kWh" if i == 0 else f"${v:,.0f}"
                 for i, v in enumerate(without_circ)],
            textposition='outside',
        ))

        fig.add_trace(go.Bar(
            name='With Circularity',
            x=categories,
            y=with_circ,
            marker_color=self.color_scheme['circularity'],
            text=[f"${v:.4f}/kWh" if i == 0 else f"${v:,.0f}"
                 for i, v in enumerate(with_circ)],
            textposition='outside',
        ))

        # Calculate savings
        lcoe_savings = lcoe_without_circularity.lcoe - lcoe_with_circularity.lcoe
        cost_savings = (lcoe_without_circularity.total_lifetime_cost -
                       lcoe_with_circularity.total_lifetime_cost)

        fig.update_layout(
            title=title,
            yaxis_title="Value",
            barmode='group',
            template=self.template,
            height=500,
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=f'LCOE Savings: ${lcoe_savings:.4f}/kWh ({lcoe_savings/lcoe_without_circularity.lcoe*100:.1f}%)<br>'
                         f'Total Cost Savings: ${cost_savings:,.0f}',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    align='left',
                )
            ]
        )

        return fig
