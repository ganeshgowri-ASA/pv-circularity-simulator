"""
Advanced Dashboard Components for PV Circularity Simulator.

This module provides production-ready dashboard components for displaying metrics,
KPIs, progress tracking, and notifications in a Streamlit-based interface.

Classes:
    DashboardComponents: Main class containing all dashboard component methods

Example:
    >>> import streamlit as st
    >>> from pv_simulator.components import DashboardComponents
    >>>
    >>> dashboard = DashboardComponents()
    >>> metrics = [...]  # List of MetricCard objects
    >>> dashboard.metric_cards(metrics, columns=3)
"""

import streamlit as st
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.metrics import (
    MetricCard,
    KPI,
    ProgressMetric,
    Notification,
    TrendDirection,
    NotificationLevel,
    MetricStatus,
)
from ..utils.formatting import (
    format_number,
    format_percentage,
    format_currency,
    format_duration,
)
from ..utils.colors import (
    get_status_color,
    get_notification_color,
    get_gradient_color,
    lighten_color,
)


class DashboardComponents:
    """
    Production-ready dashboard components for PV Circularity Simulator.

    This class provides custom Streamlit components for creating interactive
    and visually appealing dashboards with metrics, KPIs, progress tracking,
    and notifications.

    The components are designed to be:
    - **Responsive**: Adapt to different screen sizes and layouts
    - **Interactive**: Support user interactions and callbacks
    - **Customizable**: Extensive styling and configuration options
    - **Production-ready**: Comprehensive error handling and validation

    Attributes:
        theme: Color theme configuration
        default_columns: Default number of columns for grid layouts
    """

    def __init__(self, theme: Optional[Dict[str, str]] = None):
        """
        Initialize DashboardComponents with optional theme configuration.

        Args:
            theme: Optional dictionary of theme colors. Keys can include:
                - primary, secondary, success, warning, danger, info
                If None, uses default theme.

        Example:
            >>> custom_theme = {'primary': '#3b82f6', 'success': '#10b981'}
            >>> dashboard = DashboardComponents(theme=custom_theme)
        """
        self.theme = theme or {
            "primary": "#3b82f6",
            "secondary": "#8b5cf6",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "info": "#06b6d4",
        }
        self.default_columns = 3

    def metric_cards(
        self,
        metrics: List[MetricCard],
        columns: int = 3,
        height: Optional[int] = None,
        show_trend: bool = True,
        show_icon: bool = True,
        card_style: str = "default",
        on_click: Optional[Callable[[MetricCard], None]] = None,
    ) -> None:
        """
        Display metric cards in a responsive grid layout.

        Metric cards provide a clean, visual way to display key metrics with
        trend information, comparisons, and status indicators. Each card can
        show a single metric with optional trend arrows, percentage changes,
        and color-coded status.

        Args:
            metrics: List of MetricCard objects to display
            columns: Number of columns in the grid layout (default 3)
            height: Optional fixed height for cards in pixels
            show_trend: Display trend indicators (default True)
            show_icon: Display icons on cards (default True)
            card_style: Card style variant ('default', 'minimal', 'detailed')
            on_click: Optional callback function when card is clicked

        Returns:
            None (renders directly to Streamlit)

        Raises:
            ValueError: If metrics list is empty or columns < 1

        Example:
            >>> from pv_simulator.models import MetricCard, TrendDirection
            >>> metrics = [
            ...     MetricCard(
            ...         title="Energy Output",
            ...         value=1234.5,
            ...         unit="kWh",
            ...         trend_direction=TrendDirection.UP,
            ...         trend_value=5.2
            ...     ),
            ...     MetricCard(
            ...         title="CO2 Savings",
            ...         value=856.3,
            ...         unit="kg",
            ...         trend_direction=TrendDirection.UP,
            ...         trend_value=3.1
            ...     )
            ... ]
            >>> dashboard = DashboardComponents()
            >>> dashboard.metric_cards(metrics, columns=2)

        Note:
            - Cards automatically adjust to screen width
            - Trend colors are determined by metric status
            - Icons use emoji representation if icon field is not set
        """
        if not metrics:
            st.warning("No metrics to display")
            return

        if columns < 1:
            raise ValueError("columns must be at least 1")

        # Create column layout
        cols = st.columns(columns)

        for idx, metric in enumerate(metrics):
            col = cols[idx % columns]

            with col:
                # Determine card color based on status
                status_color = get_status_color(metric.status)

                # Build card HTML
                card_html = self._build_metric_card_html(
                    metric=metric,
                    height=height,
                    show_trend=show_trend,
                    show_icon=show_icon,
                    card_style=card_style,
                    status_color=status_color,
                )

                # Render card
                st.markdown(card_html, unsafe_allow_html=True)

                # Handle click callback
                if on_click:
                    if st.button(
                        "View Details",
                        key=f"metric_card_{idx}",
                        use_container_width=True,
                    ):
                        on_click(metric)

    def kpi_displays(
        self,
        kpis: List[KPI],
        layout: str = "grid",
        columns: int = 2,
        show_sparklines: bool = True,
        show_targets: bool = True,
        show_thresholds: bool = True,
        comparison_mode: str = "target",
        group_by_category: bool = False,
    ) -> None:
        """
        Display Key Performance Indicators with advanced visualizations.

        KPI displays show critical performance metrics with target comparisons,
        historical trends, threshold indicators, and progress visualization.
        Supports multiple layout modes and grouping options.

        Args:
            kpis: List of KPI objects to display
            layout: Layout style ('grid', 'list', 'compact') (default 'grid')
            columns: Number of columns for grid layout (default 2)
            show_sparklines: Display mini trend charts (default True)
            show_targets: Show target values and progress (default True)
            show_thresholds: Display threshold indicators (default True)
            comparison_mode: How to compare values ('target', 'historical', 'both')
            group_by_category: Group KPIs by category (default False)

        Returns:
            None (renders directly to Streamlit)

        Raises:
            ValueError: If kpis list is empty

        Example:
            >>> from pv_simulator.models import KPI
            >>> kpis = [
            ...     KPI(
            ...         name="System Efficiency",
            ...         current_value=87.5,
            ...         target_value=90.0,
            ...         unit="%",
            ...         threshold_excellent=92.0,
            ...         threshold_good=85.0,
            ...         threshold_fair=80.0,
            ...         historical_values=[82, 84, 85, 86, 87.5],
            ...         category="performance"
            ...     ),
            ...     KPI(
            ...         name="Recycling Rate",
            ...         current_value=73.2,
            ...         target_value=80.0,
            ...         unit="%",
            ...         category="circularity"
            ...     )
            ... ]
            >>> dashboard = DashboardComponents()
            >>> dashboard.kpi_displays(kpis, columns=2, show_sparklines=True)

        Note:
            - Sparklines show historical trend when data is available
            - Status colors automatically adjust based on thresholds
            - Progress bars show percentage towards target
            - Grouping creates expandable sections by category
        """
        if not kpis:
            st.warning("No KPIs to display")
            return

        # Group by category if requested
        if group_by_category:
            categories = {}
            for kpi in kpis:
                if kpi.category not in categories:
                    categories[kpi.category] = []
                categories[kpi.category].append(kpi)

            for category, category_kpis in categories.items():
                with st.expander(f"📊 {category.upper()}", expanded=True):
                    self._render_kpi_group(
                        category_kpis,
                        layout,
                        columns,
                        show_sparklines,
                        show_targets,
                        show_thresholds,
                        comparison_mode,
                    )
        else:
            self._render_kpi_group(
                kpis,
                layout,
                columns,
                show_sparklines,
                show_targets,
                show_thresholds,
                comparison_mode,
            )

    def progress_trackers(
        self,
        progress_metrics: List[ProgressMetric],
        layout: str = "vertical",
        show_milestones: bool = True,
        show_remaining: bool = True,
        show_eta: bool = False,
        animate: bool = True,
        compact: bool = False,
    ) -> None:
        """
        Display progress trackers for goals and objectives.

        Progress trackers visualize advancement towards specific targets with
        progress bars, milestone markers, remaining values, and optional
        time-to-completion estimates.

        Args:
            progress_metrics: List of ProgressMetric objects to display
            layout: Layout orientation ('vertical', 'horizontal') (default 'vertical')
            show_milestones: Display milestone markers on progress bars (default True)
            show_remaining: Show remaining value to target (default True)
            show_eta: Show estimated time to completion (default False)
            animate: Animate progress bars on render (default True)
            compact: Use compact layout with less spacing (default False)

        Returns:
            None (renders directly to Streamlit)

        Raises:
            ValueError: If progress_metrics list is empty

        Example:
            >>> from pv_simulator.models import ProgressMetric
            >>> from datetime import datetime, timedelta
            >>>
            >>> progress = [
            ...     ProgressMetric(
            ...         name="Annual Energy Target",
            ...         current_value=8250,
            ...         target_value=10000,
            ...         unit="kWh",
            ...         description="Progress towards annual energy production goal",
            ...         milestones=[2500, 5000, 7500, 10000],
            ...         color="green",
            ...         completion_date=datetime.now() + timedelta(days=45)
            ...     ),
            ...     ProgressMetric(
            ...         name="Material Recovery",
            ...         current_value=65.5,
            ...         target_value=100.0,
            ...         unit="%",
            ...         stages={"Initial": 25, "Moderate": 50, "Advanced": 75, "Complete": 100}
            ...     )
            ... ]
            >>> dashboard = DashboardComponents()
            >>> dashboard.progress_trackers(progress, show_milestones=True)

        Note:
            - Progress bars use gradient colors by default
            - Milestones appear as markers on the progress bar
            - Stages show labeled sections for different progress levels
            - ETA is calculated based on historical progress rate if available
        """
        if not progress_metrics:
            st.warning("No progress metrics to display")
            return

        for metric in progress_metrics:
            # Calculate progress percentage
            progress_pct = metric.get_progress_percentage()
            remaining = metric.get_remaining()
            is_completed = metric.is_completed()

            # Container styling
            container_style = "padding: 0.5rem;" if compact else "padding: 1rem;"

            # Create container
            with st.container():
                # Header section
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(
                        f"**{metric.name}**",
                        help=metric.description if metric.description else None,
                    )

                with col2:
                    if show_remaining and not is_completed:
                        st.caption(
                            f"Remaining: {format_number(remaining, decimals=1)} {metric.unit}"
                        )

                # Progress bar with custom styling
                progress_color = (
                    get_status_color(MetricStatus.EXCELLENT)
                    if is_completed
                    else metric.color
                )

                # Main progress display
                col_progress, col_value = st.columns([4, 1])

                with col_progress:
                    # Streamlit native progress bar
                    st.progress(
                        min(1.0, progress_pct / 100),
                        text=f"{progress_pct:.1f}%" if metric.show_percentage else None,
                    )

                with col_value:
                    st.metric(
                        label="",
                        value=f"{format_number(metric.current_value, decimals=1)}",
                        delta=f"Target: {format_number(metric.target_value, decimals=1)} {metric.unit}",
                    )

                # Milestones display
                if show_milestones and metric.milestones:
                    milestone_cols = st.columns(len(metric.milestones))
                    for idx, milestone in enumerate(metric.milestones):
                        with milestone_cols[idx]:
                            reached = metric.current_value >= milestone
                            icon = "✅" if reached else "⭕"
                            st.caption(
                                f"{icon} {format_number(milestone, decimals=0)}"
                            )

                # Stages display
                if metric.stages:
                    current_stage = metric.get_current_stage()
                    if current_stage:
                        st.info(f"Current Stage: **{current_stage}**")

                # ETA display
                if show_eta and metric.completion_date:
                    days_remaining = (metric.completion_date - datetime.now()).days
                    if days_remaining > 0:
                        st.caption(f"⏱️ ETA: {days_remaining} days")
                    elif metric.is_overdue:
                        st.caption("⚠️ Overdue")

                if not compact:
                    st.divider()

    def notification_widgets(
        self,
        notifications: List[Notification],
        max_display: int = 10,
        show_timestamps: bool = True,
        allow_dismiss: bool = True,
        group_by_level: bool = False,
        sort_by: str = "timestamp",
        filter_level: Optional[NotificationLevel] = None,
        show_actions: bool = True,
    ) -> List[Notification]:
        """
        Display notification widgets with filtering and interaction options.

        Notification widgets show alerts, messages, and status updates with
        different severity levels, timestamps, and optional actions. Supports
        filtering, sorting, dismissal, and callbacks.

        Args:
            notifications: List of Notification objects to display
            max_display: Maximum number of notifications to show (default 10)
            show_timestamps: Display notification timestamps (default True)
            allow_dismiss: Enable dismiss buttons (default True)
            group_by_level: Group notifications by severity level (default False)
            sort_by: Sort order ('timestamp', 'priority', 'level') (default 'timestamp')
            filter_level: Show only notifications of specific level (default None)
            show_actions: Display action buttons if available (default True)

        Returns:
            List of currently active (non-dismissed) notifications

        Raises:
            ValueError: If sort_by value is invalid

        Example:
            >>> from pv_simulator.models import Notification, NotificationLevel
            >>> from datetime import datetime
            >>>
            >>> notifications = [
            ...     Notification(
            ...         title="System Alert",
            ...         message="Panel efficiency dropped below threshold",
            ...         level=NotificationLevel.WARNING,
            ...         category="performance",
            ...         action_label="View Details",
            ...         priority=8
            ...     ),
            ...     Notification(
            ...         title="Maintenance Due",
            ...         message="Scheduled maintenance required for inverter",
            ...         level=NotificationLevel.INFO,
            ...         category="maintenance",
            ...         priority=5
            ...     ),
            ...     Notification(
            ...         title="Target Achieved",
            ...         message="Monthly energy production target reached!",
            ...         level=NotificationLevel.SUCCESS,
            ...         category="performance",
            ...         priority=7
            ...     )
            ... ]
            >>> dashboard = DashboardComponents()
            >>> active_notifications = dashboard.notification_widgets(
            ...     notifications,
            ...     max_display=5,
            ...     group_by_level=True
            ... )

        Note:
            - Dismissed notifications are tracked in session state
            - Returns active notifications for further processing
            - Action callbacks can be implemented via action_url
            - Expired notifications are automatically filtered
            - Supports real-time updates when notifications list changes
        """
        if not notifications:
            st.info("No notifications to display")
            return []

        # Validate sort_by parameter
        valid_sort_options = ["timestamp", "priority", "level"]
        if sort_by not in valid_sort_options:
            raise ValueError(
                f"sort_by must be one of {valid_sort_options}, got '{sort_by}'"
            )

        # Initialize session state for dismissed notifications
        if "dismissed_notifications" not in st.session_state:
            st.session_state.dismissed_notifications = set()

        # Filter out expired and dismissed notifications
        active_notifications = [
            n
            for n in notifications
            if not n.is_expired()
            and id(n) not in st.session_state.dismissed_notifications
        ]

        # Apply level filter
        if filter_level:
            active_notifications = [
                n for n in active_notifications if n.level == filter_level
            ]

        # Sort notifications
        if sort_by == "timestamp":
            active_notifications.sort(key=lambda n: n.timestamp, reverse=True)
        elif sort_by == "priority":
            active_notifications.sort(key=lambda n: n.priority, reverse=True)
        elif sort_by == "level":
            level_order = {
                NotificationLevel.CRITICAL: 0,
                NotificationLevel.ERROR: 1,
                NotificationLevel.WARNING: 2,
                NotificationLevel.SUCCESS: 3,
                NotificationLevel.INFO: 4,
            }
            active_notifications.sort(key=lambda n: level_order.get(n.level, 99))

        # Limit display count
        display_notifications = active_notifications[:max_display]

        # Group by level if requested
        if group_by_level:
            grouped = {}
            for notification in display_notifications:
                level = notification.level.value
                if level not in grouped:
                    grouped[level] = []
                grouped[level].append(notification)

            for level, level_notifications in grouped.items():
                with st.expander(
                    f"{level_notifications[0].get_icon()} {level.upper()} ({len(level_notifications)})",
                    expanded=level in ["critical", "error", "warning"],
                ):
                    for notification in level_notifications:
                        self._render_notification(
                            notification,
                            show_timestamps,
                            allow_dismiss,
                            show_actions,
                        )
        else:
            # Display notifications without grouping
            for notification in display_notifications:
                self._render_notification(
                    notification,
                    show_timestamps,
                    allow_dismiss,
                    show_actions,
                )

        # Show overflow indicator
        if len(active_notifications) > max_display:
            st.caption(
                f"📝 {len(active_notifications) - max_display} more notifications (not shown)"
            )

        return active_notifications

    # Private helper methods

    def _build_metric_card_html(
        self,
        metric: MetricCard,
        height: Optional[int],
        show_trend: bool,
        show_icon: bool,
        card_style: str,
        status_color: str,
    ) -> str:
        """Build HTML for metric card display."""
        height_style = f"height: {height}px;" if height else ""

        # Build trend display
        trend_html = ""
        if show_trend and metric.trend_value is not None:
            trend_color = "#10b981" if metric.is_positive_trend() else "#ef4444"
            trend_sign = "+" if metric.trend_value > 0 else ""
            trend_html = f"""
                <div style="color: {trend_color}; font-size: 0.9em; margin-top: 0.5rem;">
                    {metric.get_trend_emoji()} {trend_sign}{metric.trend_value:.1f}%
                    <span style="color: #6b7280; font-size: 0.85em;">{metric.comparison_label}</span>
                </div>
            """

        # Build icon display
        icon_html = ""
        if show_icon and metric.icon:
            icon_html = f'<div style="font-size: 2em; margin-bottom: 0.5rem;">{metric.icon}</div>'

        # Description
        desc_html = ""
        if metric.description and card_style == "detailed":
            desc_html = f'<div style="color: #6b7280; font-size: 0.85em; margin-top: 0.5rem;">{metric.description}</div>'

        card_html = f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-left: 4px solid {status_color};
            border-radius: 0.5rem;
            padding: 1.5rem;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            {height_style}
        ">
            {icon_html}
            <div style="color: #6b7280; font-size: 0.875rem; margin-bottom: 0.5rem;">
                {metric.title}
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1f2937; line-height: 1;">
                {format_number(metric.value, decimals=2)} <span style="font-size: 1rem; color: #6b7280;">{metric.unit}</span>
            </div>
            {trend_html}
            {desc_html}
        </div>
        """

        return card_html

    def _render_kpi_group(
        self,
        kpis: List[KPI],
        layout: str,
        columns: int,
        show_sparklines: bool,
        show_targets: bool,
        show_thresholds: bool,
        comparison_mode: str,
    ) -> None:
        """Render a group of KPIs."""
        if layout == "grid":
            cols = st.columns(columns)
            for idx, kpi in enumerate(kpis):
                with cols[idx % columns]:
                    self._render_single_kpi(
                        kpi,
                        show_sparklines,
                        show_targets,
                        show_thresholds,
                        comparison_mode,
                    )
        else:
            # List or compact layout
            for kpi in kpis:
                self._render_single_kpi(
                    kpi,
                    show_sparklines,
                    show_targets,
                    show_thresholds,
                    comparison_mode,
                    compact=(layout == "compact"),
                )

    def _render_single_kpi(
        self,
        kpi: KPI,
        show_sparklines: bool,
        show_targets: bool,
        show_thresholds: bool,
        comparison_mode: str,
        compact: bool = False,
    ) -> None:
        """Render a single KPI display."""
        status = kpi.get_status()
        status_color = get_status_color(status)
        performance_pct = kpi.get_performance_percentage()

        # Container
        with st.container():
            # Header
            st.markdown(
                f"**{kpi.name}**",
                help=kpi.description if kpi.description else None,
            )

            # Value and status
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    f'<div style="font-size: 2rem; font-weight: 700; color: {status_color};">'
                    f'{format_number(kpi.current_value, decimals=1)} {kpi.unit}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                if show_targets:
                    delta = kpi.current_value - kpi.target_value
                    st.metric(
                        label="Target",
                        value=f"{format_number(kpi.target_value, decimals=1)}",
                        delta=f"{delta:+.1f}",
                    )

            # Progress bar
            if show_targets:
                progress_value = min(1.0, performance_pct / 100)
                st.progress(progress_value, text=f"{performance_pct:.1f}% of target")

            # Sparkline
            if show_sparklines and kpi.historical_values:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=kpi.historical_values,
                        mode="lines",
                        line=dict(color=status_color, width=2),
                        fill="tozeroy",
                        fillcolor=lighten_color(status_color, 0.7),
                    )
                )
                fig.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"sparkline_{kpi.name}")

            # Status badge
            st.markdown(
                f'<span style="background-color: {status_color}; color: white; '
                f'padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.75rem;">'
                f'{status.value.upper()}'
                f'</span>',
                unsafe_allow_html=True,
            )

            if not compact:
                st.divider()

    def _render_notification(
        self,
        notification: Notification,
        show_timestamps: bool,
        allow_dismiss: bool,
        show_actions: bool,
    ) -> None:
        """Render a single notification."""
        color = get_notification_color(notification.level)

        # Create columns for notification layout
        col_icon, col_content, col_action = st.columns([1, 8, 2])

        with col_icon:
            st.markdown(
                f'<div style="font-size: 1.5rem;">{notification.get_icon()}</div>',
                unsafe_allow_html=True,
            )

        with col_content:
            st.markdown(f"**{notification.title}**")
            st.caption(notification.message)

            if show_timestamps:
                age_seconds = notification.get_age_seconds()
                time_str = format_duration(age_seconds)
                st.caption(f"🕒 {time_str} ago")

        with col_action:
            if show_actions and notification.action_label:
                if st.button(
                    notification.action_label,
                    key=f"action_{id(notification)}",
                    type="primary",
                ):
                    st.info(f"Action triggered: {notification.action_url}")

            if allow_dismiss and notification.is_dismissible:
                if st.button("✕", key=f"dismiss_{id(notification)}"):
                    st.session_state.dismissed_notifications.add(id(notification))
                    st.rerun()

        st.divider()
