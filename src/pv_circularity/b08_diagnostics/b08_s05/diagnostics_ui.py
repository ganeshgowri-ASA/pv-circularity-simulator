"""
Diagnostics UI Dashboard for PV System Management (B08-S05).

This module provides comprehensive interactive dashboards using Streamlit
for defect visualization, analysis, and management.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image

from ...models import (
    Defect,
    DefectType,
    DefectSeverity,
    DiagnosticResult,
    FaultReport,
    WorkOrder,
    WorkOrderStatus,
)
from .defect_database import DefectDatabase


class DiagnosticsUI:
    """
    Interactive diagnostics dashboard using Streamlit.

    This class provides comprehensive visualization and analysis tools
    for PV system defect management and diagnostics.

    Attributes:
        defect_database: DefectDatabase instance for data access
        title: Dashboard title
        page_config: Streamlit page configuration
    """

    def __init__(
        self,
        defect_database: DefectDatabase,
        title: str = "PV Diagnostics & Maintenance Dashboard",
    ):
        """
        Initialize the DiagnosticsUI.

        Args:
            defect_database: DefectDatabase instance
            title: Dashboard title
        """
        self.defect_database = defect_database
        self.title = title
        self._configure_page()

    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.title,
            page_icon="🔆",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def defect_gallery(
        self,
        defects: Optional[List[Defect]] = None,
        columns: int = 3,
        show_details: bool = True,
    ) -> None:
        """
        Display an interactive gallery of defect images with details.

        Creates a grid-based image gallery showing defect visualizations
        with filtering, sorting, and detail views.

        Args:
            defects: List of defects to display (if None, loads from database)
            columns: Number of columns in the gallery grid
            show_details: Show detailed information for each defect

        Example:
            >>> ui = DiagnosticsUI(defect_database)
            >>> ui.defect_gallery(defects=recent_defects, columns=4)
        """
        st.header("🖼️ Defect Gallery")

        # Load defects if not provided
        if defects is None:
            defects = self.defect_database.query_defects()

        if not defects:
            st.info("No defects to display.")
            return

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            type_filter = st.multiselect(
                "Filter by Type",
                options=[t.value for t in DefectType],
                default=None,
            )

        with col2:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=[s.value for s in DefectSeverity],
                default=None,
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=["Date (Newest)", "Date (Oldest)", "Severity", "Confidence"],
            )

        # Apply filters
        filtered_defects = defects
        if type_filter:
            filtered_defects = [
                d for d in filtered_defects
                if d.type.value in type_filter
            ]
        if severity_filter:
            filtered_defects = [
                d for d in filtered_defects
                if d.severity.value in severity_filter
            ]

        # Apply sorting
        if sort_by == "Date (Newest)":
            filtered_defects = sorted(
                filtered_defects,
                key=lambda d: d.created_at,
                reverse=True
            )
        elif sort_by == "Date (Oldest)":
            filtered_defects = sorted(filtered_defects, key=lambda d: d.created_at)
        elif sort_by == "Severity":
            severity_order = {
                DefectSeverity.CRITICAL: 0,
                DefectSeverity.HIGH: 1,
                DefectSeverity.MEDIUM: 2,
                DefectSeverity.LOW: 3,
            }
            filtered_defects = sorted(
                filtered_defects,
                key=lambda d: severity_order.get(d.severity, 4)
            )
        elif sort_by == "Confidence":
            filtered_defects = sorted(
                filtered_defects,
                key=lambda d: d.confidence,
                reverse=True
            )

        st.write(f"**Showing {len(filtered_defects)} of {len(defects)} defects**")

        # Display gallery
        cols = st.columns(columns)
        for idx, defect in enumerate(filtered_defects):
            col = cols[idx % columns]

            with col:
                self._render_defect_card(defect, show_details)

    def severity_heatmaps(
        self,
        defects: Optional[List[Defect]] = None,
        site_id: Optional[str] = None,
    ) -> None:
        """
        Display interactive heatmaps of defect severity distribution.

        Creates various heatmaps showing spatial distribution, temporal
        patterns, and severity trends.

        Args:
            defects: List of defects to visualize
            site_id: Optional site filter

        Example:
            >>> ui = DiagnosticsUI(defect_database)
            >>> ui.severity_heatmaps(site_id="SITE-001")
        """
        st.header("🌡️ Severity Heatmaps")

        # Load defects if not provided
        if defects is None:
            defects = self.defect_database.query_defects()

        if not defects:
            st.info("No defects to display.")
            return

        # Tabs for different heatmap types
        tab1, tab2, tab3 = st.tabs([
            "Spatial Distribution",
            "Temporal Trends",
            "Type vs Severity"
        ])

        with tab1:
            self._render_spatial_heatmap(defects)

        with tab2:
            self._render_temporal_heatmap(defects)

        with tab3:
            self._render_type_severity_heatmap(defects)

    def repair_tracking(
        self,
        work_orders: List[WorkOrder],
        show_completed: bool = True,
    ) -> None:
        """
        Display interactive repair and maintenance tracking dashboard.

        Shows work order status, progress, timelines, and technician
        assignments with real-time updates.

        Args:
            work_orders: List of work orders to track
            show_completed: Include completed work orders

        Example:
            >>> ui = DiagnosticsUI(defect_database)
            >>> ui.repair_tracking(work_orders=active_work_orders)
        """
        st.header("🔧 Repair & Maintenance Tracking")

        if not work_orders:
            st.info("No work orders to display.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_orders = len(work_orders)
        active_orders = len([
            wo for wo in work_orders
            if wo.status in [WorkOrderStatus.ASSIGNED, WorkOrderStatus.IN_PROGRESS]
        ])
        completed_orders = len([
            wo for wo in work_orders
            if wo.status == WorkOrderStatus.COMPLETED
        ])
        pending_orders = len([
            wo for wo in work_orders
            if wo.status in [WorkOrderStatus.DRAFT, WorkOrderStatus.SCHEDULED]
        ])

        with col1:
            st.metric("Total Orders", total_orders)
        with col2:
            st.metric("Active", active_orders)
        with col3:
            st.metric("Completed", completed_orders)
        with col4:
            st.metric("Pending", pending_orders)

        # Status filter
        if not show_completed:
            work_orders = [
                wo for wo in work_orders
                if wo.status != WorkOrderStatus.COMPLETED
            ]

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Work Order List",
            "Timeline View",
            "Status Distribution"
        ])

        with tab1:
            self._render_work_order_table(work_orders)

        with tab2:
            self._render_work_order_timeline(work_orders)

        with tab3:
            self._render_status_distribution(work_orders)

    def cost_analysis_dashboard(
        self,
        fault_reports: Optional[List[FaultReport]] = None,
        work_orders: Optional[List[WorkOrder]] = None,
        time_period: Optional[int] = 90,
    ) -> None:
        """
        Display comprehensive cost analysis dashboard.

        Provides detailed cost analysis including estimates, actuals,
        trends, and budget tracking.

        Args:
            fault_reports: List of fault reports with cost estimates
            work_orders: List of work orders with actual costs
            time_period: Time period in days for analysis

        Example:
            >>> ui = DiagnosticsUI(defect_database)
            >>> ui.cost_analysis_dashboard(
            ...     fault_reports=reports,
            ...     work_orders=orders,
            ...     time_period=90
            ... )
        """
        st.header("💰 Cost Analysis Dashboard")

        # Summary metrics
        total_estimated = 0.0
        total_actual = 0.0

        if fault_reports:
            total_estimated = sum(r.estimated_total_cost for r in fault_reports)

        if work_orders:
            total_actual = sum(wo.actual_cost for wo in work_orders)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Estimated Cost", f"${total_estimated:,.2f}")
        with col2:
            st.metric("Actual Cost", f"${total_actual:,.2f}")
        with col3:
            variance = total_actual - total_estimated
            st.metric(
                "Variance",
                f"${abs(variance):,.2f}",
                delta=f"{variance:,.2f}",
                delta_color="inverse"
            )
        with col4:
            accuracy = (
                (1 - abs(variance) / total_estimated) * 100
                if total_estimated > 0 else 100
            )
            st.metric("Estimate Accuracy", f"{accuracy:.1f}%")

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Cost Breakdown",
            "Trends Over Time",
            "Cost by Category"
        ])

        with tab1:
            if work_orders:
                self._render_cost_breakdown(work_orders)
            else:
                st.info("No work order data available.")

        with tab2:
            if work_orders:
                self._render_cost_trends(work_orders, time_period)
            else:
                st.info("No work order data available.")

        with tab3:
            if work_orders:
                self._render_cost_by_category(work_orders)
            else:
                st.info("No work order data available.")

    def render_dashboard(
        self,
        defects: Optional[List[Defect]] = None,
        work_orders: Optional[List[WorkOrder]] = None,
        fault_reports: Optional[List[FaultReport]] = None,
    ) -> None:
        """
        Render the complete integrated dashboard.

        Displays all dashboard components in an integrated view.

        Args:
            defects: List of defects
            work_orders: List of work orders
            fault_reports: List of fault reports
        """
        st.title(self.title)

        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select View",
                options=[
                    "Overview",
                    "Defect Gallery",
                    "Severity Analysis",
                    "Repair Tracking",
                    "Cost Analysis",
                ],
            )

        # Load data if not provided
        if defects is None:
            defects = self.defect_database.query_defects(limit=100)

        # Render selected page
        if page == "Overview":
            self._render_overview(defects, work_orders, fault_reports)
        elif page == "Defect Gallery":
            self.defect_gallery(defects)
        elif page == "Severity Analysis":
            self.severity_heatmaps(defects)
        elif page == "Repair Tracking" and work_orders:
            self.repair_tracking(work_orders)
        elif page == "Cost Analysis":
            self.cost_analysis_dashboard(fault_reports, work_orders)

    def _render_defect_card(self, defect: Defect, show_details: bool) -> None:
        """Render a single defect card."""
        # Severity color coding
        severity_colors = {
            DefectSeverity.LOW: "🟢",
            DefectSeverity.MEDIUM: "🟡",
            DefectSeverity.HIGH: "🟠",
            DefectSeverity.CRITICAL: "🔴",
        }

        with st.container():
            st.markdown(f"**{severity_colors.get(defect.severity, '⚪')} {defect.type.value.upper()}**")

            # Display image if available
            if defect.image_path and Path(defect.image_path).exists():
                try:
                    image = Image.open(defect.image_path)
                    st.image(image, use_column_width=True)
                except Exception:
                    st.info("Image unavailable")
            else:
                st.info("No image available")

            if show_details:
                st.caption(f"**Panel:** {defect.panel_id}")
                st.caption(f"**Severity:** {defect.severity.value}")
                st.caption(f"**Confidence:** {defect.confidence:.2%}")
                st.caption(f"**Power Loss:** {defect.estimated_power_loss:.1f}%")
                st.caption(f"**Date:** {defect.created_at.strftime('%Y-%m-%d')}")

            st.markdown("---")

    def _render_spatial_heatmap(self, defects: List[Defect]) -> None:
        """Render spatial distribution heatmap."""
        st.subheader("Spatial Distribution of Defects")

        # Extract coordinates
        data = []
        for defect in defects:
            data.append({
                "x": defect.location.x,
                "y": defect.location.y,
                "severity": defect.severity.value,
                "type": defect.type.value,
            })

        if not data:
            st.info("No location data available.")
            return

        df = pd.DataFrame(data)

        # Create scatter plot with color-coded severity
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="severity",
            hover_data=["type"],
            title="Defect Locations by Severity",
            color_discrete_map={
                "low": "green",
                "medium": "yellow",
                "high": "orange",
                "critical": "red",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_temporal_heatmap(self, defects: List[Defect]) -> None:
        """Render temporal trends heatmap."""
        st.subheader("Temporal Trends")

        # Group by date and severity
        data = []
        for defect in defects:
            data.append({
                "date": defect.created_at.date(),
                "severity": defect.severity.value,
                "type": defect.type.value,
            })

        if not data:
            st.info("No temporal data available.")
            return

        df = pd.DataFrame(data)

        # Count by date and severity
        counts = df.groupby(["date", "severity"]).size().reset_index(name="count")

        fig = px.bar(
            counts,
            x="date",
            y="count",
            color="severity",
            title="Defect Count Over Time by Severity",
            color_discrete_map={
                "low": "green",
                "medium": "yellow",
                "high": "orange",
                "critical": "red",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_type_severity_heatmap(self, defects: List[Defect]) -> None:
        """Render type vs severity heatmap."""
        st.subheader("Defect Type vs Severity")

        # Create matrix
        data = []
        for defect in defects:
            data.append({
                "type": defect.type.value,
                "severity": defect.severity.value,
            })

        if not data:
            st.info("No data available.")
            return

        df = pd.DataFrame(data)
        matrix = df.groupby(["type", "severity"]).size().unstack(fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="Reds",
        ))

        fig.update_layout(
            title="Defect Count by Type and Severity",
            xaxis_title="Severity",
            yaxis_title="Defect Type",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_work_order_table(self, work_orders: List[WorkOrder]) -> None:
        """Render work order table."""
        data = []
        for wo in work_orders:
            data.append({
                "WO Number": wo.work_order_number,
                "Title": wo.title,
                "Status": wo.status.value,
                "Priority": wo.priority.value,
                "Assigned To": wo.assigned_technician_id or "Unassigned",
                "Scheduled": wo.scheduled_start.strftime("%Y-%m-%d") if wo.scheduled_start else "N/A",
                "Estimated Cost": f"${wo.estimated_cost:,.2f}",
                "Actual Cost": f"${wo.actual_cost:,.2f}",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    def _render_work_order_timeline(self, work_orders: List[WorkOrder]) -> None:
        """Render work order timeline."""
        st.subheader("Work Order Timeline")

        data = []
        for wo in work_orders:
            if wo.scheduled_start:
                data.append({
                    "WO": wo.work_order_number,
                    "Start": wo.scheduled_start,
                    "End": wo.scheduled_end or wo.scheduled_start + timedelta(hours=8),
                    "Status": wo.status.value,
                })

        if not data:
            st.info("No scheduled work orders.")
            return

        df = pd.DataFrame(data)

        fig = px.timeline(
            df,
            x_start="Start",
            x_end="End",
            y="WO",
            color="Status",
            title="Work Order Schedule",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_status_distribution(self, work_orders: List[WorkOrder]) -> None:
        """Render status distribution chart."""
        st.subheader("Status Distribution")

        status_counts = pd.Series([wo.status.value for wo in work_orders]).value_counts()

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Work Orders by Status",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_cost_breakdown(self, work_orders: List[WorkOrder]) -> None:
        """Render cost breakdown."""
        st.subheader("Cost Breakdown")

        data = []
        for wo in work_orders:
            if wo.actual_cost > 0:
                data.append({
                    "WO Number": wo.work_order_number,
                    "Type": wo.maintenance_type.value,
                    "Estimated": wo.estimated_cost,
                    "Actual": wo.actual_cost,
                    "Variance": wo.actual_cost - wo.estimated_cost,
                })

        if not data:
            st.info("No cost data available.")
            return

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    def _render_cost_trends(self, work_orders: List[WorkOrder], time_period: int) -> None:
        """Render cost trends over time."""
        st.subheader(f"Cost Trends (Last {time_period} Days)")

        cutoff_date = datetime.utcnow() - timedelta(days=time_period)

        data = []
        for wo in work_orders:
            if wo.actual_end and wo.actual_end > cutoff_date:
                data.append({
                    "date": wo.actual_end.date(),
                    "cost": wo.actual_cost,
                })

        if not data:
            st.info("No cost data in the specified period.")
            return

        df = pd.DataFrame(data)
        daily_costs = df.groupby("date")["cost"].sum().reset_index()

        fig = px.line(
            daily_costs,
            x="date",
            y="cost",
            title="Daily Maintenance Costs",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_cost_by_category(self, work_orders: List[WorkOrder]) -> None:
        """Render costs by maintenance category."""
        st.subheader("Costs by Maintenance Type")

        data = []
        for wo in work_orders:
            if wo.actual_cost > 0:
                data.append({
                    "type": wo.maintenance_type.value,
                    "cost": wo.actual_cost,
                })

        if not data:
            st.info("No cost data available.")
            return

        df = pd.DataFrame(data)
        category_costs = df.groupby("type")["cost"].sum().reset_index()

        fig = px.bar(
            category_costs,
            x="type",
            y="cost",
            title="Total Costs by Maintenance Type",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_overview(
        self,
        defects: List[Defect],
        work_orders: Optional[List[WorkOrder]],
        fault_reports: Optional[List[FaultReport]],
    ) -> None:
        """Render overview page."""
        st.header("📊 System Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Defects", len(defects))

        with col2:
            critical_count = sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL)
            st.metric("Critical Defects", critical_count)

        with col3:
            if work_orders:
                active_wo = len([
                    wo for wo in work_orders
                    if wo.status in [WorkOrderStatus.ASSIGNED, WorkOrderStatus.IN_PROGRESS]
                ])
                st.metric("Active Work Orders", active_wo)
            else:
                st.metric("Active Work Orders", "N/A")

        with col4:
            if fault_reports:
                total_cost = sum(r.estimated_total_cost for r in fault_reports)
                st.metric("Estimated Costs", f"${total_cost:,.2f}")
            else:
                st.metric("Estimated Costs", "N/A")

        # Quick charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Defects by Type")
            type_counts = pd.Series([d.type.value for d in defects]).value_counts()
            fig = px.bar(x=type_counts.index, y=type_counts.values)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Defects by Severity")
            severity_counts = pd.Series([d.severity.value for d in defects]).value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index)
            st.plotly_chart(fig, use_container_width=True)
