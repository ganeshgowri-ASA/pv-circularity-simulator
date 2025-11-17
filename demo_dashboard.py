"""
Demo Dashboard Application for PV Circularity Simulator.

This demo showcases all dashboard components with realistic PV system data.
Run with: streamlit run demo_dashboard.py
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pv_simulator.components import DashboardComponents
from pv_simulator.models import (
    MetricCard,
    KPI,
    ProgressMetric,
    Notification,
    TrendDirection,
    NotificationLevel,
    MetricStatus,
)


def create_sample_metrics():
    """Create sample metric cards for demo."""
    return [
        MetricCard(
            title="Total Energy Output",
            value=15847.5,
            unit="kWh",
            description="Total energy produced this month",
            trend_direction=TrendDirection.UP,
            trend_value=8.3,
            comparison_label="vs. last month",
            icon="⚡",
            color="blue",
            status=MetricStatus.EXCELLENT,
        ),
        MetricCard(
            title="System Efficiency",
            value=89.2,
            unit="%",
            description="Overall system performance efficiency",
            trend_direction=TrendDirection.UP,
            trend_value=2.1,
            icon="📊",
            color="green",
            status=MetricStatus.GOOD,
        ),
        MetricCard(
            title="CO2 Emissions Avoided",
            value=8456.3,
            unit="kg",
            description="Carbon emissions prevented this year",
            trend_direction=TrendDirection.UP,
            trend_value=12.5,
            icon="🌱",
            color="green",
            status=MetricStatus.EXCELLENT,
        ),
        MetricCard(
            title="Material Recovery Rate",
            value=73.8,
            unit="%",
            description="Percentage of materials recovered for recycling",
            trend_direction=TrendDirection.UP,
            trend_value=5.2,
            icon="♻️",
            color="purple",
            status=MetricStatus.GOOD,
        ),
        MetricCard(
            title="System Uptime",
            value=99.7,
            unit="%",
            description="Operational uptime this month",
            trend_direction=TrendDirection.FLAT,
            trend_value=0.1,
            icon="⏱️",
            color="blue",
            status=MetricStatus.EXCELLENT,
        ),
        MetricCard(
            title="Cost Savings",
            value=3245.67,
            unit="$",
            description="Total cost savings from solar energy",
            trend_direction=TrendDirection.UP,
            trend_value=15.8,
            icon="💰",
            color="green",
            status=MetricStatus.EXCELLENT,
        ),
    ]


def create_sample_kpis():
    """Create sample KPIs for demo."""
    return [
        KPI(
            name="Annual Energy Production",
            current_value=187500,
            target_value=200000,
            unit="kWh",
            description="Total energy production target for the year",
            threshold_excellent=200000,
            threshold_good=180000,
            threshold_fair=160000,
            historical_values=[165000, 170000, 175000, 180000, 185000, 187500],
            category="performance",
            is_higher_better=True,
        ),
        KPI(
            name="Module Degradation Rate",
            current_value=0.45,
            target_value=0.50,
            unit="%/year",
            description="Annual performance degradation rate",
            threshold_excellent=0.40,
            threshold_good=0.50,
            threshold_fair=0.60,
            historical_values=[0.52, 0.50, 0.48, 0.47, 0.46, 0.45],
            category="performance",
            is_higher_better=False,
        ),
        KPI(
            name="Circular Economy Score",
            current_value=78.5,
            target_value=85.0,
            unit="points",
            description="Overall circularity performance score",
            threshold_excellent=85.0,
            threshold_good=75.0,
            threshold_fair=65.0,
            historical_values=[70, 72, 74, 75, 77, 78.5],
            category="circularity",
            is_higher_better=True,
        ),
        KPI(
            name="Recycled Material Usage",
            current_value=42.3,
            target_value=50.0,
            unit="%",
            description="Percentage of recycled materials in new modules",
            threshold_excellent=50.0,
            threshold_good=40.0,
            threshold_fair=30.0,
            historical_values=[35, 37, 38, 40, 41, 42.3],
            category="circularity",
            is_higher_better=True,
        ),
        KPI(
            name="System Availability",
            current_value=99.2,
            target_value=99.5,
            unit="%",
            description="System operational availability",
            threshold_excellent=99.5,
            threshold_good=99.0,
            threshold_fair=98.0,
            historical_values=[98.8, 99.0, 99.1, 99.0, 99.1, 99.2],
            category="reliability",
            is_higher_better=True,
        ),
        KPI(
            name="Maintenance Response Time",
            current_value=4.2,
            target_value=4.0,
            unit="hours",
            description="Average time to respond to maintenance issues",
            threshold_excellent=3.0,
            threshold_good=4.0,
            threshold_fair=6.0,
            historical_values=[5.5, 5.0, 4.8, 4.5, 4.3, 4.2],
            category="reliability",
            is_higher_better=False,
        ),
    ]


def create_sample_progress():
    """Create sample progress metrics for demo."""
    return [
        ProgressMetric(
            name="2025 Carbon Neutrality Goal",
            current_value=68.5,
            target_value=100.0,
            unit="%",
            description="Progress towards achieving carbon neutral operations",
            milestones=[25, 50, 75, 100],
            color="green",
            show_percentage=True,
            stages={
                "Initial Phase": 25,
                "Development": 50,
                "Implementation": 75,
                "Achievement": 100,
            },
            completion_date=datetime.now() + timedelta(days=180),
        ),
        ProgressMetric(
            name="Module Recycling Initiative",
            current_value=1250,
            target_value=2000,
            unit="modules",
            description="Number of end-of-life modules processed for recycling",
            milestones=[500, 1000, 1500, 2000],
            start_value=0,
            color="purple",
            show_percentage=False,
            completion_date=datetime.now() + timedelta(days=90),
        ),
        ProgressMetric(
            name="Energy Storage Capacity",
            current_value=450,
            target_value=600,
            unit="kWh",
            description="Battery storage capacity installation progress",
            milestones=[150, 300, 450, 600],
            start_value=0,
            color="blue",
            show_percentage=False,
            stages={
                "Phase 1": 150,
                "Phase 2": 300,
                "Phase 3": 450,
                "Phase 4": 600,
            },
            completion_date=datetime.now() + timedelta(days=120),
        ),
        ProgressMetric(
            name="Digital Twin Implementation",
            current_value=82.0,
            target_value=100.0,
            unit="%",
            description="Digital twin system deployment progress",
            milestones=[25, 50, 75, 100],
            color="cyan",
            show_percentage=True,
            stages={
                "Planning": 25,
                "Development": 50,
                "Testing": 75,
                "Deployment": 100,
            },
            completion_date=datetime.now() + timedelta(days=45),
        ),
    ]


def create_sample_notifications():
    """Create sample notifications for demo."""
    return [
        Notification(
            title="Critical: Inverter Malfunction",
            message="Inverter #3 in Zone B is showing error codes. Immediate attention required.",
            level=NotificationLevel.CRITICAL,
            timestamp=datetime.now() - timedelta(minutes=5),
            category="system",
            priority=10,
            action_label="View Details",
            action_url="/inverters/3",
            source="Monitoring System",
        ),
        Notification(
            title="Performance Alert",
            message="Panel array #12 efficiency dropped to 82%. May require cleaning or inspection.",
            level=NotificationLevel.WARNING,
            timestamp=datetime.now() - timedelta(minutes=25),
            category="performance",
            priority=7,
            action_label="Schedule Maintenance",
            action_url="/maintenance/schedule",
            source="Performance Monitor",
        ),
        Notification(
            title="Monthly Target Achieved",
            message="Congratulations! Monthly energy production target of 15,000 kWh has been reached.",
            level=NotificationLevel.SUCCESS,
            timestamp=datetime.now() - timedelta(hours=2),
            category="performance",
            priority=6,
            source="Analytics Engine",
        ),
        Notification(
            title="Maintenance Scheduled",
            message="Routine maintenance for Zone A scheduled for tomorrow at 9:00 AM.",
            level=NotificationLevel.INFO,
            timestamp=datetime.now() - timedelta(hours=4),
            category="maintenance",
            priority=5,
            action_label="View Schedule",
            action_url="/maintenance/calendar",
            source="Maintenance System",
        ),
        Notification(
            title="Material Recovery Milestone",
            message="Successfully recovered 1,000 kg of silicon from recycled modules this quarter.",
            level=NotificationLevel.SUCCESS,
            timestamp=datetime.now() - timedelta(hours=6),
            category="circularity",
            priority=6,
            source="Recycling Facility",
        ),
        Notification(
            title="Weather Impact Warning",
            message="Heavy cloud cover expected tomorrow. Energy production may be reduced by 30-40%.",
            level=NotificationLevel.WARNING,
            timestamp=datetime.now() - timedelta(hours=8),
            category="forecast",
            priority=5,
            source="Weather Service",
        ),
        Notification(
            title="System Update Available",
            message="New firmware version 2.4.1 is available for all inverters. Includes performance improvements.",
            level=NotificationLevel.INFO,
            timestamp=datetime.now() - timedelta(days=1),
            category="system",
            priority=4,
            action_label="Update Now",
            action_url="/system/updates",
            source="Update Manager",
        ),
        Notification(
            title="Battery Health Check",
            message="Battery bank #2 showing optimal health. No action required.",
            level=NotificationLevel.SUCCESS,
            timestamp=datetime.now() - timedelta(days=1, hours=3),
            category="storage",
            priority=3,
            source="Battery Monitor",
        ),
    ]


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="PV Circularity Simulator Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.title("⚡ PV Circularity")
        st.markdown("### Simulator Dashboard")
        st.divider()

        st.markdown("#### Component Demos")
        selected_demo = st.radio(
            "Select Component",
            [
                "🏠 All Components",
                "📊 Metric Cards",
                "🎯 KPI Displays",
                "📈 Progress Trackers",
                "🔔 Notification Widgets",
            ],
        )

        st.divider()
        st.markdown("#### Settings")
        theme_color = st.selectbox(
            "Theme",
            ["Blue", "Green", "Purple", "Custom"],
            index=0,
        )

        st.divider()
        st.caption("PV Circularity Simulator v0.1.0")
        st.caption("Demo Dashboard Application")

    # Main content
    st.title("⚡ PV Circularity Simulator")
    st.markdown("### Advanced Dashboard Components Demo")

    # Initialize dashboard
    dashboard = DashboardComponents()

    # Demo selection
    if selected_demo == "🏠 All Components" or selected_demo == "📊 Metric Cards":
        st.header("📊 Metric Cards")
        st.markdown(
            "Metric cards provide a clean, visual way to display key metrics with trend information."
        )

        metrics = create_sample_metrics()
        dashboard.metric_cards(
            metrics=metrics,
            columns=3,
            show_trend=True,
            show_icon=True,
            card_style="default",
        )

        if selected_demo != "🏠 All Components":
            with st.expander("📝 View Code Example"):
                st.code(
                    """
from pv_simulator.components import DashboardComponents
from pv_simulator.models import MetricCard, TrendDirection

metrics = [
    MetricCard(
        title="Total Energy Output",
        value=15847.5,
        unit="kWh",
        trend_direction=TrendDirection.UP,
        trend_value=8.3,
        icon="⚡",
    )
]

dashboard = DashboardComponents()
dashboard.metric_cards(metrics, columns=3)
                    """,
                    language="python",
                )

    if selected_demo == "🏠 All Components":
        st.divider()

    if selected_demo == "🏠 All Components" or selected_demo == "🎯 KPI Displays":
        st.header("🎯 Key Performance Indicators")
        st.markdown(
            "KPI displays show critical performance metrics with target comparisons and trends."
        )

        kpis = create_sample_kpis()
        dashboard.kpi_displays(
            kpis=kpis,
            layout="grid",
            columns=2,
            show_sparklines=True,
            show_targets=True,
            group_by_category=True,
        )

        if selected_demo != "🏠 All Components":
            with st.expander("📝 View Code Example"):
                st.code(
                    """
from pv_simulator.components import DashboardComponents
from pv_simulator.models import KPI

kpis = [
    KPI(
        name="Annual Energy Production",
        current_value=187500,
        target_value=200000,
        unit="kWh",
        historical_values=[165000, 170000, 175000, 180000, 185000, 187500],
        category="performance",
    )
]

dashboard = DashboardComponents()
dashboard.kpi_displays(kpis, columns=2, show_sparklines=True)
                    """,
                    language="python",
                )

    if selected_demo == "🏠 All Components":
        st.divider()

    if selected_demo == "🏠 All Components" or selected_demo == "📈 Progress Trackers":
        st.header("📈 Progress Trackers")
        st.markdown(
            "Progress trackers visualize advancement towards specific targets and goals."
        )

        progress = create_sample_progress()
        dashboard.progress_trackers(
            progress_metrics=progress,
            layout="vertical",
            show_milestones=True,
            show_remaining=True,
            show_eta=True,
        )

        if selected_demo != "🏠 All Components":
            with st.expander("📝 View Code Example"):
                st.code(
                    """
from pv_simulator.components import DashboardComponents
from pv_simulator.models import ProgressMetric

progress = [
    ProgressMetric(
        name="2025 Carbon Neutrality Goal",
        current_value=68.5,
        target_value=100.0,
        unit="%",
        milestones=[25, 50, 75, 100],
    )
]

dashboard = DashboardComponents()
dashboard.progress_trackers(progress, show_milestones=True)
                    """,
                    language="python",
                )

    if selected_demo == "🏠 All Components":
        st.divider()

    if selected_demo == "🏠 All Components" or selected_demo == "🔔 Notification Widgets":
        st.header("🔔 Notifications")
        st.markdown(
            "Notification widgets display alerts, messages, and status updates with different severity levels."
        )

        notifications = create_sample_notifications()
        active_notifications = dashboard.notification_widgets(
            notifications=notifications,
            max_display=8,
            show_timestamps=True,
            allow_dismiss=True,
            group_by_level=True,
            sort_by="priority",
        )

        st.info(f"📊 Active notifications: {len(active_notifications)}")

        if selected_demo != "🏠 All Components":
            with st.expander("📝 View Code Example"):
                st.code(
                    """
from pv_simulator.components import DashboardComponents
from pv_simulator.models import Notification, NotificationLevel

notifications = [
    Notification(
        title="Performance Alert",
        message="Panel efficiency dropped below threshold",
        level=NotificationLevel.WARNING,
        priority=7,
    )
]

dashboard = DashboardComponents()
active = dashboard.notification_widgets(notifications, group_by_level=True)
                    """,
                    language="python",
                )

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #6b7280; padding: 2rem;'>
            <p>PV Circularity Simulator - Advanced Dashboard Components</p>
            <p style='font-size: 0.875rem;'>Built with Streamlit • Production-Ready Components</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
