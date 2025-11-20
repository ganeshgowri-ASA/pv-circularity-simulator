# ⚡ PV Circularity Simulator

End-to-end PV lifecycle simulation platform with **Advanced Dashboard Components**

A comprehensive platform for photovoltaic system lifecycle management, from cell design to circular economy modeling. Features production-ready Streamlit dashboard components for real-time monitoring, KPI tracking, and performance visualization.

## 🌟 Features

### Advanced Dashboard Components

- **📊 Metric Cards**: Display key metrics with trend indicators and status colors
- **🎯 KPI Displays**: Track performance indicators with targets and sparklines
- **📈 Progress Trackers**: Visualize goal progression with milestones and stages
- **🔔 Notification Widgets**: Manage alerts and notifications with priority levels

### PV Lifecycle Coverage

- Cell design and module engineering
- System planning and performance monitoring
- Circularity analysis (Reduce, Reuse, Recycle)
- CTM loss analysis and SCAPS integration
- Reliability testing and energy forecasting
- Circular economy modeling

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Run the Demo

```bash
streamlit run demo_dashboard.py
```

This will launch an interactive dashboard showcasing all components with sample PV system data.

## 📖 Usage Examples

### Metric Cards

Display key metrics with trend information:

```python
from pv_simulator.components import DashboardComponents
from pv_simulator.models import MetricCard, TrendDirection

# Create metric cards
metrics = [
    MetricCard(
        title="Total Energy Output",
        value=15847.5,
        unit="kWh",
        description="Total energy produced this month",
        trend_direction=TrendDirection.UP,
        trend_value=8.3,
        icon="⚡",
        status=MetricStatus.EXCELLENT
    )
]

# Display in Streamlit
dashboard = DashboardComponents()
dashboard.metric_cards(metrics, columns=3, show_trend=True)
```

### KPI Displays

Track key performance indicators with targets:

```python
from pv_simulator.models import KPI

kpis = [
    KPI(
        name="System Efficiency",
        current_value=87.5,
        target_value=90.0,
        unit="%",
        threshold_excellent=92.0,
        threshold_good=85.0,
        historical_values=[82, 84, 85, 86, 87.5],
        category="performance"
    )
]

dashboard.kpi_displays(
    kpis,
    columns=2,
    show_sparklines=True,
    show_targets=True,
    group_by_category=True
)
```

### Progress Trackers

Monitor progress towards goals:

```python
from pv_simulator.models import ProgressMetric
from datetime import datetime, timedelta

progress = [
    ProgressMetric(
        name="Carbon Neutrality Goal",
        current_value=68.5,
        target_value=100.0,
        unit="%",
        milestones=[25, 50, 75, 100],
        completion_date=datetime.now() + timedelta(days=180)
    )
]

dashboard.progress_trackers(
    progress,
    show_milestones=True,
    show_remaining=True,
    show_eta=True
)
```

### Notification Widgets

Display system alerts and messages:

```python
from pv_simulator.models import Notification, NotificationLevel

notifications = [
    Notification(
        title="Performance Alert",
        message="Panel efficiency dropped below threshold",
        level=NotificationLevel.WARNING,
        category="performance",
        priority=7,
        action_label="View Details"
    )
]

active = dashboard.notification_widgets(
    notifications,
    max_display=10,
    group_by_level=True,
    allow_dismiss=True
)
```

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_simulator/
│       ├── __init__.py
│       ├── components/
│       │   ├── __init__.py
│       │   └── dashboard_components.py    # Main dashboard components
│       ├── models/
│       │   ├── __init__.py
│       │   └── metrics.py                 # Data models
│       └── utils/
│           ├── __init__.py
│           ├── formatting.py              # Formatting utilities
│           └── colors.py                  # Color utilities
├── demo_dashboard.py                       # Demo application
├── requirements.txt                        # Dependencies
├── setup.py                               # Package setup
└── README.md                              # This file
```

## 🎨 Component Features

### Metric Cards

- ✅ Responsive grid layout
- ✅ Trend indicators with arrows
- ✅ Status-based color coding
- ✅ Custom icons and styling
- ✅ Multiple card styles (default, minimal, detailed)
- ✅ Click callbacks for interactivity

### KPI Displays

- ✅ Target comparison and progress bars
- ✅ Historical trend sparklines
- ✅ Threshold indicators
- ✅ Category grouping
- ✅ Multiple layout modes (grid, list, compact)
- ✅ Performance status calculation

### Progress Trackers

- ✅ Visual progress bars with percentages
- ✅ Milestone markers
- ✅ Stage-based progression
- ✅ Remaining value calculation
- ✅ ETA estimation
- ✅ Completion status indicators

### Notification Widgets

- ✅ Severity level color coding
- ✅ Priority-based sorting
- ✅ Timestamp display
- ✅ Dismissible notifications
- ✅ Action buttons with callbacks
- ✅ Category filtering and grouping
- ✅ Automatic expiration handling

## 🎯 Data Models

All components use strongly-typed data models with comprehensive docstrings:

- **MetricCard**: Individual metric display with trends
- **KPI**: Key Performance Indicator with targets and thresholds
- **ProgressMetric**: Goal tracking with milestones
- **Notification**: Alerts and messages with priority levels

Enums for type safety:
- `TrendDirection`: UP, DOWN, FLAT
- `NotificationLevel`: INFO, SUCCESS, WARNING, ERROR, CRITICAL
- `MetricStatus`: EXCELLENT, GOOD, FAIR, POOR, CRITICAL

## 🛠️ Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black src/ demo_dashboard.py
flake8 src/
mypy src/
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

## 📊 Demo Dashboard

The included demo dashboard (`demo_dashboard.py`) showcases all components with realistic PV system data:

- 6 metric cards tracking energy, efficiency, and circularity
- 6 KPIs across performance, circularity, and reliability categories
- 4 progress trackers for various initiatives
- 8 sample notifications with different severity levels

Access different views through the sidebar:
- 🏠 All Components (comprehensive view)
- 📊 Metric Cards only
- 🎯 KPI Displays only
- 📈 Progress Trackers only
- 🔔 Notification Widgets only

## 🎓 API Documentation

### DashboardComponents Class

```python
class DashboardComponents:
    """Production-ready dashboard components for PV Circularity Simulator."""

    def __init__(self, theme: Optional[Dict[str, str]] = None):
        """Initialize with optional custom theme."""

    def metric_cards(
        self,
        metrics: List[MetricCard],
        columns: int = 3,
        height: Optional[int] = None,
        show_trend: bool = True,
        show_icon: bool = True,
        card_style: str = "default",
        on_click: Optional[Callable] = None
    ) -> None:
        """Display metric cards in responsive grid."""

    def kpi_displays(
        self,
        kpis: List[KPI],
        layout: str = "grid",
        columns: int = 2,
        show_sparklines: bool = True,
        show_targets: bool = True,
        show_thresholds: bool = True,
        comparison_mode: str = "target",
        group_by_category: bool = False
    ) -> None:
        """Display KPIs with advanced visualizations."""

    def progress_trackers(
        self,
        progress_metrics: List[ProgressMetric],
        layout: str = "vertical",
        show_milestones: bool = True,
        show_remaining: bool = True,
        show_eta: bool = False,
        animate: bool = True,
        compact: bool = False
    ) -> None:
        """Display progress trackers for goals."""

    def notification_widgets(
        self,
        notifications: List[Notification],
        max_display: int = 10,
        show_timestamps: bool = True,
        allow_dismiss: bool = True,
        group_by_level: bool = False,
        sort_by: str = "timestamp",
        filter_level: Optional[NotificationLevel] = None,
        show_actions: bool = True
    ) -> List[Notification]:
        """Display notification widgets with filtering."""
```

Full API documentation is available in the docstrings of each method.

## 🔧 Configuration

Streamlit configuration is in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f3f4f6"
textColor = "#1f2937"
```

Customize the theme by modifying these values or passing a custom theme to `DashboardComponents()`.

## 📝 Requirements

- Python 3.8+
- Streamlit 1.31.0+
- Plotly 5.18.0+
- Pandas 2.1.0+
- NumPy 1.24.0+

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web framework
- [Plotly](https://plotly.com/) - Visualization library
- Python dataclasses for robust data models

## 📬 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Status**: ✅ Production-Ready | **Version**: 0.1.0 | **Last Updated**: 2025-11-17
