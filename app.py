"""
Main Streamlit application entry point for PV Circularity Dashboard.

This is the main entry point for running the Circularity Assessment Dashboard.
Run with: streamlit run app.py

Usage:
    $ streamlit run app.py
    $ streamlit run app.py -- --sample-data
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st
from pv_circularity_simulator.dashboards.circularity_dashboard import CircularityDashboardUI
from pv_circularity_simulator.core.data_models import CircularityMetrics
from examples.sample_data_generator import generate_sample_circularity_data


def main():
    """Main application entry point."""

    # Configure Streamlit page
    st.set_page_config(
        page_title="PV Circularity Assessment Dashboard",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/4CAF50/FFFFFF?text=PV+Circularity", use_container_width=True)
        st.title("⚙️ Configuration")

        data_source = st.radio(
            "Data Source",
            options=["Sample Data", "Upload File", "Connect to Database"],
            help="Select how to load circularity assessment data"
        )

        st.markdown("---")

        # Additional filters and settings
        st.markdown("### 🔧 Settings")

        show_raw_data = st.checkbox("Show raw data tables", value=False)
        enable_caching = st.checkbox("Enable caching", value=True)

        st.markdown("---")

        st.markdown("### 📚 Resources")
        st.markdown("""
        - [Documentation](https://github.com)
        - [User Guide](https://github.com)
        - [API Reference](https://github.com)
        """)

        st.markdown("---")
        st.caption("PV Circularity Simulator v0.1.0")

    # Load data based on selection
    metrics = None

    if data_source == "Sample Data":
        with st.spinner("Loading sample circularity data..."):
            metrics = generate_sample_circularity_data()
        st.success("✅ Sample data loaded successfully!")

    elif data_source == "Upload File":
        st.info("📤 File upload functionality coming soon!")
        st.markdown("""
        **Expected JSON format:**
        ```json
        {
            "assessment_id": "ASSESS-001",
            "circularity_index": 75.5,
            "material_flows": [...],
            "reuse_metrics": {...},
            "repair_metrics": {...},
            "recycling_metrics": {...}
        }
        ```
        """)

    elif data_source == "Connect to Database":
        st.info("🔌 Database connection functionality coming soon!")

    # Initialize and render dashboard
    dashboard = CircularityDashboardUI(
        metrics=metrics,
        cache_enabled=enable_caching
    )

    # Render the dashboard
    dashboard.render()

    # Show raw data if requested
    if show_raw_data and metrics:
        with st.expander("🔍 Raw Data Inspector"):
            st.json({
                "assessment_id": metrics.assessment_id,
                "circularity_index": metrics.circularity_index,
                "timestamp": metrics.timestamp.isoformat(),
                "material_flows_count": len(metrics.material_flows) if metrics.material_flows else 0,
                "has_reuse_metrics": metrics.reuse_metrics is not None,
                "has_repair_metrics": metrics.repair_metrics is not None,
                "has_recycling_metrics": metrics.recycling_metrics is not None,
                "impact_scorecards_count": len(metrics.impact_scorecards) if metrics.impact_scorecards else 0,
                "policy_compliance_count": len(metrics.policy_compliance) if metrics.policy_compliance else 0
            })


if __name__ == "__main__":
    main()
