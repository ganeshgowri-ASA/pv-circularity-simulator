"""
Dashboard & Project Management UI Module
=========================================

This module provides a comprehensive dashboard interface for the PV Circularity Simulator,
including project metrics, module completion tracking, quick actions, and performance monitoring.

Author: PV Circularity Simulator Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import time
import os
from io import BytesIO
import json


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling to the dashboard for a professional appearance."""
    st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }

        /* Module status grid */
        .module-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .module-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid #28a745;
            transition: all 0.3s ease;
        }

        .module-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .module-item.pending {
            border-left-color: #ffc107;
        }

        /* Quick action buttons */
        .quick-action-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 6px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0.5rem;
        }

        .quick-action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }

        /* Activity log */
        .activity-log {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border-left: 3px solid #17a2b8;
        }

        .activity-time {
            color: #6c757d;
            font-size: 0.85rem;
        }

        /* Performance metrics */
        .perf-metric {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Section headers */
        .section-header {
            color: #2c3e50;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }

        /* Success/Warning badges */
        .badge-success {
            background: #28a745;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .badge-warning {
            background: #ffc107;
            color: #212529;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Progress bar customization */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_completion() -> Tuple[float, Dict[str, bool]]:
    """
    Calculate the overall project completion percentage based on module completion flags.

    Returns:
        Tuple[float, Dict[str, bool]]: Completion percentage (0-100) and dictionary of module statuses
    """
    try:
        # Define all 11 modules in the PV Circularity Simulator
        modules = {
            'cell_design': 'Cell Design & SCAPS Integration',
            'ctm_analysis': 'Cell-to-Module (CTM) Loss Analysis',
            'module_engineering': 'Module Engineering & Design',
            'reliability_testing': 'Reliability & Environmental Testing',
            'system_planning': 'System Planning & Configuration',
            'eya_forecasting': 'Energy Yield Assessment (EYA)',
            'performance_monitoring': 'Performance Monitoring & Analytics',
            'degradation_modeling': 'Degradation & Lifetime Modeling',
            'circularity_assessment': 'Circularity & 3R Assessment',
            'economic_analysis': 'Economic & LCOE Analysis',
            'reporting': 'Reporting & Documentation'
        }

        # Initialize session state for completion tracking if not exists
        if 'completion_flags' not in st.session_state:
            st.session_state.completion_flags = {key: False for key in modules.keys()}

        # Calculate completion percentage
        completed_count = sum(st.session_state.completion_flags.values())
        total_modules = len(modules)
        completion_percentage = (completed_count / total_modules) * 100 if total_modules > 0 else 0

        return completion_percentage, st.session_state.completion_flags

    except Exception as e:
        st.error(f"Error calculating completion: {str(e)}")
        return 0.0, {}


def run_full_simulation() -> bool:
    """
    Execute all simulation modules in sequence with progress tracking.

    Returns:
        bool: True if simulation completed successfully, False otherwise
    """
    try:
        st.markdown('<div class="section-header">🚀 Running Full Simulation</div>', unsafe_allow_html=True)

        # Define simulation steps
        simulation_steps = [
            ('cell_design', 'Cell Design & SCAPS Integration', 8),
            ('ctm_analysis', 'CTM Loss Analysis', 6),
            ('module_engineering', 'Module Engineering', 7),
            ('reliability_testing', 'Reliability Testing', 10),
            ('system_planning', 'System Planning', 5),
            ('eya_forecasting', 'Energy Yield Assessment', 12),
            ('performance_monitoring', 'Performance Monitoring', 8),
            ('degradation_modeling', 'Degradation Modeling', 9),
            ('circularity_assessment', 'Circularity Assessment', 7),
            ('economic_analysis', 'Economic Analysis', 6),
            ('reporting', 'Report Generation', 5)
        ]

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_steps = len(simulation_steps)

        # Execute each simulation step
        for idx, (module_key, module_name, duration) in enumerate(simulation_steps):
            status_text.info(f"⏳ Processing: {module_name}...")

            # Simulate processing time
            for i in range(duration):
                time.sleep(0.1)  # Simulate work
                progress = ((idx + (i / duration)) / total_steps)
                progress_bar.progress(progress)

            # Mark module as complete
            if 'completion_flags' in st.session_state:
                st.session_state.completion_flags[module_key] = True

            # Update progress
            progress_bar.progress((idx + 1) / total_steps)

            # Log activity
            log_activity(f"Completed: {module_name}")

        # Final status
        status_text.success("✅ Full simulation completed successfully!")
        progress_bar.progress(1.0)

        return True

    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        return False


def generate_comprehensive_report() -> Optional[BytesIO]:
    """
    Generate a comprehensive PDF report of the simulation results.

    Returns:
        Optional[BytesIO]: PDF file buffer if successful, None otherwise
    """
    try:
        st.markdown('<div class="section-header">📄 Generating Comprehensive Report</div>', unsafe_allow_html=True)

        with st.spinner('Creating PDF report...'):
            # Simulate report generation
            time.sleep(2)

            # Get project data
            completion_pct, modules_status = calculate_completion()
            project_name = st.session_state.get('project_name', 'Unnamed Project')
            system_capacity = st.session_state.get('system_capacity', 0)

            # Create report content (simplified - in production, use reportlab or similar)
            report_content = f"""
            PV CIRCULARITY SIMULATOR - COMPREHENSIVE REPORT
            ================================================

            Project: {project_name}
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            PROJECT SUMMARY
            ---------------
            System Capacity: {system_capacity} kWp
            Overall Completion: {completion_pct:.1f}%

            MODULE STATUS
            -------------
            """

            for module_key, is_complete in modules_status.items():
                status = "✅ Complete" if is_complete else "⏳ Pending"
                report_content += f"\n{module_key.replace('_', ' ').title()}: {status}"

            # Create BytesIO buffer
            buffer = BytesIO()
            buffer.write(report_content.encode('utf-8'))
            buffer.seek(0)

            # Log activity
            log_activity("Generated comprehensive PDF report")

            st.success("✅ Report generated successfully!")
            return buffer

    except Exception as e:
        st.error(f"❌ Report generation failed: {str(e)}")
        return None


def export_all_data() -> Optional[BytesIO]:
    """
    Export all simulation data to Excel format.

    Returns:
        Optional[BytesIO]: Excel file buffer if successful, None otherwise
    """
    try:
        st.markdown('<div class="section-header">📊 Exporting All Data</div>', unsafe_allow_html=True)

        with st.spinner('Creating Excel export...'):
            # Simulate export process
            time.sleep(1.5)

            # Create sample data (in production, gather actual simulation data)
            completion_pct, modules_status = calculate_completion()

            # Create DataFrame
            export_data = {
                'Module': [],
                'Status': [],
                'Completion Date': []
            }

            for module_key, is_complete in modules_status.items():
                export_data['Module'].append(module_key.replace('_', ' ').title())
                export_data['Status'].append('Complete' if is_complete else 'Pending')
                export_data['Completion Date'].append(
                    datetime.now().strftime('%Y-%m-%d') if is_complete else 'N/A'
                )

            df = pd.DataFrame(export_data)

            # Create BytesIO buffer
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Module Status', index=False)

                # Add project summary sheet
                summary_data = {
                    'Metric': ['Project Name', 'System Capacity', 'Completion %', 'Last Updated'],
                    'Value': [
                        st.session_state.get('project_name', 'Unnamed Project'),
                        f"{st.session_state.get('system_capacity', 0)} kWp",
                        f"{completion_pct:.1f}%",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Project Summary', index=False)

            buffer.seek(0)

            # Log activity
            log_activity("Exported all data to Excel")

            st.success("✅ Data exported successfully!")
            return buffer

    except Exception as e:
        st.error(f"❌ Data export failed: {str(e)}")
        return None


def display_recent_activity(max_items: int = 5) -> None:
    """
    Display the most recent activity log entries.

    Args:
        max_items: Maximum number of activities to display (default: 5)
    """
    try:
        st.markdown('<div class="section-header">📋 Recent Activity</div>', unsafe_allow_html=True)

        # Initialize activity log in session state if not exists
        if 'activity_log' not in st.session_state:
            st.session_state.activity_log = []

        # Get recent activities
        recent_activities = st.session_state.activity_log[-max_items:][::-1]  # Last N, reversed

        if not recent_activities:
            st.info("No recent activities to display.")
            return

        # Display activities
        for activity in recent_activities:
            timestamp = activity.get('timestamp', '')
            message = activity.get('message', '')
            activity_type = activity.get('type', 'info')

            # Choose icon based on type
            icon = {
                'success': '✅',
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌'
            }.get(activity_type, 'ℹ️')

            st.markdown(f"""
                <div class="activity-log">
                    <div><strong>{icon} {message}</strong></div>
                    <div class="activity-time">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying activity log: {str(e)}")


def log_activity(message: str, activity_type: str = 'info') -> None:
    """
    Log an activity to the session state activity log.

    Args:
        message: Activity message to log
        activity_type: Type of activity ('info', 'success', 'warning', 'error')
    """
    try:
        if 'activity_log' not in st.session_state:
            st.session_state.activity_log = []

        activity = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': message,
            'type': activity_type
        }

        st.session_state.activity_log.append(activity)

        # Keep only last 100 activities to prevent memory issues
        if len(st.session_state.activity_log) > 100:
            st.session_state.activity_log = st.session_state.activity_log[-100:]

    except Exception as e:
        st.error(f"Error logging activity: {str(e)}")


def initialize_session_state() -> None:
    """Initialize session state variables with default values."""
    defaults = {
        'project_name': 'PV Circularity Project',
        'system_capacity': 1000.0,  # kWp
        'num_modules': 4500,
        'completion_flags': {
            'cell_design': False,
            'ctm_analysis': False,
            'module_engineering': False,
            'reliability_testing': False,
            'system_planning': False,
            'eya_forecasting': False,
            'performance_monitoring': False,
            'degradation_modeling': False,
            'circularity_assessment': False,
            'economic_analysis': False,
            'reporting': False
        },
        'activity_log': [],
        'annual_energy': 1500000,  # kWh/year
        'performance_ratio': 84.5,  # %
        'lcoe': 0.045,  # $/kWh
        'circularity_score': 72.3  # %
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# MAIN DASHBOARD FUNCTION
# ============================================================================

def render_dashboard() -> None:
    """
    Render the main dashboard interface with all components.

    This is the primary entry point for the dashboard module, displaying:
    - Project branding header
    - Key metrics overview
    - Module completion status grid
    - Quick action buttons
    - Recent activity log
    - Key performance metrics
    """
    try:
        # Apply custom CSS
        apply_custom_css()

        # Initialize session state
        initialize_session_state()

        # ====================================================================
        # MAIN HEADER
        # ====================================================================
        st.markdown("""
            <div class="main-header">
                <h1>🔆 PV Circularity Simulator</h1>
                <p>End-to-End Photovoltaic Lifecycle Simulation Platform</p>
            </div>
        """, unsafe_allow_html=True)

        # ====================================================================
        # 4-COLUMN METRICS
        # ====================================================================
        completion_pct, modules_status = calculate_completion()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📋 Project Name",
                value=st.session_state.get('project_name', 'N/A')
            )

        with col2:
            st.metric(
                label="⚡ System Capacity",
                value=f"{st.session_state.get('system_capacity', 0):.1f} kWp"
            )

        with col3:
            st.metric(
                label="🔧 Modules",
                value=f"{st.session_state.get('num_modules', 0):,}"
            )

        with col4:
            st.metric(
                label="✅ Completion",
                value=f"{completion_pct:.1f}%",
                delta=f"{int(sum(modules_status.values()))}/11 modules"
            )

        st.markdown("---")

        # ====================================================================
        # MODULE COMPLETION STATUS GRID
        # ====================================================================
        st.markdown('<div class="section-header">📊 Module Completion Status</div>', unsafe_allow_html=True)

        # Module definitions with descriptions
        module_definitions = [
            ('cell_design', '🔬 Cell Design & SCAPS', 'Solar cell design and SCAPS-1D integration'),
            ('ctm_analysis', '🔗 CTM Loss Analysis', 'Cell-to-Module loss characterization'),
            ('module_engineering', '⚙️ Module Engineering', 'PV module design and optimization'),
            ('reliability_testing', '🧪 Reliability Testing', 'Environmental and durability testing'),
            ('system_planning', '🏗️ System Planning', 'PV system configuration and sizing'),
            ('eya_forecasting', '☀️ Energy Yield (EYA)', 'Energy production forecasting'),
            ('performance_monitoring', '📈 Performance Monitoring', 'Real-time performance analytics'),
            ('degradation_modeling', '📉 Degradation Modeling', 'Long-term degradation analysis'),
            ('circularity_assessment', '♻️ Circularity (3R)', 'Reduce, Reuse, Recycle assessment'),
            ('economic_analysis', '💰 Economic Analysis', 'LCOE and financial modeling'),
            ('reporting', '📄 Reporting', 'Documentation and report generation')
        ]

        # Create 3 columns for grid layout
        grid_cols = st.columns(3)

        for idx, (module_key, module_name, module_desc) in enumerate(module_definitions):
            col_idx = idx % 3
            with grid_cols[col_idx]:
                is_complete = modules_status.get(module_key, False)
                status_icon = "✅" if is_complete else "⏳"
                status_class = "" if is_complete else "pending"

                st.markdown(f"""
                    <div class="module-item {status_class}">
                        <div><strong>{status_icon} {module_name}</strong></div>
                        <div style="font-size: 0.85rem; color: #6c757d; margin-top: 0.25rem;">
                            {module_desc}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ====================================================================
        # QUICK ACTIONS
        # ====================================================================
        st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("🚀 Run Full Simulation", use_container_width=True, type="primary"):
                run_full_simulation()
                st.rerun()

        with action_col2:
            if st.button("📄 Generate Report", use_container_width=True):
                report_buffer = generate_comprehensive_report()
                if report_buffer:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=report_buffer,
                        file_name=f"pv_circularity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        with action_col3:
            if st.button("📊 Export Data", use_container_width=True):
                excel_buffer = export_all_data()
                if excel_buffer:
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_buffer,
                        file_name=f"pv_circularity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        st.markdown("---")

        # ====================================================================
        # RECENT ACTIVITY LOG
        # ====================================================================
        display_recent_activity(max_items=5)

        st.markdown("---")

        # ====================================================================
        # KEY PERFORMANCE METRICS (if EYA complete)
        # ====================================================================
        if modules_status.get('eya_forecasting', False):
            st.markdown('<div class="section-header">📊 Key Performance Metrics</div>', unsafe_allow_html=True)

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                st.metric(
                    label="☀️ Annual Energy",
                    value=f"{st.session_state.get('annual_energy', 0):,.0f} kWh",
                    delta="Based on EYA forecast"
                )

            with kpi_col2:
                st.metric(
                    label="📈 Performance Ratio",
                    value=f"{st.session_state.get('performance_ratio', 0):.1f}%",
                    delta="Industry benchmark: 80-85%"
                )

            with kpi_col3:
                st.metric(
                    label="💰 LCOE",
                    value=f"${st.session_state.get('lcoe', 0):.3f}/kWh",
                    delta="Levelized Cost of Energy"
                )

            with kpi_col4:
                st.metric(
                    label="♻️ Circularity Score",
                    value=f"{st.session_state.get('circularity_score', 0):.1f}%",
                    delta="3R Assessment"
                )

            # Performance visualization
            st.markdown("### 📉 Performance Overview")

            # Create sample performance data
            performance_data = pd.DataFrame({
                'Month': pd.date_range(start='2024-01', periods=12, freq='M'),
                'Energy (MWh)': [
                    120, 125, 130, 135, 140, 138, 142, 140, 135, 130, 125, 120
                ],
                'PR (%)': [
                    84.2, 84.5, 84.8, 85.1, 84.9, 84.6, 84.8, 84.5, 84.3, 84.1, 84.0, 84.2
                ]
            })

            # Display as line chart
            st.line_chart(performance_data.set_index('Month')['Energy (MWh)'])

        else:
            st.info("ℹ️ Complete the Energy Yield Assessment (EYA) module to view Key Performance Metrics")

        # ====================================================================
        # FOOTER
        # ====================================================================
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #6c757d; padding: 1rem;">
                <p><strong>PV Circularity Simulator v1.0.0</strong></p>
                <p>© 2024 | End-to-End Photovoltaic Lifecycle Simulation Platform</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Dashboard Error: {str(e)}")
        st.exception(e)


# ============================================================================
# MODULE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="PV Circularity Simulator",
        page_icon="🔆",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Render dashboard
    render_dashboard()
