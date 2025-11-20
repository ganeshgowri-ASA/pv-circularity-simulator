"""
Infrastructure & Deployment Management Module (Branch B14).

Features:
- Project lifecycle management
- Resource allocation and scheduling
- Equipment inventory tracking
- Installation progress monitoring
- Quality assurance checkpoints
- Safety compliance tracking
- Document management
- Stakeholder communication
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from enum import Enum

from utils.constants import MOUNTING_TYPES, INVERTER_TYPES
from utils.helpers import format_number


class ProjectPhase(Enum):
    """Project lifecycle phases."""
    PLANNING = "Planning"
    DESIGN = "Design"
    PROCUREMENT = "Procurement"
    CONSTRUCTION = "Construction"
    COMMISSIONING = "Commissioning"
    OPERATION = "Operation"
    MAINTENANCE = "Maintenance"


class TaskStatus(Enum):
    """Task completion status."""
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    DELAYED = "Delayed"
    BLOCKED = "Blocked"


class InfrastructureManager:
    """Infrastructure and deployment management system."""

    def __init__(self):
        """Initialize infrastructure manager."""
        self.mounting_types = MOUNTING_TYPES
        self.inverter_types = INVERTER_TYPES

    def create_project_schedule(
        self,
        system_capacity_kw: float,
        start_date: datetime,
        installation_rate_kw_per_day: float = 100
    ) -> pd.DataFrame:
        """
        Create detailed project schedule with milestones.

        Args:
            system_capacity_kw: System capacity (kW)
            start_date: Project start date
            installation_rate_kw_per_day: Installation rate (kW/day)

        Returns:
            DataFrame with project schedule
        """
        tasks = []

        # Planning phase (10% of total time)
        planning_days = max(14, int(system_capacity_kw / installation_rate_kw_per_day * 0.10))
        tasks.append({
            'phase': 'Planning',
            'task': 'Site Assessment & Survey',
            'duration_days': planning_days // 3,
            'predecessors': [],
            'resources': 'Survey Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Planning',
            'task': 'Permitting & Approvals',
            'duration_days': planning_days // 2,
            'predecessors': ['Site Assessment & Survey'],
            'resources': 'Legal Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Planning',
            'task': 'Financing Closure',
            'duration_days': planning_days // 4,
            'predecessors': ['Permitting & Approvals'],
            'resources': 'Finance Team',
            'status': 'Not Started'
        })

        # Design phase (15% of total time)
        design_days = max(21, int(system_capacity_kw / installation_rate_kw_per_day * 0.15))
        tasks.append({
            'phase': 'Design',
            'task': 'Electrical Design',
            'duration_days': design_days // 2,
            'predecessors': ['Site Assessment & Survey'],
            'resources': 'Engineering Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Design',
            'task': 'Structural Design',
            'duration_days': design_days // 2,
            'predecessors': ['Site Assessment & Survey'],
            'resources': 'Engineering Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Design',
            'task': 'Design Review & Approval',
            'duration_days': design_days // 4,
            'predecessors': ['Electrical Design', 'Structural Design'],
            'resources': 'QA Team',
            'status': 'Not Started'
        })

        # Procurement phase (20% of total time)
        procurement_days = max(30, int(system_capacity_kw / installation_rate_kw_per_day * 0.20))
        tasks.append({
            'phase': 'Procurement',
            'task': 'Equipment Procurement',
            'duration_days': procurement_days,
            'predecessors': ['Design Review & Approval'],
            'resources': 'Procurement Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Procurement',
            'task': 'Equipment Delivery',
            'duration_days': procurement_days // 3,
            'predecessors': ['Equipment Procurement'],
            'resources': 'Logistics Team',
            'status': 'Not Started'
        })

        # Construction phase (45% of total time)
        construction_days = int(system_capacity_kw / installation_rate_kw_per_day)
        tasks.append({
            'phase': 'Construction',
            'task': 'Site Preparation',
            'duration_days': construction_days // 6,
            'predecessors': ['Financing Closure', 'Equipment Delivery'],
            'resources': 'Civil Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Construction',
            'task': 'Foundation & Mounting',
            'duration_days': construction_days // 4,
            'predecessors': ['Site Preparation'],
            'resources': 'Installation Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Construction',
            'task': 'Module Installation',
            'duration_days': construction_days // 3,
            'predecessors': ['Foundation & Mounting'],
            'resources': 'Installation Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Construction',
            'task': 'Electrical Installation',
            'duration_days': construction_days // 4,
            'predecessors': ['Module Installation'],
            'resources': 'Electrical Team',
            'status': 'Not Started'
        })

        # Commissioning phase (10% of total time)
        commissioning_days = max(7, int(system_capacity_kw / installation_rate_kw_per_day * 0.10))
        tasks.append({
            'phase': 'Commissioning',
            'task': 'System Testing',
            'duration_days': commissioning_days // 2,
            'predecessors': ['Electrical Installation'],
            'resources': 'Commissioning Team',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Commissioning',
            'task': 'Grid Interconnection',
            'duration_days': commissioning_days // 3,
            'predecessors': ['System Testing'],
            'resources': 'Utility Liaison',
            'status': 'Not Started'
        })
        tasks.append({
            'phase': 'Commissioning',
            'task': 'Final Inspection',
            'duration_days': commissioning_days // 4,
            'predecessors': ['Grid Interconnection'],
            'resources': 'QA Team',
            'status': 'Not Started'
        })

        # Calculate dates
        task_df = pd.DataFrame(tasks)
        task_df['start_date'] = pd.NaT
        task_df['end_date'] = pd.NaT

        # Forward pass for scheduling
        for idx, row in task_df.iterrows():
            if not row['predecessors']:
                task_df.at[idx, 'start_date'] = start_date
            else:
                # Find latest end date of predecessors
                pred_end_dates = []
                for pred_name in row['predecessors']:
                    pred_idx = task_df[task_df['task'] == pred_name].index[0]
                    pred_end_dates.append(task_df.at[pred_idx, 'end_date'])
                task_df.at[idx, 'start_date'] = max(pred_end_dates)

            task_df.at[idx, 'end_date'] = (
                task_df.at[idx, 'start_date'] + timedelta(days=int(row['duration_days']))
            )

        # Calculate progress percentage (assuming linear progress)
        today = datetime.now()
        task_df['progress'] = 0

        for idx, row in task_df.iterrows():
            if today >= row['end_date']:
                task_df.at[idx, 'progress'] = 100
                task_df.at[idx, 'status'] = 'Completed'
            elif today >= row['start_date']:
                days_elapsed = (today - row['start_date']).days
                task_df.at[idx, 'progress'] = min(100, (days_elapsed / row['duration_days']) * 100)
                task_df.at[idx, 'status'] = 'In Progress'

        return task_df

    def create_resource_allocation(
        self,
        schedule_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create resource allocation plan.

        Args:
            schedule_df: Project schedule DataFrame

        Returns:
            DataFrame with resource allocation
        """
        resource_types = schedule_df['resources'].unique()

        # Create daily resource allocation
        start_date = schedule_df['start_date'].min()
        end_date = schedule_df['end_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        allocation_data = []

        for date in date_range:
            daily_allocation = {'date': date}

            for resource_type in resource_types:
                # Count tasks using this resource on this date
                count = 0
                for _, task in schedule_df.iterrows():
                    if task['resources'] == resource_type:
                        if task['start_date'] <= date <= task['end_date']:
                            count += 1

                daily_allocation[resource_type] = count

            allocation_data.append(daily_allocation)

        return pd.DataFrame(allocation_data)

    def create_equipment_inventory(
        self,
        system_capacity_kw: float,
        module_power_w: int = 450,
        num_inverters: int = 10
    ) -> pd.DataFrame:
        """
        Create equipment inventory tracking.

        Args:
            system_capacity_kw: System capacity (kW)
            module_power_w: Individual module power (W)
            num_inverters: Number of inverters

        Returns:
            DataFrame with equipment inventory
        """
        num_modules = int((system_capacity_kw * 1000) / module_power_w)

        equipment = [
            {
                'category': 'PV Modules',
                'item': f'{module_power_w}W Solar Module',
                'quantity_required': num_modules,
                'quantity_received': int(num_modules * 0.85),
                'quantity_installed': int(num_modules * 0.60),
                'unit': 'pcs',
                'status': 'In Progress'
            },
            {
                'category': 'Inverters',
                'item': 'String Inverter',
                'quantity_required': num_inverters,
                'quantity_received': num_inverters,
                'quantity_installed': int(num_inverters * 0.70),
                'unit': 'pcs',
                'status': 'In Progress'
            },
            {
                'category': 'Mounting System',
                'item': 'Mounting Structures',
                'quantity_required': int(num_modules / 30),
                'quantity_received': int(num_modules / 30 * 0.90),
                'quantity_installed': int(num_modules / 30 * 0.65),
                'unit': 'sets',
                'status': 'In Progress'
            },
            {
                'category': 'Electrical',
                'item': 'DC Cables',
                'quantity_required': int(num_modules * 1.5),
                'quantity_received': int(num_modules * 1.5 * 0.95),
                'quantity_installed': int(num_modules * 1.5 * 0.70),
                'unit': 'm',
                'status': 'In Progress'
            },
            {
                'category': 'Electrical',
                'item': 'AC Cables',
                'quantity_required': int(num_inverters * 50),
                'quantity_received': int(num_inverters * 50),
                'quantity_installed': int(num_inverters * 50 * 0.75),
                'unit': 'm',
                'status': 'In Progress'
            },
            {
                'category': 'Electrical',
                'item': 'Combiner Boxes',
                'quantity_required': int(num_inverters * 2),
                'quantity_received': int(num_inverters * 2),
                'quantity_installed': int(num_inverters * 2 * 0.80),
                'unit': 'pcs',
                'status': 'In Progress'
            }
        ]

        df = pd.DataFrame(equipment)
        df['completion_pct'] = (df['quantity_installed'] / df['quantity_required'] * 100).round(1)

        return df

    def create_qa_checkpoints(self) -> pd.DataFrame:
        """
        Create quality assurance checkpoint checklist.

        Returns:
            DataFrame with QA checkpoints
        """
        checkpoints = [
            {
                'phase': 'Design',
                'checkpoint': 'Electrical Design Review',
                'criteria': 'IEC 62446 Compliance',
                'status': 'Completed',
                'inspector': 'Lead Engineer',
                'date': datetime.now() - timedelta(days=45),
                'notes': 'Approved with minor revisions'
            },
            {
                'phase': 'Design',
                'checkpoint': 'Structural Analysis',
                'criteria': 'Wind & Snow Load Calculations',
                'status': 'Completed',
                'inspector': 'Structural Engineer',
                'date': datetime.now() - timedelta(days=40),
                'notes': 'Passed all requirements'
            },
            {
                'phase': 'Procurement',
                'checkpoint': 'Module Quality Inspection',
                'criteria': 'IEC 61215 Certification',
                'status': 'Completed',
                'inspector': 'QA Inspector',
                'date': datetime.now() - timedelta(days=30),
                'notes': 'Flash test reports verified'
            },
            {
                'phase': 'Construction',
                'checkpoint': 'Foundation Inspection',
                'criteria': 'Concrete Strength Test',
                'status': 'In Progress',
                'inspector': 'Civil Engineer',
                'date': datetime.now() - timedelta(days=10),
                'notes': 'Curing in progress'
            },
            {
                'phase': 'Construction',
                'checkpoint': 'Module Installation',
                'criteria': 'Torque Specifications',
                'status': 'In Progress',
                'inspector': 'Installation Supervisor',
                'date': datetime.now() - timedelta(days=5),
                'notes': '60% complete'
            },
            {
                'phase': 'Construction',
                'checkpoint': 'Electrical Continuity Test',
                'criteria': 'Zero Resistance',
                'status': 'Not Started',
                'inspector': 'Electrician',
                'date': None,
                'notes': 'Scheduled for next week'
            },
            {
                'phase': 'Commissioning',
                'checkpoint': 'String Testing',
                'criteria': 'Voc & Isc Measurements',
                'status': 'Not Started',
                'inspector': 'Commissioning Engineer',
                'date': None,
                'notes': 'Pending electrical completion'
            },
            {
                'phase': 'Commissioning',
                'checkpoint': 'Inverter Configuration',
                'criteria': 'MPPT & Protection Settings',
                'status': 'Not Started',
                'inspector': 'Commissioning Engineer',
                'date': None,
                'notes': 'Awaiting grid approval'
            },
            {
                'phase': 'Commissioning',
                'checkpoint': 'Final Performance Test',
                'criteria': 'PR > 80%',
                'status': 'Not Started',
                'inspector': 'Project Manager',
                'date': None,
                'notes': 'Final milestone'
            }
        ]

        return pd.DataFrame(checkpoints)

    def create_safety_compliance_log(self) -> pd.DataFrame:
        """
        Create safety compliance tracking log.

        Returns:
            DataFrame with safety incidents and compliance
        """
        safety_records = [
            {
                'date': datetime.now() - timedelta(days=60),
                'type': 'Safety Training',
                'description': 'Fall Protection Training',
                'personnel': 'All Construction Team',
                'status': 'Completed',
                'follow_up': 'None'
            },
            {
                'date': datetime.now() - timedelta(days=45),
                'type': 'Safety Training',
                'description': 'Electrical Safety Training',
                'personnel': 'Electrical Team',
                'status': 'Completed',
                'follow_up': 'Refresher in 6 months'
            },
            {
                'date': datetime.now() - timedelta(days=30),
                'type': 'Safety Inspection',
                'description': 'Site Safety Audit',
                'personnel': 'Safety Officer',
                'status': 'Completed',
                'follow_up': '2 minor issues resolved'
            },
            {
                'date': datetime.now() - timedelta(days=20),
                'type': 'Near Miss',
                'description': 'Tool dropped from height',
                'personnel': 'Worker #23',
                'status': 'Investigated',
                'follow_up': 'Tool lanyard policy reinforced'
            },
            {
                'date': datetime.now() - timedelta(days=15),
                'type': 'Safety Inspection',
                'description': 'PPE Compliance Check',
                'personnel': 'All Teams',
                'status': 'Completed',
                'follow_up': '100% compliance'
            },
            {
                'date': datetime.now() - timedelta(days=7),
                'type': 'Safety Meeting',
                'description': 'Weekly Safety Toolbox Talk',
                'personnel': 'All Construction Team',
                'status': 'Completed',
                'follow_up': 'Next meeting scheduled'
            },
            {
                'date': datetime.now() - timedelta(days=2),
                'type': 'Emergency Drill',
                'description': 'Fire Evacuation Drill',
                'personnel': 'All Site Personnel',
                'status': 'Completed',
                'follow_up': 'Evacuation time: 3:45 min'
            }
        ]

        return pd.DataFrame(safety_records)

    def create_document_register(self) -> pd.DataFrame:
        """
        Create project document register.

        Returns:
            DataFrame with document tracking
        """
        documents = [
            {
                'category': 'Design',
                'document': 'Electrical Single Line Diagram',
                'version': 'v3.2',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=50),
                'owner': 'Lead Electrical Engineer'
            },
            {
                'category': 'Design',
                'document': 'Array Layout Plan',
                'version': 'v2.1',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=48),
                'owner': 'Design Manager'
            },
            {
                'category': 'Permits',
                'document': 'Building Permit',
                'version': 'Final',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=55),
                'owner': 'Permitting Lead'
            },
            {
                'category': 'Permits',
                'document': 'Electrical Permit',
                'version': 'Final',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=52),
                'owner': 'Permitting Lead'
            },
            {
                'category': 'Permits',
                'document': 'Interconnection Agreement',
                'version': 'Final',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=45),
                'owner': 'Utility Liaison'
            },
            {
                'category': 'Procurement',
                'document': 'Module Datasheet',
                'version': 'Rev B',
                'status': 'Received',
                'date': datetime.now() - timedelta(days=35),
                'owner': 'Procurement Manager'
            },
            {
                'category': 'Procurement',
                'document': 'Inverter Datasheet',
                'version': 'Rev C',
                'status': 'Received',
                'date': datetime.now() - timedelta(days=35),
                'owner': 'Procurement Manager'
            },
            {
                'category': 'Quality',
                'document': 'Module Flash Test Reports',
                'version': 'Batch 1-5',
                'status': 'Verified',
                'date': datetime.now() - timedelta(days=28),
                'owner': 'QA Manager'
            },
            {
                'category': 'Quality',
                'document': 'IEC Certification Certificates',
                'version': 'Original',
                'status': 'Filed',
                'date': datetime.now() - timedelta(days=30),
                'owner': 'QA Manager'
            },
            {
                'category': 'Construction',
                'document': 'Daily Progress Reports',
                'version': 'Daily',
                'status': 'In Progress',
                'date': datetime.now(),
                'owner': 'Site Supervisor'
            },
            {
                'category': 'Safety',
                'document': 'Safety Plan',
                'version': 'v1.0',
                'status': 'Approved',
                'date': datetime.now() - timedelta(days=65),
                'owner': 'Safety Officer'
            },
            {
                'category': 'Commissioning',
                'document': 'Commissioning Checklist',
                'version': 'v1.0',
                'status': 'Draft',
                'date': datetime.now() - timedelta(days=10),
                'owner': 'Commissioning Manager'
            }
        ]

        return pd.DataFrame(documents)

    def calculate_project_kpis(
        self,
        schedule_df: pd.DataFrame,
        equipment_df: pd.DataFrame,
        budget: float,
        spent_to_date: float
    ) -> Dict[str, float]:
        """
        Calculate project management KPIs.

        Args:
            schedule_df: Project schedule DataFrame
            equipment_df: Equipment inventory DataFrame
            budget: Total project budget ($)
            spent_to_date: Amount spent to date ($)

        Returns:
            Dictionary with KPI metrics
        """
        # Schedule performance
        total_tasks = len(schedule_df)
        completed_tasks = len(schedule_df[schedule_df['status'] == 'Completed'])
        schedule_progress = (completed_tasks / total_tasks) * 100

        # Equipment installation progress
        total_equipment_pct = equipment_df['completion_pct'].mean()

        # Budget performance
        budget_used_pct = (spent_to_date / budget) * 100
        budget_variance = budget - spent_to_date

        # Time performance
        project_duration = (schedule_df['end_date'].max() - schedule_df['start_date'].min()).days
        today = datetime.now()
        elapsed_days = (today - schedule_df['start_date'].min()).days
        time_elapsed_pct = (elapsed_days / project_duration) * 100

        # Performance index
        schedule_performance_index = schedule_progress / max(time_elapsed_pct, 1)
        cost_performance_index = schedule_progress / max(budget_used_pct, 1)

        return {
            'schedule_progress': schedule_progress,
            'equipment_progress': total_equipment_pct,
            'budget_used_pct': budget_used_pct,
            'budget_variance': budget_variance,
            'time_elapsed_pct': time_elapsed_pct,
            'schedule_performance_index': schedule_performance_index,
            'cost_performance_index': cost_performance_index,
            'project_duration_days': project_duration
        }


def render_infrastructure():
    """Render infrastructure management interface in Streamlit."""
    st.header("🏗️ Infrastructure & Deployment Management")
    st.markdown("Comprehensive project lifecycle management from planning to commissioning.")

    manager = InfrastructureManager()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Project Schedule",
        "👥 Resources",
        "📦 Equipment",
        "✅ QA & Safety",
        "📄 Documents",
        "📊 Project KPIs"
    ])

    with tab1:
        st.subheader("Project Schedule & Timeline")

        col1, col2 = st.columns(2)

        with col1:
            system_capacity = st.number_input("System Capacity (kW):", min_value=100, max_value=100000, value=5000, step=500)
            start_date = st.date_input("Project Start Date:", value=datetime.now() - timedelta(days=90))

        with col2:
            installation_rate = st.slider("Installation Rate (kW/day):", 50, 500, 100, 10)

        if st.button("📅 Generate Project Schedule", key="gen_schedule"):
            with st.spinner("Creating project schedule..."):
                schedule_df = manager.create_project_schedule(
                    system_capacity,
                    datetime.combine(start_date, datetime.min.time()),
                    installation_rate
                )

            st.session_state['schedule_df'] = schedule_df

            st.success("✅ Project Schedule Created")

            # Project summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_duration = (schedule_df['end_date'].max() - schedule_df['start_date'].min()).days
                st.metric("Project Duration", f"{total_duration} days")

            with col2:
                completed_tasks = len(schedule_df[schedule_df['status'] == 'Completed'])
                st.metric("Completed Tasks", f"{completed_tasks}/{len(schedule_df)}")

            with col3:
                avg_progress = schedule_df['progress'].mean()
                st.metric("Overall Progress", f"{avg_progress:.1f}%")

            with col4:
                expected_completion = schedule_df['end_date'].max()
                st.metric("Expected Completion", expected_completion.strftime("%Y-%m-%d"))

            # Gantt chart
            st.subheader("Gantt Chart")

            fig = go.Figure()

            for phase in schedule_df['phase'].unique():
                phase_tasks = schedule_df[schedule_df['phase'] == phase]

                for _, task in phase_tasks.iterrows():
                    fig.add_trace(go.Bar(
                        x=[(task['end_date'] - task['start_date']).days],
                        y=[task['task']],
                        base=[task['start_date']],
                        orientation='h',
                        name=phase,
                        text=f"{task['progress']:.0f}%",
                        textposition='inside',
                        hovertemplate=(
                            f"<b>{task['task']}</b><br>"
                            f"Phase: {task['phase']}<br>"
                            f"Duration: {task['duration_days']} days<br>"
                            f"Start: {task['start_date'].strftime('%Y-%m-%d')}<br>"
                            f"End: {task['end_date'].strftime('%Y-%m-%d')}<br>"
                            f"Progress: {task['progress']:.1f}%<br>"
                            f"Status: {task['status']}"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))

            # Add today marker
            fig.add_vline(
                x=datetime.now(),
                line_dash="dash",
                line_color="red",
                annotation_text="Today"
            )

            fig.update_layout(
                title="Project Gantt Chart",
                xaxis_title="Timeline",
                yaxis_title="Task",
                height=600,
                barmode='overlay',
                template='plotly_white',
                xaxis=dict(type='date')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Schedule table
            st.subheader("Detailed Task List")

            display_df = schedule_df[[
                'phase', 'task', 'duration_days', 'start_date', 'end_date',
                'resources', 'status', 'progress'
            ]].copy()

            display_df['start_date'] = display_df['start_date'].dt.strftime('%Y-%m-%d')
            display_df['end_date'] = display_df['end_date'].dt.strftime('%Y-%m-%d')
            display_df['progress'] = display_df['progress'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(display_df, use_container_width=True, height=400)

            # Phase progress
            st.subheader("Progress by Phase")

            phase_progress = schedule_df.groupby('phase')['progress'].mean().reset_index()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=phase_progress['phase'],
                y=phase_progress['progress'],
                text=phase_progress['progress'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker_color='#2ECC71'
            ))

            fig.update_layout(
                title="Progress by Project Phase",
                xaxis_title="Phase",
                yaxis_title="Progress (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Resource Allocation & Scheduling")

        if 'schedule_df' not in st.session_state:
            st.warning("⚠️ Please generate project schedule first")
        else:
            if st.button("👥 Generate Resource Allocation", key="gen_resources"):
                resource_df = manager.create_resource_allocation(st.session_state['schedule_df'])

                st.session_state['resource_df'] = resource_df

                st.success("✅ Resource Allocation Created")

                # Resource utilization heatmap
                st.subheader("Resource Utilization Over Time")

                # Prepare data for heatmap
                resource_cols = [col for col in resource_df.columns if col != 'date']

                # Resample to weekly for better visualization
                resource_df_weekly = resource_df.set_index('date')
                resource_df_weekly = resource_df_weekly.resample('W').mean()

                fig = go.Figure()

                for resource in resource_cols:
                    fig.add_trace(go.Scatter(
                        x=resource_df_weekly.index,
                        y=resource_df_weekly[resource],
                        mode='lines',
                        name=resource,
                        stackgroup='one'
                    ))

                fig.update_layout(
                    title="Resource Allocation Timeline",
                    xaxis_title="Date",
                    yaxis_title="Number of Resources",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Resource summary
                st.subheader("Resource Summary")

                resource_summary = []
                for resource in resource_cols:
                    peak = resource_df[resource].max()
                    avg = resource_df[resource].mean()
                    total_days = (resource_df[resource] > 0).sum()

                    resource_summary.append({
                        'Resource': resource,
                        'Peak Utilization': int(peak),
                        'Average Utilization': f"{avg:.1f}",
                        'Active Days': int(total_days)
                    })

                st.dataframe(pd.DataFrame(resource_summary), use_container_width=True)

    with tab3:
        st.subheader("Equipment Inventory Tracking")

        col1, col2 = st.columns(2)

        with col1:
            eq_system_capacity = st.number_input("System Capacity (kW):", min_value=100, max_value=100000, value=5000, step=500, key="eq_cap")
            module_power = st.selectbox("Module Power (W):", [400, 450, 500, 550, 600], index=1)

        with col2:
            num_inverters = st.number_input("Number of Inverters:", min_value=1, max_value=100, value=10)

        if st.button("📦 Generate Equipment Inventory", key="gen_equipment"):
            equipment_df = manager.create_equipment_inventory(
                eq_system_capacity,
                module_power,
                num_inverters
            )

            st.session_state['equipment_df'] = equipment_df

            st.success("✅ Equipment Inventory Created")

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                total_received = (equipment_df['quantity_received'] / equipment_df['quantity_required'] * 100).mean()
                st.metric("Equipment Received", f"{total_received:.1f}%")

            with col2:
                total_installed = (equipment_df['quantity_installed'] / equipment_df['quantity_required'] * 100).mean()
                st.metric("Equipment Installed", f"{total_installed:.1f}%")

            with col3:
                critical_items = len(equipment_df[equipment_df['completion_pct'] < 50])
                st.metric("Critical Items (<50%)", critical_items)

            # Equipment table
            st.subheader("Equipment Inventory")

            st.dataframe(equipment_df, use_container_width=True)

            # Visual progress by category
            st.subheader("Installation Progress by Category")

            fig = go.Figure()

            categories = equipment_df['category'].unique()

            for category in categories:
                cat_data = equipment_df[equipment_df['category'] == category]

                fig.add_trace(go.Bar(
                    name=category,
                    x=cat_data['item'],
                    y=cat_data['completion_pct'],
                    text=cat_data['completion_pct'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                ))

            fig.update_layout(
                title="Equipment Installation Progress",
                xaxis_title="Item",
                yaxis_title="Completion (%)",
                height=400,
                template='plotly_white',
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Quality Assurance & Safety Compliance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Quality Assurance Checkpoints")

            qa_df = manager.create_qa_checkpoints()

            st.session_state['qa_df'] = qa_df

            # QA status summary
            qa_status_counts = qa_df['status'].value_counts()

            fig = go.Figure()

            fig.add_trace(go.Pie(
                labels=qa_status_counts.index,
                values=qa_status_counts.values,
                hole=0.4,
                marker=dict(colors=['#2ECC71', '#F39C12', '#95A5A6'])
            ))

            fig.update_layout(
                title="QA Checkpoint Status",
                height=300,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # QA table
            display_qa = qa_df[['phase', 'checkpoint', 'status', 'inspector']].copy()
            st.dataframe(display_qa, use_container_width=True, height=300)

        with col2:
            st.write("### Safety Compliance Log")

            safety_df = manager.create_safety_compliance_log()

            st.session_state['safety_df'] = safety_df

            # Safety metrics
            total_records = len(safety_df)
            incidents = len(safety_df[safety_df['type'] == 'Near Miss'])

            days_since_incident = (datetime.now() - safety_df[safety_df['type'] == 'Near Miss']['date'].max()).days

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Days Since Incident", days_since_incident)

            with col_b:
                st.metric("Total Safety Records", total_records)

            # Safety records by type
            safety_counts = safety_df['type'].value_counts()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=safety_counts.index,
                y=safety_counts.values,
                marker_color='#3498DB',
                text=safety_counts.values,
                textposition='auto'
            ))

            fig.update_layout(
                title="Safety Records by Type",
                xaxis_title="Type",
                yaxis_title="Count",
                height=300,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Combined timeline
        st.subheader("QA & Safety Timeline")

        fig = go.Figure()

        # Add QA checkpoints
        qa_completed = qa_df[qa_df['date'].notna()]
        if not qa_completed.empty:
            fig.add_trace(go.Scatter(
                x=qa_completed['date'],
                y=['QA'] * len(qa_completed),
                mode='markers',
                marker=dict(size=12, color='#2ECC71', symbol='circle'),
                text=qa_completed['checkpoint'],
                name='QA Checkpoints',
                hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
            ))

        # Add safety records
        fig.add_trace(go.Scatter(
            x=safety_df['date'],
            y=['Safety'] * len(safety_df),
            mode='markers',
            marker=dict(size=10, color='#E74C3C', symbol='diamond'),
            text=safety_df['description'],
            name='Safety Records',
            hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title="QA & Safety Events Timeline",
            xaxis_title="Date",
            yaxis_title="Category",
            height=300,
            template='plotly_white',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Document Management")

        doc_df = manager.create_document_register()

        st.session_state['doc_df'] = doc_df

        # Document statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", len(doc_df))

        with col2:
            approved = len(doc_df[doc_df['status'] == 'Approved'])
            st.metric("Approved", approved)

        with col3:
            in_progress = len(doc_df[doc_df['status'].isin(['Draft', 'In Progress'])])
            st.metric("In Progress", in_progress)

        with col4:
            categories = doc_df['category'].nunique()
            st.metric("Categories", categories)

        # Documents by category
        st.subheader("Documents by Category")

        category_counts = doc_df['category'].value_counts()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=category_counts.index,
            y=category_counts.values,
            text=category_counts.values,
            textposition='auto',
            marker_color='#9B59B6'
        ))

        fig.update_layout(
            title="Document Distribution by Category",
            xaxis_title="Category",
            yaxis_title="Count",
            height=350,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Document register table
        st.subheader("Document Register")

        st.dataframe(doc_df, use_container_width=True, height=400)

        # Filter documents
        st.subheader("Filter Documents")

        col1, col2 = st.columns(2)

        with col1:
            filter_category = st.multiselect(
                "Filter by Category:",
                options=doc_df['category'].unique(),
                default=list(doc_df['category'].unique())
            )

        with col2:
            filter_status = st.multiselect(
                "Filter by Status:",
                options=doc_df['status'].unique(),
                default=list(doc_df['status'].unique())
            )

        filtered_df = doc_df[
            (doc_df['category'].isin(filter_category)) &
            (doc_df['status'].isin(filter_status))
        ]

        st.write(f"**Showing {len(filtered_df)} of {len(doc_df)} documents**")
        st.dataframe(filtered_df, use_container_width=True)

    with tab6:
        st.subheader("Project Performance KPIs")

        if 'schedule_df' not in st.session_state or 'equipment_df' not in st.session_state:
            st.warning("⚠️ Please generate project schedule and equipment inventory first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                project_budget = st.number_input("Total Project Budget ($):", min_value=100000, max_value=100000000, value=5000000, step=100000)

            with col2:
                spent_to_date = st.number_input("Spent to Date ($):", min_value=0, max_value=project_budget, value=int(project_budget * 0.55), step=50000)

            if st.button("📊 Calculate Project KPIs", key="calc_kpis"):
                kpis = manager.calculate_project_kpis(
                    st.session_state['schedule_df'],
                    st.session_state['equipment_df'],
                    project_budget,
                    spent_to_date
                )

                st.session_state['kpis'] = kpis

                st.success("✅ Project KPIs Calculated")

                # Display KPIs
                st.subheader("Key Performance Indicators")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Schedule Progress", f"{kpis['schedule_progress']:.1f}%")

                with col2:
                    st.metric("Equipment Progress", f"{kpis['equipment_progress']:.1f}%")

                with col3:
                    st.metric("Budget Used", f"{kpis['budget_used_pct']:.1f}%")

                with col4:
                    st.metric("Time Elapsed", f"{kpis['time_elapsed_pct']:.1f}%")

                # Performance indices
                st.subheader("Performance Indices")

                col1, col2, col3 = st.columns(3)

                with col1:
                    spi = kpis['schedule_performance_index']
                    spi_status = "🟢 On Track" if spi >= 0.95 else "🟡 At Risk" if spi >= 0.85 else "🔴 Behind"
                    st.metric("Schedule Performance Index (SPI)", f"{spi:.2f}", help="SPI > 1.0 is ahead, < 1.0 is behind")
                    st.caption(spi_status)

                with col2:
                    cpi = kpis['cost_performance_index']
                    cpi_status = "🟢 Under Budget" if cpi >= 1.0 else "🟡 At Budget" if cpi >= 0.9 else "🔴 Over Budget"
                    st.metric("Cost Performance Index (CPI)", f"{cpi:.2f}", help="CPI > 1.0 is under budget, < 1.0 is over")
                    st.caption(cpi_status)

                with col3:
                    st.metric("Budget Variance", f"${format_number(kpis['budget_variance'])}")

                # Progress comparison
                st.subheader("Progress vs Time vs Budget")

                fig = go.Figure()

                categories = ['Schedule Progress', 'Time Elapsed', 'Budget Used']
                values = [
                    kpis['schedule_progress'],
                    kpis['time_elapsed_pct'],
                    kpis['budget_used_pct']
                ]
                colors = ['#2ECC71', '#3498DB', '#F39C12']

                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='auto',
                    marker_color=colors
                ))

                fig.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Target"
                )

                fig.update_layout(
                    title="Project Progress Comparison",
                    yaxis_title="Percentage (%)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Earned Value Analysis
                st.subheader("Earned Value Analysis")

                planned_value = project_budget * (kpis['time_elapsed_pct'] / 100)
                earned_value = project_budget * (kpis['schedule_progress'] / 100)
                actual_cost = spent_to_date

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=['Planned Value', 'Earned Value', 'Actual Cost'],
                    y=[planned_value, earned_value, actual_cost],
                    text=[f"${format_number(v)}" for v in [planned_value, earned_value, actual_cost]],
                    textposition='auto',
                    marker_color=['#3498DB', '#2ECC71', '#E74C3C']
                ))

                fig.update_layout(
                    title="Earned Value Management",
                    yaxis_title="Value ($)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Project health summary
                st.subheader("Project Health Summary")

                overall_health = "🟢 Healthy" if spi >= 0.95 and cpi >= 1.0 else \
                                "🟡 At Risk" if spi >= 0.85 or cpi >= 0.9 else \
                                "🔴 Critical"

                st.info(f"""
                **Project Status: {overall_health}**

                - Schedule Performance: {'On Track' if spi >= 0.95 else 'Behind Schedule'}
                - Cost Performance: {'Under Budget' if cpi >= 1.0 else 'Over Budget'}
                - Overall Progress: {kpis['schedule_progress']:.1f}%
                - Project Duration: {kpis['project_duration_days']} days
                """)

    st.divider()
    st.info("💡 **Infrastructure & Deployment Management** - Branch B14 | Complete Project Management Suite")
