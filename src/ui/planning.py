"""
Planning UI components for PV Circularity Simulator.

This module contains the core planning interface functions:
- project_wizard(): Interactive project creation wizard
- timeline_planner(): Timeline and milestone planning with date pickers
- resource_allocation_dashboard(): Resource management interface
- contract_templates(): Contract template management with file uploads
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

from src.data.models import (
    Project, Resource, Contract, Portfolio, Timeline,
    ProjectStatus, ResourceType, ContractType, ContractStatus
)
from src.core.state_manager import StateManager


def project_wizard() -> None:
    """
    Interactive multi-step project creation wizard.

    Provides a guided workflow for creating new PV projects with:
    - Basic project information (name, description, owner)
    - Technical specifications (capacity, location)
    - Budget and timeline planning
    - Initial resource estimation

    The wizard uses Streamlit forms and step-by-step navigation to ensure
    all required information is collected before project creation.

    Side Effects:
        - Creates new Project instance in session state
        - Initializes associated Timeline
        - Saves data to persistent storage
        - Updates wizard state in session_state
    """
    st.header("🧙 Project Creation Wizard")
    st.markdown("Create a new PV project with guided step-by-step setup.")

    # Initialize wizard state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0
    if "wizard_data" not in st.session_state:
        st.session_state.wizard_data = {}

    # Progress indicator
    steps = ["Basic Info", "Technical Details", "Timeline & Budget", "Review"]
    current_step = st.session_state.wizard_step

    # Display progress
    cols = st.columns(len(steps))
    for idx, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if idx < current_step:
                st.success(f"✓ {step}")
            elif idx == current_step:
                st.info(f"▶ {step}")
            else:
                st.text(f"○ {step}")

    st.divider()

    # Step 1: Basic Information
    if current_step == 0:
        st.subheader("Step 1: Basic Information")

        with st.form("step1_form"):
            project_name = st.text_input(
                "Project Name *",
                value=st.session_state.wizard_data.get("name", ""),
                placeholder="e.g., Solar Farm Alpha"
            )

            project_description = st.text_area(
                "Description",
                value=st.session_state.wizard_data.get("description", ""),
                placeholder="Describe the project scope and objectives...",
                height=100
            )

            owner = st.text_input(
                "Project Owner/Manager *",
                value=st.session_state.wizard_data.get("owner", ""),
                placeholder="e.g., John Doe"
            )

            location = st.text_input(
                "Installation Location *",
                value=st.session_state.wizard_data.get("location", ""),
                placeholder="e.g., Phoenix, Arizona, USA"
            )

            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Next →", type="primary")
            with col2:
                cancel = st.form_submit_button("Cancel")

            if submit:
                if not project_name or not owner or not location:
                    st.error("Please fill in all required fields (*)")
                else:
                    st.session_state.wizard_data.update({
                        "name": project_name,
                        "description": project_description,
                        "owner": owner,
                        "location": location
                    })
                    st.session_state.wizard_step = 1
                    st.rerun()

            if cancel:
                st.session_state.wizard_step = 0
                st.session_state.wizard_data = {}
                st.rerun()

    # Step 2: Technical Details
    elif current_step == 1:
        st.subheader("Step 2: Technical Specifications")

        with st.form("step2_form"):
            capacity = st.number_input(
                "System Capacity (kWp) *",
                min_value=0.0,
                value=st.session_state.wizard_data.get("capacity_kwp", 0.0),
                step=0.1,
                help="Total system capacity in kilowatt-peak"
            )

            status = st.selectbox(
                "Initial Project Status",
                options=[s.value for s in ProjectStatus],
                index=0
            )

            st.markdown("### Additional Specifications")

            col1, col2 = st.columns(2)
            with col1:
                module_type = st.text_input(
                    "PV Module Type",
                    value=st.session_state.wizard_data.get("module_type", ""),
                    placeholder="e.g., Monocrystalline"
                )
                inverter_type = st.text_input(
                    "Inverter Type",
                    value=st.session_state.wizard_data.get("inverter_type", ""),
                    placeholder="e.g., String Inverter"
                )

            with col2:
                module_count = st.number_input(
                    "Estimated Module Count",
                    min_value=0,
                    value=st.session_state.wizard_data.get("module_count", 0),
                    step=1
                )
                mounting_system = st.text_input(
                    "Mounting System",
                    value=st.session_state.wizard_data.get("mounting_system", ""),
                    placeholder="e.g., Fixed Tilt"
                )

            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button("← Back")
            with col2:
                submit = st.form_submit_button("Next →", type="primary")

            if submit:
                if capacity <= 0:
                    st.error("Please enter a valid system capacity")
                else:
                    st.session_state.wizard_data.update({
                        "capacity_kwp": capacity,
                        "status": status,
                        "module_type": module_type,
                        "inverter_type": inverter_type,
                        "module_count": module_count,
                        "mounting_system": mounting_system
                    })
                    st.session_state.wizard_step = 2
                    st.rerun()

            if back:
                st.session_state.wizard_step = 0
                st.rerun()

    # Step 3: Timeline & Budget
    elif current_step == 2:
        st.subheader("Step 3: Timeline & Budget Planning")

        with st.form("step3_form"):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Project Start Date *",
                    value=st.session_state.wizard_data.get("start_date", datetime.now().date())
                )

            with col2:
                end_date = st.date_input(
                    "Expected Completion Date *",
                    value=st.session_state.wizard_data.get(
                        "end_date",
                        (datetime.now() + timedelta(days=365)).date()
                    )
                )

            budget = st.number_input(
                "Total Project Budget (USD) *",
                min_value=0.0,
                value=st.session_state.wizard_data.get("budget", 0.0),
                step=1000.0,
                format="%.2f"
            )

            st.markdown("### Key Milestones (Optional)")

            milestone1 = st.text_input("Milestone 1", placeholder="e.g., Design Approval")
            milestone1_date = st.date_input("Milestone 1 Date", disabled=not milestone1)

            milestone2 = st.text_input("Milestone 2", placeholder="e.g., Equipment Procurement")
            milestone2_date = st.date_input("Milestone 2 Date", disabled=not milestone2)

            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button("← Back")
            with col2:
                submit = st.form_submit_button("Next →", type="primary")

            if submit:
                if not start_date or not end_date or budget <= 0:
                    st.error("Please fill in all required fields")
                elif end_date < start_date:
                    st.error("End date must be after start date")
                else:
                    milestones = []
                    if milestone1:
                        milestones.append({"name": milestone1, "date": milestone1_date})
                    if milestone2:
                        milestones.append({"name": milestone2, "date": milestone2_date})

                    st.session_state.wizard_data.update({
                        "start_date": start_date,
                        "end_date": end_date,
                        "budget": budget,
                        "milestones": milestones
                    })
                    st.session_state.wizard_step = 3
                    st.rerun()

            if back:
                st.session_state.wizard_step = 1
                st.rerun()

    # Step 4: Review & Create
    elif current_step == 3:
        st.subheader("Step 4: Review & Create Project")

        data = st.session_state.wizard_data

        # Display summary
        st.markdown("### Project Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {data.get('name')}")
            st.markdown(f"**Owner:** {data.get('owner')}")
            st.markdown(f"**Location:** {data.get('location')}")
            st.markdown(f"**Capacity:** {data.get('capacity_kwp')} kWp")

        with col2:
            st.markdown(f"**Budget:** ${data.get('budget'):,.2f}")
            st.markdown(f"**Start Date:** {data.get('start_date')}")
            st.markdown(f"**End Date:** {data.get('end_date')}")
            st.markdown(f"**Status:** {data.get('status')}")

        if data.get('description'):
            st.markdown(f"**Description:** {data.get('description')}")

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.wizard_step = 0
                st.session_state.wizard_data = {}
                st.rerun()

        with col3:
            if st.button("✓ Create Project", type="primary", use_container_width=True):
                # Create project
                project = Project(
                    name=data["name"],
                    description=data.get("description", ""),
                    owner=data["owner"],
                    location=data["location"],
                    capacity_kwp=data["capacity_kwp"],
                    status=ProjectStatus[data["status"].upper().replace(" ", "_")],
                    budget=data["budget"],
                    start_date=datetime.combine(data["start_date"], datetime.min.time()),
                    end_date=datetime.combine(data["end_date"], datetime.min.time()),
                    metadata={
                        "module_type": data.get("module_type", ""),
                        "inverter_type": data.get("inverter_type", ""),
                        "module_count": data.get("module_count", 0),
                        "mounting_system": data.get("mounting_system", "")
                    }
                )

                StateManager.add_project(project)

                # Create timeline with milestones
                if data.get("milestones"):
                    timeline = Timeline(project_id=project.id)
                    for milestone in data["milestones"]:
                        timeline.add_milestone(
                            name=milestone["name"],
                            date=datetime.combine(milestone["date"], datetime.min.time())
                        )
                    StateManager.add_timeline(timeline)

                st.success(f"✓ Project '{project.name}' created successfully!")
                st.balloons()

                # Reset wizard
                st.session_state.wizard_step = 0
                st.session_state.wizard_data = {}
                st.session_state.current_project_id = project.id

                st.info("Redirecting to project dashboard...")
                st.rerun()


def timeline_planner(project_id: Optional[str] = None) -> None:
    """
    Interactive timeline and milestone planning interface.

    Provides comprehensive project timeline management with:
    - Gantt chart visualization of project phases
    - Interactive milestone creation with date pickers
    - Phase duration planning and tracking
    - Critical path identification
    - Dependency management between tasks

    Args:
        project_id: Optional project identifier. If None, prompts for selection.

    Side Effects:
        - Creates/updates Timeline instances in session state
        - Renders interactive Plotly Gantt charts
        - Saves timeline data to persistent storage
    """
    st.header("📅 Timeline & Milestone Planner")

    # Project selection
    if not project_id:
        projects = StateManager.get_all_projects()
        if not projects:
            st.warning("No projects available. Please create a project first.")
            return

        project_options = {p["name"]: p["id"] for p in projects}
        selected_project_name = st.selectbox(
            "Select Project",
            options=list(project_options.keys())
        )
        project_id = project_options[selected_project_name]

    project = StateManager.get_project(project_id)
    if not project:
        st.error("Project not found")
        return

    st.subheader(f"Timeline for: {project['name']}")

    # Get or create timeline
    timeline = StateManager.get_project_timeline(project_id)
    if not timeline:
        timeline = Timeline(project_id=project_id).to_dict()
        StateManager.add_timeline(Timeline(project_id=project_id))

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Gantt Chart", "Milestones", "Phases"])

    # Tab 1: Gantt Chart Visualization
    with tab1:
        st.markdown("### Project Timeline Visualization")

        # Prepare data for Gantt chart
        gantt_data = []

        # Add project overall timeline
        if project.get("start_date") and project.get("end_date"):
            gantt_data.append({
                "Task": project["name"],
                "Start": project["start_date"],
                "Finish": project["end_date"],
                "Type": "Project"
            })

        # Add phases
        for phase in timeline.get("phases", []):
            gantt_data.append({
                "Task": phase["name"],
                "Start": phase["start_date"],
                "Finish": phase["end_date"],
                "Type": "Phase"
            })

        # Add milestones as zero-duration tasks
        for milestone in timeline.get("milestones", []):
            gantt_data.append({
                "Task": f"⭐ {milestone['name']}",
                "Start": milestone["date"],
                "Finish": milestone["date"],
                "Type": "Milestone"
            })

        if gantt_data:
            df = pd.DataFrame(gantt_data)

            fig = px.timeline(
                df,
                x_start="Start",
                x_end="Finish",
                y="Task",
                color="Type",
                title="Project Gantt Chart"
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available. Add milestones or phases below.")

    # Tab 2: Milestone Management
    with tab2:
        st.markdown("### Milestone Management")

        # Add new milestone
        with st.expander("➕ Add New Milestone", expanded=False):
            with st.form("add_milestone_form"):
                col1, col2 = st.columns(2)

                with col1:
                    milestone_name = st.text_input(
                        "Milestone Name *",
                        placeholder="e.g., Design Approval"
                    )

                with col2:
                    milestone_date = st.date_input(
                        "Target Date *",
                        value=datetime.now().date()
                    )

                milestone_description = st.text_area(
                    "Description",
                    placeholder="Describe the milestone criteria...",
                    height=100
                )

                if st.form_submit_button("Add Milestone", type="primary"):
                    if milestone_name:
                        # Add milestone to timeline
                        if "milestones" not in timeline:
                            timeline["milestones"] = []

                        new_milestone = {
                            "id": f"milestone_{len(timeline['milestones'])}",
                            "name": milestone_name,
                            "date": milestone_date.isoformat(),
                            "description": milestone_description,
                            "completed": False
                        }

                        timeline["milestones"].append(new_milestone)
                        StateManager.update_timeline(timeline["id"], {"milestones": timeline["milestones"]})

                        st.success(f"Milestone '{milestone_name}' added!")
                        st.rerun()
                    else:
                        st.error("Please enter a milestone name")

        # Display existing milestones
        st.markdown("### Existing Milestones")

        if timeline.get("milestones"):
            for idx, milestone in enumerate(timeline["milestones"]):
                with st.expander(
                    f"{'✓' if milestone.get('completed') else '○'} {milestone['name']} - {milestone['date']}"
                ):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"**Date:** {milestone['date']}")
                        if milestone.get("description"):
                            st.markdown(f"**Description:** {milestone['description']}")

                    with col2:
                        if not milestone.get("completed"):
                            if st.button("Mark Complete", key=f"complete_{idx}"):
                                timeline["milestones"][idx]["completed"] = True
                                StateManager.update_timeline(
                                    timeline["id"],
                                    {"milestones": timeline["milestones"]}
                                )
                                st.rerun()

                    with col3:
                        if st.button("Delete", key=f"delete_milestone_{idx}", type="secondary"):
                            timeline["milestones"].pop(idx)
                            StateManager.update_timeline(
                                timeline["id"],
                                {"milestones": timeline["milestones"]}
                            )
                            st.rerun()
        else:
            st.info("No milestones defined yet. Add your first milestone above.")

    # Tab 3: Phase Management
    with tab3:
        st.markdown("### Project Phases")

        # Add new phase
        with st.expander("➕ Add New Phase", expanded=False):
            with st.form("add_phase_form"):
                phase_name = st.text_input(
                    "Phase Name *",
                    placeholder="e.g., Design Phase"
                )

                col1, col2 = st.columns(2)
                with col1:
                    phase_start = st.date_input(
                        "Start Date *",
                        value=datetime.now().date()
                    )

                with col2:
                    phase_end = st.date_input(
                        "End Date *",
                        value=(datetime.now() + timedelta(days=30)).date()
                    )

                phase_description = st.text_area(
                    "Description",
                    placeholder="Describe the phase activities...",
                    height=100
                )

                if st.form_submit_button("Add Phase", type="primary"):
                    if phase_name and phase_end >= phase_start:
                        if "phases" not in timeline:
                            timeline["phases"] = []

                        new_phase = {
                            "id": f"phase_{len(timeline['phases'])}",
                            "name": phase_name,
                            "start_date": phase_start.isoformat(),
                            "end_date": phase_end.isoformat(),
                            "description": phase_description
                        }

                        timeline["phases"].append(new_phase)
                        StateManager.update_timeline(timeline["id"], {"phases": timeline["phases"]})

                        st.success(f"Phase '{phase_name}' added!")
                        st.rerun()
                    else:
                        st.error("Please check your inputs")

        # Display existing phases
        if timeline.get("phases"):
            for idx, phase in enumerate(timeline["phases"]):
                with st.expander(f"📊 {phase['name']}"):
                    st.markdown(f"**Duration:** {phase['start_date']} to {phase['end_date']}")
                    if phase.get("description"):
                        st.markdown(f"**Description:** {phase['description']}")

                    if st.button("Delete Phase", key=f"delete_phase_{idx}"):
                        timeline["phases"].pop(idx)
                        StateManager.update_timeline(timeline["id"], {"phases": timeline["phases"]})
                        st.rerun()
        else:
            st.info("No phases defined yet. Add your first phase above.")


def resource_allocation_dashboard(project_id: Optional[str] = None) -> None:
    """
    Interactive resource allocation and management dashboard.

    Provides comprehensive resource planning and tracking with:
    - Resource inventory management (modules, inverters, labor, etc.)
    - Allocation status tracking and visualization
    - Cost analysis and budget tracking
    - Supplier management
    - Availability timeline visualization
    - Resource utilization metrics

    Args:
        project_id: Optional project identifier. If None, shows all resources.

    Side Effects:
        - Creates/updates Resource instances in session state
        - Renders resource allocation charts and tables
        - Saves resource data to persistent storage
        - Updates project cost calculations
    """
    st.header("📦 Resource Allocation Dashboard")

    # Project selection
    projects = StateManager.get_all_projects()
    if not projects:
        st.warning("No projects available. Please create a project first.")
        return

    if not project_id:
        project_options = {"All Projects": None}
        project_options.update({p["name"]: p["id"] for p in projects})

        selected_project_name = st.selectbox(
            "Filter by Project",
            options=list(project_options.keys())
        )
        project_id = project_options[selected_project_name]

    # Get resources
    if project_id:
        resources = StateManager.get_project_resources(project_id)
        project = StateManager.get_project(project_id)
        st.subheader(f"Resources for: {project['name']}")
    else:
        resources = list(st.session_state.resources.values())
        st.subheader("All Resources")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_resources = len(resources)
        st.metric("Total Resources", total_resources)

    with col2:
        allocated_resources = sum(1 for r in resources if r.get("allocated"))
        st.metric("Allocated", allocated_resources)

    with col3:
        total_cost = sum(r.get("total_cost", 0) for r in resources)
        st.metric("Total Cost", f"${total_cost:,.2f}")

    with col4:
        if project_id and project:
            budget_used = (total_cost / project["budget"] * 100) if project["budget"] > 0 else 0
            st.metric("Budget Used", f"{budget_used:.1f}%")

    st.divider()

    # Resource management tabs
    tab1, tab2, tab3 = st.tabs(["Resource List", "Add Resource", "Analytics"])

    # Tab 1: Resource List
    with tab1:
        st.markdown("### Resource Inventory")

        if resources:
            # Create dataframe for display
            df_data = []
            for resource in resources:
                df_data.append({
                    "Name": resource["name"],
                    "Type": resource["resource_type"],
                    "Quantity": resource["quantity"],
                    "Unit": resource["unit"],
                    "Unit Cost": f"${resource['unit_cost']:,.2f}",
                    "Total Cost": f"${resource['total_cost']:,.2f}",
                    "Supplier": resource["supplier"],
                    "Allocated": "✓" if resource["allocated"] else "○",
                    "ID": resource["id"]
                })

            df = pd.DataFrame(df_data)

            # Display with selection
            st.dataframe(
                df.drop("ID", axis=1),
                use_container_width=True,
                hide_index=True
            )

            # Resource details in expanders
            st.markdown("### Resource Details")
            for resource in resources:
                with st.expander(f"📦 {resource['name']} - {resource['resource_type']}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Quantity:** {resource['quantity']} {resource['unit']}")
                        st.markdown(f"**Unit Cost:** ${resource['unit_cost']:,.2f}")
                        st.markdown(f"**Total Cost:** ${resource['total_cost']:,.2f}")

                    with col2:
                        st.markdown(f"**Supplier:** {resource['supplier']}")
                        st.markdown(f"**Allocated:** {'Yes' if resource['allocated'] else 'No'}")

                    with col3:
                        if resource.get("availability_start"):
                            st.markdown(f"**Available From:** {resource['availability_start']}")
                        if resource.get("availability_end"):
                            st.markdown(f"**Available Until:** {resource['availability_end']}")

                    if resource.get("constraints"):
                        st.markdown("**Constraints:**")
                        for constraint in resource["constraints"]:
                            st.markdown(f"- {constraint}")

                    # Actions
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        new_allocated = st.checkbox(
                            "Allocated",
                            value=resource["allocated"],
                            key=f"alloc_{resource['id']}"
                        )
                        if new_allocated != resource["allocated"]:
                            StateManager.update_resource(
                                resource["id"],
                                {"allocated": new_allocated}
                            )
                            st.rerun()

                    with col2:
                        if st.button("Edit", key=f"edit_{resource['id']}"):
                            st.info("Edit functionality - implementation note")

                    with col3:
                        if st.button("Delete", key=f"delete_{resource['id']}", type="secondary"):
                            StateManager.delete_resource(resource["id"])
                            st.success("Resource deleted")
                            st.rerun()

        else:
            st.info("No resources allocated yet. Add resources in the 'Add Resource' tab.")

    # Tab 2: Add Resource
    with tab2:
        st.markdown("### Add New Resource")

        with st.form("add_resource_form"):
            # Select project if not already selected
            if not project_id:
                project_options = {p["name"]: p["id"] for p in projects}
                selected_project_name = st.selectbox(
                    "Project *",
                    options=list(project_options.keys())
                )
                selected_project_id = project_options[selected_project_name]
            else:
                selected_project_id = project_id

            resource_name = st.text_input(
                "Resource Name *",
                placeholder="e.g., Monocrystalline PV Module 400W"
            )

            col1, col2 = st.columns(2)

            with col1:
                resource_type = st.selectbox(
                    "Resource Type *",
                    options=[rt.value for rt in ResourceType]
                )

                quantity = st.number_input(
                    "Quantity *",
                    min_value=0.0,
                    value=1.0,
                    step=1.0
                )

                unit_cost = st.number_input(
                    "Unit Cost (USD) *",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f"
                )

            with col2:
                unit = st.text_input(
                    "Unit *",
                    value="pieces",
                    placeholder="e.g., pieces, hours, kW"
                )

                supplier = st.text_input(
                    "Supplier/Vendor",
                    placeholder="e.g., SolarTech Inc."
                )

            st.markdown("### Availability Window")

            col1, col2 = st.columns(2)
            with col1:
                availability_start = st.date_input(
                    "Available From",
                    value=None
                )

            with col2:
                availability_end = st.date_input(
                    "Available Until",
                    value=None
                )

            constraints_text = st.text_area(
                "Constraints (one per line)",
                placeholder="e.g., Requires advance payment\nLead time: 6 weeks",
                height=100
            )

            allocated = st.checkbox("Mark as Allocated", value=False)

            if st.form_submit_button("Add Resource", type="primary"):
                if resource_name and quantity > 0:
                    # Parse constraints
                    constraints = [
                        c.strip() for c in constraints_text.split("\n")
                        if c.strip()
                    ]

                    # Create resource
                    resource = Resource(
                        project_id=selected_project_id,
                        name=resource_name,
                        resource_type=ResourceType[resource_type.upper().replace(" ", "_")],
                        quantity=quantity,
                        unit=unit,
                        unit_cost=unit_cost,
                        supplier=supplier,
                        availability_start=datetime.combine(availability_start, datetime.min.time()) if availability_start else None,
                        availability_end=datetime.combine(availability_end, datetime.min.time()) if availability_end else None,
                        allocated=allocated,
                        constraints=constraints
                    )

                    resource.calculate_total_cost()
                    StateManager.add_resource(resource)

                    st.success(f"Resource '{resource_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

    # Tab 3: Analytics
    with tab3:
        st.markdown("### Resource Analytics")

        if resources:
            # Cost breakdown by type
            st.markdown("#### Cost Breakdown by Resource Type")

            cost_by_type = {}
            for resource in resources:
                rtype = resource["resource_type"]
                cost_by_type[rtype] = cost_by_type.get(rtype, 0) + resource["total_cost"]

            fig = px.pie(
                values=list(cost_by_type.values()),
                names=list(cost_by_type.keys()),
                title="Cost Distribution by Resource Type"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Allocation status
            st.markdown("#### Allocation Status")

            allocated_count = sum(1 for r in resources if r["allocated"])
            unallocated_count = len(resources) - allocated_count

            fig = go.Figure(data=[
                go.Bar(
                    x=["Allocated", "Unallocated"],
                    y=[allocated_count, unallocated_count],
                    marker_color=["#00cc00", "#ff6666"]
                )
            ])
            fig.update_layout(title="Resource Allocation Status", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            # Top suppliers
            st.markdown("#### Top Suppliers by Cost")

            supplier_costs = {}
            for resource in resources:
                supplier = resource["supplier"] or "Unspecified"
                supplier_costs[supplier] = supplier_costs.get(supplier, 0) + resource["total_cost"]

            sorted_suppliers = sorted(supplier_costs.items(), key=lambda x: x[1], reverse=True)[:5]

            if sorted_suppliers:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[s[0] for s in sorted_suppliers],
                        y=[s[1] for s in sorted_suppliers],
                        marker_color="#4CAF50"
                    )
                ])
                fig.update_layout(title="Top 5 Suppliers", yaxis_title="Total Cost (USD)")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No resources available for analytics. Add resources to see insights.")


def contract_templates(project_id: Optional[str] = None) -> None:
    """
    Contract template management with file upload functionality.

    Provides comprehensive contract lifecycle management with:
    - Contract template library (upload/download)
    - Contract creation from templates
    - File upload for signed contracts
    - Contract status tracking
    - Payment schedule management
    - Deliverable tracking
    - Vendor/contractor database

    Args:
        project_id: Optional project identifier. If None, shows all contracts.

    Side Effects:
        - Creates/updates Contract instances in session state
        - Saves uploaded files to uploads/ directory
        - Renders contract management interface
        - Saves contract data to persistent storage
    """
    st.header("📄 Contract Templates & Management")

    # Ensure upload directory exists
    upload_dir = Path("uploads/contracts")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Project selection
    projects = StateManager.get_all_projects()
    if not projects:
        st.warning("No projects available. Please create a project first.")
        return

    if not project_id:
        project_options = {"All Projects": None}
        project_options.update({p["name"]: p["id"] for p in projects})

        selected_project_name = st.selectbox(
            "Filter by Project",
            options=list(project_options.keys())
        )
        project_id = project_options[selected_project_name]

    # Get contracts
    if project_id:
        contracts = StateManager.get_project_contracts(project_id)
        project = StateManager.get_project(project_id)
        st.subheader(f"Contracts for: {project['name']}")
    else:
        contracts = list(st.session_state.contracts.values())
        st.subheader("All Contracts")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_contracts = len(contracts)
        st.metric("Total Contracts", total_contracts)

    with col2:
        active_contracts = sum(
            1 for c in contracts
            if c.get("status") == ContractStatus.ACTIVE.value
        )
        st.metric("Active", active_contracts)

    with col3:
        total_value = sum(c.get("value", 0) for c in contracts)
        st.metric("Total Value", f"${total_value:,.2f}")

    with col4:
        pending_contracts = sum(
            1 for c in contracts
            if c.get("status") == ContractStatus.PENDING.value
        )
        st.metric("Pending Approval", pending_contracts)

    st.divider()

    # Contract management tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Contract List",
        "Create Contract",
        "Upload Template",
        "Template Library"
    ])

    # Tab 1: Contract List
    with tab1:
        st.markdown("### Active Contracts")

        if contracts:
            for contract in contracts:
                with st.expander(
                    f"📄 {contract['title']} - {contract['vendor']} "
                    f"({contract['status']})"
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Type:** {contract['contract_type']}")
                        st.markdown(f"**Vendor:** {contract['vendor']}")
                        st.markdown(f"**Value:** ${contract['value']:,.2f} {contract['currency']}")

                    with col2:
                        st.markdown(f"**Status:** {contract['status']}")
                        if contract.get("start_date"):
                            st.markdown(f"**Start:** {contract['start_date']}")
                        if contract.get("end_date"):
                            st.markdown(f"**End:** {contract['end_date']}")

                    with col3:
                        # Status update
                        new_status = st.selectbox(
                            "Update Status",
                            options=[s.value for s in ContractStatus],
                            index=[s.value for s in ContractStatus].index(contract["status"]),
                            key=f"status_{contract['id']}"
                        )

                        if st.button("Update", key=f"update_status_{contract['id']}"):
                            StateManager.update_contract(
                                contract["id"],
                                {"status": new_status}
                            )
                            st.success("Status updated!")
                            st.rerun()

                    if contract.get("description"):
                        st.markdown(f"**Description:** {contract['description']}")

                    if contract.get("terms"):
                        st.markdown("**Terms & Conditions:**")
                        st.text(contract["terms"])

                    if contract.get("deliverables"):
                        st.markdown("**Deliverables:**")
                        for deliverable in contract["deliverables"]:
                            st.markdown(f"- {deliverable}")

                    if contract.get("payment_schedule"):
                        st.markdown("**Payment Schedule:**")
                        for payment in contract["payment_schedule"]:
                            st.markdown(
                                f"- {payment.get('description', 'Payment')}: "
                                f"${payment.get('amount', 0):,.2f} "
                                f"(Due: {payment.get('due_date', 'TBD')})"
                            )

                    # File attachments
                    if contract.get("template_file"):
                        st.markdown(f"**Template:** {contract['template_file']}")

                    if contract.get("signed_file"):
                        st.markdown(f"**Signed Contract:** {contract['signed_file']}")

                    # Actions
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Download", key=f"download_{contract['id']}"):
                            st.info("Download functionality - file would be served here")

                    with col2:
                        if st.button(
                            "Delete Contract",
                            key=f"delete_contract_{contract['id']}",
                            type="secondary"
                        ):
                            StateManager.delete_contract(contract["id"])
                            st.success("Contract deleted")
                            st.rerun()

        else:
            st.info("No contracts created yet. Create your first contract in the 'Create Contract' tab.")

    # Tab 2: Create Contract
    with tab2:
        st.markdown("### Create New Contract")

        with st.form("create_contract_form"):
            # Select project if not already selected
            if not project_id:
                project_options = {p["name"]: p["id"] for p in projects}
                selected_project_name = st.selectbox(
                    "Project *",
                    options=list(project_options.keys())
                )
                selected_project_id = project_options[selected_project_name]
            else:
                selected_project_id = project_id

            col1, col2 = st.columns(2)

            with col1:
                contract_title = st.text_input(
                    "Contract Title *",
                    placeholder="e.g., PV Module Supply Agreement"
                )

                vendor = st.text_input(
                    "Vendor/Contractor *",
                    placeholder="e.g., SolarTech Inc."
                )

                contract_type = st.selectbox(
                    "Contract Type *",
                    options=[ct.value for ct in ContractType]
                )

            with col2:
                contract_value = st.number_input(
                    "Contract Value *",
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )

                currency = st.selectbox(
                    "Currency",
                    options=["USD", "EUR", "GBP", "JPY", "CNY"],
                    index=0
                )

                contract_status = st.selectbox(
                    "Initial Status",
                    options=[cs.value for cs in ContractStatus],
                    index=0
                )

            description = st.text_area(
                "Description",
                placeholder="Describe the contract scope and objectives...",
                height=100
            )

            terms = st.text_area(
                "Terms & Conditions",
                placeholder="Enter contract terms and conditions...",
                height=150
            )

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=None
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=None
                )

            deliverables_text = st.text_area(
                "Deliverables (one per line)",
                placeholder="e.g., 500 PV modules\nInstallation services\nWarranty documentation",
                height=100
            )

            st.markdown("### Payment Schedule")

            num_payments = st.number_input(
                "Number of Payment Milestones",
                min_value=0,
                max_value=10,
                value=1,
                step=1
            )

            payment_schedule = []
            for i in range(int(num_payments)):
                st.markdown(f"**Payment {i+1}**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    payment_desc = st.text_input(
                        "Description",
                        value=f"Payment {i+1}",
                        key=f"payment_desc_{i}"
                    )

                with col2:
                    payment_amount = st.number_input(
                        "Amount",
                        min_value=0.0,
                        value=0.0,
                        key=f"payment_amount_{i}"
                    )

                with col3:
                    payment_date = st.date_input(
                        "Due Date",
                        value=None,
                        key=f"payment_date_{i}"
                    )

                if payment_amount > 0:
                    payment_schedule.append({
                        "description": payment_desc,
                        "amount": payment_amount,
                        "due_date": payment_date.isoformat() if payment_date else None
                    })

            # File upload for signed contract
            st.markdown("### Attachments")
            signed_file = st.file_uploader(
                "Upload Signed Contract (PDF)",
                type=["pdf"],
                help="Upload the signed contract document"
            )

            if st.form_submit_button("Create Contract", type="primary"):
                if contract_title and vendor and contract_value >= 0:
                    # Parse deliverables
                    deliverables = [
                        d.strip() for d in deliverables_text.split("\n")
                        if d.strip()
                    ]

                    # Handle file upload
                    signed_file_path = None
                    if signed_file:
                        file_path = upload_dir / f"{contract_title.replace(' ', '_')}_{signed_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(signed_file.getbuffer())
                        signed_file_path = str(file_path)

                    # Create contract
                    contract = Contract(
                        project_id=selected_project_id,
                        title=contract_title,
                        vendor=vendor,
                        contract_type=ContractType[contract_type.upper().replace(" ", "_")],
                        value=contract_value,
                        currency=currency,
                        status=ContractStatus[contract_status.upper().replace(" ", "_")],
                        description=description,
                        terms=terms,
                        start_date=datetime.combine(start_date, datetime.min.time()) if start_date else None,
                        end_date=datetime.combine(end_date, datetime.min.time()) if end_date else None,
                        deliverables=deliverables,
                        payment_schedule=payment_schedule,
                        signed_file=signed_file_path
                    )

                    StateManager.add_contract(contract)

                    st.success(f"Contract '{contract_title}' created successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

    # Tab 3: Upload Template
    with tab3:
        st.markdown("### Upload Contract Template")

        st.info(
            "Upload contract templates that can be reused for multiple projects. "
            "Supported formats: PDF, DOCX, TXT"
        )

        with st.form("upload_template_form"):
            template_name = st.text_input(
                "Template Name *",
                placeholder="e.g., Standard Supply Agreement"
            )

            template_type = st.selectbox(
                "Template Type",
                options=[ct.value for ct in ContractType]
            )

            template_description = st.text_area(
                "Description",
                placeholder="Describe when to use this template...",
                height=100
            )

            template_file = st.file_uploader(
                "Upload Template File *",
                type=["pdf", "docx", "txt"],
                help="Upload the contract template document"
            )

            if st.form_submit_button("Upload Template", type="primary"):
                if template_name and template_file:
                    # Save template file
                    template_dir = upload_dir / "templates"
                    template_dir.mkdir(exist_ok=True)

                    file_path = template_dir / f"{template_name.replace(' ', '_')}_{template_file.name}"

                    with open(file_path, "wb") as f:
                        f.write(template_file.getbuffer())

                    # Save template metadata
                    template_metadata = {
                        "name": template_name,
                        "type": template_type,
                        "description": template_description,
                        "file_path": str(file_path),
                        "uploaded_date": datetime.now().isoformat()
                    }

                    # Store in session state
                    if "contract_templates" not in st.session_state:
                        st.session_state.contract_templates = {}

                    st.session_state.contract_templates[template_name] = template_metadata
                    StateManager.save_all()

                    st.success(f"Template '{template_name}' uploaded successfully!")
                    st.rerun()
                else:
                    st.error("Please provide template name and file")

    # Tab 4: Template Library
    with tab4:
        st.markdown("### Contract Template Library")

        if "contract_templates" in st.session_state and st.session_state.contract_templates:
            for template_name, template_data in st.session_state.contract_templates.items():
                with st.expander(f"📑 {template_name} ({template_data['type']})"):
                    st.markdown(f"**Type:** {template_data['type']}")
                    st.markdown(f"**Uploaded:** {template_data['uploaded_date']}")

                    if template_data.get("description"):
                        st.markdown(f"**Description:** {template_data['description']}")

                    st.markdown(f"**File:** {template_data['file_path']}")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Use Template", key=f"use_{template_name}"):
                            st.info("Template would be loaded into contract creation form")

                    with col2:
                        if st.button("Delete Template", key=f"delete_template_{template_name}"):
                            # Delete file
                            try:
                                os.remove(template_data['file_path'])
                            except:
                                pass

                            # Remove from session state
                            del st.session_state.contract_templates[template_name]
                            StateManager.save_all()

                            st.success("Template deleted")
                            st.rerun()
        else:
            st.info("No templates uploaded yet. Upload your first template in the 'Upload Template' tab.")

            # Provide sample templates info
            st.markdown("### Suggested Template Types")
            st.markdown("""
            - **Supply Agreement**: For equipment and material procurement
            - **Labor Contract**: For installation and construction services
            - **Service Agreement**: For ongoing maintenance and support
            - **Maintenance Contract**: For system upkeep and repairs
            - **Consulting Agreement**: For design and engineering services
            """)
