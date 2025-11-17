"""
System Design UI - Main Streamlit Application

This is the main Streamlit application for PV system design, integrating all B05 modules
including dashboard, configurator, visualization, validation, and performance preview.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from typing import Optional

from src.models.pv_components import SystemDesign, ValidationResult
from src.b05_system_design.dashboard.system_design_dashboard import SystemDesignDashboard
from src.b05_system_design.configurator.interactive_configurator import InteractiveConfigurator
from src.b05_system_design.visualization.visualization_3d import Visualization3D
from src.b05_system_design.validation.design_validation import DesignValidation
from src.b05_system_design.performance.performance_preview import PerformancePreview


def initialize_app() -> None:
    """Initialize the Streamlit application."""
    st.set_page_config(
        page_title="PV System Design Studio",
        page_icon="🌞",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #555;
            text-align: center;
            padding-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and controls.

    Returns:
        Selected page name
    """
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=PV+Design+Studio", use_container_width=True)

        st.markdown("---")

        # Page selection
        page = st.radio(
            "Navigation",
            options=[
                "🏠 Home",
                "📍 Site Configuration",
                "🔲 Module Selection",
                "⚡ Inverters",
                "📐 Layout Designer",
                "🏗️ Mounting",
                "🎯 Optimization",
                "✅ Validation",
                "📊 Performance",
                "🎨 3D Visualization",
                "📈 Results & Export"
            ],
            index=0
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### 📊 Quick Stats")

        if st.session_state.get('selected_modules'):
            dc_power = sum(
                config.total_power
                for config in st.session_state.selected_modules
            ) / 1000.0
            st.metric("DC Capacity", f"{dc_power:.1f} kW")

        if st.session_state.get('selected_inverters'):
            ac_power = sum(
                inv.p_ac_rated
                for inv in st.session_state.selected_inverters
            ) / 1000.0
            st.metric("AC Capacity", f"{ac_power:.1f} kW")

        if st.session_state.get('current_design'):
            st.metric("Design Status", "✅ Complete")
        else:
            st.metric("Design Status", "🔄 In Progress")

        st.markdown("---")

        # Design actions
        st.markdown("### 🎯 Actions")

        if st.button("💾 Save Design", use_container_width=True):
            save_design()

        if st.button("📂 Load Design", use_container_width=True):
            load_design()

        if st.button("🔄 New Design", use_container_width=True):
            new_design()

        st.markdown("---")

        # App info
        st.markdown("### ℹ️ About")
        st.caption("PV System Design Studio v1.0")
        st.caption("Comprehensive solar design platform")
        st.caption("© 2025 All rights reserved")

    return page


def render_home_page() -> None:
    """Render the home/welcome page."""
    st.markdown('<div class="main-header">🌞 PV System Design Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Professional Solar System Design Platform</div>',
        unsafe_allow_html=True
    )

    # Welcome message
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        Welcome to the PV System Design Studio! This comprehensive platform integrates
        all aspects of solar system design including:

        ✅ Site configuration and analysis
        ✅ Component selection (modules & inverters)
        ✅ Interactive layout design
        ✅ 3D visualization with sun path
        ✅ NEC compliance validation
        ✅ Performance and financial analysis
        ✅ Professional design reports

        Get started by selecting a page from the sidebar navigation.
        """)

    st.markdown("---")

    # Quick start guide
    st.subheader("🚀 Quick Start Guide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Configure Site")
        st.write("Set up your site location, coordinates, and environmental parameters.")
        if st.button("Go to Site Configuration", key="home_site"):
            st.session_state.nav_page = "📍 Site Configuration"
            st.rerun()

    with col2:
        st.markdown("### 2️⃣ Select Components")
        st.write("Choose PV modules and inverters from our comprehensive database.")
        if st.button("Go to Module Selection", key="home_modules"):
            st.session_state.nav_page = "🔲 Module Selection"
            st.rerun()

    with col3:
        st.markdown("### 3️⃣ Design & Validate")
        st.write("Create your layout, validate against codes, and analyze performance.")
        if st.button("Go to Layout Designer", key="home_layout"):
            st.session_state.nav_page = "📐 Layout Designer"
            st.rerun()

    st.markdown("---")

    # Recent designs
    st.subheader("📁 Recent Designs")

    if st.session_state.get('design_history'):
        for idx, design in enumerate(st.session_state.design_history[-5:]):  # Show last 5
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.write(f"**{design.get('name', 'Unnamed Design')}**")
            with col2:
                st.write(f"{design.get('dc_power', 0):.1f} kW DC")
            with col3:
                st.write(f"{design.get('date', 'N/A')}")
            with col4:
                if st.button("Load", key=f"load_design_{idx}"):
                    st.info("Design loading feature - to be implemented")
    else:
        st.info("No recent designs. Start by creating a new design!")

    st.markdown("---")

    # System features
    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Design Tools:**
        - 🎨 Interactive drag-drop layout designer
        - 🔄 Real-time validation and feedback
        - 📊 Comprehensive component database
        - 🎯 Automatic optimization algorithms
        - 🏗️ Multiple mounting configurations
        """)

    with col2:
        st.markdown("""
        **Analysis & Reporting:**
        - ✅ NEC 2023 compliance checking
        - 📈 Performance ratio calculation
        - 💰 Financial analysis (LCOE, NPV, IRR)
        - 🌅 3D visualization with sun path
        - 📄 Professional PDF/Excel reports
        """)


def save_design() -> None:
    """Save current design."""
    if st.session_state.get('current_design'):
        design = st.session_state.current_design
        if 'design_history' not in st.session_state:
            st.session_state.design_history = []

        st.session_state.design_history.append({
            'name': design.design_name,
            'dc_power': design.total_dc_power,
            'ac_power': design.total_ac_power,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.success("✅ Design saved successfully!")
    else:
        st.warning("⚠️ No design to save. Please complete the design first.")


def load_design() -> None:
    """Load a saved design."""
    st.info("📂 Design loading feature - Select from recent designs on the home page")


def new_design() -> None:
    """Start a new design."""
    # Clear session state
    keys_to_clear = [
        'selected_modules', 'selected_inverters', 'mounting_config',
        'site_location', 'current_design', 'array_layout'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.success("🔄 Started new design!")
    st.rerun()


def main():
    """Main application entry point."""
    # Initialize app
    initialize_app()

    # Initialize component instances
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = SystemDesignDashboard()
        st.session_state.configurator = InteractiveConfigurator()
        st.session_state.visualization = Visualization3D()
        st.session_state.validation = DesignValidation()
        st.session_state.performance = PerformancePreview()

    # Render sidebar and get selected page
    if 'nav_page' in st.session_state:
        page = st.session_state.nav_page
        del st.session_state.nav_page
    else:
        page = render_sidebar()

    # Route to appropriate page
    if page == "🏠 Home":
        render_home_page()

    elif page == "📍 Site Configuration":
        st.title("📍 Site Configuration")
        st.session_state.dashboard._render_site_configuration()

    elif page == "🔲 Module Selection":
        st.title("🔲 PV Module Selection")
        st.session_state.dashboard.module_selection_panel()

    elif page == "⚡ Inverters":
        st.title("⚡ Inverter Configuration")
        st.session_state.dashboard.inverter_configuration()

    elif page == "📐 Layout Designer":
        st.title("📐 Interactive Layout Designer")
        if st.session_state.get('selected_modules'):
            st.session_state.configurator.drag_drop_layout(
                st.session_state.selected_modules[0]
            )

            # Auto-optimization suggestions
            st.markdown("---")
            st.subheader("🎯 Optimization Suggestions")
            suggestions = st.session_state.configurator.auto_optimization_trigger()

            if suggestions['triggered'] and suggestions['optimizations']:
                for opt in suggestions['optimizations']:
                    st.info(f"""
                    **{opt['type'].replace('_', ' ').title()}:**
                    Current: {opt['current']:.2f} → Suggested: {opt['suggested']:.2f}
                    Reason: {opt['reason']}
                    Estimated Gain: {opt['estimated_gain']}
                    """)
            else:
                st.success("✅ Design is well optimized!")
        else:
            st.warning("⚠️ Please select modules first (Module Selection page)")

    elif page == "🏗️ Mounting":
        st.title("🏗️ Mounting Structure Configuration")
        st.session_state.dashboard.mounting_structure_selector()

    elif page == "🎯 Optimization":
        st.title("🎯 System Optimization")
        st.session_state.dashboard.optimization_controls()

    elif page == "✅ Validation":
        st.title("✅ Design Validation")

        if not st.session_state.get('site_location'):
            st.warning("⚠️ Please configure site location first")
            return

        if not st.session_state.get('selected_modules'):
            st.warning("⚠️ Please add module configurations first")
            return

        if not st.session_state.get('selected_inverters'):
            st.warning("⚠️ Please add inverters first")
            return

        # Create temporary design for validation
        design = SystemDesign(
            design_id=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            design_name="Current Design",
            site=st.session_state.site_location,
            modules=st.session_state.selected_modules,
            inverters=st.session_state.selected_inverters,
            mounting=st.session_state.get('mounting_config'),
            dc_ac_ratio=0.0  # Will be calculated
        )

        # Run validation
        with st.spinner("Running comprehensive validation..."):
            validation_result = st.session_state.validation.validate_complete_design(design)

        # Display results
        if validation_result.is_valid:
            st.success("✅ Design passed all validation checks!")
        else:
            st.error(f"❌ Design has {validation_result.error_count} error(s)")

        # Show errors
        if validation_result.errors:
            st.markdown("### ❌ Errors")
            for error in validation_result.errors:
                st.error(error)

        # Show warnings
        if validation_result.warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in validation_result.warnings:
                st.warning(warning)

        # Show checks performed
        with st.expander("✓ Validation Checks Performed"):
            for check in validation_result.checks_performed:
                st.write(f"✓ {check}")

        # Detailed validation report
        if st.button("📄 Generate Detailed Validation Report"):
            report = st.session_state.validation.generate_validation_report()

            st.markdown("### 📊 Validation Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Issues", report['total_issues'])
            with col2:
                st.metric("Errors", len(report['errors']))
            with col3:
                st.metric("Warnings", len(report['warnings']))

            # NEC references
            if report['nec_references']:
                st.markdown("### 📖 NEC Code References")
                for ref in report['nec_references']:
                    st.write(f"- {ref}")

    elif page == "📊 Performance":
        st.title("📊 Performance Analysis")

        if not st.session_state.get('site_location'):
            st.warning("⚠️ Please configure site location first")
            return

        if not st.session_state.get('selected_modules') or not st.session_state.get('selected_inverters'):
            st.warning("⚠️ Please configure modules and inverters first")
            return

        # Create design
        design = SystemDesign(
            design_id=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            design_name="Current Design",
            site=st.session_state.site_location,
            modules=st.session_state.selected_modules,
            inverters=st.session_state.selected_inverters,
            mounting=st.session_state.get('mounting_config'),
            dc_ac_ratio=0.0
        )

        # Calculate performance metrics
        with st.spinner("Calculating performance metrics..."):
            energy = st.session_state.performance.annual_energy_estimate(design, design.site)
            pr = st.session_state.performance.pr_calculation(design, design.site)
            shading = st.session_state.performance.shading_loss_summary(design)
            financial = st.session_state.performance.financial_preview(design, design.site)

        # Display metrics
        st.markdown("### ⚡ Energy Production")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Energy", f"{energy.annual_energy:,.0f} kWh")
        with col2:
            st.metric("Specific Yield", f"{energy.specific_yield:.0f} kWh/kWp")
        with col3:
            st.metric("Capacity Factor", f"{energy.capacity_factor:.1f}%")
        with col4:
            st.metric("Performance Ratio", f"{pr.pr_value:.3f}")

        # Monthly production chart
        st.plotly_chart(
            st.session_state.performance.plot_monthly_production(energy),
            use_container_width=True
        )

        st.markdown("---")

        # Financial metrics
        st.markdown("### 💰 Financial Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"${financial.total_cost:,.0f}")
        with col2:
            st.metric("LCOE", f"${financial.lcoe:.3f}/kWh")
        with col3:
            st.metric("Payback Period", f"{financial.payback_period:.1f} years")
        with col4:
            st.metric("NPV", f"${financial.npv:,.0f}")

        # Loss waterfall
        st.markdown("### 📉 Loss Analysis")
        st.plotly_chart(
            st.session_state.performance.plot_loss_waterfall(pr),
            use_container_width=True
        )

    elif page == "🎨 3D Visualization":
        st.title("🎨 3D System Visualization")

        # Camera controls in sidebar
        st.session_state.visualization.interactive_camera_controls()

        if st.session_state.get('array_layout') and st.session_state.array_layout.modules:
            # Render 3D visualization
            fig = st.session_state.visualization.render_system_3d(
                layout=st.session_state.array_layout,
                mounting=st.session_state.get('mounting_config'),
                site=st.session_state.get('site_location')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please create a layout first (Layout Designer page)")

    elif page == "📈 Results & Export":
        st.title("📈 Design Results & Export")

        if not all([
            st.session_state.get('site_location'),
            st.session_state.get('selected_modules'),
            st.session_state.get('selected_inverters')
        ]):
            st.warning("⚠️ Please complete the design configuration first")
            return

        # Create final design
        design = SystemDesign(
            design_id=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            design_name=st.text_input("Design Name", value="My PV System Design"),
            site=st.session_state.site_location,
            modules=st.session_state.selected_modules,
            inverters=st.session_state.selected_inverters,
            mounting=st.session_state.get('mounting_config'),
            dc_ac_ratio=0.0,
            designer=st.text_input("Designer Name", value=""),
            notes=st.text_area("Design Notes", value="")
        )

        st.session_state.current_design = design

        # Summary
        st.markdown("### 📋 Design Summary")
        st.session_state.dashboard._render_design_summary()

        st.markdown("---")

        # Export options
        st.markdown("### 📥 Export Design")

        col1, col2, col3 = st.columns(3)

        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["Excel", "JSON", "PDF"],
                index=0
            )

        with col2:
            include_3d = st.checkbox("Include 3D Visualization", value=True)

        with col3:
            include_validation = st.checkbox("Include Validation Report", value=True)

        if st.button("📄 Generate Export", type="primary"):
            with st.spinner("Generating export..."):
                try:
                    report_buffer = st.session_state.performance.export_design_report(
                        design=design,
                        site=design.site,
                        format=export_format.lower()
                    )

                    # Provide download
                    file_ext = "xlsx" if export_format == "Excel" else export_format.lower()
                    st.download_button(
                        label=f"⬇️ Download {export_format} Report",
                        data=report_buffer,
                        file_name=f"{design.design_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{file_ext}",
                        mime=f"application/{file_ext}"
                    )

                    st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")


if __name__ == "__main__":
    main()
