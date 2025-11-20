"""
Material Selection Module
=========================

Material selection and comparison for PV modules.
Evaluates materials based on performance, cost, and circularity metrics.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the material selection module.

    Args:
        session: Session manager instance

    Features:
        - Material database browsing
        - Material property comparison
        - Circularity scoring
        - Cost-benefit analysis
        - Custom material definition
    """
    st.header("🔬 Material Selection")

    st.info("Select and compare materials for your PV module design.")

    # Material categories
    st.subheader("Material Categories")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Front Glass",
        "Encapsulant",
        "Backsheet",
        "Frame & Others"
    ])

    with tab1:
        st.markdown("### Front Glass Selection")
        glass_type = st.selectbox(
            "Glass Type",
            ["Low-iron tempered", "Anti-reflective coated", "Self-cleaning"]
        )
        glass_thickness = st.slider("Thickness (mm)", 2.0, 4.0, 3.2, 0.1)
        st.metric("Recyclability Score", "95%")

    with tab2:
        st.markdown("### Encapsulant Selection")
        encapsulant = st.selectbox(
            "Encapsulant Type",
            ["EVA", "POE", "Hybrid EVA/POE"]
        )
        st.metric("Recyclability Score", "45%")

    with tab3:
        st.markdown("### Backsheet Selection")
        backsheet = st.selectbox(
            "Backsheet Type",
            ["TPT", "TPE", "Glass-glass (no backsheet)"]
        )
        st.metric("Recyclability Score", "60%")

    with tab4:
        st.markdown("### Frame and Components")
        frame_material = st.selectbox(
            "Frame Material",
            ["Aluminum", "Steel", "Frameless"]
        )
        junction_box = st.selectbox(
            "Junction Box",
            ["Standard", "Smart", "Integrated"]
        )
        st.metric("Overall Recyclability", "85%")

    # Material comparison
    st.markdown("---")
    st.subheader("Material Impact Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost", "$X.XX/Wp", "TBD")
    with col2:
        st.metric("Carbon Footprint", "XXX kg CO₂e", "TBD")
    with col3:
        st.metric("Recyclability", "XX%", "TBD")
    with col4:
        st.metric("Circularity Score", "XX/100", "TBD")

    # Save selections
    if st.button("Save Material Selections"):
        session.set('material_data', {
            'glass_type': glass_type,
            'glass_thickness': glass_thickness,
            'encapsulant': encapsulant,
            'backsheet': backsheet,
            'frame_material': frame_material,
            'junction_box': junction_box
        })
        st.success("Material selections saved!")
