"""
Module Design Module
===================

PV module design and specification.
Defines cell configuration, electrical parameters, and physical dimensions.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the module design module.

    Args:
        session: Session manager instance

    Features:
        - Cell type selection
        - Cell configuration (series/parallel)
        - Electrical parameter calculation
        - Physical dimension specification
        - Module power rating
        - Temperature coefficients
    """
    st.header("⚡ Module Design")

    st.info("Design your PV module configuration and specifications.")

    # Cell configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cell Configuration")
        cell_type = st.selectbox(
            "Cell Technology",
            ["Mono PERC", "Poly", "TOPCon", "HJT", "IBC", "Perovskite"]
        )
        cell_efficiency = st.slider("Cell Efficiency (%)", 15.0, 26.0, 22.5, 0.1)
        num_cells = st.selectbox("Number of Cells", [60, 72, 120, 132, 144])

        cell_config = st.text_input(
            "Cell Configuration (Series x Parallel)",
            value="12x6" if num_cells == 72 else "10x6"
        )

    with col2:
        st.subheader("Module Dimensions")
        module_length = st.number_input("Length (mm)", 1000, 2500, 2000)
        module_width = st.number_input("Width (mm)", 800, 1400, 1000)
        module_weight = st.number_input("Weight (kg)", 10.0, 35.0, 22.0, 0.5)

        cell_size = st.selectbox("Cell Size (mm)", ["156x156", "158x158", "166x166", "182x182", "210x210"])

    # Electrical parameters
    st.markdown("---")
    st.subheader("Electrical Parameters (STC)")

    col1, col2, col3 = st.columns(3)

    with col1:
        pmax = st.number_input("Pmax (W)", 100, 700, 400, 5)
        voc = st.number_input("Voc (V)", 20.0, 60.0, 49.5, 0.1)
        isc = st.number_input("Isc (A)", 5.0, 15.0, 10.5, 0.1)

    with col2:
        vmpp = st.number_input("Vmpp (V)", 20.0, 50.0, 41.2, 0.1)
        impp = st.number_input("Impp (A)", 5.0, 15.0, 9.7, 0.1)
        module_efficiency = st.number_input("Module Efficiency (%)", 10.0, 24.0, 20.0, 0.1)

    with col3:
        st.markdown("**Temperature Coefficients**")
        temp_coef_pmax = st.number_input("Pmax (%/°C)", -0.6, -0.2, -0.35, 0.01)
        temp_coef_voc = st.number_input("Voc (%/°C)", -0.4, -0.2, -0.28, 0.01)
        temp_coef_isc = st.number_input("Isc (%/°C)", 0.0, 0.1, 0.05, 0.01)

    # Performance ratings
    st.markdown("---")
    st.subheader("Performance Ratings")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Module Power", f"{pmax} W")
    with col2:
        st.metric("Efficiency", f"{module_efficiency}%")
    with col3:
        st.metric("Fill Factor", "TBD")
    with col4:
        st.metric("Area", f"{module_length*module_width/1e6:.2f} m²")

    # Save design
    if st.button("Save Module Design"):
        session.set('module_design_data', {
            'cell_type': cell_type,
            'cell_efficiency': cell_efficiency,
            'num_cells': num_cells,
            'cell_config': cell_config,
            'dimensions': {
                'length': module_length,
                'width': module_width,
                'weight': module_weight
            },
            'electrical': {
                'pmax': pmax,
                'voc': voc,
                'isc': isc,
                'vmpp': vmpp,
                'impp': impp,
                'efficiency': module_efficiency
            },
            'temp_coefficients': {
                'pmax': temp_coef_pmax,
                'voc': temp_coef_voc,
                'isc': temp_coef_isc
            }
        })
        st.success("Module design saved!")
