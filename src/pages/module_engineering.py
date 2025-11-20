"""Module Engineering page for PV Circularity Simulator."""

import streamlit as st


def render() -> None:
    """Render the Module Engineering page.

    This page provides tools for solar module design, CTM loss analysis,
    and module performance optimization.
    """
    st.title("⚡ Module Engineering")

    st.markdown("""
    Design and analyze solar modules with comprehensive performance modeling
    and CTM (Cell-to-Module) loss analysis.
    """)

    # Module configuration
    with st.expander("🔧 Module Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            cells_series = st.number_input("Cells in Series", min_value=1, value=60)
        with col2:
            cells_parallel = st.number_input("Cells in Parallel", min_value=1, value=1)
        with col3:
            total_cells = cells_series * cells_parallel
            st.metric("Total Cells", total_cells)

    # CTM Loss Analysis
    st.subheader("📉 CTM Loss Analysis")

    st.markdown("""
    Analyze Cell-to-Module power losses including:
    - Optical losses (reflection, absorption)
    - Resistive losses (interconnects, busbars)
    - Mismatch losses
    - Temperature coefficients
    """)

    loss_types = st.multiselect(
        "Loss Components to Analyze",
        ["Optical Losses", "Resistive Losses", "Mismatch", "Temperature", "LID/LETID"],
        default=["Optical Losses", "Resistive Losses"]
    )

    if st.button("🔍 Analyze CTM Losses", type="primary"):
        with st.spinner("Analyzing CTM losses..."):
            import pandas as pd

            # Example loss data
            loss_data = pd.DataFrame({
                'Loss Type': ['Optical', 'Resistive', 'Mismatch', 'Temperature'],
                'Loss (%)': [2.5, 1.8, 0.7, 1.2]
            })

            col1, col2 = st.columns([2, 1])

            with col1:
                st.bar_chart(loss_data.set_index('Loss Type'))

            with col2:
                st.metric("Total CTM Loss", "6.2%", delta="-0.3%", delta_color="normal")
                st.metric("Module Efficiency", "18.7%", delta="+0.2%", delta_color="normal")

    # Module specifications
    with st.expander("📋 Module Specifications"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Physical Dimensions**")
            st.write("- Length: 1680 mm")
            st.write("- Width: 1000 mm")
            st.write("- Thickness: 35 mm")
            st.write("- Weight: 18.5 kg")

        with col2:
            st.markdown("**Electrical Specifications**")
            st.write("- Pmax: 340 W")
            st.write("- Vmp: 34.2 V")
            st.write("- Imp: 9.94 A")
            st.write("- Efficiency: 18.7%")
