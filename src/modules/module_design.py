"""
Module Design Module - PV module configuration and layout
"""

import streamlit as st


def render():
    """Render the Module Design module"""
    st.header("📐 Module Design")
    st.markdown("---")

    st.markdown("""
    ### PV Module Configuration & Layout

    Design module configuration, cell layout, and electrical connections.
    """)

    tab1, tab2, tab3 = st.tabs(["Layout", "Electrical", "Mechanical"])

    with tab1:
        st.subheader("Module Layout Configuration")

        col1, col2 = st.columns(2)

        with col1:
            cells_series = st.number_input("Cells in Series", min_value=1, value=10)
            cells_parallel = st.number_input("Cells in Parallel", min_value=1, value=6)

            total_cells = cells_series * cells_parallel
            st.metric("Total Cells", total_cells)

        with col2:
            module_config = st.selectbox(
                "Module Configuration",
                ["Standard (60 cells)", "Standard (72 cells)",
                 "Half-cut (120 cells)", "Half-cut (144 cells)", "Custom"]
            )

            st.checkbox("Bypass Diodes (3 per module)", value=True)
            st.number_input("Junction Box Position (from top, mm)", min_value=0, value=800)

    with tab2:
        st.subheader("Electrical Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Module Voc (V)", min_value=0.0, value=40.8, step=0.1)
            st.number_input("Module Isc (A)", min_value=0.0, value=11.5, step=0.1)
            st.number_input("Module Vmp (V)", min_value=0.0, value=34.8, step=0.1)

        with col2:
            st.number_input("Module Imp (A)", min_value=0.0, value=10.8, step=0.1)
            st.number_input("Module Power (W)", min_value=0.0, value=375.0, step=1.0)
            st.number_input("Module Efficiency (%)", min_value=0.0, max_value=100.0, value=20.5)

        st.markdown("### Temperature Coefficients")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Pmax (%/°C)", value=-0.40)
        with col2:
            st.number_input("Voc (%/°C)", value=-0.30)
        with col3:
            st.number_input("Isc (%/°C)", value=0.05)

    with tab3:
        st.subheader("Mechanical Design")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Module Length (mm)", min_value=0, value=1755)
            st.number_input("Module Width (mm)", min_value=0, value=1038)
            st.number_input("Module Thickness (mm)", min_value=0, value=35)

        with col2:
            st.number_input("Module Weight (kg)", min_value=0.0, value=19.5, step=0.1)

            frame_type = st.selectbox(
                "Frame Type",
                ["Anodized Aluminum", "Black Frame", "Frameless", "Custom"]
            )

            st.number_input("Frame Height (mm)", min_value=0, value=35)

        st.markdown("### Certification & Standards")
        cert_col1, cert_col2 = st.columns(2)
        with cert_col1:
            st.checkbox("IEC 61215 (Design qualification)", value=True)
            st.checkbox("IEC 61730 (Safety qualification)", value=True)
        with cert_col2:
            st.checkbox("UL 1703 (Flat-Plate PV)", value=False)
            st.checkbox("CE Marking", value=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 View Module Datasheet", use_container_width=True):
            st.info("Module datasheet generation not yet implemented")
    with col2:
        if st.button("💾 Save Module Design", use_container_width=True):
            st.success("Module design saved successfully!")
