"""
Materials Selection Module - PV material selection and properties
"""

import streamlit as st


def render():
    """Render the Materials Selection module"""
    st.header("🧪 Materials Selection")
    st.markdown("---")

    st.markdown("""
    ### PV Material Selection and Properties

    Configure and select materials for your photovoltaic cells and modules.
    """)

    tab1, tab2, tab3 = st.tabs(["Cell Materials", "Encapsulation", "Substrate"])

    with tab1:
        st.subheader("Solar Cell Materials")

        material_type = st.selectbox(
            "Cell Technology",
            ["Silicon (c-Si)", "Thin Film (CdTe)", "Thin Film (CIGS)",
             "Perovskite", "Multi-junction", "Organic PV"]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Thickness (μm)", min_value=0.0, value=180.0)
            st.number_input("Efficiency (%)", min_value=0.0, max_value=100.0, value=22.0)

        with col2:
            st.number_input("Band Gap (eV)", min_value=0.0, value=1.12)
            st.number_input("Temperature Coefficient (%/°C)", value=-0.45)

    with tab2:
        st.subheader("Encapsulation Materials")

        encapsulant = st.selectbox(
            "Encapsulant Type",
            ["EVA (Ethylene Vinyl Acetate)", "POE (Polyolefin Elastomer)",
             "PVB (Polyvinyl Butyral)", "Silicone"]
        )

        st.number_input("Encapsulant Thickness (mm)", min_value=0.0, value=0.45)

    with tab3:
        st.subheader("Substrate & Backsheet")

        front_glass = st.selectbox(
            "Front Glass Type",
            ["Low-iron tempered glass", "Anti-reflective coated glass",
             "Textured glass", "Standard glass"]
        )

        backsheet = st.selectbox(
            "Backsheet Type",
            ["TPT (Tedlar-PET-Tedlar)", "TPE (Tedlar-PET-EVA)",
             "Glass-Glass", "Transparent backsheet"]
        )

    st.markdown("---")
    if st.button("💾 Save Material Configuration", use_container_width=True):
        st.success("Material configuration saved successfully!")
