"""
CTM Loss Module - Cell-to-Module loss analysis
"""

import streamlit as st
import pandas as pd


def render():
    """Render the CTM Loss module"""
    st.header("⚡ CTM Loss Analysis")
    st.markdown("---")

    st.markdown("""
    ### Cell-to-Module (CTM) Loss Analysis

    Analyze power losses during the cell-to-module assembly process.
    """)

    st.info("""
    **CTM Ratio** = (Module Power / (Cell Power × Number of Cells)) × 100%

    A higher CTM ratio indicates lower losses during module assembly.
    """)

    tab1, tab2, tab3 = st.tabs(["Loss Factors", "Analysis", "Optimization"])

    with tab1:
        st.subheader("CTM Loss Contributors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Optical Losses")
            optical_reflection = st.slider("Reflection Loss (%)", 0.0, 10.0, 2.5, 0.1)
            optical_absorption = st.slider("Absorption Loss (%)", 0.0, 10.0, 1.8, 0.1)
            optical_shading = st.slider("Shading Loss (Grid/Ribbons) (%)", 0.0, 10.0, 3.2, 0.1)

            total_optical = optical_reflection + optical_absorption + optical_shading
            st.metric("Total Optical Loss", f"{total_optical:.2f}%")

        with col2:
            st.markdown("#### Electrical Losses")
            resistive_loss = st.slider("Resistive Loss (%)", 0.0, 5.0, 1.5, 0.1)
            mismatch_loss = st.slider("Cell Mismatch Loss (%)", 0.0, 5.0, 0.8, 0.1)
            interconnection_loss = st.slider("Interconnection Loss (%)", 0.0, 3.0, 0.6, 0.1)

            total_electrical = resistive_loss + mismatch_loss + interconnection_loss
            st.metric("Total Electrical Loss", f"{total_electrical:.2f}%")

        st.markdown("#### Other Losses")
        thermal_loss = st.slider("Thermal Stress Loss (%)", 0.0, 3.0, 0.5, 0.1)
        degradation_loss = st.slider("Assembly-induced Degradation (%)", 0.0, 3.0, 0.4, 0.1)

        total_loss = total_optical + total_electrical + thermal_loss + degradation_loss
        ctm_ratio = 100 - total_loss

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total CTM Loss", f"{total_loss:.2f}%", delta=f"-{total_loss:.2f}%", delta_color="inverse")
        with col2:
            st.metric("CTM Ratio", f"{ctm_ratio:.2f}%", delta=f"+{ctm_ratio - 90:.2f}%")

    with tab2:
        st.subheader("CTM Analysis Results")

        # Sample data
        cell_power = st.number_input("Individual Cell Power (W)", min_value=0.0, value=5.2, step=0.1)
        num_cells = st.number_input("Number of Cells", min_value=1, value=72)

        theoretical_power = cell_power * num_cells
        actual_power = theoretical_power * (ctm_ratio / 100)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Theoretical Module Power", f"{theoretical_power:.1f} W")
        with col2:
            st.metric("Actual Module Power", f"{actual_power:.1f} W")
        with col3:
            st.metric("Power Loss", f"{theoretical_power - actual_power:.1f} W", delta_color="inverse")

        # Loss breakdown chart
        st.markdown("### Loss Breakdown")
        loss_data = pd.DataFrame({
            'Loss Category': ['Optical', 'Electrical', 'Thermal', 'Degradation'],
            'Loss (%)': [total_optical, total_electrical, thermal_loss, degradation_loss]
        })
        st.bar_chart(loss_data.set_index('Loss Category'))

    with tab3:
        st.subheader("CTM Optimization Recommendations")

        st.markdown("""
        ### Strategies to Improve CTM Ratio:

        #### 1. Optical Improvements
        - Use anti-reflective coatings (ARC) on glass
        - Optimize cell spacing and layout
        - Use thinner ribbons or multi-busbar (MBB) designs

        #### 2. Electrical Improvements
        - Implement cell binning for better matching
        - Optimize ribbon width and thickness
        - Use low-resistance encapsulants

        #### 3. Process Improvements
        - Control lamination temperature and pressure
        - Minimize mechanical stress during assembly
        - Implement quality control at each stage

        #### 4. Material Selection
        - High-transmittance glass
        - Low-resistivity metallization
        - UV-stable encapsulants
        """)

        if st.button("📊 Generate CTM Report", use_container_width=True):
            st.success("CTM analysis report generated successfully!")

    st.markdown("---")
    if st.button("💾 Save CTM Analysis", use_container_width=True):
        st.success("CTM analysis saved successfully!")
