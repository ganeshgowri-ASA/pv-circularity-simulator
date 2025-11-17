"""
CTM Loss Analysis Module
========================

Cell-to-Module (CTM) loss factor analysis.
Analyzes losses from k1-k15 and k21-k24 factors.
"""

import streamlit as st
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the CTM loss analysis module.

    Args:
        session: Session manager instance

    Features:
        - Individual k-factor adjustment (k1-k15, k21-k24)
        - Loss category analysis (optical, electrical, thermal, etc.)
        - Total CTM ratio calculation
        - Sensitivity analysis
        - Comparison with industry benchmarks
        - Improvement recommendations

    K-factors:
        Optical: k1 (reflection), k2 (shading), k3 (absorption)
        Electrical: k4 (resistive), k5 (mismatch), k6 (junction box)
        Thermal: k7 (temperature), k8 (hotspot)
        Assembly: k9 (encapsulation), k10 (lamination)
        Degradation: k11 (LID), k12 (PID), k13 (mechanical),
                     k14 (cell), k15 (interconnect)
        Environmental: k21 (humidity), k22 (UV), k23 (thermal cycling),
                       k24 (corrosion)
    """
    st.header("📉 CTM Loss Analysis")

    st.info("""
    Analyze Cell-to-Module losses using k-factors (k1-k15, k21-k24).
    Each k-factor represents a loss mechanism, where 1.0 = no loss.
    """)

    # Load CTM loss factors from config
    from core.config import CTM_LOSS_FACTORS

    # Initialize loss factors
    if 'ctm_losses' not in st.session_state:
        st.session_state.ctm_losses = {
            k: v['default'] for k, v in CTM_LOSS_FACTORS.items()
        }

    # Optical losses
    st.subheader("🔆 Optical Losses")
    col1, col2, col3 = st.columns(3)

    with col1:
        k1 = st.slider(
            "k1: Reflection",
            CTM_LOSS_FACTORS['k1_reflection']['min'],
            CTM_LOSS_FACTORS['k1_reflection']['max'],
            st.session_state.ctm_losses.get('k1_reflection', 0.98),
            0.001
        )
    with col2:
        k2 = st.slider(
            "k2: Shading",
            CTM_LOSS_FACTORS['k2_shading']['min'],
            CTM_LOSS_FACTORS['k2_shading']['max'],
            st.session_state.ctm_losses.get('k2_shading', 0.97),
            0.001
        )
    with col3:
        k3 = st.slider(
            "k3: Absorption",
            CTM_LOSS_FACTORS['k3_absorption']['min'],
            CTM_LOSS_FACTORS['k3_absorption']['max'],
            st.session_state.ctm_losses.get('k3_absorption', 0.99),
            0.001
        )

    optical_ratio = k1 * k2 * k3
    st.metric("Optical Transmission", f"{optical_ratio:.4f}", f"{(1-optical_ratio)*100:.2f}% loss")

    # Electrical losses
    st.markdown("---")
    st.subheader("⚡ Electrical Losses")
    col1, col2, col3 = st.columns(3)

    with col1:
        k4 = st.slider("k4: Resistive", 0.95, 0.99, 0.98, 0.001)
    with col2:
        k5 = st.slider("k5: Mismatch", 0.95, 0.99, 0.98, 0.001)
    with col3:
        k6 = st.slider("k6: Junction Box", 0.99, 0.999, 0.995, 0.001)

    electrical_ratio = k4 * k5 * k6
    st.metric("Electrical Efficiency", f"{electrical_ratio:.4f}", f"{(1-electrical_ratio)*100:.2f}% loss")

    # Thermal losses
    st.markdown("---")
    st.subheader("🌡️ Thermal Losses")
    col1, col2 = st.columns(2)

    with col1:
        k7 = st.slider("k7: Temperature", 0.90, 0.98, 0.96, 0.001)
    with col2:
        k8 = st.slider("k8: Hotspot", 0.95, 0.995, 0.99, 0.001)

    thermal_ratio = k7 * k8
    st.metric("Thermal Factor", f"{thermal_ratio:.4f}", f"{(1-thermal_ratio)*100:.2f}% loss")

    # Assembly losses
    st.markdown("---")
    st.subheader("🏭 Assembly Losses")
    col1, col2 = st.columns(2)

    with col1:
        k9 = st.slider("k9: Encapsulation", 0.97, 0.995, 0.99, 0.001)
    with col2:
        k10 = st.slider("k10: Lamination", 0.99, 0.999, 0.995, 0.001)

    assembly_ratio = k9 * k10
    st.metric("Assembly Quality", f"{assembly_ratio:.4f}", f"{(1-assembly_ratio)*100:.2f}% loss")

    # Degradation losses
    st.markdown("---")
    st.subheader("📊 Degradation Factors")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        k11 = st.slider("k11: LID", 0.95, 0.99, 0.98, 0.001)
    with col2:
        k12 = st.slider("k12: PID", 0.95, 0.995, 0.99, 0.001)
    with col3:
        k13 = st.slider("k13: Mechanical", 0.99, 0.999, 0.995, 0.001)
    with col4:
        k14 = st.slider("k14: Cell Deg.", 0.99, 0.999, 0.995, 0.001)
    with col5:
        k15 = st.slider("k15: Interconnect", 0.99, 0.999, 0.995, 0.001)

    degradation_ratio = k11 * k12 * k13 * k14 * k15
    st.metric("Degradation Factor", f"{degradation_ratio:.4f}", f"{(1-degradation_ratio)*100:.2f}% loss")

    # Environmental factors
    st.markdown("---")
    st.subheader("🌍 Environmental Factors")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        k21 = st.slider("k21: Humidity", 0.95, 0.995, 0.99, 0.001)
    with col2:
        k22 = st.slider("k22: UV Exposure", 0.95, 0.995, 0.99, 0.001)
    with col3:
        k23 = st.slider("k23: Thermal Cycling", 0.99, 0.999, 0.995, 0.001)
    with col4:
        k24 = st.slider("k24: Corrosion", 0.99, 0.999, 0.995, 0.001)

    environmental_ratio = k21 * k22 * k23 * k24
    st.metric("Environmental Impact", f"{environmental_ratio:.4f}", f"{(1-environmental_ratio)*100:.2f}% loss")

    # Total CTM ratio
    st.markdown("---")
    st.subheader("📊 Total CTM Ratio")

    total_ctm = (optical_ratio * electrical_ratio * thermal_ratio *
                 assembly_ratio * degradation_ratio * environmental_ratio)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total CTM Ratio", f"{total_ctm:.4f}")
    with col2:
        st.metric("Total Loss", f"{(1-total_ctm)*100:.2f}%")
    with col3:
        st.metric("Industry Benchmark", "0.94 - 0.97")

    # Loss breakdown chart
    st.markdown("---")
    st.subheader("Loss Breakdown")

    loss_data = pd.DataFrame({
        'Category': ['Optical', 'Electrical', 'Thermal', 'Assembly', 'Degradation', 'Environmental'],
        'Loss (%)': [
            (1-optical_ratio)*100,
            (1-electrical_ratio)*100,
            (1-thermal_ratio)*100,
            (1-assembly_ratio)*100,
            (1-degradation_ratio)*100,
            (1-environmental_ratio)*100
        ]
    })

    st.bar_chart(loss_data.set_index('Category'))

    # Save CTM analysis
    if st.button("Save CTM Analysis"):
        session.set('ctm_losses', {
            'k1_reflection': k1, 'k2_shading': k2, 'k3_absorption': k3,
            'k4_resistive': k4, 'k5_mismatch': k5, 'k6_junction_box': k6,
            'k7_temperature': k7, 'k8_hotspot': k8,
            'k9_encapsulation': k9, 'k10_lamination': k10,
            'k11_lid': k11, 'k12_pid': k12, 'k13_mechanical': k13,
            'k14_cell_degradation': k14, 'k15_interconnect': k15,
            'k21_humidity': k21, 'k22_uv_exposure': k22,
            'k23_thermal_cycling': k23, 'k24_corrosion': k24,
            'total_ctm_ratio': total_ctm
        })
        st.success(f"CTM analysis saved! Total CTM ratio: {total_ctm:.4f}")
