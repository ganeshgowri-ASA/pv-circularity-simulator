"""
Circularity (3R) Module
=======================

Reduce, Reuse, Recycle analysis for PV systems.
Comprehensive circular economy and sustainability assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the circularity (3R) module.

    Args:
        session: Session manager instance

    Features:
        - REDUCE: Material efficiency and waste minimization
        - REUSE: Component reuse and refurbishment analysis
        - RECYCLE: End-of-life recycling assessment
        - Circular economy scoring
        - Carbon footprint analysis
        - Material flow analysis
        - Sustainability metrics
        - Regulatory compliance (WEEE, RoHS, etc.)
    """
    st.header("♻️ Circularity Analysis (3R)")

    st.info("""
    Comprehensive circular economy analysis: Reduce, Reuse, and Recycle
    strategies for sustainable PV system lifecycle management.
    """)

    # Module data check
    module_design = session.get('module_design_data', {})
    material_data = session.get('material_data', {})

    if not module_design:
        st.warning("⚠️ Please complete Module Design first for full analysis.")

    # Circularity strategy selection
    tab1, tab2, tab3, tab4 = st.tabs([
        "♻️ Overview",
        "🔻 REDUCE",
        "🔄 REUSE",
        "♻️ RECYCLE"
    ])

    # Overview Tab
    with tab1:
        st.subheader("Circular Economy Overview")

        # Current system info
        if module_design:
            num_modules = session.get('system_design_data', {}).get('array', {}).get('num_modules', 1000)
            module_weight = module_design.get('dimensions', {}).get('weight', 22.0)
            total_weight = num_modules * module_weight / 1000  # tonnes

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Modules", f"{num_modules:,}")
            with col2:
                st.metric("Total Weight", f"{total_weight:.1f} tonnes")
            with col3:
                recyclability = 85  # Dummy value
                st.metric("Recyclability Score", f"{recyclability}%")
            with col4:
                carbon_footprint = total_weight * 800  # kg CO2e per tonne
                st.metric("Carbon Footprint", f"{carbon_footprint:,.0f} kg CO₂e")

        # 3R Strategy Summary
        st.markdown("---")
        st.subheader("3R Strategy Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔻 REDUCE")
            st.markdown("**Material Efficiency**")
            st.progress(0.75, text="75% - Good")
            st.caption("Minimize resource consumption and waste")

        with col2:
            st.markdown("### 🔄 REUSE")
            st.markdown("**Reuse Potential**")
            st.progress(0.60, text="60% - Moderate")
            st.caption("Extend component lifespan through reuse")

        with col3:
            st.markdown("### ♻️ RECYCLE")
            st.markdown("**Recycling Rate**")
            st.progress(0.85, text="85% - Excellent")
            st.caption("Material recovery at end-of-life")

        # Circular economy score
        st.markdown("---")
        st.subheader("🎯 Circular Economy Score")

        reduce_score = 75
        reuse_score = 60
        recycle_score = 85
        ce_score = (reduce_score + reuse_score + recycle_score) / 3

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Overall CE Score", f"{ce_score:.0f}/100")

            if ce_score >= 80:
                st.success("✅ Excellent circularity")
            elif ce_score >= 60:
                st.info("ℹ️ Good circularity")
            else:
                st.warning("⚠️ Needs improvement")

        with col2:
            score_breakdown = pd.DataFrame({
                'Category': ['Reduce', 'Reuse', 'Recycle'],
                'Score': [reduce_score, reuse_score, recycle_score]
            })
            st.bar_chart(score_breakdown.set_index('Category'))

    # REDUCE Tab
    with tab2:
        st.subheader("🔻 REDUCE: Material Efficiency & Waste Minimization")

        st.markdown("""
        Strategies to reduce material consumption, improve efficiency,
        and minimize waste during manufacturing and operation.
        """)

        # Material composition
        st.markdown("### Material Composition Analysis")

        # Typical PV module composition
        materials = {
            'Glass': 70,
            'Aluminum (frame)': 10,
            'Silicon (cells)': 3,
            'Encapsulant (EVA/POE)': 7,
            'Backsheet': 5,
            'Junction Box & Cables': 3,
            'Other': 2
        }

        material_df = pd.DataFrame({
            'Material': list(materials.keys()),
            'Weight (%)': list(materials.values())
        })

        col1, col2 = st.columns([2, 1])

        with col1:
            st.bar_chart(material_df.set_index('Material'))

        with col2:
            st.dataframe(material_df, use_container_width=True, hide_index=True)

        # Reduction strategies
        st.markdown("---")
        st.markdown("### Reduction Strategies")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Design Optimization**")

            thinner_glass = st.checkbox("Use thinner glass (2.5mm vs 3.2mm)")
            frameless = st.checkbox("Frameless design")
            reduced_encapsulant = st.checkbox("Optimized encapsulant thickness")
            lightweight_jbox = st.checkbox("Lightweight junction box")

            strategies_count = sum([thinner_glass, frameless, reduced_encapsulant, lightweight_jbox])

        with col2:
            st.markdown("**Material Savings**")

            glass_saving = 22 if thinner_glass else 0
            frame_saving = 10 if frameless else 0
            encap_saving = 15 if reduced_encapsulant else 0
            jbox_saving = 20 if lightweight_jbox else 0

            total_saving = (glass_saving * 0.7 + frame_saving * 0.1 +
                          encap_saving * 0.07 + jbox_saving * 0.03)

            st.metric("Total Weight Reduction", f"{total_saving:.1f}%")
            st.metric("Material Cost Savings", f"${total_saving * 0.5:.2f}/module")

            if module_design:
                co2_reduction = total_weight * 1000 * (total_saving/100) * 800  # kg CO2e
                st.metric("CO₂ Reduction", f"{co2_reduction:,.0f} kg CO₂e")

        # Manufacturing efficiency
        st.markdown("---")
        st.markdown("### Manufacturing Efficiency")

        col1, col2, col3 = st.columns(3)

        with col1:
            yield_rate = st.slider("Manufacturing Yield (%)", 80, 99, 95)
            st.caption("Higher yield = less waste")

        with col2:
            scrap_rate = 100 - yield_rate
            st.metric("Scrap Rate", f"{scrap_rate}%")

        with col3:
            if module_design:
                scrap_modules = num_modules * (scrap_rate / 100)
                st.metric("Scrap Modules", f"{scrap_modules:,.0f}")

    # REUSE Tab
    with tab3:
        st.subheader("🔄 REUSE: Component Reuse & Life Extension")

        st.markdown("""
        Strategies to extend component life through reuse, refurbishment,
        and secondary market applications.
        """)

        # Component reuse assessment
        st.markdown("### Component Reuse Potential")

        components = [
            {
                'Component': 'PV Modules',
                'Condition': 'Good (>80% capacity)',
                'Reuse Potential': 'High',
                'Applications': 'Off-grid, Residential, Repowering',
                'Value Retention': '40-60%'
            },
            {
                'Component': 'Aluminum Frame',
                'Condition': 'Intact',
                'Reuse Potential': 'High',
                'Applications': 'Frame replacement, Recycling',
                'Value Retention': '70-80%'
            },
            {
                'Component': 'Glass',
                'Condition': 'Undamaged',
                'Reuse Potential': 'Medium',
                'Applications': 'Secondary glass, Aggregates',
                'Value Retention': '20-30%'
            },
            {
                'Component': 'Junction Box',
                'Condition': 'Functional',
                'Reuse Potential': 'Medium',
                'Applications': 'Replacement parts',
                'Value Retention': '30-50%'
            },
            {
                'Component': 'Cables',
                'Condition': 'Good',
                'Reuse Potential': 'High',
                'Applications': 'Copper recovery, Reuse',
                'Value Retention': '60-70%'
            }
        ]

        reuse_df = pd.DataFrame(components)
        st.dataframe(reuse_df, use_container_width=True, hide_index=True)

        # Reuse scenario analysis
        st.markdown("---")
        st.markdown("### Reuse Scenario Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Scenario Configuration**")

            system_age = st.slider("System Age (years)", 10, 30, 20)
            avg_degradation = st.slider("Avg. Module Degradation (%)", 5, 40, 15)
            modules_reusable = st.slider("Reusable Modules (%)", 0, 100, 70)

            reuse_application = st.selectbox(
                "Reuse Application",
                ["Off-grid systems", "Residential retrofit", "Agricultural PV",
                 "Developing markets", "Training/Education"]
            )

        with col2:
            st.markdown("**Reuse Value**")

            if module_design:
                original_value = 200  # $/module
                remaining_capacity = 100 - avg_degradation
                reuse_value_per_module = original_value * (remaining_capacity / 100) * 0.5

                total_reusable = num_modules * (modules_reusable / 100)
                total_reuse_value = total_reusable * reuse_value_per_module

                st.metric("Reusable Modules", f"{total_reusable:,.0f}")
                st.metric("Value per Module", f"${reuse_value_per_module:.2f}")
                st.metric("Total Reuse Value", f"${total_reuse_value:,.0f}")

        # Life extension strategies
        st.markdown("---")
        st.markdown("### Life Extension Strategies")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Preventive Maintenance**")
            st.checkbox("Regular cleaning", value=True)
            st.checkbox("Thermal imaging inspection", value=True)
            st.checkbox("IV curve testing", value=False)
            st.caption("Extend life by 2-5 years")

        with col2:
            st.markdown("**Refurbishment**")
            st.checkbox("Bypass diode replacement")
            st.checkbox("Junction box upgrade")
            st.checkbox("Frame repair")
            st.caption("Restore 5-10% performance")

        with col3:
            st.markdown("**Repurposing**")
            st.checkbox("Lower-power applications")
            st.checkbox("BIPV integration")
            st.checkbox("Energy storage pairing")
            st.caption("Second-life applications")

    # RECYCLE Tab
    with tab4:
        st.subheader("♻️ RECYCLE: End-of-Life Material Recovery")

        st.markdown("""
        End-of-life recycling strategies and material recovery analysis
        for PV modules and components.
        """)

        # Recyclability by material
        st.markdown("### Material Recyclability")

        recycling_data = {
            'Material': ['Glass', 'Aluminum', 'Silicon', 'Copper', 'Silver', 'Encapsulant', 'Backsheet'],
            'Weight (%)': [70, 10, 3, 1, 0.05, 7, 5],
            'Recyclable (%)': [95, 100, 85, 95, 90, 30, 20],
            'Recovery Value ($/kg)': [0.05, 1.50, 3.00, 6.00, 500, 0, 0]
        }

        recycle_df = pd.DataFrame(recycling_data)

        # Calculate recovery potential
        recycle_df['Recovery Amount (%)'] = recycle_df['Weight (%)'] * recycle_df['Recyclable (%)'] / 100

        col1, col2 = st.columns([2, 1])

        with col1:
            st.bar_chart(recycle_df.set_index('Material')['Recyclable (%)'])

        with col2:
            overall_recyclability = recycle_df['Recovery Amount (%)'].sum()
            st.metric("Overall Recyclability", f"{overall_recyclability:.1f}%")

            if overall_recyclability >= 80:
                st.success("✅ Highly recyclable")
            elif overall_recyclability >= 60:
                st.info("ℹ️ Moderately recyclable")
            else:
                st.warning("⚠️ Low recyclability")

        st.dataframe(recycle_df, use_container_width=True, hide_index=True)

        # Recycling economics
        st.markdown("---")
        st.markdown("### Recycling Economics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Costs**")

            collection_cost = st.number_input("Collection Cost ($/module)", 0.0, 20.0, 5.0, 0.5)
            transport_cost = st.number_input("Transport Cost ($/module)", 0.0, 10.0, 2.0, 0.5)
            processing_cost = st.number_input("Processing Cost ($/module)", 0.0, 30.0, 15.0, 1.0)

            total_cost_per_module = collection_cost + transport_cost + processing_cost

            st.metric("Total Cost per Module", f"${total_cost_per_module:.2f}")

        with col2:
            st.markdown("**Revenue**")

            if module_design:
                module_weight_kg = module_design.get('dimensions', {}).get('weight', 22.0)

                # Calculate recovery value
                recovery_value = 0
                for _, row in recycle_df.iterrows():
                    material_weight = module_weight_kg * (row['Weight (%)'] / 100)
                    recoverable = material_weight * (row['Recyclable (%)'] / 100)
                    recovery_value += recoverable * row['Recovery Value ($/kg)']

                net_value = recovery_value - total_cost_per_module

                st.metric("Recovery Value", f"${recovery_value:.2f}/module")
                st.metric("Net Value", f"${net_value:.2f}/module")

                if net_value > 0:
                    st.success("✅ Profitable recycling")
                else:
                    st.warning("⚠️ Recycling requires subsidy")

        # Total system recycling analysis
        if module_design:
            st.markdown("---")
            st.markdown("### System-wide Recycling Analysis")

            total_recycling_cost = num_modules * total_cost_per_module
            total_recovery_value = num_modules * recovery_value
            total_net_value = num_modules * net_value

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Modules", f"{num_modules:,}")
            with col2:
                st.metric("Recycling Cost", f"${total_recycling_cost:,.0f}")
            with col3:
                st.metric("Recovery Value", f"${total_recovery_value:,.0f}")
            with col4:
                st.metric("Net Position", f"${total_net_value:,.0f}")

        # Regulatory compliance
        st.markdown("---")
        st.markdown("### Regulatory Compliance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**EU Regulations**")
            st.checkbox("WEEE Directive compliant", value=True)
            st.checkbox("RoHS compliant", value=True)
            st.checkbox("CE marking", value=True)
            st.checkbox("Extended Producer Responsibility", value=True)

        with col2:
            st.markdown("**Other Standards**")
            st.checkbox("ISO 14001 (Environmental)", value=False)
            st.checkbox("Cradle-to-Cradle certified", value=False)
            st.checkbox("PV Cycle membership", value=True)
            st.checkbox("Local recycling requirements", value=True)

    # Save circularity analysis
    if st.button("💾 Save Circularity Analysis"):
        session.set('circularity_data', {
            'ce_score': ce_score if 'ce_score' in locals() else 0,
            'recyclability': overall_recyclability if 'overall_recyclability' in locals() else 0,
            'reuse_value': total_reuse_value if 'total_reuse_value' in locals() else 0,
            'recycling_net_value': total_net_value if 'total_net_value' in locals() else 0
        })
        st.success("Circularity analysis saved!")
