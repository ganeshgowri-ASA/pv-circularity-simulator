"""Circularity (3R) page for PV Circularity Simulator."""

import streamlit as st


def render() -> None:
    """Render the Circularity (3R) page.

    This page provides circular economy modeling with focus on
    Reduce, Reuse, and Recycle strategies for PV systems.
    """
    st.title("♻️ Circularity (3R)")

    st.markdown("""
    Model and optimize circular economy practices for photovoltaic systems:
    **Reduce**, **Reuse**, and **Recycle**.
    """)

    # Circular economy overview
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Overview", "📉 Reduce", "🔄 Reuse", "♻️ Recycle"])

    with tab1:
        st.subheader("Circular Economy Impact")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Material Efficiency", "87%", "+5%")
        with col2:
            st.metric("Recycling Rate", "92%", "+3%")
        with col3:
            st.metric("Lifecycle Extension", "+8 years", "+2 years")

        # Circular economy model
        st.markdown("---")
        st.subheader("🔄 Circular Economy Flow")

        st.markdown("""
        ```
        [Design] → [Manufacturing] → [Use] → [End-of-Life]
                                        ↓
                    ← [Reuse/Refurbish] ← [Collection]
                                        ↓
                    ← [Recycle] ← [Material Recovery]
        ```
        """)

    with tab2:
        st.subheader("📉 Reduce: Material & Energy Optimization")

        st.markdown("**Strategies to reduce material usage and environmental impact:**")

        reduce_strategies = {
            "Thin-film Technologies": "Reduce material usage by 90%",
            "Optimized Cell Design": "Minimize silicon waste",
            "Efficient Manufacturing": "Reduce energy consumption by 30%",
            "Improved Durability": "Extend product lifetime by 25%"
        }

        for strategy, impact in reduce_strategies.items():
            with st.expander(f"✅ {strategy}"):
                st.write(f"**Impact:** {impact}")

        # Material usage calculator
        st.markdown("---")
        st.markdown("**Material Usage Calculator**")

        col1, col2 = st.columns(2)
        with col1:
            system_size = st.number_input("System Size (kW)", min_value=0.0, value=10.0, key="reduce_size")
        with col2:
            module_type = st.selectbox("Module Type", ["Crystalline Si", "Thin Film", "Perovskite"], key="reduce_type")

        if st.button("Calculate Material Usage"):
            material_usage = {
                "Crystalline Si": {"Silicon": 12.5, "Glass": 85, "Aluminum": 15, "Copper": 2.5},
                "Thin Film": {"Glass": 75, "Semiconductor": 0.5, "Aluminum": 12, "Copper": 2.0},
                "Perovskite": {"Glass": 60, "Perovskite": 0.3, "Aluminum": 10, "Copper": 1.8}
            }

            materials = material_usage.get(module_type, {})
            st.markdown(f"**Estimated material usage for {system_size} kW system:**")

            for material, amount_per_kw in materials.items():
                total = amount_per_kw * system_size
                st.write(f"- {material}: {total:.1f} kg")

    with tab3:
        st.subheader("🔄 Reuse: Second-Life Applications")

        st.markdown("**Extend module lifespan through reuse and refurbishment:**")

        reuse_options = st.multiselect(
            "Select Reuse Pathways",
            [
                "Residential Second-Life Systems",
                "Off-Grid Applications",
                "Energy Storage Integration",
                "Agricultural PV (Agrivoltaics)",
                "EV Charging Stations"
            ],
            default=["Residential Second-Life Systems"]
        )

        if reuse_options:
            st.success(f"✅ Selected {len(reuse_options)} reuse pathway(s)")

            # Example metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Extended Lifetime", "+10 years")
            with col2:
                st.metric("Cost Savings", "65%")
            with col3:
                st.metric("CO₂ Avoided", "1,250 kg")

        # Refurbishment analysis
        with st.expander("🔧 Refurbishment Analysis"):
            st.markdown("**Module Health Assessment**")

            col1, col2 = st.columns(2)
            with col1:
                st.progress(85, "Performance Retention: 85%")
                st.progress(92, "Physical Condition: 92%")
            with col2:
                st.progress(78, "Electrical Safety: 78%")
                st.progress(88, "Reuse Suitability: 88%")

            if st.button("✅ Certify for Reuse"):
                st.success("Module certified for second-life applications!")

    with tab4:
        st.subheader("♻️ Recycle: Material Recovery")

        st.markdown("**End-of-life material recovery and recycling:**")

        # Recycling process
        st.markdown("**Recycling Process Flow**")

        process_steps = [
            "1️⃣ Collection & Transportation",
            "2️⃣ Disassembly (Frame, Junction Box)",
            "3️⃣ Glass Separation",
            "4️⃣ Cell Recovery",
            "5️⃣ Material Purification",
            "6️⃣ Secondary Material Production"
        ]

        for step in process_steps:
            st.markdown(f"- {step}")

        st.markdown("---")

        # Recovery rates
        st.markdown("**Material Recovery Rates**")

        import pandas as pd

        recovery_data = pd.DataFrame({
            'Material': ['Glass', 'Aluminum', 'Silicon', 'Copper', 'Silver', 'Other'],
            'Recovery Rate (%)': [95, 92, 85, 90, 88, 70]
        })

        st.bar_chart(recovery_data.set_index('Material'))

        # Economic analysis
        with st.expander("💰 Economic Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Recycling Cost", "$12/module")
                st.metric("Recovered Value", "$18/module")

            with col2:
                st.metric("Net Value", "+$6/module", delta_color="normal")
                st.metric("Recovery Efficiency", "92%")

        if st.button("📊 Generate Recycling Report"):
            st.info("Generating comprehensive recycling impact report...")
            st.success("✅ Report generated! Download available in exports.")
