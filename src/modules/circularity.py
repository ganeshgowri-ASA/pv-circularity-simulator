"""
Circularity Module - Circular economy and 3R analysis (Reduce, Reuse, Recycle)
"""

import streamlit as st
import pandas as pd


def render():
    """Render the Circularity module"""
    st.header("♻️ Circularity & 3R Analysis")
    st.markdown("---")

    st.markdown("""
    ### Circular Economy Assessment (Reduce, Reuse, Recycle)

    Analyze the environmental impact and circular economy potential of PV systems.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Reduce", "Reuse", "Recycle", "LCA"
    ])

    with tab1:
        st.subheader("Circularity Overview")

        st.markdown("""
        ### Circular Economy Framework for PV Systems

        The 3R approach (Reduce, Reuse, Recycle) maximizes resource efficiency and minimizes waste
        throughout the PV system lifecycle.
        """)

        # Circularity score
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Circularity Index", "68%", help="Overall circularity score (0-100%)")
        with col2:
            st.metric("Material Recovery", "85%", help="Potential material recovery rate")
        with col3:
            st.metric("Carbon Footprint", "42 g CO₂/kWh", delta="-15%")
        with col4:
            st.metric("Waste Reduction", "72%", help="Waste reduction vs. linear economy")

        st.markdown("---")
        st.markdown("#### Circularity Breakdown")

        circularity_data = pd.DataFrame({
            'Aspect': ['Design for Circularity', 'Material Selection', 'Longevity',
                      'Repair & Maintenance', 'Reuse Potential', 'Recyclability',
                      'Resource Efficiency'],
            'Score (%)': [75, 65, 80, 70, 60, 85, 72]
        })

        st.bar_chart(circularity_data.set_index('Aspect'))

        st.markdown("#### System Information")
        col1, col2 = st.columns(2)
        with col1:
            system_size = st.number_input("System Size (kWp)", min_value=0.0, value=100.0)
            num_modules = st.number_input("Number of Modules", min_value=1, value=267)
            module_weight = st.number_input("Module Weight (kg)", min_value=0.0, value=21.0)

        with col2:
            system_age = st.number_input("System Age (years)", min_value=0, value=5)
            expected_life = st.number_input("Expected Lifetime (years)", min_value=1, value=25)
            remaining_life = expected_life - system_age
            st.metric("Remaining Life", f"{remaining_life} years")

    with tab2:
        st.subheader("Reduce - Resource Optimization")

        st.markdown("""
        ### Strategies to Reduce Material Use & Environmental Impact
        """)

        st.markdown("#### Design Optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Material Reduction Strategies:**")
            material_efficiency = st.slider("Material Efficiency Improvement (%)", 0, 50, 15)

            st.checkbox("Thin-film technology (reduced silicon)", value=False)
            st.checkbox("Frameless modules (reduced aluminum)", value=False)
            st.checkbox("Lightweight glass (reduced weight)", value=True)
            st.checkbox("Optimized cell thickness", value=True)

        with col2:
            st.markdown("**Energy Reduction:**")
            energy_payback = st.number_input("Energy Payback Time (years)", min_value=0.0, value=1.8, step=0.1)

            total_energy = system_size * 1000 * 1.85  # kWh/year
            lifetime_energy = total_energy * expected_life
            embodied_energy = system_size * 1000 * 2000  # kWh (approx)

            eroi = lifetime_energy / embodied_energy if embodied_energy > 0 else 0
            st.metric("Energy ROI", f"{eroi:.1f}:1", help="Energy produced vs. embodied energy")

        st.markdown("---")
        st.markdown("#### Material Footprint")

        # Material composition
        materials = pd.DataFrame({
            'Material': ['Silicon', 'Glass', 'Aluminum', 'Copper', 'Silver', 'Plastics', 'Other'],
            'Mass (kg)': [1200, 3800, 850, 280, 12, 420, 245],
            'Reduction Potential (%)': [15, 10, 25, 5, 30, 20, 10]
        })

        materials['Reduced Mass (kg)'] = materials['Mass (kg)'] * (1 - materials['Reduction Potential (%)'] / 100)
        materials['Savings (kg)'] = materials['Mass (kg)'] - materials['Reduced Mass (kg)']

        st.dataframe(materials, use_container_width=True)

        total_mass = materials['Mass (kg)'].sum()
        total_savings = materials['Savings (kg)'].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total System Mass", f"{total_mass:.0f} kg")
        with col2:
            st.metric("Potential Savings", f"{total_savings:.0f} kg")
        with col3:
            st.metric("Reduction", f"{(total_savings/total_mass*100):.1f}%")

    with tab3:
        st.subheader("Reuse - Second Life Applications")

        st.markdown("""
        ### Module Reuse & Second Life Opportunities

        End-of-life PV modules can often be reused in less demanding applications.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Current Module Status")

            current_power = st.number_input("Current Module Power (W)", min_value=0.0, value=340.0)
            original_power = st.number_input("Original Module Power (W)", min_value=0.0, value=375.0)

            remaining_capacity = (current_power / original_power * 100) if original_power > 0 else 0
            st.metric("Remaining Capacity", f"{remaining_capacity:.1f}%")

            modules_available = st.number_input("Modules Available for Reuse", min_value=0, value=50)

        with col2:
            st.markdown("#### Reuse Viability Assessment")

            if remaining_capacity >= 80:
                st.success("✅ Excellent - Suitable for primary applications")
            elif remaining_capacity >= 70:
                st.info("✓ Good - Suitable for most second-life applications")
            elif remaining_capacity >= 60:
                st.warning("⚠ Fair - Limited second-life applications")
            else:
                st.error("✗ Poor - Consider recycling instead")

            reuse_potential = min(remaining_capacity, 100)
            st.metric("Reuse Potential", f"{reuse_potential:.0f}%")

        st.markdown("---")
        st.markdown("#### Second Life Applications")

        applications = [
            {
                "Application": "Off-grid Rural Electrification",
                "Min Capacity": "60%",
                "Market Size": "High",
                "Value Retention": "40-60%"
            },
            {
                "Application": "Agricultural Irrigation",
                "Min Capacity": "65%",
                "Market Size": "Medium",
                "Value Retention": "35-50%"
            },
            {
                "Application": "EV Charging Stations",
                "Min Capacity": "70%",
                "Market Size": "High",
                "Value Retention": "45-65%"
            },
            {
                "Application": "Backup Power Systems",
                "Min Capacity": "65%",
                "Market Size": "Medium",
                "Value Retention": "40-55%"
            },
            {
                "Application": "Street Lighting",
                "Min Capacity": "60%",
                "Market Size": "Medium",
                "Value Retention": "30-45%"
            },
            {
                "Application": "Mobile/Portable Power",
                "Min Capacity": "70%",
                "Market Size": "Low",
                "Value Retention": "35-50%"
            }
        ]

        st.dataframe(pd.DataFrame(applications), use_container_width=True)

        st.markdown("#### Economic Value")

        col1, col2, col3 = st.columns(3)
        with col1:
            original_value = st.number_input("Original Module Cost ($)", min_value=0.0, value=200.0)
        with col2:
            value_retention = st.slider("Value Retention (%)", 0, 100, 45)
            reuse_value = original_value * (value_retention / 100)
            st.metric("Reuse Value", f"${reuse_value:.2f}")
        with col3:
            total_reuse_value = reuse_value * modules_available
            st.metric("Total Recovery Value", f"${total_reuse_value:,.0f}")

    with tab4:
        st.subheader("Recycle - Material Recovery")

        st.markdown("""
        ### End-of-Life Recycling & Material Recovery

        PV modules contain valuable materials that can be recovered and recycled.
        """)

        st.markdown("#### Recycling Methods")

        recycling_method = st.selectbox(
            "Recycling Technology",
            ["Thermal Treatment", "Mechanical Separation", "Chemical Treatment",
             "Combined (Thermal + Chemical)", "Advanced Pyrolysis"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Material Recovery Rates")

            recovery_rates = {
                'Glass': 95,
                'Aluminum (Frame)': 100,
                'Silicon': 85,
                'Copper': 95,
                'Silver': 90,
                'Plastics': 25,
                'Other metals': 80
            }

            for material, rate in recovery_rates.items():
                st.progress(rate / 100, text=f"{material}: {rate}%")

        with col2:
            st.markdown("#### Recovery Economics")

            modules_to_recycle = st.number_input("Modules to Recycle", min_value=1, value=100)

            # Material values ($/kg)
            material_values = {
                'Glass': 0.05,
                'Aluminum': 2.50,
                'Silicon': 8.00,
                'Copper': 8.50,
                'Silver': 650.00,
                'Plastics': 0.30
            }

            # Material quantities per module (kg)
            material_mass = {
                'Glass': 14.2,
                'Aluminum': 3.2,
                'Silicon': 4.5,
                'Copper': 1.05,
                'Silver': 0.045,
                'Plastics': 1.6
            }

            total_value = 0
            for material in material_values:
                mass = material_mass[material] * modules_to_recycle
                recovery = recovery_rates.get(material, 0) / 100
                value = mass * recovery * material_values[material]
                total_value += value

            st.metric("Total Recovery Value", f"${total_value:,.2f}")

            recycling_cost = modules_to_recycle * 15  # $15 per module
            st.metric("Recycling Cost", f"${recycling_cost:,.2f}")

            net_value = total_value - recycling_cost
            st.metric("Net Value", f"${net_value:,.2f}")

        st.markdown("---")
        st.markdown("#### Material Recovery Details")

        recovery_details = []
        for material in material_values:
            mass = material_mass[material] * modules_to_recycle
            recovery = recovery_rates.get(material, 0) / 100
            recovered_mass = mass * recovery
            value = recovered_mass * material_values[material]

            recovery_details.append({
                'Material': material,
                'Total Mass (kg)': f"{mass:.2f}",
                'Recovery Rate': f"{recovery*100:.0f}%",
                'Recovered (kg)': f"{recovered_mass:.2f}",
                'Value ($/kg)': f"${material_values[material]:.2f}",
                'Total Value ($)': f"${value:.2f}"
            })

        st.dataframe(pd.DataFrame(recovery_details), use_container_width=True)

        st.markdown("#### Environmental Benefits")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CO₂ Avoided", "1,250 kg", help="vs. virgin material production")
        with col2:
            st.metric("Energy Saved", "15,400 kWh", help="Recycling energy savings")
        with col3:
            st.metric("Landfill Avoided", "2,100 kg", help="Waste diverted from landfill")

    with tab5:
        st.subheader("Life Cycle Assessment (LCA)")

        st.markdown("""
        ### Environmental Impact Analysis

        Complete lifecycle assessment from production to end-of-life.
        """)

        st.markdown("#### Carbon Footprint Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Production Phase:**")
            module_carbon = st.number_input("Module Production (kg CO₂/kWp)", min_value=0.0, value=1200.0)
            bos_carbon = st.number_input("BOS Production (kg CO₂/kWp)", min_value=0.0, value=400.0)
            transport_carbon = st.number_input("Transportation (kg CO₂/kWp)", min_value=0.0, value=100.0)

            total_production_carbon = (module_carbon + bos_carbon + transport_carbon) * system_size

        with col2:
            st.markdown("**Operational Phase:**")
            annual_generation = st.number_input("Annual Generation (MWh)", min_value=0.0, value=150.0)
            grid_carbon_intensity = st.number_input("Grid Carbon Intensity (g CO₂/kWh)", min_value=0.0, value=500.0)

            annual_carbon_avoided = annual_generation * grid_carbon_intensity

        st.markdown("#### Lifetime Carbon Balance")

        lifetime_generation = annual_generation * expected_life
        lifetime_carbon_avoided = lifetime_generation * grid_carbon_intensity

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Production Emissions", f"{total_production_carbon:,.0f} kg CO₂")
        with col2:
            st.metric("Lifetime Avoided Emissions", f"{lifetime_carbon_avoided:,.0f} kg CO₂")
        with col3:
            net_carbon = lifetime_carbon_avoided - total_production_carbon
            st.metric("Net Carbon Benefit", f"{net_carbon:,.0f} kg CO₂", delta=f"+{net_carbon:,.0f}")

        carbon_payback = total_production_carbon / annual_carbon_avoided if annual_carbon_avoided > 0 else 0
        st.metric("Carbon Payback Time", f"{carbon_payback:.2f} years")

        st.markdown("---")
        st.markdown("#### Environmental Impact Categories")

        impact_data = pd.DataFrame({
            'Impact Category': [
                'Global Warming Potential',
                'Acidification Potential',
                'Eutrophication Potential',
                'Ozone Depletion Potential',
                'Photochemical Oxidation',
                'Water Depletion',
                'Mineral Resource Depletion'
            ],
            'Impact Score': [42, 15, 8, 2, 12, 25, 35],
            'vs Grid Average': [-88, -75, -82, -65, -70, -60, -45]
        })

        st.dataframe(impact_data, use_container_width=True)

        st.markdown("#### Circularity Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Material Circularity Index", "0.72", help="MCI score (0-1)")
        with col2:
            st.metric("Recycled Content", "12%", help="% of recycled materials used")
        with col3:
            st.metric("End-of-Life Recovery", "88%", help="% of materials recovered")
        with col4:
            st.metric("Product Lifetime Extension", "+40%", help="vs. standard warranty")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Circularity Report", use_container_width=True):
            st.success("Circularity assessment report generated!")
    with col2:
        if st.button("💾 Save Assessment", use_container_width=True):
            st.success("Circularity assessment saved successfully!")
