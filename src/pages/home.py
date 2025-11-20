"""Home page for PV Circularity Simulator."""

import streamlit as st


def render() -> None:
    """Render the home page.

    This is the main landing page for the PV Circularity Simulator,
    providing an overview and navigation to main features.
    """
    st.title("🌞 PV Circularity Simulator")

    st.markdown("""
    Welcome to the **PV Circularity Simulator** - an end-to-end photovoltaic lifecycle
    simulation platform with integrated circular economy modeling.

    ## 🎯 Platform Features

    Navigate through our comprehensive suite of tools:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔬 Design & Engineering
        - **Cell Design**: Optimize photovoltaic cell configurations
        - **Module Engineering**: Design and analyze solar modules
        - **System Planning**: Plan complete PV installations
        """)

    with col2:
        st.markdown("""
        ### 📊 Analysis & Monitoring
        - **Performance Monitoring**: Track real-world performance
        - **Circularity (3R)**: Model circular economy practices
        - **CTM Loss Analysis**: Analyze current transport mechanisms
        """)

    st.markdown("---")

    # Quick stats (example)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", "42", "+5")
    with col2:
        st.metric("Active Simulations", "12", "+2")
    with col3:
        st.metric("Modules Analyzed", "1,234", "+87")
    with col4:
        st.metric("CO₂ Reduced (kg)", "5,678", "+234")

    st.markdown("---")

    st.info("👈 Use the sidebar navigation to explore different sections of the platform.")

    # Getting started section
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        1. **Cell Design**: Start by designing your photovoltaic cell configuration
        2. **Module Engineering**: Build and optimize your solar modules
        3. **System Planning**: Plan your complete installation
        4. **Monitor Performance**: Track and analyze real-world data
        5. **Circularity Analysis**: Evaluate circular economy impacts
        """)
