"""
Materials Selection UI Components for PV Circularity Simulator.

This module provides comprehensive Streamlit UI components for material selection,
comparison, and analysis for photovoltaic manufacturing applications.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.material_loader import (
    MaterialLoader, Material, MaterialCategory, MaterialProperties,
    CircularityMetrics, SupplierInfo, Standard
)
from api.enf_client import ENFAPIClient, ENFSupplier, PriceQuote, MarketPrice


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state() -> None:
    """Initialize Streamlit session state for materials selection."""
    if 'material_loader' not in st.session_state:
        st.session_state.material_loader = MaterialLoader()

    if 'enf_client' not in st.session_state:
        st.session_state.enf_client = ENFAPIClient()

    if 'selected_materials' not in st.session_state:
        st.session_state.selected_materials = []

    if 'favorite_materials' not in st.session_state:
        st.session_state.favorite_materials = set()

    if 'bom_materials' not in st.session_state:
        st.session_state.bom_materials = []

    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False

    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "grid"  # grid or list

    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    if 'active_category' not in st.session_state:
        st.session_state.active_category = None

    if 'quick_filters' not in st.session_state:
        st.session_state.quick_filters = {
            'high_efficiency': False,
            'low_cost': False,
            'high_sustainability': False,
            'premium_only': False
        }


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css() -> None:
    """Inject custom CSS for enhanced styling."""
    st.markdown("""
        <style>
        /* Material Card Styling */
        .material-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .material-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }

        .material-card-selected {
            border: 2px solid #1f77b4;
            background: #f0f8ff;
        }

        .material-card-favorite {
            border-color: #ff6b6b;
        }

        /* Category Tabs */
        .category-tab {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 20px;
            background: #f5f5f5;
            cursor: pointer;
            transition: all 0.2s;
        }

        .category-tab:hover {
            background: #e0e0e0;
        }

        .category-tab-active {
            background: #1f77b4;
            color: white;
        }

        /* Metrics Display */
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 5px;
            text-align: center;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
        }

        .metric-label {
            font-size: 12px;
            opacity: 0.9;
        }

        /* Circularity Score Badge */
        .circularity-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 12px;
        }

        .circularity-high {
            background: #4caf50;
            color: white;
        }

        .circularity-medium {
            background: #ff9800;
            color: white;
        }

        .circularity-low {
            background: #f44336;
            color: white;
        }

        /* Price Tag */
        .price-tag {
            font-size: 20px;
            font-weight: bold;
            color: #2e7d32;
        }

        /* Supplier Rating */
        .supplier-rating {
            color: #ffa000;
            font-size: 14px;
        }

        /* Button Styling */
        .action-button {
            background: #1f77b4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 2px;
            transition: all 0.2s;
        }

        .action-button:hover {
            background: #1557a0;
        }

        /* Search Bar */
        .search-container {
            margin: 20px 0;
        }

        /* Comparison Table */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }

        .comparison-table th {
            background: #1f77b4;
            color: white;
            padding: 10px;
            text-align: left;
        }

        .comparison-table td {
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }

        .comparison-table tr:hover {
            background: #f5f5f5;
        }

        /* Loading Spinner */
        .loading-spinner {
            text-align: center;
            padding: 20px;
            color: #1f77b4;
        }

        /* Alert Boxes */
        .alert-info {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        .alert-success {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        .alert-warning {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        /* Responsive Grid */
        .material-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 10px;
        }

        @media (max-width: 768px) {
            .material-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_circularity_class(score: float) -> str:
    """
    Get CSS class for circularity score.

    Args:
        score: Circularity score (0-100)

    Returns:
        CSS class name
    """
    if score >= 70:
        return "circularity-high"
    elif score >= 40:
        return "circularity-medium"
    else:
        return "circularity-low"


def format_price(price: float, currency: str = "USD") -> str:
    """
    Format price with currency symbol.

    Args:
        price: Price value
        currency: Currency code

    Returns:
        Formatted price string
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "CNY": "¥"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{price:,.2f}"


def calculate_total_cost(material: Material, quantity_kg: float) -> float:
    """
    Calculate total cost for a given quantity.

    Args:
        material: Material object
        quantity_kg: Quantity in kg

    Returns:
        Total cost in USD
    """
    return material.base_price_per_kg * quantity_kg


def export_to_json(data: Any, filename: str) -> bytes:
    """
    Export data to JSON format.

    Args:
        data: Data to export
        filename: Output filename

    Returns:
        JSON bytes
    """
    json_str = json.dumps(data, indent=2, default=str)
    return json_str.encode()


def export_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to CSV.

    Args:
        df: DataFrame to export

    Returns:
        CSV bytes
    """
    return df.to_csv(index=False).encode()


def create_download_link(data: bytes, filename: str, link_text: str) -> str:
    """
    Create download link for data.

    Args:
        data: Binary data
        filename: Download filename
        link_text: Link display text

    Returns:
        HTML download link
    """
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'


# ============================================================================
# MATERIAL SELECTION INTERFACE
# ============================================================================

def render_category_tabs() -> Optional[MaterialCategory]:
    """
    Render category selection tabs.

    Returns:
        Selected MaterialCategory or None for all categories
    """
    st.markdown("### Material Categories")

    # Create columns for category buttons
    categories = [None] + list(MaterialCategory)
    category_names = ["All Materials"] + [cat.value for cat in MaterialCategory if cat]

    cols = st.columns(len(categories))

    selected_category = st.session_state.active_category

    for idx, (col, category, name) in enumerate(zip(cols, categories, category_names)):
        with col:
            if st.button(
                name,
                key=f"cat_{idx}",
                use_container_width=True,
                type="primary" if category == selected_category else "secondary"
            ):
                st.session_state.active_category = category
                st.rerun()

    return st.session_state.active_category


def render_search_and_filters() -> Tuple[str, Dict[str, bool], Dict[str, any]]:
    """
    Render search bar and filter controls.

    Returns:
        Tuple of (search_query, quick_filters, advanced_filters)
    """
    st.markdown("### Search & Filters")

    # Search bar with autocomplete simulation
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search materials",
            value=st.session_state.search_query,
            placeholder="Search by name, description, or tags...",
            key="search_input"
        )
        st.session_state.search_query = search_query

    with col2:
        view_mode = st.selectbox(
            "View",
            ["Grid", "List"],
            index=0 if st.session_state.view_mode == "grid" else 1,
            key="view_selector"
        )
        st.session_state.view_mode = view_mode.lower()

    # Quick filters
    st.markdown("#### Quick Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_eff = st.checkbox(
            "High Efficiency",
            value=st.session_state.quick_filters['high_efficiency'],
            key="filter_efficiency"
        )
        st.session_state.quick_filters['high_efficiency'] = high_eff

    with col2:
        low_cost = st.checkbox(
            "Cost Effective",
            value=st.session_state.quick_filters['low_cost'],
            key="filter_cost"
        )
        st.session_state.quick_filters['low_cost'] = low_cost

    with col3:
        high_sust = st.checkbox(
            "Sustainable",
            value=st.session_state.quick_filters['high_sustainability'],
            key="filter_sustainability"
        )
        st.session_state.quick_filters['high_sustainability'] = high_sust

    with col4:
        premium = st.checkbox(
            "Premium Only",
            value=st.session_state.quick_filters['premium_only'],
            key="filter_premium"
        )
        st.session_state.quick_filters['premium_only'] = premium

    # Advanced filters (collapsible)
    with st.expander("Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            price_range = st.slider(
                "Price Range ($/kg)",
                0.0, 1000.0, (0.0, 1000.0),
                step=10.0,
                key="price_slider"
            )

        with col2:
            circularity_min = st.slider(
                "Min Circularity Score",
                0, 100, 0,
                key="circularity_slider"
            )

        with col3:
            efficiency_min = st.slider(
                "Min Efficiency Impact (%)",
                0.0, 25.0, 0.0,
                step=0.5,
                key="efficiency_slider"
            )

    advanced_filters = {
        'price_range': price_range,
        'circularity_min': circularity_min,
        'efficiency_min': efficiency_min
    }

    return search_query, st.session_state.quick_filters, advanced_filters


def apply_filters(
    materials: List[Material],
    search_query: str,
    quick_filters: Dict[str, bool],
    advanced_filters: Dict[str, any]
) -> List[Material]:
    """
    Apply all filters to material list.

    Args:
        materials: List of materials to filter
        search_query: Search text
        quick_filters: Quick filter selections
        advanced_filters: Advanced filter values

    Returns:
        Filtered list of materials
    """
    filtered = materials

    # Search filter
    if search_query:
        query_lower = search_query.lower()
        filtered = [
            m for m in filtered
            if query_lower in m.name.lower() or
               query_lower in m.description.lower() or
               any(query_lower in tag.lower() for tag in m.tags)
        ]

    # Quick filters
    if quick_filters['high_efficiency']:
        filtered = [
            m for m in filtered
            if m.pv_efficiency_impact and m.pv_efficiency_impact >= 18.0
        ]

    if quick_filters['low_cost']:
        filtered = [m for m in filtered if m.base_price_per_kg <= 20.0]

    if quick_filters['high_sustainability']:
        filtered = [
            m for m in filtered
            if m.circularity.recyclability_score >= 70.0
        ]

    if quick_filters['premium_only']:
        filtered = [m for m in filtered if m.quality_grade == "Premium"]

    # Advanced filters
    price_min, price_max = advanced_filters['price_range']
    filtered = [
        m for m in filtered
        if price_min <= m.base_price_per_kg <= price_max
    ]

    if advanced_filters['circularity_min'] > 0:
        filtered = [
            m for m in filtered
            if m.circularity.recyclability_score >= advanced_filters['circularity_min']
        ]

    if advanced_filters['efficiency_min'] > 0:
        filtered = [
            m for m in filtered
            if m.pv_efficiency_impact and
               m.pv_efficiency_impact >= advanced_filters['efficiency_min']
        ]

    return filtered


def render_material_card(material: Material, selected: bool = False) -> None:
    """
    Render a single material card.

    Args:
        material: Material to display
        selected: Whether material is selected
    """
    card_class = "material-card"
    if selected:
        card_class += " material-card-selected"
    if material.id in st.session_state.favorite_materials:
        card_class += " material-card-favorite"

    # Card header
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**{material.name}**")
        st.caption(f"{material.category.value} • {material.subcategory}")

    with col2:
        # Favorite button
        if material.id in st.session_state.favorite_materials:
            if st.button("❤️", key=f"fav_{material.id}"):
                st.session_state.favorite_materials.remove(material.id)
                st.rerun()
        else:
            if st.button("🤍", key=f"fav_{material.id}"):
                st.session_state.favorite_materials.add(material.id)
                st.rerun()

    # Description
    st.markdown(f"<small>{material.description}</small>", unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Price",
            format_price(material.base_price_per_kg),
            delta=None
        )

    with col2:
        circ_score = material.circularity.recyclability_score
        circ_class = get_circularity_class(circ_score)
        st.metric(
            "Circularity",
            f"{circ_score:.0f}%"
        )

    with col3:
        if material.pv_efficiency_impact:
            st.metric(
                "Efficiency",
                f"{material.pv_efficiency_impact:.1f}%"
            )
        else:
            st.metric("Suppliers", len(material.suppliers))

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Select", key=f"select_{material.id}", use_container_width=True):
            if material.id not in [m.id for m in st.session_state.selected_materials]:
                st.session_state.selected_materials.append(material)
                st.success(f"Added {material.name}")
                st.rerun()

    with col2:
        if st.button("Details", key=f"details_{material.id}", use_container_width=True):
            st.session_state.detail_material = material
            st.rerun()

    with col3:
        if st.button("Add to BOM", key=f"bom_{material.id}", use_container_width=True):
            st.session_state.bom_materials.append(material)
            st.success("Added to BOM")

    st.divider()


def render_materials_grid(materials: List[Material]) -> None:
    """
    Render materials in grid layout.

    Args:
        materials: List of materials to display
    """
    if not materials:
        st.info("No materials found matching your criteria.")
        return

    # Display count
    st.markdown(f"**{len(materials)} materials found**")

    # Create grid layout
    cols_per_row = 2
    rows = (len(materials) + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            material_idx = row * cols_per_row + col_idx
            if material_idx < len(materials):
                with cols[col_idx]:
                    material = materials[material_idx]
                    selected = material.id in [m.id for m in st.session_state.selected_materials]
                    render_material_card(material, selected)


def render_materials_list(materials: List[Material]) -> None:
    """
    Render materials in list layout.

    Args:
        materials: List of materials to display
    """
    if not materials:
        st.info("No materials found matching your criteria.")
        return

    st.markdown(f"**{len(materials)} materials found**")

    for material in materials:
        selected = material.id in [m.id for m in st.session_state.selected_materials]
        render_material_card(material, selected)


# ============================================================================
# MATERIAL COMPARISON VIEW
# ============================================================================

def render_comparison_view() -> None:
    """Render material comparison interface."""
    st.markdown("## Material Comparison")

    if len(st.session_state.selected_materials) < 2:
        st.warning("Please select at least 2 materials to compare.")
        return

    materials = st.session_state.selected_materials

    # Comparison controls
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Comparing {len(materials)} materials**")

    with col2:
        if st.button("Clear Selection"):
            st.session_state.selected_materials = []
            st.rerun()

    # Tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Properties Table",
        "Radar Chart",
        "Cost Analysis",
        "Environmental Impact"
    ])

    with tab1:
        render_comparison_table(materials)

    with tab2:
        render_radar_chart(materials)

    with tab3:
        render_cost_comparison(materials)

    with tab4:
        render_environmental_comparison(materials)


def render_comparison_table(materials: List[Material]) -> None:
    """
    Render side-by-side comparison table.

    Args:
        materials: Materials to compare
    """
    st.markdown("### Property Comparison")

    # Build comparison data
    comparison_data = {
        'Property': [
            'Name',
            'Category',
            'Quality Grade',
            'Price ($/kg)',
            'Density (g/cm³)',
            'Thermal Conductivity (W/m·K)',
            'PV Efficiency Impact (%)',
            'Recyclability Score',
            'Carbon Footprint (kg CO2e/kg)',
            'Recycled Content (%)',
            'Lifetime (years)',
            'Suppliers',
            'Standards Compliance'
        ]
    }

    for material in materials:
        comparison_data[material.name] = [
            material.name,
            material.category.value,
            material.quality_grade,
            f"${material.base_price_per_kg:.2f}",
            f"{material.properties.density:.2f}",
            f"{material.properties.thermal_conductivity:.2f}",
            f"{material.pv_efficiency_impact:.1f}" if material.pv_efficiency_impact else "N/A",
            f"{material.circularity.recyclability_score:.0f}%",
            f"{material.circularity.carbon_footprint:.1f}",
            f"{material.circularity.recycled_content:.0f}%",
            str(material.lifetime_years or "N/A"),
            str(len(material.suppliers)),
            ", ".join([s.value for s in material.standards_compliance[:3]])
        ]

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export button
    csv_data = export_to_csv(df)
    st.download_button(
        "Download Comparison (CSV)",
        csv_data,
        "material_comparison.csv",
        "text/csv",
        key="download_comparison_csv"
    )


def render_radar_chart(materials: List[Material]) -> None:
    """
    Render radar chart for multi-dimensional comparison.

    Args:
        materials: Materials to compare
    """
    st.markdown("### Multi-Dimensional Analysis")

    # Normalize metrics for radar chart (0-100 scale)
    metrics = [
        'Efficiency',
        'Recyclability',
        'Cost Effectiveness',
        'Durability',
        'Sustainability'
    ]

    fig = go.Figure()

    for material in materials:
        # Calculate normalized scores
        efficiency_score = (material.pv_efficiency_impact or 0) * 4  # Scale to 100
        recyclability_score = material.circularity.recyclability_score
        cost_score = max(0, 100 - material.base_price_per_kg * 2)  # Inverse cost
        durability_score = (material.lifetime_years or 20) * 3.3  # Scale to 100
        sustainability_score = 100 - material.circularity.carbon_footprint  # Inverse carbon

        values = [
            efficiency_score,
            recyclability_score,
            cost_score,
            durability_score,
            sustainability_score
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=material.name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cost_comparison(materials: List[Material]) -> None:
    """
    Render cost comparison visualization.

    Args:
        materials: Materials to compare
    """
    st.markdown("### Cost Analysis")

    # Get quantity input
    col1, col2 = st.columns([2, 1])

    with col1:
        quantity_kg = st.number_input(
            "Quantity (kg)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            key="cost_quantity"
        )

    # Create bar chart
    names = [m.name for m in materials]
    prices = [m.base_price_per_kg for m in materials]
    total_costs = [calculate_total_cost(m, quantity_kg) for m in materials]

    # Unit price comparison
    fig1 = go.Figure(data=[
        go.Bar(
            x=names,
            y=prices,
            text=[f"${p:.2f}/kg" for p in prices],
            textposition='auto',
            marker_color='lightblue'
        )
    ])

    fig1.update_layout(
        title="Unit Price Comparison",
        xaxis_title="Material",
        yaxis_title="Price ($/kg)",
        height=400
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Total cost comparison
    fig2 = go.Figure(data=[
        go.Bar(
            x=names,
            y=total_costs,
            text=[f"${c:,.0f}" for c in total_costs],
            textposition='auto',
            marker_color='lightgreen'
        )
    ])

    fig2.update_layout(
        title=f"Total Cost Comparison ({quantity_kg} kg)",
        xaxis_title="Material",
        yaxis_title="Total Cost ($)",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Cost breakdown table
    st.markdown("#### Detailed Cost Breakdown")

    cost_data = {
        'Material': names,
        'Unit Price ($/kg)': [f"${p:.2f}" for p in prices],
        'Quantity (kg)': [quantity_kg] * len(materials),
        'Total Cost ($)': [f"${c:,.2f}" for c in total_costs],
        'Cost per Module': [f"${c/10:,.2f}" for c in total_costs]  # Assuming 10 modules
    }

    df = pd.DataFrame(cost_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_environmental_comparison(materials: List[Material]) -> None:
    """
    Render environmental impact comparison.

    Args:
        materials: Materials to compare
    """
    st.markdown("### Environmental Impact Analysis")

    # Circularity scores
    st.markdown("#### Circularity Metrics")

    names = [m.name for m in materials]
    recyclability = [m.circularity.recyclability_score for m in materials]
    carbon_footprint = [m.circularity.carbon_footprint for m in materials]
    recycled_content = [m.circularity.recycled_content for m in materials]

    # Create subplots
    col1, col2 = st.columns(2)

    with col1:
        # Recyclability comparison
        fig1 = go.Figure(data=[
            go.Bar(
                x=names,
                y=recyclability,
                text=[f"{r:.0f}%" for r in recyclability],
                textposition='auto',
                marker_color=['green' if r >= 70 else 'orange' if r >= 40 else 'red' for r in recyclability]
            )
        ])

        fig1.update_layout(
            title="Recyclability Score",
            xaxis_title="Material",
            yaxis_title="Score (%)",
            height=400
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Carbon footprint comparison
        fig2 = go.Figure(data=[
            go.Bar(
                x=names,
                y=carbon_footprint,
                text=[f"{c:.1f}" for c in carbon_footprint],
                textposition='auto',
                marker_color='coral'
            )
        ])

        fig2.update_layout(
            title="Carbon Footprint",
            xaxis_title="Material",
            yaxis_title="kg CO2e per kg",
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Recycled content comparison
    st.markdown("#### Recycled Content")

    fig3 = go.Figure(data=[
        go.Bar(
            x=names,
            y=recycled_content,
            text=[f"{r:.0f}%" for r in recycled_content],
            textposition='auto',
            marker_color='lightseagreen'
        )
    ])

    fig3.update_layout(
        title="Recycled Material Content",
        xaxis_title="Material",
        yaxis_title="Recycled Content (%)",
        height=400
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Environmental summary table
    st.markdown("#### Environmental Impact Summary")

    env_data = {
        'Material': names,
        'Recyclability': [f"{r:.0f}%" for r in recyclability],
        'Carbon Footprint (kg CO2e/kg)': [f"{c:.2f}" for c in carbon_footprint],
        'Recycled Content': [f"{r:.0f}%" for r in recycled_content],
        'Water Footprint (L/kg)': [f"{m.circularity.water_footprint:.0f}" for m in materials],
        'Toxicity': [m.circularity.toxicity_rating for m in materials],
        'EOL Recovery Rate': [f"{m.circularity.end_of_life_recovery_rate:.0f}%" for m in materials]
    }

    df = pd.DataFrame(env_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# MATERIAL DETAILS PANEL
# ============================================================================

def render_material_details(material: Material) -> None:
    """
    Render detailed material information panel.

    Args:
        material: Material to display
    """
    st.markdown(f"## {material.name}")
    st.caption(f"{material.category.value} • {material.subcategory} • {material.quality_grade}")

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Price", format_price(material.base_price_per_kg), delta=None)

    with col2:
        st.metric(
            "Recyclability",
            f"{material.circularity.recyclability_score:.0f}%"
        )

    with col3:
        if material.pv_efficiency_impact:
            st.metric("Efficiency", f"{material.pv_efficiency_impact:.1f}%")
        else:
            st.metric("Lifetime", f"{material.lifetime_years or 'N/A'} years")

    with col4:
        st.metric("Suppliers", len(material.suppliers))

    st.divider()

    # Tabs for different detail sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Properties",
        "Suppliers",
        "Pricing",
        "Environmental",
        "Compliance"
    ])

    with tab1:
        render_properties_tab(material)

    with tab2:
        render_suppliers_tab(material)

    with tab3:
        render_pricing_tab(material)

    with tab4:
        render_environmental_tab(material)

    with tab5:
        render_compliance_tab(material)

    # Action buttons at bottom
    st.divider()
    render_detail_actions(material)


def render_properties_tab(material: Material) -> None:
    """Render properties tab in detail view."""
    st.markdown("### Physical Properties")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Properties")
        props = material.properties

        properties_data = {
            'Property': [
                'Density',
                'Thermal Conductivity',
                'Specific Heat',
                'Melting Point'
            ],
            'Value': [
                f"{props.density:.2f} g/cm³",
                f"{props.thermal_conductivity:.2f} W/m·K",
                f"{props.specific_heat:.0f} J/kg·K",
                f"{props.melting_point:.1f} °C" if props.melting_point else "N/A"
            ]
        }

        df1 = pd.DataFrame(properties_data)
        st.dataframe(df1, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Electrical & Mechanical")

        elec_mech_data = []

        if props.electrical_resistivity:
            elec_mech_data.append(('Electrical Resistivity', f"{props.electrical_resistivity:.2e} Ω·m"))

        if props.tensile_strength:
            elec_mech_data.append(('Tensile Strength', f"{props.tensile_strength:.0f} MPa"))

        if props.elastic_modulus:
            elec_mech_data.append(('Elastic Modulus', f"{props.elastic_modulus:.0f} GPa"))

        if props.hardness:
            elec_mech_data.append(('Hardness', props.hardness))

        if elec_mech_data:
            df2 = pd.DataFrame(elec_mech_data, columns=['Property', 'Value'])
            st.dataframe(df2, use_container_width=True, hide_index=True)
        else:
            st.info("No electrical/mechanical data available")

    # Optical properties (if applicable)
    if material.properties.transmittance or material.properties.refractive_index:
        st.markdown("#### Optical Properties")

        col1, col2 = st.columns(2)

        with col1:
            if material.properties.transmittance:
                st.metric("Transmittance", f"{material.properties.transmittance:.1f}%")

        with col2:
            if material.properties.refractive_index:
                st.metric("Refractive Index", f"{material.properties.refractive_index:.3f}")

    # PV-specific properties
    st.markdown("#### PV Application Properties")

    col1, col2, col3 = st.columns(3)

    with col1:
        if material.pv_efficiency_impact:
            st.metric("Efficiency Impact", f"{material.pv_efficiency_impact:.1f}%")

    with col2:
        if material.typical_thickness:
            st.metric("Typical Thickness", f"{material.typical_thickness:.2f} mm")

    with col3:
        if material.lifetime_years:
            st.metric("Expected Lifetime", f"{material.lifetime_years} years")


def render_suppliers_tab(material: Material) -> None:
    """Render suppliers tab in detail view."""
    st.markdown("### Supplier Information")

    if not material.suppliers:
        st.info("No supplier information available. Fetching from ENF Solar...")

        # Try to fetch from ENF API
        enf_client = st.session_state.enf_client
        suppliers = enf_client.search_suppliers(min_rating=4.0)

        if suppliers:
            st.success(f"Found {len(suppliers)} suppliers")
            for supplier in suppliers[:3]:  # Show top 3
                render_supplier_card(supplier)
        else:
            st.warning("No suppliers found")
    else:
        for supplier in material.suppliers:
            render_supplier_info_card(supplier)


def render_supplier_info_card(supplier: SupplierInfo) -> None:
    """Render supplier information card."""
    with st.container():
        st.markdown(f"#### {supplier.name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Country:** {supplier.country}")
            if supplier.enf_rating:
                st.markdown(f"**ENF Rating:** {'⭐' * int(supplier.enf_rating)} ({supplier.enf_rating:.1f}/5.0)")

        with col2:
            if supplier.annual_capacity:
                st.markdown(f"**Capacity:** {supplier.annual_capacity}")
            if supplier.lead_time_days:
                st.markdown(f"**Lead Time:** {supplier.lead_time_days} days")

        with col3:
            if supplier.certifications:
                st.markdown(f"**Certifications:** {', '.join(supplier.certifications[:3])}")
            if supplier.minimum_order_quantity:
                st.markdown(f"**MOQ:** {supplier.minimum_order_quantity}")

        st.divider()


def render_supplier_card(supplier: ENFSupplier) -> None:
    """Render ENF supplier card."""
    with st.container():
        st.markdown(f"#### {supplier.company_name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Country:** {supplier.country}")
            st.markdown(f"**Tier:** {supplier.enf_tier}")
            st.markdown(f"**Rating:** {'⭐' * int(supplier.rating)} ({supplier.rating:.1f}/5.0)")

        with col2:
            st.markdown(f"**Type:** {supplier.supplier_type.value}")
            if supplier.production_capacity:
                st.markdown(f"**Capacity:** {supplier.production_capacity}")
            st.markdown(f"**Verified:** {'✅' if supplier.verified else '❌'}")

        with col3:
            if supplier.certifications:
                st.markdown(f"**Certifications:**")
                for cert in supplier.certifications[:3]:
                    st.markdown(f"- {cert}")

        if supplier.website:
            st.markdown(f"[Visit Website]({supplier.website})")

        st.divider()


def render_pricing_tab(material: Material) -> None:
    """Render pricing tab in detail view."""
    st.markdown("### Pricing Information")

    # Current price
    st.markdown("#### Current Price")
    st.markdown(f"## {format_price(material.base_price_per_kg)} per kg")

    # Price history chart
    if material.price_history:
        st.markdown("#### Price History")

        dates = [ph.date for ph in material.price_history]
        prices = [ph.price_per_kg for ph in material.price_history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="Price Trend",
            xaxis_title="Date",
            yaxis_title="Price ($/kg)",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Price statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average", format_price(sum(prices) / len(prices)))

        with col2:
            st.metric("Minimum", format_price(min(prices)))

        with col3:
            st.metric("Maximum", format_price(max(prices)))

    # Get price quotes from ENF
    st.markdown("#### Current Market Quotes")

    enf_client = st.session_state.enf_client
    quotes = enf_client.get_price_quotes(material.name)

    if quotes:
        for quote in quotes:
            with st.container():
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**{quote.supplier_name}**")
                    st.markdown(f"Price: {format_price(quote.unit_price)}/kg")

                with col2:
                    st.markdown(f"MOQ: {quote.minimum_order}")
                    st.markdown(f"Lead Time: {quote.lead_time_days} days")

                with col3:
                    st.markdown(f"Valid Until: {quote.valid_until}")
                    st.markdown(f"Terms: {quote.terms}")

                st.divider()
    else:
        st.info("No current quotes available")


def render_environmental_tab(material: Material) -> None:
    """Render environmental tab in detail view."""
    st.markdown("### Environmental Impact & Circularity")

    circ = material.circularity

    # Circularity score visualization
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Recyclability Score")

        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=circ.recyclability_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Circularity Metrics")

        metrics_data = {
            'Metric': [
                'Recyclability Score',
                'Recycled Content',
                'End-of-Life Recovery',
                'Reusability Potential',
                'Degradability'
            ],
            'Value': [
                f"{circ.recyclability_score:.0f}%",
                f"{circ.recycled_content:.0f}%",
                f"{circ.end_of_life_recovery_rate:.0f}%",
                circ.reusability_potential,
                circ.degradability
            ]
        }

        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Environmental footprint
    st.markdown("#### Environmental Footprint")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Carbon Footprint",
            f"{circ.carbon_footprint:.1f}",
            delta=None,
            help="kg CO2e per kg of material"
        )

    with col2:
        st.metric(
            "Embodied Energy",
            f"{circ.embodied_energy:.0f}",
            delta=None,
            help="MJ per kg of material"
        )

    with col3:
        st.metric(
            "Water Footprint",
            f"{circ.water_footprint:.0f}",
            delta=None,
            help="Liters per kg of material"
        )

    with col4:
        st.metric(
            "Toxicity",
            circ.toxicity_rating,
            delta=None
        )

    # Carbon footprint comparison
    st.markdown("#### Carbon Footprint Analysis")

    # Compare with typical materials
    comparison_materials = ['Aluminum', 'Steel', 'Plastic', 'Glass', 'Silicon']
    comparison_carbon = [8.5, 1.8, 3.2, 0.85, 45.0]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_materials + [material.name],
        y=comparison_carbon + [circ.carbon_footprint],
        marker_color=['lightblue'] * len(comparison_materials) + ['coral']
    ))

    fig.update_layout(
        title="Carbon Footprint Comparison",
        xaxis_title="Material",
        yaxis_title="kg CO2e per kg",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_compliance_tab(material: Material) -> None:
    """Render compliance tab in detail view."""
    st.markdown("### Standards & Compliance")

    # Standards compliance
    if material.standards_compliance:
        st.markdown("#### Compliant Standards")

        for standard in material.standards_compliance:
            st.markdown(f"✅ **{standard.value}**")

        st.divider()

    # Quality grade
    st.markdown("#### Quality Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Quality Grade:** {material.quality_grade}")

    with col2:
        if material.lifetime_years:
            st.markdown(f"**Expected Lifetime:** {material.lifetime_years} years")

    # Data source and metadata
    st.markdown("#### Data Information")

    metadata = {
        'Field': ['Material ID', 'Data Source', 'Last Updated', 'Tags'],
        'Value': [
            material.id,
            material.data_source or 'Internal Database',
            material.last_updated[:10] if material.last_updated else 'N/A',
            ', '.join(material.tags) if material.tags else 'None'
        ]
    }

    df = pd.DataFrame(metadata)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if material.notes:
        st.markdown("#### Notes")
        st.info(material.notes)


def render_detail_actions(material: Material) -> None:
    """Render action buttons for material details."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Add to Selection", use_container_width=True):
            if material.id not in [m.id for m in st.session_state.selected_materials]:
                st.session_state.selected_materials.append(material)
                st.success("Added to selection")
                st.rerun()

    with col2:
        if st.button("Add to BOM", use_container_width=True):
            st.session_state.bom_materials.append(material)
            st.success("Added to BOM")

    with col3:
        # Toggle favorite
        if material.id in st.session_state.favorite_materials:
            if st.button("Remove Favorite", use_container_width=True):
                st.session_state.favorite_materials.remove(material.id)
                st.rerun()
        else:
            if st.button("Add Favorite", use_container_width=True):
                st.session_state.favorite_materials.add(material.id)
                st.rerun()

    with col4:
        # Export material data
        material_dict = {
            'id': material.id,
            'name': material.name,
            'category': material.category.value,
            'base_price_per_kg': material.base_price_per_kg,
            'recyclability_score': material.circularity.recyclability_score,
            'carbon_footprint': material.circularity.carbon_footprint
        }

        json_data = export_to_json(material_dict, f"{material.id}.json")

        st.download_button(
            "Export Data",
            json_data,
            f"{material.id}.json",
            "application/json",
            use_container_width=True
        )

    with col5:
        if st.button("Generate Report", use_container_width=True):
            st.info("PDF report generation coming soon!")


# ============================================================================
# INTERACTIVE FEATURES
# ============================================================================

def render_favorites_section() -> None:
    """Render favorites section."""
    st.markdown("## Favorite Materials")

    if not st.session_state.favorite_materials:
        st.info("No favorite materials yet. Click the heart icon on material cards to add favorites.")
        return

    loader = st.session_state.material_loader
    favorites = [
        loader.get_material(mat_id)
        for mat_id in st.session_state.favorite_materials
    ]
    favorites = [m for m in favorites if m is not None]

    for material in favorites:
        render_material_card(material)


def render_bom_section() -> None:
    """Render Bill of Materials section."""
    st.markdown("## Bill of Materials (BOM)")

    if not st.session_state.bom_materials:
        st.info("No materials in BOM yet. Click 'Add to BOM' on material cards.")
        return

    # BOM table
    bom_data = []

    for idx, material in enumerate(st.session_state.bom_materials):
        bom_data.append({
            '#': idx + 1,
            'Material': material.name,
            'Category': material.category.value,
            'Unit Price': format_price(material.base_price_per_kg),
            'Quantity (kg)': 0,  # User input
            'Total Cost': format_price(0)
        })

    df = pd.DataFrame(bom_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export BOM
    col1, col2 = st.columns(2)

    with col1:
        csv_data = export_to_csv(df)
        st.download_button(
            "Download BOM (CSV)",
            csv_data,
            "bom.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        if st.button("Clear BOM", use_container_width=True):
            st.session_state.bom_materials = []
            st.rerun()


def render_share_section() -> None:
    """Render share selections section."""
    st.markdown("## Share Materials")

    if not st.session_state.selected_materials:
        st.info("No materials selected to share.")
        return

    # Generate shareable data
    share_data = {
        'materials': [m.id for m in st.session_state.selected_materials],
        'timestamp': datetime.now().isoformat(),
        'count': len(st.session_state.selected_materials)
    }

    json_data = export_to_json(share_data, "shared_materials.json")

    st.download_button(
        "Download Selection",
        json_data,
        "shared_materials.json",
        "application/json"
    )

    # Display share link (mock)
    share_link = f"https://pv-simulator.com/share/{hash(str(share_data))}"
    st.code(share_link, language=None)

    st.caption("Share this link to collaborate with others")


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_materials_selection_ui() -> None:
    """
    Main function to render the complete materials selection UI.

    This is the entry point for the materials selection interface.
    Call this function from your Streamlit app.
    """
    # Initialize session state
    initialize_session_state()

    # Inject custom CSS
    inject_custom_css()

    # Page header
    st.title("🔬 Materials Selection")
    st.markdown("Select and compare materials for PV module manufacturing")

    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## Navigation")

        page = st.radio(
            "Select View",
            [
                "Material Browser",
                "Comparison",
                "Favorites",
                "Bill of Materials",
                "Share"
            ],
            key="nav_radio"
        )

        st.divider()

        # Quick stats
        st.markdown("### Quick Stats")
        loader = st.session_state.material_loader

        st.metric("Total Materials", len(loader.get_all_materials()))
        st.metric("Selected", len(st.session_state.selected_materials))
        st.metric("Favorites", len(st.session_state.favorite_materials))
        st.metric("In BOM", len(st.session_state.bom_materials))

    # Main content based on selected page
    if page == "Material Browser":
        render_material_browser()
    elif page == "Comparison":
        render_comparison_view()
    elif page == "Favorites":
        render_favorites_section()
    elif page == "Bill of Materials":
        render_bom_section()
    elif page == "Share":
        render_share_section()


def render_material_browser() -> None:
    """Render the main material browser interface."""
    # Category tabs
    selected_category = render_category_tabs()

    st.divider()

    # Search and filters
    search_query, quick_filters, advanced_filters = render_search_and_filters()

    st.divider()

    # Get and filter materials
    loader = st.session_state.material_loader

    if selected_category:
        materials = loader.get_by_category(selected_category)
    else:
        materials = loader.get_all_materials()

    # Apply filters
    filtered_materials = apply_filters(
        materials,
        search_query,
        quick_filters,
        advanced_filters
    )

    # Render materials based on view mode
    if st.session_state.view_mode == "grid":
        render_materials_grid(filtered_materials)
    else:
        render_materials_list(filtered_materials)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="PV Materials Selection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Render the UI
    render_materials_selection_ui()
