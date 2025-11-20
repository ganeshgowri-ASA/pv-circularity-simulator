"""
Cell Design UI & Visualization - Streamlit Application

This module provides a comprehensive interface for designing, simulating,
and analyzing photovoltaic cell architectures.

Features:
- Interactive cell architecture designer
- Layer stack builder with drag-and-drop simulation
- Material property inputs with validation
- Real-time efficiency estimation
- SCAPS simulation launcher
- Comprehensive visualizations (JV curves, QE, band diagrams, loss analysis)
- Design comparison tools
- Optimization interface
- Cell template gallery
- Design history tracking
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import sys
from datetime import datetime
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.data_models.materials import MaterialDatabase, Material, DopingType
from src.core.data_models.cell_architecture import (
    CellArchitecture,
    Layer,
    LayerType
)
from src.core.constants import (
    CELL_ARCHITECTURES,
    LAYER_COLORS,
    EFFICIENCY_METRICS,
    STC_TEMPERATURE,
    STC_IRRADIANCE
)
from src.integrations.materials_db.database import get_materials_database
from src.integrations.scaps.wrapper import SCAPSWrapper, SimulationResults
from src.integrations.griddler.integration import (
    GriddlerIntegration,
    GridPattern
)
from src.simulation.device_physics import (
    DevicePhysicsEngine,
    quick_efficiency_estimate
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PV Cell Design Studio",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css() -> None:
    """Load custom CSS styling for the application."""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B35;
        --secondary-color: #004E89;
        --accent-color: #F7B801;
        --background-color: #F5F5F5;
        --text-color: #1E1E1E;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #004E89 0%, #1A659E 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .design-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }

    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary-color);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Layer stack styling */
    .layer-item {
        background: white;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #ccc;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .layer-item:hover {
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }

    /* Success/Warning/Error boxes */
    .success-box {
        background: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 6px;
        padding: 1rem;
        color: #155724;
    }

    .warning-box {
        background: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 6px;
        padding: 1rem;
        color: #856404;
    }

    .error-box {
        background: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 6px;
        padding: 1rem;
        color: #721C24;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'material_db' not in st.session_state:
        st.session_state.material_db = get_materials_database()

    if 'current_architecture' not in st.session_state:
        st.session_state.current_architecture = None

    if 'design_history' not in st.session_state:
        st.session_state.design_history = []

    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None

    if 'comparison_designs' not in st.session_state:
        st.session_state.comparison_designs = []

    if 'templates' not in st.session_state:
        st.session_state.templates = load_default_templates()


# ============================================================================
# TEMPLATE MANAGEMENT
# ============================================================================

def load_default_templates() -> Dict[str, CellArchitecture]:
    """
    Load default cell architecture templates.

    Returns:
        Dictionary of template name to CellArchitecture
    """
    templates = {}

    # Al-BSF Template
    al_bsf = CellArchitecture(
        name="Al-BSF (Standard)",
        architecture_type="Al-BSF",
        description="Traditional aluminum back surface field cell"
    )
    al_bsf.add_layer(Layer(
        name="Al rear contact",
        layer_type=LayerType.METAL,
        material_name="Al",
        thickness=20.0
    ))
    al_bsf.add_layer(Layer(
        name="p+ BSF",
        layer_type=LayerType.BSF,
        material_name="Si(p+)",
        thickness=0.5,
        doping_type=DopingType.P_TYPE,
        doping_concentration=1e19
    ))
    al_bsf.add_layer(Layer(
        name="p-type base",
        layer_type=LayerType.SUBSTRATE,
        material_name="Si(p)",
        thickness=180.0,
        doping_type=DopingType.P_TYPE,
        doping_concentration=1e16
    ))
    al_bsf.add_layer(Layer(
        name="n+ emitter",
        layer_type=LayerType.EMITTER,
        material_name="Si(n+)",
        thickness=0.3,
        doping_type=DopingType.N_TYPE,
        doping_concentration=1e20
    ))
    al_bsf.add_layer(Layer(
        name="SiNx ARC",
        layer_type=LayerType.ARC,
        material_name="SiNx",
        thickness=0.08
    ))
    templates["Al-BSF"] = al_bsf

    # PERC Template
    perc = CellArchitecture(
        name="PERC",
        architecture_type="PERC",
        description="Passivated emitter and rear cell"
    )
    perc.add_layer(Layer(
        name="Al rear contact",
        layer_type=LayerType.METAL,
        material_name="Al",
        thickness=20.0
    ))
    perc.add_layer(Layer(
        name="Al2O3 passivation",
        layer_type=LayerType.PASSIVATION,
        material_name="Al2O3",
        thickness=0.01
    ))
    perc.add_layer(Layer(
        name="p-type base",
        layer_type=LayerType.SUBSTRATE,
        material_name="Si(p)",
        thickness=160.0,
        doping_type=DopingType.P_TYPE,
        doping_concentration=1e16
    ))
    perc.add_layer(Layer(
        name="n+ emitter",
        layer_type=LayerType.EMITTER,
        material_name="Si(n+)",
        thickness=0.3,
        doping_type=DopingType.N_TYPE,
        doping_concentration=5e19
    ))
    perc.add_layer(Layer(
        name="SiNx ARC",
        layer_type=LayerType.ARC,
        material_name="SiNx",
        thickness=0.075
    ))
    templates["PERC"] = perc

    # TOPCon Template
    topcon = CellArchitecture(
        name="TOPCon",
        architecture_type="TOPCon",
        description="Tunnel oxide passivated contact"
    )
    topcon.add_layer(Layer(
        name="Al rear contact",
        layer_type=LayerType.METAL,
        material_name="Al",
        thickness=20.0
    ))
    topcon.add_layer(Layer(
        name="Poly-Si(n+)",
        layer_type=LayerType.CONTACT,
        material_name="Poly-Si(n+)",
        thickness=0.15,
        doping_type=DopingType.N_TYPE,
        doping_concentration=1e20
    ))
    topcon.add_layer(Layer(
        name="SiO2 tunnel oxide",
        layer_type=LayerType.PASSIVATION,
        material_name="SiO2",
        thickness=0.0015
    ))
    topcon.add_layer(Layer(
        name="n-type base",
        layer_type=LayerType.SUBSTRATE,
        material_name="Si(n)",
        thickness=170.0,
        doping_type=DopingType.N_TYPE,
        doping_concentration=1e16
    ))
    topcon.add_layer(Layer(
        name="p+ emitter",
        layer_type=LayerType.EMITTER,
        material_name="Si(p+)",
        thickness=0.2,
        doping_type=DopingType.P_TYPE,
        doping_concentration=1e19
    ))
    topcon.add_layer(Layer(
        name="SiNx ARC",
        layer_type=LayerType.ARC,
        material_name="SiNx",
        thickness=0.07
    ))
    templates["TOPCon"] = topcon

    # HJT Template
    hjt = CellArchitecture(
        name="HJT (SHJ)",
        architecture_type="HJT",
        description="Heterojunction with intrinsic thin layer"
    )
    hjt.add_layer(Layer(
        name="Ag rear contact",
        layer_type=LayerType.METAL,
        material_name="Ag",
        thickness=15.0
    ))
    hjt.add_layer(Layer(
        name="ITO (rear)",
        layer_type=LayerType.TCO,
        material_name="ITO",
        thickness=0.08
    ))
    hjt.add_layer(Layer(
        name="a-Si:H(n)",
        layer_type=LayerType.CONTACT,
        material_name="a-Si:H(n)",
        thickness=0.008,
        doping_type=DopingType.N_TYPE,
        doping_concentration=1e19
    ))
    hjt.add_layer(Layer(
        name="a-Si:H(i) rear",
        layer_type=LayerType.PASSIVATION,
        material_name="a-Si:H(i)",
        thickness=0.005,
        doping_type=DopingType.INTRINSIC
    ))
    hjt.add_layer(Layer(
        name="n-type c-Si wafer",
        layer_type=LayerType.SUBSTRATE,
        material_name="Si(n)",
        thickness=170.0,
        doping_type=DopingType.N_TYPE,
        doping_concentration=1e16
    ))
    hjt.add_layer(Layer(
        name="a-Si:H(i) front",
        layer_type=LayerType.PASSIVATION,
        material_name="a-Si:H(i)",
        thickness=0.005,
        doping_type=DopingType.INTRINSIC
    ))
    hjt.add_layer(Layer(
        name="a-Si:H(p)",
        layer_type=LayerType.CONTACT,
        material_name="a-Si:H(p)",
        thickness=0.008,
        doping_type=DopingType.P_TYPE,
        doping_concentration=1e19
    ))
    hjt.add_layer(Layer(
        name="ITO (front)",
        layer_type=LayerType.TCO,
        material_name="ITO",
        thickness=0.075
    ))
    templates["HJT"] = hjt

    return templates


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cell_cross_section(architecture: CellArchitecture) -> go.Figure:
    """
    Create cross-sectional diagram of cell structure.

    Args:
        architecture: Cell architecture to visualize

    Returns:
        Plotly figure object
    """
    if not architecture or not architecture.layers:
        return go.Figure()

    fig = go.Figure()

    # Calculate positions
    y_position = 0
    layer_data = []

    for layer in architecture.layers:
        layer_color = LAYER_COLORS.get(layer.material_name, LAYER_COLORS['default'])

        layer_data.append({
            'y_start': y_position,
            'y_end': y_position + layer.thickness,
            'name': layer.name,
            'material': layer.material_name,
            'thickness': layer.thickness,
            'color': layer_color
        })

        y_position += layer.thickness

    # Plot layers as rectangles
    for i, layer in enumerate(layer_data):
        # Main rectangle
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=layer['y_start'], y1=layer['y_end'],
            fillcolor=layer['color'],
            line=dict(color="black", width=1),
            opacity=0.8
        )

        # Label
        y_mid = (layer['y_start'] + layer['y_end']) / 2
        text = f"{layer['name']}<br>{layer['thickness']:.3f} µm"

        fig.add_annotation(
            x=0.5, y=y_mid,
            text=text,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=4
        )

    # Update layout
    total_thickness = sum(ld['thickness'] for ld in layer_data)

    fig.update_layout(
        title=f"Cell Cross-Section: {architecture.name}",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            title="Position (µm)",
            showgrid=True,
            zeroline=False,
            range=[0, total_thickness * 1.05]
        ),
        height=max(400, min(800, len(layer_data) * 80)),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=80, r=20, t=60, b=40)
    )

    return fig


def plot_jv_curve(results: SimulationResults) -> go.Figure:
    """
    Plot JV curve with interactive features.

    Args:
        results: Simulation results

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # JV curve
    fig.add_trace(go.Scatter(
        x=results.voltage,
        y=results.current_density,
        mode='lines',
        name='JV Curve',
        line=dict(color='#004E89', width=3),
        hovertemplate='V: %{x:.3f} V<br>J: %{y:.2f} mA/cm²<extra></extra>'
    ))

    # Mark key points
    # Voc
    fig.add_trace(go.Scatter(
        x=[results.voc],
        y=[0],
        mode='markers',
        name=f'Voc = {results.voc:.3f} V',
        marker=dict(size=12, color='#F7B801', symbol='diamond'),
        hovertemplate='Voc: %{x:.3f} V<extra></extra>'
    ))

    # Jsc
    fig.add_trace(go.Scatter(
        x=[0],
        y=[results.jsc],
        mode='markers',
        name=f'Jsc = {results.jsc:.2f} mA/cm²',
        marker=dict(size=12, color='#FF6B35', symbol='diamond'),
        hovertemplate='Jsc: %{y:.2f} mA/cm²<extra></extra>'
    ))

    # MPP
    fig.add_trace(go.Scatter(
        x=[results.vmpp],
        y=[results.jmpp],
        mode='markers',
        name=f'MPP: {results.pmpp:.2f} mW/cm²',
        marker=dict(size=15, color='#1A659E', symbol='star'),
        hovertemplate='Vmpp: %{x:.3f} V<br>Jmpp: %{y:.2f} mA/cm²<extra></extra>'
    ))

    # Power curve (on secondary y-axis would be ideal, but simplified here)

    fig.update_layout(
        title="Current-Voltage (JV) Characteristic",
        xaxis=dict(title="Voltage (V)", showgrid=True),
        yaxis=dict(title="Current Density (mA/cm²)", showgrid=True),
        hovermode='closest',
        height=500,
        plot_bgcolor='white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig


def plot_quantum_efficiency(results: SimulationResults) -> go.Figure:
    """
    Plot external and internal quantum efficiency.

    Args:
        results: Simulation results

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # EQE
    fig.add_trace(go.Scatter(
        x=results.wavelength,
        y=results.eqe * 100,  # Convert to percentage
        mode='lines',
        name='EQE',
        line=dict(color='#004E89', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,78,137,0.2)',
        hovertemplate='λ: %{x:.0f} nm<br>EQE: %{y:.1f}%<extra></extra>'
    ))

    # IQE
    fig.add_trace(go.Scatter(
        x=results.wavelength,
        y=results.iqe * 100,
        mode='lines',
        name='IQE',
        line=dict(color='#FF6B35', width=2, dash='dash'),
        hovertemplate='λ: %{x:.0f} nm<br>IQE: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Quantum Efficiency Spectrum",
        xaxis=dict(title="Wavelength (nm)", showgrid=True),
        yaxis=dict(title="Quantum Efficiency (%)", showgrid=True, range=[0, 105]),
        hovermode='x unified',
        height=450,
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_band_diagram(results: SimulationResults) -> go.Figure:
    """
    Plot energy band diagram.

    Args:
        results: Simulation results

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Conduction band
    fig.add_trace(go.Scatter(
        x=results.position,
        y=results.conduction_band,
        mode='lines',
        name='Conduction Band',
        line=dict(color='#004E89', width=2),
        hovertemplate='Position: %{x:.2f} µm<br>Ec: %{y:.2f} eV<extra></extra>'
    ))

    # Valence band
    fig.add_trace(go.Scatter(
        x=results.position,
        y=results.valence_band,
        mode='lines',
        name='Valence Band',
        line=dict(color='#FF6B35', width=2),
        fill='tonexty',
        fillcolor='rgba(200,200,200,0.3)',
        hovertemplate='Position: %{x:.2f} µm<br>Ev: %{y:.2f} eV<extra></extra>'
    ))

    # Fermi level
    fig.add_trace(go.Scatter(
        x=results.position,
        y=results.fermi_level,
        mode='lines',
        name='Fermi Level',
        line=dict(color='#1A659E', width=2, dash='dash'),
        hovertemplate='Position: %{x:.2f} µm<br>Ef: %{y:.2f} eV<extra></extra>'
    ))

    fig.update_layout(
        title="Energy Band Diagram",
        xaxis=dict(title="Position (µm)", showgrid=True),
        yaxis=dict(title="Energy (eV)", showgrid=True),
        hovermode='x unified',
        height=450,
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_loss_waterfall(results: SimulationResults) -> go.Figure:
    """
    Create waterfall chart showing loss mechanisms.

    Args:
        results: Simulation results

    Returns:
        Plotly figure object
    """
    # Start with theoretical maximum (Shockley-Queisser limit ~33% for Si)
    sq_limit = 33.0

    # Calculate cumulative losses
    categories = ['SQ Limit']
    values = [sq_limit]
    measures = ['absolute']

    for loss_name, loss_value in results.losses.items():
        categories.append(loss_name)
        values.append(-loss_value)
        measures.append('relative')

    categories.append('Final Efficiency')
    values.append(results.efficiency)
    measures.append('total')

    fig = go.Figure(go.Waterfall(
        name="Efficiency",
        orientation="v",
        measure=measures,
        x=categories,
        y=values,
        textposition="outside",
        text=[f"{v:+.1f}%" if v != sq_limit and v != results.efficiency
              else f"{v:.1f}%" for v in values],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#FF6B35"}},
        increasing={"marker": {"color": "#1A659E"}},
        totals={"marker": {"color": "#004E89"}}
    ))

    fig.update_layout(
        title="Loss Mechanism Analysis",
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title="Efficiency (%)", showgrid=True),
        height=500,
        plot_bgcolor='white',
        showlegend=False
    )

    return fig


def plot_design_comparison(
    designs: List[Tuple[str, SimulationResults]]
) -> go.Figure:
    """
    Create comparison chart for multiple designs.

    Args:
        designs: List of (name, results) tuples

    Returns:
        Plotly figure object
    """
    if not designs:
        return go.Figure()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Efficiency", "Voc", "Jsc", "Fill Factor"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    names = [d[0] for d in designs]
    colors = px.colors.qualitative.Set2

    # Efficiency
    fig.add_trace(
        go.Bar(
            x=names,
            y=[d[1].efficiency for d in designs],
            name="Efficiency",
            marker_color=colors[0],
            text=[f"{d[1].efficiency:.2f}%" for d in designs],
            textposition='outside'
        ),
        row=1, col=1
    )

    # Voc
    fig.add_trace(
        go.Bar(
            x=names,
            y=[d[1].voc for d in designs],
            name="Voc",
            marker_color=colors[1],
            text=[f"{d[1].voc:.3f} V" for d in designs],
            textposition='outside'
        ),
        row=1, col=2
    )

    # Jsc
    fig.add_trace(
        go.Bar(
            x=names,
            y=[d[1].jsc for d in designs],
            name="Jsc",
            marker_color=colors[2],
            text=[f"{d[1].jsc:.2f}" for d in designs],
            textposition='outside'
        ),
        row=2, col=1
    )

    # FF
    fig.add_trace(
        go.Bar(
            x=names,
            y=[d[1].ff for d in designs],
            name="FF",
            marker_color=colors[3],
            text=[f"{d[1].ff:.2f}%" for d in designs],
            textposition='outside'
        ),
        row=2, col=2
    )

    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Current (mA/cm²)", row=2, col=1)
    fig.update_yaxes(title_text="Fill Factor (%)", row=2, col=2)

    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='white'
    )

    return fig


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header() -> None:
    """Render application header."""
    st.markdown("""
    <div class="main-header">
        <h1>☀️ PV Cell Design Studio</h1>
        <p>Design, simulate, and optimize photovoltaic cell architectures</p>
    </div>
    """, unsafe_allow_html=True)


def render_architecture_selector() -> Optional[CellArchitecture]:
    """
    Render architecture selection interface.

    Returns:
        Selected or created cell architecture
    """
    st.subheader("🏗️ Cell Architecture")

    selection_mode = st.radio(
        "Choose design mode:",
        ["Start from template", "Create custom design", "Load from history"],
        horizontal=True
    )

    architecture = None

    if selection_mode == "Start from template":
        col1, col2 = st.columns([1, 1])

        with col1:
            template_name = st.selectbox(
                "Select template:",
                list(st.session_state.templates.keys()),
                help="Choose a pre-configured cell architecture"
            )

        with col2:
            if st.button("Load Template", type="primary", use_container_width=True):
                architecture = st.session_state.templates[template_name].clone()
                st.success(f"Loaded {template_name} template")

    elif selection_mode == "Create custom design":
        col1, col2 = st.columns([2, 1])

        with col1:
            arch_name = st.text_input(
                "Design name:",
                value="My Custom Cell",
                help="Enter a name for your design"
            )

        with col2:
            arch_type = st.selectbox(
                "Architecture type:",
                list(CELL_ARCHITECTURES.keys())
            )

        if st.button("Create New Design", type="primary", use_container_width=True):
            architecture = CellArchitecture(
                name=arch_name,
                architecture_type=arch_type,
                description=f"Custom {arch_type} cell design"
            )
            st.success(f"Created new {arch_type} design")

    elif selection_mode == "Load from history":
        if st.session_state.design_history:
            history_names = [
                f"{d['name']} ({d['timestamp']})"
                for d in st.session_state.design_history
            ]
            selected = st.selectbox("Select from history:", history_names)

            if st.button("Load from History", type="primary"):
                idx = history_names.index(selected)
                arch_dict = st.session_state.design_history[idx]['architecture']
                architecture = CellArchitecture.from_dict(arch_dict)
                st.success(f"Loaded design from history")
        else:
            st.info("No designs in history yet")

    return architecture


def render_layer_stack_editor(architecture: CellArchitecture) -> None:
    """
    Render layer stack editor interface.

    Args:
        architecture: Cell architecture to edit
    """
    st.subheader("📚 Layer Stack Editor")

    if not architecture:
        st.warning("Please select or create an architecture first")
        return

    # Display current layers
    st.write(f"**Total thickness:** {architecture.get_total_thickness():.3f} µm")

    # Layer list with controls
    for i, layer in enumerate(architecture.layers):
        with st.expander(
            f"Layer {i+1}: {layer.name} ({layer.thickness:.3f} µm)",
            expanded=False
        ):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                new_name = st.text_input(
                    "Layer name:",
                    value=layer.name,
                    key=f"layer_name_{layer.id}"
                )
                layer.name = new_name

            with col2:
                new_material = st.selectbox(
                    "Material:",
                    st.session_state.material_db.list_materials(),
                    index=st.session_state.material_db.list_materials().index(layer.material_name)
                    if layer.material_name in st.session_state.material_db.list_materials()
                    else 0,
                    key=f"layer_material_{layer.id}"
                )
                layer.material_name = new_material

            with col3:
                new_thickness = st.number_input(
                    "Thickness (µm):",
                    min_value=0.001,
                    max_value=500.0,
                    value=float(layer.thickness),
                    format="%.4f",
                    key=f"layer_thickness_{layer.id}"
                )
                layer.thickness = new_thickness

            # Doping parameters
            if layer.doping_type:
                col1, col2 = st.columns(2)

                with col1:
                    new_doping_type = st.selectbox(
                        "Doping type:",
                        ["n", "p", "i"],
                        index=["n", "p", "i"].index(layer.doping_type),
                        key=f"layer_doping_type_{layer.id}"
                    )
                    layer.doping_type = DopingType(new_doping_type)

                with col2:
                    if layer.doping_concentration:
                        new_doping = st.number_input(
                            "Doping (cm⁻³):",
                            min_value=1e10,
                            max_value=1e22,
                            value=float(layer.doping_concentration),
                            format="%.2e",
                            key=f"layer_doping_{layer.id}"
                        )
                        layer.doping_concentration = new_doping

            # Layer controls
            col1, col2, col3 = st.columns(3)

            with col1:
                if i > 0 and st.button("Move Up ↑", key=f"move_up_{layer.id}", use_container_width=True):
                    architecture.move_layer(layer.id, i - 1)
                    st.rerun()

            with col2:
                if i < len(architecture.layers) - 1 and st.button(
                    "Move Down ↓",
                    key=f"move_down_{layer.id}",
                    use_container_width=True
                ):
                    architecture.move_layer(layer.id, i + 1)
                    st.rerun()

            with col3:
                if st.button("Remove ✕", key=f"remove_{layer.id}", use_container_width=True):
                    architecture.remove_layer(layer.id)
                    st.rerun()

    # Add new layer
    st.markdown("---")
    st.write("**Add New Layer**")

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        new_layer_name = st.text_input("Layer name:", value="New Layer")

    with col2:
        new_layer_material = st.selectbox(
            "Material:",
            st.session_state.material_db.list_materials(),
            key="new_layer_material"
        )

    with col3:
        new_layer_thickness = st.number_input(
            "Thickness (µm):",
            min_value=0.001,
            max_value=500.0,
            value=1.0,
            key="new_layer_thickness"
        )

    with col4:
        new_layer_type = st.selectbox(
            "Layer type:",
            [lt.value for lt in LayerType],
            key="new_layer_type"
        )

    if st.button("➕ Add Layer", type="primary", use_container_width=True):
        new_layer = Layer(
            name=new_layer_name,
            layer_type=LayerType(new_layer_type),
            material_name=new_layer_material,
            thickness=new_layer_thickness
        )
        architecture.add_layer(new_layer)
        st.success(f"Added layer: {new_layer_name}")
        st.rerun()


def render_material_selector() -> None:
    """Render material database browser."""
    st.subheader("🧪 Materials Database")

    material_db = st.session_state.material_db

    # Filter materials
    material_type_filter = st.selectbox(
        "Filter by type:",
        ["All"] + [mt.value for mt in material_db._materials.values()]
    )

    # List materials
    materials = material_db.list_materials()

    if materials:
        selected_material_name = st.selectbox(
            "Select material to view:",
            materials
        )

        material = material_db.get_material(selected_material_name)

        if material:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Basic Properties:**")
                st.write(f"- Bandgap: {material.bandgap:.2f} eV")
                st.write(f"- Electron Affinity: {material.electron_affinity:.2f} eV")
                st.write(f"- Dielectric Constant: {material.dielectric_constant:.2f}")
                if material.density:
                    st.write(f"- Density: {material.density:.2f} g/cm³")

            with col2:
                if material.electrical:
                    st.write("**Electrical Properties:**")
                    st.write(f"- Electron Mobility: {material.electrical.electron_mobility:.0f} cm²/V·s")
                    st.write(f"- Hole Mobility: {material.electrical.hole_mobility:.0f} cm²/V·s")
                    st.write(f"- Electron Lifetime: {material.electrical.electron_lifetime:.2e} s")
                elif material.conductivity:
                    st.write("**Conductivity:**")
                    st.write(f"- {material.conductivity:.2e} S/cm")

            if material.description:
                st.info(material.description)
    else:
        st.warning("No materials in database")


def render_simulation_controls(architecture: CellArchitecture) -> None:
    """
    Render simulation control panel.

    Args:
        architecture: Cell architecture to simulate
    """
    st.subheader("⚙️ Simulation Controls")

    if not architecture or not architecture.layers:
        st.warning("Please create a cell design with layers first")
        return

    # Simulation parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C):",
            min_value=-50.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Cell operating temperature"
        )

    with col2:
        illumination = st.number_input(
            "Irradiance (W/m²):",
            min_value=0.0,
            max_value=1500.0,
            value=1000.0,
            step=50.0,
            help="Incident light intensity"
        )

    with col3:
        simulator = st.selectbox(
            "Simulator:",
            ["Fast Physics Engine", "SCAPS (Demo)"],
            help="Choose simulation engine"
        )

    # Run simulation
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                # Convert temperature to Kelvin
                temp_k = temperature + 273.15

                if simulator == "Fast Physics Engine":
                    # Use fast device physics engine
                    engine = DevicePhysicsEngine()

                    # Simulate JV curve
                    voltage, current = engine.simulate_jv_curve(
                        architecture,
                        st.session_state.material_db,
                        temp_k,
                        illumination
                    )

                    # Create results object
                    results = SimulationResults()
                    results.voltage = voltage
                    results.current_density = current
                    results.calculate_metrics()

                    # Simulate QE
                    wl, eqe, iqe = engine.simulate_quantum_efficiency(
                        architecture,
                        st.session_state.material_db
                    )
                    results.wavelength = wl
                    results.eqe = eqe
                    results.iqe = iqe

                    # Simplified band diagram
                    total_thick = architecture.get_total_thickness()
                    results.position = np.linspace(0, total_thick, 500)
                    results.conduction_band = -4.05 - 0.2 * np.sin(
                        results.position / total_thick * np.pi
                    )
                    results.valence_band = results.conduction_band - 1.12
                    results.fermi_level = results.conduction_band - 0.3

                    # Loss analysis
                    results.losses = {
                        "Thermalization": 32.0,
                        "Transmission": 5.0,
                        "Reflection": 3.5,
                        "Recombination (bulk)": 2.5,
                        "Recombination (surface)": 1.5,
                        "Contact resistance": 0.5,
                    }

                else:  # SCAPS
                    scaps = SCAPSWrapper()
                    results = scaps.run_simulation(
                        architecture,
                        st.session_state.material_db,
                        temp_k,
                        illumination
                    )

                # Store results
                st.session_state.simulation_results = results

                # Add to history
                st.session_state.design_history.append({
                    'name': architecture.name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'architecture': architecture.to_dict(),
                    'results': {
                        'efficiency': results.efficiency,
                        'voc': results.voc,
                        'jsc': results.jsc,
                        'ff': results.ff
                    }
                })

                st.success("Simulation completed!")
                st.rerun()

    with col2:
        if st.button("📊 Add to Comparison", use_container_width=True):
            if st.session_state.simulation_results:
                st.session_state.comparison_designs.append(
                    (architecture.name, st.session_state.simulation_results)
                )
                st.success(f"Added {architecture.name} to comparison")
            else:
                st.warning("Run simulation first")


def render_performance_metrics(results: SimulationResults) -> None:
    """
    Render performance metrics dashboard.

    Args:
        results: Simulation results
    """
    if not results:
        return

    st.subheader("📈 Performance Metrics")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Efficiency</div>
            <div class="metric-value">{results.efficiency:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Voc</div>
            <div class="metric-value">{results.voc:.3f} V</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Jsc</div>
            <div class="metric-value">{results.jsc:.2f} mA/cm²</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fill Factor</div>
            <div class="metric-value">{results.ff:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Additional metrics
    with st.expander("Additional Metrics"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Vmpp", f"{results.vmpp:.3f} V")
            st.metric("Jmpp", f"{results.jmpp:.2f} mA/cm²")

        with col2:
            st.metric("Pmpp", f"{results.pmpp:.2f} mW/cm²")

        with col3:
            pass


def render_visualization_tabs(
    architecture: CellArchitecture,
    results: SimulationResults
) -> None:
    """
    Render visualization tabs.

    Args:
        architecture: Cell architecture
        results: Simulation results
    """
    tabs = st.tabs([
        "Cell Structure",
        "JV Curve",
        "Quantum Efficiency",
        "Band Diagram",
        "Loss Analysis"
    ])

    with tabs[0]:
        if architecture and architecture.layers:
            fig = plot_cell_cross_section(architecture)
            st.plotly_chart(fig, use_container_width=True)

            # Validation warnings
            issues = architecture.validate_structure()
            if issues:
                st.warning("Design Issues:")
                for issue in issues:
                    st.write(f"- {issue}")
        else:
            st.info("Create a cell design to view structure")

    with tabs[1]:
        if results and len(results.voltage) > 0:
            fig = plot_jv_curve(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to view JV curve")

    with tabs[2]:
        if results and len(results.wavelength) > 0:
            fig = plot_quantum_efficiency(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to view quantum efficiency")

    with tabs[3]:
        if results and len(results.position) > 0:
            fig = plot_band_diagram(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to view band diagram")

    with tabs[4]:
        if results and results.losses:
            fig = plot_loss_waterfall(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to view loss analysis")


def render_comparison_tool() -> None:
    """Render design comparison interface."""
    st.subheader("🔄 Design Comparison")

    if not st.session_state.comparison_designs:
        st.info("Add designs to comparison using the 'Add to Comparison' button")
        return

    # Show comparison table
    comparison_data = []
    for name, results in st.session_state.comparison_designs:
        comparison_data.append({
            "Design": name,
            "Efficiency (%)": f"{results.efficiency:.2f}",
            "Voc (V)": f"{results.voc:.3f}",
            "Jsc (mA/cm²)": f"{results.jsc:.2f}",
            "FF (%)": f"{results.ff:.2f}",
            "Pmpp (mW/cm²)": f"{results.pmpp:.2f}"
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    # Comparison charts
    if len(st.session_state.comparison_designs) > 1:
        fig = plot_design_comparison(st.session_state.comparison_designs)
        st.plotly_chart(fig, use_container_width=True)

    # Clear comparison
    if st.button("Clear Comparison"):
        st.session_state.comparison_designs = []
        st.rerun()


def render_optimization_interface(architecture: CellArchitecture) -> None:
    """
    Render optimization interface.

    Args:
        architecture: Cell architecture to optimize
    """
    st.subheader("🎯 Design Optimization")

    if not architecture or not architecture.layers:
        st.warning("Please create a cell design first")
        return

    # Optimization target
    opt_target = st.selectbox(
        "Optimization target:",
        ["Maximum Efficiency", "Maximum Voc", "Maximum Jsc", "Minimum Cost"],
        help="Choose what to optimize for"
    )

    # Parameter sweep configuration
    with st.expander("Parameter Sweep Configuration", expanded=True):
        param_to_sweep = st.selectbox(
            "Parameter to sweep:",
            ["Layer Thickness", "Doping Concentration", "Temperature"],
            help="Choose which parameter to vary"
        )

        if param_to_sweep == "Layer Thickness":
            layer_names = [layer.name for layer in architecture.layers]
            selected_layer = st.selectbox("Select layer:", layer_names)

            col1, col2, col3 = st.columns(3)
            with col1:
                min_val = st.number_input("Min (µm):", value=0.1, min_value=0.001)
            with col2:
                max_val = st.number_input("Max (µm):", value=10.0, min_value=0.001)
            with col3:
                steps = st.number_input("Steps:", value=20, min_value=5, max_value=100)

    # Run optimization
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        with st.spinner("Running optimization..."):
            progress_bar = st.progress(0)

            # Simulated optimization (replace with real optimization)
            results_list = []
            param_values = np.linspace(min_val, max_val, steps)

            for i, val in enumerate(param_values):
                # Update parameter
                # Run simulation
                # Store result
                progress_bar.progress((i + 1) / steps)

                # Dummy result for demo
                eff = 20.0 + np.random.normal(0, 0.5)
                results_list.append(eff)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=param_values,
                y=results_list,
                mode='lines+markers',
                name='Efficiency'
            ))

            fig.update_layout(
                title="Optimization Results",
                xaxis_title=f"{param_to_sweep} ({selected_layer if param_to_sweep == 'Layer Thickness' else 'units'})",
                yaxis_title="Efficiency (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Find optimum
            opt_idx = np.argmax(results_list)
            st.success(f"Optimum found at {param_values[opt_idx]:.3f} with efficiency {results_list[opt_idx]:.2f}%")


def render_griddler_interface(architecture: CellArchitecture) -> None:
    """
    Render Griddler integration interface.

    Args:
        architecture: Cell architecture
    """
    st.subheader("🔲 Grid Design Optimization")

    if not st.session_state.simulation_results:
        st.warning("Run cell simulation first to optimize grid")
        return

    results = st.session_state.simulation_results

    # Grid parameters
    col1, col2 = st.columns(2)

    with col1:
        num_busbars = st.slider("Number of busbars:", 3, 6, 4)
        num_fingers = st.slider("Number of fingers:", 60, 120, 90)

    with col2:
        finger_width = st.slider("Finger width (mm):", 0.03, 0.08, 0.05, 0.01)
        cell_size = st.selectbox("Cell size:", ["156x156 mm", "166x166 mm", "182x182 mm"])

    if st.button("Optimize Grid", type="primary", use_container_width=True):
        griddler = GriddlerIntegration()

        cell_dim = float(cell_size.split('x')[0])

        grid_results = griddler.optimize_grid(
            cell_width=cell_dim,
            cell_height=cell_dim,
            current_density=results.jsc,
            voltage=results.vmpp,
            num_busbars=num_busbars
        )

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Shading Loss", f"{grid_results.shading_loss:.2f}%")

        with col2:
            st.metric("Resistance Loss", f"{grid_results.resistance_loss:.2f}%")

        with col3:
            st.metric("Total Loss", f"{grid_results.total_loss:.2f}%")

        st.info(f"Optimized pattern: {grid_results.pattern.num_busbars} busbars, "
                f"{grid_results.pattern.num_fingers} fingers")


def render_export_controls(architecture: CellArchitecture) -> None:
    """
    Render export/save controls.

    Args:
        architecture: Cell architecture
    """
    st.subheader("💾 Export & Save")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Design", use_container_width=True):
            if architecture:
                design_json = json.dumps(architecture.to_dict(), indent=2)
                st.download_button(
                    label="Download JSON",
                    data=design_json,
                    file_name=f"{architecture.name.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    with col2:
        if st.button("📊 Export Results", use_container_width=True):
            if st.session_state.simulation_results:
                results = st.session_state.simulation_results

                # Create CSV
                df = pd.DataFrame({
                    'Voltage (V)': results.voltage,
                    'Current Density (mA/cm2)': results.current_density
                })

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="simulation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col3:
        uploaded_file = st.file_uploader(
            "Load Design",
            type=['json'],
            help="Upload a previously saved design"
        )

        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                loaded_arch = CellArchitecture.from_dict(data)
                st.session_state.current_architecture = loaded_arch
                st.success(f"Loaded design: {loaded_arch.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading design: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Main application entry point."""
    # Load CSS
    load_custom_css()

    # Initialize session state
    initialize_session_state()

    # Render header
    render_header()

    # Sidebar
    with st.sidebar:
        st.title("Navigation")

        page = st.radio(
            "Select page:",
            [
                "Cell Designer",
                "Results & Analysis",
                "Design Comparison",
                "Optimization",
                "Grid Design",
                "Materials Database",
                "Template Gallery"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        if st.session_state.current_architecture:
            st.write("**Current Design:**")
            st.write(f"Name: {st.session_state.current_architecture.name}")
            st.write(f"Layers: {len(st.session_state.current_architecture.layers)}")
            st.write(f"Total: {st.session_state.current_architecture.get_total_thickness():.2f} µm")

        if st.session_state.simulation_results:
            st.write("**Latest Results:**")
            st.write(f"η: {st.session_state.simulation_results.efficiency:.2f}%")
            st.write(f"Voc: {st.session_state.simulation_results.voc:.3f} V")
            st.write(f"Jsc: {st.session_state.simulation_results.jsc:.2f} mA/cm²")

    # Main content area
    if page == "Cell Designer":
        col1, col2 = st.columns([1, 1])

        with col1:
            # Architecture selector
            new_arch = render_architecture_selector()
            if new_arch:
                st.session_state.current_architecture = new_arch

            # Layer editor
            if st.session_state.current_architecture:
                render_layer_stack_editor(st.session_state.current_architecture)

        with col2:
            # Simulation controls
            if st.session_state.current_architecture:
                render_simulation_controls(st.session_state.current_architecture)

            # Performance metrics
            if st.session_state.simulation_results:
                render_performance_metrics(st.session_state.simulation_results)

    elif page == "Results & Analysis":
        if st.session_state.current_architecture or st.session_state.simulation_results:
            render_visualization_tabs(
                st.session_state.current_architecture,
                st.session_state.simulation_results
            )
        else:
            st.info("Create and simulate a design to view results")

    elif page == "Design Comparison":
        render_comparison_tool()

    elif page == "Optimization":
        if st.session_state.current_architecture:
            render_optimization_interface(st.session_state.current_architecture)
        else:
            st.info("Create a design first")

    elif page == "Grid Design":
        render_griddler_interface(st.session_state.current_architecture)

    elif page == "Materials Database":
        render_material_selector()

    elif page == "Template Gallery":
        st.subheader("📚 Cell Design Templates")

        for template_name, template in st.session_state.templates.items():
            with st.expander(f"{template_name} - {template.description}"):
                st.write(f"**Layers:** {len(template.layers)}")
                st.write(f"**Total thickness:** {template.get_total_thickness():.3f} µm")

                # Quick estimate
                has_pass = any('passivation' in l.layer_type.lower() for l in template.layers)
                has_arc = any('arc' in l.layer_type.lower() for l in template.layers)
                estimate = quick_efficiency_estimate(
                    template.architecture_type,
                    has_pass,
                    has_arc
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Est. Efficiency", f"{estimate['efficiency']:.1f}%")
                with col2:
                    st.metric("Est. Voc", f"{estimate['voc']:.3f} V")
                with col3:
                    st.metric("Est. Jsc", f"{estimate['jsc']:.1f} mA/cm²")

                if st.button(f"Load {template_name}", key=f"load_{template_name}"):
                    st.session_state.current_architecture = template.clone()
                    st.success(f"Loaded {template_name}")
                    st.rerun()

    # Footer
    st.markdown("---")
    render_export_controls(st.session_state.current_architecture)


if __name__ == "__main__":
    main()
