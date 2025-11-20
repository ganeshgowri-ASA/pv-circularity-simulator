# PV Cell Design UI & Visualization

Comprehensive Streamlit interface for designing, simulating, and analyzing photovoltaic cell architectures.

## Features

### 1. Interactive Cell Designer Interface
- **Cell Architecture Selector**: Choose from Al-BSF, PERC, TOPCon, HJT, IBC architectures
- **Layer Stack Builder**: Add, remove, reorder layers with intuitive controls
- **Material Property Inputs**: Full material database with validation
- **Real-time Efficiency Estimation**: Quick performance estimates
- **SCAPS Simulation Launcher**: Integration with SCAPS for detailed physics

### 2. Visualization Components
- **Cross-sectional Cell Diagram**: Color-coded layer visualization
- **J-V Curve Plotter**: Interactive Plotly charts with MPP markers
- **Quantum Efficiency Charts**: EQE and IQE spectrum visualization
- **Band Diagram Visualization**: Energy band structure across device
- **Loss Mechanism Waterfall**: Breakdown of efficiency losses

### 3. Design Comparison Tool
- **Side-by-side Comparison**: Compare multiple architectures
- **Performance Metrics Table**: Tabular comparison of key metrics
- **Interactive Charts**: Multi-panel performance comparison

### 4. Optimization Interface
- **Parameter Sweep Configuration**: Vary design parameters systematically
- **Optimization Target Selection**: Optimize for efficiency, Voc, Jsc, or cost
- **Progress Visualization**: Real-time optimization progress
- **Results Analysis**: Interactive result visualization

### 5. Cell Template Gallery
- **Pre-configured Designs**: Commercial cell architectures (Al-BSF, PERC, TOPCon, HJT)
- **Save/Load Custom Designs**: Export and import JSON designs
- **Design History Tracking**: Automatically track simulation history
- **Quick Efficiency Estimates**: Instant performance estimates for templates

### 6. Integration
- **SCAPS Wrapper**: Interface to Solar Cell Capacitance Simulator
- **Device Physics Engine**: Fast analytical/semi-analytical simulation
- **Griddler Integration**: Front contact grid optimization
- **Materials Database**: Comprehensive material property database

### 7. Additional Features
- **Responsive Layout**: Multi-column layout with st.columns
- **Custom CSS Styling**: Professional color scheme and card designs
- **Loading States**: Progress bars and spinners for long operations
- **Error Handling**: Input validation and user-friendly error messages
- **Full Documentation**: Complete docstrings and type hints

## Installation

```bash
# Clone the repository
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
# From the project root directory
streamlit run src/ui/cell_design.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Quick Start Guide

1. **Start with a Template**:
   - Select "Cell Designer" from the sidebar
   - Choose "Start from template" and select a cell architecture (e.g., "PERC")
   - Click "Load Template"

2. **Customize the Design**:
   - Use the Layer Stack Editor to modify layers
   - Adjust thicknesses, materials, and doping concentrations
   - Add or remove layers as needed

3. **Run Simulation**:
   - Set simulation parameters (temperature, irradiance)
   - Click "Run Simulation"
   - View performance metrics in real-time

4. **Analyze Results**:
   - Navigate to "Results & Analysis" page
   - Explore JV curves, QE spectra, band diagrams, and loss analysis
   - Export results as needed

5. **Compare Designs**:
   - Run simulations for multiple designs
   - Use "Add to Comparison" button
   - View side-by-side comparison in "Design Comparison" page

6. **Optimize**:
   - Go to "Optimization" page
   - Configure parameter sweeps
   - Run optimization to find optimal design parameters

7. **Grid Design**:
   - After cell simulation, navigate to "Grid Design"
   - Configure front contact grid parameters
   - Optimize for minimum losses

## Architecture

### Project Structure

```
src/
├── core/
│   ├── data_models/          # Pydantic models for materials, layers, cells
│   │   ├── materials.py      # Material and MaterialDatabase classes
│   │   └── cell_architecture.py  # Layer and CellArchitecture classes
│   └── constants.py          # Physical constants and defaults
│
├── integrations/
│   ├── scaps/                # SCAPS wrapper
│   │   └── wrapper.py        # SCAPSWrapper and SimulationResults
│   ├── griddler/             # Grid optimization
│   │   └── integration.py    # GriddlerIntegration
│   └── materials_db/         # Materials database
│       └── database.py       # MaterialDatabase loader
│
├── simulation/
│   └── device_physics.py     # Fast physics engine
│
└── ui/
    └── cell_design.py        # Main Streamlit application
```

### Data Models

**Material**: Comprehensive material properties (optical, electrical, physical)
**Layer**: Single layer definition (material, thickness, doping)
**CellArchitecture**: Complete cell stack with layers
**SimulationResults**: Container for simulation outputs

### Simulation Engines

1. **Fast Physics Engine**: Analytical/semi-analytical models for quick feedback
   - Two-diode JV model
   - Simplified QE calculation
   - Basic band bending

2. **SCAPS Wrapper**: Interface to detailed device simulator (demo mode)
   - Input file generation
   - Process execution
   - Output parsing

## Materials Database

The application includes a comprehensive materials database with:

- **Semiconductors**: Si(p+), Si(p), Si(n), Si(n+), a-Si:H variants, Poly-Si
- **TCOs**: ITO, AZO
- **Passivation**: SiO2, Al2O3, SiNx
- **Metals**: Al, Ag

Each material includes:
- Bandgap and electron affinity
- Electrical properties (mobility, lifetime)
- Optical properties (n, k, absorption)
- Default parameters for simulation

## Cell Templates

### Al-BSF (18-19% efficiency)
Traditional aluminum back surface field cell with full-area rear contact.

### PERC (21-22% efficiency)
Passivated emitter and rear cell with Al2O3 rear passivation and local contacts.

### TOPCon (23-24% efficiency)
Tunnel oxide passivated contact with ultra-thin SiO2 and doped poly-Si.

### HJT (24-25% efficiency)
Heterojunction with intrinsic thin layer using a-Si:H passivation and ITO contacts.

## Export Formats

- **Design Export**: JSON format with complete architecture
- **Results Export**: CSV format with JV curve data
- **Comparison Tables**: Downloadable performance comparison

## Customization

### Adding Custom Materials

Materials can be added programmatically to the database:

```python
from src.core.data_models.materials import Material, MaterialType
from src.integrations.materials_db.database import get_materials_database

# Get database
db = get_materials_database()

# Create custom material
custom_mat = Material(
    name="Custom-Si",
    material_type=MaterialType.SEMICONDUCTOR,
    bandgap=1.12,
    electron_affinity=4.05,
    # ... other properties
)

# Add to database
db.add_material(custom_mat)
```

### Creating Custom Templates

Templates are defined in `load_default_templates()` function. Add new templates by creating `CellArchitecture` objects with appropriate layers.

## Performance Tips

1. Use "Fast Physics Engine" for interactive design and parameter exploration
2. Use SCAPS integration for final detailed analysis (when available)
3. Enable parameter sweep caching for optimization runs
4. Export intermediate results to avoid re-simulation

## Troubleshooting

### Import Errors
Ensure all dependencies are installed: `pip install -r requirements.txt`

### Streamlit Issues
Update Streamlit to latest version: `pip install --upgrade streamlit`

### Simulation Errors
Check that cell architecture has:
- At least one layer
- Valid material assignments
- Reasonable thickness values (0.001 - 500 µm)
- Proper doping concentrations (1e10 - 1e22 cm⁻³)

## Future Enhancements

- [ ] Real SCAPS integration (currently demo mode)
- [ ] Advanced optical modeling (transfer matrix method)
- [ ] 2D/3D device simulation
- [ ] Temperature-dependent parameters
- [ ] Spectral irradiance support
- [ ] Reliability modeling integration
- [ ] Cost modeling and LCOE calculation
- [ ] Multi-objective optimization
- [ ] Collaborative design sharing

## References

- SCAPS Manual: https://scaps.elis.ugent.be/
- PV Device Physics: Green, M.A., "Solar Cells"
- Silicon Solar Cells: Luque & Hegedus, "Handbook of Photovoltaic Science"

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue on GitHub.
