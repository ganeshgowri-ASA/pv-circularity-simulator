# PV Circularity Simulator

End-to-End PV Lifecycle Simulation Platform with comprehensive module design and analysis tools.

## Features

### Module Design UI (`src/ui/module_design.py`)

A production-ready Streamlit interface providing:

1. **Interactive Module Designer**
   - Cell selection from pre-defined templates (PERC, TOPCon, HJT, IBC)
   - Layout configuration (60/72/120/132/144 cells)
   - Cell configurations (full-cell, half-cut, quarter-cut, shingled)
   - String and bypass diode configuration
   - Module dimensions calculator
   - Real-time power rating estimation

2. **CTM Loss Visualization**
   - Interactive k-factor editor (k1-k24)
   - Loss waterfall chart showing Cell-to-Module losses
   - Sensitivity analysis for individual k-factors
   - Cell-to-Module ratio display
   - Optical, electrical, thermal, and manufacturing loss breakdown

3. **Module Configuration Tools**
   - BOM (Bill of Materials) generator
   - Cost calculator with detailed breakdown
   - Weight calculator with component analysis
   - CSV export functionality
   - LCOE contribution analysis

4. **Performance Prediction**
   - STC (Standard Test Conditions) power output
   - NOCT (Nominal Operating Cell Temperature) performance
   - Temperature coefficient effects visualization
   - Bifacial gain estimation with albedo sensitivity
   - 25-year degradation projections
   - Warranty compliance checking

5. **PAN File Generator**
   - PVsyst-compatible PAN file export
   - Preview generated PAN file
   - Download functionality
   - Validation against PVsyst format

6. **Module Comparison Tool**
   - Side-by-side comparison (up to 4 modules)
   - Performance metrics table
   - Cost per watt analysis
   - Technology comparison visualizations

7. **3D Visualization**
   - Module cross-section view
   - Cell layout visualization
   - Layer stack representation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── models/          # Data models (cell, module, material)
│   ├── ctm/             # Cell-to-Module calculators
│   ├── bom/             # BOM, cost, weight calculators
│   ├── performance/     # Performance prediction & degradation
│   ├── pan/             # PAN file generator
│   ├── templates/       # Cell templates
│   └── ui/              # Streamlit UI
│       └── module_design.py  # Main module design interface
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Technology Stack

- **Streamlit**: Interactive web interface
- **Pydantic**: Data validation and models
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

## Cell Templates

Pre-defined templates available:
- M10 PERC Monocrystalline
- M10 TOPCon Bifacial
- M12 TOPCon Bifacial
- M6 PERC Monocrystalline
- M10 HJT (Heterojunction) Bifacial
- M10 IBC (Interdigitated Back Contact)
- G12 PERC Multicrystalline

## CTM K-Factors

The system models 24 different loss mechanisms:

**Optical Losses (k1-k8):**
- Reflection, glass transmission, encapsulant, soiling, spectral mismatch, angular losses, inactive area, optical coupling

**Electrical Losses (k9-k16):**
- Interconnection, series resistance, shunt resistance, cell mismatch, diode losses, junction box, cable resistance, contact resistance

**Thermal Losses (k17-k20):**
- Thermal mismatch, heat dissipation, NOCT effect, hot spot risk

**Manufacturing Losses (k21-k24):**
- Manufacturing tolerance, lamination quality, edge deletion, measurement uncertainty

## Usage Example

```python
from src.templates.cell_templates import get_cell_templates
from src.models.cell import CellDesign
from src.ctm.calculator import CTMCalculator

# Get cell template
templates = get_cell_templates()
cell_template = templates["M10_TOPCon_Mono_Bifacial"]

# Create cell design
cell_design = CellDesign(
    template=cell_template,
    quantity=144,
    configuration="half-cut"
)

# Calculate CTM
ctm_calc = CTMCalculator()
result = ctm_calc.calculate(
    cell_pmax_w=cell_template.pmax_w,
    num_cells=144,
    cell_configuration="half-cut"
)

print(f"Module Power: {result.module_pmax_w:.2f} W")
print(f"CTM Ratio: {result.ctm_ratio:.4f}")
```

## License

See LICENSE file for details.

## Contributing

This project is part of the PV Circularity Simulator initiative for end-to-end photovoltaic lifecycle analysis.
