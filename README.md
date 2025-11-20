# pv-circularity-simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Components

### Module Configuration Builder

Comprehensive PV module design and analysis tool (`src/modules/module_builder.py`).

**Features:**
- Multiple layout types: Standard, Half-Cut, Quarter-Cut, Shingled, IBC, Bifacial
- Cell technologies: Mono PERC, TOPCon, HJT, IBC, Multi-Si, Perovskite, Tandem
- Complete electrical specifications calculator
- CTM (Cell-to-Module) loss analysis
- PVsyst PAN file generator
- Design validation and optimization
- Export to JSON, CSV, PAN formats

**Quick Start:**

```python
from src.modules import create_standard_module, ModuleConfigBuilder, CellType, LayoutType

# Create a standard 450W module
module = create_standard_module(450, CellType.MONO_PERC, LayoutType.HALF_CUT)

# Calculate specifications
builder = ModuleConfigBuilder()
specs = builder.calculate_module_specs(module)
print(f"Power: {specs.pmax:.1f}W, Efficiency: {specs.efficiency*100:.2f}%")

# Generate PVsyst PAN file
pan_content = builder.generate_pvsyst_pan_file(module)
```

**Documentation:** See `src/modules/README.md` for comprehensive documentation.

**Examples:** Run `python3 examples/module_builder_demo.py` for full demonstration.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python3 examples/module_builder_demo.py
```

## Requirements

- Python 3.11+
- Pydantic 2.0+

## License

MIT License - See LICENSE file for details.
