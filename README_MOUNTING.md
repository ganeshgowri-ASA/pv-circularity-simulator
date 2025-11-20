# PV Mounting Structure Design Module

Comprehensive mounting structure design and engineering module for all PV mounting types with ASCE 7-22 structural calculations.

## Features

### Supported Mounting Types

1. **Ground Mount Systems**
   - Fixed-tilt racking (1P, 2P, 3P, 4P configurations)
   - Single-axis trackers with backtracking algorithms
   - Dual-axis trackers with sun tracking
   - Optimized row spacing and GCR calculations

2. **Rooftop Systems**
   - Flat roof (ballasted and attached systems)
   - Pitched roof (rail-mounted, shared rail)
   - Fire setback calculations per IFC/NFPA
   - Roof load capacity verification

3. **Carport/Canopy Systems**
   - Single cantilever design
   - Double cantilever design
   - Four-post canopy structures
   - ADA-compliant clearance heights

4. **Floating PV Systems**
   - HDPE pontoon layout
   - Mooring and anchoring design
   - Wave impact analysis
   - Evaporative cooling benefits

5. **Agrivoltaic Systems**
   - High-clearance structures (4-6m)
   - Crop-specific row spacing
   - Bifacial module optimization
   - Equipment access pathways

6. **Building-Integrated PV (BIPV)**
   - Facade integration
   - Skylight/canopy systems
   - Curtain wall integration
   - Thermal performance analysis

## Structural Calculations

### ASCE 7-22 Compliance
- **Wind Load Analysis** (Chapters 26-29)
  - Velocity pressure coefficients (Kz)
  - Exposure categories (B, C, D)
  - Rooftop wind pressure zones
  - Module pressure coefficients

- **Snow Load Analysis** (Chapter 7)
  - Ground snow load conversion
  - Roof snow loads with slope factor
  - Drift load calculations
  - Exposure and thermal factors

- **Seismic Analysis** (Chapters 11-12)
  - Seismic design categories (A-F)
  - Response modification factors
  - Base shear calculations

- **Load Combinations** (Chapter 2)
  - LRFD strength design combinations
  - Governing load case determination

### Foundation Engineering
- Driven pile design (capacity, depth)
- Helical pile design (torque requirements)
- Ballasted systems (weight, sliding resistance)
- Spread footings (bearing capacity)
- Geotechnical requirements
- Frost heave protection

### Structural Members
- Beam sizing (steel W-beams, timber)
- Column design (HSS sections)
- Deflection analysis (L/180, L/240 limits)
- Connection design (bolted, welded)
- Material specifications

## Installation

```bash
# Install from source
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from pv_simulator.mounting_structure import (
    GroundMountDesign,
    GroundMountConfig,
    SiteParameters,
    ModuleDimensions,
    MountingType,
    ModuleOrientation,
    RackingConfiguration,
    FoundationType,
    ExposureCategory,
    SeismicDesignCategory,
    SoilType,
)

# Define site parameters
site = SiteParameters(
    latitude=35.0,
    longitude=-95.0,
    elevation=300.0,
    wind_speed=35.0,  # m/s
    exposure_category=ExposureCategory.C,
    ground_snow_load=1.0,  # kN/m²
    seismic_category=SeismicDesignCategory.C,
    soil_type=SoilType.SAND,
    frost_depth=0.8,  # m
)

# Define module dimensions
module = ModuleDimensions(
    length=2.0,  # m
    width=1.0,  # m
    thickness=0.04,  # m
    weight=25.0,  # kg
)

# Configure ground mount system
config = GroundMountConfig(
    mounting_type=MountingType.GROUND_FIXED_TILT,
    site_parameters=site,
    module_dimensions=module,
    num_modules=1000,
    tilt_angle=30.0,
    azimuth=180.0,
    orientation=ModuleOrientation.PORTRAIT,
    racking_config=RackingConfiguration.TWO_PORTRAIT,
    foundation_type=FoundationType.DRIVEN_PILE,
    post_spacing=3.0,
)

# Design the system
designer = GroundMountDesign(config)
result = designer.fixed_tilt_structure()

# Access results
print(f"Total steel weight: {result.total_steel_weight:.0f} kg")
print(f"Total cost estimate: ${result.total_cost_estimate:,.2f}")
print(f"Wind uplift: {result.load_analysis.wind_load_uplift:.2f} kN/m²")
print(f"Foundation type: {result.foundation_design.foundation_type.value}")
```

### Streamlit UI

Launch the interactive web interface:

```bash
streamlit run ui/mounting_structure_ui.py
```

Features:
- Interactive mounting type selector
- Site parameter input forms
- Real-time structural calculations
- Load analysis visualization
- Bill of Materials export
- Cost estimation
- 3D layout visualization

## Design Classes

### GroundMountDesign
- `fixed_tilt_structure()`: Design fixed-tilt racking
- `single_axis_tracker()`: Design single-axis tracker with backtracking
- `dual_axis_tracker()`: Design dual-axis tracker
- `calculate_row_spacing()`: Optimize row spacing for GCR
- `foundation_design()`: Complete foundation design
- `calculate_post_spacing()`: Determine optimal post spacing
- `racking_bom()`: Generate bill of materials

### RooftopMountDesign
- `flat_roof_design()`: Ballasted or attached flat roof systems
- `pitched_roof_design()`: Rail-mounted pitched roof systems
- `calculate_roof_loading()`: Total roof load analysis
- `attachment_point_design()`: Roof penetration design
- `setback_requirements()`: Fire setback per IFC
- `wind_zone_analysis()`: ASCE 7 wind zones for rooftop
- `structural_capacity_check()`: Verify roof capacity

### CarportCanopyDesign
- `single_cantilever_carport()`: Single-sided cantilever design
- `double_cantilever_carport()`: Center-post cantilever design
- `four_post_canopy()`: Traditional four-post structure
- `calculate_beam_sizing()`: Steel or timber beam sizing
- `column_foundation()`: Column foundation design
- `clearance_requirements()`: ADA compliance check
- `drainage_design()`: Rainwater management

### FloatingPVDesign
- `pontoon_layout()`: HDPE pontoon spacing and layout
- `anchoring_system()`: Mooring and anchoring design
- `wave_impact_analysis()`: Wave loads and freeboard
- `cooling_benefit_modeling()`: Evaporative cooling effects
- `environmental_considerations()`: Aquatic impact assessment
- `tilt_angle_optimization()`: Low-tilt optimization

### AgrivoltaicDesign
- `high_clearance_structure()`: Elevated structure design
- `row_spacing_for_crops()`: Crop sunlight optimization
- `bifacial_agrivoltaic()`: Bifacial module design
- `crop_specific_design()`: Crop-specific parameters
- `irrigation_integration()`: Irrigation system integration
- `seasonal_tilt_adjustment()`: Adjustable tilt design

### BIPVDesign
- `facade_integration()`: Building facade integration
- `skylight_canopy()`: Translucent skylight systems
- `curtain_wall_integration()`: Curtain wall BIPV
- `structural_glazing()`: Glass-glass module analysis
- `thermal_performance()`: U-value and SHGC
- `electrical_integration()`: Conduit and junction box placement

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=src/pv_simulator --cov-report=html

# Specific test file
pytest tests/mounting_structure/test_structural_calculator.py
```

## Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Documentation

All classes and methods include comprehensive docstrings with:
- Function description
- Parameter descriptions with types
- Return value descriptions
- Usage examples

## Example Output

### Load Analysis
```
Dead Load: 0.15 kN/m²
Live Load: 0.50 kN/m²
Wind Uplift: -2.35 kN/m²
Wind Downward: 1.82 kN/m²
Snow Load: 0.68 kN/m²
Total Load Combination: 3.24 kN/m²
```

### Foundation Design
```
Type: Driven Pile
Depth: 2.5 m
Diameter: 0.15 m
Capacity: 48.5 kN
Spacing: 3.0 m
Quantity: 850 foundations
```

### Bill of Materials
```
Item         Description              Qty    Unit   Cost
FND-001      Driven pile foundation   850    ea     $63,750
STR-001      Purlin C6x8.2           2,400   ea     $120,000
STR-002      Post HSS4x4x1/4         850     ea     $29,750
HW-001       Module clamps           4,000   ea     $10,000

Total Cost: $223,500
Total Weight: 125,400 kg
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- ASCE 7-22: Minimum Design Loads for Buildings and Other Structures
- IBC 2021: International Building Code
- IFC 2021: International Fire Code
- AAMA: American Architectural Manufacturers Association standards
- NEC Article 690: Solar Photovoltaic Systems
