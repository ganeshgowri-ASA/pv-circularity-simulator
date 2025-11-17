# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### B10-S01: Asset Management & Portfolio Tracking (Production-Ready)

Comprehensive asset management system for PV installations with support for circular economy principles.

**Core Components:**
- **AssetManager**: Central management class for sites, equipment, and performance tracking
- **Site Inventory**: Track and manage solar installation sites with capacity and status monitoring
- **Equipment Tracking**: Detailed equipment management with lifecycle and circular economy features
- **Performance History**: Historical performance data tracking and analysis

**Key Capabilities:**
- Site portfolio management with geographic tracking
- Equipment lifecycle tracking (manufacturing → installation → maintenance → decommissioning)
- Performance monitoring with environmental conditions
- Material composition tracking for circular economy
- Recyclability and end-of-life value estimation
- Comprehensive inventory summaries and analytics

## Installation

### Requirements

- Python 3.9 or higher
- SQLAlchemy 2.0+
- Pydantic 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Usage

```python
from datetime import datetime
from src.pv_circularity.database.session import init_db
from src.pv_circularity.managers.asset_manager import AssetManager
from src.pv_circularity.database.models import AssetStatus, EquipmentType
from src.pv_circularity.models.schemas import SiteCreate, EquipmentCreate

# Initialize database
db_manager = init_db("sqlite:///./pv_circularity.db", create_tables=True)
db_session = next(db_manager.get_session())
asset_manager = AssetManager(db_session)

# Create a site
site_data = SiteCreate(
    name="Solar Farm Alpha",
    location="Phoenix, AZ",
    latitude=33.4484,
    longitude=-112.0740,
    capacity_kw=5000.0,
    installation_date=datetime(2020, 1, 1),
    status=AssetStatus.ACTIVE
)
site = asset_manager.create_site(site_data)

# Add equipment
equipment_data = EquipmentCreate(
    equipment_id="PANEL-001",
    site_id=site.id,
    equipment_type=EquipmentType.SOLAR_PANEL,
    name="High Efficiency Panel",
    manufacturer="SolarTech",
    model="ST-550W",
    installation_date=datetime(2020, 1, 1),
    rated_power_w=550.0,
    efficiency_percent=22.5,
    recyclable=True,
    material_composition={"silicon": 0.4, "glass": 0.3, "aluminum": 0.2, "other": 0.1}
)
equipment = asset_manager.create_equipment(equipment_data)

# Get site inventory
inventory = asset_manager.site_inventory(include_summary=True)
print(f"Total Sites: {inventory['summary'].total_sites}")
print(f"Total Capacity: {inventory['summary'].total_capacity_kw} kW")

# Track equipment
tracking = asset_manager.equipment_tracking(include_summary=True)
print(f"Total Equipment: {tracking['summary'].total_equipment}")

# View performance history
history = asset_manager.performance_history(include_summary=True)
print(f"Total Energy: {history['summary'].total_energy_kwh} kWh")
```

### Core Methods

#### site_inventory()
Get comprehensive site inventory with summary statistics.

```python
inventory = asset_manager.site_inventory(
    status=AssetStatus.ACTIVE,
    include_summary=True
)
# Returns: {'sites': [...], 'summary': SiteInventorySummary(...)}
```

#### equipment_tracking()
Track equipment with filtering and analytics.

```python
tracking = asset_manager.equipment_tracking(
    site_id=1,
    equipment_type=EquipmentType.SOLAR_PANEL,
    status=AssetStatus.ACTIVE,
    include_summary=True
)
# Returns: {'equipment': [...], 'summary': EquipmentInventorySummary(...)}
```

#### performance_history()
Retrieve performance data with time-based filtering.

```python
history = asset_manager.performance_history(
    site_id=1,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    include_summary=True
)
# Returns: {'records': [...], 'summary': PerformanceHistorySummary(...)}
```

## Database Schema

### Sites
- Site metadata (name, location, coordinates)
- Capacity and installation information
- Status tracking (planned, active, maintenance, decommissioned)

### Equipment
- Equipment identification and specifications
- Technical parameters (power, efficiency, degradation)
- Lifecycle tracking (manufacturing → end-of-life)
- Circular economy attributes (recyclability, material composition)

### Performance Records
- Energy generation and power output
- Efficiency and capacity factor
- Environmental conditions (irradiance, temperature, wind)
- System health metrics

### Assets
- Generic asset tracking
- Financial information (cost, current value)
- Warranty and lifetime management

## Circular Economy Features

### Material Composition Tracking
Track material composition of equipment for recycling assessment:

```python
equipment_data = EquipmentCreate(
    # ... other fields ...
    recyclable=True,
    material_composition={
        "silicon": 0.35,
        "glass": 0.30,
        "aluminum": 0.20,
        "copper": 0.05,
        "other": 0.10
    },
    recycling_value=50.0
)
```

### Lifecycle Management
- Manufacturing date tracking
- Installation and commissioning dates
- Maintenance scheduling
- Expected lifetime estimation
- Degradation rate monitoring
- End-of-life planning

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_circularity --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_asset_manager.py

# Run specific test class
pytest tests/unit/test_asset_manager.py::TestSiteManagement
```

## Examples

See the `examples/` directory for complete usage examples:

- `asset_management_example.py`: Comprehensive demonstration of AssetManager features

Run the example:

```bash
python examples/asset_management_example.py
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity/
│       ├── database/           # Database models and session management
│       │   ├── models.py       # SQLAlchemy ORM models
│       │   └── session.py      # Database session management
│       ├── managers/           # Business logic managers
│       │   └── asset_manager.py  # AssetManager implementation
│       ├── models/             # Pydantic schemas
│       │   └── schemas.py      # Validation and serialization schemas
│       └── utils/              # Utility functions
├── tests/
│   ├── unit/                   # Unit tests
│   │   └── test_asset_manager.py
│   ├── integration/            # Integration tests
│   └── conftest.py            # Pytest fixtures
├── examples/                   # Usage examples
│   └── asset_management_example.py
├── pyproject.toml             # Project configuration
├── requirements.txt           # Production dependencies
└── requirements-dev.txt       # Development dependencies
```

## Technology Stack

- **Database**: SQLAlchemy 2.0+ (SQLite, PostgreSQL compatible)
- **Validation**: Pydantic 2.0+ with comprehensive schemas
- **Testing**: pytest with coverage
- **Code Quality**: ruff, black, mypy

## Roadmap

### Completed (B10-S01)
- ✅ Asset Management & Portfolio Tracking
- ✅ Site Inventory Management
- ✅ Equipment Tracking with Circular Economy Features
- ✅ Performance History Tracking

### Upcoming
- **B10-S02**: Repower Analysis & Technology Upgrade
  - Technology comparison engine
  - Upgrade scenario modeling
  - ROI analysis
- **B10-S03**: ROI Calculations & Financial Analysis
  - Payback period calculation
  - NPV and IRR analysis
  - Sensitivity analysis
- **B10-S04**: Revamp Planning UI & Project Management
  - Streamlit-based interface
  - Project timeline management
  - Budget and vendor tracking

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows style guidelines (ruff, black)
- Type hints are included (mypy)
- Documentation is updated

## License

Apache-2.0 License - See LICENSE file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub.
