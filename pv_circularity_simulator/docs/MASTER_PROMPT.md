# PV Circularity Simulator - Master Prompt

## Vision and Purpose

The **PV Circularity Simulator** is a comprehensive, end-to-end tool for simulating the complete lifecycle of photovoltaic (PV) systems, from initial design through operational monitoring to end-of-life circularity strategies. This application bridges the gap between traditional PV system design tools and circular economy principles, enabling stakeholders to make informed decisions that optimize both performance and sustainability.

### Core Objectives

1. **Design Optimization**: Enable data-driven material selection and module design with circularity considerations
2. **Performance Simulation**: Provide accurate energy yield and performance predictions
3. **Operational Excellence**: Support real-time monitoring, diagnostics, and maintenance optimization
4. **Circular Economy**: Integrate reduce-reuse-recycle (3R) strategies throughout the system lifecycle
5. **Financial Viability**: Demonstrate economic benefits of circular PV system design and operation

## Application Scope

### Target Users

- **PV System Designers**: Engineers designing new PV installations
- **Asset Managers**: Operators managing existing PV portfolios
- **Sustainability Officers**: Professionals implementing circular economy strategies
- **Researchers**: Academic and industry researchers studying PV sustainability
- **Policy Makers**: Government officials developing PV recycling regulations

### Technology Stack

- **Framework**: Streamlit (Python-based web application)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **PV Calculations**: PVlib
- **Validation**: Pydantic
- **Scientific Computing**: SciPy

## Module Architecture

The application consists of 12 interconnected modules organized in a logical workflow:

### 1. Dashboard
**Purpose**: Central hub for project management and overview
**Features**:
- Project creation and management
- Workflow progress tracking
- Quick access to all modules
- Key performance indicators summary

### 2. Material Selection
**Purpose**: Select and evaluate PV module materials
**Features**:
- Material database browsing
- Property comparison
- Circularity scoring
- Cost-benefit analysis
- Carbon footprint calculation

**Data Flow Output**:
- Selected materials with properties
- Recyclability scores
- Cost estimates
- Carbon footprint data

### 3. Module Design
**Purpose**: Design PV module specifications
**Features**:
- Cell technology selection
- Module configuration (series/parallel)
- Electrical parameter calculation
- Physical dimension specification
- Temperature coefficient definition

**Data Flow**:
- **Input**: Material selections
- **Output**: Complete module design specification
- **Key Parameters**: Pmax, Voc, Isc, Vmpp, Impp, efficiency

### 4. CTM Loss Analysis
**Purpose**: Analyze Cell-to-Module (CTM) losses
**Features**:
- Individual k-factor adjustment (k1-k15, k21-k24)
- Loss category analysis
- Total CTM ratio calculation
- Sensitivity analysis
- Benchmark comparison

**CTM K-Factors**:

#### Optical Losses (k1-k3)
- **k1**: Reflection losses (0.90-0.99, default 0.98)
- **k2**: Shading losses from busbars/fingers (0.90-0.99, default 0.97)
- **k3**: Absorption losses in encapsulant/glass (0.95-0.995, default 0.99)

#### Electrical Losses (k4-k6)
- **k4**: Resistive losses in metallization (0.95-0.99, default 0.98)
- **k5**: Cell mismatch losses (0.95-0.99, default 0.98)
- **k6**: Junction box losses (0.99-0.999, default 0.995)

#### Thermal Losses (k7-k8)
- **k7**: Temperature coefficient effects (0.90-0.98, default 0.96)
- **k8**: Hotspot losses (0.95-0.995, default 0.99)

#### Assembly Losses (k9-k10)
- **k9**: Encapsulation losses (0.97-0.995, default 0.99)
- **k10**: Lamination process losses (0.99-0.999, default 0.995)

#### Degradation Factors (k11-k15)
- **k11**: Light-Induced Degradation (LID) (0.95-0.99, default 0.98)
- **k12**: Potential-Induced Degradation (PID) (0.95-0.995, default 0.99)
- **k13**: Mechanical stress losses (0.99-0.999, default 0.995)
- **k14**: Cell degradation (0.99-0.999, default 0.995)
- **k15**: Interconnect degradation (0.99-0.999, default 0.995)

#### Environmental Factors (k21-k24)
- **k21**: Humidity-induced degradation (0.95-0.995, default 0.99)
- **k22**: UV exposure effects (0.95-0.995, default 0.99)
- **k23**: Thermal cycling stress (0.99-0.999, default 0.995)
- **k24**: Corrosion effects (0.99-0.999, default 0.995)

**Total CTM Ratio Calculation**:
```
CTM_total = k1 × k2 × k3 × k4 × k5 × k6 × k7 × k8 × k9 × k10 × k11 × k12 × k13 × k14 × k15 × k21 × k22 × k23 × k24
```

Typical range: 0.88-0.97 (industry benchmark: 0.94-0.97)

**Data Flow**:
- **Input**: Module design parameters
- **Output**: CTM loss factors, total CTM ratio
- **Application**: Adjusts module power rating from cell-level to module-level

### 5. System Design
**Purpose**: Design complete PV system configuration
**Features**:
- Site location and solar resource
- Array layout (tilt, azimuth, tracking)
- Inverter selection and sizing
- String configuration
- DC/AC ratio optimization

**Data Flow**:
- **Input**: Module design, CTM losses
- **Output**: System configuration, capacity rating
- **Parameters**: Location, mounting type, inverter specs

### 6. EYA Simulation (Energy Yield Assessment)
**Purpose**: Pre-construction energy production estimation
**Features**:
- TMY weather data integration
- Hourly/monthly energy calculations
- Performance ratio estimation
- Degradation modeling
- Financial metrics (LCOE, NPV, IRR)

**Data Flow**:
- **Input**: System design, weather data
- **Output**: Annual/monthly energy estimates, PR, LCOE
- **Timeline**: Pre-construction phase

### 7. Performance Monitoring
**Purpose**: Real-time operational performance tracking
**Features**:
- Live performance metrics
- Expected vs. actual comparison
- PR tracking
- Availability monitoring
- Alert generation

**Data Flow**:
- **Input**: Real-time SCADA data, expected performance
- **Output**: Live metrics, alerts, performance trends
- **Timeline**: Operational phase

### 8. Fault Diagnostics
**Purpose**: Automated fault detection and diagnostics
**Features**:
- Performance-based fault detection
- IV curve analysis
- Thermal imaging processing
- String current analysis
- Predictive maintenance

**Data Flow**:
- **Input**: Performance data, sensor data
- **Output**: Fault classifications, maintenance recommendations
- **Methods**: Statistical analysis, anomaly detection

### 9. HYA Simulation (Historical Yield Analysis)
**Purpose**: Post-construction performance validation
**Features**:
- Actual vs. expected comparison
- PR trending
- Loss attribution
- Degradation rate calculation
- Warranty compliance verification

**Data Flow**:
- **Input**: Historical production data, EYA predictions
- **Output**: Performance gaps, degradation rates, financial impact
- **Timeline**: Post-construction (after 1+ years operation)

### 10. Energy Forecasting
**Purpose**: Short-term and long-term production forecasting
**Features**:
- Intraday to seasonal forecasting
- Weather forecast integration
- Machine learning models
- Forecast accuracy tracking
- Confidence intervals

**Data Flow**:
- **Input**: Historical performance, weather forecasts
- **Output**: Energy forecasts with confidence intervals
- **Horizons**: 24h, day-ahead, week-ahead, seasonal

### 11. Revamp & Repower
**Purpose**: System upgrade planning and analysis
**Features**:
- System assessment
- Upgrade scenario modeling
- Component replacement planning
- Performance improvement estimation
- Financial analysis (NPV, IRR, payback)

**Data Flow**:
- **Input**: Existing system data, upgrade options
- **Output**: Upgrade scenarios, ROI analysis
- **Timeline**: Mid-life (10-15 years) or end-of-life

### 12. Circularity (3R)
**Purpose**: Circular economy and sustainability analysis
**Features**:
- **REDUCE**: Material efficiency optimization
- **REUSE**: Component life extension strategies
- **RECYCLE**: End-of-life material recovery
- Circular economy scoring
- Regulatory compliance (WEEE, RoHS)

**Data Flow**:
- **Input**: Material selections, module design, system data
- **Output**: Circularity score, recyclability %, recovery value
- **Timeline**: Design phase + end-of-life

## Data Flow Architecture

### Session State Management

All modules share data through a centralized session manager:

```python
session_state = {
    'project_name': str,
    'material_data': dict,
    'module_design_data': dict,
    'ctm_losses': dict,
    'system_design_data': dict,
    'eya_results': dict,
    'performance_data': dict,
    'fault_data': dict,
    'hya_results': dict,
    'forecast_data': dict,
    'revamp_data': dict,
    'circularity_data': dict
}
```

### Module Dependencies

```
Dashboard
    ↓
Material Selection → Module Design → CTM Loss Analysis → System Design
                                                              ↓
                                                          EYA Simulation
                                                              ↓
                                    ┌─────────────────────────┴─────────────────────────┐
                                    ↓                                                   ↓
                        Performance Monitoring                              Energy Forecasting
                                    ↓
                            Fault Diagnostics
                                    ↓
                            HYA Simulation
                                    ↓
                            Revamp & Repower

Circularity (3R) ← integrates data from all modules
```

## Development Strategy

### Phase 1: Core Design Modules (Weeks 1-3)
- Dashboard
- Material Selection
- Module Design
- CTM Loss Analysis
- System Design

### Phase 2: Simulation & Analysis (Weeks 4-6)
- EYA Simulation
- HYA Simulation
- Energy Forecasting

### Phase 3: Operational Modules (Weeks 7-9)
- Performance Monitoring
- Fault Diagnostics
- Revamp & Repower

### Phase 4: Circularity Integration (Weeks 10-12)
- Circularity (3R) module
- Cross-module integration
- Reporting and export features

## Parallel Development Strategy

### Branch Structure

```
main
  ├── feature/material-selection
  ├── feature/module-design
  ├── feature/ctm-analysis
  ├── feature/system-design
  ├── feature/eya-simulation
  ├── feature/performance-monitoring
  ├── feature/fault-diagnostics
  ├── feature/hya-simulation
  ├── feature/energy-forecasting
  ├── feature/revamp-repower
  ├── feature/circularity
  └── feature/documentation
```

### Team Assignment Recommendations

**Team A - Design Track**:
- Material Selection
- Module Design
- CTM Loss Analysis

**Team B - System Track**:
- System Design
- EYA Simulation
- HYA Simulation

**Team C - Operations Track**:
- Performance Monitoring
- Fault Diagnostics
- Energy Forecasting

**Team D - Lifecycle Track**:
- Revamp & Repower
- Circularity (3R)

**Team E - Platform Track**:
- Dashboard
- Core utilities (charts, calculations, validators)
- Session management

## Key Technical Specifications

### Module Power Calculation

```python
module_power = cell_power × num_cells × CTM_ratio
where:
    cell_power = cell_efficiency × cell_area × 1000 W/m²
    CTM_ratio = k1 × k2 × ... × k24
```

### Performance Ratio

```python
PR = (Actual_Energy_kWh / Expected_Energy_kWh) × 100%
where:
    Expected_Energy = Nameplate_kWp × Irradiation_kWh/m² / 1.0
```

### LCOE Calculation

```python
LCOE = NPV(Costs) / NPV(Energy)
where:
    NPV(Costs) = CAPEX + Σ(OPEX_year / (1 + discount_rate)^year)
    NPV(Energy) = Σ(Energy_year × (1 - degradation)^year / (1 + discount_rate)^year)
```

## Data Standards and Validation

### Required Data Formats

- **Weather Data**: TMY3, PVGIS, or custom CSV
- **Performance Data**: SCADA, inverter data logger, or CSV
- **IV Curves**: Voltage (V) and Current (A) arrays
- **Thermal Images**: JPG, PNG, or TIF format

### Validation Rules

- Latitude: -90° to 90°
- Longitude: -180° to 180°
- Efficiency: 0-30% (technological limit check)
- Tilt: 0-90°
- Azimuth: 0-360°
- DC/AC ratio: 0.8-2.0 (typical range)
- Performance Ratio: 0-100%

## Success Metrics

### Application Performance
- Module load time: <2 seconds
- Simulation runtime: <10 seconds for annual EYA
- Dashboard refresh: <1 second

### User Experience
- Intuitive workflow progression
- Clear data validation feedback
- Comprehensive help documentation
- Export capabilities for all results

### Technical Accuracy
- EYA accuracy: ±5% vs. actual (after calibration)
- PR calculation: Industry-standard methodology
- CTM ratio: Validated against manufacturer datasheets
- LCOE: Aligned with NREL methodology

## Future Enhancements

### Version 2.0
- Machine learning for fault classification
- Automated optimization algorithms
- Multi-site portfolio management
- Advanced battery storage integration
- API for third-party integrations

### Version 3.0
- Real-time data streaming
- Cloud deployment with multi-user support
- Mobile application
- Advanced predictive analytics
- Blockchain for circularity tracking

## References and Standards

- IEC 61215: PV Module Design Qualification
- IEC 61853: PV Module Performance Testing
- IEC 62804: PID Testing
- IEEE 1547: Grid Interconnection
- WEEE Directive: E-waste Management
- RoHS Directive: Hazardous Substances
- NREL: LCOE Methodology
- PVGIS: Solar Radiation Database
- PVlib: Open-source PV modeling

---

**Document Version**: 1.0
**Last Updated**: 2024
**Maintained By**: PV Circularity Simulator Development Team
