# PV Circularity Simulator - Module Specifications

## Module Implementation Guide

This document provides detailed specifications for each module, including inputs, outputs, calculations, and UI components.

---

## 1. Dashboard Module

### Purpose
Central hub for project management, navigation, and high-level overview.

### Inputs
- User-created project name
- Session state from all modules

### Outputs
- Project metadata
- Module completion status
- Navigation to other modules

### UI Components
- Project name input
- Create/Clear project buttons
- Module status indicators (Pending/In Progress/Complete)
- Quick action buttons (Export, Report, Clear)
- Module completion metrics

### Data Storage
```python
{
    'project_name': str,
    'created_at': datetime,
    'last_modified': datetime
}
```

### Implementation Priority
**High** - Required for application navigation

---

## 2. Material Selection Module

### Purpose
Select and compare materials for PV module components based on performance, cost, and circularity.

### Inputs
- Material database (`data/materials_db.json`)
- User selections for each component

### Outputs
```python
{
    'glass_type': str,
    'glass_thickness': float,
    'encapsulant': str,
    'backsheet': str,
    'frame_material': str,
    'junction_box': str,
    'total_cost': float,
    'carbon_footprint': float,
    'recyclability_score': float,
    'circularity_score': float
}
```

### Calculations
```python
# Recyclability score (weighted average)
total_weight = sum(component_weights)
recyclability = sum(weight_i * recyclability_i) / total_weight

# Carbon footprint
carbon_total = sum(component_carbon_footprints)

# Total cost per module
cost_total = sum(component_costs)
```

### UI Components
- Material category tabs (Glass, Encapsulant, Backsheet, Frame)
- Selection dropdowns for each material
- Property comparison table
- Metrics cards (Cost, Carbon, Recyclability)
- Save button

### Validation
- Material compatibility checks
- Cost range validation
- Recyclability score calculation

### Implementation Priority
**High** - Feeds into Module Design

---

## 3. Module Design Module

### Purpose
Design complete PV module specification including cell configuration and electrical parameters.

### Inputs
- Cell technology selection
- Cell and module dimensions
- Electrical parameters

### Outputs
```python
{
    'cell_type': str,
    'cell_efficiency': float,  # %
    'num_cells': int,
    'cell_config': str,  # e.g., "12x6"
    'dimensions': {
        'length': float,  # mm
        'width': float,   # mm
        'weight': float   # kg
    },
    'electrical': {
        'pmax': float,    # W
        'voc': float,     # V
        'isc': float,     # A
        'vmpp': float,    # V
        'impp': float,    # A
        'efficiency': float  # %
    },
    'temp_coefficients': {
        'pmax': float,    # %/°C
        'voc': float,     # %/°C
        'isc': float      # %/°C
    }
}
```

### Calculations
```python
# Module area
area_m2 = (length_mm / 1000) * (width_mm / 1000)

# Module efficiency
efficiency = (pmax / (area_m2 * 1000)) * 100

# Fill Factor
ff = (vmpp * impp) / (voc * isc)

# Validation: Pmax should match Vmpp × Impp
assert abs(pmax - (vmpp * impp)) < 1.0  # Within 1W tolerance
```

### UI Components
- Cell technology selector
- Number of cells selector
- Cell configuration input
- Dimension inputs (length, width, weight)
- Electrical parameter inputs
- Temperature coefficient inputs
- Calculated metrics display
- Save button

### Validation
- Efficiency range: 15-26%
- Voltage range: 20-60V (Voc)
- Current range: 5-15A (Isc)
- Power consistency: Pmax ≈ Vmpp × Impp

### Implementation Priority
**High** - Critical for CTM analysis

---

## 4. CTM Loss Analysis Module

### Purpose
Analyze Cell-to-Module losses using detailed k-factor model.

### Inputs
- Module design parameters
- Individual k-factors (k1-k15, k21-k24)

### Outputs
```python
{
    # Optical losses
    'k1_reflection': float,
    'k2_shading': float,
    'k3_absorption': float,

    # Electrical losses
    'k4_resistive': float,
    'k5_mismatch': float,
    'k6_junction_box': float,

    # Thermal losses
    'k7_temperature': float,
    'k8_hotspot': float,

    # Assembly losses
    'k9_encapsulation': float,
    'k10_lamination': float,

    # Degradation
    'k11_lid': float,
    'k12_pid': float,
    'k13_mechanical': float,
    'k14_cell_degradation': float,
    'k15_interconnect': float,

    # Environmental
    'k21_humidity': float,
    'k22_uv_exposure': float,
    'k23_thermal_cycling': float,
    'k24_corrosion': float,

    # Results
    'total_ctm_ratio': float,
    'optical_ratio': float,
    'electrical_ratio': float,
    'thermal_ratio': float,
    'assembly_ratio': float,
    'degradation_ratio': float,
    'environmental_ratio': float
}
```

### Calculations
```python
# Category ratios
optical_ratio = k1 * k2 * k3
electrical_ratio = k4 * k5 * k6
thermal_ratio = k7 * k8
assembly_ratio = k9 * k10
degradation_ratio = k11 * k12 * k13 * k14 * k15
environmental_ratio = k21 * k22 * k23 * k24

# Total CTM ratio
total_ctm = (optical_ratio * electrical_ratio * thermal_ratio *
             assembly_ratio * degradation_ratio * environmental_ratio)

# Module power with CTM
module_power_actual = cell_power * num_cells * total_ctm
```

### UI Components
- Category sections (Optical, Electrical, Thermal, etc.)
- Slider for each k-factor with default values
- Category loss metrics
- Total CTM ratio display
- Loss breakdown bar chart
- Industry benchmark comparison
- Save button

### Validation
- Each k-factor: 0.5-1.0 (reasonable range)
- Total CTM: typically 0.88-0.97
- Warning if CTM < 0.88 or > 0.97

### Implementation Priority
**High** - Critical for accurate module rating

---

## 5. System Design Module

### Purpose
Design complete PV system configuration including site, array, and inverter.

### Inputs
- Site location and climate data
- Array configuration
- Inverter specifications
- Loss factors

### Outputs
```python
{
    'site': {
        'name': str,
        'latitude': float,
        'longitude': float,
        'altitude': float,
        'avg_ghi': float,
        'avg_temp': float,
        'wind_speed': float
    },
    'array': {
        'mounting_type': str,
        'tilt_angle': float,
        'azimuth': float,
        'num_modules': int,
        'module_power': float,
        'modules_per_string': int,
        'num_strings': int,
        'ground_coverage': float
    },
    'inverter': {
        'type': str,
        'capacity': float,
        'num_inverters': int,
        'efficiency': float,
        'dc_ac_ratio': float,
        'max_dc_voltage': float,
        'mppt_channels': int
    },
    'losses': {
        'soiling': float,
        'shading': float,
        'snow': float,
        'dc_wiring': float,
        'ac_wiring': float,
        'transformer': float
    }
}
```

### Calculations
```python
# System capacity
system_capacity_kWp = (num_modules * module_power) / 1000

# DC/AC ratio
dc_ac_ratio = system_capacity_kWp / (inverter_capacity * num_inverters)

# Strings per MPPT
strings_per_mppt = num_strings / mppt_channels

# String voltage validation
string_voc = module_voc * modules_per_string
assert string_voc <= max_dc_voltage
```

### UI Components
- Site information inputs (lat/lon, altitude, climate)
- Array configuration (tilt, azimuth, mounting)
- Module count and string configuration
- Inverter selection and parameters
- Loss factor sliders
- System metrics (capacity, DC/AC ratio)
- Validation warnings
- Save button

### Validation
- Latitude: -90 to 90
- Longitude: -180 to 180
- Tilt: 0-90°
- Azimuth: 0-360°
- DC/AC ratio: warning if < 1.1 or > 1.5
- String voltage: must not exceed max DC voltage

### Implementation Priority
**High** - Required for EYA simulation

---

## 6. EYA Simulation Module

### Purpose
Pre-construction energy yield assessment and financial analysis.

### Inputs
- System design parameters
- Weather data (TMY or uploaded)
- Simulation settings
- Financial parameters

### Outputs
```python
{
    'annual_energy': float,  # MWh
    'specific_yield': float,  # MWh/kWp
    'capacity_factor': float,  # %
    'performance_ratio': float,  # %
    'monthly_production': DataFrame,
    'multi_year_projection': DataFrame,
    'lcoe': float,  # $/kWh
    'npv': float,  # $
    'irr': float,  # %
    'payback_period': float  # years
}
```

### Calculations
```python
# Annual energy (simplified)
annual_energy_MWh = (system_capacity_kWp * avg_ghi * 365 *
                      performance_ratio / 100) / 1000

# Specific yield
specific_yield = annual_energy_MWh / system_capacity_kWp

# Capacity factor
capacity_factor = (annual_energy_MWh * 1000) / (system_capacity_kWp * 8760) * 100

# LCOE
lcoe = NPV(costs) / NPV(energy)

# Multi-year with degradation
energy_year_i = annual_energy * (1 - degradation_rate/100)^(i-1)
```

### UI Components
- Simulation parameters (years, degradation, PR)
- Weather data source selector
- File upload for custom TMY
- Run simulation button
- Results metrics (annual energy, CF, PR)
- Monthly production chart
- Multi-year projection chart
- Financial inputs (CAPEX, OPEX, rates)
- Financial metrics (LCOE, NPV, IRR)

### Implementation Priority
**Medium** - Important for project planning

---

## 7. Performance Monitoring Module

### Purpose
Real-time operational performance tracking and analysis.

### Inputs
- Real-time SCADA data or uploaded data
- Expected performance from EYA
- Time range selection

### Outputs
```python
{
    'current_power': float,  # kW
    'daily_energy': float,   # kWh
    'current_pr': float,     # %
    'availability': float,   # %
    'performance_trends': DataFrame,
    'alerts': List[dict]
}
```

### Calculations
```python
# Instantaneous PR
pr_instant = (actual_power / expected_power) * 100

# Daily energy
daily_energy = integral(power_over_time)

# Availability
availability = (uptime_hours / total_hours) * 100
```

### UI Components
- Time range selector
- Live metrics cards (Power, Energy, PR, Availability)
- Performance trend charts (24h, 7d, 30d)
- Expected vs. actual comparison
- System health indicators
- Alert notifications
- Export/report buttons

### Implementation Priority
**Medium** - Operational phase tool

---

## 8. Fault Diagnostics Module

### Purpose
Automated fault detection and classification.

### Inputs
- Performance data
- IV curves (optional)
- Thermal images (optional)
- String current data

### Outputs
```python
{
    'faults': List[{
        'id': str,
        'type': str,
        'location': str,
        'severity': str,
        'impact': str,
        'detected': datetime,
        'status': str
    }],
    'recommendations': List[str]
}
```

### Algorithms
```python
# Performance-based fault detection
if pr < threshold and duration > min_duration:
    classify_fault(pr_pattern, historical_data)

# String current analysis
if string_current < (avg_current - 2*std_dev):
    flag_as_underperforming()
```

### UI Components
- Diagnostic mode selector
- Input parameters (thresholds, sensitivity)
- Run diagnostic button
- Fault summary metrics
- Fault details table
- Detailed fault analysis
- Recommendation display
- IV curve analyzer
- Thermal image upload and analysis

### Implementation Priority
**Medium** - Advanced operational tool

---

## 9. HYA Simulation Module

### Purpose
Historical yield analysis and performance validation.

### Inputs
- Historical production data (CSV or API)
- Expected yield from EYA
- Analysis period

### Outputs
```python
{
    'total_actual': float,  # MWh
    'total_expected': float,  # MWh
    'avg_pr': float,  # %
    'degradation_rate': float,  # %/year
    'revenue_loss': float,  # $
    'loss_attribution': dict,
    'monthly_trends': DataFrame
}
```

### Calculations
```python
# Performance gap
gap = total_expected - total_actual
gap_percent = (gap / total_expected) * 100

# Degradation rate (linear regression)
degradation_rate, r_squared = linear_regression(pr_values, dates)

# Loss attribution
attribute_losses(pr_gap, environmental_data, system_data)
```

### UI Components
- Date range selector
- Data import (CSV upload or data source selector)
- Run HYA button
- Performance summary metrics
- Monthly performance charts
- PR trend analysis
- Loss breakdown visualization
- Degradation analysis
- Financial impact summary

### Implementation Priority
**Medium** - Post-construction analysis

---

## 10. Energy Forecasting Module

### Purpose
Generate short-term and long-term energy production forecasts.

### Inputs
- Historical performance data
- Weather forecasts
- Forecasting model selection

### Outputs
```python
{
    'forecast_horizon': str,
    'forecast_data': DataFrame,  # timestamp, forecast, lower_bound, upper_bound
    'forecast_accuracy': {
        'rmse': float,
        'mae': float,
        'mape': float,
        'r2': float
    }
}
```

### Algorithms
```python
# Persistence model
forecast_t = actual_t-1

# Statistical model (ARIMA)
forecast = arima_model.predict(steps=horizon)

# Machine learning
forecast = ml_model.predict(features)
```

### UI Components
- Forecast horizon selector
- Time resolution selector
- Model selection
- Weather data source
- Generate forecast button
- Forecast visualization (historical + forecast)
- Confidence intervals
- Accuracy metrics
- Detailed forecast table

### Implementation Priority
**Low** - Advanced operational feature

---

## 11. Revamp & Repower Module

### Purpose
Evaluate system upgrade and repowering scenarios.

### Inputs
- Existing system data
- Current performance
- Upgrade options (new modules, inverters, etc.)
- Financial parameters

### Outputs
```python
{
    'project_type': str,
    'capacity_increase': float,  # %
    'generation_increase': float,  # %
    'total_investment': float,  # $
    'npv': float,  # $
    'payback_period': float,  # years
    'irr': float,  # %
    'lcoe_improvement': float  # %
}
```

### Calculations
```python
# Performance improvement
generation_increase = (new_capacity / old_capacity) *
                      (new_efficiency / old_efficiency) *
                      (new_pr / old_pr)

# NPV
npv = -investment + sum(additional_revenue_i / (1+r)^i)

# Payback
cumulative_cash_flow[payback_year] >= 0
```

### UI Components
- Existing system inputs
- Upgrade scenario selector
- New technology configuration
- Financial inputs
- Calculate improvements button
- Performance projection metrics
- Financial analysis (NPV, IRR, payback)
- Cash flow projection chart
- Recommendation summary

### Implementation Priority
**Medium** - Mid-life planning tool

---

## 12. Circularity (3R) Module

### Purpose
Comprehensive circular economy analysis across reduce, reuse, recycle strategies.

### Inputs
- Material selections
- Module design
- System configuration
- End-of-life assumptions

### Outputs
```python
{
    'ce_score': float,  # 0-100
    'reduce_score': float,
    'reuse_score': float,
    'recycle_score': float,
    'recyclability': float,  # %
    'reuse_value': float,  # $
    'recycling_net_value': float,  # $
    'material_recovery': dict,
    'carbon_impact': float  # kg CO2e
}
```

### Calculations
```python
# Circular Economy Score
ce_score = (reduce_score + reuse_score + recycle_score) / 3

# Material recovery value
recovery_value = sum(material_weight_i * recovery_rate_i * price_i)

# Recyclability
recyclability = sum(weight_i * recyclable_i) / total_weight * 100
```

### UI Components
- Overview tab with CE score
- REDUCE tab (material efficiency strategies)
- REUSE tab (component reuse analysis)
- RECYCLE tab (end-of-life material recovery)
- Material composition visualization
- Strategy selection checkboxes
- Impact metrics
- Economic analysis
- Regulatory compliance checklist
- Save analysis button

### Implementation Priority
**High** - Core differentiator of application

---

## Cross-Module Data Flow

### Material Selection → Module Design
- Selected materials inform BOM and weight calculations

### Module Design → CTM Analysis
- Cell configuration and parameters used in loss calculations

### CTM Analysis → System Design
- Adjusted module power used for system sizing

### System Design → EYA
- Complete system parameters feed into energy simulation

### EYA → Performance Monitoring
- Expected performance used as baseline for monitoring

### Performance → Fault Diagnostics
- Performance anomalies trigger diagnostic algorithms

### HYA ← EYA
- Compares actual historical data with EYA predictions

### All Modules → Circularity
- Aggregates lifecycle data for comprehensive analysis

---

## Implementation Checklist

### Phase 1: Core Design (Weeks 1-3)
- [ ] Dashboard - basic navigation
- [ ] Material Selection - full implementation
- [ ] Module Design - full implementation
- [ ] CTM Loss Analysis - full implementation
- [ ] System Design - full implementation

### Phase 2: Simulation (Weeks 4-6)
- [ ] EYA Simulation - basic implementation
- [ ] HYA Simulation - basic implementation
- [ ] Energy Forecasting - basic implementation

### Phase 3: Operations (Weeks 7-9)
- [ ] Performance Monitoring - basic implementation
- [ ] Fault Diagnostics - basic implementation
- [ ] Revamp & Repower - basic implementation

### Phase 4: Integration (Weeks 10-12)
- [ ] Circularity (3R) - full implementation
- [ ] Cross-module data validation
- [ ] Export/import functionality
- [ ] Documentation and help

---

**Version**: 1.0
**Last Updated**: 2024
