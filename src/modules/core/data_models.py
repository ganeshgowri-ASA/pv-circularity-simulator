"""
B14-S02: Comprehensive Data Models & Utilities
Production-ready Pydantic models for all PV circularity simulator modules.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, field_validator
from enum import Enum


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EnergyStorageTechnology(str, Enum):
    """Battery storage technologies."""
    LITHIUM_ION = "lithium_ion"
    LEAD_ACID = "lead_acid"
    FLOW_BATTERY = "flow_battery"
    SODIUM_ION = "sodium_ion"
    SOLID_STATE = "solid_state"


class GridServiceType(str, Enum):
    """Types of grid services."""
    FREQUENCY_REGULATION = "frequency_regulation"
    VOLTAGE_SUPPORT = "voltage_support"
    PEAK_SHAVING = "peak_shaving"
    DEMAND_RESPONSE = "demand_response"
    BLACK_START = "black_start"


class FinancingType(str, Enum):
    """Project financing types."""
    EQUITY = "equity"
    DEBT = "debt"
    HYBRID = "hybrid"
    LEASE = "lease"
    PPA = "ppa"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# HYBRID ENERGY DATA MODELS (B12)
# ============================================================================

class BatterySpecification(BaseModel):
    """Battery system specifications."""
    technology: EnergyStorageTechnology
    capacity_kwh: float = Field(gt=0, description="Battery capacity in kWh")
    power_rating_kw: float = Field(gt=0, description="Power rating in kW")
    round_trip_efficiency: float = Field(ge=0.7, le=1.0, description="Round-trip efficiency")
    depth_of_discharge: float = Field(ge=0.1, le=1.0, description="Maximum DoD")
    cycle_life: int = Field(gt=0, description="Expected cycle life")
    initial_cost_per_kwh: float = Field(gt=0, description="Initial cost per kWh")
    degradation_rate_per_year: float = Field(ge=0, le=0.1, description="Annual degradation")

    class Config:
        json_schema_extra = {
            "example": {
                "technology": "lithium_ion",
                "capacity_kwh": 1000.0,
                "power_rating_kw": 500.0,
                "round_trip_efficiency": 0.92,
                "depth_of_discharge": 0.9,
                "cycle_life": 5000,
                "initial_cost_per_kwh": 300.0,
                "degradation_rate_per_year": 0.02
            }
        }


class BatteryOperationSchedule(BaseModel):
    """Battery charge/discharge schedule."""
    timestamp: datetime
    power_kw: float = Field(description="Positive=discharge, Negative=charge")
    state_of_charge: float = Field(ge=0, le=1, description="SoC fraction")
    grid_price: float = Field(description="Grid electricity price $/kWh")
    arbitrage_profit: float = Field(ge=0, description="Profit from arbitrage")


class WindResourceData(BaseModel):
    """Wind resource characteristics."""
    location_name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    hub_height_m: float = Field(gt=0, description="Hub height in meters")
    average_wind_speed_ms: float = Field(gt=0, description="Average wind speed m/s")
    weibull_k: float = Field(gt=0, description="Weibull shape parameter")
    weibull_c: float = Field(gt=0, description="Weibull scale parameter")
    air_density: float = Field(default=1.225, gt=0, description="Air density kg/m³")
    roughness_length: float = Field(default=0.1, gt=0, description="Surface roughness")


class HybridSystemConfiguration(BaseModel):
    """Hybrid renewable energy system configuration."""
    system_name: str
    solar_capacity_kw: float = Field(ge=0)
    wind_capacity_kw: float = Field(ge=0)
    battery_capacity_kwh: float = Field(ge=0)
    location: str
    optimization_objective: Literal["cost", "reliability", "carbon", "balanced"]
    grid_connected: bool = Field(default=True)

    @field_validator('solar_capacity_kw', 'wind_capacity_kw')
    @classmethod
    def validate_capacity(cls, v, info):
        """Ensure at least one generation source exists."""
        if info.data.get('solar_capacity_kw', 0) + info.data.get('wind_capacity_kw', 0) <= 0:
            raise ValueError("Must have at least solar or wind capacity")
        return v


class HydrogenSystemSpec(BaseModel):
    """Hydrogen production and storage system."""
    electrolyzer_capacity_kw: float = Field(gt=0, description="Electrolyzer power rating")
    electrolyzer_efficiency: float = Field(ge=0.5, le=0.9, description="H2 production efficiency")
    h2_storage_capacity_kg: float = Field(gt=0, description="H2 storage capacity in kg")
    storage_pressure_bar: float = Field(gt=0, description="Storage pressure in bar")
    fuel_cell_capacity_kw: Optional[float] = Field(default=None, ge=0, description="Fuel cell rating")
    fuel_cell_efficiency: Optional[float] = Field(default=0.55, ge=0.4, le=0.7)
    capex_per_kw: float = Field(gt=0, description="Capital cost per kW")
    h2_production_cost: Optional[float] = Field(default=None, description="$/kg H2")


class GridConnectionSpec(BaseModel):
    """Grid connection specifications."""
    connection_point: str
    voltage_level_kv: float = Field(gt=0, description="Connection voltage level")
    max_export_capacity_kw: float = Field(ge=0)
    max_import_capacity_kw: float = Field(ge=0)
    grid_services: List[GridServiceType] = Field(default_factory=list)
    interconnection_cost: float = Field(ge=0)
    annual_grid_fee: float = Field(ge=0)


# ============================================================================
# FINANCIAL DATA MODELS (B13)
# ============================================================================

class ProjectFinancials(BaseModel):
    """Core project financial parameters."""
    project_name: str
    capacity_kw: float = Field(gt=0)
    capex_usd: float = Field(gt=0, description="Capital expenditure")
    opex_annual_usd: float = Field(ge=0, description="Annual operating expenses")
    project_lifetime_years: int = Field(gt=0, le=50)
    discount_rate: float = Field(gt=0, le=0.3, description="Discount rate (WACC)")
    inflation_rate: float = Field(ge=0, le=0.1, description="Annual inflation")
    tax_rate: float = Field(ge=0, le=0.5, description="Corporate tax rate")
    degradation_rate: float = Field(ge=0, le=0.05, description="Annual output degradation")

    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "Solar Farm Alpha",
                "capacity_kw": 50000,
                "capex_usd": 50000000,
                "opex_annual_usd": 500000,
                "project_lifetime_years": 25,
                "discount_rate": 0.08,
                "inflation_rate": 0.02,
                "tax_rate": 0.21,
                "degradation_rate": 0.005
            }
        }


class CashFlowProjection(BaseModel):
    """Annual cash flow projection."""
    year: int = Field(ge=0)
    revenue: float
    operating_expenses: float
    depreciation: float
    taxable_income: float
    taxes: float
    net_income: float
    free_cash_flow: float
    cumulative_cash_flow: float


class LCOEResult(BaseModel):
    """Levelized Cost of Energy calculation result."""
    lcoe_usd_per_kwh: float = Field(description="LCOE in $/kWh")
    total_lifetime_cost: float
    total_lifetime_energy_kwh: float
    real_lcoe: float = Field(description="Real LCOE (inflation-adjusted)")
    nominal_lcoe: float = Field(description="Nominal LCOE")
    sensitivity_range: Optional[Dict[str, float]] = Field(default=None)


class NPVResult(BaseModel):
    """Net Present Value analysis result."""
    npv_usd: float = Field(description="Net Present Value")
    benefit_cost_ratio: float = Field(gt=0)
    payback_period_years: Optional[float] = Field(default=None, ge=0)
    discounted_payback_years: Optional[float] = Field(default=None, ge=0)
    profitability_index: float = Field(gt=0)
    annual_cash_flows: List[CashFlowProjection]


class IRRResult(BaseModel):
    """Internal Rate of Return analysis."""
    irr_percent: float = Field(description="Internal Rate of Return %")
    mirr_percent: Optional[float] = Field(default=None, description="Modified IRR %")
    hurdle_rate_percent: float = Field(description="Required hurdle rate")
    exceeds_hurdle: bool = Field(description="Whether IRR exceeds hurdle rate")
    equity_irr: Optional[float] = Field(default=None)
    project_irr: float


class DebtServiceCoverage(BaseModel):
    """Debt service coverage metrics."""
    year: int
    ebitda: float = Field(description="Earnings before interest, taxes, depreciation")
    debt_service: float = Field(description="Annual debt payments")
    dscr: float = Field(description="Debt Service Coverage Ratio")
    minimum_dscr: float = Field(default=1.2, description="Minimum required DSCR")
    meets_covenant: bool


class BankabilityAssessment(BaseModel):
    """Bankability and credit assessment."""
    overall_rating: RiskLevel
    credit_score: float = Field(ge=0, le=100)
    minimum_dscr: float
    average_dscr: float
    loan_to_value_ratio: float = Field(ge=0, le=1)
    debt_to_equity_ratio: float = Field(ge=0)
    risk_factors: List[str]
    mitigation_strategies: List[str]
    bankable: bool
    recommended_debt_capacity: float


class FinancingStructure(BaseModel):
    """Project financing structure."""
    financing_type: FinancingType
    total_project_cost: float = Field(gt=0)
    equity_amount: float = Field(ge=0)
    debt_amount: float = Field(ge=0)
    debt_interest_rate: float = Field(ge=0, le=0.2)
    debt_term_years: int = Field(gt=0)
    equity_return_target: float = Field(ge=0)

    @field_validator('equity_amount', 'debt_amount')
    @classmethod
    def validate_financing(cls, v, info):
        """Validate equity + debt = total cost."""
        if 'equity_amount' in info.data and 'debt_amount' in info.data:
            total = info.data['equity_amount'] + info.data['debt_amount']
            expected = info.data.get('total_project_cost', 0)
            if abs(total - expected) > 0.01 * expected:  # 1% tolerance
                raise ValueError(f"Equity + Debt must equal total project cost")
        return v


# ============================================================================
# INTEGRATION & SYSTEM MODELS (B14)
# ============================================================================

class ModuleDataExchange(BaseModel):
    """Data exchange between modules."""
    source_module: str
    target_module: str
    data_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class SimulationConfiguration(BaseModel):
    """Overall simulation configuration."""
    simulation_name: str
    start_date: date
    end_date: date
    time_step_hours: float = Field(gt=0, le=24, default=1.0)
    location: str
    enabled_modules: List[str]
    integration_endpoints: Dict[str, str] = Field(default_factory=dict)
    output_format: Literal["json", "csv", "parquet", "excel"] = "json"


class ValidationResult(BaseModel):
    """Data validation result."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.now)
    validator_version: str = Field(default="1.0.0")


# ============================================================================
# UI & VISUALIZATION MODELS (B15)
# ============================================================================

class NavigationItem(BaseModel):
    """Navigation menu item."""
    label: str
    page_id: str
    icon: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)
    order: int = Field(default=0)
    enabled: bool = Field(default=True)
    roles_required: List[str] = Field(default_factory=list)


class ChartConfiguration(BaseModel):
    """Chart visualization configuration."""
    chart_type: Literal["line", "bar", "scatter", "area", "pie", "heatmap", "sankey"]
    title: str
    x_axis_label: Optional[str] = Field(default=None)
    y_axis_label: Optional[str] = Field(default=None)
    data_series: List[Dict[str, Any]]
    color_scheme: Optional[str] = Field(default="plotly")
    interactive: bool = Field(default=True)
    export_formats: List[str] = Field(default_factory=lambda: ["png", "svg", "pdf"])


class DashboardLayout(BaseModel):
    """Dashboard layout configuration."""
    dashboard_id: str
    title: str
    layout_type: Literal["single_column", "two_column", "grid", "tabs"]
    widgets: List[Dict[str, Any]]
    refresh_interval_seconds: Optional[int] = Field(default=None, ge=1)
    theme: Literal["light", "dark", "auto"] = "light"


# ============================================================================
# EXPORT & REPORTING MODELS
# ============================================================================

class ReportSection(BaseModel):
    """Report section configuration."""
    section_id: str
    title: str
    content_type: Literal["text", "table", "chart", "metrics"]
    data: Any
    order: int = Field(default=0)


class ExportConfiguration(BaseModel):
    """Data export configuration."""
    export_format: Literal["json", "csv", "excel", "pdf", "html"]
    include_charts: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    compression: Optional[Literal["zip", "gzip"]] = Field(default=None)
    file_path: str


# ============================================================================
# UTILITY FUNCTIONS FOR DATA MODELS
# ============================================================================

def validate_energy_balance(generation: float, consumption: float, storage_delta: float,
                           tolerance: float = 0.01) -> bool:
    """Validate energy balance in the system."""
    balance = abs(generation - consumption - storage_delta)
    return balance <= tolerance * max(generation, consumption)


def create_default_battery_spec(capacity_kwh: float) -> BatterySpecification:
    """Create default battery specification."""
    return BatterySpecification(
        technology=EnergyStorageTechnology.LITHIUM_ION,
        capacity_kwh=capacity_kwh,
        power_rating_kw=capacity_kwh * 0.5,  # 2-hour discharge
        round_trip_efficiency=0.92,
        depth_of_discharge=0.9,
        cycle_life=5000,
        initial_cost_per_kwh=300.0,
        degradation_rate_per_year=0.02
    )


def create_default_financial_params(capacity_kw: float,
                                    capex_per_kw: float = 1000.0) -> ProjectFinancials:
    """Create default financial parameters."""
    capex = capacity_kw * capex_per_kw
    opex = capex * 0.01  # 1% of CAPEX as OPEX

    return ProjectFinancials(
        project_name="Default Project",
        capacity_kw=capacity_kw,
        capex_usd=capex,
        opex_annual_usd=opex,
        project_lifetime_years=25,
        discount_rate=0.08,
        inflation_rate=0.02,
        tax_rate=0.21,
        degradation_rate=0.005
    )


__all__ = [
    # Enums
    "EnergyStorageTechnology",
    "GridServiceType",
    "FinancingType",
    "RiskLevel",
    # Hybrid Energy Models
    "BatterySpecification",
    "BatteryOperationSchedule",
    "WindResourceData",
    "HybridSystemConfiguration",
    "HydrogenSystemSpec",
    "GridConnectionSpec",
    # Financial Models
    "ProjectFinancials",
    "CashFlowProjection",
    "LCOEResult",
    "NPVResult",
    "IRRResult",
    "DebtServiceCoverage",
    "BankabilityAssessment",
    "FinancingStructure",
    # Integration Models
    "ModuleDataExchange",
    "SimulationConfiguration",
    "ValidationResult",
    # UI Models
    "NavigationItem",
    "ChartConfiguration",
    "DashboardLayout",
    # Export Models
    "ReportSection",
    "ExportConfiguration",
    # Utility Functions
    "validate_energy_balance",
    "create_default_battery_spec",
    "create_default_financial_params",
]
