"""Pydantic models for BOM Generator and Cost Calculator."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Currency(str, Enum):
    """Supported currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"


class ComponentCategory(str, Enum):
    """Categories of BOM components."""

    CELL = "cell"
    MODULE = "module"
    INTERCONNECT = "interconnect"
    ADHESIVE = "adhesive"
    PACKAGING = "packaging"


class Material(BaseModel):
    """Material definition with pricing and properties."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    id: str = Field(..., description="Unique material identifier")
    name: str = Field(..., description="Material name")
    category: ComponentCategory = Field(..., description="Component category")
    supplier: str = Field(..., description="Supplier name")
    unit: str = Field(..., description="Unit of measurement (kg, m2, pieces, etc.)")
    base_price: Decimal = Field(..., gt=0, description="Base price per unit")
    currency: Currency = Field(default=Currency.USD, description="Currency code")
    minimum_order_quantity: Optional[Decimal] = Field(
        default=None, ge=0, description="Minimum order quantity"
    )
    lead_time_days: int = Field(default=30, ge=0, description="Lead time in days")
    waste_factor: float = Field(
        default=0.05, ge=0, le=1, description="Material waste factor (0-1)"
    )
    transportation_cost_per_unit: Decimal = Field(
        default=Decimal("0"), ge=0, description="Transportation cost per unit"
    )
    specifications: Dict[str, str] = Field(
        default_factory=dict, description="Technical specifications"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last price update timestamp"
    )

    @field_validator("base_price", "transportation_cost_per_unit", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class PricingTier(BaseModel):
    """Volume-based pricing tier."""

    model_config = ConfigDict(validate_assignment=True)

    min_quantity: Decimal = Field(..., ge=0, description="Minimum quantity for tier")
    max_quantity: Optional[Decimal] = Field(
        default=None, ge=0, description="Maximum quantity for tier (None = unlimited)"
    )
    discount_percentage: float = Field(
        ..., ge=0, le=100, description="Discount percentage for this tier"
    )
    unit_price: Decimal = Field(..., gt=0, description="Price per unit at this tier")

    @field_validator("min_quantity", "max_quantity", "unit_price", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str | None) -> Decimal | None:
        """Convert numeric values to Decimal for precision."""
        if v is None:
            return None
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @field_validator("max_quantity")
    @classmethod
    def validate_max_greater_than_min(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Ensure max_quantity is greater than min_quantity."""
        if v is not None and "min_quantity" in info.data:
            min_qty = info.data["min_quantity"]
            if v <= min_qty:
                raise ValueError("max_quantity must be greater than min_quantity")
        return v


class MaterialWithTiers(Material):
    """Material with volume-based pricing tiers."""

    pricing_tiers: List[PricingTier] = Field(
        default_factory=list, description="Volume-based pricing tiers"
    )


class BOMItem(BaseModel):
    """Individual Bill of Materials item."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    material_id: str = Field(..., description="Reference to material ID")
    component_name: str = Field(..., description="Component name in module")
    category: ComponentCategory = Field(..., description="Component category")
    quantity: Decimal = Field(..., gt=0, description="Required quantity")
    unit: str = Field(..., description="Unit of measurement")
    unit_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Cost per unit")
    total_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Total line cost")
    waste_adjusted_quantity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Quantity including waste"
    )
    supplier: str = Field(default="", description="Supplier name")
    notes: str = Field(default="", description="Additional notes")

    @field_validator("quantity", "unit_cost", "total_cost", "waste_adjusted_quantity", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class CostBreakdown(BaseModel):
    """Detailed cost breakdown by category."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cell_costs: Decimal = Field(default=Decimal("0"), ge=0, description="Total cell costs")
    module_costs: Decimal = Field(default=Decimal("0"), ge=0, description="Total module costs")
    interconnect_costs: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total interconnect costs"
    )
    adhesive_costs: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total adhesive costs"
    )
    packaging_costs: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total packaging costs"
    )
    material_subtotal: Decimal = Field(
        default=Decimal("0"), ge=0, description="Subtotal of all materials"
    )
    waste_costs: Decimal = Field(default=Decimal("0"), ge=0, description="Costs from waste")
    transportation_costs: Decimal = Field(
        default=Decimal("0"), ge=0, description="Transportation costs"
    )
    manufacturing_overhead: Decimal = Field(
        default=Decimal("0"), ge=0, description="Manufacturing overhead"
    )
    total_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Grand total cost")
    currency: Currency = Field(default=Currency.USD, description="Currency code")

    @field_validator(
        "cell_costs",
        "module_costs",
        "interconnect_costs",
        "adhesive_costs",
        "packaging_costs",
        "material_subtotal",
        "waste_costs",
        "transportation_costs",
        "manufacturing_overhead",
        "total_cost",
        mode="before",
    )
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class ModuleDesign(BaseModel):
    """PV Module design specification."""

    model_config = ConfigDict(validate_assignment=True)

    module_id: str = Field(..., description="Module identifier")
    module_type: str = Field(..., description="Module type (mono-Si, poly-Si, etc.)")
    power_rating: float = Field(..., gt=0, description="Power rating in Watts")
    efficiency: float = Field(..., gt=0, le=100, description="Module efficiency %")
    dimensions: Dict[str, float] = Field(
        ..., description="Dimensions (length, width, thickness in mm)"
    )
    num_cells: int = Field(..., gt=0, description="Number of cells in module")
    cell_size: float = Field(..., gt=0, description="Cell size in mm")
    frame_type: str = Field(default="aluminum", description="Frame material type")
    glass_type: str = Field(default="tempered", description="Glass type")
    backsheet_type: str = Field(default="standard", description="Backsheet type")
    encapsulant_type: str = Field(default="EVA", description="Encapsulant type")
    junction_box_type: str = Field(default="standard", description="Junction box type")


class SupplierComparison(BaseModel):
    """Supplier comparison for a specific material."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    material_name: str = Field(..., description="Material name")
    suppliers: Dict[str, Decimal] = Field(
        ..., description="Supplier name to price mapping"
    )
    recommended_supplier: str = Field(..., description="Recommended supplier")
    potential_savings: Decimal = Field(
        ..., ge=0, description="Potential savings vs current supplier"
    )

    @field_validator("suppliers", mode="before")
    @classmethod
    def convert_prices_to_decimal(cls, v: Dict[str, float | Decimal | str]) -> Dict[str, Decimal]:
        """Convert supplier prices to Decimal."""
        return {k: Decimal(str(price)) if not isinstance(price, Decimal) else price for k, price in v.items()}

    @field_validator("potential_savings", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class BudgetAnalysis(BaseModel):
    """Budget vs actual cost analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    budgeted_cost: Decimal = Field(..., ge=0, description="Budgeted cost")
    actual_cost: Decimal = Field(..., ge=0, description="Actual cost")
    variance: Decimal = Field(..., description="Cost variance (actual - budget)")
    variance_percentage: float = Field(..., description="Variance as percentage")
    over_budget: bool = Field(..., description="Whether over budget")

    @field_validator("budgeted_cost", "actual_cost", "variance", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class CurrencyExchangeRate(BaseModel):
    """Currency exchange rate information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    from_currency: Currency = Field(..., description="Source currency")
    to_currency: Currency = Field(..., description="Target currency")
    rate: Decimal = Field(..., gt=0, description="Exchange rate")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    @field_validator("rate", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
