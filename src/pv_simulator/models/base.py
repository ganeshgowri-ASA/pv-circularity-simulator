"""Base Pydantic models for PV simulation components."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SimulationBase(BaseModel):
    """Base class for all simulation models.

    This base class provides common configuration and validation settings
    for all Pydantic models used throughout the PV circularity simulator.

    Attributes:
        model_config: Pydantic configuration with strict validation enabled.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": "See individual model implementations for examples"
        },
    )


class FinancialBase(SimulationBase):
    """Base class for financial models and calculations.

    Extends SimulationBase with common financial attributes and metadata.

    Attributes:
        calculation_date: Date when the calculation is performed.
        notes: Optional notes or comments about the calculation.
        metadata: Additional metadata for tracking and auditing.
    """

    calculation_date: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the calculation was performed",
    )
    notes: str | None = Field(
        default=None,
        description="Optional notes or comments about this calculation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for tracking and auditing purposes",
    )
