"""
Core base models and utilities for PV circularity simulator.

This module provides base Pydantic models and common utilities that are used
across all other models in the simulator. It includes:
- Base model with common configuration
- Timestamped model for tracking creation/update times
- UUID-based model for unique identification
- Common validators and custom types
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, field_validator


class BaseModel(PydanticBaseModel):
    """
    Base model for all PV circularity simulator models.

    This model provides common configuration and behavior for all models
    in the system, including:
    - Strict validation (no extra fields allowed)
    - Serialization of complex types
    - Frozen models for immutability (can be overridden)
    - Population by field name

    All models in the simulator should inherit from this base class.
    """

    model_config = ConfigDict(
        # Validation settings
        validate_assignment=True,  # Validate on attribute assignment
        validate_default=True,  # Validate default values
        str_strip_whitespace=True,  # Strip whitespace from strings
        # Serialization settings
        use_enum_values=False,  # Keep enum instances in serialization
        arbitrary_types_allowed=False,  # Don't allow arbitrary types by default
        # Extra fields handling
        extra="forbid",  # Forbid extra fields not defined in model
        # Documentation
        json_schema_extra={
            "example": {},
        },
    )

    def model_dump_json_safe(self) -> Dict[str, Any]:
        """
        Safely dump model to dictionary for JSON serialization.

        This method ensures all fields are serializable to JSON,
        converting complex types like datetime to ISO format strings.

        Returns:
            Dict[str, Any]: Dictionary representation safe for JSON serialization
        """
        return self.model_dump(mode="json")

    def update(self, **kwargs: Any) -> "BaseModel":
        """
        Create a new instance with updated fields.

        Since models can be frozen, this method creates a new instance
        with the specified fields updated.

        Args:
            **kwargs: Field values to update

        Returns:
            BaseModel: New instance with updated fields
        """
        current_data = self.model_dump()
        current_data.update(kwargs)
        return self.__class__(**current_data)


class TimestampedModel(BaseModel):
    """
    Base model with automatic timestamp tracking.

    This model automatically tracks creation and last update times.
    Useful for models that need to maintain temporal information.

    Attributes:
        created_at: Timestamp when the model instance was created
        updated_at: Timestamp when the model instance was last updated
    """

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this instance was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this instance was last updated",
    )

    def touch(self) -> "TimestampedModel":
        """
        Update the updated_at timestamp to current time.

        Returns:
            TimestampedModel: New instance with updated timestamp
        """
        return self.update(updated_at=datetime.utcnow())


class UUIDModel(TimestampedModel):
    """
    Base model with UUID-based identification and timestamps.

    This model provides a unique identifier (UUID4) for each instance,
    along with creation and update timestamps. Useful for models that
    need to be uniquely identified across the system.

    Attributes:
        id: Unique identifier (UUID4) for this instance
        created_at: Timestamp when the model instance was created
        updated_at: Timestamp when the model instance was last updated
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier (UUID4) for this instance",
    )

    def __hash__(self) -> int:
        """
        Make the model hashable based on its UUID.

        Returns:
            int: Hash of the UUID
        """
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """
        Compare models based on their UUID.

        Args:
            other: Object to compare with

        Returns:
            bool: True if both models have the same UUID
        """
        if not isinstance(other, UUIDModel):
            return False
        return self.id == other.id


class NamedModel(BaseModel):
    """
    Base model with name and optional description.

    This model provides common fields for named entities with
    optional descriptions and metadata.

    Attributes:
        name: Human-readable name for this instance
        description: Optional detailed description
        metadata: Optional dictionary for additional metadata
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for this instance",
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional detailed description",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata dictionary for additional information",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """
        Validate that name is not empty after stripping whitespace.

        Args:
            v: Name value to validate

        Returns:
            str: Validated name

        Raises:
            ValueError: If name is empty after stripping
        """
        if not v or not v.strip():
            raise ValueError("Name cannot be empty or whitespace only")
        return v.strip()


class QuantityModel(BaseModel):
    """
    Base model for physical quantities with units.

    This model ensures that physical quantities are always associated
    with their units, preventing unit confusion errors.

    Attributes:
        value: Numerical value of the quantity
        unit: Unit of measurement (e.g., 'W', 'V', 'A', 'm²', 'kg')
    """

    value: float = Field(
        ...,
        description="Numerical value of the quantity",
    )
    unit: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Unit of measurement (e.g., 'W', 'V', 'A', 'm²', 'kg')",
    )

    def __str__(self) -> str:
        """
        String representation of the quantity.

        Returns:
            str: Formatted string like "100.5 W"
        """
        return f"{self.value} {self.unit}"

    @field_validator("value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """
        Validate that value is finite (not NaN or infinity).

        Args:
            v: Value to validate

        Returns:
            float: Validated value

        Raises:
            ValueError: If value is NaN or infinite
        """
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Value must be finite (not NaN or infinity)")
        return v
