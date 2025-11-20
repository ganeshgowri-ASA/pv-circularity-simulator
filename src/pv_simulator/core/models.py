"""
Core Pydantic models and base classes for the PV Circularity Simulator.

This module provides base models and shared configurations for all domain models
in the PV simulator ecosystem.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseSimulatorModel(BaseModel):
    """
    Base Pydantic model with common configuration for all simulator models.

    This base class provides:
    - Consistent JSON encoding for datetime objects
    - Validation on assignment
    - Arbitrary types support
    - Populate by name for flexibility

    All domain models in the PV simulator should inherit from this class
    to ensure consistent behavior and configuration.

    Attributes:
        model_config: Pydantic configuration dictionary

    Examples:
        >>> class MyModel(BaseSimulatorModel):
        ...     value: float
        ...     timestamp: datetime
        >>> model = MyModel(value=42.0, timestamp=datetime.now())
        >>> model.value
        42.0
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
        },
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the model

        Examples:
            >>> model = BaseSimulatorModel()
            >>> isinstance(model.to_dict(), dict)
            True
        """
        return self.model_dump()

    def to_json(self, **kwargs: Any) -> str:
        """
        Convert the model to a JSON string.

        Args:
            **kwargs: Additional keyword arguments for json serialization

        Returns:
            str: JSON string representation of the model

        Examples:
            >>> model = BaseSimulatorModel()
            >>> json_str = model.to_json()
            >>> isinstance(json_str, str)
            True
        """
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSimulatorModel":
        """
        Create a model instance from a dictionary.

        Args:
            data: Dictionary containing model data

        Returns:
            BaseSimulatorModel: New instance of the model

        Examples:
            >>> data = {"value": 42}
            >>> # Assuming a subclass with a 'value' field
            >>> # model = MyModel.from_dict(data)
        """
        return cls(**data)


class TimestampedModel(BaseSimulatorModel):
    """
    Base model with automatic timestamp tracking.

    This model automatically captures the creation timestamp for any instance.
    Useful for tracking when data was generated or recorded.

    Attributes:
        timestamp: ISO format datetime string of when the model was created

    Examples:
        >>> class DataPoint(TimestampedModel):
        ...     value: float
        >>> point = DataPoint(value=42.0)
        >>> isinstance(point.timestamp, datetime)
        True
    """

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when this model instance was created"
    )


class IdentifiableModel(BaseSimulatorModel):
    """
    Base model with optional identifier field.

    Provides a consistent way to identify model instances across the system.

    Attributes:
        id: Optional identifier for the model instance
        name: Optional human-readable name for the model
        description: Optional detailed description

    Examples:
        >>> class Component(IdentifiableModel):
        ...     power: float
        >>> component = Component(id="pv-001", name="Panel A", power=300.0)
        >>> component.id
        'pv-001'
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this instance"
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description"
    )
