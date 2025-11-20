"""
Base Pydantic models for PV Circularity Simulator.

This module provides foundational model classes used throughout the application.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    """
    Base schema with common fields for all models.

    Attributes:
        id: Unique identifier for the record
        created_at: Timestamp when the record was created
        updated_at: Timestamp when the record was last updated
    """

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        populate_by_name=True,
        validate_assignment=True,
    )

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )


class Coordinates(BaseModel):
    """
    Spatial coordinates for defect location.

    Attributes:
        x: X-coordinate (horizontal position)
        y: Y-coordinate (vertical position)
        width: Width of the defect area (optional)
        height: Height of the defect area (optional)
    """

    x: float = Field(ge=0, description="X-coordinate")
    y: float = Field(ge=0, description="Y-coordinate")
    width: Optional[float] = Field(None, ge=0, description="Width of defect area")
    height: Optional[float] = Field(None, ge=0, description="Height of defect area")


class GeoLocation(BaseModel):
    """
    Geographic location for PV installations.

    Attributes:
        latitude: Geographic latitude
        longitude: Geographic longitude
        altitude: Altitude in meters (optional)
        site_name: Name of the installation site
    """

    latitude: float = Field(ge=-90, le=90, description="Latitude")
    longitude: float = Field(ge=-180, le=180, description="Longitude")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    site_name: Optional[str] = Field(None, description="Site name")
