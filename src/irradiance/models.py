"""Pydantic models for irradiance calculations."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class LocationConfig(BaseModel):
    """Configuration for geographic location.

    Attributes:
        latitude: Latitude in decimal degrees (positive North)
        longitude: Longitude in decimal degrees (positive East)
        altitude: Altitude above sea level in meters
        timezone: Timezone string (e.g., 'UTC', 'America/New_York')
        name: Optional location name
    """

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    altitude: float = Field(default=0.0, ge=0, description="Altitude in meters")
    timezone: str = Field(default="UTC", description="Timezone string")
    name: Optional[str] = Field(default=None, description="Location name")


class SurfaceConfig(BaseModel):
    """Configuration for tilted surface orientation.

    Attributes:
        tilt: Surface tilt angle in degrees (0=horizontal, 90=vertical)
        azimuth: Surface azimuth angle in degrees (180=south in Northern hemisphere)
        albedo: Ground reflectance (0-1), default 0.2 for typical ground
    """

    tilt: float = Field(..., ge=0, le=90, description="Surface tilt in degrees")
    azimuth: float = Field(..., ge=0, le=360, description="Surface azimuth in degrees")
    albedo: float = Field(default=0.2, ge=0, le=1, description="Ground albedo")


class IrradianceComponents(BaseModel):
    """Irradiance components in W/m²."""

    ghi: pd.Series = Field(..., description="Global Horizontal Irradiance")
    dni: pd.Series = Field(..., description="Direct Normal Irradiance")
    dhi: pd.Series = Field(..., description="Diffuse Horizontal Irradiance")

    class Config:
        arbitrary_types_allowed = True


class POAComponents(BaseModel):
    """Plane-of-Array irradiance components in W/m²."""

    poa_global: pd.Series = Field(..., description="Total POA irradiance")
    poa_direct: pd.Series = Field(..., description="Direct beam component")
    poa_diffuse: pd.Series = Field(..., description="Sky diffuse component")
    poa_ground: pd.Series = Field(..., description="Ground reflected component")

    class Config:
        arbitrary_types_allowed = True


class SolarPosition(BaseModel):
    """Solar position angles."""

    zenith: pd.Series = Field(..., description="Solar zenith angle in degrees")
    azimuth: pd.Series = Field(..., description="Solar azimuth angle in degrees")
    elevation: pd.Series = Field(..., description="Solar elevation angle in degrees")
    equation_of_time: Optional[pd.Series] = Field(default=None, description="Equation of time")

    class Config:
        arbitrary_types_allowed = True


class ResourceStatistics(BaseModel):
    """Solar resource statistical analysis results."""

    mean: float = Field(..., description="Mean value")
    median: float = Field(..., description="Median value")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    p10: float = Field(..., description="10th percentile (P90 exceedance)")
    p50: float = Field(..., description="50th percentile (P50 exceedance)")
    p90: float = Field(..., description="90th percentile (P10 exceedance)")
    coefficient_of_variation: float = Field(..., description="Coefficient of variation")

    @classmethod
    def from_series(cls, data: pd.Series) -> "ResourceStatistics":
        """Calculate statistics from a pandas Series.

        Args:
            data: Time series data

        Returns:
            ResourceStatistics object
        """
        return cls(
            mean=float(data.mean()),
            median=float(data.median()),
            std=float(data.std()),
            min=float(data.min()),
            max=float(data.max()),
            p10=float(data.quantile(0.10)),
            p50=float(data.quantile(0.50)),
            p90=float(data.quantile(0.90)),
            coefficient_of_variation=float(data.std() / data.mean()) if data.mean() > 0 else 0.0,
        )
