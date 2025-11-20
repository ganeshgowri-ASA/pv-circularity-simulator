"""Cell design data models."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class CellTemplate(BaseModel):
    """Template for solar cell specifications."""

    name: str = Field(..., description="Cell template name")
    technology: Literal["mono-Si", "multi-Si", "PERC", "TOPCon", "HJT", "IBC", "Perovskite", "Tandem"] = Field(
        ..., description="Cell technology type"
    )
    cell_type: Literal["M2", "M6", "M10", "M12", "G1", "G12"] = Field(
        ..., description="Cell size type"
    )
    length_mm: float = Field(..., gt=0, description="Cell length in mm")
    width_mm: float = Field(..., gt=0, description="Cell width in mm")
    thickness_um: float = Field(..., gt=0, description="Cell thickness in micrometers")
    efficiency_pct: float = Field(..., gt=0, le=50, description="Cell efficiency in %")
    pmax_w: float = Field(..., gt=0, description="Maximum power in watts")
    voc_v: float = Field(..., gt=0, description="Open circuit voltage in V")
    isc_a: float = Field(..., gt=0, description="Short circuit current in A")
    vmp_v: float = Field(..., gt=0, description="Voltage at maximum power in V")
    imp_a: float = Field(..., gt=0, description="Current at maximum power in A")
    fill_factor: float = Field(..., gt=0, le=1, description="Fill factor")
    temp_coeff_pmax: float = Field(..., description="Temperature coefficient of Pmax in %/°C")
    temp_coeff_voc: float = Field(..., description="Temperature coefficient of Voc in %/°C")
    temp_coeff_isc: float = Field(..., description="Temperature coefficient of Isc in %/°C")
    bifacial: bool = Field(default=False, description="Whether cell is bifacial")
    bifaciality_factor: Optional[float] = Field(None, ge=0, le=1, description="Bifaciality factor (0-1)")

    @field_validator('bifaciality_factor')
    @classmethod
    def validate_bifaciality(cls, v, info):
        """Validate bifaciality factor based on bifacial flag."""
        if info.data.get('bifacial') and v is None:
            raise ValueError("Bifaciality factor must be provided for bifacial cells")
        if not info.data.get('bifacial') and v is not None:
            raise ValueError("Bifaciality factor should not be provided for monofacial cells")
        return v


class CellDesign(BaseModel):
    """Complete cell design specification."""

    template: CellTemplate
    quantity: int = Field(..., gt=0, description="Number of cells in the design")
    configuration: Literal["full-cell", "half-cut", "quarter-cut", "shingled"] = Field(
        default="full-cell", description="Cell cutting configuration"
    )

    @property
    def total_area_m2(self) -> float:
        """Calculate total cell area in m²."""
        cell_area = (self.template.length_mm * self.template.width_mm) / 1_000_000
        return cell_area * self.quantity

    @property
    def total_pmax_w(self) -> float:
        """Calculate total maximum power in watts."""
        if self.configuration == "full-cell":
            return self.template.pmax_w * self.quantity
        elif self.configuration == "half-cut":
            return self.template.pmax_w * self.quantity  # Same total power
        elif self.configuration == "quarter-cut":
            return self.template.pmax_w * self.quantity  # Same total power
        elif self.configuration == "shingled":
            return self.template.pmax_w * self.quantity * 0.98  # 2% loss for shingling
        return self.template.pmax_w * self.quantity
