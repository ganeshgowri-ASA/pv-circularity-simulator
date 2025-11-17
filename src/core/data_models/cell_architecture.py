"""
Cell architecture and layer stack data models.

This module defines the Layer and CellArchitecture classes for building
and managing photovoltaic cell designs.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

from .materials import Material, DopingType


class LayerType(str, Enum):
    """Types of layers in a PV cell."""
    SUBSTRATE = "substrate"
    EMITTER = "emitter"
    BSF = "bsf"
    PASSIVATION = "passivation"
    CONTACT = "contact"
    TCO = "tco"
    ARC = "arc"
    METAL = "metal"


class Layer(BaseModel):
    """
    Single layer in a photovoltaic cell stack.

    Represents a physical layer with material, thickness, doping, and other properties.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique layer identifier"
    )

    name: str = Field(
        ...,
        description="Layer name (e.g., 'n+ emitter', 'SiNx ARC')"
    )

    layer_type: LayerType = Field(
        LayerType.SUBSTRATE,
        description="Type of layer"
    )

    material_name: str = Field(
        ...,
        description="Name of material used"
    )

    thickness: float = Field(
        ...,
        description="Layer thickness (µm)",
        gt=0
    )

    doping_type: Optional[DopingType] = Field(
        None,
        description="Doping type (n, p, or i)"
    )

    doping_concentration: Optional[float] = Field(
        None,
        description="Doping concentration (cm⁻³)",
        ge=0
    )

    defect_density: Optional[float] = Field(
        1e10,
        description="Defect density (cm⁻³)",
        ge=0
    )

    interface_quality: Optional[float] = Field(
        1.0,
        description="Interface quality factor (0-1, 1=perfect)",
        ge=0,
        le=1
    )

    # Layer position metadata
    position: int = Field(
        0,
        description="Layer position in stack (0=bottom)",
        ge=0
    )

    # Optional properties
    custom_properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional custom properties"
    )

    class Config:
        use_enum_values = True

    @validator('doping_concentration')
    def validate_doping(cls, v, values):
        """Ensure doping concentration is reasonable."""
        if v is not None:
            if v < 1e10 or v > 1e22:
                raise ValueError(
                    f"Doping concentration {v:.2e} cm⁻³ is outside "
                    f"reasonable range (1e10 - 1e22)"
                )
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary representation."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Layer":
        """Create Layer instance from dictionary."""
        return cls(**data)


class CellArchitecture(BaseModel):
    """
    Complete photovoltaic cell architecture.

    Manages a stack of layers and provides methods for manipulation,
    validation, and analysis.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique architecture identifier"
    )

    name: str = Field(
        ...,
        description="Architecture name (e.g., 'PERC', 'HJT')"
    )

    architecture_type: str = Field(
        ...,
        description="Type of cell architecture"
    )

    layers: List[Layer] = Field(
        default_factory=list,
        description="Stack of layers (bottom to top)"
    )

    area: float = Field(
        1.0,
        description="Cell area (cm²)",
        gt=0
    )

    temperature: float = Field(
        298.15,
        description="Operating temperature (K)",
        gt=0
    )

    description: Optional[str] = Field(
        None,
        description="Architecture description"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        arbitrary_types_allowed = True

    def add_layer(
        self,
        layer: Layer,
        position: Optional[int] = None
    ) -> None:
        """
        Add a layer to the stack.

        Args:
            layer: Layer to add
            position: Position to insert (None = append to top)
        """
        if position is None:
            position = len(self.layers)

        layer.position = position
        self.layers.insert(position, layer)

        # Update positions of layers above
        for i in range(position + 1, len(self.layers)):
            self.layers[i].position = i

    def remove_layer(self, layer_id: str) -> bool:
        """
        Remove a layer by ID.

        Args:
            layer_id: ID of layer to remove

        Returns:
            True if removed, False if not found
        """
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                self.layers.pop(i)
                # Update positions
                for j in range(i, len(self.layers)):
                    self.layers[j].position = j
                return True
        return False

    def move_layer(self, layer_id: str, new_position: int) -> bool:
        """
        Move a layer to a new position.

        Args:
            layer_id: ID of layer to move
            new_position: New position index

        Returns:
            True if moved, False if layer not found
        """
        # Find and remove layer
        layer = None
        for i, lyr in enumerate(self.layers):
            if lyr.id == layer_id:
                layer = self.layers.pop(i)
                break

        if layer is None:
            return False

        # Insert at new position
        self.add_layer(layer, new_position)
        return True

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """
        Get a layer by ID.

        Args:
            layer_id: Layer ID

        Returns:
            Layer instance or None if not found
        """
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def get_total_thickness(self) -> float:
        """
        Calculate total thickness of all layers.

        Returns:
            Total thickness in µm
        """
        return sum(layer.thickness for layer in self.layers)

    def get_layer_interfaces(self) -> List[Tuple[Layer, Layer]]:
        """
        Get all layer interfaces (pairs of adjacent layers).

        Returns:
            List of (bottom_layer, top_layer) tuples
        """
        interfaces = []
        for i in range(len(self.layers) - 1):
            interfaces.append((self.layers[i], self.layers[i + 1]))
        return interfaces

    def validate_structure(self) -> List[str]:
        """
        Validate cell structure and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        issues = []

        # Check for empty stack
        if not self.layers:
            issues.append("Cell has no layers")
            return issues

        # Check for substrate
        has_substrate = any(
            layer.layer_type == LayerType.SUBSTRATE
            for layer in self.layers
        )
        if not has_substrate:
            issues.append("Warning: No substrate layer defined")

        # Check for junction
        has_p_type = any(
            layer.doping_type == DopingType.P_TYPE
            for layer in self.layers
        )
        has_n_type = any(
            layer.doping_type == DopingType.N_TYPE
            for layer in self.layers
        )
        if not (has_p_type and has_n_type):
            issues.append("Warning: No p-n junction found")

        # Check for very thin layers
        for layer in self.layers:
            if layer.thickness < 0.001:  # < 1 nm
                issues.append(
                    f"Warning: Layer '{layer.name}' is very thin "
                    f"({layer.thickness:.4f} µm)"
                )

        # Check for very thick layers
        for layer in self.layers:
            if layer.thickness > 500:  # > 500 µm
                issues.append(
                    f"Warning: Layer '{layer.name}' is very thick "
                    f"({layer.thickness:.1f} µm)"
                )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "architecture_type": self.architecture_type,
            "layers": [layer.to_dict() for layer in self.layers],
            "area": self.area,
            "temperature": self.temperature,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellArchitecture":
        """Create CellArchitecture instance from dictionary."""
        # Convert layer dicts to Layer objects
        if "layers" in data:
            data["layers"] = [
                Layer.from_dict(layer_data)
                for layer_data in data["layers"]
            ]
        return cls(**data)

    def clone(self, new_name: Optional[str] = None) -> "CellArchitecture":
        """
        Create a deep copy of this architecture.

        Args:
            new_name: Name for the cloned architecture

        Returns:
            New CellArchitecture instance
        """
        data = self.to_dict()
        data["id"] = str(uuid.uuid4())  # New ID
        if new_name:
            data["name"] = new_name

        # Clone all layers with new IDs
        new_layers = []
        for layer_data in data["layers"]:
            layer_data["id"] = str(uuid.uuid4())
            new_layers.append(layer_data)
        data["layers"] = new_layers

        return self.from_dict(data)
