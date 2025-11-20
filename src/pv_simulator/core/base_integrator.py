"""
Base integrator class for PV Circularity Simulator.

This module provides the abstract base class that all system integrators
must inherit from, ensuring consistent interfaces across the platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel


class IntegratorMetadata(BaseModel):
    """Metadata for integrator instances.

    Attributes:
        integrator_id: Unique identifier for the integrator
        created_at: Timestamp when integrator was created
        version: Version of the integrator implementation
        description: Human-readable description
    """

    integrator_id: str
    created_at: datetime
    version: str
    description: str


class BaseIntegrator(ABC):
    """Abstract base class for all system integrators.

    This class defines the standard interface that all integrators
    (PV, Wind, Hybrid) must implement to ensure consistency across
    the simulation platform.

    Attributes:
        metadata: Metadata about this integrator instance
        config: Configuration for the integrator
        _initialized: Whether the integrator has been initialized
    """

    def __init__(self, config: BaseModel, integrator_id: str, description: str) -> None:
        """Initialize the base integrator.

        Args:
            config: Configuration object (must be a Pydantic BaseModel)
            integrator_id: Unique identifier for this integrator
            description: Human-readable description

        Raises:
            TypeError: If config is not a Pydantic BaseModel
        """
        if not isinstance(config, BaseModel):
            raise TypeError("Config must be a Pydantic BaseModel")

        self.config = config
        self.metadata = IntegratorMetadata(
            integrator_id=integrator_id,
            created_at=datetime.now(),
            version="0.1.0",
            description=description,
        )
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the integrator.

        This method should perform any setup operations required
        before the integrator can be used for simulations.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate the integrator configuration.

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def run_simulation(self) -> Dict[str, Any]:
        """Run the simulation.

        Returns:
            Dictionary containing simulation results

        Raises:
            NotImplementedError: Must be implemented by subclasses
            RuntimeError: If integrator is not initialized
        """
        pass

    def get_metadata(self) -> IntegratorMetadata:
        """Get integrator metadata.

        Returns:
            IntegratorMetadata object with current metadata
        """
        return self.metadata

    def is_initialized(self) -> bool:
        """Check if integrator is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def reset(self) -> None:
        """Reset the integrator to uninitialized state.

        This allows the integrator to be reconfigured and reinitialized.
        """
        self._initialized = False

    def __repr__(self) -> str:
        """String representation of the integrator.

        Returns:
            String representation including ID and type
        """
        return f"{self.__class__.__name__}(id={self.metadata.integrator_id})"
