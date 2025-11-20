"""Base classes and abstract interfaces for the PV Circularity Simulator."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class AnalysisResult(BaseModel):
    """Base model for analysis results.

    This is the base class for all analysis result models in the simulator.
    It provides common metadata fields that track when the analysis was performed
    and the configuration used.

    Attributes:
        timestamp: When the analysis was performed (ISO 8601 format).
        analysis_type: Type of analysis performed (e.g., "NPV", "IRR").
        metadata: Additional metadata about the analysis.
    """

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the analysis was performed"
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis performed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "timestamp": "2025-11-17T10:30:00",
                "analysis_type": "NPV",
                "metadata": {"version": "0.1.0"}
            }
        }


class BaseAnalyzer(ABC, Generic[TInput, TOutput]):
    """Abstract base class for all analyzers in the PV simulator.

    This class provides a common interface for all analyzer classes.
    Analyzers take structured input data, perform calculations, and return
    structured output results.

    Type Parameters:
        TInput: The Pydantic model type for input data.
        TOutput: The Pydantic model type for output results.

    Attributes:
        name: Name of the analyzer.
        version: Version string of the analyzer implementation.
    """

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        """Initialize the analyzer.

        Args:
            name: Name identifier for this analyzer.
            version: Version string of the analyzer implementation.
        """
        self.name = name
        self.version = version

    @abstractmethod
    def analyze(self, input_data: TInput) -> TOutput:
        """Perform the analysis.

        This is the main entry point for running the analysis. Implementations
        should validate inputs, perform calculations, and return structured results.

        Args:
            input_data: Validated input data as a Pydantic model.

        Returns:
            Analysis results as a Pydantic model.

        Raises:
            PVSimulatorError: If the analysis encounters an error.
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: TInput) -> None:
        """Validate input data for the analysis.

        This method performs additional validation beyond Pydantic's built-in
        validation. It should check business rules, data consistency, and
        raise appropriate exceptions for invalid inputs.

        Args:
            input_data: Input data to validate.

        Raises:
            InvalidCashFlowError: If input data is invalid.
            FinancialAnalysisError: If validation fails for other reasons.
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this analyzer.

        Returns:
            Dictionary containing analyzer name, version, and other metadata.
        """
        return {
            "name": self.name,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
        }
