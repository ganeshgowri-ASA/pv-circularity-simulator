"""
Base calculator abstract class for all financial and technical calculators.

This module provides the foundational abstract base class that all calculator
implementations should inherit from, ensuring consistent interfaces and validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from pydantic import BaseModel
import logging

from ..exceptions import ValidationError, CalculationError

# Type variables for input and output models
InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


logger = logging.getLogger(__name__)


class BaseCalculator(ABC, Generic[InputModel, OutputModel]):
    """
    Abstract base class for all calculators in the PV Simulator.

    This class provides a common interface and shared functionality for all
    calculator implementations. All calculators should inherit from this class
    and implement the required abstract methods.

    Type Parameters:
        InputModel: Pydantic model type for calculator inputs
        OutputModel: Pydantic model type for calculator outputs

    Example:
        >>> class MyCalculator(BaseCalculator[MyInput, MyOutput]):
        ...     def validate(self, inputs: MyInput) -> bool:
        ...         return True
        ...     def calculate(self, inputs: MyInput) -> MyOutput:
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, name: str = None):
        """
        Initialize the calculator.

        Args:
            name: Optional name for the calculator instance (for logging)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def validate(self, inputs: InputModel) -> bool:
        """
        Validate input data before calculation.

        This method should check that all inputs are valid and that the
        calculation can be performed with the given inputs.

        Args:
            inputs: Input model instance to validate

        Returns:
            True if inputs are valid

        Raises:
            ValidationError: If inputs are invalid
        """
        pass

    @abstractmethod
    def calculate(self, inputs: InputModel) -> OutputModel:
        """
        Execute the calculation.

        This is the main method that performs the calculation and returns results.
        It should call validate() before performing calculations.

        Args:
            inputs: Validated input model instance

        Returns:
            Output model instance with calculation results

        Raises:
            ValidationError: If inputs are invalid
            CalculationError: If calculation fails
        """
        pass

    def run(self, inputs: InputModel, validate_first: bool = True) -> OutputModel:
        """
        Run the calculator with validation.

        This is a convenience method that combines validation and calculation
        in a single call with proper error handling and logging.

        Args:
            inputs: Input model instance
            validate_first: Whether to validate inputs before calculation

        Returns:
            Output model instance with calculation results

        Raises:
            ValidationError: If inputs are invalid
            CalculationError: If calculation fails
        """
        self.logger.info(f"Running {self.name} calculator")

        try:
            # Validate inputs if requested
            if validate_first:
                self.logger.debug("Validating inputs")
                is_valid = self.validate(inputs)
                if not is_valid:
                    raise ValidationError(
                        f"{self.name}: Input validation returned False"
                    )

            # Perform calculation
            self.logger.debug("Performing calculation")
            result = self.calculate(inputs)

            self.logger.info(f"{self.name} calculation completed successfully")
            return result

        except ValidationError:
            self.logger.error(f"{self.name}: Input validation failed")
            raise
        except CalculationError:
            self.logger.error(f"{self.name}: Calculation failed")
            raise
        except Exception as e:
            self.logger.error(f"{self.name}: Unexpected error: {str(e)}")
            raise CalculationError(
                f"{self.name}: Unexpected error during calculation: {str(e)}"
            ) from e

    def __repr__(self) -> str:
        """String representation of the calculator."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.name
