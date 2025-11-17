"""
Configuration management for hybrid energy systems.

This module provides Pydantic-based configuration models for validating
and managing hybrid energy system configurations.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import yaml


class ComponentConfig(BaseModel):
    """
    Configuration for a single energy system component.

    Attributes:
        component_id: Unique identifier for the component
        component_type: Type of component (pv_array, battery, wind_turbine, etc.)
        name: Human-readable name for the component
        capacity: Nominal capacity of the component
        capacity_unit: Unit of capacity (kW, kWh, etc.)
        efficiency: Efficiency rating (0.0-1.0)
        parameters: Additional component-specific parameters
        enabled: Whether the component is active in the system
    """

    component_id: str = Field(..., description="Unique component identifier")
    component_type: Literal[
        "pv_array",
        "battery",
        "wind_turbine",
        "diesel_generator",
        "fuel_cell",
        "electrolyzer",
        "grid_connection",
    ] = Field(..., description="Type of energy component")
    name: str = Field(..., description="Component name")
    capacity: float = Field(gt=0, description="Nominal capacity")
    capacity_unit: str = Field(default="kW", description="Unit of capacity")
    efficiency: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Component efficiency"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )
    enabled: bool = Field(default=True, description="Component status")

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Validate component ID format."""
        if not v or len(v) < 3:
            raise ValueError("Component ID must be at least 3 characters")
        return v


class OperationStrategy(BaseModel):
    """
    Operation strategy configuration for hybrid energy system.

    Attributes:
        strategy_name: Name of the operation strategy
        strategy_type: Type of strategy (rule_based, optimal, predictive)
        priority_order: Priority order for component utilization
        control_parameters: Parameters for the control algorithm
        constraints: Operating constraints
    """

    strategy_name: str = Field(..., description="Strategy name")
    strategy_type: Literal["rule_based", "optimal", "predictive"] = Field(
        ..., description="Strategy type"
    )
    priority_order: List[str] = Field(
        default_factory=list, description="Component priority order"
    )
    control_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Control algorithm parameters"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Operating constraints"
    )


class SimulationConfig(BaseModel):
    """
    Simulation configuration settings.

    Attributes:
        time_step_minutes: Simulation time step in minutes
        simulation_duration_hours: Total simulation duration
        start_time: Simulation start time
        weather_data_source: Source for weather data
        load_profile_source: Source for load profile data
    """

    time_step_minutes: int = Field(
        default=5, ge=1, le=60, description="Time step in minutes"
    )
    simulation_duration_hours: int = Field(
        default=24, ge=1, description="Simulation duration in hours"
    )
    start_time: Optional[datetime] = Field(
        default=None, description="Simulation start time"
    )
    weather_data_source: str = Field(
        default="default", description="Weather data source"
    )
    load_profile_source: str = Field(
        default="default", description="Load profile source"
    )


class MonitoringConfig(BaseModel):
    """
    Monitoring and dashboard configuration.

    Attributes:
        refresh_rate_seconds: Dashboard refresh rate
        log_level: Logging level
        metrics_retention_hours: How long to retain metrics
        alert_thresholds: Thresholds for alerts
    """

    refresh_rate_seconds: int = Field(
        default=10, ge=1, description="Refresh rate in seconds"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    metrics_retention_hours: int = Field(
        default=168, ge=1, description="Metrics retention period"
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Alert threshold values"
    )


class SystemConfiguration(BaseModel):
    """
    Complete hybrid energy system configuration.

    This is the main configuration class that encompasses all aspects
    of the hybrid energy system including components, operation strategy,
    simulation parameters, and monitoring settings.

    Attributes:
        system_name: Name of the hybrid energy system
        system_description: Description of the system
        components: List of energy components
        operation_strategy: Operation strategy configuration
        simulation: Simulation settings
        monitoring: Monitoring settings
        metadata: Additional metadata
    """

    system_name: str = Field(..., description="System name")
    system_description: str = Field(
        default="", description="System description"
    )
    components: List[ComponentConfig] = Field(
        default_factory=list, description="System components"
    )
    operation_strategy: Optional[OperationStrategy] = Field(
        default=None, description="Operation strategy"
    )
    simulation: SimulationConfig = Field(
        default_factory=SimulationConfig, description="Simulation settings"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring settings"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("components")
    @classmethod
    def validate_components(cls, v: List[ComponentConfig]) -> List[ComponentConfig]:
        """Validate that component IDs are unique."""
        if len(v) != len(set(c.component_id for c in v)):
            raise ValueError("Component IDs must be unique")
        return v

    def get_component_by_id(self, component_id: str) -> Optional[ComponentConfig]:
        """
        Retrieve a component by its ID.

        Args:
            component_id: The unique identifier of the component

        Returns:
            ComponentConfig if found, None otherwise
        """
        for component in self.components:
            if component.component_id == component_id:
                return component
        return None

    def get_components_by_type(self, component_type: str) -> List[ComponentConfig]:
        """
        Retrieve all components of a specific type.

        Args:
            component_type: The type of components to retrieve

        Returns:
            List of matching ComponentConfig objects
        """
        return [c for c in self.components if c.component_type == component_type]

    def to_yaml(self, file_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save the YAML file
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SystemConfiguration":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            SystemConfiguration instance
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate_operation_strategy(self) -> bool:
        """
        Validate that the operation strategy references valid components.

        Returns:
            True if valid, False otherwise
        """
        if not self.operation_strategy:
            return True

        component_ids = {c.component_id for c in self.components}
        for comp_id in self.operation_strategy.priority_order:
            if comp_id not in component_ids:
                return False
        return True


class ConfigManager:
    """
    Manager class for loading and saving system configurations.

    This class provides convenient methods for managing configuration
    files and creating default configurations.
    """

    @staticmethod
    def create_default_config() -> SystemConfiguration:
        """
        Create a default hybrid energy system configuration.

        Returns:
            SystemConfiguration with default settings
        """
        return SystemConfiguration(
            system_name="Default Hybrid System",
            system_description="Default configuration for hybrid energy system",
            components=[
                ComponentConfig(
                    component_id="pv_001",
                    component_type="pv_array",
                    name="PV Array 1",
                    capacity=10.0,
                    capacity_unit="kW",
                    efficiency=0.85,
                ),
                ComponentConfig(
                    component_id="battery_001",
                    component_type="battery",
                    name="Battery Storage 1",
                    capacity=20.0,
                    capacity_unit="kWh",
                    efficiency=0.90,
                ),
            ],
            operation_strategy=OperationStrategy(
                strategy_name="Basic Rule-Based",
                strategy_type="rule_based",
                priority_order=["pv_001", "battery_001"],
            ),
        )

    @staticmethod
    def load_config(file_path: str) -> SystemConfiguration:
        """
        Load configuration from file.

        Args:
            file_path: Path to configuration file (YAML or JSON)

        Returns:
            SystemConfiguration instance
        """
        return SystemConfiguration.from_yaml(file_path)

    @staticmethod
    def save_config(config: SystemConfiguration, file_path: str) -> None:
        """
        Save configuration to file.

        Args:
            config: SystemConfiguration to save
            file_path: Path to save the configuration
        """
        config.to_yaml(file_path)
