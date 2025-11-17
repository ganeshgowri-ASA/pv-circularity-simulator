"""
Data models for PV Circularity Simulator.

This module contains all core data models for projects, resources, contracts,
and portfolio management in the PV lifecycle simulation platform.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class ProjectStatus(Enum):
    """Project lifecycle status enumeration."""
    DESIGN = "Design"
    ENGINEERING = "Engineering"
    PLANNING = "Planning"
    IMPLEMENTATION = "Implementation"
    MONITORING = "Monitoring"
    EOL = "End of Life"


class ResourceType(Enum):
    """Resource type classification."""
    MODULE = "PV Module"
    INVERTER = "Inverter"
    CABLE = "Cable"
    MOUNTING = "Mounting System"
    LABOR = "Labor"
    CAPITAL = "Capital"
    EQUIPMENT = "Equipment"
    OTHER = "Other"


class ContractType(Enum):
    """Contract classification types."""
    SUPPLY = "Supply Agreement"
    LABOR = "Labor Contract"
    SERVICE = "Service Agreement"
    MAINTENANCE = "Maintenance Contract"
    CONSULTING = "Consulting Agreement"


class ContractStatus(Enum):
    """Contract lifecycle status."""
    DRAFT = "Draft"
    PENDING = "Pending Approval"
    ACTIVE = "Active"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


@dataclass
class Project:
    """
    Represents a PV system installation/simulation project.

    Attributes:
        id: Unique project identifier
        name: Project display name
        description: Detailed project description
        status: Current lifecycle status
        capacity_kwp: System capacity in kWp (kilowatt-peak)
        location: Installation location/address
        created_date: Project creation timestamp
        updated_date: Last modification timestamp
        start_date: Planned/actual project start
        end_date: Planned/actual project completion
        owner: Project owner/manager name
        budget: Total project budget
        metadata: Additional flexible data storage
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: ProjectStatus = ProjectStatus.DESIGN
    capacity_kwp: float = 0.0
    location: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    updated_date: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    owner: str = ""
    budget: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "capacity_kwp": self.capacity_kwp,
            "location": self.location,
            "created_date": self.created_date.isoformat(),
            "updated_date": self.updated_date.isoformat(),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "owner": self.owner,
            "budget": self.budget,
            "metadata": self.metadata,
        }


@dataclass
class Resource:
    """
    Represents materials, components, labor, and financial resources.

    Attributes:
        id: Unique resource identifier
        project_id: Associated project identifier
        name: Resource display name
        resource_type: Classification of resource
        quantity: Resource quantity
        unit: Unit of measurement (e.g., pieces, hours, kW)
        unit_cost: Cost per unit
        total_cost: Total resource cost (quantity × unit_cost)
        supplier: Resource supplier/vendor name
        availability_start: Resource availability start date
        availability_end: Resource availability end date
        allocated: Whether resource is allocated to project
        constraints: List of constraint descriptions
        metadata: Additional flexible data storage
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    name: str = ""
    resource_type: ResourceType = ResourceType.OTHER
    quantity: float = 0.0
    unit: str = ""
    unit_cost: float = 0.0
    total_cost: float = 0.0
    supplier: str = ""
    availability_start: Optional[datetime] = None
    availability_end: Optional[datetime] = None
    allocated: bool = False
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_total_cost(self) -> float:
        """Calculate and update total cost."""
        self.total_cost = self.quantity * self.unit_cost
        return self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary representation."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "resource_type": self.resource_type.value,
            "quantity": self.quantity,
            "unit": self.unit,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "supplier": self.supplier,
            "availability_start": self.availability_start.isoformat() if self.availability_start else None,
            "availability_end": self.availability_end.isoformat() if self.availability_end else None,
            "allocated": self.allocated,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }


@dataclass
class Contract:
    """
    Represents supply agreements, labor contracts, and service agreements.

    Attributes:
        id: Unique contract identifier
        project_id: Associated project identifier
        contract_type: Classification of contract
        vendor: Vendor/contractor name
        title: Contract title/name
        description: Detailed contract description
        start_date: Contract start date
        end_date: Contract end date
        value: Total contract value
        currency: Currency code (e.g., USD, EUR)
        terms: Contract terms and conditions
        status: Current contract status
        payment_schedule: Payment milestones and dates
        deliverables: List of contract deliverables
        template_file: Path to contract template file
        signed_file: Path to signed contract file
        metadata: Additional flexible data storage
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    contract_type: ContractType = ContractType.SUPPLY
    vendor: str = ""
    title: str = ""
    description: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    value: float = 0.0
    currency: str = "USD"
    terms: str = ""
    status: ContractStatus = ContractStatus.DRAFT
    payment_schedule: List[Dict[str, Any]] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    template_file: Optional[str] = None
    signed_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary representation."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "contract_type": self.contract_type.value,
            "vendor": self.vendor,
            "title": self.title,
            "description": self.description,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "value": self.value,
            "currency": self.currency,
            "terms": self.terms,
            "status": self.status.value,
            "payment_schedule": self.payment_schedule,
            "deliverables": self.deliverables,
            "template_file": self.template_file,
            "signed_file": self.signed_file,
            "metadata": self.metadata,
        }


@dataclass
class Portfolio:
    """
    Collection of related PV projects for portfolio management.

    Attributes:
        id: Unique portfolio identifier
        name: Portfolio display name
        description: Portfolio description
        owner: Portfolio owner/manager name
        project_ids: List of project IDs in portfolio
        created_date: Portfolio creation timestamp
        total_capacity_kwp: Sum of all project capacities
        total_budget: Sum of all project budgets
        roi_target: Return on investment target percentage
        metadata: Additional flexible data storage
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    owner: str = ""
    project_ids: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    total_capacity_kwp: float = 0.0
    total_budget: float = 0.0
    roi_target: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "project_ids": self.project_ids,
            "created_date": self.created_date.isoformat(),
            "total_capacity_kwp": self.total_capacity_kwp,
            "total_budget": self.total_budget,
            "roi_target": self.roi_target,
            "metadata": self.metadata,
        }


@dataclass
class Timeline:
    """
    Project timeline and milestone tracking.

    Attributes:
        id: Unique timeline identifier
        project_id: Associated project identifier
        milestones: List of milestone dictionaries with name, date, status
        phases: Project phases with start/end dates
        critical_path: List of critical tasks
        dependencies: Task dependency mapping
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)

    def add_milestone(self, name: str, date: datetime, description: str = "") -> None:
        """
        Add a milestone to the timeline.

        Args:
            name: Milestone name
            date: Milestone target date
            description: Optional milestone description
        """
        milestone = {
            "id": str(uuid.uuid4()),
            "name": name,
            "date": date,
            "description": description,
            "completed": False,
        }
        self.milestones.append(milestone)

    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline to dictionary representation."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "milestones": [
                {
                    **m,
                    "date": m["date"].isoformat() if isinstance(m["date"], datetime) else m["date"]
                }
                for m in self.milestones
            ],
            "phases": self.phases,
            "critical_path": self.critical_path,
            "dependencies": self.dependencies,
        }
