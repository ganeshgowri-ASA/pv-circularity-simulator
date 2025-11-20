"""
Unit tests for data models.

Tests all core data models to ensure proper functionality,
serialization, and validation.
"""

import pytest
from datetime import datetime
from src.data.models import (
    Project, Resource, Contract, Portfolio, Timeline,
    ProjectStatus, ResourceType, ContractType, ContractStatus
)


class TestProject:
    """Test suite for Project model."""

    def test_project_creation(self):
        """Test basic project creation."""
        project = Project(
            name="Test Solar Farm",
            description="Test project for unit testing",
            status=ProjectStatus.DESIGN,
            capacity_kwp=100.0,
            location="Phoenix, AZ",
            owner="Test Owner",
            budget=150000.0
        )

        assert project.name == "Test Solar Farm"
        assert project.status == ProjectStatus.DESIGN
        assert project.capacity_kwp == 100.0
        assert project.budget == 150000.0
        assert project.id is not None

    def test_project_to_dict(self):
        """Test project serialization to dictionary."""
        project = Project(
            name="Test Project",
            owner="Owner",
            location="Location",
            budget=100000.0
        )

        data = project.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test Project"
        assert data["owner"] == "Owner"
        assert "id" in data
        assert "created_date" in data
        assert data["status"] == ProjectStatus.DESIGN.value

    def test_project_with_dates(self):
        """Test project with timeline dates."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        project = Project(
            name="Timed Project",
            owner="Owner",
            location="Location",
            budget=0,
            start_date=start,
            end_date=end
        )

        assert project.start_date == start
        assert project.end_date == end

        data = project.to_dict()
        assert data["start_date"] == start.isoformat()
        assert data["end_date"] == end.isoformat()


class TestResource:
    """Test suite for Resource model."""

    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = Resource(
            project_id="test-project-123",
            name="PV Module 400W",
            resource_type=ResourceType.MODULE,
            quantity=250,
            unit="pieces",
            unit_cost=200.0,
            supplier="SolarTech Inc."
        )

        assert resource.name == "PV Module 400W"
        assert resource.resource_type == ResourceType.MODULE
        assert resource.quantity == 250
        assert resource.unit_cost == 200.0

    def test_resource_cost_calculation(self):
        """Test automatic cost calculation."""
        resource = Resource(
            project_id="test-project",
            name="Test Resource",
            quantity=100,
            unit_cost=50.0
        )

        total = resource.calculate_total_cost()

        assert total == 5000.0
        assert resource.total_cost == 5000.0

    def test_resource_serialization(self):
        """Test resource to_dict method."""
        resource = Resource(
            project_id="test",
            name="Test",
            resource_type=ResourceType.INVERTER,
            quantity=5,
            unit="pieces",
            unit_cost=1000.0
        )

        data = resource.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test"
        assert data["resource_type"] == ResourceType.INVERTER.value
        assert "id" in data


class TestContract:
    """Test suite for Contract model."""

    def test_contract_creation(self):
        """Test basic contract creation."""
        contract = Contract(
            project_id="test-project",
            title="Supply Agreement",
            vendor="Vendor Inc.",
            contract_type=ContractType.SUPPLY,
            value=50000.0,
            currency="USD",
            status=ContractStatus.DRAFT
        )

        assert contract.title == "Supply Agreement"
        assert contract.vendor == "Vendor Inc."
        assert contract.value == 50000.0
        assert contract.status == ContractStatus.DRAFT

    def test_contract_with_deliverables(self):
        """Test contract with deliverables list."""
        deliverables = [
            "250 PV Modules",
            "Installation Support",
            "Documentation"
        ]

        contract = Contract(
            project_id="test",
            title="Test Contract",
            vendor="Vendor",
            value=0,
            deliverables=deliverables
        )

        assert len(contract.deliverables) == 3
        assert contract.deliverables[0] == "250 PV Modules"

    def test_contract_payment_schedule(self):
        """Test contract with payment schedule."""
        payment_schedule = [
            {
                "description": "Deposit",
                "amount": 10000.0,
                "due_date": "2024-01-15"
            },
            {
                "description": "Final Payment",
                "amount": 40000.0,
                "due_date": "2024-06-30"
            }
        ]

        contract = Contract(
            project_id="test",
            title="Test",
            vendor="Vendor",
            value=50000.0,
            payment_schedule=payment_schedule
        )

        assert len(contract.payment_schedule) == 2
        assert contract.payment_schedule[0]["amount"] == 10000.0

    def test_contract_serialization(self):
        """Test contract to_dict method."""
        contract = Contract(
            project_id="test",
            title="Test Contract",
            vendor="Test Vendor",
            contract_type=ContractType.SERVICE,
            value=25000.0,
            status=ContractStatus.ACTIVE
        )

        data = contract.to_dict()

        assert isinstance(data, dict)
        assert data["title"] == "Test Contract"
        assert data["contract_type"] == ContractType.SERVICE.value
        assert data["status"] == ContractStatus.ACTIVE.value


class TestPortfolio:
    """Test suite for Portfolio model."""

    def test_portfolio_creation(self):
        """Test basic portfolio creation."""
        portfolio = Portfolio(
            name="Test Portfolio",
            description="Portfolio for testing",
            owner="Portfolio Manager",
            total_capacity_kwp=500.0,
            total_budget=750000.0,
            roi_target=15.0
        )

        assert portfolio.name == "Test Portfolio"
        assert portfolio.total_capacity_kwp == 500.0
        assert portfolio.roi_target == 15.0

    def test_portfolio_with_projects(self):
        """Test portfolio with project IDs."""
        project_ids = ["project-1", "project-2", "project-3"]

        portfolio = Portfolio(
            name="Multi-Project Portfolio",
            owner="Owner",
            project_ids=project_ids
        )

        assert len(portfolio.project_ids) == 3
        assert "project-1" in portfolio.project_ids

    def test_portfolio_serialization(self):
        """Test portfolio to_dict method."""
        portfolio = Portfolio(
            name="Test",
            owner="Owner",
            total_capacity_kwp=100.0
        )

        data = portfolio.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test"
        assert "created_date" in data


class TestTimeline:
    """Test suite for Timeline model."""

    def test_timeline_creation(self):
        """Test basic timeline creation."""
        timeline = Timeline(
            project_id="test-project"
        )

        assert timeline.project_id == "test-project"
        assert timeline.milestones == []
        assert timeline.phases == []

    def test_add_milestone(self):
        """Test adding milestones to timeline."""
        timeline = Timeline(project_id="test")

        milestone_date = datetime(2024, 6, 1)
        timeline.add_milestone(
            name="Design Approval",
            date=milestone_date,
            description="Complete design phase"
        )

        assert len(timeline.milestones) == 1
        assert timeline.milestones[0]["name"] == "Design Approval"
        assert timeline.milestones[0]["completed"] is False

    def test_timeline_serialization(self):
        """Test timeline to_dict method."""
        timeline = Timeline(project_id="test")
        timeline.add_milestone("Milestone 1", datetime.now())

        data = timeline.to_dict()

        assert isinstance(data, dict)
        assert data["project_id"] == "test"
        assert len(data["milestones"]) == 1


class TestEnums:
    """Test suite for enumeration types."""

    def test_project_status_enum(self):
        """Test ProjectStatus enum values."""
        assert ProjectStatus.DESIGN.value == "Design"
        assert ProjectStatus.ENGINEERING.value == "Engineering"
        assert ProjectStatus.PLANNING.value == "Planning"
        assert ProjectStatus.IMPLEMENTATION.value == "Implementation"
        assert ProjectStatus.MONITORING.value == "Monitoring"
        assert ProjectStatus.EOL.value == "End of Life"

    def test_resource_type_enum(self):
        """Test ResourceType enum values."""
        assert ResourceType.MODULE.value == "PV Module"
        assert ResourceType.INVERTER.value == "Inverter"
        assert ResourceType.CABLE.value == "Cable"
        assert ResourceType.LABOR.value == "Labor"
        assert ResourceType.CAPITAL.value == "Capital"

    def test_contract_type_enum(self):
        """Test ContractType enum values."""
        assert ContractType.SUPPLY.value == "Supply Agreement"
        assert ContractType.LABOR.value == "Labor Contract"
        assert ContractType.SERVICE.value == "Service Agreement"
        assert ContractType.MAINTENANCE.value == "Maintenance Contract"

    def test_contract_status_enum(self):
        """Test ContractStatus enum values."""
        assert ContractStatus.DRAFT.value == "Draft"
        assert ContractStatus.PENDING.value == "Pending Approval"
        assert ContractStatus.ACTIVE.value == "Active"
        assert ContractStatus.COMPLETED.value == "Completed"
        assert ContractStatus.CANCELLED.value == "Cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
