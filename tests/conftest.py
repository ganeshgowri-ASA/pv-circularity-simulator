"""Pytest configuration and shared fixtures for tests."""

import pytest

from pv_simulator.financial.models import (
    CashFlowInput,
    CashFlowProjection,
    DiscountRateConfig,
)


@pytest.fixture
def simple_cash_flow_input() -> CashFlowInput:
    """Simple cash flow input for basic testing."""
    return CashFlowInput(
        initial_investment=100000.0,
        cash_flows=[20000.0, 25000.0, 30000.0, 35000.0, 40000.0],
        discount_rate=0.10,
        project_name="Test Project"
    )


@pytest.fixture
def negative_npv_cash_flow() -> CashFlowInput:
    """Cash flow input that results in negative NPV."""
    return CashFlowInput(
        initial_investment=100000.0,
        cash_flows=[5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
        discount_rate=0.10,
        project_name="Negative NPV Project"
    )


@pytest.fixture
def high_return_cash_flow() -> CashFlowInput:
    """Cash flow input with high returns."""
    return CashFlowInput(
        initial_investment=50000.0,
        cash_flows=[25000.0, 30000.0, 35000.0, 40000.0, 45000.0],
        discount_rate=0.08,
        project_name="High Return Project"
    )


@pytest.fixture
def discount_rate_config() -> DiscountRateConfig:
    """Standard discount rate configuration for sensitivity analysis."""
    return DiscountRateConfig(
        base_rate=0.10,
        min_rate=0.05,
        max_rate=0.20,
        step_size=0.01
    )


@pytest.fixture
def cash_flow_projection() -> CashFlowProjection:
    """Cash flow projection for modeling tests."""
    return CashFlowProjection(
        periods=[1, 2, 3, 4, 5],
        revenues=[50000.0, 55000.0, 60000.0, 65000.0, 70000.0],
        operating_costs=[20000.0, 21000.0, 22000.0, 23000.0, 24000.0],
        capital_expenditures=[5000.0, 0.0, 0.0, 0.0, 10000.0],
        initial_investment=100000.0,
        terminal_value=50000.0
    )


@pytest.fixture
def pv_project_cash_flow() -> CashFlowInput:
    """Realistic PV solar project cash flow."""
    return CashFlowInput(
        initial_investment=500000.0,  # $500k installation
        cash_flows=[
            80000.0,   # Year 1: Energy sales
            82000.0,   # Year 2: Slight increase
            84000.0,   # Year 3
            86000.0,   # Year 4
            88000.0,   # Year 5
            90000.0,   # Year 6
            92000.0,   # Year 7
            94000.0,   # Year 8
            96000.0,   # Year 9
            98000.0,   # Year 10: Last year
        ],
        discount_rate=0.08,
        project_name="PV Solar Installation"
    )
