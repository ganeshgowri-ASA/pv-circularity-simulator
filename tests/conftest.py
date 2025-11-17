"""
Pytest configuration and shared fixtures for PV Circularity Simulator tests.

This module provides common test fixtures and configuration used across
all test modules.
"""

import pytest
from typing import List

from src.pv_simulator.core.models import (
    InvestmentInput,
    CashFlow,
    SensitivityInput,
)
from src.pv_simulator.core.enums import CurrencyType, SensitivityParameter


@pytest.fixture
def basic_investment_input():
    """Basic investment scenario for testing."""
    return InvestmentInput(
        initial_investment=100000,
        annual_revenue=25000,
        annual_costs=5000,
        discount_rate=0.10,
        project_lifetime=25,
        currency=CurrencyType.USD,
        tax_rate=0.0,
        salvage_value=0.0,
        inflation_rate=0.0,
    )


@pytest.fixture
def complex_investment_input():
    """Complex investment scenario with tax, salvage value, and inflation."""
    return InvestmentInput(
        initial_investment=500000,
        annual_revenue=120000,
        annual_costs=30000,
        discount_rate=0.08,
        project_lifetime=30,
        currency=CurrencyType.EUR,
        tax_rate=0.21,
        salvage_value=50000,
        inflation_rate=0.02,
    )


@pytest.fixture
def simple_cash_flows() -> List[CashFlow]:
    """Simple cash flow scenario for testing."""
    return [
        CashFlow(
            year=0,
            inflow=0,
            outflow=100000,
            net_flow=-100000,
            cumulative_flow=-100000,
            discounted_flow=-100000,
        ),
        CashFlow(
            year=1,
            inflow=25000,
            outflow=5000,
            net_flow=20000,
            cumulative_flow=-80000,
            discounted_flow=18181.82,
        ),
        CashFlow(
            year=2,
            inflow=25000,
            outflow=5000,
            net_flow=20000,
            cumulative_flow=-60000,
            discounted_flow=16528.93,
        ),
        CashFlow(
            year=3,
            inflow=25000,
            outflow=5000,
            net_flow=20000,
            cumulative_flow=-40000,
            discounted_flow=15026.30,
        ),
        CashFlow(
            year=4,
            inflow=25000,
            outflow=5000,
            net_flow=20000,
            cumulative_flow=-20000,
            discounted_flow=13660.27,
        ),
        CashFlow(
            year=5,
            inflow=25000,
            outflow=5000,
            net_flow=20000,
            cumulative_flow=0,
            discounted_flow=12418.43,
        ),
    ]


@pytest.fixture
def sensitivity_input_discount_rate():
    """Sensitivity input for discount rate analysis."""
    return SensitivityInput(
        parameter=SensitivityParameter.DISCOUNT_RATE,
        base_value=0.10,
        variation_range=[-20, -10, 0, 10, 20],
    )


@pytest.fixture
def sensitivity_input_revenue():
    """Sensitivity input for revenue analysis."""
    return SensitivityInput(
        parameter=SensitivityParameter.ANNUAL_REVENUE,
        base_value=25000,
        variation_range=[-30, -15, 0, 15, 30],
    )
