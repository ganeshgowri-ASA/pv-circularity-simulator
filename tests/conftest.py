"""Shared pytest fixtures for test suite."""

from datetime import date

import pytest

from pv_simulator.models.incentives import (
    DepreciationMethod,
    ITCConfiguration,
    PTCConfiguration,
    SystemConfiguration,
    TaxEquityConfiguration,
)
from pv_simulator.simulators.incentive_modeler import IncentiveModeler


@pytest.fixture
def sample_system_config() -> SystemConfiguration:
    """Create a sample system configuration for testing.

    Returns:
        SystemConfiguration with typical commercial solar system parameters.
    """
    return SystemConfiguration(
        system_size_kw=100.0,
        installation_cost_total=250_000.0,
        installation_date=date(2024, 1, 15),
        location_state="CA",
        expected_annual_production_kwh=150_000.0,
        system_lifetime_years=25,
        module_efficiency=0.20,
        inverter_efficiency=0.96,
    )


@pytest.fixture
def large_system_config() -> SystemConfiguration:
    """Create a large commercial system configuration.

    Returns:
        SystemConfiguration for a 1MW commercial installation.
    """
    return SystemConfiguration(
        system_size_kw=1000.0,
        installation_cost_total=2_500_000.0,
        installation_date=date(2024, 6, 1),
        location_state="TX",
        expected_annual_production_kwh=1_500_000.0,
        system_lifetime_years=30,
        module_efficiency=0.22,
        inverter_efficiency=0.98,
    )


@pytest.fixture
def itc_config_basic(sample_system_config: SystemConfiguration) -> ITCConfiguration:
    """Create a basic ITC configuration.

    Args:
        sample_system_config: Sample system configuration fixture.

    Returns:
        ITCConfiguration with standard 30% ITC.
    """
    return ITCConfiguration(
        system_config=sample_system_config,
        itc_rate=0.30,
        apply_bonus=False,
    )


@pytest.fixture
def itc_config_with_bonus(sample_system_config: SystemConfiguration) -> ITCConfiguration:
    """Create an ITC configuration with bonus credits.

    Args:
        sample_system_config: Sample system configuration fixture.

    Returns:
        ITCConfiguration with bonus credits for domestic content.
    """
    return ITCConfiguration(
        system_config=sample_system_config,
        itc_rate=0.30,
        apply_bonus=True,
        bonus_rate=0.10,
        meets_domestic_content=True,
        is_energy_community=False,
    )


@pytest.fixture
def ptc_config_basic(sample_system_config: SystemConfiguration) -> PTCConfiguration:
    """Create a basic PTC configuration.

    Args:
        sample_system_config: Sample system configuration fixture.

    Returns:
        PTCConfiguration with standard parameters.
    """
    return PTCConfiguration(
        system_config=sample_system_config,
        ptc_rate_per_kwh=0.0275,
        credit_period_years=10,
        inflation_adjustment=True,
        inflation_rate=0.025,
        production_degradation_rate=0.005,
    )


@pytest.fixture
def ptc_config_with_bonus(sample_system_config: SystemConfiguration) -> PTCConfiguration:
    """Create a PTC configuration with bonus multiplier.

    Args:
        sample_system_config: Sample system configuration fixture.

    Returns:
        PTCConfiguration with 5x bonus multiplier.
    """
    return PTCConfiguration(
        system_config=sample_system_config,
        ptc_rate_per_kwh=0.0275,
        credit_period_years=10,
        inflation_adjustment=True,
        apply_bonus=True,
        bonus_multiplier=5.0,
    )


@pytest.fixture
def tax_equity_config(sample_system_config: SystemConfiguration) -> TaxEquityConfiguration:
    """Create a tax equity configuration.

    Args:
        sample_system_config: Sample system configuration fixture.

    Returns:
        TaxEquityConfiguration with typical partnership flip parameters.
    """
    return TaxEquityConfiguration(
        system_config=sample_system_config,
        investor_equity_percentage=0.99,
        target_flip_irr=0.08,
        post_flip_investor_percentage=0.05,
        tax_rate=0.40,
        project_lifetime_years=25,
        include_itc=True,
        include_depreciation=True,
    )


@pytest.fixture
def incentive_modeler() -> IncentiveModeler:
    """Create an IncentiveModeler instance.

    Returns:
        Fresh IncentiveModeler instance for testing.
    """
    return IncentiveModeler()
