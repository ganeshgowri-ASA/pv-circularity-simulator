"""
Enumerations for the PV Circularity Simulator.

This module defines all enum types used across the application for type safety
and validation.
"""

from enum import Enum


class CurrencyType(str, Enum):
    """Supported currency types for financial calculations."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"
    CNY = "CNY"
    JPY = "JPY"


class SensitivityParameter(str, Enum):
    """Parameters that can be varied in sensitivity analysis."""

    DISCOUNT_RATE = "discount_rate"
    INITIAL_INVESTMENT = "initial_investment"
    ANNUAL_REVENUE = "annual_revenue"
    ANNUAL_COSTS = "annual_costs"
    PROJECT_LIFETIME = "project_lifetime"
    DEGRADATION_RATE = "degradation_rate"
    ELECTRICITY_PRICE = "electricity_price"


class AnalysisType(str, Enum):
    """Types of financial analysis."""

    ROI = "roi"
    NPV = "npv"
    IRR = "irr"
    PAYBACK = "payback"
    SENSITIVITY = "sensitivity"
