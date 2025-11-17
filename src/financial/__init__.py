"""
PV Circularity Simulator - Financial Analysis Module.

This module provides comprehensive financial analysis capabilities for
photovoltaic systems including:
- LCOE (Levelized Cost of Energy) calculations
- Cash flow modeling and analysis
- Sensitivity analysis and risk assessment
- Circular economy impact quantification
- Professional report generation
- Interactive dashboard interface

Quick Start:
    >>> from financial.dashboard import run_dashboard
    >>> run_dashboard()

Or use components individually:
    >>> from financial.models import CostStructure, RevenueStream
    >>> from financial.calculators import LCOECalculator
    >>> calculator = LCOECalculator(cost_structure, revenue_stream)
    >>> result = calculator.calculate_lcoe()
"""

from .models import (
    CostStructure,
    RevenueStream,
    CircularityMetrics,
    CashFlowModel,
    SensitivityParameter,
)
from .calculators import LCOECalculator, SensitivityAnalyzer
from .visualization import FinancialChartBuilder
from .reporting import FinancialReportGenerator
from .dashboard import FinancialDashboardUI, run_dashboard

__version__ = '1.0.0'

__all__ = [
    'CostStructure',
    'RevenueStream',
    'CircularityMetrics',
    'CashFlowModel',
    'SensitivityParameter',
    'LCOECalculator',
    'SensitivityAnalyzer',
    'FinancialChartBuilder',
    'FinancialReportGenerator',
    'FinancialDashboardUI',
    'run_dashboard',
]
