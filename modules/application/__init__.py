"""
Application Suite Module - Group 5.

This package contains the application-level modules for the PV Circularity Simulator:
- Financial Analysis & Bankability (B13)
- Infrastructure & Deployment Management (B14)
- Integrated Analytics & Reporting

These modules provide comprehensive business intelligence, project management,
and financial analysis capabilities for PV system lifecycle management.
"""

from modules.application.financial_analysis import (
    FinancialAnalyzer,
    render_financial_analysis
)

from modules.application.infrastructure import (
    InfrastructureManager,
    ProjectPhase,
    TaskStatus,
    render_infrastructure
)

from modules.application.analytics_reporting import (
    AnalyticsReporter,
    render_analytics_reporting
)

__all__ = [
    # Financial Analysis
    'FinancialAnalyzer',
    'render_financial_analysis',

    # Infrastructure Management
    'InfrastructureManager',
    'ProjectPhase',
    'TaskStatus',
    'render_infrastructure',

    # Analytics & Reporting
    'AnalyticsReporter',
    'render_analytics_reporting'
]

__version__ = '1.0.0'
__author__ = 'PV Circularity Simulator Team'
