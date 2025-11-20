"""
Monitoring Suite Package - Group 3.

This package provides comprehensive monitoring and diagnostics capabilities:

Branch B07 - Real-time Performance Monitoring:
    - Real-time KPI tracking (PR, capacity factor, specific yield)
    - Power and energy monitoring (DC, AC, daily, monthly, annual)
    - Inverter performance tracking
    - System availability calculations
    - Performance benchmarking
    - Alerts and threshold monitoring

Branch B08 - Fault Detection & Diagnostics:
    - Hot spot detection (thermal imaging)
    - Cell crack detection (EL imaging)
    - Bypass diode failure detection
    - Soiling detection and quantification
    - Delamination detection
    - PID detection and mitigation
    - Fault severity classification

Branch B09 - Energy Forecasting:
    - ML ensemble forecasting (7-day)
    - Statistical models (ARIMA-like)
    - Prophet-like time series forecasting
    - LSTM deep learning simulation
    - Hybrid weather-based forecasting
    - Confidence intervals and uncertainty quantification
    - Forecast accuracy metrics (MAE, RMSE, MAPE, R²)
"""

from modules.monitoring.performance_monitoring import (
    PerformanceMonitor,
    render_performance_monitoring
)

from modules.monitoring.fault_diagnostics import (
    FaultDiagnostics,
    render_fault_diagnostics
)

from modules.monitoring.energy_forecasting import (
    EnergyForecaster,
    render_energy_forecasting
)

__all__ = [
    # Performance Monitoring (B07)
    'PerformanceMonitor',
    'render_performance_monitoring',

    # Fault Diagnostics (B08)
    'FaultDiagnostics',
    'render_fault_diagnostics',

    # Energy Forecasting (B09)
    'EnergyForecaster',
    'render_energy_forecasting',
]

__version__ = '1.0.0'
__author__ = 'PV Circularity Simulator Team'
