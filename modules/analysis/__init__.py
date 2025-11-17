"""
Analysis Suite - Group 2.

This package contains comprehensive analysis modules for:
- IEC standards testing and certification (Branch B04)
- System design and optimization (Branch B05)
- Weather data and energy yield assessment (Branch B06)
"""

from modules.analysis.iec_testing import IECTestingSimulator, render_iec_testing
from modules.analysis.system_design import SystemDesignOptimizer, render_system_design
from modules.analysis.weather_eya import WeatherEnergyAnalyzer, render_weather_eya

__all__ = [
    'IECTestingSimulator',
    'render_iec_testing',
    'SystemDesignOptimizer',
    'render_system_design',
    'WeatherEnergyAnalyzer',
    'render_weather_eya',
]
