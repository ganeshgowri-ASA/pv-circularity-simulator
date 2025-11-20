"""
Weather API integration module for PV Circularity Simulator.

This module provides comprehensive weather data integration from multiple providers,
including current conditions, forecasts, and historical data.
"""

from pv_simulator.weather.cache import CacheManager, create_cache_manager

__all__ = ["CacheManager", "create_cache_manager"]
