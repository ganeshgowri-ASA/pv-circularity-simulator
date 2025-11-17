"""SCADA system integration components."""

from .connector import SCADAConnector
from .aggregator import DataAggregator

__all__ = [
    "SCADAConnector",
    "DataAggregator",
]
