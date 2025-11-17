"""
Modules Package
===============

Contains all feature modules for the PV Circularity Simulator.
"""

from . import dashboard
from . import material_selection
from . import module_design
from . import ctm_loss_analysis
from . import system_design
from . import eya_simulation
from . import performance_monitoring
from . import fault_diagnostics
from . import hya_simulation
from . import energy_forecasting
from . import revamp_repower
from . import circularity_3r

__all__ = [
    'dashboard',
    'material_selection',
    'module_design',
    'ctm_loss_analysis',
    'system_design',
    'eya_simulation',
    'performance_monitoring',
    'fault_diagnostics',
    'hya_simulation',
    'energy_forecasting',
    'revamp_repower',
    'circularity_3r',
]
