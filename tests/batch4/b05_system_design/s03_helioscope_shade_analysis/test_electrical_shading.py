"""
Unit tests for ElectricalShadingModel.
"""

import unittest

from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.electrical_shading import (
    ElectricalShadingModel,
)
from src.pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.models import (
    ModuleElectricalParams,
)


class TestElectricalShadingModel(unittest.TestCase):
    """Test cases for ElectricalShadingModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.module_params = ModuleElectricalParams(
            cells_in_series=72,
            cell_rows=6,
            cell_columns=12,
            bypass_diodes=3,
            cells_per_diode=24,
            v_oc=48.0,
            i_sc=10.0,
            v_mp=40.0,
            i_mp=9.5,
            p_max=400.0
        )

        self.model = ElectricalShadingModel(self.module_params)

    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.cells_per_substring, 24)

    def test_bypass_diode_simulation(self):
        """Test bypass diode simulation."""
        shaded_cells = [0, 1, 2, 3, 4]

        result = self.model.bypass_diode_simulation(shaded_cells)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.power_loss, 0)
        self.assertLessEqual(result.power_loss, 1)

    def test_substring_shading(self):
        """Test substring shading calculation."""
        result = self.model.substring_shading(
            substring_index=0,
            shading_fraction=0.5
        )

        self.assertIn('power_fraction', result)
        self.assertIn('bypass_active', result)
        self.assertGreaterEqual(result['power_fraction'], 0)
        self.assertLessEqual(result['power_fraction'], 1)

    def test_mismatch_losses(self):
        """Test current mismatch calculation."""
        module_irradiances = [1000.0, 900.0, 800.0, 1000.0]

        mismatch_loss = self.model.mismatch_losses(module_irradiances)

        self.assertGreaterEqual(mismatch_loss, 0)
        self.assertLessEqual(mismatch_loss, 1)

    def test_module_iv_under_shade(self):
        """Test I-V curve generation under shading."""
        shaded_cells = [0, 1, 2]
        irradiance_on_cells = [0 if i in shaded_cells else 1000 for i in range(72)]

        voltage, current = self.model.module_iv_under_shade(
            shaded_cells,
            irradiance_on_cells
        )

        self.assertEqual(len(voltage), len(current))
        self.assertGreater(len(voltage), 0)

    def test_mppt_behavior_under_shade(self):
        """Test MPPT behavior simulation."""
        shaded_cells = [0, 1, 2, 3, 4]
        irradiance_on_cells = [0 if i in shaded_cells else 1000 for i in range(72)]

        result = self.model.mppt_behavior_under_shade(shaded_cells, irradiance_on_cells)

        self.assertIn('v_mpp', result)
        self.assertIn('i_mpp', result)
        self.assertIn('p_mpp', result)
        self.assertIn('mppt_efficiency', result)

    def test_hotspot_risk_analysis(self):
        """Test hotspot risk analysis."""
        shaded_cells = [0, 1, 2]
        irradiance_on_cells = [0 if i in shaded_cells else 1000 for i in range(72)]

        result = self.model.hotspot_risk_analysis(shaded_cells, irradiance_on_cells)

        self.assertIn('hotspot_risk', result)
        self.assertIn('hotspot_cells', result)
        self.assertIsInstance(result['hotspot_risk'], bool)


if __name__ == '__main__':
    unittest.main()
