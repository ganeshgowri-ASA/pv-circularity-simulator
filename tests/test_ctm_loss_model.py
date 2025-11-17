"""
Comprehensive tests for CTM Loss Model implementation.
Validates against expected values and Cell-to-Module.com data.
"""

import pytest
import numpy as np
from src.modules.ctm_loss_model import (
    CTMLossModel,
    CellParameters,
    ModuleParameters,
    ModuleType,
    EncapsulantType,
)


class TestCellParameters:
    """Test CellParameters validation."""

    def test_valid_cell_parameters(self):
        """Test creation of valid cell parameters."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        assert cell.power_stc == 5.2
        assert cell.efficiency == 22.5

    def test_invalid_efficiency(self):
        """Test that efficiency must be <= 100%."""
        with pytest.raises(ValueError):
            CellParameters(
                power_stc=5.2,
                voltage_mpp=0.65,
                current_mpp=8.0,
                voltage_oc=0.72,
                current_sc=8.5,
                efficiency=150,  # Invalid
                width=166,
                height=166,
            )

    def test_power_validation(self):
        """Test power validation range."""
        with pytest.raises(ValueError):
            CellParameters(
                power_stc=15.0,  # Too high for single cell
                voltage_mpp=0.65,
                current_mpp=8.0,
                voltage_oc=0.72,
                current_sc=8.5,
                efficiency=22.5,
                width=166,
                height=166,
            )


class TestModuleParameters:
    """Test ModuleParameters validation."""

    def test_standard_module(self):
        """Test standard 60-cell module."""
        module = ModuleParameters(
            cells_in_series=60,
            cells_in_parallel=1,
        )
        assert module.total_cells == 60
        assert module.module_type == ModuleType.STANDARD

    def test_half_cut_module(self):
        """Test half-cut module configuration."""
        module = ModuleParameters(
            module_type=ModuleType.HALF_CUT,
            cells_in_series=60,
            cells_in_parallel=2,
        )
        assert module.total_cells == 120

    def test_bifacial_module(self):
        """Test bifacial module parameters."""
        module = ModuleParameters(
            cells_in_series=60,
            is_bifacial=True,
            bifaciality_factor=0.75,
        )
        assert module.is_bifacial
        assert module.bifaciality_factor == 0.75


class TestCTMLossModelBasic:
    """Basic CTM Loss Model tests."""

    @pytest.fixture
    def standard_cell(self):
        """Standard PERC cell parameters."""
        return CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
            front_grid_coverage=2.5,
            temp_coeff_power=-0.40,
        )

    @pytest.fixture
    def standard_module(self):
        """Standard 60-cell module."""
        return ModuleParameters(
            cells_in_series=60,
            cells_in_parallel=1,
            module_type=ModuleType.STANDARD,
        )

    def test_model_initialization(self, standard_cell, standard_module):
        """Test model initialization."""
        model = CTMLossModel(standard_cell, standard_module)
        assert model.cell == standard_cell
        assert model.module == standard_module

    def test_k_factor_calculation(self, standard_cell, standard_module):
        """Test that all k-factors are calculated."""
        model = CTMLossModel(standard_cell, standard_module)
        k_factors = model.calculate_all_k_factors()

        # Check all k-factors exist
        expected_factors = [
            'k1_glass_reflection', 'k2_encapsulant_gain', 'k3_grid_correction',
            'k4_inactive_area', 'k5_glass_absorption', 'k6_encapsulant_absorption',
            'k7_rear_optical', 'k8_cell_gaps', 'k9_internal_mismatch',
            'k10_module_mismatch', 'k11_lid_letid', 'k12_resistive',
            'k13_interconnection', 'k14_manufacturing', 'k15_inactive_electrical',
            'k21_temperature', 'k22_low_irradiance', 'k23_spectral', 'k24_aoi',
        ]

        for factor in expected_factors:
            assert factor in k_factors
            assert isinstance(k_factors[factor], (int, float))

    def test_module_power_calculation(self, standard_cell, standard_module):
        """Test module power calculation."""
        model = CTMLossModel(standard_cell, standard_module)
        module_power = model.calculate_module_power()

        # Module power should be less than sum of cell powers (losses > gains)
        total_cell_power = standard_cell.power_stc * 60
        assert 0 < module_power < total_cell_power * 1.1  # Allow for small gains

    def test_ctm_ratio(self, standard_cell, standard_module):
        """Test CTM ratio calculation."""
        model = CTMLossModel(standard_cell, standard_module)
        ctm_ratio = model.get_ctm_ratio()

        # CTM ratio typically 0.94-0.99 for standard modules
        assert 0.90 < ctm_ratio < 1.05


class TestOpticalFactors:
    """Test optical k-factors (k1-k7)."""

    @pytest.fixture
    def model(self):
        """Create standard model."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(cells_in_series=60)
        return CTMLossModel(cell, module)

    def test_k1_glass_reflection(self, model):
        """Test glass reflection gain (k1)."""
        k1 = model.calculate_k1_glass_reflection_gain()
        # Should be small gain (1.005 to 1.02)
        assert 1.0 < k1 < 1.03

    def test_k2_encapsulant_gain(self, model):
        """Test encapsulant gain (k2)."""
        k2 = model.calculate_k2_encapsulant_gain()
        # Should be small gain
        assert 1.0 < k2 < 1.02

    def test_k3_grid_correction(self, model):
        """Test front grid shading correction (k3)."""
        k3 = model.calculate_k3_front_grid_correction()
        # Should be close to 1 or slight loss
        assert 0.95 < k3 < 1.01

    def test_k4_inactive_area(self, model):
        """Test inactive area loss (k4)."""
        k4 = model.calculate_k4_inactive_area_loss()
        # Should be small loss
        assert 0.99 < k4 <= 1.0

    def test_k5_glass_absorption(self, model):
        """Test glass absorption (k5)."""
        k5 = model.calculate_k5_glass_absorption()
        # Glass absorbs some light
        assert 0.96 < k5 < 0.99

    def test_k6_encapsulant_absorption(self, model):
        """Test encapsulant absorption (k6)."""
        k6 = model.calculate_k6_encapsulant_absorption()
        # Small absorption
        assert 0.97 < k6 <= 1.0

    def test_k7_rear_optical_monofacial(self, model):
        """Test rear optical for monofacial (k7)."""
        k7 = model.calculate_k7_rear_optical_properties()
        # Monofacial: minimal effect
        assert 0.99 < k7 < 1.02

    def test_k7_rear_optical_bifacial(self):
        """Test rear optical for bifacial (k7)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            is_bifacial=True,
            bifaciality_factor=0.75,
        )
        model = CTMLossModel(cell, module)
        k7 = model.calculate_k7_rear_optical_properties()
        # Bifacial: significant gain
        assert 1.1 < k7 < 1.3


class TestCouplingFactors:
    """Test coupling k-factors (k8-k11)."""

    @pytest.fixture
    def model(self):
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(cells_in_series=60)
        return CTMLossModel(cell, module)

    def test_k8_cell_gaps(self, model):
        """Test cell gap losses (k8)."""
        k8 = model.calculate_k8_cell_gap_losses()
        # Gaps reduce active area
        assert 0.95 < k8 < 1.0

    def test_k9_internal_mismatch(self, model):
        """Test internal mismatch (k9)."""
        k9 = model.calculate_k9_internal_mismatch()
        # Small mismatch loss
        assert 0.985 < k9 < 1.0

    def test_k10_module_mismatch(self, model):
        """Test module mismatch (k10)."""
        k10 = model.calculate_k10_module_mismatch()
        # Very small loss
        assert 0.995 < k10 <= 1.0

    def test_k11_lid_letid(self, model):
        """Test LID/LETID (k11)."""
        k11 = model.calculate_k11_lid_letid()
        # LID causes loss
        assert 0.97 < k11 < 1.0


class TestElectricalFactors:
    """Test electrical k-factors (k12-k15)."""

    @pytest.fixture
    def model(self):
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(cells_in_series=60)
        return CTMLossModel(cell, module)

    def test_k12_resistive_losses(self, model):
        """Test resistive losses (k12)."""
        k12 = model.calculate_k12_resistive_losses()
        # I²R losses
        assert 0.97 < k12 < 1.0

    def test_k13_interconnection(self, model):
        """Test interconnection resistance (k13)."""
        k13 = model.calculate_k13_interconnection_resistance()
        # Small contact loss
        assert 0.995 < k13 <= 1.0

    def test_k14_manufacturing_damage(self, model):
        """Test manufacturing damage (k14)."""
        k14 = model.calculate_k14_manufacturing_damage()
        # Small damage loss
        assert 0.995 < k14 <= 1.0

    def test_k15_inactive_electrical(self, model):
        """Test inactive electrical loss (k15)."""
        k15 = model.calculate_k15_inactive_electrical_loss()
        # Very small loss
        assert 0.997 < k15 <= 1.0


class TestEnvironmentalFactors:
    """Test environmental k-factors (k21-k24)."""

    def test_k21_temperature_stc(self):
        """Test temperature coefficient at STC (k21)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            operating_temperature=25.0,  # STC
        )
        model = CTMLossModel(cell, module)
        k21 = model.calculate_k21_temperature_coefficient()
        # At 25°C, no loss
        assert abs(k21 - 1.0) < 0.001

    def test_k21_temperature_high(self):
        """Test temperature coefficient at high temp (k21)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
            temp_coeff_power=-0.40,
        )
        module = ModuleParameters(
            cells_in_series=60,
            operating_temperature=65.0,  # Hot operation
        )
        model = CTMLossModel(cell, module)
        k21 = model.calculate_k21_temperature_coefficient()
        # At 65°C, significant loss
        assert 0.84 < k21 < 0.88

    def test_k22_low_irradiance(self):
        """Test low irradiance losses (k22)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            irradiance=200.0,  # Low light
        )
        model = CTMLossModel(cell, module)
        k22 = model.calculate_k22_low_irradiance_losses()
        # Loss at low irradiance
        assert k22 < 1.0

    def test_k23_spectral_response(self):
        """Test spectral response (k23)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
            spectral_mismatch=1.0,  # 1% gain
        )
        module = ModuleParameters(cells_in_series=60)
        model = CTMLossModel(cell, module)
        k23 = model.calculate_k23_spectral_response()
        assert abs(k23 - 1.01) < 0.001

    def test_k24_aoi_normal(self):
        """Test AOI at normal incidence (k24)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            aoi_angle=0.0,  # Normal incidence
        )
        model = CTMLossModel(cell, module)
        k24 = model.calculate_k24_angle_of_incidence()
        # At 0°, minimal loss
        assert abs(k24 - 1.0) < 0.01

    def test_k24_aoi_oblique(self):
        """Test AOI at oblique angle (k24)."""
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            aoi_angle=60.0,  # 60° angle
        )
        model = CTMLossModel(cell, module)
        k24 = model.calculate_k24_angle_of_incidence()
        # At 60°, significant loss
        assert 0.4 < k24 < 0.6


class TestAdvancedModuleTypes:
    """Test advanced module architectures."""

    @pytest.fixture
    def cell(self):
        return CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )

    def test_half_cut_module(self, cell):
        """Test half-cut module shows reduced resistive losses."""
        standard = ModuleParameters(
            cells_in_series=60,
            module_type=ModuleType.STANDARD,
        )
        half_cut = ModuleParameters(
            cells_in_series=60,
            cells_in_parallel=2,
            module_type=ModuleType.HALF_CUT,
        )

        model_std = CTMLossModel(cell, standard)
        model_hc = CTMLossModel(cell, half_cut)

        # Half-cut should have lower resistive losses
        k12_std = model_std.calculate_k12_resistive_losses()
        k12_hc = model_hc.calculate_k12_resistive_losses()
        assert k12_hc > k12_std

    def test_shingled_module(self, cell):
        """Test shingled module has no gap losses."""
        shingled = ModuleParameters(
            cells_in_series=60,
            module_type=ModuleType.SHINGLED,
        )
        model = CTMLossModel(cell, shingled)

        k8 = model.calculate_k8_cell_gap_losses()
        # Shingled has no gaps
        assert abs(k8 - 1.0) < 0.001

    def test_ibc_module(self, cell):
        """Test IBC module has no front grid shading."""
        ibc = ModuleParameters(
            cells_in_series=60,
            module_type=ModuleType.IBC,
        )
        model = CTMLossModel(cell, ibc)

        k3 = model.calculate_k3_front_grid_correction()
        # IBC has better optical performance
        assert k3 > 1.0


class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    @pytest.fixture
    def model(self):
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(cells_in_series=60)
        return CTMLossModel(cell, module)

    def test_sensitivity_analysis(self, model):
        """Test sensitivity analysis on cell efficiency."""
        results = model.sensitivity_analysis('cell.efficiency', (0.9, 1.1), 10)

        assert 'parameter_values' in results
        assert 'module_power' in results
        assert 'ctm_ratio' in results
        assert len(results['parameter_values']) == 10
        assert len(results['module_power']) == 10

    def test_multi_parameter_sensitivity(self, model):
        """Test multi-parameter sensitivity analysis."""
        params = ['cell.efficiency', 'module.glass_thickness']
        results = model.multi_parameter_sensitivity(params, num_points=5)

        assert len(results) == 2
        for param in params:
            assert param in results
            assert 'module_power' in results[param]


class TestReporting:
    """Test reporting functionality."""

    @pytest.fixture
    def model(self):
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(cells_in_series=60)
        return CTMLossModel(cell, module)

    def test_loss_breakdown(self, model):
        """Test loss breakdown by category."""
        breakdown = model.get_loss_breakdown()

        expected_categories = [
            'Optical (k1-k7)',
            'Coupling (k8-k11)',
            'Electrical (k12-k15)',
            'Environmental (k21-k24)',
            'Total CTM',
        ]

        for cat in expected_categories:
            assert cat in breakdown
            assert isinstance(breakdown[cat], (int, float))

    def test_generate_report(self, model):
        """Test text report generation."""
        report = model.generate_report()

        assert isinstance(report, str)
        assert 'CTM LOSS MODELING REPORT' in report
        assert 'CELL PARAMETERS' in report
        assert 'MODULE PARAMETERS' in report
        assert 'RESULTS' in report
        assert 'CTM Ratio' in report


class TestValidationAgainstKnownValues:
    """Validate against known CTM ratios from literature."""

    def test_standard_60cell_perc(self):
        """
        Test standard 60-cell PERC module.
        Expected CTM ratio: ~96-97% based on Cell-to-Module.com data.
        """
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=60,
            module_type=ModuleType.STANDARD,
        )
        model = CTMLossModel(cell, module)
        ctm_ratio = model.get_ctm_ratio()

        # Should be in expected range
        assert 0.95 < ctm_ratio < 0.98

    def test_half_cut_72cell(self):
        """
        Test half-cut 72-cell module.
        Expected CTM ratio: ~97-98% (better than standard).
        """
        cell = CellParameters(
            power_stc=5.2,
            voltage_mpp=0.65,
            current_mpp=8.0,
            voltage_oc=0.72,
            current_sc=8.5,
            efficiency=22.5,
            width=166,
            height=166,
        )
        module = ModuleParameters(
            cells_in_series=72,
            cells_in_parallel=2,
            module_type=ModuleType.HALF_CUT,
        )
        model = CTMLossModel(cell, module)
        ctm_ratio = model.get_ctm_ratio()

        # Half-cut should be better than standard
        assert 0.96 < ctm_ratio < 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
