"""
Unit tests for bifacial module modeling.

Tests cover:
- View factor calculations
- Backside irradiance computation
- Bifacial gain calculations
- Performance simulation
- Edge cases and validation
"""

import pytest
import numpy as np
import pandas as pd
from src.modules.bifacial_model import (
    BifacialModuleModel,
    BifacialModuleParams,
    BifacialSystemConfig,
    MountingStructure,
    GroundSurface,
    TMY,
    AlbedoType,
    MountingType,
    ViewFactorModel,
    ViewFactorCalculator,
    ALBEDO_VALUES,
    calculate_gcr,
    validate_bifacial_system,
    get_albedo_seasonal_variation,
)


class TestViewFactorCalculator:
    """Test view factor calculations."""

    def test_simple_view_factor_horizontal(self):
        """Test view factor for horizontal module (0° tilt)."""
        calc = ViewFactorCalculator()
        vf = calc.simple_view_factor(tilt=0.0)

        assert abs(vf['f_sky'] - 0.5) < 0.01
        assert abs(vf['f_gnd_beam'] - 0.5) < 0.01
        assert abs(vf['f_gnd_diff'] - 0.5) < 0.01

    def test_simple_view_factor_vertical(self):
        """Test view factor for vertical module (90° tilt)."""
        calc = ViewFactorCalculator()
        vf = calc.simple_view_factor(tilt=90.0)

        assert abs(vf['f_sky'] - 0.5) < 0.01
        assert abs(vf['f_gnd_beam'] - 0.5) < 0.01

    def test_simple_view_factor_typical(self):
        """Test view factor for typical tilt (30°)."""
        calc = ViewFactorCalculator()
        vf = calc.simple_view_factor(tilt=30.0)

        # At 30°, sky view factor should be > 0.5
        assert vf['f_sky'] > 0.5
        assert vf['f_gnd_beam'] < 0.5
        assert abs(vf['f_sky'] + vf['f_gnd_beam'] - 1.0) < 0.01

    def test_perez_view_factor_single_row(self):
        """Test Perez view factor for single row."""
        calc = ViewFactorCalculator()
        vf = calc.perez_view_factor(
            tilt=30.0,
            clearance=1.0,
            row_spacing=None,
            row_width=None,
            total_rows=1
        )

        assert 0 <= vf['f_gnd_beam'] <= 1
        assert 0 <= vf['f_sky'] <= 1
        assert vf['f_row'] == 0.0

    def test_perez_view_factor_multi_row(self):
        """Test Perez view factor for multi-row system."""
        calc = ViewFactorCalculator()
        vf = calc.perez_view_factor(
            tilt=30.0,
            clearance=1.0,
            row_spacing=4.0,
            row_width=1.1,
            row_number=5,
            total_rows=10
        )

        assert 0 <= vf['f_gnd_beam'] <= 1
        assert 0 <= vf['f_sky'] <= 1
        assert 0 <= vf['f_row'] <= 1

    def test_perez_view_factor_edge_row(self):
        """Test Perez view factor for edge rows."""
        calc = ViewFactorCalculator()

        # First row
        vf_first = calc.perez_view_factor(
            tilt=30.0,
            clearance=1.0,
            row_spacing=4.0,
            row_width=1.1,
            row_number=1,
            total_rows=10
        )

        # Interior row
        vf_interior = calc.perez_view_factor(
            tilt=30.0,
            clearance=1.0,
            row_spacing=4.0,
            row_width=1.1,
            row_number=5,
            total_rows=10
        )

        # Edge rows should have higher ground view factor
        assert vf_first['f_gnd_beam'] > vf_interior['f_gnd_beam']

    def test_durusoy_view_factor(self):
        """Test Durusoy view factor calculation."""
        calc = ViewFactorCalculator()
        vf = calc.durusoy_view_factor(
            tilt=30.0,
            clearance=1.0,
            row_spacing=4.0,
            row_width=1.1
        )

        assert 0 <= vf['f_gnd_beam'] <= 1
        assert 0 <= vf['f_sky'] <= 1
        assert abs(vf['f_sky'] + vf['f_gnd_beam'] + vf['f_row'] - 1.0) < 0.1


class TestBifacialModuleParams:
    """Test bifacial module parameters validation."""

    def test_valid_params(self):
        """Test creation with valid parameters."""
        params = BifacialModuleParams(
            bifaciality=0.70,
            front_efficiency=0.21
        )
        assert params.bifaciality == 0.70
        assert params.rear_efficiency == pytest.approx(0.147, rel=0.01)

    def test_explicit_rear_efficiency(self):
        """Test explicit rear efficiency."""
        params = BifacialModuleParams(
            bifaciality=0.70,
            front_efficiency=0.21,
            rear_efficiency=0.15
        )
        assert params.rear_efficiency == 0.15

    def test_invalid_bifaciality(self):
        """Test invalid bifaciality value."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BifacialModuleParams(
                bifaciality=1.5,  # Invalid: > 1.0
                front_efficiency=0.21
            )


class TestMountingStructure:
    """Test mounting structure validation."""

    def test_fixed_tilt_structure(self):
        """Test fixed tilt mounting structure."""
        structure = MountingStructure(
            mounting_type=MountingType.FIXED_TILT,
            tilt=30.0,
            azimuth=180.0,
            clearance_height=1.0,
            row_spacing=4.0,
            row_width=1.1,
            n_rows=10
        )
        assert structure.mounting_type == MountingType.FIXED_TILT
        assert structure.tilt == 30.0

    def test_tracker_structure(self):
        """Test tracker mounting structure."""
        structure = MountingStructure(
            mounting_type=MountingType.SINGLE_AXIS_TRACKER,
            tilt=0.0,
            clearance_height=2.0,
            row_spacing=6.0,
            row_width=2.0,
            n_rows=5,
            tracker_max_angle=60.0
        )
        assert structure.mounting_type == MountingType.SINGLE_AXIS_TRACKER
        assert structure.tracker_max_angle == 60.0


class TestGroundSurface:
    """Test ground surface parameters."""

    def test_explicit_albedo(self):
        """Test explicit albedo value."""
        ground = GroundSurface(albedo=0.25)
        assert ground.albedo == 0.25

    def test_albedo_from_type(self):
        """Test albedo set from standard type."""
        ground = GroundSurface(
            albedo=0.0,  # Will be overridden
            albedo_type=AlbedoType.GRASS
        )
        assert ground.albedo == ALBEDO_VALUES[AlbedoType.GRASS]

    def test_high_albedo_surfaces(self):
        """Test high albedo surface types."""
        ground_white = GroundSurface(albedo=0.0, albedo_type=AlbedoType.WHITE_MEMBRANE)
        ground_snow = GroundSurface(albedo=0.0, albedo_type=AlbedoType.SNOW)

        assert ground_white.albedo >= 0.70
        assert ground_snow.albedo >= 0.80


class TestBifacialModuleModel:
    """Test bifacial module model calculations."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return BifacialModuleModel()

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        module = BifacialModuleParams(
            bifaciality=0.70,
            front_efficiency=0.21
        )
        structure = MountingStructure(
            mounting_type=MountingType.FIXED_TILT,
            tilt=30.0,
            clearance_height=1.0,
            row_spacing=4.0,
            row_width=1.1,
            n_rows=10
        )
        ground = GroundSurface(albedo=0.25)

        return BifacialSystemConfig(
            module=module,
            structure=structure,
            ground=ground,
            location_latitude=35.0,
            location_longitude=-106.0
        )

    def test_backside_irradiance_positive(self, model):
        """Test that backside irradiance is positive."""
        back_irr = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=30.0,
            clearance=1.0,
            front_poa_global=1000.0,
            front_poa_beam=700.0,
            front_poa_diffuse=300.0,
            dhi=100.0
        )
        assert back_irr > 0
        assert back_irr < 1000.0  # Should be less than front

    def test_backside_irradiance_albedo_effect(self, model):
        """Test effect of ground albedo on backside irradiance."""
        back_irr_low = model.calculate_backside_irradiance(
            ground_albedo=0.15,  # Dark surface
            tilt=30.0,
            clearance=1.0,
            front_poa_global=1000.0
        )

        back_irr_high = model.calculate_backside_irradiance(
            ground_albedo=0.70,  # White surface
            tilt=30.0,
            clearance=1.0,
            front_poa_global=1000.0
        )

        assert back_irr_high > back_irr_low

    def test_backside_irradiance_clearance_effect(self, model):
        """Test effect of clearance height."""
        back_irr_low = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=30.0,
            clearance=0.5,  # Low clearance
            front_poa_global=1000.0
        )

        back_irr_high = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=30.0,
            clearance=2.0,  # High clearance
            front_poa_global=1000.0
        )

        # Higher clearance typically gives more backside irradiance
        assert back_irr_high >= back_irr_low * 0.9  # Allow some variation

    def test_bifacial_gain_calculation(self, model):
        """Test bifacial gain calculation."""
        gain = model.calculate_bifacial_gain(
            front_irr=1000.0,
            back_irr=200.0,
            bifaciality=0.70
        )

        expected_gain = (200.0 * 0.70) / 1000.0
        assert abs(gain - expected_gain) < 0.01

    def test_bifacial_gain_zero_front(self, model):
        """Test bifacial gain with zero front irradiance."""
        gain = model.calculate_bifacial_gain(
            front_irr=0.0,
            back_irr=100.0,
            bifaciality=0.70
        )
        assert gain == 0.0

    def test_effective_irradiance(self, model):
        """Test effective irradiance calculation."""
        eff_irr = model.calculate_effective_irradiance(
            front=1000.0,
            back=200.0,
            bifaciality=0.70,
            glass_transmission_front=0.91,
            glass_transmission_rear=0.88
        )

        # Should be greater than monofacial
        mono_eff = 1000.0 * 0.91
        assert eff_irr > mono_eff

    def test_temperature_effect(self, model):
        """Test temperature coefficient calculation."""
        cell_temp, temp_loss = model.calculate_temperature_effect(
            front_irr=1000.0,
            back_irr=200.0,
            ambient_temp=25.0,
            wind_speed=1.0
        )

        assert cell_temp > 25.0  # Should be warmer than ambient
        assert temp_loss < 1.0  # Should have losses at high temperature

    def test_temperature_wind_effect(self, model):
        """Test wind speed cooling effect."""
        cell_temp_low_wind, _ = model.calculate_temperature_effect(
            front_irr=1000.0,
            back_irr=200.0,
            ambient_temp=25.0,
            wind_speed=0.5
        )

        cell_temp_high_wind, _ = model.calculate_temperature_effect(
            front_irr=1000.0,
            back_irr=200.0,
            ambient_temp=25.0,
            wind_speed=5.0
        )

        # Higher wind should result in cooler cells
        assert cell_temp_high_wind < cell_temp_low_wind

    def test_mismatch_losses(self, model):
        """Test mismatch loss calculation."""
        # Uniform irradiance (low mismatch)
        uniform_irr = np.array([200.0] * 10)
        loss_uniform = model.calculate_mismatch_losses(uniform_irr, 1000.0)

        # Non-uniform irradiance (high mismatch)
        nonuniform_irr = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 200.0, 180.0, 160.0, 140.0, 120.0])
        loss_nonuniform = model.calculate_mismatch_losses(nonuniform_irr, 1000.0)

        assert loss_nonuniform > loss_uniform
        assert 0 <= loss_uniform <= 0.15
        assert 0 <= loss_nonuniform <= 0.15

    def test_soiling_impact(self, model):
        """Test soiling impact calculation."""
        front_clean, back_clean = 1000.0, 200.0

        eff_front, eff_back = model.calculate_soiling_impact(
            front_soiling=0.95,
            rear_soiling=0.90,
            front_irr=front_clean,
            back_irr=back_clean,
            bifaciality=0.70
        )

        assert eff_front < front_clean
        assert eff_back < back_clean
        assert eff_front == front_clean * 0.95
        assert eff_back == back_clean * 0.90

    def test_model_view_factors(self, model, test_config):
        """Test view factor modeling."""
        vf_results = model.model_view_factors(
            test_config.structure,
            ViewFactorModel.PEREZ
        )

        assert 'rows' in vf_results
        assert len(vf_results['rows']) == test_config.structure.n_rows
        assert 0 <= vf_results['average_f_gnd_beam'] <= 1
        assert 0 <= vf_results['average_f_sky'] <= 1

    def test_optimize_row_spacing(self, model):
        """Test row spacing optimization."""
        results = model.optimize_row_spacing(
            module_width=1.1,
            tilt=30.0,
            ground_albedo=0.25,
            clearance=1.0,
            latitude=35.0,
            max_gcr=0.5,
            min_gcr=0.2,
            n_points=10
        )

        assert 'optimal_gcr' in results
        assert 'optimal_spacing' in results
        assert 'optimization_curve' in results
        assert 0.2 <= results['optimal_gcr'] <= 0.5

    def test_simulate_bifacial_performance(self, model, test_config):
        """Test performance simulation."""
        # Create simple TMY data
        n_hours = 24
        tmy = TMY(
            ghi=[800.0] * n_hours,
            dni=[600.0] * n_hours,
            dhi=[200.0] * n_hours,
            temp_air=[25.0] * n_hours,
            wind_speed=[2.0] * n_hours
        )

        system = {
            'module': test_config.module,
            'structure': test_config.structure,
            'ground': test_config.ground,
            'latitude': test_config.location_latitude,
            'longitude': test_config.location_longitude
        }

        results = model.simulate_bifacial_performance(system, tmy)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == n_hours
        assert 'front_poa_global' in results.columns
        assert 'back_irradiance' in results.columns
        assert 'bifacial_gain' in results.columns
        assert 'power_output' in results.columns

        # Check that bifacial gain is positive
        assert (results['bifacial_gain'] >= 0).all()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_gcr(self):
        """Test GCR calculation."""
        gcr = calculate_gcr(row_width=1.1, row_spacing=4.0)
        assert abs(gcr - 0.275) < 0.01

    def test_calculate_gcr_invalid(self):
        """Test GCR with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_gcr(row_width=1.1, row_spacing=0.0)

    def test_seasonal_albedo_variation(self):
        """Test seasonal albedo variation."""
        # Winter grass (dormant, higher albedo)
        albedo_winter = get_albedo_seasonal_variation(
            base_albedo=ALBEDO_VALUES[AlbedoType.GRASS],
            month=1,
            snow_cover=False
        )

        # Summer grass (green, lower albedo)
        albedo_summer = get_albedo_seasonal_variation(
            base_albedo=ALBEDO_VALUES[AlbedoType.GRASS],
            month=6,
            snow_cover=False
        )

        assert albedo_summer < albedo_winter

    def test_seasonal_albedo_snow(self):
        """Test albedo with snow cover."""
        albedo_snow = get_albedo_seasonal_variation(
            base_albedo=ALBEDO_VALUES[AlbedoType.GRASS],
            month=1,
            snow_cover=True
        )
        assert albedo_snow == ALBEDO_VALUES[AlbedoType.SNOW]

    def test_validate_bifacial_system(self):
        """Test system validation."""
        module = BifacialModuleParams(bifaciality=0.70, front_efficiency=0.21)
        structure = MountingStructure(
            mounting_type=MountingType.FIXED_TILT,
            tilt=30.0,
            clearance_height=0.3,  # Very low - should trigger warning
            row_spacing=2.0,
            row_width=1.1,
            n_rows=10
        )
        ground = GroundSurface(albedo=0.10)  # Low albedo - should trigger warning

        config = BifacialSystemConfig(
            module=module,
            structure=structure,
            ground=ground,
            location_latitude=35.0,
            location_longitude=-106.0
        )

        warnings = validate_bifacial_system(config)
        assert len(warnings) > 0  # Should have warnings for low clearance and albedo


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test complete bifacial analysis workflow."""
        # 1. Create system configuration
        module = BifacialModuleParams(
            bifaciality=0.75,
            front_efficiency=0.22,
            module_width=1.1,
            module_length=2.3
        )

        structure = MountingStructure(
            mounting_type=MountingType.FIXED_TILT,
            tilt=35.0,
            clearance_height=1.5,
            row_spacing=5.0,
            row_width=1.1,
            n_rows=20
        )

        ground = GroundSurface(
            albedo=0.0,
            albedo_type=AlbedoType.WHITE_MEMBRANE  # High albedo for bifacial
        )

        config = BifacialSystemConfig(
            module=module,
            structure=structure,
            ground=ground,
            location_latitude=40.0,
            location_longitude=-105.0
        )

        # 2. Validate configuration
        warnings = validate_bifacial_system(config)
        # With good configuration, should have minimal warnings

        # 3. Create model
        model = BifacialModuleModel(config)

        # 4. Calculate backside irradiance
        back_irr = model.calculate_backside_irradiance(
            ground_albedo=config.ground.albedo,
            tilt=config.structure.tilt,
            clearance=config.structure.clearance_height,
            front_poa_global=1000.0,
            row_spacing=config.structure.row_spacing,
            row_width=config.structure.row_width,
            total_rows=config.structure.n_rows
        )

        assert back_irr > 0

        # 5. Calculate bifacial gain
        gain = model.calculate_bifacial_gain(1000.0, back_irr, config.module.bifaciality)
        assert gain > 0

        # 6. With high albedo, should have significant gain
        assert gain > 0.10  # At least 10% gain with white membrane

    def test_tracker_vs_fixed_comparison(self):
        """Compare tracker vs fixed tilt systems."""
        model = BifacialModuleModel()

        # Fixed tilt
        back_irr_fixed = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=30.0,
            clearance=1.0,
            front_poa_global=1000.0,
            row_spacing=4.0,
            row_width=1.1
        )

        # Tracker (higher clearance, different spacing)
        back_irr_tracker = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=20.0,  # Typical tracker angle at this time
            clearance=2.0,  # Higher clearance
            front_poa_global=1000.0,
            row_spacing=6.0,
            row_width=2.0
        )

        # Both should produce reasonable values
        assert 0 < back_irr_fixed < 500
        assert 0 < back_irr_tracker < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
