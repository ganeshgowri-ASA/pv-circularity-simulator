"""Tests for cell temperature modeling."""

import pytest
import numpy as np
from pv_simulator.core.cell_temperature import (
    CellTemperatureModel,
    ModuleTemperatureCalculator,
)


class TestCellTemperatureModel:
    """Test CellTemperatureModel class."""

    def test_sandia_model(self, standard_conditions, mounting_config, thermal_parameters):
        """Test Sandia temperature model."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.sandia_model()

        # Check result structure
        assert result.cell_temperature > standard_conditions.ambient_temp
        assert result.module_temperature > standard_conditions.ambient_temp
        assert result.model_name == "Sandia"
        assert result.conditions == standard_conditions

        # Check reasonable temperature range
        assert 30.0 < result.cell_temperature < 80.0

    def test_pvsyst_model(self, standard_conditions, mounting_config, thermal_parameters):
        """Test PVsyst temperature model."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.pvsyst_model()

        assert result.cell_temperature > standard_conditions.ambient_temp
        assert result.model_name == "PVsyst"
        assert 30.0 < result.cell_temperature < 80.0

    def test_faiman_model(self, standard_conditions, mounting_config, thermal_parameters):
        """Test Faiman temperature model."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.faiman_model()

        assert result.cell_temperature > standard_conditions.ambient_temp
        assert result.model_name == "Faiman"
        assert 30.0 < result.cell_temperature < 80.0

    def test_noct_based_model(self, standard_conditions, mounting_config, thermal_parameters):
        """Test NOCT-based temperature model."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.noct_based(noct=45.0)

        assert result.cell_temperature > standard_conditions.ambient_temp
        assert result.model_name == "NOCT-based"
        assert 30.0 < result.cell_temperature < 80.0

    def test_noct_with_module_data(
        self, standard_conditions, mounting_config, thermal_parameters, sample_module_noct_data
    ):
        """Test NOCT-based model with ModuleNOCTData."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.noct_based(noct=sample_module_noct_data)

        assert result.cell_temperature > standard_conditions.ambient_temp
        assert 30.0 < result.cell_temperature < 80.0

    def test_custom_thermal_model(self, standard_conditions, mounting_config, thermal_parameters):
        """Test custom thermal model function."""

        def custom_model(conditions, thermal_params, mounting, k=1.0):
            return conditions.ambient_temp + k * conditions.irradiance / 30.0

        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )
        result = model.custom_thermal_models(
            model_func=custom_model,
            model_params={"k": 1.0},
        )

        assert result.cell_temperature > standard_conditions.ambient_temp
        assert result.model_name == "Custom"

    def test_custom_model_error_handling(
        self, standard_conditions, mounting_config, thermal_parameters
    ):
        """Test error handling for custom models."""
        model = CellTemperatureModel(
            conditions=standard_conditions,
            thermal_params=thermal_parameters,
            mounting=mounting_config,
        )

        # No function provided
        with pytest.raises(ValueError):
            model.custom_thermal_models(model_func=None)

        # Non-callable function
        with pytest.raises(ValueError):
            model.custom_thermal_models(model_func="not_a_function")

    def test_wind_speed_effect(self, mounting_config, thermal_parameters):
        """Test that higher wind speed reduces temperature."""
        from pv_simulator.models.thermal import TemperatureConditions

        # Low wind
        cond_low_wind = TemperatureConditions(
            ambient_temp=25.0, irradiance=1000.0, wind_speed=1.0
        )
        model_low = CellTemperatureModel(cond_low_wind, thermal_parameters, mounting_config)
        result_low = model_low.sandia_model()

        # High wind
        cond_high_wind = TemperatureConditions(
            ambient_temp=25.0, irradiance=1000.0, wind_speed=10.0
        )
        model_high = CellTemperatureModel(cond_high_wind, thermal_parameters, mounting_config)
        result_high = model_high.sandia_model()

        # Higher wind should result in lower temperature
        assert result_high.cell_temperature < result_low.cell_temperature


class TestModuleTemperatureCalculator:
    """Test ModuleTemperatureCalculator class."""

    def test_heat_transfer_coefficients(
        self, thermal_parameters, mounting_config, standard_conditions
    ):
        """Test heat transfer coefficient calculation."""
        calculator = ModuleTemperatureCalculator(
            thermal_params=thermal_parameters,
            mounting=mounting_config,
            conditions=standard_conditions,
        )
        coeffs = calculator.heat_transfer_coefficients()

        # Check all coefficients are positive
        assert coeffs.convective_front > 0
        assert coeffs.convective_back > 0
        assert coeffs.radiative_front > 0
        assert coeffs.radiative_back > 0

        # Front should have higher convection than back
        assert coeffs.convective_front > coeffs.convective_back

    def test_wind_speed_effects(self, thermal_parameters, mounting_config, standard_conditions):
        """Test wind speed effects analysis."""
        calculator = ModuleTemperatureCalculator(
            thermal_params=thermal_parameters,
            mounting=mounting_config,
            conditions=standard_conditions,
        )
        wind_results = calculator.wind_speed_effects()

        # Check dataframe structure
        assert "wind_speed_ms" in wind_results.columns
        assert "h_conv_front" in wind_results.columns
        assert "temp_reduction_c" in wind_results.columns

        # Higher wind speed should increase heat transfer
        assert wind_results["h_conv_front"].iloc[-1] > wind_results["h_conv_front"].iloc[0]

    def test_mounting_configuration_effects(
        self, thermal_parameters, mounting_config, standard_conditions
    ):
        """Test mounting configuration comparison."""
        calculator = ModuleTemperatureCalculator(
            thermal_params=thermal_parameters,
            mounting=mounting_config,
            conditions=standard_conditions,
        )
        mount_results = calculator.mounting_configuration_effects()

        # Check dataframe structure
        assert "mounting_type" in mount_results.columns
        assert "avg_cell_temp_c" in mount_results.columns

        # Should have 4 mounting types
        assert len(mount_results) == 4

        # Building integrated should be hottest
        bi_temp = mount_results[mount_results["mounting_type"] == "building_integrated"][
            "avg_cell_temp_c"
        ].values[0]
        or_temp = mount_results[mount_results["mounting_type"] == "open_rack"][
            "avg_cell_temp_c"
        ].values[0]
        assert bi_temp > or_temp

    def test_back_surface_temperature(
        self, thermal_parameters, mounting_config, standard_conditions
    ):
        """Test back surface temperature calculation."""
        calculator = ModuleTemperatureCalculator(
            thermal_params=thermal_parameters,
            mounting=mounting_config,
            conditions=standard_conditions,
        )
        back_temp = calculator.back_surface_temperature(
            front_surface_temp=60.0,
            irradiance=1000.0,
            ambient_temp=25.0,
        )

        # Back temp should be between ambient and front
        assert 25.0 < back_temp < 60.0

    def test_thermal_time_constants(
        self, thermal_parameters, mounting_config, standard_conditions
    ):
        """Test thermal time constant calculations."""
        calculator = ModuleTemperatureCalculator(
            thermal_params=thermal_parameters,
            mounting=mounting_config,
            conditions=standard_conditions,
        )
        tau_results = calculator.thermal_time_constants(wind_speed=3.0)

        # Check all required keys
        assert "tau_heating" in tau_results
        assert "tau_cooling" in tau_results
        assert "response_time_63" in tau_results
        assert "response_time_95" in tau_results

        # Time constants should be positive
        assert tau_results["tau_heating"] > 0
        assert tau_results["tau_cooling"] > 0

        # Cooling should be faster than heating
        assert tau_results["tau_cooling"] < tau_results["tau_heating"]

        # 95% response should be ~3x the 63% response
        assert abs(tau_results["response_time_95"] - 3 * tau_results["response_time_63"]) < 1.0
