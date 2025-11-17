"""
Unit tests for B03 CTM Loss Model.

Tests the B03 CTM loss model with k1-k24 factors including:
- Individual factor calculation
- Category-level loss aggregation
- Total CTM ratio calculation
- Scenario comparison
- Sensitivity analysis
"""

import pytest
from pv_circularity_simulator.core.ctm.b03_ctm_loss_model import (
    B03CTMLossModel,
    B03CTMConfiguration,
    B03CTMLossResult,
)


class TestB03CTMConfiguration:
    """Test suite for B03CTMConfiguration model."""

    def test_default_configuration(self):
        """Test creation of default configuration."""
        config = B03CTMConfiguration()

        assert config.k1_cell_binning == "medium"
        assert config.k10_interconnect_shading == "standard_3bb"
        assert config.k20_quality_control_process == "standard"

    def test_from_scenario_premium(self):
        """Test configuration from premium quality scenario."""
        config = B03CTMConfiguration.from_scenario("premium_quality")

        assert config.k1_cell_binning == "tight"
        assert config.k10_interconnect_shading == "mbb_5bb"
        assert config.k20_quality_control_process == "stringent"

    def test_from_scenario_economy(self):
        """Test configuration from economy quality scenario."""
        config = B03CTMConfiguration.from_scenario("economy_quality")

        assert config.k1_cell_binning == "loose"
        assert config.k10_interconnect_shading == "conventional_2bb"
        assert config.k20_quality_control_process == "minimal"

    def test_from_scenario_invalid(self):
        """Test error handling for invalid scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            B03CTMConfiguration.from_scenario("invalid_scenario")


class TestB03CTMLossModel:
    """Test suite for B03CTMLossModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = B03CTMLossModel()

        assert model.loss_factors is not None
        assert len(model.loss_factors) == 24

    def test_calculate_ctm_losses_standard(self):
        """Test CTM loss calculation with standard quality."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("standard_quality")

        result = model.calculate_ctm_losses(config)

        assert isinstance(result, B03CTMLossResult)
        assert result.total_ctm_factor > 0
        assert result.total_ctm_factor < 1.0
        assert 90 <= result.total_ctm_ratio_percent <= 100
        assert result.total_loss_percent > 0

    def test_calculate_ctm_losses_premium(self):
        """Test CTM loss calculation with premium quality."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("premium_quality")

        result = model.calculate_ctm_losses(config)

        # Premium should have higher CTM ratio than standard
        standard_config = B03CTMConfiguration.from_scenario("standard_quality")
        standard_result = model.calculate_ctm_losses(standard_config)

        assert result.total_ctm_ratio_percent > standard_result.total_ctm_ratio_percent

    def test_calculate_ctm_losses_economy(self):
        """Test CTM loss calculation with economy quality."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("economy_quality")

        result = model.calculate_ctm_losses(config)

        # Economy should have lower CTM ratio
        assert result.total_ctm_ratio_percent < 97.0

    def test_loss_breakdown(self):
        """Test loss breakdown by category."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("standard_quality")

        result = model.calculate_ctm_losses(config)
        breakdown = result.get_loss_breakdown()

        assert "cell_level_loss" in breakdown
        assert "interconnection_loss" in breakdown
        assert "encapsulation_loss" in breakdown
        assert "assembly_loss" in breakdown
        assert "measurement_loss" in breakdown

        # All losses should be non-negative
        for loss_value in breakdown.values():
            assert loss_value >= 0

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        model = B03CTMLossModel()

        comparison = model.compare_scenarios()

        assert "premium_quality" in comparison
        assert "standard_quality" in comparison
        assert "economy_quality" in comparison

        # Verify ordering: premium > standard > economy
        premium_ratio = comparison["premium_quality"].total_ctm_ratio_percent
        standard_ratio = comparison["standard_quality"].total_ctm_ratio_percent
        economy_ratio = comparison["economy_quality"].total_ctm_ratio_percent

        assert premium_ratio > standard_ratio > economy_ratio

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis for a single factor."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("standard_quality")

        sensitivity = model.sensitivity_analysis(
            base_configuration=config,
            factor_to_vary="k10_interconnect_shading",
        )

        assert "mbb_5bb" in sensitivity
        assert "standard_3bb" in sensitivity
        assert "conventional_2bb" in sensitivity

        # More busbars should give higher CTM ratio
        assert sensitivity["mbb_5bb"] > sensitivity["conventional_2bb"]

    def test_sensitivity_analysis_invalid_factor(self):
        """Test sensitivity analysis with invalid factor."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("standard_quality")

        with pytest.raises(ValueError, match="Unknown factor"):
            model.sensitivity_analysis(
                base_configuration=config,
                factor_to_vary="invalid_factor",
            )

    def test_individual_factors_extraction(self):
        """Test extraction of individual k factors."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("premium_quality")

        result = model.calculate_ctm_losses(config)

        # Check that all 24 factors are present
        assert len(result.individual_factors) == 24

        # Check that all factors are multiplicative (around 1.0)
        for factor_name, factor_value in result.individual_factors.items():
            assert 0.9 <= factor_value <= 1.05  # Factors can be gains or losses

    def test_category_factors(self):
        """Test category-level factor calculations."""
        model = B03CTMLossModel()
        config = B03CTMConfiguration.from_scenario("standard_quality")

        result = model.calculate_ctm_losses(config)

        # Verify that total factor equals product of category factors
        calculated_total = (
            result.cell_level_factor *
            result.interconnection_factor *
            result.encapsulation_factor *
            result.assembly_factor *
            result.measurement_factor
        )

        assert abs(calculated_total - result.total_ctm_factor) < 1e-6
