"""Unit tests for GroundMountDesign class."""

import pytest
from pv_simulator.mounting_structure.ground_mount import GroundMountDesign
from pv_simulator.mounting_structure.models import (
    GroundMountConfig,
    SiteParameters,
    ModuleDimensions,
    MountingType,
    ModuleOrientation,
    RackingConfiguration,
    FoundationType,
    ExposureCategory,
    SeismicDesignCategory,
    SoilType,
)


@pytest.fixture
def ground_config():
    """Sample ground mount configuration."""
    site = SiteParameters(
        latitude=35.0,
        longitude=-95.0,
        elevation=300.0,
        wind_speed=30.0,
        exposure_category=ExposureCategory.C,
        ground_snow_load=0.5,
        seismic_category=SeismicDesignCategory.B,
        soil_type=SoilType.SAND,
        frost_depth=0.5,
    )

    module = ModuleDimensions(
        length=2.0,
        width=1.0,
        thickness=0.04,
        weight=25.0,
        frame_width=0.035,
        glass_thickness=0.0032,
    )

    return GroundMountConfig(
        mounting_type=MountingType.GROUND_FIXED_TILT,
        site_parameters=site,
        module_dimensions=module,
        num_modules=1000,
        tilt_angle=30.0,
        azimuth=180.0,
        orientation=ModuleOrientation.PORTRAIT,
        racking_config=RackingConfiguration.TWO_PORTRAIT,
        foundation_type=FoundationType.DRIVEN_PILE,
        post_spacing=3.0,
    )


class TestGroundMountDesign:
    """Test suite for GroundMountDesign."""

    def test_fixed_tilt_structure(self, ground_config):
        """Test fixed-tilt structure design."""
        designer = GroundMountDesign(ground_config)
        result = designer.fixed_tilt_structure()

        assert result.mounting_type == MountingType.GROUND_FIXED_TILT
        assert result.load_analysis is not None
        assert result.foundation_design is not None
        assert len(result.structural_members) > 0
        assert len(result.bill_of_materials) > 0
        assert result.total_steel_weight > 0

    def test_calculate_row_spacing(self, ground_config):
        """Test row spacing calculation."""
        designer = GroundMountDesign(ground_config)
        spacing = designer.calculate_row_spacing(min_solar_access=0.85)

        assert spacing["row_spacing"] > 0
        assert 0 < spacing["ground_coverage_ratio"] < 1
        assert spacing["shadow_length_winter"] > 0
        assert spacing["solar_altitude_winter"] > 0

    def test_calculate_post_spacing(self, ground_config):
        """Test post spacing calculation."""
        designer = GroundMountDesign(ground_config)
        result = designer.calculate_post_spacing()

        assert result["post_spacing"] > 0
        assert result["post_spacing"] >= 1.5  # Minimum practical
        assert result["post_spacing"] <= 4.0  # Maximum practical
        assert result["governing_criteria"] in ["deflection", "stress"]

    def test_foundation_design(self, ground_config):
        """Test foundation design."""
        designer = GroundMountDesign(ground_config)
        result = designer.foundation_design()

        assert "foundations" in result
        assert "geotechnical_requirements" in result
        assert "frost_protection" in result

    def test_racking_bom(self, ground_config):
        """Test BOM generation."""
        designer = GroundMountDesign(ground_config)
        bom = designer.racking_bom()

        assert len(bom) > 0
        # Should have foundations, structural members, and hardware
        assert any("foundation" in item.description.lower() for item in bom)

    def test_single_axis_tracker(self, ground_config):
        """Test single-axis tracker design."""
        ground_config.mounting_type = MountingType.GROUND_SINGLE_AXIS
        ground_config.max_tracking_angle = 60.0
        ground_config.backtracking_enabled = True

        designer = GroundMountDesign(ground_config)
        result = designer.single_axis_tracker()

        assert result.mounting_type == MountingType.GROUND_SINGLE_AXIS
        assert "backtracking" in result.connection_details
        assert result.max_deflection > 0

    def test_dual_axis_tracker(self, ground_config):
        """Test dual-axis tracker design."""
        ground_config.mounting_type = MountingType.GROUND_DUAL_AXIS

        designer = GroundMountDesign(ground_config)
        result = designer.dual_axis_tracker()

        assert result.mounting_type == MountingType.GROUND_DUAL_AXIS
        assert "azimuth_motor" in result.connection_details
        assert "tilt_motor" in result.connection_details

    def test_different_racking_configs(self, ground_config):
        """Test different racking configurations."""
        for config in [RackingConfiguration.ONE_PORTRAIT, RackingConfiguration.THREE_PORTRAIT, RackingConfiguration.FOUR_PORTRAIT]:
            ground_config.racking_config = config
            designer = GroundMountDesign(ground_config)
            result = designer.fixed_tilt_structure()

            assert result is not None
            assert len(result.structural_members) > 0

    def test_different_orientations(self, ground_config):
        """Test portrait vs landscape orientation."""
        # Portrait
        ground_config.orientation = ModuleOrientation.PORTRAIT
        designer = GroundMountDesign(ground_config)
        result_portrait = designer.fixed_tilt_structure()

        # Landscape
        ground_config.orientation = ModuleOrientation.LANDSCAPE
        designer = GroundMountDesign(ground_config)
        result_landscape = designer.fixed_tilt_structure()

        assert result_portrait is not None
        assert result_landscape is not None
