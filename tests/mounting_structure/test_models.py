"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError
from pv_simulator.mounting_structure.models import (
    SiteParameters,
    ModuleDimensions,
    LoadAnalysis,
    FoundationDesign,
    GroundMountConfig,
    ExposureCategory,
    SeismicDesignCategory,
    SoilType,
    FoundationType,
    MaterialType,
    MountingType,
    ModuleOrientation,
    RackingConfiguration,
)


class TestSiteParameters:
    """Test SiteParameters model."""

    def test_valid_site_parameters(self):
        """Test creation with valid parameters."""
        site = SiteParameters(
            latitude=40.0,
            longitude=-105.0,
            elevation=1000.0,
            wind_speed=35.0,
            exposure_category=ExposureCategory.C,
            ground_snow_load=1.0,
            seismic_category=SeismicDesignCategory.C,
            soil_type=SoilType.SAND,
            frost_depth=0.8,
        )

        assert site.latitude == 40.0
        assert site.wind_speed == 35.0
        assert site.exposure_category == ExposureCategory.C

    def test_invalid_latitude(self):
        """Test invalid latitude raises validation error."""
        with pytest.raises(ValidationError):
            SiteParameters(
                latitude=100.0,  # Invalid: > 90
                longitude=-105.0,
                elevation=1000.0,
                wind_speed=35.0,
                exposure_category=ExposureCategory.C,
                ground_snow_load=1.0,
                seismic_category=SeismicDesignCategory.C,
                soil_type=SoilType.SAND,
            )

    def test_negative_wind_speed(self):
        """Test negative wind speed raises error."""
        with pytest.raises(ValidationError):
            SiteParameters(
                latitude=40.0,
                longitude=-105.0,
                elevation=1000.0,
                wind_speed=-35.0,  # Invalid: negative
                exposure_category=ExposureCategory.C,
                ground_snow_load=1.0,
                seismic_category=SeismicDesignCategory.C,
                soil_type=SoilType.SAND,
            )


class TestModuleDimensions:
    """Test ModuleDimensions model."""

    def test_valid_module_dimensions(self):
        """Test creation with valid dimensions."""
        module = ModuleDimensions(
            length=2.0,
            width=1.0,
            thickness=0.04,
            weight=25.0,
            frame_width=0.035,
            glass_thickness=0.0032,
        )

        assert module.length == 2.0
        assert module.weight == 25.0

    def test_zero_dimensions(self):
        """Test zero dimensions raise error."""
        with pytest.raises(ValidationError):
            ModuleDimensions(
                length=0.0,  # Invalid: must be > 0
                width=1.0,
                thickness=0.04,
                weight=25.0,
            )


class TestLoadAnalysis:
    """Test LoadAnalysis model."""

    def test_valid_load_analysis(self):
        """Test creation with valid loads."""
        load = LoadAnalysis(
            dead_load=0.15,
            live_load=0.5,
            wind_load_uplift=-2.5,
            wind_load_downward=1.8,
            snow_load=0.8,
            total_load_combination=3.2,
            safety_factor=2.0,
        )

        assert load.dead_load == 0.15
        assert load.safety_factor == 2.0

    def test_invalid_safety_factor(self):
        """Test safety factor must be > 1.0."""
        with pytest.raises(ValidationError):
            LoadAnalysis(
                dead_load=0.15,
                live_load=0.5,
                wind_load_uplift=-2.5,
                wind_load_downward=1.8,
                snow_load=0.8,
                total_load_combination=3.2,
                safety_factor=0.5,  # Invalid: <= 1.0
            )


class TestFoundationDesign:
    """Test FoundationDesign model."""

    def test_valid_foundation_design(self):
        """Test creation with valid foundation."""
        foundation = FoundationDesign(
            foundation_type=FoundationType.DRIVEN_PILE,
            depth=2.5,
            diameter=0.15,
            length=2.5,
            width=0.15,
            capacity=50.0,
            spacing=3.0,
            quantity=100,
            material=MaterialType.STEEL_GALVANIZED,
            embedment_depth=2.5,
        )

        assert foundation.foundation_type == FoundationType.DRIVEN_PILE
        assert foundation.capacity == 50.0

    def test_negative_capacity(self):
        """Test negative capacity raises error."""
        with pytest.raises(ValidationError):
            FoundationDesign(
                foundation_type=FoundationType.DRIVEN_PILE,
                depth=2.5,
                capacity=-50.0,  # Invalid: negative
                spacing=3.0,
                quantity=100,
                material=MaterialType.STEEL_GALVANIZED,
                embedment_depth=2.5,
            )


class TestGroundMountConfig:
    """Test GroundMountConfig model."""

    def test_valid_ground_mount_config(self):
        """Test creation with valid config."""
        site = SiteParameters(
            latitude=40.0,
            longitude=-105.0,
            elevation=1000.0,
            wind_speed=35.0,
            exposure_category=ExposureCategory.C,
            ground_snow_load=1.0,
            seismic_category=SeismicDesignCategory.C,
            soil_type=SoilType.SAND,
        )

        module = ModuleDimensions(
            length=2.0,
            width=1.0,
            thickness=0.04,
            weight=25.0,
        )

        config = GroundMountConfig(
            mounting_type=MountingType.GROUND_FIXED_TILT,
            site_parameters=site,
            module_dimensions=module,
            num_modules=1000,
            tilt_angle=30.0,
            orientation=ModuleOrientation.PORTRAIT,
            racking_config=RackingConfiguration.TWO_PORTRAIT,
            foundation_type=FoundationType.DRIVEN_PILE,
        )

        assert config.num_modules == 1000
        assert config.tilt_angle == 30.0

    def test_invalid_tilt_angle(self):
        """Test invalid tilt angle raises error."""
        site = SiteParameters(
            latitude=40.0,
            longitude=-105.0,
            elevation=1000.0,
            wind_speed=35.0,
            exposure_category=ExposureCategory.C,
            ground_snow_load=1.0,
            seismic_category=SeismicDesignCategory.C,
            soil_type=SoilType.SAND,
        )

        module = ModuleDimensions(
            length=2.0,
            width=1.0,
            thickness=0.04,
            weight=25.0,
        )

        with pytest.raises(ValidationError):
            GroundMountConfig(
                mounting_type=MountingType.GROUND_FIXED_TILT,
                site_parameters=site,
                module_dimensions=module,
                num_modules=1000,
                tilt_angle=95.0,  # Invalid: > 90
                orientation=ModuleOrientation.PORTRAIT,
                racking_config=RackingConfiguration.TWO_PORTRAIT,
                foundation_type=FoundationType.DRIVEN_PILE,
            )


class TestEnums:
    """Test enum types."""

    def test_mounting_type_enum(self):
        """Test MountingType enum values."""
        assert MountingType.GROUND_FIXED_TILT.value == "ground_fixed_tilt"
        assert MountingType.ROOFTOP_FLAT.value == "rooftop_flat"

    def test_foundation_type_enum(self):
        """Test FoundationType enum values."""
        assert FoundationType.DRIVEN_PILE.value == "driven_pile"
        assert FoundationType.HELICAL_PILE.value == "helical_pile"

    def test_material_type_enum(self):
        """Test MaterialType enum values."""
        assert MaterialType.STEEL_GALVANIZED.value == "steel_galvanized"
        assert MaterialType.ALUMINUM.value == "aluminum"
