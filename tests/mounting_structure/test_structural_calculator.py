"""Unit tests for StructuralCalculator class."""

import pytest
import math
from pv_simulator.mounting_structure.structural_calculator import StructuralCalculator
from pv_simulator.mounting_structure.models import (
    SiteParameters,
    ModuleDimensions,
    ExposureCategory,
    SeismicDesignCategory,
    SoilType,
)


@pytest.fixture
def site_params():
    """Sample site parameters for testing."""
    return SiteParameters(
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


@pytest.fixture
def module_dims():
    """Sample module dimensions for testing."""
    return ModuleDimensions(
        length=2.0,
        width=1.0,
        thickness=0.04,
        weight=25.0,
        frame_width=0.035,
        glass_thickness=0.0032,
    )


@pytest.fixture
def calculator(site_params):
    """StructuralCalculator instance for testing."""
    return StructuralCalculator(site_params)


class TestStructuralCalculator:
    """Test suite for StructuralCalculator."""

    def test_wind_load_analysis_basic(self, calculator, module_dims):
        """Test basic wind load calculation."""
        result = calculator.wind_load_analysis(
            tilt_angle=30.0,
            height=2.0,
            module_dimensions=module_dims,
        )

        assert "velocity_pressure_qz" in result
        assert "uplift_pressure" in result
        assert "downward_pressure" in result
        assert result["velocity_pressure_qz"] > 0
        assert result["uplift_pressure"] < 0 or result["uplift_pressure"] > 0
        assert result["wind_speed"] == 35.0

    def test_wind_load_rooftop(self, calculator, module_dims):
        """Test wind load for rooftop installation."""
        result = calculator.wind_load_analysis(
            tilt_angle=15.0,
            height=1.0,
            module_dimensions=module_dims,
            is_rooftop=True,
            roof_height=10.0,
        )

        assert result["uplift_pressure"] != 0
        # Rooftop should have additional pressure

    def test_snow_load_analysis(self, calculator):
        """Test snow load calculation."""
        result = calculator.snow_load_analysis(
            tilt_angle=30.0,
            module_length=2.0,
        )

        assert "ground_snow_load" in result
        assert "flat_roof_snow_load" in result
        assert "sloped_roof_snow_load" in result
        assert result["ground_snow_load"] == 1.0
        assert 0 <= result["slope_factor_cs"] <= 1.0

    def test_snow_load_steep_slope(self, calculator):
        """Test snow load on steep slope (>70°)."""
        result = calculator.snow_load_analysis(
            tilt_angle=80.0,
            module_length=2.0,
        )

        # Steep slopes should have minimal snow load
        assert result["sloped_roof_snow_load"] < result["flat_roof_snow_load"]

    def test_seismic_analysis_low_category(self, calculator):
        """Test seismic analysis for low category."""
        result = calculator.seismic_analysis(
            total_weight=100.0,
            height=2.0,
        )

        # Category C should require seismic design
        assert result["requires_seismic_design"] is True
        assert result["seismic_force"] > 0

    def test_seismic_analysis_category_a(self):
        """Test seismic analysis for category A (minimal)."""
        site = SiteParameters(
            latitude=40.0,
            longitude=-105.0,
            elevation=1000.0,
            wind_speed=35.0,
            exposure_category=ExposureCategory.C,
            ground_snow_load=1.0,
            seismic_category=SeismicDesignCategory.A,
            soil_type=SoilType.SAND,
            frost_depth=0.8,
        )
        calc = StructuralCalculator(site)

        result = calc.seismic_analysis(
            total_weight=100.0,
            height=2.0,
        )

        assert result["requires_seismic_design"] is False
        assert result["seismic_force"] == 0.0

    def test_deflection_analysis_simple_support(self, calculator):
        """Test deflection analysis for simply supported beam."""
        result = calculator.deflection_analysis(
            span_length=3.0,
            applied_load=2.0,
            moment_of_inertia=5e-6,
            support_type="simple",
        )

        assert result["max_deflection"] > 0
        assert result["deflection_limit"] > 0
        assert "passes_deflection" in result

    def test_deflection_analysis_cantilever(self, calculator):
        """Test deflection analysis for cantilever."""
        result = calculator.deflection_analysis(
            span_length=2.0,
            applied_load=2.0,
            moment_of_inertia=5e-6,
            support_type="cantilever",
        )

        # Cantilever should have larger deflection than simple support
        assert result["max_deflection"] > 0

    def test_connection_design_bolted(self, calculator):
        """Test bolted connection design."""
        result = calculator.connection_design(
            applied_force=10.0,
            connection_type="bolted",
            num_fasteners=4,
        )

        assert result["connection_type"] == "bolted"
        assert result["num_fasteners"] == 4
        assert result["total_capacity"] > 0
        assert "passes" in result

    def test_connection_design_welded(self, calculator):
        """Test welded connection design."""
        result = calculator.connection_design(
            applied_force=20.0,
            connection_type="welded",
            num_fasteners=2,  # Used for weld length
        )

        assert result["connection_type"] == "welded"
        assert result["total_capacity"] > 0

    def test_safety_factors_load_combinations(self, calculator):
        """Test load combination calculations."""
        result = calculator.safety_factors(
            dead_load=1.0,
            live_load=0.5,
            wind_load=2.0,
            snow_load=0.8,
        )

        assert "combinations" in result
        assert "governing_combination" in result
        assert "governing_load" in result
        assert len(result["combinations"]) >= 5

    def test_calculate_total_loads(self, calculator, module_dims):
        """Test complete load analysis."""
        load_analysis = calculator.calculate_total_loads(
            module_dimensions=module_dims,
            num_modules=100,
            tilt_angle=30.0,
            height=2.0,
        )

        assert load_analysis.dead_load > 0
        assert load_analysis.live_load > 0
        assert load_analysis.wind_load_uplift != 0
        assert load_analysis.snow_load >= 0
        assert load_analysis.safety_factor > 1.0

    def test_exposure_coefficient_calculation(self, calculator):
        """Test Kz calculation for different heights."""
        kz_15ft = calculator._calculate_kz(4.57)  # 15 ft
        kz_30ft = calculator._calculate_kz(9.14)  # 30 ft

        # Higher elevation should have higher Kz
        assert kz_30ft > kz_15ft
        assert kz_15ft > 0

    def test_wind_load_different_exposures(self, module_dims):
        """Test wind loads for different exposure categories."""
        results = {}

        for exposure in [ExposureCategory.B, ExposureCategory.C, ExposureCategory.D]:
            site = SiteParameters(
                latitude=40.0,
                longitude=-105.0,
                elevation=1000.0,
                wind_speed=35.0,
                exposure_category=exposure,
                ground_snow_load=1.0,
                seismic_category=SeismicDesignCategory.C,
                soil_type=SoilType.SAND,
                frost_depth=0.8,
            )
            calc = StructuralCalculator(site)
            result = calc.wind_load_analysis(
                tilt_angle=30.0,
                height=2.0,
                module_dimensions=module_dims,
            )
            results[exposure.value] = result["uplift_pressure"]

        # Exposure D (coastal) should have highest wind loads
        assert abs(results["D"]) >= abs(results["C"])
        assert abs(results["C"]) >= abs(results["B"])
