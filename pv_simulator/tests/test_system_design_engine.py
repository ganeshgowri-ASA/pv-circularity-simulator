"""Unit tests for SystemDesignEngine."""

import pytest
import numpy as np

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    InverterType,
    SystemType,
    MountingType,
)
from pv_simulator.system_design.system_design_engine import SystemDesignEngine


@pytest.fixture
def sample_module():
    """Create sample module parameters."""
    return ModuleParameters(
        manufacturer="Test Solar",
        model="TS400",
        technology="mtSiMono",
        pmax=400.0,
        voc=48.5,
        isc=10.5,
        vmp=40.8,
        imp=9.8,
        temp_coeff_pmax=-0.37,
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        length=1.776,
        width=1.052,
        thickness=0.035,
        weight=21.5,
        cells_in_series=72,
        efficiency=21.4,
    )


@pytest.fixture
def sample_inverter():
    """Create sample inverter parameters."""
    return InverterParameters(
        manufacturer="Test Inverter",
        model="TI100K",
        inverter_type=InverterType.STRING,
        pac_max=100000,
        vac_nom=480,
        iac_max=150,
        pdc_max=150000,
        vdc_max=1000,
        vdc_nom=600,
        vdc_min=200,
        idc_max=200,
        num_mppt=6,
        mppt_vmin=200,
        mppt_vmax=850,
        strings_per_mppt=3,
        max_efficiency=98.5,
        weight=65,
    )


@pytest.fixture
def design_engine():
    """Create system design engine."""
    return SystemDesignEngine(
        project_name="Test Project",
        system_type=SystemType.COMMERCIAL,
        location="Test Location",
        latitude=35.0,
        longitude=-100.0,
        elevation=100.0,
    )


class TestSystemDesignEngine:
    """Test suite for SystemDesignEngine."""

    def test_initialization(self, design_engine):
        """Test engine initialization."""
        assert design_engine.project_name == "Test Project"
        assert design_engine.system_type == SystemType.COMMERCIAL
        assert design_engine.latitude == 35.0

    def test_calculate_string_sizing(self, design_engine, sample_module, sample_inverter):
        """Test string sizing calculation."""
        string_config = design_engine.calculate_string_sizing(
            module=sample_module,
            inverter=sample_inverter,
            site_temp_min=-10.0,
            site_temp_max=70.0,
        )

        assert string_config.modules_per_string > 0
        assert string_config.strings_per_mppt > 0

    def test_design_system_configuration(
        self, design_engine, sample_module, sample_inverter
    ):
        """Test complete system configuration design."""
        system_config = design_engine.design_system_configuration(
            module=sample_module,
            inverter=sample_inverter,
            target_dc_capacity_kw=500.0,
            mounting_type=MountingType.GROUND_FIXED,
            site_temp_min=-10.0,
            site_temp_max=70.0,
        )

        assert system_config.project_name == "Test Project"
        assert system_config.num_modules > 0
        assert system_config.num_inverters > 0
        assert system_config.dc_capacity is not None
        assert system_config.ac_capacity is not None
        assert system_config.dc_ac_ratio is not None

    def test_design_dc_wiring(self, design_engine, sample_module, sample_inverter):
        """Test DC wiring design."""
        # First create a system configuration
        system_config = design_engine.design_system_configuration(
            module=sample_module,
            inverter=sample_inverter,
            target_dc_capacity_kw=100.0,
            mounting_type=MountingType.GROUND_FIXED,
        )

        # Test DC wiring design
        dc_wiring = design_engine.design_dc_wiring(
            array_layout=system_config.array_layout,
            string_config=system_config.string_config,
            cable_cross_section_mm2=6.0,
        )

        assert "num_strings" in dc_wiring
        assert "num_combiners" in dc_wiring
        assert "total_dc_cable_estimate_m" in dc_wiring

    def test_design_ac_collection(self, design_engine):
        """Test AC collection system design."""
        ac_design = design_engine.design_ac_collection(
            num_inverters=5,
            inverter_ac_power_kw=100.0,
            distance_to_poc_m=500.0,
        )

        assert ac_design["total_ac_power_kw"] == 500.0
        assert ac_design["transformer_size_kva"] > 0
        assert ac_design["collection_voltage_v"] > 0
        assert ac_design["num_phases"] in [1, 3]

    def test_calculate_system_losses(
        self, design_engine, sample_module, sample_inverter
    ):
        """Test system losses calculation."""
        losses = design_engine.calculate_system_losses(
            module=sample_module,
            inverter=sample_inverter,
            mounting_type=MountingType.GROUND_FIXED,
            location_type="temperate",
        )

        assert losses.soiling > 0
        assert losses.dc_wiring > 0
        assert losses.inverter > 0
        assert losses.total_losses() > 0

    def test_design_array_layout_ground(
        self, design_engine, sample_module, sample_inverter
    ):
        """Test ground-mounted array layout design."""
        # Initialize design engine components
        design_engine.string_calculator = None
        design_engine.layout_designer = None

        system_config = design_engine.design_system_configuration(
            module=sample_module,
            inverter=sample_inverter,
            target_dc_capacity_kw=100.0,
            mounting_type=MountingType.GROUND_FIXED,
        )

        layout = system_config.array_layout
        assert layout.mounting_type == MountingType.GROUND_FIXED
        assert layout.rows > 0
        assert layout.modules_per_row > 0

    def test_optimize_system_layout(
        self, design_engine, sample_module, sample_inverter
    ):
        """Test system layout optimization."""
        optimized_config = design_engine.optimize_system_layout(
            module=sample_module,
            inverter=sample_inverter,
            available_area_m2=10000.0,
            mounting_type=MountingType.GROUND_FIXED,
        )

        assert optimized_config.num_modules > 0
        assert optimized_config.dc_capacity > 0
        # Verify modules fit in available area
        total_area_needed = (
            optimized_config.num_modules * sample_module.area
        )
        assert total_area_needed <= 10000.0 * 1.5  # Allow some margin
