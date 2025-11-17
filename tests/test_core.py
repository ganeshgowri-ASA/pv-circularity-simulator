"""
Comprehensive unit tests for core module.

Tests all Pydantic data models, SessionManager, and related functionality.
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Import all models and classes to test
from src.core.data_models import (
    # Enums
    CellTechnology,
    MountingType,
    MaterialType,
    # Material models
    Material,
    MaterialProperties,
    CircularityMetrics,
    # Cell models
    Cell,
    TemperatureCoefficients,
    # Module models
    Module,
    CuttingPattern,
    ModuleLayout,
    # System models
    PVSystem,
    Location,
    # Performance & Financial models
    PerformanceData,
    FinancialModel,
    # Helper functions
    create_default_monocrystalline_cell,
    create_example_silicon_material,
)

from src.core.session_manager import (
    SessionManager,
    SessionState,
    ProjectMetadata,
    ModuleCompletionStatus,
    SimulationModule,
)


# ============================================================================
# Test Material Models
# ============================================================================

class TestMaterialProperties:
    """Test MaterialProperties model."""

    def test_valid_material_properties(self):
        """Test creating valid MaterialProperties."""
        props = MaterialProperties(
            density=2330.0,
            thermal_conductivity=150.0,
            specific_heat=700.0,
            melting_point=1414.0,
            recyclability_rate=0.95,
            embodied_energy=200.0,
            carbon_footprint=15.0,
            toxicity_score=1.0
        )
        assert props.density == 2330.0
        assert props.recyclability_rate == 0.95
        assert props.toxicity_score == 1.0

    def test_recyclability_rate_bounds(self):
        """Test recyclability rate is bounded 0-1."""
        with pytest.raises(ValueError):
            MaterialProperties(
                density=2330.0,
                thermal_conductivity=150.0,
                specific_heat=700.0,
                recyclability_rate=1.5,  # Invalid > 1
                embodied_energy=200.0,
                carbon_footprint=15.0
            )

    def test_melting_point_validation(self):
        """Test melting point validation."""
        with pytest.raises(ValueError):
            MaterialProperties(
                density=2330.0,
                thermal_conductivity=150.0,
                specific_heat=700.0,
                melting_point=-500.0,  # Below absolute zero
                recyclability_rate=0.95,
                embodied_energy=200.0,
                carbon_footprint=15.0
            )


class TestMaterial:
    """Test Material model."""

    def test_create_material(self):
        """Test creating a complete material."""
        material = create_example_silicon_material()
        assert material.name == "High-purity Polysilicon"
        assert material.material_type == MaterialType.SILICON
        assert material.properties.density == 2330.0

    def test_material_cost_calculation(self):
        """Test material cost calculations."""
        material = create_example_silicon_material()
        expected_cost = material.mass_per_module * material.cost_per_kg
        assert material.total_cost == expected_cost

    def test_embodied_energy_calculation(self):
        """Test embodied energy calculation."""
        material = create_example_silicon_material()
        expected_energy = material.mass_per_module * material.properties.embodied_energy
        assert material.total_embodied_energy == expected_energy

    def test_carbon_footprint_calculation(self):
        """Test carbon footprint calculation."""
        material = create_example_silicon_material()
        expected_carbon = material.mass_per_module * material.properties.carbon_footprint
        assert material.total_carbon_footprint == expected_carbon


class TestCircularityMetrics:
    """Test CircularityMetrics model."""

    def test_valid_circularity_metrics(self):
        """Test creating valid circularity metrics."""
        metrics = CircularityMetrics(
            recyclability_score=0.9,
            recycled_content_ratio=0.3,
            reusability_score=0.7,
            repairability_score=0.6,
            material_recovery_potential=0.85
        )
        assert metrics.recyclability_score == 0.9
        assert metrics.circular_economy_index is not None

    def test_circular_economy_index_calculation(self):
        """Test automatic calculation of circular economy index."""
        metrics = CircularityMetrics(
            recyclability_score=0.8,
            recycled_content_ratio=0.2,
            reusability_score=0.6,
            repairability_score=0.7,
            material_recovery_potential=0.9,
            lifetime_extension_potential=0.5
        )
        # Should be auto-calculated
        assert 0 <= metrics.circular_economy_index <= 100

    def test_circularity_scores_bounded(self):
        """Test all circularity scores are bounded 0-1."""
        with pytest.raises(ValueError):
            CircularityMetrics(
                recyclability_score=1.5,  # Invalid
                reusability_score=0.7,
                repairability_score=0.6,
                material_recovery_potential=0.85
            )


# ============================================================================
# Test Cell Models
# ============================================================================

class TestTemperatureCoefficients:
    """Test TemperatureCoefficients model."""

    def test_valid_temperature_coefficients(self):
        """Test creating valid temperature coefficients."""
        coeffs = TemperatureCoefficients(
            power=-0.35,
            voltage=-0.27,
            current=0.05
        )
        assert coeffs.power == -0.35
        assert coeffs.voltage == -0.27
        assert coeffs.current == 0.05

    def test_power_coefficient_should_be_negative(self):
        """Test power coefficient validation (should be negative)."""
        with pytest.raises(ValueError):
            TemperatureCoefficients(
                power=0.35,  # Should be negative
                voltage=-0.27,
                current=0.05
            )

    def test_voltage_coefficient_should_be_negative(self):
        """Test voltage coefficient validation (should be negative)."""
        with pytest.raises(ValueError):
            TemperatureCoefficients(
                power=-0.35,
                voltage=0.27,  # Should be negative
                current=0.05
            )


class TestCell:
    """Test Cell model."""

    def test_create_default_cell(self):
        """Test creating a default cell."""
        cell = create_default_monocrystalline_cell()
        assert cell.technology == CellTechnology.MONOCRYSTALLINE
        assert cell.efficiency == 0.225
        assert cell.power_output == 5.5

    def test_electrical_parameter_validation(self):
        """Test electrical parameters are consistent."""
        cell = create_default_monocrystalline_cell()
        # Vmp * Imp should equal power_output
        calculated_power = cell.voltage_at_max_power * cell.current_at_max_power
        assert abs(calculated_power - cell.power_output) < 0.1

    def test_vmp_less_than_voc(self):
        """Test Vmp < Voc validation."""
        with pytest.raises(ValueError):
            Cell(
                technology=CellTechnology.MONOCRYSTALLINE,
                efficiency=0.22,
                area=0.0244,
                thickness=180.0,
                power_output=5.5,
                voltage_at_max_power=0.70,  # Greater than Voc
                current_at_max_power=7.86,
                open_circuit_voltage=0.66,
                short_circuit_current=8.3,
                fill_factor=0.80,
                temperature_coefficients=TemperatureCoefficients(
                    power=-0.35, voltage=-0.27, current=0.05
                ),
                manufacturing_cost=0.50
            )

    def test_fill_factor_validation(self):
        """Test fill factor calculation and validation."""
        cell = create_default_monocrystalline_cell()
        max_power_theoretical = cell.open_circuit_voltage * cell.short_circuit_current
        calculated_ff = cell.power_output / max_power_theoretical
        assert abs(calculated_ff - cell.fill_factor) < 0.01


# ============================================================================
# Test Module Models
# ============================================================================

class TestCuttingPattern:
    """Test CuttingPattern model."""

    def test_valid_cutting_pattern(self):
        """Test creating valid cutting pattern."""
        pattern = CuttingPattern(
            pattern_type="half-cut",
            segments_per_cell=2,
            cutting_loss=0.01,
            efficiency_gain=0.02,
            cost_increase=0.05
        )
        assert pattern.pattern_type == "half-cut"
        assert pattern.segments_per_cell == 2

    def test_pattern_type_validation(self):
        """Test pattern type validation."""
        with pytest.raises(ValueError):
            CuttingPattern(
                pattern_type="invalid-pattern",
                segments_per_cell=2
            )


class TestModuleLayout:
    """Test ModuleLayout model."""

    def test_valid_module_layout(self):
        """Test creating valid module layout."""
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        assert layout.cells_in_series == 60
        assert layout.rows * layout.columns == 60

    def test_layout_consistency_validation(self):
        """Test layout consistency validation."""
        with pytest.raises(ValueError):
            ModuleLayout(
                cells_in_series=60,
                cells_in_parallel=1,
                bypass_diodes=3,
                rows=10,
                columns=5  # 10 * 5 = 50, not 60
            )

    def test_bypass_diode_validation(self):
        """Test bypass diode count validation."""
        with pytest.raises(ValueError):
            ModuleLayout(
                cells_in_series=120,
                cells_in_parallel=1,
                bypass_diodes=1,  # Too few for 120 cells
                rows=10,
                columns=12
            )


class TestModule:
    """Test Module model."""

    def test_create_module(self):
        """Test creating a complete module."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module 330W",
            manufacturer="Test Manufacturer",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        assert module.model_name == "Test Module 330W"
        assert module.total_cells == 60

    def test_module_area_calculation(self):
        """Test module area calculation."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module",
            manufacturer="Test",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        expected_area = 1.65 * 0.992
        assert abs(module.area - expected_area) < 0.001

    def test_power_density_calculation(self):
        """Test power density calculation."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module",
            manufacturer="Test",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        expected_power_density = 330.0 / module.area
        assert abs(module.power_density - expected_power_density) < 0.1


# ============================================================================
# Test System Models
# ============================================================================

class TestLocation:
    """Test Location model."""

    def test_valid_location(self):
        """Test creating valid location."""
        location = Location(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=10.0,
            timezone="America/New_York",
            city="New York",
            country="USA"
        )
        assert location.latitude == 40.7128
        assert location.city == "New York"

    def test_latitude_bounds(self):
        """Test latitude validation."""
        with pytest.raises(ValueError):
            Location(
                latitude=95.0,  # Invalid > 90
                longitude=-74.0,
                timezone="America/New_York"
            )

    def test_longitude_bounds(self):
        """Test longitude validation."""
        with pytest.raises(ValueError):
            Location(
                latitude=40.0,
                longitude=185.0,  # Invalid > 180
                timezone="America/New_York"
            )


class TestPVSystem:
    """Test PVSystem model."""

    def test_create_pv_system(self):
        """Test creating a complete PV system."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module 330W",
            manufacturer="Test Manufacturer",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        location = Location(
            latitude=40.7128,
            longitude=-74.0060,
            timezone="America/New_York"
        )
        system = PVSystem(
            system_name="Test System",
            location=location,
            modules=[module],
            module_quantity=100,
            mounting_type=MountingType.FIXED_TILT,
            tilt_angle=30.0,
            azimuth_angle=180.0,
            dc_capacity=33.0,
            ac_capacity=30.0
        )
        assert system.system_name == "Test System"
        assert system.module_quantity == 100

    def test_dc_ac_ratio_calculation(self):
        """Test DC/AC ratio calculation."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module",
            manufacturer="Test",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        location = Location(
            latitude=40.0,
            longitude=-74.0,
            timezone="America/New_York"
        )
        system = PVSystem(
            system_name="Test",
            location=location,
            modules=[module],
            module_quantity=100,
            mounting_type=MountingType.FIXED_TILT,
            tilt_angle=30.0,
            azimuth_angle=180.0,
            dc_capacity=33.0,
            ac_capacity=30.0
        )
        expected_ratio = 33.0 / 30.0
        assert abs(system.dc_ac_ratio - expected_ratio) < 0.01

    def test_ac_capacity_cannot_exceed_dc(self):
        """Test AC capacity validation."""
        cell = create_default_monocrystalline_cell()
        layout = ModuleLayout(
            cells_in_series=60,
            cells_in_parallel=1,
            bypass_diodes=3,
            rows=10,
            columns=6
        )
        module = Module(
            model_name="Test Module",
            manufacturer="Test",
            cell=cell,
            layout=layout,
            rated_power=330.0,
            length=1.65,
            width=0.992,
            thickness=0.035,
            weight=18.5,
            efficiency=0.201
        )
        location = Location(
            latitude=40.0,
            longitude=-74.0,
            timezone="America/New_York"
        )
        with pytest.raises(ValueError):
            PVSystem(
                system_name="Test",
                location=location,
                modules=[module],
                module_quantity=100,
                mounting_type=MountingType.FIXED_TILT,
                tilt_angle=30.0,
                azimuth_angle=180.0,
                dc_capacity=30.0,
                ac_capacity=35.0  # Invalid: exceeds DC
            )


# ============================================================================
# Test Performance & Financial Models
# ============================================================================

class TestPerformanceData:
    """Test PerformanceData model."""

    def test_valid_performance_data(self):
        """Test creating valid performance data."""
        data = PerformanceData(
            timestamp=datetime.now(),
            dc_power=3000.0,
            ac_power=2880.0,
            dc_voltage=600.0,
            dc_current=5.0,
            ac_voltage=240.0,
            ac_current=12.0,
            irradiance=800.0,
            module_temperature=45.0,
            ambient_temperature=25.0,
            wind_speed=3.5,
            energy_today=15.5,
            energy_total=5000.0
        )
        assert data.dc_power == 3000.0
        assert data.ac_power == 2880.0

    def test_dc_power_consistency(self):
        """Test DC power calculation consistency."""
        data = PerformanceData(
            timestamp=datetime.now(),
            dc_power=3000.0,
            ac_power=2880.0,
            dc_voltage=600.0,
            dc_current=5.0,
            ac_voltage=240.0,
            ac_current=12.0,
            irradiance=800.0,
            module_temperature=45.0,
            ambient_temperature=25.0
        )
        # Validation should pass as 600V * 5A = 3000W
        assert data.dc_power == 3000.0

    def test_ac_cannot_exceed_dc(self):
        """Test AC power cannot exceed DC power."""
        with pytest.raises(ValueError):
            PerformanceData(
                timestamp=datetime.now(),
                dc_power=2000.0,
                ac_power=2500.0,  # Exceeds DC
                dc_voltage=400.0,
                dc_current=5.0,
                ac_voltage=240.0,
                ac_current=10.4,
                irradiance=800.0,
                module_temperature=45.0,
                ambient_temperature=25.0
            )


class TestFinancialModel:
    """Test FinancialModel model."""

    def test_valid_financial_model(self):
        """Test creating valid financial model."""
        model = FinancialModel(
            system_cost=50000.0,
            module_cost=25000.0,
            inverter_cost=8000.0,
            balance_of_system_cost=12000.0,
            installation_cost=5000.0,
            annual_om_cost=500.0,
            electricity_rate=0.12
        )
        assert model.system_cost == 50000.0
        assert model.electricity_rate == 0.12

    def test_cost_breakdown_validation(self):
        """Test cost breakdown sums correctly."""
        model = FinancialModel(
            system_cost=50000.0,
            module_cost=25000.0,
            inverter_cost=8000.0,
            balance_of_system_cost=12000.0,
            installation_cost=5000.0,
            annual_om_cost=500.0,
            electricity_rate=0.12
        )
        # 25000 + 8000 + 12000 + 5000 = 50000
        assert model.system_cost == 50000.0

    def test_lcoe_calculation(self):
        """Test LCOE calculation."""
        model = FinancialModel(
            system_cost=50000.0,
            module_cost=25000.0,
            inverter_cost=8000.0,
            balance_of_system_cost=12000.0,
            installation_cost=5000.0,
            annual_om_cost=500.0,
            electricity_rate=0.12,
            system_lifetime=25
        )
        lcoe = model.calculate_lcoe(annual_energy_kwh=10000.0)
        assert lcoe > 0
        assert model.lcoe == lcoe

    def test_npv_calculation(self):
        """Test NPV calculation."""
        model = FinancialModel(
            system_cost=50000.0,
            module_cost=25000.0,
            inverter_cost=8000.0,
            balance_of_system_cost=12000.0,
            installation_cost=5000.0,
            annual_om_cost=500.0,
            electricity_rate=0.12,
            system_lifetime=25
        )
        npv = model.calculate_npv(annual_energy_kwh=10000.0)
        assert isinstance(npv, float)
        assert model.npv == npv

    def test_payback_calculation(self):
        """Test simple payback calculation."""
        model = FinancialModel(
            system_cost=50000.0,
            module_cost=25000.0,
            inverter_cost=8000.0,
            balance_of_system_cost=12000.0,
            installation_cost=5000.0,
            annual_om_cost=500.0,
            electricity_rate=0.12,
            system_lifetime=25
        )
        payback = model.calculate_simple_payback(annual_energy_kwh=10000.0)
        assert payback > 0
        assert model.payback_period == payback


# ============================================================================
# Test Session Manager
# ============================================================================

class TestSessionManager:
    """Test SessionManager class."""

    @pytest.fixture
    def temp_projects_dir(self):
        """Create temporary projects directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_projects_dir):
        """Create SessionManager with temporary directory."""
        return SessionManager(projects_directory=temp_projects_dir)

    def test_initialize_session_state(self, session_manager):
        """Test initializing session state."""
        state = session_manager.initialize_session_state(
            project_name="Test Project",
            project_id="test_001",
            description="Test description",
            author="Test Author"
        )
        assert state.metadata.project_name == "Test Project"
        assert state.metadata.project_id == "test_001"
        assert len(state.module_status) == 11  # All 11 modules

    def test_all_11_modules_initialized(self, session_manager):
        """Test all 11 simulation modules are initialized."""
        state = session_manager.initialize_session_state(
            project_name="Test",
            project_id="test"
        )
        expected_modules = [
            "material_selection",
            "cell_design",
            "module_engineering",
            "cutting_pattern",
            "system_design",
            "performance_simulation",
            "financial_analysis",
            "reliability_testing",
            "circularity_assessment",
            "scaps_integration",
            "energy_forecasting"
        ]
        for module in expected_modules:
            assert module in state.module_status
            assert state.module_status[module].completed is False
            assert state.module_status[module].completion_percentage == 0.0

    def test_create_new_project(self, session_manager):
        """Test creating a new project."""
        state = session_manager.create_new_project(
            project_name="New Project",
            description="Test description"
        )
        assert state.metadata.project_name == "New Project"
        assert session_manager.current_session is not None

    def test_save_and_load_project(self, session_manager, temp_projects_dir):
        """Test saving and loading a project."""
        # Create and save project
        state = session_manager.create_new_project(
            project_name="Save Test",
            description="Test save/load"
        )
        project_path = session_manager.save_project()
        assert project_path.exists()

        # Load project
        loaded_state = session_manager.load_project(project_path)
        assert loaded_state.metadata.project_name == "Save Test"
        assert len(loaded_state.module_status) == 11

    def test_list_projects(self, session_manager):
        """Test listing projects."""
        # Create multiple projects
        session_manager.create_new_project("Project 1")
        session_manager.create_new_project("Project 2")
        session_manager.create_new_project("Project 3")

        # List projects
        projects = session_manager.list_projects()
        assert len(projects) == 3
        assert all('project_name' in p for p in projects)

    def test_log_activity(self, session_manager):
        """Test logging activity."""
        session_manager.create_new_project("Activity Test")
        session_manager.log_activity(
            module="cell_design",
            action="create_cell",
            details="Created new cell design"
        )
        assert len(session_manager.current_session.activity_log) > 0

    def test_get_completion_percentage(self, session_manager):
        """Test getting completion percentage."""
        session_manager.create_new_project("Completion Test")

        # Initially 0%
        overall = session_manager.get_completion_percentage()
        assert overall == 0.0

        # Update one module
        session_manager.update_module_status(
            module="cell_design",
            completion_percentage=50.0
        )

        # Overall should be ~4.5% (50% of 1/11 modules)
        overall = session_manager.get_completion_percentage()
        assert 4.0 <= overall <= 5.0

    def test_update_module_status(self, session_manager):
        """Test updating module status."""
        session_manager.create_new_project("Update Test")

        session_manager.update_module_status(
            module="cell_design",
            completion_percentage=75.0,
            completed=False,
            data={'test_key': 'test_value'}
        )

        status = session_manager.get_module_status("cell_design")
        assert status.completion_percentage == 75.0
        assert status.completed is False
        assert status.data['test_key'] == 'test_value'

    def test_mark_module_complete(self, session_manager):
        """Test marking module as complete."""
        session_manager.create_new_project("Complete Test")

        session_manager.update_module_status(
            module="cell_design",
            completed=True
        )

        status = session_manager.get_module_status("cell_design")
        assert status.completed is True
        assert status.completion_percentage == 100.0

    def test_get_all_module_statuses(self, session_manager):
        """Test getting all module statuses."""
        session_manager.create_new_project("All Status Test")

        statuses = session_manager.get_all_module_statuses()
        assert len(statuses) == 11
        assert all(isinstance(s, ModuleCompletionStatus) for s in statuses.values())

    def test_get_activity_log(self, session_manager):
        """Test retrieving activity log."""
        session_manager.create_new_project("Log Test")

        session_manager.log_activity("module1", "action1", "details1")
        session_manager.log_activity("module2", "action2", "details2")
        session_manager.log_activity("module1", "action3", "details3")

        # Get all logs
        all_logs = session_manager.get_activity_log()
        assert len(all_logs) > 0

        # Get filtered logs
        module1_logs = session_manager.get_activity_log(module="module1")
        assert len(module1_logs) == 2

        # Get limited logs
        limited_logs = session_manager.get_activity_log(limit=1)
        assert len(limited_logs) == 1

    def test_export_session_summary(self, session_manager):
        """Test exporting session summary."""
        session_manager.create_new_project("Summary Test")

        session_manager.update_module_status("cell_design", completion_percentage=100, completed=True)
        session_manager.update_module_status("module_engineering", completion_percentage=50)

        summary = session_manager.export_session_summary()
        assert summary['project_name'] == "Summary Test"
        assert 'overall_completion' in summary
        assert 'modules' in summary
        assert len(summary['modules']) == 11

    def test_close_session(self, session_manager):
        """Test closing session."""
        session_manager.create_new_project("Close Test")
        assert session_manager.current_session is not None

        session_manager.close_session(save=True)
        assert session_manager.current_session is None
        assert session_manager.current_project_path is None

    def test_invalid_module_name(self, session_manager):
        """Test handling invalid module name."""
        session_manager.create_new_project("Invalid Module Test")

        with pytest.raises(ValueError):
            session_manager.update_module_status(
                module="invalid_module",
                completion_percentage=50.0
            )

    def test_completion_percentage_bounds(self, session_manager):
        """Test completion percentage validation."""
        session_manager.create_new_project("Bounds Test")

        with pytest.raises(ValueError):
            session_manager.update_module_status(
                module="cell_design",
                completion_percentage=150.0  # Invalid > 100
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def temp_projects_dir(self):
        """Create temporary projects directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow(self, temp_projects_dir):
        """Test complete workflow from project creation to completion."""
        # Initialize session manager
        manager = SessionManager(projects_directory=temp_projects_dir)

        # Create new project
        manager.create_new_project(
            project_name="Full Workflow Test",
            description="Testing complete workflow",
            author="Test User"
        )

        # Create materials
        silicon = create_example_silicon_material()
        assert silicon.material_type == MaterialType.SILICON

        # Create cell
        cell = create_default_monocrystalline_cell()
        assert cell.technology == CellTechnology.MONOCRYSTALLINE

        # Update module status
        manager.update_module_status(
            module="material_selection",
            completion_percentage=100,
            completed=True,
            data={'silicon_cost': silicon.total_cost}
        )

        manager.update_module_status(
            module="cell_design",
            completion_percentage=100,
            completed=True,
            data={'cell_efficiency': cell.efficiency}
        )

        # Check overall completion
        overall = manager.get_completion_percentage()
        assert overall > 0

        # Save project
        project_path = manager.save_project()
        assert project_path.exists()

        # Create new manager and load project
        manager2 = SessionManager(projects_directory=temp_projects_dir)
        loaded_state = manager2.load_project(project_path)

        assert loaded_state.metadata.project_name == "Full Workflow Test"
        assert loaded_state.module_status["material_selection"].completed is True
        assert loaded_state.module_status["cell_design"].completed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
