"""
Comprehensive unit tests for SCAPS wrapper module.

Tests cover:
- Pydantic model validation
- SCAPSInterface functionality
- File I/O and parsing
- Batch processing
- Caching mechanisms
- Cell templates
- Error handling
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.modules.scaps_wrapper import (
    CellArchitecture,
    CellTemplates,
    Contact,
    ContactType,
    DefectDistribution,
    DefectType,
    DeviceParams,
    DopingProfile,
    DopingType,
    InterfaceProperties,
    Layer,
    MaterialProperties,
    MaterialType,
    OpticalProperties,
    SCAPSInterface,
    SimulationResults,
    SimulationSettings,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def silicon_properties():
    """Standard silicon material properties."""
    return MaterialProperties(
        material=MaterialType.SILICON,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.7,
        electron_mobility=1400.0,
        hole_mobility=450.0,
        nc=2.8e19,
        nv=1.04e19,
        electron_lifetime=1e-6,
        hole_lifetime=1e-6,
    )


@pytest.fixture
def simple_layer(silicon_properties):
    """Simple silicon layer for testing."""
    return Layer(
        name="test_layer",
        thickness=1000.0,
        material_properties=silicon_properties,
        doping=DopingProfile(
            doping_type=DopingType.P_TYPE,
            concentration=1e16,
            uniform=True
        )
    )


@pytest.fixture
def simple_device_params(simple_layer):
    """Simple device parameters for testing."""
    return DeviceParams(
        architecture=CellArchitecture.PERC,
        device_name="Test Device",
        layers=[simple_layer],
        front_contact=Contact(
            contact_type=ContactType.FRONT,
            work_function=4.3,
            surface_recombination_electron=1e6,
            surface_recombination_hole=1e6,
        ),
        back_contact=Contact(
            contact_type=ContactType.BACK,
            work_function=4.3,
            surface_recombination_electron=1e6,
            surface_recombination_hole=1e6,
        ),
        optics=OpticalProperties(
            illumination_spectrum="AM1.5G",
            light_intensity=1000.0,
        ),
        settings=SimulationSettings(
            temperature=300.0,
            voltage_min=0.0,
            voltage_max=1.0,
            voltage_step=0.01,
        )
    )


@pytest.fixture
def scaps_interface(temp_dir):
    """SCAPS interface instance for testing."""
    return SCAPSInterface(
        working_directory=temp_dir / "work",
        cache_directory=temp_dir / "cache",
        enable_cache=True
    )


# ============================================================================
# Material Properties Tests
# ============================================================================


class TestMaterialProperties:
    """Test MaterialProperties model."""

    def test_valid_silicon_properties(self, silicon_properties):
        """Test valid silicon properties."""
        assert silicon_properties.material == MaterialType.SILICON
        assert silicon_properties.bandgap == 1.12
        assert silicon_properties.electron_mobility == 1400.0

    def test_invalid_bandgap(self):
        """Test validation of invalid bandgap."""
        with pytest.raises(ValidationError):
            MaterialProperties(
                material=MaterialType.SILICON,
                bandgap=-0.5,  # Negative bandgap
                electron_affinity=4.05,
                dielectric_constant=11.7,
                electron_mobility=1400.0,
                hole_mobility=450.0,
                nc=2.8e19,
                nv=1.04e19,
            )

    def test_invalid_mobility(self):
        """Test validation of invalid mobility."""
        with pytest.raises(ValidationError):
            MaterialProperties(
                material=MaterialType.SILICON,
                bandgap=1.12,
                electron_affinity=4.05,
                dielectric_constant=11.7,
                electron_mobility=-100.0,  # Negative mobility
                hole_mobility=450.0,
                nc=2.8e19,
                nv=1.04e19,
            )

    def test_optical_properties_optional(self, silicon_properties):
        """Test that optical properties are optional."""
        assert silicon_properties.absorption_coefficient_file is None
        assert silicon_properties.refractive_index == 3.5  # Default value


# ============================================================================
# Layer Tests
# ============================================================================


class TestLayer:
    """Test Layer model."""

    def test_valid_layer(self, simple_layer):
        """Test valid layer creation."""
        assert simple_layer.name == "test_layer"
        assert simple_layer.thickness == 1000.0
        assert simple_layer.doping.doping_type == DopingType.P_TYPE

    def test_negative_thickness(self, silicon_properties):
        """Test validation of negative thickness."""
        with pytest.raises(ValidationError):
            Layer(
                name="invalid",
                thickness=-100.0,
                material_properties=silicon_properties,
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1e16,
                    uniform=True
                )
            )

    def test_excessive_thickness(self, silicon_properties):
        """Test validation of excessive thickness."""
        with pytest.raises(ValidationError):
            Layer(
                name="invalid",
                thickness=2e6,  # 2 mm
                material_properties=silicon_properties,
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1e16,
                    uniform=True
                )
            )

    def test_bulk_defects(self, silicon_properties):
        """Test layer with bulk defects."""
        defect = DefectDistribution(
            defect_type=DefectType.BULK,
            energy_level=0.56,
            total_density=1e12,
            electron_capture_cross_section=1e-15,
            hole_capture_cross_section=1e-15,
        )

        layer = Layer(
            name="defective_layer",
            thickness=1000.0,
            material_properties=silicon_properties,
            doping=DopingProfile(
                doping_type=DopingType.P_TYPE,
                concentration=1e16,
                uniform=True
            ),
            bulk_defects=[defect]
        )

        assert len(layer.bulk_defects) == 1
        assert layer.bulk_defects[0].total_density == 1e12


# ============================================================================
# Device Parameters Tests
# ============================================================================


class TestDeviceParams:
    """Test DeviceParams model."""

    def test_valid_device_params(self, simple_device_params):
        """Test valid device parameters."""
        assert simple_device_params.architecture == CellArchitecture.PERC
        assert len(simple_device_params.layers) == 1
        assert simple_device_params.settings.temperature == 300.0

    def test_interface_validation(self, simple_layer):
        """Test interface layer index validation."""
        # Create device with invalid interface
        with pytest.raises(ValidationError):
            DeviceParams(
                architecture=CellArchitecture.PERC,
                device_name="Invalid Device",
                layers=[simple_layer],
                interfaces=[
                    InterfaceProperties(
                        name="invalid",
                        layer1_index=0,
                        layer2_index=5,  # Index out of range
                        sn=1e5,
                        sp=1e5,
                    )
                ],
                front_contact=Contact(
                    contact_type=ContactType.FRONT,
                    work_function=4.3,
                    surface_recombination_electron=1e6,
                    surface_recombination_hole=1e6,
                ),
                back_contact=Contact(
                    contact_type=ContactType.BACK,
                    work_function=4.3,
                    surface_recombination_electron=1e6,
                    surface_recombination_hole=1e6,
                ),
                optics=OpticalProperties(
                    illumination_spectrum="AM1.5G",
                    light_intensity=1000.0,
                ),
                settings=SimulationSettings()
            )

    def test_self_interface_validation(self, simple_layer):
        """Test that interface cannot connect layer to itself."""
        with pytest.raises(ValidationError):
            DeviceParams(
                architecture=CellArchitecture.PERC,
                device_name="Invalid Device",
                layers=[simple_layer],
                interfaces=[
                    InterfaceProperties(
                        name="self",
                        layer1_index=0,
                        layer2_index=0,  # Same layer
                        sn=1e5,
                        sp=1e5,
                    )
                ],
                front_contact=Contact(
                    contact_type=ContactType.FRONT,
                    work_function=4.3,
                    surface_recombination_electron=1e6,
                    surface_recombination_hole=1e6,
                ),
                back_contact=Contact(
                    contact_type=ContactType.BACK,
                    work_function=4.3,
                    surface_recombination_electron=1e6,
                    surface_recombination_hole=1e6,
                ),
                optics=OpticalProperties(
                    illumination_spectrum="AM1.5G",
                    light_intensity=1000.0,
                ),
                settings=SimulationSettings()
            )


# ============================================================================
# SCAPS Interface Tests
# ============================================================================


class TestSCAPSInterface:
    """Test SCAPSInterface class."""

    def test_initialization(self, scaps_interface, temp_dir):
        """Test SCAPS interface initialization."""
        assert scaps_interface.working_directory.exists()
        assert scaps_interface.cache_directory.exists()
        assert scaps_interface.enable_cache is True

    def test_configure_simulation(self, scaps_interface, simple_device_params):
        """Test simulation configuration."""
        config = scaps_interface.configure_simulation(simple_device_params)

        assert 'simulation_id' in config
        assert 'input_file' in config
        assert config['device_name'] == "Test Device"
        assert config['architecture'] == CellArchitecture.PERC
        assert config['num_layers'] == 1

    def test_generate_input_file(self, scaps_interface, simple_device_params):
        """Test SCAPS input file generation."""
        input_content = scaps_interface.generate_input_file(simple_device_params)

        assert isinstance(input_content, str)
        assert "[problem]" in input_content
        assert "[layer1]" in input_content
        assert "[contacts]" in input_content
        assert "[optics]" in input_content
        assert "[simulation]" in input_content
        assert "Test Device" in input_content

    def test_run_simulation(self, scaps_interface, simple_device_params):
        """Test running a simulation."""
        results = scaps_interface.run_simulation(simple_device_params)

        assert isinstance(results, SimulationResults)
        assert results.voc > 0
        assert results.jsc > 0
        assert 0 < results.ff < 1
        assert 0 < results.efficiency < 1
        assert results.convergence_achieved is True

    def test_simulation_caching(self, scaps_interface, simple_device_params):
        """Test result caching mechanism."""
        # First run
        results1 = scaps_interface.run_simulation(simple_device_params)

        # Second run should be cached
        results2 = scaps_interface.run_simulation(simple_device_params)

        # Results should be identical
        assert results1.voc == results2.voc
        assert results1.jsc == results2.jsc
        assert results1.efficiency == results2.efficiency

    def test_export_json(self, scaps_interface, simple_device_params, temp_dir):
        """Test JSON export."""
        results = scaps_interface.run_simulation(simple_device_params)
        output_file = temp_dir / "test_results.json"

        scaps_interface.export_results(results, output_file, format="json")

        assert output_file.exists()

        # Verify JSON content
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert 'voc' in data
        assert 'jsc' in data
        assert 'efficiency' in data

    def test_export_csv(self, scaps_interface, simple_device_params, temp_dir):
        """Test CSV export."""
        results = scaps_interface.run_simulation(simple_device_params)
        output_file = temp_dir / "test_results.csv"

        scaps_interface.export_results(results, output_file, format="csv")

        assert output_file.exists()

        # Verify CSV content
        content = output_file.read_text()
        assert 'Voltage' in content
        assert 'Current Density' in content
        assert 'Voc' in content

    def test_invalid_export_format(self, scaps_interface, simple_device_params, temp_dir):
        """Test invalid export format."""
        results = scaps_interface.run_simulation(simple_device_params)
        output_file = temp_dir / "test_results.xml"

        with pytest.raises(ValueError):
            scaps_interface.export_results(results, output_file, format="xml")


# ============================================================================
# Batch Processing Tests
# ============================================================================


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_execution(self, scaps_interface, simple_device_params):
        """Test batch simulation execution."""
        # Create variations
        simulations = []
        for temp in [280, 300, 320]:
            params_dict = simple_device_params.model_dump()
            params_dict['settings']['temperature'] = temp
            simulations.append(params_dict)

        results = scaps_interface.execute_scaps_batch(simulations, max_workers=2)

        assert len(results) == 3
        assert all(isinstance(r, SimulationResults) for r in results)

        # Check temperature dependence
        assert results[0].voc != results[1].voc  # Different temps should give different Voc

    def test_temperature_coefficients(self, scaps_interface, simple_device_params):
        """Test temperature coefficient calculation."""
        coefficients = scaps_interface.calculate_temperature_coefficients(
            simple_device_params,
            temp_range=(290.0, 310.0),
            temp_step=5.0
        )

        assert 'temperature_coefficient_voc' in coefficients
        assert 'temperature_coefficient_jsc' in coefficients
        assert 'temperature_coefficient_efficiency' in coefficients
        assert 'temperatures' in coefficients
        assert 'voc_values' in coefficients

        # TC Voc should be negative for silicon
        assert coefficients['temperature_coefficient_voc'] < 0


# ============================================================================
# Cell Template Tests
# ============================================================================


class TestCellTemplates:
    """Test standard cell templates."""

    def test_perc_cell_template(self):
        """Test PERC cell template."""
        perc = CellTemplates.create_perc_cell()

        assert perc.architecture == CellArchitecture.PERC
        assert len(perc.layers) == 3  # Emitter, base, BSF
        assert perc.layers[0].name == "n+ emitter"
        assert perc.layers[1].name == "p-type base"
        assert perc.layers[2].name == "p+ BSF"
        assert perc.optics.arc_enabled is True

    def test_topcon_cell_template(self):
        """Test TOPCon cell template."""
        topcon = CellTemplates.create_topcon_cell()

        assert topcon.architecture == CellArchitecture.TOPCON
        assert len(topcon.layers) == 4  # Emitter, base, oxide, poly-Si
        assert topcon.layers[2].name == "tunnel oxide"
        assert topcon.layers[3].name == "n++ poly-Si"

        # Check tunneling enabled
        tunnel_interface = None
        for interface in topcon.interfaces:
            if interface.tunneling_enabled:
                tunnel_interface = interface
                break

        assert tunnel_interface is not None

    def test_hjt_cell_template(self):
        """Test HJT cell template."""
        hjt = CellTemplates.create_hjt_cell()

        assert hjt.architecture == CellArchitecture.HJT
        assert len(hjt.layers) == 7  # ITO, n-aSi, i-aSi, c-Si, i-aSi, p-aSi, ITO
        assert len(hjt.interfaces) == 6  # Multiple interfaces

        # Check for c-Si base
        c_si_layer = hjt.layers[3]
        assert "c-Si" in c_si_layer.name

    def test_perc_simulation(self, scaps_interface):
        """Test running simulation with PERC template."""
        perc = CellTemplates.create_perc_cell()
        results = scaps_interface.run_simulation(perc)

        assert results.voc > 0.6  # Reasonable Voc for silicon
        assert results.jsc > 30.0  # Reasonable Jsc
        assert results.efficiency > 0.15  # At least 15% efficiency

    def test_topcon_simulation(self, scaps_interface):
        """Test running simulation with TOPCon template."""
        topcon = CellTemplates.create_topcon_cell()
        results = scaps_interface.run_simulation(topcon)

        assert results.voc > 0.6
        assert results.jsc > 30.0
        assert results.efficiency > 0.15

    def test_hjt_simulation(self, scaps_interface):
        """Test running simulation with HJT template."""
        hjt = CellTemplates.create_hjt_cell()
        results = scaps_interface.run_simulation(hjt)

        assert results.voc > 0.6
        assert results.jsc > 30.0
        assert results.efficiency > 0.15


# ============================================================================
# Simulation Results Tests
# ============================================================================


class TestSimulationResults:
    """Test simulation results processing."""

    def test_results_structure(self, scaps_interface, simple_device_params):
        """Test structure of simulation results."""
        results = scaps_interface.run_simulation(simple_device_params)

        # Check required fields
        assert len(results.voltage) > 0
        assert len(results.current_density) > 0
        assert len(results.power_density) > 0

        # Check metrics
        assert results.voc > 0
        assert results.jsc > 0
        assert results.ff > 0
        assert results.efficiency > 0
        assert results.vmp > 0
        assert results.jmp > 0
        assert results.pmax > 0

    def test_qe_data(self, scaps_interface, simple_device_params):
        """Test quantum efficiency data."""
        results = scaps_interface.run_simulation(simple_device_params)

        # QE data should be present
        assert results.wavelength is not None
        assert results.eqe is not None
        assert len(results.wavelength) == len(results.eqe)

        # EQE should be between 0 and 1
        assert all(0 <= e <= 1.1 for e in results.eqe)

    def test_band_diagram(self, scaps_interface, simple_device_params):
        """Test band diagram data."""
        results = scaps_interface.run_simulation(simple_device_params)

        assert results.position is not None
        assert results.ec is not None
        assert results.ev is not None
        assert results.ef is not None

        # Ec should be above Ev
        ec_array = np.array(results.ec)
        ev_array = np.array(results.ev)
        assert all(ec_array > ev_array)

    def test_generation_profile(self, scaps_interface, simple_device_params):
        """Test generation/recombination profiles."""
        results = scaps_interface.run_simulation(simple_device_params)

        assert results.generation_rate is not None
        assert results.recombination_rate is not None

        # Generation should be positive
        assert all(g >= 0 for g in results.generation_rate)
        assert all(r >= 0 for r in results.recombination_rate)

    def test_metrics_consistency(self, scaps_interface, simple_device_params):
        """Test consistency of calculated metrics."""
        results = scaps_interface.run_simulation(simple_device_params)

        # Pmax should equal Vmp * Jmp
        calculated_pmax = results.vmp * results.jmp
        assert abs(results.pmax - calculated_pmax) < 0.01

        # FF should equal Pmax / (Voc * Jsc)
        calculated_ff = results.pmax / (results.voc * results.jsc)
        assert abs(results.ff - calculated_ff) < 0.01


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and validation."""

    def test_missing_executable(self, temp_dir):
        """Test handling of missing SCAPS executable."""
        scaps = SCAPSInterface(
            scaps_executable=Path("/nonexistent/scaps"),
            working_directory=temp_dir / "work",
            enable_cache=False
        )

        # Should still initialize (executable check is optional)
        assert scaps.scaps_executable == Path("/nonexistent/scaps")

    def test_invalid_device_params(self, scaps_interface):
        """Test handling of invalid device parameters."""
        # Create device with unreasonable total thickness
        with pytest.raises(ValueError):
            thick_layer = Layer(
                name="thick",
                thickness=2e6,  # 2 mm
                material_properties=MaterialProperties(
                    material=MaterialType.SILICON,
                    bandgap=1.12,
                    electron_affinity=4.05,
                    dielectric_constant=11.7,
                    electron_mobility=1400.0,
                    hole_mobility=450.0,
                    nc=2.8e19,
                    nv=1.04e19,
                ),
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1e16,
                    uniform=True
                )
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, scaps_interface, temp_dir):
        """Test complete workflow from template to export."""
        # Create cell
        perc = CellTemplates.create_perc_cell()

        # Run simulation
        results = scaps_interface.run_simulation(perc)

        # Export results
        json_file = temp_dir / "integration_test.json"
        csv_file = temp_dir / "integration_test.csv"

        scaps_interface.export_results(results, json_file, format="json")
        scaps_interface.export_results(results, csv_file, format="csv")

        # Verify exports
        assert json_file.exists()
        assert csv_file.exists()

        # Verify JSON can be loaded back
        with open(json_file, 'r') as f:
            data = json.load(f)

        loaded_results = SimulationResults(**data)
        assert loaded_results.voc == results.voc
        assert loaded_results.efficiency == results.efficiency

    def test_parametric_sweep(self, scaps_interface):
        """Test parametric sweep of emitter doping."""
        results_list = []

        for doping in [1e18, 5e18, 1e19, 5e19]:
            perc = CellTemplates.create_perc_cell(emitter_doping=doping)
            results = scaps_interface.run_simulation(perc)
            results_list.append((doping, results.efficiency))

        # Check that we have results for all doping levels
        assert len(results_list) == 4

        # All efficiencies should be reasonable
        assert all(0.1 < eff < 0.3 for _, eff in results_list)

    def test_architecture_comparison(self, scaps_interface):
        """Test comparison of different cell architectures."""
        perc = CellTemplates.create_perc_cell()
        topcon = CellTemplates.create_topcon_cell()
        hjt = CellTemplates.create_hjt_cell()

        perc_results = scaps_interface.run_simulation(perc)
        topcon_results = scaps_interface.run_simulation(topcon)
        hjt_results = scaps_interface.run_simulation(hjt)

        # All should have reasonable efficiencies
        assert perc_results.efficiency > 0.15
        assert topcon_results.efficiency > 0.15
        assert hjt_results.efficiency > 0.15

        # Print comparison
        print(f"\nArchitecture Comparison:")
        print(f"  PERC: {perc_results.efficiency*100:.2f}%")
        print(f"  TOPCon: {topcon_results.efficiency*100:.2f}%")
        print(f"  HJT: {hjt_results.efficiency*100:.2f}%")


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
