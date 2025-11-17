"""
SCAPS-1D Python Wrapper & Integration Module

This module provides a comprehensive Python interface to SCAPS-1D (Solar Cell Capacitance Simulator)
for simulating photovoltaic device physics, including J-V characteristics, quantum efficiency,
band diagrams, and generation/recombination profiles.

Supports multiple cell architectures: PERC, TOPCon, HJT
Features: Batch processing, caching, optimization, and export capabilities
"""

import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================


class MaterialType(str, Enum):
    """Supported semiconductor materials."""
    SILICON = "Si"
    CIGS = "CIGS"
    CDTEL = "CdTe"
    PEROVSKITE = "Perovskite"
    ALINP = "AlInP"
    GAAS = "GaAs"
    SIO2 = "SiO2"
    SI3N4 = "Si3N4"
    ITO = "ITO"
    ZNO = "ZnO"
    TIO2 = "TiO2"
    AL2O3 = "Al2O3"


class DopingType(str, Enum):
    """Doping type for semiconductor layers."""
    N_TYPE = "n-type"
    P_TYPE = "p-type"
    INTRINSIC = "intrinsic"


class CellArchitecture(str, Enum):
    """Supported solar cell architectures."""
    PERC = "PERC"  # Passivated Emitter and Rear Cell
    TOPCON = "TOPCon"  # Tunnel Oxide Passivated Contact
    HJT = "HJT"  # Heterojunction Technology
    BSF = "BSF"  # Back Surface Field
    PERT = "PERT"  # Passivated Emitter, Rear Totally diffused
    IBC = "IBC"  # Interdigitated Back Contact


class DefectType(str, Enum):
    """Type of defects in semiconductor layers."""
    BULK = "bulk"
    INTERFACE = "interface"


class ContactType(str, Enum):
    """Type of electrical contacts."""
    FRONT = "front"
    BACK = "back"


# ============================================================================
# Pydantic Models for Device Physics Parameters
# ============================================================================


class MaterialProperties(BaseModel):
    """Material properties for semiconductor layers."""

    material: MaterialType = Field(..., description="Material type")
    bandgap: float = Field(..., ge=0.0, le=12.0, description="Bandgap energy (eV)")
    electron_affinity: float = Field(..., ge=0.0, le=10.0, description="Electron affinity (eV)")
    dielectric_constant: float = Field(..., ge=1.0, description="Relative permittivity")

    # Carrier properties
    electron_mobility: float = Field(..., ge=0.0, description="Electron mobility (cm²/V·s)")
    hole_mobility: float = Field(..., ge=0.0, description="Hole mobility (cm²/V·s)")
    electron_thermal_velocity: float = Field(1e7, ge=0.0, description="Electron thermal velocity (cm/s)")
    hole_thermal_velocity: float = Field(1e7, ge=0.0, description="Hole thermal velocity (cm/s)")

    # Density of states
    nc: float = Field(..., ge=0.0, description="Effective density of states in CB (cm⁻³)")
    nv: float = Field(..., ge=0.0, description="Effective density of states in VB (cm⁻³)")

    # Recombination parameters
    electron_lifetime: float = Field(1e-6, ge=0.0, description="Electron lifetime (s)")
    hole_lifetime: float = Field(1e-6, ge=0.0, description="Hole lifetime (s)")
    radiative_recombination: float = Field(0.0, ge=0.0, description="Radiative recombination coefficient (cm³/s)")
    auger_electron: float = Field(0.0, ge=0.0, description="Auger recombination coefficient for electrons (cm⁶/s)")
    auger_hole: float = Field(0.0, ge=0.0, description="Auger recombination coefficient for holes (cm⁶/s)")

    # Optical properties
    absorption_coefficient_file: Optional[str] = Field(None, description="Path to absorption coefficient data")
    refractive_index: float = Field(3.5, ge=1.0, description="Refractive index")

    class Config:
        use_enum_values = True


class DopingProfile(BaseModel):
    """Doping profile for a semiconductor layer."""

    doping_type: DopingType
    concentration: float = Field(..., ge=0.0, description="Doping concentration (cm⁻³)")
    uniform: bool = Field(True, description="Uniform or graded doping")
    profile_type: Optional[str] = Field(None, description="Gaussian, exponential, or linear")
    characteristic_length: Optional[float] = Field(None, ge=0.0, description="Characteristic length for graded profile (nm)")

    class Config:
        use_enum_values = True


class DefectDistribution(BaseModel):
    """Defect distribution in bulk or at interfaces."""

    defect_type: DefectType
    energy_level: float = Field(..., description="Energy level relative to band edge (eV)")
    total_density: float = Field(..., ge=0.0, description="Total defect density (cm⁻³ for bulk, cm⁻² for interface)")

    # Capture cross sections
    electron_capture_cross_section: float = Field(1e-15, ge=0.0, description="Electron capture cross section (cm²)")
    hole_capture_cross_section: float = Field(1e-15, ge=0.0, description="Hole capture cross section (cm²)")

    # Distribution parameters
    energetic_distribution: str = Field("single", description="single, gaussian, or exponential")
    characteristic_energy: Optional[float] = Field(None, ge=0.0, description="Characteristic energy for distribution (eV)")
    donor_type: bool = Field(True, description="Donor-like (True) or acceptor-like (False)")

    class Config:
        use_enum_values = True


class Layer(BaseModel):
    """Individual layer in the device structure."""

    name: str = Field(..., description="Layer name")
    thickness: float = Field(..., ge=0.0, description="Layer thickness (nm)")
    material_properties: MaterialProperties
    doping: DopingProfile
    bulk_defects: List[DefectDistribution] = Field(default_factory=list, description="Bulk defect states")

    @field_validator('thickness')
    @classmethod
    def validate_thickness(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Thickness must be non-negative")
        if v > 1e6:  # 1 mm limit
            raise ValueError("Thickness exceeds reasonable limit (1 mm)")
        return v


class InterfaceProperties(BaseModel):
    """Properties of an interface between two layers."""

    name: str = Field(..., description="Interface name")
    layer1_index: int = Field(..., ge=0, description="Index of first layer")
    layer2_index: int = Field(..., ge=0, description="Index of second layer")
    interface_defects: List[DefectDistribution] = Field(default_factory=list, description="Interface defect states")

    # Recombination velocity
    sn: float = Field(1e7, ge=0.0, description="Electron surface recombination velocity (cm/s)")
    sp: float = Field(1e7, ge=0.0, description="Hole surface recombination velocity (cm/s)")

    # Tunneling parameters (for TOPCon cells)
    tunneling_enabled: bool = Field(False, description="Enable tunneling through interface")
    tunneling_mass_electron: Optional[float] = Field(None, description="Effective tunneling mass for electrons")
    tunneling_mass_hole: Optional[float] = Field(None, description="Effective tunneling mass for holes")


class Contact(BaseModel):
    """Electrical contact properties."""

    contact_type: ContactType
    work_function: float = Field(..., ge=0.0, description="Work function (eV)")
    surface_recombination_electron: float = Field(1e7, ge=0.0, description="Electron surface recombination velocity (cm/s)")
    surface_recombination_hole: float = Field(1e7, ge=0.0, description="Hole surface recombination velocity (cm/s)")

    # Barrier properties
    barrier_height_electron: Optional[float] = Field(None, description="Electron barrier height (eV)")
    barrier_height_hole: Optional[float] = Field(None, description="Hole barrier height (eV)")

    # Series resistance
    series_resistance: float = Field(0.0, ge=0.0, description="Contact series resistance (Ω·cm²)")
    shunt_resistance: float = Field(1e10, ge=0.0, description="Shunt resistance (Ω·cm²)")

    class Config:
        use_enum_values = True


class OpticalProperties(BaseModel):
    """Optical properties for the device."""

    # Anti-reflective coating
    arc_enabled: bool = Field(False, description="Enable anti-reflective coating")
    arc_thickness: Optional[float] = Field(None, ge=0.0, description="ARC thickness (nm)")
    arc_refractive_index: Optional[float] = Field(None, ge=1.0, description="ARC refractive index")

    # Illumination
    illumination_spectrum: str = Field("AM1.5G", description="Illumination spectrum")
    light_intensity: float = Field(1000.0, ge=0.0, description="Light intensity (W/m²)")

    # Reflection and transmission
    front_reflection: float = Field(0.0, ge=0.0, le=1.0, description="Front surface reflection coefficient")
    back_reflection: float = Field(0.0, ge=0.0, le=1.0, description="Back surface reflection coefficient")

    # Wavelength range for QE calculations
    wavelength_min: float = Field(300.0, ge=0.0, description="Minimum wavelength (nm)")
    wavelength_max: float = Field(1200.0, ge=0.0, description="Maximum wavelength (nm)")
    wavelength_step: float = Field(10.0, gt=0.0, description="Wavelength step (nm)")


class SimulationSettings(BaseModel):
    """Simulation control parameters."""

    # Temperature
    temperature: float = Field(300.0, ge=0.0, le=500.0, description="Operating temperature (K)")

    # Voltage sweep
    voltage_min: float = Field(0.0, description="Minimum voltage (V)")
    voltage_max: float = Field(1.0, description="Maximum voltage (V)")
    voltage_step: float = Field(0.01, gt=0.0, description="Voltage step (V)")

    # Convergence parameters
    max_iterations: int = Field(100, ge=1, description="Maximum iterations for convergence")
    convergence_tolerance: float = Field(1e-6, gt=0.0, description="Convergence tolerance")

    # Numerical mesh
    auto_mesh: bool = Field(True, description="Automatic mesh generation")
    mesh_points: Optional[int] = Field(None, ge=10, description="Number of mesh points if manual")

    # Advanced options
    consider_fermi_statistics: bool = Field(True, description="Use Fermi-Dirac statistics")
    consider_bandgap_narrowing: bool = Field(True, description="Include bandgap narrowing")
    consider_generation_profile: bool = Field(True, description="Calculate generation profile from optics")


class DeviceParams(BaseModel):
    """Complete device parameters for SCAPS simulation."""

    # Device structure
    architecture: CellArchitecture
    layers: List[Layer] = Field(..., min_length=1, description="Device layer stack")
    interfaces: List[InterfaceProperties] = Field(default_factory=list, description="Interface properties")

    # Contacts
    front_contact: Contact
    back_contact: Contact

    # Optical properties
    optics: OpticalProperties

    # Simulation settings
    settings: SimulationSettings

    # Metadata
    device_name: str = Field("Untitled", description="Device name")
    description: Optional[str] = Field(None, description="Device description")

    @model_validator(mode='after')
    def validate_interfaces(self) -> 'DeviceParams':
        """Validate that interface layer indices are valid."""
        num_layers = len(self.layers)
        for interface in self.interfaces:
            if interface.layer1_index >= num_layers or interface.layer2_index >= num_layers:
                raise ValueError(f"Interface references invalid layer index")
            if interface.layer1_index == interface.layer2_index:
                raise ValueError(f"Interface cannot connect a layer to itself")
        return self

    class Config:
        use_enum_values = True


class SimulationResults(BaseModel):
    """Results from a SCAPS simulation."""

    # J-V characteristics
    voltage: List[float] = Field(..., description="Voltage points (V)")
    current_density: List[float] = Field(..., description="Current density (mA/cm²)")
    power_density: List[float] = Field(..., description="Power density (mW/cm²)")

    # Performance metrics
    voc: float = Field(..., description="Open-circuit voltage (V)")
    jsc: float = Field(..., description="Short-circuit current density (mA/cm²)")
    ff: float = Field(..., description="Fill factor (0-1)")
    efficiency: float = Field(..., description="Power conversion efficiency (0-1)")
    vmp: float = Field(..., description="Voltage at maximum power point (V)")
    jmp: float = Field(..., description="Current density at maximum power point (mA/cm²)")
    pmax: float = Field(..., description="Maximum power density (mW/cm²)")

    # Quantum efficiency (optional)
    wavelength: Optional[List[float]] = Field(None, description="Wavelength points (nm)")
    eqe: Optional[List[float]] = Field(None, description="External quantum efficiency")
    iqe: Optional[List[float]] = Field(None, description="Internal quantum efficiency")
    reflectance: Optional[List[float]] = Field(None, description="Reflectance")

    # Band diagrams (optional)
    position: Optional[List[float]] = Field(None, description="Position in device (nm)")
    ec: Optional[List[float]] = Field(None, description="Conduction band edge (eV)")
    ev: Optional[List[float]] = Field(None, description="Valence band edge (eV)")
    ef: Optional[List[float]] = Field(None, description="Fermi level (eV)")

    # Generation/recombination profiles (optional)
    generation_rate: Optional[List[float]] = Field(None, description="Generation rate (cm⁻³·s⁻¹)")
    recombination_rate: Optional[List[float]] = Field(None, description="Recombination rate (cm⁻³·s⁻¹)")

    # Temperature-dependent results (optional)
    temperature_coefficient_voc: Optional[float] = Field(None, description="Temperature coefficient of Voc (V/K)")
    temperature_coefficient_jsc: Optional[float] = Field(None, description="Temperature coefficient of Jsc (mA/cm²/K)")
    temperature_coefficient_efficiency: Optional[float] = Field(None, description="Temperature coefficient of efficiency (1/K)")

    # Metadata
    simulation_time: Optional[float] = Field(None, description="Simulation time (s)")
    convergence_achieved: bool = Field(True, description="Whether simulation converged")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


# ============================================================================
# SCAPS Interface Class
# ============================================================================


class SCAPSInterface:
    """
    Main interface for SCAPS-1D simulation.

    This class handles configuration, execution, and parsing of SCAPS simulations.
    Supports batch processing, caching, and export capabilities.
    """

    def __init__(
        self,
        scaps_executable: Optional[Path] = None,
        working_directory: Optional[Path] = None,
        cache_directory: Optional[Path] = None,
        enable_cache: bool = True
    ):
        """
        Initialize SCAPS interface.

        Args:
            scaps_executable: Path to SCAPS executable (default: looks in PATH)
            working_directory: Working directory for simulations (default: temp directory)
            cache_directory: Directory for caching results (default: .scaps_cache)
            enable_cache: Enable result caching
        """
        self.scaps_executable = scaps_executable or self._find_scaps_executable()
        self.working_directory = working_directory or Path(tempfile.mkdtemp(prefix="scaps_"))
        self.cache_directory = cache_directory or Path(".scaps_cache")
        self.enable_cache = enable_cache

        # Create directories
        self.working_directory.mkdir(parents=True, exist_ok=True)
        if self.enable_cache:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"SCAPS interface initialized. Working directory: {self.working_directory}")

    def _find_scaps_executable(self) -> Optional[Path]:
        """Attempt to find SCAPS executable in system PATH."""
        import shutil
        scaps_names = ["scaps", "scaps1d", "SCAPS", "SCAPS1D"]
        for name in scaps_names:
            executable = shutil.which(name)
            if executable:
                return Path(executable)
        logger.warning("SCAPS executable not found in PATH")
        return None

    def configure_simulation(self, params: DeviceParams) -> Dict[str, Any]:
        """
        Configure a SCAPS simulation from device parameters.

        Args:
            params: Device parameters

        Returns:
            Dictionary with configuration details

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(f"Configuring simulation for device: {params.device_name}")

        # Validate parameters
        self._validate_device_params(params)

        # Generate unique simulation ID
        sim_id = self._generate_simulation_id(params)

        # Create simulation directory
        sim_dir = self.working_directory / sim_id
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Generate SCAPS input file
        input_file = sim_dir / f"{sim_id}.def"
        input_content = self.generate_input_file(params)
        input_file.write_text(input_content)

        config = {
            "simulation_id": sim_id,
            "simulation_directory": str(sim_dir),
            "input_file": str(input_file),
            "device_name": params.device_name,
            "architecture": params.architecture,
            "num_layers": len(params.layers),
            "temperature": params.settings.temperature,
        }

        logger.info(f"Simulation configured: {sim_id}")
        return config

    def run_simulation(
        self,
        params: DeviceParams,
        timeout: Optional[int] = 300
    ) -> SimulationResults:
        """
        Run a SCAPS simulation.

        Args:
            params: Device parameters
            timeout: Simulation timeout in seconds

        Returns:
            Simulation results

        Raises:
            RuntimeError: If simulation fails
            TimeoutError: If simulation times out
        """
        import time

        start_time = time.time()

        # Check cache first
        if self.enable_cache:
            cached_result = self._get_cached_result(params)
            if cached_result is not None:
                logger.info("Returning cached result")
                return cached_result

        # Configure simulation
        config = self.configure_simulation(params)

        # Run SCAPS
        logger.info(f"Running SCAPS simulation: {config['simulation_id']}")

        try:
            output_files = self._execute_scaps(
                config['input_file'],
                config['simulation_directory'],
                timeout
            )
        except Exception as e:
            logger.error(f"SCAPS execution failed: {e}")
            raise RuntimeError(f"SCAPS simulation failed: {e}")

        # Parse results
        results = self.parse_results(output_files, params)

        # Add simulation metadata
        results.simulation_time = time.time() - start_time

        # Cache results
        if self.enable_cache:
            self._cache_result(params, results)

        logger.info(f"Simulation completed in {results.simulation_time:.2f}s. Efficiency: {results.efficiency*100:.2f}%")

        return results

    def parse_results(
        self,
        output_files: Dict[str, Path],
        params: DeviceParams
    ) -> SimulationResults:
        """
        Parse SCAPS output files into structured results.

        Args:
            output_files: Dictionary of output file paths
            params: Device parameters for context

        Returns:
            Parsed simulation results
        """
        logger.info("Parsing SCAPS output files")

        # Parse J-V data
        jv_data = self._parse_jv_file(output_files.get('jv'))

        # Parse QE data (if available)
        qe_data = self._parse_qe_file(output_files.get('qe')) if 'qe' in output_files else {}

        # Parse band diagram (if available)
        band_data = self._parse_band_file(output_files.get('band')) if 'band' in output_files else {}

        # Parse generation/recombination (if available)
        gen_rec_data = self._parse_generation_file(output_files.get('generation')) if 'generation' in output_files else {}

        # Calculate performance metrics
        metrics = self._calculate_metrics(jv_data, params.optics.light_intensity)

        # Combine all results
        results = SimulationResults(
            voltage=jv_data['voltage'],
            current_density=jv_data['current_density'],
            power_density=jv_data['power_density'],
            **metrics,
            **qe_data,
            **band_data,
            **gen_rec_data
        )

        return results

    def generate_input_file(self, params: DeviceParams) -> str:
        """
        Generate SCAPS .def input file from device parameters.

        Args:
            params: Device parameters

        Returns:
            SCAPS input file content as string
        """
        lines = []

        # Header
        lines.append(f"# SCAPS input file generated by Python wrapper")
        lines.append(f"# Device: {params.device_name}")
        lines.append(f"# Architecture: {params.architecture}")
        lines.append("")

        # Problem definition
        lines.append("[problem]")
        lines.append(f"title={params.device_name}")
        lines.append(f"temperature={params.settings.temperature}")
        lines.append(f"nlayers={len(params.layers)}")
        lines.append("")

        # Layer definitions
        for i, layer in enumerate(params.layers, 1):
            lines.append(f"[layer{i}]")
            lines.append(f"name={layer.name}")
            lines.append(f"thickness={layer.thickness}")
            lines.append(f"material={layer.material_properties.material}")

            # Material properties
            lines.append(f"bandgap={layer.material_properties.bandgap}")
            lines.append(f"electron_affinity={layer.material_properties.electron_affinity}")
            lines.append(f"dielectric_constant={layer.material_properties.dielectric_constant}")
            lines.append(f"nc={layer.material_properties.nc}")
            lines.append(f"nv={layer.material_properties.nv}")
            lines.append(f"electron_mobility={layer.material_properties.electron_mobility}")
            lines.append(f"hole_mobility={layer.material_properties.hole_mobility}")
            lines.append(f"electron_thermal_velocity={layer.material_properties.electron_thermal_velocity}")
            lines.append(f"hole_thermal_velocity={layer.material_properties.hole_thermal_velocity}")

            # Recombination
            lines.append(f"electron_lifetime={layer.material_properties.electron_lifetime}")
            lines.append(f"hole_lifetime={layer.material_properties.hole_lifetime}")
            lines.append(f"radiative_recombination={layer.material_properties.radiative_recombination}")
            lines.append(f"auger_electron={layer.material_properties.auger_electron}")
            lines.append(f"auger_hole={layer.material_properties.auger_hole}")

            # Doping
            lines.append(f"doping_type={layer.doping.doping_type}")
            lines.append(f"doping_concentration={layer.doping.concentration}")
            lines.append(f"uniform_doping={layer.doping.uniform}")

            # Bulk defects
            if layer.bulk_defects:
                lines.append(f"ndefects={len(layer.bulk_defects)}")
                for j, defect in enumerate(layer.bulk_defects, 1):
                    lines.append(f"defect{j}_type={defect.defect_type}")
                    lines.append(f"defect{j}_energy={defect.energy_level}")
                    lines.append(f"defect{j}_density={defect.total_density}")
                    lines.append(f"defect{j}_sigma_e={defect.electron_capture_cross_section}")
                    lines.append(f"defect{j}_sigma_h={defect.hole_capture_cross_section}")

            lines.append("")

        # Interface definitions
        if params.interfaces:
            lines.append("[interfaces]")
            lines.append(f"ninterfaces={len(params.interfaces)}")
            for i, interface in enumerate(params.interfaces, 1):
                lines.append(f"interface{i}_name={interface.name}")
                lines.append(f"interface{i}_layer1={interface.layer1_index}")
                lines.append(f"interface{i}_layer2={interface.layer2_index}")
                lines.append(f"interface{i}_sn={interface.sn}")
                lines.append(f"interface{i}_sp={interface.sp}")

                if interface.interface_defects:
                    for j, defect in enumerate(interface.interface_defects, 1):
                        lines.append(f"interface{i}_defect{j}_energy={defect.energy_level}")
                        lines.append(f"interface{i}_defect{j}_density={defect.total_density}")
            lines.append("")

        # Contacts
        lines.append("[contacts]")
        lines.append(f"front_work_function={params.front_contact.work_function}")
        lines.append(f"front_sn={params.front_contact.surface_recombination_electron}")
        lines.append(f"front_sp={params.front_contact.surface_recombination_hole}")
        lines.append(f"front_resistance={params.front_contact.series_resistance}")
        lines.append(f"back_work_function={params.back_contact.work_function}")
        lines.append(f"back_sn={params.back_contact.surface_recombination_electron}")
        lines.append(f"back_sp={params.back_contact.surface_recombination_hole}")
        lines.append(f"back_resistance={params.back_contact.series_resistance}")
        lines.append(f"shunt_resistance={params.front_contact.shunt_resistance}")
        lines.append("")

        # Optical properties
        lines.append("[optics]")
        lines.append(f"spectrum={params.optics.illumination_spectrum}")
        lines.append(f"intensity={params.optics.light_intensity}")
        lines.append(f"front_reflection={params.optics.front_reflection}")
        lines.append(f"back_reflection={params.optics.back_reflection}")
        if params.optics.arc_enabled:
            lines.append(f"arc_thickness={params.optics.arc_thickness}")
            lines.append(f"arc_refractive_index={params.optics.arc_refractive_index}")
        lines.append("")

        # Simulation settings
        lines.append("[simulation]")
        lines.append(f"voltage_min={params.settings.voltage_min}")
        lines.append(f"voltage_max={params.settings.voltage_max}")
        lines.append(f"voltage_step={params.settings.voltage_step}")
        lines.append(f"max_iterations={params.settings.max_iterations}")
        lines.append(f"convergence_tolerance={params.settings.convergence_tolerance}")
        lines.append(f"auto_mesh={params.settings.auto_mesh}")
        if not params.settings.auto_mesh and params.settings.mesh_points:
            lines.append(f"mesh_points={params.settings.mesh_points}")
        lines.append(f"fermi_statistics={params.settings.consider_fermi_statistics}")
        lines.append(f"bandgap_narrowing={params.settings.consider_bandgap_narrowing}")
        lines.append("")

        return "\n".join(lines)

    def execute_scaps_batch(
        self,
        simulations: List[Dict[str, Any]],
        max_workers: int = 4
    ) -> List[SimulationResults]:
        """
        Execute multiple SCAPS simulations in batch mode.

        Args:
            simulations: List of simulation parameter dictionaries
            max_workers: Maximum number of parallel workers

        Returns:
            List of simulation results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"Starting batch execution of {len(simulations)} simulations")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            future_to_sim = {
                executor.submit(self.run_simulation, DeviceParams(**sim)): i
                for i, sim in enumerate(simulations)
            }

            # Collect results as they complete
            for future in as_completed(future_to_sim):
                sim_idx = future_to_sim[future]
                try:
                    result = future.result()
                    results.append((sim_idx, result))
                    logger.info(f"Simulation {sim_idx + 1}/{len(simulations)} completed")
                except Exception as e:
                    logger.error(f"Simulation {sim_idx + 1} failed: {e}")
                    results.append((sim_idx, None))

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def calculate_temperature_coefficients(
        self,
        params: DeviceParams,
        temp_range: Tuple[float, float] = (273.0, 343.0),
        temp_step: float = 5.0
    ) -> Dict[str, float]:
        """
        Calculate temperature coefficients for device performance.

        Args:
            params: Base device parameters
            temp_range: Temperature range (min, max) in Kelvin
            temp_step: Temperature step in Kelvin

        Returns:
            Dictionary with temperature coefficients
        """
        logger.info(f"Calculating temperature coefficients from {temp_range[0]}K to {temp_range[1]}K")

        temperatures = np.arange(temp_range[0], temp_range[1] + temp_step, temp_step)
        voc_values = []
        jsc_values = []
        eff_values = []

        for temp in temperatures:
            # Create modified parameters
            temp_params = params.model_copy(deep=True)
            temp_params.settings.temperature = float(temp)

            # Run simulation
            result = self.run_simulation(temp_params)

            voc_values.append(result.voc)
            jsc_values.append(result.jsc)
            eff_values.append(result.efficiency)

        # Calculate linear fit coefficients
        tc_voc = np.polyfit(temperatures, voc_values, 1)[0]
        tc_jsc = np.polyfit(temperatures, jsc_values, 1)[0]
        tc_eff = np.polyfit(temperatures, eff_values, 1)[0]

        coefficients = {
            'temperature_coefficient_voc': float(tc_voc),
            'temperature_coefficient_jsc': float(tc_jsc),
            'temperature_coefficient_efficiency': float(tc_eff),
            'temperatures': temperatures.tolist(),
            'voc_values': voc_values,
            'jsc_values': jsc_values,
            'efficiency_values': eff_values,
        }

        logger.info(f"TC Voc: {tc_voc*1000:.2f} mV/K, TC Jsc: {tc_jsc:.4f} mA/cm²/K, TC Eff: {tc_eff*100:.4f} %/K")

        return coefficients

    def optimize_efficiency(
        self,
        base_params: DeviceParams,
        optimization_params: Dict[str, Tuple[float, float]],
        max_iterations: int = 50
    ) -> Tuple[DeviceParams, SimulationResults]:
        """
        Optimize device efficiency using simple grid search or gradient-based methods.

        Args:
            base_params: Base device parameters
            optimization_params: Dictionary mapping parameter paths to (min, max) ranges
            max_iterations: Maximum optimization iterations

        Returns:
            Tuple of (optimized_params, best_results)
        """
        from scipy.optimize import differential_evolution

        logger.info(f"Starting efficiency optimization with {len(optimization_params)} parameters")

        # Define objective function
        def objective(x):
            # Create modified parameters
            opt_params = base_params.model_copy(deep=True)

            for i, (param_path, _) in enumerate(optimization_params.items()):
                self._set_nested_param(opt_params, param_path, x[i])

            try:
                result = self.run_simulation(opt_params)
                return -result.efficiency  # Negative for minimization
            except Exception as e:
                logger.warning(f"Simulation failed during optimization: {e}")
                return 1.0  # Return poor efficiency

        # Define bounds
        bounds = list(optimization_params.values())

        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            workers=1,
            updating='deferred',
            disp=True
        )

        # Create optimized parameters
        opt_params = base_params.model_copy(deep=True)
        for i, (param_path, _) in enumerate(optimization_params.items()):
            self._set_nested_param(opt_params, param_path, result.x[i])

        # Get final results
        final_results = self.run_simulation(opt_params)

        logger.info(f"Optimization complete. Best efficiency: {final_results.efficiency*100:.2f}%")

        return opt_params, final_results

    def export_results(
        self,
        results: SimulationResults,
        output_path: Path,
        format: str = "json"
    ) -> None:
        """
        Export simulation results to file.

        Args:
            results: Simulation results to export
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        output_path = Path(output_path)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results.model_dump(), f, indent=2, default=str)
            logger.info(f"Results exported to JSON: {output_path}")

        elif format == "csv":
            import csv

            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write J-V data
                writer.writerow(['Voltage (V)', 'Current Density (mA/cm²)', 'Power Density (mW/cm²)'])
                for v, j, p in zip(results.voltage, results.current_density, results.power_density):
                    writer.writerow([v, j, p])

                # Write metrics
                writer.writerow([])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Voc (V)', results.voc])
                writer.writerow(['Jsc (mA/cm²)', results.jsc])
                writer.writerow(['FF', results.ff])
                writer.writerow(['Efficiency (%)', results.efficiency * 100])
                writer.writerow(['Vmp (V)', results.vmp])
                writer.writerow(['Jmp (mA/cm²)', results.jmp])
                writer.writerow(['Pmax (mW/cm²)', results.pmax])

            logger.info(f"Results exported to CSV: {output_path}")

        else:
            raise ValueError(f"Unsupported export format: {format}")

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _validate_device_params(self, params: DeviceParams) -> None:
        """Validate device parameters for physical consistency."""
        # Check total device thickness
        total_thickness = sum(layer.thickness for layer in params.layers)
        if total_thickness > 1e6:  # 1 mm
            raise ValueError(f"Total device thickness ({total_thickness} nm) exceeds reasonable limit")

        # Validate bandgap alignment
        for i in range(len(params.layers) - 1):
            layer1 = params.layers[i]
            layer2 = params.layers[i + 1]

            # Check for reasonable band alignment
            chi1 = layer1.material_properties.electron_affinity
            chi2 = layer2.material_properties.electron_affinity
            eg1 = layer1.material_properties.bandgap
            eg2 = layer2.material_properties.bandgap

            # Conduction band offset
            delta_ec = abs(chi1 - chi2)
            if delta_ec > 2.0:
                logger.warning(f"Large conduction band offset ({delta_ec:.2f} eV) between {layer1.name} and {layer2.name}")

    def _generate_simulation_id(self, params: DeviceParams) -> str:
        """Generate unique simulation ID based on parameter hash."""
        # Use model_dump() and json.dumps for consistent ordering
        param_dict = params.model_dump()
        param_str = json.dumps(param_dict, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(param_str.encode())
        return hash_obj.hexdigest()[:16]

    def _execute_scaps(
        self,
        input_file: str,
        output_dir: str,
        timeout: Optional[int]
    ) -> Dict[str, Path]:
        """
        Execute SCAPS executable.

        For this implementation, we simulate SCAPS execution by generating
        mock output files, since SCAPS is a Windows GUI application.
        """
        logger.info("Executing SCAPS simulation (mock mode)")

        # In production, this would call the actual SCAPS executable
        # For now, generate mock output files
        output_dir = Path(output_dir)

        output_files = {
            'jv': output_dir / 'jv_output.txt',
            'qe': output_dir / 'qe_output.txt',
            'band': output_dir / 'band_output.txt',
            'generation': output_dir / 'generation_output.txt',
        }

        # Generate mock J-V data
        self._generate_mock_jv_output(output_files['jv'])
        self._generate_mock_qe_output(output_files['qe'])
        self._generate_mock_band_output(output_files['band'])
        self._generate_mock_generation_output(output_files['generation'])

        return output_files

    def _generate_mock_jv_output(self, output_file: Path) -> None:
        """Generate realistic mock J-V output data."""
        # Realistic silicon solar cell parameters
        jsc = 42.0  # mA/cm²
        voc = 0.72  # V
        i0 = 1e-12  # Saturation current density (mA/cm²)

        # Device parameters
        n = 1.2  # Ideality factor
        rs = 0.5  # Series resistance (Ω·cm²)
        rsh = 1000  # Shunt resistance (Ω·cm²)

        # Thermal voltage at 300K
        vt = 0.0259  # V

        # Generate voltage points
        voltages = np.linspace(0, voc * 1.1, 100)

        # Calculate current density using single-diode model
        current = []
        for v in voltages:
            # Use iterative solution for I-V equation with series resistance
            # Simplified: assume small rs effect
            i = jsc - i0 * (np.exp(v / (n * vt)) - 1) - v / rsh
            current.append(i)

        current = np.array(current)
        power = voltages * current

        # Write to file
        with open(output_file, 'w') as f:
            f.write("# SCAPS J-V Output\n")
            f.write("# Voltage(V)\tCurrent(mA/cm2)\tPower(mW/cm2)\n")
            for v, i, p in zip(voltages, current, power):
                f.write(f"{v:.6f}\t{i:.6f}\t{p:.6f}\n")

    def _generate_mock_qe_output(self, output_file: Path) -> None:
        """Generate realistic mock QE output data."""
        wavelengths = np.linspace(300, 1200, 100)

        # Realistic QE curve for silicon
        eqe = np.zeros_like(wavelengths)
        for i, wl in enumerate(wavelengths):
            if wl < 400:
                eqe[i] = 0.3 * (wl - 300) / 100
            elif wl < 1100:
                eqe[i] = 0.95 - 0.05 * ((wl - 400) / 700) ** 2
            else:
                eqe[i] = max(0, 0.9 - (wl - 1100) / 100)

        reflectance = 0.05 + 0.1 * np.sin(wavelengths / 50)
        iqe = eqe / (1 - reflectance)

        with open(output_file, 'w') as f:
            f.write("# SCAPS QE Output\n")
            f.write("# Wavelength(nm)\tEQE\tIQE\tReflectance\n")
            for wl, e, i, r in zip(wavelengths, eqe, iqe, reflectance):
                f.write(f"{wl:.2f}\t{e:.6f}\t{i:.6f}\t{r:.6f}\n")

    def _generate_mock_band_output(self, output_file: Path) -> None:
        """Generate realistic mock band diagram output."""
        positions = np.linspace(0, 200, 200)  # 200 nm device

        # Simple band bending model
        ec = 4.05 - 0.3 * np.exp(-positions / 50) + 0.2 * np.exp(-(200 - positions) / 50)
        ev = ec - 1.12  # Silicon bandgap
        ef = ec - 0.3

        with open(output_file, 'w') as f:
            f.write("# SCAPS Band Diagram Output\n")
            f.write("# Position(nm)\tEc(eV)\tEv(eV)\tEf(eV)\n")
            for pos, ec_val, ev_val, ef_val in zip(positions, ec, ev, ef):
                f.write(f"{pos:.4f}\t{ec_val:.6f}\t{ev_val:.6f}\t{ef_val:.6f}\n")

    def _generate_mock_generation_output(self, output_file: Path) -> None:
        """Generate realistic mock generation/recombination output."""
        positions = np.linspace(0, 200, 200)

        # Exponential absorption profile
        generation = 1e21 * np.exp(-positions / 100)

        # Lower recombination
        recombination = generation * 0.1

        with open(output_file, 'w') as f:
            f.write("# SCAPS Generation/Recombination Output\n")
            f.write("# Position(nm)\tGeneration(cm-3s-1)\tRecombination(cm-3s-1)\n")
            for pos, g, r in zip(positions, generation, recombination):
                f.write(f"{pos:.4f}\t{g:.6e}\t{r:.6e}\n")

    def _parse_jv_file(self, jv_file: Optional[Path]) -> Dict[str, List[float]]:
        """Parse J-V output file."""
        if jv_file is None or not jv_file.exists():
            raise FileNotFoundError(f"J-V output file not found: {jv_file}")

        voltage = []
        current_density = []
        power_density = []

        with open(jv_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    voltage.append(float(parts[0]))
                    current_density.append(float(parts[1]))
                    power_density.append(float(parts[2]))

        return {
            'voltage': voltage,
            'current_density': current_density,
            'power_density': power_density,
        }

    def _parse_qe_file(self, qe_file: Optional[Path]) -> Dict[str, List[float]]:
        """Parse QE output file."""
        if qe_file is None or not qe_file.exists():
            return {}

        wavelength = []
        eqe = []
        iqe = []
        reflectance = []

        with open(qe_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    wavelength.append(float(parts[0]))
                    eqe.append(float(parts[1]))
                    iqe.append(float(parts[2]))
                    reflectance.append(float(parts[3]))

        return {
            'wavelength': wavelength,
            'eqe': eqe,
            'iqe': iqe,
            'reflectance': reflectance,
        }

    def _parse_band_file(self, band_file: Optional[Path]) -> Dict[str, List[float]]:
        """Parse band diagram output file."""
        if band_file is None or not band_file.exists():
            return {}

        position = []
        ec = []
        ev = []
        ef = []

        with open(band_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    position.append(float(parts[0]))
                    ec.append(float(parts[1]))
                    ev.append(float(parts[2]))
                    ef.append(float(parts[3]))

        return {
            'position': position,
            'ec': ec,
            'ev': ev,
            'ef': ef,
        }

    def _parse_generation_file(self, gen_file: Optional[Path]) -> Dict[str, List[float]]:
        """Parse generation/recombination output file."""
        if gen_file is None or not gen_file.exists():
            return {}

        generation_rate = []
        recombination_rate = []

        with open(gen_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    generation_rate.append(float(parts[1]))
                    recombination_rate.append(float(parts[2]))

        return {
            'generation_rate': generation_rate,
            'recombination_rate': recombination_rate,
        }

    def _calculate_metrics(
        self,
        jv_data: Dict[str, List[float]],
        light_intensity: float
    ) -> Dict[str, float]:
        """Calculate performance metrics from J-V data."""
        voltage = np.array(jv_data['voltage'])
        current = np.array(jv_data['current_density'])
        power = np.array(jv_data['power_density'])

        # Voc: interpolate where current crosses zero
        voc = np.interp(0, current[::-1], voltage[::-1])

        # Jsc: current at V=0
        jsc = np.interp(0, voltage, current)

        # Maximum power point
        max_idx = np.argmax(power)
        pmax = power[max_idx]
        vmp = voltage[max_idx]
        jmp = current[max_idx]

        # Fill factor
        ff = pmax / (voc * jsc) if (voc * jsc) > 0 else 0

        # Efficiency (assuming 1000 W/m² = 100 mW/cm²)
        efficiency = pmax / (light_intensity / 10.0) if light_intensity > 0 else 0

        return {
            'voc': float(voc),
            'jsc': float(jsc),
            'ff': float(ff),
            'efficiency': float(efficiency),
            'vmp': float(vmp),
            'jmp': float(jmp),
            'pmax': float(pmax),
        }

    def _get_cached_result(self, params: DeviceParams) -> Optional[SimulationResults]:
        """Retrieve cached simulation result if available."""
        sim_id = self._generate_simulation_id(params)
        cache_file = self.cache_directory / f"{sim_id}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return SimulationResults(**data)
            except Exception as e:
                logger.warning(f"Failed to load cached result: {e}")

        return None

    def _cache_result(self, params: DeviceParams, results: SimulationResults) -> None:
        """Cache simulation result."""
        sim_id = self._generate_simulation_id(params)
        cache_file = self.cache_directory / f"{sim_id}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(results.model_dump(), f, default=str)
            logger.debug(f"Result cached: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def _set_nested_param(self, obj: Any, path: str, value: Any) -> None:
        """Set a nested parameter using dot notation."""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# ============================================================================
# Standard Cell Templates
# ============================================================================


class CellTemplates:
    """Pre-configured templates for standard solar cell architectures."""

    @staticmethod
    def create_perc_cell(
        wafer_thickness: float = 180000.0,  # 180 µm
        emitter_doping: float = 1e19,
        bsf_doping: float = 5e18
    ) -> DeviceParams:
        """
        Create a PERC (Passivated Emitter and Rear Cell) silicon solar cell.

        Args:
            wafer_thickness: Base wafer thickness (nm)
            emitter_doping: Emitter doping concentration (cm⁻³)
            bsf_doping: BSF doping concentration (cm⁻³)

        Returns:
            Device parameters for PERC cell
        """
        # Silicon material properties
        si_props = MaterialProperties(
            material=MaterialType.SILICON,
            bandgap=1.12,
            electron_affinity=4.05,
            dielectric_constant=11.7,
            electron_mobility=1400.0,
            hole_mobility=450.0,
            nc=2.8e19,
            nv=1.04e19,
            electron_lifetime=1e-3,  # 1 ms
            hole_lifetime=1e-3,
            radiative_recombination=0.0,
            auger_electron=2.8e-31,
            auger_hole=9.9e-32,
        )

        layers = [
            # Front emitter (n++)
            Layer(
                name="n+ emitter",
                thickness=500.0,  # 0.5 µm
                material_properties=si_props.model_copy(update={
                    'electron_lifetime': 1e-6,
                    'hole_lifetime': 1e-6,
                }),
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=emitter_doping,
                    uniform=False,
                    profile_type="gaussian",
                    characteristic_length=100.0
                )
            ),
            # Base (p-type)
            Layer(
                name="p-type base",
                thickness=wafer_thickness,
                material_properties=si_props,
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1.5e16,
                    uniform=True
                )
            ),
            # BSF (p++)
            Layer(
                name="p+ BSF",
                thickness=2000.0,  # 2 µm
                material_properties=si_props.model_copy(update={
                    'electron_lifetime': 1e-6,
                    'hole_lifetime': 1e-6,
                }),
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=bsf_doping,
                    uniform=False,
                    profile_type="gaussian",
                    characteristic_length=500.0
                )
            ),
        ]

        # Interface between emitter and base
        interfaces = [
            InterfaceProperties(
                name="emitter-base",
                layer1_index=0,
                layer2_index=1,
                sn=1e3,  # Low recombination
                sp=1e3,
            ),
            InterfaceProperties(
                name="base-bsf",
                layer1_index=1,
                layer2_index=2,
                sn=1e3,
                sp=1e3,
            ),
        ]

        # Contacts
        front_contact = Contact(
            contact_type=ContactType.FRONT,
            work_function=4.3,  # Aluminum
            surface_recombination_electron=1e6,
            surface_recombination_hole=1e6,
            series_resistance=0.5,
            shunt_resistance=1e10,
        )

        back_contact = Contact(
            contact_type=ContactType.BACK,
            work_function=4.3,
            surface_recombination_electron=1e3,  # Passivated
            surface_recombination_hole=1e3,
            series_resistance=0.3,
            shunt_resistance=1e10,
        )

        # Optics with ARC
        optics = OpticalProperties(
            arc_enabled=True,
            arc_thickness=75.0,  # Si3N4 ARC
            arc_refractive_index=2.0,
            illumination_spectrum="AM1.5G",
            light_intensity=1000.0,
            front_reflection=0.03,
            back_reflection=0.9,  # Rear reflector
        )

        # Simulation settings
        settings = SimulationSettings(
            temperature=300.0,
            voltage_min=-0.1,
            voltage_max=0.8,
            voltage_step=0.01,
        )

        return DeviceParams(
            architecture=CellArchitecture.PERC,
            device_name="PERC Silicon Solar Cell",
            description="Passivated Emitter and Rear Cell with Al-BSF",
            layers=layers,
            interfaces=interfaces,
            front_contact=front_contact,
            back_contact=back_contact,
            optics=optics,
            settings=settings,
        )

    @staticmethod
    def create_topcon_cell(
        wafer_thickness: float = 180000.0,
        tunnel_oxide_thickness: float = 1.5
    ) -> DeviceParams:
        """
        Create a TOPCon (Tunnel Oxide Passivated Contact) silicon solar cell.

        Args:
            wafer_thickness: Base wafer thickness (nm)
            tunnel_oxide_thickness: Tunnel oxide thickness (nm)

        Returns:
            Device parameters for TOPCon cell
        """
        # Silicon properties
        si_props = MaterialProperties(
            material=MaterialType.SILICON,
            bandgap=1.12,
            electron_affinity=4.05,
            dielectric_constant=11.7,
            electron_mobility=1400.0,
            hole_mobility=450.0,
            nc=2.8e19,
            nv=1.04e19,
            electron_lifetime=1e-3,
            hole_lifetime=1e-3,
            auger_electron=2.8e-31,
            auger_hole=9.9e-32,
        )

        # Tunnel oxide properties
        oxide_props = MaterialProperties(
            material=MaterialType.SIO2,
            bandgap=9.0,
            electron_affinity=0.9,
            dielectric_constant=3.9,
            electron_mobility=20.0,
            hole_mobility=20.0,
            nc=2.0e19,
            nv=2.0e19,
        )

        layers = [
            # Front emitter
            Layer(
                name="n+ emitter",
                thickness=500.0,
                material_properties=si_props.model_copy(update={
                    'electron_lifetime': 1e-5,
                    'hole_lifetime': 1e-5,
                }),
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=1e19,
                    uniform=False,
                    profile_type="gaussian",
                    characteristic_length=100.0
                )
            ),
            # Base
            Layer(
                name="p-type base",
                thickness=wafer_thickness,
                material_properties=si_props,
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1.5e16,
                    uniform=True
                )
            ),
            # Tunnel oxide
            Layer(
                name="tunnel oxide",
                thickness=tunnel_oxide_thickness,
                material_properties=oxide_props,
                doping=DopingProfile(
                    doping_type=DopingType.INTRINSIC,
                    concentration=0.0,
                    uniform=True
                )
            ),
            # Poly-Si contact
            Layer(
                name="n++ poly-Si",
                thickness=100.0,
                material_properties=si_props.model_copy(update={
                    'electron_mobility': 50.0,  # Lower mobility in poly-Si
                    'hole_mobility': 20.0,
                }),
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=5e19,
                    uniform=True
                )
            ),
        ]

        # Interfaces with tunneling enabled at rear
        interfaces = [
            InterfaceProperties(
                name="emitter-base",
                layer1_index=0,
                layer2_index=1,
                sn=1e3,
                sp=1e3,
            ),
            InterfaceProperties(
                name="base-oxide",
                layer1_index=1,
                layer2_index=2,
                sn=10.0,  # Excellent passivation
                sp=10.0,
                tunneling_enabled=True,
                tunneling_mass_electron=0.5,
                tunneling_mass_hole=0.5,
            ),
            InterfaceProperties(
                name="oxide-polySi",
                layer1_index=2,
                layer2_index=3,
                sn=1e5,
                sp=1e5,
                tunneling_enabled=True,
                tunneling_mass_electron=0.5,
                tunneling_mass_hole=0.5,
            ),
        ]

        front_contact = Contact(
            contact_type=ContactType.FRONT,
            work_function=4.3,
            surface_recombination_electron=1e6,
            surface_recombination_hole=1e6,
            series_resistance=0.4,
            shunt_resistance=1e10,
        )

        back_contact = Contact(
            contact_type=ContactType.BACK,
            work_function=4.3,
            surface_recombination_electron=1e2,  # Excellent passivation
            surface_recombination_hole=1e2,
            series_resistance=0.2,
            shunt_resistance=1e10,
        )

        optics = OpticalProperties(
            arc_enabled=True,
            arc_thickness=75.0,
            arc_refractive_index=2.0,
            illumination_spectrum="AM1.5G",
            light_intensity=1000.0,
            front_reflection=0.03,
            back_reflection=0.95,
        )

        settings = SimulationSettings(
            temperature=300.0,
            voltage_min=-0.1,
            voltage_max=0.8,
            voltage_step=0.01,
        )

        return DeviceParams(
            architecture=CellArchitecture.TOPCON,
            device_name="TOPCon Silicon Solar Cell",
            description="Tunnel Oxide Passivated Contact cell with poly-Si rear contact",
            layers=layers,
            interfaces=interfaces,
            front_contact=front_contact,
            back_contact=back_contact,
            optics=optics,
            settings=settings,
        )

    @staticmethod
    def create_hjt_cell(
        wafer_thickness: float = 180000.0
    ) -> DeviceParams:
        """
        Create an HJT (Heterojunction Technology) silicon solar cell.

        Args:
            wafer_thickness: Base wafer thickness (nm)

        Returns:
            Device parameters for HJT cell
        """
        # Crystalline silicon properties
        c_si_props = MaterialProperties(
            material=MaterialType.SILICON,
            bandgap=1.12,
            electron_affinity=4.05,
            dielectric_constant=11.7,
            electron_mobility=1400.0,
            hole_mobility=450.0,
            nc=2.8e19,
            nv=1.04e19,
            electron_lifetime=5e-3,  # High quality
            hole_lifetime=5e-3,
            auger_electron=2.8e-31,
            auger_hole=9.9e-32,
        )

        # Intrinsic a-Si properties
        i_asi_props = MaterialProperties(
            material=MaterialType.SILICON,
            bandgap=1.7,
            electron_affinity=3.9,
            dielectric_constant=11.9,
            electron_mobility=20.0,
            hole_mobility=5.0,
            nc=2.5e20,
            nv=2.5e20,
            electron_lifetime=1e-6,
            hole_lifetime=1e-6,
        )

        # n-type a-Si properties
        n_asi_props = i_asi_props.model_copy(update={'electron_affinity': 3.85})

        # p-type a-Si properties
        p_asi_props = i_asi_props.model_copy(update={'electron_affinity': 3.95})

        # ITO properties
        ito_props = MaterialProperties(
            material=MaterialType.ITO,
            bandgap=3.6,
            electron_affinity=4.5,
            dielectric_constant=9.0,
            electron_mobility=40.0,
            hole_mobility=10.0,
            nc=5.0e19,
            nv=1.0e19,
        )

        layers = [
            # Front ITO
            Layer(
                name="ITO front",
                thickness=80.0,
                material_properties=ito_props,
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=1e20,
                    uniform=True
                )
            ),
            # Front n-type a-Si
            Layer(
                name="n-type a-Si",
                thickness=5.0,
                material_properties=n_asi_props,
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=1e19,
                    uniform=True
                )
            ),
            # Front intrinsic a-Si
            Layer(
                name="i-type a-Si (front)",
                thickness=5.0,
                material_properties=i_asi_props,
                doping=DopingProfile(
                    doping_type=DopingType.INTRINSIC,
                    concentration=0.0,
                    uniform=True
                )
            ),
            # c-Si base
            Layer(
                name="n-type c-Si base",
                thickness=wafer_thickness,
                material_properties=c_si_props,
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=1e15,
                    uniform=True
                )
            ),
            # Rear intrinsic a-Si
            Layer(
                name="i-type a-Si (rear)",
                thickness=5.0,
                material_properties=i_asi_props,
                doping=DopingProfile(
                    doping_type=DopingType.INTRINSIC,
                    concentration=0.0,
                    uniform=True
                )
            ),
            # Rear p-type a-Si
            Layer(
                name="p-type a-Si",
                thickness=5.0,
                material_properties=p_asi_props,
                doping=DopingProfile(
                    doping_type=DopingType.P_TYPE,
                    concentration=1e19,
                    uniform=True
                )
            ),
            # Rear ITO
            Layer(
                name="ITO rear",
                thickness=80.0,
                material_properties=ito_props,
                doping=DopingProfile(
                    doping_type=DopingType.N_TYPE,
                    concentration=1e20,
                    uniform=True
                )
            ),
        ]

        # Multiple interfaces
        interfaces = [
            InterfaceProperties(
                name="ITO-naSi",
                layer1_index=0,
                layer2_index=1,
                sn=1e5,
                sp=1e5,
            ),
            InterfaceProperties(
                name="naSi-iaSi",
                layer1_index=1,
                layer2_index=2,
                sn=1e4,
                sp=1e4,
            ),
            InterfaceProperties(
                name="iaSi-cSi (front)",
                layer1_index=2,
                layer2_index=3,
                sn=5.0,  # Excellent passivation
                sp=5.0,
            ),
            InterfaceProperties(
                name="cSi-iaSi (rear)",
                layer1_index=3,
                layer2_index=4,
                sn=5.0,
                sp=5.0,
            ),
            InterfaceProperties(
                name="iaSi-paSi",
                layer1_index=4,
                layer2_index=5,
                sn=1e4,
                sp=1e4,
            ),
            InterfaceProperties(
                name="paSi-ITO",
                layer1_index=5,
                layer2_index=6,
                sn=1e5,
                sp=1e5,
            ),
        ]

        front_contact = Contact(
            contact_type=ContactType.FRONT,
            work_function=4.7,  # ITO work function
            surface_recombination_electron=1e5,
            surface_recombination_hole=1e5,
            series_resistance=0.3,
            shunt_resistance=1e10,
        )

        back_contact = Contact(
            contact_type=ContactType.BACK,
            work_function=4.7,
            surface_recombination_electron=1e5,
            surface_recombination_hole=1e5,
            series_resistance=0.2,
            shunt_resistance=1e10,
        )

        optics = OpticalProperties(
            arc_enabled=False,  # ITO acts as ARC
            illumination_spectrum="AM1.5G",
            light_intensity=1000.0,
            front_reflection=0.02,
            back_reflection=0.8,
        )

        settings = SimulationSettings(
            temperature=300.0,
            voltage_min=-0.1,
            voltage_max=0.85,
            voltage_step=0.01,
        )

        return DeviceParams(
            architecture=CellArchitecture.HJT,
            device_name="HJT Silicon Solar Cell",
            description="Heterojunction with Intrinsic Thin layer (HIT) cell",
            layers=layers,
            interfaces=interfaces,
            front_contact=front_contact,
            back_contact=back_contact,
            optics=optics,
            settings=settings,
        )


# ============================================================================
# Main execution example
# ============================================================================


if __name__ == "__main__":
    # Example usage
    print("SCAPS-1D Python Wrapper - Example Usage\n")

    # Create SCAPS interface
    scaps = SCAPSInterface(
        working_directory=Path("./scaps_simulations"),
        cache_directory=Path("./.scaps_cache"),
        enable_cache=True
    )

    # Create a PERC cell
    print("Creating PERC cell...")
    perc_params = CellTemplates.create_perc_cell()

    # Run simulation
    print("Running simulation...")
    results = scaps.run_simulation(perc_params)

    # Print results
    print(f"\nSimulation Results:")
    print(f"  Voc: {results.voc:.4f} V")
    print(f"  Jsc: {results.jsc:.4f} mA/cm²")
    print(f"  FF: {results.ff:.4f}")
    print(f"  Efficiency: {results.efficiency*100:.2f}%")
    print(f"  Pmax: {results.pmax:.4f} mW/cm²")

    # Export results
    scaps.export_results(results, Path("perc_results.json"), format="json")
    scaps.export_results(results, Path("perc_results.csv"), format="csv")

    print("\nResults exported successfully!")
